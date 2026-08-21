"""Unit tests for Phase 3: Mathematical Feature Synthesis Engine & Provenance."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.synthesizer import (
    DiscoveryFeatureSynthesizer,
    evaluate_discovery_formula,
    generate_discovery_features_from_dataset,
    is_eligible_base_feature,
    zscore,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    GeneratorStrategy,
    compute_formula_hash,
)


class TestDiscoveryFeatureSynthesizer(unittest.TestCase):
    """Test suite for mathematical feature synthesis, formula evaluation, and budget constraints."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.df = pd.DataFrame({
            "feat_price": np.random.normal(100.0, 15.0, n),
            "feat_vol": np.random.exponential(10.0, n) + 1.0,
            "feat_skew": np.random.uniform(-2.0, 2.0, n),
            "feat_const": np.ones(n) * 5.0,  # Zero variance
            "label_up_5pct_5m": np.random.choice([0, 1], n),  # Target column
            "target_volatility": np.random.normal(0.2, 0.05, n),  # Target column
            "timestamp": pd.date_range("2026-08-20", periods=n, freq="6s"),
            "symbol": ["NIFTY"] * n,
        })

    def test_eligibility_and_target_exclusion(self):
        """Verify strict exclusion of targets, constants, timestamps, and symbols."""
        self.assertTrue(is_eligible_base_feature("feat_price", self.df["feat_price"]))
        self.assertTrue(is_eligible_base_feature("feat_vol", self.df["feat_vol"]))
        self.assertTrue(is_eligible_base_feature("feat_skew", self.df["feat_skew"]))

        # Excluded targets
        self.assertFalse(is_eligible_base_feature("label_up_5pct_5m", self.df["label_up_5pct_5m"]))
        self.assertFalse(is_eligible_base_feature("target_volatility", self.df["target_volatility"]))
        self.assertFalse(is_eligible_base_feature("timestamp", self.df["timestamp"]))
        self.assertFalse(is_eligible_base_feature("symbol", self.df["symbol"]))
        self.assertFalse(is_eligible_base_feature("feat_const", self.df["feat_const"]))

    def test_ratio_synthesis(self):
        """Verify ratio synthesis formula and numerical stability."""
        syn = DiscoveryFeatureSynthesizer.synthesize_ratio(
            "feat_price", "feat_vol", self.df["feat_price"], self.df["feat_vol"]
        )
        self.assertIsNotNone(syn)
        name, formula, s_val = syn
        self.assertEqual(name, "synth_ratio__feat_price__div__feat_vol")
        self.assertTrue(formula.startswith("col('feat_price') / (abs(col('feat_vol'))"))
        self.assertEqual(len(s_val), len(self.df))
        self.assertFalse(s_val.isna().any())
        self.assertFalse(np.isinf(s_val).any())

    def test_interaction_synthesis_and_symmetry(self):
        """Verify multiplicative interaction and lexicographical symmetry ordering."""
        syn1 = DiscoveryFeatureSynthesizer.synthesize_interaction(
            "feat_vol", "feat_price", self.df["feat_vol"], self.df["feat_price"]
        )
        syn2 = DiscoveryFeatureSynthesizer.synthesize_interaction(
            "feat_price", "feat_vol", self.df["feat_price"], self.df["feat_vol"]
        )
        self.assertIsNotNone(syn1)
        self.assertIsNotNone(syn2)

        name1, form1, s1 = syn1
        name2, form2, s2 = syn2

        # Symmetrical inputs produce identical canonical name and formula
        self.assertEqual(name1, name2)
        self.assertEqual(form1, form2)
        self.assertEqual(name1, "synth_inter__feat_price__x__feat_vol")
        np.testing.assert_allclose(s1.values, s2.values)

    def test_nonlinear_synthesis(self):
        """Verify log1p, tanh_z, and sq non-linear transforms."""
        syn_log = DiscoveryFeatureSynthesizer.synthesize_nonlinear("feat_vol", self.df["feat_vol"], "log1p")
        self.assertIsNotNone(syn_log)
        self.assertEqual(syn_log[0], "synth_log1p__feat_vol")

        syn_tanh = DiscoveryFeatureSynthesizer.synthesize_nonlinear("feat_skew", self.df["feat_skew"], "tanh_z")
        self.assertIsNotNone(syn_tanh)
        self.assertEqual(syn_tanh[0], "synth_tanh_z__feat_skew")

        syn_sq = DiscoveryFeatureSynthesizer.synthesize_nonlinear("feat_skew", self.df["feat_skew"], "sq")
        self.assertIsNotNone(syn_sq)
        self.assertEqual(syn_sq[0], "synth_sq__feat_skew")

    def test_spread_and_composite_synthesis(self):
        """Verify spread and 3-way composite synthesis."""
        syn_sp = DiscoveryFeatureSynthesizer.synthesize_spread(
            "feat_price", "feat_vol", self.df["feat_price"], self.df["feat_vol"]
        )
        self.assertIsNotNone(syn_sp)
        self.assertEqual(syn_sp[0], "synth_spread__feat_price__minus__feat_vol")

        syn_comp = DiscoveryFeatureSynthesizer.synthesize_composite(
            "feat_price", "feat_vol", "feat_skew",
            self.df["feat_price"], self.df["feat_vol"], self.df["feat_skew"],
        )
        self.assertIsNotNone(syn_comp)
        self.assertTrue(syn_comp[0].startswith("synth_comp__"))

    def test_vectorized_formula_evaluator(self):
        """Verify evaluate_discovery_formula matches vectorized numpy calculations."""
        formula = "col('feat_price') / (abs(col('feat_vol')) + 0.050000)"
        evaluated = evaluate_discovery_formula(self.df, formula)
        expected = self.df["feat_price"] / (self.df["feat_vol"].abs() + 0.05)
        np.testing.assert_allclose(evaluated.values, expected.values, rtol=1e-5)

    def test_batch_feature_generation_and_budget(self):
        """Verify bounded batch generation, provenance, and hash deduplication."""
        budget = DiscoveryPipelineBudget(max_new_features_per_gen=10)
        pipe_id = "DP_CAMP_SYNTH_001"

        specs, out_df = generate_discovery_features_from_dataset(
            self.df,
            pipeline_id=pipe_id,
            generation_number=1,
            budget=budget,
        )

        self.assertLessEqual(len(specs), 10)
        self.assertEqual(len(out_df.columns), len(specs))
        for spec in specs:
            self.assertEqual(spec.pipeline_id, pipe_id)
            self.assertEqual(spec.generation_discovered, 1)
            self.assertEqual(spec.lifecycle_status, DiscoveryLifecycleStatus.CANDIDATE)
            self.assertEqual(len(spec.formula_hash), 16)
            self.assertIn(spec.feature_name, out_df.columns)
            # Ensure none of the parents are targets
            for p in spec.parent_features:
                self.assertFalse(p.startswith("label_"))
                self.assertFalse(p.startswith("target_"))

        # Test duplicate prevention with existing hashes
        existing = {s.formula_hash for s in specs}
        specs_dup, _ = generate_discovery_features_from_dataset(
            self.df,
            pipeline_id=pipe_id,
            generation_number=2,
            existing_formula_hashes=existing,
            budget=budget,
        )
        for s in specs_dup:
            self.assertNotIn(s.formula_hash, existing)


if __name__ == "__main__":
    unittest.main()
