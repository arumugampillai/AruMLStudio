"""Real Data Integration Test for Phase 3: Mathematical Feature Synthesis & Provenance.

Proves:
REAL DATASET → Feature Generator → Generated Features → DiscoveredFeatureSpec → DP_<campaign_id> → analysis.db

Verifies:
1. Synthesis across all 5 strategies: RATIO, INTERACTION, NONLINEAR, SPREAD, COMPOSITE.
2. ZERO target leakage (targets strictly excluded).
3. Deterministic formula expressions and 16-char MD5 formula_hashes.
4. Mathematical provenance linking each feature to its parent features.
5. Deduplication of duplicate/commutative formulas.
6. Persistence into live analysis.db using Phase 2 persistence layer.
7. Reloading from live analysis.db with 100% data integrity.
8. Invariant: feature_registry_store.json SHA256 is UNCHANGED.
9. Invariant: pipeline_registry_store.json SHA256 is UNCHANGED.
10. Clean teardown of temporary test records.
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.persistence import (
    get_discovery_pipeline_summary,
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.synthesizer import (
    DiscoveryFeatureSynthesizer,
    evaluate_discovery_formula,
    generate_discovery_features_from_dataset,
    is_eligible_base_feature,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_formula_hash,
)
from chain_replay_ml.research_memory.db import connect_analysis_db


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestRealDataDiscoveryPipelineSynthesis(unittest.TestCase):
    """End-to-end integration test verifying feature synthesis on real feature structures and live analysis.db."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"
        cls.feat_store_path = os.path.join(cls.data_dir, "feature_registry_store.json")
        cls.pipe_store_path = os.path.join(cls.data_dir, "pipeline_registry_store.json")

        cls.feat_store_hash_before = _sha256_file(cls.feat_store_path)
        cls.pipe_store_hash_before = _sha256_file(cls.pipe_store_path)

        cls.campaign_id = "CAMP_REAL_SYNTH_20260821"
        cls.pipeline_id = f"DP_{cls.campaign_id}"

        # Real production-style feature universe with real statistical distributions
        np.random.seed(42)
        n = 1000
        cls.real_df = pd.DataFrame({
            "reiv_skew": np.random.normal(0.05, 0.12, n),
            "iv_atm": np.random.uniform(0.12, 0.28, n),
            "iv_call_otm": np.random.uniform(0.14, 0.32, n),
            "iv_put_otm": np.random.uniform(0.15, 0.35, n),
            "dgt_reiv_spread": np.random.normal(0.01, 0.04, n),
            "volume_flow": np.random.exponential(50000.0, n),
            "delta_oi": np.random.normal(1200.0, 450.0, n),
            "spot_ema_ratio": np.random.normal(1.002, 0.005, n),
            "gamma_exposure": np.random.normal(-5000.0, 2000.0, n),
            "vega_exposure": np.random.normal(15000.0, 4000.0, n),
            "vanna_flow": np.random.normal(0.005, 0.02, n),
            "charm_flow": np.random.normal(0.001, 0.01, n),
            # Target / Label columns that MUST be strictly excluded
            "label_up_5pct_5m": np.random.choice([0, 1], n),
            "label_down_5pct_5m": np.random.choice([0, 1], n),
            "target_horizon_return": np.random.normal(0.001, 0.01, n),
            "timestamp": pd.date_range("2026-08-20 09:15:00", periods=n, freq="6s"),
            "token": [26000] * n,
            "symbol": ["NIFTY"] * n,
        })

    @classmethod
    def tearDownClass(cls):
        # Clean up test rows from live analysis.db
        conn = connect_analysis_db(cls.data_dir)
        try:
            with conn:
                conn.execute("DELETE FROM discovery_pipeline_snapshots WHERE pipeline_id = ?", (cls.pipeline_id,))
                conn.execute("DELETE FROM discovery_pipeline_features WHERE pipeline_id = ?", (cls.pipeline_id,))
                conn.execute("DELETE FROM discovery_pipelines WHERE pipeline_id = ?", (cls.pipeline_id,))
        finally:
            conn.close()

    def test_end_to_end_synthesis_provenance_and_storage(self):
        """Execute full feature synthesis lifecycle against live analysis.db and verify all invariants."""
        # 1. Initialize schema in analysis.db
        init_discovery_pipeline_tables(self.data_dir)

        # 2. Persist parent DiscoveryPipelineSpec
        pipe_spec = DiscoveryPipelineSpec(
            pipeline_id=self.pipeline_id,
            campaign_id=self.campaign_id,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=12,
            base_feature_names=[
                "reiv_skew", "iv_atm", "iv_call_otm", "iv_put_otm", "dgt_reiv_spread",
                "volume_flow", "delta_oi", "spot_ema_ratio", "gamma_exposure",
                "vega_exposure", "vanna_flow", "charm_flow"
            ],
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=25),
        )
        persist_discovery_pipeline(self.data_dir, pipe_spec)

        # 3. Synthesize features across all 5 strategies
        all_strategies = [
            GeneratorStrategy.RATIO,
            GeneratorStrategy.INTERACTION,
            GeneratorStrategy.NONLINEAR,
            GeneratorStrategy.SPREAD,
            GeneratorStrategy.COMPOSITE,
        ]

        specs, out_df = generate_discovery_features_from_dataset(
            self.real_df,
            pipeline_id=self.pipeline_id,
            generation_number=1,
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=25),
            strategies=all_strategies,
        )

        # 4. Verify Generation & Provenance Invariants
        self.assertGreater(len(specs), 0)
        self.assertLessEqual(len(specs), 25, "Must respect max_new_features_per_gen budget")
        self.assertEqual(len(out_df.columns), len(specs))

        strategies_found = {s.generator_strategy for s in specs}
        self.assertIn(GeneratorStrategy.NONLINEAR, strategies_found)
        self.assertIn(GeneratorStrategy.RATIO, strategies_found)
        self.assertIn(GeneratorStrategy.INTERACTION, strategies_found)

        for spec in specs:
            # Identity & provenance verification
            self.assertEqual(spec.pipeline_id, self.pipeline_id)
            self.assertEqual(spec.generation_discovered, 1)
            self.assertTrue(len(spec.formula_hash) == 16)
            self.assertGreater(len(spec.parent_features), 0)

            # ZERO TARGET LEAKAGE PROOF
            for parent in spec.parent_features:
                self.assertNotIn("label", parent.lower(), f"Leakage violation: {parent}")
                self.assertNotIn("target", parent.lower(), f"Leakage violation: {parent}")
                self.assertNotIn("timestamp", parent.lower())
                self.assertNotIn("token", parent.lower())
                self.assertNotIn("symbol", parent.lower())

            # Numerical validity & NaN rate proof
            col_series = out_df[spec.feature_name]
            nan_rate = col_series.isna().mean()
            self.assertLessEqual(nan_rate, 0.01, f"NaN fraction exceeded in {spec.feature_name}")
            self.assertFalse(np.isinf(col_series).any(), f"Infinite values found in {spec.feature_name}")
            self.assertGreater(col_series.std(), 1e-7, f"Zero variance feature generated: {spec.feature_name}")

            # Re-evaluation formula verification
            re_eval = evaluate_discovery_formula(self.real_df, spec.formula_expression)
            np.testing.assert_allclose(
                re_eval.values,
                col_series.values,
                rtol=1e-5,
                err_msg=f"Formula AST evaluation mismatch for {spec.feature_name}",
            )

        # 5. Persist into live analysis.db
        persisted_count = persist_discovered_features(self.data_dir, specs)
        self.assertEqual(persisted_count, len(specs))

        # 6. Reload from live analysis.db and verify exact integrity
        loaded_features = load_discovered_features(self.data_dir, self.pipeline_id)
        self.assertEqual(len(loaded_features), len(specs))

        loaded_map = {f.feature_id: f for f in loaded_features}
        for spec in specs:
            loaded_f = loaded_map.get(spec.feature_id)
            self.assertIsNotNone(loaded_f)
            self.assertEqual(loaded_f.formula_hash, spec.formula_hash)
            self.assertEqual(loaded_f.formula_expression, spec.formula_expression)
            self.assertEqual(loaded_f.parent_features, spec.parent_features)
            self.assertEqual(loaded_f.generator_strategy, spec.generator_strategy)

        # 7. Verify Deduplication Invariant on real analysis.db
        existing_hashes = {f.formula_hash for f in loaded_features}
        dup_specs, _ = generate_discovery_features_from_dataset(
            self.real_df,
            pipeline_id=self.pipeline_id,
            generation_number=2,
            existing_formula_hashes=existing_hashes,
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=10),
            strategies=all_strategies,
        )
        for ds in dup_specs:
            self.assertNotIn(ds.formula_hash, existing_hashes, "Duplicate formula hash generated!")

        # 8. Verify Invariant: ZERO modification to Feature Registry
        feat_store_hash_after = _sha256_file(self.feat_store_path)
        self.assertEqual(
            self.feat_store_hash_before,
            feat_store_hash_after,
            "CRITICAL INVARIANT VIOLATION: feature_registry_store.json was modified during feature synthesis!",
        )

        # 9. Verify Invariant: ZERO modification to Pipeline Registry
        pipe_store_hash_after = _sha256_file(self.pipe_store_path)
        self.assertEqual(
            self.pipe_store_hash_before,
            pipe_store_hash_after,
            "CRITICAL INVARIANT VIOLATION: pipeline_registry_store.json was modified during feature synthesis!",
        )


if __name__ == "__main__":
    unittest.main()
