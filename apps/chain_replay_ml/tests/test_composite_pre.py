"""Focused unit tests for Phase 4B.1: Pre-Training Composite Feature Selection Engine."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.feature_selection.composite_pre import PreTrainingCompositeSelector
from chain_replay_ml.feature_selection.types import (
    AttributionStage,
    CanonicalFeatureAction,
    CompositeSelectionConfig,
    DiscoveryDiagnosticAction,
)


class TestCompositePreSelection(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n_samples = 500
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        # Highly non-linear feature
        x_nonlinear = np.random.uniform(-3, 3, n_samples)
        # Collinear clone of x1
        x1_collinear = x1 + np.random.normal(0, 0.01, n_samples)
        # Pure random noise
        x_noise = np.random.normal(0, 1, n_samples)
        # Constant zero-variance feature
        x_const = np.zeros(n_samples)
        # High null feature
        x_null = np.random.normal(0, 1, n_samples)
        x_null[:50] = np.nan  # 10% nulls (> 5% max_null_pct)

        # Target: linear in x1, non-linear in x_nonlinear
        y = 2.0 * x1 + 3.0 * (x_nonlinear ** 2) + np.random.normal(0, 0.2, n_samples)

        self.df = pd.DataFrame({
            "feat_linear": x1,
            "feat_secondary": x2,
            "feat_nonlinear": x_nonlinear,
            "feat_collinear": x1_collinear,
            "feat_noise": x_noise,
            "feat_constant": x_const,
            "feat_high_null": x_null,
            "target": y,
        })
        self.selector = PreTrainingCompositeSelector()

    def test_no_shap_import(self) -> None:
        """Verify SHAP is strictly prohibited and not imported in composite_pre."""
        import sys
        import chain_replay_ml.feature_selection.composite_pre as comp_module
        self.assertNotIn("shap", dir(comp_module))

    def test_quarantine_high_null_and_zero_variance(self) -> None:
        """Verify features with >5% nulls or zero variance are quarantined into REMOVE."""
        res = self.selector.select_features(self.df, "target", run_id="test_run", dataset_id="test_ds")
        
        self.assertIn("feat_constant", res.quarantined_features)
        self.assertIn("feat_high_null", res.quarantined_features)
        
        const_attr = res.attributions["feat_constant"]
        self.assertEqual(const_attr.canonical_action, CanonicalFeatureAction.REMOVE)
        self.assertEqual(const_attr.diagnostic_action, DiscoveryDiagnosticAction.RETIRE_CANDIDATE.value)

        null_attr = res.attributions["feat_high_null"]
        self.assertEqual(null_attr.canonical_action, CanonicalFeatureAction.REMOVE)

    def test_non_linear_relationship_detection(self) -> None:
        """Verify that a purely non-linear relationship (y = x^2) is detected with high score."""
        res = self.selector.select_features(self.df, "target", run_id="test_run", dataset_id="test_ds")
        
        nl_attr = res.attributions["feat_nonlinear"]
        self.assertGreater(nl_attr.mi_raw, 0.20)
        self.assertEqual(nl_attr.canonical_action, CanonicalFeatureAction.KEEP)
        self.assertIn("feat_nonlinear", res.selected_features)

    def test_collinear_greedy_pruning(self) -> None:
        """Verify that collinear peers (r >= 0.95) have the lower-scoring feature pruned."""
        res = self.selector.select_features(self.df, "target", run_id="test_run", dataset_id="test_ds")
        
        # Exactly one of feat_linear or feat_collinear should survive in selected_features
        linear_in = "feat_linear" in res.selected_features
        collinear_in = "feat_collinear" in res.selected_features
        self.assertTrue(linear_in ^ collinear_in)
        self.assertTrue("feat_linear" in res.pruned_collinear_features or "feat_collinear" in res.pruned_collinear_features)

    def test_determinism_with_seed(self) -> None:
        """Verify identical results when run twice with same seed and config."""
        res1 = self.selector.select_features(self.df, "target", run_id="run_1", dataset_id="ds_1")
        res2 = self.selector.select_features(self.df, "target", run_id="run_1", dataset_id="ds_1")

        self.assertEqual(res1.selected_features, res2.selected_features)
        self.assertEqual(res1.quarantined_features, res2.quarantined_features)
        for col in res1.attributions:
            self.assertAlmostEqual(res1.attributions[col].composite_score, res2.attributions[col].composite_score, places=5)
            self.assertEqual(res1.attributions[col].canonical_action, res2.attributions[col].canonical_action)

    def test_discrete_classification_target(self) -> None:
        """Verify discrete classification targets use mutual_info_classif properly."""
        df_class = self.df.copy()
        df_class["class_target"] = (df_class["target"] > df_class["target"].median()).astype(int)
        res = self.selector.select_features(df_class, "class_target", run_id="class_run")
        self.assertGreater(res.selected_feature_count, 0)
        self.assertEqual(res.stage, AttributionStage.STAGE_DISCOVERY)


if __name__ == "__main__":
    unittest.main()
