"""Focused unit and integration validation tests for Phase 4A.6 Feature Analysis Lab Qualification Bridge."""

import os
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_feature_roles import (
    ROLE_PREDICTOR,
    classify_feature_role,
    predictor_columns,
)
from chain_replay_ml.dataset_builder.feature_registry_store import load_store
from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store as load_pipeline_store
from chain_replay_ml.surface_math.analysis_bridge import FeatureAnalysisLabBridge


class TestPhase4AAnalysisBridge(unittest.TestCase):
    """Rigorous tests for Feature Analysis Lab bridge, HCA/correlation clustering, and candidate selection."""

    def setUp(self) -> None:
        self.bridge = FeatureAnalysisLabBridge()
        self.phase4a_feature_names = [
            "color", "zomma", "ultima",
            "svi_param_a", "svi_param_b", "svi_param_rho", "svi_param_m", "svi_param_sigma", "svi_calibration_rmse",
            "sabr_param_alpha", "sabr_param_rho", "sabr_param_nu", "sabr_calibration_rmse",
            "iv_skew_25d", "iv_skew_10d", "iv_curvature_25d", "iv_term_slope_near_next",
            "surface_displacement_5m", "surface_displacement_15m", "surface_acceleration_15m", "vrp_proxy_30m",
        ]

    def test_phase4a_role_classification_all_predictors(self) -> None:
        """Verify that all Phase 4A features are correctly classified as ROLE_PREDICTOR."""
        for feat in self.phase4a_feature_names:
            role = classify_feature_role(feat)
            self.assertEqual(
                role,
                ROLE_PREDICTOR,
                f"Feature '{feat}' must be classified as ROLE_PREDICTOR, got '{role}'",
            )

    def test_registry_identity_preservation_and_status(self) -> None:
        """Verify registry stability and experimental status for all Phase 4A features."""
        store = load_store()
        f_ids = store.get("feature_ids", {})
        f_overrides = store.get("overrides", {})

        # 1. Verify exact existing IDs
        self.assertEqual(f_ids.get("vanna"), "FR0198")
        self.assertEqual(f_ids.get("volga"), "FR0201")
        self.assertEqual(f_ids.get("charm"), "FR0019")
        self.assertEqual(f_ids.get("speed"), "FR0120")
        self.assertEqual(f_ids.get("iv_skew_25d"), "FR0063")

        # 2. Verify all newly registered IDs (FR0391 - FR0410)
        for i in range(391, 411):
            fid = f"FR{i:04d}"
            self.assertIn(fid, f_overrides)
            self.assertEqual(f_overrides[fid].get("implementation_status"), "experimental")

    def test_base_pipeline_pl0001_immutability(self) -> None:
        """Verify that PL_0001 remains 100% immutable."""
        p_store = load_pipeline_store()
        pl_0001 = p_store.get("pipelines", {}).get("PL_0001", {})
        self.assertEqual(pl_0001.get("type"), "base")
        pl_features = set(pl_0001.get("registry_feature_ids", []))
        for i in range(391, 411):
            self.assertNotIn(f"FR{i:04d}", pl_features)

    def test_full_analysis_lab_bridge_pipeline_execution(self) -> None:
        """Verify end-to-end execution of the existing Feature Analysis Lab via the bridge."""
        np.random.seed(42)
        n_rows = 150

        # Synthetic multi-feature dataset with existing Greeks + Phase 4A features + metadata
        df = pd.DataFrame({
            # Metadata
            "trading_day": ["2026-05-26"] * n_rows,
            "timestamp": [1779769680.0 + 60.0 * i for i in range(n_rows)],
            # Existing Base Predictors
            "delta": np.linspace(0.48, 0.52, n_rows),
            "gamma": np.full(n_rows, 0.00045),
            "vega": np.full(n_rows, 12.5),
            "vanna": np.linspace(-0.05, -0.04, n_rows),
            "current_iv": np.linspace(0.14, 0.16, n_rows),
            # Phase 4A Eligible Predictors
            "color": np.random.normal(0.00005, 0.00001, n_rows),
            "zomma": np.random.normal(-0.005, 0.0005, n_rows),
            "ultima": np.random.normal(-750.0, 15.0, n_rows),
            "svi_calibration_rmse": np.linspace(0.015, 0.025, n_rows),
            "sabr_calibration_rmse": np.linspace(0.018, 0.028, n_rows),
            "iv_skew_25d": np.linspace(-0.01, 0.01, n_rows),
            "iv_skew_10d": np.linspace(0.02, 0.05, n_rows),
            "iv_curvature_25d": np.linspace(0.005, 0.015, n_rows),
            "vrp_proxy_30m": np.random.normal(0.002, 0.0002, n_rows),
            # Phase 4A Quarantined / High Missingness Predictors
            "svi_param_a": [None] * n_rows,  # 100% missing
            "sabr_param_alpha": [None] * n_rows,  # 100% missing
            "iv_term_slope_near_next": [None] * n_rows,  # 100% missing
        })

        # Run bridge
        result = self.bridge.run_analysis_pipeline(
            df=df,
            run_id="test_run_bridge",
            dataset_id="test_ds_bridge",
            corr_threshold=0.95,
        )

        # 1. Check profiling and predictor counts
        self.assertGreater(result.total_features_profiled, 15)
        self.assertGreater(result.predictor_count, 12)

        # 2. Check candidate dataset selection
        self.assertGreater(result.selected_feature_count, 5)

        # 3. Prove that 100% missing / quarantined features are REJECTED
        self.assertIn("svi_param_a", result.rejected_features)
        self.assertIn("sabr_param_alpha", result.rejected_features)
        self.assertIn("iv_term_slope_near_next", result.rejected_features)
        self.assertNotIn("svi_param_a", result.selected_features)

        # 4. Prove that eligible surface features enter final candidate dataset
        self.assertIn("color", result.selected_features)
        self.assertIn("zomma", result.selected_features)
        self.assertIn("ultima", result.selected_features)
        self.assertIn("vrp_proxy_30m", result.selected_features)

        # 5. Check HCA families formed
        self.assertGreater(result.hca_family_count, 0)


if __name__ == "__main__":
    unittest.main()
