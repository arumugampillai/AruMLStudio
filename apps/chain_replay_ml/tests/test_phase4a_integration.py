"""Focused unit and integration validation tests for Phase 4A.5 Feature Transformation & Dataset Builder Integration."""

import os
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.feature_registry_store import load_store
from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store as load_pipeline_store
from chain_replay_ml.surface_math.feature_extractor import (
    FeatureQualificationReport,
    OptionSurfaceFeatureExtractor,
)
from chain_replay_ml.surface_math.types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
)


class TestPhase4AIntegration(unittest.TestCase):
    """Rigorous tests for Feature Registry integrity, qualification gating, and pipeline immutability."""

    def test_feature_registry_integrity_and_no_duplication(self) -> None:
        """Verify that new Phase 4A features have valid IDs and existing Greek IDs are strictly preserved."""
        store = load_store()
        f_ids = store.get("feature_ids", {})
        f_idents = store.get("feature_identities", {})

        # 1. Verify exact IDs for pre-existing features (must NOT be duplicated or reassigned)
        self.assertEqual(f_ids.get("vanna"), "FR0198")
        self.assertEqual(f_ids.get("volga"), "FR0201")
        self.assertEqual(f_ids.get("charm"), "FR0019")
        self.assertEqual(f_ids.get("speed"), "FR0120")
        self.assertEqual(f_ids.get("iv_skew_25d"), "FR0063")

        # 2. Verify new Phase 4A features are present with stable IDs
        expected_new_features = [
            ("color", "FR0391"),
            ("zomma", "FR0392"),
            ("ultima", "FR0393"),
            ("svi_param_a", "FR0394"),
            ("svi_param_b", "FR0395"),
            ("svi_param_rho", "FR0396"),
            ("svi_param_m", "FR0397"),
            ("svi_param_sigma", "FR0398"),
            ("svi_calibration_rmse", "FR0399"),
            ("sabr_param_alpha", "FR0400"),
            ("sabr_param_rho", "FR0401"),
            ("sabr_param_nu", "FR0402"),
            ("sabr_calibration_rmse", "FR0403"),
            ("iv_skew_10d", "FR0404"),
            ("iv_curvature_25d", "FR0405"),
            ("iv_term_slope_near_next", "FR0406"),
            ("surface_displacement_5m", "FR0407"),
            ("surface_displacement_15m", "FR0408"),
            ("surface_acceleration_15m", "FR0409"),
            ("vrp_proxy_30m", "FR0410"),
        ]

        for feat_name, expected_fid in expected_new_features:
            self.assertIn(feat_name, f_ids)
            self.assertEqual(f_ids[feat_name], expected_fid)
            self.assertIn(expected_fid, f_idents)
            ident = f_idents[expected_fid]
            self.assertEqual(ident["name"], feat_name)
            self.assertIn(ident["group_id"], ["greeks", "surface_svi", "surface_sabr", "surface_topology"])

    def test_base_pipeline_pl0001_immutability(self) -> None:
        """Verify that Base Pipeline PL_0001 remains 100% immutable."""
        p_store = load_pipeline_store()
        pipelines = p_store.get("pipelines", {})

        self.assertIn("PL_0001", pipelines)
        pl_0001 = pipelines["PL_0001"]
        self.assertEqual(pl_0001["type"], "base")
        self.assertEqual(pl_0001["status"], "ready")

        # Ensure newly registered experimental features are NOT injected into PL_0001
        pl_0001_features = set(pl_0001.get("registry_feature_ids", []))
        for fid in [f"FR{i:04d}" for i in range(391, 411)]:
            self.assertNotIn(fid, pl_0001_features)

    def test_pre_training_feature_qualification_gating(self) -> None:
        """Verify that Feature Analysis Lab qualifies features and rejects bad candidates."""
        extractor = OptionSurfaceFeatureExtractor()

        # Synthetic DataFrame with eligible, high-missingness, and zero-variance features
        n_rows = 100
        df = pd.DataFrame({
            "color": np.random.normal(0.0001, 0.00002, n_rows),  # Eligible
            "zomma": np.random.normal(-0.005, 0.001, n_rows),     # Eligible
            "svi_param_b": [None if i % 2 == 0 else 0.08 for i in range(n_rows)],  # 50% missing -> Reject
            "sabr_param_rho": np.full(n_rows, -0.35),             # Constant variance=0 -> Reject
        })

        reports = extractor.qualify_candidate_features(
            df,
            feature_names=["color", "zomma", "svi_param_b", "sabr_param_rho"],
            max_missingness_pct=5.0,
        )

        # 1. Eligible features
        self.assertTrue(reports["color"].is_eligible)
        self.assertTrue(reports["zomma"].is_eligible)
        self.assertEqual(reports["color"].missingness_pct, 0.0)
        self.assertGreater(reports["color"].variance, 0.0)

        # 2. High missingness rejected
        self.assertFalse(reports["svi_param_b"].is_eligible)
        self.assertTrue(any("EXCESSIVE_MISSINGNESS" in r for r in reports["svi_param_b"].rejection_reasons))

        # 3. Constant / Zero variance rejected
        self.assertFalse(reports["sabr_param_rho"].is_eligible)
        self.assertTrue(any("CONSTANT_OR_NEAR_ZERO_VARIANCE" in r for r in reports["sabr_param_rho"].rejection_reasons))

    def test_snapshot_feature_extraction_and_quality_gating(self) -> None:
        """Verify snapshot extraction and quality containment."""
        extractor = OptionSurfaceFeatureExtractor()

        strikes = np.linspace(23000.0, 26000.0, 31)
        flat_ivs = np.full_like(strikes, 0.15)
        spot = 24500.0
        t_exp = 14.0 / 365.0
        r = 0.07

        # Extract features for snapshot
        feats = extractor.extract_snapshot_features(
            underlying_spot=spot,
            time_to_expiry_years=t_exp,
            risk_free_rate=r,
            strikes=strikes,
            implied_volatilities=flat_ivs,
            as_of_timestamp=1782881800.0,
            expiry_date="2026-05-28",
        )

        self.assertIn("svi_param_a", feats)
        self.assertIn("sabr_param_alpha", feats)
        self.assertIn("iv_skew_25d", feats)
        self.assertIn("iv_curvature_25d", feats)

        # On flat smile, skew and curvature should be ~0
        self.assertAlmostEqual(feats["iv_skew_25d"], 0.0, places=3)
        self.assertAlmostEqual(feats["iv_curvature_25d"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
