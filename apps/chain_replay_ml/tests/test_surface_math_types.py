"""Focused unit and smoke tests for Phase 4A.0 surface math data contracts and types."""

import json
import math
import unittest

from chain_replay_ml.surface_math.types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
    FeatureProvenanceRecord,
    HigherOrderGreeksRecord,
    MathematicalFamily,
    SabrBetaMode,
    SabrCalibrationResult,
    SabrParameters,
    SurfaceMathConfig,
    SurfaceTopologicalFeatures,
    SviCalibrationResult,
    SviParameters,
)


class TestSurfaceMathTypes(unittest.TestCase):
    """Verify Phase 4A.0 data contracts, serialization, validation, and configuration."""

    def test_higher_order_greeks_record(self) -> None:
        """Verify HigherOrderGreeksRecord creation and serialization."""
        rec = HigherOrderGreeksRecord(
            delta=0.52,
            gamma=0.0012,
            theta=-12.5,
            vega=4.8,
            vanna=-0.1438,
            volga=36.61,
            charm=-0.0021,
            color=0.000056,
            speed=-0.000000147,
            zomma=-0.00519,
            ultima=-757.12,
            strike=24500.0,
            time_to_expiry_years=0.01918,
            implied_volatility=0.15,
            underlying_spot=24520.0,
            option_type="CE",
        )
        d = rec.to_dict()
        self.assertEqual(d["vanna"], -0.1438)
        self.assertEqual(d["volga"], 36.61)
        self.assertEqual(d["color"], 0.000056)
        restored = HigherOrderGreeksRecord.from_dict(d)
        self.assertEqual(restored, rec)

    def test_svi_parameters_and_no_arbitrage_validation(self) -> None:
        """Verify SVI total variance evaluation, IV calculation, and strict no-arbitrage boundary checks."""
        # 1. Valid NIFTY smile SVI parameters
        params = SviParameters(
            a=0.002,
            b=0.08,
            rho=-0.45,
            m=-0.02,
            sigma=0.05,
        )
        # Total variance at ATM (k = 0)
        w_atm = params.total_variance(0.0)
        self.assertGreater(w_atm, 0.0)

        # Implied vol for 7 days to expiry (7/365 years)
        t_exp = 7.0 / 365.0
        iv_atm = params.implied_volatility(0.0, t_exp)
        self.assertGreater(iv_atm, 0.05)
        self.assertLess(iv_atm, 0.80)

        # Verify no-arbitrage
        is_valid, violations = params.verify_no_arbitrage(t_exp)
        self.assertTrue(is_valid)
        self.assertEqual(violations, [])

        # 2. Invalid parameter: negative b slope
        invalid_b = SviParameters(a=0.002, b=-0.05, rho=-0.45, m=-0.02, sigma=0.05)
        is_valid_b, viol_b = invalid_b.verify_no_arbitrage(t_exp)
        self.assertFalse(is_valid_b)
        self.assertTrue(any("NEGATIVE_B_SLOPE" in v for v in viol_b))

        # 3. Invalid parameter: |rho| >= 1
        invalid_rho = SviParameters(a=0.002, b=0.08, rho=-1.2, m=-0.02, sigma=0.05)
        is_valid_rho, viol_rho = invalid_rho.verify_no_arbitrage(t_exp)
        self.assertFalse(is_valid_rho)
        self.assertTrue(any("INVALID_RHO_RANGE" in v for v in viol_rho))

        # 4. Invalid parameter: Lee wing slope violation (excessive b)
        invalid_lee = SviParameters(a=0.002, b=150.0, rho=0.5, m=0.0, sigma=0.05)
        is_valid_lee, viol_lee = invalid_lee.verify_no_arbitrage(t_exp)
        self.assertFalse(is_valid_lee)
        self.assertTrue(any("LEE_WING_SLOPE_VIOLATION" in v for v in viol_lee))

        # 5. Dict serialization round-trip
        d = params.to_dict()
        self.assertEqual(SviParameters.from_dict(d), params)

    def test_svi_calibration_result_container(self) -> None:
        """Verify SviCalibrationResult serialization and tiered quality classification."""
        params = SviParameters(a=0.002, b=0.08, rho=-0.45, m=-0.02, sigma=0.05)
        res = SviCalibrationResult(
            parameters=params,
            status=CalibrationStatus.CONVERGED,
            quality_tier=CalibrationQualityTier.TIER_1_HIGH_PRECISION,
            rmse=0.0125,
            mae=0.0098,
            max_error=0.028,
            strikes_used=24,
            optimization_iterations=65,
            as_of_timestamp=1782880200.0,
            expiry_date="2026-05-28",
            time_to_expiry_years=0.01918,
            warnings=[],
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "CONVERGED")
        self.assertEqual(d["quality_tier"], "TIER_1_HIGH_PRECISION")
        self.assertEqual(d["rmse"], 0.0125)

        restored = SviCalibrationResult.from_dict(d)
        self.assertEqual(restored.parameters, params)
        self.assertEqual(restored.status, CalibrationStatus.CONVERGED)
        self.assertEqual(restored.quality_tier, CalibrationQualityTier.TIER_1_HIGH_PRECISION)

    def test_sabr_parameters_and_closed_form_iv(self) -> None:
        """Verify SABR implied volatility evaluation for ATM and OTM strikes."""
        # Standard equity index SABR (beta = 0.5, negative correlation rho = -0.35)
        sabr = SabrParameters(alpha=22.5, beta=0.5, rho=-0.35, nu=0.85)
        F = 24500.0
        t_exp = 14.0 / 365.0

        # ATM volatility (K == F)
        iv_atm = sabr.implied_volatility(strike=24500.0, forward=F, time_to_expiry_years=t_exp)
        self.assertGreater(iv_atm, 0.05)
        self.assertLess(iv_atm, 0.40)

        # OTM Put volatility (K = 24000 < F) -> should have higher IV due to negative skew rho
        iv_otm_put = sabr.implied_volatility(strike=24000.0, forward=F, time_to_expiry_years=t_exp)
        # OTM Call volatility (K = 25000 > F)
        iv_otm_call = sabr.implied_volatility(strike=25000.0, forward=F, time_to_expiry_years=t_exp)

        self.assertGreater(iv_otm_put, iv_atm)
        self.assertGreater(iv_otm_put, iv_otm_call)

        # Dict round-trip
        d = sabr.to_dict()
        self.assertEqual(SabrParameters.from_dict(d), sabr)

    def test_sabr_calibration_result_container(self) -> None:
        """Verify SabrCalibrationResult serialization and diagnostics."""
        sabr = SabrParameters(alpha=22.5, beta=0.5, rho=-0.35, nu=0.85)
        res = SabrCalibrationResult(
            parameters=sabr,
            status=CalibrationStatus.CONVERGED,
            quality_tier=CalibrationQualityTier.TIER_1_HIGH_PRECISION,
            rmse=0.0185,
            mae=0.0142,
            strikes_used=22,
            forward_price=24510.0,
            atm_implied_volatility=0.145,
            as_of_timestamp=1782880200.0,
            expiry_date="2026-05-28",
            time_to_expiry_years=0.01918,
            warnings=[],
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "CONVERGED")
        self.assertEqual(d["quality_tier"], "TIER_1_HIGH_PRECISION")
        restored = SabrCalibrationResult.from_dict(d)
        self.assertEqual(restored.parameters, sabr)

    def test_surface_topological_features_container(self) -> None:
        """Verify SurfaceTopologicalFeatures instantiation and round-trip."""
        topo = SurfaceTopologicalFeatures(
            iv_skew_25d=0.028,
            iv_skew_10d=0.052,
            iv_curvature_25d=0.015,
            iv_term_slope_near_next=0.0085,
            surface_displacement_5m=0.004,
            surface_displacement_15m=0.009,
            surface_acceleration_15m=0.0015,
            vrp_proxy_30m=0.0022,
            atm_iv=0.148,
            forward_price=24520.0,
            as_of_timestamp=1782880200.0,
        )
        d = topo.to_dict()
        self.assertEqual(d["iv_skew_25d"], 0.028)
        self.assertEqual(d["surface_acceleration_15m"], 0.0015)
        restored = SurfaceTopologicalFeatures.from_dict(d)
        self.assertEqual(restored, topo)

    def test_feature_provenance_record_and_hashing(self) -> None:
        """Verify FeatureProvenanceRecord creation, JSON serialization, and cryptographic hashing."""
        prov = FeatureProvenanceRecord(
            feature_name="vanna_atm",
            mathematical_family=MathematicalFamily.HIGHER_ORDER_GREEKS,
            formula_expression="-phi(d1) * (d2 / sigma)",
            source_fields=["underlying_spot", "strike", "implied_volatility", "time_to_expiry_years", "risk_free_rate"],
            source_snapshot_hash="snap_abc123",
            calibration_rmse=None,
            calibration_status=None,
            calibration_tier=None,
            units="ratio",
            implementation_version="1.1.0",
        )
        h1 = prov.compute_provenance_hash()
        h2 = prov.compute_provenance_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

        d = prov.to_dict()
        self.assertEqual(d["mathematical_family"], "HIGHER_ORDER_GREEKS")
        restored = FeatureProvenanceRecord.from_dict(d)
        self.assertEqual(restored.feature_name, "vanna_atm")
        self.assertEqual(restored.mathematical_family, MathematicalFamily.HIGHER_ORDER_GREEKS)

    def test_surface_math_config_defaults_and_customization(self) -> None:
        """Verify SurfaceMathConfig defaults (beta=0.5, 5m calibration grid) and customization."""
        # 1. Test canonical defaults
        cfg = DEFAULT_SURFACE_MATH_CONFIG
        self.assertEqual(cfg.sabr_beta, 0.5)
        self.assertEqual(cfg.sabr_beta_mode, SabrBetaMode.CIR_SQUARE_ROOT)
        self.assertEqual(cfg.calibration_interval_minutes, 5)
        self.assertEqual(cfg.min_liquid_strikes, 5)
        self.assertEqual(cfg.tier1_rmse_threshold, 0.03)
        self.assertEqual(cfg.tier2_rmse_threshold, 0.06)
        self.assertEqual(cfg.max_cpu_workers, 4)

        # Hash determinism
        hash1 = cfg.compute_config_hash()
        hash2 = cfg.compute_config_hash()
        self.assertEqual(hash1, hash2)

        # 2. Test customization to lognormal beta = 1.0
        custom_cfg = SurfaceMathConfig(
            sabr_beta=1.0,
            sabr_beta_mode=SabrBetaMode.LOGNORMAL,
            calibration_interval_minutes=15,
        )
        self.assertEqual(custom_cfg.sabr_beta, 1.0)
        self.assertEqual(custom_cfg.sabr_beta_mode, SabrBetaMode.LOGNORMAL)
        self.assertEqual(custom_cfg.calibration_interval_minutes, 15)

        # Round-trip serialization
        d = custom_cfg.to_dict()
        self.assertEqual(d["sabr_beta"], 1.0)
        self.assertEqual(d["sabr_beta_mode"], "LOGNORMAL")
        restored = SurfaceMathConfig.from_dict(d)
        self.assertEqual(restored, custom_cfg)


if __name__ == "__main__":
    unittest.main()
