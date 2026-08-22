"""Focused unit & numerical validation tests for Phase 4A.4 surface topology and dynamic derivatives."""

import math
import numpy as np
import unittest

from chain_replay_ml.surface_math.surface import (
    SurfaceDynamicsEngine,
    SurfaceTopologyEngine,
    SurfaceTopologyEvaluation,
    TopologySource,
    find_strike_for_delta,
)
from chain_replay_ml.surface_math.types import (
    CalibrationQualityTier,
    CalibrationStatus,
    SviCalibrationResult,
    SviParameters,
)


class TestSurfaceTopologyEngine(unittest.TestCase):
    """Rigorous tests for cross-sectional skews, smile butterfly curvature, and term slopes."""

    def setUp(self) -> None:
        self.engine = SurfaceTopologyEngine()
        self.spot = 24500.0
        self.r = 0.07
        self.t_exp = 14.0 / 365.0
        self.forward = self.spot * math.exp(self.r * self.t_exp)

    def test_flat_surface_topology(self) -> None:
        """Verify that a flat volatility smile yields zero skew and zero smile curvature."""
        strikes = np.linspace(23000.0, 26000.0, 31)
        flat_ivs = np.full_like(strikes, 0.15)

        eval_res = self.engine.evaluate_cross_sectional_topology(
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            strikes=strikes,
            implied_volatilities=flat_ivs,
        )

        self.assertTrue(eval_res.is_valid)
        self.assertIsNotNone(eval_res.features.iv_skew_25d)
        self.assertAlmostEqual(eval_res.features.iv_skew_25d, 0.0, places=4)
        self.assertAlmostEqual(eval_res.features.iv_curvature_25d, 0.0, places=4)

    def test_downside_skew_and_convex_curvature(self) -> None:
        """Verify positive skew and positive smile butterfly for realistic equity smile."""
        # High quality Tier 1 SVI parameters with negative skew rho = -0.45
        svi_params = SviParameters(a=0.002, b=0.08, rho=-0.45, m=-0.015, sigma=0.06)
        svi_res = SviCalibrationResult(
            parameters=svi_params,
            status=CalibrationStatus.CONVERGED,
            quality_tier=CalibrationQualityTier.TIER_1_HIGH_PRECISION,
            rmse=0.012,
            mae=0.009,
            max_error=0.02,
            strikes_used=25,
            optimization_iterations=50,
            as_of_timestamp=1782880200.0,
            expiry_date="2026-05-28",
            time_to_expiry_years=self.t_exp,
        )

        eval_res = self.engine.evaluate_cross_sectional_topology(
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            svi_result=svi_res,
        )

        self.assertEqual(eval_res.source, TopologySource.PARAMETRIC_SVI)
        self.assertEqual(eval_res.quality_tier, CalibrationQualityTier.TIER_1_HIGH_PRECISION)
        self.assertTrue(eval_res.is_valid)
        
        # In negative rho SVI (equity put skew), Put IV > Call IV => positive skew_25d
        self.assertGreater(eval_res.features.iv_skew_25d, 0.0)
        # Tail 10d skew should exceed 25d skew
        self.assertGreater(eval_res.features.iv_skew_10d, eval_res.features.iv_skew_25d)
        # Smile curvature (butterfly) is positive
        self.assertGreater(eval_res.features.iv_curvature_25d, 0.0)

    def test_quality_gating_tier3_rejection(self) -> None:
        """Verify that Tier 3 SVI calibration is rejected and falls back safely."""
        # Tier 3 SVI result
        bad_svi = SviCalibrationResult(
            parameters=SviParameters(a=0.002, b=0.08, rho=-0.45, m=-0.015, sigma=0.06),
            status=CalibrationStatus.CALIB_WARNING,
            quality_tier=CalibrationQualityTier.TIER_3_FAILED,
            rmse=0.145,  # High error
            mae=0.110,
            max_error=0.25,
            strikes_used=25,
            optimization_iterations=100,
            as_of_timestamp=1782880200.0,
            expiry_date="2026-05-28",
            time_to_expiry_years=self.t_exp,
        )

        # Without empirical chain fallback
        eval_res_no_chain = self.engine.evaluate_cross_sectional_topology(
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            svi_result=bad_svi,
        )
        self.assertFalse(eval_res_no_chain.is_valid)
        self.assertEqual(eval_res_no_chain.source, TopologySource.UNAVAILABLE)
        self.assertTrue(any("SVI_CALIBRATION_REJECTED" in f for f in eval_res_no_chain.quality_flags))

        # With empirical chain fallback
        strikes = np.linspace(23000.0, 26000.0, 25)
        ivs = np.linspace(0.18, 0.13, 25)  # Downward sloping
        eval_res_with_chain = self.engine.evaluate_cross_sectional_topology(
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            strikes=strikes,
            implied_volatilities=ivs,
            svi_result=bad_svi,
        )
        self.assertEqual(eval_res_with_chain.source, TopologySource.EMPIRICAL_CHAIN)
        self.assertTrue(eval_res_with_chain.is_valid)
        self.assertTrue(any("SVI_CALIBRATION_REJECTED" in f for f in eval_res_with_chain.quality_flags))
        self.assertTrue(any("EMPIRICAL_CHAIN_EVALUATION" in f for f in eval_res_with_chain.quality_flags))

    def test_term_structure_slope(self) -> None:
        """Verify contango, backwardation, and missing next expiry handling for term slope."""
        # 1. Contango: IV rises from 14% at 7d to 16% at 30d
        t1, t2 = 7.0 / 365.0, 30.0 / 365.0
        slope_contango = self.engine.calculate_term_structure_slope(
            near_expiry_iv=0.14, near_expiry_years=t1, next_expiry_iv=0.16, next_expiry_years=t2
        )
        self.assertIsNotNone(slope_contango)
        self.assertGreater(slope_contango, 0.0)

        # 2. Backwardation: IV falls from 22% at 7d to 18% at 30d
        slope_back = self.engine.calculate_term_structure_slope(
            near_expiry_iv=0.22, near_expiry_years=t1, next_expiry_iv=0.18, next_expiry_years=t2
        )
        self.assertIsNotNone(slope_back)
        self.assertLess(slope_back, 0.0)

        # 3. Invalid / missing next expiry
        slope_invalid = self.engine.calculate_term_structure_slope(
            near_expiry_iv=0.14, near_expiry_years=t1, next_expiry_iv=0.14, next_expiry_years=t1
        )
        self.assertIsNone(slope_invalid)


class TestSurfaceDynamicsEngine(unittest.TestCase):
    """Rigorous tests for backward-looking volatility dynamics and Variance Risk Premium."""

    def setUp(self) -> None:
        self.dynamics = SurfaceDynamicsEngine(max_lookback_gap_seconds=120.0)

    def test_surface_displacement_and_acceleration(self) -> None:
        """Verify velocity and acceleration with timestamp-aligned historical observations."""
        now_ts = 1782881800.0  # T
        # History snapshots at T-30m (1782880000), T-15m (1782880900), T-5m (1782881500)
        history = [
            (1782880000.0, 0.140),  # T-30m (IV = 14.0%)
            (1782880900.0, 0.145),  # T-15m (IV = 14.5%)
            (1782881500.0, 0.148),  # T-5m  (IV = 14.8%)
        ]
        now_iv = 0.152  # Current IV = 15.2%

        disp_5m = self.dynamics.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, target_lag_seconds=300.0
        )
        # Expected disp_5m = 0.152 - 0.148 = +0.004 (+0.4% vol)
        self.assertIsNotNone(disp_5m)
        self.assertAlmostEqual(disp_5m, 0.004, places=4)

        disp_15m = self.dynamics.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, target_lag_seconds=900.0
        )
        # Expected disp_15m = 0.152 - 0.145 = +0.007 (+0.7% vol)
        self.assertIsNotNone(disp_15m)
        self.assertAlmostEqual(disp_15m, 0.007, places=4)

        accel_15m = self.dynamics.compute_surface_acceleration(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, lag_seconds=900.0
        )
        # Accel = IV(T) - 2*IV(T-15m) + IV(T-30m) = 0.152 - 2*(0.145) + 0.140 = 0.152 - 0.290 + 0.140 = +0.002
        self.assertIsNotNone(accel_15m)
        self.assertAlmostEqual(accel_15m, 0.002, places=4)

    def test_lookback_gap_rejection(self) -> None:
        """Verify that data gaps exceeding max_lookback_gap_seconds cleanly return None."""
        now_ts = 1782881800.0
        # Nearest snapshot to T-300s is 10 minutes away (600s gap > 120s tolerance)
        history = [(1782881000.0, 0.140)]

        disp = self.dynamics.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=0.150, history=history, target_lag_seconds=300.0
        )
        self.assertIsNone(disp)

    def test_leakage_audit_strictly_backward_looking(self) -> None:
        """Prove that future timestamps > current_timestamp are strictly ignored."""
        now_ts = 1782881800.0
        history = [
            (1782881500.0, 0.145),  # Past T-5m
            (1782882100.0, 0.999),  # Future T+5m (must be ignored)
            (1782883000.0, 0.999),  # Future T+20m (must be ignored)
        ]
        now_iv = 0.150

        disp = self.dynamics.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, target_lag_seconds=300.0
        )
        # Should compare 0.150 against past 0.145, NOT future 0.999
        self.assertIsNotNone(disp)
        self.assertAlmostEqual(disp, 0.005, places=4)

    def test_vrp_proxy_30m(self) -> None:
        """Verify Variance Risk Premium calculation: IV^2 - RV_30m^2."""
        now_ts = 1782881800.0
        # 30-minute spot history with small constant drift
        spot_history = [(now_ts - 60.0 * i, 24500.0 * (1.0 + 0.0001 * i)) for i in reversed(range(31))]

        vrp = self.dynamics.compute_vrp_proxy(
            current_atm_iv=0.15,  # IV = 15% -> IV^2 = 0.0225
            spot_price_history_30m=spot_history,
            current_timestamp=now_ts,
        )
        self.assertIsNotNone(vrp)
        # VRP should be positive for calm market (implied variance exceeds realized variance)
        self.assertGreater(vrp, 0.0)


if __name__ == "__main__":
    unittest.main()
