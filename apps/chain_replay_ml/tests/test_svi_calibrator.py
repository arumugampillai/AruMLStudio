"""Focused unit & numerical validation tests for Phase 4A.2 Raw SVI surface calibrator."""

import math
import numpy as np
import unittest

from chain_replay_ml.surface_math.svi import (
    SviCalibrator,
    compute_log_moneyness,
    evaluate_raw_svi,
    evaluate_svi_implied_volatility,
)
from chain_replay_ml.surface_math.types import (
    CalibrationQualityTier,
    CalibrationStatus,
    SviCalibrationResult,
    SviParameters,
)


class TestSviCalibrator(unittest.TestCase):
    """Rigorous mathematical tests for Zeliade quasi-explicit SVI calibration."""

    def setUp(self) -> None:
        self.calibrator = SviCalibrator()
        self.spot = 24500.0
        self.r = 0.07
        self.t_exp = 14.0 / 365.0  # 14 days
        self.forward = self.spot * math.exp(self.r * self.t_exp)

        # True benchmark SVI parameters for NIFTY
        self.true_params = SviParameters(
            a=0.002,
            b=0.08,
            rho=-0.45,
            m=-0.015,
            sigma=0.06,
        )

    def test_synthetic_exact_reconstruction(self) -> None:
        """Verify exact SVI reconstruction when true parameters generate observations."""
        strikes = np.linspace(23000.0, 26000.0, 25)
        log_k = np.log(strikes / self.forward)
        true_w = evaluate_raw_svi(log_k, self.true_params)
        true_iv = np.sqrt(true_w / self.t_exp)

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=true_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
        )

        self.assertEqual(res.status, CalibrationStatus.CONVERGED)
        self.assertEqual(res.quality_tier, CalibrationQualityTier.TIER_1_HIGH_PRECISION)
        self.assertLess(res.rmse, 0.005)
        self.assertLess(res.mae, 0.005)

        # Check parameter recovery
        self.assertAlmostEqual(res.parameters.a, self.true_params.a, places=2)
        self.assertAlmostEqual(res.parameters.b, self.true_params.b, places=2)
        self.assertAlmostEqual(res.parameters.rho, self.true_params.rho, places=1)
        self.assertAlmostEqual(res.parameters.m, self.true_params.m, places=2)
        self.assertAlmostEqual(res.parameters.sigma, self.true_params.sigma, places=2)

    def test_noisy_observation_fit_tier(self) -> None:
        """Verify calibration under synthetic market microstructure noise."""
        np.random.seed(42)
        strikes = np.linspace(23000.0, 26000.0, 31)
        log_k = np.log(strikes / self.forward)
        true_w = evaluate_raw_svi(log_k, self.true_params)
        true_iv = np.sqrt(true_w / self.t_exp)
        
        # Add 0.5% (0.005) gaussian noise to IV
        noise = np.random.normal(0.0, 0.005, size=len(strikes))
        noisy_iv = np.maximum(0.05, true_iv + noise)

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=noisy_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
        )

        self.assertIn(res.quality_tier, [CalibrationQualityTier.TIER_1_HIGH_PRECISION, CalibrationQualityTier.TIER_2_ACCEPTABLE])
        self.assertLess(res.rmse, 0.03)
        self.assertGreater(res.parameters.b, 0.0)
        self.assertLess(abs(res.parameters.rho), 1.0)
        self.assertGreater(res.parameters.sigma, 0.0)

    def test_insufficient_strikes_path_b(self) -> None:
        """Verify that < 5 strikes returns INSUFFICIENT_DATA and TIER_3_FAILED without raising."""
        # Only 3 strikes available (ATM band slice)
        sparse_strikes = [24400.0, 24500.0, 24600.0]
        sparse_iv = [0.155, 0.150, 0.148]

        res = self.calibrator.calibrate_slice(
            strikes=sparse_strikes,
            implied_volatilities=sparse_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
        )

        self.assertEqual(res.status, CalibrationStatus.INSUFFICIENT_DATA)
        self.assertEqual(res.quality_tier, CalibrationQualityTier.TIER_3_FAILED)
        self.assertEqual(res.strikes_used, 3)
        self.assertEqual(res.parameters.b, 0.0)
        self.assertTrue(any("Insufficient liquid strikes" in w for w in res.warnings))

    def test_duplicate_strikes_handling(self) -> None:
        """Verify that duplicate strikes are cleanly averaged."""
        strikes = [24000.0, 24000.0, 24500.0, 24500.0, 25000.0, 25500.0, 26000.0]
        ivs = [0.18, 0.182, 0.15, 0.152, 0.14, 0.145, 0.155]

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=ivs,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
        )
        self.assertEqual(res.strikes_used, 5)  # 5 unique strikes
        self.assertLess(res.rmse, 0.05)

    def test_nan_and_negative_inputs_safety(self) -> None:
        """Verify graceful error containment for NaN, Inf, and non-positive inputs."""
        # 1. Non-positive spot
        res_zero_s = self.calibrator.calibrate_slice(
            strikes=[24000.0, 24500.0, 25000.0, 25500.0, 26000.0],
            implied_volatilities=[0.16, 0.15, 0.14, 0.145, 0.15],
            underlying_spot=0.0,
            time_to_expiry_years=0.02,
        )
        self.assertEqual(res_zero_s.status, CalibrationStatus.INSUFFICIENT_DATA)

        # 2. NaN IVs
        res_nan = self.calibrator.calibrate_slice(
            strikes=[24000.0, 24500.0, 25000.0, 25500.0, 26000.0],
            implied_volatilities=[0.16, float("nan"), 0.14, 0.145, 0.15],
            underlying_spot=24500.0,
            time_to_expiry_years=0.02,
        )
        # NaN is filtered out, leaving 4 strikes -> Insufficient data (< 5)
        self.assertEqual(res_nan.status, CalibrationStatus.INSUFFICIENT_DATA)

    def test_deterministic_repeated_calibration(self) -> None:
        """Verify calibration is 100% deterministic given identical inputs."""
        strikes = np.linspace(23500.0, 25500.0, 15)
        log_k = np.log(strikes / self.forward)
        w = evaluate_raw_svi(log_k, self.true_params)
        iv = np.sqrt(w / self.t_exp)

        res1 = self.calibrator.calibrate_slice(
            strikes=strikes, implied_volatilities=iv, underlying_spot=self.spot, time_to_expiry_years=self.t_exp
        )
        res2 = self.calibrator.calibrate_slice(
            strikes=strikes, implied_volatilities=iv, underlying_spot=self.spot, time_to_expiry_years=self.t_exp
        )

        self.assertEqual(res1.parameters, res2.parameters)
        self.assertEqual(res1.rmse, res2.rmse)


if __name__ == "__main__":
    unittest.main()
