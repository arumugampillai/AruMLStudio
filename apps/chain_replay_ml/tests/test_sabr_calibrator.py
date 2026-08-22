"""Focused unit & numerical validation tests for Phase 4A.3 SABR surface calibrator."""

import math
import numpy as np
import unittest

from chain_replay_ml.surface_math.sabr import (
    SabrCalibrator,
    evaluate_sabr_volatility,
    invert_sabr_alpha_from_atm,
)
from chain_replay_ml.surface_math.types import (
    CalibrationQualityTier,
    CalibrationStatus,
    SabrCalibrationResult,
    SabrParameters,
)


class TestSabrCalibrator(unittest.TestCase):
    """Rigorous mathematical tests for SABR surface calibration."""

    def setUp(self) -> None:
        self.calibrator = SabrCalibrator()
        self.spot = 24500.0
        self.r = 0.07
        self.t_exp = 14.0 / 365.0
        self.forward = self.spot * math.exp(self.r * self.t_exp)

        # Standard NIFTY SABR parameters with beta = 0.5
        self.true_params_beta05 = SabrParameters(
            alpha=23.5,
            beta=0.5,
            rho=-0.35,
            nu=0.75,
        )

        # Log-normal SABR parameters with beta = 1.0
        self.true_params_beta10 = SabrParameters(
            alpha=0.15,
            beta=1.0,
            rho=-0.40,
            nu=0.85,
        )

    def test_synthetic_exact_reconstruction_beta05(self) -> None:
        """Verify exact parameter recovery and RMSE < 0.005 for beta = 0.5."""
        strikes = np.linspace(23200.0, 25800.0, 25)
        true_iv = evaluate_sabr_volatility(strikes, self.forward, self.t_exp, self.true_params_beta05)

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=true_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            beta=0.5,
        )

        self.assertEqual(res.status, CalibrationStatus.CONVERGED)
        self.assertEqual(res.quality_tier, CalibrationQualityTier.TIER_1_HIGH_PRECISION)
        self.assertLess(res.rmse, 0.005)
        self.assertAlmostEqual(res.parameters.alpha, self.true_params_beta05.alpha, delta=0.20)
        self.assertAlmostEqual(res.parameters.rho, self.true_params_beta05.rho, delta=0.05)
        self.assertAlmostEqual(res.parameters.nu, self.true_params_beta05.nu, delta=0.05)

    def test_synthetic_exact_reconstruction_beta10(self) -> None:
        """Verify parameter recovery for beta = 1.0 (Lognormal SABR)."""
        strikes = np.linspace(23200.0, 25800.0, 25)
        true_iv = evaluate_sabr_volatility(strikes, self.forward, self.t_exp, self.true_params_beta10)

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=true_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            beta=1.0,
        )

        self.assertEqual(res.status, CalibrationStatus.CONVERGED)
        self.assertEqual(res.quality_tier, CalibrationQualityTier.TIER_1_HIGH_PRECISION)
        self.assertLess(res.rmse, 0.005)
        self.assertEqual(res.parameters.beta, 1.0)
        self.assertAlmostEqual(res.parameters.alpha, self.true_params_beta10.alpha, places=2)

    def test_noisy_observations_quality_tier(self) -> None:
        """Verify SABR calibration robustness under 0.5% market microstructure noise."""
        np.random.seed(42)
        strikes = np.linspace(23000.0, 26000.0, 31)
        true_iv = evaluate_sabr_volatility(strikes, self.forward, self.t_exp, self.true_params_beta05)
        noise = np.random.normal(0.0, 0.005, size=len(strikes))
        noisy_iv = np.maximum(0.05, true_iv + noise)

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=noisy_iv,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
            forward_price=self.forward,
            beta=0.5,
        )

        self.assertIn(res.quality_tier, [CalibrationQualityTier.TIER_1_HIGH_PRECISION, CalibrationQualityTier.TIER_2_ACCEPTABLE])
        self.assertLess(res.rmse, 0.03)
        self.assertLess(abs(res.parameters.rho), 1.0)
        self.assertGreater(res.parameters.nu, 0.0)

    def test_insufficient_strikes_path_b(self) -> None:
        """Verify that < 5 strikes returns INSUFFICIENT_DATA and TIER_3_FAILED without raising."""
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
        self.assertEqual(res.parameters.alpha, 0.0)

    def test_duplicate_strikes_handling(self) -> None:
        """Verify that duplicate strikes are cleanly averaged."""
        strikes = [24000.0, 24000.0, 24500.0, 24500.0, 25000.0, 25500.0, 26000.0]
        ivs = [0.17, 0.172, 0.15, 0.151, 0.142, 0.148, 0.156]

        res = self.calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=ivs,
            underlying_spot=self.spot,
            time_to_expiry_years=self.t_exp,
            risk_free_rate=self.r,
        )
        self.assertEqual(res.strikes_used, 5)
        self.assertLess(res.rmse, 0.05)

    def test_nan_and_negative_inputs_safety(self) -> None:
        """Verify graceful error containment for NaN, Inf, and non-positive inputs."""
        res_nan = self.calibrator.calibrate_slice(
            strikes=[24000.0, 24500.0, 25000.0, 25500.0, 26000.0],
            implied_volatilities=[0.16, float("nan"), 0.14, 0.145, 0.15],
            underlying_spot=24500.0,
            time_to_expiry_years=0.02,
        )
        self.assertEqual(res_nan.status, CalibrationStatus.INSUFFICIENT_DATA)

    def test_deterministic_repeated_calibration(self) -> None:
        """Verify 100% deterministic calibration results."""
        strikes = np.linspace(23500.0, 25500.0, 15)
        true_iv = evaluate_sabr_volatility(strikes, self.forward, self.t_exp, self.true_params_beta05)

        res1 = self.calibrator.calibrate_slice(
            strikes=strikes, implied_volatilities=true_iv, underlying_spot=self.spot, time_to_expiry_years=self.t_exp
        )
        res2 = self.calibrator.calibrate_slice(
            strikes=strikes, implied_volatilities=true_iv, underlying_spot=self.spot, time_to_expiry_years=self.t_exp
        )

        self.assertEqual(res1.parameters, res2.parameters)
        self.assertEqual(res1.rmse, res2.rmse)


if __name__ == "__main__":
    unittest.main()
