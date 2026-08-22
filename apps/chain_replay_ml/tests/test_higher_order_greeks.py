"""Focused unit & numerical validation tests for Phase 4A.1 analytical higher-order Greeks."""

import math
import numpy as np
import unittest

from chain_replay_ml import bs
from chain_replay_ml.surface_math.greeks import (
    calculate_charm,
    calculate_color,
    calculate_d1_d2,
    calculate_higher_order_greeks,
    calculate_higher_order_greeks_vectorized,
    calculate_speed,
    calculate_ultima,
    calculate_vanna,
    calculate_volga,
    calculate_zomma,
)
from chain_replay_ml.surface_math.types import HigherOrderGreeksRecord


class TestHigherOrderGreeksEngine(unittest.TestCase):
    """Rigorous analytical, finite-difference, and unit scaling validation for Greeks."""

    def setUp(self) -> None:
        # Standard NIFTY 50 baseline parameters
        self.spot = 24500.0
        self.strike_atm = 24500.0
        self.strike_itm_call = 24000.0
        self.strike_otm_call = 25000.0
        self.rate = 0.07
        self.expiry_7d = 7.0 / 365.0
        self.expiry_30d = 30.0 / 365.0
        self.vol = 0.15
        self.eps = 1e-6

    # =========================================================================
    # 1. PARITY & SCALING TESTS AGAINST BS.PY
    # =========================================================================

    def test_parity_with_existing_bs_module(self) -> None:
        """Verify that base Greeks and common higher-order Greeks match bs.py exactly."""
        # 1. Call ATM
        bs_res = bs.greeks("CE", self.spot, self.strike_atm, self.rate, self.expiry_7d, self.vol)
        eng_res = calculate_higher_order_greeks(
            option_type="CE",
            underlying_spot=self.spot,
            strike=self.strike_atm,
            risk_free_rate=self.rate,
            time_to_expiry_years=self.expiry_7d,
            implied_volatility=self.vol,
        )

        self.assertAlmostEqual(eng_res.delta, bs_res["delta"], places=7)
        self.assertAlmostEqual(eng_res.gamma, bs_res["gamma"], places=9)
        self.assertAlmostEqual(eng_res.theta, bs_res["theta"], places=7)
        self.assertAlmostEqual(eng_res.vega, bs_res["vega"], places=7)
        self.assertAlmostEqual(eng_res.vanna, bs_res["vanna"], places=7)
        self.assertAlmostEqual(eng_res.volga, bs_res["volga"], places=7)
        self.assertAlmostEqual(eng_res.charm, bs_res["charm"], places=9)
        self.assertAlmostEqual(eng_res.speed, bs_res["speed"], places=11)

        # 2. Put ATM
        bs_put = bs.greeks("PE", self.spot, self.strike_atm, self.rate, self.expiry_7d, self.vol)
        eng_put = calculate_higher_order_greeks(
            option_type="PE",
            underlying_spot=self.spot,
            strike=self.strike_atm,
            risk_free_rate=self.rate,
            time_to_expiry_years=self.expiry_7d,
            implied_volatility=self.vol,
        )
        self.assertAlmostEqual(eng_put.delta, bs_put["delta"], places=7)
        self.assertAlmostEqual(eng_put.theta, bs_put["theta"], places=7)
        # Put and Call share identical Gamma, Vega, Vanna, Volga, Charm, Speed
        self.assertAlmostEqual(eng_put.vanna, eng_res.vanna, places=7)
        self.assertAlmostEqual(eng_put.volga, eng_res.volga, places=7)
        self.assertAlmostEqual(eng_put.charm, eng_res.charm, places=7)

    # =========================================================================
    # 2. FINITE-DIFFERENCE NUMERICAL VALIDATION (CENTRAL DIFFERENCES)
    # =========================================================================

    def test_vanna_finite_difference(self) -> None:
        """Verify Vanna = d(Delta)/d(sigma) = d(Vega_raw)/dS via central differences."""
        eps = self.eps
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_vanna = calculate_vanna(s, k, r, t, sig)

        # Numerical derivative of Delta wrt sigma
        d_plus = bs.greeks("CE", s, k, r, t, sig + eps)["delta"]
        d_minus = bs.greeks("CE", s, k, r, t, sig - eps)["delta"]
        num_vanna_vol = (d_plus - d_minus) / (2.0 * eps)

        rel_err = abs(ana_vanna - num_vanna_vol) / max(1e-6, abs(ana_vanna))
        self.assertLess(rel_err, 1e-4)

    def test_volga_finite_difference(self) -> None:
        """Verify Volga = d(Vega)/d(sigma) in emitted bs.py units."""
        eps = self.eps
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_volga = calculate_volga(s, k, r, t, sig)

        # Numerical derivative of Vega (per vol pt) wrt sigma
        v_plus = bs.greeks("CE", s, k, r, t, sig + eps)["vega"]
        v_minus = bs.greeks("CE", s, k, r, t, sig - eps)["vega"]
        num_volga = (v_plus - v_minus) / (2.0 * eps)

        rel_err = abs(ana_volga - num_volga) / max(1e-6, abs(ana_volga))
        self.assertLess(rel_err, 1e-4)

    def test_charm_finite_difference(self) -> None:
        """Verify Charm = d(Delta)/dt = -d(Delta)/dT (scaled per calendar day / 365)."""
        eps = 1e-7
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_charm = calculate_charm(s, k, r, t, sig)

        # Numerical derivative of Delta wrt time to expiry T (negative of d/dT)
        d_plus = bs.greeks("CE", s, k, r, t + eps, sig)["delta"]
        d_minus = bs.greeks("CE", s, k, r, t - eps, sig)["delta"]
        num_charm = -((d_plus - d_minus) / (2.0 * eps)) / 365.0

        rel_err = abs(ana_charm - num_charm) / max(1e-6, abs(ana_charm))
        self.assertLess(rel_err, 1e-4)

    def test_color_finite_difference(self) -> None:
        """Verify Color = d(Gamma)/dt = -d(Gamma)/dT (scaled per calendar day / 365) with corrected sign."""
        eps = 1e-7
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_color = calculate_color(s, k, r, t, sig)

        # Numerical derivative of Gamma wrt time to expiry T (negative of d/dT)
        g_plus = bs.greeks("CE", s, k, r, t + eps, sig)["gamma"]
        g_minus = bs.greeks("CE", s, k, r, t - eps, sig)["gamma"]
        num_color = -((g_plus - g_minus) / (2.0 * eps)) / 365.0

        # Verify positive Color for ATM gamma decay
        self.assertGreater(ana_color, 0.0)
        rel_err = abs(ana_color - num_color) / max(1e-6, abs(ana_color))
        self.assertLess(rel_err, 1e-4)

    def test_speed_finite_difference(self) -> None:
        """Verify Speed = d(Gamma)/dS via central differences."""
        eps = 1e-4
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_speed = calculate_speed(s, k, r, t, sig)

        g_plus = bs.greeks("CE", s + eps, k, r, t, sig)["gamma"]
        g_minus = bs.greeks("CE", s - eps, k, r, t, sig)["gamma"]
        num_speed = (g_plus - g_minus) / (2.0 * eps)

        rel_err = abs(ana_speed - num_speed) / max(1e-8, abs(ana_speed))
        self.assertLess(rel_err, 1e-4)

    def test_zomma_finite_difference(self) -> None:
        """Verify Zomma = d(Gamma)/d(sigma) via central differences."""
        eps = self.eps
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_zomma = calculate_zomma(s, k, r, t, sig)

        g_plus = bs.greeks("CE", s, k, r, t, sig + eps)["gamma"]
        g_minus = bs.greeks("CE", s, k, r, t, sig - eps)["gamma"]
        num_zomma = (g_plus - g_minus) / (2.0 * eps)

        rel_err = abs(ana_zomma - num_zomma) / max(1e-6, abs(ana_zomma))
        self.assertLess(rel_err, 1e-4)

    def test_ultima_finite_difference(self) -> None:
        """Verify Ultima = d(Volga)/d(sigma) via central differences."""
        eps = self.eps
        s, k, r, t, sig = self.spot, self.strike_atm, self.rate, self.expiry_30d, self.vol

        ana_ultima = calculate_ultima(s, k, r, t, sig)

        volga_plus = calculate_volga(s, k, r, t, sig + eps)
        volga_minus = calculate_volga(s, k, r, t, sig - eps)
        num_ultima = (volga_plus - volga_minus) / (2.0 * eps)

        rel_err = abs(ana_ultima - num_ultima) / max(1e-6, abs(ana_ultima))
        self.assertLess(rel_err, 1e-4)

    # =========================================================================
    # 3. ATM / ITM / OTM & NEAR-EXPIRY CASES
    # =========================================================================

    def test_moneyness_profiles(self) -> None:
        """Verify Greeks across ATM, deep ITM, and deep OTM strikes."""
        # 1. ATM Call
        g_atm = calculate_higher_order_greeks(
            underlying_spot=self.spot,
            strike=self.strike_atm,
            risk_free_rate=self.rate,
            time_to_expiry_years=self.expiry_7d,
            implied_volatility=self.vol,
        )
        # Gamma and Volga should peak near ATM
        self.assertGreater(g_atm.gamma, 0.0)
        self.assertGreater(g_atm.volga, 0.0)

        # 2. Deep ITM Call (K = 20000 << S = 24500)
        g_itm = calculate_higher_order_greeks(
            underlying_spot=self.spot,
            strike=20000.0,
            risk_free_rate=self.rate,
            time_to_expiry_years=self.expiry_7d,
            implied_volatility=self.vol,
        )
        self.assertAlmostEqual(g_itm.delta, 1.0, places=2)
        self.assertAlmostEqual(g_itm.gamma, 0.0, places=5)
        self.assertAlmostEqual(g_itm.vega, 0.0, places=3)
        self.assertAlmostEqual(g_itm.volga, 0.0, places=3)

        # 3. Deep OTM Call (K = 29000 >> S = 24500)
        g_otm = calculate_higher_order_greeks(
            underlying_spot=self.spot,
            strike=29000.0,
            risk_free_rate=self.rate,
            time_to_expiry_years=self.expiry_7d,
            implied_volatility=self.vol,
        )
        self.assertAlmostEqual(g_otm.delta, 0.0, places=3)
        self.assertAlmostEqual(g_otm.gamma, 0.0, places=5)

    def test_near_expiry_stability(self) -> None:
        """Verify numerical stability on expiry day (T = 0.05 / 365, ~30 mins to expiry)."""
        t_near = 0.05 / 365.0
        g_near = calculate_higher_order_greeks(
            underlying_spot=self.spot,
            strike=self.strike_atm,
            risk_free_rate=self.rate,
            time_to_expiry_years=t_near,
            implied_volatility=self.vol,
        )
        self.assertTrue(math.isfinite(g_near.gamma))
        self.assertTrue(math.isfinite(g_near.color))
        self.assertTrue(math.isfinite(g_near.speed))
        self.assertTrue(math.isfinite(g_near.zomma))
        self.assertTrue(math.isfinite(g_near.ultima))

    # =========================================================================
    # 4. NUMERICAL EDGE CASES & INVALID INPUTS
    # =========================================================================

    def test_invalid_and_zero_inputs(self) -> None:
        """Verify that non-positive spot, strike, T, or vol return safe zeros without raising."""
        # Non-positive spot
        g_zero_s = calculate_higher_order_greeks(
            underlying_spot=0.0, strike=24500.0, risk_free_rate=0.07, time_to_expiry_years=0.02, implied_volatility=0.15
        )
        self.assertEqual(g_zero_s.delta, 0.0)
        self.assertEqual(g_zero_s.vanna, 0.0)

        # Zero time to expiry
        g_zero_t = calculate_higher_order_greeks(
            underlying_spot=24500.0, strike=24500.0, risk_free_rate=0.07, time_to_expiry_years=0.0, implied_volatility=0.15
        )
        self.assertEqual(g_zero_t.gamma, 0.0)
        self.assertEqual(g_zero_t.volga, 0.0)

        # Zero volatility
        g_zero_vol = calculate_higher_order_greeks(
            underlying_spot=24500.0, strike=24500.0, risk_free_rate=0.07, time_to_expiry_years=0.02, implied_volatility=0.0
        )
        self.assertEqual(g_zero_vol.vega, 0.0)
        self.assertEqual(g_zero_vol.ultima, 0.0)

        # NaN input
        g_nan = calculate_higher_order_greeks(
            underlying_spot=float("nan"), strike=24500.0, risk_free_rate=0.07, time_to_expiry_years=0.02, implied_volatility=0.15
        )
        self.assertEqual(g_nan.delta, 0.0)

    # =========================================================================
    # 5. VECTORIZED ENGINE & SCALAR/VECTOR PARITY
    # =========================================================================

    def test_scalar_vector_parity(self) -> None:
        """Verify 100% numerical identity between scalar and vectorized Greeks implementations."""
        n_samples = 50
        spots = np.linspace(24000.0, 25000.0, n_samples)
        strikes = np.full(n_samples, 24500.0)
        times = np.linspace(1.0 / 365.0, 45.0 / 365.0, n_samples)
        vols = np.linspace(0.10, 0.35, n_samples)

        # Add invalid edge cases into array
        spots[0] = 0.0
        times[1] = -0.01
        vols[2] = float("nan")

        vec_res = calculate_higher_order_greeks_vectorized(
            underlying_spots=spots,
            strikes=strikes,
            risk_free_rate=0.07,
            times_to_expiry_years=times,
            implied_volatilities=vols,
            option_types="CE",
        )

        for i in range(n_samples):
            scalar_res = calculate_higher_order_greeks(
                option_type="CE",
                underlying_spot=spots[i],
                strike=strikes[i],
                risk_free_rate=0.07,
                time_to_expiry_years=times[i],
                implied_volatility=vols[i],
            )
            for k in ["delta", "gamma", "theta", "vega", "vanna", "volga", "charm", "color", "speed", "zomma", "ultima"]:
                v_val = vec_res[k][i]
                s_val = getattr(scalar_res, k)
                if math.isnan(s_val):
                    self.assertTrue(math.isnan(v_val))
                else:
                    self.assertAlmostEqual(v_val, s_val, places=8, msg=f"Mismatch for {k} at index {i}")

    def test_deterministic_reproducibility(self) -> None:
        """Verify deterministic calculation across repeated calls."""
        res1 = calculate_higher_order_greeks(
            underlying_spot=self.spot, strike=self.strike_atm, risk_free_rate=self.rate, time_to_expiry_years=self.expiry_7d, implied_volatility=self.vol
        )
        res2 = calculate_higher_order_greeks(
            underlying_spot=self.spot, strike=self.strike_atm, risk_free_rate=self.rate, time_to_expiry_years=self.expiry_7d, implied_volatility=self.vol
        )
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
