"""Charm / Speed / IV Butterfly 25Δ / Gamma Flip Registry wave (v27)."""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

from chain_replay_ml import bs
from chain_replay_ml.dataset_builder.chain_gex import (
    _gamma_flip_spot_from_net_by_strike,
    compute_chain_gex_at,
)
from chain_replay_ml.dataset_builder.chain_iv_skew import compute_chain_iv_skew_at
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    FEATURE_REGISTRY_VERSION,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.transformations.lag_ui import classify_feature
from chain_replay_ml.ticks import TickTimeline


class TestCharmSpeed(unittest.TestCase):
    def test_registry_computed_base(self) -> None:
        self.assertEqual(FEATURE_REGISTRY_VERSION, 30)
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in ("charm", "speed"):
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["greeks"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
            self.assertEqual(classify_feature(name), "Greeks")

    def test_greeks_keys_and_zeros(self) -> None:
        empty = bs.greeks("CE", 0.0, 100.0, 0.07, 0.1, 0.2)
        for key in ("charm", "speed", "vanna", "volga"):
            self.assertIn(key, empty)
            self.assertEqual(empty[key], 0.0)

    def test_formulas_match_bs(self) -> None:
        s, k, r, t, sigma = 24500.0, 24500.0, 0.07, 7.0 / 365.0, 0.18
        for opt in ("CE", "PE"):
            g = bs.greeks(opt, s, k, r, t, sigma)
            sqrt_t = math.sqrt(t)
            d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t
            pdf = bs.norm_pdf(d1)
            gamma = pdf / (s * sigma * sqrt_t)
            charm_annual = -pdf * (2.0 * r * t - d2 * sigma * sqrt_t) / (
                2.0 * t * sigma * sqrt_t
            )
            self.assertAlmostEqual(g["charm"], charm_annual / 365.0, places=10)
            self.assertAlmostEqual(
                g["speed"], -gamma / s * (d1 / (sigma * sqrt_t) + 1.0), places=10
            )
            other = "PE" if opt == "CE" else "CE"
            g2 = bs.greeks(other, s, k, r, t, sigma)
            self.assertAlmostEqual(g["charm"], g2["charm"], places=10)
            self.assertAlmostEqual(g["speed"], g2["speed"], places=10)

    def test_finite_difference_speed(self) -> None:
        s, k, r, t, sigma = 24500.0, 24600.0, 0.07, 14.0 / 365.0, 0.20
        eps = 1.0  # ₹1 spot bump
        g_up = bs.greeks("CE", s + eps, k, r, t, sigma)["gamma"]
        g_dn = bs.greeks("CE", s - eps, k, r, t, sigma)["gamma"]
        speed_fd = (g_up - g_dn) / (2.0 * eps)
        speed = bs.greeks("CE", s, k, r, t, sigma)["speed"]
        self.assertAlmostEqual(speed, speed_fd, places=6)


class TestIvButterfly25d(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertIn("iv_butterfly_25d", _REGISTRY_FEATURES["chain"])
        self.assertEqual(ownership_of("iv_butterfly_25d"), OWNERSHIP_COMPUTED_BASE)
        self.assertIn("iv_butterfly_25d", CONTROLLER_FEATURES["token.chain"])

    def test_formula_from_25d_and_atm(self) -> None:
        # Reuse the Wave B skew fixture pattern with synthetic timelines.
        def _tl(ltp: float, oi: int = 100) -> TickTimeline:
            tl = TickTimeline()
            tl.append(1000.0, int(ltp * 100), volume=10, oi=oi)
            return tl

        # Build a small chain: ATM + 25Δ-ish wings via mocked IV path is heavy;
        # unit-test the butterfly arithmetic through compute with patched IVs.
        from chain_replay_ml.dataset_builder import chain_iv_skew as mod

        strike_mapping = {
            (24500.0, "CE"): ("1", "CE", _tl(150)),
            (24500.0, "PE"): ("2", "PE", _tl(140)),
            (24600.0, "CE"): ("3", "CE", _tl(90)),
            (24400.0, "PE"): ("4", "PE", _tl(95)),
        }
        orig = mod._option_iv_delta_at
        orig_iv = mod._option_iv_at

        def fake_iv_delta(*, strike, option_type, **kwargs):
            if option_type == "CE" and strike == 24600.0:
                return 0.12, 0.25
            if option_type == "PE" and strike == 24400.0:
                return 0.18, -0.25
            if option_type == "CE" and strike == 24500.0:
                return 0.14, 0.50
            if option_type == "PE" and strike == 24500.0:
                return 0.15, -0.50
            return None, None

        def fake_iv(*, strike, option_type, **kwargs):
            iv, _ = fake_iv_delta(strike=strike, option_type=option_type)
            return iv

        mod._option_iv_delta_at = lambda *a, **k: fake_iv_delta(**k)  # type: ignore
        mod._option_iv_at = lambda *a, **k: fake_iv(**k)  # type: ignore
        try:
            out = compute_chain_iv_skew_at(
                strike_mapping,
                ts=1000.0,
                spot=24500.0,
                atm_strike=24500.0,
                strike_step=100,
                expiry_ts=1000.0 + 7 * 86400,
            )
        finally:
            mod._option_iv_delta_at = orig  # type: ignore
            mod._option_iv_at = orig_iv  # type: ignore

        self.assertAlmostEqual(out["iv_skew_25d"], 0.18 - 0.12)
        atm = 0.5 * (0.14 + 0.15)
        self.assertAlmostEqual(out["iv_butterfly_25d"], 0.5 * (0.12 + 0.18) - atm)


class TestGammaFlip(unittest.TestCase):
    def test_registry(self) -> None:
        for name in ("gamma_flip_spot", "gamma_flip_distance"):
            self.assertIn(name, _REGISTRY_FEATURES["chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])

    def test_zero_crossing_interpolation(self) -> None:
        # cum: 24400→+10, 24500→+10-12=-2 → zero between 24400 and 24500
        flip = _gamma_flip_spot_from_net_by_strike({
            24400.0: 10.0,
            24500.0: -12.0,
            24600.0: -5.0,
        })
        self.assertIsNotNone(flip)
        # After 24400 cum=+10; after 24500 cum=-2 → zero at 10/(10+2) along the segment.
        self.assertAlmostEqual(float(flip), 24400.0 + 100.0 * (10.0 / 12.0), places=6)

    def test_distance_formula(self) -> None:
        index_tl = TickTimeline()
        index_tl.append(1000.0, 2_450_000, volume=1, oi=0)
        # Empty chain → no flip
        out = compute_chain_gex_at({}, index_tl=index_tl, ts=1000.0, expiry_ts=2000.0)
        self.assertIsNone(out["gamma_flip_spot"])
        self.assertIsNone(out["gamma_flip_distance"])


if __name__ == "__main__":
    unittest.main()
