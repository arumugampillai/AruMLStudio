"""Wave B: Vanna / Volga as Registry Base greeks (extend bs.greeks)."""

from __future__ import annotations

import math
import unittest

from chain_replay_ml import bs
from chain_replay_ml.dataset_builder.feature_ownership import OWNERSHIP_BASE, ownership_of
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.transformations.lag_ui import classify_feature


class TestVannaVolga(unittest.TestCase):
    def test_registry_base(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in ("vanna", "volga"):
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["greeks"])
            self.assertEqual(ownership_of(name), OWNERSHIP_BASE)
            self.assertEqual(classify_feature(name), "Greeks")

    def test_greeks_keys_and_zeros(self) -> None:
        empty = bs.greeks("CE", 0.0, 100.0, 0.07, 0.1, 0.2)
        for key in ("delta", "gamma", "theta", "vega", "vanna", "volga"):
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
            vega_expected = bs.bs_vega_raw(s, k, r, t, sigma) / 100.0
            self.assertAlmostEqual(g["vega"], vega_expected, places=10)
            self.assertAlmostEqual(g["vanna"], -pdf * d2 / sigma, places=10)
            self.assertAlmostEqual(g["volga"], vega_expected * d1 * d2 / sigma, places=10)
            # CE/PE share vanna and volga under BS (delta differs).
            other = "PE" if opt == "CE" else "CE"
            g2 = bs.greeks(other, s, k, r, t, sigma)
            self.assertAlmostEqual(g["vanna"], g2["vanna"], places=10)
            self.assertAlmostEqual(g["volga"], g2["volga"], places=10)

    def test_finite_difference_vanna(self) -> None:
        """Vanna ≈ ∂Δ/∂σ via central difference on sigma."""
        s, k, r, t, sigma = 24500.0, 24600.0, 0.07, 14.0 / 365.0, 0.20
        eps = 1e-5
        d_up = bs.greeks("CE", s, k, r, t, sigma + eps)["delta"]
        d_dn = bs.greeks("CE", s, k, r, t, sigma - eps)["delta"]
        vanna_fd = (d_up - d_dn) / (2.0 * eps)
        vanna = bs.greeks("CE", s, k, r, t, sigma)["vanna"]
        self.assertAlmostEqual(vanna, vanna_fd, places=4)

    def test_finite_difference_volga(self) -> None:
        """Volga ≈ ∂vega/∂σ via central difference (vega = raw/100)."""
        s, k, r, t, sigma = 24500.0, 24400.0, 0.07, 21.0 / 365.0, 0.22
        eps = 1e-5
        v_up = bs.greeks("PE", s, k, r, t, sigma + eps)["vega"]
        v_dn = bs.greeks("PE", s, k, r, t, sigma - eps)["vega"]
        volga_fd = (v_up - v_dn) / (2.0 * eps)
        volga = bs.greeks("PE", s, k, r, t, sigma)["volga"]
        self.assertAlmostEqual(volga, volga_fd, places=4)


if __name__ == "__main__":
    unittest.main()
