"""OI by abs(delta) band Registry wave (v30)."""

from __future__ import annotations

import unittest

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.dataset_builder.chain_oi_delta_bands import (
    OI_ABS_DELTA_BAND_FEATURES,
    _band_suffix,
    compute_oi_abs_delta_bands_at,
    needs_oi_abs_delta_bands,
)
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    FEATURE_REGISTRY_VERSION,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.ticks import TickTimeline


class TestOiAbsDeltaBands(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertEqual(FEATURE_REGISTRY_VERSION, 30)
        self.assertEqual(len(OI_ABS_DELTA_BAND_FEATURES), 10)
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in OI_ABS_DELTA_BAND_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["oi"])
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertTrue(needs_oi_abs_delta_bands(None))
        self.assertTrue(needs_oi_abs_delta_bands({"oi_abs_delta_40_60_ce"}))
        self.assertFalse(needs_oi_abs_delta_bands({"chain_pcr"}))

    def test_band_suffix_edges(self) -> None:
        self.assertEqual(_band_suffix(0.0), "0_20")
        self.assertEqual(_band_suffix(0.1999), "0_20")
        self.assertEqual(_band_suffix(0.2), "20_40")
        self.assertEqual(_band_suffix(0.4), "40_60")
        self.assertEqual(_band_suffix(0.6), "60_80")
        self.assertEqual(_band_suffix(0.8), "80_100")
        self.assertEqual(_band_suffix(1.0), "80_100")
        self.assertEqual(_band_suffix(1.05), "80_100")

    def test_buckets_split_ce_pe(self) -> None:
        """Two options with known deltas land in expected CE/PE bands."""
        spot = 24500.0
        t_exp = 7.0 / 365.0
        ts = 1_000_000.0
        expiry_ts = ts + 7 * 86400.0

        # ATM-ish call ≈ 0.5Δ, OTM put ≈ 0.25Δ (approx; assert via computed δ).
        ce_k, pe_k = 24500.0, 24200.0
        sigma = 0.18
        ce_delta = abs(
            bs.greeks("CE", spot, ce_k, RISK_FREE_RATE, t_exp, sigma)["delta"]
        )
        pe_delta = abs(
            bs.greeks("PE", spot, pe_k, RISK_FREE_RATE, t_exp, sigma)["delta"]
        )
        ce_band = _band_suffix(ce_delta)
        pe_band = _band_suffix(pe_delta)
        self.assertIsNotNone(ce_band)
        self.assertIsNotNone(pe_band)

        ce_ltp = bs.bs_price("CE", spot, ce_k, RISK_FREE_RATE, t_exp, sigma)
        pe_ltp = bs.bs_price("PE", spot, pe_k, RISK_FREE_RATE, t_exp, sigma)
        self.assertGreater(ce_ltp, 0)
        self.assertGreater(pe_ltp, 0)

        index_tl = TickTimeline()
        index_tl.append(ts, int(spot * 100))

        ce_tl = TickTimeline()
        ce_tl.append(ts, int(ce_ltp * 100), oi=1000)
        pe_tl = TickTimeline()
        pe_tl.append(ts, int(pe_ltp * 100), oi=2500)

        strike_mapping = {
            (ce_k, "CE"): ("ce", "CE", ce_tl),
            (pe_k, "PE"): ("pe", "PE", pe_tl),
        }
        out = compute_oi_abs_delta_bands_at(
            strike_mapping,
            index_tl=index_tl,
            ts=ts,
            expiry_ts=expiry_ts,
        )
        for name in OI_ABS_DELTA_BAND_FEATURES:
            self.assertIsNotNone(out[name])
        self.assertAlmostEqual(out[f"oi_abs_delta_{ce_band}_ce"], 1000.0)
        self.assertAlmostEqual(out[f"oi_abs_delta_{pe_band}_pe"], 2500.0)
        # Other CE/PE buckets empty.
        for name in OI_ABS_DELTA_BAND_FEATURES:
            if name.endswith("_ce") and name != f"oi_abs_delta_{ce_band}_ce":
                self.assertAlmostEqual(out[name], 0.0)
            if name.endswith("_pe") and name != f"oi_abs_delta_{pe_band}_pe":
                self.assertAlmostEqual(out[name], 0.0)


if __name__ == "__main__":
    unittest.main()
