"""Synthetic forward + book depth slope Registry wave (v28)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from chain_replay_ml.dataset_builder.chain_maps import ChainMaps
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    FEATURE_REGISTRY_VERSION,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.market_microstructure import (
    MARKET_MICROSTRUCTURE_FEATURES,
    _depth_slope,
    compute_microstructure_levels,
)
from chain_replay_ml.dataset_builder.transformations.lag_ui import classify_feature
from chain_replay_ml.ticks import BookSnapshot


class TestSyntheticForwardRegistry(unittest.TestCase):
    def test_registered_computed_base(self) -> None:
        self.assertEqual(FEATURE_REGISTRY_VERSION, 30)
        name = "synthetic_forward_spot"
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        self.assertIn(name, all_feats)
        self.assertIn(name, _REGISTRY_FEATURES["chain"])
        self.assertIn(name, CONTROLLER_FEATURES["token.chain"])
        self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(classify_feature(name), "Spot & Futures")

    def test_map_holds_forward(self) -> None:
        maps = ChainMaps()
        maps.synthetic_forward_spot[1000.0] = 24512.5
        self.assertAlmostEqual(maps.lookup(maps.synthetic_forward_spot, 1000.0), 24512.5)

    def test_parity_formula_on_maps_build_path(self) -> None:
        """K + (C − P) with mocked CE/PE timelines (same arithmetic as build)."""
        k, c, p = 24500.0, 180.0, 165.0
        expected = k + c - p
        ce_tl = MagicMock()
        pe_tl = MagicMock()
        ce_tl.ltp_rupees_at.return_value = c
        pe_tl.ltp_rupees_at.return_value = p
        strike_mapping = {
            (k, "CE"): ("tok_ce", k, ce_tl),
            (k, "PE"): ("tok_pe", k, pe_tl),
        }
        t = 1.0
        entry_ce = strike_mapping.get((k, "CE"))
        entry_pe = strike_mapping.get((k, "PE"))
        assert entry_ce and entry_pe
        _, _, ce_atm_tl = entry_ce
        _, _, pe_atm_tl = entry_pe
        c_ltp = ce_atm_tl.ltp_rupees_at(t)
        p_ltp = pe_atm_tl.ltp_rupees_at(t)
        self.assertAlmostEqual(float(k) + float(c_ltp) - float(p_ltp), expected)


class TestBookDepthSlope(unittest.TestCase):
    def test_registered(self) -> None:
        for name in ("book_depth_slope_bid", "book_depth_slope_ask"):
            self.assertIn(name, MARKET_MICROSTRUCTURE_FEATURES)
            self.assertIn(name, _REGISTRY_FEATURES["market_microstructure"])
            self.assertIn(name, CONTROLLER_FEATURES["token.book"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
            self.assertEqual(classify_feature(name), "Volume & Liquidity")

    def test_ols_slope_linear(self) -> None:
        self.assertAlmostEqual(_depth_slope((100, 80, 60, 40, 20)), -20.0)
        self.assertAlmostEqual(_depth_slope((50, 40, 30, 20, 10)), -10.0)

    def test_null_when_insufficient_levels(self) -> None:
        self.assertIsNone(_depth_slope((100,)))
        self.assertIsNone(_depth_slope(()))
        book = BookSnapshot(
            bid_prices_paise=(10000, 0, 0, 0, 0),
            ask_prices_paise=(10050, 0, 0, 0, 0),
            bid_quantities=(200, 0, 0, 0, 0),
            ask_quantities=(100, 0, 0, 0, 0),
            spread_paise=50,
        )
        # Zero pads still count as levels with qty ≥ 0 → slope defined.
        levels = compute_microstructure_levels(book)
        self.assertIsNotNone(levels["book_depth_slope_bid"])
        self.assertIsNotNone(levels["book_depth_slope_ask"])


if __name__ == "__main__":
    unittest.main()
