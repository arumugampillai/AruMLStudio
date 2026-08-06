"""Wave B: chain GEX levels."""

from __future__ import annotations

import unittest
from unittest import mock

from chain_replay_ml.dataset_builder.chain_gex import (
    CHAIN_GEX_FEATURES,
    compute_chain_gex_at,
)
from chain_replay_ml.dataset_builder.chain_maps import chain_features_at, precompute_chain_maps
from chain_replay_ml.dataset_builder.controller_registry import CONTROLLER_FEATURES
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.ticks import TickTimeline


def _tl(ts: float, ltp: float, oi: int) -> TickTimeline:
    t = TickTimeline()
    t.append(ts, int(round(ltp * 100)), oi=oi)
    return t


class TestChainGex(unittest.TestCase):
    def test_registry(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in CHAIN_GEX_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])

    def test_net_and_total(self) -> None:
        ts = 1000.0
        index = TickTimeline()
        index.append(ts, 2450000)
        mapping = {
            (24500.0, "CE"): ("c", "C", _tl(ts, 100.0, 1000)),
            (24500.0, "PE"): ("p", "P", _tl(ts, 100.0, 2000)),
        }

        def fake_contrib(**kwargs):
            return 10.0 if kwargs["option_type"] == "CE" else 15.0

        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_gex._gamma_oi_contrib",
            side_effect=fake_contrib,
        ):
            out = compute_chain_gex_at(
                mapping,
                index_tl=index,
                ts=ts,
                expiry_ts=ts + 7 * 86400,
            )
        self.assertAlmostEqual(out["call_gex"], 10.0)
        self.assertAlmostEqual(out["put_gex"], 15.0)
        self.assertAlmostEqual(out["net_gex"], -5.0)
        self.assertAlmostEqual(out["chain_gex"], 25.0)

    def test_precompute_emit(self) -> None:
        ts = 1_700_000_000.0
        index = TickTimeline()
        index.append(ts, 2450000)
        mapping = {
            (24500.0, "CE"): ("c", "C", _tl(ts, 100.0, 500)),
            (24500.0, "PE"): ("p", "P", _tl(ts, 100.0, 500)),
        }
        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_gex.compute_chain_gex_at",
            return_value={
                "call_gex": 1.0,
                "put_gex": 2.0,
                "net_gex": -1.0,
                "chain_gex": 3.0,
            },
        ):
            maps = precompute_chain_maps(
                index_tl=index,
                strike_mapping=mapping,
                timestamps=[ts],
                strike_step=50,
                expiry_ts=ts + 86400,
                include_gex=True,
            )
        self.assertAlmostEqual(maps.net_gex[ts], -1.0)
        feats = chain_features_at(
            maps,
            ts,
            expiry_ts=ts + 86400,
            strike_mapping=mapping,
            index_tl=index,
            atm_strike=24500,
        )
        self.assertAlmostEqual(feats["call_gex"], 1.0)
        self.assertAlmostEqual(feats["put_gex"], 2.0)
        self.assertAlmostEqual(feats["net_gex"], -1.0)
        self.assertAlmostEqual(feats["chain_gex"], 3.0)


if __name__ == "__main__":
    unittest.main()
