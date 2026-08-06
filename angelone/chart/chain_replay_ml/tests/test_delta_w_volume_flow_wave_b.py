"""Wave B: chain delta-weighted volume flow."""

from __future__ import annotations

import unittest
from unittest import mock

from chain_replay_ml.dataset_builder.chain_delta_volume_flow import (
    DELTA_W_VOLUME_FLOW_FEATURES,
    compute_delta_w_volume_flow_at,
)
from chain_replay_ml.dataset_builder.chain_maps import chain_features_at, precompute_chain_maps
from chain_replay_ml.dataset_builder.controller_registry import CONTROLLER_FEATURES
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.ticks import TickTimeline


def _tl_with_volume(ts: float, ltp: float, vol_now: int, vol_past: int, lookback: float = 60.0) -> TickTimeline:
    t = TickTimeline()
    t.append(ts - lookback, int(round(ltp * 100)), volume=vol_past)
    t.append(ts, int(round(ltp * 100)), volume=vol_now)
    return t


class TestDeltaWVolumeFlow(unittest.TestCase):
    def test_registry(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in DELTA_W_VOLUME_FLOW_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        for name in DELTA_W_VOLUME_FLOW_FEATURES:
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])

    def test_signed_sum(self) -> None:
        ts = 1000.0
        index = TickTimeline()
        index.append(ts, 2450000)
        mapping = {
            (24500.0, "CE"): ("c", "C", _tl_with_volume(ts, 100.0, 200, 100)),
            (24500.0, "PE"): ("p", "P", _tl_with_volume(ts, 100.0, 150, 100)),
        }

        def fake_delta(**kwargs):
            return 0.5 if kwargs["option_type"] == "CE" else -0.4

        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_delta_volume_flow.compute_delta_at_ts",
            side_effect=fake_delta,
        ):
            # CE: 0.5 * 100 = 50; PE: -0.4 * 50 = -20; total = 30
            val = compute_delta_w_volume_flow_at(
                mapping,
                index_tl=index,
                ts=ts,
                expiry_ts=ts + 7 * 86400,
                lookback_sec=60.0,
            )
        self.assertAlmostEqual(val, 30.0)

    def test_precompute_emit(self) -> None:
        ts = 1_700_000_000.0
        index = TickTimeline()
        index.append(ts, 2450000)
        mapping = {
            (24500.0, "CE"): ("c", "C", _tl_with_volume(ts, 100.0, 200, 100)),
            (24500.0, "PE"): ("p", "P", _tl_with_volume(ts, 100.0, 100, 100)),
        }
        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_delta_volume_flow.compute_all_delta_w_volume_flows_at",
            return_value={
                "delta_w_volume_flow_1m": 12.5,
                "delta_w_volume_flow_5m": 40.0,
            },
        ):
            maps = precompute_chain_maps(
                index_tl=index,
                strike_mapping=mapping,
                timestamps=[ts],
                strike_step=50,
                expiry_ts=ts + 86400,
                include_delta_flow=True,
            )
        self.assertAlmostEqual(maps.delta_w_volume_flow_1m[ts], 12.5)
        feats = chain_features_at(
            maps,
            ts,
            expiry_ts=ts + 86400,
            strike_mapping=mapping,
            index_tl=index,
            atm_strike=24500,
        )
        self.assertAlmostEqual(feats["delta_w_volume_flow_1m"], 12.5)
        self.assertAlmostEqual(feats["delta_w_volume_flow_5m"], 40.0)


if __name__ == "__main__":
    unittest.main()
