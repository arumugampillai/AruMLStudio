"""Wave B: Chain Controller IV skew levels."""

from __future__ import annotations

import unittest
from unittest import mock

from chain_replay_ml.dataset_builder.chain_iv_skew import (
    CHAIN_IV_SKEW_FEATURES,
    compute_chain_iv_skew_at,
)
from chain_replay_ml.dataset_builder.chain_maps import (
    ChainMaps,
    chain_features_at,
    precompute_chain_maps,
)
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.ticks import TickTimeline


def _tl(ts: float, ltp: float) -> TickTimeline:
    t = TickTimeline()
    t.append(ts, int(round(ltp * 100)))
    return t


class TestChainIvSkew(unittest.TestCase):
    def test_registry_and_controller(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in CHAIN_IV_SKEW_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertIn("token.chain", CONTROLLER_REGISTRY)
        for name in CHAIN_IV_SKEW_FEATURES:
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])

    def test_formulas_from_injected_ivs(self) -> None:
        strike_mapping = {
            (24500.0, "CE"): ("1", "C", _tl(1000.0, 100.0)),
            (24500.0, "PE"): ("2", "P", _tl(1000.0, 100.0)),
            (24750.0, "CE"): ("3", "C", _tl(1000.0, 50.0)),
            (24250.0, "PE"): ("4", "P", _tl(1000.0, 50.0)),
        }

        def fake_iv(_sm, *, strike, option_type, **_kw):
            table = {
                (24500.0, "CE"): 0.12,
                (24500.0, "PE"): 0.13,
                (24750.0, "CE"): 0.11,
                (24250.0, "PE"): 0.15,
            }
            return table.get((float(strike), option_type))

        def fake_iv_delta(_sm, *, strike, option_type, **_kw):
            # Force 25Δ picks on wing strikes.
            if option_type == "CE" and float(strike) == 24750.0:
                return 0.11, 0.25
            if option_type == "PE" and float(strike) == 24250.0:
                return 0.15, -0.25
            if option_type == "CE":
                return 0.12, 0.50
            return 0.13, -0.50

        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_iv_skew._option_iv_at",
            side_effect=fake_iv,
        ), mock.patch(
            "chain_replay_ml.dataset_builder.chain_iv_skew._option_iv_delta_at",
            side_effect=fake_iv_delta,
        ):
            out = compute_chain_iv_skew_at(
                strike_mapping,
                ts=1000.0,
                spot=24500.0,
                atm_strike=24500.0,
                strike_step=50,
                expiry_ts=1000.0 + 7 * 86400,
                wing_steps=5,  # 5*50 = 250 → 24250 / 24750
            )
        self.assertAlmostEqual(out["iv_call_put_skew"], 0.12 - 0.13)
        self.assertAlmostEqual(out["iv_skew_atm"], 0.15 - 0.11)
        self.assertAlmostEqual(out["iv_skew_25d"], 0.15 - 0.11)

    def test_precompute_and_emit(self) -> None:
        ts = 1_700_000_000.0
        index = _tl(ts, 24500.0)
        mapping = {}
        for i in range(-10, 11):
            k = 24500.0 + i * 50.0
            mapping[(k, "CE")] = (f"c{i}", "C", _tl(ts, max(5.0, 120.0 - abs(i) * 8)))
            mapping[(k, "PE")] = (f"p{i}", "P", _tl(ts, max(5.0, 120.0 - abs(i) * 8)))

        with mock.patch(
            "chain_replay_ml.dataset_builder.chain_iv_skew.compute_chain_iv_skew_at",
            return_value={
                "atm_iv_ce": 0.12,
                "atm_iv_pe": 0.13,
                "iv_call_put_skew": -0.01,
                "iv_skew_atm": 0.04,
                "iv_skew_25d": 0.035,
            },
        ):
            maps = precompute_chain_maps(
                index_tl=index,
                strike_mapping=mapping,
                timestamps=[ts],
                strike_step=50,
                expiry_ts=ts + 7 * 86400,
                include_iv_skew=True,
            )
        self.assertAlmostEqual(maps.iv_skew_atm[ts], 0.04)
        self.assertAlmostEqual(maps.iv_call_put_skew[ts], -0.01)
        self.assertAlmostEqual(maps.iv_skew_25d[ts], 0.035)

        feats = chain_features_at(
            maps,
            ts,
            expiry_ts=ts + 7 * 86400,
            strike_mapping=mapping,
            index_tl=index,
            atm_strike=24500,
        )
        self.assertAlmostEqual(feats["iv_skew_atm"], 0.04)
        self.assertAlmostEqual(feats["iv_call_put_skew"], -0.01)
        self.assertAlmostEqual(feats["iv_skew_25d"], 0.035)

    def test_skip_skew_when_disabled(self) -> None:
        ts = 1_700_000_000.0
        index = _tl(ts, 24500.0)
        mapping = {
            (24500.0, "CE"): ("c", "C", _tl(ts, 100.0)),
            (24500.0, "PE"): ("p", "P", _tl(ts, 100.0)),
        }
        maps = precompute_chain_maps(
            index_tl=index,
            strike_mapping=mapping,
            timestamps=[ts],
            strike_step=50,
            expiry_ts=ts + 86400,
            include_iv_skew=False,
        )
        self.assertEqual(maps.iv_skew_atm, {})
        self.assertIsInstance(maps, ChainMaps)


if __name__ == "__main__":
    unittest.main()
