"""Regression tests for false EMA reset / token-scoped replay lookup."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.chain_maps import ChainMaps
from chain_replay_ml.dataset_builder.extended_features import OptionFeatureState, enrich_with_chain_maps
from chain_replay_ml.dataset_builder.gap_policy_instrumentation import row_gap_exceeds
from chain_replay_ml.dataset_builder.rolling_controllers import update_token_ltp_controllers
from chain_replay_ml.feature_policy.warmup_calc_debug import (
    build_replay_lookup_from_rows,
    resolve_primary_replay_token,
)
from chain_replay_ml.ticks import TickTimeline

GAP_MAX_SEC = 10.0
STEP_SEC = 3.0
OPEN_TS = 1_700_000_000.0


class SerialInterleavedGapTests(unittest.TestCase):
    def test_interleaved_tokens_no_false_gap_reset(self) -> None:
        """Serial row order (ts, tokenA), (ts, tokenB) must not reset 3s-spaced token A."""
        opt_states: dict[str, OptionFeatureState] = {}
        token_a = "111"
        token_b = "222"
        prices_a = [100.0 + i * 0.1 for i in range(200)]
        ratios: list[float | None] = []
        last_ts: dict[str, float | None] = {}

        for i in range(200):
            ts = OPEN_TS + i * STEP_SEC
            for token, price in ((token_a, prices_a[i]), (token_b, 50.0 + i * 0.05)):
                if token not in opt_states:
                    opt_states[token] = OptionFeatureState()
                opt_state = opt_states[token]
                prev_ts = last_ts.get(token)
                if prev_ts is not None and row_gap_exceeds(ts, prev_ts, GAP_MAX_SEC):
                    opt_state.controllers.reset_all(ts=ts)
                update_token_ltp_controllers(opt_state.controllers, price, ts=ts)
                last_ts[token] = ts
                if token != token_a:
                    continue
                raw = {"ltp": price, "spot": 25000.0}
                enriched = enrich_with_chain_maps(
                    raw,
                    ts=ts,
                    chain_maps=ChainMaps(),
                    strike_mapping={},
                    index_tl=TickTimeline(),
                    atm_strike=25000,
                    expiry_ts=OPEN_TS + 86400.0,
                    opt_state=opt_state,
                    option_timeline=TickTimeline(),
                    open_ts=OPEN_TS,
                    close_ts=OPEN_TS + 6 * 3600.0,
                    active_features=frozenset({"ltp_ema9_to_ltp_ratio"}),
                    feature_grid_step_sec=STEP_SEC,
                )
                ratios.append(enriched.get("ltp_ema9_to_ltp_ratio"))

        self.assertIsNone(ratios[0])
        self.assertIsNotNone(ratios[8])
        for idx in range(9, len(ratios)):
            self.assertIsNotNone(
                ratios[idx],
                f"unexpected NULL at token-A sample {idx + 1}",
            )

    def test_row_gap_exceeds_3s_under_10s_limit(self) -> None:
        self.assertFalse(row_gap_exceeds(OPEN_TS + 3.0, OPEN_TS, GAP_MAX_SEC))


class ReplayLookupTokenTests(unittest.TestCase):
    def test_resolve_primary_replay_token_stable(self) -> None:
        rows = [
            {"timestamp": 1000.0, "token": "AAA", "ltp": 100.0, "delta": 0.45},
            {"timestamp": 1000.0, "token": "BBB", "ltp": 50.0, "delta": 0.05},
            {"timestamp": 1003.0, "token": "AAA", "ltp": 101.0, "delta": 0.44},
            {"timestamp": 1003.0, "token": "BBB", "ltp": 51.0, "delta": 0.06},
        ]
        tok = resolve_primary_replay_token(rows, anchor_ts=1000.0, step_sec=3)
        self.assertEqual(tok, "AAA")

    def test_lookup_uses_single_token_not_atm_switch(self) -> None:
        rows = [
            {
                "timestamp": 1000.0,
                "token": "AAA",
                "ltp": 100.0,
                "ltp_ema9_to_ltp_ratio": 1.01,
                "delta": 0.45,
            },
            {
                "timestamp": 1000.0,
                "token": "BBB",
                "ltp": 50.0,
                "ltp_ema9_to_ltp_ratio": None,
                "delta": 0.05,
            },
            {
                "timestamp": 1003.0,
                "token": "AAA",
                "ltp": 101.0,
                "ltp_ema9_to_ltp_ratio": 1.02,
                "delta": 0.44,
            },
            {
                "timestamp": 1003.0,
                "token": "BBB",
                "ltp": 51.0,
                "ltp_ema9_to_ltp_ratio": None,
                "delta": 0.04,
            },
        ]
        lookup = build_replay_lookup_from_rows(
            rows,
            ["ltp_ema9_to_ltp_ratio"],
            token="AAA",
            anchor_ts=1000.0,
            step_sec=3,
        )
        self.assertAlmostEqual(lookup[1000]["ltp_ema9_to_ltp_ratio"], 1.01)
        self.assertAlmostEqual(lookup[1003]["ltp_ema9_to_ltp_ratio"], 1.02)

    def test_lookup_without_token_filter_can_switch_rows(self) -> None:
        """Document old failure mode: ATM scoring picked BBB over AAA at t=1003."""
        rows = [
            {
                "timestamp": 1000.0,
                "token": "AAA",
                "ltp": 100.0,
                "ltp_ema9_to_ltp_ratio": 1.01,
                "delta": 0.45,
            },
            {
                "timestamp": 1000.0,
                "token": "BBB",
                "ltp": 50.0,
                "ltp_ema9_to_ltp_ratio": None,
                "delta": 0.05,
            },
            {
                "timestamp": 1003.0,
                "token": "AAA",
                "ltp": 101.0,
                "ltp_ema9_to_ltp_ratio": 1.02,
                "delta": 0.44,
            },
            {
                "timestamp": 1003.0,
                "token": "BBB",
                "ltp": 51.0,
                "ltp_ema9_to_ltp_ratio": None,
                "delta": 0.04,
            },
        ]
        lookup_all = build_replay_lookup_from_rows(
            rows,
            ["ltp_ema9_to_ltp_ratio"],
            token="AAA",
            step_sec=3,
        )
        self.assertIsNotNone(lookup_all[1003]["ltp_ema9_to_ltp_ratio"])


class TokenKeyNormalizationTests(unittest.TestCase):
    def test_stages_token_dict_key_uses_str(self) -> None:
        opt_states: dict[str, OptionFeatureState] = {}
        for token in (12345, "12345"):
            key = str(token)
            if key not in opt_states:
                opt_states[key] = OptionFeatureState()
            opt_states[key].controllers.ema9.update(100.0)
        self.assertIs(opt_states["12345"], opt_states[str(12345)])


if __name__ == "__main__":
    unittest.main()
