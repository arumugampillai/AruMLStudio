"""Tests for warm-up calculation debugger."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy.warmup_calc_debug import (
    build_formula_spec,
    build_replay_lookup,
    build_replay_lookup_from_rows,
    build_row_breakdown,
    lookup_replay_hit_rate,
    lookup_replay_values,
    normalize_epoch_ts,
    replay_columns_for,
    _operand_values,
)


class WarmupCalcDebugTests(unittest.TestCase):
    def test_replay_columns_ltp_ema_level(self) -> None:
        cols = replay_columns_for("ltp_ema200")
        self.assertIn("ltp", cols)
        self.assertIn("ltp_ema200", cols)

    def test_replay_columns_channel_width(self) -> None:
        cols = replay_columns_for("ltp_to_spot_ema200_channel_width_ratio")
        self.assertIn("spot_high_ema200", cols)
        self.assertIn("spot_low_ema200", cols)

    def test_operands_ltp_ema_level(self) -> None:
        ops = _operand_values(
            "ltp_ema200",
            kind="ltp_ema_level",
            period=200,
            replay_vals={"ltp": 242.3, "spot": 25280.0, "ltp_ema200": 239.02},
            feature_ready=True,
        )
        self.assertAlmostEqual(ops["ema200"], 239.02, places=2)

    def test_operands_channel_width(self) -> None:
        ops = _operand_values(
            "ltp_to_spot_ema200_channel_width_ratio",
            kind="channel_width",
            period=200,
            replay_vals={
                "ltp": 210.0,
                "ltp_to_spot_ema200_channel_width_ratio": 5.25,
                "spot_high_ema200": 24519.6,
                "spot_low_ema200": 24490.2,
            },
            feature_ready=True,
        )
        self.assertAlmostEqual(ops["high_ema200"], 24519.6, places=1)
        self.assertAlmostEqual(ops["low_ema200"], 24490.2, places=1)
        self.assertAlmostEqual(ops["channel_width"], 24519.6 - 24490.2, places=1)

    def test_row_breakdown_shows_substitution(self) -> None:
        spec = build_formula_spec("ltp_ema200")
        ops = {
            "ltp": 242.30,
            "spot": 25280.15,
            "ema200": 239.02,
            "feature_value": 239.02,
        }
        bd = build_row_breakdown(
            "ltp_ema200",
            sample=351,
            time="10:17:33",
            operands=ops,
            formula_spec=spec,
        )
        self.assertTrue(any("EMA200" in d for d in bd["dependencies"]))
        self.assertIn("EMA200", "\n".join(bd["tree_lines"]))

    def test_build_calculation_rows_all_trace(self) -> None:
        from chain_replay_ml.feature_policy.warmup_calc_debug import build_calculation_rows

        trace = [
            {"samples": i + 1, "time": f"09:{i:02d}", "ts": 1000.0 + i * 3, "feature_ready": i >= 2}
            for i in range(5)
        ]
        lookup = {
            int(1000.0 + i * 3): {"ltp": 100.0 + i, "spot": 25000.0, "ltp_ema200": 100.0 + i}
            for i in range(5)
        }
        rows, _spec = build_calculation_rows(
            trace,
            feature_name="ltp_ema200",
            replay_lookup=lookup,
            step_sec=3,
        )
        self.assertEqual(len(rows), 5)

    def test_normalize_epoch_ts_datetime_nanoseconds(self) -> None:
        import pandas as pd

        ts = pd.Timestamp("2026-07-01 09:59:33", tz="Asia/Kolkata")
        key = normalize_epoch_ts(ts)
        self.assertIsNotNone(key)
        bad = normalize_epoch_ts(float(ts.value))
        self.assertEqual(bad, key)

    def test_build_replay_lookup_datetime_column(self) -> None:
        import pandas as pd

        epoch = 1782811773.0
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([epoch], unit="s", utc=True),
            "ltp": [20.35],
            "spot": [23981.75],
            "weighted_ltp_ema_to_ltp_ratio": [1.0],
        })
        lookup = build_replay_lookup(
            df,
            ["ltp", "spot", "weighted_ltp_ema_to_ltp_ratio"],
            step_sec=3,
        )
        key = normalize_epoch_ts(epoch)
        assert key is not None
        self.assertIn(key, lookup)
        self.assertAlmostEqual(float(lookup[key]["ltp"]), 20.35)

    def test_lookup_from_chain_rows_matches_trace(self) -> None:
        trace = [{"ts": 1782811773.0 + i * 3, "samples": i + 1, "feature_ready": True} for i in range(5)]
        rows = [
            {
                "timestamp": 1782811773.0 + i * 3,
                "ltp": 20.0 + i,
                "spot": 24000.0,
                "weighted_ltp_ema_to_ltp_ratio": 1.0,
                "delta": 0.5,
            }
            for i in range(5)
        ]
        lookup = build_replay_lookup_from_rows(
            rows,
            ["ltp", "spot", "weighted_ltp_ema_to_ltp_ratio"],
        )
        self.assertAlmostEqual(lookup_replay_hit_rate(lookup, trace, step_sec=3), 1.0)
        vals = lookup_replay_values(lookup, trace[2]["ts"], step_sec=3)
        self.assertAlmostEqual(float(vals["ltp"]), 22.0)

    def test_weighted_ltp_replay_columns(self) -> None:
        cols = replay_columns_for("weighted_ltp_ema_to_ltp_ratio")
        self.assertIn("ltp_ema200", cols)

    def test_weighted_ltp_operands_show_ltp_before_ready(self) -> None:
        ops = _operand_values(
            "weighted_ltp_ema_to_ltp_ratio",
            kind="weighted_ltp_ema",
            period=None,
            replay_vals={"ltp": 20.35, "spot": 23981.75, "weighted_ltp_ema_to_ltp_ratio": 1.0},
            feature_ready=False,
        )
        self.assertAlmostEqual(ops["ltp"], 20.35)
        self.assertAlmostEqual(ops["spot"], 23981.75)
        self.assertIsNone(ops["feature_value"])

    def test_lookup_retains_null_warmup_columns(self) -> None:
        ts = 1782880173.0
        rows = [
            {"timestamp": ts, "ltp": 530.15, "spot": 23981.75, "iv_change_1m": ""},
            {"timestamp": ts + 3, "ltp": 530.2, "spot": 23981.25, "iv_change_1m": -0.28},
        ]
        lookup = build_replay_lookup_from_rows(rows, ["ltp", "spot", "iv_change_1m"])
        self.assertIn(int(round(ts)), lookup)
        self.assertIn("iv_change_1m", lookup[int(round(ts))])
        self.assertIsNone(lookup[int(round(ts))]["iv_change_1m"])
        vals = lookup_replay_values(lookup, ts, step_sec=3)
        self.assertIn("iv_change_1m", vals)
        self.assertIsNone(vals["iv_change_1m"])


if __name__ == "__main__":
    unittest.main()
