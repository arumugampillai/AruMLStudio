"""Unit tests for the OHLC Aggregation feature transformation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.transformations import (
    default_transformation_config,
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation import (
    OHLC_TIMEFRAMES,
    ohlc_aggregation_column_name,
    period_rows_for_timeframe,
)
from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
    build_ohlc_aggregation_config,
    merge_ohlc_aggregation_into_config,
    timeframe_display_label,
)
from chain_replay_ml.dataset_builder.transformations.ohlc_history_profiles import (
    available_ohlc_timeframes,
    clear_ohlc_history_profile_cache,
    get_ohlc_interval_profile,
    load_ohlc_history_profiles,
    resolve_timeframe_spec,
)
from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
    merge_rolling_into_config,
)
from chain_replay_ml.dataset_builder.transformations.time_shift import LagConfigError


class OhlcAggregationTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_ohlc_history_profile_cache()

    def test_ohlc_aggregation_column_naming(self) -> None:
        self.assertEqual(
            ohlc_aggregation_column_name("spot_ltp", "3m", 1, "close"),
            "spot_ltp_3m_1_close",
        )
        self.assertEqual(
            ohlc_aggregation_column_name("delta", "5m", 2, "high"),
            "delta_5m_2_high",
        )
        self.assertEqual(
            ohlc_aggregation_column_name("iv", "15m", 1, "open"),
            "iv_15m_1_open",
        )

    def test_interval_history_profiles(self) -> None:
        p3 = get_ohlc_interval_profile(3)
        self.assertEqual(p3.get("3m").history, 6)
        self.assertEqual(p3.get("5m").history, 3)
        self.assertEqual(p3.get("15m").history, 1)

        p6 = get_ohlc_interval_profile(6)
        self.assertEqual(p6.get("3m").history, 6)
        self.assertEqual(p6.get("5m").history, 6)
        self.assertEqual(p6.get("15m").history, 2)
        self.assertEqual(p6.get("30m").history, 1)
        self.assertIn("30m", available_ohlc_timeframes(6))

        p9 = get_ohlc_interval_profile(9)
        self.assertEqual(p9.get("3m").history, 6)
        self.assertEqual(p9.get("15m").history, 3)
        self.assertIn("5m", available_ohlc_timeframes(9))
        self.assertNotIn("30m", available_ohlc_timeframes(9))
        # 5m @ 9s is an explicit 297s approximation (33 samples), not silent.
        spec5 = resolve_timeframe_spec(9, "5m")
        self.assertEqual(spec5.actual_duration_sec, 297)
        self.assertEqual(spec5.nominal_duration_sec, 300)
        self.assertEqual(spec5.sample_count(9), 33)
        self.assertTrue(spec5.is_approximate)
        meta = spec5.to_metadata(9)
        self.assertEqual(meta["timeframe_label"], "5m")
        self.assertEqual(meta["actual_duration_sec"], 297)
        self.assertEqual(meta["sample_count"], 33)
        self.assertTrue(meta["is_approximate"])
        from chain_replay_ml.dataset_builder.transformations.ohlc_history_profiles import (
            format_ohlc_approximation_hint,
            unavailable_ohlc_timeframe_messages,
        )

        self.assertEqual(unavailable_ohlc_timeframe_messages(9), {})
        hint = format_ohlc_approximation_hint(9)
        self.assertIn("5m", hint)
        self.assertIn("297s", hint)
        self.assertEqual(period_rows_for_timeframe("5m", 9.0), 33)
        label = timeframe_display_label("5m", sample_interval_sec=9)
        self.assertIn("297", label)
        self.assertIn("33", label)
        cfg = build_ohlc_aggregation_config(
            enabled=True,
            features=["x"],
            timeframes=["5m"],
            outputs=["close"],
            sample_interval_sec=9,
        )
        specs = cfg["transformations"][0]["params"]["timeframe_specs"]
        self.assertEqual(specs[0]["timeframe_label"], "5m")
        self.assertEqual(specs[0]["actual_duration_sec"], 297)
        self.assertEqual(specs[0]["sample_count"], 33)

    def test_history_not_from_warmup_configurable_document(self) -> None:
        custom = load_ohlc_history_profiles(
            document={
                "profiles": {
                    "3": {
                        "3m": {"seconds": 180, "history": 2},
                    }
                }
            }
        )
        self.assertEqual(custom[3].get("3m").history, 2)
        spec = resolve_timeframe_spec(3, "3m", profiles=custom)
        self.assertEqual(spec.history, 2)

    def test_disabled_passthrough(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cfg = build_ohlc_aggregation_config(
            enabled=False,
            features=["x"],
            timeframes=["3m"],
            outputs=["close"],
        )
        self.assertEqual(cfg["transformations"], [])
        result = run_transformation_pipeline(df, cfg)
        self.assertIs(result.frame, df)
        self.assertEqual(result.executed, 0)
        self.assertTrue(all("_3m_" not in c for c in result.frame.columns))

        cfg2 = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "ohlc_aggregation",
                "enabled": False,
                "params": {
                    "features": ["x"],
                    "timeframes": ["3m"],
                    "outputs": ["close"],
                },
            }],
        }
        result2 = run_transformation_pipeline(df, cfg2)
        self.assertEqual(result2.executed, 0)
        self.assertNotIn("x_3m_1_close", result2.frame.columns)

    def test_3s_interval_3m_period_and_history_1(self) -> None:
        period = period_rows_for_timeframe("3m", 3.0)
        self.assertEqual(period, 60)
        n = 65
        vals = np.arange(1.0, n + 1.0)
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * n,
            "token": ["T"] * n,
            "x": vals,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "ohlc_aggregation",
                "enabled": True,
                "params": {
                    "features": ["x"],
                    "timeframes": ["3m"],
                    "outputs": ["open", "high", "low", "close"],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        col = "x_3m_1_close"
        self.assertIn(col, result.frame.columns)
        self.assertTrue(result.frame[col].iloc[:59].isna().all())
        self.assertFalse(pd.isna(result.frame[col].iloc[59]))
        self.assertEqual(float(result.frame[col].iloc[59]), 60.0)

    def test_history_index_1_is_newest_completed(self) -> None:
        n = 120
        vals = np.arange(1.0, n + 1.0)
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * n,
            "token": ["T"] * n,
            "x": vals,
        })
        cfg = build_ohlc_aggregation_config(
            enabled=True,
            features=["x"],
            timeframes=["3m"],
            outputs=["close"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        # At last row: candle1 = rows 61..120 close=120; candle2 = rows 1..60 close=60
        self.assertEqual(float(result.frame["x_3m_1_close"].iloc[-1]), 120.0)
        self.assertEqual(float(result.frame["x_3m_2_close"].iloc[-1]), 60.0)

    def test_history_lengths_per_timeframe_3s_profile(self) -> None:
        # Legacy export mirrors 3s profile
        self.assertEqual(OHLC_TIMEFRAMES["3m"].history, 6)
        self.assertEqual(OHLC_TIMEFRAMES["5m"].history, 3)
        self.assertEqual(OHLC_TIMEFRAMES["15m"].history, 1)

    def test_6s_profile_emits_30m_and_deeper_5m_history(self) -> None:
        period = period_rows_for_timeframe("3m", 6.0)
        self.assertEqual(period, 30)
        # 6 completed 3m candles + a bit → history index 6 present
        n = 30 * 6 + 5
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * n,
            "token": ["T"] * n,
            "x": np.arange(1.0, n + 1.0),
        })
        cfg = build_ohlc_aggregation_config(
            enabled=True,
            features=["x"],
            timeframes=["3m", "5m", "30m"],
            outputs=["close"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=6.0,
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=6.0)
        )
        self.assertIn("x_3m_6_close", result.frame.columns)
        self.assertIn("x_5m_6_close", result.frame.columns)
        self.assertIn("x_30m_1_close", result.frame.columns)
        # Not enough rows for a completed 30m candle yet
        self.assertTrue(result.frame["x_30m_1_close"].isna().all())

    def test_partial_candle_never_emitted(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 59,
            "token": ["T"] * 59,
            "x": np.arange(1.0, 60.0),
        })
        cfg = build_ohlc_aggregation_config(
            enabled=True,
            features=["x"],
            timeframes=["3m"],
            outputs=["close"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        self.assertTrue(result.frame["x_3m_1_close"].isna().all())

    def test_ohlc_values_match_window(self) -> None:
        vals = list(range(1, 61))
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 60,
            "token": ["T"] * 60,
            "x": [float(v) for v in vals],
        })
        cfg = build_ohlc_aggregation_config(
            enabled=True,
            features=["x"],
            timeframes=["3m"],
            outputs=["open", "high", "low", "close"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        last = result.frame.iloc[-1]
        self.assertEqual(float(last["x_3m_1_open"]), 1.0)
        self.assertEqual(float(last["x_3m_1_high"]), 60.0)
        self.assertEqual(float(last["x_3m_1_low"]), 1.0)
        self.assertEqual(float(last["x_3m_1_close"]), 60.0)

    def test_order_after_rolling_before_interaction(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Need enough rows for a tiny custom override — use history override via params
        # Stick with rolling + ohlc disabled history path: use 3m with few rows just to check order ids
        n = 60
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * n,
            "token": ["A"] * n,
            "x": np.arange(1.0, n + 1.0),
            "y": np.arange(1.0, n + 1.0) * 2,
        })
        cfg = default_transformation_config()
        cfg = merge_rolling_into_config(
            cfg,
            enabled=True,
            features=["x"],
            windows=[3],
            operations=["mean"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        cfg = merge_ohlc_aggregation_into_config(
            cfg,
            enabled=True,
            features=["x"],
            timeframes=["3m"],
            outputs=["close"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        cfg["transformations"].append({
            "id": "interaction",
            "enabled": True,
            "order": 50,
            "params": {
                "pairs": [{
                    "left": "x",
                    "right": "y",
                    "op": "multiply",
                    "output": "x_x_y",
                }],
            },
        })
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        self.assertEqual(
            result.executed_ids,
            ["rolling", "ohlc_aggregation", "interaction"],
        )

    def test_reject_duplicate_existing_column(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 60,
            "token": ["A"] * 60,
            "x": np.arange(1.0, 61.0),
            "x_3m_1_close": [0.0] * 60,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "ohlc_aggregation",
                "enabled": True,
                "params": {
                    "features": ["x"],
                    "timeframes": ["3m"],
                    "outputs": ["close"],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        with self.assertRaises(LagConfigError) as err:
            run_transformation_pipeline(
                df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
            )
        self.assertIn("already exists", str(err.exception))

    def test_unknown_interval_profile_fails(self) -> None:
        with self.assertRaises(LagConfigError):
            get_ohlc_interval_profile(12)


if __name__ == "__main__":
    unittest.main()
