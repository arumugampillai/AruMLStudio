"""Phase 3: Triple Barrier — TP/SL/TIME, close truncation, stale validity, encoding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.outcome_label_engine import (
    ENGINE_VERSION,
    TRIPLE_BARRIER_LABEL_ENCODING,
    TRIPLE_BARRIER_STRATEGY_ID,
    DayChunkRunner,
    ImmutableArtifactWriter,
    InMemoryPredictionDaySource,
    LabelSourceContext,
    LabelStrategyConfig,
    ensure_builtin_strategies,
    get_strategy,
    label_triple_barrier_sample,
    list_strategy_ids,
    mint_artifact_id,
    resolve_barrier_prices,
)
from chain_replay_ml.outcome_label_engine.triple_barrier import TripleBarrierStrategy


def _path(*marks: tuple[float, float]) -> list[dict]:
    """Build explicit path points as (timestamp, ltp)."""
    return [{"timestamp": ts, "ltp": px} for ts, px in marks]


class TripleBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        ensure_builtin_strategies()

    def test_registry_has_triple_barrier(self) -> None:
        self.assertIn(TRIPLE_BARRIER_STRATEGY_ID, list_strategy_ids())
        s = get_strategy(TRIPLE_BARRIER_STRATEGY_ID)
        self.assertIsInstance(s, TripleBarrierStrategy)
        self.assertEqual(s.metadata.display_name, "Triple Barrier")
        self.assertEqual(s.get_target_definitions().primary_target, "label_id")
        self.assertEqual(s.get_target_definitions().label_encoding, TRIPLE_BARRIER_LABEL_ENCODING)
        schema = s.get_config_schema()
        self.assertEqual(schema["barrier_type"]["default"], "percentage")
        self.assertEqual(schema["tp_value"]["default"], 20.0)

    def test_resolve_barrier_prices_modes(self) -> None:
        tp, sl = resolve_barrier_prices(100.0, barrier_type="points", tp_value=10.0, sl_value=5.0)
        self.assertEqual((tp, sl), (110.0, 95.0))
        tp, sl = resolve_barrier_prices(
            100.0, barrier_type="percentage", tp_value=20.0, sl_value=10.0
        )
        self.assertEqual((tp, sl), (120.0, 90.0))
        with self.assertRaises(ValueError):
            resolve_barrier_prices(100.0, barrier_type="atr", tp_value=1.0, sl_value=1.0)

    def test_tp_first(self) -> None:
        entry_ts = 1000.0
        sample = {
            "prediction_id": "p1",
            "trading_day": "2024-01-02",
            "token": "T1",
            "timestamp": entry_ts,
            "current_ltp": 100.0,
            "path": _path((1003.0, 105.0), (1006.0, 112.0), (1010.0, 90.0)),
            "session_close_ts": 2000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            barrier_type="points",
            tp_value=10.0,
            sl_value=5.0,
            truncate_at_close=True,
        )
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "TP")
        self.assertEqual(out["label_id"], 0)
        self.assertEqual(out["exit_reason"], "TP")
        self.assertEqual(out["tp_price"], 110.0)
        self.assertEqual(out["sl_price"], 95.0)
        self.assertEqual(out["exit_price"], 112.0)
        self.assertAlmostEqual(out["realized_return"], 0.12)

    def test_percentage_tp(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": _path((1003.0, 121.0)),
            "session_close_ts": 2000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            barrier_type="percentage",
            tp_value=20.0,
            sl_value=10.0,
        )
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "TP")
        self.assertEqual(out["tp_price"], 120.0)
        self.assertEqual(out["sl_price"], 90.0)

    def test_legacy_tp_points_alias(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": _path((1003.0, 112.0)),
            "session_close_ts": 2000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
        )
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "TP")
        self.assertEqual(out["tp_price"], 110.0)

    def test_sl_first(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": _path((1003.0, 94.0), (1006.0, 120.0)),
            "session_close_ts": 2000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
        )
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "SL")
        self.assertEqual(out["label_id"], 1)
        self.assertEqual(out["exit_price"], 94.0)

    def test_time_exit(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": _path((1100.0, 102.0), (1200.0, 103.0), (1400.0, 104.0)),
            "session_close_ts": 5000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=250,
            tp_points=10.0,
            sl_points=5.0,
        )
        # hold_end = 1250 → last mark in window is 1200 @ 103
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "TIME")
        self.assertEqual(out["label_id"], 2)
        self.assertEqual(out["exit_timestamp"], 1200.0)
        self.assertEqual(out["exit_price"], 103.0)

    def test_truncate_at_close(self) -> None:
        # Entry 15:28-ish, close 15:30, holding 5m → only ~2m window.
        entry_ts = 1_000_000.0
        close_ts = entry_ts + 120.0
        sample = {
            "timestamp": entry_ts,
            "current_ltp": 100.0,
            "path": _path(
                (entry_ts + 60.0, 101.0),
                (entry_ts + 100.0, 102.0),
                (entry_ts + 200.0, 150.0),  # after close — must be ignored
            ),
            "session_close_ts": close_ts,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
            truncate_at_close=True,
        )
        self.assertTrue(out["is_valid"])
        self.assertEqual(out["label_name"], "TIME")
        self.assertEqual(out["exit_timestamp"], entry_ts + 100.0)
        self.assertEqual(out["exit_price"], 102.0)
        self.assertLess(out["holding_seconds"], 300.0)

    def test_session_closed_before_entry(self) -> None:
        sample = {
            "timestamp": 2000.0,
            "current_ltp": 100.0,
            "path": _path((2003.0, 110.0)),
            "session_close_ts": 1500.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
            truncate_at_close=True,
        )
        self.assertFalse(out["is_valid"])
        self.assertEqual(out["invalid_reason"], "session_closed_before_entry")
        self.assertIsNone(out["label_id"])

    def test_empty_path_invalid(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": [],
            "session_close_ts": 5000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
        )
        self.assertFalse(out["is_valid"])
        self.assertEqual(out["invalid_reason"], "empty_path")
        self.assertIsNone(out["label_name"])
        self.assertIsNone(out["label_id"])

    def test_stale_path_gap(self) -> None:
        sample = {
            "timestamp": 1000.0,
            "current_ltp": 100.0,
            "path": _path((1003.0, 101.0), (1300.0, 102.0)),
            "session_close_ts": 5000.0,
        }
        out = label_triple_barrier_sample(
            sample,
            holding_seconds=600,
            tp_points=10.0,
            sl_points=5.0,
            max_path_gap_sec=60.0,
        )
        self.assertFalse(out["is_valid"])
        self.assertEqual(out["invalid_reason"], "stale_path")
        self.assertNotEqual(out.get("label_id"), -1)
        self.assertIsNone(out["label_id"])

    def test_encoding_map_clean(self) -> None:
        self.assertEqual(TRIPLE_BARRIER_LABEL_ENCODING, {"TP": 0, "SL": 1, "TIME": 2})
        self.assertNotIn(-1, TRIPLE_BARRIER_LABEL_ENCODING.values())

    def test_build_labels_from_day_grid(self) -> None:
        """Day cohort without explicit path — group by token and walk forward."""
        day = "2024-01-02"
        rows = [
            {
                "prediction_id": "a",
                "trading_day": day,
                "token": "T1",
                "timestamp": 1000.0,
                "row_index": 0,
                "current_ltp": 100.0,
                "session_close_ts": 5000.0,
            },
            {
                "prediction_id": "b",
                "trading_day": day,
                "token": "T1",
                "timestamp": 1003.0,
                "row_index": 1,
                "current_ltp": 112.0,
                "session_close_ts": 5000.0,
            },
            {
                "prediction_id": "c",
                "trading_day": day,
                "token": "T1",
                "timestamp": 1006.0,
                "row_index": 2,
                "current_ltp": 90.0,
                "session_close_ts": 5000.0,
            },
        ]
        batch = TripleBarrierStrategy().build_labels(
            LabelSourceContext(source_kind="prediction", day=day),
            rows,
            LabelStrategyConfig(
                strategy_id="triple_barrier",
                version="1.0",
                params={"holding_seconds": 300, "tp_points": 10.0, "sl_points": 5.0},
            ),
        )
        by_pid = {r["prediction_id"]: r for r in batch.rows}
        self.assertEqual(by_pid["a"]["label_name"], "TP")
        self.assertEqual(by_pid["a"]["exit_price"], 112.0)
        self.assertEqual(by_pid["b"]["label_name"], "SL")
        self.assertEqual(by_pid["b"]["exit_price"], 90.0)
        # Last sample has empty forward path → invalid empty_path
        self.assertFalse(by_pid["c"]["is_valid"])
        self.assertEqual(by_pid["c"]["invalid_reason"], "empty_path")

    def test_day_chunk_end_to_end(self) -> None:
        source = InMemoryPredictionDaySource(
            days={
                "2024-01-02": [
                    {
                        "prediction_id": "d1",
                        "trading_day": "2024-01-02",
                        "token": "T1",
                        "timestamp": 1000.0,
                        "current_ltp": 100.0,
                        "path": _path((1005.0, 111.0)),
                    }
                ],
                "2024-01-03": [
                    {
                        "prediction_id": "d2",
                        "trading_day": "2024-01-03",
                        "token": "T1",
                        "timestamp": 2000.0,
                        "current_ltp": 50.0,
                        "path": _path((2010.0, 44.0)),
                    }
                ],
            },
            session_close_by_day={"2024-01-02": 5000.0, "2024-01-03": 5000.0},
        )
        strategy = get_strategy(TRIPLE_BARRIER_STRATEGY_ID)
        with tempfile.TemporaryDirectory() as tmp:
            artifact_id = mint_artifact_id("triple_barrier", suffix="e2e")
            writer = ImmutableArtifactWriter(tmp, artifact_id)
            result = DayChunkRunner().run(
                strategy,
                source,
                LabelStrategyConfig(
                    strategy_id="triple_barrier",
                    version="1.0",
                    params={"holding_seconds": 300, "tp_points": 10.0, "sl_points": 5.0},
                ),
                writer,
            )
            self.assertEqual(result.days_processed, ["2024-01-02", "2024-01-03"])
            self.assertEqual(result.run_meta.rows, 2)
            self.assertEqual(result.run_meta.engine_version, ENGINE_VERSION)
            self.assertEqual(result.run_meta.strategy, "triple_barrier")
            meta = json.loads(
                Path(result.artifact_dir, "run_meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["target_definitions"]["primary_target"], "label_id")
            self.assertEqual(
                meta["target_definitions"]["label_encoding"],
                {"TP": 0, "SL": 1, "TIME": 2},
            )


if __name__ == "__main__":
    unittest.main()
