"""Phase 5: production hardening — immutable artifacts, contracts, benchmarks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.outcome_label_engine import (
    ENGINE_VERSION,
    OleContractError,
    assert_run_meta_complete,
    assert_truncate_at_close_default,
    benchmark_fixed_horizon_regression,
    benchmark_streaming_day_chunks,
    create_immutable_writer,
    ensure_builtin_strategies,
    get_triple_barrier_strategy,
    label_triple_barrier_sample,
    run_all_benchmarks,
    run_labeling,
    sanitize_label_row,
)
from chain_replay_ml.outcome_label_engine.prediction_source import InMemoryPredictionDaySource


class OleHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        ensure_builtin_strategies()

    def test_truncate_at_close_default_contract(self) -> None:
        schema = get_triple_barrier_strategy().get_config_schema()
        assert_truncate_at_close_default(schema)

    def test_sentinel_label_id_rejected(self) -> None:
        with self.assertRaises(OleContractError):
            sanitize_label_row({"is_valid": True, "label_id": -1, "label_name": "X"})

    def test_invalid_row_clears_labels(self) -> None:
        out = sanitize_label_row(
            {
                "is_valid": False,
                "invalid_reason": "stale_path",
                "label_id": 2,
                "label_name": "TIME",
            }
        )
        self.assertIsNone(out["label_id"])
        self.assertIsNone(out["label_name"])
        self.assertEqual(out["invalid_reason"], "stale_path")

    def test_immutable_second_run_new_artifact(self) -> None:
        source = InMemoryPredictionDaySource(
            days={
                "2024-01-02": [
                    {
                        "prediction_id": "p1",
                        "trading_day": "2024-01-02",
                        "token": "T1",
                        "timestamp": 1000.0,
                        "current_ltp": 100.0,
                        "path": [{"timestamp": 1005.0, "ltp": 111.0}],
                        "session_close_ts": 5000.0,
                    }
                ]
            }
        )
        strategy = get_triple_barrier_strategy()
        with tempfile.TemporaryDirectory() as tmp:
            r1 = run_labeling(
                strategy=strategy,
                source=source,
                root=tmp,
                params={"holding_seconds": 300, "tp_points": 10.0, "sl_points": 5.0},
            )
            r2 = run_labeling(
                strategy=strategy,
                source=source,
                root=tmp,
                params={"holding_seconds": 300, "tp_points": 10.0, "sl_points": 5.0},
            )
            self.assertNotEqual(r1.artifact_id, r2.artifact_id)
            self.assertTrue((Path(tmp) / r1.artifact_id / "run_meta.json").is_file())
            self.assertTrue((Path(tmp) / r2.artifact_id / "run_meta.json").is_file())
            # Same explicit id must refuse overwrite.
            w = create_immutable_writer(tmp, "triple_barrier", suffix="same")
            w.open()
            with self.assertRaises(FileExistsError):
                create_immutable_writer(tmp, "triple_barrier", suffix="same").open()

    def test_run_meta_audit_fields(self) -> None:
        source = InMemoryPredictionDaySource(
            days={
                "2024-01-02": [
                    {
                        "prediction_id": "p1",
                        "trading_day": "2024-01-02",
                        "token": "T1",
                        "timestamp": 1000.0,
                        "current_ltp": 100.0,
                        "path": [],
                        "session_close_ts": 5000.0,
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_labeling(
                strategy=get_triple_barrier_strategy(),
                source=source,
                root=tmp,
                params={"holding_seconds": 300, "tp_points": 10.0, "sl_points": 5.0},
            )
            meta = assert_run_meta_complete(result.run_meta)
            self.assertEqual(meta["strategy"], "triple_barrier")
            self.assertEqual(meta["engine_version"], ENGINE_VERSION)
            self.assertIn("holding_seconds", meta["params"])
            self.assertEqual(meta["rows"], 1)
            self.assertGreaterEqual(meta["compute_time_sec"], 0.0)
            self.assertEqual(meta["target_definitions"]["primary_target"], "label_id")
            self.assertEqual(
                meta["target_definitions"]["label_encoding"],
                {"TP": 0, "SL": 1, "TIME": 2},
            )
            self.assertIsNotNone(meta.get("created_at_utc"))
            self.assertEqual(result.run_meta.invalid_rows, 1)
            self.assertEqual(result.run_meta.valid_rows, 0)
            disk = json.loads(
                (Path(result.artifact_dir) / "run_meta.json").read_text(encoding="utf-8")
            )
            assert_run_meta_complete(disk)

    def test_writer_rejects_sentinel_on_append(self) -> None:
        from chain_replay_ml.outcome_label_engine import (
            ImmutableArtifactWriter,
            LabelBatchResult,
            TargetDefinitions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            w = create_immutable_writer(tmp, "triple_barrier", suffix="bad")
            w.open()
            batch = LabelBatchResult(
                rows=[{"is_valid": True, "label_id": -1, "label_name": "X"}],
                target_columns=["label_id"],
                target_definitions=TargetDefinitions(primary_target="label_id"),
            )
            with self.assertRaises(OleContractError):
                w.append_day("2024-01-02", batch)

    def test_benchmarks_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stream = benchmark_streaming_day_chunks(
                root=tmp, n_days=3, rows_per_day=40
            )
            self.assertEqual(stream.rows, 120)
            self.assertEqual(stream.days, 3)
            self.assertIsNotNone(stream.peak_memory_kib)
            fh = benchmark_fixed_horizon_regression(n_rows=200)
            self.assertEqual(fh.extras["mismatches"], 0)
            results = run_all_benchmarks(tmp)
            self.assertEqual(len(results), 4)
            names = {r.name for r in results}
            self.assertIn("streaming_day_chunks", names)
            self.assertIn("fixed_horizon_regression", names)

    def test_empty_path_validity_not_sentinel(self) -> None:
        out = label_triple_barrier_sample(
            {
                "timestamp": 1000.0,
                "current_ltp": 100.0,
                "path": [],
                "session_close_ts": 5000.0,
            },
            holding_seconds=300,
            tp_points=10.0,
            sl_points=5.0,
        )
        self.assertFalse(out["is_valid"])
        self.assertEqual(out["invalid_reason"], "empty_path")
        self.assertIsNone(out["label_id"])


if __name__ == "__main__":
    unittest.main()
