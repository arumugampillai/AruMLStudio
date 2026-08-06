"""Tests for Phase 1 prediction run store and registry."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.prediction_runs.registry import compare_runs, get_fold_rows, get_run_detail, list_runs
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter


class PredictionRunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _create_run_with_fold_rows(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "TestModel_v1",
                "model_version": "1.0",
                "dataset_name": "MS_test",
                "target": "future_ltp_5m",
                "dataset_fingerprint": "ds_abc123",
                "feature_snapshot_hash": "feat_def456",
                "walk_forward_config_hash": "wf_ghi789",
                "training_config_hash": "cfg_jkl012",
                "status": "running",
                "run_kind": "walk_forward_production",
            })
            run_id = run["run_id"]
            writer = PredictionRunWriter(store, run_id)
            ctx = pd.DataFrame({
                "trading_day": ["2026-07-01", "2026-07-01"],
                "timestamp": [1.0, 2.0],
                "token": ["NIFTY", "NIFTY"],
                "strike": [24000.0, 24100.0],
                "option_type": ["CE", "PE"],
                "spot": [24050.0, 24055.0],
                "ltp": [120.0, 95.0],
            })
            writer.write_fold_predictions(
                fold_number=1,
                fold_def={
                    "train": {"start": 0, "stop": 10, "rows": 10},
                    "validation": {"start": 10, "stop": 12, "rows": 2},
                    "window_mode": "rolling",
                },
                metrics={"mae": 1.5, "rmse": 2.0, "directional_accuracy_pct": 50.0},
                val_context=ctx,
                val_pred=np.array([121.0, 94.0]),
                val_y=pd.Series([120.0, 96.0]),
                baseline_ltp=pd.Series([119.0, 95.0]),
            )
            store.finalize_run(
                run_id,
                status="completed",
                prediction_count=writer.row_count,
                fold_count=1,
            )
            folds = store.list_folds(run_id)
            return run_id, folds[0]["fold_id"]

    def test_create_run_and_list_rows(self) -> None:
        run_id, fold_id = self._create_run_with_fold_rows()
        detail = get_run_detail(self.tmp, run_id)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["model_id"], "TestModel_v1")
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(len(detail["folds"]), 1)
        self.assertEqual(detail["prediction_count_stored"], 2)

        rows_doc = get_fold_rows(self.tmp, run_id, fold_id, limit=10)
        self.assertTrue(rows_doc["ok"])
        self.assertEqual(rows_doc["total"], 2)
        self.assertEqual(len(rows_doc["rows"]), 2)
        self.assertAlmostEqual(rows_doc["rows"][0]["predicted_ltp"], 121.0)
        self.assertEqual(rows_doc["rows"][0]["direction_correct"], 1)

    def test_list_runs_for_model(self) -> None:
        run_id, _ = self._create_run_with_fold_rows()
        runs = list_runs(self.tmp, "TestModel_v1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], run_id)

    def test_compare_runs_by_fold(self) -> None:
        run_a, _ = self._create_run_with_fold_rows()
        with PredictionRunStore(self.tmp) as store:
            run_b = store.create_run({
                "model_id": "TestModel_v1",
                "dataset_name": "MS_test",
                "target": "future_ltp_5m",
                "status": "completed",
            })
            store.insert_fold({
                "run_id": run_b["run_id"],
                "fold_number": 1,
                "mae": 2.0,
                "rmse": 2.5,
                "directional_accuracy_pct": 40.0,
                "prediction_count": 2,
            })
            run_b_id = run_b["run_id"]

        result = compare_runs(self.tmp, run_a, run_b_id)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["fold_comparisons"]), 1)
        self.assertAlmostEqual(result["fold_comparisons"][0]["run_a"]["mae"], 1.5)
        self.assertAlmostEqual(result["fold_comparisons"][0]["run_b"]["mae"], 2.0)

    def test_list_all_runs(self) -> None:
        run_id, _ = self._create_run_with_fold_rows()
        from chain_replay_ml.prediction_runs.registry import list_all_runs

        runs = list_all_runs(self.tmp)
        self.assertGreaterEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], run_id)

        from chain_replay_ml.prediction_runs.paths import prediction_runs_db_path

        path = prediction_runs_db_path(self.tmp)
        self.assertTrue(path.endswith(os.path.join("prediction_runs", "registry.db")))


if __name__ == "__main__":
    unittest.main()
