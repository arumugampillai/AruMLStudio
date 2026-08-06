"""Confidence Filter Comparison — SQL-only side-by-side classifier impact."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.writer import datasets_dir
from chain_replay_ml.model_lab.confidence import create_confidence_dataset
from chain_replay_ml.model_lab.confidence_inference import run_confidence_inference
from chain_replay_ml.model_lab.confidence_manifest import set_operating_threshold
from chain_replay_ml.model_lab.confidence_train import train_confidence_model
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    DATASET_TYPE_UNSEEN,
)
from chain_replay_ml.model_lab.research_dashboard import (
    compute_confidence_filter_comparison,
    list_comparable_confidence_models,
)
from chain_replay_ml.model_lab.store import ModelLabStore


def _write_train(data_dir: str, name: str, rows: list[dict]) -> None:
    out = datasets_dir(data_dir)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
    with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_name": name,
                "feature_columns": ["f1", "f2"],
                "selected_features": ["f1", "f2"],
                "feature_count": 2,
                "prediction_target_columns": ["future_ltp_5m"],
                "row_count": len(rows),
            },
            fh,
            indent=2,
        )


def _seed_lab(path: str, *, data_dir: str, parent_dataset: str, pred_rows: list[dict]) -> None:
    with ModelLabStore(path) as store:
        store._ensure_schema()
        store.ensure_prediction_schema()
        store.ensure_feature_columns(["sf_f1", "sf_f2"])
        store.write_info(
            lab_uuid="u1",
            lab_id="lab1",
            lab_name="Test Lab",
            parent_model_id="m1",
            parent_model_name="Future_LTP_5m_Test",
            model_checksum=None,
            description=None,
            purpose=None,
            version=1,
            original_feature_count=2,
            selected_feature_count=2,
            training_rows=len(pred_rows),
            target="future_ltp_5m",
            algorithm="xgboost",
            dataset_snapshot={"dataset_name": parent_dataset},
            model_snapshot=None,
            training_config_snapshot=None,
            wf_snapshot=None,
            metrics_snapshot=None,
            selected_features_snapshot=["f1", "f2"],
            feature_ranking_snapshot=None,
            artifact_pointers={
                "package_dir": {"path": os.path.join(data_dir, "models", "Future_LTP_5m_Test")}
            },
        )
        store.write_prediction_summary(
            lab_uuid="u1",
            status="ready",
            row_count=len(pred_rows),
            trading_days=len({r.get("trading_day") for r in pred_rows}),
            target_column="future_ltp_5m",
            parent_dataset=parent_dataset,
            parent_model_name="Future_LTP_5m_Test",
            created_at="2026-07-16T10:00:00+00:00",
            feature_storage_mode="embedded",
            feature_columns_json=json.dumps({"f1": "sf_f1", "f2": "sf_f2"}),
            selected_feature_count=2,
        )
        store.insert_prediction_rows(pred_rows, feature_columns=["sf_f1", "sf_f2"])
        days = sorted({str(r.get("trading_day")) for r in pred_rows if r.get("trading_day")})
        store.ensure_build_days(
            "u1",
            days,
            day_dataset_types={
                "2024-01-02": DATASET_TYPE_SEEN,
                "2024-01-03": DATASET_TYPE_UNSEEN,
            },
        )


class _FakeBooster:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        vals = pd.to_numeric(X["f1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(vals, 0.0, 1.0)


class ConfidenceFilterComparisonTests(unittest.TestCase):
    def _lab(self, tmp: str) -> tuple[str, str]:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_rows = []
        pred_rows = []
        for i in range(60):
            day = "2024-01-02" if i < 40 else "2024-01-03"
            f1 = 0.9 if i % 2 == 0 else 0.1
            hit = 1 if f1 > 0.5 else 0
            pred_rows.append(
                {
                    "lab_uuid": "u1",
                    "prediction_id": f"p{i}",
                    "trading_day": day,
                    "timestamp": float(1000 + i),
                    "token": f"T{i}",
                    "master_row_id": i + 1,
                    "sf_f1": f1,
                    "sf_f2": float(i % 7),
                    "current_ltp": 100.0,
                    "expected_move": 1.0,
                    "actual_move": 1.0 if hit else -1.0,
                    "predicted_trend": "up",
                    "actual_trend": "up" if hit else "down",
                    "direction_correct": hit,
                    "target_reached": hit,
                    "time_to_target": 30.0 if hit else None,
                    "dd_before_target": 0.5 if hit else 2.0,
                    "maximum_profit": 3.0 if hit else 0.5,
                    "maximum_drawdown": 1.0 if hit else 2.5,
                    "absolute_error": 0.5 if hit else 1.5,
                    "prediction_error": 0.5,
                    "premium_error_pct": 2.0 if hit else 5.0,
                }
            )
            if i < 40:
                train_rows.append(
                    {
                        "trading_day": day,
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": i + 1,
                        "f1": f1,
                        "f2": float(i % 7),
                        "future_ltp_5m": 100.0 + i,
                    }
                )
        _write_train(data_dir, "parent_ds", train_rows)
        lab_path = os.path.join(tmp, "lab.db")
        _seed_lab(
            lab_path, data_dir=data_dir, parent_dataset="parent_ds", pred_rows=pred_rows
        )
        created = create_confidence_dataset(lab_path, data_dir=data_dir)
        self.assertTrue(created.get("ok"), created)
        trained = train_confidence_model(
            lab_path,
            "target_hit",
            parameters={"n_estimators": 40, "early_stopping_rounds": 10},
        )
        self.assertTrue(trained.get("ok"), trained)
        return lab_path, data_dir

    def test_comparison_baseline_and_target_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab(tmp)
            set_operating_threshold(lab_path, "target_hit", 0.70)
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)

            result = compute_confidence_filter_comparison(
                lab_path, evaluation_set="all"
            )
            self.assertTrue(result.get("ok"), result.get("error"))
            self.assertTrue(result.get("available"))
            self.assertEqual(result.get("evaluation_set"), "all")
            rows = result.get("rows") or []
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "None")
            self.assertTrue(rows[0]["is_baseline"])
            self.assertEqual(int(rows[0]["rows_left"]), 60)
            self.assertEqual(int(rows[0]["rows_removed"]), 0)

            target = next(r for r in rows if r["filter_key"] == "target_hit")
            self.assertEqual(int(target["rows_left"]), 30)
            self.assertEqual(int(target["rows_removed"]), 30)
            self.assertIsNotNone(target["hit_rate"])
            self.assertIsNotNone(target["profit_dd"])
            self.assertIsNotNone(target["delta_hit_pp"])
            self.assertIsNotNone(target["mae"])
            self.assertIsNotNone(target["premium_rmse"])

            # Seen-only scope
            seen = compute_confidence_filter_comparison(
                lab_path, evaluation_set="seen"
            )
            self.assertEqual(int(seen["baseline_rows"]), 40)
            self.assertEqual(seen["evaluation_set_label"], "Seen Only")
            unseen = compute_confidence_filter_comparison(
                lab_path, evaluation_set="unseen"
            )
            self.assertEqual(int(unseen["baseline_rows"]), 20)

    def test_dynamic_model_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab(tmp)
            with ModelLabStore(lab_path) as store:
                models = list_comparable_confidence_models(store)
            self.assertEqual(models, [])  # no inference yet

            set_operating_threshold(lab_path, "target_hit", 0.70)
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)

            with ModelLabStore(lab_path) as store:
                models = list_comparable_confidence_models(store)
            keys = [m["model_key"] for m in models]
            self.assertIn("target_hit", keys)
            self.assertNotIn("rr_1_1", keys)  # not inferred yet


if __name__ == "__main__":
    unittest.main()
