"""Confidence Dataset Mapping Validation — join + RR label check."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.writer import datasets_dir
from chain_replay_ml.model_lab.confidence import create_confidence_dataset
from chain_replay_ml.model_lab.confidence_mapping_validation import (
    validate_confidence_dataset_mapping,
)
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    compute_rr_hit_labels,
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


def _seed_lab(
    path: str,
    *,
    data_dir: str,
    parent_dataset: str,
    pred_rows: list[dict],
) -> None:
    with ModelLabStore(path) as store:
        store._ensure_schema()
        store.ensure_prediction_schema()
        store.write_info(
            lab_uuid="u1",
            lab_id="lab1",
            lab_name="Map Lab",
            parent_model_id="m1",
            parent_model_name="Future_LTP_5m_Map",
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
                "package_dir": {
                    "path": os.path.join(data_dir, "models", "Future_LTP_5m_Map")
                }
            },
        )
        store.write_prediction_summary(
            lab_uuid="u1",
            status="ready",
            row_count=len(pred_rows),
            trading_days=1,
            target_column="future_ltp_5m",
            parent_dataset=parent_dataset,
        )
        store.insert_prediction_rows(pred_rows)
        store.ensure_build_days(
            "u1",
            ["2026-07-01"],
            day_dataset_types={"2026-07-01": DATASET_TYPE_SEEN},
        )


class ConfidenceMappingValidationTests(unittest.TestCase):
    def test_mapping_verified_for_rr_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            # Build rows covering RR 1:2 / 1:3 / 1:4 positives
            specs = [
                (1, 12.0, 3.0),  # RR 4
                (1, 9.0, 3.0),  # RR 3
                (1, 6.0, 3.0),  # RR 2
                (0, 1.0, 3.0),  # miss
            ]
            for i, (hit, profit, dd) in enumerate(specs):
                for copy in range(10):  # enough rows for sampling
                    mid = i * 10 + copy + 1
                    rr = compute_rr_hit_labels(
                        target_reached=hit,
                        maximum_profit=profit,
                        maximum_drawdown=dd,
                    )
                    train_rows.append(
                        {
                            "trading_day": "2026-07-01",
                            "timestamp": float(1000 + mid),
                            "token": f"T{mid}",
                            "master_row_id": mid,
                            "f1": float(mid),
                            "f2": float(mid * 2),
                            "future_ltp_5m": 100.0,
                        }
                    )
                    pred_rows.append(
                        {
                            "lab_uuid": "u1",
                            "prediction_id": f"p{mid}",
                            "trading_day": "2026-07-01",
                            "timestamp": float(1000 + mid),
                            "token": f"T{mid}",
                            "master_row_id": mid,
                            "target_reached": hit,
                            "maximum_profit": profit,
                            "maximum_drawdown": dd,
                            **rr,
                        }
                    )
            _write_train(data_dir, "MS_map", train_rows)
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_map",
                pred_rows=pred_rows,
            )
            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))

            report = validate_confidence_dataset_mapping(lab_path, seed=7)
            self.assertTrue(report["ok"], report.get("error"))
            self.assertEqual(report["join_keys"], ["master_row_id"])
            by_key = {s["key"]: s for s in report["samples"]}
            for key in ("rr_1_1", "rr_2_3", "rr_1_2", "rr_1_3", "rr_1_4"):
                sample = by_key[key]
                self.assertTrue(sample["available"], sample.get("message"))
                self.assertTrue(sample["mapping_ok"], sample.get("message"))
                fields = sample["fields"]
                self.assertEqual(fields["dataset_label"], 1)
                self.assertEqual(fields["prediction_label"], 1)
                self.assertIsNotNone(fields["profit_dd_ratio"])
                self.assertIsNotNone(fields["dataset_row_id"])
                self.assertIsNotNone(fields["prediction_row_id"])
                self.assertEqual(fields["dataset_row_id"], fields["prediction_row_id"])
                self.assertTrue(fields["row_ids_match"])
                # Payload must not expose feature columns
                self.assertNotIn("f1", fields)
                self.assertNotIn("f2", fields)
            self.assertTrue(report["all_mapping_ok"])

    def test_no_positive_sample_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            for i in range(20):
                # Only RR 1:2 positives (profit=6, dd=3); never RR 1:4
                hit = 1
                profit, dd = 6.0, 3.0
                rr = compute_rr_hit_labels(
                    target_reached=hit, maximum_profit=profit, maximum_drawdown=dd
                )
                self.assertEqual(rr["rr_1_2_hit"], 1)
                self.assertEqual(rr["rr_1_4_hit"], 0)
                train_rows.append(
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": i + 1,
                        "f1": float(i),
                        "f2": float(i * 2),
                        "future_ltp_5m": 100.0,
                    }
                )
                pred_rows.append(
                    {
                        "lab_uuid": "u1",
                        "prediction_id": f"p{i}",
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": i + 1,
                        "target_reached": hit,
                        "maximum_profit": profit,
                        "maximum_drawdown": dd,
                        **rr,
                    }
                )
            _write_train(data_dir, "MS_nopos", train_rows)
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_nopos",
                pred_rows=pred_rows,
            )
            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))

            report = validate_confidence_dataset_mapping(lab_path, seed=1)
            by_key = {s["key"]: s for s in report["samples"]}
            self.assertTrue(by_key["rr_1_2"]["available"])
            self.assertTrue(by_key["rr_1_2"]["mapping_ok"])
            self.assertFalse(by_key["rr_1_4"]["available"])
            self.assertIn("No positive sample available for RR 1:4", by_key["rr_1_4"]["message"])

    def test_missing_confidence_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path = os.path.join(tmp, "empty.db")
            with ModelLabStore(lab_path) as store:
                store._ensure_schema()
                store.ensure_prediction_schema()
            report = validate_confidence_dataset_mapping(lab_path)
            self.assertFalse(report["ok"])
            self.assertIn("Confidence Dataset not found", report["error"])


if __name__ == "__main__":
    unittest.main()
