"""Confidence Model subsystem — lab-local dataset + classifiers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.writer import datasets_dir
from chain_replay_ml.model_lab.confidence import (
    confidence_status,
    create_confidence_dataset,
    link_trained_confidence_model,
    read_confidence_link,
)
from chain_replay_ml.model_lab.confidence_dataset import confidence_dataset_paths
from chain_replay_ml.model_lab.confidence_manifest import (
    delete_confidence_model,
    read_manifest,
    set_active_model,
)
from chain_replay_ml.model_lab.confidence_train import (
    evaluate_confidence_model,
    train_confidence_model,
)
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    DATASET_TYPE_UNSEEN,
    compute_rr_hit_labels,
)
from chain_replay_ml.model_lab.store import ModelLabStore


def _write_train(data_dir: str, name: str, rows: list[dict]) -> None:
    out = datasets_dir(data_dir)
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
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
    day_types: dict[str, str] | None = None,
) -> None:
    with ModelLabStore(path) as store:
        store._ensure_schema()
        store.ensure_prediction_schema()
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
        )
        store.insert_prediction_rows(pred_rows)
        days = sorted({str(r.get("trading_day")) for r in pred_rows if r.get("trading_day")})
        store.ensure_build_days(
            "u1",
            days,
            day_dataset_types=day_types or {d: DATASET_TYPE_SEEN for d in days},
        )


class ConfidenceSubsystemTests(unittest.TestCase):
    def test_create_dataset_seen_only_and_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            for i in range(40):
                mid = i + 1
                train_rows.append(
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "f1": float(i % 5),
                        "f2": float((i * 3) % 7),
                        "future_ltp_5m": 100.0 + i,
                    }
                )
                hit = 1 if i % 2 == 0 else 0
                profit = 12.0 if i % 3 == 0 else 6.0
                dd = 3.0
                rr = compute_rr_hit_labels(
                    target_reached=hit, maximum_profit=profit, maximum_drawdown=dd
                )
                pred_rows.append(
                    {
                        "lab_uuid": "u1",
                        "prediction_id": f"p{i}",
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "target_reached": hit,
                        "maximum_profit": profit,
                        "maximum_drawdown": dd,
                        **rr,
                    }
                )
            # Unseen day — must be ignored
            rr_u = compute_rr_hit_labels(
                target_reached=1, maximum_profit=20.0, maximum_drawdown=2.0
            )
            pred_rows.append(
                {
                    "lab_uuid": "u1",
                    "prediction_id": "unseen",
                    "trading_day": "2026-07-15",
                    "timestamp": 9999.0,
                    "token": "U",
                    "master_row_id": 9999,
                    "target_reached": 1,
                    "maximum_profit": 20.0,
                    "maximum_drawdown": 2.0,
                    **rr_u,
                }
            )
            _write_train(data_dir, "MS_conf_src", train_rows)
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_conf_src",
                pred_rows=pred_rows,
                day_types={
                    "2026-07-01": DATASET_TYPE_SEEN,
                    "2026-07-15": DATASET_TYPE_UNSEEN,
                },
            )

            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))
            self.assertEqual(created["row_count"], 40)
            self.assertEqual(created["report"]["unseen_prediction_rows"], 1)
            paths = confidence_dataset_paths(lab_path)
            self.assertTrue(os.path.isfile(paths["parquet"]))
            # Must not appear in Dataset Registry folder as a new global export name
            self.assertFalse(
                os.path.isfile(os.path.join(datasets_dir(data_dir), "confidence_dataset.parquet"))
            )

            enriched = pd.read_parquet(paths["parquet"])
            self.assertIn("rr_1_2_hit", enriched.columns)
            self.assertIn("target_reached", enriched.columns)
            self.assertIn("f1", enriched.columns)

            from chain_replay_ml.model_lab.confidence_manifest import CONFIDENCE_TARGETS

            st = confidence_status(lab_path, data_dir=data_dir)
            self.assertEqual(st["confidence_dataset_status"], "ready")
            self.assertEqual(len(st["models"]), len(CONFIDENCE_TARGETS))

            trained = train_confidence_model(
                lab_path,
                "rr_1_2",
                parameters={"n_estimators": 40, "early_stopping_rounds": 10},
            )
            self.assertTrue(trained["ok"], trained.get("error"))
            self.assertIn("accuracy_pct", trained["metrics"])

            evaluated = evaluate_confidence_model(lab_path, "rr_1_2")
            self.assertTrue(evaluated["ok"])
            self.assertEqual(evaluated["label"], "RR 1:2")

            set_active_model(lab_path, "rr_1_2")
            man = read_manifest(lab_path)
            self.assertEqual(man["active_model_key"], "rr_1_2")

            delete_confidence_model(lab_path, "rr_1_2")
            man2 = read_manifest(lab_path)
            self.assertEqual(man2["models"]["rr_1_2"]["status"], "not_created")

    def test_trim_unmatched_training_rows(self) -> None:
        """Missing training rows (e.g. last-horizon) are dropped, not aborted."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            for i in range(20):
                mid = i + 1
                train_rows.append(
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "f1": float(i),
                        "f2": float(i * 2),
                        "future_ltp_5m": 100.0 + i,
                    }
                )
                if i < 17:  # 3 training rows have no Seen prediction
                    hit = 1 if i % 2 == 0 else 0
                    rr = compute_rr_hit_labels(
                        target_reached=hit, maximum_profit=9.0, maximum_drawdown=3.0
                    )
                    pred_rows.append(
                        {
                            "lab_uuid": "u1",
                            "prediction_id": f"p{i}",
                            "trading_day": "2026-07-01",
                            "timestamp": float(1000 + i),
                            "token": f"T{i}",
                            "master_row_id": mid,
                            "target_reached": hit,
                            "maximum_profit": 9.0,
                            "maximum_drawdown": 3.0,
                            **rr,
                        }
                    )
            _write_train(data_dir, "MS_trim", train_rows)
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_trim",
                pred_rows=pred_rows,
            )
            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))
            self.assertEqual(created["row_count"], 17)
            self.assertEqual(created["report"]["missing"], 0)
            self.assertEqual(created["report"]["dropped_unmatched"], 3)
            self.assertEqual(created["report"]["dataset_rows"], 20)
            self.assertEqual(created["report"]["dataset_rows_after_trim"], 17)

    def test_null_labels_warn_and_continue(self) -> None:
        """All-null label columns are omitted with a warning; build still succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            for i in range(40):
                mid = i + 1
                hit = 1 if i % 2 == 0 else 0
                rr = compute_rr_hit_labels(
                    target_reached=hit, maximum_profit=12.0, maximum_drawdown=3.0
                )
                # Simulate legacy prediction rows before RR 1:1 / 2:3 existed
                rr["rr_1_1_hit"] = None
                rr["rr_2_3_hit"] = None
                train_rows.append(
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "f1": float(i),
                        "f2": float(i * 2),
                        "future_ltp_5m": 100.0 + i,
                    }
                )
                pred_rows.append(
                    {
                        "lab_uuid": "u1",
                        "prediction_id": f"p{i}",
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "target_reached": hit,
                        "maximum_profit": 12.0,
                        "maximum_drawdown": 3.0,
                        **rr,
                    }
                )
            _write_train(data_dir, "MS_null_lbl", train_rows)
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_null_lbl",
                pred_rows=pred_rows,
            )
            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))
            # Null RR 1:1 / 2:3 are backfilled from profit/dd when available
            self.assertIn("rr_1_1_hit", created["labels"])
            self.assertIn("rr_2_3_hit", created["labels"])
            self.assertIn("rr_1_2_hit", created["labels"])
            self.assertIn(
                "rr_1_1_hit",
                created["report"].get("rr_labels_backfilled_columns") or [],
            )

    def test_inherits_regression_selected_features_only(self) -> None:
        """Confidence uses regression subset, not the full training export matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = []
            pred_rows = []
            for i in range(40):
                hit = 1 if i % 2 == 0 else 0
                rr = compute_rr_hit_labels(
                    target_reached=hit, maximum_profit=12.0, maximum_drawdown=3.0
                )
                train_rows.append(
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": i + 1,
                        "f1": float(i),
                        "f2": float(i * 2),
                        "extra_a": float(i * 3),  # not in regression selection
                        "extra_b": float(i * 4),
                        "future_ltp_5m": 100.0 + i,
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
                        "maximum_profit": 12.0,
                        "maximum_drawdown": 3.0,
                        **rr,
                    }
                )
            # Training export metadata lists ALL columns (legacy / full matrix)
            out = datasets_dir(data_dir)
            os.makedirs(out, exist_ok=True)
            pd.DataFrame(train_rows).to_parquet(
                os.path.join(out, "MS_full.parquet"), index=False
            )
            with open(os.path.join(out, "MS_full.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": "MS_full",
                        "feature_columns": ["f1", "f2", "extra_a", "extra_b"],
                        "selected_features": ["f1", "f2", "extra_a", "extra_b"],
                        "feature_count": 4,
                        "prediction_target_columns": ["future_ltp_5m"],
                        "row_count": len(train_rows),
                    },
                    fh,
                    indent=2,
                )
            lab_path = os.path.join(tmp, "lab.db")
            _seed_lab(
                lab_path,
                data_dir=data_dir,
                parent_dataset="MS_full",
                pred_rows=pred_rows,
            )
            created = create_confidence_dataset(lab_path, data_dir=data_dir)
            self.assertTrue(created["ok"], created.get("error"))
            self.assertEqual(created["feature_count"], 2)
            self.assertEqual(created["features"], ["f1", "f2"])

            paths = confidence_dataset_paths(lab_path)
            with open(paths["json"], encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["feature_source"], "regression_model")
            self.assertEqual(meta["feature_columns"], ["f1", "f2"])

            enriched = pd.read_parquet(paths["parquet"])
            self.assertIn("f1", enriched.columns)
            self.assertIn("f2", enriched.columns)
            self.assertNotIn("extra_a", enriched.columns)
            self.assertNotIn("extra_b", enriched.columns)

            st = confidence_status(lab_path, data_dir=data_dir)
            self.assertEqual(st["feature_source"], "Regression Model")
            self.assertEqual(st["feature_count"], 2)

            trained = train_confidence_model(
                lab_path,
                "target_hit",
                parameters={"n_estimators": 40, "early_stopping_rounds": 10},
            )
            self.assertTrue(trained["ok"], trained.get("error"))
            self.assertEqual(trained.get("feature_count"), 2)
            pkg = os.path.join(paths["models_dir"], "target_hit")
            with open(os.path.join(pkg, "metrics.json"), encoding="utf-8") as fh:
                mmeta = json.load(fh)
            self.assertEqual(mmeta["features"], ["f1", "f2"])
            self.assertEqual(mmeta["feature_source"], "regression_model")

    def test_legacy_link_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            open(lab, "wb").close()
            link_trained_confidence_model(lab, model_name="hit_demo")
            doc = read_confidence_link(lab)
            assert doc is not None
            self.assertEqual(doc["active_model_key"], "target_hit")
            self.assertEqual(doc["status"], "ready")


if __name__ == "__main__":
    unittest.main()
