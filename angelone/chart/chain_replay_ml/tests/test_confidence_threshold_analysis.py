"""Threshold Analysis is required for every Confidence Model after train."""

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
)
from chain_replay_ml.model_lab.confidence_dataset import confidence_dataset_paths
from chain_replay_ml.model_lab.confidence_manifest import (
    CONFIDENCE_TARGETS,
    read_manifest,
    set_operating_threshold,
    write_manifest,
)
from chain_replay_ml.model_lab.confidence_train import (
    evaluate_confidence_model,
    model_has_threshold_analysis,
    train_confidence_model,
)
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    compute_rr_hit_labels,
)
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.training.evaluator import DEFAULT_THRESHOLD_SWEEP


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


def _seed_lab(path: str, *, data_dir: str, parent_dataset: str, pred_rows: list[dict]) -> None:
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
            trading_days=1,
            target_column="future_ltp_5m",
            parent_dataset=parent_dataset,
            parent_model_name="Future_LTP_5m_Test",
            created_at="2026-07-16T10:00:00+00:00",
        )
        store.insert_prediction_rows(pred_rows)
        store.ensure_build_days(
            "u1",
            ["2026-07-01"],
            day_dataset_types={"2026-07-01": DATASET_TYPE_SEEN},
        )


def _build_lab(tmp: str) -> str:
    data_dir = os.path.join(tmp, "data")
    os.makedirs(data_dir, exist_ok=True)
    train_rows = []
    pred_rows = []
    for i in range(48):
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
        rr = compute_rr_hit_labels(
            target_reached=hit, maximum_profit=profit, maximum_drawdown=3.0
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
                "maximum_drawdown": 3.0,
                **rr,
            }
        )
    _write_train(data_dir, "MS_ta", train_rows)
    lab_path = os.path.join(tmp, "lab.db")
    _seed_lab(lab_path, data_dir=data_dir, parent_dataset="MS_ta", pred_rows=pred_rows)
    created = create_confidence_dataset(lab_path, data_dir=data_dir)
    assert created["ok"], created.get("error")
    return lab_path


class ConfidenceThresholdAnalysisTests(unittest.TestCase):
    def test_every_target_persists_threshold_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path = _build_lab(tmp)
            expected = [round(float(t), 2) for t in DEFAULT_THRESHOLD_SWEEP]
            paths = confidence_dataset_paths(lab_path)

            from chain_replay_ml.model_lab.target_spec import TARGET_SPEC_BY_KEY

            for spec in CONFIDENCE_TARGETS:
                key = spec["key"]
                # Replay-Based columns require a Confidence Label Run first.
                if TARGET_SPEC_BY_KEY.get(key) and TARGET_SPEC_BY_KEY[key].is_replay_based:
                    continue
                trained = train_confidence_model(
                    lab_path,
                    key,
                    parameters={"n_estimators": 40, "early_stopping_rounds": 10},
                )
                self.assertTrue(trained["ok"], f"{key}: {trained.get('error')}")
                self.assertTrue(trained.get("has_threshold_analysis"))
                ta = trained.get("threshold_analysis") or []
                self.assertEqual(len(ta), len(expected), key)
                self.assertEqual([row["threshold"] for row in ta], expected, key)

                pkg = os.path.join(paths["models_dir"], key)
                self.assertTrue(
                    os.path.isfile(os.path.join(pkg, "eval_predictions.npz")), key
                )
                with open(os.path.join(pkg, "metrics.json"), encoding="utf-8") as fh:
                    meta = json.load(fh)
                self.assertEqual(len(meta.get("threshold_analysis") or []), len(expected), key)

                evaluated = evaluate_confidence_model(lab_path, key)
                self.assertTrue(evaluated["ok"], key)
                self.assertFalse(evaluated.get("is_legacy"), key)
                self.assertTrue(evaluated.get("has_threshold_analysis"), key)
                self.assertEqual(
                    len(evaluated.get("threshold_analysis") or []), len(expected), key
                )

                set_operating_threshold(lab_path, key, 0.70)
                man = read_manifest(lab_path)
                self.assertEqual(man["models"][key]["operating_threshold"], 0.70)

            st = confidence_status(lab_path)
            for m in st["models"]:
                key = m["key"]
                if TARGET_SPEC_BY_KEY.get(key) and TARGET_SPEC_BY_KEY[key].is_replay_based:
                    continue
                self.assertFalse(m.get("is_legacy"), key)
                self.assertEqual(m.get("status_display"), "Ready", key)

    def test_legacy_model_missing_threshold_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path = _build_lab(tmp)
            trained = train_confidence_model(
                lab_path,
                "rr_1_1",
                parameters={"n_estimators": 40, "early_stopping_rounds": 10},
            )
            self.assertTrue(trained["ok"], trained.get("error"))

            # Simulate older package: strip Threshold Analysis artifacts
            doc = read_manifest(lab_path)
            entry = doc["models"]["rr_1_1"]
            entry.pop("threshold_analysis", None)
            entry.pop("has_threshold_analysis", None)
            metrics = dict(entry.get("metrics") or {})
            metrics.pop("threshold_analysis", None)
            entry["metrics"] = metrics
            pkg = entry["package_dir"]
            pred_path = os.path.join(pkg, "eval_predictions.npz")
            if os.path.isfile(pred_path):
                os.remove(pred_path)
            metrics_path = os.path.join(pkg, "metrics.json")
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            meta.pop("threshold_analysis", None)
            meta.pop("has_threshold_analysis", None)
            if isinstance(meta.get("metrics"), dict):
                meta["metrics"].pop("threshold_analysis", None)
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            write_manifest(lab_path, doc)

            self.assertFalse(model_has_threshold_analysis(lab_path, "rr_1_1"))
            evaluated = evaluate_confidence_model(lab_path, "rr_1_1")
            self.assertTrue(evaluated["ok"])
            self.assertTrue(evaluated["is_legacy"])
            self.assertFalse(evaluated["has_threshold_analysis"])
            self.assertEqual(evaluated.get("threshold_analysis") or [], [])
            self.assertIn("Legacy", evaluated.get("legacy_message") or "")

            st = confidence_status(lab_path)
            rr = next(m for m in st["models"] if m["key"] == "rr_1_1")
            self.assertTrue(rr["is_legacy"])
            self.assertEqual(rr["status_display"], "Legacy")

            with self.assertRaises(ValueError) as ctx:
                set_operating_threshold(lab_path, "rr_1_1", 0.70)
            self.assertIn("Legacy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
