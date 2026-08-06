"""Operating Threshold + Confidence Inference over Prediction Dataset."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.writer import datasets_dir
from chain_replay_ml.model_lab.confidence import (
    create_confidence_dataset,
    set_operating_threshold,
)
from chain_replay_ml.model_lab.confidence_inference import (
    PRED_COL,
    THRESHOLD_COL,
    clear_confidence_inference,
    inference_status,
    resolve_operating_threshold,
    run_confidence_inference,
)
from chain_replay_ml.model_lab.confidence_manifest import (
    mark_inference_out_of_date,
    read_manifest,
)
from chain_replay_ml.model_lab.confidence_train import train_confidence_model
from chain_replay_ml.model_lab.prediction_schema import DATASET_TYPE_SEEN
from chain_replay_ml.model_lab.research_dashboard import compute_research_dashboard
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
) -> None:
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
                "package_dir": {
                    "path": os.path.join(data_dir, "models", "Future_LTP_5m_Test")
                }
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
            day_dataset_types={d: DATASET_TYPE_SEEN for d in days},
        )


class _FakeBooster:
    def __init__(self, thr_signal: float = 0.5) -> None:
        self.thr_signal = thr_signal

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Higher f1 → higher probability
        vals = pd.to_numeric(X["f1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(vals, 0.0, 1.0)


class ConfidenceInferenceTests(unittest.TestCase):
    def _lab_with_trained_target_hit(self, tmp: str) -> tuple[str, str]:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_rows = []
        pred_rows = []
        for i in range(60):
            day = "2024-01-02" if i < 40 else "2024-01-03"
            mid = i + 1
            f1 = 0.9 if i % 2 == 0 else 0.1
            hit = 1 if f1 > 0.5 else 0
            row = {
                "lab_uuid": "u1",
                "prediction_id": f"p{i}",
                "trading_day": day,
                "timestamp": float(1000 + i),
                "token": f"T{i}",
                "master_row_id": mid,
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
                "maximum_profit": 2.0 if hit else 0.2,
                "maximum_drawdown": 0.5,
                "absolute_error": 0.1,
                "prediction_error": 0.1,
                "premium_error_pct": 1.0,
            }
            pred_rows.append(row)
            if i < 40:
                train_rows.append(
                    {
                        "trading_day": day,
                        "timestamp": float(1000 + i),
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "f1": f1,
                        "f2": float(i % 7),
                        "future_ltp_5m": 100.0 + i,
                    }
                )
        _write_train(data_dir, "parent_ds", train_rows)
        lab_path = os.path.join(tmp, "lab.db")
        _seed_lab(
            lab_path,
            data_dir=data_dir,
            parent_dataset="parent_ds",
            pred_rows=pred_rows,
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

    def test_resolve_threshold_requires_persisted_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, _ = self._lab_with_trained_target_hit(tmp)
            res = resolve_operating_threshold(lab_path)
            self.assertFalse(res.get("ok"))
            self.assertIn("No Operating Threshold", str(res.get("error") or ""))

            set_operating_threshold(lab_path, "target_hit", 0.70)
            res2 = resolve_operating_threshold(lab_path)
            self.assertTrue(res2.get("ok"))
            self.assertAlmostEqual(float(res2["threshold"]), 0.70, places=4)

    def test_changing_threshold_marks_inference_out_of_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, _ = self._lab_with_trained_target_hit(tmp)
            set_operating_threshold(lab_path, "target_hit", 0.60)
            doc = read_manifest(lab_path)
            doc.setdefault("inference", {})["target_hit"] = {
                "status": "completed",
                "rows": 10,
                "positive": 5,
                "negative": 5,
                "threshold": 0.60,
            }
            from chain_replay_ml.model_lab.confidence_manifest import write_manifest

            write_manifest(lab_path, doc)

            saved = set_operating_threshold(lab_path, "target_hit", 0.75)
            self.assertTrue(saved.get("inference_stale"))
            inf = (read_manifest(lab_path).get("inference") or {}).get("target_hit") or {}
            self.assertEqual(inf.get("status"), "out_of_date")

    def test_inference_uses_model_threshold_not_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab_with_trained_target_hit(tmp)
            set_operating_threshold(lab_path, "target_hit", 0.70)

            # Fail if someone hard-codes 0.50 as the decision threshold
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                result = run_confidence_inference(
                    lab_path, data_dir=data_dir, batch_size=25
                )
            self.assertTrue(result.get("ok"), result)
            self.assertAlmostEqual(float(result["threshold"]), 0.70, places=4)

            with ModelLabStore(lab_path) as store:
                rows = store.conn.execute(
                    f"""
                    SELECT "{PRED_COL}", "{THRESHOLD_COL}"
                    FROM prediction_dataset
                    """
                ).fetchall()
            self.assertTrue(rows)
            self.assertTrue(all(r[0] in (0, 1) for r in rows))
            self.assertTrue(all(abs(float(r[1]) - 0.70) < 1e-9 for r in rows))

            # Probabilities from FakeBooster are f1; at thr=0.70, only f1>=0.70 → 1
            with ModelLabStore(lab_path) as store:
                pos = store.conn.execute(
                    f'SELECT COUNT(*) FROM prediction_dataset WHERE "{PRED_COL}" = 1'
                ).fetchone()[0]
                # Even rows have f1=0.9
                self.assertEqual(int(pos), 30)

            st = inference_status(lab_path)
            self.assertEqual(st.get("status"), "completed")
            self.assertAlmostEqual(float(st["threshold"]), 0.70, places=4)

    def test_inference_refuses_without_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab_with_trained_target_hit(tmp)
            result = run_confidence_inference(lab_path, data_dir=data_dir)
            self.assertFalse(result.get("ok"))
            self.assertIn("Operating Threshold", str(result.get("error") or ""))

    def test_clear_and_stale_on_mark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab_with_trained_target_hit(tmp)
            set_operating_threshold(lab_path, "target_hit", 0.55)
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)
            clear_confidence_inference(lab_path)
            st = inference_status(lab_path)
            self.assertEqual(st.get("status"), "not_run")
            with ModelLabStore(lab_path) as store:
                nulls = store.conn.execute(
                    f'SELECT COUNT(*) FROM prediction_dataset WHERE "{PRED_COL}" IS NOT NULL'
                ).fetchone()[0]
            self.assertEqual(int(nulls), 0)

            # Re-run then mark out of date
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)
            mark_inference_out_of_date(lab_path, reason="Prediction Dataset rebuilt")
            st2 = inference_status(lab_path)
            self.assertEqual(st2.get("status"), "out_of_date")

    def test_dashboard_confidence_filter_sql_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab_with_trained_target_hit(tmp)
            set_operating_threshold(lab_path, "target_hit", 0.70)
            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)

            full = compute_research_dashboard(lab_path, data_dir=data_dir)
            hit1 = compute_research_dashboard(
                lab_path, data_dir=data_dir, confidence_filter="target_hit_1"
            )
            hit0 = compute_research_dashboard(
                lab_path,
                data_dir=data_dir,
                confidence_classifier="target_hit",
                confidence_prediction=0,
            )
            self.assertTrue(full.get("available"))
            self.assertEqual(int(full["total_predictions"]), 60)
            self.assertEqual(int(hit1["total_predictions"]), 30)
            self.assertEqual(int(hit0["total_predictions"]), 30)
            meta = hit1.get("confidence_filter_meta") or {}
            self.assertEqual(meta.get("value"), 1)
            self.assertAlmostEqual(float(meta.get("threshold")), 0.70, places=4)

            # Mark second day Unseen and verify Evaluation Set scoping
            from chain_replay_ml.model_lab.prediction_schema import DATASET_TYPE_UNSEEN
            from chain_replay_ml.model_lab.store import ModelLabStore

            with ModelLabStore(lab_path) as store:
                store.apply_day_dataset_types(
                    "u1", {"2024-01-03": DATASET_TYPE_UNSEEN}
                )

            seen = compute_research_dashboard(
                lab_path, data_dir=data_dir, evaluation_set="seen"
            )
            unseen = compute_research_dashboard(
                lab_path, data_dir=data_dir, evaluation_set="unseen"
            )
            self.assertEqual(int(seen["total_predictions"]), 40)
            self.assertEqual(int(unseen["total_predictions"]), 20)
            self.assertEqual(seen.get("evaluation_set"), "seen")
            self.assertEqual(unseen.get("evaluation_set_label"), "Unseen Only")

            # Combined: Seen Only + Target Hit = 1
            seen_hit1 = compute_research_dashboard(
                lab_path,
                data_dir=data_dir,
                confidence_classifier="target_hit",
                confidence_prediction=1,
                evaluation_set="seen",
            )
            self.assertLessEqual(
                int(seen_hit1["total_predictions"]), int(seen["total_predictions"])
            )
            effect = seen_hit1.get("confidence_filter_effect") or {}
            self.assertTrue(effect.get("available"))
            self.assertEqual(int(effect.get("rows_before") or 0), 40)


    def test_filter_gate_and_action_label(self) -> None:
        from chain_replay_ml.model_lab.confidence_inference import (
            target_hit_filter_available,
        )

        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir = self._lab_with_trained_target_hit(tmp)
            gate0 = target_hit_filter_available(lab_path)
            self.assertFalse(gate0.get("available"))

            set_operating_threshold(lab_path, "target_hit", 0.70)
            st = inference_status(lab_path, "target_hit")
            self.assertIn("Run Inference", st.get("action_label") or "")

            with mock.patch(
                "chain_replay_ml.model_lab.confidence_inference._load_classifier",
                return_value=_FakeBooster(),
            ):
                run_confidence_inference(lab_path, data_dir=data_dir, batch_size=20)

            gate1 = target_hit_filter_available(lab_path)
            self.assertTrue(gate1.get("available"))
            st2 = inference_status(lab_path, "target_hit")
            self.assertIn("Re-run", st2.get("action_label") or "")

            set_operating_threshold(lab_path, "target_hit", 0.80)
            st3 = inference_status(lab_path, "target_hit")
            self.assertEqual(st3.get("status"), "out_of_date")
            self.assertIn("Update Inference", st3.get("action_label") or "")


if __name__ == "__main__":
    unittest.main()
