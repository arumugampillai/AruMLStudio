"""Phase 2 Model Lab — prediction research dataset builder."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from chain_replay_ml.model_lab.prediction_builder import (
    build_prediction_dataset,
    prediction_dataset_status,
    validate_prediction_inputs,
)
from chain_replay_ml.model_lab.prediction_export import export_prediction_dataset
from chain_replay_ml.model_lab.prediction_schema import (
    PRED_STATUS_READY,
    compute_error_metrics,
    feature_column_map,
    horizon_sec_from_target,
    sanitize_feature_column,
)
from chain_replay_ml.model_lab.service import create_model_lab
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.training.inference_runtime import InferenceRuntimeInfo
from chain_replay_ml.tests.test_model_lab_phase1 import _sample_doc


class _FakeModel:
    feature_names_in_ = ["gamma", "ltp", "delta"]  # != snapshot order on purpose

    def predict(self, X):
        if hasattr(X, "columns"):
            # Require training column order
            assert list(X.columns) == list(self.feature_names_in_), list(X.columns)
            return (X["ltp"].astype(float) + 1.0).to_numpy()
        # batch_predict_day passes an aligned numpy matrix — ltp is column 1
        import numpy as np

        return np.asarray(X, dtype=float)[:, 1] + 1.0


_FAKE_INFO = InferenceRuntimeInfo(
    algorithm="xgboost",
    device_label="CPU",
    device_param="cpu",
    gpu_requested=False,
    gpu_active=False,
)


class ModelLabPhase2Tests(unittest.TestCase):
    def test_feature_column_sanitize(self) -> None:
        self.assertEqual(sanitize_feature_column("ltp"), "sf_ltp")
        self.assertEqual(sanitize_feature_column("spot-ema"), "sf_spot_ema")
        m = feature_column_map(["ltp", "gamma", "ltp"])
        self.assertEqual(len(set(m.values())), 2)

    def test_align_features_to_model_order(self) -> None:
        from chain_replay_ml.model_lab.prediction_schema import align_features_to_model

        class _M:
            feature_names_in_ = ["gamma", "ltp", "delta"]

        # Snapshot / ranking order differs from training order
        selected = ["delta", "gamma", "ltp"]
        self.assertEqual(align_features_to_model(selected, _M()), ["gamma", "ltp", "delta"])

        self.assertEqual(horizon_sec_from_target("future_ltp_5m"), 300.0)
        self.assertEqual(horizon_sec_from_target("future_ltp_30s"), 30.0)
        err = compute_error_metrics(predicted=12.0, actual=10.0, entry_ltp=9.0)
        self.assertEqual(err["absolute_error"], 2.0)
        self.assertEqual(err["prediction_error"], 2.0)
        self.assertEqual(err["direction_correct"], 1)

    def test_build_prediction_dataset_day_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            datasets = os.path.join(data_dir, "datasets")
            models = os.path.join(data_dir, "models", "Future_LTP_5m_WF_239f_XGB_0223_13")
            os.makedirs(datasets, exist_ok=True)
            os.makedirs(models, exist_ok=True)
            with open(os.path.join(models, "model.ubj"), "wb") as fh:
                fh.write(b"fake-model-bytes")

            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-01", "2026-07-01", "2026-07-02"],
                    "timestamp": [1.0, 2.0, 3.0],
                    "token": ["t1", "t2", "t3"],
                    "strike": [24000.0, 24100.0, 24200.0],
                    "option_type": ["CE", "PE", "CE"],
                    "expiry": ["2026-07-02", "2026-07-02", "2026-07-02"],
                    "market": ["NIFTY", "NIFTY", "NIFTY"],
                    "spot": [24050.0, 24050.0, 24100.0],
                    "ltp": [100.0, 80.0, 90.0],
                    "minutes_to_expiry": [120.0, 100.0, 80.0],
                    "delta": [0.5, -0.4, 0.3],
                    "gamma": [0.01, 0.02, 0.015],
                    "future_ltp_5m": [101.0, 78.0, 95.0],
                    "noise_feature": [1.0, 2.0, 3.0],
                }
            )
            parquet_path = os.path.join(datasets, "Master_NIFTY_239f.parquet")
            df.to_parquet(parquet_path, index=False)

            doc = _sample_doc()
            info = create_model_lab(data_dir, doc, research_dir=research)
            check = validate_prediction_inputs(data_dir, info)
            self.assertTrue(check["ok"], check.get("errors"))

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                result = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )

            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("row_count"), 3)
            self.assertEqual(result.get("trading_days"), 2)
            self.assertEqual(result.get("selected_feature_count"), 3)

            st = prediction_dataset_status(info.db_path)
            self.assertEqual(st.get("status"), PRED_STATUS_READY)
            self.assertEqual(st.get("row_count"), 3)

            with ModelLabStore(info.db_path) as store:
                cols = store.list_prediction_columns()
                self.assertIn("predicted_future_ltp", cols)
                self.assertIn("sf_ltp", cols)
                self.assertIn("sf_delta", cols)
                self.assertIn("sf_gamma", cols)
                # Never invent columns for non-selected features
                self.assertNotIn("sf_noise_feature", cols)
                n = store.prediction_row_count()
                self.assertEqual(n, 3)
                summary = store.read_prediction_summary()
                self.assertIsNotNone(summary)
                assert summary is not None
                self.assertEqual(summary["status"], PRED_STATUS_READY)
                info2 = store.read_info()
                self.assertEqual(info2.phase, 2)

            # Resume with no pending days → already complete (no reprocessing)
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ):
                resumed = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    resume=True,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(resumed.get("ok"), resumed)
            self.assertTrue(resumed.get("already_complete"))
            self.assertEqual(resumed.get("row_count"), 3)
            self.assertEqual(resumed.get("days_processed"), [])

            # Legacy block when resume is disabled
            blocked = build_prediction_dataset(
                data_dir,
                info.db_path,
                overwrite=False,
                resume=False,
                enrich_path_outcomes=False,
                workers=1,
                print_timing=False,
            )
            self.assertFalse(blocked.get("ok"))
            self.assertEqual(blocked.get("code"), "exists")

            # Multi-worker run must match single-worker hash / count
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                multi = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=True,
                    enrich_path_outcomes=False,
                    workers=3,
                    print_timing=False,
                )
            self.assertTrue(multi.get("ok"), multi)
            self.assertEqual(multi.get("row_count"), 3)
            self.assertEqual(multi.get("dataset_hash"), result.get("dataset_hash"))
            self.assertEqual(multi.get("workers_used"), 2)  # only 2 days → capped

            # Partial selected-day build + resume remaining
            with ModelLabStore(info.db_path) as store:
                store.clear_prediction_dataset()
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                part = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=True,
                    selected_days=["2026-07-01"],
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(part.get("ok"), part)
            self.assertEqual(part.get("row_count"), 2)
            self.assertEqual(part.get("days_processed"), ["2026-07-01"])
            with ModelLabStore(info.db_path) as store:
                days = {d["trading_day"]: d["status"] for d in store.list_build_days(store.read_info().lab_uuid)}
                self.assertEqual(days.get("2026-07-01"), "completed")

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                rest = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    resume=True,
                    selected_days=["2026-07-01", "2026-07-02"],
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(rest.get("ok"), rest)
            self.assertEqual(rest.get("days_processed"), ["2026-07-02"])
            self.assertEqual(rest.get("row_count"), 3)

            export_csv = os.path.join(tmp, "pred.csv")
            exp = export_prediction_dataset(info.db_path, export_csv)
            self.assertTrue(exp.get("ok"))
            self.assertTrue(os.path.isfile(export_csv))

    def test_build_writes_package_probability_columns(self) -> None:
        """One build pass fans out to available ladder members; missing stay NULL."""
        import json

        class _FakeClassifier:
            feature_names_in_ = ["delta", "gamma"]

            def predict_proba(self, X):  # noqa: N803
                import numpy as np

                p = np.full(len(X), 0.75)
                return np.column_stack([1.0 - p, p])

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            datasets = os.path.join(data_dir, "datasets")
            models = os.path.join(data_dir, "models", "Future_LTP_5m_WF_239f_XGB_0223_13")
            os.makedirs(datasets, exist_ok=True)
            os.makedirs(models, exist_ok=True)
            with open(os.path.join(models, "model.ubj"), "wb") as fh:
                fh.write(b"fake-model-bytes")

            # A trained +3% ladder classifier on the same dataset
            cls_dir = os.path.join(data_dir, "models", "label_up_3pct_5m_test")
            os.makedirs(cls_dir, exist_ok=True)
            with open(os.path.join(cls_dir, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "target": "label_up_3pct_5m",
                        "dataset": "Master_NIFTY_239f",
                        "prediction_type": "binary",
                        "algorithm": "xgboost",
                        "features": ["delta", "gamma"],
                    },
                    fh,
                )
            with open(os.path.join(cls_dir, "model.ubj"), "wb") as fh:
                fh.write(b"fake-classifier-bytes")

            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-01", "2026-07-01"],
                    "timestamp": [1.0, 2.0],
                    "token": ["t1", "t2"],
                    "strike": [24000.0, 24100.0],
                    "option_type": ["CE", "PE"],
                    "expiry": ["2026-07-02", "2026-07-02"],
                    "market": ["NIFTY", "NIFTY"],
                    "spot": [24050.0, 24050.0],
                    "ltp": [100.0, 80.0],
                    "minutes_to_expiry": [120.0, 100.0],
                    "delta": [0.5, -0.4],
                    "gamma": [0.01, 0.02],
                    "future_ltp_5m": [101.0, 78.0],
                }
            )
            parquet_path = os.path.join(datasets, "Master_NIFTY_239f.parquet")
            df.to_parquet(parquet_path, index=False)

            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            def _fake_member_loader(model_path, algorithm):
                return _FakeClassifier(), 0.0, False

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.training.model_runtime.load_prediction_model_cached",
                side_effect=_fake_member_loader,
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                result = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("row_count"), 2)

            with ModelLabStore(info.db_path) as store:
                rows = store.conn.execute(
                    """
                    SELECT pred_prob_up_3pct_5m, pred_prob_up_2pct_5m,
                           pred_prob_up_gt6pct_5m
                    FROM prediction_dataset
                    """
                ).fetchall()
            self.assertEqual(len(rows), 2)
            for available, missing_2, missing_gt6 in rows:
                self.assertAlmostEqual(available, 0.75)
                self.assertIsNone(missing_2)
                self.assertIsNone(missing_gt6)


if __name__ == "__main__":
    unittest.main()
