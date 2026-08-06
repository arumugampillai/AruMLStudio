"""Tests for verified GPU/CPU inference helpers + day timing format."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.training.inference_runtime import (
    InferenceRuntimeInfo,
    batch_predict_day,
    configure_prediction_model_for_inference,
    format_day_stage_timings,
    prefer_gpu_inference,
)


class FormatDayTimingsTests(unittest.TestCase):
    def test_format_matches_required_shape(self) -> None:
        text = format_day_stage_timings(
            {
                "load_master": 1.24,
                "load_timeline": 0.83,
                "prepare_matrix": 0.41,
                "predict": 0.29,
                "outcomes": 3.87,
                "sqlite_write": 1.15,
            },
            device_label="CUDA",
            algorithm="xgboost",
        )
        self.assertIn("Load Master Dataset", text)
        self.assertIn("1.24 s", text)
        self.assertIn("XGBoost Predict", text)
        self.assertIn("0.29 s (CUDA)", text)
        self.assertIn("CPU Outcome Metrics", text)
        self.assertIn("SQLite Write", text)
        self.assertIn("Total", text)
        self.assertIn("7.79 s", text)

    def test_format_includes_outcome_sub_breakdown(self) -> None:
        text = format_day_stage_timings(
            {
                "load_master": 1.0,
                "load_timeline": 1.0,
                "prepare_matrix": 0.1,
                "predict": 0.2,
                "outcomes": 10.0,
                "outcomes_read": 4.0,
                "outcomes_path": 3.5,
                "outcomes_build": 2.0,
                "outcomes_append": 0.1,
                "outcomes_checkpoint": 0.4,
                "sqlite_write": 1.0,
            },
            device_label="CUDA",
        )
        self.assertIn("Read row (iterrows)", text)
        self.assertIn("Path outcome calculation", text)
        self.assertIn("Build Python dict", text)
        self.assertIn("Append to list", text)
        self.assertIn("Checkpoint batching", text)
        self.assertIn("4.00 s", text)

    def test_per_prediction_outcome_table(self) -> None:
        from chain_replay_ml.training.inference_runtime import (
            format_per_prediction_outcome_timings,
            resolve_outcome_profile_rows,
        )

        self.assertEqual(resolve_outcome_profile_rows(row_limit=5000), 5000)
        self.assertEqual(resolve_outcome_profile_rows(row_limit=None), 0)
        text = format_per_prediction_outcome_timings(
            [3.12, 2.98, 18.45, 3.05, 2.91], warn_mult=3.0
        )
        self.assertIn("Prediction\tOutcome Time", text)
        self.assertIn("1\t3.12 ms", text)
        self.assertIn("3\t18.45 ms", text)
        self.assertIn(" WARN", text)

    def test_path_outcome_microprofile_report(self) -> None:
        from chain_replay_ml.training.inference_runtime import (
            format_path_outcome_microprofile,
        )

        samples = {
            "timeline_lookup": [3.0, 3.2, 2.8],
            "future_window_index": [0.1, 0.1, 0.1],
            "future_window_slice": [0.2, 0.2, 0.2],
            "future_tick_scan": [0.4, 0.5, 0.3],
            "mfe_mae_update": [0.05, 0.05, 0.05],
            "target_hit_detection": [0.3, 0.3, 0.3],
            "dd_before_target_update": [0.2, 0.2, 0.2],
            "timestamp_tracking": [0.01, 0.01, 0.01],
            "result_construction": [0.05, 0.05, 0.05],
        }
        text = format_path_outcome_microprofile(samples, n_predictions=3)
        self.assertIn("MOST EXPENSIVE", text)
        self.assertIn("Timeline array access", text)
        self.assertIn("%total", text)
        self.assertIn("If that op were free", text)


class PreferGpuEnvTests(unittest.TestCase):
    def test_force_cpu(self) -> None:
        with mock.patch.dict(os.environ, {"PREDICTION_INFER_DEVICE": "cpu"}, clear=False):
            self.assertFalse(prefer_gpu_inference())

    def test_default_prefers_gpu(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PREDICTION_INFER_DEVICE", "XGB_INFER_DEVICE")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(prefer_gpu_inference())


class ConfigureInferenceTests(unittest.TestCase):
    def test_force_cpu_on_real_xgb_model(self) -> None:
        from xgboost import XGBRegressor

        X = np.random.randn(40, 3).astype(np.float32)
        y = X.sum(axis=1)
        model = XGBRegressor(
            n_estimators=4, max_depth=2, tree_method="hist", device="cpu"
        )
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.ubj")
            model.save_model(path)
            loaded = XGBRegressor()
            loaded.load_model(path)
            # After load, device is typically unset — must configure explicitly
            info = configure_prediction_model_for_inference(
                loaded, "xgboost", prefer_gpu=False
            )
            self.assertEqual(info.device_label, "CPU")
            self.assertFalse(info.gpu_active)
            preds = batch_predict_day(loaded, pd.DataFrame(X), info=info)
            self.assertEqual(len(preds), len(X))

    def test_gpu_configure_or_cpu_fallback(self) -> None:
        from xgboost import XGBRegressor

        X = np.random.randn(40, 3).astype(np.float32)
        y = X.sum(axis=1)
        model = XGBRegressor(
            n_estimators=4, max_depth=2, tree_method="hist", device="cpu"
        )
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.ubj")
            model.save_model(path)
            loaded = XGBRegressor()
            loaded.load_model(path)
            info = configure_prediction_model_for_inference(
                loaded, "xgboost", prefer_gpu=True
            )
            self.assertIn(info.device_label, ("CUDA", "CPU"))
            if not info.gpu_active:
                self.assertIsNotNone(info.fallback_reason)
            preds = batch_predict_day(loaded, pd.DataFrame(X), info=info)
            self.assertEqual(len(preds), len(X))

    def test_batch_chunk_rows(self) -> None:
        info = InferenceRuntimeInfo(
            algorithm="xgboost",
            device_label="CPU",
            device_param="cpu",
            gpu_requested=False,
            gpu_active=False,
            predict_api="predict",
        )

        class _Stub:
            def predict(self, X):  # noqa: ANN001
                return np.asarray(X).sum(axis=1)

        X = pd.DataFrame(np.ones((25, 2), dtype=np.float32))
        preds = batch_predict_day(_Stub(), X, info=info, chunk_rows=10)
        self.assertEqual(len(preds), 25)
        np.testing.assert_allclose(preds, 2.0)


if __name__ == "__main__":
    unittest.main()
