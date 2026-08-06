"""GPU-first experiment training device helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from chain_replay_ml.dataset_builder.analysis_train_device import (
    DEVICE_CPU,
    DEVICE_GPU,
    fit_xgb_regressor_gpu_first,
    format_device_label,
)
from chain_replay_ml.training.model_device import DevicePlan, clear_device_probe_cache


def _cpu_plan(**kwargs) -> DevicePlan:
    defaults = dict(
        algorithm="xgboost",
        prefer_gpu=True,
        use_gpu=False,
        device="cpu",
        device_label="CPU",
        requested="auto",
        library_params={
            "tree_method": "hist",
            "device": "cpu",
            "predictor": "cpu_predictor",
        },
        fallback_reason="No NVIDIA GPU detected",
        gpu_name=None,
    )
    defaults.update(kwargs)
    return DevicePlan(**defaults)


def _gpu_plan(**kwargs) -> DevicePlan:
    defaults = dict(
        algorithm="xgboost",
        prefer_gpu=True,
        use_gpu=True,
        device="cuda",
        device_label="GPU",
        requested="auto",
        library_params={
            "tree_method": "hist",
            "device": "cuda",
            "predictor": "gpu_predictor",
        },
        fallback_reason=None,
        gpu_name="FakeGPU",
    )
    defaults.update(kwargs)
    return DevicePlan(**defaults)


class FormatDeviceLabelTests(unittest.TestCase):
    def test_gpu_label(self) -> None:
        self.assertEqual(format_device_label("GPU"), "GPU")
        self.assertEqual(
            format_device_label("GPU", shap_device="CPU"), "GPU (SHAP CPU)"
        )

    def test_cpu_with_reason(self) -> None:
        label = format_device_label(
            "CPU", fallback_reason="No NVIDIA GPU detected"
        )
        self.assertTrue(label.startswith("CPU"))
        self.assertIn("No NVIDIA GPU detected", label)


class FitXgbGpuFirstTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_device_probe_cache()
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(80, 3))
        self.y = self.X[:, 0] * 0.5 + rng.normal(scale=0.1, size=80)

    def tearDown(self) -> None:
        clear_device_probe_cache()

    def test_falls_back_to_cpu_when_no_gpu(self) -> None:
        with mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "resolve_experiment_xgb_plan",
            return_value=_cpu_plan(),
        ):
            model, info = fit_xgb_regressor_gpu_first(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertEqual(info["train_device"], DEVICE_CPU)
        self.assertIn("fallback_reason", info)
        self.assertTrue(info["fallback_reason"])

    def test_uses_gpu_when_plan_and_booster_report_cuda(self) -> None:
        class _FakeModel:
            def fit(self, X, y):  # noqa: N803
                return self

            def get_booster(self):
                return object()

        with mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "resolve_experiment_xgb_plan",
            return_value=_gpu_plan(),
        ), mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "_build_xgb_regressor",
            return_value=_FakeModel(),
        ), mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "_verify_model_device",
            return_value="cuda:0",
        ):
            model, info = fit_xgb_regressor_gpu_first(self.X, self.y)
        self.assertIsInstance(model, _FakeModel)
        self.assertEqual(info["train_device"], DEVICE_GPU)
        self.assertEqual(info["executed_device"], "cuda:0")
        self.assertEqual(info["gpu_name"], "FakeGPU")
        self.assertIsNone(info["fallback_reason"])

    def test_gpu_fit_failure_falls_back_to_cpu(self) -> None:
        class _FailGpuThenCpu:
            calls = 0

            def fit(self, X, y):  # noqa: N803
                type(self).calls += 1
                if type(self).calls == 1:
                    raise RuntimeError("CUDA OOM")
                return self

            def get_booster(self):
                return object()

        _FailGpuThenCpu.calls = 0

        with mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "resolve_experiment_xgb_plan",
            return_value=_gpu_plan(),
        ), mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "_build_xgb_regressor",
            side_effect=lambda *_a, **_k: _FailGpuThenCpu(),
        ), mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "_verify_model_device",
            return_value="cpu",
        ):
            model, info = fit_xgb_regressor_gpu_first(self.X, self.y)
        self.assertIsInstance(model, _FailGpuThenCpu)
        self.assertEqual(info["train_device"], DEVICE_CPU)
        self.assertIn("CUDA OOM", info["fallback_reason"] or "")


if __name__ == "__main__":
    unittest.main()
