"""Tests for shared GPU/CPU device factory."""

from __future__ import annotations

import unittest
from unittest import mock

from chain_replay_ml.training.model_device import (
    LightGBMGpuUnavailableError,
    clear_device_probe_cache,
    format_startup_diagnostics,
    resolve_training_device,
    verify_xgboost_booster_device,
)


class ModelDeviceFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_device_probe_cache()

    def tearDown(self) -> None:
        clear_device_probe_cache()

    def test_xgboost_defaults_to_gpu_when_probe_ok(self) -> None:
        with mock.patch(
            "chain_replay_ml.training.model_device.detect_gpu_hardware",
            return_value={"gpu_detected": True, "gpu_name": "FakeGPU"},
        ), mock.patch(
            "chain_replay_ml.training.model_device.probe_xgboost_gpu",
            return_value={
                "supported": True,
                "version": "3.3.0",
                "detail": "probe trained on cuda:0",
                "executed_device": "cuda:0",
            },
        ):
            plan = resolve_training_device("xgboost", {})
        self.assertTrue(plan.use_gpu)
        self.assertEqual(plan.device_label, "GPU")
        self.assertEqual(plan.library_params.get("device"), "cuda")
        self.assertIsNone(plan.fallback_reason)

    def test_xgboost_cpu_when_requested(self) -> None:
        plan = resolve_training_device("xgboost", {"xgb_device": "cpu"})
        self.assertFalse(plan.use_gpu)
        self.assertEqual(plan.device_label, "CPU")
        self.assertIn("CPU requested", plan.fallback_reason or "")

    def test_lightgbm_raises_when_gpu_unsupported(self) -> None:
        with mock.patch(
            "chain_replay_ml.training.model_device.detect_gpu_hardware",
            return_value={"gpu_detected": True, "gpu_name": "FakeGPU"},
        ), mock.patch(
            "chain_replay_ml.training.model_device.probe_lightgbm_gpu",
            return_value={
                "supported": False,
                "installed": True,
                "version": "4.0.0",
                "detail": "CPU-only wheel",
            },
        ):
            with self.assertRaises(LightGBMGpuUnavailableError) as ctx:
                resolve_training_device("lightgbm", {"lgb_device": "cuda"})
        self.assertIn("Refusing to silently train on CPU", str(ctx.exception))

    def test_lightgbm_cpu_allowed_when_explicit(self) -> None:
        plan = resolve_training_device("lightgbm", {"lgb_device": "cpu"})
        self.assertFalse(plan.use_gpu)
        self.assertEqual(plan.library_params.get("num_threads"), -1)

    def test_catboost_defaults_to_gpu(self) -> None:
        with mock.patch(
            "chain_replay_ml.training.model_device.detect_gpu_hardware",
            return_value={"gpu_detected": True, "gpu_name": "FakeGPU"},
        ), mock.patch(
            "chain_replay_ml.training.model_device.probe_catboost_gpu",
            return_value={"supported": True, "version": "1.0", "detail": "ok", "installed": True},
        ):
            plan = resolve_training_device("catboost", {})
        self.assertTrue(plan.use_gpu)
        self.assertEqual(plan.library_params.get("task_type"), "GPU")

    def test_rf_falls_back_with_warning_when_cuml_missing(self) -> None:
        with mock.patch(
            "chain_replay_ml.training.model_device.detect_gpu_hardware",
            return_value={"gpu_detected": True, "gpu_name": "FakeGPU"},
        ), mock.patch(
            "chain_replay_ml.training.model_device.probe_cuml",
            return_value={"supported": False, "detail": "not installed", "version": None},
        ):
            with self.assertWarns(UserWarning):
                plan = resolve_training_device("random_forest", {})
        self.assertFalse(plan.use_gpu)
        self.assertIn("cuML unavailable", plan.fallback_reason or "")

    def test_startup_diagnostics_format(self) -> None:
        lines = format_startup_diagnostics(
            {
                "gpu_detected": True,
                "gpu_name": "NVIDIA GeForce RTX 3050",
                "driver_version": "560.94",
                "memory_total": "8192 MiB",
                "libraries": {
                    "xgboost": {"version": "3.3.0", "supported": True, "detail": "ok"},
                    "lightgbm": {
                        "version": None,
                        "supported": False,
                        "detail": "not installed",
                    },
                    "catboost": {
                        "version": None,
                        "supported": False,
                        "detail": "not installed",
                    },
                    "cuml": {"version": None, "supported": False, "detail": "missing"},
                    "sklearn": {"version": "1.9.0"},
                },
            }
        )
        text = "\n".join(lines)
        self.assertIn("GPU detected: Yes", text)
        self.assertIn("RTX 3050", text)
        self.assertIn("xgboost: 3.3.0 · GPU OK", text)
        self.assertIn("lightgbm:", text)

    def test_verify_xgboost_booster_device_parses_cuda(self) -> None:
        class _B:
            def save_config(self) -> str:
                return '{"learner":{"generic_parameter":{"device":"cuda:0"}}}'

        self.assertTrue(verify_xgboost_booster_device(_B()).startswith("cuda"))


if __name__ == "__main__":
    unittest.main()
