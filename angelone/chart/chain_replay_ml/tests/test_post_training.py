"""Phase 5.1 — post-training Feature Studio orchestrator + status + config."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PostTrainingOrchestratorTests(unittest.TestCase):
    def test_sequential_completed(self) -> None:
        from chain_replay_ml.post_training import run

        calls: list[str] = []

        def _ok(stage: str):
            def _runner(*, data_dir, model_name, package_dir, progress=None):
                calls.append(stage)
                return SimpleNamespace(ok=True, error=None, artifacts_dir=f"/a/{stage}")

            return _runner

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "DemoModel")
            os.makedirs(pkg)
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
                with patch(
                    "chain_replay_ml.post_training.orchestrator._stage_runner",
                    side_effect=lambda stage: _ok(stage),
                ):
                    result = run(pkg, tmp, model_name="DemoModel")

            status_path = os.path.join(pkg, "feature_studio_status.json")
            self.assertTrue(os.path.isfile(status_path))
            with open(status_path, encoding="utf-8") as fh:
                disk = json.load(fh)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, ["importance", "distribution", "drift"])
        self.assertEqual(result["importance"], "completed")
        self.assertEqual(result["distribution"], "completed")
        self.assertEqual(result["drift"], "completed")
        self.assertEqual(result["warnings"], [])
        self.assertIn("timings_sec", result)
        self.assertIn("telemetry", result)
        self.assertEqual(result["telemetry"]["stages_run"], ["importance", "distribution", "drift"])
        self.assertEqual(result["config"]["enabled"], True)
        self.assertEqual(disk["status"], "completed")
        self.assertEqual(disk["importance"], "completed")
        self.assertIn("started_at", disk["stages"]["importance"])

    def test_partial_when_drift_fails(self) -> None:
        from chain_replay_ml.post_training import run

        def _runner_for(stage: str):
            def _runner(*, data_dir, model_name, package_dir, progress=None):
                if stage == "drift":
                    return SimpleNamespace(ok=False, error="boom")
                return SimpleNamespace(ok=True, error=None)

            return _runner

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "DemoModel")
            os.makedirs(pkg)
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
                with patch(
                    "chain_replay_ml.post_training.orchestrator._stage_runner",
                    side_effect=lambda stage: _runner_for(stage),
                ):
                    result = run(pkg, tmp, model_name="DemoModel")
            with open(os.path.join(pkg, "feature_studio_status.json"), encoding="utf-8") as fh:
                disk = json.load(fh)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["importance"], "completed")
        self.assertEqual(result["distribution"], "completed")
        self.assertEqual(result["drift"], "failed")
        self.assertTrue(any("Drift" in w for w in result["warnings"]))
        self.assertEqual(disk["status"], "partial")
        self.assertEqual(disk["drift"], "failed")

    def test_missing_package_does_not_raise(self) -> None:
        from chain_replay_ml.post_training import run_safe

        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "NoSuchModel")
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
                result = run_safe(missing, tmp, model_name="NoSuchModel")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["warnings"])

    def test_env_off_skips(self) -> None:
        from chain_replay_ml.post_training import run

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "DemoModel")
            os.makedirs(pkg)
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "off"}, clear=False):
                result = run(pkg, tmp)
            with open(os.path.join(pkg, "feature_studio_status.json"), encoding="utf-8") as fh:
                disk = json.load(fh)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(disk["status"], "skipped")
        self.assertTrue(result["config"]["env_disabled"])

    def test_config_disables_drift(self) -> None:
        from chain_replay_ml.post_training import run

        calls: list[str] = []

        def _ok(stage: str):
            def _runner(*, data_dir, model_name, package_dir, progress=None):
                calls.append(stage)
                return SimpleNamespace(ok=True, error=None)

            return _runner

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "DemoModel")
            os.makedirs(pkg)
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
                with patch(
                    "chain_replay_ml.post_training.orchestrator._stage_runner",
                    side_effect=lambda stage: _ok(stage),
                ):
                    result = run(
                        pkg,
                        tmp,
                        model_name="DemoModel",
                        config={
                            "enabled": True,
                            "importance": True,
                            "distribution": True,
                            "drift": False,
                        },
                    )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, ["importance", "distribution"])
        self.assertEqual(result["drift"], "skipped")
        self.assertEqual(result["telemetry"]["stages_skipped"], ["drift"])
        self.assertEqual(
            result["stages"]["drift"].get("skipped_reason"),
            "disabled_by_config",
        )

    def test_config_master_off_skips(self) -> None:
        from chain_replay_ml.post_training import run

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "DemoModel")
            os.makedirs(pkg)
            with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
                with patch(
                    "chain_replay_ml.post_training.orchestrator._stage_runner",
                ) as runner:
                    result = run(pkg, tmp, config={"enabled": False})
                    runner.assert_not_called()
        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["config"]["env_disabled"])

    def test_attach_post_training_keeps_ok_true(self) -> None:
        from chain_replay_ml.training.orchestrator import _attach_post_training

        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "M")
            os.makedirs(pkg)
            saved = {"model_name": "M", "package_dir": pkg}
            result = {"ok": True, "model_name": "M"}
            with patch(
                "chain_replay_ml.post_training.run_safe",
                return_value={"status": "failed", "warnings": ["x"]},
            ) as mock_run:
                out = _attach_post_training(
                    data_dir=tmp,
                    saved=saved,
                    result=result,
                    on_progress=None,
                    post_training_config={"enabled": True, "drift": False},
                )
                kwargs = mock_run.call_args.kwargs
                self.assertEqual(kwargs.get("config"), {"enabled": True, "drift": False})
        self.assertTrue(out["ok"])
        self.assertEqual(out["post_training"]["status"], "failed")


class PostTrainingConfigTests(unittest.TestCase):
    def test_normalize_and_resolve(self) -> None:
        from chain_replay_ml.post_training import (
            normalize_post_training_config,
            resolve_post_training_config,
        )

        norm = normalize_post_training_config(
            {"enabled": True, "stages": {"importance": False, "drift": True}}
        )
        self.assertFalse(norm["importance"])
        self.assertTrue(norm["distribution"])
        self.assertTrue(norm["drift"])

        with patch.dict(os.environ, {"ARUNEO_POST_TRAINING": "on"}, clear=False):
            resolved = resolve_post_training_config(norm)
        self.assertEqual(resolved["active_stages"], ["distribution", "drift"])

    def test_training_config_includes_post_training(self) -> None:
        from chain_replay_ml.training.config import normalize_training_config

        cfg = normalize_training_config(
            {
                "dataset": "d",
                "target": "t",
                "features": ["a"],
                "post_training": {"enabled": True, "drift": False},
            }
        )
        self.assertTrue(cfg.post_training["enabled"])
        self.assertFalse(cfg.post_training["drift"])
        self.assertFalse(cfg.to_dict()["post_training"]["drift"])


class FeatureStudioStatusTests(unittest.TestCase):
    def test_write_load_roundtrip(self) -> None:
        from chain_replay_ml.post_training import (
            format_readiness_line,
            load_feature_studio_status,
            write_feature_studio_status,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = write_feature_studio_status(
                tmp,
                {
                    "status": "completed",
                    "model_name": "M",
                    "stages": {
                        "importance": {
                            "status": "completed",
                            "duration_sec": 1.0,
                            "error": None,
                        },
                        "distribution": {
                            "status": "completed",
                            "duration_sec": 0.5,
                            "error": None,
                        },
                        "drift": {
                            "status": "failed",
                            "duration_sec": 0.1,
                            "error": "x",
                        },
                    },
                    "warnings": ["Drift failed: x"],
                    "started_at": "t0",
                    "finished_at": "t1",
                    "duration_sec": 1.6,
                },
            )
            self.assertTrue(path.endswith("feature_studio_status.json"))
            loaded = load_feature_studio_status(tmp)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(loaded["importance"], "completed")
        self.assertEqual(loaded["drift"], "failed")
        line = format_readiness_line(loaded)
        self.assertIn("Importance", line)
        self.assertIn("Drift", line)

    def test_load_missing_returns_none(self) -> None:
        from chain_replay_ml.post_training import (
            format_readiness_line,
            load_feature_studio_status,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_feature_studio_status(tmp))
        self.assertIn("no auto-run status", format_readiness_line(None))


if __name__ == "__main__":
    unittest.main()
