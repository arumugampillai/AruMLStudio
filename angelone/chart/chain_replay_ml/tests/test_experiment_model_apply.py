"""Tests for Phase C experiment model apply helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.fold_research.experiment_model_apply import (
    infer_lifecycle_mode,
    read_prediction_run_id,
    resolve_feature_hints,
)
from chain_replay_ml.training.model_lifecycle import build_model_builder_preset


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _seed_model_package(data_dir: str, model_name: str) -> None:
    from chain_replay_ml.training.paths import model_artifact_paths

    paths = model_artifact_paths(data_dir, model_name)
    pkg = paths["package_dir"]
    config = {
        "dataset": "MS_test",
        "target": "future_ltp_5m",
        "algorithm": "xgboost",
        "prediction_type": "regression",
        "features": ["feat_a", "feat_b"],
        "model_version": "1.0",
        "split": {
            "train": 70,
            "validation": 15,
            "test": 15,
            "strategy": "walk_forward",
            "walk_forward": {"n_folds": 3, "window_mode": "expanding"},
            "hyperparameter_optimization": {"enabled": False},
        },
        "parameters": {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 100},
    }
    _write_json(paths["config_json"], config)
    _write_json(os.path.join(pkg, "metadata.json"), {})
    _write_json(paths["metrics_json"], {"optimization_result": {"enabled": False}})
    _write_json(paths["training_summary_json"], {})
    _write_json(paths["registry_json"], {"version": 1})


class ExperimentModelApplyTests(unittest.TestCase):
    def test_infer_lifecycle_mode(self) -> None:
        self.assertEqual(
            infer_lifecycle_mode({"feature_changes": [{"text": "add theta"}]}),
            "feature_optimization",
        )
        self.assertEqual(
            infer_lifecycle_mode({"model_changes": [{"text": "retrain"}]}),
            "retrain",
        )
        self.assertEqual(
            infer_lifecycle_mode({"optimization_changes": [{"text": "optuna"}]}),
            "complete_optimization",
        )

    def test_read_prediction_run_id(self) -> None:
        tmp = tempfile.mkdtemp()
        pkg = os.path.join(tmp, "model_pkg")
        os.makedirs(pkg, exist_ok=True)
        _write_json(os.path.join(pkg, "prediction_run.json"), {"run_id": "pred123"})
        self.assertEqual(read_prediction_run_id(pkg), "pred123")

    def test_clone_preset_for_model(self) -> None:
        tmp = tempfile.mkdtemp()
        _seed_model_package(tmp, "ExpCModel")
        preset = build_model_builder_preset(tmp, "ExpCModel", "complete_optimization")
        self.assertTrue(preset.get("ok"))
        cfg = preset.get("training_config") or {}
        self.assertEqual(cfg.get("dataset"), "MS_test")
        self.assertTrue(cfg["split"]["hyperparameter_optimization"]["enabled"])

    def test_resolve_feature_hints_without_dataset(self) -> None:
        tmp = tempfile.mkdtemp()
        out = resolve_feature_hints(tmp, "missing_dataset", ["theta"], existing_features=["feat_a"])
        self.assertIn("feat_a", out["features"])


if __name__ == "__main__":
    unittest.main()
