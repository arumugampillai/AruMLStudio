"""Tests for model lifecycle preset builder."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.training.model_lifecycle import (
    build_model_builder_preset,
    hpo_status_summary,
)
from chain_replay_ml.training.paths import model_artifact_paths


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _seed_model_package(
    data_dir: str,
    model_name: str,
    *,
    hpo_performed: bool = False,
) -> None:
    paths = model_artifact_paths(data_dir, model_name)
    pkg = paths["package_dir"]
    os.makedirs(pkg, exist_ok=True)

    config = {
        "dataset": "MS_185f_3s_0337",
        "target": "future_ltp_5m",
        "algorithm": "xgboost",
        "prediction_type": "regression",
        "features": ["feat_a", "feat_b", "feat_c"],
        "model_version": "1.2",
        "split": {
            "train": 70,
            "validation": 15,
            "test": 15,
            "strategy": "walk_forward",
            "validation_strategy_ui": "walk_forward",
            "hyperparameter_optimization": {
                "enabled": hpo_performed,
                "n_trials": 25,
                "resume": True,
            },
            "walk_forward": {
                "n_folds": 10,
                "window_mode": "expanding",
                "train_window_size": 5000,
                "validation_window_size": 1000,
                "feature_selection_method": "rfe",
                "hyperparameter_optimization": {
                    "enabled": hpo_performed,
                    "n_trials": 25,
                },
            },
        },
        "parameters": {
            "learning_rate": 0.04,
            "max_depth": 8,
            "n_estimators": 1000,
            "subsample": 0.9,
            "random_seed": 42,
        },
    }
    _write_json(paths["config_json"], config)
    _write_json(os.path.join(pkg, "metadata.json"), {"feature_elimination": {}})
    _write_json(paths["metrics_json"], {
        "optimization_result": {"enabled": hpo_performed, "winner": "tuned" if hpo_performed else "baseline"},
        "composite_scores": {"production_composite": 0.4622},
    })
    _write_json(paths["training_summary_json"], {})
    _write_json(paths["registry_json"], {"version": 3})

    wf_dir = os.path.join(pkg, "walk_forward")
    os.makedirs(wf_dir, exist_ok=True)
    _write_json(os.path.join(wf_dir, "summary.json"), {})
    if hpo_performed:
        _write_json(os.path.join(wf_dir, "best_parameters.json"), {
            "n_trials_completed": 25,
            "best_trial_number": 61,
            "best_parameters": {
                "learning_rate": 0.035,
                "max_depth": 7,
                "subsample": 0.88,
            },
            "trial_summary": {"best_trial": 61},
        })


class TestModelLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.model_name = "Test_Lifecycle_Model"

    def test_hpo_status_not_performed(self) -> None:
        _seed_model_package(self.tmp, self.model_name, hpo_performed=False)
        from chain_replay_ml.training.registry import load_model_detail

        detail = load_model_detail(self.tmp, self.model_name)
        status = hpo_status_summary(detail)
        self.assertFalse(status["performed"])
        self.assertEqual(status["status_label"], "Not Performed")
        self.assertAlmostEqual(status["best_composite"], 0.4622)
        self.assertEqual(status["baseline_parameters"]["max_depth"], 8)

    def test_hpo_status_completed(self) -> None:
        _seed_model_package(self.tmp, self.model_name, hpo_performed=True)
        from chain_replay_ml.training.registry import load_model_detail

        detail = load_model_detail(self.tmp, self.model_name)
        status = hpo_status_summary(detail)
        self.assertTrue(status["performed"])
        self.assertEqual(status["best_trial"], 61)
        self.assertEqual(status["baseline_parameters"]["max_depth"], 7)

    def test_complete_optimization_preset(self) -> None:
        _seed_model_package(self.tmp, self.model_name, hpo_performed=True)
        preset = build_model_builder_preset(self.tmp, self.model_name, "complete_optimization")
        tc = preset["training_config"]
        self.assertEqual(preset["mode"], "complete_optimization")
        self.assertEqual(tc["dataset"], "MS_185f_3s_0337")
        self.assertEqual(tc["model_name"], self.model_name)
        self.assertEqual(tc["model_version"], "v2")
        self.assertTrue(tc["split"]["hyperparameter_optimization"]["enabled"])
        self.assertFalse(tc["split"]["hyperparameter_optimization"]["resume"])
        self.assertEqual(tc["split"]["walk_forward"]["feature_selection_method"], "none")
        self.assertIn("feature_snapshot", tc["lifecycle"])
        self.assertEqual(len(tc["lifecycle"]["feature_snapshot"]), 3)
        self.assertIn("selection_method", tc["lifecycle"])
        self.assertTrue(tc["lifecycle"]["center_on_baseline"])
        self.assertEqual(tc["lifecycle"]["source_model"], self.model_name)
        self.assertEqual(tc["parameters"]["max_depth"], 7)

    def test_retrain_preset_disables_hpo(self) -> None:
        _seed_model_package(self.tmp, self.model_name, hpo_performed=True)
        with self.assertRaises(FileNotFoundError):
            build_model_builder_preset(self.tmp, self.model_name, "retrain")

    def test_retrain_training_overrides(self) -> None:
        from chain_replay_ml.training.config import TrainingConfig, apply_lifecycle_training_overrides

        config = TrainingConfig(
            dataset="ds_a",
            target="future_ltp_5m",
            features=["a", "b", "c"],
            split={
                "strategy": "walk_forward",
                "walk_forward": {
                    "n_folds": 5,
                    "feature_selection_method": "rfe",
                    "hyperparameter_optimization": {"enabled": True, "n_trials": 25},
                },
                "hyperparameter_optimization": {"enabled": True, "n_trials": 25},
            },
            parameters={"learning_rate": 0.1, "max_depth": 6},
            lifecycle={
                "mode": "retrain",
                "family_model_name": "Future_LTP_5m_WF_135f_XGB_094",
                "next_version_label": "v3",
                "feature_snapshot": ["x", "y"],
                "baseline_parameters": {"learning_rate": 0.04, "max_depth": 8},
            },
        )
        apply_lifecycle_training_overrides(config)
        self.assertEqual(config.model_name, "Future_LTP_5m_WF_135f_XGB_094__v3")
        self.assertEqual(config.features, ["a", "b", "c"])
        self.assertEqual(config.parameters["max_depth"], 8)
        self.assertFalse(config.split["hyperparameter_optimization"]["enabled"])
        self.assertEqual(config.split["walk_forward"]["feature_selection_method"], "none")
        self.assertFalse(config.split["walk_forward"]["hyperparameter_optimization"]["enabled"])
        self.assertTrue(config.skip_dataset_audit)
        self.assertTrue(config.skip_dataset_validation)

    def test_complete_optimization_locks_feature_snapshot(self) -> None:
        from chain_replay_ml.training.config import TrainingConfig, apply_lifecycle_training_overrides

        config = TrainingConfig(
            dataset="ds_a",
            target="future_ltp_5m",
            features=["a", "b", "c"],
            split={"strategy": "walk_forward", "walk_forward": {"feature_selection_method": "rfe"}},
            parameters={"learning_rate": 0.1},
            lifecycle={
                "mode": "complete_optimization",
                "family_model_name": "Family_Model",
                "next_version_label": "v2",
                "feature_snapshot": ["x", "y"],
            },
        )
        apply_lifecycle_training_overrides(config)
        self.assertEqual(config.features, ["x", "y"])

    def test_normalize_config_applies_retrain_overrides(self) -> None:
        from chain_replay_ml.training.config import normalize_training_config

        config = normalize_training_config({
            "dataset": "ds_a",
            "target": "future_ltp_5m",
            "features": ["a", "b"],
            "split": {
                "strategy": "walk_forward",
                "walk_forward": {"feature_selection_method": "rfe"},
                "hyperparameter_optimization": {"enabled": True},
            },
            "parameters": {"learning_rate": 0.1},
            "lifecycle": {
                "mode": "retrain",
                "family_model_name": "Family_Model",
                "next_version_label": "v2",
                "feature_snapshot": ["x"],
                "baseline_parameters": {"learning_rate": 0.04},
            },
        })
        self.assertEqual(config.features, ["a", "b"])
        self.assertEqual(config.model_name, "Family_Model__v2")
        self.assertEqual(config.split["walk_forward"]["feature_selection_method"], "none")
        self.assertFalse(config.split["hyperparameter_optimization"]["enabled"])

    def test_feature_optimization_preset(self) -> None:
        _seed_model_package(self.tmp, self.model_name, hpo_performed=False)
        preset = build_model_builder_preset(self.tmp, self.model_name, "feature_optimization")
        tc = preset["training_config"]
        self.assertFalse(tc["split"]["hyperparameter_optimization"]["enabled"])
        self.assertEqual(tc["split"]["walk_forward"]["feature_selection_method"], "rfe")
        self.assertEqual(tc["lifecycle"]["mode"], "feature_optimization")
        self.assertEqual(tc["model_name"], self.model_name)
        self.assertEqual(tc["model_version"], "v2")

    def test_feature_optimization_skips_audit_validation(self) -> None:
        from chain_replay_ml.training.config import TrainingConfig, apply_lifecycle_training_overrides

        config = TrainingConfig(
            dataset="ds_a",
            target="future_ltp_5m",
            features=["a", "b"],
            lifecycle={"mode": "feature_optimization", "source_model": "Parent_v1"},
        )
        apply_lifecycle_training_overrides(config)
        self.assertTrue(config.skip_dataset_audit)
        self.assertTrue(config.skip_dataset_validation)

    def test_complete_optimization_skips_audit_validation(self) -> None:
        from chain_replay_ml.training.config import TrainingConfig, apply_lifecycle_training_overrides

        config = TrainingConfig(
            dataset="ds_a",
            target="future_ltp_5m",
            features=["a", "b"],
            lifecycle={
                "mode": "complete_optimization",
                "source_model": "Parent_v1",
                "family_model_name": "Parent",
                "next_version_label": "v2",
            },
        )
        apply_lifecycle_training_overrides(config)
        self.assertTrue(config.skip_dataset_audit)
        self.assertTrue(config.skip_dataset_validation)

    def test_lifecycle_package_name(self) -> None:
        from chain_replay_ml.training.naming import lifecycle_package_name

        self.assertEqual(
            lifecycle_package_name("Future_LTP_5m_WF_135f_XGB_094", "v4"),
            "Future_LTP_5m_WF_135f_XGB_094__v4",
        )

    def test_calibration_only_not_implemented(self) -> None:
        _seed_model_package(self.tmp, self.model_name)
        with self.assertRaises(ValueError):
            build_model_builder_preset(self.tmp, self.model_name, "calibration_only")


if __name__ == "__main__":
    unittest.main()
