"""Tests for centralized algorithm capability registry."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.training.algorithm_capabilities import (
    algorithm_supports_prediction_type,
    assert_algorithm_supports_prediction_type,
    capability_matrix_rows,
    format_algorithm_support_report,
    get_algorithm_capabilities,
    list_algorithm_capabilities,
)
from chain_replay_ml.training.boost_trainer import train_regressor
from chain_replay_ml.training.model_device import clear_device_probe_cache, format_startup_diagnostics
from chain_replay_ml.training.trainers import get_trainer, supported_algorithms


class AlgorithmCapabilitiesTests(unittest.TestCase):
    def test_all_algorithms_support_binary(self) -> None:
        for algo_id, _label in supported_algorithms():
            caps = get_algorithm_capabilities(algo_id)
            self.assertTrue(caps.supports_regression, algo_id)
            self.assertTrue(caps.supports_binary_classification, algo_id)
            self.assertTrue(caps.supports_multiclass, algo_id)
            self.assertTrue(algorithm_supports_prediction_type(algo_id, "binary"), algo_id)
            self.assertTrue(algorithm_supports_prediction_type(algo_id, "classification"), algo_id)

    def test_trainer_supports_binary_via_registry(self) -> None:
        for algo_id, _label in supported_algorithms():
            trainer = get_trainer(algo_id)
            self.assertTrue(trainer.supports_binary_classification(), algo_id)
            self.assertTrue(trainer.supports_prediction_type("binary"), algo_id)

    def test_no_xgboost_only_hardcode_in_trainers(self) -> None:
        # Non-XGBoost trainers must accept binary without raising the old hardcode.
        for algo_id in ("lightgbm", "catboost", "random_forest", "extra_trees"):
            # Capability assert alone must pass (training may still fail if lib missing).
            assert_algorithm_supports_prediction_type(algo_id, "binary")

    def test_capability_matrix_has_gpu_reason(self) -> None:
        rows = capability_matrix_rows()
        self.assertGreaterEqual(len(rows), 5)
        by_id = {r["algorithm"]: r for r in rows}
        self.assertIn("extra_trees", by_id)
        et = by_id["extra_trees"]
        self.assertTrue(et["binary"])
        self.assertTrue(et["multiclass"])
        # Without cuML, GPU should be unavailable with an explicit reason.
        if not et["gpu_train_available"]:
            self.assertTrue(et["gpu_reason"])

    def test_startup_diagnostics_include_algorithm_support(self) -> None:
        clear_device_probe_cache()
        lines = format_startup_diagnostics()
        joined = "\n".join(lines)
        self.assertIn("Algorithm Support", joined)
        self.assertIn("Extra Trees", joined)
        self.assertIn("Binary", joined)

    def test_format_algorithm_support_report(self) -> None:
        lines = format_algorithm_support_report()
        self.assertEqual(lines[0], "Algorithm Support")
        self.assertTrue(any("XGBoost" == ln for ln in lines))

    def test_random_forest_binary_trains_on_cpu(self) -> None:
        rng = np.random.default_rng(0)
        n = 80
        X = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        })
        y = pd.Series((X["f1"] > 0).astype(int))
        split = 60
        result = train_regressor(
            algorithm="random_forest",
            train_X=X.iloc[:split],
            train_y=y.iloc[:split],
            val_X=X.iloc[split:],
            val_y=y.iloc[split:],
            features=["f1", "f2"],
            parameters={
                "n_estimators": 20,
                "max_depth": 4,
                "rf_device": "cpu",
                "prediction_type": "binary",
            },
            prediction_type="binary",
        )
        pred = result["model"].predict(X.iloc[split:])
        self.assertEqual(len(pred), n - split)
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))

    def test_extra_trees_binary_trains_on_cpu(self) -> None:
        rng = np.random.default_rng(1)
        n = 80
        X = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        })
        y = pd.Series((X["f1"] > 0).astype(int))
        split = 60
        result = train_regressor(
            algorithm="extra_trees",
            train_X=X.iloc[:split],
            train_y=y.iloc[:split],
            val_X=X.iloc[split:],
            val_y=y.iloc[split:],
            features=["f1", "f2"],
            parameters={
                "n_estimators": 20,
                "max_depth": 4,
                "rf_device": "cpu",
                "prediction_type": "binary",
            },
            prediction_type="binary",
        )
        pred = result["model"].predict(X.iloc[split:])
        self.assertEqual(len(pred), n - split)
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))


if __name__ == "__main__":
    unittest.main()
