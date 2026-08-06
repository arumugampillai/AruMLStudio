"""Tests for experiment recommendation routing."""

from __future__ import annotations

import unittest

from chain_replay_ml.fold_research.experiment_recommendations import categorize_recommendation


class ExperimentRecommendationTests(unittest.TestCase):
    def test_premium_routes_to_strategy(self) -> None:
        cat = categorize_recommendation("Avoid premium below ₹26", premium_threshold=26)
        self.assertEqual(cat["target"], "strategy_registry")
        self.assertEqual(cat["filters"].get("min_premium"), 26)

    def test_theta_feature_routes_to_registry(self) -> None:
        cat = categorize_recommendation("Retrain model with theta / regime features")
        self.assertEqual(cat["target"], "feature_registry")
        self.assertIn("theta", cat["feature_hints"])

    def test_optuna_routes_to_hpo(self) -> None:
        cat = categorize_recommendation("Increase Optuna trials")
        self.assertEqual(cat["target"], "hyperparameter_optimization")


if __name__ == "__main__":
    unittest.main()
