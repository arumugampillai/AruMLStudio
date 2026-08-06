"""Tests for automatic post-job pipeline."""

from __future__ import annotations

import unittest

from chain_replay_ml.fold_research.experiment_post_job import (
    build_trading_impact,
    compare_with_baseline,
    compute_information_gain,
    default_job_decision,
    generate_experiment_verdict,
    recommend_follow_up_experiment,
    suggest_next_experiments,
)


class ExperimentPostJobTests(unittest.TestCase):
    def test_regression_verdict_and_next_experiments(self) -> None:
        template = {
            "accepted_changes": [
                {"text": "Avoid premium below 24", "target": "strategy_registry", "filters": {"min_premium": 24}},
                {"text": "Retrain with theta", "target": "feature_registry", "feature_hints": ["theta"]},
                {"text": "Use 7% stop", "target": "strategy_registry", "filters": {"stop_pct": 7}},
            ],
            "routing": {
                "strategy_changes": [{"text": "Avoid premium below 24", "filters": {"min_premium": 24}}],
                "feature_changes": [{"text": "Retrain with theta", "feature_hints": ["theta"]}],
            },
        }
        collected = {
            "baseline_strategy": {"profit_factor": 7.35, "win_rate_pct": 88.0, "trade_count": 160},
            "strategy": {"profit_factor": 6.85, "win_rate_pct": 87.7, "trade_count": 122},
            "model_metrics": {"mae": 2.15, "rmse": 3.1},
        }
        comparison = compare_with_baseline(collected)
        self.assertEqual(comparison["pf_delta"], -0.5)
        self.assertEqual(comparison["trade_count_delta"], -38)

        verdict = generate_experiment_verdict(comparison, collected=collected, template=template)
        self.assertEqual(verdict["verdict"], "Regression")

        gain = compute_information_gain(comparison, template=template, verdict=verdict)
        self.assertIn(gain["label"], ("Very High", "High"))

        next_items = suggest_next_experiments(template, comparison=comparison, verdict=verdict, information_gain=gain)
        self.assertTrue(len(next_items) >= 2)
        titles = {n.get("title") for n in next_items}
        self.assertTrue(any("Retrain Only" in t for t in titles))

    def test_trading_impact_neutral_retrain(self) -> None:
        template = {
            "accepted_changes": [{"text": "Retrain with theta", "target": "feature_registry"}],
            "routing": {"feature_changes": [{"text": "Retrain with theta"}]},
            "prediction_run_id": "pred_base",
        }
        collected = {
            "baseline_strategy": {"profit_factor": 7.35, "win_rate_pct": 88.1, "trade_count": 160},
            "strategy": {"profit_factor": 7.35, "win_rate_pct": 88.1, "trade_count": 160},
            "model_metrics": {"mae": 2.1, "directional_accuracy_pct": 58.0},
            "phase": "C",
            "prediction_run_id": "pred_new",
            "baseline_prediction_run_id": "pred_base",
        }
        comparison = compare_with_baseline(collected)
        verdict = generate_experiment_verdict(comparison, collected=collected, template=template)
        impact = build_trading_impact(
            comparison,
            collected=collected,
            template=template,
            verdict=verdict,
            outputs={"phase": "C", "prediction_run_id": "pred_new", "baseline_prediction_run_id": "pred_base"},
        )
        self.assertEqual(impact["pf"], "No Change")
        self.assertEqual(impact["win_rate"], "No Change")
        self.assertEqual(impact["trades"], "No Change")
        self.assertIn(impact["prediction"], ("Improved", "Retrained"))
        self.assertIn("no trading benefit", impact["conclusion"].lower())

    def test_default_decision_archives_evidence(self) -> None:
        decision = default_job_decision(
            {"verdict": "Neutral"},
            template={"accepted_changes": [{"text": "x"}]},
            outputs={"phase": "C"},
        )
        self.assertTrue(decision["archive_as_evidence"])
        self.assertTrue(decision["repeat_modified_hypothesis"])


if __name__ == "__main__":
    unittest.main()
