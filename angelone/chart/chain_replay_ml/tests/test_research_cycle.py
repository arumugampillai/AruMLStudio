"""Tests for Phase D3 — Research Cycle and Generalization Score."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.generalization_score import (
    compute_generalization_score,
    extract_generalization_slices,
)
from chain_replay_ml.fold_research.research_cycle import (
    apply_cycle_decision,
    get_cycle_view,
    infer_exploration_stage,
    record_hypothesis_trial,
)


class ResearchCycleTests(unittest.TestCase):
    def test_exploration_stage_progression(self) -> None:
        memory = {"experiments_run": 1}
        self.assertEqual(infer_exploration_stage(memory), "explore")

        memory = {
            "experiments_run": 3,
            "best_verdict": "Improvement",
            "best_profit_factor": 2.2,
        }
        self.assertEqual(infer_exploration_stage(memory), "exploit")

        memory["best_generalization"] = {"overall": 75}
        self.assertEqual(infer_exploration_stage(memory), "validate")

    def test_hypothesis_log_and_decision(self) -> None:
        memory: dict = {}
        memory = record_hypothesis_trial(
            memory,
            template={
                "template_id": "t1",
                "goal": "Stop 7%",
                "accepted_changes": [{"text": "Stop 7%"}],
                "objective_score": {"overall": 80},
            },
            job={"job_id": "j1", "job_number": 1},
            comparison={"pf_delta": 0.15, "after_pf": 2.1},
            verdict={"verdict": "Improvement"},
        )
        self.assertEqual(len(memory.get("hypothesis_log") or []), 1)

        memory = apply_cycle_decision(
            memory,
            comparison={"after_pf": 2.1, "baseline_pf": 1.9, "after_win_rate_pct": 85, "after_trade_count": 120},
            verdict={"verdict": "Improvement", "recommendation": "Continue"},
            objective={
                "primary_goal": {"metric": "profit_factor", "direction": "maximize"},
                "constraints": [{"metric": "win_rate_pct", "op": ">=", "value": 80}],
            },
            generalization={"overall": 82, "label": "Good"},
        )
        self.assertTrue(memory.get("validation_ready"))
        view = get_cycle_view(memory)
        self.assertEqual(view["current_step"], "decision")


class GeneralizationScoreTests(unittest.TestCase):
    def test_stable_slices_score_high(self) -> None:
        slices = {
            "walk_forward_folds": [
                {"profit_factor": 2.0},
                {"profit_factor": 2.1},
                {"profit_factor": 1.95},
            ],
            "calendar_months": [
                {"profit_factor": 2.05},
                {"profit_factor": 2.0},
            ],
            "volatility_regimes": [{"profit_factor": 2.1}],
            "expiry_weeks": [{"profit_factor": 2.0}],
        }
        score = compute_generalization_score(slices, baseline_pf=1.8)
        self.assertGreaterEqual(score["overall"], 70)
        self.assertIn(score["label"], ("Good", "Excellent"))

    def test_unstable_slices_score_low(self) -> None:
        slices = {
            "walk_forward_folds": [
                {"profit_factor": 3.5},
                {"profit_factor": 0.8},
                {"profit_factor": 2.9},
            ],
            "calendar_months": [
                {"profit_factor": 0.5},
                {"profit_factor": 3.0},
            ],
            "volatility_regimes": [],
            "expiry_weeks": [],
        }
        score = compute_generalization_score(slices, baseline_pf=2.0)
        self.assertLess(score["overall"], 70)
        self.assertFalse(score["promote_recommended"])

    def test_extract_slices_empty_job(self) -> None:
        slices = extract_generalization_slices(tempfile.mkdtemp(), {"outputs": {}})
        self.assertEqual(slices["walk_forward_folds"], [])


if __name__ == "__main__":
    unittest.main()
