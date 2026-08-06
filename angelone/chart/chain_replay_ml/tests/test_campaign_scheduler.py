"""Tests for Phase D2 — Objective Score and Campaign Scheduler."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.fold_research.campaign_scheduler import (
    bootstrap_campaign_scheduler,
    check_campaign_budget,
    get_campaign_scheduler_view,
)
from chain_replay_ml.fold_research.experiment_pipeline_store import ExperimentPipelineStore
from chain_replay_ml.fold_research.objective_score import compute_objective_score
from chain_replay_ml.fold_research.research_program import (
    create_research_campaign,
    create_research_program,
    start_research_campaign,
)
from chain_replay_ml.fold_research.research_report_store import ResearchReportStore


def _minimal_report(report_id: str = "rpt1") -> dict:
    return {
        "ok": True,
        "report_id": report_id,
        "prediction_run_id": "pred1",
        "strategy_run_id": "strat1",
        "executive_summary": {
            "model_id": "model1",
            "strategy": "OTM Buyer",
            "overall_grade": "B",
        },
        "baseline_metrics": {
            "profit_factor": 2.1,
            "win_rate_pct": 82.0,
            "trade_count": 120,
        },
        "recommendations": [
            {
                "text": "Tighten stop loss to 7%",
                "target": "strategy_registry",
                "accepted_default": True,
                "filters": {"stop_pct": 7},
            },
            {
                "text": "Enforce theta filter",
                "target": "strategy_registry",
                "accepted_default": False,
                "filters": {},
            },
        ],
    }


class ObjectiveScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_objective_score_ranks_proposal(self) -> None:
        proposal = {
            "goal": "What is the optimal stop loss?",
            "selected_recommendations": [
                {"text": "Tighten stop loss to 7%", "target": "strategy_registry"},
            ],
            "baseline": {"profit_factor": 2.1, "win_rate_pct": 82, "trade_count": 120},
            "prediction_run_id": "pred1",
            "strategy_run_id": "strat1",
            "model_id": "model1",
            "strategy_label": "OTM Buyer",
        }
        objective = {
            "primary_goal": {"metric": "profit_factor", "direction": "maximize"},
            "constraints": [{"metric": "win_rate_pct", "op": ">=", "value": 80}],
        }
        scored = compute_objective_score(
            self.tmp,
            proposal=proposal,
            objective=objective,
            importance="high",
        )
        self.assertFalse(scored.get("rejected"))
        self.assertGreater(int(scored.get("overall") or 0), 0)
        self.assertIn("components", scored)

    def test_constraint_reject(self) -> None:
        proposal = {
            "goal": "test",
            "selected_recommendations": [{"text": "x", "target": "strategy_registry"}],
            "baseline": {"win_rate_pct": 70, "trade_count": 120},
        }
        objective = {
            "constraints": [{"metric": "win_rate_pct", "op": ">=", "value": 80}],
        }
        scored = compute_objective_score(self.tmp, proposal=proposal, objective=objective)
        self.assertTrue(scored.get("rejected"))
        self.assertEqual(scored.get("overall"), 0)


class CampaignSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_program_campaign(self) -> tuple[str, str]:
        prog = create_research_program(
            self.tmp,
            name="Test Program",
            budget={"max_experiments": 3},
        )
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp = create_research_campaign(
            self.tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss percentage?",
        )
        cid = str((camp.get("campaign") or {}).get("campaign_id") or "")
        return pid, cid

    def test_scheduler_view_and_budget(self) -> None:
        _, cid = self._seed_program_campaign()
        view = get_campaign_scheduler_view(self.tmp, cid)
        self.assertTrue(view.get("ok"))
        self.assertEqual(view.get("status"), "created")

        budget = check_campaign_budget(self.tmp, cid)
        self.assertFalse(budget.get("exhausted"))

    def test_bootstrap_after_start_with_baseline(self) -> None:
        _, cid = self._seed_program_campaign()
        report = _minimal_report()
        with ResearchReportStore(self.tmp) as store:
            store.save_report(report)

        from chain_replay_ml.fold_research.campaign_proposal_generator import attach_campaign_baseline

        attach_campaign_baseline(self.tmp, cid, research_report_id="rpt1")
        out = start_research_campaign(self.tmp, cid)
        self.assertTrue(out.get("ok"))
        sched = out.get("scheduler") or {}
        self.assertTrue(sched.get("ok"))

        view = get_campaign_scheduler_view(self.tmp, cid)
        self.assertEqual(view.get("status"), "running")

    def test_proposal_persisted_with_campaign_id(self) -> None:
        _, cid = self._seed_program_campaign()
        report = _minimal_report()
        with ResearchReportStore(self.tmp) as store:
            store.save_report(report)

        from chain_replay_ml.fold_research.campaign_proposal_generator import (
            attach_campaign_baseline,
            seed_proposals_from_report,
        )

        attach_campaign_baseline(self.tmp, cid, research_report_id="rpt1")
        seeded = seed_proposals_from_report(self.tmp, cid)
        self.assertTrue(seeded.get("ok"))
        self.assertGreater(seeded.get("count", 0), 0)

        with ExperimentPipelineStore(self.tmp) as store:
            proposals = store.list_proposals(campaign_id=cid, status="draft")
        self.assertTrue(all(p.get("campaign_id") == cid for p in proposals))
        self.assertTrue(all((p.get("objective_score") or {}).get("overall", 0) >= 0 for p in proposals))


if __name__ == "__main__":
    unittest.main()
