"""Tests for Phase D1 — Research Program and Campaign."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.research_objective import merge_objective, validate_research_question
from chain_replay_ml.fold_research.research_program import (
    create_research_campaign,
    create_research_program,
    get_campaign_config,
    get_research_campaign,
    list_research_campaigns,
    list_research_programs,
    retire_research_campaign,
    start_research_campaign,
)


class ResearchProgramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_program_campaign_lifecycle(self) -> None:
        out = create_research_program(
            self.tmp,
            name="NIFTY Current Expiry OTM Buyer",
            importance="high",
            objective={
                "primary_goal": {"metric": "profit_factor", "direction": "maximize"},
                "constraints": [{"metric": "win_rate_pct", "op": ">=", "value": 85}],
            },
            budget={"max_experiments": 50, "max_gpu_hours": 20},
        )
        self.assertTrue(out.get("ok"))
        program = out.get("program") or {}
        pid = str(program.get("program_id") or "")

        bad = create_research_campaign(
            self.tmp,
            pid,
            name="Everything",
            research_question="Improve entire strategy",
        )
        self.assertFalse(bad.get("ok"))

        out2 = create_research_campaign(
            self.tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss percentage?",
            budget={"max_experiments": 10},
        )
        self.assertTrue(out2.get("ok"))
        campaign = out2.get("campaign") or {}
        cid = str(campaign.get("campaign_id") or "")

        loaded = get_research_campaign(self.tmp, cid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("resolved_objective", {}).get("constraints", [{}])[0].get("value"), 85)
        self.assertEqual(loaded.get("resolved_budget", {}).get("max_experiments"), 10)
        self.assertEqual(loaded.get("resolved_budget", {}).get("max_gpu_hours"), 20)

        out3 = start_research_campaign(self.tmp, cid)
        self.assertTrue(out3.get("ok"))
        self.assertEqual((out3.get("campaign") or {}).get("status"), "running")

        out4 = retire_research_campaign(self.tmp, cid, reason="Superseded by Stop 7% winner")
        self.assertTrue(out4.get("ok"))
        self.assertEqual((out4.get("campaign") or {}).get("status"), "retired")

        programs = list_research_programs(self.tmp)
        self.assertEqual(len(programs), 1)
        self.assertEqual(programs[0].get("campaign_stats", {}).get("total"), 1)

        campaigns = list_research_campaigns(self.tmp, program_id=pid, status=None)
        self.assertFalse(any(c.get("status") != "retired" for c in campaigns))

        cfg = get_campaign_config(self.tmp, cid)
        self.assertTrue(cfg.get("ok"))

    def test_merge_objective(self) -> None:
        merged = merge_objective(
            {"primary_goal": {"metric": "profit_factor"}, "constraints": [{"metric": "trade_count", "op": ">=", "value": 100}]},
            {"constraints": [{"metric": "win_rate_pct", "op": ">=", "value": 80}]},
        )
        self.assertEqual(len(merged["constraints"]), 1)
        self.assertEqual(merged["constraints"][0]["metric"], "win_rate_pct")

    def test_validate_question(self) -> None:
        self.assertIsNotNone(validate_research_question("x", "Improve entire strategy"))


if __name__ == "__main__":
    unittest.main()
