"""Tests for Phase D5 — Campaign Report and Program Champion."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.campaign_report import build_campaign_report, get_campaign_report
from chain_replay_ml.fold_research.campaign_scheduler import mark_campaign_validated
from chain_replay_ml.fold_research.program_champion import (
    approve_program_champion,
    build_candidate_program_champion,
    get_program_champion_view,
    refresh_program_champion_candidate,
)
from chain_replay_ml.fold_research.research_program import (
    create_research_campaign,
    create_research_program,
    update_research_campaign,
)


class CampaignReportChampionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_validated_campaign(self) -> tuple[str, str]:
        prog = create_research_program(self.tmp, name="OTM Buyer", importance="high")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp = create_research_campaign(
            self.tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss percentage?",
        )
        cid = str((camp.get("campaign") or {}).get("campaign_id") or "")
        update_research_campaign(
            self.tmp,
            cid,
            status="validated",
            memory={
                "experiments_run": 5,
                "best_profit_factor": 2.3,
                "best_job_id": "job1",
                "best_template_id": "tmpl1",
                "best_generalization": {"overall": 82, "label": "Good", "promote_recommended": True},
                "hypothesis_log": [
                    {"job_number": 1, "change_text": "Stop 7%", "verdict": "Improvement", "pf_delta": 0.12},
                ],
            },
        )
        return pid, cid

    def test_build_campaign_report(self) -> None:
        _, cid = self._seed_validated_campaign()
        report = build_campaign_report(self.tmp, cid)
        self.assertTrue(report.get("ok"))
        summary = report.get("executive_summary") or {}
        self.assertIn("stop", summary.get("conclusion", "").lower())
        self.assertEqual(summary.get("experiments_run"), 5)
        self.assertEqual((summary.get("generalization") or {}).get("overall"), 82)

    def test_mark_validated_generates_report_and_candidate(self) -> None:
        pid, cid = self._seed_validated_campaign()
        update_research_campaign(self.tmp, cid, status="running")
        out = mark_campaign_validated(self.tmp, cid)
        self.assertTrue(out.get("ok"))
        self.assertIsNotNone(out.get("report"))
        got = get_campaign_report(self.tmp, cid)
        self.assertTrue(got.get("ok"))

        champ = build_candidate_program_champion(self.tmp, pid)
        self.assertTrue(champ.get("ok"))
        candidate = champ.get("candidate") or {}
        self.assertTrue(candidate)
        self.assertGreaterEqual(int((candidate.get("generalization") or {}).get("overall") or 0), 70)

    def test_approve_program_champion_human_gate(self) -> None:
        pid, cid = self._seed_validated_campaign()
        refresh_program_champion_candidate(self.tmp, pid)
        view = get_program_champion_view(self.tmp, pid)
        self.assertTrue(view.get("can_approve"))

        bad = approve_program_champion(self.tmp, "missing")
        self.assertFalse(bad.get("ok"))

        out = approve_program_champion(self.tmp, pid)
        self.assertTrue(out.get("ok"))
        view2 = get_program_champion_view(self.tmp, pid)
        self.assertTrue(view2.get("has_approved_champion"))
        self.assertFalse(view2.get("can_approve"))


if __name__ == "__main__":
    unittest.main()
