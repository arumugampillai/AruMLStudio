"""Tests for Phase D4 — Program Portfolio and Campaign Dashboard."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.campaign_dashboard import get_campaign_dashboard
from chain_replay_ml.fold_research.program_portfolio import get_program_portfolio, get_research_portfolio
from chain_replay_ml.fold_research.research_program import (
    create_research_campaign,
    create_research_program,
    start_research_campaign,
)


class PortfolioDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        prog = create_research_program(self.tmp, name="OTM Buyer", importance="high")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp = create_research_campaign(
            self.tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss percentage?",
        )
        cid = str((camp.get("campaign") or {}).get("campaign_id") or "")
        return pid, cid

    def test_program_portfolio(self) -> None:
        pid, cid = self._seed()
        port = get_program_portfolio(self.tmp, pid)
        self.assertTrue(port.get("ok"))
        self.assertEqual(len(port.get("campaigns") or []), 1)
        card = (port.get("campaigns") or [])[0]
        self.assertEqual(card.get("campaign_id"), cid)
        self.assertEqual(card.get("status"), "created")
        stats = port.get("stats") or {}
        self.assertEqual(stats.get("total_campaigns"), 1)

    def test_desk_portfolio(self) -> None:
        self._seed()
        desk = get_research_portfolio(self.tmp)
        self.assertTrue(desk.get("ok"))
        self.assertEqual(len(desk.get("programs") or []), 1)
        self.assertIn("global_stats", desk)

    def test_campaign_dashboard_running(self) -> None:
        pid, cid = self._seed()
        start_research_campaign(self.tmp, cid)
        dash = get_campaign_dashboard(self.tmp, cid)
        self.assertTrue(dash.get("ok"))
        self.assertEqual((dash.get("campaign") or {}).get("status"), "running")
        self.assertIn("funnel", dash)
        self.assertIn("budget_burn", dash)
        self.assertIn("scheduler", dash)


if __name__ == "__main__":
    unittest.main()
