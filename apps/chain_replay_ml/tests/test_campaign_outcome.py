"""Tests for campaign outcome summary."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.campaign_outcome import get_campaign_outcome
from chain_replay_ml.fold_research.research_program import create_research_campaign, create_research_program


class CampaignOutcomeTests(unittest.TestCase):
    def test_outcome_empty_campaign(self) -> None:
        tmp = tempfile.mkdtemp()
        prog = create_research_program(tmp, name="OTM", importance="medium")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp = create_research_campaign(
            tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss percentage?",
        )
        cid = str((camp.get("campaign") or {}).get("campaign_id") or "")
        out = get_campaign_outcome(tmp, cid)
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("campaign_name"), "Stop Optimization")
        self.assertIn("assessment", out.get("executive_summary") or {})


if __name__ == "__main__":
    unittest.main()
