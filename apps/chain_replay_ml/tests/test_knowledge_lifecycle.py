"""Tests for Phase E — Knowledge lifecycle and KB-driven proposals."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.finding_extraction import extract_findings_from_job
from chain_replay_ml.fold_research.kb_proposal_generator import get_knowledge_gaps_for_campaign
from chain_replay_ml.fold_research.knowledge_pipeline import (
    finalize_campaign_job_knowledge,
    get_knowledge_pipeline_view,
    promote_finding_to_knowledge,
    process_job_knowledge_pipeline,
)
from chain_replay_ml.fold_research.knowledge_store import KnowledgeStore, lifecycle_stage_for_status
from chain_replay_ml.fold_research.research_program import create_research_campaign, create_research_program


class KnowledgeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_campaign(self) -> tuple[str, str]:
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

    def test_lifecycle_stage_mapping(self) -> None:
        self.assertEqual(lifecycle_stage_for_status("candidate"), "evidence_linked")
        self.assertEqual(lifecycle_stage_for_status("supported"), "finding")
        self.assertEqual(lifecycle_stage_for_status("knowledge"), "knowledge")

    def test_extract_links_campaign_and_program(self) -> None:
        pid, cid = self._seed_campaign()
        template = {
            "goal": "Optimize stop loss",
            "accepted_changes": [{"text": "Use 7% stop loss", "target": "strategy_registry"}],
            "campaign_id": cid,
            "program_id": pid,
        }
        job = {
            "job_id": "job1",
            "job_number": 1,
            "outputs": {},
            "results": {},
        }
        comparison = {
            "baseline_pf": 1.5,
            "after_pf": 1.7,
            "baseline_win_rate_pct": 45.0,
            "after_win_rate_pct": 48.0,
        }
        out = extract_findings_from_job(
            self.tmp,
            template=template,
            job=job,
            comparison=comparison,
            trade_count=500,
            campaign_id=cid,
            program_id=pid,
        )
        self.assertGreater(out.get("findings_updated") or 0, 0)
        finding_id = (out.get("findings") or [{}])[0].get("finding_id")
        self.assertTrue(finding_id)
        with KnowledgeStore(self.tmp) as store:
            links = store.list_links_for_finding(str(finding_id))
            refs = {(l.get("link_type"), l.get("link_ref")) for l in links}
            self.assertIn(("research_campaign", cid), refs)
            self.assertIn(("research_program", pid), refs)

    def test_promote_to_knowledge(self) -> None:
        with KnowledgeStore(self.tmp) as store:
            finding = store.upsert_finding(
                finding_key="stop_7pct_improves_outcomes",
                finding="7% stop loss improves trade outcomes",
                category="strategy",
            )
            fid = str(finding.get("finding_id") or "")
            for i in range(3):
                store.add_evidence(
                    fid,
                    {
                        "experiment_id": f"exp{i}",
                        "experiment_number": i + 1,
                        "supports_finding": True,
                        "trade_count": 400,
                        "pf_change": 0.12,
                    },
                )
            store.conn.execute(
                "UPDATE knowledge_findings SET status = 'confirmed' WHERE finding_id = ?",
                (fid,),
            )
            store.conn.commit()

        out = promote_finding_to_knowledge(self.tmp, fid)
        self.assertTrue(out.get("ok"))
        promoted = out.get("finding") or {}
        self.assertEqual(promoted.get("status"), "knowledge")
        self.assertEqual(lifecycle_stage_for_status("knowledge"), "knowledge")

    def test_finalize_campaign_job_knowledge(self) -> None:
        pid, cid = self._seed_campaign()
        job = {
            "results": {
                "knowledge": {
                    "findings": [{"finding_id": "missing"}],
                },
            },
        }
        out = finalize_campaign_job_knowledge(self.tmp, job=job, campaign_id=cid, program_id=pid)
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("findings_seen"), 1)

    def test_knowledge_pipeline_view(self) -> None:
        pid, cid = self._seed_campaign()
        template = {
            "goal": "Filter theta",
            "accepted_changes": [{"text": "Add theta filter", "target": "strategy_registry"}],
        }
        job = {"job_id": "j2", "job_number": 2, "outputs": {}, "results": {}}
        comparison = {"baseline_pf": 1.4, "after_pf": 1.35, "baseline_win_rate_pct": 40, "after_win_rate_pct": 39}
        process_job_knowledge_pipeline(
            self.tmp,
            template=template,
            job=job,
            comparison=comparison,
            campaign_id=cid,
            program_id=pid,
        )
        view = get_knowledge_pipeline_view(self.tmp, campaign_id=cid)
        self.assertTrue(view.get("ok"))
        self.assertGreaterEqual(view.get("totals", {}).get("all", 0), 1)

    def test_knowledge_gaps_for_campaign(self) -> None:
        _, cid = self._seed_campaign()
        gaps = get_knowledge_gaps_for_campaign(self.tmp, cid)
        self.assertTrue(gaps.get("ok"))
        self.assertIn("knowledge_gaps", gaps)


if __name__ == "__main__":
    unittest.main()
