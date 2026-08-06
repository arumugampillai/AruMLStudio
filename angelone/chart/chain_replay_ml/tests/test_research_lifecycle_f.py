"""Tests for Phase F — reusable programs, manifest, evidence stopping."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research.campaign_manifest import build_campaign_manifest, empty_manifest
from chain_replay_ml.fold_research.campaign_stopping import evaluate_campaign_stop
from chain_replay_ml.fold_research.model_certification import build_model_certification
from chain_replay_ml.fold_research.program_execution_store import create_program_run
from chain_replay_ml.fold_research.research_objective import default_stopping_policy
from chain_replay_ml.fold_research.research_program import (
    create_research_campaign,
    create_research_program,
    update_research_campaign,
)
from chain_replay_ml.fold_research.campaign_scheduler import evaluate_campaign_should_stop


class ResearchLifecycleFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_campaign_hypothesis_and_stopping_fields(self) -> None:
        prog = create_research_program(
            self.tmp,
            name="OTM Buyer v1",
            program_type="strategy",
            stopping=default_stopping_policy(max_jobs=40),
        )
        pid = str((prog.get("program") or {}).get("program_id") or "")

        out = create_research_campaign(
            self.tmp,
            pid,
            name="Stop Optimization",
            research_question="What is the optimal stop loss?",
            hypothesis="7% stop increases PF.",
            success_criteria={"pf_delta_min": 0.15, "trade_count_min": 100},
            failure_criteria={"pf_delta_max": -0.02, "trade_count_min": 50},
            stopping={"min_jobs": 5, "max_jobs": 30, "auto_stop": True},
        )
        self.assertTrue(out.get("ok"))
        camp = out.get("campaign") or {}
        self.assertEqual(camp.get("hypothesis"), "7% stop increases PF.")
        self.assertEqual(camp.get("stopping", {}).get("max_jobs"), 30)

    def test_evidence_stop_success(self) -> None:
        decision = evaluate_campaign_stop(
            stopping={"min_jobs": 5, "max_jobs": 50, "auto_stop": True, "min_confidence_pct": 80, "min_generalization": 70},
            completed_jobs=12,
            comparison={"pf_delta": 0.25, "trade_count": 150},
            success_criteria={"pf_delta_min": 0.15},
            failure_criteria={"pf_delta_max": -0.02},
            generalization={"overall": 85},
            confidence_pct=90.0,
        )
        self.assertTrue(decision.get("should_stop"))
        self.assertEqual(decision.get("reason"), "success_criteria_met")

    def test_evidence_stop_below_min_jobs(self) -> None:
        decision = evaluate_campaign_stop(
            stopping=default_stopping_policy(),
            completed_jobs=3,
            comparison={"pf_delta": 0.5},
            success_criteria={"pf_delta_min": 0.1},
        )
        self.assertFalse(decision.get("should_stop"))
        self.assertEqual(decision.get("reason"), "below_min_jobs")

    def test_program_run_on_model(self) -> None:
        prog = create_research_program(self.tmp, name="Strategy Program", program_type="strategy")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        create_research_campaign(
            self.tmp,
            pid,
            name="Stop",
            research_question="Optimal stop?",
            hypothesis="Tighter stop helps.",
        )

        run_out = create_program_run(
            self.tmp,
            model_id="Future_LTP_5m_XGB_0035",
            program_id=pid,
            research_report_id="report-abc",
        )
        self.assertTrue(run_out.get("ok"))
        run = run_out.get("run") or {}
        self.assertEqual(run.get("model_id"), "Future_LTP_5m_XGB_0035")
        manifest = run.get("manifest") or {}
        self.assertEqual(manifest.get("program_type"), "strategy")
        self.assertGreaterEqual(manifest.get("total_campaigns", 0), 1)

    def test_manifest_checkpoint(self) -> None:
        prog = create_research_program(self.tmp, name="P1")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp_out = create_research_campaign(
            self.tmp,
            pid,
            name="Premium",
            research_question="What premium band?",
            hypothesis="20-30 band wins.",
        )
        cid = str((camp_out.get("campaign") or {}).get("campaign_id") or "")
        update_research_campaign(
            self.tmp,
            cid,
            status="running",
            manifest=empty_manifest(camp_out.get("campaign") or {}),
            memory={"experiments_run": 2, "model_id": "model_x"},
        )
        built = build_campaign_manifest(self.tmp, cid)
        self.assertTrue(built.get("ok"))
        man = built.get("manifest") or {}
        self.assertEqual(man.get("hypothesis"), "20-30 band wins.")
        self.assertEqual(man.get("model_id"), "model_x")

    def test_model_certification_empty(self) -> None:
        cert = build_model_certification(self.tmp, "SomeModel")
        self.assertTrue(cert.get("ok"))
        self.assertIn("certification", cert)
        self.assertFalse((cert.get("certification") or {}).get("production_ready"))

    def test_scheduler_stop_evaluator(self) -> None:
        prog = create_research_program(self.tmp, name="P2")
        pid = str((prog.get("program") or {}).get("program_id") or "")
        camp_out = create_research_campaign(
            self.tmp,
            pid,
            name="Hold",
            research_question="Best hold time?",
            stopping={"min_jobs": 2, "max_jobs": 5, "auto_stop": False},
            budget={"max_experiments": 5},
        )
        cid = str((camp_out.get("campaign") or {}).get("campaign_id") or "")
        update_research_campaign(
            self.tmp,
            cid,
            status="running",
            budget_used={"experiments": 5},
        )
        out = evaluate_campaign_should_stop(self.tmp, cid)
        self.assertTrue(out.get("ok"))
        self.assertTrue((out.get("decision") or {}).get("should_stop"))


if __name__ == "__main__":
    unittest.main()
