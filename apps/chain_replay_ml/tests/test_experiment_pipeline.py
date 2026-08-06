"""Tests for Phase A — Proposal → Template → Job pipeline."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.fold_research import (
    create_experiment_proposal_from_report,
    create_experiment_proposal_from_suggestion,
    create_experiment_template_from_proposal,
    get_experiment_job,
    get_experiment_proposal,
    get_experiment_template,
    get_research_report,
    list_experiment_jobs,
    list_experiment_proposals,
    list_experiment_templates,
    run_experiment_template_job,
    update_experiment_proposal_selection,
)
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation


class ExperimentPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "PipeModel",
                "dataset_name": "MS_pipe",
                "target": "future_ltp_5m",
                "status": "completed",
            })
            run_id = run["run_id"]
            writer = PredictionRunWriter(store, run_id)
            ctx = pd.DataFrame({
                "trading_day": ["2026-07-01"] * 4,
                "timestamp": [100.0, 103.0, 106.0, 109.0],
                "token": ["TOK_A"] * 4,
                "strike": [24000.0] * 4,
                "option_type": ["CE"] * 4,
                "spot": [24050.0, 24055.0, 24060.0, 24065.0],
                "ltp": [20.0, 20.5, 22.5, 23.0],
            })
            writer.write_fold_predictions(
                fold_number=2,
                fold_def={"train": {"start": 0, "stop": 5}, "validation": {"start": 5, "stop": 9}},
                metrics={"mae": 0.8, "rmse": 1.0, "directional_accuracy_pct": 80.0},
                val_context=ctx,
                val_pred=np.array([21.0, 21.0, 24.0, 24.0]),
                val_y=pd.Series([20.0, 21.0, 22.5, 23.0]),
                baseline_ltp=pd.Series([19.5, 20.0, 21.0, 22.0]),
            )
            store.finalize_run(run_id, status="completed", prediction_count=4, fold_count=1)

        strat = create_strategy(self.tmp, display_name="Pipe Strat", config=get_default_template())
        sim = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return run_id, sim["run"]["strategy_run_id"]

    def test_proposal_template_job_lifecycle(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        self.assertTrue(report.get("ok"))

        out = create_experiment_proposal_from_report(self.tmp, report)
        self.assertTrue(out.get("ok"))
        proposal = out.get("proposal") or {}
        pid = str(proposal.get("proposal_id") or "")
        self.assertEqual(proposal.get("status"), "draft")
        self.assertTrue(proposal.get("available_recommendations"))

        available = proposal.get("available_recommendations") or []
        strategy_items = [
            i for i in available
            if str(i.get("target") or "strategy_registry") == "strategy_registry"
        ]
        pick = strategy_items[:2] if len(strategy_items) >= 2 else available[:2]
        keys = [str(i.get("key") or i.get("text")) for i in pick]
        self.assertTrue(keys, "expected strategy recommendations in report")
        out2 = update_experiment_proposal_selection(self.tmp, pid, selected_keys=keys)
        self.assertTrue(out2.get("ok"))
        proposal = out2.get("proposal") or {}
        self.assertEqual(len(proposal.get("selected_recommendations") or []), 2)
        score = proposal.get("score") or {}
        self.assertIn("overall", score)
        self.assertIn("tags", score)

        out3 = create_experiment_template_from_proposal(self.tmp, pid)
        self.assertTrue(out3.get("ok"))
        template = out3.get("template") or {}
        tid = str(template.get("template_id") or "")
        self.assertTrue(template.get("accepted_changes"))
        self.assertEqual(template.get("status"), "ready")

        converted = get_experiment_proposal(self.tmp, pid)
        self.assertEqual(converted.get("status"), "converted")
        self.assertEqual(converted.get("template_id"), tid)

        drafts = list_experiment_proposals(self.tmp, status="draft")
        self.assertFalse(any(p.get("proposal_id") == pid for p in drafts))

        out4 = run_experiment_template_job(self.tmp, tid)
        self.assertTrue(out4.get("ok"), out4.get("error"))
        job = out4.get("job") or {}
        self.assertEqual(job.get("status"), "complete")
        self.assertEqual(job.get("current_step"), "complete")

        outputs = job.get("outputs") or {}
        self.assertEqual(outputs.get("phase"), "B")
        self.assertTrue(outputs.get("strategy_version_id"))
        self.assertTrue(outputs.get("strategy_run_id"))
        self.assertTrue(outputs.get("research_report_id"))

        comparison = job.get("comparison") or {}
        self.assertIn("after_pf", comparison)
        self.assertIn("baseline_pf", comparison)

        loaded = get_experiment_job(self.tmp, str(job.get("job_id") or ""))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("template_id"), tid)

        templates = list_experiment_templates(self.tmp)
        self.assertTrue(any(t.get("template_id") == tid for t in templates))
        jobs = list_experiment_jobs(self.tmp, template_id=tid)
        self.assertEqual(len(jobs), 1)

        detail = get_experiment_template(self.tmp, tid)
        self.assertEqual(len(detail.get("accepted_changes") or []), 2)

    def test_partial_run_with_deferred_changes(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        out = create_experiment_proposal_from_report(self.tmp, report)
        proposal = out.get("proposal") or {}
        available = proposal.get("available_recommendations") or []
        strategy_keys = [
            str(i.get("key") or i.get("text"))
            for i in available
            if str(i.get("target") or "") == "strategy_registry"
        ]
        feature_keys = [
            str(i.get("key") or i.get("text"))
            for i in available
            if str(i.get("target") or "") in ("feature_registry", "model_builder")
        ]
        if not strategy_keys:
            self.skipTest("no strategy recommendations in report")
        keys = strategy_keys[:1]
        if feature_keys:
            keys.append(feature_keys[0])
        out2 = update_experiment_proposal_selection(self.tmp, str(proposal.get("proposal_id") or ""), selected_keys=keys)
        out3 = create_experiment_template_from_proposal(self.tmp, str(proposal.get("proposal_id") or ""))
        tid = str((out3.get("template") or {}).get("template_id") or "")
        out4 = run_experiment_template_job(self.tmp, tid)
        self.assertTrue(out4.get("ok"), out4.get("error"))
        job = out4.get("job") or {}
        outputs = job.get("outputs") or {}
        if feature_keys:
            self.assertIn(outputs.get("phase"), ("B", "B+C", "C"))
        self.assertTrue(outputs.get("strategy_run_id"))

    def test_create_proposal_from_suggestion(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        out = create_experiment_proposal_from_report(self.tmp, report)
        proposal = out.get("proposal") or {}
        available = proposal.get("available_recommendations") or []
        if len(available) < 2:
            self.skipTest("need at least two recommendations")
        keys = [str(i.get("key") or i.get("text")) for i in available[:2]]
        update_experiment_proposal_selection(self.tmp, str(proposal.get("proposal_id") or ""), selected_keys=keys)
        out3 = create_experiment_template_from_proposal(self.tmp, str(proposal.get("proposal_id") or ""))
        template = out3.get("template") or {}
        tid = str(template.get("template_id") or "")
        accepted = template.get("accepted_changes") or []
        suggestion = {
            "title": "Isolate first change",
            "goal": f"Isolate: {accepted[0].get('text')}",
            "selection": [accepted[0]],
            "expected_information_gain": "Very High",
            "stars": 5,
            "reason": "Single-change follow-up",
        }
        out4 = create_experiment_proposal_from_suggestion(self.tmp, tid, suggestion)
        self.assertTrue(out4.get("ok"), out4.get("error"))
        follow_up = out4.get("proposal") or {}
        self.assertEqual(follow_up.get("status"), "draft")
        self.assertEqual(len(follow_up.get("selected_recommendations") or []), 1)
        self.assertEqual(len(follow_up.get("available_recommendations") or []), len(accepted))
        score = follow_up.get("score") or {}
        self.assertEqual((score.get("follow_up") or {}).get("parent_template_id"), tid)
        self.assertIn("follow_up", follow_up.get("tags") or [])


if __name__ == "__main__":
    unittest.main()
