"""Tests for Knowledge Base Phase 1 — evidence-backed findings."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research import (
    complete_experiment,
    create_experiment_from_report,
    get_knowledge_finding,
    get_research_report,
    list_knowledge_findings,
)
from chain_replay_ml.fold_research.finding_extraction import extract_findings_from_experiment
from chain_replay_ml.fold_research.knowledge_store import KnowledgeStore
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation

import numpy as np
import pandas as pd


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "KBModel_v1",
                "dataset_name": "MS_kb",
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

        strat = create_strategy(self.tmp, display_name="KB Strat", config=get_default_template())
        sim = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return run_id, sim["run"]["strategy_run_id"]

    def test_schema_tables(self) -> None:
        with KnowledgeStore(self.tmp) as store:
            tables = {
                r["name"]
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertIn("knowledge_findings", tables)
        self.assertIn("finding_evidence", tables)
        self.assertIn("finding_links", tables)

    def test_extraction_on_complete(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        items = [{
            "text": "Increase confidence threshold to 70%",
            "target": "strategy_registry",
            "target_label": "Strategy Registry",
            "filters": {"min_confidence": 70.0},
            "feature_hints": [],
        }]
        created = create_experiment_from_report(
            self.tmp,
            report,
            accepted_items=items,
            goal="Reduce theta decay failures",
        )
        exp_id = created["experiment"]["experiment_id"]
        done = complete_experiment(
            self.tmp,
            exp_id,
            results={
                "profit_factor_before": 1.2,
                "profit_factor_after": 1.45,
                "win_rate_before_pct": 62.0,
                "win_rate_after_pct": 68.0,
                "grade": "B+",
                "trade_count": 482,
            },
        )
        self.assertTrue(done.get("ok"))
        extraction = done.get("knowledge_extraction") or {}
        self.assertGreaterEqual(extraction.get("findings_updated") or 0, 1)

        findings = list_knowledge_findings(self.tmp)
        self.assertGreaterEqual(len(findings), 1)
        top = findings[0]
        self.assertIn(top["status"], ("candidate", "supported", "confirmed"))
        self.assertGreaterEqual(top["evidence_count"], 1)

        detail = get_knowledge_finding(self.tmp, top["finding_id"])
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertGreaterEqual(len(detail.get("evidence") or []), 1)
        self.assertTrue(any(e.get("experiment_id") == exp_id for e in detail["evidence"]))

    def test_lifecycle_promotion(self) -> None:
        experiment = {
            "experiment_id": "exp_test_1",
            "experiment_number": 1,
            "goal": "Test",
            "accepted_changes": [{
                "text": "Avoid premium below ₹26",
                "target": "strategy_registry",
                "filters": {"min_premium": 26},
                "feature_hints": [],
            }],
            "provenance": {"prediction_run_id": "run1", "model_id": "m1"},
            "results": {
                "profit_factor_before": 1.0,
                "profit_factor_after": 1.3,
                "win_rate_before_pct": 60,
                "win_rate_after_pct": 66,
            },
        }
        extract_findings_from_experiment(self.tmp, experiment, trade_count=500)
        with KnowledgeStore(self.tmp) as store:
            finding = store.get_finding_by_key("premium_below_26_poor")
            self.assertIsNotNone(finding)
            fid = finding["finding_id"]
            for i in range(2, 6):
                store.add_evidence(
                    fid,
                    {
                        "experiment_id": f"exp_{i}",
                        "experiment_number": i,
                        "supports_finding": True,
                        "trade_count": 400,
                        "pf_change": 0.15,
                        "evidence_quality": "moderate",
                    },
                )
            refreshed = store.get_finding(fid)
        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertIn(refreshed["status"], ("supported", "confirmed"))
        self.assertGreaterEqual(refreshed["evidence_count"], 4)


if __name__ == "__main__":
    unittest.main()
