"""Tests for Knowledge Base Phase 3 retrieval."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research import (
    complete_experiment,
    create_experiment_from_report,
    get_feature_knowledge,
    get_known_findings_for_report,
    get_research_report,
    score_experiment_proposal,
)
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation

import numpy as np
import pandas as pd


class KnowledgeRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "KB3Model",
                "dataset_name": "MS_kb3",
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

        strat = create_strategy(self.tmp, display_name="KB3 Strat", config=get_default_template())
        sim = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return run_id, sim["run"]["strategy_run_id"]

    def test_known_findings_for_report(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        kb = get_known_findings_for_report(self.tmp, report)
        self.assertTrue(kb.get("ok"))
        self.assertIn("findings", kb)

    def test_score_experiment_proposal(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        items = [{
            "text": "Increase confidence threshold to 70%",
            "target": "strategy_registry",
            "filters": {"min_confidence": 70.0},
            "feature_hints": [],
        }]
        score = score_experiment_proposal(
            self.tmp,
            report,
            accepted_items=items,
            goal="Reduce theta decay failures",
        )
        self.assertIn("novelty_score", score)
        self.assertIn("improvement_probability", score)
        self.assertIn("evidence_quality", score)

    def test_feature_knowledge_after_experiment(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        items = [{
            "text": "Retrain model with theta / regime features",
            "target": "feature_registry",
            "filters": {},
            "feature_hints": ["theta"],
        }]
        created = create_experiment_from_report(self.tmp, report, accepted_items=items, goal="Reduce theta decay")
        complete_experiment(
            self.tmp,
            created["experiment"]["experiment_id"],
            results={
                "profit_factor_before": 1.1,
                "profit_factor_after": 1.35,
                "win_rate_before_pct": 60,
                "win_rate_after_pct": 67,
                "grade": "B",
            },
        )
        hints = get_feature_knowledge(self.tmp, ["theta"])
        self.assertGreaterEqual(len(hints), 1)


if __name__ == "__main__":
    unittest.main()
