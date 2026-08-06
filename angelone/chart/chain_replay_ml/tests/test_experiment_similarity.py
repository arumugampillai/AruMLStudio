"""Tests for experiment similarity / duplicate prevention (KB Phase 2)."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.fold_research import (
    check_experiment_before_create,
    check_similar_experiments,
    complete_experiment,
    create_experiment_from_report,
    get_research_report,
)
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation

import numpy as np
import pandas as pd


class ExperimentSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "SimModel_v1",
                "dataset_name": "MS_sim",
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

        strat = create_strategy(self.tmp, display_name="Sim Strat", config=get_default_template())
        sim = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return run_id, sim["run"]["strategy_run_id"]

    def test_detects_very_similar_failed_experiment(self) -> None:
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
        complete_experiment(
            self.tmp,
            exp_id,
            results={
                "profit_factor_before": 1.3,
                "profit_factor_after": 1.25,
                "win_rate_before_pct": 65.0,
                "win_rate_after_pct": 63.0,
                "grade": "C",
            },
        )

        check = check_experiment_before_create(
            self.tmp,
            report,
            accepted_items=items,
            goal="Reduce theta decay failures",
        )
        self.assertTrue(check.get("ok"))
        self.assertGreaterEqual(check.get("top_similarity_pct") or 0, 80)
        self.assertEqual(check.get("verdict"), "very_similar")
        self.assertTrue(check.get("should_warn"))
        self.assertEqual((check.get("matches") or [{}])[0].get("outcome"), "no_improvement")

    def test_novel_experiment_low_overlap(self) -> None:
        run_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        check = check_similar_experiments(
            self.tmp,
            accepted_items=[{
                "text": "Increase Optuna trials",
                "target": "hyperparameter_optimization",
                "filters": {},
                "feature_hints": [],
            }],
            goal="Improve hyperparameter search",
            model_id="OtherModel_v9",
        )
        self.assertEqual(check.get("verdict"), "novel")
        self.assertFalse(check.get("should_warn"))


if __name__ == "__main__":
    unittest.main()
