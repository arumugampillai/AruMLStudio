"""Tests for Phase 5 fold research and replay."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.fold_research import (
    create_experiment_from_report,
    get_experiment,
    get_experiment_planner_view,
    get_fold_replay_timeline,
    get_fold_research,
    get_prediction_run_summary,
    get_research_report,
    launch_experiment,
    list_saved_research_reports,
    save_research_report_to_store,
)
from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation


class FoldResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed(self) -> tuple[str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "FoldModel_v1",
                "dataset_name": "MS_fold",
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
            fold_id = store.list_folds(run_id)[0]["fold_id"]

        strat = create_strategy(self.tmp, display_name="Fold Strat", config=get_default_template())
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        sim = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return run_id, fold_id, sim["run"]["strategy_run_id"]

    def test_fold_research_detail(self) -> None:
        run_id, fold_id, sr_id = self._seed()
        doc = get_fold_research(
            self.tmp,
            prediction_run_id=run_id,
            fold_id=fold_id,
            strategy_run_id=sr_id,
        )
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["fold"]["fold_number"], 2)
        self.assertIsNotNone(doc["prediction_quality"]["mae"])
        self.assertTrue(doc["market_summary"]["available"])
        self.assertGreater(doc["market_summary"]["row_count"], 0)
        self.assertIsNotNone(doc["trading"])
        self.assertGreater(doc["replay"]["event_count"], 0)

    def test_fold_replay_timeline(self) -> None:
        run_id, fold_id, sr_id = self._seed()
        doc = get_fold_replay_timeline(
            self.tmp,
            prediction_run_id=run_id,
            fold_id=fold_id,
            strategy_run_id=sr_id,
        )
        self.assertTrue(doc["ok"])
        self.assertGreater(doc["total"], 0)
        types = {e["event_type"] for e in doc["events"]}
        self.assertIn("prediction", types)

    def test_prediction_analysis_calibration(self) -> None:
        from chain_replay_ml.fold_research.prediction_analysis import analyze_prediction_rows

        rows = [
            {"ltp": 20.0, "predicted_ltp": 21.0, "actual_ltp": 20.5, "prediction_error": 0.5, "direction_correct": 1},
            {"ltp": 21.0, "predicted_ltp": 22.0, "actual_ltp": 21.5, "prediction_error": 0.5, "direction_correct": 1},
        ]
        q = analyze_prediction_rows(rows)
        self.assertAlmostEqual(q["mae"], 0.5)
        self.assertGreaterEqual(len(q["calibration_buckets"]), 1)

    def test_prediction_run_summary(self) -> None:
        run_id, _fold_id, sr_id = self._seed()
        doc = get_prediction_run_summary(
            self.tmp,
            run_id,
            strategy_run_id=sr_id,
        )
        self.assertTrue(doc.get("ok"))
        self.assertGreaterEqual(doc.get("fold_count") or 0, 1)
        self.assertIn("baseline_metrics", doc)
        self.assertIn("root_causes", doc)
        self.assertIn("recommendations", doc)

    def test_research_report_sections(self) -> None:
        run_id, _fold_id, sr_id = self._seed()
        doc = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        self.assertTrue(doc.get("ok"))
        self.assertIn("executive_summary", doc)
        self.assertIn("root_cause_analysis", doc)
        self.assertIn("opportunity_analysis", doc)
        self.assertIn("fold_ranking", doc)
        self.assertIn("recommendations", doc)
        self.assertIn("action_plan", doc)
        exec_sum = doc["executive_summary"]
        self.assertIn("overall_grade", exec_sum)
        self.assertIn("recommendation_flags", exec_sum)

    def test_research_report_store(self) -> None:
        run_id, _fold_id, sr_id = self._seed()
        doc = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        saved = save_research_report_to_store(self.tmp, doc)
        self.assertTrue(saved.get("report_id"))
        rows = list_saved_research_reports(self.tmp, prediction_run_id=run_id)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0].get("grade"), doc["executive_summary"].get("overall_grade"))

    def test_experiment_planner_categorization(self) -> None:
        run_id, _fold_id, sr_id = self._seed()
        doc = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        view = get_experiment_planner_view(self.tmp, doc)
        self.assertTrue(view.get("ok"))
        self.assertIn("items", view)
        self.assertTrue(view.get("suggested_goal"))

    def test_experiment_lifecycle(self) -> None:
        run_id, _fold_id, sr_id = self._seed()
        report = get_research_report(self.tmp, run_id, strategy_run_id=sr_id)
        view = get_experiment_planner_view(self.tmp, report)
        items = view.get("items") or [{"text": "Increase confidence threshold to 70%", "target": "strategy_registry", "target_label": "Strategy Registry", "filters": {}, "feature_hints": []}]
        created = create_experiment_from_report(self.tmp, report, accepted_items=items[:1], goal="Test goal")
        self.assertTrue(created.get("ok"))
        exp = created.get("experiment") or {}
        exp_id = exp.get("experiment_id")
        self.assertTrue(exp_id)
        self.assertEqual(exp.get("status"), "pending")
        self.assertEqual(exp.get("goal"), "Test goal")

        launched = launch_experiment(self.tmp, str(exp_id))
        self.assertTrue(launched.get("ok"))
        loaded = get_experiment(self.tmp, str(exp_id))
        self.assertEqual(loaded.get("status"), "launched")


if __name__ == "__main__":
    unittest.main()
