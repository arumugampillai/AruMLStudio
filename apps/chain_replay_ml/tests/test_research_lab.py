"""Tests for Phase 4 research lab."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.research_lab import (
    build_leaderboard,
    build_research_matrix,
    compare_strategy_runs,
    create_research_session,
    get_matrix,
    list_research_sessions,
)
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import run_strategy_simulation


class ResearchLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_strategy_run(self) -> str:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "ResearchModel_v1",
                "dataset_name": "MS_research",
                "target": "future_ltp_5m",
                "status": "completed",
            })
            run_id = run["run_id"]
            writer = PredictionRunWriter(store, run_id)
            ctx = pd.DataFrame({
                "trading_day": ["2026-07-01"] * 3,
                "timestamp": [100.0, 103.0, 106.0],
                "token": ["TOK_A"] * 3,
                "strike": [24000.0] * 3,
                "option_type": ["CE"] * 3,
                "spot": [24050.0] * 3,
                "ltp": [20.0, 20.5, 22.5],
            })
            writer.write_fold_predictions(
                fold_number=1,
                fold_def={"train": {"start": 0, "stop": 5}, "validation": {"start": 5, "stop": 8}},
                metrics={"mae": 1.0, "rmse": 1.1, "directional_accuracy_pct": 70.0},
                val_context=ctx,
                val_pred=np.array([21.0, 21.0, 24.0]),
                val_y=pd.Series([20.0, 21.0, 22.5]),
                baseline_ltp=pd.Series([19.5, 20.0, 21.0]),
            )
            store.finalize_run(run_id, status="completed", prediction_count=3, fold_count=1)

        strat = create_strategy(self.tmp, display_name="Research Strat", config=get_default_template())
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        detail = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=strat["champion_version"]["version_id"],
        )
        return detail["run"]["strategy_run_id"]

    def test_matrix_and_leaderboard(self) -> None:
        sr_id = self._seed_strategy_run()
        matrix = build_research_matrix(self.tmp)
        self.assertTrue(matrix["ok"])
        self.assertEqual(matrix["row_count"], 1)
        row = matrix["rows"][0]
        self.assertEqual(row["model_id"], "ResearchModel_v1")
        self.assertEqual(row["strategy_name"], "Research Strat")
        self.assertGreater(row.get("trade_count") or 0, 0)

        board = build_leaderboard(self.tmp, mode="highest_profit")
        self.assertEqual(len(board["leaderboard"]), 1)
        self.assertEqual(board["leaderboard"][0]["strategy_run_id"], sr_id)

    def test_get_matrix_includes_grid(self) -> None:
        self._seed_strategy_run()
        doc = get_matrix(self.tmp)
        self.assertIn("grid", doc)
        self.assertIn("ResearchModel_v1", doc["grid"])

    def test_compare_strategy_runs(self) -> None:
        a = self._seed_strategy_run()
        b = self._seed_strategy_run()
        result = compare_strategy_runs(self.tmp, [a, b])
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["runs"]), 2)

    def test_research_session(self) -> None:
        sr_id = self._seed_strategy_run()
        created = create_research_session(self.tmp, title="EMA Channel Study", notes="phase 4 test")
        sid = created["session"]["session_id"]
        from chain_replay_ml.research_lab import update_research_session

        updated = update_research_session(
            self.tmp,
            sid,
            {"strategy_run_ids": [sr_id]},
        )
        self.assertTrue(updated["ok"])
        sessions = list_research_sessions(self.tmp)
        self.assertEqual(len(sessions["sessions"]), 1)

    def test_sessions_db_path(self) -> None:
        from chain_replay_ml.research_lab.paths import research_sessions_db_path

        path = research_sessions_db_path(self.tmp)
        self.assertTrue(path.endswith(os.path.join("research_lab", "sessions.db")))


if __name__ == "__main__":
    unittest.main()
