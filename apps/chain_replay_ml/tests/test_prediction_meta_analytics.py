"""Tests for prediction model analytics."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.prediction_meta.model_analytics import (
    compute_model_leaderboard,
    diagnose_row_error,
    read_prediction_model_analytics,
)
from chain_replay_ml.prediction_meta.model_registry import ensure_registry_tables


class TestModelAnalytics(unittest.TestCase):
    def _seed_db(self, path: str) -> None:
        conn = sqlite3.connect(path)
        ensure_registry_tables(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                prediction_id TEXT PRIMARY KEY,
                trading_day TEXT,
                timestamp REAL,
                token TEXT,
                strike REAL,
                option_type TEXT,
                expiry TEXT,
                current_ltp REAL,
                current_spot REAL,
                minutes_to_expiry REAL,
                ensemble_mean REAL,
                ensemble_median REAL,
                ensemble_spread REAL,
                agreement REAL,
                direction_correct REAL,
                actual_5m_ltp REAL,
                actual_max_profit_5m REAL,
                model_1_pred REAL,
                model_2_pred REAL
            );
            INSERT INTO prediction_model_registry
                (prediction_version, slot, model_id, model_name, target, registered_at)
            VALUES
                (1, 'model_1', 'M1', 'Model_One', 'future_ltp_5m', '2026-01-01'),
                (1, 'model_2', 'M2', 'Model_Two', 'future_ltp_5m', '2026-01-01');
            INSERT INTO samples VALUES
                ('a', '2026-05-27', 1716782400, '1', 25000, 'CE', '2026-05-29', 10, 24500, 800, 12, 11.5, 1.0, 0.95, 1, 11, 2.0, 11, 13),
                ('b', '2026-05-27', 1716786000, '2', 24000, 'PE', '2026-05-29', 10, 24500, 800, 12, 11.5, 4.0, 0.55, 0, 9, 1.0, 14, 15),
                ('c', '2026-05-27', 1716793200, '3', 26000, 'CE', '2026-05-27', 6.8, 24500, 120, 16.5, 15, 8.0, 0.4, 0, 6.8, 0.5, 18, 17);
            """
        )
        conn.commit()
        conn.close()

    def test_leaderboard_signed_error_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pred.db")
            self._seed_db(path)
            conn = sqlite3.connect(path)
            ensure_registry_tables(conn)
            rows = compute_model_leaderboard(
                conn,
                model_slots=[
                    {"slot": "model_1", "model_name": "Model_One"},
                    {"slot": "model_2", "model_name": "Model_Two"},
                ],
                actual_col="actual_5m_ltp",
            )
            conn.close()
            m1 = next(r for r in rows if r["model_index"] == 1)
            self.assertEqual(m1["mean_positive_error"], 8.1)
            self.assertIsNone(m1["mean_negative_error"])
            self.assertEqual(m1["over_prediction_pct"], 66.7)
            self.assertEqual(m1["bias"], 5.4)

    def test_leaderboard_and_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pred.db")
            self._seed_db(path)
            out = read_prediction_model_analytics(path, data_dir=tmp)
            self.assertTrue(out["exists"])
            self.assertNotIn("error", out)
            self.assertEqual(out["target_column"], "future_ltp_5m")
            self.assertEqual(out["actual_horizon"], "5m")
            self.assertEqual(out["actual_column"], "actual_5m_ltp")
            self.assertEqual(out["eval_rows"], 3)
            self.assertEqual(len(out["leaderboard"]), 2)
            self.assertIn("ensemble_comparison", out)
            self.assertTrue(out["agreement_buckets"])
            self.assertTrue(out["premium_buckets"])
            self.assertTrue(out["calibration_scatter"])
            self.assertTrue(out["spread_buckets"])
            self.assertIn("high_error_sample", out)
            diag = out["high_error_sample"]["error_diagnosis"]
            self.assertTrue(diag["overestimate"])
            self.assertIn("premium < 10", diag["flags"])

    def test_diagnose_row_error(self) -> None:
        row = {
            "current_ltp": 6.8,
            "ensemble_mean": 16.5,
            "actual_5m_ltp": 6.8,
            "strike": 26000,
            "current_spot": 24500,
            "option_type": "CE",
            "trading_day": "2026-05-27",
            "expiry": "2026-05-27",
            "timestamp": 1716793200.0,
        }
        d = diagnose_row_error(row, actual_col="actual_5m_ltp")
        self.assertEqual(d["bias"], 9.7)
        self.assertTrue(d["deep_otm"])
        self.assertTrue(d["expiry_day"])


if __name__ == "__main__":
    unittest.main()
