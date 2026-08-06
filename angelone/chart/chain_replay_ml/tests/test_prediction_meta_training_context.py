"""Tests for training dataset scope in prediction analytics."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.prediction_meta.training_context import (
    build_training_scope_sql,
    extract_selection_criteria,
    resolve_training_context,
)


class TestTrainingContext(unittest.TestCase):
    def test_extract_selection_criteria_from_master_export(self) -> None:
        meta = {
            "selection_method": {
                "summary": "All days · ATM ±10 · LTP 15–300 · Delta off",
                "criteria": {
                    "atm_band_filter": 10,
                    "premium_enabled": True,
                    "premium_min": 15.0,
                    "premium_max": 300.0,
                    "delta_enabled": False,
                },
            }
        }
        crit = extract_selection_criteria(meta)
        self.assertEqual(crit["premium_min"], 15.0)
        self.assertEqual(crit["atm_band_filter"], 10)

    def test_build_scope_sql_premium_and_atm(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE samples (
                current_ltp REAL, strike REAL, current_spot REAL,
                actual_5m_ltp REAL, ensemble_mean REAL
            )
            """
        )
        sql, info = build_training_scope_sql(
            conn,
            {
                "premium_enabled": True,
                "premium_min": 15,
                "premium_max": 300,
                "atm_band_filter": 10,
            },
        )
        self.assertTrue(info["active"])
        self.assertIn("current_ltp >= 15", sql)
        self.assertIn("current_ltp <= 300", sql)
        self.assertIn("strike", sql)
        conn.close()

    def test_resolve_missing_models(self) -> None:
        out = resolve_training_context("/tmp", [])
        self.assertFalse(out["resolved"])


if __name__ == "__main__":
    unittest.main()
