"""Tests for unified DatasetSelectionEngine."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.dataset_builder.dataset_selection_engine import (
    DatasetSelectionEngine,
    DatasetSelectionSpec,
    build_selection_sql_where,
)
from chain_replay_ml.dataset_builder.selection_preview_calibration import (
    load_selection_calibration,
    record_selection_calibration,
)
from chain_replay_ml.prediction_meta.training_context import build_training_scope_sql


class TestDatasetSelectionEngine(unittest.TestCase):
    def test_master_sql_where_premium_atm(self) -> None:
        spec = DatasetSelectionSpec(
            single_day="2026-05-27",
            atm_band=5,
            premium_min=20.0,
            premium_max=100.0,
            premium_enabled=True,
        )
        sql, params = build_selection_sql_where(
            spec,
            profile="master_samples",
            column_names={"trading_day", "ltp", "strike_distance_from_atm"},
        )
        self.assertIn("trading_day = ?", sql)
        self.assertIn('"ltp" >= ?', sql)
        self.assertIn('ABS("strike_distance_from_atm") <= ?', sql)
        self.assertEqual(params[0], "2026-05-27")

    def test_prediction_meta_scope_sql(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE samples (
                current_ltp REAL, strike REAL, current_spot REAL,
                abs_delta REAL, actual_5m_ltp REAL
            )
            """
        )
        spec = DatasetSelectionSpec.from_registry_criteria({
            "premium_enabled": True,
            "premium_min": 15,
            "premium_max": 300,
            "atm_band_filter": 10,
        })
        sql, params = build_selection_sql_where(
            spec,
            profile="prediction_meta",
            column_names={"current_ltp", "strike", "current_spot"},
            param_style="inline",
        )
        self.assertIn("current_ltp >= 15", sql)
        self.assertIn("current_ltp <= 300", sql)
        self.assertIn("strike", sql)
        scope_sql, info = build_training_scope_sql(conn, spec.to_filter_summary_dict())
        self.assertTrue(info["active"])
        self.assertIn("current_ltp >= 15", scope_sql)
        conn.close()

    def test_preview_exact_without_filters(self) -> None:
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "master.db")
        store = MasterStore(db_path)
        store.open()
        try:
            cols = ["trading_day", "timestamp", "token", "ltp", "strike_distance_from_atm", "abs_delta"]
            store.begin_day("2026-05-27", cols)
            store.insert_rows([
                {"trading_day": "2026-05-27", "timestamp": 1.0, "token": "A", "ltp": 25.0,
                 "strike_distance_from_atm": 0, "abs_delta": 0.15},
                {"trading_day": "2026-05-27", "timestamp": 2.0, "token": "B", "ltp": 5.0,
                 "strike_distance_from_atm": 2, "abs_delta": 0.25},
            ])
            store.commit_day("2026-05-27")
        finally:
            store.close()

        spec = DatasetSelectionSpec(selected_days=["2026-05-27"])
        result = DatasetSelectionEngine(spec, db_path).preview()
        self.assertEqual(result.accuracy, "exact")
        self.assertEqual(result.estimated_rows, 2)

    def test_preview_estimated_with_premium(self) -> None:
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "master.db")
        store = MasterStore(db_path)
        store.open()
        try:
            cols = ["trading_day", "timestamp", "token", "ltp", "strike_distance_from_atm", "abs_delta"]
            store.begin_day("2026-05-27", cols)
            store.insert_rows([
                {"trading_day": "2026-05-27", "timestamp": 1.0, "token": "A", "ltp": 25.0,
                 "strike_distance_from_atm": 0, "abs_delta": 0.15},
                {"trading_day": "2026-05-27", "timestamp": 2.0, "token": "B", "ltp": 5.0,
                 "strike_distance_from_atm": 2, "abs_delta": 0.25},
            ])
            store.commit_day("2026-05-27")
        finally:
            store.close()

        spec = DatasetSelectionSpec(
            selected_days=["2026-05-27"],
            premium_min=20.0,
            premium_max=100.0,
            premium_enabled=True,
        )
        result = DatasetSelectionEngine(spec, db_path).preview()
        self.assertEqual(result.accuracy, "estimated")
        self.assertEqual(result.estimated_rows, 1)

    def test_calibration_record(self) -> None:
        tmp = tempfile.mkdtemp()
        spec = DatasetSelectionSpec(selected_days=["2026-05-27"], atm_band=5)
        preview = DatasetSelectionEngine(spec).estimate_from_metadata(
            {"metadata_version": 3, "total_rows": 100, "database_size": 4096},
            [{"trading_day": "2026-05-27", "row_count": 100, "token_count": 10}],
            [],
        )
        record = record_selection_calibration(
            tmp,
            build_kind="master_build",
            spec=spec,
            preview=preview,
            actual_rows=95,
        )
        self.assertEqual(record["actual"]["rows"], 95)
        self.assertIsNotNone(record["deltas"]["row_error_pct"])
        history = load_selection_calibration(tmp, limit=5)
        self.assertEqual(len(history), 1)

    def test_from_strike_selection(self) -> None:
        spec = DatasetSelectionSpec.from_strike_selection(
            {"mode": "premium_band", "premiumMin": 15, "premiumMax": 30},
            selected_days=["2026-01-02"],
        )
        self.assertEqual(spec.mode, "premium_band")
        self.assertEqual(spec.premium_min, 15.0)
        strike_dict = spec.to_strike_selection_dict()
        self.assertEqual(strike_dict["mode"], "premium_band")


if __name__ == "__main__":
    unittest.main()
