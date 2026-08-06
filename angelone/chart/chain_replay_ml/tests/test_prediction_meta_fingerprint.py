"""Tests for prediction build fingerprint."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from chain_replay_ml.prediction_meta.build_fingerprint import compute_build_fingerprint, merge_fingerprint
from chain_replay_ml.prediction_meta.projects import create_project, list_projects


class TestBuildFingerprint(unittest.TestCase):
    def test_compute_fingerprint_fields(self) -> None:
        project = {
            "project_id": "prediction_nifty_3s_4d_10mod",
            "display_name": "Prediction_NIFTY_3s_4d_10mod",
            "db_filename": "prediction_nifty_3s_4d_10mod.db",
            "source_master_db": "master_dataset_nifty_3s.db",
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "selected_models": ["M1", "M2"],
            "created_at": "2026-07-06T10:00:00+00:00",
            "feature_version": "fe-sc15-tp42",
            "trading_days_filter": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"],
        }
        fp = compute_build_fingerprint(
            project,
            master_row_count=1_773_188,
            prediction_version=1,
            model_registry_version="sig|models:a,b",
            inference_registry_signature="sig-part-1|sig-part-2|sig-part-3",
            model_registry_slot_count=3,
            build_status="complete",
            completed_at="2026-07-07T04:00:00+00:00",
        )
        self.assertEqual(fp["prediction_dataset"], "Prediction_NIFTY_3s_4d_10mod")
        self.assertEqual(fp["master_dataset"], "master_dataset_nifty_3s.db")
        self.assertEqual(fp["rows"], 1_773_188)
        self.assertEqual(fp["models_count"], 2)
        self.assertEqual(fp["feature_version"], "fe-sc15-tp42")
        self.assertEqual(fp["prediction_version"], 1)
        self.assertEqual(fp["model_registry_version_display"], "3")
        self.assertEqual(fp["created_label"], "2026-07-06")
        self.assertEqual(fp["completed_label"], "2026-07-07")
        self.assertTrue(fp["fingerprint_id"])

    def test_merge_updates_id(self) -> None:
        merged = merge_fingerprint({"prediction_dataset": "A"}, {"rows_written": 100})
        self.assertEqual(merged["rows_written"], 100)
        self.assertTrue(merged["fingerprint_id"])

    def test_create_project_has_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            create_project(
                tmp,
                display_name="Prediction_Test_FP",
                source_master_db="master_dataset_nifty_3s.db",
                selected_models=["Model_A"],
            )
            rows = list_projects(tmp)
            self.assertEqual(len(rows), 1)
            fp = rows[0].get("build_fingerprint")
            self.assertIsInstance(fp, dict)
            self.assertEqual(fp.get("prediction_dataset"), "Prediction_Test_FP")
            self.assertEqual(fp.get("build_status"), "pending")
            self.assertTrue(fp.get("fingerprint_id"))


if __name__ == "__main__":
    unittest.main()
