"""Tests for master dataset metadata tables."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_dataset_service import MasterDatasetService
from chain_replay_ml.dataset_builder.master_store import MasterStore


class TestMasterMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "master_test.db")

    def _store(self) -> MasterStore:
        store = MasterStore(self.db_path)
        store.open()
        return store

    def test_commit_day_updates_metadata(self) -> None:
        store = self._store()
        try:
            cols = ["trading_day", "timestamp", "token", "expiry", "option_type"]
            store.begin_day("2026-01-02", cols)
            store.insert_rows([
                {
                    "trading_day": "2026-01-02",
                    "timestamp": 1000.0,
                    "token": "A",
                    "expiry": "2026-01-08",
                    "option_type": "CE",
                },
                {
                    "trading_day": "2026-01-02",
                    "timestamp": 1003.0,
                    "token": "B",
                    "expiry": "2026-01-08",
                    "option_type": "PE",
                },
            ])
            store.commit_day("2026-01-02")
            meta = store.read_master_meta()
            self.assertEqual(meta.total_rows, 2)
            self.assertEqual(meta.total_days, 1)
            self.assertGreater(meta.metadata_version, 0)
            days = store.read_master_days()
            self.assertEqual(len(days), 1)
            self.assertEqual(days[0]["row_count"], 2)
            self.assertEqual(days[0]["token_count"], 2)
            self.assertEqual(days[0]["expiry_count"], 1)
            self.assertEqual(days[0]["is_expiry_day"], 0)
        finally:
            store.close()

    def test_commit_day_marks_expiry_day(self) -> None:
        store = self._store()
        try:
            cols = ["trading_day", "timestamp", "token", "expiry", "option_type"]
            store.begin_day("2026-01-08", cols)
            store.insert_rows([
                {
                    "trading_day": "2026-01-08",
                    "timestamp": 1000.0,
                    "token": "A",
                    "expiry": "2026-01-08",
                    "option_type": "CE",
                },
            ])
            store.commit_day("2026-01-08")
            days = store.read_master_days()
            self.assertEqual(days[0]["is_expiry_day"], 1)
            self.assertEqual(days[0]["dominant_expiry"], "2026-01-08")
        finally:
            store.close()

    def test_read_status_includes_feature_project_id(self) -> None:
        store = self._store()
        try:
            from chain_replay_ml.dataset_builder.master_feature_project import set_master_feature_project_id

            set_master_feature_project_id(store, self.tmp, "all")
            svc = MasterDatasetService(self.db_path)
            status = svc.read_status(data_dir=self.tmp, market="NIFTY", interval_sec=10)
            self.assertEqual(status.get("feature_project_id"), "all")
        finally:
            store.close()

    def test_backfill_feature_project_id_defaults_to_all(self) -> None:
        store = self._store()
        try:
            store.conn.execute(
                "UPDATE master_dataset_meta SET feature_project_id = NULL WHERE id = 1"
            )
            store.conn.commit()
            store._backfill_feature_project_id()
            meta = store.read_master_meta_dict()
            self.assertEqual(meta.get("feature_project_id"), "all")
            cfg = store.get_meta("master_config")
            self.assertIsInstance(cfg, dict)
            self.assertEqual(cfg.get("feature_project_id"), "all")
        finally:
            store.close()

    def test_delete_day_updates_metadata(self) -> None:
        store = self._store()
        try:
            cols = ["trading_day", "timestamp", "token"]
            store.begin_day("2026-01-02", cols)
            store.insert_rows([
                {"trading_day": "2026-01-02", "timestamp": 1000.0, "token": "A"},
            ])
            store.commit_day("2026-01-02")
            deleted = store.delete_day("2026-01-02")
            self.assertEqual(deleted, 1)
            meta = store.read_master_meta()
            self.assertEqual(meta.total_rows, 0)
            self.assertEqual(meta.total_days, 0)
            self.assertEqual(store.read_master_days(), [])
        finally:
            store.close()

    def test_backfill_existing_samples(self) -> None:
        store = self._store()
        try:
            store.ensure_columns(["trading_day", "timestamp", "token", "expiry"])
            store.conn.executemany(
                "INSERT INTO samples (trading_day, timestamp, token, expiry) VALUES (?, ?, ?, ?)",
                [
                    ("2026-01-02", 1000.0, "A", "2026-01-08"),
                    ("2026-01-02", 1003.0, "B", "2026-01-08"),
                    ("2026-01-03", 2000.0, "C", "2026-01-08"),
                ],
            )
            store.conn.commit()
            meta = store.refresh_metadata_from_samples(reason="BACKFILL")
            self.assertEqual(meta.total_rows, 3)
            self.assertEqual(meta.total_days, 2)
            self.assertEqual(len(store.read_master_days()), 2)
        finally:
            store.close()

    def test_service_reads_metadata_only(self) -> None:
        store = self._store()
        try:
            cols = ["trading_day", "timestamp", "token"]
            store.begin_day("2026-01-02", cols)
            store.insert_rows([
                {"trading_day": "2026-01-02", "timestamp": 1000.0, "token": "A"},
            ])
            store.commit_day("2026-01-02")
        finally:
            store.close()

        svc = MasterDatasetService(self.db_path)
        status = svc.read_status(data_dir=self.tmp, market="NIFTY", interval_sec=10)
        self.assertEqual(status["row_count"], 1)
        self.assertEqual(status["days_in_master"], ["2026-01-02"])
        details = svc.read_day_details(status.get("coverage_by_day"))
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["row_count"], 1)


    def test_distribution_incremental(self) -> None:
        store = self._store()
        try:
            cols = [
                "trading_day", "timestamp", "token", "ltp",
                "strike_distance_from_atm", "abs_delta",
            ]
            store.begin_day("2026-01-02", cols)
            store.insert_rows([
                {
                    "trading_day": "2026-01-02",
                    "timestamp": 1000.0,
                    "token": "A",
                    "ltp": 15.0,
                    "strike_distance_from_atm": 0,
                    "abs_delta": 0.15,
                },
                {
                    "trading_day": "2026-01-02",
                    "timestamp": 1003.0,
                    "token": "B",
                    "ltp": 25.0,
                    "strike_distance_from_atm": 1,
                    "abs_delta": 0.25,
                },
            ])
            store.commit_day("2026-01-02")
            dist = store.read_master_distributions()
            self.assertTrue(any(d["distribution_type"] == "PREMIUM" for d in dist))
            self.assertTrue(any(d["distribution_type"] == "ATM" for d in dist))
            self.assertTrue(any(d["distribution_type"] == "DELTA" for d in dist))
            fp = store.read_dataset_fingerprint()
            self.assertIn("features", fp)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
