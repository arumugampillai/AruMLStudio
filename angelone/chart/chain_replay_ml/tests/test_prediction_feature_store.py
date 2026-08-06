"""Tests for master_row_id + PredictionFeatureStore lean schema."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_store import MasterStore
from chain_replay_ml.model_lab.prediction_feature_store import (
    PredictionFeatureStore,
    detect_feature_storage_mode,
    referenced_feature_column_map,
)
from chain_replay_ml.model_lab.prediction_schema import (
    FEATURE_STORAGE_EMBEDDED,
    FEATURE_STORAGE_REFERENCED,
    LAB_SCHEMA_VERSION_PREDICTION,
)
from chain_replay_ml.model_lab.store import ModelLabStore


class MasterRowIdTests(unittest.TestCase):
    def test_assign_and_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "master.db")
            with MasterStore(path) as store:
                store.begin_day("2026-01-02", ["trading_day", "timestamp", "token", "ltp"])
                n = store.insert_rows(
                    [
                        {
                            "trading_day": "2026-01-02",
                            "timestamp": 1.0,
                            "token": "t1",
                            "ltp": 10.0,
                        },
                        {
                            "trading_day": "2026-01-02",
                            "timestamp": 2.0,
                            "token": "t2",
                            "ltp": 11.0,
                        },
                    ]
                )
                store.commit_day("2026-01-02")
                self.assertEqual(n, 2)
                rows = store.conn.execute(
                    "SELECT master_row_id, token FROM samples ORDER BY timestamp"
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertIsNotNone(rows[0][0])
                self.assertIsNotNone(rows[1][0])
                self.assertNotEqual(rows[0][0], rows[1][0])
                first_id = int(rows[0][0])

            # Re-open: IDs stable (no renumber)
            with MasterStore(path) as store:
                again = store.conn.execute(
                    "SELECT master_row_id FROM samples WHERE token='t1'"
                ).fetchone()
                self.assertEqual(int(again[0]), first_id)


class PredictionFeatureStoreTests(unittest.TestCase):
    def test_embedded_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            with ModelLabStore(lab) as store:
                store.ensure_prediction_schema()
                store.ensure_feature_columns(["sf_foo"])
                store.write_prediction_summary(
                    lab_uuid="u1",
                    status="ready",
                    row_count=1,
                    trading_days=1,
                    feature_columns_json='{"foo": "sf_foo"}',
                    feature_storage_mode=FEATURE_STORAGE_EMBEDDED,
                )
                store.insert_prediction_rows(
                    [
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "p1",
                            "trading_day": "2026-01-02",
                            "timestamp": 1.0,
                            "target_reached": 1,
                            "sf_foo": 3.5,
                        }
                    ],
                    feature_columns=["sf_foo"],
                )
                access = PredictionFeatureStore.from_store(store)
                self.assertEqual(access.storage_mode(), FEATURE_STORAGE_EMBEDDED)
                self.assertEqual(access.feature_map(), [("foo", "sf_foo")])
                rows = access.fetch_rows(
                    outcome_cols=["target_reached"],
                    feature_names=["foo"],
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["foo"], 3.5)
                self.assertEqual(rows[0]["target_reached"], 1)

    def test_referenced_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            master = os.path.join(tmp, "master.db")
            lab = os.path.join(tmp, "lab.db")
            with MasterStore(master) as ms:
                ms.begin_day(
                    "2026-01-02",
                    ["trading_day", "timestamp", "token", "foo"],
                )
                ms.insert_rows(
                    [
                        {
                            "trading_day": "2026-01-02",
                            "timestamp": 1.0,
                            "token": "t1",
                            "foo": 42.0,
                        }
                    ]
                )
                ms.commit_day("2026-01-02")
                rid = int(
                    ms.conn.execute(
                        "SELECT master_row_id FROM samples"
                    ).fetchone()[0]
                )

            with ModelLabStore(lab) as store:
                store.ensure_prediction_schema()
                store.write_prediction_summary(
                    lab_uuid="u1",
                    status="ready",
                    row_count=1,
                    trading_days=1,
                    feature_columns_json='{"foo": "foo"}',
                    feature_storage_mode=FEATURE_STORAGE_REFERENCED,
                    master_dataset_id="master",
                    master_db_path=master,
                )
                store.insert_prediction_rows(
                    [
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "p1",
                            "trading_day": "2026-01-02",
                            "timestamp": 1.0,
                            "token": "t1",
                            "target_reached": 1,
                            "master_row_id": rid,
                        }
                    ]
                )
                # No sf_* columns
                cols = store._prediction_table_columns()
                self.assertNotIn("sf_foo", cols)
                self.assertIn("master_row_id", cols)

                access = PredictionFeatureStore.from_store(store, data_dir=tmp)
                self.assertTrue(access.is_referenced())
                rows = access.fetch_rows(
                    outcome_cols=["target_reached"],
                    feature_names=["foo"],
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]["foo"]), 42.0)

    def test_detect_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            master = os.path.join(tmp, "m.db")
            with MasterStore(master) as ms:
                ms.ensure_master_row_id()
            mode, path = detect_feature_storage_mode(
                parquet_columns=["ltp", "foo"],
                master_db_path=master,
                data_dir=tmp,
            )
            self.assertEqual(mode, FEATURE_STORAGE_REFERENCED)
            self.assertEqual(path, os.path.abspath(master))

            mode2, path2 = detect_feature_storage_mode(
                parquet_columns=["ltp"],
                master_db_path=None,
                data_dir=tmp,
            )
            self.assertEqual(mode2, FEATURE_STORAGE_EMBEDDED)
            self.assertIsNone(path2)

    def test_referenced_map_identity(self) -> None:
        self.assertEqual(
            referenced_feature_column_map(["a", "b"]),
            {"a": "a", "b": "b"},
        )

    def test_schema_version_bumped(self) -> None:
        self.assertGreaterEqual(LAB_SCHEMA_VERSION_PREDICTION, 8)


if __name__ == "__main__":
    unittest.main()
