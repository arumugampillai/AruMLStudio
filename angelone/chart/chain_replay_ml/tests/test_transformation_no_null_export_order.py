"""Regression tests for Feature Transformations → No-Null → final export."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.master_registry_export import (
    create_master_registry_dataset,
)


class TransformationNoNullExportOrderTests(unittest.TestCase):
    def test_lag_runs_before_no_null_filter_and_final_write(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            db_path = os.path.join(data_dir, "master.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE samples ("
                    "trading_day TEXT, timestamp REAL, token TEXT, market TEXT, "
                    "expiry TEXT, ltp REAL, spot REAL)"
                )
                conn.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "2026-07-01",
                            float(i * 3),
                            "A",
                            "NIFTY",
                            "2026-07-30",
                            100.0 + i,
                            200.0 + i,
                        )
                        for i in range(5)
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            progress: list[str] = []
            result = create_master_registry_dataset(
                data_dir,
                market="NIFTY",
                interval_sec=3,
                all_days=True,
                master_db_path=db_path,
                dataset_name="pipeline_order",
                no_null_data=True,
                transformation_config={
                    "transformation_pipeline_version": 1,
                    "transformations": [{
                        "id": "lag",
                        "enabled": True,
                        "params": {
                            "features": ["ltp"],
                            "lag_seconds": [3],
                            "partition_by": ["trading_day", "token"],
                        },
                    }],
                },
                on_progress=lambda message, *_: progress.append(str(message)),
            )

            frame = pd.read_parquet(result["parquet_path"])
            # Lag creates one leading NULL; post-transform No-Null removes that row.
            self.assertEqual(len(frame), 4)
            self.assertEqual(frame["ltp_lag_3s"].tolist(), [100.0, 101.0, 102.0, 103.0])
            self.assertEqual(int(frame.isna().sum().sum()), 0)

            transform_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("Feature Transformations:")
            )
            no_null_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("No-Null filter: running after")
            )
            final_export_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("Export Dataset: writing final")
            )
            self.assertLess(transform_i, no_null_i)
            self.assertLess(no_null_i, final_export_i)

            with open(result["json_path"], "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            self.assertEqual(metadata["row_count"], 4)
            self.assertEqual(metadata["column_count"], len(frame.columns))
            self.assertEqual(metadata["no_null_report"]["stage"], "post_transformation")
            self.assertEqual(metadata["no_null_report"]["remaining_null_cells"], 0)
            # Ghost columns dropped by No-Null must not remain in feature_columns.
            dropped = set(metadata.get("no_null_dropped_columns") or [])
            feat_cols = set(metadata.get("feature_columns") or [])
            self.assertTrue(dropped.isdisjoint(feat_cols))
            self.assertEqual(metadata["feature_count"], len(feat_cols))
            lag_params = metadata["transformations"][0]["params"]
            self.assertEqual(lag_params["lag_seconds"], [3])
            self.assertEqual(lag_params["sample_interval_sec"], 3)
            # Experiment identity — top-level sample grid for row-span features.
            self.assertEqual(metadata["sample_interval_sec"], 3)
            self.assertEqual(metadata["sampling"]["interval_sec"], 3)

    def test_analysis_premium_runs_after_no_null(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            db_path = os.path.join(data_dir, "master.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE samples ("
                    "trading_day TEXT, timestamp REAL, token TEXT, market TEXT, "
                    "expiry TEXT, ltp REAL, spot REAL)"
                )
                conn.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("2026-07-01", float(i * 3), "A", "NIFTY", "2026-07-30", ltp, 200.0)
                        for i, ltp in enumerate([10.0, 20.0, 30.0, 50.0, 25.0])
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            progress: list[str] = []
            result = create_master_registry_dataset(
                data_dir,
                market="NIFTY",
                interval_sec=3,
                all_days=True,
                master_db_path=db_path,
                dataset_name="analysis_premium_order",
                dataset_kind="analysis",
                keep_pipeline_owned=True,
                no_null_data=True,
                premium_enabled=True,
                premium_min=15,
                premium_max=40,
                transformation_config={
                    "transformation_pipeline_version": 1,
                    "transformations": [{
                        "id": "lag",
                        "enabled": True,
                        "params": {
                            "features": ["ltp"],
                            "lag_seconds": [3],
                            "partition_by": ["trading_day", "token"],
                        },
                    }],
                },
                on_progress=lambda message, *_: progress.append(str(message)),
            )

            frame = pd.read_parquet(result["parquet_path"])
            # Lag drops first row via post-transform No-Null; premium then keeps LTP 15–40.
            # Remaining after lag+no-null: 20,30,50,25 → premium: 20,30,25
            self.assertEqual(sorted(frame["ltp"].tolist()), [20.0, 25.0, 30.0])

            transform_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("Feature Transformations:")
            )
            no_null_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("No-Null filter: running after")
            )
            premium_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("Premium filter:")
            )
            final_export_i = next(
                i for i, msg in enumerate(progress)
                if msg.startswith("Export Dataset: writing final")
            )
            # No-Null must not run before transforms.
            early_registry_nn = [
                i for i, msg in enumerate(progress)
                if "Registry Features" in msg and "before Pipeline" in msg
            ]
            self.assertEqual(early_registry_nn, [])
            self.assertLess(transform_i, no_null_i)
            self.assertLess(no_null_i, premium_i)
            self.assertLess(premium_i, final_export_i)

            with open(result["json_path"], "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            self.assertTrue(metadata.get("premium_filter_deferred"))
            self.assertEqual(metadata["premium_report"]["rows_after"], 3)
            self.assertEqual(metadata["row_count"], 3)
            self.assertEqual(metadata["no_null_report"]["stage"], "post_transformation")
            self.assertEqual(metadata["feature_count"], len(metadata.get("feature_columns") or []))
            for col in metadata.get("feature_columns") or []:
                self.assertIn(col, frame.columns)


if __name__ == "__main__":
    unittest.main()
