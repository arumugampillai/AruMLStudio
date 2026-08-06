"""Feature Transformation → Auto: dataset metadata must record concrete trading
dates for both All days and Selected days, mirroring the Master Dataset panel
(``trading_day_filter.selected_dates`` / ``exported_dates``) instead of a vague
"all days" flag.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.dataset_builder.analysis_dataset_export import create_analysis_dataset
from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter


def _make_master_db(db_path: str, days: list[str]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE samples ("
            "trading_day TEXT, timestamp REAL, token TEXT, market TEXT, "
            "expiry TEXT, ltp REAL, spot REAL)"
        )
        rows = []
        for day in days:
            for i in range(3):
                rows.append((day, float(i * 3), "A", "NIFTY", "2099-12-31", 100.0 + i, 200.0 + i))
        conn.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


class TestAnalysisDatasetDayScopeMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.days = ["2026-06-30", "2026-07-07", "2026-07-14", "2026-07-21"]

    def test_all_days_writes_explicit_trading_day_filter_dates(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            db_path = os.path.join(data_dir, "master.db")
            _make_master_db(db_path, self.days)

            all_days, explicit_days, filter_meta = resolve_day_scope_filter(
                scope="all",
                selected_days=set(),
                master_days=self.days,
            )
            self.assertTrue(all_days)

            result = create_analysis_dataset(
                data_dir,
                market="NIFTY",
                interval_sec=3,
                include_registry=True,
                include_pipeline=False,
                all_days=all_days,
                selected_days=explicit_days or None,
                trading_day_filter=filter_meta,
                master_db_path=db_path,
                dataset_name="auto_all_days",
            )
            self.assertEqual(result["status"], "completed")

            with open(
                os.path.join(data_dir, "datasets", "auto_all_days.json"),
                encoding="utf-8",
            ) as fh:
                meta = json.load(fh)

            # Metadata must list the actual trading dates, not just an "all" flag.
            self.assertTrue(meta["master_filter"]["all_days"])
            tdf = meta["trading_day_filter"]
            self.assertEqual(tdf["mode"], "all")
            self.assertEqual(tdf["selected_dates"], sorted(self.days))
            self.assertEqual(tdf["exported_dates"], sorted(self.days))
            self.assertEqual(
                sorted(d["trading_day"] for d in meta["days"]),
                sorted(self.days),
            )

    def test_selected_days_writes_only_chosen_dates(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            db_path = os.path.join(data_dir, "master.db")
            _make_master_db(db_path, self.days)

            chosen = {"2026-06-30", "2026-07-21"}
            all_days, explicit_days, filter_meta = resolve_day_scope_filter(
                scope="selected",
                selected_days=chosen,
                master_days=self.days,
            )
            self.assertFalse(all_days)
            self.assertEqual(explicit_days, sorted(chosen))

            result = create_analysis_dataset(
                data_dir,
                market="NIFTY",
                interval_sec=3,
                include_registry=True,
                include_pipeline=False,
                all_days=all_days,
                selected_days=explicit_days,
                trading_day_filter=filter_meta,
                master_db_path=db_path,
                dataset_name="auto_selected_days",
            )
            self.assertEqual(result["status"], "completed")

            with open(
                os.path.join(data_dir, "datasets", "auto_selected_days.json"),
                encoding="utf-8",
            ) as fh:
                meta = json.load(fh)

            self.assertFalse(meta["master_filter"]["all_days"])
            self.assertEqual(sorted(meta["master_filter"]["selected_days"]), sorted(chosen))
            tdf = meta["trading_day_filter"]
            self.assertEqual(tdf["selected_dates"], sorted(chosen))
            self.assertEqual(tdf["exported_dates"], sorted(chosen))
            self.assertEqual(
                sorted(d["trading_day"] for d in meta["days"]),
                sorted(chosen),
            )
            # Untouched day must not leak into the export.
            self.assertNotIn("2026-07-07", [d["trading_day"] for d in meta["days"]])


if __name__ == "__main__":
    unittest.main()
