"""Warm-up Simulator trading-day catalog (ticks + master completed days)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.feature_policy.warmup_simulator import list_trading_days


class ListTradingDaysTests(unittest.TestCase):
    def test_includes_tick_search_dirs_and_master_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chart = os.path.join(tmp, "chart")
            tick_dir = os.path.join(tmp, "ticks")
            master_dir = os.path.join(tmp, "masters")
            os.makedirs(os.path.join(chart, "data"))
            os.makedirs(tick_dir)
            os.makedirs(master_dir)

            # Tick only in external tick dir (not under chart/data) — old bug missed these.
            tick_path = os.path.join(tick_dir, "angel_market_2026-07-01.db")
            with open(tick_path, "wb") as fh:
                fh.write(b"x")

            # Master 3s DB with completed days (including one without tick file).
            master_path = os.path.join(master_dir, "master_dataset_nifty_3s.db")
            conn = sqlite3.connect(master_path)
            conn.executescript(
                """
                CREATE TABLE master_dataset_days (
                    trading_day TEXT PRIMARY KEY,
                    row_count INTEGER,
                    token_count INTEGER,
                    expiry_count INTEGER,
                    first_timestamp REAL,
                    last_timestamp REAL,
                    coverage_percent REAL,
                    rejected_rows INTEGER,
                    status TEXT,
                    last_updated TEXT,
                    dominant_expiry TEXT,
                    is_expiry_day INTEGER
                );
                INSERT INTO master_dataset_days
                    (trading_day, row_count, status)
                VALUES
                    ('2026-07-01', 1000, 'ok'),
                    ('2026-07-02', 2000, 'ok'),
                    ('2026-07-03', 0, 'empty');
                """
            )
            conn.commit()
            conn.close()

            with mock.patch(
                "tick_data_paths.tick_search_dirs", return_value=[tick_dir]
            ), mock.patch(
                "chain_replay_ml.dataset_builder.master_naming.resolve_master_datasets_dir",
                return_value=master_dir,
            ), mock.patch(
                "chain_replay_ml.dataset_builder.master_naming.resolve_master_db_path",
                return_value=master_path,
            ):
                days = list_trading_days(
                    chart,
                    sampling_interval_sec=3,
                    market="NIFTY",
                )

            self.assertEqual(days, ["2026-07-02", "2026-07-01"])
            self.assertNotIn("2026-07-03", days)


if __name__ == "__main__":
    unittest.main()
