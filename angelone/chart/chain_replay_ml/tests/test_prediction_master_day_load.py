"""Tests for Master-only Unseen prediction day loading."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_feature_store import (
    count_trading_day_rows_in_master,
    load_trading_day_frame_from_master,
)
from chain_replay_ml.model_lab.prediction_schema import DAY_COMPLETED, DAY_WAITING
from chain_replay_ml.model_lab.store import ModelLabStore


def _make_master(path: str, *, day: str, n: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE samples (
            trading_day TEXT,
            timestamp REAL,
            token TEXT,
            strike REAL,
            option_type TEXT,
            market TEXT,
            expiry TEXT,
            spot REAL,
            ltp REAL,
            f1 REAL,
            future_ltp_5m REAL,
            master_row_id INTEGER
        )
        """
    )
    for i in range(n):
        conn.execute(
            """
            INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                day,
                1000.0 + i,
                f"T{i}",
                25000.0,
                "CE",
                "NIFTY",
                "2026-07-16",
                25000.0,
                100.0 + i,
                float(i),
                110.0 + i,
                i + 1,
            ),
        )
    conn.commit()
    conn.close()


class MasterDayLoadTests(unittest.TestCase):
    def test_load_and_count_master_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            master = os.path.join(tmp, "master.db")
            _make_master(master, day="2026-07-14", n=5)
            self.assertEqual(count_trading_day_rows_in_master(master, "2026-07-14"), 5)
            self.assertEqual(count_trading_day_rows_in_master(master, "2026-07-13"), 0)
            df = load_trading_day_frame_from_master(
                master,
                "2026-07-14",
                ["timestamp", "token", "ltp", "f1", "future_ltp_5m", "missing_col"],
            )
            self.assertEqual(len(df), 5)
            self.assertIn("master_row_id", df.columns)
            self.assertNotIn("missing_col", df.columns)
            self.assertIn("f1", df.columns)

    def test_premium_filter_from_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            master = os.path.join(tmp, "master.db")
            conn = sqlite3.connect(master)
            conn.execute(
                """
                CREATE TABLE samples (
                    trading_day TEXT, timestamp REAL, token TEXT, strike REAL,
                    option_type TEXT, market TEXT, expiry TEXT, spot REAL, ltp REAL,
                    abs_delta REAL, f1 REAL, future_ltp_5m REAL, master_row_id INTEGER
                )
                """
            )
            # ATM distance encoded via abs(strike-spot)/50 ~= band index loosely;
            # build_selection_sql uses atm_distance or similar — inspect and match.
            rows = [
                # in premium 15-500
                ("2026-07-14", 1.0, "T1", 25000.0, "CE", "NIFTY", "2026-07-16", 25000.0, 50.0, 0.5, 1.0, 60.0, 1),
                ("2026-07-14", 2.0, "T2", 25000.0, "CE", "NIFTY", "2026-07-16", 25000.0, 200.0, 0.5, 1.0, 210.0, 2),
                # below premium
                ("2026-07-14", 3.0, "T3", 25000.0, "CE", "NIFTY", "2026-07-16", 25000.0, 5.0, 0.5, 1.0, 6.0, 3),
                # above premium
                ("2026-07-14", 4.0, "T4", 25000.0, "CE", "NIFTY", "2026-07-16", 25000.0, 800.0, 0.5, 1.0, 810.0, 4),
            ]
            conn.executemany(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            conn.close()
            filt = {
                "premium_enabled": True,
                "premium_min": 15.0,
                "premium_max": 500.0,
                "atm_band_filter": None,
                "delta_enabled": False,
                "no_null_data": False,
            }
            n = count_trading_day_rows_in_master(
                master, "2026-07-14", master_filter=filt
            )
            self.assertEqual(n, 2)
            df = load_trading_day_frame_from_master(
                master,
                "2026-07-14",
                ["timestamp", "token", "ltp", "future_ltp_5m"],
                master_filter=filt,
            )
            self.assertEqual(len(df), 2)
            self.assertTrue(all(15.0 <= float(x) <= 500.0 for x in df["ltp"]))


class PendingZeroCompleteTests(unittest.TestCase):
    def test_completed_zero_with_expected_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab = "lab-1"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab, ["2026-07-14"])
                store.set_day_rows_expected(lab, "2026-07-14", 309838)
                store.set_build_day_status(
                    lab,
                    "2026-07-14",
                    status=DAY_COMPLETED,
                    row_count=0,
                    finished=True,
                    progress_pct=100.0,
                )
                store.set_days_selected(lab, ["2026-07-14"])
                pending = store.pending_build_days(lab, selected_only=True)
                self.assertEqual(pending, ["2026-07-14"])

                store.set_build_day_status(
                    lab,
                    "2026-07-14",
                    status=DAY_WAITING,
                    row_count=0,
                    progress_pct=0.0,
                )
                pending2 = store.pending_build_days(lab, selected_only=True)
                self.assertEqual(pending2, ["2026-07-14"])


if __name__ == "__main__":
    unittest.main()
