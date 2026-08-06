"""Unit tests for diagnostic No-Null filter report (read-only)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.dataset_builder.no_null_filter_report import (
    build_no_null_filter_report_text,
)


class NoNullFilterReportTests(unittest.TestCase):
    def test_report_is_read_only_and_includes_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "master.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE samples ("
                    "trading_day TEXT, timestamp REAL, token TEXT, "
                    "market TEXT, expiry TEXT, ltp REAL, spot REAL, "
                    "atm_distance INTEGER)"
                )
                # 5 rows; one incomplete (spot NULL) after filters
                conn.executemany(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)",
                    [
                        ("2026-07-01", float(i * 3), "A", "NIFTY", "2026-07-30", 20.0 + i, spot, 0)
                        for i, spot in enumerate([100.0, 101.0, None, 103.0, 104.0])
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            before_mtime = os.path.getmtime(db_path)
            before_size = os.path.getsize(db_path)
            text = build_no_null_filter_report_text(
                db_path=db_path,
                trading_day="2026-07-01",
                atm_band_filter=10,
                premium_min=15,
                premium_max=40,
            )
            after_mtime = os.path.getmtime(db_path)
            after_size = os.path.getsize(db_path)

            self.assertEqual(before_mtime, after_mtime)
            self.assertEqual(before_size, after_size)
            self.assertIn("No-Null Data Filter Report", text)
            self.assertIn("Master dataset", text)
            self.assertIn("After ATM filter", text)
            self.assertIn("After LTP / Delta", text)
            self.assertIn("Final surviving row count:", text)
            self.assertIn("diagnostics only", text.lower())

    def test_empty_after_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "master.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE samples ("
                    "trading_day TEXT, timestamp REAL, token TEXT, "
                    "ltp REAL, spot REAL)"
                )
                conn.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?)",
                    ("2026-07-01", 0.0, "A", 5.0, 100.0),  # LTP outside 15–40
                )
                conn.commit()
            finally:
                conn.close()
            text = build_no_null_filter_report_text(
                db_path=db_path,
                trading_day="2026-07-01",
                premium_min=15,
                premium_max=40,
            )
            self.assertIn("Final surviving rows: 0", text)

    def test_ema_gap_reset_per_gap_log_and_buckets(self) -> None:
        from chain_replay_ml.dataset_builder.no_null_filter_report import (
            _format_ema_gap_reset_check,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "master.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE samples ("
                    "trading_day TEXT, timestamp REAL, token TEXT, "
                    "ltp_ema300 REAL, strike REAL, spot REAL, "
                    "strike_distance_from_atm REAL)"
                )
                # Token A: continuous. Token B: leaves ATM band for 90s then returns.
                # strike step 50; ATM 23600 → B at 23100 is dist -10; during gap spot→23650
                # mid ATM 23650 → dist 11 → Left ATM ±10.
                base = 1_700_000_000.0
                rows = []
                for i in range(5):
                    ts = base + i * 3.0
                    rows.append(
                        (
                            "2026-07-01",
                            ts,
                            "A",
                            1.0 if i >= 1 else None,
                            23600.0,
                            23600.0,
                            0.0,
                        )
                    )
                # B before gap (in band)
                rows.append(
                    ("2026-07-01", base, "B", None, 23100.0, 23600.0, -10.0)
                )
                rows.append(
                    ("2026-07-01", base + 3.0, "B", None, 23100.0, 23600.0, -10.0)
                )
                # peers continue during B's absence; spot drifts so B is OTM of band
                for i in range(2, 32):
                    ts = base + i * 3.0
                    rows.append(
                        (
                            "2026-07-01",
                            ts,
                            "A",
                            1.0,
                            23600.0,
                            23650.0,
                            -1.0,
                        )
                    )
                # B returns after 90s gap (30*3)
                rows.append(
                    (
                        "2026-07-01",
                        base + 93.0,
                        "B",
                        None,
                        23100.0,
                        23600.0,
                        -10.0,
                    )
                )
                for j in range(1, 5):
                    rows.append(
                        (
                            "2026-07-01",
                            base + 93.0 + j * 3.0,
                            "B",
                            None if j < 3 else 1.0,
                            23100.0,
                            23600.0,
                            -10.0,
                        )
                    )
                conn.executemany(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                lines = _format_ema_gap_reset_check(
                    conn,
                    where_sql="trading_day = ?",
                    params=["2026-07-01"],
                    feature="ltp_ema300",
                    ema_period=300,
                    gap_max_sec=20.0,
                    trading_day="2026-07-01",
                    atm_band=10,
                )
            finally:
                conn.close()

            text = "\n".join(lines)
            self.assertIn("Gap duration summary", text)
            self.assertIn("Per-gap reset log", text)
            self.assertIn("Why token disappeared", text)
            self.assertIn("Left ATM +/-10 band", text)
            self.assertIn("61-120 sec", text)
            self.assertIn("Yes", text)


if __name__ == "__main__":
    unittest.main()
