"""As-of alignment + diagnostics for historic NIFTY multi-TF EMA context."""

from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from chain_replay_ml.dataset_builder.historic_spot_ema_context import (
    HISTORIC_SPOT_EMA_FEATURES,
    HistoricSpotEmaBook,
    _TfEmaSeries,
    build_historic_spot_ema_book,
    compute_ema,
    enrich_historic_spot_ema_features,
    historic_spot_ema_feature_name,
)

IST = ZoneInfo("Asia/Kolkata")


def _ist_ts(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> float:
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST).timestamp()


def _session_buckets(day: tuple[int, int, int], interval_sec: int) -> list[float]:
    """NSE-like session opens: 09:15 .. last open <= 15:29."""
    start = _ist_ts(*day, 9, 15)
    end = _ist_ts(*day, 15, 29)
    out: list[float] = []
    t = start
    while t <= end + 1e-9:
        out.append(t)
        t += float(interval_sec)
    return out


def _synthetic_book() -> HistoricSpotEmaBook:
    """Session bars for prior + current day (matches historic DB continuity)."""
    periods = (9, 20, 50, 100, 200)
    series = []
    for label, interval in (("1m", 60), ("3m", 180), ("5m", 300), ("15m", 900)):
        ts = _session_buckets((2026, 7, 23), interval) + _session_buckets(
            (2026, 7, 24), interval
        )
        closes = [100.0 + i * 0.05 for i in range(len(ts))]
        series.append(
            _TfEmaSeries(
                label=label,
                interval_sec=interval,
                timestamps=tuple(ts),
                emas={p: tuple(compute_ema(closes, p)) for p in periods},
            )
        )
    return HistoricSpotEmaBook(series=tuple(series), ema_periods=periods)


class TestHistoricSpotEmaAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.book = _synthetic_book()
        self.day = (2026, 7, 24)

    def test_feature_count(self) -> None:
        self.assertEqual(len(HISTORIC_SPOT_EMA_FEATURES), 20)
        self.assertEqual(len(self.book.feature_names), 20)

    def test_0915_matches_0915_candle(self) -> None:
        tick = _ist_ts(*self.day, 9, 15, 0)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["3m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["5m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["15m"]["matched_ist"], "2026-07-24 09:15:00")

    def test_0915_45_still_matches_0915(self) -> None:
        tick = _ist_ts(*self.day, 9, 15, 45)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 09:15:00")
        # 3m/5m/15m opens are still 09:15
        self.assertEqual(diag["timeframes"]["3m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["5m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["15m"]["matched_ist"], "2026-07-24 09:15:00")

    def test_0916_02_matches_0916_1m(self) -> None:
        tick = _ist_ts(*self.day, 9, 16, 2)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 09:16:00")
        # Larger TFs still on 09:15 open
        self.assertEqual(diag["timeframes"]["3m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["5m"]["matched_ist"], "2026-07-24 09:15:00")
        self.assertEqual(diag["timeframes"]["15m"]["matched_ist"], "2026-07-24 09:15:00")

    def test_late_start_1102_matches_correct_candles(self) -> None:
        """Day that begins late (power failure) still as-of joins historic bars."""
        tick = _ist_ts(*self.day, 11, 2, 0)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 11:02:00")
        self.assertEqual(diag["timeframes"]["3m"]["matched_ist"], "2026-07-24 11:00:00")
        self.assertEqual(diag["timeframes"]["5m"]["matched_ist"], "2026-07-24 11:00:00")
        self.assertEqual(diag["timeframes"]["15m"]["matched_ist"], "2026-07-24 11:00:00")

    def test_1107_18_example_alignment(self) -> None:
        tick = _ist_ts(*self.day, 11, 7, 18)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 11:07:00")
        self.assertEqual(diag["timeframes"]["3m"]["matched_ist"], "2026-07-24 11:06:00")
        self.assertEqual(diag["timeframes"]["5m"]["matched_ist"], "2026-07-24 11:05:00")
        self.assertEqual(diag["timeframes"]["15m"]["matched_ist"], "2026-07-24 11:00:00")

    def test_never_uses_future_candle(self) -> None:
        tick = _ist_ts(*self.day, 11, 7, 18)
        levels = self.book.levels_at(tick)
        for series in self.book.series:
            idx = series.asof_index(tick)
            self.assertIsNotNone(idx)
            assert idx is not None
            self.assertLessEqual(series.timestamps[idx], tick + 1e-9)
            # Next candle (if any) must be strictly after tick
            if idx + 1 < len(series.timestamps):
                self.assertGreater(series.timestamps[idx + 1], tick)

        # Sanity: EMA values present after warmup
        self.assertIsNotNone(levels["spot_1m_ema9"])
        self.assertTrue(math.isfinite(float(levels["spot_1m_ema9"])))

    def test_missing_before_first_bar_is_null(self) -> None:
        tick = _ist_ts(2026, 7, 23, 9, 14, 59)
        levels = self.book.levels_at(tick)
        for name in self.book.feature_names:
            self.assertIsNone(levels[name])

    def test_enrich_uses_book_on_ctx(self) -> None:
        class _Ctx:
            historic_spot_ema_book = None

        ctx = _Ctx()
        ctx.historic_spot_ema_book = self.book
        tick = _ist_ts(*self.day, 11, 7, 18)
        raw = enrich_historic_spot_ema_features({}, ts=tick, ctx=ctx)
        self.assertIn("spot_1m_ema9", raw)
        self.assertIsNotNone(raw["spot_1m_ema9"])
        self.assertIsNotNone(raw["spot_15m_ema9"])

    def test_diagnostics_table_shape(self) -> None:
        tick = _ist_ts(*self.day, 11, 7, 18)
        diag = self.book.match_diagnostics(tick)
        self.assertEqual(diag["tick_ist"], "2026-07-24 11:07:18")
        for tf in ("1m", "3m", "5m", "15m"):
            self.assertIn("matched_bucket_start", diag["timeframes"][tf])
            self.assertIn("matched_ist", diag["timeframes"][tf])
        self.assertEqual(len(diag["ema_values"]), 20)


class TestHistoricSpotEmaFromSqlite(unittest.TestCase):
    def test_build_from_temp_db_preserves_close_path(self) -> None:
        day = "2026-07-24"
        token = "99926000"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "angel_historic_bars.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE angel_historic_bars (
                    token TEXT NOT NULL,
                    interval_sec INTEGER NOT NULL,
                    bucket_start REAL NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (token, interval_sec, bucket_start)
                )
                """
            )
            # Enough 1m bars for EMA9 seed + a few after
            rows = []
            t0 = _ist_ts(2026, 7, 24, 9, 15)
            for i in range(20):
                rows.append((token, 60, t0 + i * 60, 100 + i, 100 + i, 100 + i, 100.0 + i, 0))
            # Prior-day bars for warmup continuity
            t_prev = _ist_ts(2026, 7, 23, 9, 15)
            for i in range(30):
                rows.append((token, 60, t_prev + i * 60, 90 + i, 90 + i, 90 + i, 90.0 + i, 0))
            # Minimal stubs for other intervals
            for interval, step in ((180, 180), (300, 300), (900, 900)):
                for i in range(15):
                    rows.append(
                        (
                            token,
                            interval,
                            t0 + i * step,
                            100,
                            100,
                            100,
                            100.0 + i,
                            0,
                        )
                    )
            conn.executemany(
                "INSERT INTO angel_historic_bars VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            conn.close()

            book = build_historic_spot_ema_book(
                trading_day=day,
                db_path=str(db),
                timeframes=("1m",),
                ema_periods=(9,),
            )
            tick = _ist_ts(2026, 7, 24, 9, 15, 45)
            diag = book.match_diagnostics(tick)
            self.assertEqual(diag["timeframes"]["1m"]["matched_ist"], "2026-07-24 09:15:00")
            name = historic_spot_ema_feature_name("1m", 9)
            self.assertIsNotNone(diag["ema_values"][name])


if __name__ == "__main__":
    unittest.main()
