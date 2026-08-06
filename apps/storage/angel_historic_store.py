"""Persistent Angel historic OHLC store with coverage / availability checks."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_DB_NAME = "angel_historic_bars.db"

# User-requested intervals (seconds) → Angel SmartAPI interval name
HISTORIC_INTERVALS_SEC: dict[int, str] = {
    60: "ONE_MINUTE",
    180: "THREE_MINUTE",
    300: "FIVE_MINUTE",
    900: "FIFTEEN_MINUTE",
    1800: "THIRTY_MINUTE",
    3600: "ONE_HOUR",
    86400: "ONE_DAY",
}

INTERVAL_LABEL: dict[int, str] = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "60m",
    86400: "1d",
}

# Session bars per trading day (09:15–15:30 IST)
BARS_PER_TRADING_DAY: dict[int, int] = {
    60: 375,
    180: 125,
    300: 75,
    900: 25,
    1800: 13,
    3600: 6,
    86400: 1,
}

_DDL = """
CREATE TABLE IF NOT EXISTS angel_historic_bars (
    token          TEXT    NOT NULL,
    exchange       TEXT    NOT NULL,
    interval_sec   INTEGER NOT NULL,
    bucket_start   REAL    NOT NULL,
    open           REAL    NOT NULL,
    high           REAL    NOT NULL,
    low            REAL    NOT NULL,
    close          REAL    NOT NULL,
    volume         INTEGER NOT NULL DEFAULT 0,
    fetched_at     REAL,
    PRIMARY KEY (token, interval_sec, bucket_start)
);
CREATE INDEX IF NOT EXISTS idx_ahb_token_interval_ts
    ON angel_historic_bars(token, interval_sec, bucket_start);

CREATE TABLE IF NOT EXISTS angel_historic_sync (
    token          TEXT    NOT NULL,
    exchange       TEXT    NOT NULL,
    interval_sec   INTEGER NOT NULL,
    oldest_ts      REAL,
    newest_ts      REAL,
    bar_count      INTEGER NOT NULL DEFAULT 0,
    last_fetch_at  REAL,
    last_error     TEXT,
    PRIMARY KEY (token, interval_sec)
);
"""

_UPSERT = """
INSERT INTO angel_historic_bars (
    token, exchange, interval_sec, bucket_start,
    open, high, low, close, volume, fetched_at
) VALUES (
    :token, :exchange, :interval_sec, :bucket_start,
    :open, :high, :low, :close, :volume, :fetched_at
)
ON CONFLICT(token, interval_sec, bucket_start) DO UPDATE SET
    open=excluded.open,
    high=excluded.high,
    low=excluded.low,
    close=excluded.close,
    volume=excluded.volume,
    fetched_at=excluded.fetched_at
"""


def default_db_path(chart_dir: str | None = None) -> str:
    base = chart_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", DEFAULT_DB_NAME)


def parse_interval_label(text: str) -> int | None:
    raw = str(text or "").strip().lower()
    aliases = {
        "1m": 60, "1min": 60, "one_minute": 60,
        "3m": 180, "3min": 180, "three_minute": 180,
        "5m": 300, "5min": 300, "five_minute": 300,
        "15m": 900, "15min": 900, "fifteen_minute": 900,
        "30m": 1800, "30min": 1800, "thirty_minute": 1800,
        "60m": 3600, "1h": 3600, "60min": 3600, "one_hour": 3600,
        "1d": 86400, "1day": 86400, "daily": 86400, "one_day": 86400,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        sec = int(raw)
        return sec if sec in HISTORIC_INTERVALS_SEC else None
    except ValueError:
        return None


def parse_interval_list(text: str | None) -> list[int]:
    if not text or str(text).strip().lower() in ("all", "*"):
        return list(HISTORIC_INTERVALS_SEC)
    out: list[int] = []
    for part in str(text).split(","):
        sec = parse_interval_label(part.strip())
        if sec is not None and sec not in out:
            out.append(sec)
    return out


def count_weekdays(from_day: date, to_day: date) -> int:
    if to_day < from_day:
        return 0
    n = 0
    d = from_day
    while d <= to_day:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def historic_window(months: int = 6, *, end: datetime | None = None) -> tuple[float, float, date, date]:
    end_dt = end or datetime.now(tz=IST)
    start_dt = end_dt - timedelta(days=int(months) * 30)
    from_day = start_dt.date()
    to_day = end_dt.date()
    open_dt = datetime(from_day.year, from_day.month, from_day.day, 9, 15, tzinfo=IST)
    return open_dt.timestamp(), end_dt.timestamp(), from_day, to_day


def expected_bar_count(interval_sec: int, from_day: date, to_day: date) -> int:
    per_day = BARS_PER_TRADING_DAY.get(int(interval_sec), 0)
    return count_weekdays(from_day, to_day) * per_day


class AngelHistoricStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or default_db_path()
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()

    def upsert_bars(
        self,
        token: str,
        exchange: str,
        interval_sec: int,
        bars: list[dict[str, Any]],
    ) -> int:
        if not bars:
            return 0
        now = time.time()
        rows = []
        for b in bars:
            rows.append({
                "token": str(token),
                "exchange": str(exchange),
                "interval_sec": int(interval_sec),
                "bucket_start": float(b["bucket_start"]),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": int(b.get("volume") or 0),
                "fetched_at": now,
            })
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(_UPSERT, rows)
                conn.commit()
            finally:
                conn.close()
        self._refresh_sync_meta(token, exchange, interval_sec)
        return len(rows)

    def _refresh_sync_meta(self, token: str, exchange: str, interval_sec: int) -> None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt,
                           MIN(bucket_start) AS oldest,
                           MAX(bucket_start) AS newest
                    FROM angel_historic_bars
                    WHERE token=? AND interval_sec=?
                    """,
                    (str(token), int(interval_sec)),
                ).fetchone()
                conn.execute(
                    """
                    INSERT INTO angel_historic_sync (
                        token, exchange, interval_sec,
                        oldest_ts, newest_ts, bar_count, last_fetch_at, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(token, interval_sec) DO UPDATE SET
                        exchange=excluded.exchange,
                        oldest_ts=excluded.oldest_ts,
                        newest_ts=excluded.newest_ts,
                        bar_count=excluded.bar_count,
                        last_fetch_at=excluded.last_fetch_at,
                        last_error=NULL
                    """,
                    (
                        str(token),
                        str(exchange),
                        int(interval_sec),
                        row["oldest"],
                        row["newest"],
                        int(row["cnt"] or 0),
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def set_sync_error(self, token: str, interval_sec: int, error: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE angel_historic_sync
                    SET last_error=?, last_fetch_at=?
                    WHERE token=? AND interval_sec=?
                    """,
                    (str(error)[:500], time.time(), str(token), int(interval_sec)),
                )
                conn.commit()
            finally:
                conn.close()

    def coverage_in_range(
        self,
        token: str,
        interval_sec: int,
        from_ts: float,
        to_ts: float,
    ) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt,
                           MIN(bucket_start) AS oldest,
                           MAX(bucket_start) AS newest
                    FROM angel_historic_bars
                    WHERE token=? AND interval_sec=?
                      AND bucket_start >= ? AND bucket_start <= ?
                    """,
                    (str(token), int(interval_sec), float(from_ts), float(to_ts)),
                ).fetchone()
            finally:
                conn.close()
        cnt = int(row["cnt"] or 0)
        return {
            "bar_count": cnt,
            "oldest_ts": row["oldest"],
            "newest_ts": row["newest"],
            "oldest_time": _fmt_ts(row["oldest"]),
            "newest_time": _fmt_ts(row["newest"]),
        }

    def get_bounds(self, token: str, interval_sec: int) -> dict[str, Any]:
        """Full stored range for token/interval (all rows in DB)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt,
                           MIN(bucket_start) AS oldest,
                           MAX(bucket_start) AS newest
                    FROM angel_historic_bars
                    WHERE token=? AND interval_sec=?
                    """,
                    (str(token), int(interval_sec)),
                ).fetchone()
            finally:
                conn.close()
        cnt = int(row["cnt"] or 0)
        return {
            "bar_count": cnt,
            "oldest_ts": row["oldest"],
            "newest_ts": row["newest"],
            "oldest_time": _fmt_ts(row["oldest"]),
            "newest_time": _fmt_ts(row["newest"]),
        }

    def availability(
        self,
        token: str,
        *,
        months: int = 6,
        intervals: list[int] | None = None,
    ) -> dict[str, Any]:
        intervals = intervals or list(HISTORIC_INTERVALS_SEC)
        from_ts, to_ts, from_day, to_day = historic_window(months)
        db_exists = os.path.isfile(self.db_path)
        db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if db_exists else 0.0

        per_interval: dict[str, Any] = {}
        all_available = True
        any_data = False

        for interval_sec in intervals:
            label = INTERVAL_LABEL.get(interval_sec, f"{interval_sec}s")
            expected = expected_bar_count(interval_sec, from_day, to_day)
            bounds = self.get_bounds(token, interval_sec)
            cov = self.coverage_in_range(token, interval_sec, from_ts, to_ts)
            cnt_total = int(bounds["bar_count"])
            cnt_window = int(cov["bar_count"])
            pct = round(100.0 * cnt_window / expected, 1) if expected > 0 else 0.0
            has_enough = cnt_window >= int(expected * 0.85)
            if cnt_total > 0:
                any_data = True
            if not has_enough:
                all_available = False
            per_interval[label] = {
                "interval_sec": interval_sec,
                "angel_interval": HISTORIC_INTERVALS_SEC.get(interval_sec),
                "expected_bars": expected,
                "stored_bars": cnt_total,
                "stored_total": cnt_total,
                "window_6m_bars": cnt_window,
                "coverage_pct": pct,
                "window_6m_coverage_pct": pct,
                "available": has_enough,
                "oldest": bounds["oldest_time"],
                "newest": bounds["newest_time"],
                "missing_bars_est": max(0, expected - cnt_window),
                "next_fetch_mode": (
                    "extend_before" if cnt_total > 0 and has_enough else "initial"
                ),
            }

        return {
            "db_path": self.db_path,
            "db_exists": db_exists,
            "db_size_mb": db_size_mb,
            "token": str(token),
            "months": months,
            "window_from": from_day.isoformat(),
            "window_to": to_day.isoformat(),
            "trading_days_est": count_weekdays(from_day, to_day),
            "intervals": per_interval,
            "all_available": all_available,
            "any_data": any_data,
            "needs_fetch": not all_available,
        }

    def fetch_bars(
        self,
        token: str,
        interval_sec: int,
        *,
        limit: int = 1800,
        since_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return newest ``limit`` completed bars (oldest → newest)."""
        limit = max(1, int(limit))
        with self._lock:
            conn = self._connect()
            try:
                sql = """
                    SELECT bucket_start, open, high, low, close, volume
                    FROM angel_historic_bars
                    WHERE token = ? AND interval_sec = ?
                """
                params: list[Any] = [str(token), int(interval_sec)]
                if since_ts is not None:
                    sql += " AND bucket_start >= ?"
                    params.append(float(since_ts))
                sql += " ORDER BY bucket_start DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in reversed(rows):
            bucket = float(row["bucket_start"])
            out.append({
                "bucket_start": bucket,
                "time": bucket,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"] or 0),
                "is_complete": True,
                "source": "angel_historic_db",
            })
        return out

    def aggregate_60m_from_30m(self, token: str, exchange: str = "NSE") -> dict[str, Any]:
        """Build 60m OHLC bars from stored 30m bars (2×30m → 1×60m)."""
        src_sec, dst_sec = 1800, 3600
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT bucket_start, open, high, low, close, volume
                    FROM angel_historic_bars
                    WHERE token = ? AND interval_sec = ?
                    ORDER BY bucket_start
                    """,
                    (str(token), src_sec),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return {"ok": False, "error": "No 30m bars in DB — fetch 30m first", "built": 0}

        out_bars: list[dict[str, Any]] = []
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            ts_a = float(a["bucket_start"])
            ts_b = float(b["bucket_start"])
            if abs((ts_b - ts_a) - src_sec) > 1:
                continue
            dt_a = datetime.fromtimestamp(ts_a, tz=IST)
            dt_b = datetime.fromtimestamp(ts_b, tz=IST)
            if dt_a.date() != dt_b.date():
                continue
            out_bars.append({
                "bucket_start": ts_a,
                "open": float(a["open"]),
                "high": max(float(a["high"]), float(b["high"])),
                "low": min(float(a["low"]), float(b["low"])),
                "close": float(b["close"]),
                "volume": int(a["volume"] or 0) + int(b["volume"] or 0),
                "interval_sec": dst_sec,
            })

        before = self.get_bounds(token, dst_sec)
        stored = self.upsert_bars(token, exchange, dst_sec, out_bars)
        after = self.get_bounds(token, dst_sec)
        return {
            "ok": True,
            "source": "aggregated_from_30m",
            "pairs_used": len(out_bars),
            "stored": stored,
            "added_bars": int(after["bar_count"] or 0) - int(before["bar_count"] or 0),
            "bounds_before": before,
            "bounds_after": after,
        }

    def get_meta(self, token: str | None = None) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                if token:
                    rows = conn.execute(
                        """
                        SELECT interval_sec, bar_count, oldest_ts, newest_ts, last_fetch_at, last_error
                        FROM angel_historic_sync
                        WHERE token = ?
                        ORDER BY interval_sec
                        """,
                        (str(token),),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT token, interval_sec, bar_count, oldest_ts, newest_ts,
                               last_fetch_at, last_error
                        FROM angel_historic_sync
                        ORDER BY token, interval_sec
                        """
                    ).fetchall()
            finally:
                conn.close()

        intervals: dict[str, Any] = {}
        for row in rows:
            key = str(row["interval_sec"])
            meta_row = {
                "interval_sec": row["interval_sec"],
                "label": INTERVAL_LABEL.get(int(row["interval_sec"]), key),
                "bar_count": row["bar_count"],
                "oldest": _fmt_ts(row["oldest_ts"]),
                "newest": _fmt_ts(row["newest_ts"]),
                "last_fetch_at": _fmt_ts(row["last_fetch_at"]),
                "last_error": row["last_error"],
            }
            if token:
                intervals[key] = meta_row
            else:
                tok = str(row["token"])
                intervals[f"{tok}:{key}"] = {**meta_row, "token": tok}
        exists = os.path.isfile(self.db_path)
        return {
            "db_path": self.db_path,
            "db_exists": exists,
            "db_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if exists else 0.0,
            "token": str(token) if token else None,
            "intervals": intervals,
        }


angel_historic_store = AngelHistoricStore()


def _fmt_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=IST).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return None
