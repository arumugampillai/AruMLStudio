"""Read-only 1-minute NIFTY spot candle loader (SQLite historic bars)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator
from zoneinfo import ZoneInfo

from .config import DEFAULT_INTERVAL_SEC, DEFAULT_NIFTY_TOKEN, PriceSource

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Candle:
    trading_day: str  # YYYY-MM-DD
    timestamp: float  # unix seconds (bucket start)
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    def assignment_price(self, source: PriceSource) -> float:
        if source == "close":
            return float(self.close)
        if source in ("hlc3", "typical_price"):
            return float((self.high + self.low + self.close) / 3.0)
        if source == "ohlc4":
            return float((self.open + self.high + self.low + self.close) / 4.0)
        raise ValueError(f"unsupported price source: {source}")


def _day_bounds_unix(trading_day: str) -> tuple[float, float]:
    d = date.fromisoformat(trading_day)
    start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).timestamp()
    # Inclusive last 1m bar at 15:29
    end = datetime(d.year, d.month, d.day, 15, 29, tzinfo=IST).timestamp()
    return start, end


class CandleLoader:
    """Load NIFTY 1m OHLC from angel_historic_bars (or compatible schema)."""

    def __init__(
        self,
        db_path: str,
        *,
        token: str = DEFAULT_NIFTY_TOKEN,
        interval_sec: int = DEFAULT_INTERVAL_SEC,
    ) -> None:
        self.db_path = db_path
        self.token = str(token)
        self.interval_sec = int(interval_sec)

    def list_trading_days(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[str]:
        sql = """
            SELECT DISTINCT date(bucket_start, 'unixepoch', '+5 hours', '+30 minutes') AS d
            FROM angel_historic_bars
            WHERE token = ? AND interval_sec = ?
        """
        params: list[object] = [self.token, self.interval_sec]
        if from_date:
            sql += " AND date(bucket_start, 'unixepoch', '+5 hours', '+30 minutes') >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND date(bucket_start, 'unixepoch', '+5 hours', '+30 minutes') <= ?"
            params.append(to_date)
        sql += " ORDER BY d"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows if r[0]]

    def load_day(self, trading_day: str) -> list[Candle]:
        start_ts, end_ts = _day_bounds_unix(trading_day)
        return self.load_range(
            interval_sec=self.interval_sec,
            from_ts=start_ts,
            to_ts=end_ts,
            trading_day=trading_day,
        )

    def load_range(
        self,
        *,
        interval_sec: int | None = None,
        from_ts: float | None = None,
        to_ts: float | None = None,
        trading_day: str | None = None,
    ) -> list[Candle]:
        """Load OHLC bars for any interval, optionally bounded by unix timestamps."""
        iv = int(self.interval_sec if interval_sec is None else interval_sec)
        sql = """
            SELECT bucket_start, open, high, low, close, volume
            FROM angel_historic_bars
            WHERE token = ? AND interval_sec = ?
        """
        params: list[object] = [self.token, iv]
        if from_ts is not None:
            sql += " AND bucket_start >= ?"
            params.append(float(from_ts))
        if to_ts is not None:
            sql += " AND bucket_start <= ?"
            params.append(float(to_ts))
        sql += " ORDER BY bucket_start ASC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[Candle] = []
        for bucket, o, h, l, c, vol in rows:
            ts = float(bucket)
            day = trading_day
            if day is None:
                day = datetime.fromtimestamp(ts, tz=IST).date().isoformat()
            out.append(
                Candle(
                    trading_day=day,
                    timestamp=ts,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=int(vol or 0),
                )
            )
        return out

    def iter_days(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> Iterator[tuple[str, list[Candle]]]:
        for day in self.list_trading_days(from_date=from_date, to_date=to_date):
            candles = self.load_day(day)
            if candles:
                yield day, candles
