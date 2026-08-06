"""Multi-timeframe EMA → spot ratio features (Market Context).

EMAs are precomputed once per timeframe over the loaded history.
Per ORMP row we only advance TF pointers and divide by spot_ltp(T).

No look-ahead: for each TF use the latest *completed* bar with
``timestamp <= T`` (1m prefers an exact match when present).

Timeframes / EMA periods are configurable so 30m / 60m / 1d can be
added later without changing the join architecture.

Slope features (e.g. ``5m_ema20_slope``) are intentionally not emitted yet;
the same EMA series can feed them later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from .data_loader import CandleLoader

IST = ZoneInfo("Asia/Kolkata")

# Extensible label → seconds (matches angel_historic_bars intervals).
TIMEFRAME_INTERVAL_SEC: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1d": 86400,
}

DEFAULT_MARKET_CONTEXT_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")
DEFAULT_EMA_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200)


def ratio_feature_name(timeframe: str, period: int) -> str:
    return f"{timeframe}_ema{int(period)}_to_spot_ratio"


def market_context_feature_names(
    timeframes: Sequence[str] = DEFAULT_MARKET_CONTEXT_TIMEFRAMES,
    ema_periods: Sequence[int] = DEFAULT_EMA_PERIODS,
) -> tuple[str, ...]:
    names: list[str] = []
    for tf in timeframes:
        for p in ema_periods:
            names.append(ratio_feature_name(tf, p))
    return tuple(names)


MARKET_CONTEXT_COLUMNS: tuple[str, ...] = market_context_feature_names()


def compute_ema(closes: Sequence[float], period: int) -> list[float | None]:
    """EMA with SMA seed at index ``period - 1``; earlier values are None."""
    n = len(closes)
    out: list[float | None] = [None] * n
    p = int(period)
    if p <= 0 or n < p:
        return out
    alpha = 2.0 / (p + 1.0)
    seed = sum(float(closes[i]) for i in range(p)) / float(p)
    out[p - 1] = seed
    prev = seed
    for i in range(p, n):
        prev = alpha * float(closes[i]) + (1.0 - alpha) * prev
        out[i] = prev
    return out


@dataclass
class _TimeframeSeries:
    label: str
    interval_sec: int
    timestamps: list[float]
    emas: dict[int, list[float | None]]
    cursor: int = -1  # last index with timestamp <= current T

    def advance_to(self, t: float, *, require_exact: bool) -> int | None:
        """Move cursor to latest completed bar <= t. Return index or None."""
        ts = self.timestamps
        n = len(ts)
        while self.cursor + 1 < n and ts[self.cursor + 1] <= t:
            self.cursor += 1
        if self.cursor < 0:
            return None
        if require_exact and ts[self.cursor] != t:
            return None
        return self.cursor


class MarketContextBook:
    """Precomputed multi-TF EMA book with advancing-pointer alignment."""

    def __init__(
        self,
        series: list[_TimeframeSeries],
        *,
        ema_periods: Sequence[int],
        base_interval_sec: int = 60,
    ) -> None:
        self._series = series
        self._ema_periods = tuple(int(p) for p in ema_periods)
        self._base_interval_sec = int(base_interval_sec)
        self._feature_names = market_context_feature_names(
            [s.label for s in series],
            self._ema_periods,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def empty_features(self) -> dict[str, None]:
        return {name: None for name in self._feature_names}

    def ratios_at(self, timestamp: float, spot_ltp: float) -> dict[str, float | None]:
        """Align each TF to ``timestamp`` and return EMA/spot ratios."""
        t = float(timestamp)
        spot = float(spot_ltp)
        out: dict[str, float | None] = {}
        if spot == 0.0:
            return {name: None for name in self._feature_names}

        for series in self._series:
            require_exact = series.interval_sec == self._base_interval_sec
            idx = series.advance_to(t, require_exact=require_exact)
            for period in self._ema_periods:
                key = ratio_feature_name(series.label, period)
                if idx is None:
                    out[key] = None
                    continue
                ema_val = series.emas[period][idx]
                if ema_val is None:
                    out[key] = None
                else:
                    out[key] = float(ema_val) / spot
        return out

    @classmethod
    def build(
        cls,
        loader: CandleLoader,
        *,
        timeframes: Sequence[str] = DEFAULT_MARKET_CONTEXT_TIMEFRAMES,
        ema_periods: Sequence[int] = DEFAULT_EMA_PERIODS,
        from_date: str | None = None,
        to_date: str | None = None,
        base_interval_sec: int = 60,
    ) -> "MarketContextBook":
        periods = tuple(int(p) for p in ema_periods)
        if not periods:
            raise ValueError("ema_periods must be non-empty")
        max_period = max(periods)

        series_list: list[_TimeframeSeries] = []
        for label in timeframes:
            label = str(label).strip().lower()
            if label not in TIMEFRAME_INTERVAL_SEC:
                raise ValueError(f"unsupported market-context timeframe: {label}")
            interval_sec = TIMEFRAME_INTERVAL_SEC[label]
            from_ts, to_ts = _load_window_unix(
                from_date=from_date,
                to_date=to_date,
                interval_sec=interval_sec,
                ema_period=max_period,
            )
            candles = loader.load_range(
                interval_sec=interval_sec,
                from_ts=from_ts,
                to_ts=to_ts,
            )
            timestamps = [float(c.timestamp) for c in candles]
            closes = [float(c.close) for c in candles]
            emas = {p: compute_ema(closes, p) for p in periods}
            series_list.append(
                _TimeframeSeries(
                    label=label,
                    interval_sec=interval_sec,
                    timestamps=timestamps,
                    emas=emas,
                )
            )
        return cls(
            series_list,
            ema_periods=periods,
            base_interval_sec=base_interval_sec,
        )


def _load_window_unix(
    *,
    from_date: str | None,
    to_date: str | None,
    interval_sec: int,
    ema_period: int,
) -> tuple[float | None, float | None]:
    """Expand from_date backward so EMA(period) can warm up before the ORMP range."""
    to_ts: float | None = None
    if to_date:
        d = date.fromisoformat(to_date)
        # Inclusive last session minute
        to_ts = datetime(d.year, d.month, d.day, 15, 29, tzinfo=IST).timestamp()

    if not from_date:
        return None, to_ts

    d0 = date.fromisoformat(from_date)
    # ~3x period in calendar days covers weekends / holidays for daily & intraday.
    bars_needed = max(int(ema_period) * 3, int(ema_period) + 5)
    if interval_sec >= 86400:
        lookback_days = bars_needed
    else:
        # Session ≈ 6.25h; convert bars → calendar days with buffer.
        bars_per_day = max(1, int(6.25 * 3600 / interval_sec))
        lookback_days = max(5, (bars_needed + bars_per_day - 1) // bars_per_day + 2)
    start = d0 - timedelta(days=lookback_days)
    from_ts = datetime(start.year, start.month, start.day, 9, 15, tzinfo=IST).timestamp()
    return from_ts, to_ts


def null_market_context_features(
    timeframes: Iterable[str] = DEFAULT_MARKET_CONTEXT_TIMEFRAMES,
    ema_periods: Iterable[int] = DEFAULT_EMA_PERIODS,
) -> dict[str, None]:
    return {name: None for name in market_context_feature_names(tuple(timeframes), tuple(ema_periods))}
