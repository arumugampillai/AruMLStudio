"""Historical NIFTY multi-TF EMA context from angel_historic_bars.db.

These features are **not** computed from Trading Day ticks. EMAs are derived
once from the continuous historic candle series (warmup included), then each
sample timestamp receives an as-of (backward-only) lookup:

  spot_{1m|3m|5m|15m}_ema{9|20|50|100|200}

Lookup rule: latest candle with ``bucket_start <= tick_ts`` (no future bars,
no interpolation, no nearest-neighbour ahead).
"""

from __future__ import annotations

import bisect
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

NIFTY_INDEX_TOKEN = "99926000"

HISTORIC_SPOT_EMA_TIMEFRAMES: tuple[str, ...] = ("1m", "3m", "5m", "15m")
HISTORIC_SPOT_EMA_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200)

TIMEFRAME_INTERVAL_SEC: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
}


def historic_spot_ema_feature_name(timeframe: str, period: int) -> str:
    return f"spot_{str(timeframe).strip().lower()}_ema{int(period)}"


def historic_spot_ema_feature_names(
    timeframes: Sequence[str] = HISTORIC_SPOT_EMA_TIMEFRAMES,
    periods: Sequence[int] = HISTORIC_SPOT_EMA_PERIODS,
) -> tuple[str, ...]:
    return tuple(
        historic_spot_ema_feature_name(tf, p)
        for tf in timeframes
        for p in periods
    )


HISTORIC_SPOT_EMA_FEATURES: tuple[str, ...] = historic_spot_ema_feature_names()
HISTORIC_SPOT_EMA_FEATURE_SET: frozenset[str] = frozenset(HISTORIC_SPOT_EMA_FEATURES)


def active_historic_spot_ema_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return HISTORIC_SPOT_EMA_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in HISTORIC_SPOT_EMA_FEATURE_SET)


def needs_historic_spot_ema(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in HISTORIC_SPOT_EMA_FEATURE_SET for f in active)


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


@dataclass(frozen=True)
class _TfEmaSeries:
    label: str
    interval_sec: int
    timestamps: tuple[float, ...]
    emas: dict[int, tuple[float | None, ...]]

    def asof_index(self, tick_ts: float) -> int | None:
        """Largest i with timestamps[i] <= tick_ts, or None."""
        ts = self.timestamps
        if not ts:
            return None
        i = bisect.bisect_right(ts, float(tick_ts)) - 1
        if i < 0:
            return None
        return i


@dataclass
class HistoricSpotEmaBook:
    """Precomputed multi-TF EMA levels with O(log N) as-of lookup."""

    series: tuple[_TfEmaSeries, ...]
    ema_periods: tuple[int, ...]
    feature_names: tuple[str, ...] = field(init=False)
    db_path: str = ""

    def __post_init__(self) -> None:
        labels = [s.label for s in self.series]
        object.__setattr__(
            self,
            "feature_names",
            historic_spot_ema_feature_names(labels, self.ema_periods),
        )

    def empty_features(self) -> dict[str, None]:
        return {name: None for name in self.feature_names}

    def levels_at(self, tick_ts: float) -> dict[str, float | None]:
        """Backward-only as-of EMA levels for ``tick_ts``."""
        t = float(tick_ts)
        out: dict[str, float | None] = {}
        for series in self.series:
            idx = series.asof_index(t)
            for period in self.ema_periods:
                key = historic_spot_ema_feature_name(series.label, period)
                if idx is None:
                    out[key] = None
                    continue
                val = series.emas[period][idx]
                out[key] = float(val) if val is not None else None
        return out

    def match_diagnostics(self, tick_ts: float) -> dict[str, Any]:
        """Human-readable as-of match report for validation."""
        t = float(tick_ts)
        tick_ist = datetime.fromtimestamp(t, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
        matched: dict[str, Any] = {"tick_ts": t, "tick_ist": tick_ist, "timeframes": {}}
        levels = self.levels_at(t)
        for series in self.series:
            idx = series.asof_index(t)
            if idx is None:
                matched["timeframes"][series.label] = {
                    "matched_bucket_start": None,
                    "matched_ist": None,
                    "index": None,
                }
                continue
            bucket = float(series.timestamps[idx])
            # Guard: never ahead of tick.
            assert bucket <= t + 1e-9, (series.label, bucket, t)
            matched["timeframes"][series.label] = {
                "matched_bucket_start": bucket,
                "matched_ist": datetime.fromtimestamp(bucket, tz=IST).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "index": idx,
            }
        matched["ema_values"] = {
            name: levels.get(name)
            for name in self.feature_names
        }
        return matched


def resolve_historic_bars_db_path(chart_dir: str | None = None) -> str:
    from storage.angel_historic_store import default_db_path

    return default_db_path(chart_dir)


def _lookback_from_ts(
    trading_day: str,
    *,
    interval_sec: int,
    max_period: int,
) -> float:
    d0 = date.fromisoformat(str(trading_day))
    bars_needed = max(int(max_period) * 3, int(max_period) + 5)
    bars_per_day = max(1, int(6.25 * 3600 / max(1, interval_sec)))
    lookback_days = max(5, (bars_needed + bars_per_day - 1) // bars_per_day + 2)
    start = d0 - timedelta(days=lookback_days)
    return datetime(start.year, start.month, start.day, 9, 15, tzinfo=IST).timestamp()


def _day_end_ts(trading_day: str) -> float:
    d = date.fromisoformat(str(trading_day))
    # Inclusive last 1m open at 15:29; larger TFs still have opens <= 15:29.
    return datetime(d.year, d.month, d.day, 15, 29, tzinfo=IST).timestamp()


def _load_closes(
    db_path: str,
    *,
    token: str,
    interval_sec: int,
    from_ts: float,
    to_ts: float,
) -> tuple[list[float], list[float]]:
    sql = """
        SELECT bucket_start, close
        FROM angel_historic_bars
        WHERE token = ? AND interval_sec = ?
          AND bucket_start >= ? AND bucket_start <= ?
        ORDER BY bucket_start ASC
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            sql, (str(token), int(interval_sec), float(from_ts), float(to_ts))
        ).fetchall()
    timestamps = [float(r[0]) for r in rows]
    closes = [float(r[1]) for r in rows]
    return timestamps, closes


def build_historic_spot_ema_book(
    *,
    trading_day: str,
    chart_dir: str | None = None,
    db_path: str | None = None,
    token: str = NIFTY_INDEX_TOKEN,
    timeframes: Sequence[str] = HISTORIC_SPOT_EMA_TIMEFRAMES,
    ema_periods: Sequence[int] = HISTORIC_SPOT_EMA_PERIODS,
) -> HistoricSpotEmaBook:
    """Load historic bars once and precompute EMA series for as-of joins."""
    path = db_path or resolve_historic_bars_db_path(chart_dir)
    if not path or not os.path.isfile(path):
        # Empty book — enricher returns NULLs.
        return HistoricSpotEmaBook(series=tuple(), ema_periods=tuple(int(p) for p in ema_periods), db_path=str(path or ""))

    periods = tuple(int(p) for p in ema_periods)
    if not periods:
        raise ValueError("ema_periods must be non-empty")
    max_period = max(periods)
    to_ts = _day_end_ts(trading_day)

    series_list: list[_TfEmaSeries] = []
    for label in timeframes:
        label_n = str(label).strip().lower()
        if label_n not in TIMEFRAME_INTERVAL_SEC:
            raise ValueError(f"unsupported historic spot EMA timeframe: {label_n}")
        interval_sec = TIMEFRAME_INTERVAL_SEC[label_n]
        from_ts = _lookback_from_ts(
            trading_day, interval_sec=interval_sec, max_period=max_period
        )
        timestamps, closes = _load_closes(
            path,
            token=token,
            interval_sec=interval_sec,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        emas = {
            p: tuple(compute_ema(closes, p))
            for p in periods
        }
        series_list.append(
            _TfEmaSeries(
                label=label_n,
                interval_sec=interval_sec,
                timestamps=tuple(timestamps),
                emas=emas,
            )
        )
    return HistoricSpotEmaBook(
        series=tuple(series_list),
        ema_periods=periods,
        db_path=path,
    )


def ensure_historic_spot_ema_book(
    ctx: Any,
    *,
    chart_dir: str | None = None,
    active_features: Iterable[str] | None = None,
) -> HistoricSpotEmaBook | None:
    """Attach a day-scoped book on ``ctx`` when any historic EMA feature is active."""
    if not needs_historic_spot_ema(active_features):
        return None
    existing = getattr(ctx, "historic_spot_ema_book", None)
    if isinstance(existing, HistoricSpotEmaBook):
        return existing
    trading_day = str(getattr(getattr(ctx, "source", None), "trading_day", "") or "")
    if not trading_day:
        return None
    book = build_historic_spot_ema_book(trading_day=trading_day, chart_dir=chart_dir)
    try:
        ctx.historic_spot_ema_book = book
    except Exception:
        pass
    return book


def enrich_historic_spot_ema_features(
    raw: dict[str, Any],
    *,
    ts: float,
    ctx: Any | None = None,
    book: HistoricSpotEmaBook | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_historic_spot_ema_features(active_features)
    if not wanted:
        return raw
    resolved = book
    if resolved is None and ctx is not None:
        resolved = getattr(ctx, "historic_spot_ema_book", None)
    out = dict(raw)
    if resolved is None:
        for name in wanted:
            out.setdefault(name, None)
        return out
    levels = resolved.levels_at(float(ts))
    for name in wanted:
        out[name] = levels.get(name)
    return out


def format_historic_bar_check(
    trading_day: str,
    *,
    chart_dir: str | None = None,
    db_path: str | None = None,
) -> list[str]:
    """Lines for Builder / No-Null diagnostics: bars loaded + EMA200 readiness."""
    day = str(trading_day or "").strip()
    if not day:
        return ["Historical Bar Check: (skipped — no trading_day)"]

    book = build_historic_spot_ema_book(
        trading_day=day, chart_dir=chart_dir, db_path=db_path
    )
    lines = [
        "Historical Bar Check",
        "-" * 72,
        f"Trading day: {day}",
        f"Historic DB: {book.db_path or '(missing)'}",
    ]
    if not book.series:
        lines.append(
            "  (no series loaded — historic bar DB missing or empty; "
            "spot_*_ema* will be NULL)"
        )
        return lines

    def _fmt(ts: float | None) -> str:
        if ts is None:
            return "(none)"
        return datetime.fromtimestamp(float(ts), tz=IST).strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )

    for s in book.series:
        lines.append(f"{s.label} bars loaded: {len(s.timestamps):,}")
        if s.timestamps:
            lines.append(f"  Earliest bar timestamp: {_fmt(s.timestamps[0])}")
            lines.append(f"  Latest bar timestamp:   {_fmt(s.timestamps[-1])}")

    lines.append("")
    lines.append("EMA200 first non-null:")
    for s in book.series:
        ema = s.emas.get(200)
        first_i = next((i for i, v in enumerate(ema or ()) if v is not None), None)
        if first_i is None:
            lines.append(f"  {s.label:4s}: (never — not enough bars for EMA200)")
        else:
            lines.append(f"  {s.label:4s}: {_fmt(s.timestamps[first_i])}")

    try:
        d = date.fromisoformat(day)
        open_ts = datetime(d.year, d.month, d.day, 9, 16, tzinfo=IST).timestamp()
        levels = book.levels_at(open_ts)
        lines.append("")
        lines.append(f"As-of lookup at {day} 09:16 IST:")
        for label in HISTORIC_SPOT_EMA_TIMEFRAMES:
            key = historic_spot_ema_feature_name(label, 200)
            val = levels.get(key)
            lines.append(
                f"  {key}: " + ("NULL" if val is None else f"{float(val):.4f}")
            )
        ready = all(
            levels.get(historic_spot_ema_feature_name(label, 200)) is not None
            for label in HISTORIC_SPOT_EMA_TIMEFRAMES
        )
        lines.append("")
        if ready:
            lines.append(
                "Verdict: historic bars + EMA200 are ready at session open. "
                "If Master still shows identical early-session NULLs across "
                "spot_*_ema200, Feature Policy grid warm-up is the gate "
                "(not missing history)."
            )
        else:
            lines.append(
                "Verdict: EMA200 not ready at session open — check lookback "
                "window / historic DB coverage."
            )
    except Exception as exc:
        lines.append(f"  (as-of probe failed: {exc})")
    return lines
