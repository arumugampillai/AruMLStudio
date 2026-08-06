"""Spot tick coverage — grid clipping, freshness, and dataset_meta statistics."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from chain_replay_ml.ticks import TickTimeline

from .day_context import DayContext

IST = ZoneInfo("Asia/Kolkata")
LOOKBACK_START_SEC = 60.0
DEFAULT_MAX_STALE_SEC = 10.0
COVERAGE_OK_PCT = 95.0


def _fmt_ist_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%H:%M:%S")


def _fmt_ist_range(first_ts: float, last_ts: float) -> str:
    return f"{_fmt_ist_time(first_ts)} → {_fmt_ist_time(last_ts)}"


def spot_tick_bounds(index_tl: TickTimeline) -> tuple[float, float] | None:
    if not index_tl.timestamps:
        return None
    return float(index_tl.timestamps[0]), float(index_tl.timestamps[-1])


def resolve_effective_session_start_ts(ctx: DayContext) -> float:
    """First available spot tick, else exchange session open (calendar only)."""
    ess = float(getattr(ctx, "effective_session_start_ts", 0.0) or 0.0)
    if ess > 0:
        return ess
    bounds = spot_tick_bounds(ctx.index_tl)
    if bounds:
        return float(bounds[0])
    return float(ctx.open_ts)


def sync_feature_grid_step(
    ctx: DayContext,
    step_sec: int,
    *,
    gap_max_sec: float | None = None,
) -> None:
    """Record dataset sampling interval on DayContext (no EMA precompute)."""
    step = max(int(step_sec), 1)
    gap_key = float(gap_max_sec) if gap_max_sec is not None else 0.0
    if int(ctx.feature_grid_step_sec) == step and float(ctx.feature_grid_gap_max_sec or 0.0) == gap_key:
        return
    ctx.feature_grid_step_sec = step
    ctx.feature_grid_gap_max_sec = gap_key


def feature_grid_origin_ts(ctx: DayContext) -> float:
    """First aligned feature-grid timestamp for spot.hl bar indexing."""
    from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC, uniform_grid

    origin = resolve_effective_session_start_ts(ctx)
    step = max(int(ctx.feature_grid_step_sec or EMA_BAR_INTERVAL_SEC), 1)
    grid = uniform_grid(origin, ctx.close_ts, float(step))
    if grid:
        return float(grid[0])
    return origin


def compute_spot_coverage(
    ctx: DayContext,
    *,
    max_stale_sec: float = DEFAULT_MAX_STALE_SEC,
) -> dict[str, Any]:
    """Coverage stats from spot ticks vs calendar session (Rule 1 + 4)."""
    bounds = spot_tick_bounds(ctx.index_tl)
    session_sec = max(0.0, ctx.close_ts - ctx.open_ts)
    if not bounds or session_sec <= 0:
        return {
            "first_tick": None,
            "last_tick": None,
            "first_tick_ts": None,
            "last_tick_ts": None,
            "tick_range": None,
            "coverage_pct": 0.0,
            "missing_start_sec": 0.0,
            "missing_end_sec": 0.0,
            "session_sec": session_sec,
            "max_stale_sec": max_stale_sec,
            "status": "partial",
        }

    first_ts, last_ts = bounds
    tick_span = max(0.0, last_ts - first_ts)
    coverage_pct = round(100.0 * tick_span / session_sec, 1) if session_sec > 0 else 0.0
    missing_start = max(0.0, first_ts - ctx.open_ts)
    missing_end = max(0.0, ctx.close_ts - last_ts)
    status = "ok" if coverage_pct >= COVERAGE_OK_PCT else "partial"

    return {
        "first_tick": _fmt_ist_time(first_ts),
        "last_tick": _fmt_ist_time(last_ts),
        "first_tick_ts": first_ts,
        "last_tick_ts": last_ts,
        "tick_range": _fmt_ist_range(first_ts, last_ts),
        "coverage_pct": coverage_pct,
        "missing_start_sec": round(missing_start, 1),
        "missing_end_sec": round(missing_end, 1),
        "session_sec": session_sec,
        "max_stale_sec": max_stale_sec,
        "status": status,
    }


def clipped_grid_bounds(
    ctx: DayContext,
    *,
    max_horizon_sec: int,
) -> tuple[float, float] | None:
    """Rule 3 — clip sampling grid so every sample has a full target horizon.

    End bound is driven by available spot data, not a hardcoded market close:
    ``grid_end = last_tick - max_horizon_sec``. Samples after that cannot be
    evaluated over the complete prediction window (regression + path outcomes).
    """
    bounds = spot_tick_bounds(ctx.index_tl)
    if not bounds:
        return None
    first_tick, last_tick = bounds
    grid_start = max(ctx.open_ts + LOOKBACK_START_SEC, first_tick)
    # Exclude the final ``max_horizon_sec`` of available ticks (no close clock).
    grid_end = float(last_tick) - float(max_horizon_sec)
    if grid_end <= grid_start:
        return None
    return grid_start, grid_end


def list_clipped_grid_timestamps(
    ctx: DayContext,
    *,
    step_sec: int,
    max_horizon_sec: int,
) -> list[float]:
    """All grid points on clipped spot span (before fresh filter)."""
    bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_horizon_sec)
    if not bounds:
        return []
    grid_start, grid_end = bounds
    out: list[float] = []
    t = grid_start
    while t <= grid_end + 0.001:
        out.append(t)
        t += step_sec
    return out


def build_clipped_sample_timestamps(
    ctx: DayContext,
    *,
    step_sec: int,
    max_horizon_sec: int,
    max_stale_sec: float = DEFAULT_MAX_STALE_SEC,
) -> tuple[list[float], dict[str, Any]]:
    """Build 10s grid clipped to spot coverage; keep only fresh spot timestamps (Rule 2+3)."""
    bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_horizon_sec)
    coverage = compute_spot_coverage(ctx, max_stale_sec=max_stale_sec)
    if not bounds:
        coverage["grid_points"] = 0
        coverage["fresh_grid_points"] = 0
        return [], coverage

    grid_start, grid_end = bounds
    timestamps: list[float] = []
    t = grid_start
    while t <= grid_end + 0.001:
        if ctx.index_tl.is_fresh_at(t, max_stale_sec):
            timestamps.append(t)
        t += step_sec

    coverage["grid_start"] = _fmt_ist_time(grid_start)
    coverage["grid_end"] = _fmt_ist_time(grid_end)
    coverage["grid_points"] = len(list_clipped_grid_timestamps(ctx, step_sec=step_sec, max_horizon_sec=max_horizon_sec))
    coverage["fresh_grid_points"] = len(timestamps)
    return timestamps, coverage


def spot_coverage_preview(
    chart_dir: str,
    trading_day: str,
    market: str,
    expiry: str,
) -> dict[str, Any]:
    """Lightweight tick coverage from spot timeline (no full feature build)."""
    from .day_context import SourceSpec, load_day_context

    try:
        ctx = load_day_context(
            chart_dir,
            SourceSpec(
                source_id=f"{trading_day}|{market}|{expiry}",
                trading_day=trading_day,
                market=market,
                expiry=expiry,
            ),
        )
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}

    return compute_spot_coverage(ctx)


def snap_to_training_grid(clock_ts: float, *, grid_start: float, step_sec: int) -> float:
    step = max(int(step_sec), 1)
    rel = float(clock_ts) - float(grid_start)
    slot = int(rel // step) if rel >= 0 else 0
    return float(grid_start) + slot * step


def resolve_inference_sample_ts(
    only_timestamp: float,
    *,
    horizons_sec: list[int],
    training_step_sec: int,
    grid_start: float | None,
) -> float:
    """Live inference uses exact replay clock; training dataset builds snap to model grid."""
    ts = float(only_timestamp)
    if not horizons_sec:
        return ts
    step_i = max(int(training_step_sec), 1)
    if grid_start is None:
        return ts
    return snap_to_training_grid(ts, grid_start=float(grid_start), step_sec=step_i)
