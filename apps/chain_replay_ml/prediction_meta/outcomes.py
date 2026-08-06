"""Outcome columns — map master targets + optional tick-path enrichment."""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np

from chain_replay_ml.ticks import TickTimeline

Trend = Literal["UP", "DOWN", "FLAT"]

# Micro-profile keys for compute_path_outcomes (seconds accumulated per call).
PATH_OUTCOME_PROFILE_KEYS: tuple[str, ...] = (
    "timeline_lookup",
    "future_window_index",
    "future_window_slice",
    "future_tick_scan",
    "mfe_mae_update",
    "target_hit_detection",
    "dd_before_target_update",
    "timestamp_tracking",
    "result_construction",
)


def prepare_path_outcome_timelines(
    timelines: dict[str, TickTimeline] | None,
) -> int:
    """
    Convert tick timelines to path arrays once per day (before prediction loop).

    Returns number of timelines prepared.
    """
    if not timelines:
        return 0
    n = 0
    for tl in timelines.values():
        if tl is None:
            continue
        if not getattr(tl, "timestamps", None):
            continue
        tl.ensure_path_arrays()
        n += 1
    return n


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def move_trend(move: float | None) -> Trend | None:
    """UP / DOWN / FLAT from a signed move (predicted or actual)."""
    if move is None:
        return None
    try:
        m = float(move)
    except (TypeError, ValueError):
        return None
    if m > 0:
        return "UP"
    if m < 0:
        return "DOWN"
    return "FLAT"


def map_point_outcomes(row: dict[str, Any], entry_ltp: float | None) -> dict[str, float | None]:
    """Copy horizon LTP targets from master row to actual_* columns."""
    out = {
        "actual_30s_ltp": _f(row.get("future_ltp_30s")),
        "actual_1m_ltp": _f(row.get("future_ltp_1m")),
        "actual_3m_ltp": _f(row.get("future_ltp_3m") or row.get("future_ltp_180s")),
        "actual_5m_ltp": _f(row.get("future_ltp_5m")),
    }
    return out


def compute_time_to_target(
    seg_ts: list[float],
    seg_ltp: list[float],
    *,
    entry_ts: float,
    entry_ltp: float,
    predicted_ltp: float,
) -> float | None:
    """
    Seconds from entry until Predicted LTP is first touched.

    UP: first tick with LTP >= predicted
    DOWN: first tick with LTP <= predicted
    FLAT: 0.0

    Returns -1.0 when never reached; None when inputs invalid.
    """
    reached = first_target_reached_ts(
        seg_ts,
        seg_ltp,
        entry_ts=entry_ts,
        entry_ltp=entry_ltp,
        predicted_ltp=predicted_ltp,
    )
    if reached is None:
        if not seg_ts or not seg_ltp or len(seg_ts) != len(seg_ltp):
            return None
        try:
            entry = float(entry_ltp)
            target = float(predicted_ltp)
            float(entry_ts)
        except (TypeError, ValueError):
            return None
        if entry != entry or target != target:
            return None
        return -1.0
    return float(reached) - float(entry_ts)


def first_target_reached_ts(
    seg_ts: Sequence[float],
    seg_ltp: Sequence[float],
    *,
    entry_ts: float,
    entry_ltp: float,
    predicted_ltp: float,
) -> float | None:
    """Absolute unix timestamp when Predicted LTP is first touched; NULL if never."""
    n = len(seg_ts)
    if n == 0 or n != len(seg_ltp):
        return None
    try:
        entry = float(entry_ltp)
        target = float(predicted_ltp)
        t0 = float(entry_ts)
    except (TypeError, ValueError):
        return None
    if entry != entry or target != target:
        return None

    if target == entry:
        return t0

    arr_ltp = np.asarray(seg_ltp, dtype=np.float64)
    arr_ts = np.asarray(seg_ts, dtype=np.float64)
    if target > entry:
        hits = np.flatnonzero(arr_ltp >= target)
    else:
        hits = np.flatnonzero(arr_ltp <= target)
    if hits.size == 0:
        return None
    return float(arr_ts[int(hits[0])])


def _direction_sign(entry_ltp: float, predicted_ltp: float | None) -> int:
    """+1 UP, -1 DOWN, 0 FLAT (treated like UP for MFE/MAE extremes)."""
    if predicted_ltp is None:
        return 1
    if predicted_ltp > entry_ltp:
        return 1
    if predicted_ltp < entry_ltp:
        return -1
    return 0


def compute_dd_before_target(
    seg_ts: Sequence[float],
    seg_ltp: Sequence[float],
    *,
    entry_ts: float,
    entry_ltp: float,
    predicted_ltp: float | None,
    target_reached_at: float | None,
    maximum_drawdown: float | None,
    time_to_max_drawdown: float | None,
) -> tuple[float | None, float | None]:
    """
    Max adverse excursion before first target hit (non-negative magnitude).

    UP:  current_ltp − min(LTP) on [entry, target_reached_at]
    DOWN: max(LTP) − current_ltp on that window

    If target never reached → (maximum_drawdown, time_to_max_drawdown).
    If dd_before_target == 0 → time_to_dd_before_target forced to 0.
    Time of extreme = first occurrence of min/max.
    """
    if predicted_ltp is None:
        return None, None

    try:
        entry = float(entry_ltp)
        t0 = float(entry_ts)
    except (TypeError, ValueError):
        return None, None
    if entry != entry:  # NaN
        return None, None

    if target_reached_at is None:
        dd = float(maximum_drawdown) if maximum_drawdown is not None else None
        t_dd = float(time_to_max_drawdown) if time_to_max_drawdown is not None else None
        if dd is not None:
            dd = max(0.0, dd)
            if dd == 0.0:
                t_dd = 0.0
        return dd, t_dd

    try:
        t_hit = float(target_reached_at)
    except (TypeError, ValueError):
        return None, None

    n = len(seg_ts)
    if n == 0 or n != len(seg_ltp):
        return 0.0, 0.0

    # Integer window on prebuilt arrays (no list copy of full segment).
    if isinstance(seg_ts, np.ndarray) and isinstance(seg_ltp, np.ndarray):
        i0 = int(np.searchsorted(seg_ts, t0, side="left"))
        i1 = int(np.searchsorted(seg_ts, t_hit, side="right"))
        if i0 >= i1:
            return 0.0, 0.0
        win_ts = seg_ts[i0:i1]
        win_ltp = seg_ltp[i0:i1]
        direction = _direction_sign(entry, float(predicted_ltp))
        if direction >= 0:
            extreme_i = int(np.argmin(win_ltp))
            extreme = float(win_ltp[extreme_i])
            dd = max(0.0, entry - extreme)
        else:
            extreme_i = int(np.argmax(win_ltp))
            extreme = float(win_ltp[extreme_i])
            dd = max(0.0, extreme - entry)
        if dd == 0.0:
            return 0.0, 0.0
        return dd, float(win_ts[extreme_i]) - t0

    win_ts: list[float] = []
    win_ltp: list[float] = []
    for i in range(n):
        tf = float(seg_ts[i])
        if t0 <= tf <= t_hit:
            win_ts.append(tf)
            win_ltp.append(float(seg_ltp[i]))
    if not win_ltp:
        return 0.0, 0.0

    direction = _direction_sign(entry, float(predicted_ltp))
    if direction >= 0:
        # UP / FLAT — adverse = dip below entry
        extreme = min(win_ltp)
        dd = max(0.0, entry - extreme)
        extreme_i = win_ltp.index(extreme)
    else:
        # DOWN — adverse = rally above entry
        extreme = max(win_ltp)
        dd = max(0.0, extreme - entry)
        extreme_i = win_ltp.index(extreme)

    if dd == 0.0:
        return 0.0, 0.0
    return dd, float(win_ts[extreme_i]) - t0


def compute_path_outcomes(
    timeline: TickTimeline | None,
    *,
    ts: float,
    entry_ltp: float,
    horizon_sec: float,
    predicted_ltp: float | None = None,
    profile: dict[str, float] | None = None,
) -> dict[str, float | None]:
    """
    Intra-horizon path stats (direction-aware MFE / MAE).

    ``horizon_sec`` is required and must match the configured regression target
    horizon (e.g. 180 for future_ltp_3m). There is no default 5-minute window.

    UP prediction:
      max_profit = high - entry, max_drawdown = entry - low
    DOWN prediction:
      max_profit = entry - low, max_drawdown = high - entry
    Both amounts stored as non-negative.

    dd_before_target: adverse excursion before first target hit (also non-negative).

    Return keys use horizon-neutral names (``actual_max_profit``, …). Legacy
    ``*_5m`` aliases are also populated with the same values for older
    prediction_meta schemas — they are naming leftovers, not a 5m window.

    If ``profile`` is a dict, accumulates wall seconds into PATH_OUTCOME_PROFILE_KEYS
    for this call (no algorithm change — instrumentation only).

    Hot path uses prebuilt NumPy arrays from ``TickTimeline.ensure_path_arrays()``
    (convert once per timeline / day — never rebuild LTP lists per prediction).
    """
    def _mark() -> float:
        return time.perf_counter()

    def _add(key: str, t0: float) -> None:
        if profile is not None:
            profile[key] = float(profile.get(key) or 0.0) + (_mark() - t0)

    hz = float(horizon_sec)
    if hz <= 0:
        raise ValueError(f"horizon_sec must be positive, got {horizon_sec!r}")

    exit_at = float(ts) + hz
    empty = {
        "actual_high": None,
        "actual_low": None,
        "actual_max_profit": None,
        "actual_max_drawdown": None,
        "ticks_above_entry": None,
        "ticks_below_entry": None,
        # Legacy aliases (same values; not a hardcoded 5m window)
        "actual_high_5m": None,
        "actual_low_5m": None,
        "actual_max_profit_5m": None,
        "actual_max_drawdown_5m": None,
        "ticks_above_entry_5m": None,
        "ticks_below_entry_5m": None,
        "time_to_max_profit": None,
        "time_to_max_drawdown": None,
        "time_to_target": None,
        "target_reached_at": None,
        "max_profit_at": None,
        "max_drawdown_at": None,
        "dd_before_target": None,
        "time_to_dd_before_target": None,
        "exit_at": exit_at,
        "horizon_sec": hz,
    }
    if timeline is None or not timeline.timestamps or entry_ltp <= 0:
        return empty

    t0 = _mark()
    # Cached arrays — O(1) after prepare_path_outcome_timelines / first ensure.
    stamps, ltps = timeline.ensure_path_arrays()
    _add("timeline_lookup", t0)

    t0 = _mark()
    start_i = int(np.searchsorted(stamps, ts, side="left"))
    end_i = int(np.searchsorted(stamps, exit_at, side="right"))
    _add("future_window_index", t0)
    if start_i >= end_i:
        return empty

    t0 = _mark()
    seg_ts = stamps[start_i:end_i]
    seg_ltp = ltps[start_i:end_i]
    _add("future_window_slice", t0)

    t0 = _mark()
    high = float(seg_ltp.max())
    low = float(seg_ltp.min())
    max_i = int(seg_ltp.argmax())
    min_i = int(seg_ltp.argmin())
    entry_f = float(entry_ltp)
    above = int(np.count_nonzero(seg_ltp > entry_f))
    below = int(np.count_nonzero(seg_ltp < entry_f))
    _add("future_tick_scan", t0)

    t0 = _mark()
    direction = _direction_sign(entry_f, float(predicted_ltp) if predicted_ltp is not None else None)
    if direction >= 0:
        # UP or FLAT
        max_profit = max(0.0, high - entry_f)
        max_dd = max(0.0, entry_f - low)
        max_profit_at = float(seg_ts[max_i])
        max_drawdown_at = float(seg_ts[min_i])
    else:
        # DOWN
        max_profit = max(0.0, entry_f - low)
        max_dd = max(0.0, high - entry_f)
        max_profit_at = float(seg_ts[min_i])
        max_drawdown_at = float(seg_ts[max_i])
    _add("mfe_mae_update", t0)

    t0 = _mark()
    t_mfe = max_profit_at - float(ts)
    t_mae = max_drawdown_at - float(ts)
    _add("timestamp_tracking", t0)

    t0 = _mark()
    target_reached_at: float | None = None
    ttt: float | None = None
    if predicted_ltp is not None:
        target_reached_at = first_target_reached_ts(
            seg_ts,
            seg_ltp,
            entry_ts=float(ts),
            entry_ltp=entry_f,
            predicted_ltp=float(predicted_ltp),
        )
        if target_reached_at is None:
            ttt = -1.0
        else:
            ttt = float(target_reached_at) - float(ts)
    _add("target_hit_detection", t0)

    t0 = _mark()
    dd_bt, t_dd_bt = compute_dd_before_target(
        seg_ts,
        seg_ltp,
        entry_ts=float(ts),
        entry_ltp=entry_f,
        predicted_ltp=float(predicted_ltp) if predicted_ltp is not None else None,
        target_reached_at=target_reached_at,
        maximum_drawdown=max_dd,
        time_to_max_drawdown=t_mae,
    )
    _add("dd_before_target_update", t0)

    t0 = _mark()
    out = {
        "actual_high": high,
        "actual_low": low,
        "actual_max_profit": max_profit,
        "actual_max_drawdown": max_dd,
        "ticks_above_entry": float(above),
        "ticks_below_entry": float(below),
        # Legacy aliases — same configured-horizon values
        "actual_high_5m": high,
        "actual_low_5m": low,
        "actual_max_profit_5m": max_profit,
        "actual_max_drawdown_5m": max_dd,
        "ticks_above_entry_5m": float(above),
        "ticks_below_entry_5m": float(below),
        "time_to_max_profit": t_mfe,
        "time_to_max_drawdown": t_mae,
        "time_to_target": ttt,
        "target_reached_at": target_reached_at,
        "max_profit_at": max_profit_at,
        "max_drawdown_at": max_drawdown_at,
        "dd_before_target": dd_bt,
        "time_to_dd_before_target": t_dd_bt,
        "exit_at": exit_at,
        "horizon_sec": hz,
    }
    _add("result_construction", t0)
    return out


def compute_prediction_quality(
    *,
    ensemble_mean: float | None,
    entry_ltp: float | None,
    actual_ltp: float | None,
) -> dict[str, float | None]:
    """Endpoint Model Quality error + direction (canonical evaluator rules)."""
    if ensemble_mean is None or entry_ltp is None or actual_ltp is None:
        return {"prediction_error": None, "direction_correct": None}
    from chain_replay_ml.training.evaluator import direction_correct_flag

    err = float(ensemble_mean) - float(actual_ltp)
    direction = direction_correct_flag(
        float(ensemble_mean), float(actual_ltp), float(entry_ltp)
    )
    return {
        "prediction_error": round(err, 4),
        "direction_correct": float(direction) if direction is not None else None,
    }
