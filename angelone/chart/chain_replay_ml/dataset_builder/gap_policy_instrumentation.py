"""Lightweight gap-policy instrumentation for hot build paths (no cProfile)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator
from zoneinfo import ZoneInfo

_EMA_RESET_LOG = logging.getLogger("chain_replay_ml.ema_reset")
_IST = ZoneInfo("Asia/Kolkata")


@dataclass
class GapPolicyProfilerStats:
    gap_checks: int = 0
    gaps_detected: int = 0
    reset_count: int = 0
    ltp_ema_cache_hits: int = 0
    ltp_ema_rebuilds: int = 0
    wall_sec: float = 0.0
    cprofile_totals: dict[str, float] = field(default_factory=dict)
    cprofile_calls: dict[str, int] = field(default_factory=dict)
    gap_max_sec: float | None = None
    function_times_sec: dict[str, float] = field(default_factory=dict)
    function_calls: dict[str, int] = field(default_factory=dict)
    cprofile_top: list[dict[str, Any]] = field(default_factory=list)

    def gap_function_times(self) -> dict[str, float]:
        return {
            name: sec
            for name, sec in sorted(
                self.function_times_sec.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    def top_functions(self, n: int = 20) -> list[dict[str, Any]]:
        if self.cprofile_top:
            return list(self.cprofile_top[:n])
        ranked = sorted(
            self.function_times_sec.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for name, sec in ranked[:n]:
            out.append({
                "function": name,
                "total_sec": round(sec, 6),
                "calls": int(self.function_calls.get(name, 0)),
            })
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_checks": self.gap_checks,
            "gaps_detected": self.gaps_detected,
            "reset_count": self.reset_count,
            "ltp_ema_cache_hits": self.ltp_ema_cache_hits,
            "ltp_ema_rebuilds": self.ltp_ema_rebuilds,
            "wall_sec": round(self.wall_sec, 3),
            "cprofile_totals": dict(self.cprofile_totals),
            "cprofile_calls": dict(self.cprofile_calls),
            "gap_max_sec": self.gap_max_sec,
            "function_times_sec": dict(self.function_times_sec),
            "function_calls": dict(self.function_calls),
            "gap_function_times_sec": self.gap_function_times(),
            "top_functions": self.top_functions(20),
            "cprofile_top": list(self.cprofile_top),
        }


_active: ContextVar[GapPolicyProfilerStats | None] = ContextVar("gap_policy_profiler", default=None)
_wall_start: ContextVar[float | None] = ContextVar("gap_policy_wall_start", default=None)


def profiler_active() -> bool:
    return _active.get() is not None


def get_profiler() -> GapPolicyProfilerStats | None:
    return _active.get()


def _begin_profiler_session(*, gap_max_sec: float | None = None) -> GapPolicyProfilerStats:
    stats = GapPolicyProfilerStats(gap_max_sec=gap_max_sec)
    _active.set(stats)
    _wall_start.set(time.perf_counter())
    return stats


def _end_profiler_session() -> GapPolicyProfilerStats | None:
    stats = _active.get()
    wall_start = _wall_start.get()
    if stats is not None and wall_start is not None:
        stats.wall_sec = time.perf_counter() - wall_start
    _wall_start.set(None)
    _active.set(None)
    return stats


def reset_day_context_feature_grid(ctx: Any) -> None:
    """Clear cached feature-grid step so the next build_day_rows pass resyncs it."""
    if ctx is None:
        return
    ctx.feature_grid_step_sec = 0
    ctx.feature_grid_gap_max_sec = float("nan")


@contextmanager
def gap_policy_profile_block(name: str) -> Iterator[None]:
    stats = _active.get()
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        stats.function_times_sec[name] = stats.function_times_sec.get(name, 0.0) + elapsed
        stats.function_calls[name] = stats.function_calls.get(name, 0) + 1


def record_gap_check(*, is_gap: bool) -> None:
    stats = _active.get()
    if stats is None:
        return
    stats.gap_checks += 1
    if is_gap:
        stats.gaps_detected += 1


def record_gap_reset() -> None:
    stats = _active.get()
    if stats is None:
        return
    stats.reset_count += 1


def record_gap_ema_cache_hit() -> None:
    stats = _active.get()
    if stats is None:
        return
    stats.ltp_ema_cache_hits += 1


def record_gap_ema_rebuild() -> None:
    stats = _active.get()
    if stats is None:
        return
    stats.ltp_ema_rebuilds += 1


def gap_policy_enabled(gap_max_sec: float | None) -> bool:
    return gap_max_sec is not None and float(gap_max_sec) > 0


def row_gap_exceeds(
    ts: float,
    last_row_ts: float | None,
    gap_max_sec: float | None,
) -> bool:
    """O(1): one subtraction and one comparison when gap policy is active."""
    if not gap_policy_enabled(gap_max_sec) or last_row_ts is None:
        return False
    return ts - last_row_ts > float(gap_max_sec)


def ema_reset_debug_enabled() -> bool:
    return os.getenv("EMA_RESET_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _fmt_reset_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts), tz=_IST).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return str(ts)


def log_controller_reset(
    *,
    token: str | None = None,
    feature: str,
    previous_ts: float | None = None,
    current_ts: float | None = None,
    gap: float | None = None,
    gap_limit: float | None = None,
    reason: str = "row_gap",
) -> None:
    """Temporary debug hook — enable with EMA_RESET_DEBUG=1."""
    if not ema_reset_debug_enabled():
        return
    if gap is None and previous_ts is not None and current_ts is not None:
        gap = float(current_ts) - float(previous_ts)
    _EMA_RESET_LOG.warning(
        "[EMA RESET]\n"
        "Token        : %s\n"
        "Feature      : %s\n"
        "Previous ts  : %s\n"
        "Current ts   : %s\n"
        "Gap          : %s\n"
        "Gap limit    : %s\n"
        "Reason       : %s",
        token or "—",
        feature,
        _fmt_reset_ts(previous_ts),
        _fmt_reset_ts(current_ts),
        f"{gap:.1f}" if gap is not None else "—",
        f"{gap_limit:.1f}" if gap_limit is not None else "—",
        reason,
    )
