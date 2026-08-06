"""Profiler for feature-policy readiness enforcement during dataset builds."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ReadinessProfilerStats:
    enforce_wall_sec: float = 0.0
    validate_wall_sec: float = 0.0
    function_times_sec: dict[str, float] = field(default_factory=dict)
    function_calls: dict[str, int] = field(default_factory=dict)
    gap_max_sec: float | None = None
    feature_count: int = 0
    row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        rows = self.by_function_table()
        return {
            "enforce_wall_sec": round(self.enforce_wall_sec, 6),
            "validate_wall_sec": round(self.validate_wall_sec, 6),
            "total_wall_sec": round(self.enforce_wall_sec + self.validate_wall_sec, 6),
            "gap_max_sec": self.gap_max_sec,
            "feature_count": self.feature_count,
            "row_count": self.row_count,
            "by_function": rows,
        }

    def by_function_table(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        keys = set(self.function_times_sec) | set(self.function_calls)
        rows: list[dict[str, Any]] = []
        for key in keys:
            sec = float(self.function_times_sec.get(key, 0.0))
            calls = int(self.function_calls.get(key, 0))
            if sec <= 0 and calls == 0:
                continue
            avg_us = (sec / calls * 1_000_000) if calls else 0.0
            rows.append({
                "function": key,
                "time_sec": round(sec, 6),
                "calls": calls,
                "avg_time_us": round(avg_us, 3),
            })
        rows.sort(key=lambda r: float(r["time_sec"]), reverse=True)
        if limit is not None:
            return rows[:limit]
        return rows


_active: ContextVar[ReadinessProfilerStats | None] = ContextVar("readiness_profiler", default=None)


def profiler_active() -> bool:
    return _active.get() is not None


def get_profiler() -> ReadinessProfilerStats | None:
    return _active.get()


def start_readiness_profiler(
    *,
    gap_max_sec: float | None = None,
    feature_count: int = 0,
    row_count: int = 0,
) -> ReadinessProfilerStats:
    stats = ReadinessProfilerStats(
        gap_max_sec=gap_max_sec,
        feature_count=feature_count,
        row_count=row_count,
    )
    _active.set(stats)
    return stats


def stop_readiness_profiler() -> ReadinessProfilerStats | None:
    stats = _active.get()
    _active.set(None)
    return stats


@contextmanager
def readiness_profile_block(name: str) -> Iterator[None]:
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


def record_readiness_call(name: str, *, elapsed_sec: float = 0.0) -> None:
    stats = _active.get()
    if stats is None:
        return
    if elapsed_sec:
        stats.function_times_sec[name] = stats.function_times_sec.get(name, 0.0) + elapsed_sec
    stats.function_calls[name] = stats.function_calls.get(name, 0) + 1


def compare_readiness_profiles(
    off_stats: ReadinessProfilerStats | dict[str, Any],
    on_stats: ReadinessProfilerStats | dict[str, Any],
) -> dict[str, Any]:
    """Diff readiness profiler tables (gap OFF vs ON)."""
    off_doc = off_stats.to_dict() if isinstance(off_stats, ReadinessProfilerStats) else dict(off_stats)
    on_doc = on_stats.to_dict() if isinstance(on_stats, ReadinessProfilerStats) else dict(on_stats)
    off_map = {str(r["function"]): r for r in off_doc.get("by_function") or []}
    on_map = {str(r["function"]): r for r in on_doc.get("by_function") or []}
    keys = set(off_map) | set(on_map)
    rows: list[dict[str, Any]] = []
    for key in keys:
        off = off_map.get(key, {})
        on = on_map.get(key, {})
        off_sec = float(off.get("time_sec", 0.0))
        on_sec = float(on.get("time_sec", 0.0))
        delta = on_sec - off_sec
        if abs(delta) < 1e-6 and int(off.get("calls", 0)) == int(on.get("calls", 0)):
            continue
        rows.append({
            "function": key,
            "off_sec": round(off_sec, 6),
            "on_sec": round(on_sec, 6),
            "delta_sec": round(delta, 6),
            "calls_off": int(off.get("calls", 0)),
            "calls_on": int(on.get("calls", 0)),
        })
    rows.sort(key=lambda r: float(r["delta_sec"]), reverse=True)
    return {
        "off_total_sec": round(float(off_doc.get("total_wall_sec", 0.0)), 6),
        "on_total_sec": round(float(on_doc.get("total_wall_sec", 0.0)), 6),
        "delta_total_sec": round(
            float(on_doc.get("total_wall_sec", 0.0)) - float(off_doc.get("total_wall_sec", 0.0)),
            6,
        ),
        "by_function": rows,
    }
