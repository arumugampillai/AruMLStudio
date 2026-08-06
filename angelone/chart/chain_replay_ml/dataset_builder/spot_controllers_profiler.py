"""Fine-grained profiler for SpotControllers.update() call paths."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

_lock = threading.Lock()
_stats: SpotControllersUpdateProfiler | None = None


@dataclass
class PathStat:
    calls: int = 0
    total_sec: float = 0.0

    def record(self, duration_sec: float) -> None:
        self.calls += 1
        self.total_sec += max(0.0, float(duration_sec))

    def merge(self, other: PathStat) -> None:
        self.calls += other.calls
        self.total_sec += other.total_sec

    def avg_ms(self) -> float:
        if self.calls <= 0:
            return 0.0
        return (self.total_sec / self.calls) * 1000.0


@dataclass
class SpotControllersUpdateProfiler:
    total_calls: int = 0
    duplicate_timestamp: PathStat = field(default_factory=PathStat)
    invalid_spot: PathStat = field(default_factory=PathStat)
    full_update: PathStat = field(default_factory=PathStat)
    ema: PathStat = field(default_factory=PathStat)
    rv: PathStat = field(default_factory=PathStat)
    momentum: PathStat = field(default_factory=PathStat)
    spot_hl: PathStat = field(default_factory=PathStat)

    def merge(self, other: SpotControllersUpdateProfiler) -> None:
        self.total_calls += other.total_calls
        self.duplicate_timestamp.merge(other.duplicate_timestamp)
        self.invalid_spot.merge(other.invalid_spot)
        self.full_update.merge(other.full_update)
        self.ema.merge(other.ema)
        self.rv.merge(other.rv)
        self.momentum.merge(other.momentum)
        self.spot_hl.merge(other.spot_hl)

    def _row(self, path: str, stat: PathStat) -> dict[str, Any]:
        return {
            "path": path,
            "calls": stat.calls,
            "total_sec": round(stat.total_sec, 6),
            "avg_ms": round(stat.avg_ms(), 4),
        }

    def summary_table(self) -> list[dict[str, Any]]:
        return [
            self._row("Duplicate timestamp", self.duplicate_timestamp),
            self._row("Invalid spot", self.invalid_spot),
            self._row("Full update", self.full_update),
            self._row("├─ EMA", self.ema),
            self._row("├─ RV", self.rv),
            self._row("├─ Momentum", self.momentum),
            self._row("└─ Spot HL", self.spot_hl),
        ]

    def _pct_of_full_update(self, stat: PathStat) -> float:
        full_sec = self.full_update.total_sec
        if full_sec <= 0:
            return 0.0
        return 100.0 * stat.total_sec / full_sec

    def full_update_breakdown_table(self) -> list[dict[str, Any]]:
        """Per-section time as % of full-update wall time (for parallelization targeting)."""
        full_sec = self.full_update.total_sec
        sections: list[tuple[str, PathStat]] = [
            ("EMA9–200", self.ema),
            ("RV", self.rv),
            ("Momentum", self.momentum),
            ("Spot HL", self.spot_hl),
        ]
        rows: list[dict[str, Any]] = []
        attributed = 0.0
        for name, stat in sections:
            attributed += stat.total_sec
            rows.append({
                "section": name,
                "calls": stat.calls,
                "total_sec": round(stat.total_sec, 6),
                "pct_of_full_update": round(self._pct_of_full_update(stat), 2),
                "avg_ms": round(stat.avg_ms(), 4),
            })
        overhead = max(0.0, full_sec - attributed)
        if overhead >= 1e-9:
            rows.append({
                "section": "Other (setup/overhead)",
                "calls": self.full_update.calls,
                "total_sec": round(overhead, 6),
                "pct_of_full_update": round(100.0 * overhead / full_sec, 2) if full_sec > 0 else 0.0,
                "avg_ms": round((overhead / self.full_update.calls * 1000.0) if self.full_update.calls else 0.0, 4),
            })
        return rows

    def case_classification(self) -> str:
        early_sec = self.duplicate_timestamp.total_sec + self.invalid_spot.total_sec
        full_sec = self.full_update.total_sec
        if full_sec <= 0 and early_sec <= 0:
            return "none"
        if full_sec >= early_sec * 1.5:
            return "case_1_full_updates_dominate"
        if early_sec >= full_sec * 1.5:
            return "case_2_duplicates_dominate"
        return "mixed"

    def interpretation(self) -> str:
        early_sec = self.duplicate_timestamp.total_sec + self.invalid_spot.total_sec
        full_sec = self.full_update.total_sec
        tracked = early_sec + full_sec
        if tracked <= 0:
            return "No SpotControllers.update() time recorded."

        case = self.case_classification()
        if case == "case_1_full_updates_dominate":
            dominant = max(
                [
                    ("EMA9–200", self.ema),
                    ("RV", self.rv),
                    ("Momentum", self.momentum),
                    ("Spot HL", self.spot_hl),
                ],
                key=lambda item: item[1].total_sec,
            )
            return (
                f"Case 1 — full updates dominate ({_fmt_short_sec(full_sec)} vs {_fmt_short_sec(early_sec)} early-return). "
                f"Deduplication works well; bottleneck is real computation. "
                f"Largest full-update section: {dominant[0]} ({self._pct_of_full_update(dominant[1]):.1f}% of full-update time). "
                f"Parallel workers may help."
            )
        if case == "case_2_duplicates_dominate":
            return (
                f"Case 2 — duplicate early returns dominate ({_fmt_short_sec(early_sec)} vs {_fmt_short_sec(full_sec)} full-update). "
                f"Repeated function calls are expensive — reduce call frequency before adding workers."
            )
        return (
            f"Mixed — early-return ({_fmt_short_sec(early_sec)}) and full-update ({_fmt_short_sec(full_sec)}) "
            f"time are comparable. Profile both caller dedupe and full-update sections."
        )

    def to_dict(self) -> dict[str, Any]:
        early_sec = self.duplicate_timestamp.total_sec + self.invalid_spot.total_sec
        early_calls = self.duplicate_timestamp.calls + self.invalid_spot.calls
        full_sec = self.full_update.total_sec
        tracked_sec = early_sec + full_sec
        return {
            "total_calls": self.total_calls,
            "early_returns_duplicate_timestamp": self.duplicate_timestamp.calls,
            "early_returns_invalid_spot": self.invalid_spot.calls,
            "full_updates_executed": self.full_update.calls,
            "total_time_early_return_sec": round(early_sec, 6),
            "total_time_full_update_sec": round(full_sec, 6),
            "avg_ms_early_return": round(
                (early_sec / early_calls * 1000.0) if early_calls else 0.0,
                4,
            ),
            "avg_ms_full_update": round(self.full_update.avg_ms(), 4),
            "pct_time_early_returns": round(
                100.0 * early_sec / tracked_sec if tracked_sec > 0 else 0.0,
                2,
            ),
            "pct_time_full_updates": round(
                100.0 * full_sec / tracked_sec if tracked_sec > 0 else 0.0,
                2,
            ),
            "interpretation": self.interpretation(),
            "case": self.case_classification(),
            "summary_table": self.summary_table(),
            "full_update_breakdown": self.full_update_breakdown_table(),
        }


def _fmt_short_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{sec / 60:.1f}m"


def spot_update_profiler_active() -> bool:
    from .build_profiler import get_profiler

    return get_profiler() is not None


def reset_spot_controllers_profiler() -> None:
    global _stats
    with _lock:
        _stats = SpotControllersUpdateProfiler() if spot_update_profiler_active() else None


def snapshot_spot_controllers_profiler() -> SpotControllersUpdateProfiler | None:
    with _lock:
        if _stats is None:
            return None
        return _stats


def _get_stats() -> SpotControllersUpdateProfiler | None:
    if not spot_update_profiler_active():
        return None
    global _stats
    with _lock:
        if _stats is None:
            _stats = SpotControllersUpdateProfiler()
        return _stats


def record_duplicate_timestamp_return(duration_sec: float) -> None:
    stats = _get_stats()
    if stats is None:
        return
    stats.total_calls += 1
    stats.duplicate_timestamp.record(duration_sec)


def record_invalid_spot_return(duration_sec: float) -> None:
    stats = _get_stats()
    if stats is None:
        return
    stats.total_calls += 1
    stats.invalid_spot.record(duration_sec)


def record_full_update(
    *,
    total_sec: float,
    ema_sec: float,
    rv_sec: float,
    momentum_sec: float | None = None,
    spot_hl_sec: float | None = None,
) -> None:
    stats = _get_stats()
    if stats is None:
        return
    stats.total_calls += 1
    stats.full_update.record(total_sec)
    stats.ema.record(ema_sec)
    stats.rv.record(rv_sec)
    if momentum_sec is not None:
        stats.momentum.record(momentum_sec)
    if spot_hl_sec is not None:
        stats.spot_hl.record(spot_hl_sec)


def time_section() -> float:
    return perf_counter()
