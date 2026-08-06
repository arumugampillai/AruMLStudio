"""Lightweight profiler for feature-engine hotspots (synthetic or cProfile)."""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ProfileResult:
    label: str
    total_sec: float
    rows: int
    features: int
    rows_per_sec: float
    features_per_sec: float
    peak_memory_mib: float | None = None
    hottest: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total_sec": round(self.total_sec, 6),
            "rows": self.rows,
            "features": self.features,
            "rows_per_sec": round(self.rows_per_sec, 2),
            "features_per_sec": round(self.features_per_sec, 2),
            "peak_memory_mib": (
                None if self.peak_memory_mib is None else round(self.peak_memory_mib, 3)
            ),
            "hottest": self.hottest,
            "extra": self.extra,
        }


def _rows_features_rate(total_sec: float, rows: int, features: int) -> tuple[float, float]:
    if total_sec <= 0:
        return 0.0, 0.0
    # features = per-row feature count; throughput = rows * features / sec
    return rows / total_sec, (rows * features) / total_sec


def profile_callable(
    fn: Callable[[], Any],
    *,
    label: str,
    rows: int,
    features: int,
    use_cprofile: bool = True,
    track_memory: bool = True,
    top_n: int = 15,
) -> tuple[Any, ProfileResult]:
    """Run ``fn`` once under optional cProfile + tracemalloc."""
    pr: cProfile.Profile | None = None
    if use_cprofile:
        pr = cProfile.Profile()
        pr.enable()
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    total_sec = time.perf_counter() - t0
    peak_mib = None
    if track_memory:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak / (1024.0 * 1024.0)
    if pr is not None:
        pr.disable()
    hottest = _extract_hottest(pr, top_n=top_n) if pr is not None else []
    rps, fps = _rows_features_rate(total_sec, rows, features)
    return result, ProfileResult(
        label=label,
        total_sec=total_sec,
        rows=rows,
        features=features,
        rows_per_sec=rps,
        features_per_sec=fps,
        peak_memory_mib=peak_mib,
        hottest=hottest,
    )


def _extract_hottest(pr: cProfile.Profile, *, top_n: int) -> list[dict[str, Any]]:
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(top_n)
    rows: list[dict[str, Any]] = []
    # Prefer structured stats over parsing text.
    for (filename, line, func), (cc, _nc, tt, ct, _callers) in sorted(
        stats.stats.items(),
        key=lambda item: item[1][3],
        reverse=True,
    )[:top_n]:
        rows.append(
            {
                "function": func,
                "file": os.path.basename(filename),
                "line": line,
                "cumtime": round(ct, 6),
                "tottime": round(tt, 6),
                "calls": cc,
            }
        )
    return rows


def format_profile_markdown(result: ProfileResult) -> str:
    lines = [
        f"### {result.label}",
        "",
        f"- total_sec: **{result.total_sec:.4f}**",
        f"- rows: {result.rows} → **{result.rows_per_sec:,.0f} rows/sec**",
        f"- features: {result.features} → **{result.features_per_sec:,.0f} features/sec**",
    ]
    if result.peak_memory_mib is not None:
        lines.append(f"- peak_memory_mib: {result.peak_memory_mib:.3f}")
    if result.hottest:
        lines.extend(["", "| function | file | cumtime | tottime | calls |", "|---|---|---:|---:|---:|"])
        for h in result.hottest[:12]:
            lines.append(
                f"| `{h['function']}` | {h['file']}:{h['line']} | "
                f"{h['cumtime']:.4f} | {h['tottime']:.4f} | {h['calls']} |"
            )
    return "\n".join(lines)
