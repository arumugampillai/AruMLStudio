"""Benchmark utility stubs for Feature Intelligence Core (Sprint 0).

Measures execution time, memory usage, and database latency.
No feature-specific benchmarks yet.
"""

from __future__ import annotations

import sqlite3
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    elapsed_seconds: float
    peak_memory_bytes: int | None = None
    extra: dict[str, Any] | None = None


def measure_time(fn: Callable[[], T], *, label: str = "operation") -> tuple[T, BenchmarkResult]:
    """Run ``fn`` and record wall-clock elapsed time."""
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return value, BenchmarkResult(label=label, elapsed_seconds=elapsed)


def measure_memory(fn: Callable[[], T], *, label: str = "operation") -> tuple[T, BenchmarkResult]:
    """Run ``fn`` while tracking peak allocated memory via tracemalloc."""
    tracemalloc.start()
    try:
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, BenchmarkResult(
        label=label,
        elapsed_seconds=elapsed,
        peak_memory_bytes=peak,
    )


def measure_db_latency(
    db_path: Path,
    *,
    iterations: int = 10,
    label: str = "db_latency",
) -> BenchmarkResult:
    """Measure average SQLite round-trip latency for a trivial query."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[float] = []
    conn = sqlite3.connect(str(path))
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            conn.execute("SELECT 1").fetchone()
            samples.append(time.perf_counter() - start)
    finally:
        conn.close()
    avg = sum(samples) / len(samples)
    return BenchmarkResult(
        label=label,
        elapsed_seconds=avg,
        extra={"iterations": iterations, "samples": samples},
    )
