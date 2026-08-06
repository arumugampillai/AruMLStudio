"""Phase 5 performance benchmarks — streaming, scale, memory, FH regression."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import DayChunkRunner, ImmutableArtifactWriter, mint_artifact_id
from .fixed_horizon import compute_fixed_horizon_targets
from .prediction_source import CallablePredictionDaySource
from .triple_barrier import get_triple_barrier_strategy
from .types import LabelStrategyConfig


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    elapsed_sec: float
    rows: int
    days: int
    peak_memory_kib: float | None = None
    extras: dict[str, Any] | None = None


def _tb_day_rows(day: str, n: int, *, entry_ts: float = 1_000_000.0) -> list[dict[str, Any]]:
    """Synthetic prediction-grid day: each sample has a short explicit path."""
    close_ts = entry_ts + 20_000.0
    rows: list[dict[str, Any]] = []
    for i in range(n):
        ts = entry_ts + float(i) * 3.0
        rows.append(
            {
                "prediction_id": f"{day}-{i}",
                "trading_day": day,
                "token": "T1",
                "timestamp": ts,
                "current_ltp": 100.0,
                "session_close_ts": close_ts,
                "path": [
                    {"timestamp": ts + 3.0, "ltp": 111.0 if i % 3 == 0 else 101.0},
                    {"timestamp": ts + 6.0, "ltp": 94.0 if i % 3 == 1 else 102.0},
                ],
            }
        )
    return rows


def benchmark_streaming_day_chunks(
    *,
    root: str | Path,
    n_days: int = 20,
    rows_per_day: int = 200,
    suffix: str | None = None,
) -> BenchmarkResult:
    """Day → label → append; only one day loaded at a time via callable source."""
    days = [f"2024-01-{i:02d}" for i in range(1, n_days + 1)]
    loads: list[str] = []

    def load_fn(day: str) -> list[dict[str, Any]]:
        loads.append(day)
        # Ensure previous day is not retained by the source (streaming contract).
        return _tb_day_rows(day, rows_per_day)

    source = CallablePredictionDaySource(days=days, load_fn=load_fn)
    strategy = get_triple_barrier_strategy()
    artifact_id = mint_artifact_id("triple_barrier", suffix=suffix)
    writer = ImmutableArtifactWriter(root, artifact_id)

    tracemalloc.start()
    t0 = time.perf_counter()
    result = DayChunkRunner().run(
        strategy,
        source,
        LabelStrategyConfig(
            strategy_id="triple_barrier",
            version="1.0",
            params={
                "barrier_type": "points",
                "holding_seconds": 300,
                "tp_value": 10.0,
                "sl_value": 5.0,
            },
        ),
        writer,
    )
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if loads != days:
        raise AssertionError(f"expected day-at-a-time loads {days}, got {loads}")
    if result.run_meta.rows != n_days * rows_per_day:
        raise AssertionError("row count mismatch in streaming benchmark")

    return BenchmarkResult(
        name="streaming_day_chunks",
        elapsed_sec=elapsed,
        rows=result.run_meta.rows,
        days=n_days,
        peak_memory_kib=peak / 1024.0,
        extras={"loads": len(loads), "artifact_id": artifact_id},
    )


def benchmark_large_dataset(
    *,
    root: str | Path,
    n_days: int = 10,
    rows_per_day: int = 2_000,
) -> BenchmarkResult:
    """Larger day-chunk run — production-scale smoke without loading all days."""
    return benchmark_streaming_day_chunks(
        root=root, n_days=n_days, rows_per_day=rows_per_day, suffix="bench_large"
    )


def benchmark_memory_bound(
    *,
    root: str | Path,
    n_days: int = 30,
    rows_per_day: int = 100,
) -> BenchmarkResult:
    """Peak RSS during streaming should stay bounded (day-local load)."""
    return benchmark_streaming_day_chunks(
        root=root, n_days=n_days, rows_per_day=rows_per_day, suffix="bench_mem"
    )


class _Tl:
    def __init__(self, points: dict[float, float]) -> None:
        self._points = points

    def is_fresh_at(self, ts: float, max_stale: float) -> bool:
        return any(abs(t - ts) <= max_stale for t in self._points)

    def ltp_rupees_at(self, ts: float) -> float | None:
        best = None
        best_d = 1e18
        for t, px in self._points.items():
            d = abs(t - ts)
            if d < best_d:
                best_d = d
                best = px
        return best


def benchmark_fixed_horizon_regression(
    *,
    n_rows: int = 5_000,
) -> BenchmarkResult:
    """FH compute stays identical to legacy Stage 5 loop on a synthetic grid."""
    from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name

    horizons = [5, 10, 60, 300]
    base_ts = 1_000_000.0

    def legacy(ts: float, opt_tl: _Tl, max_stale: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for h in horizons:
            col = horizon_column_name(h)
            future_ts = ts + float(h)
            if opt_tl.is_fresh_at(future_ts, max_stale):
                out[col] = opt_tl.ltp_rupees_at(future_ts)
            else:
                out[col] = None
        return out

    t0 = time.perf_counter()
    mismatches = 0
    for i in range(n_rows):
        ts = base_ts + float(i)
        tl = _Tl({ts: 100.0, ts + 5.0: 101.0, ts + 10.0: 102.0})
        a = legacy(ts, tl, 10.0)
        b = compute_fixed_horizon_targets(
            ts=ts,
            opt_tl=tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        if a != b:
            mismatches += 1
    elapsed = time.perf_counter() - t0
    if mismatches:
        raise AssertionError(f"FH regression mismatches: {mismatches}/{n_rows}")
    return BenchmarkResult(
        name="fixed_horizon_regression",
        elapsed_sec=elapsed,
        rows=n_rows,
        days=0,
        extras={"mismatches": 0},
    )


def run_all_benchmarks(root: str | Path) -> list[BenchmarkResult]:
    return [
        benchmark_streaming_day_chunks(
            root=root, n_days=8, rows_per_day=150, suffix="bench_stream"
        ),
        benchmark_large_dataset(root=root, n_days=5, rows_per_day=800),
        benchmark_memory_bound(root=root, n_days=12, rows_per_day=120),
        benchmark_fixed_horizon_regression(n_rows=2_000),
    ]
