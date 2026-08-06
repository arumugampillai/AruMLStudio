"""Benchmark harness for CPU vs optional GPU Pearson correlation.

Generates synthetic numeric frames at 100k / 500k / 1M / 5M / 10M rows.
When RAPIDS is unavailable (typical on Windows), CPU timings are still written
and GPU fields are marked N/A.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import CorrelationEngine, is_gpu_available
from .types import normalize_preference

DEFAULT_ROW_COUNTS = (100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)
DEFAULT_N_FEATURES = 20

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _PKG_DIR / "benchmarks"


def _synthetic_frame(n_rows: int, n_features: int, *, seed: int = 42) -> Any:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(int(seed))
    # Correlated block + noise so Pearson is non-trivial.
    base = rng.standard_normal(n_rows)
    data: dict[str, Any] = {}
    for i in range(n_features):
        noise = rng.standard_normal(n_rows) * 0.1
        data[f"f{i:02d}"] = base * (0.5 + 0.02 * i) + noise
    # Inject a few NaNs to exercise pairwise missing handling.
    for i in range(min(3, n_features)):
        idx = rng.integers(0, n_rows, size=max(1, n_rows // 10_000))
        col = data[f"f{i:02d}"]
        col = col.copy()
        col[idx] = np.nan
        data[f"f{i:02d}"] = col
    return pd.DataFrame(data)


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        # Linux: ru_maxrss in KB; macOS: bytes
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss > 1e9:  # likely bytes (mac)
            return rss / (1024 * 1024)
        return rss / 1024.0
    except Exception:
        pass
    # Windows without psutil
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_proc = ctypes.windll.kernel32.GetCurrentProcess
        get_proc.restype = wintypes.HANDLE
        get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_mem.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if get_mem(get_proc(), ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024 * 1024)
    except Exception:
        pass
    return None


def run_one(
    n_rows: int,
    n_features: int,
    *,
    preference: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    engine = CorrelationEngine(preference=preference)
    gpu_ok = is_gpu_available()
    frame = _synthetic_frame(n_rows, n_features, seed=seed)

    # Always time CPU path for speedup denominator.
    cpu_engine = CorrelationEngine(preference="cpu")
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    cpu_res = cpu_engine.compute(frame)
    cpu_wall = max(time.perf_counter() - t0, 0.0)
    mem_after_cpu = _rss_mb()

    row: dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "gpu_available": bool(gpu_ok),
        "preference": normalize_preference(preference),
        "cpu_time_sec": float(cpu_res.timing.cpu_compute_sec or cpu_wall),
        "cpu_wall_sec": cpu_wall,
        "gpu_transfer_sec": None,
        "gpu_compute_sec": None,
        "gpu_total_sec": None,
        "backend_used": "cpu",
        "speedup": None,
        "memory_mb_before": mem_before,
        "memory_mb_after_cpu": mem_after_cpu,
        "memory_mb_after": mem_after_cpu,
        "fallback_reason": None,
        "status": "ok",
    }

    if not gpu_ok:
        row["status"] = "gpu_na"
        row["gpu_note"] = (
            "GPU N/A — RAPIDS cuDF/CuPy not importable "
            "(common on Windows; use Linux+CUDA for GPU path)"
        )
        return row

    # GPU (or auto→GPU) path
    t1 = time.perf_counter()
    gpu_res = engine.compute(frame, preference="gpu")
    gpu_wall = max(time.perf_counter() - t1, 0.0)
    mem_after = _rss_mb()
    row["backend_used"] = gpu_res.backend_used
    row["gpu_transfer_sec"] = gpu_res.timing.gpu_transfer_sec
    row["gpu_compute_sec"] = gpu_res.timing.gpu_compute_sec
    row["gpu_total_sec"] = float(gpu_res.timing.total_sec or gpu_wall)
    row["fallback_reason"] = gpu_res.timing.fallback_reason
    row["memory_mb_after"] = mem_after
    cpu_t = float(row["cpu_time_sec"] or 0.0)
    gpu_t = float(row["gpu_total_sec"] or 0.0)
    if gpu_res.backend_used == "gpu" and gpu_t > 0:
        row["speedup"] = cpu_t / gpu_t
        row["status"] = "ok"
    else:
        row["status"] = "gpu_fallback"
    return row


def run_benchmark(
    row_counts: tuple[int, ...] | list[int] = DEFAULT_ROW_COUNTS,
    *,
    n_features: int = DEFAULT_N_FEATURES,
    preference: str = "auto",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    gpu_ok = is_gpu_available()
    results: list[dict[str, Any]] = []
    for n in row_counts:
        print(f"Benchmarking n_rows={n:,} n_features={n_features} …")
        results.append(
            run_one(int(n), int(n_features), preference=preference)
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": os.name,
        "gpu_available": bool(gpu_ok),
        "n_features": int(n_features),
        "preference": normalize_preference(preference),
        "rows": results,
        "notes": [
            "CPU remains the default Analysis Lab backend.",
            "RAPIDS cuDF is typically Linux+CUDA only; Windows runs CPU-only.",
            "Speedup = cpu_time_sec / gpu_total_sec when GPU succeeds.",
        ],
    }

    json_path = out / "benchmark.json"
    md_path = out / "benchmark.md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md_path.write_text(_format_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Correlation Engine Benchmark (Phase 1)",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Platform: `{report.get('platform')}`",
        f"- GPU available: **{report.get('gpu_available')}**",
        f"- Features: `{report.get('n_features')}`",
        f"- Preference: `{report.get('preference')}`",
        "",
        "| n_rows | n_features | CPU (s) | GPU transfer (s) | GPU compute (s) | GPU total (s) | Speedup | Memory (MB) | Backend | Status |",
        "|-------:|----------:|--------:|-----------------:|----------------:|--------------:|--------:|------------:|---------|--------|",
    ]
    for r in report.get("rows") or []:
        def _fmt(v: Any) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        lines.append(
            "| {n_rows:,} | {n_features} | {cpu} | {xfer} | {gcomp} | {gtot} | {spd} | {mem} | {be} | {st} |".format(
                n_rows=int(r.get("n_rows") or 0),
                n_features=int(r.get("n_features") or 0),
                cpu=_fmt(r.get("cpu_time_sec")),
                xfer=_fmt(r.get("gpu_transfer_sec")),
                gcomp=_fmt(r.get("gpu_compute_sec")),
                gtot=_fmt(r.get("gpu_total_sec")),
                spd=_fmt(r.get("speedup")),
                mem=_fmt(r.get("memory_mb_after")),
                be=r.get("backend_used") or "cpu",
                st=r.get("status") or "",
            )
        )
    lines.append("")
    lines.append("## Notes")
    for n in report.get("notes") or []:
        lines.append(f"- {n}")
    for r in report.get("rows") or []:
        note = r.get("gpu_note") or r.get("fallback_reason")
        if note:
            lines.append(f"- n_rows={r.get('n_rows')}: {note}")
            break
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlation CPU/GPU benchmark")
    parser.add_argument(
        "--rows",
        type=str,
        default=",".join(str(x) for x in DEFAULT_ROW_COUNTS),
        help="Comma-separated row counts",
    )
    parser.add_argument("--features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument(
        "--preference",
        choices=("auto", "cpu", "gpu"),
        default="auto",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller row counts for smoke runs (10k,50k,100k)",
    )
    args = parser.parse_args(argv)
    if args.quick:
        counts = (10_000, 50_000, 100_000)
    else:
        counts = tuple(int(x.strip()) for x in str(args.rows).split(",") if x.strip())
    run_benchmark(
        counts,
        n_features=int(args.features),
        preference=str(args.preference),
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
