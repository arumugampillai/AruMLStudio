"""Helpers for Dataset Engine vs Pandas training-load measurement."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

LoadBackend = Literal["pandas", "dataset_engine", "auto"]


def resolve_training_load_backend(explicit: str | None = None) -> Literal["pandas", "dataset_engine"]:
    """Resolve which load path to try first.

    Env ``ARUNEO_DATASET_ENGINE`` (Phase 1 default: ``auto``):
      - ``off`` / ``0`` / ``pandas`` → pandas only
      - ``on`` / ``1`` / ``engine`` → prefer dataset_engine
      - ``auto`` / unset → dataset_engine when ``duckdb`` importable, else pandas

    Runtime Engine failures always fall back to pandas in
    ``dataset_loader.load_training_xy`` (``engine_fallback=True`` in telemetry).
    """
    raw = (explicit if explicit is not None else os.getenv("ARUNEO_DATASET_ENGINE", "auto")).strip().lower()
    if raw in ("off", "0", "false", "pandas", "pd"):
        return "pandas"
    if raw in ("on", "1", "true", "engine", "dataset_engine"):
        return "dataset_engine"
    # auto
    try:
        import duckdb  # noqa: F401

        return "dataset_engine"
    except ImportError:
        return "pandas"


def process_rss_mb() -> float | None:
    """Current process RSS in MiB (best effort)."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    # Windows fallback without psutil
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

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_mem = psapi.GetProcessMemoryInfo
        get_mem.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_mem.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if get_mem(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def measure_span(fn):
    """Run ``fn`` and return (result, metrics dict with load_time_sec + peak_rss_mb)."""
    rss0 = process_rss_mb()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    rss1 = process_rss_mb()
    peak = None
    if rss0 is not None and rss1 is not None:
        peak = max(rss0, rss1)
    elif rss1 is not None:
        peak = rss1
    return result, {
        "load_time_sec": round(elapsed, 6),
        "rss_before_mb": None if rss0 is None else round(rss0, 3),
        "rss_after_mb": None if rss1 is None else round(rss1, 3),
        "peak_rss_mb": None if peak is None else round(peak, 3),
    }
