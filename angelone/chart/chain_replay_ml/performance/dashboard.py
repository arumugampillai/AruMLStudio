"""CLI performance dashboard for the Feature Engine.

Reuse profiler/benchmark metrics and runtime counters::

    python -m chain_replay_ml.performance.dashboard
    python -m chain_replay_ml.performance.dashboard --report path/to/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

_CHART = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CHART not in sys.path:
    sys.path.insert(0, _CHART)


def _default_report_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "benchmark_report.json")


def _default_dashboard_json_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "dashboard_report.json")


def _numba_status_label(runtime_mod: Any, report: dict[str, Any] | None) -> str:
    stats = runtime_mod.performance_stats()
    if stats.get("python_fallback"):
        reason = stats.get("python_fallback_reason") or "auto"
        return f"Python fallback ({reason})"
    if not stats.get("numba_available"):
        return "Unavailable"
    if not stats.get("numba_enabled"):
        return "Disabled (ARUNEO_FEATURE_NUMBA=off)"
    if report is not None and report.get("numba_available") is False:
        return "Unavailable (last report)"
    return "Active (Numba)"


def build_dashboard(
    *,
    report_path: str | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble dashboard metrics from runtime stats + optional benchmark JSON."""
    from chain_replay_ml.performance import runtime

    loaded: dict[str, Any] | None = report
    path = report_path or _default_report_path()
    if loaded is None and path and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)

    stats = runtime.performance_stats()
    suite_opt: dict[str, Any] = {}
    if loaded:
        suite_opt = (loaded.get("suite") or {}).get("optimized") or {}

    compile_total = float(stats.get("compile_total_sec") or 0.0)
    if compile_total <= 0 and loaded:
        compile_total = float(sum((loaded.get("compile_overhead_sec") or {}).values()))

    dashboard = {
        "title": "Feature Engine Performance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_path": path if loaded is not None else None,
        "rows_per_sec": suite_opt.get("rows_per_sec"),
        "features_per_sec": suite_opt.get("features_per_sec"),
        "numba_status": _numba_status_label(runtime, loaded),
        "compile_time_sec": round(compile_total, 6),
        "kernel_hits": int(stats.get("kernel_hits") or 0),
        "cache_hits": int(stats.get("cache_hits") or 0),
        "runtime": stats,
        "suite_label": suite_opt.get("label"),
        "rows": (loaded or {}).get("rows") or suite_opt.get("rows"),
        "features_assumed": (loaded or {}).get("features_assumed") or suite_opt.get("features"),
    }
    return dashboard


def format_dashboard_text(dashboard: dict[str, Any]) -> str:
    def _fmt_rate(v: Any) -> str:
        if v is None:
            return "n/a (run benchmark)"
        try:
            return f"{float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "Feature Engine Performance",
        f"Rows/sec          {_fmt_rate(dashboard.get('rows_per_sec'))}",
        f"Features/sec      {_fmt_rate(dashboard.get('features_per_sec'))}",
        f"Numba Status      {dashboard.get('numba_status')}",
        f"Compile Time      {dashboard.get('compile_time_sec'):.4f}s",
        f"Kernel Hits       {dashboard.get('kernel_hits')}",
        f"Cache Hits        {dashboard.get('cache_hits')}",
    ]
    return "\n".join(lines)


def write_dashboard_json(dashboard: dict[str, Any], path: str | None = None) -> str:
    out = path or _default_dashboard_json_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dashboard, fh, indent=2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature Engine performance dashboard")
    parser.add_argument(
        "--report",
        default=None,
        help="Path to benchmark_report.json (default: performance/benchmarks/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write dashboard JSON (default: benchmarks/dashboard_report.json)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print only; do not write dashboard_report.json",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="Warm kernels before collecting runtime counters",
    )
    args = parser.parse_args(argv)

    if args.warm:
        from chain_replay_ml.performance import warm_kernels

        warm_kernels(verbose=True)

    dash = build_dashboard(report_path=args.report)
    print(format_dashboard_text(dash))
    if not args.no_write:
        written = write_dashboard_json(dash, args.out)
        print(f"\nWrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
