"""Benchmark harness for Phase 6.0 feature kernels + controller hotspots.

Run from ``angelone/chart``::

    python -m chain_replay_ml.performance.benchmark
    python -m chain_replay_ml.performance.benchmark --rows 200000 --out-dir ...
    python -m chain_replay_ml.performance.benchmark --check-regression
    python -m chain_replay_ml.performance.benchmark --dashboard

Uses synthetic price/IV streams (no production DB required).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Ensure angelone/chart is on path when run as __main__.
_CHART = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CHART not in sys.path:
    sys.path.insert(0, _CHART)


def _synthetic_prices(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.15, size=n)
    return np.cumsum(noise).astype(np.float64) + 100.0


def _synthetic_iv(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.12 + rng.normal(0.0, 0.01, size=n)).astype(np.float64)


def _bench_legacy_std_list_np(prices: np.ndarray, window: int = 20) -> dict[str, Any]:
    """Pre-Phase-6.0 path: ``np.std(list(deque), ddof=0)`` every ready row."""
    from collections import deque

    buf: deque[float] = deque(maxlen=window)
    t0 = time.perf_counter()
    last = 0.0
    for px in prices:
        buf.append(float(px))
        if len(buf) >= window:
            last = float(np.std(list(buf), ddof=0))
    elapsed = time.perf_counter() - t0
    n = int(prices.shape[0])
    return {
        "kernel": "legacy_std_list_np",
        "use_numba": False,
        "rows": n,
        "window": window,
        "total_sec": elapsed,
        "rows_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _bench_population_std(prices: np.ndarray, window: int, use_numba: bool) -> dict[str, Any]:
    from chain_replay_ml.performance import runtime
    from chain_replay_ml.performance.feature_kernels import population_std_numpy
    from chain_replay_ml.performance.feature_kernels import population_std_kernel

    runtime.set_numba_enabled(use_numba)
    n = prices.shape[0]
    t0 = time.perf_counter()
    last = 0.0
    for i in range(window, n + 1):
        buf = prices[i - window : i]
        if use_numba and runtime.numba_available():
            last = float(population_std_kernel(buf))
        else:
            last = population_std_numpy(buf)
    elapsed = time.perf_counter() - t0
    calls = max(n - window + 1, 0)
    return {
        "kernel": "population_std",
        "use_numba": use_numba and runtime.numba_available(),
        "calls": calls,
        "window": window,
        "total_sec": elapsed,
        "calls_per_sec": calls / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _bench_ema_series(prices: np.ndarray, period: int, use_numba: bool) -> dict[str, Any]:
    from chain_replay_ml.performance import runtime

    runtime.set_numba_enabled(use_numba)
    # Warm once outside timing for fair steady-state; compile recorded separately.
    _ = runtime.ema_series(prices[: min(64, prices.shape[0])], period)
    t0 = time.perf_counter()
    out = runtime.ema_series(prices, period)
    elapsed = time.perf_counter() - t0
    return {
        "kernel": "ema_series",
        "use_numba": use_numba and runtime.numba_available(),
        "rows": int(prices.shape[0]),
        "period": period,
        "total_sec": elapsed,
        "rows_per_sec": prices.shape[0] / elapsed if elapsed > 0 else 0.0,
        "last_value": float(out[-1]) if out.size else None,
    }


def _bench_iv_zscore(ivs: np.ndarray, window: int, use_numba: bool) -> dict[str, Any]:
    from chain_replay_ml.performance import runtime
    from chain_replay_ml.performance.feature_kernels import iv_zscore_kernel, iv_zscore_python

    runtime.set_numba_enabled(use_numba)
    n = ivs.shape[0]
    t0 = time.perf_counter()
    last = 0.0
    for i in range(window, n):
        priors = ivs[i - window : i]
        cur = float(ivs[i])
        if use_numba and runtime.numba_available():
            last = float(iv_zscore_kernel(priors, cur))
        else:
            last = iv_zscore_python(priors.tolist(), cur)
    elapsed = time.perf_counter() - t0
    calls = max(n - window, 0)
    return {
        "kernel": "iv_zscore",
        "use_numba": use_numba and runtime.numba_available(),
        "calls": calls,
        "window": window,
        "total_sec": elapsed,
        "calls_per_sec": calls / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _bench_std_controller(prices: np.ndarray, use_numba: bool) -> dict[str, Any]:
    """End-to-end StdController updates+value (wired path)."""
    from chain_replay_ml.dataset_builder.rolling_controllers import StdController
    from chain_replay_ml.performance import runtime

    runtime.set_numba_enabled(use_numba)
    ctrl = StdController(20)
    t0 = time.perf_counter()
    last = None
    for i, px in enumerate(prices):
        ctrl.update(float(px), ts=float(i))
        last = ctrl.value()
    elapsed = time.perf_counter() - t0
    n = int(prices.shape[0])
    return {
        "kernel": "StdController",
        "use_numba": use_numba and runtime.numba_available(),
        "rows": n,
        "total_sec": elapsed,
        "rows_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _bench_rv_controller(prices: np.ndarray, period: int, use_numba: bool) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.rolling_controllers import RvController
    from chain_replay_ml.performance import runtime

    runtime.set_numba_enabled(use_numba)
    ctrl = RvController(period)
    t0 = time.perf_counter()
    last = None
    for i, px in enumerate(prices):
        ctrl.update(float(px), ts=float(i))
        last = ctrl.value()
    elapsed = time.perf_counter() - t0
    n = int(prices.shape[0])
    return {
        "kernel": "RvController",
        "use_numba": use_numba and runtime.numba_available(),
        "rows": n,
        "period": period,
        "total_sec": elapsed,
        "rows_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _bench_iv_zscore_controller(ivs: np.ndarray, use_numba: bool) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.rolling_controllers import (
        IV_GRID_STEP_SEC,
        IvZscoreWindowController,
    )
    from chain_replay_ml.performance import runtime

    runtime.set_numba_enabled(use_numba)
    ctrl = IvZscoreWindowController(300.0, int(300.0 / IV_GRID_STEP_SEC))
    t0 = time.perf_counter()
    last = None
    for i, iv in enumerate(ivs):
        ctrl.update(float(iv), ts=float(i) * IV_GRID_STEP_SEC)
        last = ctrl.value()
    elapsed = time.perf_counter() - t0
    n = int(ivs.shape[0])
    return {
        "kernel": "IvZscoreWindowController",
        "use_numba": use_numba and runtime.numba_available(),
        "rows": n,
        "total_sec": elapsed,
        "rows_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "last_value": last,
    }


def _speedup(baseline_sec: float, optimized_sec: float) -> float | None:
    if optimized_sec <= 0 or baseline_sec <= 0:
        return None
    return baseline_sec / optimized_sec


def _default_benchmarks_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")


def _default_baseline_path() -> str:
    return os.path.join(_default_benchmarks_dir(), "baseline.json")


def load_baseline(path: str | None = None) -> dict[str, Any]:
    baseline_path = path or _default_baseline_path()
    with open(baseline_path, encoding="utf-8") as fh:
        return json.load(fh)


def extract_suite_rows_per_sec(report: dict[str, Any]) -> float:
    suite = report.get("suite") or {}
    optimized = suite.get("optimized") or {}
    rps = optimized.get("rows_per_sec")
    if rps is None:
        raise KeyError("suite.optimized.rows_per_sec missing from report")
    return float(rps)


def check_regression(
    report: dict[str, Any],
    *,
    baseline_path: str | None = None,
    max_drop_pct: float | None = None,
) -> dict[str, Any]:
    """Compare current suite rows/sec to a checked-in baseline.

    Fails (``ok=False``) when rows/sec drops by more than ``max_drop_pct``
    (default 20, or baseline JSON ``max_drop_pct``).
    """
    baseline = load_baseline(baseline_path)
    base_rps = float(baseline["rows_per_sec"])
    cur_rps = extract_suite_rows_per_sec(report)
    drop_limit = float(
        max_drop_pct if max_drop_pct is not None else baseline.get("max_drop_pct", 20.0)
    )
    if base_rps <= 0:
        raise ValueError("baseline rows_per_sec must be positive")
    drop_pct = max(0.0, (base_rps - cur_rps) / base_rps * 100.0)
    floor = base_rps * (1.0 - drop_limit / 100.0)
    ok = cur_rps >= floor
    return {
        "ok": ok,
        "baseline_path": baseline_path or _default_baseline_path(),
        "baseline_rows_per_sec": base_rps,
        "current_rows_per_sec": cur_rps,
        "drop_pct": round(drop_pct, 3),
        "max_drop_pct": drop_limit,
        "min_allowed_rows_per_sec": round(floor, 2),
        "message": (
            "PASS"
            if ok
            else (
                f"REGRESSION: rows/sec {cur_rps:.2f} is {drop_pct:.1f}% below baseline "
                f"{base_rps:.2f} (limit {drop_limit:.0f}%)"
            )
        ),
    }


def run_benchmark(
    *,
    rows: int = 100_000,
    features_assumed: int = 206,
    out_dir: str | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.performance import runtime
    from chain_replay_ml.performance.profiler import profile_callable, format_profile_markdown

    prices = _synthetic_prices(rows)
    ivs = _synthetic_iv(rows)

    compile_sec = runtime.warm_kernels(verbose=True) if runtime.numba_available() else {}

    kernel_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    # Kernel microbenchmarks
    legacy_std = _bench_legacy_std_list_np(prices, 20)
    std_numba = _bench_std_controller(prices, True)
    legacy_vs_std = {
        "name": "StdController_vs_legacy_list_np",
        "baseline": legacy_std,
        "optimized": std_numba,
        "speedup": _speedup(float(legacy_std["total_sec"]), float(std_numba["total_sec"])),
    }
    if legacy_vs_std["speedup"] is not None:
        legacy_vs_std["speedup"] = round(float(legacy_vs_std["speedup"]), 3)

    for name, base_fn, opt_fn in (
        (
            "population_std",
            lambda: _bench_population_std(prices, 20, False),
            lambda: _bench_population_std(prices, 20, True),
        ),
        (
            "ema_series",
            lambda: _bench_ema_series(prices, 20, False),
            lambda: _bench_ema_series(prices, 20, True),
        ),
        (
            "iv_zscore",
            lambda: _bench_iv_zscore(ivs, 100, False),
            lambda: _bench_iv_zscore(ivs, 100, True),
        ),
        (
            "StdController",
            lambda: _bench_std_controller(prices, False),
            lambda: _bench_std_controller(prices, True),
        ),
        (
            "RvController",
            lambda: _bench_rv_controller(prices, 30, False),
            lambda: _bench_rv_controller(prices, 30, True),
        ),
        (
            "IvZscoreWindowController",
            lambda: _bench_iv_zscore_controller(ivs, False),
            lambda: _bench_iv_zscore_controller(ivs, True),
        ),
    ):
        baseline = base_fn()
        optimized = opt_fn()
        kernel_pairs.append((name, baseline, optimized))

    per_kernel = []
    for name, base, opt in kernel_pairs:
        sp = _speedup(float(base["total_sec"]), float(opt["total_sec"]))
        per_kernel.append(
            {
                "name": name,
                "baseline": base,
                "optimized": opt,
                "speedup": None if sp is None else round(sp, 3),
            }
        )
    per_kernel.insert(0, legacy_vs_std)

    # Combined controller suite (baseline vs numba) under profiler
    def _suite(use_numba: bool) -> None:
        runtime.set_numba_enabled(use_numba)
        _bench_std_controller(prices, use_numba)
        _bench_rv_controller(prices, 30, use_numba)
        _bench_iv_zscore_controller(ivs, use_numba)
        runtime.ema_series(prices, 50)

    _, baseline_prof = profile_callable(
        lambda: _suite(False),
        label="controller_suite_baseline",
        rows=rows,
        features=features_assumed,
        use_cprofile=True,
        track_memory=True,
    )
    _, optimized_prof = profile_callable(
        lambda: _suite(True),
        label="controller_suite_numba",
        rows=rows,
        features=features_assumed,
        use_cprofile=True,
        track_memory=True,
    )
    runtime.set_numba_enabled(None)

    overall_speedup = _speedup(baseline_prof.total_sec, optimized_prof.total_sec)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "features_assumed": features_assumed,
        "numba_available": runtime.numba_available(),
        "numba_enabled": runtime.numba_enabled(),
        "python_fallback": runtime.performance_stats().get("python_fallback"),
        "compile_overhead_sec": {k: round(v, 6) for k, v in compile_sec.items()},
        "runtime_stats": runtime.performance_stats(),
        "limitations": [
            "Synthetic price/IV streams — not a full production day build.",
            "Does not exercise build_feature_raw_for_row end-to-end (requires tick DB).",
            "Throughput approximates controller hotspots inside the per-row feature path.",
            f"features/sec uses assumed feature count={features_assumed} (registry-sized).",
        ],
        "per_kernel": per_kernel,
        "suite": {
            "baseline": baseline_prof.to_dict(),
            "optimized": optimized_prof.to_dict(),
            "overall_speedup": None if overall_speedup is None else round(overall_speedup, 3),
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "benchmark_report.json")
        md_path = os.path.join(out_dir, "benchmark_report.md")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(_render_markdown(report, baseline_prof, optimized_prof))
        report["written"] = {"json": json_path, "markdown": md_path}

    return report


def _render_markdown(report: dict[str, Any], baseline_prof: Any, optimized_prof: Any) -> str:
    from chain_replay_ml.performance.profiler import format_profile_markdown

    lines = [
        "# Feature Engine Performance Benchmark (Phase 6.0)",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"- rows: **{report['rows']}**",
        f"- features_assumed: **{report['features_assumed']}**",
        f"- numba_available: **{report['numba_available']}**",
        f"- overall suite speedup: **{report['suite']['overall_speedup']}x**",
        "",
        "## Limitations",
        "",
    ]
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.extend(["", "## Compile overhead (first call)", ""])
    if report["compile_overhead_sec"]:
        for k, v in report["compile_overhead_sec"].items():
            lines.append(f"- `{k}`: {v:.4f}s")
    else:
        lines.append("- (Numba not available — no compile)")
    lines.extend(["", "## Per-kernel speedup", "", "| kernel | baseline_sec | numba_sec | speedup |", "|---|---:|---:|---:|"])
    for row in report["per_kernel"]:
        b = row["baseline"]["total_sec"]
        o = row["optimized"]["total_sec"]
        lines.append(f"| `{row['name']}` | {b:.4f} | {o:.4f} | {row['speedup']}x |")
    lines.extend(["", "## Suite profiles", ""])
    lines.append(format_profile_markdown(baseline_prof))
    lines.append("")
    lines.append(format_profile_markdown(optimized_prof))
    lines.extend(
        [
            "",
            "## How to re-run",
            "",
            "```bash",
            "cd angelone/chart",
            "python -m chain_replay_ml.performance.benchmark --rows 100000",
            "python -m chain_replay_ml.performance.benchmark --check-regression",
            "python -m chain_replay_ml.performance.dashboard",
            "```",
            "",
            "Disable Numba: `set ARUNEO_FEATURE_NUMBA=off` (Windows) / `export ARUNEO_FEATURE_NUMBA=off`.",
            "Auto Python fallback applies when Numba is missing or JIT fails (no env flag required).",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6.0 feature performance benchmark")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=206)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for JSON/MD reports (default: performance/benchmarks/)",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail if suite rows/sec drops >20%% vs benchmarks/baseline.json",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline JSON for --check-regression (default: benchmarks/baseline.json)",
    )
    parser.add_argument(
        "--max-drop-pct",
        type=float,
        default=None,
        help="Override baseline max_drop_pct (default 20)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Print Feature Engine Performance dashboard after the run",
    )
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Skip re-benchmark; check existing benchmark_report.json vs baseline",
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir or _default_benchmarks_dir()

    if args.regression_only:
        report_path = os.path.join(out_dir, "benchmark_report.json")
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
        result = check_regression(
            report, baseline_path=args.baseline, max_drop_pct=args.max_drop_pct
        )
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    report = run_benchmark(rows=args.rows, features_assumed=args.features, out_dir=out_dir)
    summary: dict[str, Any] = {
        "rows": report["rows"],
        "numba_available": report["numba_available"],
        "numba_enabled": report.get("numba_enabled"),
        "overall_speedup": report["suite"]["overall_speedup"],
        "written": report.get("written"),
        "per_kernel_speedup": {r["name"]: r["speedup"] for r in report["per_kernel"]},
    }

    exit_code = 0
    if args.check_regression:
        result = check_regression(
            report, baseline_path=args.baseline, max_drop_pct=args.max_drop_pct
        )
        summary["regression"] = result
        if not result["ok"]:
            exit_code = 1

    print(json.dumps(summary, indent=2))

    if args.dashboard or args.check_regression:
        from chain_replay_ml.performance.dashboard import (
            build_dashboard,
            format_dashboard_text,
            write_dashboard_json,
        )

        dash = build_dashboard(report=report)
        print()
        print(format_dashboard_text(dash))
        write_dashboard_json(dash, os.path.join(out_dir, "dashboard_report.json"))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
