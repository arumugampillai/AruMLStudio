"""Dataset Engine Qualification Test — stress / edge-case parity + telemetry.

Generates synthetic Analysis-like parquets that vary in size and column
characteristics, then compares Engine vs Pandas materialization and records
scalability metrics.

Cases cover:
  - row scales (100k / 1M / optional 5M+)
  - feature counts (50 / 200 / 500+)
  - missing values, constant columns, high-cardinality
  - mixed numeric dtypes (float32/64, int32/64)

Qualification metrics (per case):
  wall_time_sec, peak_rss_mb, rows_per_sec, columns_per_sec,
  partitions_scanned / partitions_pruned, fallback_count, engine_exceptions

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.dataset_engine.stress_test
  python -m chain_replay_ml.dataset_engine.stress_test --include-5m
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_LOG = logging.getLogger("dataset_engine_stress")


def build_stress_frame(
    *,
    n_rows: int,
    n_features: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a frame with missing / constant / high-card / mixed dtypes."""
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {
        "trading_day": np.where(rng.random(n_rows) < 0.5, "2026-07-23", "2026-07-24"),
        "token": (rng.integers(1, max(n_rows // 10, 2), size=n_rows)).astype(np.int64),
        "timestamp": np.arange(n_rows, dtype=np.int64),
        "ltp": rng.uniform(5.0, 200.0, size=n_rows).astype(np.float64),
    }
    data["const_f32"] = np.full(n_rows, 1.5, dtype=np.float32)
    data["const_i32"] = np.full(n_rows, 7, dtype=np.int32)
    data["high_card"] = rng.integers(0, max(n_rows, 2), size=n_rows, dtype=np.int64)

    n_feat = max(int(n_features), 4)
    for i in range(n_feat):
        if i % 4 == 0:
            col = rng.normal(size=n_rows).astype(np.float32)
            col[rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)] = np.nan
            data[f"f32_{i}"] = col
        elif i % 4 == 1:
            data[f"f64_{i}"] = rng.normal(size=n_rows).astype(np.float64)
        elif i % 4 == 2:
            data[f"i32_{i}"] = rng.integers(-1000, 1000, size=n_rows, dtype=np.int32)
        else:
            data[f"i64_{i}"] = rng.integers(0, 10_000_000, size=n_rows, dtype=np.int64)

    return pd.DataFrame(data)


def write_stress_parquet(
    path: Path,
    *,
    n_rows: int,
    n_features: int,
    seed: int = 42,
    chunk_rows: int = 500_000,
) -> None:
    """Write stress parquet, chunking when large to avoid build OOMs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if n_rows <= chunk_rows:
        build_stress_frame(n_rows=n_rows, n_features=n_features, seed=seed).to_parquet(
            path, index=False
        )
        return

    writer: Any = None
    written = 0
    chunk_i = 0
    try:
        while written < n_rows:
            n = min(chunk_rows, n_rows - written)
            chunk = build_stress_frame(
                n_rows=n, n_features=n_features, seed=seed + chunk_i
            )
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            del chunk
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            written += n
            chunk_i += 1
            del table
    finally:
        if writer is not None:
            writer.close()


def _read_columns_from_names(schema_names: list[str], n_features: int) -> list[str]:
    cols = ["trading_day", "token", "ltp", "const_f32", "const_i32", "high_card"]
    feat_cols = [c for c in schema_names if c.startswith(("f32_", "f64_", "i32_", "i64_"))]
    take = min(len(feat_cols), max(32, min(n_features, 256)))
    cols.extend(feat_cols[:take])
    return [c for c in cols if c in schema_names]


def _read_columns(df: pd.DataFrame, n_features: int) -> list[str]:
    """Identity + constants + enough feature cols to exercise wide tables."""
    return _read_columns_from_names([str(c) for c in df.columns], n_features)


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, *, rtol: float, atol: float) -> dict[str, Any]:
    if list(a.columns) != list(b.columns):
        return {
            "ok": False,
            "reason": "columns_differ",
            "a_cols": list(a.columns),
            "b_cols": list(b.columns),
        }
    if len(a) != len(b):
        return {"ok": False, "reason": "row_count", "a": len(a), "b": len(b)}
    max_diff = 0.0
    for c in a.columns:
        sa, sb = a[c], b[c]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            xa = pd.to_numeric(sa, errors="coerce").to_numpy(dtype=float)
            xb = pd.to_numeric(sb, errors="coerce").to_numpy(dtype=float)
            if not np.allclose(xa, xb, rtol=rtol, atol=atol, equal_nan=True):
                diff = float(np.nanmax(np.abs(xa - xb)))
                return {"ok": False, "reason": f"values:{c}", "max_abs_diff": diff}
            diff = float(np.nanmax(np.abs(xa - xb))) if len(xa) else 0.0
            if math.isfinite(diff):
                max_diff = max(max_diff, diff)
        else:
            if not sa.astype(str).equals(sb.astype(str)):
                return {"ok": False, "reason": f"values:{c}"}
    return {"ok": True, "max_abs_diff": max_diff}


def _throughput(rows: int, cols: int, wall_sec: float) -> dict[str, float | None]:
    if wall_sec <= 0:
        return {"rows_per_sec": None, "columns_per_sec": None, "cells_per_sec": None}
    return {
        "rows_per_sec": round(rows / wall_sec, 1),
        "columns_per_sec": round(cols / wall_sec, 1),
        "cells_per_sec": round((rows * cols) / wall_sec, 1),
    }


def run_case(
    *,
    n_rows: int,
    n_features: int,
    max_rows: int | None = None,
    work_dir: str | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_engine import SampleSpec, query_dataset
    from chain_replay_ml.training.load_backend import measure_span, process_rss_mb

    own_tmp = work_dir is None
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="de_stress_"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"stress_r{n_rows}_f{n_features}.parquet"

    wall0 = time.perf_counter()
    rss_case0 = process_rss_mb()

    t_build = time.perf_counter()
    try:
        write_stress_parquet(path, n_rows=n_rows, n_features=n_features)
    except MemoryError as exc:
        return {
            "n_rows": n_rows,
            "n_features": n_features,
            "max_rows": max_rows,
            "ok": False,
            "parity": {"ok": False, "reason": "build_oom", "detail": str(exc)},
            "fallback_count": 0,
            "engine_exceptions": 0,
            "wall_time_sec": round(time.perf_counter() - wall0, 3),
            "peak_rss_mb": process_rss_mb(),
            "error": f"MemoryError during build: {exc}",
        }
    build_sec = time.perf_counter() - t_build
    # Column list from schema (avoid holding full frame).
    import pyarrow.parquet as pq

    schema_names = [str(n) for n in pq.read_schema(path).names]
    probe = pd.DataFrame(columns=schema_names)
    cols = _read_columns(probe, n_features)
    # Ensure requested cols exist
    cols = [c for c in cols if c in schema_names]

    sample = SampleSpec(max_rows=max_rows, seed=42) if max_rows is not None else None
    engine_exception: str | None = None
    fallback_count = 0
    engine_stats: dict[str, Any] = {}
    eng: pd.DataFrame | None = None

    def _load_engine():
        return query_dataset(
            str(path),
            columns=cols,
            sample=sample,
            parquet_path=str(path),
        )

    try:
        result, eng_span = measure_span(_load_engine)
        eng = result.table.to_pandas()
        st = result.stats
        engine_stats = {
            "backend": st.backend,
            "partitions_scanned": st.partitions_scanned,
            "partitions_pruned": st.partitions_pruned,
            "engine_execution_time_sec": st.execution_time_sec,
            "rows_returned_stats": st.rows_returned,
            **eng_span,
        }
    except Exception as exc:
        engine_exception = f"{exc.__class__.__name__}: {exc}"
        fallback_count = 1
        _LOG.warning("engine exception → pandas reference only: %s", engine_exception)
        eng_span = {"load_time_sec": None, "peak_rss_mb": None}

    def _load_pandas():
        frame = pd.read_parquet(path, columns=cols)
        if max_rows is not None and len(frame) > int(max_rows):
            frame = frame.head(int(max_rows))
        return frame

    pd_df, pd_span = measure_span(_load_pandas)

    parity: dict[str, Any]
    if eng is None:
        parity = {"ok": False, "reason": "engine_exception", "detail": engine_exception}
        eng = pd_df.iloc[0:0].copy()
        eng_sec = float(eng_span.get("load_time_sec") or 0.0)
    else:
        eng = eng[cols]
        pd_df = pd_df[cols]
        parity = _frames_equal(eng, pd_df, rtol=rtol, atol=atol)
        eng_sec = float(engine_stats.get("load_time_sec") or 0.0)

    pd_sec = float(pd_span.get("load_time_sec") or 0.0)
    rows_ret = int(len(eng)) if eng is not None else 0
    cols_ret = int(len(cols))
    wall_sec = time.perf_counter() - wall0
    rss_case1 = process_rss_mb()
    peak_case = None
    if rss_case0 is not None and rss_case1 is not None:
        peak_case = max(float(rss_case0), float(rss_case1))
        for key in ("peak_rss_mb",):
            for span in (engine_stats, pd_span):
                v = span.get(key)
                if v is not None:
                    peak_case = max(peak_case, float(v))

    report = {
        "n_rows": n_rows,
        "n_features": n_features,
        "max_rows": max_rows,
        "parquet": str(path),
        "build_sec": round(build_sec, 3),
        # Qualification metrics
        "wall_time_sec": round(wall_sec, 3),
        "peak_rss_mb": None if peak_case is None else round(peak_case, 3),
        "engine_load_sec": round(eng_sec, 3) if eng_sec else None,
        "pandas_load_sec": round(pd_sec, 3),
        "rows_returned": rows_ret,
        "columns_returned": cols_ret,
        **_throughput(rows_ret, cols_ret, eng_sec if eng_sec else wall_sec),
        "partitions_scanned": engine_stats.get("partitions_scanned"),
        "partitions_pruned": engine_stats.get("partitions_pruned"),
        "fallback_count": fallback_count,
        "engine_exceptions": 0 if engine_exception is None else 1,
        "engine_exception": engine_exception,
        "engine_stats": engine_stats,
        "pandas_span": pd_span,
        "parity": parity,
        "ok": bool(parity.get("ok")) and engine_exception is None and fallback_count == 0,
        "kept_temp": not own_tmp,
    }
    if own_tmp:
        try:
            path.unlink(missing_ok=True)
            root.rmdir()
        except Exception:
            pass
    return report


# Default qualification matrix (without 5M).
DEFAULT_CASES: list[tuple[int, int]] = [
    (100_000, 50),
    (100_000, 200),
    (100_000, 500),
    (1_000_000, 50),
    (1_000_000, 200),
    (1_000_000, 500),  # maturity gate
]


def run_suite(
    cases: list[tuple[int, int]] | None = None,
    *,
    include_500_features: bool = True,
    include_5m: bool = False,
) -> dict[str, Any]:
    if cases is not None:
        selected = list(cases)
    else:
        selected = list(DEFAULT_CASES)
        if not include_500_features:
            selected = [(r, f) for r, f in selected if f < 500]
        if include_5m:
            # 5M×50 only — wider 5M frames are build-time memory heavy even chunked.
            selected.append((5_000_000, 50))

    results: list[dict[str, Any]] = []
    for n_rows, n_feat in selected:
        _LOG.info("case rows=%s features=%s", n_rows, n_feat)
        # Cap materialization for 5M+ so machines stay usable; parity vs head().
        max_rows = None if n_rows <= 1_000_000 else 1_000_000
        results.append(run_case(n_rows=n_rows, n_features=n_feat, max_rows=max_rows))

    fallback_total = sum(int(r.get("fallback_count") or 0) for r in results)
    exception_total = sum(int(r.get("engine_exceptions") or 0) for r in results)
    summary_metrics = {
        "fallback_count_total": fallback_total,
        "engine_exceptions_total": exception_total,
        "wall_time_sec_total": round(sum(float(r.get("wall_time_sec") or 0) for r in results), 3),
        "peak_rss_mb_max": max(
            (float(r["peak_rss_mb"]) for r in results if r.get("peak_rss_mb") is not None),
            default=None,
        ),
        "maturity_gate_1m_x_500": next(
            (
                {
                    "ok": r.get("ok"),
                    "wall_time_sec": r.get("wall_time_sec"),
                    "peak_rss_mb": r.get("peak_rss_mb"),
                    "rows_per_sec": r.get("rows_per_sec"),
                    "columns_per_sec": r.get("columns_per_sec"),
                    "partitions_scanned": r.get("partitions_scanned"),
                    "partitions_pruned": r.get("partitions_pruned"),
                    "fallback_count": r.get("fallback_count"),
                    "engine_exceptions": r.get("engine_exceptions"),
                    "parity": r.get("parity"),
                }
                for r in results
                if r.get("n_rows") == 1_000_000 and r.get("n_features") == 500
            ),
            None,
        ),
    }
    ok = (
        all(r.get("ok") for r in results)
        and fallback_total == 0
        and exception_total == 0
    )
    return {
        "name": "Dataset Engine Qualification Test",
        "ok": ok,
        "n_cases": len(results),
        "passed": sum(1 for r in results if r.get("ok")),
        "failed": [r for r in results if not r.get("ok")],
        "summary_metrics": summary_metrics,
        "cases": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("\n=== Dataset Engine Qualification Test ===")
    print(f"ok={report.get('ok')}  passed={report.get('passed')}/{report.get('n_cases')}")
    sm = report.get("summary_metrics") or {}
    print(
        f"fallbacks={sm.get('fallback_count_total')}  "
        f"exceptions={sm.get('engine_exceptions_total')}  "
        f"wall_total_sec={sm.get('wall_time_sec_total')}  "
        f"peak_rss_mb_max={sm.get('peak_rss_mb_max')}"
    )
    gate = sm.get("maturity_gate_1m_x_500")
    if gate:
        print("maturity gate 1M×500:", json.dumps(gate, default=str))
    print(
        f"{'rows':>10} {'feats':>6} {'ok':>5} {'wall_s':>8} {'eng_s':>8} "
        f"{'rss_mb':>8} {'rows/s':>12} {'cols/s':>10} {'part':>6}"
    )
    for r in report.get("cases") or []:
        print(
            f"{r.get('n_rows'):>10} {r.get('n_features'):>6} "
            f"{'PASS' if r.get('ok') else 'FAIL':>5} "
            f"{float(r.get('wall_time_sec') or 0):>8.2f} "
            f"{float(r.get('engine_load_sec') or 0):>8.3f} "
            f"{float(r.get('peak_rss_mb') or 0):>8.0f} "
            f"{float(r.get('rows_per_sec') or 0):>12.0f} "
            f"{float(r.get('columns_per_sec') or 0):>10.0f} "
            f"{str(r.get('partitions_scanned')):>6}"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Dataset Engine Qualification Test (stress + metrics)"
    )
    parser.add_argument("--rows", default="", help="Comma-separated row counts")
    parser.add_argument("--features", default="", help="Comma-separated feature counts")
    parser.add_argument("--include-5m", action="store_true")
    parser.add_argument("--no-500", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--quiet", action="store_true", help="JSON only")
    args = parser.parse_args(argv)

    cases = None
    if args.rows.strip() and args.features.strip():
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
        feats = [int(x) for x in args.features.split(",") if x.strip()]
        cases = [(r, f) for r in rows for f in feats]

    report = run_suite(
        cases,
        include_500_features=not args.no_500,
        include_5m=bool(args.include_5m),
    )
    if not args.quiet:
        _print_summary(report)
    text = json.dumps(report, indent=2, default=str)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        _LOG.info("Wrote %s", out)
    if args.quiet:
        print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
