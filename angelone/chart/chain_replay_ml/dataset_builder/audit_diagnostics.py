"""Deep audit diagnostics: sampling breakdown, missing features, rows math, targets."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.day_context import SourceSpec, load_day_context
from chain_replay_ml.dataset_builder.stages import LOOKBACK_START_SEC, build_sample_timestamps

_CHART_DIR = None  # set by auditor

# Longest feature lookback used in extended_features (15m).
FEATURE_LOOKBACK_SEC = 900.0
# Classify missing samples in first/last N grid slots per day.
EDGE_SAMPLE_SLOTS = 30


def _horizon_sec_from_column(col: str) -> int:
    if col.endswith("_1m"):
        return 60
    if col.endswith("_5m"):
        return 300
    m = re.search(r"_(\d+)s$", col)
    if m:
        return int(m.group(1))
    return 0


def _max_horizon_sec(target_columns: list[str]) -> int:
    if not target_columns:
        return 0
    return max(_horizon_sec_from_column(c) for c in target_columns)


def _reason_for_feature(feat: str) -> str:
    rules: list[tuple[str, str]] = [
        (r"body_pct_prev|range_pct_prev", "No previous candle"),
        (r"^oi_|oi_change|oi_wall|pinning", "Missing OI"),
        (r"roll_|bs_reiv|dgt_reiv|rows_since_roll", "Roll / re-anchor state"),
        (r"atm_straddle", "ATM straddle unavailable"),
        (r"iv_rank|iv_zscore|iv_change_15m|iv_change_5m", "Insufficient IV history"),
        (r"_15m|change_15m|slope_15m", "Long lookback window (15m)"),
        (r"_5m|change_5m|slope_5m|rv_5m|dist_.*_5m", "Long lookback window (5m)"),
        (r"chain_pcr|atm_pcr|max_call_oi|max_put_oi|build_wall", "Chain-wide aggregation"),
        (r"bid_ask", "Missing bid-ask spread ticks"),
        (r"minutes_since_open|is_first_hour", "Beginning of session"),
    ]
    for pattern, label in rules:
        if re.search(pattern, feat):
            return label
    return "Sparse / conditional feature"


def _reason_all_null_feature(feat: str, *, row_count: int, column_present: bool) -> str:
    if not column_present:
        return (
            f"100% NULL — column absent from parquet ({row_count:,} rows). "
            "Feature was expected but never written during dataset build."
        )
    if feat == "price_dist_from_cross_pct":
        return (
            f"100% NULL on all {row_count:,} rows. "
            "Usually caused by feature_plugins alias mapping to a key that extract_timeline_features does not emit."
        )
    sparse_reason = _reason_for_feature(feat)
    if sparse_reason != "Sparse / conditional feature":
        return (
            f"100% NULL on all {row_count:,} rows — unexpected for a conditional feature "
            f"(typical gap reason: {sparse_reason}). Likely not calculated or not mapped into parquet."
        )
    return (
        f"100% NULL on all {row_count:,} rows — no non-null values in parquet. "
        "Check extractor output key, feature_plugins alias, and builder pick_features_from_row mapping."
    )


def audit_all_null_features(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Fail when any expected feature is entirely NULL (or missing from parquet)."""
    n_rows = len(df)
    if n_rows == 0 or not feature_columns:
        return {"status": "warn", "count": 0, "features": [], "row_count": n_rows}

    failures: list[dict[str, Any]] = []
    for col in feature_columns:
        present = col in df.columns
        if not present:
            failures.append({
                "feature": col,
                "missing_count": n_rows,
                "row_count": n_rows,
                "pct": 100.0,
                "column_present": False,
                "reason": _reason_all_null_feature(col, row_count=n_rows, column_present=False),
            })
            continue
        n_missing = int(pd.isna(df[col]).sum())
        if n_missing >= n_rows:
            failures.append({
                "feature": col,
                "missing_count": n_missing,
                "row_count": n_rows,
                "pct": 100.0,
                "column_present": True,
                "reason": _reason_all_null_feature(col, row_count=n_rows, column_present=True),
            })

    return {
        "status": "fail" if failures else "pass",
        "label": "FAIL" if failures else "PASS",
        "count": len(failures),
        "row_count": n_rows,
        "features": failures,
    }


def _coverage_status(coverage_pct: float) -> str:
    if coverage_pct <= 0.0:
        return "fail"
    if coverage_pct >= 99.5:
        return "pass"
    return "warn"


def audit_feature_coverage(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Per-feature non-null coverage for the audit report table."""
    n_rows = len(df)
    rows: list[dict[str, Any]] = []

    for col in feature_columns:
        if n_rows == 0:
            missing = 0
            coverage = 0.0
        elif col not in df.columns:
            missing = n_rows
            coverage = 0.0
        else:
            missing = int(pd.isna(df[col]).sum())
            coverage = round(100.0 * (n_rows - missing) / n_rows, 1)

        status = _coverage_status(coverage)
        rows.append({
            "feature": col,
            "coverage_pct": coverage,
            "missing_count": missing,
            "row_count": n_rows,
            "status": status,
            "icon": "✅" if status == "pass" else ("❌" if status == "fail" else "⚠"),
        })

    order = {"fail": 0, "warn": 1, "pass": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["coverage_pct"], r["feature"]))

    fail_n = sum(1 for r in rows if r["status"] == "fail")
    warn_n = sum(1 for r in rows if r["status"] == "warn")
    pass_n = sum(1 for r in rows if r["status"] == "pass")

    return {
        "status": "fail" if fail_n else ("warn" if warn_n else "pass"),
        "row_count": n_rows,
        "features": rows,
        "fail_count": fail_n,
        "warn_count": warn_n,
        "pass_count": pass_n,
    }


def audit_missing_features(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Top missing features and aggregated reasons."""
    top: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    total_missing = 0

    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col]
        n_missing = int(pd.isna(series).sum())
        if n_missing <= 0:
            continue
        total_missing += n_missing
        reason = _reason_for_feature(col)
        reason_counts[reason] = reason_counts.get(reason, 0) + n_missing
        top.append({"feature": col, "missing_count": n_missing, "reason": reason})

    top.sort(key=lambda r: r["missing_count"], reverse=True)
    top = top[:15]

    # Beginning-of-session rows with any feature null
    session_missing = 0
    if "minutes_since_open" in df.columns and feature_columns:
        early = df["minutes_since_open"] <= 15.0
        if early.any():
            feat_cols = [c for c in feature_columns if c in df.columns]
            if feat_cols:
                session_missing = int(df.loc[early, feat_cols].map(
                    lambda v: v is None or (isinstance(v, float) and math.isnan(v))
                ).sum().sum())
    if session_missing > 0:
        reason_counts["Beginning of session"] = reason_counts.get("Beginning of session", 0) + session_missing

    reasons = [
        {"reason": k, "missing_count": v}
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "total_missing": total_missing,
        "top_missing": top,
        "reasons": reasons,
    }


def _classify_missing_timestamp(
    ts: float,
    *,
    grid_start: float,
    grid_end: float,
    open_ts: float,
    close_ts: float,
    step_sec: int,
    max_horizon: int,
    spot_ok: bool,
) -> str:
    if not spot_ok:
        return "missing_tick_timestamps"
    slot = int(round((ts - grid_start) / step_sec)) if step_sec > 0 else 0
    end_slot = int(round((grid_end - ts) / step_sec)) if step_sec > 0 else 0
    if slot < EDGE_SAMPLE_SLOTS:
        return "session_start_trimming"
    if end_slot < EDGE_SAMPLE_SLOTS:
        return "future_target_trimming"
    if ts - (open_ts + LOOKBACK_START_SEC) < FEATURE_LOOKBACK_SEC:
        return "lookback_trimming"
    return "unexpected_missing"


def audit_sampling_breakdown(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    step_sec: int,
    max_horizon_sec: int,
) -> dict[str, Any]:
    """Explain expected vs actual sample counts."""
    step_sec = max(int(step_sec or 10), 1)
    max_horizon_sec = int(max_horizon_sec or 0)

    days_meta = list(meta_doc.get("days") or [])
    if not days_meta:
        sources = meta_doc.get("sources") or []
        for s in sources:
            if s.get("status") == "loaded" or s.get("trading_day"):
                days_meta.append({
                    "trading_day": s.get("trading_day"),
                    "market": s.get("market"),
                    "expiry": s.get("expiry"),
                    "source_id": s.get("source_id", ""),
                })

    actual_by_day: dict[str, set[float]] = {}
    if "timestamp" in df.columns:
        if "trading_day" in df.columns:
            for day, grp in df.groupby("trading_day", dropna=False):
                actual_by_day[str(day)] = {float(x) for x in grp["timestamp"].unique()}
        else:
            actual_by_day["_all"] = {float(x) for x in df["timestamp"].unique()}

    raw_session_samples = 0
    raw_session_start_trim = 0
    raw_future_target_trim = 0
    session_start_trimming = 0
    future_target_trimming = 0
    lookback_trimming = 0
    missing_tick_timestamps = 0
    unexpected_missing = 0
    expected_samples = 0
    actual_samples = 0

    for day_info in days_meta:
        trading_day = str(day_info.get("trading_day") or "")
        if not trading_day:
            continue
        try:
            ctx = load_day_context(
                chart_dir,
                SourceSpec(
                    source_id=str(day_info.get("source_id") or trading_day),
                    trading_day=trading_day,
                    market=str(day_info.get("market") or "NIFTY"),
                    expiry=str(day_info.get("expiry") or ""),
                ),
            )
        except Exception:
            continue

        open_ts, close_ts = ctx.open_ts, ctx.close_ts
        grid_start = open_ts + LOOKBACK_START_SEC
        grid_end = close_ts - float(max_horizon_sec)

        full: list[float] = []
        t = open_ts
        while t <= close_ts + 0.001:
            full.append(t)
            t += step_sec
        raw_session_samples += len(full)
        raw_session_start_trim += sum(1 for ts in full if ts < grid_start)
        raw_future_target_trim += sum(1 for ts in full if ts > grid_end)

        expected_ts = build_sample_timestamps(ctx, step_sec=step_sec, max_horizon_sec=max_horizon_sec)
        expected_samples += len(expected_ts)
        actual_ts = actual_by_day.get(trading_day, set())
        actual_samples += len(actual_ts)

        for ts in expected_ts:
            if ts in actual_ts:
                continue
            spot = ctx.index_tl.ltp_rupees_at(ts)
            spot_ok = spot is not None and spot > 0
            bucket = _classify_missing_timestamp(
                ts,
                grid_start=grid_start,
                grid_end=grid_end,
                open_ts=open_ts,
                close_ts=close_ts,
                step_sec=step_sec,
                max_horizon=max_horizon_sec,
                spot_ok=spot_ok,
            )
            if bucket == "session_start_trimming":
                session_start_trimming += 1
            elif bucket == "future_target_trimming":
                future_target_trimming += 1
            elif bucket == "lookback_trimming":
                lookback_trimming += 1
            elif bucket == "missing_tick_timestamps":
                missing_tick_timestamps += 1
            else:
                unexpected_missing += 1

    if expected_samples == 0:
        expected_samples = int(meta_doc.get("sample_points_estimate") or 0)
    if actual_samples == 0 and "timestamp" in df.columns:
        if "trading_day" in df.columns:
            actual_samples = int(df.groupby(["trading_day", "timestamp"], dropna=False).ngroups)
        else:
            actual_samples = int(df["timestamp"].nunique())

    missing_samples = max(0, expected_samples - actual_samples)
    explained_missing = (
        session_start_trimming + lookback_trimming + future_target_trimming
        + missing_tick_timestamps + unexpected_missing
    )

    missing_breakdown = [
        {"id": "session_start_trimming", "label": "Session start trimming", "count": session_start_trimming, "expected": True},
        {"id": "lookback_trimming", "label": "Lookback trimming", "count": lookback_trimming, "expected": True},
        {"id": "future_target_trimming", "label": "Future target trimming", "count": future_target_trimming, "expected": True},
        {"id": "missing_tick_timestamps", "label": "Missing tick timestamps", "count": missing_tick_timestamps, "expected": True},
    ]

    intentional_missing = (
        session_start_trimming + lookback_trimming + future_target_trimming + missing_tick_timestamps
    )
    skip_explanation = {
        "requested_interval_sec": step_sec,
        "spot_tick_unavailable": missing_tick_timestamps,
        "intentional_skip_total": intentional_missing,
        "intentional_skip_note": (
            "These timestamps were skipped intentionally when spot ticks were unavailable "
            "or fell in session edge / lookback / future-target windows."
        ),
        "unexpected_skipped": unexpected_missing,
        "builder_ok": unexpected_missing == 0,
    }

    return {
        "interval_sec": step_sec,
        "max_horizon_sec": max_horizon_sec,
        "raw_session_samples": raw_session_samples,
        "raw_session_start_trim": raw_session_start_trim,
        "raw_future_target_trim": raw_future_target_trim,
        "expected_samples": expected_samples,
        "actual_samples": actual_samples,
        "missing_samples": missing_samples,
        "missing_breakdown": missing_breakdown,
        "unexpected_missing": unexpected_missing,
        "explained_missing": explained_missing,
        "duplicate_samples": 0,
        "skip_explanation": skip_explanation,
    }


def audit_rows_calculation(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    band: int,
    actual_samples: int,
) -> dict[str, Any]:
    """Mathematical proof of expected row count."""
    trading_days = 0
    if "trading_day" in df.columns:
        trading_days = int(df["trading_day"].nunique())
    elif meta_doc.get("days"):
        trading_days = len(meta_doc["days"])

    sample_timestamps = actual_samples
    if sample_timestamps == 0 and "timestamp" in df.columns:
        if "trading_day" in df.columns:
            sample_timestamps = int(df.groupby(["trading_day", "timestamp"], dropna=False).ngroups)
        else:
            sample_timestamps = int(df["timestamp"].nunique())

    rows_per_sample = (2 * int(band) + 1) * 2 if band else 0
    expected_rows = sample_timestamps * rows_per_sample
    actual_rows = len(df)
    delta = actual_rows - expected_rows
    status = "pass" if expected_rows == actual_rows else "warn"
    if expected_rows > 0 and abs(delta) <= max(1, int(expected_rows * 0.001)):
        status = "pass"

    return {
        "trading_days": trading_days,
        "sample_timestamps": sample_timestamps,
        "rows_per_sample": rows_per_sample,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "delta": delta,
        "status": status,
        "formula": f"{sample_timestamps} × {rows_per_sample} = {expected_rows:,}",
    }


def audit_targets_quality(df: pd.DataFrame, target_columns: list[str]) -> dict[str, Any]:
    """Per-target NaN / Inf / negative checks."""
    columns: list[dict[str, Any]] = []
    total_nan = 0
    total_inf = 0
    total_negative = 0

    for col in target_columns:
        present = col in df.columns
        nan_c = inf_c = neg_c = 0
        if present:
            series = pd.to_numeric(df[col], errors="coerce")
            nan_c = int(series.isna().sum())
            inf_c = int(np.isinf(series.to_numpy()).sum())
            neg_c = int((series < 0).sum())
        total_nan += nan_c
        total_inf += inf_c
        total_negative += neg_c
        columns.append({
            "column": col,
            "present": present,
            "nan": nan_c,
            "inf": inf_c,
            "negative": neg_c,
        })

    all_pass = total_nan == 0 and total_inf == 0 and total_negative == 0
    return {
        "columns": columns,
        "nan": total_nan,
        "inf": total_inf,
        "negative": total_negative,
        "status": "pass" if all_pass else "fail",
        "missing_values": total_nan,
    }
