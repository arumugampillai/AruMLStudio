"""Extended audit validations — independent of dataset builder code paths."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec, load_day_context
from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, find_atm_strike, normalize_index_name
from chain_replay_ml.ticks import TickTimeline

from .audit_progress import AuditStageTracker
from .lookback_policy import (
    detect_configuration_mismatch,
    greek_at_lookback,
    lookback_policy,
    policy_alignment_view,
    policy_label,
    read_validator_policy,
    read_dataset_configuration,
    normalize_policy_doc,
)

IST = ZoneInfo("Asia/Kolkata")

from .validation_rules import (
    distribution_features,
    independent_checks,
    validation_identity_material,
)


def _corr_status(corr: float | None, *, expect_positive: bool) -> str:
    if corr is None or math.isnan(corr):
        return "info"
    if expect_positive:
        if corr >= 0.15:
            return "pass"
        if corr >= 0.0:
            return "warn"
        return "fail"
    if corr <= -0.15:
        return "pass"
    if corr <= 0.0:
        return "warn"
    return "fail"


def _prepare_correlation_tests(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Runnable correlation test definitions for the current dataframe."""
    tests: list[dict[str, Any]] = []
    if "spot" in df.columns and "delta" in df.columns and "option_type" in df.columns:
        tests.append({
            "id": "ce_spot_delta",
            "check": "Spot ↑ → Call Delta ↑",
            "pair_label": "Spot vs Delta (CE)",
            "column_x": "spot",
            "column_y": "delta",
            "filter_col": "option_type",
            "filter_val": "CE",
            "expect_positive": True,
            "threshold_label": "r ≥ 0.15",
            "min_rows": 30,
        })
        tests.append({
            "id": "pe_spot_delta",
            "check": "Spot ↑ → Put Delta ↓",
            "pair_label": "Spot vs Delta (PE)",
            "column_x": "spot",
            "column_y": "delta",
            "filter_col": "option_type",
            "filter_val": "PE",
            "expect_positive": False,
            "threshold_label": "r ≤ −0.15",
            "min_rows": 30,
        })

    iv_col = "current_iv" if "current_iv" in df.columns else None
    price_col = "ltp" if "ltp" in df.columns else None
    if iv_col and price_col:
        tests.append({
            "id": "iv_price",
            "check": "IV ↑ → Option Price ↑",
            "pair_label": "IV vs Price",
            "column_x": iv_col,
            "column_y": price_col,
            "filter_col": None,
            "filter_val": None,
            "expect_positive": True,
            "threshold_label": "r ≥ 0.15",
            "min_rows": 30,
        })
    return tests


def _subset_for_correlation_test(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    cols = [spec["column_x"], spec["column_y"]]
    if spec.get("filter_col"):
        sub = df.loc[df[spec["filter_col"]] == spec["filter_val"], cols]
    else:
        sub = df[cols]
    return sub.apply(pd.to_numeric, errors="coerce").dropna()


def audit_correlation_checks(
    df: pd.DataFrame,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    chunk_rows: int = 0,
) -> dict[str, Any]:
    """Directional sanity: spot vs delta, IV vs price — with optional live progress."""
    t0 = time.monotonic()
    specs = _prepare_correlation_tests(df)
    checklist = [
        {"id": s["id"], "label": s["pair_label"], "check": s["check"], "status": "pending"}
        for s in specs
    ]
    results: list[dict[str, Any]] = []

    def emit(**payload: Any) -> None:
        if on_progress:
            on_progress({"phase": "correlation", "checks": [dict(c) for c in checklist], **payload})

    emit(
        status="starting",
        pair_label="—",
        rows_processed=0,
        rows_total=0,
        pct_complete=0,
        elapsed_sec=0,
        eta_sec=None,
        rows_per_sec=0,
        correlation=None,
        threshold_label="—",
    )

    for idx, spec in enumerate(specs):
        checklist[idx]["status"] = "running"
        sub = _subset_for_correlation_test(df, spec)
        total = len(sub)
        min_rows = int(spec["min_rows"])

        if total < min_rows:
            checklist[idx]["status"] = "info"
            row = {
                "id": spec["id"],
                "check": spec["check"],
                "pair_label": spec["pair_label"],
                "correlation": None,
                "status": "info",
                "rows": total,
                "threshold_label": spec["threshold_label"],
                "execution_time_sec": 0.0,
            }
            results.append(row)
            emit(
                check_id=spec["id"],
                check_label=spec["check"],
                pair_label=spec["pair_label"],
                status="info",
                rows_processed=total,
                rows_total=total,
                pct_complete=100,
                elapsed_sec=round(time.monotonic() - t0, 2),
                eta_sec=0,
                rows_per_sec=0,
                correlation=None,
                threshold_label=spec["threshold_label"],
                completed_check=row,
            )
            continue

        cs = chunk_rows or max(50, min(500, total // 40))
        check_t0 = time.monotonic()
        partial_corr: float | None = None
        processed = 0

        for end in range(cs, total + cs, cs):
            end = min(end, total)
            chunk = sub.iloc[:end]
            if len(chunk) >= 2:
                partial_corr = float(chunk[spec["column_x"]].corr(chunk[spec["column_y"]]))
            processed = end
            check_elapsed = time.monotonic() - check_t0
            elapsed = time.monotonic() - t0
            speed = processed / check_elapsed if check_elapsed > 0 else 0.0
            remaining = total - processed
            eta = remaining / speed if speed > 0 else None
            pct = round(100.0 * processed / total, 1)
            emit(
                check_id=spec["id"],
                check_label=spec["check"],
                pair_label=spec["pair_label"],
                status="running",
                rows_processed=processed,
                rows_total=total,
                pct_complete=pct,
                elapsed_sec=round(elapsed, 2),
                eta_sec=round(eta, 2) if eta is not None else None,
                rows_per_sec=round(speed, 1),
                correlation=round(partial_corr, 4) if partial_corr is not None else None,
                threshold_label=spec["threshold_label"],
            )

        final_status = _corr_status(partial_corr, expect_positive=bool(spec["expect_positive"]))
        checklist[idx]["status"] = (
            "passed" if final_status == "pass"
            else "failed" if final_status == "fail"
            else final_status
        )
        exec_time = round(time.monotonic() - check_t0, 3)
        row = {
            "id": spec["id"],
            "check": spec["check"],
            "pair_label": spec["pair_label"],
            "correlation": round(partial_corr, 4) if partial_corr is not None else None,
            "status": final_status,
            "rows": total,
            "threshold_label": spec["threshold_label"],
            "execution_time_sec": exec_time,
        }
        results.append(row)
        emit(
            check_id=spec["id"],
            check_label=spec["check"],
            pair_label=spec["pair_label"],
            status=final_status,
            rows_processed=total,
            rows_total=total,
            pct_complete=100,
            elapsed_sec=round(time.monotonic() - t0, 2),
            eta_sec=0,
            rows_per_sec=round(total / exec_time, 1) if exec_time > 0 else 0,
            correlation=row["correlation"],
            threshold_label=spec["threshold_label"],
            completed_check=row,
        )

    statuses = [c["status"] for c in results if c["status"] != "info"]
    if not statuses:
        overall = "warn"
    elif any(s == "fail" for s in statuses):
        overall = "fail"
    elif any(s == "warn" for s in statuses):
        overall = "warn"
    else:
        overall = "pass"

    passed_count = sum(1 for c in results if c["status"] == "pass")
    summary = {
        "status": overall,
        "label": overall.upper(),
        "checks": results,
        "passed_count": passed_count,
        "total_checks": len(results),
        "execution_time_sec": round(time.monotonic() - t0, 3),
    }
    if on_progress:
        on_progress({
            "phase": "correlation",
            "status": "done",
            "checks": [dict(c) for c in checklist],
            "summary": summary,
        })
    return summary


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def _independent_iv(
    index_tl: TickTimeline,
    opt_tl: TickTimeline,
    ts: float,
    option_type: str,
    strike: float,
    expiry_ts: float,
) -> float | None:
    spot = index_tl.ltp_rupees_at(ts)
    ltp = opt_tl.ltp_rupees_at(ts)
    if spot is None or ltp is None or ltp <= 0 or spot <= 0:
        return None
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    if t_exp <= 0:
        return None
    iv = bs.implied_volatility(option_type, ltp, spot, strike, RISK_FREE_RATE, t_exp)
    return iv * 100.0 if iv is not None else None


def _independent_delta(
    index_tl: TickTimeline,
    opt_tl: TickTimeline,
    ts: float,
    option_type: str,
    strike: float,
    expiry_ts: float,
) -> float | None:
    spot = index_tl.ltp_rupees_at(ts)
    ltp = opt_tl.ltp_rupees_at(ts)
    if spot is None or ltp is None or ltp <= 0 or spot <= 0:
        return None
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    if t_exp <= 0:
        return None
    iv = bs.implied_volatility(option_type, ltp, spot, strike, RISK_FREE_RATE, t_exp)
    if iv is None:
        return None
    return float(bs.greeks(option_type, spot, strike, RISK_FREE_RATE, t_exp, iv)["delta"])


def _independent_value(
    check: dict[str, Any],
    *,
    index_tl: TickTimeline,
    opt_tl: TickTimeline,
    ts: float,
    option_type: str,
    strike: float,
    expiry_ts: float,
    policy: dict[str, Any] | None = None,
    greek_snapshots: list[tuple[float, dict[str, float]]] | None = None,
) -> float | None:
    kind = check["kind"]
    if kind == "future_ltp":
        h = int(check["horizon_sec"])
        return opt_tl.ltp_rupees_at(ts + float(h))
    if kind == "ltp_change":
        lb = float(check["lookback_sec"])
        cur = opt_tl.ltp_rupees_at(ts)
        past = opt_tl.ltp_rupees_at(ts - lb)
        if cur is None or past is None:
            return None
        return float(cur - past)
    if kind == "oi_change":
        lb = float(check["lookback_sec"])
        cur = opt_tl.oi_at(ts)
        past = opt_tl.oi_at(ts - lb)
        if cur is None or past is None:
            return None
        return float(cur - past)
    if kind == "delta_change":
        lb = float(check["lookback_sec"])
        d_now = _independent_delta(index_tl, opt_tl, ts, option_type, strike, expiry_ts)
        if d_now is None:
            return None
        pol = normalize_policy_doc(policy)
        if greek_snapshots is not None:
            d_past = greek_at_lookback(greek_snapshots, ts, lb, "delta", pol)
        else:
            d_past = _independent_delta(index_tl, opt_tl, ts - lb, option_type, strike, expiry_ts)
        if d_past is None:
            return None
        return float(d_now - d_past)
    return None


def _values_close(a: Any, b: Any, *, atol: float = 1e-4) -> bool:
    if _is_null(a) and _is_null(b):
        return True
    if _is_null(a) or _is_null(b):
        return False
    try:
        return abs(float(a) - float(b)) <= atol + 1e-9 * max(abs(float(a)), abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return a == b


def audit_independent_formulas(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    expected_doc: dict[str, Any] | None = None,
    n_sample: int = 50,
) -> dict[str, Any]:
    """Recalculate features from replay ticks using dataset configuration policy."""
    if df.empty:
        return {"status": "warn", "rows_checked": 0, "checks": [], "failures": []}

    dataset_config = read_dataset_configuration(meta_doc, expected_doc)
    validator_policy = read_validator_policy(meta_doc, expected_doc)
    policy = lookback_policy(dataset_config)
    alignment = policy_alignment_view(meta_doc, expected_doc, validator_policy=validator_policy)
    config_mismatch, mismatch_reason = detect_configuration_mismatch(
        meta_doc, expected_doc, validator_policy=validator_policy,
    )
    expected_doc = expected_doc or {}

    if config_mismatch:
        return {
            "status": "configuration_mismatch",
            "label": "CONFIG MISMATCH",
            "rows_checked": 0,
            "comparisons": 0,
            "comparisons_failed": 0,
            "checks": [
                {
                    "feature": c["feature"],
                    "label": c["feature"].replace("_", " "),
                    "checked": 0,
                    "failed": 0,
                    "status": "info",
                    "method": "skipped — configuration mismatch",
                }
                for c in independent_checks()
            ],
            "failures": [],
            "dataset_configuration": dataset_config,
            "builder_policy": alignment.get("builder_policy"),
            "validator_policy": alignment.get("validator_policy"),
            "policies_match": False,
            "policy_alignment": alignment,
            "configuration_mismatch": True,
            "mismatch_reason": mismatch_reason,
            "note": mismatch_reason or "Builder and validator lookback policies differ.",
        }

    sample_n = min(max(1, n_sample), len(df))
    sample = df.sample(n=sample_n, random_state=7)
    ctx_cache: dict[str, DayContext] = {}
    replay_cache: dict[Any, dict[str, Any] | None] = {}
    check_stats: dict[str, dict[str, int]] = {
        c["feature"]: {"checked": 0, "passed": 0, "failed": 0}
        for c in independent_checks()
    }
    failures: list[dict[str, Any]] = []

    def _replay_for_row(idx: Any, row: pd.Series) -> dict[str, Any] | None:
        if idx in replay_cache:
            return replay_cache[idx]
        try:
            replay_cache[idx] = replay_row_context(df, meta_doc, expected_doc, chart_dir, row)
        except Exception:
            replay_cache[idx] = None
        return replay_cache[idx]

    for idx, row in sample.iterrows():
        day = str(row.get("trading_day") or "")
        if day not in ctx_cache:
            day_info = next((d for d in (meta_doc.get("days") or []) if str(d.get("trading_day")) == day), None)
            if not day_info:
                continue
            try:
                ctx_cache[day] = load_day_context(
                    chart_dir,
                    SourceSpec(
                        source_id=str(day_info.get("source_id") or day),
                        trading_day=day,
                        market=str(day_info.get("market") or "NIFTY"),
                        expiry=str(day_info.get("expiry") or ""),
                    ),
                )
            except Exception:
                continue

        ctx = ctx_cache[day]
        strike = float(row["strike"])
        opt_type = str(row["option_type"])
        entry = ctx.strike_mapping.get((strike, opt_type))
        if not entry:
            continue
        _tok, _sym, opt_tl = entry
        ts = float(row["timestamp"])
        replay = None
        snapshots = None
        needs_snapshots = any(
            c["feature"] in df.columns and c.get("kind") == "delta_change"
            for c in independent_checks()
        )
        if needs_snapshots:
            replay = _replay_for_row(idx, row)
            if replay:
                snapshots = replay["opt_state"].greek_snapshots

        for check in independent_checks():
            feat = check["feature"]
            if feat not in df.columns:
                continue
            expected_ds = row[feat]
            if _is_null(expected_ds):
                continue
            independent = _independent_value(
                check,
                index_tl=ctx.index_tl,
                opt_tl=opt_tl,
                ts=ts,
                option_type=opt_type,
                strike=strike,
                expiry_ts=ctx.expiry_ts,
                policy=validator_policy,
                greek_snapshots=snapshots,
            )
            check_stats[feat]["checked"] += 1
            if _values_close(expected_ds, independent, atol=0.02 if "ltp" in feat or "oi" in feat else 1e-3):
                check_stats[feat]["passed"] += 1
            else:
                check_stats[feat]["failed"] += 1
                if len(failures) < 20:
                    failures.append({
                        "feature": feat,
                        "dataset": float(expected_ds),
                        "independent": independent,
                        "difference": (
                            abs(float(expected_ds) - float(independent))
                            if independent is not None and not _is_null(expected_ds)
                            else None
                        ),
                        "trading_day": day,
                        "timestamp": ts,
                        "timestamp_label": _fmt_ts(ts),
                        "strike": strike,
                        "option_type": opt_type,
                        "token": str(row.get("token") or ""),
                        "strike_label": f"{int(strike) if strike == int(strike) else strike} {opt_type}",
                    })

    rows_out = []
    total_fail = 0
    total_chk = 0
    for check in independent_checks():
        feat = check["feature"]
        st = check_stats.get(feat, {})
        checked = int(st.get("checked") or 0)
        failed = int(st.get("failed") or 0)
        total_chk += checked
        total_fail += failed
        if checked == 0:
            status = "info"
        elif failed == 0:
            status = "pass"
        else:
            status = "fail"
        rows_out.append({
            "feature": feat,
            "label": feat.replace("_", " "),
            "checked": checked,
            "failed": failed,
            "status": status,
            "method": "replay_tick + BS (independent)",
        })

    if total_chk == 0:
        status = "warn"
    elif total_fail == 0:
        status = "pass"
    else:
        status = "fail"

    return {
        "status": status,
        "label": "PASS" if status == "pass" else ("FAIL" if status == "fail" else "WARN"),
        "rows_checked": int(len(sample)),
        "comparisons": total_chk,
        "comparisons_failed": total_fail,
        "checks": rows_out,
        "failures": failures,
        "dataset_configuration": dataset_config,
        "builder_policy": alignment.get("builder_policy"),
        "validator_policy": alignment.get("validator_policy"),
        "policies_match": alignment.get("policies_match", True),
        "policy_alignment": alignment,
        "configuration_mismatch": False,
        "note": (
            f"Validator uses dataset lookback policy: {policy_label(validator_policy)}. "
            "Replay tick lookup + Black-Scholes — does not call dataset builder."
        ),
    }


def audit_feature_distributions(df: pd.DataFrame) -> dict[str, Any]:
    """Min / max / mean / median / std with range sanity checks."""
    rows: list[dict[str, Any]] = []
    any_fail = False

    for spec in distribution_features():
        col = spec["column"]
        if col not in df.columns and spec.get("alt"):
            col = spec["alt"]
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            rows.append({
                "feature": col,
                "label": spec["label"],
                "status": "info",
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "note": "No non-null values",
            })
            continue

        vmin = float(series.min())
        vmax = float(series.max())
        status = "pass"
        issues: list[str] = []
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and vmin < float(lo):
            status = "fail"
            issues.append(f"min {vmin:.4g} < {lo}")
            any_fail = True
        if hi is not None and vmax > float(hi):
            status = "fail"
            issues.append(f"max {vmax:.4g} > {hi}")
            any_fail = True

        rows.append({
            "feature": col,
            "label": spec["label"],
            "status": status,
            "min": round(vmin, 6),
            "max": round(vmax, 6),
            "mean": round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "std": round(float(series.std()), 6) if len(series) > 1 else 0.0,
            "bounds": {"min": lo, "max": hi},
            "issues": issues,
        })

    return {
        "status": "fail" if any_fail else "pass",
        "label": "FAIL" if any_fail else "PASS",
        "features": rows,
    }


def audit_replay_verification(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    target_columns: list[str],
    n_sample: int = 10,
) -> dict[str, Any]:
    """Random timestamps with spot, ATM, LTP, and target columns for visual verification."""
    if df.empty:
        return {"status": "warn", "samples": []}

    sample_n = min(max(1, n_sample), len(df))
    sample = df.sample(n=sample_n, random_state=11)
    ctx_cache: dict[str, DayContext] = {}
    samples: list[dict[str, Any]] = []

    for _, row in sample.iterrows():
        day = str(row.get("trading_day") or "")
        if day not in ctx_cache:
            day_info = next((d for d in (meta_doc.get("days") or []) if str(d.get("trading_day")) == day), None)
            if not day_info:
                continue
            try:
                ctx_cache[day] = load_day_context(
                    chart_dir,
                    SourceSpec(
                        source_id=str(day_info.get("source_id") or day),
                        trading_day=day,
                        market=str(day_info.get("market") or "NIFTY"),
                        expiry=str(day_info.get("expiry") or ""),
                    ),
                )
            except Exception:
                continue

        ctx = ctx_cache[day]
        ts = float(row["timestamp"])
        strike = float(row["strike"])
        opt_type = str(row["option_type"])
        index_key = normalize_index_name(ctx.source.market)
        strike_step = STRIKE_STEP.get(index_key, 50)
        spot = ctx.index_tl.ltp_rupees_at(ts)
        atm = find_atm_strike(spot, strike_step) if spot else None

        entry = ctx.strike_mapping.get((strike, opt_type))
        replay_ltp = None
        if entry:
            _tok, _sym, opt_tl = entry
            replay_ltp = opt_tl.ltp_rupees_at(ts)

        targets: dict[str, Any] = {}
        for col in target_columns:
            if col in df.columns:
                v = row[col]
                targets[col] = None if _is_null(v) else float(v)

        horizon_labels = {}
        for col in target_columns:
            m = re.search(r"_(\d+)s$", col)
            if m:
                horizon_labels[col] = f"{m.group(1)}s Future"
            elif col.endswith("_1m"):
                horizon_labels[col] = "1m Future"
            elif col.endswith("_5m"):
                horizon_labels[col] = "5m Future"

        samples.append({
            "trading_day": day,
            "timestamp": ts,
            "timestamp_label": _fmt_ts(ts),
            "spot": round(spot, 2) if spot is not None else None,
            "atm_strike": atm,
            "selected_strike": int(strike) if strike == int(strike) else strike,
            "option_type": opt_type,
            "strike_label": f"{int(strike) if strike == int(strike) else strike} {opt_type}",
            "current_ltp": round(float(row["ltp"]), 2) if "ltp" in df.columns and not _is_null(row.get("ltp")) else replay_ltp,
            "replay_ltp": round(replay_ltp, 2) if replay_ltp is not None else None,
            "targets": targets,
            "target_labels": horizon_labels,
        })

    return {"status": "pass" if samples else "warn", "samples": samples}


def audit_feature_heatmap(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Per-feature missing value counts (full list, sorted descending)."""
    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    max_missing = 0

    for col in feature_columns:
        if col not in df.columns:
            rows.append({"feature": col, "missing": total_rows, "pct": 100.0})
            max_missing = max(max_missing, total_rows)
            continue
        n_missing = int(pd.isna(df[col]).sum())
        if n_missing <= 0:
            continue
        max_missing = max(max_missing, n_missing)
        rows.append({
            "feature": col,
            "missing": n_missing,
            "pct": round(100.0 * n_missing / total_rows, 2) if total_rows else 0.0,
        })

    rows.sort(key=lambda r: r["missing"], reverse=True)

    return {
        "total_rows": total_rows,
        "features_with_gaps": len(rows),
        "total_missing_cells": sum(r["missing"] for r in rows),
        "max_missing": max_missing,
        "rows": rows,
    }


def run_extended_audits(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    feature_columns: list[str],
    target_columns: list[str],
    expected_doc: dict[str, Any] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    tracker: AuditStageTracker | None = None,
    audit_options: Any | None = None,
) -> dict[str, Any]:
    """Run extended validation blocks (subset skippable for fast experiment builds)."""
    from .audit_options import AuditOptions

    opts = audit_options if isinstance(audit_options, AuditOptions) else AuditOptions()
    stage = tracker or (AuditStageTracker(on_progress) if on_progress else None)

    def step(name: str, status: str = "running") -> None:
        if stage:
            stage.emit(name, status)

    out: dict[str, Any] = {"skipped": opts.to_dict()}

    if not opts.skip_feature_audit:
        step("independent_formulas")
        out["independent_formulas"] = audit_independent_formulas(
            df, meta_doc, chart_dir=chart_dir, expected_doc=expected_doc,
        )
        step("independent_formulas", "done")
    else:
        out["independent_formulas"] = {"status": "skipped", "reason": "fast_experiment"}

    if not opts.skip_distribution_report:
        step("distribution_checks")
        out["feature_distributions"] = audit_feature_distributions(df)
        step("distribution_checks", "done")
    else:
        out["feature_distributions"] = {"status": "skipped", "reason": "fast_experiment"}

    if not opts.skip_leakage_audit:
        step("correlation_checks")
        out["correlation_checks"] = audit_correlation_checks(
            df,
            on_progress=stage.progress_callback if stage else on_progress,
        )
        step("correlation_checks", "done")
        step("replay_verification")
        out["replay_verification"] = audit_replay_verification(
            df, meta_doc, chart_dir=chart_dir, target_columns=target_columns,
        )
        step("replay_verification", "done")
    else:
        out["correlation_checks"] = {"status": "skipped", "reason": "fast_experiment"}
        out["replay_verification"] = {"status": "skipped", "reason": "fast_experiment"}

    if not opts.skip_feature_audit:
        step("feature_heatmap")
        out["feature_heatmap"] = audit_feature_heatmap(df, feature_columns)
        step("feature_heatmap", "done")
    else:
        out["feature_heatmap"] = {"status": "skipped", "reason": "fast_experiment"}

    return out
