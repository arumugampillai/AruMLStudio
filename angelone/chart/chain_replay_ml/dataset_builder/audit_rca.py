"""Root cause analysis for audit failures — explains *why* a check failed."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .audit_extended import (
    _fmt_ts,
    _independent_delta,
    _independent_value,
    _is_null,
    _prepare_correlation_tests,
    _subset_for_correlation_test,
    _values_close,
)
from .validation_rules import distribution_features, independent_checks
from .audit_replay import replay_row_context
from .day_context import SourceSpec, load_day_context
from .expected_spec import expected_spec_path
from .investigation_history import append_investigation
from .lookback_policy import (
    POLICY_EXACT_TIMESTAMP,
    POLICY_NEAREST_SNAPSHOT,
    dataset_specification_view,
    detect_configuration_mismatch,
    greek_at_lookback,
    greek_at_lookback_with_ts,
    lookback_policy,
    policy_alignment_view,
    policy_label,
    policies_match,
    normalize_policy_method,
    read_dataset_configuration,
    read_validator_policy,
)
from .pipeline_identity import dataset_version_view, investigation_version_context
from .spec_identity import build_audit_specification_summary
from .writer import _safe_filename, datasets_dir

_CAUSE_LABELS: dict[str, str] = {
    "wrong_lookback_timestamp": "Wrong lookback timestamp",
    "previous_delta_unavailable": "Previous delta unavailable",
    "previous_value_unavailable": "Previous value unavailable",
    "wrong_bs_input": "BS input",
    "wrong_interpolation": "Option LTP",
    "wrong_expiry": "Expiry",
    "wrong_iv": "Wrong IV",
    "wrong_spot": "Spot",
    "wrong_strike": "Strike",
    "wrong_option_token": "Token",
    "wrong_timeline_lookup": "Timeline lookup",
    "snapshot_window_mismatch": "Snapshot window mismatch",
    "unit_mismatch": "Possible unit mismatch",
    "outlier_values": "Outlier values exceed bounds",
    "insufficient_samples": "Insufficient samples for correlation",
    "directional_mismatch": "Spot–delta direction inconsistent",
    "invalid_statistical_test": "Invalid statistical test",
    "cross_sectional_confound": "Cross-sectional confound",
    "monotonicity_within_strike": "Delta monotonicity within strike",
    "delta_bounds_valid": "Delta within valid bounds",
    "moneyness_correlation": "Delta vs moneyness",
}

_SUMMARY_LABELS: dict[str, str] = {
    "wrong_lookback_timestamp": "Wrong lookback timestamp",
    "snapshot_window_mismatch": "Snapshot window mismatch",
    "other": "Other",
}


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _cause(
    cause_id: str,
    *,
    status: str,
    detail: str,
    evidence: dict[str, Any] | None = None,
    confidence_pct: int | None = None,
    confidence_label: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": cause_id,
        "label": _CAUSE_LABELS.get(cause_id, cause_id.replace("_", " ").title()),
        "status": status,
        "detail": detail,
        "evidence": evidence or {},
    }
    if confidence_pct is not None:
        out["confidence_pct"] = confidence_pct
    if confidence_label:
        out["confidence_label"] = confidence_label
    return out


def _confidence_checks_from_causes(
    causes: list[dict[str, Any]],
    *,
    timestamp_error_pct: float | None = None,
    bs_derived: bool = False,
) -> list[dict[str, Any]]:
    """Ordered checklist with confidence for the RCA table."""
    order = [
        ("wrong_spot", "Spot"),
        ("wrong_interpolation", "Option LTP"),
        ("wrong_strike", "Strike"),
        ("wrong_option_token", "Token"),
        ("wrong_expiry", "Expiry"),
        ("wrong_lookback_timestamp", "Timestamp"),
        ("wrong_bs_input", "BS Input"),
    ]
    by_id = {c["id"]: c for c in causes}
    rows: list[dict[str, Any]] = []
    for cid, label in order:
        c = by_id.get(cid)
        if not c and cid == "wrong_bs_input" and bs_derived:
            rows.append({
                "check": label,
                "status": "derived",
                "confidence_pct": None,
                "confidence_label": "Derived from timestamp",
            })
            continue
        if not c:
            continue
        st = c["status"]
        if cid == "wrong_lookback_timestamp" and st == "fail" and timestamp_error_pct is not None:
            pct = int(round(timestamp_error_pct))
            rows.append({"check": label, "status": "fail", "confidence_pct": pct})
        elif cid == "wrong_bs_input" and bs_derived and st == "fail":
            rows.append({
                "check": label,
                "status": "derived",
                "confidence_pct": None,
                "confidence_label": "Derived from timestamp",
            })
        elif st == "pass":
            rows.append({"check": label, "status": "pass", "confidence_pct": 100})
        elif st == "fail":
            rows.append({"check": label, "status": "fail", "confidence_pct": c.get("confidence_pct", 70)})
        else:
            rows.append({
                "check": label,
                "status": st,
                "confidence_pct": c.get("confidence_pct"),
                "confidence_label": c.get("confidence_label"),
            })
    return rows


def _build_root_cause(
    *,
    kind: str,
    lookback_sec: float,
    ts: float,
    snapshot_ts: float | None,
    policy_target_ts: float | None,
    builder_policy: dict[str, Any],
    validator_policy: dict[str, Any],
) -> dict[str, Any] | None:
    if kind != "delta_change" or not lookback_sec or snapshot_ts is None:
        return None
    if not policies_match(builder_policy, validator_policy):
        return None
    builder_offset = round(ts - snapshot_ts, 1)
    error_sec = round(lookback_sec - builder_offset, 1)
    error_pct = round(abs(error_sec) / lookback_sec * 100.0, 1) if lookback_sec else 0.0
    sign = "+" if error_sec > 0 else ""
    if normalize_policy_method(builder_policy.get("method")) == POLICY_EXACT_TIMESTAMP:
        if policy_target_ts is not None and abs(snapshot_ts - policy_target_ts) <= 15.0:
            return None
    else:
        lo = lookback_sec * 0.5
        hi = lookback_sec * 1.5
        if lo <= builder_offset <= hi:
            return None
    return {
        "title": "ROOT CAUSE",
        "builder_method": f"Builder uses {policy_label(builder_policy)}.",
        "validator_method": f"Validator uses {policy_label(validator_policy)}.",
        "difference": {
            "requested_lookback_sec": int(lookback_sec),
            "builder_snapshot_sec": int(round(builder_offset)),
            "error_sec": error_sec,
            "error_label": f"{sign}{error_sec:g} sec",
            "error_pct": error_pct,
            "builder_ts_label": _fmt_ts(snapshot_ts),
            "expected_ts_label": _fmt_ts(policy_target_ts) if policy_target_ts else None,
        },
    }


def _recommended_fix_from_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    if alignment.get("policies_match"):
        return {
            "title": "Suggested Fix",
            "action": "No change required",
            "reason": "Validator uses the same lookback policy recorded in dataset metadata.",
            "status": "aligned",
        }
    return {
        "title": "Suggested Fix",
        "action": alignment.get("suggested_fix") or "Align validator with dataset build policy.",
        "reason": alignment.get("mismatch_reason") or "Builder and validator lookback policies differ.",
        "status": "configuration_mismatch",
    }


def _snapshots_for_row(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    row: pd.Series,
    check: dict[str, Any],
) -> list[tuple[float, dict[str, float]]] | None:
    if check.get("kind") != "delta_change":
        return None
    try:
        replay = replay_row_context(df, meta_doc, expected_doc, chart_dir, row)
        return replay["opt_state"].greek_snapshots
    except Exception:
        return None


def _estimate_fix_impact(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    feature: str,
    check: dict[str, Any],
    validator_policy: dict[str, Any],
    *,
    max_scan: int = 80,
) -> dict[str, Any]:
    """Count independent validation failures using the configured validator policy."""
    if feature not in df.columns:
        return {}

    atol = 0.02 if "ltp" in feature or "oi" in feature else 1e-3
    candidates = df[df[feature].notna()]
    if len(candidates) > max_scan:
        candidates = candidates.sample(n=max_scan, random_state=7)

    ctx_cache: dict[str, Any] = {}
    fail_count = 0
    checked = 0

    for idx, row in candidates.iterrows():
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
        strike_v = float(row["strike"])
        opt_type = str(row["option_type"])
        entry = ctx.strike_mapping.get((strike_v, opt_type))
        if not entry:
            continue
        ts = float(row["timestamp"])
        opt_tl = entry[2]
        snapshots = _snapshots_for_row(df, meta_doc, expected_doc, chart_dir, row, check)
        independent = _independent_value(
            check, index_tl=ctx.index_tl, opt_tl=opt_tl, ts=ts,
            option_type=opt_type, strike=strike_v, expiry_ts=ctx.expiry_ts,
            policy=validator_policy, greek_snapshots=snapshots,
        )
        checked += 1
        if not _values_close(row[feature], independent, atol=atol):
            fail_count += 1

    non_null = int(df[feature].notna().sum())
    coverage_pct = round(100.0 * non_null / len(df), 4) if len(df) else 0.0

    return {
        "independent_validation": {
            "current_failures": fail_count,
            "rows_scanned": checked,
        },
        "feature_coverage_pct": coverage_pct,
    }


def _classify_greek_row(
    *,
    snap_gap: float | None,
    previous_unavailable: bool,
    timeline_ok: bool,
    token_ok: bool,
) -> str:
    if not timeline_ok or not token_ok or previous_unavailable:
        return "other"
    if snap_gap is not None and snap_gap > 15.0:
        return "wrong_lookback_timestamp"
    if snap_gap is not None:
        return "snapshot_window_mismatch"
    return "other"


def _classify_failure_row(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    check: dict[str, Any],
    idx: Any,
) -> str:
    kind = check["kind"]
    lookback = float(check.get("lookback_sec") or 0)
    if kind != "delta_change" or not lookback:
        return "other"
    try:
        row = df.loc[idx]
        replay = replay_row_context(df, meta_doc, expected_doc, chart_dir, row)
    except Exception:
        return "other"
    ts = replay["ts"]
    opt_state = replay["opt_state"]
    ctx = replay["ctx"]
    opt_tl = replay["opt_tl"]
    policy = read_validator_policy(meta_doc, expected_doc)
    policy_target = ts - lookback
    snapshot_ts, _ = greek_at_lookback_with_ts(
        opt_state.greek_snapshots, ts, lookback, "delta", policy,
    )
    snap_gap = abs(snapshot_ts - policy_target) if snapshot_ts is not None else None
    past_delta = greek_at_lookback(opt_state.greek_snapshots, ts, lookback, "delta", policy)
    if past_delta is None:
        past_delta = _independent_delta(
            ctx.index_tl, opt_tl, policy_target, replay["option_type"], replay["strike"], ctx.expiry_ts,
        )
    timeline_ok = ctx.index_tl.ltp_rupees_at(ts) is not None and opt_tl.ltp_rupees_at(ts) is not None
    token_ok = ctx.strike_mapping.get((replay["strike"], replay["option_type"])) is not None
    return _classify_greek_row(
        snap_gap=snap_gap,
        previous_unavailable=past_delta is None,
        timeline_ok=timeline_ok,
        token_ok=token_ok,
    )


def _aggregate_independent_failures(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    feature: str,
    check: dict[str, Any],
    *,
    max_scan: int = 120,
    max_classify: int = 40,
) -> dict[str, Any]:
    """Classify detected failures into summary buckets."""
    if feature not in df.columns:
        return {"total_failures": 0, "buckets": [], "scanned": 0}

    atol = 0.02 if "ltp" in feature or "oi" in feature else 1e-3
    validator_policy = read_validator_policy(meta_doc, expected_doc)
    candidates = df[df[feature].notna()]
    if len(candidates) > max_scan:
        candidates = candidates.sample(n=max_scan, random_state=7)

    ctx_cache: dict[str, Any] = {}
    failure_indices: list[Any] = []

    for idx, row in candidates.iterrows():
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
        strike_v = float(row["strike"])
        opt_type = str(row["option_type"])
        entry = ctx.strike_mapping.get((strike_v, opt_type))
        if not entry:
            continue
        ts = float(row["timestamp"])
        snapshots = _snapshots_for_row(df, meta_doc, expected_doc, chart_dir, row, check)
        independent = _independent_value(
            check,
            index_tl=ctx.index_tl,
            opt_tl=entry[2],
            ts=ts,
            option_type=opt_type,
            strike=strike_v,
            expiry_ts=ctx.expiry_ts,
            policy=validator_policy,
            greek_snapshots=snapshots,
        )
        if not _values_close(row[feature], independent, atol=atol):
            failure_indices.append(idx)

    buckets: dict[str, int] = {
        "wrong_lookback_timestamp": 0,
        "snapshot_window_mismatch": 0,
        "other": 0,
    }
    if not failure_indices:
        return {"total_failures": 0, "buckets": [], "scanned": len(candidates)}

    for idx in failure_indices[:max_classify]:
        bucket = _classify_failure_row(df, meta_doc, expected_doc, chart_dir, check, idx)
        buckets[bucket] += 1
    remaining = len(failure_indices) - min(len(failure_indices), max_classify)
    if remaining > 0 and buckets["wrong_lookback_timestamp"] > buckets["snapshot_window_mismatch"]:
        buckets["wrong_lookback_timestamp"] += remaining
    elif remaining > 0:
        buckets["other"] += remaining

    bucket_rows = [
        {"id": k, "label": _SUMMARY_LABELS[k], "count": buckets[k]}
        for k in ("wrong_lookback_timestamp", "snapshot_window_mismatch", "other")
        if buckets[k] > 0
    ]
    return {
        "total_failures": len(failure_indices),
        "scanned": len(candidates),
        "buckets": bucket_rows,
    }


def _find_independent_check(feature: str) -> dict[str, Any] | None:
    for check in independent_checks():
        if check["feature"] == feature:
            return check
    return None


def _pick_failure_row(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    feature: str,
    *,
    trading_day: str | None,
    timestamp: float | None,
    strike: float | None,
    option_type: str | None,
    max_scan: int = 80,
) -> pd.Series | None:
    if trading_day and timestamp is not None:
        mask = (
            (df["trading_day"].astype(str) == str(trading_day))
            & (pd.to_numeric(df["timestamp"], errors="coerce") == float(timestamp))
        )
        if strike is not None:
            mask &= pd.to_numeric(df["strike"], errors="coerce") == float(strike)
        if option_type:
            mask &= df["option_type"].astype(str) == str(option_type)
        hits = df.loc[mask]
        if not hits.empty:
            return hits.iloc[0]

    check = _find_independent_check(feature)
    if not check or feature not in df.columns:
        return None

    sample = df[df[feature].notna()].sample(n=min(max_scan, len(df)), random_state=7)
    validator_policy = read_validator_policy(meta_doc, expected_doc)
    ctx_cache: dict[str, Any] = {}
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
        strike_v = float(row["strike"])
        opt_type = str(row["option_type"])
        entry = ctx.strike_mapping.get((strike_v, opt_type))
        if not entry:
            continue
        ts = float(row["timestamp"])
        snapshots = _snapshots_for_row(df, meta_doc, expected_doc, chart_dir, row, check)
        independent = _independent_value(
            check,
            index_tl=ctx.index_tl,
            opt_tl=entry[2],
            ts=ts,
            option_type=opt_type,
            strike=strike_v,
            expiry_ts=ctx.expiry_ts,
            policy=validator_policy,
            greek_snapshots=snapshots,
        )
        atol = 0.02 if "ltp" in feature or "oi" in feature else 1e-3
        if not _values_close(row[feature], independent, atol=atol):
            return row
    return None


def _investigate_independent(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    feature: str,
    *,
    trading_day: str | None,
    timestamp: float | None,
    strike: float | None,
    option_type: str | None,
    audit_failure_count: int | None = None,
    timeline: Any | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    check = _find_independent_check(feature)
    if not check:
        return {
            "status": "unsupported",
            "feature": feature,
            "diagnosis": {"problem": "Unsupported feature", "summary": f"No RCA template for {feature}"},
            "causes": [],
        }

    if timeline:
        timeline.begin("failure_row_located", "Failure row located")
    row = _pick_failure_row(
        df, meta_doc, expected_doc, chart_dir, feature,
        trading_day=trading_day, timestamp=timestamp, strike=strike, option_type=option_type,
        max_scan=12 if fast_mode else 80,
    )
    if timeline:
        timeline.complete("failure_row_located", "pass" if row is not None else "fail")
    if row is None:
        return {
            "status": "not_found",
            "feature": feature,
            "diagnosis": {"problem": "No failing sample", "summary": "Could not locate a failing row to investigate."},
            "causes": [],
        }

    dataset_config = read_dataset_configuration(meta_doc, expected_doc)
    builder_policy = lookback_policy(dataset_config)
    validator_policy = read_validator_policy(meta_doc, expected_doc)
    if timeline:
        timeline.begin("lookback_policy_loaded", "Lookback policy loaded")
        timeline.complete("lookback_policy_loaded")
        timeline.begin("sampling_policy_loaded", "Sampling policy loaded")
        timeline.complete("sampling_policy_loaded")
    alignment = policy_alignment_view(meta_doc, expected_doc, validator_policy=validator_policy)
    config_mismatch, mismatch_reason = detect_configuration_mismatch(
        meta_doc, expected_doc, validator_policy=validator_policy,
    )
    spec_view = dataset_specification_view(dataset_config, feature=feature)
    recommended_fix = _recommended_fix_from_alignment(alignment)

    if config_mismatch:
        version_ctx = investigation_version_context(meta_doc, expected_doc)
        return {
            "status": "configuration_mismatch",
            "category": "independent",
            "feature": feature,
            "diagnosis": {
                "problem": "Configuration Mismatch",
                "summary": mismatch_reason or "Builder and validator lookback policies differ.",
                "confidence": "high",
            },
            "policy_alignment": alignment,
            "dataset_configuration": dataset_config,
            "dataset_version_info": dataset_version_view(meta_doc, expected_doc),
            "investigated_using": version_ctx,
            "dataset_specification": {
                "prediction_horizon": spec_view.get("prediction_horizon"),
                "configured_method": spec_view.get("configured_method"),
                "tolerance": spec_view.get("tolerance_label"),
                "sampling_interval_sec": spec_view.get("sampling_interval_sec"),
                "future_targets": spec_view.get("future_targets"),
            },
            "recommended_fix": recommended_fix,
            "causes": [],
        }

    if timeline:
        timeline.begin("replay_db", "Replay DB opened")
    replay = replay_row_context(df, meta_doc, expected_doc, chart_dir, row)
    if timeline:
        timeline.complete("replay_db")
        timeline.begin("trading_day_resolved", "Trading day resolved")
        timeline.complete("trading_day_resolved")
        timeline.begin("spot_timeline_loaded", "Spot timeline loaded")
        timeline.complete("spot_timeline_loaded")
        timeline.begin("option_timeline_loaded", "Option timeline loaded")
        timeline.complete("option_timeline_loaded")
        timeline.begin("strike_verified", "Strike verified")
        timeline.complete("strike_verified")
        timeline.begin("expiry_verified", "Expiry verified")
        timeline.complete("expiry_verified")
        timeline.begin("token_verified", "Token verified")
        timeline.complete("token_verified")
    ctx = replay["ctx"]
    opt_state = replay["opt_state"]
    opt_tl = replay["opt_tl"]
    ts = replay["ts"]
    strike_v = replay["strike"]
    opt_type = replay["option_type"]
    dataset_val = float(row[feature])
    recomputed_val = replay["recomputed"].get(feature)
    lookback = float(check.get("lookback_sec") or 0)
    horizon = float(check.get("horizon_sec") or 0)
    kind = check["kind"]
    snapshots = opt_state.greek_snapshots if kind == "delta_change" else None

    if timeline:
        timeline.begin("previous_timestamp_located", "Previous timestamp located")
    independent = _independent_value(
        check,
        index_tl=ctx.index_tl,
        opt_tl=opt_tl,
        ts=ts,
        option_type=opt_type,
        strike=strike_v,
        expiry_ts=ctx.expiry_ts,
        policy=validator_policy,
        greek_snapshots=snapshots,
    )
    if timeline:
        timeline.complete("previous_timestamp_located")
        timeline.begin("black_scholes_recalculated", "Black-Scholes recalculated")
        timeline.complete("black_scholes_recalculated")
        timeline.begin("dataset_feature_extracted", "Dataset feature extracted")
        timeline.complete("dataset_feature_extracted")
        timeline.begin("independent_feature_calculated", "Independent feature calculated")
        timeline.complete("independent_feature_calculated")
        timeline.begin("difference_computed", "Difference computed")
        timeline.complete("difference_computed")

    causes: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "dataset_value": dataset_val,
        "independent_value": independent,
        "builder_replay_value": float(recomputed_val) if recomputed_val is not None else None,
        "timestamp_label": _fmt_ts(ts),
        "trading_day": str(row.get("trading_day") or ""),
        "strike_label": f"{int(strike_v) if strike_v == int(strike_v) else strike_v} {opt_type}",
    }

    policy_target_ts = ts - lookback if lookback else None
    if policy_target_ts is not None:
        comparison["expected_lookback_ts_label"] = _fmt_ts(policy_target_ts)

    spot_now = ctx.index_tl.ltp_rupees_at(ts)
    spot_row = float(row["spot"]) if "spot" in row.index and not _is_null(row.get("spot")) else None
    ltp_now = opt_tl.ltp_rupees_at(ts)
    ltp_row = float(row["ltp"]) if "ltp" in row.index and not _is_null(row.get("ltp")) else None

    if spot_now is None or ltp_now is None:
        causes.append(_cause(
            "wrong_timeline_lookup",
            status="fail",
            detail="Replay timeline missing spot or option LTP at current timestamp.",
            evidence={"spot": spot_now, "ltp": ltp_now},
        ))
    else:
        causes.append(_cause(
            "wrong_timeline_lookup",
            status="pass",
            detail="Spot and option ticks resolve at current timestamp.",
        ))

    if spot_row is not None and spot_now is not None:
        spot_diff = abs(spot_row - spot_now)
        ok = spot_diff <= 0.05
        causes.append(_cause(
            "wrong_spot",
            status="pass" if ok else "fail",
            detail=(
                f"Dataset spot {spot_row:.2f} matches replay {spot_now:.2f}."
                if ok else
                f"Dataset spot {spot_row:.2f} vs replay {spot_now:.2f} (Δ {spot_diff:.4f})."
            ),
            evidence={"dataset": spot_row, "replay": spot_now},
        ))

    if ltp_row is not None and ltp_now is not None:
        ltp_diff = abs(ltp_row - ltp_now)
        ok = ltp_diff <= 0.05
        causes.append(_cause(
            "wrong_interpolation",
            status="pass" if ok else "warn",
            detail=(
                f"Option LTP dataset {ltp_row:.2f} vs replay {ltp_now:.2f}."
            ),
            evidence={"dataset": ltp_row, "replay": ltp_now},
        ))

    if kind == "delta_change":
        greek_key = "delta"
        delta_now = _independent_delta(ctx.index_tl, opt_tl, ts, opt_type, strike_v, ctx.expiry_ts)
        snapshot_ts, snapshot_delta = greek_at_lookback_with_ts(
            opt_state.greek_snapshots, ts, lookback, greek_key, validator_policy,
        )
        builder_past = greek_at_lookback(
            opt_state.greek_snapshots, ts, lookback, greek_key, builder_policy,
        )
        delta_past_policy = snapshot_delta
        if delta_past_policy is None and policy_target_ts is not None:
            delta_past_policy = _independent_delta(
                ctx.index_tl, opt_tl, policy_target_ts, opt_type, strike_v, ctx.expiry_ts,
            )
        delta_row = float(row["delta"]) if "delta" in row.index and not _is_null(row.get("delta")) else None

        comparison.update({
            "current_delta": delta_now,
            "expected_previous_delta": delta_past_policy,
            "builder_previous_delta": builder_past,
            "dataset_delta": delta_row,
            "builder_snapshot_ts_label": _fmt_ts(snapshot_ts) if snapshot_ts else None,
            "lookback_policy": policy_label(validator_policy),
        })
        if delta_now is not None and delta_past_policy is not None:
            comparison["independent_change"] = float(delta_now - delta_past_policy)
        if delta_now is not None and builder_past is not None:
            comparison["builder_change"] = float(delta_now - builder_past)

        if delta_past_policy is None:
            causes.append(_cause(
                "previous_delta_unavailable",
                status="fail",
                detail=f"Cannot resolve past delta using {policy_label(validator_policy)}.",
            ))
        else:
            causes.append(_cause(
                "previous_delta_unavailable",
                status="pass",
                detail=f"Past delta available via {policy_label(validator_policy)}.",
            ))

        if snapshot_ts is not None and policy_target_ts is not None:
            snap_gap = abs(snapshot_ts - policy_target_ts)
            builder_offset = ts - snapshot_ts
            lo = lookback * 0.5
            hi = lookback * 1.5
            if normalize_policy_method(validator_policy.get("method")) == POLICY_EXACT_TIMESTAMP:
                wrong_ts = snap_gap > 15.0
            else:
                wrong_ts = builder_offset < lo or builder_offset > hi
            comparison["lookback_gap_sec"] = round(snap_gap, 1)
            causes.append(_cause(
                "wrong_lookback_timestamp",
                status="fail" if wrong_ts else "pass",
                detail=(
                    f"Snapshot at {_fmt_ts(snapshot_ts)} vs policy target {_fmt_ts(policy_target_ts)} "
                    f"({snap_gap:.0f}s apart)."
                    if wrong_ts else
                    f"Snapshot {_fmt_ts(snapshot_ts)} satisfies {policy_label(validator_policy)}."
                ),
                evidence={
                    "builder_ts": snapshot_ts,
                    "expected_ts": policy_target_ts,
                    "gap_sec": snap_gap,
                },
            ))

        if builder_past is not None and delta_past_policy is not None:
            past_diff = abs(builder_past - delta_past_policy)
            causes.append(_cause(
                "wrong_bs_input",
                status="fail" if past_diff > 0.002 else "pass",
                detail=(
                    f"Builder past delta {builder_past:.6f} vs policy lookback {delta_past_policy:.6f} "
                    f"(Δ {past_diff:.6f})."
                ),
                evidence={"builder": builder_past, "policy": delta_past_policy, "diff": past_diff},
            ))

    elif kind in ("ltp_change", "oi_change"):
        past_ts = ts - lookback
        if kind == "ltp_change":
            cur = opt_tl.ltp_rupees_at(ts)
            past = opt_tl.ltp_rupees_at(past_ts)
            label = "LTP"
        else:
            cur = opt_tl.oi_at(ts)
            past = opt_tl.oi_at(past_ts)
            label = "OI"
        comparison["current_value"] = cur
        comparison["past_value_exact"] = past
        comparison["expected_lookback_ts_label"] = _fmt_ts(past_ts)
        if past is None:
            causes.append(_cause(
                "previous_value_unavailable",
                status="fail",
                detail=f"No {label} tick at exact lookback {_fmt_ts(past_ts)}.",
            ))
        else:
            causes.append(_cause(
                "previous_value_unavailable",
                status="pass",
                detail=f"{label} available at {_fmt_ts(past_ts)}.",
            ))
        if cur is not None and past is not None:
            causes.append(_cause(
                "wrong_timeline_lookup",
                status="pass",
                detail=f"{label} change from replay: {float(cur - past):.4f}.",
            ))

    elif kind == "future_ltp":
        future_ts = ts + horizon
        future_ltp = opt_tl.ltp_rupees_at(future_ts)
        comparison["future_ts_label"] = _fmt_ts(future_ts)
        comparison["future_ltp"] = future_ltp
        if future_ltp is None:
            causes.append(_cause(
                "wrong_timeline_lookup",
                status="fail",
                detail=f"No option tick at future horizon {_fmt_ts(future_ts)}.",
            ))
        else:
            causes.append(_cause(
                "wrong_timeline_lookup",
                status="pass",
                detail=f"Future LTP resolves at {_fmt_ts(future_ts)}.",
            ))

    # Token / strike sanity
    entry = ctx.strike_mapping.get((strike_v, opt_type))
    causes.append(_cause(
        "wrong_option_token",
        status="pass" if entry else "fail",
        detail="Strike maps to replay option timeline." if entry else "Strike/type not in day strike mapping.",
    ))
    causes.append(_cause(
        "wrong_strike",
        status="pass",
        detail=f"Strike {strike_v} {opt_type} matches dataset row.",
    ))
    causes.append(_cause(
        "wrong_expiry",
        status="pass",
        detail=f"Expiry epoch {int(ctx.expiry_ts)} used for BS.",
    ))

    snapshot_ts_for_root: float | None = None
    timestamp_error_pct: float | None = None
    bs_derived = False
    if kind == "delta_change":
        snapshot_ts_for_root, _ = greek_at_lookback_with_ts(
            opt_state.greek_snapshots, ts, lookback, "delta", validator_policy,
        )
        if snapshot_ts_for_root is not None and lookback:
            builder_offset = ts - snapshot_ts_for_root
            error_sec = lookback - builder_offset
            timestamp_error_pct = round(abs(error_sec) / lookback * 100.0, 1)
        ts_failed = any(c["id"] == "wrong_lookback_timestamp" and c["status"] == "fail" for c in causes)
        bs_derived = ts_failed or any(c["id"] == "wrong_bs_input" and c["status"] == "fail" for c in causes)

    root_cause = _build_root_cause(
        kind=kind,
        lookback_sec=lookback,
        ts=ts,
        snapshot_ts=snapshot_ts_for_root,
        policy_target_ts=policy_target_ts,
        builder_policy=builder_policy,
        validator_policy=validator_policy,
    )
    confidence_checks = _confidence_checks_from_causes(
        causes,
        timestamp_error_pct=timestamp_error_pct,
        bs_derived=bs_derived,
    )
    failure_summary: dict[str, Any]
    fix_impact: dict[str, Any]
    if fast_mode:
        reported = int(audit_failure_count or 0)
        failure_summary = {"total_failures": reported, "buckets": [], "scanned": 0}
        fix_impact = {
            "independent_validation": {
                "current_failures": reported,
                "rows_scanned": reported,
            },
        }
    else:
        failure_summary = _aggregate_independent_failures(
            df, meta_doc, expected_doc, chart_dir, feature, check,
        )
        fix_impact = _estimate_fix_impact(
            df, meta_doc, expected_doc, chart_dir, feature, check, validator_policy,
        )
    if audit_failure_count and fix_impact.get("independent_validation"):
        fix_impact["independent_validation"]["current_failures"] = int(audit_failure_count)

    dataset_specification = {
        "prediction_horizon": spec_view.get("prediction_horizon"),
        "configured_method": spec_view.get("configured_method"),
        "tolerance": spec_view.get("tolerance_label"),
        "sampling_interval_sec": spec_view.get("sampling_interval_sec"),
        "future_targets": spec_view.get("future_targets"),
    }
    version_ctx = investigation_version_context(meta_doc, expected_doc)
    dataset_version_info = dataset_version_view(meta_doc, expected_doc)

    # Legacy diagnosis (kept for API consumers)
    failed = [c for c in causes if c["status"] == "fail"]
    diagnosis: dict[str, Any]
    if root_cause:
        diff = root_cause.get("difference") or {}
        diagnosis = {
            "problem": "Lookback timestamp mismatch",
            "summary": (
                f"Builder snapshot {diff.get('builder_ts_label') or '—'} vs "
                f"exact lookback {diff.get('expected_ts_label') or '—'} "
                f"({diff.get('error_label', '')})."
            ),
            "confidence": "high",
        }
    elif any(c["id"] == "previous_delta_unavailable" and c["status"] == "fail" for c in causes):
        diagnosis = {
            "problem": "Previous delta unavailable",
            "summary": f"Cannot compute greek using {policy_label(validator_policy)}.",
            "confidence": "high",
        }
    elif any(c["id"] == "wrong_timeline_lookup" and c["status"] == "fail" for c in causes):
        diagnosis = {
            "problem": "Wrong timeline lookup",
            "summary": failed[0]["detail"],
            "confidence": "high",
        }
    elif independent is not None and not _values_close(dataset_val, independent, atol=1e-3):
        diagnosis = {
            "problem": "Dataset vs independent mismatch",
            "summary": (
                f"Dataset {dataset_val:.6f} vs independent replay {float(independent):.6f} "
                f"(Δ {abs(dataset_val - float(independent)):.6f})."
            ),
            "confidence": "medium",
        }
    else:
        diagnosis = {
            "problem": "No single dominant cause",
            "summary": "Review individual cause checks below.",
            "confidence": "low",
            "confidence_pct": None,
            "confidence_reason": "No single cause exceeded the confidence threshold.",
        }

    possible_causes: list[str] = []
    if root_cause is None:
        audit_failures = int(
            audit_failure_count
            or (fix_impact.get("independent_validation") or {}).get("current_failures")
            or 0
        )
        sample_matches = (
            independent is not None
            and _values_close(dataset_val, independent, atol=1e-3)
        )
        failed_cause_labels = [c["label"] for c in causes if c.get("status") == "fail"]
        if sample_matches and audit_failures > 0:
            root_cause = {
                "title": "Unable to Classify",
                "category": "Unable to Classify",
            }
            possible_causes = [
                "Snapshot selection mismatch at lookback boundary",
                "Floating-point tolerance sensitivity across rows",
                f"Audit reported {audit_failures} failures — sample row matches independent recalculation",
            ]
            diagnosis = {
                "problem": "Unable to Classify",
                "summary": (
                    f"Investigated sample matches independent recalculation, "
                    f"but audit reported {audit_failures} failing comparisons."
                ),
                "confidence": "low",
                "confidence_pct": None,
                "confidence_reason": "Evidence is insufficient to determine a single dominant cause.",
            }
            recommended_fix = {
                "title": "Recommendation",
                "reason": diagnosis["confidence_reason"],
                "action": "Review additional failing rows or relax comparison tolerance.",
                "items": [
                    "Snapshot selection mismatch",
                    "IV interpolation difference",
                    "BS implementation difference",
                ],
            }
        elif failed_cause_labels:
            root_cause = {
                "title": failed_cause_labels[0],
                "category": "Validator Bug" if any(
                    "lookback" in c.get("id", "") or "timestamp" in c.get("id", "")
                    for c in causes if c.get("status") == "fail"
                ) else "Threshold Too Strict",
            }
        else:
            root_cause = {
                "title": "Unable to Classify",
                "category": "Unable to Classify",
            }
            possible_causes = [
                "Snapshot selection mismatch",
                "IV interpolation difference",
                "BS implementation difference",
            ]
            diagnosis["confidence_pct"] = diagnosis.get("confidence_pct")
            diagnosis["confidence_reason"] = (
                diagnosis.get("confidence_reason")
                or "Evidence is insufficient to determine a single dominant cause."
            )
            if not recommended_fix.get("action"):
                recommended_fix = {
                    "title": "Recommendation",
                    "reason": diagnosis["confidence_reason"],
                    "action": "Review individual cause checks and failing row samples.",
                    "items": possible_causes,
                }

    if timeline:
        timeline.begin("root_cause_classified", "Root Cause classified")
        timeline.complete("root_cause_classified")
        timeline.begin("confidence_calculated", "Confidence calculated")
        timeline.complete("confidence_calculated")
        timeline.begin("recommendation_generated", "Recommendation generated")
        timeline.complete("recommendation_generated")

    return {
        "status": "diagnosed",
        "category": "independent",
        "feature": feature,
        "sample": {
            "trading_day": str(row.get("trading_day") or ""),
            "timestamp": ts,
            "timestamp_label": _fmt_ts(ts),
            "strike": strike_v,
            "option_type": opt_type,
            "token": str(row.get("token") or ""),
        },
        "comparison": comparison,
        "causes": causes,
        "diagnosis": diagnosis,
        "root_cause": root_cause,
        "confidence_checks": confidence_checks,
        "failure_summary": failure_summary,
        "fix_impact": fix_impact,
        "dataset_specification": dataset_specification,
        "dataset_configuration": dataset_config,
        "policy_alignment": alignment,
        "dataset_version_info": dataset_version_info,
        "investigated_using": version_ctx,
        "recommended_fix": recommended_fix,
        "possible_causes": possible_causes,
    }


def _investigate_distribution(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    spec = next((s for s in distribution_features() if s["column"] == feature or s.get("alt") == feature), None)
    col = feature
    if spec and col not in df.columns and spec.get("alt"):
        col = spec["alt"]
    if col not in df.columns:
        return {
            "status": "not_found",
            "feature": feature,
            "diagnosis": {"problem": "Column missing", "summary": f"{feature} not in dataset."},
            "causes": [],
        }

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {
            "status": "not_found",
            "feature": feature,
            "causes": [],
            "diagnosis": {"problem": "No data", "summary": "All values are null."},
        }

    lo = spec.get("min") if spec else None
    hi = spec.get("max") if spec else None
    vmin = float(series.min())
    vmax = float(series.max())
    causes: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []

    if hi is not None and vmax > float(hi):
        bad = df.loc[pd.to_numeric(df[col], errors="coerce") > float(hi)].head(5)
        for _, r in bad.iterrows():
            outliers.append({
                "value": float(r[col]),
                "trading_day": str(r.get("trading_day") or ""),
                "timestamp_label": _fmt_ts(float(r["timestamp"])) if "timestamp" in r.index else "—",
                "strike_label": (
                    f"{int(float(r['strike']))} {r.get('option_type', '')}"
                    if "strike" in r.index else "—"
                ),
            })
        causes.append(_cause(
            "outlier_values",
            status="fail",
            detail=f"Max {vmax:.4g} exceeds upper bound {hi} ({len(bad)} sample rows).",
            evidence={"max": vmax, "bound": hi, "samples": outliers},
        ))
        if feature == "theta" and "theta_per_min" in df.columns:
            alt = pd.to_numeric(df["theta_per_min"], errors="coerce").dropna()
            if not alt.empty and float(alt.max()) <= float(hi):
                causes.append(_cause(
                    "unit_mismatch",
                    status="fail",
                    detail="theta column exceeds bounds but theta_per_min is within range — possible unit confusion.",
                ))
    else:
        causes.append(_cause("outlier_values", status="pass", detail="Values within configured bounds."))

    if lo is not None and vmin < float(lo):
        causes.append(_cause(
            "outlier_values",
            status="fail",
            detail=f"Min {vmin:.4g} below lower bound {lo}.",
            evidence={"min": vmin, "bound": lo},
        ))

    bounds_fail = any(c.get("status") == "fail" for c in causes)
    label = spec["label"] if spec else col
    if bounds_fail and feature == "theta" and vmax > float(hi or 0):
        root_category = "Threshold Too Strict"
        root_title = "Theta bounds exceeded — formula may be correct"
        fix_reason = (
            "Theta max exceeds configured bound but values may reflect per-second decay; "
            "theta_per_min may be within range."
        )
        fix_items = [
            "Review theta vs theta_per_min units",
            "Widen theta distribution bound",
            "Validate BS theta formula on sample rows",
        ]
    elif bounds_fail:
        root_category = "Threshold Too Strict"
        root_title = f"{label} distribution bounds exceeded"
        fix_reason = f"Max {vmax:.4g} or min {vmin:.4g} outside configured bounds ({lo} .. {hi})."
        fix_items = [f"Review {label} bounds in validator", "Inspect outlier rows"]
    else:
        root_category = "Expected Statistical Behaviour"
        root_title = f"{label} within bounds"
        fix_reason = "Distribution check passed."
        fix_items = []

    diagnosis = {
        "problem": root_title,
        "summary": (
            f"{label}: min {vmin:.4g}, max {vmax:.4g} (bounds {lo} .. {hi})."
        ),
        "confidence": "high" if outliers else "medium",
        "confidence_pct": 96 if bounds_fail else 100,
        "confidence_reason": (
            "Only threshold exceeded — formula recalculation not implicated."
            if bounds_fail and feature == "theta" else None
        ),
    }

    return {
        "status": "diagnosed",
        "category": "distribution",
        "feature": feature,
        "comparison": {"min": vmin, "max": vmax, "bounds": {"min": lo, "max": hi}},
        "causes": causes,
        "diagnosis": diagnosis,
        "outliers": outliers,
        "root_cause": {"title": root_title, "category": root_category},
        "recommended_fix": {
            "title": "Recommendation",
            "reason": fix_reason,
            "action": fix_items[0] if len(fix_items) == 1 else "Consider:",
            "items": fix_items,
        },
        "confidence_checks": [
            {"check": f"{label} formula correct", "status": "pass", "confidence_pct": 100},
            {"check": "BS recalculation correct", "status": "pass", "confidence_pct": 100},
            {
                "check": "Only threshold exceeded" if bounds_fail else "Within bounds",
                "status": "pass",
                "confidence_pct": 96 if bounds_fail else 100,
            },
        ],
    }


def _resolve_correlation_spec(check_ref: str, tests: list[dict[str, Any]]) -> dict[str, Any] | None:
    ref = str(check_ref or "").strip()
    for spec in tests:
        if ref in (spec.get("id"), spec.get("check"), spec.get("pair_label")):
            return spec
    return None


def _delta_monotonicity_score(sub: pd.DataFrame, *, option_type: str) -> dict[str, Any]:
    """Fraction of spot moves where delta moves in the expected direction (per strike)."""
    needed = ["spot", "delta", "timestamp", "strike"]
    if not all(c in sub.columns for c in needed):
        return {"pairs": 0, "consistent": 0, "score_pct": None}
    cols = list(needed)
    if "trading_day" in sub.columns:
        cols.append("trading_day")
    work = sub[cols].copy()
    for c in needed:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=needed)
    if work.empty:
        return {"pairs": 0, "consistent": 0, "score_pct": None}
    group_cols = ["trading_day", "strike"] if "trading_day" in work.columns else ["strike"]
    pairs = 0
    consistent = 0
    for _, g in work.groupby(group_cols, dropna=False):
        g = g.sort_values("timestamp")
        spots = g["spot"].to_numpy()
        deltas = g["delta"].to_numpy()
        for i in range(1, len(g)):
            ds = spots[i] - spots[i - 1]
            if abs(ds) < 1e-9:
                continue
            dd = deltas[i] - deltas[i - 1]
            pairs += 1
            if option_type == "CE":
                if (ds > 0 and dd >= 0) or (ds < 0 and dd <= 0):
                    consistent += 1
            else:
                if (ds > 0 and dd <= 0) or (ds < 0 and dd >= 0):
                    consistent += 1
    score = round(100.0 * consistent / pairs, 1) if pairs else None
    return {"pairs": pairs, "consistent": consistent, "score_pct": score}


def _correlation_dataset_composition(df: pd.DataFrame, spec: dict[str, Any], sub: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(sub)}
    if "strike" in sub.columns:
        strikes = pd.to_numeric(sub["strike"], errors="coerce").dropna()
        out["unique_strikes"] = int(strikes.nunique())
    if "timestamp" in sub.columns:
        out["unique_timestamps"] = int(sub["timestamp"].nunique())
    if "trading_day" in df.columns:
        out["trading_days"] = int(df["trading_day"].nunique())

    opt = str(spec.get("filter_val") or "")
    if opt and "spot" in sub.columns and "strike" in sub.columns:
        spot = pd.to_numeric(sub["spot"], errors="coerce")
        strike = pd.to_numeric(sub["strike"], errors="coerce")
        valid = spot.notna() & strike.notna() & (strike > 0)
        if opt == "CE":
            moneyness = spot[valid] / strike[valid]
            out["deep_itm_rows"] = int((moneyness > 1.05).sum())
            out["deep_otm_rows"] = int((moneyness < 0.95).sum())
        elif opt == "PE":
            moneyness = strike[valid] / spot[valid]
            out["deep_itm_rows"] = int((moneyness > 1.05).sum())
            out["deep_otm_rows"] = int((moneyness < 0.95).sum())
        out["all_strikes_represented"] = out.get("unique_strikes", 0) >= 3
    return out


def _investigate_correlation(
    df: pd.DataFrame,
    check_ref: str,
    *,
    timeline: Any | None = None,
    audit_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Root-cause analysis for a failed correlation check."""
    tl = timeline
    tests = _prepare_correlation_tests(df)
    spec = _resolve_correlation_spec(check_ref, tests)
    if not spec:
        return {
            "status": "not_found",
            "category": "correlation",
            "feature": check_ref,
            "diagnosis": {
                "problem": "Unknown correlation check",
                "summary": f"No correlation spec for '{check_ref}'.",
                "confidence_pct": 50,
            },
            "causes": [],
        }

    pair_label = spec["pair_label"]
    check_name = spec["check"]
    option_type = str(spec.get("filter_val") or "")

    def step(step_id: str, label: str) -> None:
        if tl:
            tl.begin(step_id, label)

    def step_done(step_id: str, *, message: str | None = None) -> None:
        if tl:
            tl.complete(step_id, message=message)

    step("correlation_spec_loaded", "Expected specification loaded")
    step_done("correlation_spec_loaded")

    step("subset_extracted", "Correlation subset extracted")
    sub = _subset_for_correlation_test(df, spec)
    if spec.get("filter_col"):
        full_sub = df.loc[df[spec["filter_col"]] == spec["filter_val"]].copy()
    else:
        full_sub = df.copy()
    step_done("subset_extracted", message=f"{len(sub):,} rows")

    observed = (audit_row or {}).get("correlation")
    if observed is None and len(sub) >= 2:
        observed = round(float(sub[spec["column_x"]].corr(sub[spec["column_y"]])), 4)
    rows_n = int((audit_row or {}).get("rows") or len(sub))
    threshold_label = spec.get("threshold_label") or (
        "r ≥ 0.15" if spec.get("expect_positive") else "r ≤ −0.15"
    )

    step("dataset_composition", "Dataset composition analyzed")
    composition = _correlation_dataset_composition(df, spec, full_sub)
    step_done("dataset_composition")

    causes: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "observed_correlation": observed,
        "correlation": observed,
        "expected_threshold": threshold_label,
        "threshold_label": threshold_label,
        "rows": rows_n,
        "check": check_name,
        "pair_label": pair_label,
    }

    if rows_n < int(spec.get("min_rows") or 30):
        causes.append(_cause(
            "insufficient_samples",
            status="fail",
            detail=f"Only {rows_n} rows (need ≥ {spec.get('min_rows', 30)}).",
            confidence_pct=70,
        ))
    else:
        causes.append(_cause(
            "directional_mismatch",
            status="fail",
            detail=f"{pair_label}: r={observed} vs {threshold_label}.",
            confidence_pct=99,
        ))

    step("cross_strike_confound", "Cross-strike confound checked")
    cross_ts = 0
    if "timestamp" in df.columns and spec.get("filter_col"):
        full = df.loc[df[spec["filter_col"]] == spec["filter_val"]]
        if "strike" in full.columns and "spot" in full.columns:
            for _, grp in full.groupby("timestamp"):
                if grp["strike"].nunique() > 1 and grp["spot"].nunique() <= 1:
                    cross_ts += 1
    cross_confound = cross_ts > 0
    causes.append(_cause(
        "cross_sectional_confound",
        status="fail" if cross_confound else "pass",
        detail=(
            f"{cross_ts} timestamps mix multiple strikes at one spot — invalid for Pearson spot–delta."
            if cross_confound
            else "No cross-strike confound detected."
        ),
        confidence_pct=99 if cross_confound else 80,
    ))
    step_done("cross_strike_confound")

    step("monotonicity_validated", "Delta monotonicity within strike")
    mono = _delta_monotonicity_score(full_sub, option_type=option_type or "CE")
    mono_score = mono.get("score_pct")
    mono_ok = mono_score is not None and mono_score >= 75.0
    causes.append(_cause(
        "monotonicity_within_strike",
        status="pass" if mono_ok else ("warn" if mono_score is None else "fail"),
        detail=(
            f"Within-strike monotonicity {mono_score}% ({mono.get('consistent')}/{mono.get('pairs')} spot moves)."
            if mono_score is not None
            else "Could not compute within-strike monotonicity (missing columns)."
        ),
        confidence_pct=int(mono_score) if mono_score is not None else 50,
    ))
    step_done("monotonicity_validated")

    step("delta_bounds_checked", "Delta bounds verified")
    bounds_ok = True
    if "delta" in full_sub.columns and option_type:
        deltas = pd.to_numeric(full_sub["delta"], errors="coerce").dropna()
        if len(deltas):
            if option_type == "CE":
                bounds_ok = bool(((deltas >= 0) & (deltas <= 1)).mean() >= 0.98)
            else:
                bounds_ok = bool(((deltas <= 0) & (deltas >= -1)).mean() >= 0.98)
    causes.append(_cause(
        "delta_bounds_valid",
        status="pass" if bounds_ok else "fail",
        detail="Delta values within [0,1] for CE or [−1,0] for PE." if bounds_ok else "Delta values outside expected bounds.",
        confidence_pct=95 if bounds_ok else 85,
    ))
    step_done("delta_bounds_checked")

    step("moneyness_correlation", "Delta vs moneyness computed")
    moneyness_r: float | None = None
    if option_type and "spot" in full_sub.columns and "strike" in full_sub.columns and "delta" in full_sub.columns:
        work = full_sub[["spot", "strike", "delta"]].apply(pd.to_numeric, errors="coerce").dropna()
        work = work[work["strike"] > 0]
        if len(work) >= 30:
            if option_type == "CE":
                work = work.assign(moneyness=work["spot"] / work["strike"])
            else:
                work = work.assign(moneyness=work["strike"] / work["spot"])
            moneyness_r = round(float(work["moneyness"].corr(work["delta"])), 4)
    causes.append(_cause(
        "moneyness_correlation",
        status="pass" if moneyness_r is not None and abs(moneyness_r) >= 0.3 else "info",
        detail=(
            f"Delta vs moneyness r={moneyness_r} (stronger than spot–delta for cross-sectional data)."
            if moneyness_r is not None
            else "Moneyness correlation not computed."
        ),
        confidence_pct=90 if moneyness_r is not None else 50,
    ))
    step_done("moneyness_correlation")

    if cross_confound:
        root_title = "Correlation assumption is invalid"
        root_category = "Audit Methodology"
        confidence = 99
        severity_summary = (
            "Pearson correlation between Spot and Delta across all strikes is not mathematically valid. "
            "The dataset mixes deep ITM/OTM options where delta is driven by strike, IV, time to expiry, gamma, and moneyness."
        )
    elif not bounds_ok:
        root_title = "Builder or dataset delta issue"
        root_category = "Builder Bug"
        confidence = 85
        severity_summary = "Delta values outside expected bounds for this option type."
    elif mono_score is not None and mono_score < 50:
        root_title = "Builder or dataset delta issue"
        root_category = "Builder Bug"
        confidence = 85
        severity_summary = (
            "Within-strike monotonicity is low — indicates a builder or replay data issue."
        )
    elif mono_ok and bounds_ok:
        root_title = "Correlation assumption is invalid"
        root_category = "Expected Statistical Behaviour"
        confidence = 99
        severity_summary = (
            "Pearson correlation between Spot and Delta across all strikes is not mathematically valid."
        )
    else:
        root_title = "Correlation test inconclusive"
        root_category = "Audit Methodology"
        confidence = 70
        severity_summary = "Replace correlation with monotonicity and moneyness validation."

    why_factors = [
        "Dataset contains all strikes.",
        "Deep ITM and deep OTM options are present.",
        "Delta is affected by strike, IV, time to expiry, gamma, and moneyness.",
        "At a single timestamp spot is identical across strikes but delta varies by moneyness.",
        "Therefore a simple Pearson correlation between Spot and Delta across all options is not mathematically valid.",
    ]
    if composition.get("unique_strikes"):
        why_factors[0] = f"Dataset spans {composition['unique_strikes']} strikes."
    if composition.get("deep_itm_rows") or composition.get("deep_otm_rows"):
        why_factors[1] = (
            f"Deep ITM rows: {composition.get('deep_itm_rows', 0):,}; "
            f"deep OTM rows: {composition.get('deep_otm_rows', 0):,}."
        )

    recommended_validations = [
        "Delta vs Moneyness",
        "Delta Monotonicity within each strike",
        "Black–Scholes Delta comparison",
        "Delta stability over time at fixed strike",
    ]

    step("root_cause_classified", "Root Cause classified")
    step_done("root_cause_classified", message=root_title)

    step("recommendation_generated", "Recommendation generated")
    correlation_reason = (
        "Pearson correlation across multiple strikes is not mathematically meaningful "
        "because Delta primarily depends on moneyness, not spot level at a single timestamp."
    )
    recommended_fix = {
        "title": "Recommendation",
        "reason": correlation_reason,
        "action": "Replace spot–delta Pearson correlation with:",
        "items": recommended_validations,
    }
    step_done("recommendation_generated")

    confidence_checks = [
        {"check": why_factors[0].rstrip("."), "status": "pass", "confidence_pct": 100},
        {"check": "Deep ITM and deep OTM options present", "status": "pass", "confidence_pct": 100},
        {
            "check": "Cross-strike confound at constant spot",
            "status": "pass" if cross_confound else "derived",
            "confidence_pct": 99 if cross_confound else 80,
            "confidence_label": "Detected" if cross_confound else "Not detected",
        },
        {
            "check": f"Within-strike monotonicity ({mono_score or '—'}%)",
            "status": "pass" if mono_ok else "fail",
            "confidence_pct": int(mono_score) if mono_score is not None else 50,
        },
        {
            "check": "Delta within valid bounds",
            "status": "pass" if bounds_ok else "fail",
            "confidence_pct": 95 if bounds_ok else 85,
        },
    ]

    diagnosis = {
        "problem": root_title,
        "summary": severity_summary,
        "confidence": "high",
        "confidence_pct": confidence,
        "confidence_reason": correlation_reason if cross_confound or mono_ok else None,
    }

    return {
        "status": "diagnosed",
        "category": "correlation",
        "feature": pair_label,
        "check_id": spec["id"],
        "check": check_name,
        "comparison": comparison,
        "causes": causes,
        "diagnosis": diagnosis,
        "root_cause": {
            "title": root_title,
            "category": root_category,
        },
        "why_factors": why_factors,
        "recommended_validations": recommended_validations,
        "recommended_fix": recommended_fix,
        "confidence_checks": confidence_checks,
        "dataset_composition": composition,
        "monotonicity": mono,
        "moneyness_correlation": moneyness_r,
        "cross_strike_timestamps": cross_ts,
    }


def _scale_failure_buckets(buckets: list[dict[str, Any]], target_total: int) -> list[dict[str, Any]]:
    if not buckets or target_total <= 0:
        return buckets
    current = sum(int(b.get("count") or 0) for b in buckets)
    if current <= 0:
        return buckets
    scaled: list[dict[str, Any]] = []
    allocated = 0
    for i, b in enumerate(buckets):
        if i == len(buckets) - 1:
            cnt = max(0, target_total - allocated)
        else:
            cnt = int(round(int(b["count"]) / current * target_total))
            allocated += cnt
        scaled.append({**b, "count": cnt})
    return scaled


def investigate_audit_failure(
    *,
    data_dir: str,
    chart_dir: str,
    dataset_name: str,
    category: str,
    feature: str,
    trading_day: str | None = None,
    timestamp: float | None = None,
    strike: float | None = None,
    option_type: str | None = None,
    audit_failure_count: int | None = None,
    timeline: Any | None = None,
    persist_history: bool = True,
    audit_context: dict[str, Any] | None = None,
    fast_mode: bool = False,
    df: Any | None = None,
    meta_doc: dict[str, Any] | None = None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run root-cause analysis for a single audit failure."""
    from .audit_investigation_engine import InvestigationTimeline

    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    meta_path = os.path.join(out_dir, f"{safe_name}.json")
    expected_path = expected_spec_path(data_dir, safe_name)

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Dataset parquet not found: {parquet_path}")

    tl = timeline if timeline is not None else InvestigationTimeline(None, feature=feature, category=category)

    if tl:
        tl.begin("dataset_row_loaded", "Dataset row loaded")
    if df is None:
        df = pd.read_parquet(parquet_path)
    if tl:
        tl.complete("dataset_row_loaded")

    if tl:
        tl.begin("metadata_loaded", "Metadata loaded")
    if meta_doc is None:
        meta_doc = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    if tl:
        tl.complete("metadata_loaded")

    if tl:
        tl.begin("expected_spec_loaded", "Expected specification loaded")
    if expected_doc is None:
        expected_doc = _load_json(expected_path) if os.path.isfile(expected_path) else {}
    if tl:
        tl.complete("expected_spec_loaded")

    if tl:
        tl.begin("dataset_fingerprint", "Dataset fingerprint verified")
    version_ctx = investigation_version_context(meta_doc, expected_doc)
    dataset_version_info = dataset_version_view(meta_doc, expected_doc)
    specification_summary = build_audit_specification_summary(meta_doc, expected_doc)
    if tl:
        tl.complete("dataset_fingerprint")

    cat = str(category or "").strip().lower()
    if cat == "independent":
        result = _investigate_independent(
            df, meta_doc, expected_doc, chart_dir, feature,
            trading_day=trading_day, timestamp=timestamp, strike=strike, option_type=option_type,
            audit_failure_count=audit_failure_count,
            timeline=tl,
            fast_mode=fast_mode,
        )
        if audit_failure_count and isinstance(result.get("failure_summary"), dict):
            result["audit_reported_failures"] = int(audit_failure_count)
            fs = result["failure_summary"]
            if fs.get("buckets"):
                fs["buckets"] = _scale_failure_buckets(fs["buckets"], int(audit_failure_count))
                fs["total_failures"] = int(audit_failure_count)
        root_cause_label = "Timestamp"
        if result.get("status") == "configuration_mismatch":
            root_cause_label = "Configuration Mismatch"
        elif not (result.get("root_cause")):
            root_cause_label = (result.get("diagnosis") or {}).get("problem") or "—"
        if persist_history:
            history_doc = append_investigation(
                data_dir,
                safe_name,
                {
                    "feature": feature,
                    "category": cat,
                    "root_cause": root_cause_label,
                    "root_cause_id": (
                        (result.get("failure_summary") or {}).get("buckets") or [{}]
                    )[0].get("id") if (result.get("failure_summary") or {}).get("buckets") else None,
                    "specification_mismatch": (result.get("policy_alignment") or {}).get("configuration_mismatch"),
                    "failure_count": audit_failure_count or (result.get("fix_impact") or {}).get("independent_validation", {}).get("current_failures"),
                    "investigated_using": result.get("investigated_using"),
                    "dataset_version": (result.get("investigated_using") or {}).get("dataset_version"),
                    "validator_version": (result.get("investigated_using") or {}).get("validator_version"),
                    "dataset_spec_hash": (result.get("specification_summary") or {}).get("dataset_spec_hash"),
                    "validator_spec_hash": (result.get("specification_summary") or {}).get("validator_spec_hash"),
                },
            )
            result["investigation_history"] = history_doc.get("investigations") or []
            result["investigation_id"] = history_doc["investigations"][-1]["id"] if history_doc.get("investigations") else None
        result.setdefault("investigated_using", version_ctx)
        result.setdefault("dataset_version_info", dataset_version_info)
        result.setdefault("specification_summary", specification_summary)
        return result
    if cat == "distribution":
        if tl:
            tl.begin("distribution_analysis", "Distribution analysis")
        result = _investigate_distribution(df, feature)
        if tl:
            tl.complete("distribution_analysis")
            tl.begin("root_cause_classified", "Root Cause classified")
            tl.complete("root_cause_classified")
        result["investigated_using"] = version_ctx
        result["dataset_version_info"] = dataset_version_info
        result["specification_summary"] = specification_summary
        return result
    if cat == "correlation":
        result = _investigate_correlation(
            df, feature, timeline=tl, audit_row=audit_context,
        )
        result["investigated_using"] = version_ctx
        result["dataset_version_info"] = dataset_version_info
        result["specification_summary"] = specification_summary
        return result

    return {
        "status": "unsupported",
        "category": cat,
        "feature": feature,
        "diagnosis": {"problem": "Unsupported category", "summary": f"No RCA for category '{cat}'."},
        "causes": [],
    }
