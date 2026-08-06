"""Recalculate features on sample rows to validate formula correctness."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name
from chain_replay_ml.features_atm_band import find_atm_strike

from .audit_diagnostics import _horizon_sec_from_column
from .chain_maps import precompute_chain_maps
from .day_context import DayContext, SourceSpec, load_day_context
from .extended_features import OptionFeatureState
from .lookback_policy import read_dataset_configuration, lookback_policy
from .feature_plugins import horizon_column_name
from .rolling_controllers import SpotControllers
from .writer import normalize_days_meta

SPOT_CHECK_FEATURES = [
    "delta",
    "gamma",
    "theta",
    "vega",
    "current_iv",
    "moneyness",
    "oi",
    "future_ltp_3s",
    "future_ltp_5s",
    "future_ltp_10s",
    "future_ltp_30s",
    "future_ltp_1m",
    "future_ltp_3m",
    "future_ltp_5m",
]

_COMPARE_ATOL = {
    "delta": 1e-4,
    "gamma": 1e-6,
    "theta": 1e-3,
    "vega": 1e-3,
    "current_iv": 0.05,
    "moneyness": 1e-4,
    "oi": 1.0,
}
_DEFAULT_ATOL = 0.01
_DEFAULT_RTOL = 1e-3


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _values_close(feature: str, expected: Any, actual: Any, *, tolerance: float | None = None) -> tuple[bool, float | None]:
    if _is_null(expected) and _is_null(actual):
        return True, 0.0
    if _is_null(expected) or _is_null(actual):
        return False, None
    try:
        a = float(expected)
        b = float(actual)
    except (TypeError, ValueError):
        return expected == actual, None
    diff = abs(a - b)
    atol = float(tolerance) if tolerance is not None else _COMPARE_ATOL.get(feature, _DEFAULT_ATOL)
    rtol = _DEFAULT_RTOL
    ok = diff <= atol + rtol * max(abs(a), abs(b), 1e-9)
    return ok, diff


def _recompute_row(
    *,
    ctx: DayContext,
    chain_maps: Any,
    opt_state: OptionFeatureState,
    strike_step: int,
    ts: float,
    strike: float,
    option_type: str,
    token: str,
    opt_tl: Any,
    enabled_groups: list[str],
    horizons_sec: list[int],
    lookback_policy_doc: dict[str, Any] | None = None,
    spot_controllers: SpotControllers | None = None,
) -> dict[str, Any]:
    from .registry_features import build_registry_features_at_ts

    spot = ctx.index_tl.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return {}
    atm = find_atm_strike(spot, strike_step)
    picked = build_registry_features_at_ts(
        ts=float(ts),
        strike=float(strike),
        option_type=str(option_type),
        opt_tl=opt_tl,
        index_tl=ctx.index_tl,
        strike_mapping=ctx.strike_mapping,
        chain_maps=chain_maps,
        opt_state=opt_state,
        strike_step=int(strike_step),
        expiry_ts=float(ctx.expiry_ts),
        open_ts=float(ctx.open_ts),
        close_ts=float(ctx.close_ts),
        enabled_groups=enabled_groups,
        trading_day=str(ctx.source.trading_day),
        expiry_norm=str(ctx.expiry_norm),
        lookback_policy_doc=lookback_policy_doc,
        atm_strike=atm,
        spot_controllers=spot_controllers,
    )
    for h in horizons_sec:
        col = horizon_column_name(h)
        picked[col] = opt_tl.ltp_rupees_at(ts + float(h)) if opt_tl else None
    return picked


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def validate_dataset_features(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    registry: dict[str, Any],
    enabled_groups: list[str],
    target_columns: list[str],
    n_sample: int = 100,
    tolerance: float = 1e-6,
    on_progress: Any | None = None,
    max_failures: int = 50,
) -> dict[str, Any]:
    """Validate features by recomputing on a random sample with optional progress callbacks."""
    if df.empty:
        return {
            "status": "fail",
            "label": "FAIL",
            "rows_checked": 0,
            "features_checked": 0,
            "comparisons": 0,
            "comparisons_failed": 0,
            "failures": 0,
            "max_difference": None,
            "groups": [],
            "spot_checks": [],
            "failure_samples": [],
        }

    horizons_sec = [_horizon_sec_from_column(c) for c in target_columns if _horizon_sec_from_column(c) > 0]
    lb_policy_doc = lookback_policy(read_dataset_configuration(meta_doc, None))
    groups_meta = registry.get("groups") or {}
    sample_n = min(max(1, int(n_sample)), len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    sample_idx = set(sample_df.index.tolist())
    sample_order = list(sample_df.index)

    ctx_cache: dict[str, DayContext] = {}
    chain_cache: dict[str, Any] = {}
    opt_states: dict[str, OptionFeatureState] = {}
    spot_controllers_by_day: dict[str, SpotControllers] = {}

    group_stats: dict[str, dict[str, int]] = {
        gid: {"checked": 0, "passed": 0, "failed": 0}
        for gid in enabled_groups
    }
    spot_accum: dict[str, list[tuple[float, float, float]]] = {f: [] for f in SPOT_CHECK_FEATURES}
    failure_samples: list[dict[str, Any]] = []
    feature_stats: dict[str, dict[str, Any]] = {}
    features_seen: set[str] = set()
    total_checked = 0
    total_failed = 0
    max_diff: float | None = None
    rows_done = 0

    def _emit_progress(**kwargs: Any) -> None:
        if on_progress:
            on_progress({
                "rows_done": rows_done,
                "rows_total": sample_n,
                **kwargs,
            })

    token_groups = sample_df.groupby(["trading_day", "token"], dropna=False).groups

    days_meta = normalize_days_meta(meta_doc)
    days_by_trading_day = {str(d.get("trading_day")): d for d in days_meta if d.get("trading_day")}

    for (trading_day, token), _ in token_groups.items():
        day = str(trading_day)
        tok = str(token)
        if day not in ctx_cache:
            day_info = days_by_trading_day.get(day)
            if not day_info:
                continue
            expiry = str(day_info.get("expiry") or "").strip()
            if not expiry and "expiry" in df.columns:
                exp_vals = df.loc[df["trading_day"] == trading_day, "expiry"].dropna().unique()
                if len(exp_vals):
                    expiry = str(exp_vals[0])
            try:
                ctx_cache[day] = load_day_context(
                    chart_dir,
                    SourceSpec(
                        source_id=str(day_info.get("source_id") or day),
                        trading_day=day,
                        market=str(day_info.get("market") or "NIFTY"),
                        expiry=expiry,
                    ),
                )
            except Exception:
                continue

        ctx = ctx_cache[day]
        index_key = normalize_index_name(ctx.source.market)
        strike_step = STRIKE_STEP.get(index_key, 50)

        token_df = df[(df["trading_day"] == trading_day) & (df["token"] == tok)].sort_values("timestamp")
        if day not in chain_cache:
            all_day_ts = sorted(df.loc[df["trading_day"] == trading_day, "timestamp"].unique())
            chain_cache[day] = precompute_chain_maps(
                index_tl=ctx.index_tl,
                strike_mapping=ctx.strike_mapping,
                timestamps=all_day_ts,
                strike_step=strike_step,
            )
        chain_maps = chain_cache[day]
        state_key = f"{day}:{tok}"
        if state_key not in opt_states:
            opt_states[state_key] = OptionFeatureState()

        if day not in spot_controllers_by_day:
            spot_controllers_by_day[day] = SpotControllers()
        spot_ctrl = spot_controllers_by_day[day]

        for idx, row in token_df.iterrows():
            strike_r = float(row["strike"])
            opt_type = str(row["option_type"])
            entry = ctx.strike_mapping.get((strike_r, opt_type))
            if not entry:
                continue
            _tok, _sym, opt_tl = entry
            ts = float(row["timestamp"])
            recomputed = _recompute_row(
                ctx=ctx,
                chain_maps=chain_maps,
                opt_state=opt_states[state_key],
                strike_step=strike_step,
                ts=ts,
                strike=strike_r,
                option_type=opt_type,
                token=tok,
                opt_tl=opt_tl,
                enabled_groups=enabled_groups,
                horizons_sec=horizons_sec,
                lookback_policy_doc=lb_policy_doc,
                spot_controllers=spot_ctrl,
            )
            if idx not in sample_idx:
                continue

            current_feature = None
            for gid in enabled_groups:
                feats = list((groups_meta.get(gid) or {}).get("features") or [])
                for feat in feats:
                    if feat not in df.columns or feat not in recomputed:
                        continue
                    expected = row[feat]
                    actual = recomputed.get(feat)
                    if _is_null(expected):
                        continue
                    current_feature = feat
                    features_seen.add(feat)
                    group_stats[gid]["checked"] += 1
                    total_checked += 1
                    ok, diff = _values_close(feat, expected, actual, tolerance=tolerance)
                    atol = float(tolerance) if tolerance is not None else _COMPARE_ATOL.get(feat, _DEFAULT_ATOL)
                    fstat = feature_stats.setdefault(
                        feat,
                        {"checked": 0, "failed": 0, "max_diff": 0.0, "atol": atol, "group": gid},
                    )
                    fstat["checked"] += 1
                    if diff is not None:
                        max_diff = diff if max_diff is None else max(max_diff, diff)
                        if diff > float(fstat.get("max_diff") or 0):
                            fstat["max_diff"] = float(diff)
                    if ok:
                        group_stats[gid]["passed"] += 1
                    else:
                        group_stats[gid]["failed"] += 1
                        fstat["failed"] += 1
                        total_failed += 1
                        if len(failure_samples) < max_failures:
                            failure_samples.append({
                                "feature": feat,
                                "dataset": float(expected) if not _is_null(expected) else None,
                                "expected": float(actual) if not _is_null(actual) else None,
                                "difference": diff,
                                "trading_day": day,
                                "timestamp": ts,
                                "timestamp_label": _fmt_ts(ts),
                                "strike": strike_r,
                                "option_type": opt_type,
                                "strike_label": f"{int(strike_r) if strike_r == int(strike_r) else strike_r} {opt_type}",
                            })
                    if feat in spot_accum and len(spot_accum[feat]) < 1 and not _is_null(expected):
                        try:
                            spot_accum[feat].append((
                                float(expected),
                                float(actual) if not _is_null(actual) else float("nan"),
                                float(diff or 0),
                            ))
                        except (TypeError, ValueError):
                            pass
                    _emit_progress(
                        trading_day=day,
                        timestamp=ts,
                        timestamp_label=_fmt_ts(ts),
                        strike=strike_r,
                        option_type=opt_type,
                        strike_label=f"{int(strike_r) if strike_r == int(strike_r) else strike_r} {opt_type}",
                        feature=current_feature,
                    )

            rows_done += 1
            _emit_progress(
                trading_day=day,
                timestamp=ts,
                timestamp_label=_fmt_ts(ts),
                strike=strike_r,
                option_type=opt_type,
                strike_label=f"{int(strike_r) if strike_r == int(strike_r) else strike_r} {opt_type}",
                feature=current_feature,
            )

    group_rows = []
    for gid in enabled_groups:
        st = group_stats.get(gid, {})
        checked = int(st.get("checked") or 0)
        failed = int(st.get("failed") or 0)
        if checked == 0:
            status = "info"
        elif failed == 0:
            status = "pass"
        elif failed < checked * 0.05:
            status = "warn"
        else:
            status = "fail"
        group_rows.append({
            "id": gid,
            "label": str((groups_meta.get(gid) or {}).get("label") or gid),
            "checked": checked,
            "failed": failed,
            "status": status,
        })

    spot_checks = []
    for feat in SPOT_CHECK_FEATURES:
        if feat not in df.columns:
            continue
        examples = spot_accum.get(feat) or []
        if examples:
            exp, act, diff = examples[0]
            ok = not math.isnan(act) and _values_close(feat, exp, act, tolerance=tolerance)[0]
            spot_checks.append({
                "feature": feat,
                "dataset": exp,
                "recalculated": act if not math.isnan(act) else None,
                "difference": diff,
                "status": "pass" if ok else "fail",
            })
        else:
            spot_checks.append({
                "feature": feat,
                "dataset": None,
                "recalculated": None,
                "difference": None,
                "status": "info",
            })

    if total_checked == 0:
        status = "warn"
    elif total_failed == 0:
        status = "pass"
    elif total_failed < total_checked * 0.02:
        status = "warn"
    else:
        status = "fail"

    failure_by_feature = sorted(
        [
            {
                "feature": feat,
                "group": st.get("group"),
                "failures": int(st["failed"]),
                "checked": int(st["checked"]),
                "max_difference": round(float(st["max_diff"]), 6) if st.get("max_diff") else 0.0,
                "atol": float(st.get("atol") or _DEFAULT_ATOL),
                "failure_rate_pct": round(100.0 * int(st["failed"]) / max(int(st["checked"]), 1), 2),
            }
            for feat, st in feature_stats.items()
            if int(st.get("failed") or 0) > 0
        ],
        key=lambda r: (-int(r["failures"]), -float(r["max_difference"])),
    )

    return {
        "status": status,
        "label": "PASS" if status == "pass" else ("WARN" if status == "warn" else "FAIL"),
        "rows_checked": rows_done,
        "features_checked": len(features_seen),
        "comparisons": total_checked,
        "comparisons_failed": total_failed,
        "failures": total_failed,
        "max_difference": max_diff,
        "groups": group_rows,
        "spot_checks": spot_checks,
        "failure_samples": failure_samples,
        "failure_by_feature": failure_by_feature,
    }


def audit_formula_recalc(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    *,
    chart_dir: str,
    registry: dict[str, Any],
    enabled_groups: list[str],
    target_columns: list[str],
    n_sample: int = 100,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate features by recomputing on a random sample (audit dialog)."""
    return validate_dataset_features(
        df,
        meta_doc,
        chart_dir=chart_dir,
        registry=registry,
        enabled_groups=enabled_groups,
        target_columns=target_columns,
        n_sample=n_sample,
        tolerance=None,
        on_progress=on_progress,
    )
