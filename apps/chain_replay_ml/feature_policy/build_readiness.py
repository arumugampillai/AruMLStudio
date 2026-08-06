"""Enforce Feature Policy readiness during dataset generation — NULL when NOT READY."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .engine import FeaturePolicyEngine
from .readiness_profiler import profiler_active
from .registry import FeaturePolicyRegistry, load_feature_policy_registry
from .types import DEFAULT_GAP_MAX_SEC, FeatureCategory

FEATURE_READINESS_POLICY_VERSION = "1"


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def build_feature_readiness_manifest(
    *,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    rolling_enforcement: bool = True,
    not_ready_output: str = "NULL",
    dependency_propagation: bool = True,
) -> dict[str, Any]:
    return {
        "version": FEATURE_READINESS_POLICY_VERSION,
        "gap_max_sec": float(gap_max_sec),
        "rolling_enforcement": bool(rolling_enforcement),
        "not_ready_output": not_ready_output,
        "dependency_propagation": bool(dependency_propagation),
        "engine": "FeaturePolicyEngine",
    }


def enforce_readiness_on_rows(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    sampling_interval_sec: float,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    session_key: str = "trading_day",
    readiness_profile: bool = False,
) -> dict[str, Any]:
    """Null-out feature columns that are NOT READY per FeaturePolicyEngine.

    Advances the engine once per unique timestamp (session-scoped), matching the
    warm-up simulator. Every non-NULL written value is therefore mature.
    """
    names = list(dict.fromkeys(n for n in feature_names if n and not str(n).startswith("__roll.")))
    if not rows or not names:
        return _empty_enforcement_stats()

    if readiness_profile:
        from .readiness_profiler import start_readiness_profiler

        if not profiler_active():
            start_readiness_profiler(
                gap_max_sec=float(gap_max_sec),
                feature_count=len(names),
                row_count=len(rows),
            )

    reg = load_feature_policy_registry(feature_names=names)
    eng = FeaturePolicyEngine(
        reg,
        sampling_interval_sec=float(sampling_interval_sec),
        gap_max_sec=float(gap_max_sec),
        reset_on_gap_enabled=True,
    )

    stats = _empty_enforcement_stats()
    stats["feature_count"] = len(names)
    stats["gap_max_sec"] = float(gap_max_sec)

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get(session_key) or "__session__")].append(row)

    for session_rows in by_session.values():
        eng.on_session_start()
        by_ts: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in session_rows:
            by_ts[float(row["timestamp"])].append(row)
        for ts in sorted(by_ts.keys()):
            eng.on_sample(ts)
            ready_at_ts = {feat: eng.is_ready(feat) for feat in names}
            for row in by_ts[ts]:
                stats["enforced_rows"] += 1
                for feat in names:
                    if feat not in row:
                        continue
                    before = row[feat]
                    if ready_at_ts.get(feat):
                        after = before
                    else:
                        after = None
                        meta = reg.get(feat)
                        if meta and meta.feature_category == FeatureCategory.DERIVED:
                            eng._stats.derived_null_propagations += 1
                        elif meta and meta.feature_category == FeatureCategory.ROLLING:
                            eng._stats.rolling_not_ready_outputs += 1
                    if after is None and not _is_null(before):
                        stats["nulled_cells"] += 1
                        meta = reg.get(feat)
                        if meta and meta.feature_category == FeatureCategory.ROLLING:
                            stats["rolling_nulled"] += 1
                        elif meta and meta.feature_category == FeatureCategory.DERIVED:
                            stats["derived_nulled"] += 1
                    row[feat] = after

    eng_stats = eng.stats_dict()
    stats["gap_resets"] = eng_stats.get("gap_resets", 0)
    stats["rolling_not_ready_outputs"] = eng_stats.get("rolling_not_ready_outputs", 0)
    stats["derived_null_propagations"] = eng_stats.get("derived_null_propagations", 0)
    return stats


def validate_readiness_compliance(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    sampling_interval_sec: float,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    session_key: str = "trading_day",
    registry: FeaturePolicyRegistry | None = None,
    readiness_profile: bool = False,
) -> dict[str, Any]:
    """Post-build audit: count rows where NOT READY features still have numeric values."""
    names = list(dict.fromkeys(n for n in feature_names if n and not str(n).startswith("__roll.")))
    if not rows or not names:
        return _compliance_report(0, 0, {}, total_rows=0)

    if readiness_profile and not profiler_active():
        from .readiness_profiler import start_readiness_profiler

        start_readiness_profiler(
            gap_max_sec=float(gap_max_sec),
            feature_count=len(names),
            row_count=len(rows),
        )

    reg = registry or load_feature_policy_registry(feature_names=names)
    eng = FeaturePolicyEngine(
        reg,
        sampling_interval_sec=float(sampling_interval_sec),
        gap_max_sec=float(gap_max_sec),
        reset_on_gap_enabled=True,
    )

    violations = 0
    derived_violations = 0
    per_feature: dict[str, int] = defaultdict(int)

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get(session_key) or "__session__")].append(row)

    for session_rows in by_session.values():
        eng.on_session_start()
        by_ts: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in session_rows:
            by_ts[float(row["timestamp"])].append(row)
        for ts in sorted(by_ts.keys()):
            eng.on_sample(ts)
            ready_at_ts = {feat: eng.is_ready(feat) for feat in names}
            for row in by_ts[ts]:
                for feat in names:
                    if feat not in row:
                        continue
                    if ready_at_ts.get(feat):
                        continue
                    if _is_null(row[feat]):
                        continue
                    violations += 1
                    per_feature[feat] += 1
                    meta = reg.get(feat)
                    if meta and meta.feature_category == FeatureCategory.DERIVED:
                        derived_violations += 1

    report = _compliance_report(violations, derived_violations, dict(per_feature), total_rows=len(rows))
    return report


def _compliance_report(
    violations: int,
    derived_violations: int,
    per_feature: dict[str, int],
    *,
    total_rows: int,
) -> dict[str, Any]:
    compliance_pct = 100.0 if violations == 0 else max(0.0, 100.0 * (1.0 - violations / max(total_rows, 1)))
    checks: list[dict[str, Any]] = []

    def _anchor_check(anchor_suffix: str, label: str) -> None:
        immature = sum(
            cnt for feat, cnt in per_feature.items()
            if anchor_suffix in feat.lower() or f"ema{anchor_suffix}" in feat.lower()
        )
        checks.append({
            "id": f"immature_{anchor_suffix}",
            "label": f"{label} immature rows",
            "count": immature,
            "status": "pass" if immature == 0 else "fail",
        })

    _anchor_check("200", "EMA200")
    _anchor_check("300", "EMA300")
    checks.append({
        "id": "derived_violations",
        "label": "Derived violations",
        "count": derived_violations,
        "status": "pass" if derived_violations == 0 else "fail",
    })
    checks.append({
        "id": "policy_compliance",
        "label": "Policy compliance",
        "count": violations,
        "status": "pass" if violations == 0 else "fail",
        "pct": round(compliance_pct, 2),
    })

    return {
        "violations": violations,
        "derived_violations": derived_violations,
        "per_feature_violations": per_feature,
        "policy_compliance_pct": round(compliance_pct, 2),
        "checks": checks,
        "total_rows_audited": total_rows,
    }


def _empty_enforcement_stats() -> dict[str, Any]:
    return {
        "enforced_rows": 0,
        "nulled_cells": 0,
        "rolling_nulled": 0,
        "derived_nulled": 0,
        "gap_resets": 0,
        "rolling_not_ready_outputs": 0,
        "derived_null_propagations": 0,
        "feature_count": 0,
        "gap_max_sec": float(DEFAULT_GAP_MAX_SEC),
        "readiness_enforcement": True,
    }


def merge_readiness_stats(
    total: dict[str, Any],
    day: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-day readiness enforcement counters."""
    out = dict(total or _empty_enforcement_stats())
    day = day or {}
    for key in (
        "enforced_rows", "nulled_cells", "rolling_nulled", "derived_nulled",
        "gap_resets", "rolling_not_ready_outputs", "derived_null_propagations",
    ):
        out[key] = int(out.get(key) or 0) + int(day.get(key) or 0)
    if day.get("readiness_compliance"):
        out["readiness_compliance"] = day["readiness_compliance"]
    out["readiness_enforcement"] = True
    return out


# Back-compat alias
_merge_readiness_stats = merge_readiness_stats
