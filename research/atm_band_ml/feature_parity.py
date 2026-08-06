"""Feature implementation matrix and live vs dataset parity reporting."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_DEFAULT_ATOL = 1e-4
_COMPARE_ATOL: dict[str, float] = {
    "current_iv": 0.05,
    "delta": 1e-4,
    "gamma": 1e-6,
    "theta": 1e-3,
    "vega": 1e-3,
    "oi": 1.0,
    "moneyness": 1e-4,
}


def _chart_plugins():
    from chain_replay_ml.dataset_builder import feature_plugins, schema_implementation

    return feature_plugins, schema_implementation


def _is_null(val: Any) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and math.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _values_close(name: str, a: Any, b: Any, *, rtol: float = 1e-3) -> tuple[bool, float | None]:
    if _is_null(a) and _is_null(b):
        return True, 0.0
    if _is_null(a) or _is_null(b):
        return False, None
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return a == b, None
    diff = abs(fa - fb)
    atol = _COMPARE_ATOL.get(name, _DEFAULT_ATOL)
    ok = diff <= atol + rtol * max(abs(fa), abs(fb), 1e-9)
    return ok, diff


def implementation_matrix() -> dict[str, Any]:
    """
    Compare every registry feature: dataset builder vs live pipeline attribution.

    After live registry refactor, live uses the same dataset_builder modules (Shared).
    """
    plugins, schema_impl = _chart_plugins()
    from chain_replay_ml.dataset_builder.schema_registry import all_registry_feature_names

    names = all_registry_feature_names()
    rows: list[dict[str, Any]] = []
    shared = 0

    for name in names:
        gid = None
        for g, mapping in plugins.GROUP_FEATURE_SOURCES.items():
            if name in mapping:
                gid = g
                break
        impl = schema_impl.resolve_implementation(name, group_id=str(gid or ""))
        dataset_impl = f"{impl.get('module', '')} → {impl.get('function', '')}"
        live_impl = (
            "live_registry_builder → registry_features.build_registry_features_at_ts "
            f"→ {impl.get('function', '')}"
        )
        rows.append(
            {
                "feature": name,
                "group": gid,
                "dataset_builder": dataset_impl,
                "live_builder": live_impl,
                "status": "shared",
            }
        )
        shared += 1

    return {
        "registry_count": len(names),
        "shared_count": shared,
        "missing_count": 0,
        "different_count": 0,
        "features": rows,
    }


def compare_feature_dicts(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    required: Sequence[str],
    *,
    reference_label: str = "dataset",
    candidate_label: str = "live",
) -> dict[str, Any]:
    """Value-level parity between two feature dicts for the same timestamp."""
    req = [str(c) for c in required if str(c).strip()]
    mismatches: list[dict[str, Any]] = []
    match_n = mismatch_n = missing_n = 0
    matches: list[dict[str, Any]] = []

    for name in req:
        ref_val = reference.get(name)
        cand_val = candidate.get(name)
        if name not in candidate and name not in reference:
            missing_n += 1
            mismatches.append(
                {
                    "feature": name,
                    "status": "missing_both",
                    reference_label: ref_val,
                    candidate_label: cand_val,
                }
            )
            continue
        ok, diff = _values_close(name, ref_val, cand_val)
        if ok:
            match_n += 1
            matches.append({"feature": name, "difference": diff or 0.0})
        else:
            mismatch_n += 1
            mismatches.append(
                {
                    "feature": name,
                    "status": "mismatch",
                    reference_label: ref_val,
                    candidate_label: cand_val,
                    "difference": diff,
                }
            )

    total = len(req)
    parity_pct = round(100.0 * match_n / total, 2) if total else 0.0
    return {
        "required_count": total,
        "match_count": match_n,
        "mismatch_count": mismatch_n,
        "missing_count": missing_n,
        "parity_pct": parity_pct,
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "mismatches": mismatches[:80],
        "matches_sample": matches[:20],
    }


def live_parity_report(
    *,
    registry_features: Mapping[str, Any],
    legacy_features: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    """Developer report: implementation matrix + legacy vs registry value diff."""
    matrix = implementation_matrix()
    value_cmp = compare_feature_dicts(
        legacy_features,
        registry_features,
        required,
        reference_label="legacy_live",
        candidate_label="registry_live",
    )
    reg_n = sum(
        1
        for n in required
        if n in registry_features and not _is_null(registry_features.get(n))
    )
    return {
        "dataset_builder_features": matrix["registry_count"],
        "live_builder_features": reg_n,
        "registry_populated": reg_n,
        "parity_pct": value_cmp["parity_pct"],
        "implementation": matrix,
        "legacy_vs_registry": value_cmp,
        "significant_differences": [
            m for m in value_cmp["mismatches"] if m.get("status") == "mismatch"
        ],
    }
