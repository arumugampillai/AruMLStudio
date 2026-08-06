"""Live registry feature coverage audit (investigation — no scoring fixes)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# Registry group → live pipeline stage (dataset builder is single source of truth).
_LIVE_MODULE_BY_GROUP: dict[str, str] = {
    "price": "registry_features → extract_timeline_features",
    "greeks": "registry_features → enrich_dataset_features / bs.greeks",
    "iv": "registry_features → enrich_dataset_features",
    "oi": "registry_features → enrich_dataset_features + chain_maps",
    "volume": "registry_features → extract_timeline_features",
    "momentum": "registry_features → extract_timeline_features",
    "time": "registry_features → extract_timeline_features + enrich_dataset_features",
    "moneyness": "registry_features → extract_timeline_features + enrich_dataset_features",
    "atm_straddle": "registry_features → enrich_with_chain_maps",
    "chain": "registry_features → enrich_with_chain_maps",
    "historical": "registry_features → extract_ohlc_features_for_timeline",
    "advanced": "registry_features → enrich_dataset_features",
}

MISSING_CATEGORY_LABELS: dict[str, str] = {
    "not_implemented": "Not Implemented",
    "waiting_for_history": "Waiting for History",
    "missing_market_data": "Missing Market Data",
    "feature_error": "Feature Error",
}

_HISTORY_GROUPS = frozenset({"historical", "advanced"})
_HISTORY_REASON_MARKERS = (
    "history",
    "lookback",
    "window",
    "incomplete",
    "ohlc",
    "insufficient",
    "early session",
    "chain context history",
)
_HISTORY_NAME_MARKERS = (
    "_prev",
    "zscore",
    "_rank_",
    "slope",
    "roll_age",
    "rows_since_roll",
    "iv_drift",
    "change_5m",
    "change_15m",
    "change_30m",
    "change_1m",
    "change_accel",
    "pct_change",
    "since_roll",
    "rv_",
    "cross_age",
    "time_since_cross",
)
_MARKET_DATA_NAME_MARKERS = (
    "ltp",
    "spot",
    "bid_ask",
    "current_iv",
    "roll_iv",
    "oi",
    "volume",
    "delta",
    "gamma",
    "theta",
    "vega",
    "moneyness",
    "atm_straddle",
    "chain_pcr",
    "atm_pcr",
)


def empty_missing_breakdown() -> dict[str, int]:
    return {k: 0 for k in MISSING_CATEGORY_LABELS}


def _chart_feature_plugins() -> Any:
    from chain_replay_ml.dataset_builder import feature_plugins

    return feature_plugins


def _feature_group(feature: str) -> str | None:
    plugins = _chart_feature_plugins()
    for gid, mapping in plugins.GROUP_FEATURE_SOURCES.items():
        if feature in mapping:
            return gid
    return None


def live_source_module(feature: str) -> str:
    gid = _feature_group(feature)
    if gid is None:
        return "unknown"
    plugins = _chart_feature_plugins()
    src = (plugins.GROUP_FEATURE_SOURCES.get(gid) or {}).get(feature)
    if src is None:
        return f"{_LIVE_MODULE_BY_GROUP.get(gid, 'dataset_builder')} — not mapped for live"
    return _LIVE_MODULE_BY_GROUP.get(gid, f"dataset_builder.{gid}")


def _coerce_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval) or math.isinf(fval):
        return None
    return fval


def _missing_reason(feature: str, features: Mapping[str, Any]) -> str:
    gid = _feature_group(feature)
    if feature not in features:
        if gid is None:
            return "Feature not in registry / live pipeline"
        if gid in _HISTORY_GROUPS:
            return "Insufficient tick/OHLC history at signal_ts"
        if gid in ("atm_straddle", "chain"):
            return "Chain straddle/OI history unavailable at signal_ts"
        if gid in ("volume", "oi"):
            return "Option tick or OI window incomplete at signal_ts"
        if gid == "iv" and feature == "current_iv":
            return "Missing option LTP or spot at signal_ts"
        if gid == "iv":
            return "IV history or roll state not warm enough at signal_ts"
        return "Lookback window or market tick unavailable at signal_ts"
    val = features[feature]
    if val is None:
        return "Null — market tick or quote missing at signal_ts"
    if isinstance(val, float) and math.isnan(val):
        return "NaN — feature computation failed at signal_ts"
    try:
        float(val)
    except (TypeError, ValueError):
        return "Non-numeric value at signal_ts"
    return "Unavailable"


def _is_history_feature(feature: str, gid: str | None) -> bool:
    if gid in _HISTORY_GROUPS:
        return True
    name = str(feature).lower()
    return any(marker in name for marker in _HISTORY_NAME_MARKERS)


def _is_point_in_time_feature(feature: str) -> bool:
    name = str(feature).lower()
    if any(marker in name for marker in _HISTORY_NAME_MARKERS):
        return False
    return any(marker in name for marker in _MARKET_DATA_NAME_MARKERS)


def classify_missing_feature(
    feature: str,
    features: Mapping[str, Any],
    *,
    reason: str = "",
    raw: Any = None,
) -> str:
    """
    Bucket a missing/unavailable feature by root cause (implementation vs runtime data).
    """
    gid = _feature_group(feature)
    reason_l = str(reason or "").lower()

    if "non-numeric" in reason_l or "nan" in reason_l:
        return "feature_error"
    if feature in features and raw is not None and _coerce_float(raw) is None:
        return "feature_error"

    plugins = _chart_feature_plugins()
    if gid is None:
        return "not_implemented"
    mapping = plugins.GROUP_FEATURE_SOURCES.get(gid) or {}
    if feature not in mapping or mapping.get(feature) is None:
        return "not_implemented"
    if "not in registry" in reason_l or "not implemented" in reason_l:
        return "not_implemented"

    absent = feature not in features or features.get(feature) is None
    if absent and _is_point_in_time_feature(feature):
        return "missing_market_data"

    if _is_history_feature(feature, gid):
        return "waiting_for_history"
    if any(marker in reason_l for marker in _HISTORY_REASON_MARKERS):
        return "waiting_for_history"

    if feature not in features:
        return "waiting_for_history"

    if features.get(feature) is None:
        return "missing_market_data"

    return "feature_error"


def audit_registry_feature_coverage(
    features: Mapping[str, Any],
    required: Sequence[str],
    *,
    model_name: str = "",
    probe_symbol: str = "",
    probe_token: str = "",
    signal_ts: float | None = None,
) -> dict[str, Any]:
    """
  Compare model registry columns vs a live feature dict (strict — no fill_missing).

  Returns a JSON-serializable report for Trade Decision Server UI.
  """
    req = [str(c) for c in (required or []) if str(c).strip()]
    feat = dict(features or {})
    rows: list[dict[str, Any]] = []
    present_ok = 0
    missing_names: list[str] = []
    type_mismatches: list[str] = []
    missing_breakdown = empty_missing_breakdown()

    for name in req:
        raw = feat.get(name)
        fval = _coerce_float(raw)
        if name not in feat or fval is None:
            reason = _missing_reason(name, feat)
            category = classify_missing_feature(name, feat, reason=reason, raw=raw)
            missing_breakdown[category] = int(missing_breakdown.get(category, 0)) + 1
            rows.append(
                {
                    "name": name,
                    "status": "missing",
                    "value": None,
                    "display_value": "—",
                    "reason": reason,
                    "missing_category": category,
                    "missing_category_label": MISSING_CATEGORY_LABELS.get(category, category),
                    "source_module": live_source_module(name),
                }
            )
            missing_names.append(name)
            if name in feat and raw is not None:
                type_mismatches.append(name)
            continue
        present_ok += 1
        display = fval
        if abs(fval) >= 1000:
            display = round(fval, 2)
        elif abs(fval) >= 1:
            display = round(fval, 4)
        else:
            display = round(fval, 6)
        rows.append(
            {
                "name": name,
                "status": "ok",
                "value": fval,
                "display_value": str(display),
                "reason": "",
                "missing_category": "",
                "missing_category_label": "",
                "source_module": live_source_module(name),
            }
        )

    extra = sorted(set(feat.keys()) - set(req))
    generated_keys = [k for k, v in feat.items() if _coerce_float(v) is not None]
    order_in_generated = [c for c in req if c in feat and _coerce_float(feat.get(c)) is not None]
    order_match = order_in_generated == [c for c in req if c in order_in_generated]

    required_n = len(req)
    missing_n = len(missing_names)
    coverage_pct = round(100.0 * present_ok / required_n, 1) if required_n else 0.0
    can_score = missing_n == 0 and not type_mismatches

    if can_score:
        inference_status = "Can Score"
        inference_reason = "All required features available."
    else:
        inference_status = "Blocked"
        parts = []
        if missing_n:
            parts.append(f"{missing_n} required feature{'s' if missing_n != 1 else ''} unavailable")
            hist_n = missing_breakdown.get("waiting_for_history", 0)
            mkt_n = missing_breakdown.get("missing_market_data", 0)
            if hist_n:
                parts.append(f"{hist_n} waiting for history")
            if mkt_n:
                parts.append(f"{mkt_n} missing market data")
            if missing_breakdown.get("not_implemented"):
                parts.append(f"{missing_breakdown['not_implemented']} not implemented")
            if missing_breakdown.get("feature_error"):
                parts.append(f"{missing_breakdown['feature_error']} feature error(s)")
        if type_mismatches:
            parts.append(f"{len(type_mismatches)} type mismatch(es)")
        inference_reason = "; ".join(parts) + "."

    breakdown_rows = [
        {"category": key, "label": MISSING_CATEGORY_LABELS[key], "count": int(missing_breakdown.get(key, 0))}
        for key in MISSING_CATEGORY_LABELS
    ]

    return {
        "model_name": str(model_name or ""),
        "probe_symbol": str(probe_symbol or ""),
        "probe_token": str(probe_token or ""),
        "signal_ts": float(signal_ts) if signal_ts is not None else None,
        "required_count": required_n,
        "generated_count": present_ok,
        "missing_count": missing_n,
        "missing_breakdown": missing_breakdown,
        "missing_breakdown_rows": breakdown_rows,
        "coverage_pct": coverage_pct,
        "inference_status": inference_status,
        "inference_reason": inference_reason,
        "can_score_strict": can_score,
        "validation": {
            "missing_features": missing_names,
            "extra_features": extra,
            "extra_count": len(extra),
            "feature_order_match": order_match,
            "type_mismatches": type_mismatches,
            "required_order": list(req),
            "generated_order": order_in_generated,
        },
        "features": rows,
        "pipeline": [
            "Tick Data (tick_ring_store / LIVE_LTP / chart volume)",
            "Registry Feature Builder (live_registry_builder)",
            "extract_timeline_features → enrich_dataset_features → enrich_with_chain_maps",
            "pick_features_from_row (registry aliases)",
            "Live LTP overlay (_overlay_live_option_features)",
            "Inference (AtmBandModelScorer.score_features — strict audit, fill_missing=0 at score)",
        ],
    }


def empty_feature_coverage(*, model_name: str = "", reason: str = "No evaluation yet") -> dict[str, Any]:
    return {
        "model_name": str(model_name or ""),
        "probe_symbol": "",
        "probe_token": "",
        "signal_ts": None,
        "required_count": 0,
        "generated_count": 0,
        "missing_count": 0,
        "missing_breakdown": empty_missing_breakdown(),
        "missing_breakdown_rows": [
            {"category": k, "label": v, "count": 0} for k, v in MISSING_CATEGORY_LABELS.items()
        ],
        "coverage_pct": 0.0,
        "inference_status": "—",
        "inference_reason": reason,
        "can_score_strict": False,
        "validation": {
            "missing_features": [],
            "extra_features": [],
            "extra_count": 0,
            "feature_order_match": True,
            "type_mismatches": [],
            "required_order": [],
            "generated_order": [],
        },
        "features": [],
        "pipeline": [],
    }
