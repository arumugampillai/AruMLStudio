"""Feature family resolution for Experiment Planner v2.

Prefer an optional name→family map (e.g. from Feature Registry metadata).
Otherwise use deterministic name heuristics (prefixes / keywords).

Planner families (display labels):
  IV, EMA, OI, PCR, Volume, Greeks, Spot, Time, Straddle, Chain, GEX, Targets, Other

Registry group ids (when a map is supplied) are remapped via
``REGISTRY_GROUP_TO_FAMILY``; unknown groups fall through to heuristics.
"""

from __future__ import annotations

from typing import Any

# Canonical planner families (stable order for docs / ranking ties).
PLANNER_FAMILIES: tuple[str, ...] = (
    "IV",
    "EMA",
    "OI",
    "PCR",
    "Volume",
    "Greeks",
    "Spot",
    "Time",
    "Straddle",
    "Chain",
    "GEX",
    "Targets",
    "Other",
)

# Feature-registry / day-metadata group id → planner family.
REGISTRY_GROUP_TO_FAMILY: dict[str, str] = {
    "iv": "IV",
    "iv_zscore": "IV",
    "iv_ema_ratio": "IV",
    "volatility": "IV",
    "greeks": "Greeks",
    "oi": "OI",
    "pcr": "PCR",
    "atm_straddle": "Straddle",
    "straddle": "Straddle",
    "chain": "Chain",
    "gex": "GEX",
    "time": "Time",
    "price": "Spot",
    "spot": "Spot",
    "volume": "Volume",
    "ema": "EMA",
    "target": "Targets",
    "targets": "Targets",
    # Broader day-metadata labels
    "market structure": "OI",
    "order book": "Volume",
    "meta": "Time",
    "prediction": "Other",
    "derived": "Other",
    "base": "Spot",
}

# Alias / UI label → planner family (case-insensitive).
_LABEL_ALIASES: dict[str, str] = {
    f.lower(): f for f in PLANNER_FAMILIES
}
_LABEL_ALIASES.update(
    {
        "volatility": "IV",
        "vol": "IV",
        "implied volatility": "IV",
        "open interest": "OI",
        "open_interest": "OI",
        "market structure": "OI",
        "price": "Spot",
        "target": "Targets",
        "momentum": "Other",
        "order book": "Volume",
        "meta": "Time",
        "prediction": "Other",
        "derived": "Other",
        "base": "Spot",
    }
)


def normalize_family_label(raw: Any) -> str | None:
    """Map an arbitrary family/group label to a planner family, or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    low = text.lower()
    if low in _LABEL_ALIASES:
        return _LABEL_ALIASES[low]
    if low in REGISTRY_GROUP_TO_FAMILY:
        return REGISTRY_GROUP_TO_FAMILY[low]
    # Compact forms: "feature_family:IV"
    if ":" in low:
        return normalize_family_label(low.split(":")[-1])
    return None


def map_registry_families(family_by_name: dict[str, Any] | None) -> dict[str, str]:
    """Normalize a name→family map to planner family labels (drop unknowns)."""
    out: dict[str, str] = {}
    if not isinstance(family_by_name, dict):
        return out
    for name, fam in family_by_name.items():
        feat = str(name or "").strip()
        if not feat:
            continue
        mapped = normalize_family_label(fam)
        if mapped:
            out[feat] = mapped
        else:
            # Leave unresolved so heuristics can still classify the name.
            continue
    return out


def resolve_feature_family(
    name: str,
    *,
    family_by_name: dict[str, str] | None = None,
) -> str:
    """Resolve a feature name to a planner family.

    Order: explicit map → name heuristics → Other.
    """
    feat = str(name or "").strip()
    if not feat:
        return "Other"
    if family_by_name:
        mapped = family_by_name.get(feat)
        if mapped:
            norm = normalize_family_label(mapped) or str(mapped).strip()
            if norm in PLANNER_FAMILIES:
                return norm
    return _heuristic_family(feat)


def _heuristic_family(name: str) -> str:
    """Deterministic keyword / prefix heuristics (first match wins)."""
    n = str(name or "").strip()
    low = n.lower()

    # Targets first (labels / futures).
    if low.startswith("future_") or low.endswith("_target") or low.startswith("target_"):
        return "Targets"

    # Specific market-structure tokens before broader ones.
    if "gex" in low:
        return "GEX"
    if "pcr" in low:
        return "PCR"
    if "straddle" in low:
        return "Straddle"
    if low.startswith("chain_") or "_chain_" in low or low.endswith("_chain"):
        return "Chain"

    # IV before EMA so iv_ema* stays IV.
    if any(
        tok in low
        for tok in (
            "iv_",
            "_iv",
            "atm_iv",
            "implied",
            "iv_skew",
            "iv_vs",
            "current_iv",
            "reiv",
        )
    ) or low in ("iv",) or low.startswith("iv"):
        # Avoid classifying pure volume tokens as IV via "vol" substring alone.
        return "IV"

    if "ema" in low:
        return "EMA"

    if any(
        tok in low
        for tok in (
            "delta",
            "gamma",
            "theta",
            "vega",
            "charm",
            "vanna",
            "vomma",
            "volga",
            "rho",
        )
    ):
        return "Greeks"

    if any(tok in low for tok in ("open_interest", "_oi_", "oi_", "_oi")) or low.endswith(
        "_oi"
    ) or low.startswith("oi_") or "max_call_oi" in low or "max_put_oi" in low or "oi_wall" in low:
        return "OI"
    if "oi" in low and "noise" not in low:
        # Broad fallback for *oi* tokens (distance_to_max_*_oi_strikes, etc.).
        return "OI"

    if any(tok in low for tok in ("volume", "vwap", "turnover", "vol_flow")):
        return "Volume"

    if any(
        tok in low
        for tok in ("spot", "underlying", "ltp", "fut_ltp", "premium", "price")
    ):
        return "Spot"

    if any(
        tok in low
        for tok in (
            "time_to",
            "days_to",
            "session",
            "minute",
            "_lag_",
            "timestamp",
            "expiry",
        )
    ):
        return "Time"

    return "Other"


def group_features_by_family(
    features: list[Any],
    *,
    family_by_name: dict[str, str] | None = None,
) -> dict[str, list[Any]]:
    """Group affected-feature items (dicts or names) by planner family.

    Preserves encounter order within each family. Family keys follow
    ``PLANNER_FAMILIES`` order, then any unexpected labels.
    """
    buckets: dict[str, list[Any]] = {}
    for item in features:
        if isinstance(item, dict):
            feat = str(item.get("feature") or "").strip()
        else:
            feat = str(item).strip()
        if not feat:
            continue
        fam = resolve_feature_family(feat, family_by_name=family_by_name)
        buckets.setdefault(fam, []).append(item)

    ordered: dict[str, list[Any]] = {}
    for fam in PLANNER_FAMILIES:
        if fam in buckets:
            ordered[fam] = buckets[fam]
    for fam, items in buckets.items():
        if fam not in ordered:
            ordered[fam] = items
    return ordered
