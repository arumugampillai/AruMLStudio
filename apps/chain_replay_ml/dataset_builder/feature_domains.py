"""Feature Registry primary domains — business taxonomy for Auto Feature Generation.

Plugin ``group_id`` in ``_REGISTRY_FEATURES`` remains the *implementation* grouping
(schema / extractors). This module is the *business* primary domain every
canonical feature belongs to exactly once.

Does not rename feature IDs or change formulas.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from .feature_ownership import (
    OWNERSHIP_BASE,
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from .feature_plugins import _REGISTRY_FEATURES

DomainId = Literal[
    "price_premium",
    "spot_futures",
    "greeks",
    "implied_volatility",
    "open_interest",
    "volume_liquidity",
    "chain_analytics",
    "historical_context",
    "market_structure",
    "time_session",
    "metadata",
]

DataType = Literal[
    "Price",
    "Level",
    "Ratio",
    "Percentage",
    "Count",
    "Boolean",
    "Time",
    "Greek",
    "Volatility",
    "Score",
    "Spread",
]

OwnershipKind = Literal["base", "computed_base"]


class FeatureDomainMeta(TypedDict):
    primary_domain: DomainId
    ownership: OwnershipKind
    data_type: DataType
    can_apply_lag: bool
    can_apply_difference: bool
    can_apply_return: bool
    can_apply_rolling: bool
    can_apply_zscore: bool
    can_participate_in_interaction: bool


# Stable UI order (labels match product language).
DOMAIN_ORDER: tuple[DomainId, ...] = (
    "price_premium",
    "spot_futures",
    "greeks",
    "implied_volatility",
    "open_interest",
    "volume_liquidity",
    "chain_analytics",
    "historical_context",
    "market_structure",
    "time_session",
    "metadata",
)

DOMAIN_LABELS: dict[DomainId, str] = {
    "price_premium": "Price & Premium",
    "spot_futures": "Spot & Futures",
    "greeks": "Greeks",
    "implied_volatility": "Implied Volatility",
    "open_interest": "Open Interest",
    "volume_liquidity": "Volume & Liquidity",
    "chain_analytics": "Chain Analytics",
    "historical_context": "Historical Context",
    "market_structure": "Market Structure",
    "time_session": "Time & Session",
    "metadata": "Metadata",
}

# ---------------------------------------------------------------------------
# Explicit primary-domain membership (exactly one domain per feature).
# ---------------------------------------------------------------------------

_DOMAIN_MEMBERS: dict[DomainId, frozenset[str]] = {
    "price_premium": frozenset({
        "ltp",
        "option_vwap",
        "option_bid",
        "option_ask",
        "option_open",
        "option_high",
        "option_low",
        "option_prev_close",
        "bid_ask_spread",
        "bs_reiv_pred",
        "dgt_reiv_pred",
        "dgt_prediction_error",
        "ltp_ema9",
        "ltp_ema20",
        "ltp_ema50",
        "ltp_ema100",
        "ltp_ema200",
        "ltp_ema300",
        "ltp_std20",
        "side_to_ltp_ratio",
        "weighted_ltp_ema",
        "ltp_to_spot_ratio",
        "ce_atm6_ltp_sum",
        "pe_atm6_ltp_sum",
        "ce_minus_pe_atm6_ltp",
        "moneyness",
        "distance_from_spot_pct",
        "distance_from_atm_pct",
        "distance_from_atm_points",
        "strike_distance_from_atm",
        "strike_to_spot_ratio",
    }),
    "spot_futures": frozenset({
        "spot",
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_prev_close",
        "futures_ltp",
        "futures_vwap",
        "futures_bid",
        "futures_ask",
        "futures_spread",
        "futures_day_volume",
        "futures_oi",
        "synthetic_forward_spot",
        "spot_ema9",
        "spot_ema20",
        "spot_ema50",
        "spot_ema100",
        "spot_ema200",
        "spot_ema300",
        "spot_high_ema20",
        "spot_high_ema50",
        "spot_high_ema100",
        "spot_high_ema200",
        "spot_high_ema300",
        "spot_low_ema20",
        "spot_low_ema50",
        "spot_low_ema100",
        "spot_low_ema200",
        "spot_low_ema300",
        "weighted_spot_ema",
        "weighted_spot_high_ema",
        "weighted_spot_low_ema",
        "weighted_spot_close_ema",
        "spot_ema20_channel_width",
        "spot_ema50_channel_width",
        "spot_ema100_channel_width",
        "spot_ema200_channel_width",
        "spot_ema300_channel_width",
        "gamma_flip_spot",
    }),
    "greeks": frozenset({
        "delta",
        "abs_delta",
        "gamma",
        "theta",
        "vega",
        "vanna",
        "volga",
        "charm",
        "speed",
        "color",
        "zomma",
        "ultima",
        "theta_per_min",
        "vega_per_ivpt",
    }),
    "implied_volatility": frozenset({
        "current_iv",
        "roll_iv",
        "iv_drift_from_roll",
        "iv_rank_session",
        "iv_vs_atm",
        "atm_iv_ce",
        "atm_iv_pe",
        "iv_ema9",
        "iv_ema20",
        "iv_ema50",
        "iv_ema100",
        "iv_ema200",
        "iv_ema300",
        "iv_skew_atm",
        "iv_call_put_skew",
        "iv_skew_25d",
        "iv_skew_10d",
        "iv_curvature_25d",
        "iv_butterfly_25d",
        "iv_term_slope_near_next",
        "surface_displacement_5m",
        "surface_displacement_15m",
        "surface_acceleration_15m",
        "vrp_proxy_30m",
        "svi_param_a",
        "svi_param_b",
        "svi_param_rho",
        "svi_param_m",
        "svi_param_sigma",
        "svi_calibration_rmse",
        "sabr_param_alpha",
        "sabr_param_rho",
        "sabr_param_nu",
        "sabr_calibration_rmse",
        "iv_rv_spread_5m",
        "iv_rv_spread_10m",
    }),
    "open_interest": frozenset({
        "option_oi",
        "total_call_oi",
        "total_put_oi",
        "max_call_oi_pct",
        "max_put_oi_pct",
        "distance_to_max_call_oi_strikes",
        "distance_to_max_put_oi_strikes",
        "distance_to_call_build_wall",
        "distance_to_put_build_wall",
        "oi_wall_bias",
        "pinning_pressure",
        "chain_pcr",
        "atm_pcr",
        "oi_abs_delta_0_20_ce",
        "oi_abs_delta_0_20_pe",
        "oi_abs_delta_20_40_ce",
        "oi_abs_delta_20_40_pe",
        "oi_abs_delta_40_60_ce",
        "oi_abs_delta_40_60_pe",
        "oi_abs_delta_60_80_ce",
        "oi_abs_delta_60_80_pe",
        "oi_abs_delta_80_100_ce",
        "oi_abs_delta_80_100_pe",
    }),
    "volume_liquidity": frozenset({
        "option_day_volume",
        "ltq",
        "total_buy_qty",
        "total_sell_qty",
        "total_ce_volume",
        "total_pe_volume",
        "otm_ce_volume",
        "otm_pe_volume",
        "otm_pcr_volume",
        "mid_price",
        "microprice",
        "microprice_bias",
        "book_imbalance_l1",
        "book_imbalance_l1_5",
        "bid_depth_l1_5",
        "ask_depth_l1_5",
        "book_depth_slope_bid",
        "book_depth_slope_ask",
    }),
    "chain_analytics": frozenset({
        "atm_straddle",
        "call_gex",
        "put_gex",
        "net_gex",
        "chain_gex",
        "gamma_flip_distance",
        "delta_w_volume_flow_1m",
        "delta_w_volume_flow_5m",
    }),
    "historical_context": frozenset({
        "spot_1m_ema9",
        "spot_1m_ema20",
        "spot_1m_ema50",
        "spot_1m_ema100",
        "spot_1m_ema200",
        "spot_3m_ema9",
        "spot_3m_ema20",
        "spot_3m_ema50",
        "spot_3m_ema100",
        "spot_3m_ema200",
        "spot_5m_ema9",
        "spot_5m_ema20",
        "spot_5m_ema50",
        "spot_5m_ema100",
        "spot_5m_ema200",
        "spot_15m_ema9",
        "spot_15m_ema20",
        "spot_15m_ema50",
        "spot_15m_ema100",
        "spot_15m_ema200",
        "spot_up_score_1m",
        "spot_up_score_3m",
        "spot_up_score_5m",
        "spot_up_score_10m",
        "spot_down_score_1m",
        "spot_down_score_3m",
        "spot_down_score_5m",
        "spot_down_score_10m",
        "spot_up_sample_count_1m",
        "spot_up_sample_count_3m",
        "spot_up_sample_count_5m",
        "spot_up_sample_count_10m",
        "spot_down_sample_count_1m",
        "spot_down_sample_count_3m",
        "spot_down_sample_count_5m",
        "spot_down_sample_count_10m",
    }),
    "market_structure": frozenset({
        "ema9_slope",
        "ema9_gt_ema20",
        "time_since_cross_min",
        "cross_age_decay",
        "price_dist_from_cross_pct",
        "spot_rv_5m",
        "spot_rv_10m",
        "opt_rv_5m",
        "opt_rv_10m",
    }),
    "time_session": frozenset({
        "minutes_to_expiry",
        "minutes_since_open",
        "minutes_to_close",
        "is_first_hour",
        "is_last_hour",
        "minute_of_day",
        "days_to_expiry",
        "is_expiry_day",
    }),
    "metadata": frozenset({
        "strike",
        "is_call",
        "roll_age_min",
        "rows_since_roll",
    }),
}

# Per-feature data_type overrides (default inferred from name / domain).
_DATA_TYPE_OVERRIDES: dict[str, DataType] = {
    "bid_ask_spread": "Spread",
    "futures_spread": "Spread",
    "microprice_bias": "Ratio",
    "book_imbalance_l1": "Ratio",
    "book_imbalance_l1_5": "Ratio",
    "book_depth_slope_bid": "Level",
    "book_depth_slope_ask": "Level",
    "ltp_to_spot_ratio": "Ratio",
    "strike_to_spot_ratio": "Ratio",
    "side_to_ltp_ratio": "Ratio",
    "moneyness": "Ratio",
    "distance_from_spot_pct": "Percentage",
    "distance_from_atm_pct": "Percentage",
    "price_dist_from_cross_pct": "Percentage",
    "max_call_oi_pct": "Percentage",
    "max_put_oi_pct": "Percentage",
    "iv_rank_session": "Percentage",
    "chain_pcr": "Ratio",
    "atm_pcr": "Ratio",
    "otm_pcr_volume": "Ratio",
    "gamma_flip_distance": "Ratio",
    "iv_vs_atm": "Ratio",
    "iv_skew_atm": "Volatility",
    "iv_call_put_skew": "Volatility",
    "iv_skew_25d": "Volatility",
    "iv_butterfly_25d": "Volatility",
    "iv_rv_spread_5m": "Volatility",
    "iv_rv_spread_10m": "Volatility",
    "iv_drift_from_roll": "Volatility",
    "current_iv": "Volatility",
    "roll_iv": "Volatility",
    "atm_iv_ce": "Volatility",
    "atm_iv_pe": "Volatility",
    "spot_rv_5m": "Volatility",
    "spot_rv_10m": "Volatility",
    "opt_rv_5m": "Volatility",
    "opt_rv_10m": "Volatility",
    "ema9_gt_ema20": "Boolean",
    "is_call": "Boolean",
    "is_first_hour": "Boolean",
    "is_last_hour": "Boolean",
    "is_expiry_day": "Boolean",
    "cross_age_decay": "Score",
    "ema9_slope": "Score",
    "dgt_prediction_error": "Price",
    "theta_per_min": "Greek",
    "vega_per_ivpt": "Greek",
    "color": "Greek",
    "zomma": "Greek",
    "ultima": "Greek",
    "iv_skew_10d": "Volatility",
    "iv_curvature_25d": "Volatility",
    "iv_term_slope_near_next": "Volatility",
    "surface_displacement_5m": "Level",
    "surface_displacement_15m": "Level",
    "surface_acceleration_15m": "Level",
    "vrp_proxy_30m": "Volatility",
    "svi_param_a": "Level",
    "svi_param_b": "Level",
    "svi_param_rho": "Ratio",
    "svi_param_m": "Level",
    "svi_param_sigma": "Volatility",
    "svi_calibration_rmse": "Score",
    "sabr_param_alpha": "Volatility",
    "sabr_param_rho": "Ratio",
    "sabr_param_nu": "Volatility",
    "sabr_calibration_rmse": "Score",
}

# Transform-flag overrides (True/False). Unset → inferred from data_type.
_FLAG_OVERRIDES: dict[str, dict[str, bool]] = {
    # Booleans / clocks: lag ok for some clocks; no return / limited interaction.
    "is_call": {
        "can_apply_lag": False,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
    "is_first_hour": {
        "can_apply_lag": False,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
    "is_last_hour": {
        "can_apply_lag": False,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
    "is_expiry_day": {
        "can_apply_lag": False,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
    "ema9_gt_ema20": {
        "can_apply_lag": True,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
    "strike": {
        "can_apply_lag": False,
        "can_apply_difference": False,
        "can_apply_return": False,
        "can_apply_rolling": False,
        "can_apply_zscore": False,
        "can_participate_in_interaction": True,
    },
}


def _registry_feature_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for feats in _REGISTRY_FEATURES.values():
        for name in feats:
            n = str(name)
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _domain_of_name(name: str) -> DomainId:
    for domain_id, members in _DOMAIN_MEMBERS.items():
        if name in members:
            return domain_id
    raise KeyError(f"Feature {name!r} has no primary domain assignment")


def _infer_data_type(name: str, domain: DomainId) -> DataType:
    if name in _DATA_TYPE_OVERRIDES:
        return _DATA_TYPE_OVERRIDES[name]
    if domain == "greeks":
        return "Greek"
    if domain == "implied_volatility":
        return "Volatility"
    if domain == "time_session":
        if name.startswith("is_"):
            return "Boolean"
        return "Time"
    if domain == "metadata":
        if name.startswith("is_"):
            return "Boolean"
        if "age" in name or "rows_" in name:
            return "Time"
        return "Level"
    if name.endswith("_pct") or "pct" in name:
        return "Percentage"
    if name.endswith("_ratio") or name.endswith("_pcr") or "imbalance" in name:
        return "Ratio"
    if any(
        k in name
        for k in (
            "volume",
            "oi",
            "qty",
            "ltq",
            "depth",
            "count",
            "gex",
        )
    ):
        return "Count"
    if any(k in name for k in ("score", "decay", "slope")):
        return "Score"
    if domain in ("price_premium", "spot_futures") or name in (
        "mid_price",
        "microprice",
        "atm_straddle",
    ):
        return "Price"
    if domain == "volume_liquidity":
        return "Count"
    if domain == "open_interest":
        return "Count"
    if domain == "chain_analytics":
        return "Level"
    if domain == "historical_context":
        if "count" in name:
            return "Count"
        if "score" in name:
            return "Score"
        return "Price"
    if domain == "market_structure":
        return "Level"
    return "Level"


def _flags_for_data_type(data_type: DataType) -> dict[str, bool]:
    if data_type == "Boolean":
        return {
            "can_apply_lag": True,
            "can_apply_difference": False,
            "can_apply_return": False,
            "can_apply_rolling": False,
            "can_apply_zscore": False,
            "can_participate_in_interaction": True,
        }
    if data_type == "Time":
        return {
            "can_apply_lag": True,
            "can_apply_difference": True,
            "can_apply_return": False,
            "can_apply_rolling": True,
            "can_apply_zscore": False,
            "can_participate_in_interaction": True,
        }
    if data_type in ("Price", "Level", "Greek", "Volatility", "Score", "Spread"):
        return {
            "can_apply_lag": True,
            "can_apply_difference": True,
            "can_apply_return": data_type in ("Price", "Level", "Volatility"),
            "can_apply_rolling": True,
            "can_apply_zscore": True,
            "can_participate_in_interaction": True,
        }
    if data_type == "Count":
        return {
            "can_apply_lag": True,
            "can_apply_difference": True,
            "can_apply_return": True,
            "can_apply_rolling": True,
            "can_apply_zscore": True,
            "can_participate_in_interaction": True,
        }
    if data_type in ("Ratio", "Percentage"):
        return {
            "can_apply_lag": True,
            "can_apply_difference": True,
            "can_apply_return": False,
            "can_apply_rolling": True,
            "can_apply_zscore": True,
            "can_participate_in_interaction": True,
        }
    return {
        "can_apply_lag": True,
        "can_apply_difference": True,
        "can_apply_return": False,
        "can_apply_rolling": True,
        "can_apply_zscore": True,
        "can_participate_in_interaction": True,
    }


def _ownership_kind(name: str) -> OwnershipKind:
    cat = ownership_of(name)
    if cat == OWNERSHIP_COMPUTED_BASE:
        return "computed_base"
    if cat == OWNERSHIP_BASE:
        return "base"
    # Registry admission should never leave pipeline/retired in canonical set.
    return "base"


def build_feature_domain_meta(name: str) -> FeatureDomainMeta:
    domain = _domain_of_name(name)
    data_type = _infer_data_type(name, domain)
    flags = _flags_for_data_type(data_type)
    if name in _FLAG_OVERRIDES:
        flags.update(_FLAG_OVERRIDES[name])
    return {
        "primary_domain": domain,
        "ownership": _ownership_kind(name),
        "data_type": data_type,
        "can_apply_lag": bool(flags["can_apply_lag"]),
        "can_apply_difference": bool(flags["can_apply_difference"]),
        "can_apply_return": bool(flags["can_apply_return"]),
        "can_apply_rolling": bool(flags["can_apply_rolling"]),
        "can_apply_zscore": bool(flags["can_apply_zscore"]),
        "can_participate_in_interaction": bool(flags["can_participate_in_interaction"]),
    }


def all_feature_domain_meta() -> dict[str, FeatureDomainMeta]:
    return {name: build_feature_domain_meta(name) for name in _registry_feature_names()}


def primary_domain_of(name: str) -> DomainId:
    return _domain_of_name(str(name))


def primary_domain_label(name: str) -> str:
    return DOMAIN_LABELS[primary_domain_of(name)]


def domain_label(domain_id: str) -> str:
    return DOMAIN_LABELS.get(domain_id, str(domain_id))  # type: ignore[arg-type]


def features_by_domain() -> dict[DomainId, list[str]]:
    out: dict[DomainId, list[str]] = {d: [] for d in DOMAIN_ORDER}
    for name in _registry_feature_names():
        out[primary_domain_of(name)].append(name)
    for d in DOMAIN_ORDER:
        out[d].sort()
    return out


def domain_counts() -> dict[DomainId, int]:
    return {d: len(names) for d, names in features_by_domain().items()}


def validate_domain_coverage(*, expected_total: int = 226) -> dict[str, Any]:
    """Validate 1:1 coverage of registry features ↔ domain members."""
    registry = set(_registry_feature_names())
    assigned: set[str] = set()
    dupes: list[str] = []
    for domain_id, members in _DOMAIN_MEMBERS.items():
        for name in members:
            if name in assigned:
                dupes.append(name)
            assigned.add(name)

    missing = sorted(registry - assigned)
    extra = sorted(assigned - registry)
    counts = domain_counts()
    ok = (
        not missing
        and not extra
        and not dupes
        and len(registry) == expected_total
        and sum(counts.values()) == expected_total
    )
    return {
        "ok": ok,
        "registry_count": len(registry),
        "assigned_count": len(assigned),
        "expected_total": expected_total,
        "missing": missing,
        "extra": extra,
        "duplicates": sorted(set(dupes)),
        "domain_counts": {DOMAIN_LABELS[d]: counts[d] for d in DOMAIN_ORDER},
        "domain_counts_by_id": dict(counts),
    }


def group_features_by_domain(features: list[str]) -> list[tuple[str, list[str]]]:
    """Ordered (domain_label, feature_names) for UI trees — skips empty domains."""
    buckets: dict[DomainId, list[str]] = {d: [] for d in DOMAIN_ORDER}
    other: list[str] = []
    for raw in features:
        name = str(raw)
        try:
            buckets[primary_domain_of(name)].append(name)
        except KeyError:
            other.append(name)
    out: list[tuple[str, list[str]]] = []
    for d in DOMAIN_ORDER:
        if buckets[d]:
            out.append((DOMAIN_LABELS[d], buckets[d]))
    if other:
        out.append(("Other", other))
    return out


__all__ = [
    "DOMAIN_LABELS",
    "DOMAIN_ORDER",
    "DataType",
    "DomainId",
    "FeatureDomainMeta",
    "all_feature_domain_meta",
    "build_feature_domain_meta",
    "domain_counts",
    "domain_label",
    "features_by_domain",
    "group_features_by_domain",
    "primary_domain_label",
    "primary_domain_of",
    "validate_domain_coverage",
]
