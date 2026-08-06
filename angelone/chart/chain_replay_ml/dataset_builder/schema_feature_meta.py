"""Tags, dependencies, compute cost, and status for ml_schema_registry columns."""

from __future__ import annotations

import re
from typing import Any

# ── Prediction target registry (versioned alongside features) ─────────────────

TARGET_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "future_ltp_3s",
        "display_name": "Future LTP (3 Seconds)",
        "description": "Option LTP exactly 3 seconds after the current sample timestamp.",
        "interpretation": "Matches the default 3s sampling grid; shortest production horizon.",
        "example": "146.50",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 3,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 3 seconds",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Beginner",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_5s",
        "display_name": "Future LTP (5 Seconds)",
        "description": "Option LTP exactly 5 seconds after the current sample timestamp.",
        "interpretation": "Ultra short-horizon regression target for fast sampling grids (e.g. 3s/5s).",
        "example": "146.80",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 5,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 5 seconds",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Beginner",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_10s",
        "display_name": "Future LTP (10 Seconds)",
        "description": "Option LTP exactly 10 seconds after the current sample timestamp.",
        "interpretation": "Short-horizon regression target for intraday premium movement.",
        "example": "147.25",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 10,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 10 seconds",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Beginner",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_30s",
        "display_name": "Future LTP (30 Seconds)",
        "description": "Option LTP exactly 30 seconds after the current sample timestamp.",
        "interpretation": "Aligns with common 10s/30s sampling; captures sub-minute repricing.",
        "example": "148.10",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 30,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 30 seconds",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Beginner",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_1m",
        "display_name": "Future LTP (1 Minute)",
        "description": "Option LTP exactly one minute after the current sample timestamp.",
        "interpretation": "Medium short-term target; smoother than 10s/30s horizons.",
        "example": "149.50",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 60,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 1 minute",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Beginner",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_3m",
        "display_name": "Future LTP (3 Minutes)",
        "description": "Option LTP exactly three minutes after the current sample timestamp.",
        "interpretation": "Between 1m and 5m horizons; captures medium intraday repricing.",
        "example": "151.20",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 180,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 3 minutes",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Intermediate",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
    {
        "name": "future_ltp_5m",
        "display_name": "Future LTP (5 Minutes)",
        "description": "Option LTP exactly five minutes after the current sample timestamp.",
        "interpretation": "Longest default horizon; captures sustained directional premium moves.",
        "example": "152.80",
        "expected_range": "0 to spot-level",
        "prediction_horizon_sec": 300,
        "unit": "₹",
        "formula_ref": "future_ltp_at_horizon",
        "formula_doc": "LTP at timestamp + 5 minutes",
        "used_by": ["training", "prediction", "audit"],
        "learning_level": "Intermediate",
        "tags": ["target", "price", "horizon"],
        "depends_on": ["ltp", "timestamp", "token"],
        "compute_cost": "low",
        "status": "production",
    },
]

# Columns explicitly marked experimental (default: production).
_EXPERIMENTAL: set[str] = set()

_DEP_ORDER: dict[str, int] = {
    "spot": 0,
    "strike": 1,
    "ltp": 2,
    "current_iv": 3,
    "minutes_to_expiry": 4,
    "expiry": 5,
    "risk_free_rate": 6,
    "option_type": 7,
    "timestamp": 8,
    "token": 9,
}

_RELATED_FEATURES: dict[str, list[str]] = {
    "delta": ["gamma", "theta", "vega", "current_iv"],
    "gamma": ["delta", "theta", "vega", "current_iv"],
    "theta": ["delta", "gamma", "vega", "current_iv"],
    "vega": ["delta", "gamma", "theta", "current_iv"],
    "current_iv": ["roll_iv", "iv_rank_session", "delta", "vega"],
    "spot": ["ltp", "atm_straddle", "current_iv", "moneyness"],
    "ltp": ["current_iv", "delta", "bid_ask_spread", "option_vwap"],
    "option_vwap": ["ltp", "bid_ask_spread"],
    "spot_open": ["spot", "spot_high", "spot_low", "spot_prev_close"],
    "spot_high": ["spot", "spot_open", "spot_low"],
    "spot_low": ["spot", "spot_open", "spot_high"],
    "spot_prev_close": ["spot", "spot_open"],
    "option_open": ["ltp", "option_high", "option_low", "option_prev_close"],
    "option_high": ["ltp", "option_open", "option_low"],
    "option_low": ["ltp", "option_open", "option_high"],
    "option_prev_close": ["ltp", "option_open"],
    "futures_ltp": ["futures_vwap", "spot", "futures_bid", "futures_ask"],
    "futures_vwap": ["futures_ltp", "spot"],
    "futures_day_volume": ["futures_ltp"],
    "futures_bid": ["futures_ask", "futures_spread", "futures_ltp"],
    "futures_ask": ["futures_bid", "futures_spread", "futures_ltp"],
    "futures_spread": ["futures_bid", "futures_ask"],
    "atm_straddle": ["current_iv", "spot", "chain_pcr"],
}

# Per-column overrides (merged on top of inference).
_COLUMN_META_OVERRIDES: dict[str, dict[str, Any]] = {}

_COMPUTE_COST_HIGH = re.compile(
    r"chain_pcr|atm_pcr|max_.*oi|build_wall|oi_wall|pinning|atm_straddle|iv_vs_atm|distance_to_max",
    re.I,
)
_COMPUTE_COST_MEDIUM = re.compile(
    r"change_|return_|zscore|rv_|ema|greek|iv_|reiv|roll_|oi_change|volume_change|flow_|body_pct|range_pct|straddle",
    re.I,
)

_TAG_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"volume|flow|spread|bid_ask|body_pct|range_pct|wick"), ["microstructure"]),
    (re.compile(r"return_|_change_|slope|ema|cross|momentum"), ["momentum"]),
    (re.compile(r"_rv_|rv_ratio|rv_5m|rv_10m"), ["realized_vol", "momentum"]),
    (re.compile(r"roll_|reiv|rows_since_roll"), ["roll"]),
    (re.compile(r"chain_|pcr|max_.*oi|build_wall|oi_wall|pinning"), ["chain_wide", "positioning"]),
    (re.compile(r"straddle"), ["atm_straddle", "volatility"]),
    (re.compile(r"delta|gamma|theta|vega|greeks"), ["greeks"]),
    (re.compile(r"^iv_|current_iv|roll_iv"), ["volatility"]),
    (re.compile(r"minutes_|days_to_expiry|expiry|is_first_hour|is_last_hour|minute_of_day"), ["session", "expiry"]),
    (re.compile(r"oi_"), ["positioning", "open_interest"]),
    (re.compile(r"prev\d|historical"), ["ohlc"]),
    (re.compile(r"bs_|dgt_|theoretical"), ["theoretical"]),
]

_GROUP_SEMANTIC_TAGS: dict[str, list[str]] = {
    "price": ["price"],
    "dgt_reiv": ["price", "theoretical", "greek_walk"],
    "ratio": ["price", "theoretical", "ratio"],
    "greeks": ["greeks", "theoretical"],
    "iv": ["volatility"],
    "oi": ["open_interest", "positioning"],
    "volume": ["microstructure", "flow"],
    "momentum": ["momentum"],
    "sharp_momentum": ["momentum", "sharp_momentum"],
    "iv_zscore": ["volatility", "iv_zscore"],
    "iv_ema_ratio": ["volatility", "ratio", "iv_ema"],
    "spot_hl": ["momentum", "spot_hl", "channel"],
    "time": ["session", "expiry"],
    "moneyness": ["moneyness"],
    "ltp_to_spot": ["moneyness", "price", "ratio"],
    "ltp_to_others": ["price", "theoretical", "ratio"],
    "spot_and_other_ratio": ["price", "ratio", "moneyness"],
    "atm_straddle": ["atm_straddle", "volatility"],
    "atm6_ltp": ["atm_straddle", "chain_wide", "price"],
    "chain": ["chain_wide", "positioning"],
    "historical": ["ohlc"],
    "advanced": ["roll", "flow", "microstructure", "ratio"],
}


def _sort_deps(deps: list[str] | set[str]) -> list[str]:
    items = list(dict.fromkeys(deps))
    return sorted(items, key=lambda x: (_DEP_ORDER.get(x, 99), x))


def infer_related_features(name: str, group_id: str = "") -> list[str]:
    override = _COLUMN_META_OVERRIDES.get(name, {}).get("related_features")
    if override is not None:
        return list(override)
    if name in _RELATED_FEATURES:
        return list(_RELATED_FEATURES[name])
    return []


def infer_tags(name: str, group_id: str = "", col_type: str = "feature") -> list[str]:
    tags: set[str] = set()
    if col_type == "target":
        tags.update(["target", "price", "horizon"])
    elif col_type == "metadata":
        tags.add("metadata")
    else:
        tags.update(_GROUP_SEMANTIC_TAGS.get(group_id, []))
        if group_id:
            tags.add(group_id)
    for pattern, extra in _TAG_PATTERNS:
        if pattern.search(name):
            tags.update(extra)
    override = _COLUMN_META_OVERRIDES.get(name, {}).get("tags")
    if override:
        tags.update(override)
    return sorted(tags)


def infer_compute_cost(name: str, group_id: str = "", col_type: str = "feature") -> str:
    override = _COLUMN_META_OVERRIDES.get(name, {}).get("compute_cost")
    if override:
        return str(override)
    if col_type in ("metadata", "target"):
        return "low"
    if group_id == "chain" or _COMPUTE_COST_HIGH.search(name):
        return "high"
    if group_id in ("oi", "atm_straddle") and (
        "distance" in name or "max_" in name or "wall" in name or "pinning" in name or name == "atm_straddle"
    ):
        return "high"
    if _COMPUTE_COST_MEDIUM.search(name) or group_id in (
        "greeks", "iv", "iv_zscore", "iv_ema_ratio", "momentum", "sharp_momentum", "spot_hl", "historical",
    ):
        return "medium"
    return "low"


def infer_status(name: str, col_type: str = "feature") -> str:
    override = _COLUMN_META_OVERRIDES.get(name, {}).get("status")
    if override:
        return str(override)
    if name in _EXPERIMENTAL:
        return "experimental"
    return "production"


def infer_depends_on(name: str, group_id: str = "", col_type: str = "feature") -> list[str]:
    override = _COLUMN_META_OVERRIDES.get(name, {}).get("depends_on")
    if override is not None:
        return list(override)

    if col_type == "metadata":
        if name == "timestamp":
            return []
        if name in ("trading_day", "market", "expiry"):
            return []
        if name in ("strike", "option_type", "token", "symbol"):
            return ["token"]
        return []

    if col_type == "target":
        return ["ltp", "timestamp", "token"]

    if name == "roll_iv":
        return sorted({"timestamp", "token", "ltp", "spot", "strike", "minutes_to_expiry", "option_type"})

    deps: set[str] = {"timestamp"}

    # Atomic / identity inputs
    if name == "spot":
        return ["timestamp"]
    if name in ("ltp", "bid_ask_spread", "option_vwap"):
        return sorted({"timestamp", "token"})
    if name in (
        "spot_open", "spot_high", "spot_low", "spot_prev_close",
    ):
        return ["timestamp"]
    if name in (
        "option_open", "option_high", "option_low", "option_prev_close",
    ):
        return sorted({"timestamp", "token"})
    if name in ("futures_ltp", "futures_vwap", "futures_day_volume",
                "futures_bid", "futures_ask", "futures_spread"):
        return ["timestamp"]
    if name in ("strike", "option_type", "token", "symbol"):
        return ["token"] if name != "token" else []

    # Time features
    if group_id == "time" or name in ("minute_of_day",):
        if name in ("minutes_to_expiry", "days_to_expiry", "is_expiry_day"):
            return sorted({"timestamp", "expiry"})
        if name in ("minutes_since_open", "minutes_to_close", "is_first_hour", "is_last_hour"):
            return ["timestamp"]
        return ["timestamp"]

    # Moneyness
    if group_id == "moneyness" or name in (
        "distance_from_spot_pct", "distance_from_atm_pct", "distance_from_atm_points",
        "strike_distance_from_atm", "moneyness", "strike_to_spot_ratio", "ltp_to_spot_ratio", "is_call",
        "moneyness_delta_ltp_to_spot_ratio",
    ) or name.startswith("ltp_to_spot_ratio"):
        deps.update(["spot", "strike", "option_type"])
        if name == "ltp_to_spot_ratio" or name.startswith("ltp_to_spot_ratio"):
            deps.add("ltp")
        if name == "moneyness_delta_ltp_to_spot_ratio":
            deps.update(["abs_delta", "delta", "ltp"])
        return sorted(deps)

    # Roll / re-anchor family
    if name in ("roll_iv", "roll_age_min", "rows_since_roll", "bs_reiv_pred", "dgt_reiv_pred", "iv_drift_from_roll"):
        base = {"spot", "strike", "ltp", "minutes_to_expiry", "option_type", "timestamp"}
        if name in ("bs_reiv_pred", "iv_drift_from_roll"):
            base.add("roll_iv")
        if name == "dgt_reiv_pred":
            base.update(["roll_iv", "delta", "gamma", "theta"])
        if name in ("roll_age_min", "rows_since_roll"):
            base.add("roll_iv")
        return sorted(base)

    if name == "ltp_to_dgt_reiv_ratio":
        return sorted({"ltp", "dgt_reiv_pred", "timestamp", "token"})
    if name == "ltp_to_bs_reiv_ratio":
        return sorted({"ltp", "bs_reiv_pred", "timestamp", "token"})
    if name == "dgt_reiv_to_ltp_ratio":
        return sorted({"ltp", "dgt_reiv_pred", "timestamp", "token"})
    if name == "bs_reiv_to_ltp_ratio":
        return sorted({"ltp", "bs_reiv_pred", "timestamp", "token"})
    if name == "side_to_ltp_ratio":
        return sorted({"ce_atm6_ltp_sum", "ltp", "option_type", "pe_atm6_ltp_sum", "timestamp", "token"})
    if name == "current_to_atm6_flow_delta_ltp_to_spot_ratio":
        return sorted({
            "abs_delta", "delta", "ltp", "oi", "option_type", "spot", "strike", "timestamp", "token", "volume",
        })
    if name == "delta_ltp_to_spot_ratio":
        return sorted({"delta", "ltp", "spot", "timestamp", "token"})
    if name == "gamma_ltp_to_spot_ratio":
        return sorted({"gamma", "ltp", "spot", "timestamp", "token"})
    if name == "moneyness_delta_ltp_to_spot_ratio":
        return sorted({"moneyness", "abs_delta", "delta", "ltp", "spot", "strike", "timestamp", "token"})
    if name == "atm6_total_to_ltp_ratio":
        return sorted({"ce_atm6_ltp_sum", "ltp", "pe_atm6_ltp_sum", "timestamp", "token"})
    if name == "atm6_total_to_spot_ratio":
        return sorted({"ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "spot", "timestamp"})
    if name in (
        "ltp_ema9_to_spot_ratio", "ltp_ema20_to_spot_ratio",
        "ltp_ema50_to_spot_ratio", "ltp_ema100_to_spot_ratio", "ltp_ema200_to_spot_ratio",
    ):
        return sorted({"ltp", "spot", "timestamp", "token"})
    if name.endswith("_to_spot_ratio_x_moneyness") and name.startswith("ltp_ema"):
        return sorted({"ltp", "spot", "moneyness", "timestamp", "token"})
    if name.endswith("_to_ltp_ratio_x_moneyness") and name.startswith("spot_ema"):
        return sorted({"ltp", "spot", "moneyness", "timestamp", "token"})
    if name == "ltp_std20_to_spot_ratio":
        return sorted({"ltp", "spot", "timestamp", "token"})
    if name.startswith("ltp_ema") and name.endswith("_to_ltp_ratio"):
        return sorted({"ltp", "timestamp", "token"})
    if name == "ltp_std20_to_ltp_ratio":
        return sorted({"ltp", "timestamp", "token"})
    if name.startswith("spot_ema") and name.endswith("_to_ltp_ratio"):
        return sorted({"ltp", "spot", "timestamp", "token"})
    if name == "dgt_to_spot_ratio":
        return sorted({"dgt_reiv_pred", "spot", "timestamp", "token"})
    if name == "bs_to_spot_ratio":
        return sorted({"bs_reiv_pred", "spot", "timestamp", "token"})
    if name.startswith("dgt_reiv_pred_lag_") or name.startswith("dgt_reiv_pred_change_"):
        return sorted({"dgt_reiv_pred", "timestamp", "token"})
    if name == "dgt_prediction_error" or name.startswith("dgt_prediction_error"):
        return sorted({"dgt_reiv_pred", "ltp", "timestamp", "token"})

    # IV family
    if group_id == "iv" or name.startswith("iv_") or name == "current_iv":
        deps.update(["ltp", "spot", "strike", "minutes_to_expiry", "option_type"])
        if name in ("iv_rank_session", "iv_zscore_30m", "iv_zscore_1m", "iv_zscore_5m", "iv_zscore_15m"):
            deps.add("current_iv")
        if name == "iv_vs_atm":
            deps.update(["current_iv", "atm_straddle"])
        if name in ("iv_drift_from_roll", "roll_iv"):
            deps.add("roll_iv")
        return sorted(deps)

    # Greeks
    if group_id == "greeks":
        base = ["spot", "strike", "current_iv", "minutes_to_expiry", "risk_free_rate"]
        if name.endswith("_change_5m"):
            base.append(name.replace("_change_5m", ""))
        return _sort_deps(base)

    # Chain-wide (scan entire chain)
    if group_id == "chain":
        deps.update(["spot", "timestamp"])
        if "pcr" in name:
            deps.update(["oi"])
        if name.startswith("spot_dist") or name == "spot_range_pos_5m":
            return sorted({"spot", "timestamp"})
        return sorted(deps)

    # OI walls / chain aggregates
    if group_id == "oi":
        deps.update(["timestamp", "token"])
        if any(x in name for x in ("distance", "max_", "wall", "pinning", "bias")):
            deps.add("oi")
            return sorted(deps)
        if "oi" in name:
            deps.add("oi")
        return sorted(deps)

    # ATM+6 wing LTP aggregates
    if name in ("ce_atm6_ltp_sum", "pe_atm6_ltp_sum"):
        return sorted({"spot", "timestamp", "ltp"})
    if name in ("ce_atm6_ltp_to_spot_ratio", "pe_atm6_ltp_to_spot_ratio"):
        base = name.replace("_to_spot_ratio", "_sum")
        return sorted({"spot", "timestamp", base})
    if name == "ce_pe_atm6_ltp_ratio":
        return sorted({"timestamp", "ce_atm6_ltp_sum", "pe_atm6_ltp_sum"})
    if name == "ce_minus_pe_atm6_ltp":
        return sorted({"timestamp", "ce_atm6_ltp_sum", "pe_atm6_ltp_sum"})
    if name == "ce_pe_atm6_ltp_diff_pct":
        return sorted({"timestamp", "ce_atm6_ltp_sum", "pe_atm6_ltp_sum"})

    # ATM straddle
    if group_id == "atm_straddle":
        deps.update(["spot", "strike", "timestamp", "ltp"])
        if "zscore" in name:
            deps.add("atm_straddle")
        return sorted(deps)

    # Volume / microstructure
    if name.startswith("ltp_x_volume_change_pct_"):
        return sorted({"timestamp", "token", "volume", "ltp"})
    if group_id == "volume":
        return sorted({"timestamp", "token", "volume"})

    # Momentum / EMA / RV
    if group_id == "momentum":
        deps.add("spot")
        if name.startswith("opt_rv"):
            deps.update(["ltp", "token"])
        if "ema" in name or "cross" in name:
            deps.add("spot")
        return sorted(deps)

    if group_id == "sharp_momentum":
        deps.update(["spot", "ltp"])
        if name == "weighted_ltp_ema_to_ltp_ratio":
            deps.update([
                "ltp_ema9_to_ltp_ratio",
                "ltp_ema20_to_ltp_ratio",
                "ltp_ema50_to_ltp_ratio",
                "ltp_ema200_to_ltp_ratio",
            ])
        elif name == "weighted_spot_ema_to_ltp_ratio":
            deps.update([
                "spot_ema9_to_ltp_ratio",
                "spot_ema20_to_ltp_ratio",
                "spot_ema50_to_ltp_ratio",
                "spot_ema200_to_ltp_ratio",
            ])
        elif name.startswith("weighted_ltp") or name.startswith("ltp_to_"):
            deps.add("token")
        return sorted(deps)

    if group_id == "iv_zscore":
        deps.update(["current_iv", "ltp", "spot"])
        if name.startswith("weighted_spot_ema") or name == "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio":
            deps.add("weighted_spot_ema_to_ltp_ratio")
        if "delta" in name:
            deps.add("delta")
        if "iv_zscore_1m" in name:
            deps.add("iv_zscore_1m")
        if "iv_zscore_5m" in name:
            deps.add("iv_zscore_5m")
        if "iv_zscore_15m" in name:
            deps.add("iv_zscore_15m")
        if name == "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio":
            deps.update(["iv_zscore_1m", "iv_zscore_5m", "iv_zscore_15m"])
        return sorted(deps)

    if group_id == "iv_ema_ratio":
        deps.update(["current_iv", "ltp", "spot", "timestamp", "token"])
        if "moneyness" in name:
            deps.add("moneyness")
        return sorted(deps)

    if group_id == "spot_hl":
        deps.update(["spot", "ltp"])
        return sorted(deps)

    # OHLC / historical bars
    if group_id == "historical" or "body_pct" in name or "range_pct" in name:
        if name.startswith("opt_"):
            deps.update(["ltp", "token"])
        else:
            deps.add("spot")
        return sorted(deps)

    # Price returns / changes
    if group_id == "price":
        if name.startswith("spot_"):
            deps.add("spot")
        elif name.startswith("ltp") or name.startswith("opt_"):
            deps.update(["ltp", "token"])
        if name in ("bs_reiv_pred", "dgt_reiv_pred", "ltp_to_dgt_reiv_ratio"):
            return infer_depends_on(name, "advanced", col_type)
        return sorted(deps)

    if group_id == "dgt_reiv":
        return infer_depends_on(name, "advanced", col_type)

    # Advanced
    if group_id == "advanced":
        if name == "opt_volume_acc_5s_1m":
            return sorted({"timestamp", "token", "volume"})
        if name == "spot_vol_ratio_10s_1m":
            return sorted({"spot", "timestamp"})
        if name in ("roll_age_min", "rows_since_roll"):
            return infer_depends_on(name, "", col_type)
        if name.startswith("weighted_spot_ema_to_ltp_ratio_x_moneyness"):
            deps.update(["weighted_spot_ema_to_ltp_ratio", "moneyness"])
            if "delta" in name:
                deps.add("delta")
            return sorted(deps)

    return sorted(deps)


def build_column_meta_extras(
    name: str,
    *,
    group_id: str = "",
    col_type: str = "feature",
    doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tags, depends_on, compute_cost, status, and optional importance for a column."""
    doc = doc or {}
    extras: dict[str, Any] = {
        "tags": infer_tags(name, group_id, col_type),
        "depends_on": infer_depends_on(name, group_id, col_type),
        "compute_cost": infer_compute_cost(name, group_id, col_type),
        "status": infer_status(name, col_type),
    }
    if doc.get("tags"):
        extras["tags"] = sorted(set(extras["tags"]) | set(doc["tags"]))
    if doc.get("depends_on") is not None:
        extras["depends_on"] = list(doc["depends_on"])
    if doc.get("compute_cost"):
        extras["compute_cost"] = doc["compute_cost"]
    if doc.get("status"):
        extras["status"] = doc["status"]
    importance = doc.get("importance")
    if isinstance(importance, dict) and importance:
        extras["importance"] = dict(importance)
    if doc.get("related_features") is not None:
        extras["related_features"] = list(doc["related_features"])
    elif col_type == "feature":
        related = infer_related_features(name, group_id)
        if related:
            extras["related_features"] = related
    return extras


def build_targets_registry(target_defs: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Top-level targets map for ml_schema_registry.json."""
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(target_defs or TARGET_DEFINITIONS, start=1):
        name = str(raw["name"])
        entry = {
            "id": f"target_{i:03d}",
            "name": name,
            **{k: v for k, v in raw.items() if k != "name"},
        }
        if "tags" not in entry:
            entry["tags"] = infer_tags(name, "", "target")
        if "depends_on" not in entry:
            entry["depends_on"] = infer_depends_on(name, "", "target")
        if "compute_cost" not in entry:
            entry["compute_cost"] = "low"
        if "status" not in entry:
            entry["status"] = "production"
        out[name] = entry
    return out


def all_tags_from_columns(columns: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for col in columns.values():
        for tag in col.get("tags") or []:
            found.add(str(tag))
    return sorted(found)
