"""Map registry feature groups to columns produced by dataset feature extractors."""

from __future__ import annotations

from typing import Any

# Registry names on the left where they differ from extractor keys.
_EXTRACTOR_ALIASES: dict[str, str] = {
    "current_iv": "iv",
    "oi": "oi",
    "volume": "volume",
    # price_dist_from_cross_pct is already the extractor key in features_atm_band
}

# All registry features map to extractor column names (identity unless aliased).
_REGISTRY_FEATURES: dict[str, list[str]] = {
    "price": [
        "spot", "ltp", "bid_ask_spread", "option_vwap",
        "option_bid", "option_ask",
        "spot_open", "spot_high", "spot_low", "spot_prev_close",
        "option_open", "option_high", "option_low", "option_prev_close",
        "futures_ltp", "futures_vwap",
        "futures_day_volume", "futures_oi", "futures_bid", "futures_ask", "futures_spread",
        "bs_reiv_pred", "dgt_reiv_pred",
        # Packaging / transforms → Pipeline only
    ],
    # Wave A: Market Microstructure Controller levels (token.book).
    "market_microstructure": [
        "mid_price",
        "microprice",
        "microprice_bias",
        "book_imbalance_l1",
        "book_imbalance_l1_5",
        "bid_depth_l1_5",
        "ask_depth_l1_5",
        "book_depth_slope_bid",
        "book_depth_slope_ask",
    ],
    "dgt_reiv": [
        # Lag/change/error_lag/error_change → Pipeline Owned or retired (10s).
        "dgt_prediction_error",
    ],
    "ratio": [
        # ltp_to_bs_reiv_ratio / dgt_to_spot_ratio / bs_to_spot_ratio → Interaction
    ],
    "greeks": [
        "delta", "abs_delta", "gamma", "theta", "vega",
        "vanna", "volga", "charm", "speed",
        # delta_x_spot / gamma_x_spot → Interaction pipeline only
        "theta_per_min", "vega_per_ivpt",
        # delta/gamma/theta_change_5m → Pipeline Owned
    ],
    "iv": [
        "current_iv", "roll_iv",
        # iv_change_* / iv_pct_change_1m → Pipeline Owned
        # iv_zscore_30m → Pipeline Owned (rolling_statistics)
        "iv_drift_from_roll", "iv_rank_session", "iv_vs_atm",
        "atm_iv_ce", "atm_iv_pe",
    ],
    # iv_zscore_* → Pipeline Owned; weighted_*_x_* composites → Interaction pipeline only
    "iv_zscore": [],
    "iv_ema_ratio": [
        # Wave 2: canonical IV EMA levels (ratios → Interaction)
        "iv_ema9", "iv_ema20", "iv_ema50", "iv_ema100", "iv_ema200", "iv_ema300",
    ],
    "oi": [
        "option_oi",
        # oi_change_* / oi_change_pct_* / oi_velocity_1m → Pipeline Owned
        "distance_to_max_call_oi_strikes", "distance_to_max_put_oi_strikes",
        "max_call_oi_pct", "max_put_oi_pct",
        "distance_to_call_build_wall", "distance_to_put_build_wall",
        "oi_wall_bias", "pinning_pressure",
        "total_call_oi", "total_put_oi",
        "oi_abs_delta_0_20_ce", "oi_abs_delta_0_20_pe",
        "oi_abs_delta_20_40_ce", "oi_abs_delta_20_40_pe",
        "oi_abs_delta_40_60_ce", "oi_abs_delta_40_60_pe",
        "oi_abs_delta_60_80_ce", "oi_abs_delta_60_80_pe",
        "oi_abs_delta_80_100_ce", "oi_abs_delta_80_100_pe",
    ],
    "volume": [
        "option_day_volume", "ltq", "total_buy_qty", "total_sell_qty",
        "total_ce_volume", "total_pe_volume",
        "otm_ce_volume", "otm_pe_volume",
        # volume_change_* → Pipeline Owned (5s retired)
        # opt_volume_flow_{15s,30s,1m} → Pipeline Owned; *_flow_5s retired
        # ltp_x_volume_change_pct_* → Pipeline Owned (÷3); *_10s retired
    ],
    "momentum": [
        # Wave 6: spot_vs_ema20_pct / ema_spread_pct / ema_spread_vs_spot_pct → Interaction
        "ema9_slope", "ema9_gt_ema20",
        "time_since_cross_min", "cross_age_decay",
        "price_dist_from_cross_pct",
        "spot_rv_5m", "spot_rv_10m",
        # spot_rv_ratio → Interaction (spot_rv_5m / spot_rv_10m)
        # Wave B: IV − spot RV
        "iv_rv_spread_5m", "iv_rv_spread_10m",
        "opt_rv_5m", "opt_rv_10m",
        # opt_rv_ratio → Interaction (opt_rv_5m / opt_rv_10m)
    ],
    "sharp_momentum": [
        # Wave 3/4: weighted + sharp levels; packaging → Interaction
        "weighted_spot_ema",
        "weighted_ltp_ema",
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
    ],
    "spot_hl": [
        # Wave 2: canonical HL EMA levels (÷ltp / channel width → Interaction)
        "spot_high_ema20", "spot_high_ema50", "spot_high_ema100",
        "spot_high_ema200", "spot_high_ema300",
        "spot_low_ema20", "spot_low_ema50", "spot_low_ema100",
        "spot_low_ema200", "spot_low_ema300",
        # Wave 3: weighted HL blend levels (packaging → Interaction)
        "weighted_spot_high_ema",
        "weighted_spot_low_ema",
        "weighted_spot_close_ema",  # HL close blend; used by historical cross-features
        # Wave 5: channel width levels (ltp÷width → Interaction)
        "spot_ema20_channel_width",
        "spot_ema50_channel_width",
        "spot_ema100_channel_width",
        "spot_ema200_channel_width",
        "spot_ema300_channel_width",
    ],
    "time": [
        "minutes_to_expiry", "minutes_since_open", "minutes_to_close",
        "is_first_hour", "is_last_hour", "minute_of_day", "days_to_expiry", "is_expiry_day",
    ],
    "moneyness": [
        "strike", "distance_from_spot_pct", "distance_from_atm_pct",
        "distance_from_atm_points", "strike_distance_from_atm", "moneyness", "is_call",
        # KEEP foundational Base: moneyness, strike_to_spot_ratio, ltp_to_spot_ratio
        "strike_to_spot_ratio", "ltp_to_spot_ratio",
    ],
    # ltp_to_spot lag/change families → Pipeline Owned (10s retired); group removed.
    "ltp_to_others": [
        # Wave 2: canonical LTP EMA + std20 levels (÷ltp/÷spot → Interaction)
        "ltp_ema9", "ltp_ema20", "ltp_ema50", "ltp_ema100", "ltp_ema200", "ltp_ema300",
        "ltp_std20",
        "side_to_ltp_ratio",  # EDGE: CONDITIONAL_COLUMN_SELECTION (see EDGE_FEATURES)
    ],
    "spot_and_other_ratio": [
        # Wave 2: canonical spot EMA levels (÷ltp → Interaction)
        "spot_ema9", "spot_ema20", "spot_ema50", "spot_ema100", "spot_ema200", "spot_ema300",
    ],
    # Multi-TF NIFTY EMAs from angel_historic_bars (as-of join; not tick-stream EMAs).
    "historic_spot_ema": [
        "spot_1m_ema9", "spot_1m_ema20", "spot_1m_ema50", "spot_1m_ema100", "spot_1m_ema200",
        "spot_3m_ema9", "spot_3m_ema20", "spot_3m_ema50", "spot_3m_ema100", "spot_3m_ema200",
        "spot_5m_ema9", "spot_5m_ema20", "spot_5m_ema50", "spot_5m_ema100", "spot_5m_ema200",
        "spot_15m_ema9", "spot_15m_ema20", "spot_15m_ema50", "spot_15m_ema100", "spot_15m_ema200",
    ],
    "atm_straddle": [
        "atm_straddle",
        # change / change_pct → Pipeline Owned
        # zscore_30m / slope / accel / pct_change_from_open → Pipeline Owned
        # atm_straddle_zscore_change_5m → Pipeline Owned (Difference of zscore_30m)
        # Wave 6: ce_pe_atm6_ltp_diff_pct → Interaction
    ],
    "atm6_ltp": [
        "ce_atm6_ltp_sum", "pe_atm6_ltp_sum",
        # ce/pe_atm6_ltp_to_spot_ratio / ce_pe_atm6_ltp_ratio → Interaction
        # Wave 6: ce_pe_atm6_ltp_diff_pct → Interaction
        "ce_minus_pe_atm6_ltp",
    ],
    "chain": [
        "chain_pcr", "atm_pcr",
        # Wave B: token.chain IV skew + delta flow + GEX
        "iv_skew_atm", "iv_call_put_skew", "iv_skew_25d", "iv_butterfly_25d",
        "delta_w_volume_flow_1m", "delta_w_volume_flow_5m",
        "call_gex", "put_gex", "net_gex", "chain_gex",
        "gamma_flip_spot", "gamma_flip_distance",
        "synthetic_forward_spot",
        "otm_pcr_volume",
        # chain_pcr_change_5m / atm_pcr_change_5m → Pipeline Owned
        # spot_dist_*_5m / spot_range_pos_5m → Pipeline Owned (rolling_ohlc)
    ],
    # historical prev-candle / tick-OHLC 10s family → retired (∤3); group cleared.
    "historical": [],
    "advanced": [
        "roll_age_min", "rows_since_roll",
        # opt_volume_acc_5s_1m / spot_vol_ratio_10s_1m → retired (5∤3 / 10∤3)
        # delta/gamma/moneyness_delta_ltp_to_spot_ratio → Interaction (generic registry math)
        # current_to_atm6_flow_delta_ltp_to_spot_ratio → pipeline experiment engineering
        # weighted_spot_ema_to_ltp_ratio_x_* → Interaction pipeline only
    ],
}


def _build_group_sources() -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for gid, feats in _REGISTRY_FEATURES.items():
        mapping: dict[str, str | None] = {}
        for feat in feats:
            mapping[feat] = _EXTRACTOR_ALIASES.get(feat, feat)
        out[gid] = mapping
    return out


GROUP_FEATURE_SOURCES: dict[str, dict[str, str | None]] = _build_group_sources()


def plugin_columns_missing_from_schema(columns: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build column metadata for implemented plugin features absent from on-disk schema."""
    from .feature_registry_catalog import _build_column_from_plugins

    out: dict[str, dict[str, Any]] = {}
    for gid, feats in _REGISTRY_FEATURES.items():
        for feat in feats:
            if feat not in columns:
                out[feat] = _build_column_from_plugins(feat, gid)
    return out


def plugin_groups_missing_from_schema(
    groups: dict[str, Any],
    *,
    group_labels: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Merge plugin group definitions into legacy schema groups metadata."""
    from .feature_registry_catalog import _GROUP_LABELS

    labels = group_labels or _GROUP_LABELS
    merged = dict(groups)
    order_add: list[str] = []
    for gid, feats in _REGISTRY_FEATURES.items():
        existing = list((merged.get(gid) or {}).get("features") or [])
        plugin_feats = list(feats)
        if gid not in merged:
            merged[gid] = {"label": labels.get(gid, gid), "features": plugin_feats}
            order_add.append(gid)
            continue
        updated = list(existing)
        changed = False
        for feat in plugin_feats:
            if feat not in updated:
                updated.append(feat)
                changed = True
        if changed:
            merged[gid] = {**merged[gid], "features": updated}
    return merged, order_add


def horizon_column_name(sec: int) -> str:
    if sec == 60:
        return "future_ltp_1m"
    if sec == 180:
        return "future_ltp_3m"
    if sec == 300:
        return "future_ltp_5m"
    if sec % 60 == 0 and sec >= 60:
        return f"future_ltp_{sec // 60}m"
    return f"future_ltp_{sec}s"


def horizon_label(sec: int) -> str:
    if sec == 60:
        return "1m"
    if sec == 180:
        return "3m"
    if sec == 300:
        return "5m"
    if sec % 60 == 0 and sec >= 60:
        return f"{sec // 60}m"
    return f"{sec}s"


def implemented_features_from_names(
    feature_names: list[str],
    registry: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Return (implemented_cols, pending_cols, per_group_implemented) for explicit feature names."""
    groups_meta = registry.get("groups") or {}
    group_order = registry.get("groupOrder") or []
    feat_to_gid: dict[str, str] = {}
    for gid in group_order:
        for feat in (groups_meta.get(gid) or {}).get("features") or []:
            feat_to_gid[feat] = gid

    implemented: list[str] = []
    pending: list[str] = []
    per_group: dict[str, list[str]] = {}
    for feat in feature_names:
        gid = feat_to_gid.get(feat)
        if not gid:
            pending.append(feat)
            continue
        mapping = GROUP_FEATURE_SOURCES.get(gid, {})
        src = mapping.get(feat)
        if src is None:
            pending.append(feat)
            continue
        implemented.append(feat)
        per_group.setdefault(gid, []).append(feat)
    return implemented, pending, per_group


def resolve_implemented_features_for_selection(
    feature_selection: dict[str, Any],
    registry: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """Return (enabled_groups, implemented, pending, per_group) from feature_selection config."""
    from .schema_registry import resolve_feature_selection

    enabled_groups, feature_names = resolve_feature_selection(feature_selection, registry)
    if data_dir:
        from .feature_registry_store import disabled_registry_feature_names, load_store

        excluded = disabled_registry_feature_names(load_store(data_dir))
        if excluded:
            feature_names = [f for f in feature_names if f not in excluded]
    implemented, pending, per_group = implemented_features_from_names(feature_names, registry)
    return enabled_groups, implemented, pending, per_group


def implemented_features_for_groups(
    enabled_groups: list[str],
    registry: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Return (implemented_cols, pending_cols, per_group_implemented)."""
    implemented: list[str] = []
    pending: list[str] = []
    per_group: dict[str, list[str]] = {}
    groups_meta = registry.get("groups") or {}
    for gid in enabled_groups:
        group_feats = list((groups_meta.get(gid) or {}).get("features") or [])
        if not group_feats:
            group_feats = list(_REGISTRY_FEATURES.get(gid) or [])
        mapping = GROUP_FEATURE_SOURCES.get(gid, {})
        group_impl: list[str] = []
        for feat in group_feats:
            src = mapping.get(feat)
            if src is None and feat not in mapping:
                pending.append(feat)
                continue
            if src is None:
                pending.append(feat)
                continue
            group_impl.append(feat)
            if feat not in implemented:
                implemented.append(feat)
        if group_impl:
            per_group[gid] = group_impl
    return implemented, pending, per_group


def pick_features_from_row(
    raw: dict[str, Any],
    feature_names: list[str],
    enabled_groups: list[str],
) -> dict[str, Any]:
    from .feature_migration import is_pipeline_owned, is_retired
    from .feature_ownership import is_interaction_feature

    out: dict[str, Any] = {}
    for gid in enabled_groups:
        mapping = GROUP_FEATURE_SOURCES.get(gid, {})
        for feat in feature_names:
            if feat not in mapping:
                continue
            if is_pipeline_owned(feat) or is_retired(feat) or is_interaction_feature(feat):
                continue
            src = mapping.get(feat)
            if not src:
                continue
            val = raw.get(src)
            if feat == "current_iv" and val is not None:
                val = float(val) * 100.0 if val <= 1.0 else float(val)
            out[feat] = val
    return out
