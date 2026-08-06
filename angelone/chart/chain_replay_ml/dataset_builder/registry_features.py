"""Single registry feature row builder — shared by dataset builder and live prediction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from chain_replay_ml.features_atm_band import extract_timeline_features, find_atm_strike

from .chain_maps import ChainMaps
from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC
from .extended_features import OptionFeatureState, enrich_dataset_features, enrich_with_chain_maps
from .current_to_atm6_flow import enrich_current_to_atm6_flow_features, needs_current_to_atm6_flow
from .iv_zscore_features import enrich_iv_zscore_features, needs_iv_zscore_composites
from .iv_ema_ratio_features import enrich_iv_ema_ratio_features, needs_iv_ema_ratio_features
from .spot_ratio_moneyness_features import enrich_spot_ratio_moneyness_features, needs_spot_ratio_moneyness
from .advanced_composite_features import enrich_advanced_composite_features, needs_advanced_composites
from .feature_grid_policy import resolve_feature_grid_step_sec
from .feature_plugins import GROUP_FEATURE_SOURCES, pick_features_from_row
from .rolling_controllers import SpotControllers
from .spot_hl_registry import (
    enrich_spot_hl_composite_registry_features,
    enrich_spot_hl_ratio_registry_features,
    needs_spot_hl_composite_registry,
    needs_spot_hl_ratio_registry,
)
from .spot_momentum_registry import enrich_spot_momentum_registry_features, needs_spot_momentum_registry


def enabled_registry_feature_names(enabled_groups: Sequence[str]) -> list[str]:
    all_feats: list[str] = []
    for gid in enabled_groups:
        mapping = GROUP_FEATURE_SOURCES.get(str(gid), {})
        for feat in mapping:
            if feat not in all_feats:
                all_feats.append(feat)
    return all_feats


def build_registry_features_at_ts(
    *,
    ts: float,
    strike: float,
    option_type: str,
    opt_tl: Any,
    index_tl: Any,
    strike_mapping: Mapping[tuple[float, str], tuple[str, str, Any]],
    chain_maps: ChainMaps,
    opt_state: OptionFeatureState,
    strike_step: int,
    expiry_ts: float,
    open_ts: float,
    close_ts: float,
    enabled_groups: Sequence[str],
    trading_day: str = "",
    expiry_norm: str = "",
    lookback_policy_doc: dict[str, Any] | None = None,
    atm_strike: int | None = None,
    active_features: frozenset[str] | None = None,
    feature_grid_step_sec: float | None = None,
    spot_controllers: SpotControllers | None = None,
) -> dict[str, Any]:
    """
    Build registry-named features for one option at ``ts``.

    Same pipeline as dataset builder Stage 6:
    extract_timeline_features → enrich_dataset_features → enrich_with_chain_maps → pick_features_from_row
    """
    spot = index_tl.ltp_rupees_at(float(ts))
    if spot is None or spot <= 0:
        return {}
    atm = int(atm_strike) if atm_strike is not None else find_atm_strike(float(spot), int(strike_step))
    groups = [str(g) for g in enabled_groups if str(g) in GROUP_FEATURE_SOURCES]
    all_feats = enabled_registry_feature_names(groups)
    feat_active = active_features or frozenset(all_feats)
    grid_step = resolve_feature_grid_step_sec(
        sampling=None,
        fallback=int(feature_grid_step_sec) if feature_grid_step_sec is not None else None,
    )
    raw = extract_timeline_features(
        ts=float(ts),
        index_timeline=index_tl,
        option_timeline=opt_tl,
        option_type=str(option_type).upper(),
        strike_rupees=float(strike),
        atm_strike_price=atm,
        expiry_ts=float(expiry_ts),
        open_ts=float(open_ts),
        close_ts=float(close_ts),
        feature_grid_step_sec=grid_step,
    )
    raw["option_type"] = str(option_type).upper()
    raw = enrich_dataset_features(
        raw,
        ts=float(ts),
        option_timeline=opt_tl,
        index_timeline=index_tl,
        option_type=str(option_type).upper(),
        strike_rupees=float(strike),
        atm_strike=atm,
        strike_step=int(strike_step),
        expiry_ts=float(expiry_ts),
        open_ts=float(open_ts),
        close_ts=float(close_ts),
        trading_day=str(trading_day or ""),
        expiry_norm=str(expiry_norm or ""),
        opt_state=opt_state,
        lookback_policy_doc=lookback_policy_doc,
    )
    raw = enrich_with_chain_maps(
        raw,
        ts=float(ts),
        chain_maps=chain_maps,
        strike_mapping=dict(strike_mapping),
        index_tl=index_tl,
        atm_strike=atm,
        expiry_ts=float(expiry_ts),
        opt_state=opt_state,
        option_timeline=opt_tl,
        open_ts=float(open_ts),
        close_ts=float(close_ts),
        active_features=feat_active,
        feature_grid_step_sec=float(grid_step),
    )
    if needs_current_to_atm6_flow(feat_active):
        raw = enrich_current_to_atm6_flow_features(
            raw,
            ts=float(ts),
            strike_mapping=dict(strike_mapping),
            strike_rupees=float(strike),
            strike_step=int(strike_step),
            option_type=str(option_type).upper(),
            active_features=feat_active,
        )
    if spot_controllers is not None:
        from .tick_coverage import spot_tick_bounds

        bounds = spot_tick_bounds(index_tl)
        grid_origin = float(bounds[0]) if bounds else float(open_ts)
        spot_controllers.update(
            spot,
            ts=float(ts),
            grid_step_sec=grid_step,
            index_tl=index_tl,
            grid_origin_ts=grid_origin,
        )
    if needs_spot_momentum_registry(feat_active):
        raw = enrich_spot_momentum_registry_features(
            raw,
            ts=float(ts),
            spot_controllers=spot_controllers,
            active_features=feat_active,
        )
    if needs_spot_hl_ratio_registry(feat_active):
        raw = enrich_spot_hl_ratio_registry_features(
            raw,
            spot_controllers=spot_controllers,
            active_features=feat_active,
        )
    if needs_spot_hl_composite_registry(feat_active):
        raw = enrich_spot_hl_composite_registry_features(
            raw,
            spot_controllers=spot_controllers,
            active_features=feat_active,
        )
    if needs_iv_zscore_composites(feat_active):
        raw = enrich_iv_zscore_features(
            raw,
            active_features=feat_active,
            spot_controllers=spot_controllers,
            ts=float(ts),
        )
    if needs_iv_ema_ratio_features(feat_active):
        raw = enrich_iv_ema_ratio_features(
            raw,
            opt_state=opt_state,
            active_features=feat_active,
        )
    if needs_spot_ratio_moneyness(feat_active):
        raw = enrich_spot_ratio_moneyness_features(
            raw,
            opt_state=opt_state,
            spot_controllers=spot_controllers,
            ts=float(ts),
            active_features=feat_active,
        )
    if needs_advanced_composites(feat_active):
        raw = enrich_advanced_composite_features(
            raw,
            active_features=feat_active,
            spot_controllers=spot_controllers,
            ts=float(ts),
        )
    return pick_features_from_row(raw, all_feats, groups)
