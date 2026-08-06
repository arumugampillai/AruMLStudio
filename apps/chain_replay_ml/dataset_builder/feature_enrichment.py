"""Shared per-row feature enrichment for serial and parallel dataset builds."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.features_atm_band import extract_timeline_features

from .chain_maps import ChainMaps
from .day_context import DayContext
from .extended_features import OptionFeatureState, enrich_dataset_features, enrich_with_chain_maps, reset_option_rolling_state
from .gap_policy_instrumentation import gap_policy_profile_block, record_gap_check, record_gap_reset, row_gap_exceeds
from .feature_grid_policy import resolve_feature_grid_step_sec
from .rolling_controllers import (
    SpotControllers,
    update_token_ltp_controllers,
    update_token_rv_controllers,
)
from .sharp_momentum import enrich_sharp_momentum_features, needs_sharp_momentum
from .iv_zscore_features import enrich_iv_zscore_features, needs_iv_zscore_composites
from .iv_ema_ratio_features import enrich_iv_ema_ratio_features, needs_iv_ema_ratio_features
from .iv_rv_spread_features import enrich_iv_rv_spread_features, needs_iv_rv_spread
from .historic_spot_ema_context import (
    enrich_historic_spot_ema_features,
    needs_historic_spot_ema,
)
from .market_microstructure import (
    enrich_market_microstructure_features,
    needs_market_microstructure,
)
from .futures_context import emit_futures_timeline_features
from .option_tape_features import emit_option_tape_features
from .session_ohlc import emit_session_ohlc_features
from .spot_ratio_moneyness_features import enrich_spot_ratio_moneyness_features, needs_spot_ratio_moneyness
from .advanced_composite_features import enrich_advanced_composite_features, needs_advanced_composites
from .spot_hl_registry import (
    enrich_spot_hl_composite_registry_features,
    enrich_spot_hl_ratio_registry_features,
    needs_spot_hl_composite_registry,
    needs_spot_hl_ratio_registry,
)
from .current_to_atm6_flow import enrich_current_to_atm6_flow_features, needs_current_to_atm6_flow
from .spot_momentum_registry import enrich_spot_momentum_registry_features, needs_spot_momentum_registry
from .build_profiler import profile_block

# Always required for replay backtest scoring (delta band filter), even if not model inputs.
SCORING_INFRA_COLUMNS: tuple[str, ...] = ("ltp", "delta", "abs_delta")


def attach_scoring_infra_columns(row: dict[str, Any], raw: dict[str, Any]) -> None:
    """Ensure ltp/delta exist on a feature row for registry replay scoring."""
    for col in SCORING_INFRA_COLUMNS:
        if col not in row or row.get(col) is None:
            val = raw.get(col)
            if val is not None:
                row[col] = val
    if "delta" not in row:
        row["delta"] = raw.get("delta", 0.0)
    if "ltp" not in row and raw.get("ltp") is not None:
        row["ltp"] = raw["ltp"]


def build_feature_raw_for_row(
    row: dict[str, Any],
    *,
    ctx: DayContext,
    chain_maps: ChainMaps,
    strike_step: int,
    lookback_policy_doc: dict[str, Any] | None,
    opt_state: OptionFeatureState,
    active_features: frozenset[str] | None = None,
    gap_max_sec: float | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    with profile_block("function.build_feature_raw_for_row", rows=1):
        return _build_feature_raw_for_row_impl(
            row,
            ctx=ctx,
            chain_maps=chain_maps,
            strike_step=strike_step,
            lookback_policy_doc=lookback_policy_doc,
            opt_state=opt_state,
            active_features=active_features,
            gap_max_sec=gap_max_sec,
            spot_controllers=spot_controllers,
            spot_rv_cache=spot_rv_cache,
        )


def _build_feature_raw_for_row_impl(
    row: dict[str, Any],
    *,
    ctx: DayContext,
    chain_maps: ChainMaps,
    strike_step: int,
    lookback_policy_doc: dict[str, Any] | None,
    opt_state: OptionFeatureState,
    active_features: frozenset[str] | None = None,
    gap_max_sec: float | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    ts = float(row["timestamp"])
    token_key = str(row.get("token") or "")
    if gap_max_sec is not None and float(gap_max_sec) > 0:
        with gap_policy_profile_block("build_feature_raw_for_row.gap_check"):
            prev_ts = opt_state.last_row_ts
            is_gap = row_gap_exceeds(ts, prev_ts, gap_max_sec)
            record_gap_check(is_gap=is_gap)
            if is_gap:
                with gap_policy_profile_block("reset_option_rolling_state"):
                    reset_option_rolling_state(
                        opt_state,
                        ts=ts,
                        token=token_key or None,
                        previous_ts=prev_ts,
                        gap_limit=float(gap_max_sec),
                        reason="row_gap",
                    )
                record_gap_reset()
        opt_state.last_row_ts = ts
    opt_tl = row["_opt_tl"]
    atm = row["_atm"]
    grid_step = resolve_feature_grid_step_sec(ctx=ctx)
    with profile_block("function.extract_timeline_features", rows=1):
        raw = extract_timeline_features(
            ts=row["timestamp"],
            index_timeline=ctx.index_tl,
            option_timeline=opt_tl,
            option_type=row["option_type"],
            strike_rupees=row["strike"],
            atm_strike_price=atm,
            expiry_ts=ctx.expiry_ts,
            open_ts=ctx.open_ts,
            close_ts=ctx.close_ts,
            feature_grid_step_sec=grid_step,
        )
    with profile_block("function.update_token_ltp_controllers", rows=1):
        update_token_ltp_controllers(opt_state.controllers, raw.get("ltp"), ts=ts)
    with profile_block("function.update_token_rv_controllers", rows=1):
        update_token_rv_controllers(opt_state.controllers, raw.get("ltp"), ts=ts)
    if spot_controllers is not None:
        from .tick_coverage import feature_grid_origin_ts

        with profile_block("function.spot_controllers.update", rows=1):
            spot_controllers.update(
                raw.get("spot"),
                ts=ts,
                grid_step_sec=grid_step,
                index_tl=ctx.index_tl,
                grid_origin_ts=feature_grid_origin_ts(ctx),
            )
    if needs_spot_momentum_registry(active_features):
        with profile_block("function.enrich_spot_momentum_registry", rows=1):
            raw = enrich_spot_momentum_registry_features(
                raw,
                ts=ts,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                active_features=active_features,
            )
    raw["option_type"] = row["option_type"]
    with profile_block("function.enrich_dataset_features", rows=1):
        raw = enrich_dataset_features(
            raw,
            ts=row["timestamp"],
            option_timeline=opt_tl,
            index_timeline=ctx.index_tl,
            option_type=row["option_type"],
            strike_rupees=row["strike"],
            atm_strike=atm,
            strike_step=strike_step,
            expiry_ts=ctx.expiry_ts,
            open_ts=ctx.open_ts,
            close_ts=ctx.close_ts,
            trading_day=ctx.source.trading_day,
            expiry_norm=ctx.expiry_norm,
            opt_state=opt_state,
            lookback_policy_doc=lookback_policy_doc,
        )
    with profile_block("function.enrich_futures_timeline", rows=1):
        raw = emit_futures_timeline_features(
            raw,
            ts=ts,
            futures_tl=getattr(ctx, "futures_tl", None),
        )
    with profile_block("function.enrich_option_tape", rows=1):
        raw = emit_option_tape_features(
            raw,
            ts=ts,
            option_timeline=opt_tl,
        )
    with profile_block("function.enrich_session_ohlc", rows=1):
        spot_session = {
            "open": getattr(ctx, "spot_open", None),
            "high": getattr(ctx, "spot_high", None),
            "low": getattr(ctx, "spot_low", None),
            "prev_close": getattr(ctx, "spot_prev_close", None),
        }
        opt_session = (getattr(ctx, "option_session_ohlc", None) or {}).get(token_key)
        raw = emit_session_ohlc_features(
            raw,
            spot_session=spot_session,
            option_session=opt_session,
        )
    if needs_market_microstructure(active_features):
        with profile_block("function.enrich_market_microstructure", rows=1):
            raw = enrich_market_microstructure_features(
                raw,
                ts=ts,
                option_timeline=opt_tl,
                active_features=active_features,
            )
    with profile_block("function.enrich_with_chain_maps", rows=1):
        raw = enrich_with_chain_maps(
            raw,
            ts=row["timestamp"],
            chain_maps=chain_maps,
            strike_mapping=ctx.strike_mapping,
            index_tl=ctx.index_tl,
            atm_strike=atm,
            expiry_ts=ctx.expiry_ts,
            opt_state=opt_state,
            option_timeline=opt_tl,
            open_ts=ctx.open_ts,
            close_ts=ctx.close_ts,
            active_features=active_features,
            feature_grid_step_sec=grid_step,
            gap_max_sec=gap_max_sec if gap_max_sec is not None else getattr(ctx, "feature_grid_gap_max_sec", None),
            spot_controllers=spot_controllers,
            spot_rv_cache=spot_rv_cache,
        )
    if needs_iv_rv_spread(active_features):
        with profile_block("function.enrich_iv_rv_spread", rows=1):
            raw = enrich_iv_rv_spread_features(
                raw,
                ts=ts,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                active_features=active_features,
            )
    if needs_sharp_momentum(active_features):
        with profile_block("function.enrich_sharp_momentum", rows=1):
            raw = enrich_sharp_momentum_features(
                raw,
                ts=row["timestamp"],
                ctx=ctx,
                opt_state=opt_state,
                option_timeline=opt_tl,
                open_ts=ctx.open_ts,
                close_ts=ctx.close_ts,
                active_features=active_features,
                feature_grid_step_sec=grid_step,
                gap_max_sec=gap_max_sec if gap_max_sec is not None else getattr(ctx, "feature_grid_gap_max_sec", None),
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
            )
    if needs_iv_zscore_composites(active_features):
        with profile_block("function.enrich_iv_zscore", rows=1):
            raw = enrich_iv_zscore_features(
                raw,
                active_features=active_features,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                ts=ts,
            )
    if needs_iv_ema_ratio_features(active_features):
        with profile_block("function.enrich_iv_ema_ratio", rows=1):
            raw = enrich_iv_ema_ratio_features(
                raw,
                opt_state=opt_state,
                active_features=active_features,
            )
    if needs_historic_spot_ema(active_features):
        with profile_block("function.enrich_historic_spot_ema", rows=1):
            raw = enrich_historic_spot_ema_features(
                raw,
                ts=ts,
                ctx=ctx,
                active_features=active_features,
            )
    if needs_spot_ratio_moneyness(active_features):
        with profile_block("function.enrich_spot_ratio_moneyness", rows=1):
            raw = enrich_spot_ratio_moneyness_features(
                raw,
                opt_state=opt_state,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                ts=ts,
                active_features=active_features,
            )
    if needs_advanced_composites(active_features):
        with profile_block("function.enrich_advanced_composites", rows=1):
            raw = enrich_advanced_composite_features(
                raw,
                active_features=active_features,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                ts=ts,
            )
    if needs_spot_hl_ratio_registry(active_features):
        with profile_block("function.enrich_spot_hl_ratio", rows=1):
            raw = enrich_spot_hl_ratio_registry_features(
                raw,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                ts=ts,
                active_features=active_features,
            )
    if needs_spot_hl_composite_registry(active_features):
        with profile_block("function.enrich_spot_hl_composite", rows=1):
            raw = enrich_spot_hl_composite_registry_features(
                raw,
                spot_controllers=spot_controllers,
                spot_rv_cache=spot_rv_cache,
                ts=ts,
                active_features=active_features,
            )
    if needs_current_to_atm6_flow(active_features):
        with profile_block("function.enrich_current_to_atm6_flow", rows=1):
            raw = enrich_current_to_atm6_flow_features(
                raw,
                ts=row["timestamp"],
                strike_mapping=ctx.strike_mapping,
                strike_rupees=row["strike"],
                strike_step=strike_step,
                option_type=row["option_type"],
                active_features=active_features,
            )
    return raw
