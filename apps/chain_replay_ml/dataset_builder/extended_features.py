"""Roll-state and session IV features for dataset builder rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.reanchor import ReanchorThresholds
from chain_replay_ml.surface_math.greeks import calculate_higher_order_greeks
from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC, TickTimeline

from .chain_maps import ChainMaps, chain_features_at
from .lookback_policy import DEFAULT_LOOKBACK_POLICY, greek_at_lookback, normalize_policy_doc
from .rolling_controllers import (
    TokenControllers,
    emit_controller_value,
    emit_dgt_features,
    emit_iv_history_features,
    emit_roll_features,
    guard_controller_derived_rv_features,
    update_token_dgt_controller,
    update_token_iv_controllers,
    update_token_roll_controller,
)


@dataclass
class OptionFeatureState:
    greek_snapshots: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    controllers: TokenControllers = field(default_factory=TokenControllers)
    last_row_ts: float | None = None


_SPOT_EMA_LEVEL_FEATS: frozenset[str] = frozenset({
    "spot_ema9", "spot_ema20", "spot_ema50", "spot_ema100", "spot_ema200", "spot_ema300",
})
_LTP_CONTROLLER_LEVEL_FEATS: frozenset[str] = frozenset({
    "ltp_ema9", "ltp_ema20", "ltp_ema50", "ltp_ema100", "ltp_ema200", "ltp_ema300", "ltp_std20",
})


def _needs_spot_ema_levels(active_features: frozenset[str] | None) -> bool:
    if active_features is None:
        return True
    return any(f in active_features for f in _SPOT_EMA_LEVEL_FEATS)


def _needs_ltp_controller_levels(active_features: frozenset[str] | None) -> bool:
    if active_features is None:
        return True
    return any(f in active_features for f in _LTP_CONTROLLER_LEVEL_FEATS)


def _ema_series_from_prices(
    prices: np.ndarray,
    period: int,
    *,
    last_tick_ts: np.ndarray | None = None,
    gap_max_sec: float | None = None,
) -> np.ndarray:
    from .gap_policy_instrumentation import gap_policy_profile_block

    with gap_policy_profile_block("_ema_series_from_prices"):
        ema = np.zeros_like(prices)
        if len(prices) == 0:
            return ema
        ema[0] = prices[0]
        alpha = 2.0 / (float(period) + 1.0)
        use_gap = (
            gap_max_sec is not None
            and float(gap_max_sec) > 0
            and last_tick_ts is not None
            and len(last_tick_ts) == len(prices)
        )
        if not use_gap:
            for idx in range(1, len(prices)):
                ema[idx] = prices[idx] * alpha + ema[idx - 1] * (1.0 - alpha)
            return ema
        gap_limit = float(gap_max_sec)
        for idx in range(1, len(prices)):
            if float(last_tick_ts[idx] - last_tick_ts[idx - 1]) > gap_limit:
                ema[idx] = prices[idx]
            else:
                ema[idx] = prices[idx] * alpha + ema[idx - 1] * (1.0 - alpha)
        return ema


def reset_option_rolling_state(
    opt_state: OptionFeatureState,
    *,
    ts: float | None = None,
    token: str | None = None,
    previous_ts: float | None = None,
    gap_limit: float | None = None,
    reason: str = "row_gap",
) -> None:
    """Clear incremental rolling history after a sample gap."""
    opt_state.greek_snapshots.clear()
    opt_state.controllers.reset_all(
        ts,
        token=token,
        previous_ts=previous_ts,
        gap_limit=gap_limit,
        reason=reason,
    )


def _ratio_to_ltp(numerator: float | None, ltp: float | None) -> float | None:
    if numerator is not None and ltp is not None and float(ltp) > 0:
        return float(numerator) / float(ltp)
    return None


def _ratio_to_spot(numerator: float | None, spot: float | None) -> float | None:
    if numerator is not None and spot is not None and float(spot) > 0:
        return float(numerator) / float(spot)
    return None


def _abs_change(timeline: TickTimeline | None, ts: float, lookback_sec: float, *, attr: str = "ltp") -> float | None:
    if timeline is None:
        return None
    if attr == "ltp":
        cur = timeline.ltp_rupees_at(ts)
        past = timeline.ltp_rupees_at(ts - lookback_sec)
    elif attr == "oi":
        cur = timeline.oi_at(ts)
        past = timeline.oi_at(ts - lookback_sec)
    elif attr == "volume":
        cur = timeline.volume_at(ts)
        past = timeline.volume_at(ts - lookback_sec)
    else:
        return None
    if cur is None or past is None:
        return None
    return float(cur - past)


def _pct_change_val(cur: float | None, past: float | None) -> float | None:
    if cur is None or past is None:
        return None
    if past <= 0:
        return 0.0 if cur == 0 else None
    return float((cur - past) / past * 100.0)


def _iv_at(
    option_timeline: TickTimeline | None,
    index_timeline: TickTimeline,
    ts: float,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
) -> float | None:
    if option_timeline is None:
        return None
    spot = index_timeline.ltp_rupees_at(ts)
    ltp = option_timeline.ltp_rupees_at(ts)
    if spot is None or ltp is None or ltp <= 0:
        return None
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    return bs.implied_volatility(option_type, ltp, spot, strike_rupees, RISK_FREE_RATE, t_exp)


def _greek_at_ts(
    snapshots: list[tuple[float, dict[str, float]]],
    ts: float,
    lookback_sec: float,
    key: str,
    *,
    lookback_policy_doc: dict[str, Any] | None = None,
) -> float | None:
    return greek_at_lookback(
        snapshots, ts, lookback_sec, key,
        normalize_policy_doc(lookback_policy_doc or DEFAULT_LOOKBACK_POLICY),
    )


def enrich_dataset_features(
    raw: dict[str, Any],
    *,
    ts: float,
    option_timeline: TickTimeline | None,
    index_timeline: TickTimeline,
    option_type: str,
    strike_rupees: float,
    atm_strike: int,
    strike_step: int,
    expiry_ts: float,
    open_ts: float,
    close_ts: float,
    trading_day: str,
    expiry_norm: str,
    opt_state: OptionFeatureState,
    thresholds: ReanchorThresholds | None = None,
    lookback_policy_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add registry feature columns not produced by extract_timeline_features."""
    thresholds = thresholds or ReanchorThresholds()
    spot = raw.get("spot") or index_timeline.ltp_rupees_at(ts)
    ltp = raw.get("ltp")
    if option_timeline is not None and ltp is None:
        ltp = option_timeline.ltp_rupees_at(ts)

    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    actual_iv = _iv_at(option_timeline, index_timeline, ts, option_type, strike_rupees, expiry_ts)

    update_token_roll_controller(
        opt_state.controllers,
        actual_iv=actual_iv,
        spot=spot,
        ltp=ltp,
        ts=ts,
        option_type=option_type,
        strike_rupees=strike_rupees,
        expiry_ts=expiry_ts,
        thresholds=thresholds,
    )
    roll_feats = emit_roll_features(
        opt_state.controllers.roll,
        actual_iv=actual_iv,
        spot=spot,
        ltp=ltp,
        t_exp=t_exp,
        option_type=option_type,
        strike_rupees=strike_rupees,
        ts=ts,
    )
    bs_reiv = roll_feats["bs_reiv_pred"]
    dgt_reiv = roll_feats["dgt_reiv_pred"]

    dgt_feats = emit_dgt_features(
        opt_state.controllers.dgt,
        ts=ts,
        ltp=ltp,
        dgt_reiv=dgt_reiv,
        spot=spot,
    )
    update_token_dgt_controller(
        opt_state.controllers,
        ts=ts,
        ltp=ltp,
        dgt_reiv=dgt_reiv,
    )

    greeks_now = {}
    if actual_iv is not None and actual_iv > 0 and spot is not None and t_exp > 0:
        g_rec = calculate_higher_order_greeks(
            option_type=option_type,
            underlying_spot=spot,
            strike=strike_rupees,
            risk_free_rate=RISK_FREE_RATE,
            time_to_expiry_years=t_exp,
            implied_volatility=actual_iv,
        )
        greeks_now = g_rec.to_dict()
        opt_state.greek_snapshots.append((ts, dict(greeks_now)))
        if len(opt_state.greek_snapshots) > 120:
            opt_state.greek_snapshots = opt_state.greek_snapshots[-120:]

    vega = greeks_now.get("vega")
    vanna = greeks_now.get("vanna")
    volga = greeks_now.get("volga")
    charm = greeks_now.get("charm")
    color = greeks_now.get("color")
    speed = greeks_now.get("speed")
    zomma = greeks_now.get("zomma")
    ultima = greeks_now.get("ultima")
    delta = greeks_now.get("delta", raw.get("delta"))
    gamma = greeks_now.get("gamma", raw.get("gamma"))
    theta = greeks_now.get("theta", raw.get("theta"))

    iv_pct = actual_iv * 100.0 if actual_iv is not None else None

    if actual_iv is not None:
        update_token_iv_controllers(opt_state.controllers, actual_iv, ts=ts)

    spread_paise = option_timeline.spread_paise_at(ts) if option_timeline else None
    bid_ask_spread = spread_paise / 100.0 if spread_paise is not None else None
    # Exchange day ATP = session Option VWAP (rupees). Distances → Interaction.
    option_vwap = option_timeline.atp_rupees_at(ts) if option_timeline else None

    dte = bs.days_to_expiry(trading_day, expiry_norm)
    step = max(strike_step, 1)
    lb_policy = normalize_policy_doc(lookback_policy_doc or DEFAULT_LOOKBACK_POLICY)

    # Fair-value ÷ ltp/spot normalizations → Interaction (both operands Registry).

    out = dict(raw)
    out.update({
        # Price
        "bid_ask_spread": bid_ask_spread,
        "option_vwap": option_vwap,
        "bs_reiv_pred": bs_reiv,
        "dgt_reiv_pred": dgt_reiv,
        **dgt_feats,
        "ltp_change_1m": _abs_change(option_timeline, ts, 60.0),
        "ltp_change_5m": _abs_change(option_timeline, ts, 300.0),
        "ltp_change_15m": _abs_change(option_timeline, ts, 900.0),
        # Greeks
        "vega": vega,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
        "color": color,
        "speed": speed,
        "zomma": zomma,
        "ultima": ultima,
        "theta_per_min": float(theta / 1440.0) if theta is not None else None,
        "vega_per_ivpt": float(vega / iv_pct) if vega is not None and iv_pct and iv_pct > 0 else None,
        "delta_change_5m": (
            float(delta - _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "delta", lookback_policy_doc=lb_policy))
            if delta is not None and _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "delta", lookback_policy_doc=lb_policy) is not None
            else None
        ),
        "gamma_change_5m": (
            float(gamma - _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "gamma", lookback_policy_doc=lb_policy))
            if gamma is not None and _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "gamma", lookback_policy_doc=lb_policy) is not None
            else None
        ),
        "theta_change_5m": (
            float(theta - _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "theta", lookback_policy_doc=lb_policy))
            if theta is not None and _greek_at_ts(opt_state.greek_snapshots, ts, 300.0, "theta", lookback_policy_doc=lb_policy) is not None
            else None
        ),
        # IV (roll + drift — token.roll; change features via iv_history controller in enrich_with_chain_maps)
        **roll_feats,
        # OI / volume longer windows
        "oi_change_5m": _abs_change(option_timeline, ts, 300.0, attr="oi"),
        "oi_change_15m": _abs_change(option_timeline, ts, 900.0, attr="oi"),
        "oi_change_pct_1m": _pct_change_val(
            float(option_timeline.oi_at(ts)) if option_timeline and option_timeline.oi_at(ts) is not None else None,
            float(option_timeline.oi_at(ts - 60)) if option_timeline and option_timeline.oi_at(ts - 60) is not None else None,
        ),
        "oi_change_pct_5m": _pct_change_val(
            option_timeline.oi_at(ts) if option_timeline else None,
            option_timeline.oi_at(ts - 300) if option_timeline else None,
        ),
        "oi_change_pct_15m": _pct_change_val(
            option_timeline.oi_at(ts) if option_timeline else None,
            option_timeline.oi_at(ts - 900) if option_timeline else None,
        ),
        "oi_velocity_1m": _abs_change(option_timeline, ts, 60.0, attr="oi"),
        "volume_change_5m": _abs_change(option_timeline, ts, 300.0, attr="volume"),
        "volume_change_15m": _abs_change(option_timeline, ts, 900.0, attr="volume"),
        # Time
        "minute_of_day": bs.minute_of_day_ist(ts),
        "days_to_expiry": dte,
        "is_expiry_day": 1.0 if dte == 0 else 0.0,
        # Moneyness
        "distance_from_atm_points": float(spot - atm_strike) if spot is not None else None,
        "strike_distance_from_atm": int(round((strike_rupees - atm_strike) / step)) if step > 0 else 0,
        "moneyness": float(spot / strike_rupees) if spot and strike_rupees > 0 else None,
        "is_call": 1.0 if option_type.upper() == "CE" else 0.0,
        # Historical candles
        "spot_body_pct_prev1": raw.get("spot_body_pct_prev1"),
        "spot_body_pct_prev2": raw.get("spot_body_pct_prev2"),
        "spot_body_pct_prev3": raw.get("spot_body_pct_prev3"),
        "spot_range_pct_prev1": raw.get("spot_range_pct_prev1"),
        "opt_body_pct_prev1": raw.get("opt_body_pct_prev1"),
        "opt_range_pct_prev1": raw.get("opt_range_pct_prev1"),
        # Advanced
        "opt_volume_acc_5s_1m": raw.get("opt_volume_acc_5s_1m"),
        "spot_vol_ratio_10s_1m": raw.get("spot_vol_ratio_10s_1m"),
        # Chain spot range
        "spot_dist_high_5m_pct": raw.get("spot_dist_high_5m_pct"),
        "spot_dist_low_5m_pct": raw.get("spot_dist_low_5m_pct"),
        "spot_range_pos_5m": raw.get("spot_range_pos_5m"),
    })

    return out


def enrich_with_chain_maps(
    raw: dict[str, Any],
    *,
    ts: float,
    chain_maps: ChainMaps,
    strike_mapping: dict,
    index_tl: TickTimeline,
    atm_strike: int,
    expiry_ts: float,
    opt_state: OptionFeatureState | None = None,
    option_timeline: TickTimeline | None = None,
    open_ts: float | None = None,
    close_ts: float | None = None,
    active_features: frozenset[str] | None = None,
    feature_grid_step_sec: float = EMA_BAR_INTERVAL_SEC,
    gap_max_sec: float | None = None,
    spot_controllers: Any | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Merge chain-level columns; used after enrich_dataset_features for full chain context."""
    chain_feats = chain_features_at(
        chain_maps, ts,
        expiry_ts=expiry_ts,
        strike_mapping=strike_mapping,
        index_tl=index_tl,
        atm_strike=atm_strike,
    )
    atm_iv = chain_feats.pop("_atm_iv", None)
    out = dict(raw)
    out.update(chain_feats)

    ltp = out.get("ltp")
    option_type = str(out.get("option_type") or "").upper()
    if not option_type and out.get("is_call") is not None:
        option_type = "CE" if float(out["is_call"]) >= 0.5 else "PE"
    ce_atm6_sum = out.get("ce_atm6_ltp_sum")
    pe_atm6_sum = out.get("pe_atm6_ltp_sum")

    side_to_ltp_ratio = None
    if ltp is not None and float(ltp) > 0:
        if option_type == "CE" and ce_atm6_sum is not None:
            side_to_ltp_ratio = float(ce_atm6_sum) / float(ltp)
        elif option_type == "PE" and pe_atm6_sum is not None:
            side_to_ltp_ratio = float(pe_atm6_sum) / float(ltp)

    atm6_total_to_ltp_ratio = None
    if (
        ltp is not None
        and float(ltp) > 0
        and ce_atm6_sum is not None
        and pe_atm6_sum is not None
    ):
        atm6_total_to_ltp_ratio = float(ce_atm6_sum + pe_atm6_sum) / float(ltp)

    out["side_to_ltp_ratio"] = side_to_ltp_ratio
    # atm6_total_to_{ltp,spot}_ratio → Interaction (Wave 1 normalization migration)

    spot = out.get("spot")
    spot_f = float(spot) if spot is not None and float(spot) > 0 else None

    # Wave 2: canonical spot EMA levels (÷ltp → Interaction)
    if _needs_spot_ema_levels(active_features):
        spot_ema_map = {
            "spot_ema9": "ema9",
            "spot_ema20": "ema20",
            "spot_ema50": "ema50",
            "spot_ema100": "ema100",
            "spot_ema200": "ema200",
            "spot_ema300": "ema300",
        }
        if spot_controllers is not None:
            for feat, attr in spot_ema_map.items():
                if active_features is not None and feat not in active_features:
                    continue
                out[feat] = emit_controller_value(getattr(spot_controllers, attr))
        elif spot_rv_cache is not None:
            cached = spot_rv_cache.get(float(ts), {})
            for feat in spot_ema_map:
                if active_features is not None and feat not in active_features:
                    continue
                cached_v = cached.get(feat)
                out[feat] = float(cached_v) if cached_v is not None else None

    # Wave 2: canonical LTP EMA / std20 levels (÷ltp/÷spot → Interaction)
    if opt_state is not None and _needs_ltp_controller_levels(active_features):
        ctrl = opt_state.controllers
        ltp_level_map = {
            "ltp_ema9": ctrl.ema9,
            "ltp_ema20": ctrl.ema20,
            "ltp_ema50": ctrl.ema50,
            "ltp_ema100": ctrl.ema100,
            "ltp_ema200": ctrl.ema200,
            "ltp_ema300": ctrl.ema300,
            "ltp_std20": ctrl.std20,
        }
        for feat, ema_ctrl in ltp_level_map.items():
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = emit_controller_value(ema_ctrl)

    if opt_state is not None:
        ctrl = opt_state.controllers
        rv_feats = {
            "opt_rv_5m": ctrl.rv5m,
            "opt_rv_10m": ctrl.rv10m,
        }
        for feat, rv_ctrl in rv_feats.items():
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = emit_controller_value(rv_ctrl)
        # opt_rv_ratio → Interaction (opt_rv_5m / opt_rv_10m)

    if opt_state is not None:
        iv_ctrl = opt_state.controllers
        iv_feats = {
            "iv_zscore_1m": iv_ctrl.iv_zscore_1m,
            "iv_zscore_5m": iv_ctrl.iv_zscore_5m,
            "iv_zscore_15m": iv_ctrl.iv_zscore_15m,
            "iv_zscore_30m": iv_ctrl.iv_zscore_30m,
            "iv_rank_session": iv_ctrl.iv_session_rank,
        }
        for feat, controller in iv_feats.items():
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = emit_controller_value(controller)

        hist_feats = emit_iv_history_features(iv_ctrl.iv_history)
        for feat, val in hist_feats.items():
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = val

    if spot_rv_cache is not None:
        cached = spot_rv_cache.get(float(ts), {})
        for feat in ("spot_rv_5m", "spot_rv_10m"):
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = cached.get(feat)
        # spot_rv_ratio → Interaction
    elif spot_controllers is not None:
        for feat, ctrl in (
            ("spot_rv_5m", spot_controllers.rv5m),
            ("spot_rv_10m", spot_controllers.rv10m),
        ):
            if active_features is not None and feat not in active_features:
                continue
            out[feat] = emit_controller_value(ctrl)
        # spot_rv_ratio → Interaction

    guard_controller_derived_rv_features(out)

    actual_iv = out.get("iv")
    if actual_iv is not None and atm_iv is not None and atm_iv > 0:
        iv_dec = actual_iv / 100.0 if actual_iv > 3.0 else actual_iv
        out["iv_vs_atm"] = float((iv_dec - atm_iv) / atm_iv * 100.0)
    return out
