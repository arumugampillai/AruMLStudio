from __future__ import annotations

import math
from typing import Any, Callable
from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.ticks import TickTimeline

def find_atm_strike(spot_price: float, strike_step: int) -> int:
    """Find the ATM strike price given spot price and strike step."""
    return int(round(spot_price / strike_step) * strike_step)

def select_atm_band_strikes(atm_strike: int, strike_step: int, band_size: int = 10) -> list[int]:
    """Return a list of 21 strikes from ATM-10 to ATM+10."""
    return [atm_strike + i * strike_step for i in range(-band_size, band_size + 1)]


def compute_delta_at_ts(
    *,
    ts: float,
    index_timeline: TickTimeline,
    option_timeline: TickTimeline,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
) -> float | None:
    """Black–Scholes delta at ``ts`` from spot + option LTP (IV inverted)."""
    spot = index_timeline.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return None
    ltp = option_timeline.ltp_rupees_at(ts)
    if ltp is None or ltp <= 0:
        return None
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    iv = bs.implied_volatility(option_type, ltp, spot, strike_rupees, RISK_FREE_RATE, t_exp)
    if iv is None or iv <= 0:
        return None
    return float(bs.greeks(option_type, spot, strike_rupees, RISK_FREE_RATE, t_exp, iv).get("delta", 0.0))


def delta_matches_selection(
    delta: float,
    option_type: str,
    *,
    delta_type: str,
    delta_min: float,
    delta_max: float,
) -> bool:
    """Return True when option delta falls in the configured band."""
    dt = str(delta_type or "absolute").lower()
    lo = float(delta_min)
    hi = float(delta_max)
    if lo > hi:
        lo, hi = hi, lo
    if dt == "ce":
        if str(option_type).upper() != "CE":
            return False
        return lo <= delta <= hi
    if dt == "pe":
        if str(option_type).upper() != "PE":
            return False
        return (-hi) <= delta <= (-lo)
    return lo <= abs(delta) <= hi


def select_option_entries_for_timestamp(
    *,
    ts: float,
    spot: float,
    strike_step: int,
    index_timeline: TickTimeline,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    expiry_ts: float,
    strike_selection: dict[str, Any],
) -> list[tuple[float, str, str, str, TickTimeline, int]]:
    """Pick (strike, side, token, symbol, timeline, atm) rows for one sample time."""
    mode = str(strike_selection.get("mode") or "atm_band").lower()
    atm = find_atm_strike(spot, strike_step)
    entries: list[tuple[float, str, str, str, TickTimeline, int]] = []

    def _append(strike_r: float, opt_type: str) -> None:
        hit = strike_mapping.get((strike_r, opt_type))
        if not hit:
            return
        tok, symbol, opt_tl = hit
        entries.append((strike_r, opt_type, tok, symbol, opt_tl, atm))

    if mode == "delta_range":
        delta_type = str(strike_selection.get("deltaType") or "absolute").lower()
        dmin = float(strike_selection.get("deltaMin") or 0.15)
        dmax = float(strike_selection.get("deltaMax") or 0.50)
        for (strike_r, opt_type), (_tok, _sym, opt_tl) in strike_mapping.items():
            if delta_type == "ce" and opt_type != "CE":
                continue
            if delta_type == "pe" and opt_type != "PE":
                continue
            delta = compute_delta_at_ts(
                ts=ts,
                index_timeline=index_timeline,
                option_timeline=opt_tl,
                option_type=opt_type,
                strike_rupees=strike_r,
                expiry_ts=expiry_ts,
            )
            if delta is None:
                continue
            if not delta_matches_selection(
                delta,
                opt_type,
                delta_type=delta_type,
                delta_min=dmin,
                delta_max=dmax,
            ):
                continue
            _append(strike_r, opt_type)
        return entries

    if mode == "premium_band":
        pmin = float(strike_selection.get("premiumMin") or 15)
        pmax = float(strike_selection.get("premiumMax") or 30)
        for (strike_r, opt_type), (_tok, _sym, opt_tl) in strike_mapping.items():
            ltp = opt_tl.ltp_rupees_at(ts)
            if ltp is None or ltp < pmin or ltp > pmax:
                continue
            _append(strike_r, opt_type)
        return entries

    if mode == "custom":
        for off in strike_selection.get("customOffsets") or []:
            try:
                strike_r = float(atm + int(off) * strike_step)
            except (TypeError, ValueError):
                continue
            for opt_type in ("CE", "PE"):
                _append(strike_r, opt_type)
        return entries

    # atm_band (default)
    band_raw = strike_selection.get("atmBand", 10)
    if str(band_raw).lower() == "all":
        strikes = sorted({s for s, _ in strike_mapping.keys()})
    else:
        band_size = int(band_raw or 10)
        strikes = select_atm_band_strikes(atm, strike_step, band_size=band_size)
    for strike_r in strikes:
        for opt_type in ("CE", "PE"):
            _append(float(strike_r), opt_type)
    return entries

def compute_pct_change(current_val: float | None, past_val: float | None) -> float | None:
    """Compute percentage change. Return None if values are invalid."""
    if current_val is None or past_val is None or past_val <= 0:
        return None
    return float((current_val - past_val) / past_val * 100.0)


def compute_ltp_to_spot_ratio_at(
    index_timeline: TickTimeline,
    option_timeline: TickTimeline | None,
    ts: float,
) -> float | None:
    """Option LTP divided by spot at ``ts`` (normalized premium scale)."""
    if option_timeline is None:
        return None
    spot = index_timeline.ltp_rupees_at(ts)
    ltp = option_timeline.ltp_rupees_at(ts)
    if ltp is None or ltp <= 0 or spot is None or spot <= 0:
        return None
    return float(ltp / spot)


# (suffix, lag seconds, change interpretation for docs/UI)
# Change lookbacks are Pipeline Owned; *_change_10s retired (10∤3 at default 3s).
_LTP_TO_SPOT_RATIO_HORIZONS: tuple[tuple[str, float, str], ...] = ()


def ltp_to_spot_ratio_lag_change_features(
    index_timeline: TickTimeline,
    option_timeline: TickTimeline | None,
    ts: float,
) -> dict[str, float | None]:
    """No-op: ``ltp_to_spot_ratio`` lag/change columns are Pipeline Owned or retired."""
    del index_timeline, option_timeline, ts
    return {}


def ltp_to_spot_ratio_lag_change_nulls() -> dict[str, None]:
    """Null placeholders when option timeline is unavailable (none — pipeline owned)."""
    return {}


_VOLUME_CHANGE_EPS = 1e-9

# (suffix, lag seconds, interpretation for docs/UI)
_LTP_X_VOLUME_CHANGE_HORIZONS: tuple[tuple[str, float, str], ...] = (
    ("10s", 10.0, "Immediate volume surge scaled by premium — tick-level activity burst."),
    ("30s", 30.0, "Short-term volume momentum weighted by LTP."),
    ("1m", 60.0, "Very short participation trend in premium terms."),
    ("3m", 180.0, "Intraday micro-trend in volume acceleration × premium."),
    ("5m", 300.0, "Strong short-term flow signal scaled by option price."),
    ("15m", 15.0 * 60.0, "Broader participation context weighted by current premium."),
)


def compute_volume_change_pct(
    current_vol: float | int | None,
    past_vol: float | int | None,
) -> float | None:
    """Fractional volume change: (current − past) / (past + ε). Not multiplied by 100."""
    if current_vol is None or past_vol is None:
        return None
    return float((float(current_vol) - float(past_vol)) / (float(past_vol) + _VOLUME_CHANGE_EPS))


def ltp_x_volume_change_pct_features(
    option_timeline: TickTimeline | None,
    ts: float,
    ltp: float | None,
) -> dict[str, float | None]:
    """LTP × fractional volume change over multiple horizons."""
    out: dict[str, float | None] = {}
    for suffix, _, _ in _LTP_X_VOLUME_CHANGE_HORIZONS:
        out[f"ltp_x_volume_change_pct_{suffix}"] = None
    if option_timeline is None or ltp is None or ltp <= 0:
        return out
    volume = option_timeline.volume_at(ts)
    if volume is None:
        return out
    for suffix, lag_sec, _ in _LTP_X_VOLUME_CHANGE_HORIZONS:
        vol_lag = option_timeline.volume_at(ts - lag_sec)
        vol_chg_pct = compute_volume_change_pct(volume, vol_lag)
        if vol_chg_pct is not None:
            out[f"ltp_x_volume_change_pct_{suffix}"] = float(ltp * vol_chg_pct)
    return out

def extract_timeline_features(
    ts: float,
    index_timeline: TickTimeline,
    option_timeline: TickTimeline | None,
    option_type: str,
    strike_rupees: float,
    atm_strike_price: int,
    expiry_ts: float,
    ema9_now: float | None = None,
    ema20_now: float | None = None,
    ema9_1m_ago: float | None = None,
    ema9_gt_ema20: float | None = None,
    ema_spread_vs_spot_pct: float | None = None,
    time_since_cross_min: float | None = None,
    price_dist_from_cross: float | None = None,
    open_ts: float | None = None,
    close_ts: float | None = None,
    feature_grid_step_sec: float | None = None,
) -> dict[str, Any]:
    """Extract all features for a single option strike at a given timestamp."""
    spot = index_timeline.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return {}

    # Distance features
    distance_from_spot_pct = float(100.0 * (strike_rupees - spot) / spot)
    distance_from_atm_pct = float(100.0 * (strike_rupees - atm_strike_price) / atm_strike_price)
    t_expiry = bs.time_to_expiry_years(expiry_ts, ts)
    minutes_to_expiry = max(0.0, (expiry_ts - ts) / 60.0)

    # Session time features
    minutes_since_open = None
    minutes_to_close = None
    is_first_hour = None
    is_last_hour = None
    if open_ts is not None and close_ts is not None:
        minutes_since_open = float(max(0.0, (ts - open_ts) / 60.0))
        minutes_to_close = float(max(0.0, (close_ts - ts) / 60.0))
        is_first_hour = 1.0 if minutes_since_open <= 60.0 else 0.0
        is_last_hour = 1.0 if minutes_to_close <= 60.0 else 0.0

    # Spot return windows (5s, 15s, 30s, 1m)
    spot_5s = index_timeline.ltp_rupees_at(ts - 5)
    spot_15s = index_timeline.ltp_rupees_at(ts - 15)
    spot_30s = index_timeline.ltp_rupees_at(ts - 30)
    spot_1m = index_timeline.ltp_rupees_at(ts - 60)

    # Spot momentum registry features — legacy export path when EMA inputs are supplied.
    # Dataset builder emits these from SpotControllers (spot_momentum_registry).
    # Wave 6: spot_vs_ema20_pct / ema_spread_pct / ema_spread_vs_spot_pct → Interaction.
    ema9_slope = None
    cross_age_decay = None
    momentum_defaults = ema9_now is None and ema20_now is None and ema9_1m_ago is None
    if momentum_defaults:
        ema9_gt_ema20_out: float | None = None
        time_since_cross_min_out: float | None = None
        price_dist_from_cross_out: float | None = None
    else:
        ema9_gt_ema20_out = ema9_gt_ema20 if ema9_gt_ema20 is not None else 0.0
        time_since_cross_min_out = time_since_cross_min if time_since_cross_min is not None else 60.0
        price_dist_from_cross_out = price_dist_from_cross if price_dist_from_cross is not None else 0.0
        # Keep signature compatibility; packaging moved to Interaction.
        _ = ema_spread_vs_spot_pct

        if ema9_now is not None and ema9_1m_ago is not None and ema9_1m_ago > 0:
            ema9_slope = float(100.0 * (ema9_now - ema9_1m_ago) / ema9_1m_ago)

        cross_age_decay = (
            float(math.exp(-time_since_cross_min_out / 30.0))
            if time_since_cross_min_out is not None
            else None
        )

    features = {
        "strike": strike_rupees,
        "option_type": option_type,
        "spot": spot,
        "distance_from_spot_pct": distance_from_spot_pct,
        "distance_from_atm_pct": distance_from_atm_pct,
        "strike_to_spot_ratio": float(strike_rupees / spot) if spot > 0 else None,
        "minutes_to_expiry": minutes_to_expiry,
        "minutes_since_open": minutes_since_open,
        "minutes_to_close": minutes_to_close,
        "is_first_hour": is_first_hour,
        "is_last_hour": is_last_hour,
        "spot_change_5s": compute_pct_change(spot, spot_5s),
        "spot_change_15s": compute_pct_change(spot, spot_15s),
        "spot_change_30s": compute_pct_change(spot, spot_30s),
        "spot_change_1m": compute_pct_change(spot, spot_1m),
        "ema9_slope": ema9_slope,
        "ema9_gt_ema20": ema9_gt_ema20_out,
        "time_since_cross_min": time_since_cross_min_out,
        "cross_age_decay": cross_age_decay,
        "price_dist_from_cross_pct": price_dist_from_cross_out,
    }

    # Default option features when the timeline is missing or inactive
    ltp = None
    oi = None
    volume = None
    iv = None
    delta = 0.0
    gamma = 0.0
    theta = 0.0

    # If option contract is active/exists
    if option_timeline is not None:
        ltp = option_timeline.ltp_rupees_at(ts)
        oi = option_timeline.oi_at(ts)
        volume = option_timeline.volume_at(ts)

        if ltp is not None and ltp > 0:
            iv = bs.implied_volatility(option_type, ltp, spot, strike_rupees, RISK_FREE_RATE, t_expiry)
            if iv is not None and iv > 0:
                greeks_dict = bs.greeks(option_type, spot, strike_rupees, RISK_FREE_RATE, t_expiry, iv)
                delta = greeks_dict.get("delta", 0.0)
                gamma = greeks_dict.get("gamma", 0.0)
                theta = greeks_dict.get("theta", 0.0)

        # Option LTP returns (5s, 15s, 30s, 1m)
        ltp_5s = option_timeline.ltp_rupees_at(ts - 5)
        ltp_15s = option_timeline.ltp_rupees_at(ts - 15)
        ltp_30s = option_timeline.ltp_rupees_at(ts - 30)
        ltp_1m = option_timeline.ltp_rupees_at(ts - 60)

        features.update({
            "ltp": ltp,
            "ltp_to_spot_ratio": float(ltp / spot) if ltp is not None and ltp > 0 and spot > 0 else None,
            "ltp_return_5s": compute_pct_change(ltp, ltp_5s),
            "ltp_return_15s": compute_pct_change(ltp, ltp_15s),
            "ltp_return_30s": compute_pct_change(ltp, ltp_30s),
            "ltp_return_1m": compute_pct_change(ltp, ltp_1m),
        })
        features.update(ltp_to_spot_ratio_lag_change_features(index_timeline, option_timeline, ts))

        # OI changes (5s, 15s, 30s, 1m)
        oi_5s = option_timeline.oi_at(ts - 5)
        oi_15s = option_timeline.oi_at(ts - 15)
        oi_30s = option_timeline.oi_at(ts - 30)
        oi_1m = option_timeline.oi_at(ts - 60)

        features.update({
            "oi": oi,
            "oi_change_5s": compute_pct_change(oi, oi_5s),
            "oi_change_15s": compute_pct_change(oi, oi_15s),
            "oi_change_30s": compute_pct_change(oi, oi_30s),
            "oi_change_1m": compute_pct_change(oi, oi_1m),
        })

        # Volume changes (5s, 15s, 30s, 1m)
        vol_5s = option_timeline.volume_at(ts - 5)
        vol_15s = option_timeline.volume_at(ts - 15)
        vol_30s = option_timeline.volume_at(ts - 30)
        vol_1m = option_timeline.volume_at(ts - 60)

        flow_5s = max(0.0, volume - vol_5s) if vol_5s is not None else 0.0
        flow_15s = max(0.0, volume - vol_15s) if vol_15s is not None else 0.0
        flow_30s = max(0.0, volume - vol_30s) if vol_30s is not None else 0.0
        flow_1m = max(0.0, volume - vol_1m) if vol_1m is not None else 0.0

        features.update({
            "volume": volume,
            "volume_change_5s": compute_pct_change(volume, vol_5s),
            "volume_change_15s": compute_pct_change(volume, vol_15s),
            "volume_change_30s": compute_pct_change(volume, vol_30s),
            "volume_change_1m": compute_pct_change(volume, vol_1m),
            "opt_volume_flow_5s": flow_5s,
            "opt_volume_flow_15s": flow_15s,
            "opt_volume_flow_30s": flow_30s,
            "opt_volume_flow_1m": flow_1m,
            "opt_volume_acc_5s_1m": float(flow_5s / (flow_1m + 1e-9)),
        })
        features.update(ltp_x_volume_change_pct_features(option_timeline, ts, ltp))
    else:
        # Fallback values when timeline is missing
        features.update({
            "ltp": None,
            "ltp_to_spot_ratio": None,
            "ltp_return_5s": None,
            "ltp_return_15s": None,
            "ltp_return_30s": None,
            "ltp_return_1m": None,
            **ltp_to_spot_ratio_lag_change_nulls(),
            "oi": None,
            "oi_change_5s": None,
            "oi_change_15s": None,
            "oi_change_30s": None,
            "oi_change_1m": None,
            "volume": None,
            "volume_change_5s": None,
            "volume_change_15s": None,
            "volume_change_30s": None,
            "volume_change_1m": None,
            "opt_volume_flow_5s": None,
            "opt_volume_flow_15s": None,
            "opt_volume_flow_30s": None,
            "opt_volume_flow_1m": None,
            "opt_volume_acc_5s_1m": None,
            **{f"ltp_x_volume_change_pct_{suffix}": None for suffix, _, _ in _LTP_X_VOLUME_CHANGE_HORIZONS},
        })

    features.update({
        "iv": iv,
        "delta": delta,
        "abs_delta": abs(delta) if delta is not None else None,
        "gamma": gamma,
        "theta": theta,
    })

    # Spot OHLC features (calendar windows; RV subsample uses feature grid)
    spot_ohlc_feats = extract_ohlc_features_for_timeline(
        index_timeline, ts, "spot_", feature_grid_step_sec=feature_grid_step_sec,
    )
    features.update(spot_ohlc_feats)

    # Option OHLC features
    opt_ohlc_feats = extract_ohlc_features_for_timeline(
        option_timeline, ts, "opt_", feature_grid_step_sec=feature_grid_step_sec,
    )
    features.update(opt_ohlc_feats)

    return features


def get_ohlc(timeline: TickTimeline | None, start_ts: float, end_ts: float) -> tuple[float, float, float, float] | None:
    if timeline is None or not timeline.timestamps:
        return None
    
    import bisect
    start_idx = bisect.bisect_left(timeline.timestamps, start_ts)
    end_idx = bisect.bisect_right(timeline.timestamps, end_ts)
    
    if start_idx >= len(timeline.timestamps) or start_idx >= end_idx:
        val = timeline.ltp_rupees_at(end_ts)
        if val is None or val <= 0:
            return None
        return val, val, val, val
        
    open_val = timeline.ltps_paise[start_idx] / 100.0
    close_val = timeline.ltps_paise[end_idx - 1] / 100.0
    high_val = max(timeline.ltps_paise[start_idx:end_idx]) / 100.0
    low_val = min(timeline.ltps_paise[start_idx:end_idx]) / 100.0
    
    return open_val, high_val, low_val, close_val


def get_realized_volatility(
    timeline: TickTimeline | None,
    ts: float,
    lookback_sec: float,
    *,
    grid_step_sec: float | None = None,
) -> float | None:
    if timeline is None or not timeline.timestamps:
        return None

    from chain_replay_ml.dataset_builder.feature_grid_policy import rv_subsample_step_sec

    step = rv_subsample_step_sec(grid_step_sec if grid_step_sec is not None else 10.0)
    prices = []
    num_intervals = max(1, int(lookback_sec / step))
    for i in range(num_intervals + 1):
        t = ts - lookback_sec + i * step
        p = timeline.ltp_rupees_at(t)
        if p is not None:
            prices.append(p)
            
    if len(prices) < 5:
        return None
        
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1] * 100.0
            returns.append(ret)
            
    if not returns:
        return None
        
    import numpy as np
    return float(np.std(returns))


def extract_ohlc_features_for_timeline(
    timeline: TickTimeline | None,
    ts: float,
    prefix: str,
    *,
    feature_grid_step_sec: float | None = None,
) -> dict[str, Any]:
    out = {}
    
    def add_nulls(keys):
        for k in keys:
            out[prefix + k] = None

    keys_all = [
        "body_pct_10s", "range_pct_10s", "upper_wick_pct_10s", "lower_wick_pct_10s",
        "body_pct_30s", "range_pct_30s", "upper_wick_pct_30s", "lower_wick_pct_30s",
        "body_pct_1m", "range_pct_1m", "upper_wick_pct_1m", "lower_wick_pct_1m",
        "close_vs_high_1m_pct", "close_vs_low_1m_pct",
        "body_pct_prev1", "body_pct_prev2", "body_pct_prev3", "range_pct_prev1",
        "bullish_candle", "bearish_candle", "inside_bar", "outside_bar", "higher_high", "lower_low",
        "high_vs_prev_high_pct", "low_vs_prev_low_pct",
        "vol_ratio_10s_1m", "vol_ratio_1m_5m",
        "dist_high_5m_pct", "dist_low_5m_pct",
        "range_pos_5m"
    ]

    if timeline is None or not timeline.timestamps:
        add_nulls(keys_all)
        return out

    c10s = get_ohlc(timeline, ts - 10.0, ts)
    c30s = get_ohlc(timeline, ts - 30.0, ts)
    c1m = get_ohlc(timeline, ts - 60.0, ts)
    
    prev1 = get_ohlc(timeline, ts - 20.0, ts - 10.0)
    prev2 = get_ohlc(timeline, ts - 30.0, ts - 20.0)
    prev3 = get_ohlc(timeline, ts - 40.0, ts - 30.0)

    if c10s:
        o10, h10, l10, c10 = c10s
        out[prefix + "body_pct_10s"] = float((c10 - o10) / o10 * 100.0) if o10 > 0 else 0.0
        out[prefix + "range_pct_10s"] = float((h10 - l10) / o10 * 100.0) if o10 > 0 else 0.0
        out[prefix + "upper_wick_pct_10s"] = float((h10 - max(o10, c10)) / o10 * 100.0) if o10 > 0 else 0.0
        out[prefix + "lower_wick_pct_10s"] = float((min(o10, c10) - l10) / o10 * 100.0) if o10 > 0 else 0.0
        out[prefix + "bullish_candle"] = 1.0 if c10 > o10 else 0.0
        out[prefix + "bearish_candle"] = 1.0 if c10 < o10 else 0.0
    else:
        for k in ["body_pct_10s", "range_pct_10s", "upper_wick_pct_10s", "lower_wick_pct_10s", "bullish_candle", "bearish_candle"]:
            out[prefix + k] = None

    if c30s:
        o30, h30, l30, c30 = c30s
        out[prefix + "body_pct_30s"] = float((c30 - o30) / o30 * 100.0) if o30 > 0 else 0.0
        out[prefix + "range_pct_30s"] = float((h30 - l30) / o30 * 100.0) if o30 > 0 else 0.0
        out[prefix + "upper_wick_pct_30s"] = float((h30 - max(o30, c30)) / o30 * 100.0) if o30 > 0 else 0.0
        out[prefix + "lower_wick_pct_30s"] = float((min(o30, c30) - l30) / o30 * 100.0) if o30 > 0 else 0.0
    else:
        for k in ["body_pct_30s", "range_pct_30s", "upper_wick_pct_30s", "lower_wick_pct_30s"]:
            out[prefix + k] = None

    if c1m:
        o1m, h1m, l1m, c1m_val = c1m
        out[prefix + "body_pct_1m"] = float((c1m_val - o1m) / o1m * 100.0) if o1m > 0 else 0.0
        out[prefix + "range_pct_1m"] = float((h1m - l1m) / o1m * 100.0) if o1m > 0 else 0.0
        out[prefix + "upper_wick_pct_1m"] = float((h1m - max(o1m, c1m_val)) / o1m * 100.0) if o1m > 0 else 0.0
        out[prefix + "lower_wick_pct_1m"] = float((min(o1m, c1m_val) - l1m) / o1m * 100.0) if o1m > 0 else 0.0
        out[prefix + "close_vs_high_1m_pct"] = float((c1m_val - h1m) / h1m * 100.0) if h1m > 0 else 0.0
        out[prefix + "close_vs_low_1m_pct"] = float((c1m_val - l1m) / l1m * 100.0) if l1m > 0 else 0.0
    else:
        for k in ["body_pct_1m", "range_pct_1m", "upper_wick_pct_1m", "lower_wick_pct_1m", "close_vs_high_1m_pct", "close_vs_low_1m_pct"]:
            out[prefix + k] = None

    if prev1:
        o_p1, h_p1, l_p1, c_p1 = prev1
        out[prefix + "body_pct_prev1"] = float((c_p1 - o_p1) / o_p1 * 100.0) if o_p1 > 0 else 0.0
        out[prefix + "range_pct_prev1"] = float((h_p1 - l_p1) / o_p1 * 100.0) if o_p1 > 0 else 0.0
    else:
        out[prefix + "body_pct_prev1"] = None
        out[prefix + "range_pct_prev1"] = None

    if prev2:
        o_p2, h_p2, l_p2, c_p2 = prev2
        out[prefix + "body_pct_prev2"] = float((c_p2 - o_p2) / o_p2 * 100.0) if o_p2 > 0 else 0.0
    else:
        out[prefix + "body_pct_prev2"] = None

    if prev3:
        o_p3, h_p3, l_p3, c_p3 = prev3
        out[prefix + "body_pct_prev3"] = float((c_p3 - o_p3) / o_p3 * 100.0) if o_p3 > 0 else 0.0
    else:
        out[prefix + "body_pct_prev3"] = None

    if c10s and prev1:
        o10, h10, l10, c10 = c10s
        o_p1, h_p1, l_p1, c_p1 = prev1
        out[prefix + "inside_bar"] = 1.0 if (h10 < h_p1 and l10 > l_p1) else 0.0
        out[prefix + "outside_bar"] = 1.0 if (h10 > h_p1 and l10 < l_p1) else 0.0
        out[prefix + "higher_high"] = 1.0 if h10 > h_p1 else 0.0
        out[prefix + "lower_low"] = 1.0 if l10 < l_p1 else 0.0
        out[prefix + "high_vs_prev_high_pct"] = float(100.0 * (h10 - h_p1) / h_p1) if h_p1 > 0 else 0.0
        out[prefix + "low_vs_prev_low_pct"] = float(100.0 * (l10 - l_p1) / l_p1) if l_p1 > 0 else 0.0
    else:
        for k in ["inside_bar", "outside_bar", "higher_high", "lower_low", "high_vs_prev_high_pct", "low_vs_prev_low_pct"]:
            out[prefix + k] = None

    ranges_10s = []
    for i in range(6):
        c_i = get_ohlc(timeline, ts - (i + 1) * 10.0, ts - i * 10.0)
        if c_i:
            ranges_10s.append(c_i[1] - c_i[2])
    
    if ranges_10s and c10s:
        range_10s = c10s[1] - c10s[2]
        avg_range_1m = sum(ranges_10s) / len(ranges_10s)
        out[prefix + "vol_ratio_10s_1m"] = float(range_10s / avg_range_1m) if avg_range_1m > 0 else 1.0
    else:
        out[prefix + "vol_ratio_10s_1m"] = None

    ranges_1m = []
    for i in range(5):
        c_i = get_ohlc(timeline, ts - (i + 1) * 60.0, ts - i * 60.0)
        if c_i:
            ranges_1m.append(c_i[1] - c_i[2])
            
    if ranges_1m and c1m:
        range_1m = c1m[1] - c1m[2]
        avg_range_5m = sum(ranges_1m) / len(ranges_1m)
        out[prefix + "vol_ratio_1m_5m"] = float(range_1m / avg_range_5m) if avg_range_5m > 0 else 1.0
    else:
        out[prefix + "vol_ratio_1m_5m"] = None

    c5m = get_ohlc(timeline, ts - 300.0, ts)
    if c5m:
        _, h5m, l5m, c_curr = c5m
        out[prefix + "dist_high_5m_pct"] = float((c_curr - h5m) / h5m * 100.0) if h5m > 0 else 0.0
        out[prefix + "dist_low_5m_pct"] = float((c_curr - l5m) / l5m * 100.0) if l5m > 0 else 0.0
        out[prefix + "range_pos_5m"] = float((c_curr - l5m) / (h5m - l5m + 1e-9))
    else:
        out[prefix + "dist_high_5m_pct"] = None
        out[prefix + "dist_low_5m_pct"] = None
        out[prefix + "range_pos_5m"] = None

    return out


def filter_dataset_for_experiment_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Experiment 1: ATM + OTM strikes only, premium >= ₹10 to ATM premium.
    - CE: Keep distance_from_atm_pct >= 0
    - PE: Keep distance_from_atm_pct <= 0
    - Premium bounds: ltp >= 10.0 and ltp <= ATM premium
    """
    import pandas as pd
    import numpy as np

    if df.empty:
        return df

    # 1. Keep CE (distance >= 0) and PE (distance <= 0)
    ce_mask = (df["option_type"] == "CE") & (df["distance_from_atm_pct"] >= -1e-5)
    pe_mask = (df["option_type"] == "PE") & (df["distance_from_atm_pct"] <= 1e-5)
    df_filtered = df[ce_mask | pe_mask].copy()

    # 2. Keep premiums >= 5.0
    df_filtered = df_filtered[df_filtered["ltp"] >= 5.0]

    # 3. Find ATM premium for each timestamp/date/option_type combination
    df_atm = df[df["distance_from_atm_pct"].abs() < 1e-5].copy()
    if df_atm.empty:
        # Fallback to absolute closest to ATM strike
        idx = df.groupby(["date", "timestamp", "option_type"])["distance_from_atm_pct"].transform(lambda x: x.abs().idxmin())
        df_atm = df.loc[idx].copy()

    # Create key map for ATM LTP lookup
    atm_lookup = df_atm.drop_duplicates(subset=["date", "timestamp", "option_type"]).set_index(["date", "timestamp", "option_type"])["ltp"].to_dict()

    # Map ATM LTP and filter out options with LTP > ATM LTP
    keys = list(zip(df_filtered["date"], df_filtered["timestamp"], df_filtered["option_type"]))
    df_filtered["atm_ltp"] = [atm_lookup.get(k, float("inf")) for k in keys]
    df_filtered = df_filtered[df_filtered["ltp"] <= df_filtered["atm_ltp"] + 1e-5]

    df_filtered = df_filtered.drop(columns=["atm_ltp"])
    return df_filtered

