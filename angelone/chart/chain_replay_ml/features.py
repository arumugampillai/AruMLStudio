"""Build Phase 1 canonical 1-minute feature rows for one option."""

from __future__ import annotations

from typing import Any

from . import bs, reanchor
from .constants import (
    LABEL_FORWARD_MIN,
    LOOKBACK_MINUTES,
    MIN_LTP_RUPEES,
    RISK_FREE_RATE,
    STRIKE_STEP_BY_INDEX,
    WARMUP_MINUTES,
)
from .reanchor import ReanchorThresholds, RollState, apply_roll, evaluate_triggers
from .ticks import TickTimeline


def atm_strike(spot_rupees: float, step: int) -> int:
    if spot_rupees <= 0 or step <= 0:
        return 0
    return int(round(spot_rupees / step) * step)


def _change_at(
    timeline: TickTimeline,
    ts: float,
    lookback_min: int,
    *,
    iv_fn=None,
) -> float | None:
    if iv_fn is not None:
        cur = iv_fn(ts)
        past = iv_fn(ts - lookback_min * 60)
        if cur is None or past is None:
            return None
        return cur - past
    cur = timeline.ltp_rupees_at(ts)
    past = timeline.ltp_rupees_at(ts - lookback_min * 60)
    if cur is None or past is None:
        return None
    return cur - past


def _volume_change_at(timeline: TickTimeline, ts: float, lookback_min: int) -> int | None:
    cur = timeline.volume_at(ts)
    past = timeline.volume_at(ts - lookback_min * 60)
    if cur is None or past is None:
        return None
    return cur - past


def _oi_change_at(timeline: TickTimeline, ts: float, lookback_min: int) -> int | None:
    cur = timeline.oi_at(ts)
    past = timeline.oi_at(ts - lookback_min * 60)
    if cur is None or past is None:
        return None
    return cur - past


def _oi_change_pct_at(timeline: TickTimeline, ts: float, lookback_min: int) -> float | None:
    cur = timeline.oi_at(ts)
    past = timeline.oi_at(ts - lookback_min * 60)
    if cur is None or past is None:
        return None
    if past <= 0:
        return 0.0 if cur == 0 else None
    return (cur - past) / past * 100.0


def build_option_rows(
    *,
    date: str,
    underlying: str,
    expiry: str,
    token: str,
    symbol: str,
    option_type: str,
    strike_rupees: float,
    index_timeline: TickTimeline,
    option_timeline: TickTimeline,
    minute_times: list[float],
    open_ts: float,
    close_ts: float,
    expiry_ts: float,
    thresholds: ReanchorThresholds | None = None,
    # Maps for straddle and OI features:
    straddle_map: dict[float, float] | None = None,
    zscore_map: dict[float, float] | None = None,
    max_call_oi_dist_map: dict[float, float] | None = None,
    max_put_oi_dist_map: dict[float, float] | None = None,
    max_call_oi_pct_map: dict[float, float] | None = None,
    max_put_oi_pct_map: dict[float, float] | None = None,
    chain_pcr_map: dict[float, float] | None = None,
    atm_pcr_map: dict[float, float] | None = None,
    oi_wall_bias_map: dict[float, float] | None = None,
    dist_call_build_map: dict[float, float] | None = None,
    dist_put_build_map: dict[float, float] | None = None,
    pinning_pressure_map: dict[float, float] | None = None,
    atm_straddle_open: float | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or ReanchorThresholds()
    step = STRIKE_STEP_BY_INDEX.get(underlying.upper(), 50)
    dte = bs.days_to_expiry(date, expiry)
    is_expiry_day = 1 if dte == 0 else 0

    roll = RollState()
    session_initialized = False
    rows_since_roll = -1
    rows: list[dict[str, Any]] = []

    def spot_at(ts: float) -> float | None:
        return index_timeline.ltp_rupees_at(ts)

    def ltp_at(ts: float) -> float | None:
        return option_timeline.ltp_rupees_at(ts)

    def iv_at(ts: float) -> float | None:
        s = spot_at(ts)
        l = ltp_at(ts)
        if s is None or l is None or l <= 0:
            return None
        t = bs.time_to_expiry_years(expiry_ts, ts)
        return bs.implied_volatility(option_type, l, s, strike_rupees, RISK_FREE_RATE, t)

    for i, row_ts in enumerate(minute_times):
        actual_spot = spot_at(row_ts)
        actual_ltp = ltp_at(row_ts)
        if actual_spot is None or actual_ltp is None or actual_ltp < MIN_LTP_RUPEES:
            continue

        t_row = bs.time_to_expiry_years(expiry_ts, row_ts)
        if t_row <= 0:
            continue

        actual_iv = bs.implied_volatility(
            option_type, actual_ltp, actual_spot, strike_rupees, RISK_FREE_RATE, t_row,
        )
        if actual_iv is None:
            continue

        if not session_initialized:
            roll.roll_iv = actual_iv
            roll.roll_anchor_ts = row_ts
            roll.roll_spot = actual_spot
            roll.roll_ltp = actual_ltp
            roll.roll_greeks = bs.greeks(
                option_type, actual_spot, strike_rupees, RISK_FREE_RATE, t_row, actual_iv,
            )
            session_initialized = True

        reanchor_reason = "no"
        is_reanchor_row = 0
        if row_ts > roll.roll_anchor_ts + 0.001:
            should_roll, reasons = evaluate_triggers(
                actual_iv=actual_iv,
                actual_spot=actual_spot,
                roll=roll,
                row_ts=row_ts,
                thresholds=thresholds,
            )
            if should_roll:
                apply_roll(
                    roll,
                    actual_iv=actual_iv,
                    actual_spot=actual_spot,
                    actual_ltp=actual_ltp,
                    row_ts=row_ts,
                    option_type=option_type,
                    strike_rupees=strike_rupees,
                    expiry_ts=expiry_ts,
                )
                is_reanchor_row = 1
                reanchor_reason = "+".join(reasons)

        roll_iv = roll.roll_iv
        roll_age_min = (row_ts - roll.roll_anchor_ts) / 60.0
        roll_fwd_min = max(0.0, roll_age_min)

        bs_reiv = None
        if roll_iv and roll_iv > 0:
            bs_reiv = max(
                0.0,
                bs.bs_price(option_type, actual_spot, strike_rupees, RISK_FREE_RATE, t_row, roll_iv),
            )

        dgt_reiv = None
        if roll.roll_greeks and roll.roll_ltp and roll.roll_spot is not None:
            dgt_reiv = bs.greek_predicted_ltp(
                roll.roll_ltp,
                roll.roll_greeks,
                actual_spot - roll.roll_spot,
                roll_fwd_min,
                0.0,
            )

        ltp_fwd = ltp_at(row_ts + LABEL_FORWARD_MIN * 60)
        residual_5m = None
        residual_pct_5m = None
        if ltp_fwd is not None and bs_reiv is not None:
            residual_5m = ltp_fwd - bs_reiv
            if bs_reiv > 0:
                residual_pct_5m = residual_5m / bs_reiv * 100.0

        atm = atm_strike(actual_spot, step)
        dist_pts = actual_spot - atm
        dist_pct = (dist_pts / actual_spot * 100.0) if actual_spot > 0 else None

        minutes_from_open = (row_ts - open_ts) / 60.0
        warmup_row = 1 if minutes_from_open < WARMUP_MINUTES else 0

        if is_reanchor_row:
            rows_since_roll = 0
        else:
            rows_since_roll += 1

        spread = option_timeline.spread_paise_at(row_ts)
        spread_rupees = spread / 100.0 if spread is not None else None

        # Resolve feature maps
        s_map = straddle_map or {}
        z_map = zscore_map or {}
        c_dist_map = max_call_oi_dist_map or {}
        p_dist_map = max_put_oi_dist_map or {}
        c_pct_map = max_call_oi_pct_map or {}
        p_pct_map = max_put_oi_pct_map or {}
        pcr_map = chain_pcr_map or {}
        a_pcr_map = atm_pcr_map or {}
        bias_map = oi_wall_bias_map or {}
        c_build_map = dist_call_build_map or {}
        p_build_map = dist_put_build_map or {}
        pin_map = pinning_pressure_map or {}

        curr_straddle = s_map.get(row_ts)
        straddle_change_1m = None
        straddle_change_5m = None
        straddle_change_15m = None
        straddle_change_pct_1m = None
        straddle_change_pct_5m = None
        straddle_change_pct_15m = None
        straddle_slope_5m = None
        straddle_slope_15m = None
        straddle_pct_change_from_open = None
        straddle_change_accel = None

        if curr_straddle is not None:
            past_1m = s_map.get(row_ts - 60)
            past_5m = s_map.get(row_ts - 300)
            past_15m = s_map.get(row_ts - 900)

            if past_1m is not None:
                straddle_change_1m = curr_straddle - past_1m
                straddle_change_pct_1m = (straddle_change_1m / past_1m * 100.0) if past_1m > 0 else 0.0
            if past_5m is not None:
                straddle_change_5m = curr_straddle - past_5m
                straddle_change_pct_5m = (straddle_change_5m / past_5m * 100.0) if past_5m > 0 else 0.0
                straddle_slope_5m = straddle_change_5m / 5.0
            if past_15m is not None:
                straddle_change_15m = curr_straddle - past_15m
                straddle_change_pct_15m = (straddle_change_15m / past_15m * 100.0) if past_15m > 0 else 0.0
                straddle_slope_15m = straddle_change_15m / 15.0

            if straddle_change_1m is not None and straddle_change_5m is not None:
                straddle_change_accel = straddle_change_1m - (straddle_change_5m / 5.0)

            if atm_straddle_open is not None and atm_straddle_open > 0:
                straddle_pct_change_from_open = 100.0 * (curr_straddle - atm_straddle_open) / atm_straddle_open

        curr_zscore = z_map.get(row_ts)
        zscore_change_5m = None
        if curr_zscore is not None:
            past_z_5m = z_map.get(row_ts - 300)
            if past_z_5m is not None:
                zscore_change_5m = curr_zscore - past_z_5m

        chain_pcr = pcr_map.get(row_ts)
        chain_pcr_change_5m = None
        if chain_pcr is not None:
            past_pcr_5m = pcr_map.get(row_ts - 300)
            if past_pcr_5m is not None:
                chain_pcr_change_5m = chain_pcr - past_pcr_5m

        atm_pcr = a_pcr_map.get(row_ts)
        atm_pcr_change_5m = None
        if atm_pcr is not None:
            past_atm_pcr_5m = a_pcr_map.get(row_ts - 300)
            if past_atm_pcr_5m is not None:
                atm_pcr_change_5m = atm_pcr - past_atm_pcr_5m

        def _r4(val: float | None) -> float | None:
            return round(val, 4) if val is not None else None

        row: dict[str, Any] = {
            "ts": row_ts,
            "date": date,
            "underlying": underlying,
            "expiry": expiry,
            "token": token,
            "symbol": symbol,
            "warmup_row": warmup_row,
            "time": bs.format_time_hhmm(row_ts),
            "minute_of_day": bs.minute_of_day_ist(row_ts),
            "minutes_to_close": (close_ts - row_ts) / 60.0,
            "days_to_expiry": dte,
            "is_expiry_day": is_expiry_day,
            "spot": round(actual_spot, 2),
            "ltp": round(actual_ltp, 2),
            "current_iv": round(actual_iv * 100, 4),
            "roll_iv": round(roll_iv * 100, 4) if roll_iv else None,
            "roll_age_min": round(roll_age_min, 2),
            "rows_since_roll": rows_since_roll,
            "roll_reason": reanchor_reason,
            "roll_reason_code": reanchor.encode_roll_reason(reanchor_reason),
            "roll_count": roll.roll_count,
            "strike": strike_rupees,
            "distance_from_atm_points": round(actual_spot - atm, 2),
            "distance_from_atm_pct": round(dist_pct, 4) if dist_pct is not None else None,
            "strike_distance_from_atm": int(round((strike_rupees - atm) / step)) if step > 0 else 0,
            "is_call": 1 if option_type == "CE" else -1,
            "moneyness": round(actual_spot / strike_rupees, 6) if strike_rupees > 0 else None,
            "delta": round(roll.roll_greeks.get("delta", 0.0), 6),
            "gamma": round(roll.roll_greeks.get("gamma", 0.0), 6),
            "theta": round(roll.roll_greeks.get("theta", 0.0), 4),
            "vega": round(roll.roll_greeks.get("vega", 0.0), 4),
            "iv_drift_from_roll": _round_or_none(
                reanchor.iv_drift_from_roll_pct(actual_iv, roll_iv), 4,
            ),
            "spot_drift_from_roll": _round_or_none(
                reanchor.spot_drift_from_roll_pct(actual_spot, roll.roll_spot), 4,
            ),
            "is_reanchor_row": is_reanchor_row,
            "bid_ask_spread": _round_or_none(spread_rupees, 4),
            "atm_straddle": _r4(curr_straddle),
            "atm_straddle_change_1m": _r4(straddle_change_1m),
            "atm_straddle_change_5m": _r4(straddle_change_5m),
            "atm_straddle_change_15m": _r4(straddle_change_15m),
            "atm_straddle_change_pct_1m": _r4(straddle_change_pct_1m),
            "atm_straddle_change_pct_5m": _r4(straddle_change_pct_5m),
            "atm_straddle_change_pct_15m": _r4(straddle_change_pct_15m),
            "atm_straddle_zscore_30m": _r4(curr_zscore),
            "atm_straddle_zscore_change_5m": _r4(zscore_change_5m),
            "atm_straddle_change_accel": _r4(straddle_change_accel),
            "atm_straddle_slope_5m": _r4(straddle_slope_5m),
            "atm_straddle_slope_15m": _r4(straddle_slope_15m),
            "atm_straddle_pct_change_from_open": _r4(straddle_pct_change_from_open),
            "distance_to_max_call_oi_strikes": _r4(c_dist_map.get(row_ts)),
            "distance_to_max_put_oi_strikes": _r4(p_dist_map.get(row_ts)),
            "max_call_oi_pct": _r4(c_pct_map.get(row_ts)),
            "max_put_oi_pct": _r4(p_pct_map.get(row_ts)),
            "chain_pcr": _r4(chain_pcr),
            "atm_pcr": _r4(atm_pcr),
            "oi_wall_bias": _r4(bias_map.get(row_ts)),
            "distance_to_call_build_wall": _r4(c_build_map.get(row_ts)),
            "distance_to_put_build_wall": _r4(p_build_map.get(row_ts)),
            "chain_pcr_change_5m": _r4(chain_pcr_change_5m),
            "atm_pcr_change_5m": _r4(atm_pcr_change_5m),
            "pinning_pressure": _r4(pin_map.get(row_ts)),
            "bs_reiv_pred": round(bs_reiv, 4) if bs_reiv is not None else None,
            "dgt_reiv_pred": round(dgt_reiv, 4) if dgt_reiv is not None else None,
            "actual_ltp_t": round(actual_ltp, 4),
            "actual_ltp_t_plus_5m": round(ltp_fwd, 4) if ltp_fwd is not None else None,
            "residual_5m": round(residual_5m, 4) if residual_5m is not None else None,
            "residual_pct_5m": round(residual_pct_5m, 4) if residual_pct_5m is not None else None,
        }

        # Future trajectory targets (10m window)
        traj_10m = option_timeline.analyze_future_trajectory(row_ts, 600.0)
        if traj_10m:
            base_paise = traj_10m["baseline_paise"]
            high_paise = traj_10m["future_high_paise"]
            low_paise = traj_10m["future_low_paise"]
            time_to_high = traj_10m["time_to_high_sec"]
            time_to_low = traj_10m["time_to_low_sec"]
            high_first_flag = traj_10m["high_first"]
            low_first_flag = traj_10m["low_first"]

            mfe_pct_10m = 100.0 * (high_paise - base_paise) / base_paise
            mae_pct_10m = 100.0 * (base_paise - low_paise) / base_paise

            high_first_bonus = 20.0 if high_first_flag else -20.0
            speed_bonus = 60.0 / max(time_to_high, 10.0)
            entry_quality_score = (
                0.35 * mfe_pct_10m
                - 0.25 * mae_pct_10m
                + 0.20 * high_first_bonus
                + 0.20 * speed_bonus
            )

            speed_score = 30.0 * (120.0 - min(time_to_high, 120.0)) / 120.0
            scalp_expectancy_score = (
                0.6 * mfe_pct_10m
                - 0.5 * mae_pct_10m
                + speed_score
            )

            scalp_score = (
                0.5 * mfe_pct_10m
                - 0.4 * mae_pct_10m
                + speed_bonus
            )

            if option_timeline.check_scalp_outcome_seconds(row_ts, 120.0, 20.0, 10.0) == 1:
                scalper_score = 100.0
            elif option_timeline.check_scalp_outcome_seconds(row_ts, 120.0, 15.0, 7.0) == 1:
                scalper_score = 70.0
            elif option_timeline.check_scalp_outcome_seconds(row_ts, 120.0, 10.0, 5.0) == 1:
                scalper_score = 40.0
            else:
                scalper_score = 0.0
        else:
            mfe_pct_10m = None
            mae_pct_10m = None
            high_first_flag = None
            low_first_flag = None
            time_to_high = None
            time_to_low = None
            entry_quality_score = None
            scalp_expectancy_score = None
            scalp_score = None
            scalper_score = None

        hit_targets = {}
        for H in (60, 120, 300):
            if traj_10m:
                hit_targets[f"hit_10pct_before_5pct_down_{H}s"] = option_timeline.check_scalp_outcome_seconds(row_ts, float(H), 10.0, 5.0)
                hit_targets[f"hit_15pct_before_7pct_down_{H}s"] = option_timeline.check_scalp_outcome_seconds(row_ts, float(H), 15.0, 7.0)
                hit_targets[f"hit_20pct_before_10pct_down_{H}s"] = option_timeline.check_scalp_outcome_seconds(row_ts, float(H), 20.0, 10.0)
            else:
                hit_targets[f"hit_10pct_before_5pct_down_{H}s"] = None
                hit_targets[f"hit_15pct_before_7pct_down_{H}s"] = None
                hit_targets[f"hit_20pct_before_10pct_down_{H}s"] = None

        if traj_10m:
            hit_7_3_60 = option_timeline.check_scalp_outcome_seconds(row_ts, 60.0, 7.0, 3.0)
            hit_5_2_30 = option_timeline.check_scalp_outcome_seconds(row_ts, 30.0, 5.0, 2.0)
        else:
            hit_7_3_60 = None
            hit_5_2_30 = None

        row.update({
            "mfe_pct_10m": _round_or_none(mfe_pct_10m, 4),
            "mae_pct_10m": _round_or_none(mae_pct_10m, 4),
            "future_high_first_10m": high_first_flag,
            "future_low_first_10m": low_first_flag,
            "time_to_high_sec_10m": _round_or_none(time_to_high, 2),
            "time_to_low_sec_10m": _round_or_none(time_to_low, 2),
            "entry_quality_score": _round_or_none(entry_quality_score, 4),
            "scalp_expectancy_score": _round_or_none(scalp_expectancy_score, 4),
            "scalp_score": _round_or_none(scalp_score, 4),
            "scalper_score": _round_or_none(scalper_score, 2),
            "hit_7pct_before_3pct_down_60s": hit_7_3_60,
            "hit_5pct_before_2pct_down_30s": hit_5_2_30,
            "hit_10pct_before_5pct_down_60s": hit_targets["hit_10pct_before_5pct_down_60s"],
            "hit_15pct_before_7pct_down_60s": hit_targets["hit_15pct_before_7pct_down_60s"],
            "hit_20pct_before_10pct_down_60s": hit_targets["hit_20pct_before_10pct_down_60s"],
            "hit_10pct_before_5pct_down_120s": hit_targets["hit_10pct_before_5pct_down_120s"],
            "hit_15pct_before_7pct_down_120s": hit_targets["hit_15pct_before_7pct_down_120s"],
            "hit_20pct_before_10pct_down_120s": hit_targets["hit_20pct_before_10pct_down_120s"],
            "hit_10pct_before_5pct_down_300s": hit_targets["hit_10pct_before_5pct_down_300s"],
            "hit_15pct_before_7pct_down_300s": hit_targets["hit_15pct_before_7pct_down_300s"],
            "hit_20pct_before_10pct_down_300s": hit_targets["hit_20pct_before_10pct_down_300s"],
        })


        for lb in LOOKBACK_MINUTES:
            row[f"spot_change_{lb}m"] = _round_or_none(_change_at(index_timeline, row_ts, lb), 4)
            iv_chg = _change_at(index_timeline, row_ts, lb, iv_fn=iv_at)
            row[f"iv_change_{lb}m"] = _round_or_none(iv_chg * 100.0 if iv_chg is not None else None, 4)
            row[f"ltp_change_{lb}m"] = _round_or_none(_change_at(option_timeline, row_ts, lb), 4)
            row[f"volume_change_{lb}m"] = _volume_change_at(option_timeline, row_ts, lb)
            row[f"oi_change_{lb}m"] = _oi_change_at(option_timeline, row_ts, lb)
            row[f"oi_change_pct_{lb}m"] = _round_or_none(_oi_change_pct_at(option_timeline, row_ts, lb), 4)

        rows.append(row)

    return rows


def _round_or_none(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)
