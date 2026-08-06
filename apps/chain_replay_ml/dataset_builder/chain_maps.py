"""Precompute chain-wide feature maps at sample timestamps."""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from chain_replay_ml.ticks import TickTimeline


def _atm6_side_ltp_sum(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    ts: float,
    atm_strike: float,
    step: int,
    option_type: str,
) -> float | None:
    """Sum LTP for ATM + 5 OTM strikes on one side (CE → higher strikes, PE → lower)."""
    if step <= 0:
        return None
    direction = 1 if option_type.upper() == "CE" else -1
    total = 0.0
    for i in range(6):
        strike = atm_strike + direction * i * step
        entry = strike_mapping.get((strike, option_type))
        if not entry:
            return None
        _, _, tl = entry
        ltp = tl.ltp_rupees_at(ts)
        if ltp is None or ltp <= 0:
            return None
        total += ltp
    return total


@dataclass
class ChainMaps:
    grid_timestamps: list[float] = field(default_factory=list)
    straddle: dict[float, float] = field(default_factory=dict)
    zscore: dict[float, float] = field(default_factory=dict)
    max_call_oi_dist: dict[float, float] = field(default_factory=dict)
    max_put_oi_dist: dict[float, float] = field(default_factory=dict)
    max_call_oi_pct: dict[float, float] = field(default_factory=dict)
    max_put_oi_pct: dict[float, float] = field(default_factory=dict)
    chain_pcr: dict[float, float] = field(default_factory=dict)
    atm_pcr: dict[float, float] = field(default_factory=dict)
    oi_wall_bias: dict[float, float] = field(default_factory=dict)
    dist_call_build: dict[float, float] = field(default_factory=dict)
    dist_put_build: dict[float, float] = field(default_factory=dict)
    pinning_pressure: dict[float, float] = field(default_factory=dict)
    atm_iv_ce: dict[float, float] = field(default_factory=dict)
    atm_iv_pe: dict[float, float] = field(default_factory=dict)
    total_call_oi: dict[float, float] = field(default_factory=dict)
    total_put_oi: dict[float, float] = field(default_factory=dict)
    total_ce_volume: dict[float, float] = field(default_factory=dict)
    total_pe_volume: dict[float, float] = field(default_factory=dict)
    otm_ce_volume: dict[float, float] = field(default_factory=dict)
    otm_pe_volume: dict[float, float] = field(default_factory=dict)
    otm_pcr_volume: dict[float, float] = field(default_factory=dict)
    iv_skew_atm: dict[float, float] = field(default_factory=dict)
    iv_call_put_skew: dict[float, float] = field(default_factory=dict)
    iv_skew_25d: dict[float, float] = field(default_factory=dict)
    iv_butterfly_25d: dict[float, float] = field(default_factory=dict)
    delta_w_volume_flow_1m: dict[float, float] = field(default_factory=dict)
    delta_w_volume_flow_5m: dict[float, float] = field(default_factory=dict)
    call_gex: dict[float, float] = field(default_factory=dict)
    put_gex: dict[float, float] = field(default_factory=dict)
    net_gex: dict[float, float] = field(default_factory=dict)
    chain_gex: dict[float, float] = field(default_factory=dict)
    gamma_flip_spot: dict[float, float] = field(default_factory=dict)
    gamma_flip_distance: dict[float, float] = field(default_factory=dict)
    synthetic_forward_spot: dict[float, float] = field(default_factory=dict)
    # feature_name → {ts → oi sum}
    oi_abs_delta: dict[str, dict[float, float]] = field(default_factory=dict)
    ce_atm6_ltp_sum: dict[float, float] = field(default_factory=dict)
    pe_atm6_ltp_sum: dict[float, float] = field(default_factory=dict)
    straddle_open: float | None = None

    def lookup(self, maps: dict[float, float], ts: float, lookback_sec: float = 0.0) -> float | None:
        key = ts - lookback_sec
        val = maps.get(key)
        if val is not None:
            return val
        if not maps:
            return None
        grid = self.grid_timestamps or sorted(maps.keys())
        if not grid:
            return None
        idx = bisect.bisect_right(grid, key) - 1
        if idx < 0:
            return None
        t = grid[idx]
        dt = key - t
        if 0 <= dt < 1.0:
            return maps.get(t)
        return None


def precompute_chain_maps(
    *,
    index_tl: TickTimeline,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    timestamps: list[float],
    strike_step: int,
    expiry_ts: float | None = None,
    include_iv_skew: bool = False,
    include_atm_iv: bool = False,
    include_delta_flow: bool = False,
    include_gex: bool = False,
    include_oi_delta_bands: bool = False,
) -> ChainMaps:
    """Build straddle, PCR, OI-wall maps at each sample timestamp.

    When Wave B flags and ``expiry_ts`` are set, also fills chain analytics
    for controller ``token.chain`` (IV skew, delta flow, GEX).
    """
    unique_strikes = sorted({k[0] for k in strike_mapping})
    if not unique_strikes:
        return ChainMaps()

    out = ChainMaps()
    grid_ts = sorted(set(timestamps))
    out.grid_timestamps = grid_ts
    window: deque[float] = deque(maxlen=30)
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    step = max(strike_step, 1)
    have_expiry = expiry_ts is not None and float(expiry_ts) > 0
    skew_expiry = float(expiry_ts) if have_expiry else 0.0
    do_skew = bool(include_iv_skew and have_expiry)
    do_atm_iv = bool((include_atm_iv or include_iv_skew) and have_expiry)
    do_delta_flow = bool(include_delta_flow and have_expiry)
    do_gex = bool(include_gex and have_expiry)
    do_oi_delta_bands = bool(include_oi_delta_bands and have_expiry)

    for t in grid_ts:
        spot = index_tl.ltp_rupees_at(t)
        if spot is None or spot <= 0:
            continue

        idx = bisect.bisect_right(unique_strikes, spot)
        if idx == 0:
            k_lower = k_upper = unique_strikes[0]
        elif idx >= len(unique_strikes):
            k_lower = k_upper = unique_strikes[-1]
        else:
            k_lower = unique_strikes[idx - 1]
            k_upper = unique_strikes[idx]

        if k_upper == k_lower:
            w_upper = w_lower = 0.5
        else:
            w_upper = (spot - k_lower) / (k_upper - k_lower)
            w_lower = 1.0 - w_upper

        def _straddle_at(strike: float) -> float | None:
            ce_tl = strike_mapping.get((strike, "CE"))
            pe_tl = strike_mapping.get((strike, "PE"))
            if not ce_tl or not pe_tl:
                return None
            _, _, ce = ce_tl
            _, _, pe = pe_tl
            l_ce = ce.ltp_rupees_at(t)
            l_pe = pe.ltp_rupees_at(t)
            if l_ce is None or l_pe is None or l_ce <= 0 or l_pe <= 0:
                return None
            return l_ce + l_pe

        s_lower = _straddle_at(k_lower)
        s_upper = _straddle_at(k_upper)
        blended = None
        if s_lower is not None and s_upper is not None:
            blended = w_lower * s_lower + w_upper * s_upper
        elif s_lower is not None:
            blended = s_lower
        elif s_upper is not None:
            blended = s_upper

        if blended is not None:
            out.straddle[t] = blended
            if out.straddle_open is None:
                out.straddle_open = blended
            if len(window) == 30:
                old = window.popleft()
                rolling_sum -= old
                rolling_sum_sq -= old ** 2
            window.append(blended)
            rolling_sum += blended
            rolling_sum_sq += blended ** 2
            n = len(window)
            if n >= 2:
                mean = rolling_sum / n
                variance = max(0.0, (rolling_sum_sq / n) - (mean ** 2))
                std = math.sqrt(variance)
                out.zscore[t] = (blended - mean) / std if std > 1e-6 else 0.0
            else:
                out.zscore[t] = 0.0

        nearest_atm = min(unique_strikes, key=lambda s: abs(s - spot))
        atm_idx = unique_strikes.index(nearest_atm)

        # Put–call parity synthetic forward at nearest ATM: K + (C − P).
        ce_atm_entry = strike_mapping.get((nearest_atm, "CE"))
        pe_atm_entry = strike_mapping.get((nearest_atm, "PE"))
        if ce_atm_entry and pe_atm_entry:
            _, _, ce_atm_tl = ce_atm_entry
            _, _, pe_atm_tl = pe_atm_entry
            c_ltp = ce_atm_tl.ltp_rupees_at(t)
            p_ltp = pe_atm_tl.ltp_rupees_at(t)
            if (
                c_ltp is not None
                and p_ltp is not None
                and c_ltp > 0
                and p_ltp > 0
            ):
                out.synthetic_forward_spot[t] = float(nearest_atm) + float(c_ltp) - float(p_ltp)

        ce_sum = _atm6_side_ltp_sum(strike_mapping, t, nearest_atm, step, "CE")
        if ce_sum is not None:
            out.ce_atm6_ltp_sum[t] = ce_sum
        pe_sum = _atm6_side_ltp_sum(strike_mapping, t, nearest_atm, step, "PE")
        if pe_sum is not None:
            out.pe_atm6_ltp_sum[t] = pe_sum

        local_strikes = set(
            unique_strikes[max(0, atm_idx - 5): min(len(unique_strikes), atm_idx + 6)]
        )

        total_call_oi = 0
        total_put_oi = 0
        total_ce_volume = 0
        total_pe_volume = 0
        otm_ce_volume = 0
        otm_pe_volume = 0
        max_call_oi = -1
        max_call_strike = None
        max_put_oi = -1
        max_put_strike = None
        call_builds: list[tuple[int, float]] = []
        put_builds: list[tuple[int, float]] = []
        local_call_oi = 0
        local_put_oi = 0

        for strike_r in unique_strikes:
            ce_entry = strike_mapping.get((strike_r, "CE"))
            pe_entry = strike_mapping.get((strike_r, "PE"))
            if ce_entry:
                _, _, ce_tl = ce_entry
                oi_ce = ce_tl.oi_at(t)
                if oi_ce is not None and oi_ce > 0:
                    total_call_oi += oi_ce
                    if oi_ce > max_call_oi:
                        max_call_oi = oi_ce
                        max_call_strike = strike_r
                    oi_past = ce_tl.oi_at(t - 15 * 60)
                    if oi_past is not None:
                        build = oi_ce - oi_past
                        if build > 0:
                            call_builds.append((build, strike_r))
                    if strike_r in local_strikes:
                        local_call_oi += oi_ce
                vol_ce = ce_tl.volume_at(t)
                if vol_ce is not None and vol_ce > 0:
                    total_ce_volume += int(vol_ce)
                    # OTM call: strike strictly above spot (ATM excluded).
                    if float(strike_r) > float(spot):
                        otm_ce_volume += int(vol_ce)
            if pe_entry:
                _, _, pe_tl = pe_entry
                oi_pe = pe_tl.oi_at(t)
                if oi_pe is not None and oi_pe > 0:
                    total_put_oi += oi_pe
                    if oi_pe > max_put_oi:
                        max_put_oi = oi_pe
                        max_put_strike = strike_r
                    oi_past = pe_tl.oi_at(t - 15 * 60)
                    if oi_past is not None:
                        build = oi_pe - oi_past
                        if build > 0:
                            put_builds.append((build, strike_r))
                    if strike_r in local_strikes:
                        local_put_oi += oi_pe
                vol_pe = pe_tl.volume_at(t)
                if vol_pe is not None and vol_pe > 0:
                    total_pe_volume += int(vol_pe)
                    # OTM put: strike strictly below spot (ATM excluded).
                    if float(strike_r) < float(spot):
                        otm_pe_volume += int(vol_pe)

        out.total_call_oi[t] = float(total_call_oi)
        out.total_put_oi[t] = float(total_put_oi)
        out.total_ce_volume[t] = float(total_ce_volume)
        out.total_pe_volume[t] = float(total_pe_volume)
        out.otm_ce_volume[t] = float(otm_ce_volume)
        out.otm_pe_volume[t] = float(otm_pe_volume)
        if otm_ce_volume > 0:
            out.otm_pcr_volume[t] = float(otm_pe_volume) / float(otm_ce_volume)

        if total_call_oi > 0:
            out.chain_pcr[t] = total_put_oi / total_call_oi
        if local_call_oi > 0:
            out.atm_pcr[t] = local_put_oi / local_call_oi

        if max_call_strike is not None:
            out.max_call_oi_dist[t] = (spot - max_call_strike) / step
            if total_call_oi > 0:
                out.max_call_oi_pct[t] = max_call_oi / total_call_oi
        if max_put_strike is not None:
            out.max_put_oi_dist[t] = (spot - max_put_strike) / step
            if total_put_oi > 0:
                out.max_put_oi_pct[t] = max_put_oi / total_put_oi

        if max_call_strike is not None and max_put_strike is not None:
            out.oi_wall_bias[t] = out.max_put_oi_dist[t] - out.max_call_oi_dist[t]
            out.pinning_pressure[t] = abs(out.max_call_oi_dist[t]) + abs(out.max_put_oi_dist[t])

        if call_builds:
            best = max(call_builds, key=lambda x: x[0])[1]
            out.dist_call_build[t] = (spot - best) / step
        elif max_call_strike is not None and t in out.max_call_oi_dist:
            out.dist_call_build[t] = out.max_call_oi_dist[t]

        if put_builds:
            best = max(put_builds, key=lambda x: x[0])[1]
            out.dist_put_build[t] = (spot - best) / step
        elif max_put_strike is not None and t in out.max_put_oi_dist:
            out.dist_put_build[t] = out.max_put_oi_dist[t]

        if do_skew:
            from .chain_iv_skew import compute_chain_iv_skew_at

            skew = compute_chain_iv_skew_at(
                strike_mapping,
                ts=t,
                spot=float(spot),
                atm_strike=float(nearest_atm),
                strike_step=step,
                expiry_ts=skew_expiry,
            )
            if skew.get("atm_iv_ce") is not None:
                out.atm_iv_ce[t] = float(skew["atm_iv_ce"])
            if skew.get("atm_iv_pe") is not None:
                out.atm_iv_pe[t] = float(skew["atm_iv_pe"])
            if skew.get("iv_skew_atm") is not None:
                out.iv_skew_atm[t] = float(skew["iv_skew_atm"])
            if skew.get("iv_call_put_skew") is not None:
                out.iv_call_put_skew[t] = float(skew["iv_call_put_skew"])
            if skew.get("iv_skew_25d") is not None:
                out.iv_skew_25d[t] = float(skew["iv_skew_25d"])
            if skew.get("iv_butterfly_25d") is not None:
                out.iv_butterfly_25d[t] = float(skew["iv_butterfly_25d"])
        elif do_atm_iv:
            from .chain_iv_skew import _option_iv_at
            from chain_replay_ml import bs as _bs

            t_exp = _bs.time_to_expiry_years(skew_expiry, t)
            iv_ce = _option_iv_at(
                strike_mapping,
                strike=float(nearest_atm),
                option_type="CE",
                ts=t,
                spot=float(spot),
                t_exp=t_exp,
            )
            iv_pe = _option_iv_at(
                strike_mapping,
                strike=float(nearest_atm),
                option_type="PE",
                ts=t,
                spot=float(spot),
                t_exp=t_exp,
            )
            if iv_ce is not None:
                out.atm_iv_ce[t] = float(iv_ce)
            if iv_pe is not None:
                out.atm_iv_pe[t] = float(iv_pe)

        if do_delta_flow:
            from .chain_delta_volume_flow import compute_all_delta_w_volume_flows_at

            flows = compute_all_delta_w_volume_flows_at(
                strike_mapping,
                index_tl=index_tl,
                ts=t,
                expiry_ts=skew_expiry,
            )
            if flows.get("delta_w_volume_flow_1m") is not None:
                out.delta_w_volume_flow_1m[t] = float(flows["delta_w_volume_flow_1m"])
            if flows.get("delta_w_volume_flow_5m") is not None:
                out.delta_w_volume_flow_5m[t] = float(flows["delta_w_volume_flow_5m"])

        if do_gex:
            from .chain_gex import compute_chain_gex_at

            gex = compute_chain_gex_at(
                strike_mapping,
                index_tl=index_tl,
                ts=t,
                expiry_ts=skew_expiry,
            )
            for key, store in (
                ("call_gex", out.call_gex),
                ("put_gex", out.put_gex),
                ("net_gex", out.net_gex),
                ("chain_gex", out.chain_gex),
                ("gamma_flip_spot", out.gamma_flip_spot),
                ("gamma_flip_distance", out.gamma_flip_distance),
            ):
                if gex.get(key) is not None:
                    store[t] = float(gex[key])

        if do_oi_delta_bands:
            from .chain_oi_delta_bands import compute_oi_abs_delta_bands_at

            bands = compute_oi_abs_delta_bands_at(
                strike_mapping,
                index_tl=index_tl,
                ts=t,
                expiry_ts=skew_expiry,
            )
            for key, val in bands.items():
                if val is None:
                    continue
                out.oi_abs_delta.setdefault(key, {})[t] = float(val)

    return out


def chain_features_at(
    maps: ChainMaps,
    ts: float,
    *,
    expiry_ts: float,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    index_tl: TickTimeline,
    atm_strike: int,
) -> dict[str, Any]:
    """Resolve chain-level registry columns at timestamp ts."""
    from chain_replay_ml import bs
    from chain_replay_ml.constants import EPS_T, RISK_FREE_RATE

    def _map_val(m: dict[float, float], lookback: float = 0.0) -> float | None:
        return maps.lookup(m, ts, lookback)

    curr_straddle = _map_val(maps.straddle)
    past_1m = _map_val(maps.straddle, 60.0)
    past_5m = _map_val(maps.straddle, 300.0)
    past_15m = _map_val(maps.straddle, 900.0)

    straddle_change_1m = (curr_straddle - past_1m) if curr_straddle is not None and past_1m is not None else None
    straddle_change_5m = (curr_straddle - past_5m) if curr_straddle is not None and past_5m is not None else None
    straddle_change_pct_1m = (
        straddle_change_1m / past_1m * 100.0
        if straddle_change_1m is not None and past_1m and past_1m > 0 else None
    )
    straddle_change_pct_5m = (
        straddle_change_5m / past_5m * 100.0
        if straddle_change_5m is not None and past_5m and past_5m > 0 else None
    )
    straddle_slope_5m = straddle_change_5m / 5.0 if straddle_change_5m is not None else None
    straddle_slope_15m = None
    if curr_straddle is not None and past_15m is not None:
        straddle_slope_15m = (curr_straddle - past_15m) / 15.0
    straddle_accel = None
    if straddle_change_1m is not None and straddle_change_5m is not None:
        straddle_accel = straddle_change_1m - (straddle_change_5m / 5.0)
    straddle_pct_from_open = None
    if curr_straddle is not None and maps.straddle_open and maps.straddle_open > 0:
        straddle_pct_from_open = 100.0 * (curr_straddle - maps.straddle_open) / maps.straddle_open

    curr_z = _map_val(maps.zscore)
    past_z_5m = _map_val(maps.zscore, 300.0)
    zscore_change_5m = (curr_z - past_z_5m) if curr_z is not None and past_z_5m is not None else None

    chain_pcr = _map_val(maps.chain_pcr)
    past_pcr_5m = _map_val(maps.chain_pcr, 300.0)
    chain_pcr_chg = (chain_pcr - past_pcr_5m) if chain_pcr is not None and past_pcr_5m is not None else None

    atm_pcr = _map_val(maps.atm_pcr)
    past_atm_pcr_5m = _map_val(maps.atm_pcr, 300.0)
    atm_pcr_chg = (atm_pcr - past_atm_pcr_5m) if atm_pcr is not None and past_atm_pcr_5m is not None else None

    spot = index_tl.ltp_rupees_at(ts)
    ce_atm6_sum = _map_val(maps.ce_atm6_ltp_sum)
    pe_atm6_sum = _map_val(maps.pe_atm6_ltp_sum)
    ce_atm6_ltp_to_spot_ratio = (
        float(ce_atm6_sum / spot)
        if ce_atm6_sum is not None and spot is not None and spot > 0
        else None
    )
    pe_atm6_ltp_to_spot_ratio = (
        float(pe_atm6_sum / spot)
        if pe_atm6_sum is not None and spot is not None and spot > 0
        else None
    )
    ce_pe_atm6_ltp_ratio = (
        float(ce_atm6_sum / (pe_atm6_sum + EPS_T))
        if ce_atm6_sum is not None and pe_atm6_sum is not None
        else None
    )
    ce_minus_pe_atm6_ltp = (
        float(ce_atm6_sum - pe_atm6_sum)
        if ce_atm6_sum is not None and pe_atm6_sum is not None
        else None
    )
    # Wave 6: ce_pe_atm6_ltp_diff_pct → Interaction (ce−pe)/(ce+pe)

    atm_iv = None
    if spot is not None and spot > 0:
        t_exp = bs.time_to_expiry_years(expiry_ts, ts)
        ivs: list[float] = []
        for opt_type in ("CE", "PE"):
            entry = strike_mapping.get((float(atm_strike), opt_type))
            if not entry:
                continue
            _, _, tl = entry
            ltp = tl.ltp_rupees_at(ts)
            if ltp is None or ltp <= 0:
                continue
            iv = bs.implied_volatility(opt_type, ltp, spot, float(atm_strike), RISK_FREE_RATE, t_exp)
            if iv is not None and iv > 0:
                ivs.append(iv)
        if ivs:
            atm_iv = sum(ivs) / len(ivs)

    out = {
        "atm_straddle": curr_straddle,
        "atm_straddle_change_1m": straddle_change_1m,
        "atm_straddle_change_5m": straddle_change_5m,
        "atm_straddle_change_pct_1m": straddle_change_pct_1m,
        "atm_straddle_change_pct_5m": straddle_change_pct_5m,
        "atm_straddle_zscore_30m": curr_z,
        "atm_straddle_zscore_change_5m": zscore_change_5m,
        "atm_straddle_change_accel": straddle_accel,
        "atm_straddle_slope_5m": straddle_slope_5m,
        "atm_straddle_slope_15m": straddle_slope_15m,
        "atm_straddle_pct_change_from_open": straddle_pct_from_open,
        "ce_atm6_ltp_sum": ce_atm6_sum,
        "pe_atm6_ltp_sum": pe_atm6_sum,
        "ce_atm6_ltp_to_spot_ratio": ce_atm6_ltp_to_spot_ratio,
        "pe_atm6_ltp_to_spot_ratio": pe_atm6_ltp_to_spot_ratio,
        "ce_pe_atm6_ltp_ratio": ce_pe_atm6_ltp_ratio,
        "ce_minus_pe_atm6_ltp": ce_minus_pe_atm6_ltp,
        # Wave 6: ce_pe_atm6_ltp_diff_pct → Interaction
        "distance_to_max_call_oi_strikes": _map_val(maps.max_call_oi_dist),
        "distance_to_max_put_oi_strikes": _map_val(maps.max_put_oi_dist),
        "max_call_oi_pct": _map_val(maps.max_call_oi_pct),
        "max_put_oi_pct": _map_val(maps.max_put_oi_pct),
        "distance_to_call_build_wall": _map_val(maps.dist_call_build),
        "distance_to_put_build_wall": _map_val(maps.dist_put_build),
        "oi_wall_bias": _map_val(maps.oi_wall_bias),
        "pinning_pressure": _map_val(maps.pinning_pressure),
        "chain_pcr": chain_pcr,
        "atm_pcr": atm_pcr,
        "chain_pcr_change_5m": chain_pcr_chg,
        "atm_pcr_change_5m": atm_pcr_chg,
        "iv_skew_atm": _map_val(maps.iv_skew_atm),
        "iv_call_put_skew": _map_val(maps.iv_call_put_skew),
        "iv_skew_25d": _map_val(maps.iv_skew_25d),
        "iv_butterfly_25d": _map_val(maps.iv_butterfly_25d),
        "atm_iv_ce": _map_val(maps.atm_iv_ce),
        "atm_iv_pe": _map_val(maps.atm_iv_pe),
        "total_call_oi": _map_val(maps.total_call_oi),
        "total_put_oi": _map_val(maps.total_put_oi),
        "total_ce_volume": _map_val(maps.total_ce_volume),
        "total_pe_volume": _map_val(maps.total_pe_volume),
        "otm_ce_volume": _map_val(maps.otm_ce_volume),
        "otm_pe_volume": _map_val(maps.otm_pe_volume),
        "otm_pcr_volume": _map_val(maps.otm_pcr_volume),
        "delta_w_volume_flow_1m": _map_val(maps.delta_w_volume_flow_1m),
        "delta_w_volume_flow_5m": _map_val(maps.delta_w_volume_flow_5m),
        "call_gex": _map_val(maps.call_gex),
        "put_gex": _map_val(maps.put_gex),
        "net_gex": _map_val(maps.net_gex),
        "chain_gex": _map_val(maps.chain_gex),
        "gamma_flip_spot": _map_val(maps.gamma_flip_spot),
        "gamma_flip_distance": _map_val(maps.gamma_flip_distance),
        "synthetic_forward_spot": _map_val(maps.synthetic_forward_spot),
        "_atm_iv": atm_iv,
    }
    from .chain_oi_delta_bands import OI_ABS_DELTA_BAND_FEATURES

    for _band_name in OI_ABS_DELTA_BAND_FEATURES:
        out[_band_name] = _map_val(maps.oi_abs_delta.get(_band_name, {}))
    return out
