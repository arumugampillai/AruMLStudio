"""Batch export pipeline: SQLite replay DB → Phase 1 feature rows."""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from typing import Any, Iterable

from path_config import CHART_DATA_ROOT as _CHART_DIR
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()
from storage.chain_replay_export import (
    ChainReplayError,
    bootstrap_provider_for_underlying,
    ist_market_session_bounds,
    normalize_expiry_param,
    require_v1_ticks_schema,
    resolve_chain_tokens,
)

from .bs import expiry_close_ts, normalize_strike_rupees
from .constants import MIN_LTP_RUPEES, PHASE1_COLUMNS, RISK_FREE_RATE
from .features import atm_strike, build_option_rows
from . import bs
from .reanchor import ReanchorThresholds
from .ticks import TickTimeline, load_tick_timelines, minute_grid

INDEX_CONFIG = {
    "NIFTY": {
        "title": "NIFTY",
        "display_symbol": "Nifty 50",
        "exch_seg": "NFO",
        "index_token": "99926000",
        "index_exchange_type": 1,
    },
    "SENSEX": {
        "title": "SENSEX",
        "display_symbol": "SENSEX",
        "exch_seg": "BFO",
        "index_token": "99919000",
        "index_exchange_type": 3,
    },
}

STRIKE_STEP = {"NIFTY": 50, "SENSEX": 100}


def normalize_index_name(raw: str) -> str:
    key = str(raw or "NIFTY").strip().upper()
    return key if key in INDEX_CONFIG else "NIFTY"


def replay_db_path(chart_dir: str, day: str) -> str | None:
    from tick_data_paths import replay_db_path as _replay_db_path

    return _replay_db_path(chart_dir, day)


def select_option_tokens(
    token_meta: dict[str, dict[str, Any]],
    *,
    strikes: Iterable[float] | None = None,
    types: Iterable[str] | None = None,
    atm_band: int | None = None,
    spot_rupees: float | None = None,
    underlying: str = "NIFTY",
) -> list[tuple[str, dict[str, Any]]]:
    type_set = {t.upper() for t in types} if types else {"CE", "PE"}
    strike_set = {float(s) for s in strikes} if strikes else None

    if atm_band is not None and spot_rupees and spot_rupees > 0:
        step = STRIKE_STEP.get(underlying.upper(), 50)
        atm = atm_strike(spot_rupees, step)
        strike_set = {atm + i * step for i in range(-atm_band, atm_band + 1)}

    selected: list[tuple[str, dict[str, Any]]] = []
    for tok, meta in token_meta.items():
        if meta.get("type") == "INDEX":
            continue
        opt_type = str(meta.get("type") or "").upper()
        if opt_type not in type_set:
            continue
        strike_r = normalize_strike_rupees(meta.get("strike"))
        if strike_set is not None and strike_r not in strike_set:
            continue
        selected.append((tok, meta))
    return sorted(selected, key=lambda x: (normalize_strike_rupees(x[1].get("strike")), x[1].get("type")))


def _delta_at_seed(
    index_tl: TickTimeline,
    option_tl: TickTimeline,
    seed_ts: float,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
) -> float | None:
    spot = index_tl.ltp_rupees_at(seed_ts)
    ltp = option_tl.ltp_rupees_at(seed_ts)
    if spot is None or ltp is None or ltp < MIN_LTP_RUPEES:
        return None
    t = bs.time_to_expiry_years(expiry_ts, seed_ts)
    if t <= 0:
        return None
    iv = bs.implied_volatility(option_type, ltp, spot, strike_rupees, RISK_FREE_RATE, t)
    if iv is None:
        return None
    return bs.greeks(option_type, spot, strike_rupees, RISK_FREE_RATE, t, iv)["delta"]


def select_delta_profile_tokens(
    token_meta: dict[str, dict[str, Any]],
    *,
    index_tl: TickTimeline,
    timelines: dict[str, TickTimeline],
    spot_rupees: float,
    seed_ts: float,
    expiry_ts: float,
    underlying: str,
    target_delta: float = 0.15,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Pick all CE/PE with delta in (target, ATM] band at seed time.

    CE: target_delta < delta <= delta_ATM
    PE: delta_ATM <= delta < -target_delta
    """
    step = STRIKE_STEP.get(underlying.upper(), 50)
    atm = float(atm_strike(spot_rupees, step))

    scored: dict[tuple[str, float], tuple[str, dict[str, Any], float]] = {}
    for tok, meta in token_meta.items():
        if meta.get("type") == "INDEX":
            continue
        opt_type = str(meta.get("type") or "").upper()
        if opt_type not in ("CE", "PE"):
            continue
        strike_r = normalize_strike_rupees(meta.get("strike"))
        opt_tl = timelines.get(tok)
        if not opt_tl or not opt_tl.timestamps:
            continue
        delta = _delta_at_seed(index_tl, opt_tl, seed_ts, opt_type, strike_r, expiry_ts)
        if delta is None:
            continue
        scored[(opt_type, strike_r)] = (tok, meta, delta)

    if not scored:
        return [], {
            "atm_strike": atm,
            "target_delta": target_delta,
            "atm_ce_delta": None,
            "atm_pe_delta": None,
            "picked": [],
        }

    def atm_delta(opt_type: str) -> float | None:
        entry = scored.get((opt_type, atm))
        if entry:
            return entry[2]
        same_type = [(strike, val[2]) for (ot, strike), val in scored.items() if ot == opt_type]
        if not same_type:
            return None
        return min(same_type, key=lambda x: abs(x[0] - atm))[1]

    atm_ce_delta = atm_delta("CE")
    atm_pe_delta = atm_delta("PE")

    selected: list[tuple[str, dict[str, Any]]] = []
    picked_info: list[dict[str, Any]] = []

    for (opt_type, strike_r), (tok, meta, delta) in scored.items():
        in_band = False
        if opt_type == "CE" and atm_ce_delta is not None:
            in_band = target_delta < delta <= atm_ce_delta
        elif opt_type == "PE" and atm_pe_delta is not None:
            in_band = atm_pe_delta <= delta < -target_delta
        if not in_band:
            continue
        selected.append((tok, meta))
        picked_info.append({
            "type": opt_type,
            "strike": strike_r,
            "delta": round(delta, 4),
            "symbol": meta.get("symbol"),
            "token": tok,
        })

    picked_info.sort(
        key=lambda p: (
            0 if p["type"] == "CE" else 1,
            -p["delta"] if p["type"] == "CE" else p["delta"],
        ),
    )
    meta_out = {
        "atm_strike": atm,
        "spot_seed": round(spot_rupees, 2),
        "target_delta": target_delta,
        "atm_ce_delta": round(atm_ce_delta, 4) if atm_ce_delta is not None else None,
        "atm_pe_delta": round(atm_pe_delta, 4) if atm_pe_delta is not None else None,
        "picked": picked_info,
    }
    return (
        sorted(selected, key=lambda x: (x[1].get("type"), normalize_strike_rupees(x[1].get("strike")))),
        meta_out,
    )


def export_day_features(
    *,
    chart_dir: str,
    underlying: str,
    expiry: str,
    date: str,
    strikes: Iterable[float] | None = None,
    types: Iterable[str] | None = None,
    atm_band: int | None = None,
    delta_profile: float | None = None,
    thresholds: ReanchorThresholds | None = None,
    skip_warmup: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    db_path = replay_db_path(chart_dir, date)
    if not db_path:
        raise ChainReplayError(f"No replay DB for date={date}")

    index_key = normalize_index_name(underlying)
    bootstrap_provider_for_underlying(underlying, INDEX_CONFIG, normalize_index_name=normalize_index_name)
    expiry_norm = normalize_expiry_param(expiry)
    open_ts, close_ts = ist_market_session_bounds(date)
    expiry_ts = expiry_close_ts(expiry_norm)

    conn = sqlite3.connect(db_path)
    try:
        require_v1_ticks_schema(conn)
        token_meta = resolve_chain_tokens(
            conn,
            underlying=underlying,
            expiry=expiry_norm,
            as_of_date=date,
            index_config=INDEX_CONFIG,
            normalize_index_name=normalize_index_name,
        )
        index_token = None
        for tok, meta in token_meta.items():
            if meta.get("type") == "INDEX":
                index_token = tok
                break
        if not index_token:
            raise ChainReplayError("Index token not found in chain meta")

        index_tl = load_tick_timelines(conn, [index_token], open_ts, close_ts).get(index_token)
        if not index_tl or not index_tl.timestamps:
            raise ChainReplayError("No index ticks in session")

        spot_seed = index_tl.ltp_rupees_at(open_ts + 60) or index_tl.ltp_rupees_at(index_tl.timestamps[0])
        profile_meta: dict[str, Any] | None = None

        # Always load all CE/PE timelines for chain-level computations
        opt_tokens_all = [
            tok for tok, meta in token_meta.items()
            if meta.get("type") in ("CE", "PE")
        ]
        all_timelines = load_tick_timelines(conn, opt_tokens_all, open_ts, close_ts)

        if delta_profile is not None:
            options, profile_meta = select_delta_profile_tokens(
                token_meta,
                index_tl=index_tl,
                timelines=all_timelines,
                spot_rupees=spot_seed or 0.0,
                seed_ts=open_ts + 60,
                expiry_ts=expiry_ts,
                underlying=index_key,
                target_delta=delta_profile,
            )
        else:
            options = select_option_tokens(
                token_meta,
                strikes=strikes,
                types=types,
                atm_band=atm_band,
                spot_rupees=spot_seed,
                underlying=index_key,
            )
        if not options:
            raise ChainReplayError("No options matched strike/type filters")

        opt_tokens = [tok for tok, _ in options]
        timelines = {tok: all_timelines[tok] for tok in opt_tokens if tok in all_timelines}
        minute_times = minute_grid(open_ts, close_ts)

        # Build mapping of (strike, type) -> TickTimeline
        strike_to_timeline = {}
        for tok, meta in token_meta.items():
            if meta.get("type") in ("CE", "PE"):
                strike_r = normalize_strike_rupees(meta.get("strike"))
                opt_type = meta.get("type")
                opt_tl = all_timelines.get(tok)
                if opt_tl:
                    strike_to_timeline[(strike_r, opt_type)] = opt_tl

        unique_strikes = sorted(list({k[0] for k in strike_to_timeline.keys()}))

        # Precompute volatility and market structure maps over minute_times
        straddle_map = {}
        zscore_map = {}
        max_call_oi_dist_map = {}
        max_put_oi_dist_map = {}
        max_call_oi_pct_map = {}
        max_put_oi_pct_map = {}
        chain_pcr_map = {}
        atm_pcr_map = {}
        oi_wall_bias_map = {}
        dist_call_build_map = {}
        dist_put_build_map = {}
        pinning_pressure_map = {}

        from collections import deque
        import math
        import bisect

        window = deque(maxlen=30)
        rolling_sum = 0.0
        rolling_sum_sq = 0.0
        step = STRIKE_STEP.get(index_key, 50)

        for t in minute_times:
            spot = index_tl.ltp_rupees_at(t)
            if spot is None or spot <= 0:
                continue

            # 1. Blended Straddle Calculation
            if not unique_strikes:
                continue

            idx = bisect.bisect_right(unique_strikes, spot)
            if idx == 0:
                K_lower = unique_strikes[0]
                K_upper = unique_strikes[0]
            elif idx >= len(unique_strikes):
                K_lower = unique_strikes[-1]
                K_upper = unique_strikes[-1]
            else:
                K_lower = unique_strikes[idx - 1]
                K_upper = unique_strikes[idx]

            if K_upper == K_lower:
                w_upper = 0.5
                w_lower = 0.5
            else:
                w_upper = (spot - K_lower) / (K_upper - K_lower)
                w_lower = 1.0 - w_upper

            straddle_lower = None
            ce_lower_tl = strike_to_timeline.get((K_lower, "CE"))
            pe_lower_tl = strike_to_timeline.get((K_lower, "PE"))
            if ce_lower_tl and pe_lower_tl:
                l_ce = ce_lower_tl.ltp_rupees_at(t)
                l_pe = pe_lower_tl.ltp_rupees_at(t)
                if l_ce is not None and l_pe is not None and l_ce > 0 and l_pe > 0:
                    straddle_lower = l_ce + l_pe

            straddle_upper = None
            ce_upper_tl = strike_to_timeline.get((K_upper, "CE"))
            pe_upper_tl = strike_to_timeline.get((K_upper, "PE"))
            if ce_upper_tl and pe_upper_tl:
                u_ce = ce_upper_tl.ltp_rupees_at(t)
                u_pe = pe_upper_tl.ltp_rupees_at(t)
                if u_ce is not None and u_pe is not None and u_ce > 0 and u_pe > 0:
                    straddle_upper = u_ce + u_pe

            blended = None
            if straddle_lower is not None and straddle_upper is not None:
                blended = w_lower * straddle_lower + w_upper * straddle_upper
            elif straddle_lower is not None:
                blended = straddle_lower
            elif straddle_upper is not None:
                blended = straddle_upper

            if blended is not None:
                straddle_map[t] = blended
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
                    zscore = (blended - mean) / std if std > 1e-6 else 0.0
                    zscore_map[t] = zscore
                else:
                    zscore_map[t] = 0.0

            # 2. Total & Max OI, PCR, and Walls
            total_call_oi = 0
            total_put_oi = 0
            max_call_oi = -1
            max_call_strike = None
            max_put_oi = -1
            max_put_strike = None

            call_builds = []
            put_builds = []

            # Local ATM range strikes (+/- 5 strikes around current nearest ATM)
            nearest_atm = min(unique_strikes, key=lambda s: abs(s - spot))
            atm_idx = unique_strikes.index(nearest_atm)
            start_idx = max(0, atm_idx - 5)
            end_idx = min(len(unique_strikes) - 1, atm_idx + 5)
            local_atm_strikes = set(unique_strikes[start_idx:end_idx + 1])

            local_call_oi = 0
            local_put_oi = 0

            for strike_r in unique_strikes:
                ce_tl = strike_to_timeline.get((strike_r, "CE"))
                pe_tl = strike_to_timeline.get((strike_r, "PE"))

                if ce_tl:
                    oi_ce = ce_tl.oi_at(t)
                    if oi_ce is not None and oi_ce > 0:
                        total_call_oi += oi_ce
                        if oi_ce > max_call_oi:
                            max_call_oi = oi_ce
                            max_call_strike = strike_r

                        oi_ce_past = ce_tl.oi_at(t - 15 * 60)
                        if oi_ce_past is not None:
                            build_ce = oi_ce - oi_ce_past
                            if build_ce > 0:
                                call_builds.append((build_ce, strike_r))

                        if strike_r in local_atm_strikes:
                            local_call_oi += oi_ce

                if pe_tl:
                    oi_pe = pe_tl.oi_at(t)
                    if oi_pe is not None and oi_pe > 0:
                        total_put_oi += oi_pe
                        if oi_pe > max_put_oi:
                            max_put_oi = oi_pe
                            max_put_strike = strike_r

                        oi_pe_past = pe_tl.oi_at(t - 15 * 60)
                        if oi_pe_past is not None:
                            build_pe = oi_pe - oi_pe_past
                            if build_pe > 0:
                                put_builds.append((build_pe, strike_r))

                        if strike_r in local_atm_strikes:
                            local_put_oi += oi_pe

            if total_call_oi > 0:
                chain_pcr_map[t] = total_put_oi / total_call_oi
            if local_call_oi > 0:
                atm_pcr_map[t] = local_put_oi / local_call_oi

            if max_call_strike is not None:
                max_call_oi_dist_map[t] = (spot - max_call_strike) / step
                if total_call_oi > 0:
                    max_call_oi_pct_map[t] = max_call_oi / total_call_oi
            if max_put_strike is not None:
                max_put_oi_dist_map[t] = (spot - max_put_strike) / step
                if total_put_oi > 0:
                    max_put_oi_pct_map[t] = max_put_oi / total_put_oi

            if max_call_strike is not None and max_put_strike is not None:
                oi_wall_bias_map[t] = max_put_oi_dist_map[t] - max_call_oi_dist_map[t]
                pinning_pressure_map[t] = abs(max_call_oi_dist_map[t]) + abs(max_put_oi_dist_map[t])

            if call_builds:
                best_call_build_strike = max(call_builds, key=lambda x: x[0])[1]
                dist_call_build_map[t] = (spot - best_call_build_strike) / step
            elif max_call_strike is not None:
                dist_call_build_map[t] = max_call_oi_dist_map[t]

            if put_builds:
                best_put_build_strike = max(put_builds, key=lambda x: x[0])[1]
                dist_put_build_map[t] = (spot - best_put_build_strike) / step
            elif max_put_strike is not None:
                dist_put_build_map[t] = max_put_oi_dist_map[t]

        # Find the first valid blended straddle value of the day
        atm_straddle_open = None
        for t in minute_times:
            if t in straddle_map:
                atm_straddle_open = straddle_map[t]
                break

        all_rows: list[dict[str, Any]] = []
        for tok, meta in options:
            opt_tl = timelines.get(tok)
            if not opt_tl or not opt_tl.timestamps:
                continue
            opt_type = str(meta.get("type") or "CE").upper()
            if opt_type not in ("CE", "PE"):
                continue
            strike_r = normalize_strike_rupees(meta.get("strike"))
            rows = build_option_rows(
                date=date,
                underlying=index_key,
                expiry=expiry_norm,
                token=tok,
                symbol=str(meta.get("symbol") or ""),
                option_type=opt_type,
                strike_rupees=strike_r,
                index_timeline=index_tl,
                option_timeline=opt_tl,
                minute_times=minute_times,
                open_ts=open_ts,
                close_ts=close_ts,
                expiry_ts=expiry_ts,
                thresholds=thresholds,
                # Passed maps:
                straddle_map=straddle_map,
                zscore_map=zscore_map,
                max_call_oi_dist_map=max_call_oi_dist_map,
                max_put_oi_dist_map=max_put_oi_dist_map,
                max_call_oi_pct_map=max_call_oi_pct_map,
                max_put_oi_pct_map=max_put_oi_pct_map,
                chain_pcr_map=chain_pcr_map,
                atm_pcr_map=atm_pcr_map,
                oi_wall_bias_map=oi_wall_bias_map,
                dist_call_build_map=dist_call_build_map,
                dist_put_build_map=dist_put_build_map,
                pinning_pressure_map=pinning_pressure_map,
                atm_straddle_open=atm_straddle_open,
            )
            if skip_warmup:
                rows = [r for r in rows if not r.get("warmup_row")]
            all_rows.extend(rows)
        return all_rows, profile_meta
    finally:
        conn.close()


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE1_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_parquet(path: str, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ChainReplayError("Parquet export requires pandas and pyarrow") from exc
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(rows)
    for col in PHASE1_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[PHASE1_COLUMNS]
    df.to_parquet(path, index=False)


def default_out_path(
    chart_dir: str,
    underlying: str,
    expiry: str,
    date: str,
    ext: str,
    *,
    profile_suffix: str | None = None,
) -> str:
    out_dir = os.path.join(chart_dir, "data", "ml_features")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{profile_suffix}" if profile_suffix else ""
    return os.path.join(out_dir, f"phase1_{underlying}_{expiry}_{date}{suffix}.{ext}")


def delta_profile_suffix(target_delta: float) -> str:
    pct = int(round(target_delta * 100))
    return f"atm{pct:02d}"
