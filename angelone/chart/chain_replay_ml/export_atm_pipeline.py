#!/usr/bin/env python3
"""
Pipeline to export ATM ± 10 strikes ML dataset with 5-minute future returns and direction.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from typing import Any
import numpy as np

# Add chart directory to sys.path to resolve imports correctly
_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)

from storage.chain_replay_export import (
    ChainReplayError,
    bootstrap_provider_for_underlying,
    ist_market_session_bounds,
    normalize_expiry_param,
    require_v1_ticks_schema,
    resolve_chain_tokens,
)
from chain_replay_ml.bs import expiry_close_ts, normalize_strike_rupees
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.ticks import TickTimeline, load_tick_timelines
from chain_replay_ml.features_atm_band import (
    find_atm_strike,
    select_atm_band_strikes,
    extract_timeline_features,
)

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


def atm_features_csv_path(
    chart_dir: str,
    underlying: str,
    date: str,
) -> str:
    index_key = normalize_index_name(underlying)
    out_dir = os.path.join(chart_dir, "data", "ml_features", "atm_band_exports")
    return os.path.join(out_dir, f"atm_features_{index_key}_{date}.csv")


def resolve_nearest_expiry(
    db_path: str,
    date_str: str,
    *,
    underlying: str = "NIFTY",
) -> str | None:
    index_key = normalize_index_name(underlying)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT DISTINCT expiry_date
            FROM token_day_meta
            WHERE expiry_date IS NOT NULL
              AND expiry_date >= ?
              AND name = ?
            ORDER BY expiry_date ASC
            LIMIT 1
            """,
            (date_str, index_key),
        ).fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        conn.close()


def ensure_atm_features_csv(
    *,
    chart_dir: str,
    underlying: str,
    date: str,
    expiry: str | None = None,
    step_sec: int = 10,
) -> str:
    """Return path to per-day ATM feature CSV, exporting from tick DB when missing."""
    csv_path = atm_features_csv_path(chart_dir, underlying, date)
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path

    db_path = replay_db_path(chart_dir, date)
    if not db_path:
        raise ChainReplayError(f"No replay DB for date={date}")

    expiry_resolved = str(expiry or "").strip()
    if not expiry_resolved or expiry_resolved.lower() == "nearest":
        expiry_resolved = resolve_nearest_expiry(db_path, date, underlying=underlying) or ""
    if not expiry_resolved:
        raise ChainReplayError(f"No option expiry found in DB for {underlying} on {date}")

    out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    return export_atm_features(
        chart_dir=chart_dir,
        underlying=underlying,
        expiry=expiry_resolved,
        date=date,
        step_sec=step_sec,
        out_dir=out_dir,
    )


def export_atm_features(
    *,
    chart_dir: str,
    underlying: str,
    expiry: str,
    date: str,
    step_sec: int = 10,
    direction_threshold_pct: float = 0.1,
    out_dir: str,
) -> str:
    db_path = replay_db_path(chart_dir, date)
    if not db_path:
        raise ChainReplayError(f"No replay DB for date={date}")

    index_key = normalize_index_name(underlying)
    bootstrap_provider_for_underlying(underlying, INDEX_CONFIG, normalize_index_name=normalize_index_name)
    expiry_norm = normalize_expiry_param(expiry)
    open_ts, close_ts = ist_market_session_bounds(date)
    expiry_ts = expiry_close_ts(expiry_norm)

    print(f"Connecting to database: {db_path}")
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

        print(f"Loading spot timeline for index token {index_token}...")
        index_tl = load_tick_timelines(conn, [index_token], open_ts, close_ts).get(index_token)
        if not index_tl or not index_tl.timestamps:
            raise ChainReplayError("No index ticks in session")

        # Get all options timelines
        opt_tokens_all = [
            tok for tok, meta in token_meta.items()
            if meta.get("type") in ("CE", "PE")
        ]
        print(f"Loading tick timelines for {len(opt_tokens_all)} option contracts...")
        all_timelines = load_tick_timelines(conn, opt_tokens_all, open_ts, close_ts)

        # Mapping of (strike_rupees, type) -> (token, symbol, timeline)
        strike_mapping = {}
        for tok, meta in token_meta.items():
            if meta.get("type") in ("CE", "PE"):
                strike_r = normalize_strike_rupees(meta.get("strike"))
                opt_type = meta.get("type")
                opt_tl = all_timelines.get(tok)
                if opt_tl:
                    strike_mapping[(strike_r, opt_type)] = (tok, meta.get("symbol", ""), opt_tl)

        step = STRIKE_STEP.get(index_key, 50)

        # Generate sampling timestamps:
        # We need 60 seconds at start for lookbacks and 300 seconds (5m) at end for forward labels
        grid_start = open_ts + 60.0
        grid_end = close_ts - 300.0
        if grid_end <= grid_start:
            raise ChainReplayError(f"Market session for {date} is too short.")

        timestamps = []
        t = grid_start
        while t <= grid_end + 0.001:
            timestamps.append(t)
            t += step_sec

        # Precompute Spot EMA9 and EMA20 on a 1-minute grid
        print("Precomputing spot index EMA9 and EMA20 on a 1-minute grid...")
        minutes = np.arange(open_ts, close_ts + 1.0, 60.0)
        prices_1m = []
        last_p = index_tl.ltps_paise[0]/100.0 if index_tl.ltps_paise else 0.0
        for m in minutes:
            p = index_tl.ltp_rupees_at(m)
            if p is not None:
                last_p = p
            prices_1m.append(last_p)
        prices_1m = np.array(prices_1m)
        
        alpha_9 = 2.0 / (9 + 1)
        alpha_20 = 2.0 / (20 + 1)
        
        ema9 = np.zeros_like(prices_1m)
        ema20 = np.zeros_like(prices_1m)
        
        if len(prices_1m) > 0:
            ema9[0] = prices_1m[0]
            ema20[0] = prices_1m[0]
            for idx_m in range(1, len(prices_1m)):
                ema9[idx_m] = prices_1m[idx_m] * alpha_9 + ema9[idx_m-1] * (1.0 - alpha_9)
                ema20[idx_m] = prices_1m[idx_m] * alpha_20 + ema20[idx_m-1] * (1.0 - alpha_20)

        # Identify crossovers in the 1-minute grid
        crossovers = []
        if len(prices_1m) > 0:
            last_state = ema9[0] > ema20[0]
            # Seed crossover at the start of the day
            crossovers.append({'ts': minutes[0], 'price': prices_1m[0], 'dir': 1 if last_state else -1})
            for idx_m in range(1, len(prices_1m)):
                curr_state = ema9[idx_m] > ema20[idx_m]
                if curr_state != last_state:
                    crossovers.append({
                        'ts': minutes[idx_m],
                        'price': prices_1m[idx_m],
                        'dir': 1 if curr_state else -1
                    })
                    last_state = curr_state

        print(f"Generating features for {len(timestamps)} timestamps (every {step_sec}s)...")

        rows = []
        for ts in timestamps:
            spot = index_tl.ltp_rupees_at(ts)
            if spot is None or spot <= 0:
                continue

            # Lookup precomputed EMA values
            idx_ts = int((ts - open_ts) / 60.0)
            idx_ts = max(0, min(idx_ts, len(minutes) - 1))
            ema9_now = float(ema9[idx_ts]) if len(ema9) > 0 else None
            ema20_now = float(ema20[idx_ts]) if len(ema20) > 0 else None
            
            idx_1m_ago = max(0, idx_ts - 1)
            ema9_1m_ago = float(ema9[idx_1m_ago]) if len(ema9) > 0 else None

            # Calculate Spot EMA crossover features
            ema9_gt_ema20 = 0.0
            ema_spread_vs_spot_pct = 0.0
            time_since_cross_min = 60.0
            price_dist_from_cross = 0.0
            
            if ema9_now is not None and ema20_now is not None:
                ema9_gt_ema20 = 1.0 if ema9_now > ema20_now else 0.0
                ema_spread_vs_spot_pct = float(100.0 * (ema9_now - ema20_now) / spot)
                
                # Find the latest crossover before or at ts
                latest_cross = crossovers[0]
                for cross in crossovers:
                    if cross['ts'] <= ts:
                        latest_cross = cross
                    else:
                        break
                time_since_cross_min = float(max(0.0, (ts - latest_cross['ts']) / 60.0))
                price_dist_from_cross = float(100.0 * (spot - latest_cross['price']) / latest_cross['price'])

            atm = find_atm_strike(spot, step)
            band_strikes = select_atm_band_strikes(atm, step, band_size=10)

            # Spot future targets (maximum return and minimum return)
            traj_spot = index_tl.analyze_future_trajectory(ts, 300.0)
            target_spot_max_return_5m_pct = None
            target_spot_min_return_5m_pct = None
            target_spot_direction_5m = None
            target_spot_minutes_to_target_5m = None
            target_spot_first_event_5m = None
            if traj_spot:
                high_paise = traj_spot.get("future_high_paise")
                low_paise = traj_spot.get("future_low_paise")
                baseline_paise = traj_spot.get("baseline_paise")
                if high_paise is not None and baseline_paise is not None and baseline_paise > 0:
                    target_spot_max_return_5m_pct = float((high_paise - baseline_paise) / baseline_paise * 100.0)
                    target_spot_minutes_to_target_5m = float(traj_spot.get("time_to_high_sec") / 60.0)
                    target_spot_direction_5m = 1 if target_spot_max_return_5m_pct >= direction_threshold_pct else 0
                if low_paise is not None and baseline_paise is not None and baseline_paise > 0:
                    target_spot_min_return_5m_pct = float((low_paise - baseline_paise) / baseline_paise * 100.0)
                target_spot_first_event_5m = index_tl.check_scalp_outcome_seconds(ts, 300.0, direction_threshold_pct, direction_threshold_pct)

            # Collect features for each strike in ATM ± 10 band
            for strike_r in band_strikes:
                for opt_type in ("CE", "PE"):
                    map_entry = strike_mapping.get((strike_r, opt_type))
                    if not map_entry:
                        continue

                    tok, symbol, opt_tl = map_entry
                    
                    # Extract historical features
                    feats = extract_timeline_features(
                        ts=ts,
                        index_timeline=index_tl,
                        option_timeline=opt_tl,
                        option_type=opt_type,
                        strike_rupees=strike_r,
                        atm_strike_price=atm,
                        expiry_ts=expiry_ts,
                        ema9_now=ema9_now,
                        ema20_now=ema20_now,
                        ema9_1m_ago=ema9_1m_ago,
                        ema9_gt_ema20=ema9_gt_ema20,
                        ema_spread_vs_spot_pct=ema_spread_vs_spot_pct,
                        time_since_cross_min=time_since_cross_min,
                        price_dist_from_cross=price_dist_from_cross,
                        open_ts=open_ts,
                        close_ts=close_ts,
                    )
                    if not feats:
                        continue

                    # Future target calculation for option price (maximum return and minimum return)
                    ltp = feats.get("ltp")
                    target_max_return_5m_pct = None
                    target_min_return_5m_pct = None
                    target_direction_5m = None
                    target_minutes_to_target_5m = None
                    target_first_event_5m = None
                    if ltp is not None and ltp > 0:
                        # Dynamic Target
                        if ltp > 100.0:
                            tgt = 2.0
                        elif ltp >= 50.0:
                            tgt = 3.0
                        elif ltp >= 20.0:
                            tgt = 5.0
                        else:
                            tgt = 10.0
                        sl = 5.0

                        traj_opt = opt_tl.analyze_future_trajectory(ts, 300.0)
                        if traj_opt:
                            high_paise = traj_opt.get("future_high_paise")
                            low_paise = traj_opt.get("future_low_paise")
                            baseline_paise = traj_opt.get("baseline_paise")
                            if high_paise is not None and baseline_paise is not None and baseline_paise > 0:
                                target_max_return_5m_pct = float((high_paise - baseline_paise) / baseline_paise * 100.0)
                                target_minutes_to_target_5m = float(traj_opt.get("time_to_high_sec") / 60.0)
                                target_direction_5m = 1 if target_max_return_5m_pct >= tgt else 0
                            if low_paise is not None and baseline_paise is not None and baseline_paise > 0:
                                target_min_return_5m_pct = float((low_paise - baseline_paise) / baseline_paise * 100.0)
                        
                        target_first_event_5m = opt_tl.check_scalp_outcome_seconds(ts, 300.0, tgt, sl)

                    row = {
                        "date": date,
                        "underlying": underlying,
                        "expiry": expiry_norm,
                        "token": tok,
                        "symbol": symbol,
                        "option_type": opt_type,
                        "timestamp": ts,
                        "target_max_return_5m_pct": target_max_return_5m_pct,
                        "target_min_return_5m_pct": target_min_return_5m_pct,
                        "target_direction_5m": target_direction_5m,
                        "target_minutes_to_target_5m": target_minutes_to_target_5m,
                        "target_first_event_5m": target_first_event_5m,
                        "target_spot_max_return_5m_pct": target_spot_max_return_5m_pct,
                        "target_spot_min_return_5m_pct": target_spot_min_return_5m_pct,
                        "target_spot_direction_5m": target_spot_direction_5m,
                        "target_spot_minutes_to_target_5m": target_spot_minutes_to_target_5m,
                        "target_spot_first_event_5m": target_spot_first_event_5m,
                    }
                    row.update(feats)
                    rows.append(row)

        if not rows:
            raise ChainReplayError("No feature rows were successfully generated.")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"atm_features_{underlying}_{date}.csv")
        
        # Determine CSV fieldnames
        fieldnames = list(rows[0].keys())
        meta_keys = ["date", "underlying", "expiry", "token", "symbol", "option_type", "timestamp", "strike", "spot"]
        target_keys = [
            "target_max_return_5m_pct",
            "target_min_return_5m_pct",
            "target_direction_5m",
            "target_minutes_to_target_5m",
            "target_first_event_5m",
            "target_spot_max_return_5m_pct",
            "target_spot_min_return_5m_pct",
            "target_spot_direction_5m",
            "target_spot_minutes_to_target_5m",
            "target_spot_first_event_5m",
        ]
        other_keys = [k for k in fieldnames if k not in meta_keys and k not in target_keys]
        ordered_fieldnames = meta_keys + other_keys + target_keys

        print(f"Writing {len(rows)} feature rows to {out_path}...")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Successfully exported dataset to: {out_path}")
        return out_path

    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ATM ± 10 strikes option features & targets.")
    parser.add_argument("--underlying", default="NIFTY", help="Underlying index name (default: NIFTY)")
    parser.add_argument("--expiry", default="nearest", help="Expiry date YYYY-MM-DD or 'nearest' (default: nearest)")
    parser.add_argument("--date", default="2026-06-22", help="Trading date to process YYYY-MM-DD (default: 2026-06-22)")
    parser.add_argument("--step-sec", type=int, default=10, help="Grid step size in seconds (default: 10)")
    parser.add_argument("--threshold-pct", type=float, default=5.0, help="Direction target percent threshold (default: 5.0)")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV features")

    args = parser.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = os.path.join(_CHART_DIR, "data", "ml_features", "atm_band_exports")

    try:
        t0 = time.monotonic()
        export_atm_features(
            chart_dir=_CHART_DIR,
            underlying=args.underlying,
            expiry=args.expiry,
            date=args.date,
            step_sec=args.step_sec,
            direction_threshold_pct=args.threshold_pct,
            out_dir=args.out_dir,
        )
        print(f"Pipeline finished in {time.monotonic() - t0:.2f}s")
        return 0
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
