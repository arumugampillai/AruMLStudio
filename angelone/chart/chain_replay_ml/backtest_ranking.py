#!/usr/bin/env python3
"""
Backtest Option Strike Ranking and Selection using trained XGBoost Classifier and Regressors.
Formula (Expectancy-based):
    score = P_hit * MaxReturn - (1 - P_hit) * abs(MinReturn)
    
Simulates the actual trade path with tick-by-tick data from SQLite DB:
    - Entry: Buy highest score strike.
    - Stop loss: -5%
    - Target: Dynamic (+2% for premium > 100, +3% for 50-100, +5% for 20-50, +10% for <20)
    - Exit: Target hit, Stop loss hit, or 5-minute timeout (whichever comes first).
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# Add chart directory to sys.path
_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)

from chain_replay_ml.ticks import load_tick_timelines
from storage.chain_replay_export import ist_market_session_bounds

DEFAULT_DATA_DIR = os.path.join(_CHART_DIR, "data")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_DATA_DIR, "models")
DEFAULT_DATA_PATH = os.path.join(_CHART_DIR, "data", "ml_features", "atm_band_exports", "atm_features_NIFTY_all_days.csv")

FEATURE_COLUMNS = [
    "minutes_to_expiry",
    "delta",
    "abs_delta",
    "distance_from_spot_pct",
    "distance_from_atm_pct",
    "spot_change_5s",
    "spot_change_15s",
    "spot_change_30s",
    "spot_change_1m",
    "ltp_return_5s",
    "ltp_return_15s",
    "ltp_return_30s",
    "ltp_return_1m",
    "volume_change_5s",
    "volume_change_15s",
    "volume_change_30s",
    "volume_change_1m",
    "spot_vs_ema20_pct",
    "ema_spread_pct",
    "ema9_slope",
    "ema9_gt_ema20",
    "ema_spread_vs_spot_pct",
    "time_since_cross_min",
    "cross_age_decay",
    "price_dist_from_cross_pct",
    "minutes_since_open",
    "minutes_to_close",
    "is_first_hour",
    "is_last_hour",
    # Spot OHLC Normalized Features
    "spot_body_pct_10s",
    "spot_range_pct_10s",
    "spot_upper_wick_pct_10s",
    "spot_lower_wick_pct_10s",
    "spot_body_pct_30s",
    "spot_range_pct_30s",
    "spot_upper_wick_pct_30s",
    "spot_lower_wick_pct_30s",
    "spot_body_pct_1m",
    "spot_range_pct_1m",
    "spot_upper_wick_pct_1m",
    "spot_lower_wick_pct_1m",
    "spot_close_vs_high_1m_pct",
    "spot_close_vs_low_1m_pct",
    "spot_body_pct_prev1",
    "spot_body_pct_prev2",
    "spot_body_pct_prev3",
    "spot_range_pct_prev1",
    "spot_bullish_candle",
    "spot_bearish_candle",
    "spot_inside_bar",
    "spot_outside_bar",
    "spot_higher_high",
    "spot_lower_low",
    "spot_high_vs_prev_high_pct",
    "spot_low_vs_prev_low_pct",
    "spot_vol_ratio_10s_1m",
    "spot_vol_ratio_1m_5m",
    "spot_rv_5m",
    "spot_rv_10m",
    "spot_rv_ratio",
    "spot_dist_high_5m_pct",
    "spot_dist_low_5m_pct",
    "spot_range_pos_5m",
    # Option OHLC Normalized Features
    "opt_body_pct_10s",
    "opt_range_pct_10s",
    "opt_upper_wick_pct_10s",
    "opt_lower_wick_pct_10s",
    "opt_body_pct_30s",
    "opt_range_pct_30s",
    "opt_upper_wick_pct_30s",
    "opt_lower_wick_pct_30s",
    "opt_body_pct_1m",
    "opt_range_pct_1m",
    "opt_upper_wick_pct_1m",
    "opt_lower_wick_pct_1m",
    "opt_close_vs_high_1m_pct",
    "opt_close_vs_low_1m_pct",
    "opt_body_pct_prev1",
    "opt_body_pct_prev2",
    "opt_body_pct_prev3",
    "opt_range_pct_prev1",
    "opt_bullish_candle",
    "opt_bearish_candle",
    "opt_inside_bar",
    "opt_outside_bar",
    "opt_higher_high",
    "opt_lower_low",
    "opt_high_vs_prev_high_pct",
    "opt_low_vs_prev_low_pct",
    "opt_vol_ratio_10s_1m",
    "opt_vol_ratio_1m_5m",
    "opt_rv_5m",
    "opt_rv_10m",
    "opt_rv_ratio",
    "opt_dist_high_5m_pct",
    "opt_dist_low_5m_pct",
    "opt_range_pos_5m",
    "opt_volume_flow_5s",
    "opt_volume_flow_15s",
    "opt_volume_flow_30s",
    "opt_volume_flow_1m",
    "opt_volume_acc_5s_1m",
]


def replay_db_path(chart_dir: str, day: str) -> str | None:
    from tick_data_paths import replay_db_path as _replay_db_path

    return _replay_db_path(chart_dir, day)


def discover_model_stamps(model_dir: str) -> list[str]:
    from research.atm_band_ml.xgb_inference import discover_model_stamps as _discover

    return _discover(model_dir)


def load_models_for_stamp(model_dir: str, stamp: str) -> dict[str, dict[str, any]]:
    from research.atm_band_ml.xgb_inference import load_models_for_stamp as _load

    return _load(stamp, model_dir=model_dir)


def filter_by_delta_band_name(df: pd.DataFrame) -> pd.Series:
    abs_delta = df["delta"].abs()
    band = pd.Series(None, index=df.index, dtype=object)
    band.loc[(abs_delta >= 0.40) & (abs_delta <= 0.50)] = "A"
    band.loc[(abs_delta >= 0.25) & (abs_delta < 0.40)] = "B"
    band.loc[(abs_delta >= 0.15) & (abs_delta < 0.25)] = "C"
    return band


def check_scalp_outcome_seconds_with_time(
    timeline,
    start_ts: float,
    seconds: float,
    up_pct: float,
    down_pct: float,
) -> tuple[int, float]:
    if not timeline or not timeline.timestamps:
        return 0, seconds

    baseline = timeline.ltp_paise_at(start_ts)
    if baseline is None or baseline <= 0:
        return 0, seconds

    import bisect
    entry_idx = bisect.bisect_right(timeline.timestamps, start_ts) - 1
    end_idx = bisect.bisect_right(timeline.timestamps, start_ts + seconds)

    up_threshold = baseline * (1.0 + up_pct / 100.0)
    down_threshold = baseline * (1.0 - down_pct / 100.0)

    for idx in range(entry_idx + 1, end_idx):
        ltp = timeline.ltps_paise[idx]
        ts = timeline.timestamps[idx]
        if ltp >= up_threshold:
            return 1, max(0.0, ts - start_ts)
        if ltp <= down_threshold:
            return -1, max(0.0, ts - start_ts)

    return 0, seconds


def run_path_dependent_backtest(res_df: pd.DataFrame, chart_dir: str) -> pd.DataFrame:
    # Enrich dataframe with path-dependent exit returns and exit types
    res_df["strat_outcome_return"] = 0.0
    res_df["strat_outcome_type"] = "timeout"
    res_df["strat_elapsed_min"] = 5.0
    res_df["atm_outcome_return"] = 0.0
    res_df["atm_outcome_type"] = "timeout"
    
    unique_dates = res_df["date"].unique()
    for d in unique_dates:
        db_p = replay_db_path(chart_dir, d)
        if not db_p:
            print(f"Warning: No database found for date {d}. Skipping path validation for this day.")
            continue
            
        day_rows = res_df[res_df["date"] == d]
        tokens = list(set(day_rows["strat_token"].astype(str).tolist() + day_rows["atm_token"].astype(str).tolist()))
        
        print(f"  Simulating path for {d} using database {os.path.basename(db_p)} ({len(day_rows)} trades, {len(tokens)} option timelines)...")
        
        conn = sqlite3.connect(db_p)
        try:
            open_ts, close_ts = ist_market_session_bounds(d)
            timelines = load_tick_timelines(conn, tokens, open_ts, close_ts)
        finally:
            conn.close()
            
        for idx in day_rows.index:
            ts = res_df.loc[idx, "timestamp"]
            
            # --- Strategy Option Simulation ---
            strat_tok = str(res_df.loc[idx, "strat_token"])
            strat_ltp = res_df.loc[idx, "strat_ltp"]
            
            # Dynamic Target
            if strat_ltp > 100.0:
                strat_tgt = 2.0
            elif strat_ltp >= 50.0:
                strat_tgt = 3.0
            elif strat_ltp >= 20.0:
                strat_tgt = 5.0
            else:
                strat_tgt = 10.0
            strat_sl = 5.0
            
            strat_tl = timelines.get(strat_tok)
            if strat_tl:
                outcome, elapsed_sec = check_scalp_outcome_seconds_with_time(strat_tl, ts, 300.0, strat_tgt, strat_sl)
                res_df.loc[idx, "strat_elapsed_min"] = elapsed_sec / 60.0
                if outcome == 1:
                    res_df.loc[idx, "strat_outcome_return"] = strat_tgt
                    res_df.loc[idx, "strat_outcome_type"] = "target"
                elif outcome == -1:
                    res_df.loc[idx, "strat_outcome_return"] = -strat_sl
                    res_df.loc[idx, "strat_outcome_type"] = "sl"
                else:
                    # Timeout: exit at end price
                    entry_p = strat_tl.ltp_rupees_at(ts)
                    exit_p = strat_tl.ltp_rupees_at(ts + 300.0)
                    if entry_p and exit_p and entry_p > 0:
                        res_df.loc[idx, "strat_outcome_return"] = float((exit_p - entry_p) / entry_p * 100.0)
                    res_df.loc[idx, "strat_outcome_type"] = "timeout"
            
            # --- ATM Baseline Option Simulation ---
            atm_tok = str(res_df.loc[idx, "atm_token"])
            atm_tl = timelines.get(atm_tok)
            if atm_tl:
                atm_entry_ltp = atm_tl.ltp_rupees_at(ts)
                if atm_entry_ltp and atm_entry_ltp > 0:
                    if atm_entry_ltp > 100.0:
                        atm_tgt = 2.0
                    elif atm_entry_ltp >= 50.0:
                        atm_tgt = 3.0
                    elif atm_entry_ltp >= 20.0:
                        atm_tgt = 5.0
                    else:
                        atm_tgt = 10.0
                    atm_sl = 5.0
                    
                    outcome = atm_tl.check_scalp_outcome_seconds(ts, 300.0, atm_tgt, atm_sl)
                    if outcome == 1:
                        res_df.loc[idx, "atm_outcome_return"] = atm_tgt
                        res_df.loc[idx, "atm_outcome_type"] = "target"
                    elif outcome == -1:
                        res_df.loc[idx, "atm_outcome_return"] = -atm_sl
                        res_df.loc[idx, "atm_outcome_type"] = "sl"
                    else:
                        exit_p = atm_tl.ltp_rupees_at(ts + 300.0)
                        if exit_p:
                            res_df.loc[idx, "atm_outcome_return"] = float((exit_p - atm_entry_ltp) / atm_entry_ltp * 100.0)
                        res_df.loc[idx, "atm_outcome_type"] = "timeout"
                        
    return res_df


def compute_expectancy_metrics(df: pd.DataFrame, prefix: str) -> dict[str, any]:
    returns = df[f"{prefix}_outcome_return"].values
    types = df[f"{prefix}_outcome_type"].values
    
    total_trades = len(returns)
    if total_trades == 0:
        return {}
        
    # Winner = positive return, Loser = negative/zero return
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    win_pct = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_winner = wins.mean() if len(wins) > 0 else 0.0
    avg_loser = losses.mean() if len(losses) > 0 else 0.0
    
    sum_gains = wins.sum()
    sum_losses = abs(losses.sum())
    profit_factor = sum_gains / sum_losses if sum_losses > 0 else float("inf")
    
    net_return = returns.sum()
    
    # Calculate Max Drawdown from Cumulative Returns
    cum_returns = returns.cumsum()
    peak = -999999.0
    max_dd = 0.0
    for val in cum_returns:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
            
    # Max consecutive losses
    consecutive_losses = 0
    curr_consecutive = 0
    for r in returns:
        if r <= 0:
            curr_consecutive += 1
            if curr_consecutive > consecutive_losses:
                consecutive_losses = curr_consecutive
        else:
            curr_consecutive = 0
            
    # Exit outcome distribution
    type_counts = pd.Series(types).value_counts().to_dict()
    
    return {
        "total_trades": total_trades,
        "win_pct": win_pct,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "consecutive_losses": consecutive_losses,
        "net_return": net_return,
        "target_exits": type_counts.get("target", 0),
        "sl_exits": type_counts.get("sl", 0),
        "timeout_exits": type_counts.get("timeout", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest Option Strike Ranking and Expectancy.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to NIFTY features dataset")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory containing trained models")
    parser.add_argument("--stamp", help="Model timestamp (YYYYMMDD_HHMMSS). If omitted, finds the latest.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Min score required to take a trade (default: 0.0)")
    parser.add_argument("--val-only", action="store_true", help="Backtest only on the validation dates (June 22)")
    parser.add_argument("--backtest-date", help="Specific date YYYY-MM-DD to backtest on (overrides other filters)")
    parser.add_argument("--report-json", help="Path to save backtest metrics as JSON")
    parser.add_argument("--strike-filter", choices=["none", "experiment1"], default="none", help="Strike filtering type (default: none)")
    
    args = parser.parse_args(argv)
    
    if not os.path.exists(args.data):
        print(f"Data file not found: {args.data}", file=sys.stderr)
        return 1
        
    stamps = discover_model_stamps(args.model_dir)
    if not stamps:
        print(f"No models found in {args.model_dir}", file=sys.stderr)
        return 1
        
    stamp = args.stamp or stamps[0]
    print(f"Using model stamp: {stamp}")
    
    try:
        models = load_models_for_stamp(args.model_dir, stamp)
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        return 1
        
    print(f"Loading dataset: {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"Loaded dataset with {len(df)} rows.")

    if args.strike_filter == "experiment1":
        from chain_replay_ml.features_atm_band import filter_dataset_for_experiment_1
        print("Applying strike filter 'experiment1' to backtest data (ATM + OTM strikes only, premium Rs. 10 to ATM premium)...")
        df = filter_dataset_for_experiment_1(df)
        print(f"Dataset size after strike filter: {len(df)} rows.")
    
    # Exclude rows without features or targets
    required_cols = FEATURE_COLUMNS + ["target_max_return_5m_pct", "target_min_return_5m_pct", "ltp"]
    df = df.dropna(subset=required_cols).copy()
    
    # Assign Delta Band
    df["delta_band"] = filter_by_delta_band_name(df)
    df = df.dropna(subset=["delta_band"]).copy()
    
    # Handle Train/Val split based on date
    dates = sorted(df["date"].astype(str).unique())
    if args.backtest_date:
        df = df[df["date"].astype(str) == args.backtest_date].copy()
        print(f"Backtesting only on date {args.backtest_date} with {len(df)} rows.")
    elif args.val_only:
        val_frac = 0.2
        cut = max(1, int(len(dates) * (1.0 - val_frac)))
        val_dates = set(dates[cut:])
        df = df[df["date"].astype(str).isin(val_dates)].copy()
        print(f"Backtesting only on Validation Dates ({len(val_dates)} days: {sorted(list(val_dates))}) with {len(df)} rows.")
    else:
        print(f"Backtesting on All Dates ({len(dates)} days: {dates}) with {len(df)} rows.")
        
    if len(df) == 0:
        print("No rows to backtest.", file=sys.stderr)
        return 1
        
    # Generate predictions band-by-band
    print("Generating model predictions for each strike/band...")
    df["P_hit"] = np.nan
    df["pred_max_return"] = np.nan
    df["pred_min_return"] = np.nan
    
    for band, b_models in models.items():
        band_mask = df["delta_band"] == band
        if not band_mask.any():
            continue
            
        X = df.loc[band_mask, FEATURE_COLUMNS]
        
        # Classifier probability
        try:
            probs = b_models["clf"].predict_proba(X)
            df.loc[band_mask, "P_hit"] = probs[:, 1]
        except Exception:
            preds = b_models["clf"].predict(X)
            df.loc[band_mask, "P_hit"] = 1.0 / (1.0 + np.exp(-preds))
            
        # Regressors
        df.loc[band_mask, "pred_max_return"] = b_models["reg_max"].predict(X)
        df.loc[band_mask, "pred_min_return"] = b_models["reg_min"].predict(X)
        
    # Compute Expectancy Score
    df["score"] = df["P_hit"] * df["pred_max_return"] - (1.0 - df["P_hit"]) * df["pred_min_return"].abs()
    
    # Sort and rank within each timestamp
    print("Grouping options by timestamp and ranking...")
    results = []
    
    df["abs_dist_from_atm"] = df["distance_from_atm_pct"].abs()
    
    grouped = df.groupby(["date", "timestamp"])
    
    for (date_val, ts_val), group in grouped:
        # Strategy selection: Highest score
        top_opt = group.sort_values(by="score", ascending=False).iloc[0]
        
        # ATM baseline selection
        atm_options = group[group["distance_from_atm_pct"] == 0]
        if not atm_options.empty:
            atm_opt = atm_options.iloc[0]
        else:
            atm_opt = group.sort_values(by="abs_dist_from_atm").iloc[0]
            
        results.append({
            "date": date_val,
            "timestamp": ts_val,
            
            # Strategy Chosen
            "strat_token": top_opt["token"],
            "strat_symbol": top_opt["symbol"],
            "strat_option_type": top_opt["option_type"],
            "strat_delta": top_opt["delta"],
            "strat_ltp": top_opt["ltp"],
            "strat_band": top_opt["delta_band"],
            "strat_score": top_opt["score"],
            "strat_max_return": top_opt["target_max_return_5m_pct"],
            
            # ATM Baseline Chosen
            "atm_token": atm_opt["token"],
            "atm_symbol": atm_opt["symbol"],
        })
        
    res_df = pd.DataFrame(results)
    
    # Filter by score threshold
    strat_traded = res_df[res_df["strat_score"] >= args.score_threshold].copy()
    
    if len(strat_traded) == 0:
        print(f"No trades taken with current score threshold >= {args.score_threshold}.")
        return 0
        
    print(f"Simulating path-dependent outcomes for {len(strat_traded)} trades...")
    strat_traded = run_path_dependent_backtest(strat_traded, _CHART_DIR)
    
    # Compute expectancy metrics
    strat_m = compute_expectancy_metrics(strat_traded, "strat")
    atm_m = compute_expectancy_metrics(strat_traded, "atm")
    
    # Backtest Metrics Report
    print("\n" + "="*95)
    print(" PATH-DEPENDENT BACKTEST RESULTS (EXPECTANCY & TRADE RULES) ")
    print("="*95)
    print(f"Total Periods: {len(res_df)} | Trades Taken (score >= {args.score_threshold}): {len(strat_traded)} ({len(strat_traded)/len(res_df):.2%})")
    
    print("\nEXPECTANCY VS ATM BASELINE:")
    print(f"  {'Metric':<30} | {'Strategy (Score-Ranked)':<28} | {'ATM Baseline':<20}")
    print(f"  {'-'*30}-|-{'-'*28}-|-{'-'*20}")
    print(f"  {'Win Rate (Positive Returns)':<30} | {strat_m['win_pct']:<28.2%} | {atm_m['win_pct']:<20.2%}")
    print(f"  {'Average Winner':<30} | {strat_m['avg_winner']:<27.2f}% | {atm_m['avg_winner']:<19.2f}%")
    print(f"  {'Average Loser':<30} | {strat_m['avg_loser']:<27.2f}% | {atm_m['avg_loser']:<19.2f}%")
    print(f"  {'Profit Factor':<30} | {strat_m['profit_factor']:<28.4f} | {atm_m['profit_factor']:<20.4f}")
    print(f"  {'Maximum Drawdown':<30} | {strat_m['max_drawdown']:<27.2f}% | {atm_m['max_drawdown']:<19.2f}%")
    print(f"  {'Consecutive Losses':<30} | {strat_m['consecutive_losses']:<28d} | {atm_m['consecutive_losses']:<20d}")
    print(f"  {'Net Return (Sum of Returns)':<30} | {strat_m['net_return']:<27.2f}% | {atm_m['net_return']:<19.2f}%")
    
    print("\nEXIT DISTRIBUTION COMPARISON:")
    print(f"  {'Exit Type':<30} | {'Strategy (Score-Ranked)':<28} | {'ATM Baseline':<20}")
    print(f"  {'-'*30}-|-{'-'*28}-|-{'-'*20}")
    print(f"  {'Target Hit Exits':<30} | {strat_m['target_exits']:<28d} | {atm_m['target_exits']:<20d}")
    print(f"  {'Stop Loss Exits':<30} | {strat_m['sl_exits']:<28d} | {atm_m['sl_exits']:<20d}")
    print(f"  {'Timeout (5m) Exits':<30} | {strat_m['timeout_exits']:<28d} | {atm_m['timeout_exits']:<20d}")
    
    print("\nSTRATEGY CONTRACT DETAILS:")
    print("  Delta Band Distribution:")
    print(strat_traded["strat_band"].value_counts(normalize=True).to_string())
    print("  Option Type Distribution:")
    print(strat_traded["strat_option_type"].value_counts(normalize=True).to_string())
    print("  Average LTP (Premium): {:.2f}".format(strat_traded["strat_ltp"].mean()))
    print("="*95)
    
    if args.report_json:
        import json
        # Convert NumPy types to native Python types for JSON serialization
        def sanitize_dict(d):
            sanitized = {}
            for k, v in d.items():
                if isinstance(v, (np.integer, np.int64)):
                    sanitized[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    sanitized[k] = float(v)
                else:
                    sanitized[k] = v
            return sanitized
        
        trade_details = []
        for idx, row in strat_traded.iterrows():
            trade_details.append({
                "date": str(row["date"]),
                "timestamp": float(row["timestamp"]),
                "token": str(row["strat_token"]),
                "symbol": str(row["strat_symbol"]),
                "option_type": str(row["strat_option_type"]),
                "delta": float(row["strat_delta"]),
                "ltp": float(row["strat_ltp"]),
                "band": str(row["strat_band"]),
                "outcome_return": float(row["strat_outcome_return"]),
                "outcome_type": str(row["strat_outcome_type"]),
                "elapsed_min": float(row.get("strat_elapsed_min", 5.0)),
                "max_return": float(row.get("strat_max_return", 0.0)),
            })
        
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump({
                "strat": sanitize_dict(strat_m),
                "atm": sanitize_dict(atm_m),
                "trades": trade_details,
            }, f, indent=2)
        print(f"Saved backtest metrics JSON to: {args.report_json}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
