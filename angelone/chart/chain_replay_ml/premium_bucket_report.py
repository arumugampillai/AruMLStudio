#!/usr/bin/env python3
"""
Premium Bucket Report for Experiment 1 filtered trades.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# Add chart directory to sys.path
_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)

from chain_replay_ml.ticks import load_tick_timelines
from chain_replay_ml.features_atm_band import filter_dataset_for_experiment_1
from chain_replay_ml.train_atm_model import FEATURE_COLUMNS
from chain_replay_ml.backtest_ranking import filter_by_delta_band_name, load_models_for_stamp, replay_db_path
from storage.chain_replay_export import ist_market_session_bounds


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


def run_bucket_analysis_for_date(
    date_str: str,
    stamp: str,
    score_threshold: float = 3.0,
) -> list[dict[str, any]]:
    models_dir = os.path.join(_CHART_DIR, "data", "ml_models")
    models = load_models_for_stamp(models_dir, stamp)
    
    csv_path = os.path.join(_CHART_DIR, "data", "ml_features", "atm_band_exports", f"atm_features_NIFTY_{date_str}.csv")
    df = pd.read_csv(csv_path)
    
    # Apply Experiment 1 filter
    df = filter_dataset_for_experiment_1(df)
    
    # Drop rows missing features
    required_cols = FEATURE_COLUMNS + ["target_max_return_5m_pct", "target_min_return_5m_pct", "ltp"]
    df = df.dropna(subset=required_cols).copy()
    
    # Assign Delta Band
    df["delta_band"] = filter_by_delta_band_name(df)
    df = df.dropna(subset=["delta_band"]).copy()
    
    # Generate predictions
    df["P_hit"] = np.nan
    df["pred_max_return"] = np.nan
    df["pred_min_return"] = np.nan
    
    for band, b_models in models.items():
        band_mask = df["delta_band"] == band
        if not band_mask.any():
            continue
            
        X = df.loc[band_mask, FEATURE_COLUMNS]
        try:
            probs = b_models["clf"].predict_proba(X)
            df.loc[band_mask, "P_hit"] = probs[:, 1]
        except Exception:
            preds = b_models["clf"].predict(X)
            df.loc[band_mask, "P_hit"] = 1.0 / (1.0 + np.exp(-preds))
            
        df.loc[band_mask, "pred_max_return"] = b_models["reg_max"].predict(X)
        df.loc[band_mask, "pred_min_return"] = b_models["reg_min"].predict(X)
        
    df["score"] = df["P_hit"] * df["pred_max_return"] - (1.0 - df["P_hit"]) * df["pred_min_return"].abs()
    
    # Group by timestamp and select highest score
    grouped = df.groupby("timestamp")
    trades = []
    
    for ts_val, group in grouped:
        top_opt = group.sort_values(by="score", ascending=False).iloc[0]
        if top_opt["score"] >= score_threshold:
            trades.append(top_opt)
            
    if not trades:
        print(f"No trades taken for {date_str} at threshold {score_threshold}")
        return []
        
    df_trades = pd.DataFrame(trades)
    
    # Load timelines to perform path-dependent checks and get exit times
    df_trades["token"] = df_trades["token"].astype(str)
    tokens = df_trades["token"].unique().tolist()
    db_p = replay_db_path(_CHART_DIR, date_str)
    
    conn = sqlite3.connect(db_p)
    try:
        open_ts, close_ts = ist_market_session_bounds(date_str)
        timelines = load_tick_timelines(conn, tokens, open_ts, close_ts)
    finally:
        conn.close()
        
    results = []
    for idx, row in df_trades.iterrows():
        ts = row["timestamp"]
        tok = str(row["token"])
        ltp = row["ltp"]
        delta = row["delta"]
        band = row["delta_band"]
        opt_type = row["option_type"]
        max_ret_5m = row["target_max_return_5m_pct"]
        min_ret_5m = row["target_min_return_5m_pct"]
        
        # Dynamic Target
        if ltp > 100.0:
            strat_tgt = 2.0
        elif ltp >= 50.0:
            strat_tgt = 3.0
        elif ltp >= 20.0:
            strat_tgt = 5.0
        else:
            strat_tgt = 10.0
        strat_sl = 5.0
        
        strat_tl = timelines.get(tok)
        outcome_return = 0.0
        outcome_type = "timeout"
        elapsed_sec = 300.0
        
        if strat_tl:
            outcome, elapsed_sec = check_scalp_outcome_seconds_with_time(strat_tl, ts, 300.0, strat_tgt, strat_sl)
            if outcome == 1:
                outcome_return = strat_tgt
                outcome_type = "target"
            elif outcome == -1:
                outcome_return = -strat_sl
                outcome_type = "sl"
            else:
                entry_p = strat_tl.ltp_rupees_at(ts)
                exit_p = strat_tl.ltp_rupees_at(ts + 300.0)
                if entry_p and exit_p and entry_p > 0:
                    outcome_return = float((exit_p - entry_p) / entry_p * 100.0)
                outcome_type = "timeout"
                
        # Premium Bucket
        if 5.0 <= ltp < 10.0:
            bucket = "5-10"
        elif 10.0 <= ltp < 15.0:
            bucket = "10-15"
        elif 15.0 <= ltp < 20.0:
            bucket = "15-20"
        elif 20.0 <= ltp < 30.0:
            bucket = "20-30"
        elif 30.0 <= ltp < 50.0:
            bucket = "30-50"
        elif ltp >= 50.0:
            bucket = "50-ATM"
        else:
            bucket = "under_5"
            
        results.append({
            "bucket": bucket,
            "ltp": ltp,
            "delta": delta,
            "band": band,
            "opt_type": opt_type,
            "outcome_return": outcome_return,
            "outcome_type": outcome_type,
            "elapsed_min": elapsed_sec / 60.0,
            "max_ret_5m": max_ret_5m,
            "min_ret_5m": min_ret_5m,
        })
        
    return results


def compile_bucket_report(results: list[dict[str, any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    buckets = ["5-10", "10-15", "15-20", "20-30", "30-50", "50-ATM"]
    compiled = []
    
    for b in buckets:
        df_b = df[df["bucket"] == b]
        total_trades = len(df_b)
        
        if total_trades == 0:
            compiled.append({
                "Bucket": b,
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Timeouts": 0,
                "Win Rate": "0.00%",
                "Avg Time to Target (min)": "0.00",
                "Median Time to Target (min)": "0.00",
                "Avg Max Return 5m": "0.00%",
                "Avg Min Return 5m": "0.00%",
                "Profit Factor": "0.0000",
                "Net Return": "0.00%",
                "Avg Delta": "0.00",
                "Delta Band (A/B/C)": "0 / 0 / 0",
                "CE Trades": 0,
                "PE Trades": 0,
            })
            continue
            
        wins = df_b[df_b["outcome_return"] > 0]
        losses = df_b[df_b["outcome_return"] <= 0]
        timeouts = df_b[df_b["outcome_type"] == "timeout"]
        
        win_rate = len(wins) / total_trades
        
        # Win or SL time to target
        active_exits = df_b[df_b["outcome_type"].isin(["target", "sl"])]
        if not active_exits.empty:
            avg_time = active_exits["elapsed_min"].mean()
            med_time = active_exits["elapsed_min"].median()
        else:
            avg_time = 5.0
            med_time = 5.0
            
        avg_max = df_b["max_ret_5m"].mean()
        avg_min = df_b["min_ret_5m"].mean()
        
        sum_gains = wins["outcome_return"].sum()
        sum_losses = abs(losses["outcome_return"].sum())
        profit_factor = sum_gains / sum_losses if sum_losses > 0 else float("inf")
        net_ret = df_b["outcome_return"].sum()
        
        avg_delta = df_b["delta"].mean()
        
        # Band distribution
        band_counts = df_b["band"].value_counts()
        band_str = f"{band_counts.get('A', 0)} / {band_counts.get('B', 0)} / {band_counts.get('C', 0)}"
        
        ce_count = (df_b["opt_type"] == "CE").sum()
        pe_count = (df_b["opt_type"] == "PE").sum()
        
        compiled.append({
            "Bucket": b,
            "Trades": total_trades,
            "Wins": len(wins),
            "Losses": len(losses),
            "Timeouts": len(timeouts),
            "Win Rate": f"{win_rate:.2%}",
            "Avg Time to Target (min)": f"{avg_time:.2f}",
            "Median Time to Target (min)": f"{med_time:.2f}",
            "Avg Max Return 5m": f"{avg_max:.2f}%",
            "Avg Min Return 5m": f"{avg_min:.2f}%",
            "Profit Factor": f"{profit_factor:.4f}" if profit_factor != float("inf") else "inf",
            "Net Return": f"{net_ret:+.2f}%",
            "Avg Delta": f"{avg_delta:.2f}",
            "Delta Band (A/B/C)": band_str,
            "CE Trades": ce_count,
            "PE Trades": pe_count,
        })
        
    return pd.DataFrame(compiled)


def to_markdown_custom(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = df.columns.tolist()
    header = " | ".join(cols)
    divider = " | ".join(["---"] * len(cols))
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(row[c]) for c in cols))
    return f"| {header} |\n| {divider} |\n" + "\n".join(f"| {r} |" for r in rows)


def main():
    print("Generating premium bucket report for Fold 1 (2026-06-22) and Fold 2 (2026-06-23)...")
    
    # 1. Fold 1
    fold1_res = run_bucket_analysis_for_date("2026-06-22", "fold_1_2026-06-22", 3.0)
    df_fold1 = compile_bucket_report(fold1_res)
    
    # 2. Fold 2
    fold2_res = run_bucket_analysis_for_date("2026-06-23", "fold_2_2026-06-23", 3.0)
    df_fold2 = compile_bucket_report(fold2_res)
    
    # Write output report to stdout
    print("\n" + "="*80)
    print(" FOLD 1 (2026-06-22) PREMIUM BUCKET REPORT ")
    print("="*80)
    print(to_markdown_custom(df_fold1))
    
    print("\n" + "="*80)
    print(" FOLD 2 (2026-06-23) PREMIUM BUCKET REPORT ")
    print("="*80)
    print(to_markdown_custom(df_fold2))
    
    # Profit/loss contribution analysis
    def get_analysis_str(results, date_name):
        if not results:
            return f"No trades for {date_name}."
        df = pd.DataFrame(results)
        # Net return per bucket
        net_ret_per_bucket = df.groupby("bucket")["outcome_return"].sum()
        
        # Max profit bucket
        best_bucket = net_ret_per_bucket.idxmax()
        best_val = net_ret_per_bucket.max()
        
        # Max loss bucket
        worst_bucket = net_ret_per_bucket.idxmin()
        worst_val = net_ret_per_bucket.min()
        
        return (
            f"**{date_name} Profit/Loss Analysis:**\n"
            f"* **Highest Profit Contributor**: Premium bucket `{best_bucket}` with a net return of `{best_val:+.2f}%`.\n"
            f"* **Highest Loss Contributor**: Premium bucket `{worst_bucket}` with a net return of `{worst_val:+.2f}%`.\n"
        )
        
    print("\n" + "="*80)
    print(" PROFIT / LOSS CONTRIBUTION ANALYSIS ")
    print("="*80)
    print(get_analysis_str(fold1_res, "Fold 1 (2026-06-22)"))
    print(get_analysis_str(fold2_res, "Fold 2 (2026-06-23)"))


if __name__ == "__main__":
    main()
