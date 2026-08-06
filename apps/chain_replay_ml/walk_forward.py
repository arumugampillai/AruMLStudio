#!/usr/bin/env python3
"""
Rolling walk-forward tests for ATM options ML model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import pandas as pd
import numpy as np

# Add chart directory to sys.path
from path_config import CHART_DATA_ROOT as _CHART_DIR
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()
from chain_replay_ml.export_atm_pipeline import export_atm_features
from chain_replay_ml.train_atm_model import main as train_main
from chain_replay_ml.backtest_ranking import main as backtest_main


def discover_valid_days() -> list[str]:
    """Find all valid daily database files and return sorted date strings."""
    data_dir = os.path.join(_CHART_DIR, "data")
    db_pattern = os.path.join(data_dir, "angel_market_*.db")
    db_files = sorted(glob.glob(db_pattern))
    
    valid_dates = []
    for db_file in db_files:
        size_mb = os.path.getsize(db_file) / (1024 * 1024)
        if size_mb < 10.0:
            continue
            
        basename = os.path.basename(db_file)
        date_str = basename.replace("angel_market_", "").replace(".db", "")
        
        # Verify if NIFTY option metadata exists
        import sqlite3
        conn = sqlite3.connect(db_file)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT expiry_date 
                FROM token_day_meta 
                WHERE expiry_date IS NOT NULL 
                  AND expiry_date >= ?
                  AND name = 'NIFTY'
                  AND exchange = 'NFO'
                LIMIT 1
            """, (date_str,))
            row = cursor.fetchone()
            if row:
                valid_dates.append(date_str)
        except Exception as e:
            print(f"Error checking metadata in {basename}: {e}")
        finally:
            conn.close()
            
    return sorted(valid_dates)


def verify_or_export_features(dates: list[str]) -> list[str]:
    """Ensure that feature CSV exists for each date. Export if missing."""
    out_dir = os.path.join(_CHART_DIR, "data", "ml_features", "atm_band_exports")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_paths = []
    for date_str in dates:
        csv_path = os.path.join(out_dir, f"atm_features_NIFTY_{date_str}.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
            csv_paths.append(csv_path)
            continue
            
        print(f"\nFeature file missing for {date_str}. Generating it now...")
        # Resolve nearest expiry date
        import sqlite3
        db_path = os.path.join(_CHART_DIR, "data", f"angel_market_{date_str}.db")
        conn = sqlite3.connect(db_path)
        expiry_date = None
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT expiry_date 
                FROM token_day_meta 
                WHERE expiry_date IS NOT NULL 
                  AND expiry_date >= ?
                  AND name = 'NIFTY'
                  AND exchange = 'NFO'
                ORDER BY expiry_date ASC
                LIMIT 1
            """, (date_str,))
            row = cursor.fetchone()
            if row:
                expiry_date = row[0]
        finally:
            conn.close()
            
        if not expiry_date:
            raise ValueError(f"Could not resolve NIFTY option expiry for date {date_str}")
            
        print(f"Exporting features for {date_str} (Expiry: {expiry_date})...")
        exported_path = export_atm_features(
            chart_dir=_CHART_DIR,
            underlying="NIFTY",
            expiry=expiry_date,
            date=date_str,
            step_sec=10,
            direction_threshold_pct=5.0,
            out_dir=out_dir,
        )
        csv_paths.append(exported_path)
        
    return csv_paths


def calculate_max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cum_returns = np.cumsum(returns)
    peak = -999999.0
    max_dd = 0.0
    for val in cum_returns:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


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


def generate_expectancy_breakdown(
    trades: list[dict[str, any]],
    group_col_func: any,
    group_labels: list[str],
    title_label: str,
) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
        
    df = pd.DataFrame(trades)
    if "premium_bucket" not in df.columns:
        df["premium_bucket"] = df["ltp"].apply(lambda x: "5-10" if 5.0 <= x < 10.0 else ("10-15" if 10.0 <= x < 15.0 else ("15-20" if 15.0 <= x < 20.0 else ("20-30" if 20.0 <= x < 30.0 else ("30-50" if 30.0 <= x < 50.0 else ("50-ATM" if x >= 50.0 else "under_5"))))))
        
    compiled = []
    for label in group_labels:
        if isinstance(group_col_func, str):
            df_g = df[df[group_col_func] == label]
        else:
            df_g = df[df.apply(group_col_func, axis=1) == label]
            
        total_trades = len(df_g)
        if total_trades == 0:
            compiled.append({
                title_label: label, "Trades": 0, "Wins": 0, "Losses": 0, "Timeouts": 0,
                "Win Rate": "0.00%", "Avg Time (min)": "0.00", "Avg Max Return": "0.00%",
                "Net Return": "0.00%", "Max DD": "0.00%", "Profit Factor": "0.0000"
            })
            continue
            
        wins = df_g[df_g["outcome_return"] > 0]
        losses = df_g[df_g["outcome_return"] <= 0]
        timeouts = df_g[df_g["outcome_type"] == "timeout"]
        
        win_rate = len(wins) / total_trades
        avg_time = df_g["elapsed_min"].mean()
        avg_max = df_g["max_return"].mean()
        net_ret = df_g["outcome_return"].sum()
        
        # Max Drawdown (chronological)
        sorted_returns = df_g.sort_values("timestamp")["outcome_return"].tolist()
        max_dd = calculate_max_drawdown(sorted_returns)
        
        # Profit Factor
        sum_gains = wins["outcome_return"].sum()
        sum_losses = abs(losses["outcome_return"].sum())
        pf = sum_gains / sum_losses if sum_losses > 0 else float("inf")
        pf_str = f"{pf:.4f}" if pf != float("inf") else "inf"
        
        compiled.append({
            title_label: label,
            "Trades": total_trades,
            "Wins": len(wins),
            "Losses": len(losses),
            "Timeouts": len(timeouts),
            "Win Rate": f"{win_rate:.2%}",
            "Avg Time (min)": f"{avg_time:.2f}",
            "Avg Max Return": f"{avg_max:.2f}%",
            "Net Return": f"{net_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
            "Profit Factor": pf_str,
        })
        
    return pd.DataFrame(compiled)


def main():
    parser = argparse.ArgumentParser(description="Rolling walk-forward tests.")
    parser.add_argument("--loss-type", choices=["logloss", "focal"], default="logloss", help="Model loss type (default: logloss)")
    parser.add_argument("--score-thresholds", default="0.0,2.0,3.0", help="Comma-separated score thresholds (default: 0.0,2.0,3.0)")
    parser.add_argument("--min-train-days", type=int, default=3, help="Minimum number of training days for the first fold")
    parser.add_argument("--strike-filter", choices=["none", "experiment1"], default="none", help="Strike filtering type (default: none)")
    
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.score_thresholds.split(",") if x.strip()]
    
    print("=" * 70)
    print(" ROLLING WALK-FORWARD VALIDATION ")
    print("=" * 70)
    
    dates = discover_valid_days()
    print(f"Discovered {len(dates)} valid trading days with NIFTY options metadata:")
    for d in dates:
        print(f"  - {d}")
        
    if len(dates) <= args.min_train_days:
        print(f"Error: Need at least {args.min_train_days + 1} dates for walk-forward, but only found {len(dates)}.")
        return 1
        
    print("\nEnsuring all feature CSV files are exported...")
    csv_paths = verify_or_export_features(dates)
    
    out_dir = os.path.join(_CHART_DIR, "data", "ml_features", "atm_band_exports")
    models_dir = os.path.join(_CHART_DIR, "data", "ml_models")
    
    # Store results
    # dict: fold_number -> { val_date, thresh_results: { thresh -> metrics } }
    fold_results = {}
    
    for i in range(args.min_train_days, len(dates)):
        fold_idx = i - args.min_train_days + 1
        train_dates = dates[:i]
        val_date = dates[i]
        
        print("\n" + "#" * 70)
        print(f" STARTING WALK-FORWARD FOLD {fold_idx}")
        print(f"   Train Dates:      {', '.join(train_dates)}")
        print(f"   Validation Date:  {val_date}")
        print("#" * 70)
        
        # 1. Concatenate training feature files
        train_dfs = []
        for d in train_dates:
            path = os.path.join(out_dir, f"atm_features_NIFTY_{d}.csv")
            train_dfs.append(pd.read_csv(path))
        df_train = pd.concat(train_dfs, ignore_index=True)
        
        temp_train_path = os.path.join(out_dir, f"temp_train_fold_{fold_idx}.csv")
        df_train.to_csv(temp_train_path, index=False)
        print(f"Concatenated training data for Fold {fold_idx}: {len(df_train)} rows.")
        
        # 2. Train model
        stamp = f"fold_{fold_idx}_{val_date}"
        print(f"Training models for Fold {fold_idx} with stamp: {stamp}...")
        train_args = [
            "--data", temp_train_path,
            "--target-type", "option",
            "--val-frac", "0.0",  # Train on 100% of training dates
            "--loss-type", args.loss_type,
            "--stamp", stamp,
            "--n-estimators", "500",
            "--max-depth", "5",
            "--learning-rate", "0.03",
            "--strike-filter", args.strike_filter,
        ]
        
        try:
            train_main(train_args)
        finally:
            # Clean up temp train file to save disk space
            if os.path.exists(temp_train_path):
                os.remove(temp_train_path)
                
        # 3. Backtest on validation date for each score threshold
        val_csv_path = os.path.join(out_dir, f"atm_features_NIFTY_{val_date}.csv")
        fold_results[fold_idx] = {
            "val_date": val_date,
            "train_dates": train_dates,
            "train_rows": int(len(df_train)),
            "thresholds": {},
        }
        
        for t in thresholds:
            report_path = os.path.join(out_dir, f"temp_report_fold_{fold_idx}_t_{t:.1f}.json")
            backtest_args = [
                "--data", val_csv_path,
                "--model-dir", models_dir,
                "--stamp", stamp,
                "--score-threshold", str(t),
                "--backtest-date", val_date,
                "--report-json", report_path,
                "--strike-filter", args.strike_filter,
            ]
            
            print(f"Backtesting Fold {fold_idx} on {val_date} at threshold {t}...")
            try:
                backtest_main(backtest_args)
                
                # Read result metrics
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        metrics_data = json.load(f)
                    fold_results[fold_idx]["thresholds"][t] = metrics_data
                    if abs(t - 3.0) < 1e-5:
                        fold_results[fold_idx]["trades"] = metrics_data.get("trades", [])
                else:
                    print(f"Warning: Backtest report JSON not found for threshold {t}.")
            except Exception as e:
                print(f"Error backtesting threshold {t}: {e}")
            finally:
                if os.path.exists(report_path):
                    os.remove(report_path)

    # 4. Print summary report
    print("\n" + "=" * 120)
    print(f" WALK-FORWARD VALIDATION SUMMARY REPORT ({args.loss_type.upper()} LOSS - STRIKE FILTER: {args.strike_filter.upper()}) ")
    print("=" * 120)
    
    headers = [
        "Fold", "Val Date", "Thresh", "Trades", "Win Rate", 
        "Net Return", "Max DD", "Profit Factor", "ATM Win Rate", "ATM Return"
    ]
    print(f"{headers[0]:<5} | {headers[1]:<10} | {headers[2]:<6} | {headers[3]:<6} | {headers[4]:<8} | {headers[5]:<10} | {headers[6]:<8} | {headers[7]:<13} | {headers[8]:<12} | {headers[9]:<10}")
    print("-" * 120)
    
    rows_for_artifact = []
    
    for fold_idx, f_info in sorted(fold_results.items()):
        val_date = f_info["val_date"]
        for t, m_data in sorted(f_info["thresholds"].items()):
            strat = m_data.get("strat", {})
            atm = m_data.get("atm", {})
            
            if not strat:
                continue
                
            trades = strat.get("total_trades", 0)
            win_rate = f"{strat.get('win_pct', 0.0):.2%}"
            net_ret = f"{strat.get('net_return', 0.0):+.2f}%"
            max_dd = f"{strat.get('max_drawdown', 0.0):.2f}%"
            pf = f"{strat.get('profit_factor', 0.0):.4f}"
            
            atm_win = f"{atm.get('win_pct', 0.0):.2%}"
            atm_ret = f"{atm.get('net_return', 0.0):+.2f}%"
            
            print(f"{fold_idx:<5} | {val_date:<10} | {t:<6.1f} | {trades:<6d} | {win_rate:<8} | {net_ret:<10} | {max_dd:<8} | {pf:<13} | {atm_win:<12} | {atm_ret:<10}")
            
            rows_for_artifact.append({
                "Fold": fold_idx,
                "Val Date": val_date,
                "Threshold": t,
                "Trades": trades,
                "Win Rate": win_rate,
                "Net Return": net_ret,
                "Max DD": max_dd,
                "Profit Factor": pf,
                "ATM Win": atm_win,
                "ATM Return": atm_ret,
            })
            
    print("=" * 120)
    
    # 5. Print breakdown reports for threshold 3.0
    for fold_idx, f_info in sorted(fold_results.items()):
        val_date = f_info["val_date"]
        trades_list = f_info.get("trades", [])
        
        if not trades_list:
            continue
            
        print("\n" + "=" * 90)
        print(f" FOLD {fold_idx} ({val_date}) POST-BACKTEST DETAILED ANALYSIS (THRESHOLD 3.0) ")
        print("=" * 90)
        
        # A. Premium Bucket Breakdown
        print("\nPremium Bucket Analysis:")
        bucket_labels = ["5-10", "10-15", "15-20", "20-30", "30-50", "50-ATM"]
        df_bucket = generate_expectancy_breakdown(trades_list, "premium_bucket", bucket_labels, "Premium Bucket")
        print(to_markdown_custom(df_bucket))
        
        # B. CE vs PE Breakdown
        print("\nCE vs PE Analysis:")
        df_side = generate_expectancy_breakdown(trades_list, "option_type", ["CE", "PE"], "Option Side")
        print(to_markdown_custom(df_side))
        
        # C. Combined Breakdown
        print("\nCombined Breakdown (Side + Premium Bucket):")
        combined_labels = [f"{side} {bucket}" for side in ["CE", "PE"] for bucket in bucket_labels]
        combined_func = lambda row: f"{row['option_type']} {row['premium_bucket']}"
        df_comb = generate_expectancy_breakdown(trades_list, combined_func, combined_labels, "Option Side + Bucket")
        print(to_markdown_custom(df_comb))
        print("=" * 90)
        
    # Save the walk-forward results report to a JSON file in data/
    report_out_path = os.path.join(_CHART_DIR, "data", f"walk_forward_report_{args.loss_type}_sf_{args.strike_filter}_{int(time.time())}.json")
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved walk-forward results to {report_out_path}")
    
    return 0


if __name__ == "__main__":
    main()
