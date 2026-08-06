#!/usr/bin/env python3
"""
Automate feature export for all valid database days, concatenate them, and train XGBoost models.
"""

from __future__ import annotations

import os
import glob
import sqlite3
import sys
import time
import pandas as pd

# Add chart directory to sys.path
from path_config import CHART_DATA_ROOT as _CHART_DIR
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()
from chain_replay_ml.export_atm_pipeline import export_atm_features
from chain_replay_ml.train_atm_model import main as train_main


def resolve_nearest_expiry(db_path: str, date_str: str) -> str | None:
    conn = sqlite3.connect(db_path)
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
        """, (date_str,))
        rows = cursor.fetchall()
        if rows:
            return rows[0][0]
    except Exception as e:
        print(f"Error querying {db_path}: {e}")
    finally:
        conn.close()
    return None


def run_pipeline_for_all_days():
    t_start = time.monotonic()
    data_dir = os.path.join(_CHART_DIR, "data")
    db_pattern = os.path.join(data_dir, "angel_market_*.db")
    db_files = sorted(glob.glob(db_pattern))
    
    valid_configs = []
    for db_file in db_files:
        size_mb = os.path.getsize(db_file) / (1024 * 1024)
        if size_mb < 10.0:
            continue
            
        basename = os.path.basename(db_file)
        date_str = basename.replace("angel_market_", "").replace(".db", "")
        
        expiry_date = resolve_nearest_expiry(db_file, date_str)
        if expiry_date:
            valid_configs.append((date_str, expiry_date))
            print(f"Found active trading day: {date_str} (NIFTY option expiry: {expiry_date})")
        else:
            print(f"Skipping {date_str}: No NIFTY options found in metadata.")

    if not valid_configs:
        print("No valid daily database files found.")
        return 1

    out_dir = os.path.join(data_dir, "ml_features", "atm_band_exports")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_paths = []
    for date_str, expiry_date in valid_configs:
        print("\n" + "="*60)
        print(f"EXPORTING DATE: {date_str} (Expiry: {expiry_date})")
        print("="*60)
        
        try:
            csv_path = export_atm_features(
                chart_dir=_CHART_DIR,
                underlying="NIFTY",
                expiry=expiry_date,
                date=date_str,
                step_sec=10,
                direction_threshold_pct=5.0,
                out_dir=out_dir,
            )
            csv_paths.append(csv_path)
        except Exception as e:
            print(f"Failed to export data for {date_str}: {e}")
            import traceback
            traceback.print_exc()

    if not csv_paths:
        print("No CSV files were successfully exported.")
        return 1

    # Concatenate all CSVs
    print("\n" + "="*60)
    print("CONCATENATING ALL EXPORTED CSVs...")
    print("="*60)
    
    dfs = []
    for path in csv_paths:
        print(f"Loading {path}...")
        dfs.append(pd.read_csv(path))
        
    df_all = pd.concat(dfs, ignore_index=True)
    concatenated_path = os.path.join(out_dir, "atm_features_NIFTY_all_days.csv")
    print(f"Saving concatenated dataset with {len(df_all)} rows to: {concatenated_path}")
    df_all.to_csv(concatenated_path, index=False)
    
    print("\n" + "="*60)
    print("TRAINING MODELS ON CONCATENATED DATASET...")
    print("="*60)
    
    train_args = [
        "--data", concatenated_path,
        "--target-type", "option",
        "--val-frac", "0.2"
    ]
    train_main(train_args)
    
    print(f"\nAll days export and training finished in {time.monotonic() - t_start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline_for_all_days())
