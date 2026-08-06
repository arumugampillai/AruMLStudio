#!/usr/bin/env python3
"""
Train XGBoost Classifier and Regressor on exported ATM ± 10 strikes feature datasets,
supporting strike distance separation (ATM ±3, ATM ±5, ATM ±10).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor

def focal_loss_obj(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom objective function for binary Focal Loss.
    y_true: True labels (0 or 1)
    y_pred: Raw margin scores (pre-sigmoid predictions)
    """
    gamma = 2.0
    alpha = 0.25
    # Convert margins to probabilities using sigmoid
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Calculate gradient and hessian
    p_t = p * y_true + (1 - p) * (1 - y_true)
    grad = -alpha * (1 - p_t)**gamma * (y_true - p)
    
    # Hessian: Second derivative approximation
    hess = alpha * (1 - p_t)**gamma * (gamma * (y_true - p)**2 / p_t + p * (1 - p))
    # Clip hessian to avoid negative or zero values which breaks tree splits
    hess = np.clip(hess, 1e-4, None)
    return grad, hess

# Add chart directory to sys.path
_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)

DEFAULT_MODEL_DIR = os.path.join(_CHART_DIR, "data", "ml_models")

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


def resolve_input_paths(raw_path: str) -> list[str]:
    paths = sorted(glob.glob(raw_path))
    if not paths and os.path.isfile(raw_path):
        paths = [raw_path]
    if not paths:
        raise FileNotFoundError(f"No files matched input path pattern: {raw_path}")
    return paths


def filter_by_delta_band(df: pd.DataFrame, band_name: str) -> pd.DataFrame:
    if "delta" not in df.columns:
        raise ValueError("Dataset must have a 'delta' column to filter by Delta bands.")
    
    abs_delta = df["delta"].abs()
    if band_name == "A":
        return df[(abs_delta >= 0.40) & (abs_delta <= 0.50)].copy()
    elif band_name == "B":
        return df[(abs_delta >= 0.25) & (abs_delta < 0.40)].copy()
    elif band_name == "C":
        return df[(abs_delta >= 0.15) & (abs_delta < 0.25)].copy()
    else:
        raise ValueError(f"Unknown delta band: {band_name}. Expected 'A', 'B', or 'C'.")


def split_train_val_data(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("Dataset must have a 'date' column for time-split validation.")
    
    dates = sorted(df["date"].astype(str).unique())
    if len(dates) < 2:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1.0 - val_frac))
        return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
    
    cut = max(1, int(len(dates) * (1.0 - val_frac)))
    train_dates = set(dates[:cut])
    val_dates = set(dates[cut:])
    
    train_df = df[df["date"].astype(str).isin(train_dates)].copy()
    val_df = df[df["date"].astype(str).isin(val_dates)].copy()
    return train_df, val_df


def train_and_eval_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    **xgb_kwargs: Any,
) -> tuple[XGBRegressor, dict[str, float]]:
    print(f"Training XGBoost Regressor with {len(X_train)} samples...")
    model = XGBRegressor(**xgb_kwargs)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)
    
    metrics = {
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_r2": float(r2),
    }
    return model, metrics


def train_and_eval_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    loss_type: str = "logloss",
    **xgb_kwargs: Any,
) -> tuple[XGBClassifier, dict[str, Any]]:
    print(f"Training XGBoost Binary Classifier with {len(X_train)} samples...")
    print(f"Loss function: {loss_type}")
    
    if loss_type == "focal":
        model = XGBClassifier(
            objective=focal_loss_obj,
            eval_metric="aucpr",
            **xgb_kwargs,
        )
    else:
        print(f"Using scale_pos_weight: {scale_pos_weight:.4f}")
        model = XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            **xgb_kwargs,
        )
        
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    
    # Calculate precision and recall specifically for the positive class (Hit (1))
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    
    # Class distribution
    class_1_pct = float(y_val.mean())
    class_0_pct = float(1.0 - class_1_pct)
    
    # Calculate precision and recall across thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    threshold_metrics = {}
    probs = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, probs)
    
    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        prec_t = precision_score(y_val, preds_t, zero_division=0)
        rec_t = recall_score(y_val, preds_t, zero_division=0)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, preds_t).ravel()
        threshold_metrics[f"{t:.2f}"] = {
            "precision": float(prec_t),
            "recall": float(rec_t),
            "tp": int(tp_t),
            "fp": int(fp_t),
            "tn": int(tn_t),
            "fn": int(fn_t),
        }
        
    report_dict = classification_report(
        y_val, preds, 
        target_names=["No Hit (0)", "Hit (1)"], 
        output_dict=True,
        zero_division=0,
    )
    
    metrics = {
        "val_accuracy": float(acc),
        "val_pr_auc": float(pr_auc),
        "val_precision": float(precision),
        "val_recall": float(recall),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "class_distribution": {
            "class_1": class_1_pct,
            "class_0": class_0_pct,
        },
        "threshold_metrics": threshold_metrics,
        "classification_report": report_dict,
    }
    return model, metrics


def run_training_for_delta_band(df: pd.DataFrame, band_name: str, stamp: str, args: argparse.Namespace) -> dict[str, Any]:
    print("\n" + "#"*60)
    print(f" TRAINING MODEL FOR DELTA BAND: {band_name} ")
    if band_name == "A":
        print(" Range: 0.40 <= abs(delta) <= 0.50 ")
    elif band_name == "B":
        print(" Range: 0.25 <= abs(delta) < 0.40 ")
    elif band_name == "C":
        print(" Range: 0.15 <= abs(delta) < 0.25 ")
    print("#"*60)

    # Filter data to the given Delta band
    df_band = filter_by_delta_band(df, band_name)
    print(f"Rows in Delta Band {band_name}: {len(df_band)} (Filtered from {len(df)})")

    # Select targets
    if args.target_type == "option":
        target_reg_max = "target_max_return_5m_pct"
        target_reg_min = "target_min_return_5m_pct"
        target_clf = "target_first_event_5m"
    else:
        target_reg_max = "target_spot_max_return_5m_pct"
        target_reg_min = "target_spot_min_return_5m_pct"
        target_clf = "target_spot_first_event_5m"

    # Clean up NaNs in targets
    df_reg_max = df_band.dropna(subset=[target_reg_max]).copy()
    df_reg_min = df_band.dropna(subset=[target_reg_min]).copy()
    df_clf = df_band.dropna(subset=[target_clf]).copy()

    # Split train/val
    train_reg_max_df, val_reg_max_df = split_train_val_data(df_reg_max, args.val_frac)
    train_reg_min_df, val_reg_min_df = split_train_val_data(df_reg_min, args.val_frac)
    train_clf_df, val_clf_df = split_train_val_data(df_clf, args.val_frac)

    X_train_reg_max = train_reg_max_df[FEATURE_COLUMNS]
    y_train_reg_max = train_reg_max_df[target_reg_max]
    X_val_reg_max = val_reg_max_df[FEATURE_COLUMNS]
    y_val_reg_max = val_reg_max_df[target_reg_max]

    X_train_reg_min = train_reg_min_df[FEATURE_COLUMNS]
    y_train_reg_min = train_reg_min_df[target_reg_min]
    X_val_reg_min = val_reg_min_df[FEATURE_COLUMNS]
    y_val_reg_min = val_reg_min_df[target_reg_min]

    X_train_clf = train_clf_df[FEATURE_COLUMNS]
    y_train_clf = (train_clf_df[target_clf] == 1).astype(int)
    X_val_clf = val_clf_df[FEATURE_COLUMNS]
    y_val_clf = (val_clf_df[target_clf] == 1).astype(int)

    # If validation sets are empty (e.g. val_frac = 0.0), fall back to using training data as validation eval_set
    if len(X_val_reg_max) == 0:
        X_val_reg_max, y_val_reg_max = X_train_reg_max, y_train_reg_max
    if len(X_val_reg_min) == 0:
        X_val_reg_min, y_val_reg_min = X_train_reg_min, y_train_reg_min
    if len(X_val_clf) == 0:
        X_val_clf, y_val_clf = X_train_clf, y_train_clf

    # Hyperparameters
    xgb_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Train Regressors
    print("Training Max Return Regressor...")
    reg_max_model, reg_max_metrics = train_and_eval_regressor(
        X_train_reg_max, y_train_reg_max, X_val_reg_max, y_val_reg_max, **xgb_kwargs
    )
    print("Training Min Return Regressor...")
    reg_min_model, reg_min_metrics = train_and_eval_regressor(
        X_train_reg_min, y_train_reg_min, X_val_reg_min, y_val_reg_min, **xgb_kwargs
    )

    # Calculate scale_pos_weight dynamically on training dataset targets
    pos_count = int((y_train_clf == 1).sum())
    neg_count = int((y_train_clf == 0).sum())
    scale_pos_weight = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
    print(f"Dynamic scale_pos_weight calculated: {scale_pos_weight:.4f} (Pos: {pos_count}, Neg: {neg_count})")

    # Train Classifier
    clf_model, clf_metrics = train_and_eval_classifier(
        X_train_clf, y_train_clf, X_val_clf, y_val_clf,
        scale_pos_weight=scale_pos_weight,
        loss_type=args.loss_type,
        **xgb_kwargs
    )

    # Model saving
    os.makedirs(args.out_dir, exist_ok=True)
    reg_max_path = os.path.join(args.out_dir, f"xgboost_reg_max_{args.target_type}_delta_{band_name}_{stamp}.json")
    reg_min_path = os.path.join(args.out_dir, f"xgboost_reg_min_{args.target_type}_delta_{band_name}_{stamp}.json")
    clf_path = os.path.join(args.out_dir, f"xgboost_clf_{args.target_type}_delta_{band_name}_{stamp}.json")
    report_path = os.path.join(args.out_dir, f"training_report_{args.target_type}_delta_{band_name}_{stamp}.json")

    reg_max_model.save_model(reg_max_path)
    reg_min_model.save_model(reg_min_path)
    clf_model.save_model(clf_path)

    # Feature importances
    reg_max_importances = dict(zip(FEATURE_COLUMNS, map(float, reg_max_model.feature_importances_)))
    reg_min_importances = dict(zip(FEATURE_COLUMNS, map(float, reg_min_model.feature_importances_)))
    clf_importances = dict(zip(FEATURE_COLUMNS, map(float, clf_model.feature_importances_)))
    sorted_reg_max_importances = sorted(reg_max_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_reg_min_importances = sorted(reg_min_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_clf_importances = sorted(clf_importances.items(), key=lambda x: x[1], reverse=True)

    report = {
        "delta_band": band_name,
        "samples": len(df_band),
        "model_paths": {
            "regressor_max": reg_max_path,
            "regressor_min": reg_min_path,
            "classifier": clf_path
        },
        "metrics": {
            "regressor_max": reg_max_metrics,
            "regressor_min": reg_min_metrics,
            "classifier": clf_metrics
        },
        "feature_importances": {
            "regressor_max": sorted_reg_max_importances,
            "regressor_min": sorted_reg_min_importances,
            "classifier": sorted_clf_importances,
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Regressor Max saved: {reg_max_path}")
    print(f"Regressor Min saved: {reg_min_path}")
    print(f"Classifier saved: {clf_path}")
    clf_m = clf_metrics
    cm = clf_m["confusion_matrix"]
    dist = clf_m["class_distribution"]
    print(f"Regression MAE (Max): {reg_max_metrics['val_mae']:.4f}% | (Min): {reg_min_metrics['val_mae']:.4f}% | Classification Accuracy: {clf_metrics['val_accuracy']:.2%} | PR-AUC: {clf_m['val_pr_auc']:.4f}")
    print(f"Classifier Validation Metrics:")
    print(f"  Precision (Class 1): {clf_m['val_precision']:.2%}")
    print(f"  Recall (Class 1):    {clf_m['val_recall']:.2%}")
    print(f"  Class Distribution (Val): Class 1 (Hit): {dist['class_1']:.2%}, Class 0 (No Hit): {dist['class_0']:.2%}")
    print(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
    print(f"  Probability Threshold Analysis:")
    print(f"    {'Threshold':<10} | {'Precision':<10} | {'Recall':<8} | {'TP / FP / TN / FN':<18}")
    print(f"    {'-'*55}")
    for t_str, t_metrics in clf_m["threshold_metrics"].items():
        prec_str = f"{t_metrics['precision']:.2%}"
        rec_str = f"{t_metrics['recall']:.2%}"
        tp_fp_tn_fn = f"{t_metrics['tp']} / {t_metrics['fp']} / {t_metrics['tn']} / {t_metrics['fn']}"
        print(f"    {t_str:<10} | {prec_str:<10} | {rec_str:<8} | {tp_fp_tn_fn:<18}")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost models on Delta bands data.")
    parser.add_argument("--data", required=True, help="CSV path or glob matching exported files")
    parser.add_argument("--target-type", choices=["option", "spot"], default="option", help="Target options or spot (default: option)")
    parser.add_argument("--out-dir", default=DEFAULT_MODEL_DIR, help="Directory to save trained models and reports")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of dates held out for validation (default: 0.2)")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--loss-type", choices=["logloss", "focal"], default="logloss", help="Loss function type for classifier (default: logloss)")
    parser.add_argument("--delta-band", default="all", choices=["A", "B", "C", "all"], help="Delta bands to train models on (default: all)")
    parser.add_argument("--stamp", default=None, help="Custom timestamp/identifier for model filenames (default: current date/time)")
    parser.add_argument("--strike-filter", choices=["none", "experiment1"], default="none", help="Strike filtering type (default: none)")
    
    args = parser.parse_args(argv)
    
    try:
        t0 = time.monotonic()
        paths = resolve_input_paths(args.data)
        print(f"Loading {len(paths)} CSV files...")
        dfs = [pd.read_csv(p) for p in paths]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded dataset with {len(df)} total rows.")

        if args.strike_filter == "experiment1":
            from chain_replay_ml.features_atm_band import filter_dataset_for_experiment_1
            print("Applying strike filter 'experiment1' (ATM + OTM strikes only, premium Rs. 10 to ATM premium)...")
            df = filter_dataset_for_experiment_1(df)
            print(f"Dataset size after strike filter: {len(df)} rows.")

        # If target-type is option, dynamically calculate target_direction_5m based on premium (LTP)
        if args.target_type == "option" and "ltp" in df.columns:
            print("\nApplying dynamic premium-based profit targets to normalize difficulty:")
            print("  Premium > 100   => target >= 2%")
            print("  Premium 50-100  => target >= 3%")
            print("  Premium 20-50   => target >= 5%")
            print("  Premium < 20    => target >= 10%")
            
            thresholds = pd.Series(10.0, index=df.index)
            thresholds.loc[df["ltp"] > 100.0] = 2.0
            thresholds.loc[(df["ltp"] >= 50.0) & (df["ltp"] <= 100.0)] = 3.0
            thresholds.loc[(df["ltp"] >= 20.0) & (df["ltp"] < 50.0)] = 5.0
            
            if "target_min_return_5m_pct" in df.columns:
                print("Applying option drawdown constraint: target_min_return_5m_pct > -5.0%")
                old_hits = (df["target_max_return_5m_pct"] >= thresholds).sum()
                df["target_direction_5m"] = (
                    (df["target_max_return_5m_pct"] >= thresholds) & 
                    (df["target_min_return_5m_pct"] > -5.0)
                ).astype(int)
                new_hits = df["target_direction_5m"].sum()
                print(f"Option Hits changed from {old_hits} to {new_hits} due to drawdown constraint (dropped {old_hits - new_hits} bad trades).")
            else:
                df["target_direction_5m"] = (df["target_max_return_5m_pct"] >= thresholds).astype(int)
        
        # Verify feature columns exist in dataset
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from input data: {missing}")

        bands_to_train = ["A", "B", "C"] if args.delta_band == "all" else [args.delta_band]
        results = []
        stamp = args.stamp or time.strftime("%Y%m%d_%H%M%S")
        for band in bands_to_train:
            band_report = run_training_for_delta_band(df, band, stamp, args)
            results.append(band_report)

        # Print comparison table at the end
        print("\n" + "="*136)
        print(f" MODEL PERFORMANCE COMPARISON ({args.target_type.upper()} TARGETS - {args.loss_type.upper()} LOSS) ")
        print("="*136)
        print(f"{'Band':<12} | {'Rows':<8} | {'Max MAE':<9} | {'Min MAE':<9} | {'Clf Acc':<8} | {'Clf PR-AUC':<10} | {'Class 1%':<9} | {'Precision':<10} | {'Recall':<8} | {'TP/FP/TN/FN':<18}")
        print("-"*136)
        for r in results:
            band_name = f"Delta Band {r['delta_band']}"
            rows = r["samples"]
            reg_max = r["metrics"]["regressor_max"]
            reg_min = r["metrics"]["regressor_min"]
            clf_m = r["metrics"]["classifier"]
            mae_max = f"{reg_max['val_mae']:.4f}%"
            mae_min = f"{reg_min['val_mae']:.4f}%"
            acc = f"{clf_m['val_accuracy']:.2%}"
            pr_auc = f"{clf_m['val_pr_auc']:.4f}"
            c1_pct = f"{clf_m['class_distribution']['class_1']:.2%}"
            prec = f"{clf_m['val_precision']:.2%}"
            rec = f"{clf_m['val_recall']:.2%}"
            cm = clf_m["confusion_matrix"]
            cm_str = f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}"
            print(f"{band_name:<12} | {rows:<8} | {mae_max:<9} | {mae_min:<9} | {acc:<8} | {pr_auc:<10} | {c1_pct:<9} | {prec:<10} | {rec:<8} | {cm_str:<18}")
        print("="*136)

        print(f"\nCompleted all delta band models training in {time.monotonic() - t0:.2f}s")
        return 0
    except Exception as e:
        print(f"Error training models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
