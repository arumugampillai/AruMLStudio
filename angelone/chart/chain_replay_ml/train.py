"""Train XGBoost regressor on Phase 1 feature exports."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


from .constants import (
    DEFAULT_TARGET,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_SIDE,
    OPTION_SIDES,
    SUPPORTED_TARGETS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.dirname(HERE)
DEFAULT_MODEL_DIR = os.path.join(CHART_DIR, "data", "ml_models")


def _resolve_input_paths(raw: str) -> list[str]:
    paths = sorted(glob.glob(raw))
    if not paths and os.path.isfile(raw):
        paths = [raw]
    if not paths:
        raise FileNotFoundError(f"No input files matched: {raw}")
    return paths


def load_training_frame(
    paths: list[str],
    *,
    skip_warmup: bool = True,
) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if skip_warmup and "warmup_row" in df.columns:
        df = df[df["warmup_row"].fillna(0).astype(int) == 0]
    return df.reset_index(drop=True)


def split_by_date(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("Dataset needs a date column for time-ordered split")
    dates = sorted(df["date"].astype(str).unique())
    if len(dates) < 2:
        raise ValueError("Need at least 2 trading days for train/val split")
    cut = max(1, int(len(dates) * (1.0 - val_frac)))
    train_dates = set(dates[:cut])
    val_dates = set(dates[cut:])
    train_df = df[df["date"].astype(str).isin(train_dates)].copy()
    val_df = df[df["date"].astype(str).isin(val_dates)].copy()
    return train_df, val_df


def split_train_val(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("Dataset needs a date column for split")
    dates = sorted(df["date"].astype(str).unique())
    if len(dates) >= 2:
        return split_by_date(df, val_frac=val_frac)

    work = df.copy()
    if "minute_of_day" in work.columns:
        work = work.sort_values("minute_of_day")
    cut = max(1, int(len(work) * (1.0 - val_frac)))
    return work.iloc[:cut].copy(), work.iloc[cut:].copy()


def filter_option_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if side not in OPTION_SIDES:
        raise ValueError(f"side must be one of {OPTION_SIDES}")
    if "is_call" not in df.columns:
        raise ValueError("Dataset needs is_call column (+1 CE, -1 PE)")
    want = 1 if side == "CE" else -1
    return df[df["is_call"].fillna(0).astype(int) == want].copy()


def prepare_xy(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cols = feature_columns or FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    work = df.dropna(subset=[target]).copy()
    X = work[cols].copy()
    y = work[target].astype(float)
    if target.startswith("hit_"):
        y = (y == 1.0).astype(int)
    return X, y, work


def _reconstructed_ltp_mae(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target: str,
) -> float | None:
    if target in ("residual_pct_5m", "residual_5m"):
        if "bs_reiv_pred" not in df.columns or "actual_ltp_t_plus_5m" not in df.columns:
            return None
        bs = df["bs_reiv_pred"].astype(float).to_numpy()
        actual = df["actual_ltp_t_plus_5m"].astype(float).to_numpy()
        if target == "residual_pct_5m":
            final_ltp = bs * (1.0 + y_pred / 100.0)
        else:
            final_ltp = bs + y_pred
        mask = np.isfinite(bs) & np.isfinite(actual) & (bs > 0)
        if not mask.any():
            return None
        return float(mean_absolute_error(actual[mask], final_ltp[mask]))
    elif target in ("mfe_pct_10m", "mae_pct_10m"):
        if "ltp" not in df.columns:
            return None
        ltp = df["ltp"].astype(float).to_numpy()
        if target == "mfe_pct_10m":
            pred_price = ltp * (1.0 + y_pred / 100.0)
            true_price = ltp * (1.0 + y_true / 100.0)
        else:
            pred_price = ltp * (1.0 - y_pred / 100.0)
            true_price = ltp * (1.0 - y_true / 100.0)
        mask = np.isfinite(ltp) & np.isfinite(y_true) & np.isfinite(y_pred)
        if not mask.any():
            return None
        return float(mean_absolute_error(true_price[mask], pred_price[mask]))
    else:
        return None


def evaluate_split(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target: str,
) -> dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "final_ltp_mae": _reconstructed_ltp_mae(df, y_true, y_pred, target=target),
    }
    if target.startswith("hit_"):
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            metrics["auc"] = None
        try:
            metrics["logloss"] = float(log_loss(y_true, y_pred, labels=[0, 1]))
        except ValueError:
            metrics["logloss"] = None
    return metrics


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_columns: list[str] | None = None,
    model_kind: str = "combined",
    side: str | None = None,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"target must be one of {SUPPORTED_TARGETS}")

    feat_cols = feature_columns or FEATURE_COLUMNS
    X_train, y_train, train_eval_df = prepare_xy(train_df, target=target, feature_columns=feat_cols)
    X_val, y_val, val_eval_df = prepare_xy(val_df, target=target, feature_columns=feat_cols)

    is_classifier = target.startswith("hit_")
    if is_classifier:
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=6,
            tree_method="hist",
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=6,
            tree_method="hist",
            early_stopping_rounds=early_stopping_rounds,
        )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if is_classifier:
        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]
    else:
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

    importance = sorted(
        zip(feat_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    report: dict[str, Any] = {
        "target": target,
        "model_kind": model_kind,
        "side": side,
        "n_features": len(X_train.columns),
        "feature_columns": list(X_train.columns),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_dates": sorted(train_df["date"].astype(str).unique().tolist()) if "date" in train_df else [],
        "val_dates": sorted(val_df["date"].astype(str).unique().tolist()) if "date" in val_df else [],
        "metrics": {
            "train": evaluate_split(train_eval_df, y_train.to_numpy(), train_pred, target=target),
            "val": evaluate_split(val_eval_df, y_val.to_numpy(), val_pred, target=target),
        },
        "feature_importance_top15": [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in importance[:15]
        ],
        "best_iteration": int(getattr(model, "best_iteration", n_estimators)),
    }
    return model, report


def save_artifacts(
    model: Any,
    report: dict[str, Any],
    *,
    out_dir: str,
    model_name: str,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}.json")
    report_path = os.path.join(out_dir, f"{model_name}_report.json")
    model.save_model(model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {"model": model_path, "report": report_path}


def default_model_name(target: str, algorithm: str = "xgboost") -> str:
    from chain_replay_ml.training.naming import suggest_model_name

    return suggest_model_name(target, algorithm)


def run_training(
    *,
    data: str,
    target: str = DEFAULT_TARGET,
    out_dir: str = DEFAULT_MODEL_DIR,
    model_name: str | None = None,
    val_frac: float = 0.2,
    skip_warmup: bool = True,
    side: str | None = None,
    drop_is_call: bool = False,
    save_model: bool = True,
    drop_features: list[str] | None = None,
    **model_kwargs: Any,
) -> dict[str, Any]:
    paths = _resolve_input_paths(data)
    df = load_training_frame(paths, skip_warmup=skip_warmup)
    train_df, val_df = split_train_val(df, val_frac=val_frac)

    feat_cols = FEATURE_COLUMNS
    model_kind = "combined"
    if side is not None:
        train_df = filter_option_side(train_df, side)
        val_df = filter_option_side(val_df, side)
        feat_cols = FEATURE_COLUMNS_SIDE
        model_kind = side.lower()
        drop_is_call = True
    elif drop_is_call:
        feat_cols = FEATURE_COLUMNS_SIDE

    if drop_features:
        feat_cols = [c for c in feat_cols if c not in drop_features]

    model, report = train_model(
        train_df,
        val_df,
        target=target,
        feature_columns=feat_cols,
        model_kind=model_kind,
        side=side,
        **model_kwargs,
    )
    report["inputs"] = paths
    if save_model:
        suffix = f"_{side.lower()}" if side else ""
        name = model_name or default_model_name(f"{target}{suffix}")
        report["artifacts"] = save_artifacts(model, report, out_dir=out_dir, model_name=name)
    return report


def _predict_with_model(
    model: Any,
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, work = prepare_xy(df, target=target, feature_columns=feature_columns)
    if target.startswith("hit_"):
        pred = model.predict_proba(X)[:, 1]
    else:
        pred = model.predict(X)
    return y.to_numpy(), pred, work


def run_side_comparison(
    *,
    data: str,
    target: str = DEFAULT_TARGET,
    out_dir: str = DEFAULT_MODEL_DIR,
    model_name: str | None = None,
    val_frac: float = 0.2,
    skip_warmup: bool = True,
    **model_kwargs: Any,
) -> dict[str, Any]:
    paths = _resolve_input_paths(data)
    df = load_training_frame(paths, skip_warmup=skip_warmup)
    train_df, val_df = split_train_val(df, val_frac=val_frac)

    combined_model, combined_report = train_model(
        train_df,
        val_df,
        target=target,
        feature_columns=FEATURE_COLUMNS,
        model_kind="combined",
        **model_kwargs,
    )

    ce_train = filter_option_side(train_df, "CE")
    ce_val = filter_option_side(val_df, "CE")
    pe_train = filter_option_side(train_df, "PE")
    pe_val = filter_option_side(val_df, "PE")

    ce_model, ce_report = train_model(
        ce_train,
        ce_val,
        target=target,
        feature_columns=FEATURE_COLUMNS_SIDE,
        model_kind="ce_only",
        side="CE",
        **model_kwargs,
    )
    pe_model, pe_report = train_model(
        pe_train,
        pe_val,
        target=target,
        feature_columns=FEATURE_COLUMNS_SIDE,
        model_kind="pe_only",
        side="PE",
        **model_kwargs,
    )

    y_ce, pred_ce, ce_val_work = _predict_with_model(
        ce_model, ce_val, target=target, feature_columns=FEATURE_COLUMNS_SIDE,
    )
    y_pe, pred_pe, pe_val_work = _predict_with_model(
        pe_model, pe_val, target=target, feature_columns=FEATURE_COLUMNS_SIDE,
    )

    split_val_work = pd.concat([ce_val_work, pe_val_work], ignore_index=True)
    y_split = np.concatenate([y_ce, y_pe])
    pred_split = np.concatenate([pred_ce, pred_pe])

    comparison = {
        "target": target,
        "inputs": paths,
        "train_rows": {
            "combined": combined_report["train_rows"],
            "ce": ce_report["train_rows"],
            "pe": pe_report["train_rows"],
        },
        "val_rows": {
            "combined": combined_report["val_rows"],
            "ce": ce_report["val_rows"],
            "pe": pe_report["val_rows"],
            "split_blended": int(len(y_split)),
        },
        "val_metrics": {
            "combined_with_is_call": combined_report["metrics"]["val"],
            "ce_only": ce_report["metrics"]["val"],
            "pe_only": pe_report["metrics"]["val"],
            "split_blended": evaluate_split(split_val_work, y_split, pred_split, target=target),
        },
        "models": {
            "combined": combined_report,
            "ce": ce_report,
            "pe": pe_report,
        },
    }

    stamp = model_name or default_model_name(f"{target}_side_compare")
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, f"{stamp}_combined.json")
    ce_path = os.path.join(out_dir, f"{stamp}_ce.json")
    pe_path = os.path.join(out_dir, f"{stamp}_pe.json")
    report_path = os.path.join(out_dir, f"{stamp}_comparison.json")
    combined_model.save_model(combined_path)
    ce_model.save_model(ce_path)
    pe_model.save_model(pe_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    comparison["artifacts"] = {
        "combined_model": combined_path,
        "ce_model": ce_path,
        "pe_model": pe_path,
        "comparison_report": report_path,
    }
    return comparison


def run_split_blended_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    target: str,
    **model_kwargs: Any,
) -> tuple[dict[str, Any], Any, Any]:
    ce_train = filter_option_side(train_df, "CE")
    ce_val = filter_option_side(val_df, "CE")
    pe_train = filter_option_side(train_df, "PE")
    pe_val = filter_option_side(val_df, "PE")

    ce_model, ce_report = train_model(
        ce_train,
        ce_val,
        target=target,
        feature_columns=FEATURE_COLUMNS_SIDE,
        model_kind="ce_only",
        side="CE",
        **model_kwargs,
    )
    pe_model, pe_report = train_model(
        pe_train,
        pe_val,
        target=target,
        feature_columns=FEATURE_COLUMNS_SIDE,
        model_kind="pe_only",
        side="PE",
        **model_kwargs,
    )

    y_ce, pred_ce, ce_val_work = _predict_with_model(
        ce_model, ce_val, target=target, feature_columns=FEATURE_COLUMNS_SIDE,
    )
    y_pe, pred_pe, pe_val_work = _predict_with_model(
        pe_model, pe_val, target=target, feature_columns=FEATURE_COLUMNS_SIDE,
    )

    split_val_work = pd.concat([ce_val_work, pe_val_work], ignore_index=True)
    y_split = np.concatenate([y_ce, y_pe])
    pred_split = np.concatenate([pred_ce, pred_pe])

    report = {
        "target": target,
        "train_rows": {"ce": ce_report["train_rows"], "pe": pe_report["train_rows"]},
        "val_rows": {"ce": ce_report["val_rows"], "pe": pe_report["val_rows"], "total": int(len(y_split))},
        "val_metrics": {
            "ce_only": ce_report["metrics"]["val"],
            "pe_only": pe_report["metrics"]["val"],
            "split_blended": evaluate_split(split_val_work, y_split, pred_split, target=target),
        },
        "ce": ce_report,
        "pe": pe_report,
    }
    return report, ce_model, pe_model


def run_target_comparison(
    *,
    data: str,
    out_dir: str = DEFAULT_MODEL_DIR,
    model_name: str | None = None,
    val_frac: float = 0.2,
    skip_warmup: bool = True,
    **model_kwargs: Any,
) -> dict[str, Any]:
    paths = _resolve_input_paths(data)
    df = load_training_frame(paths, skip_warmup=skip_warmup)
    train_df, val_df = split_train_val(df, val_frac=val_frac)

    by_target: dict[str, Any] = {}
    models: dict[str, dict[str, str]] = {}
    stamp = model_name or default_model_name("target_compare")

    for target in SUPPORTED_TARGETS:
        report, ce_model, pe_model = run_split_blended_training(
            train_df, val_df, target=target, **model_kwargs,
        )
        by_target[target] = report
        os.makedirs(out_dir, exist_ok=True)
        ce_path = os.path.join(out_dir, f"{stamp}_{target}_ce.json")
        pe_path = os.path.join(out_dir, f"{stamp}_{target}_pe.json")
        ce_model.save_model(ce_path)
        pe_model.save_model(pe_path)
        models[target] = {"ce": ce_path, "pe": pe_path}

    comparison = {
        "inputs": paths,
        "model_setup": "split_ce_pe_no_is_call",
        "by_target": by_target,
        "summary": {
            target: by_target[target]["val_metrics"]["split_blended"]
            for target in SUPPORTED_TARGETS
        },
    }
    report_path = os.path.join(out_dir, f"{stamp}_target_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    comparison["artifacts"] = {"comparison_report": report_path, "models": models}
    return comparison


def _print_target_row(target: str, metrics: dict[str, float | None]) -> None:
    if target.startswith("hit_"):
        auc_s = f"{metrics['auc']:.4f}" if metrics.get("auc") is not None else "n/a"
        logloss_s = f"{metrics['logloss']:.4f}" if metrics.get("logloss") is not None else "n/a"
        print(
            f"  {target:<18} AUC={auc_s} LogLoss={logloss_s} "
            f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
        )
    else:
        unit = "%" if target in ("residual_pct_5m", "mfe_pct_10m", "mae_pct_10m") else " Rs"
        ltp = metrics["final_ltp_mae"]
        ltp_s = f"{ltp:.4f}" if ltp is not None else "n/a"
        print(
            f"  {target:<18} target_MAE={metrics['mae']:.4f}{unit} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} "
            f"final_ltp_mae={ltp_s} Rs"
        )


def _print_metrics(label: str, metrics: dict[str, float | None]) -> None:
    auc_part = f" AUC={metrics['auc']:.4f}" if metrics.get("auc") is not None else ""
    logloss_part = f" LogLoss={metrics['logloss']:.4f}" if metrics.get("logloss") is not None else ""
    final_ltp = metrics.get("final_ltp_mae")
    final_ltp_part = f" final_ltp_mae={final_ltp:.4f} Rs" if final_ltp is not None else ""
    print(
        f"  {label:<22} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
        f"R2={metrics['r2']:.4f}{auc_part}{logloss_part}{final_ltp_part}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost on Phase 1 ML feature CSVs.")
    parser.add_argument(
        "--data",
        required=True,
        help="CSV path or glob, e.g. data/ml_features/phase1_*.csv",
    )
    parser.add_argument(
        "--target",
        choices=SUPPORTED_TARGETS,
        default=DEFAULT_TARGET,
        help=f"Label column (default: {DEFAULT_TARGET})",
    )
    parser.add_argument("--out-dir", default=DEFAULT_MODEL_DIR, help="Directory for model + report JSON")
    parser.add_argument("--model-name", default=None, help="Base filename (without extension)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Hold out last fraction of dates")
    parser.add_argument("--keep-warmup", action="store_true", help="Include warmup_row rows in training")
    parser.add_argument("--side", choices=OPTION_SIDES, default=None, help="Train one side only (drops is_call)")
    parser.add_argument("--drop-is-call", action="store_true", help="Combined model without is_call feature")
    parser.add_argument(
        "--compare-sides",
        action="store_true",
        help="Train combined (with is_call) vs separate CE/PE models (without is_call)",
    )
    parser.add_argument(
        "--compare-targets",
        action="store_true",
        help="Compare residual_pct_5m vs residual_5m (split CE+PE, no is_call)",
    )
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--drop-features", default=None, help="Comma-separated list of features to drop from inputs")
    args = parser.parse_args(argv)

    try:
        t0 = time.monotonic()
        drop_list = [f.strip() for f in args.drop_features.split(",")] if args.drop_features else None
        common = dict(
            data=args.data,
            target=args.target,
            out_dir=args.out_dir,
            model_name=args.model_name,
            val_frac=args.val_frac,
            skip_warmup=not args.keep_warmup,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            drop_features=drop_list,
        )
        if args.compare_sides:
            if args.side or args.drop_is_call or args.compare_targets:
                raise ValueError("--compare-sides cannot be used with --side, --drop-is-call, or --compare-targets")
            result = run_side_comparison(**common)
            vm = result["val_metrics"]
            print(f"Side comparison done in {time.monotonic() - t0:.1f}s")
            print(f"Val rows: combined={result['val_rows']['combined']} "
                  f"ce={result['val_rows']['ce']} pe={result['val_rows']['pe']}")
            print("Validation metrics:")
            _print_metrics("combined (is_call)", vm["combined_with_is_call"])
            _print_metrics("CE-only model", vm["ce_only"])
            _print_metrics("PE-only model", vm["pe_only"])
            _print_metrics("split blended", vm["split_blended"])
            print(f"Comparison report -> {result['artifacts']['comparison_report']}")
            return 0

        if args.compare_targets:
            if args.side or args.drop_is_call or args.compare_sides:
                raise ValueError("--compare-targets cannot be used with --side, --drop-is-call, or --compare-sides")
            result = run_target_comparison(
                data=args.data,
                out_dir=args.out_dir,
                model_name=args.model_name,
                val_frac=args.val_frac,
                skip_warmup=not args.keep_warmup,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
            )
            print(f"Target comparison done in {time.monotonic() - t0:.1f}s (split CE+PE, no is_call)")
            print(f"Val rows per target: {result['by_target']['residual_pct_5m']['val_rows']['total']}")
            print("Split-blended validation (compare final_ltp_mae in Rs):")
            for target in SUPPORTED_TARGETS:
                _print_target_row(target, result["summary"][target])
            pct_ltp = result["summary"]["residual_pct_5m"]["final_ltp_mae"]
            rupee_ltp = result["summary"]["residual_5m"]["final_ltp_mae"]
            if pct_ltp is not None and rupee_ltp is not None:
                winner = "residual_5m" if rupee_ltp < pct_ltp else "residual_pct_5m"
                print(f"Winner on final_ltp_mae: {winner}")
            print(f"Comparison report -> {result['artifacts']['comparison_report']}")
            return 0

        report = run_training(
            side=args.side,
            drop_is_call=args.drop_is_call,
            **common,
        )
        val = report["metrics"]["val"]
        kind = report.get("model_kind", "combined")
        print(
            f"Trained [{kind}] {report['artifacts']['model']} in {time.monotonic() - t0:.1f}s "
            f"({report['train_rows']} train / {report['val_rows']} val rows, "
            f"{report['n_features']} features)"
        )
        auc_part = f" AUC={val['auc']:.4f}" if val.get("auc") is not None else ""
        logloss_part = f" LogLoss={val['logloss']:.4f}" if val.get("logloss") is not None else ""
        final_ltp = val.get("final_ltp_mae")
        final_ltp_part = f" final_ltp_mae={final_ltp:.4f} Rs" if final_ltp is not None else " n/a"
        print(
            f"Val  target MAE={val['mae']:.4f} RMSE={val['rmse']:.4f} R2={val['r2']:.4f}"
            f"{auc_part}{logloss_part} final_ltp_mae={final_ltp_part}"
        )
        print(f"Report -> {report['artifacts']['report']}")
        print("Top features:")
        for item in report["feature_importance_top15"][:8]:
            print(f"  {item['feature']}: {item['importance']:.4f}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
