"""Rehydrate feature vectors for prediction rows via dataset join."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from chain_replay_ml.training.config import TrainingConfig, normalize_training_config
from chain_replay_ml.training.dataset_loader import load_training_xy
from chain_replay_ml.training.feature_matrix import drop_invalid_rows


def load_package_config(package_dir: str) -> dict[str, Any]:
    for name in ("training_config.json", "config.json"):
        path = os.path.join(package_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def load_aligned_training_frame(
    data_dir: str,
    *,
    package_dir: str | None,
    dataset_name: str | None,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame, TrainingConfig | None]:
    """Load X/y/context aligned with walk-forward prediction row indices."""
    cfg_dict = load_package_config(package_dir or "") if package_dir else {}
    if not cfg_dict and dataset_name:
        cfg_dict = {"dataset": dataset_name, "features": [], "target": ""}
    if not cfg_dict.get("dataset") and dataset_name:
        cfg_dict["dataset"] = dataset_name
    if not cfg_dict.get("dataset"):
        raise ValueError("dataset_name or package config required")

    config = normalize_training_config(cfg_dict)
    X, y, features, _meta, _expected, context_df = load_training_xy(data_dir, config)
    X, y, context_df = drop_invalid_rows(X, y, context_df)
    return X, y, features, context_df, config


def global_row_index(fold: dict[str, Any], prediction_row: dict[str, Any]) -> int:
    return int(fold.get("validation_start") or 0) + int(prediction_row.get("row_index") or 0)


def rehydrate_feature_row(
    data_dir: str,
    *,
    run: dict[str, Any],
    fold: dict[str, Any],
    prediction_row: dict[str, Any],
) -> dict[str, Any]:
    package_dir = run.get("package_dir")
    dataset_name = run.get("dataset_name")
    try:
        X, _y, features, _ctx, config = load_aligned_training_frame(
            data_dir,
            package_dir=package_dir,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    idx = global_row_index(fold, prediction_row)
    if idx < 0 or idx >= len(X):
        return {"ok": False, "error": f"row index {idx} out of range (matrix rows={len(X)})"}

    row = X.iloc[idx]
    feat_dict: dict[str, Any] = {}
    for feat in features:
        val = row.get(feat)
        try:
            feat_dict[feat] = round(float(val), 6) if pd.notna(val) else None
        except (TypeError, ValueError):
            feat_dict[feat] = None

    return {
        "ok": True,
        "global_index": idx,
        "features": features,
        "feature_values": feat_dict,
        "algorithm": config.algorithm if config else None,
        "package_dir": package_dir,
    }


def slice_feature_matrix(
    data_dir: str,
    *,
    run: dict[str, Any],
    fold: dict[str, Any],
) -> dict[str, Any]:
    """Train and validation feature slices for drift analysis."""
    try:
        X, _y, features, _ctx, _config = load_aligned_training_frame(
            data_dir,
            package_dir=run.get("package_dir"),
            dataset_name=run.get("dataset_name"),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    tr_start = int(fold.get("train_start") or 0)
    tr_end = int(fold.get("train_end") or 0)
    va_start = int(fold.get("validation_start") or 0)
    va_end = int(fold.get("validation_end") or 0)

    if tr_end <= tr_start or va_end <= va_start:
        return {"ok": False, "error": "invalid fold window indices"}

    return {
        "ok": True,
        "features": features,
        "train": X.iloc[tr_start:tr_end],
        "validation": X.iloc[va_start:va_end],
    }
