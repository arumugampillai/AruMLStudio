"""Feature matrix validation — shape, NaN, infinity, duplicates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureMatrixError(Exception):
    pass


def check_duplicate_features(features: list[str]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for f in features:
        if f in seen:
            dups.append(f)
        seen.add(f)
    return dups


def validate_feature_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    min_rows: int = 500,
) -> dict[str, Any]:
    """Validate X/y before training. Raises FeatureMatrixError on hard failures."""
    dup_feats = check_duplicate_features(list(X.columns))
    if dup_feats:
        raise FeatureMatrixError(f"Duplicate features: {', '.join(sorted(set(dup_feats)))}")

    n_rows = len(X)
    if n_rows != len(y):
        raise FeatureMatrixError(f"X and y row count mismatch: {n_rows} vs {len(y)}")

    if n_rows < min_rows:
        raise FeatureMatrixError(f"Not enough rows for training ({n_rows} < {min_rows})")

    target_nan = int(y.isna().sum())
    if target_nan > 0:
        raise FeatureMatrixError(f"Target has {target_nan:,} NaN values")

    x_inf = int(np.isinf(X.select_dtypes(include=[np.number])).sum().sum())
    y_inf = int(np.isinf(pd.to_numeric(y, errors="coerce")).sum())
    feature_nan = int(X.isna().sum().sum())

    report: dict[str, Any] = {
        "x_shape": [int(n_rows), int(X.shape[1])],
        "y_shape": [int(n_rows)],
        "feature_nan_count": feature_nan,
        "target_nan_count": target_nan,
        "feature_inf_count": x_inf,
        "target_inf_count": y_inf,
        "duplicate_features": [],
        "ready": True,
    }

    if y_inf > 0:
        raise FeatureMatrixError(f"Target has {y_inf:,} infinite values")

    return report


def drop_invalid_rows(
    X: pd.DataFrame,
    y: pd.Series,
    context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Drop rows where target is NaN/inf (features may still have NaN — XGBoost handles)."""
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna() & np.isfinite(y_num)
    X_out = X.loc[mask].reset_index(drop=True)
    y_out = y_num.loc[mask].reset_index(drop=True)
    if context is not None and len(context) == len(X):
        ctx_out = context.loc[mask].reset_index(drop=True)
    else:
        ctx_out = pd.DataFrame()
    return X_out, y_out, ctx_out
