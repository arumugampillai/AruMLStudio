"""SHAP feature attribution for trained XGBoost boosters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SHAP_SAMPLE_SIZE = 400


def compute_shap_importance(
    booster: Any,
    X: pd.DataFrame,
    features: list[str],
    *,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    try:
        import shap
    except ImportError:
        return []
    if X.empty or not features:
        return []
    use = X[features].replace([np.inf, -np.inf], np.nan).dropna()
    if use.empty:
        return []
    if len(use) > sample_size:
        use = use.sample(sample_size, random_state=42)
    try:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(use)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_abs = np.abs(np.asarray(shap_vals)).mean(axis=0)
    except Exception:
        return []
    total = float(mean_abs.sum()) or 1.0
    rows = [
        {
            "feature": feat,
            "importance_pct": round(float(val) / total * 100.0, 2),
            "source": "shap",
        }
        for feat, val in zip(features, mean_abs)
    ]
    rows.sort(key=lambda r: r.get("importance_pct") or 0, reverse=True)
    return rows
