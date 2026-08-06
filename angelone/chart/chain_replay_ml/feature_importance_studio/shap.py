"""Holdout TreeSHAP — mean |SHAP| ranking (v1; no dependence plots).

Prefers the ``shap`` package (TreeExplainer). If unavailable, falls back to
XGBoost ``pred_contribs`` (same family of tree attributions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from chain_replay_ml.feature_importance_studio.native import _booster_from_model


def _mean_abs_from_contribs(contribs: np.ndarray) -> np.ndarray:
    arr = np.asarray(contribs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros(0, dtype=float)
    # Last column is bias for XGBoost pred_contribs.
    return np.abs(arr[:, :-1]).mean(axis=0)


def _shap_via_package(booster: Any, use: pd.DataFrame) -> np.ndarray | None:
    try:
        import shap
    except ImportError:
        return None
    try:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(use)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return np.abs(np.asarray(shap_vals, dtype=float)).mean(axis=0)
    except Exception:
        return None


def _shap_via_pred_contribs(booster: Any, use: pd.DataFrame, features: list[str]) -> np.ndarray | None:
    try:
        import xgboost as xgb
    except ImportError:
        return None
    try:
        dmat = xgb.DMatrix(use, feature_names=features)
        contribs = booster.predict(dmat, pred_contribs=True)
        mean_abs = _mean_abs_from_contribs(contribs)
        if mean_abs.size != len(features):
            return None
        return mean_abs
    except Exception:
        return None


def compute_shap_holdout(
    model: Any,
    X: pd.DataFrame,
    features: list[str],
    *,
    sample_size: int = 400,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Tree attributions on a holdout sample; return mean |SHAP| and rank."""
    use_feats = [f for f in features if f in X.columns]
    if not use_feats or X.empty:
        return []

    use = (
        X[use_feats]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if use.empty:
        return []
    if len(use) > int(sample_size):
        use = use.sample(n=int(sample_size), random_state=random_state)

    booster = _booster_from_model(model)
    mean_abs = _shap_via_package(booster, use)
    source = "shap_treeshap"
    if mean_abs is None:
        mean_abs = _shap_via_pred_contribs(booster, use, use_feats)
        source = "xgboost_pred_contribs"
    if mean_abs is None or len(mean_abs) != len(use_feats):
        return []

    rows: list[dict[str, Any]] = []
    for feat, val in zip(use_feats, mean_abs):
        rows.append(
            {
                "feature": feat,
                "shap_mean_abs": float(val),
                "source": source,
            }
        )
    ranked = sorted(rows, key=lambda r: r["shap_mean_abs"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank_shap"] = i
    by_feat = {r["feature"]: r for r in ranked}
    return [by_feat[f] for f in use_feats if f in by_feat]
