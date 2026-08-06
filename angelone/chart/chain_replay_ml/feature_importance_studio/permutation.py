"""Holdout permutation importance (model-agnostic)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


PredictionKind = Literal["regression", "binary"]


def _baseline_score(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    kind: PredictionKind,
) -> float:
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if kind == "binary":
        # Higher is better → return accuracy; importance = drop in accuracy
        labels = (pred >= 0.5).astype(int) if pred.min() >= 0.0 and pred.max() <= 1.0 else pred.astype(int)
        y_int = y.astype(int)
        return float((labels == y_int).mean()) if len(y_int) else 0.0
    # regression: lower RMSE is better — return negative RMSE so higher=better
    err = pred - y
    return -float(np.sqrt(np.mean(err * err))) if len(err) else 0.0


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    features: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    kind: PredictionKind = "regression",
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Permute each feature on holdout; return mean/std importance + rank.

    Importance = baseline_score - permuted_score (higher = more important).
    """
    use_feats = [f for f in features if f in X.columns]
    if not use_feats or X.empty:
        return []

    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    X_use = X[use_feats].apply(pd.to_numeric, errors="coerce").astype("float32")
    mask = np.isfinite(y_arr) & np.isfinite(X_use.to_numpy(dtype=float)).all(axis=1)
    X_use = X_use.loc[mask].reset_index(drop=True)
    y_arr = y_arr[mask]
    if len(X_use) < 10:
        return []

    baseline = _baseline_score(model, X_use, y_arr, kind=kind)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    n_feat = len(use_feats)

    # In-place column swap avoids a full DataFrame copy per (feature × repeat).
    # Same permutation semantics; restore original values after each score.
    for i, feat in enumerate(use_feats):
        if progress:
            progress({"stage": "permutation", "feature": feat, "done": i, "total": n_feat})
        deltas: list[float] = []
        values = X_use[feat].to_numpy(copy=True)
        for _ in range(max(1, int(n_repeats))):
            shuffled = values.copy()
            rng.shuffle(shuffled)
            X_use[feat] = shuffled
            score = _baseline_score(model, X_use, y_arr, kind=kind)
            deltas.append(baseline - score)
        X_use[feat] = values
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        rows.append(
            {
                "feature": feat,
                "permutation_mean": mean,
                "permutation_std": std,
                "baseline_score": baseline,
                "n_repeats": int(n_repeats),
            }
        )

    ranked = sorted(rows, key=lambda r: r["permutation_mean"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank_permutation"] = i
    by_feat = {r["feature"]: r for r in ranked}
    return [by_feat[f] for f in use_feats if f in by_feat]
