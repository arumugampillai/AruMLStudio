"""Top 1% error driver attribution — separation scores and error-linked importance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_PREMIUM_EPS = 1e-9

PRIMARY_DRIVERS: list[dict[str, Any]] = [
    {
        "key": "gamma",
        "label": "Gamma",
        "columns": ["gamma", "gamma_x_spot"],
    },
    {
        "key": "time_to_expiry",
        "label": "Time to Expiry",
        "columns": ["minutes_to_expiry", "days_to_expiry"],
    },
    {
        "key": "spot_movement",
        "label": "Spot Movement",
        "columns": ["spot_change_5m", "spot_change_1m", "spot_change"],
    },
    {
        "key": "current_iv",
        "label": "Current IV",
        "columns": ["current_iv", "roll_iv"],
    },
]

_BASELINE_TRIO_KEYS = frozenset({"gamma", "time_to_expiry", "current_iv"})


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sq_rel_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > _PREMIUM_EPS)
    out = np.zeros(len(y_true), dtype=float)
    if not mask.any():
        return out
    rel = (y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])
    out[mask] = np.square(rel)
    return out


def _mean_sq_rel_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = _sq_rel_error(y_true, y_pred)
    v = err[err > 0]
    if len(v) == 0:
        return 0.0
    return float(np.mean(v))


def _separation_score(top_vals: pd.Series, rest_vals: pd.Series) -> float:
    """Log-ratio of medians plus Cohen's d blend — robust across scales."""
    top = top_vals.dropna()
    rest = rest_vals.dropna()
    if top.empty or rest.empty:
        return 0.0
    top_med = float(top.median())
    rest_med = float(rest.median())
    denom = max(abs(rest_med), _PREMIUM_EPS)
    log_ratio = abs(np.log(max(abs(top_med), _PREMIUM_EPS) / denom))
    pooled_std = float(np.sqrt((top.var(ddof=1) + rest.var(ddof=1)) / 2.0)) or _PREMIUM_EPS
    cohens_d = abs((float(top.mean()) - float(rest.mean())) / pooled_std)
    return round(float(log_ratio * 0.6 + min(cohens_d, 10.0) * 0.4), 4)


def score_primary_driver_separation(
    *,
    top_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    compare_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rank Gamma / TTE / Spot / IV by how strongly top 1% differs from the rest."""
    compare_map = {
        str(r.get("key") or ""): r
        for r in (compare_rows or [])
        if isinstance(r, dict)
    }
    rows: list[dict[str, Any]] = []
    for spec in PRIMARY_DRIVERS:
        col = _resolve_column(top_df, list(spec["columns"]))
        if col:
            top_s = pd.to_numeric(top_df[col], errors="coerce")
            rest_s = pd.to_numeric(rest_df[col], errors="coerce")
            separation = _separation_score(top_s, rest_s)
            top_med = float(top_s.median()) if not top_s.dropna().empty else None
            rest_med = float(rest_s.median()) if not rest_s.dropna().empty else None
        else:
            separation = 0.0
            top_med = rest_med = None

        cmp_row = compare_map.get(str(spec["key"]), {})
        rows.append({
            "key": spec["key"],
            "driver": spec["label"],
            "column": col,
            "separation_score": separation,
            "top_median": round(top_med, 4) if top_med is not None else None,
            "rest_median": round(rest_med, 4) if rest_med is not None else None,
            "mean_difference_pct": cmp_row.get("difference_pct"),
        })
    rows.sort(key=lambda r: float(r.get("separation_score") or 0), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        row["error_contribution_pct"] = None
    return rows


class _PredictAdapter:
    def __init__(self, model: Any, features: list[str]) -> None:
        self._model = model
        self._features = features

    def fit(self, X, y):  # noqa: ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        if isinstance(X, pd.DataFrame):
            return np.asarray(self._model.predict(X[self._features]), dtype=float)
        return np.asarray(self._model.predict(X), dtype=float)


def compute_top1_error_permutation_importance(
    *,
    model: Any,
    X_top1: pd.DataFrame,
    y_top1: np.ndarray,
    features: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Permutation importance on mean squared relative error for top-1% rows."""
    use_features = [f for f in features if f in X_top1.columns]
    if not use_features or X_top1.empty:
        return []

    y_true = np.asarray(y_top1, dtype=float)
    adapter = _PredictAdapter(model, use_features)
    try:
        base_pred = adapter.predict(X_top1)
    except Exception:
        return []

    baseline = _mean_sq_rel_error(y_true, base_pred)
    if baseline <= 0:
        return []

    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        return []

    def _scorer(estimator, X, y):  # noqa: ANN001
        pred = estimator.predict(X)
        return -_mean_sq_rel_error(np.asarray(y, dtype=float), pred)

    try:
        result = permutation_importance(
            adapter,
            X_top1[use_features],
            y_true,
            scoring=_scorer,
            n_repeats=max(1, int(n_repeats)),
            random_state=int(random_state),
            n_jobs=1,
        )
    except Exception:
        return []

    raw = np.maximum(np.asarray(result.importances_mean, dtype=float), 0.0)
    total = float(raw.sum()) or 1.0
    rows = [
        {
            "feature": feat,
            "error_contribution_pct": round(float(val) / total * 100.0, 2),
            "error_increase": round(float(val), 6),
            "method": "permutation",
        }
        for feat, val in zip(use_features, raw)
    ]
    rows.sort(key=lambda r: r.get("error_contribution_pct") or 0, reverse=True)
    return rows


def compute_top1_error_shap_importance(
    *,
    model: Any,
    X_top1: pd.DataFrame,
    y_top1: np.ndarray,
    features: list[str],
    sample_size: int = 400,
) -> list[dict[str, Any]]:
    """SHAP attribution weighted by per-row squared relative error on top 1%."""
    from .shap_importance import SHAP_SAMPLE_SIZE

    use_features = [f for f in features if f in X_top1.columns]
    if not use_features or X_top1.empty:
        return []

    try:
        import shap
    except ImportError:
        return []

    y_true = np.asarray(y_top1, dtype=float)
    X = X_top1[use_features].replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(y_true) & (np.abs(y_true) > _PREMIUM_EPS)
    if not mask.any():
        return []
    X = X.loc[mask]
    y_true = y_true[mask]
    if X.empty:
        return []

    try:
        pred = np.asarray(model.predict(X), dtype=float)
    except Exception:
        return []

    row_weights = _sq_rel_error(y_true, pred)
    weight_sum = float(row_weights.sum()) or 1.0

    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_use = X.iloc[idx]
        weights_use = row_weights[idx]
        weight_sum = float(weights_use.sum()) or 1.0
    else:
        X_use = X
        weights_use = row_weights

    booster = getattr(model, "get_booster", lambda: None)()
    explainer_target = booster if booster is not None else model
    try:
        explainer = shap.TreeExplainer(explainer_target)
        shap_vals = explainer.shap_values(X_use)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_arr = np.abs(np.asarray(shap_vals, dtype=float))
        weighted = (shap_arr.T * weights_use).T.sum(axis=0) / weight_sum
    except Exception:
        return []

    total = float(weighted.sum()) or 1.0
    rows = [
        {
            "feature": feat,
            "error_contribution_pct": round(float(val) / total * 100.0, 2),
            "method": "shap_weighted",
        }
        for feat, val in zip(use_features, weighted)
    ]
    rows.sort(key=lambda r: r.get("error_contribution_pct") or 0, reverse=True)
    return rows


def _map_feature_to_driver(feature: str, driver_rows: list[dict[str, Any]]) -> str | None:
    feat = str(feature).lower()
    for spec in PRIMARY_DRIVERS:
        for col in spec["columns"]:
            if feat == col.lower() or feat.startswith(col.lower()):
                return str(spec["key"])
    alias_map = {
        "minutes_to_expiry": "time_to_expiry",
        "days_to_expiry": "time_to_expiry",
        "spot_change": "spot_movement",
        "spot_change_5m": "spot_movement",
        "spot_change_1m": "spot_movement",
        "roll_iv": "current_iv",
        "gamma_x_spot": "gamma",
    }
    return alias_map.get(feat)


def aggregate_driver_error_contribution(
    importance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Roll feature-level error importance into the four primary drivers."""
    totals: dict[str, float] = {spec["key"]: 0.0 for spec in PRIMARY_DRIVERS}
    for row in importance_rows:
        feat = str(row.get("feature") or "")
        pct = float(row.get("error_contribution_pct") or 0.0)
        driver = _map_feature_to_driver(feat, [])
        if driver:
            totals[driver] += pct
    grand = sum(totals.values()) or 1.0
    out = []
    label_map = {spec["key"]: spec["label"] for spec in PRIMARY_DRIVERS}
    for key, val in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        out.append({
            "key": key,
            "driver": label_map[key],
            "error_contribution_pct": round(val / grand * 100.0, 2) if grand > 0 else 0.0,
        })
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def recommend_features_beyond_baseline(
    importance_rows: list[dict[str, Any]],
    *,
    min_pct: float = 3.0,
) -> dict[str, Any]:
    """Suggest new features only when they materially add beyond Gamma/IV/TTE."""
    if not importance_rows:
        return {
            "recommend_new_features": False,
            "baseline_trio_pct": None,
            "notes": "Insufficient error-importance data.",
            "candidates": [],
        }

    baseline_pct = 0.0
    outside: list[dict[str, Any]] = []
    for row in importance_rows:
        feat = str(row.get("feature") or "")
        pct = float(row.get("error_contribution_pct") or 0.0)
        driver = _map_feature_to_driver(feat, [])
        if driver in _BASELINE_TRIO_KEYS:
            baseline_pct += pct
        elif pct >= min_pct:
            outside.append({"feature": feat, "error_contribution_pct": pct})

    outside.sort(key=lambda r: r["error_contribution_pct"], reverse=True)
    recommend = bool(outside)
    notes = (
        f"Gamma + IV + Time-to-Expiry explain {baseline_pct:.1f}% of measured error contribution."
    )
    if recommend:
        top = outside[0]
        notes += (
            f" {top['feature']} adds {top['error_contribution_pct']:.1f}% beyond the baseline trio"
            " and may warrant engineering."
        )
    else:
        notes += " No non-baseline feature exceeds the materiality threshold; focus on expiry-gamma regime modeling."

    return {
        "recommend_new_features": recommend,
        "baseline_trio_pct": round(baseline_pct, 2),
        "notes": notes,
        "candidates": outside[:5],
    }


def analyze_top1_error_drivers(
    *,
    top_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    compare_rows: list[dict[str, Any]] | None = None,
    model: Any | None = None,
    X_top1: pd.DataFrame | None = None,
    y_top1: np.ndarray | None = None,
    features: list[str] | None = None,
    importance_method: str = "permutation",
) -> dict[str, Any]:
    separation = score_primary_driver_separation(
        top_df=top_df,
        rest_df=rest_df,
        compare_rows=compare_rows,
    )

    importance_rows: list[dict[str, Any]] = []
    method_used = None
    if model is not None and X_top1 is not None and y_top1 is not None and features:
        if importance_method == "shap":
            importance_rows = compute_top1_error_shap_importance(
                model=model,
                X_top1=X_top1,
                y_top1=y_top1,
                features=features,
            )
            method_used = "shap_weighted" if importance_rows else None
        if not importance_rows:
            importance_rows = compute_top1_error_permutation_importance(
                model=model,
                X_top1=X_top1,
                y_top1=y_top1,
                features=features,
            )
            method_used = "permutation" if importance_rows else None

    driver_error = aggregate_driver_error_contribution(importance_rows)
    if driver_error:
        sep_map = {r["key"]: r for r in separation}
        for row in separation:
            err_row = next((d for d in driver_error if d["key"] == row["key"]), None)
            if err_row:
                row["error_contribution_pct"] = err_row.get("error_contribution_pct")

    primary_by_separation = separation[0]["driver"] if separation else None
    primary_by_error = driver_error[0]["driver"] if driver_error else None
    primary_driver = primary_by_error or primary_by_separation

    recommendations = recommend_features_beyond_baseline(importance_rows)

    return {
        "primary_driver": primary_driver,
        "primary_driver_by_separation": primary_by_separation,
        "primary_driver_by_error": primary_by_error,
        "driver_separation_ranking": separation,
        "feature_error_importance": importance_rows[:25],
        "driver_error_contribution": driver_error,
        "importance_method": method_used,
        "feature_recommendations": recommendations,
    }
