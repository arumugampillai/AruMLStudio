"""Univariate holdout distribution stats per feature."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite_arr(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def feature_distribution_row(feature: str, series: pd.Series) -> dict[str, Any]:
    """Build one stats row for a feature column (full slice including nulls)."""
    try:
        from chain_replay_ml.frame_backend.studio_stats import (
            feature_distribution_rows_via_polars,
        )

        rows = feature_distribution_rows_via_polars(
            pd.DataFrame({feature: series}),
            [feature],
        )
        if rows:
            return rows[0]
    except Exception:
        pass
    return _feature_distribution_row_pandas(feature, series)


def _feature_distribution_row_pandas(feature: str, series: pd.Series) -> dict[str, Any]:
    n_total = int(len(series))
    numeric = pd.to_numeric(series, errors="coerce")
    finite = _finite_arr(series)
    n_finite = int(len(finite))
    try:
        n_unique = int(series.nunique(dropna=True))
    except TypeError:
        n_unique = int(pd.Series(finite).nunique()) if n_finite else 0

    null_pct = (100.0 * (n_total - n_finite) / n_total) if n_total else 0.0

    def pct(q: float) -> float | None:
        if n_finite == 0:
            return None
        return float(np.percentile(finite, q))

    skew: float | None = None
    if n_finite >= 3:
        try:
            skew = float(pd.Series(finite).skew())
            if not np.isfinite(skew):
                skew = None
        except (ValueError, TypeError):
            skew = None

    return {
        "feature": feature,
        "count": n_total,
        "n_finite": n_finite,
        "null_count": int(numeric.isna().sum()) if n_total else 0,
        "null_pct": round(null_pct, 4),
        "n_unique": n_unique,
        "mean": float(np.mean(finite)) if n_finite else None,
        "std": float(np.std(finite)) if n_finite else None,
        "min": float(np.min(finite)) if n_finite else None,
        "p1": pct(1),
        "p5": pct(5),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "p99": pct(99),
        "max": float(np.max(finite)) if n_finite else None,
        "skew": skew,
    }


def compute_holdout_stats(
    X: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    try:
        from chain_replay_ml.frame_backend.studio_stats import (
            feature_distribution_rows_via_polars,
        )

        return feature_distribution_rows_via_polars(X, features)
    except Exception:
        rows: list[dict[str, Any]] = []
        for name in features:
            if name not in X.columns:
                rows.append(
                    {
                        "feature": name,
                        "count": 0,
                        "n_finite": 0,
                        "null_count": 0,
                        "null_pct": 100.0,
                        "n_unique": 0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "p1": None,
                        "p5": None,
                        "p25": None,
                        "p50": None,
                        "p75": None,
                        "p95": None,
                        "p99": None,
                        "max": None,
                        "skew": None,
                    }
                )
                continue
            rows.append(_feature_distribution_row_pandas(name, X[name]))
        return rows
