"""Polars univariate / drift helpers for Feature Studio (post-P2)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _empty_distribution_summary() -> dict[str, float | None]:
    return {
        "mean": None,
        "std": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "min": None,
        "max": None,
        "count": 0,
    }


def distribution_summary_via_polars(values: Any) -> dict[str, float | None]:
    """Finite-value summary matching ``holdout_performance.distribution_summary``."""
    from .convert import require_polars

    pl = require_polars()
    if hasattr(values, "to_numpy") and not isinstance(values, (list, tuple, np.ndarray)):
        # pandas Series / Index
        try:
            import pandas as pd

            if isinstance(values, pd.Series):
                values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        except Exception:
            pass
    s = pl.Series(values).cast(pl.Float64, strict=False)
    s = s.filter(s.is_finite())
    n = int(s.len())
    if n == 0:
        return _empty_distribution_summary()
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
        "count": n,
    }


def _empty_feature_distribution_row(feature: str) -> dict[str, Any]:
    return {
        "feature": feature,
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


def feature_distribution_rows_via_polars(
    frame: Any,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    """Batch univariate holdout stats (Distribution Studio schema)."""
    from .convert import arrow_table_to_polars, require_polars

    pl = require_polars()
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        try:
            import pyarrow as pa

            pl_df = arrow_table_to_polars(pa.Table.from_pandas(frame, preserve_index=False))
        except Exception:
            pl_df = pl.from_pandas(frame)
    elif hasattr(frame, "with_columns"):
        pl_df = frame
    else:
        pl_df = pl.DataFrame(frame)

    n_total = int(pl_df.height)
    rows: list[dict[str, Any]] = []
    for name in features:
        if name not in pl_df.columns:
            rows.append(_empty_feature_distribution_row(name))
            continue
        col = pl.col(name).cast(pl.Float64, strict=False)
        finite = pl.when(col.is_finite()).then(col).otherwise(None)
        # Single-pass aggregates for this column
        agg = pl_df.select(
            col.is_null().sum().alias("null_count"),
            finite.is_not_null().sum().alias("n_finite"),
            finite.n_unique().alias("n_unique"),
            finite.mean().alias("mean"),
            finite.std(ddof=0).alias("std"),
            finite.min().alias("min"),
            finite.quantile(0.01).alias("p1"),
            finite.quantile(0.05).alias("p5"),
            finite.quantile(0.25).alias("p25"),
            finite.quantile(0.50).alias("p50"),
            finite.quantile(0.75).alias("p75"),
            finite.quantile(0.95).alias("p95"),
            finite.quantile(0.99).alias("p99"),
            finite.max().alias("max"),
            finite.skew().alias("skew"),
        ).row(0, named=True)
        n_finite = int(agg["n_finite"] or 0)
        null_pct = (100.0 * (n_total - n_finite) / n_total) if n_total else 0.0
        skew_out: float | None = None
        if n_finite >= 3 and agg.get("skew") is not None:
            try:
                skew_f = float(agg["skew"])
                if np.isfinite(skew_f):
                    skew_out = skew_f
            except (TypeError, ValueError):
                skew_out = None

        def _f(key: str) -> float | None:
            if n_finite == 0:
                return None
            val = agg.get(key)
            if val is None:
                return None
            try:
                out = float(val)
            except (TypeError, ValueError):
                return None
            return out if np.isfinite(out) else None

        rows.append(
            {
                "feature": name,
                "count": n_total,
                "n_finite": n_finite,
                "null_count": int(agg["null_count"] or 0),
                "null_pct": round(null_pct, 4),
                "n_unique": int(agg["n_unique"] or 0) if n_finite else 0,
                "mean": _f("mean"),
                "std": _f("std"),
                "min": _f("min"),
                "p1": _f("p1"),
                "p5": _f("p5"),
                "p25": _f("p25"),
                "p50": _f("p50"),
                "p75": _f("p75"),
                "p95": _f("p95"),
                "p99": _f("p99"),
                "max": _f("max"),
                "skew": skew_out,
            }
        )
    return rows
