"""LTP premium band filter for dataset export (post No-Null).

Same semantics as Master Dataset Sample Preview: keep rows where
``ltp`` is within [premium_min, premium_max] (inclusive).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_premium_bounds(
    premium_min: float | None,
    premium_max: float | None,
) -> tuple[float, float] | None:
    """Return sorted (lo, hi) or None when either bound is missing."""
    if premium_min is None or premium_max is None:
        return None
    lo = float(premium_min)
    hi = float(premium_max)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def apply_premium_ltp_filter_frame(
    frame: pd.DataFrame,
    *,
    premium_min: float,
    premium_max: float,
    ltp_column: str = "ltp",
) -> dict[str, Any]:
    """Filter ``frame`` to the LTP premium band.

    Returns ``{"frame": DataFrame, "report": dict}``.
    """
    bounds = normalize_premium_bounds(premium_min, premium_max)
    if bounds is None:
        raise ValueError("premium_min and premium_max are required")
    lo, hi = bounds
    col = str(ltp_column or "ltp").strip() or "ltp"
    if col not in frame.columns:
        raise ValueError(f"Premium filter requires column '{col}'")

    rows_before = int(len(frame))
    series = pd.to_numeric(frame[col], errors="coerce")
    mask = series.notna() & (series >= lo) & (series <= hi)
    out = frame.loc[mask]
    rows_after = int(len(out))
    return {
        "frame": out,
        "report": {
            "stage": "post_no_null",
            "ltp_column": col,
            "premium_min": lo,
            "premium_max": hi,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_dropped": rows_before - rows_after,
        },
    }


__all__ = [
    "apply_premium_ltp_filter_frame",
    "normalize_premium_bounds",
]
