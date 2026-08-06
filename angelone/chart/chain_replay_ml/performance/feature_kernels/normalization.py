"""Ratio / distance normalization helpers."""

from __future__ import annotations

import numpy as np

from ..numba_utils import njit


@njit(cache=True)
def safe_ratio_kernel(numer: float, denom: float) -> float:
    """numer/denom; returns nan if denom <= 0 (caller maps to None)."""
    if denom <= 0.0:
        return np.nan
    return numer / denom


@njit(cache=True)
def distance_pct_kernel(price: float, ref: float) -> float:
    """(price - ref) / ref * 100; nan if ref == 0."""
    if ref == 0.0:
        return np.nan
    return (price - ref) / ref * 100.0
