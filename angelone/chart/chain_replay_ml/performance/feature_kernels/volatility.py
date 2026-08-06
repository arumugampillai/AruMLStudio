"""Volatility / return / IV z-score kernels."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..numba_utils import njit


@njit(cache=True)
def pct_returns_kernel(prices: np.ndarray) -> np.ndarray:
    """Percentage returns; length n-1 for n prices. Skips non-positive prices."""
    n = prices.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)
    # Worst-case size n-1; compact afterward is awkward in nopython — write valid slots.
    out = np.empty(n - 1, dtype=np.float64)
    count = 0
    last = prices[0]
    for i in range(1, n):
        price = prices[i]
        if last <= 0.0 or price <= 0.0:
            last = price
            continue
        out[count] = (price - last) / last * 100.0
        count += 1
        last = price
    return out[:count]


@njit(cache=True)
def iv_zscore_kernel(priors: np.ndarray, iv: float, eps: float = 1e-8) -> float:
    """Z-score of ``iv`` vs prior population; 0.0 when std <= eps (matches controller)."""
    n = priors.shape[0]
    if n <= 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += priors[i]
    mean /= n
    var = 0.0
    for i in range(n):
        d = priors[i] - mean
        var += d * d
    var /= n
    std = math.sqrt(var) if var > 0.0 else 0.0
    if std <= eps:
        return 0.0
    return (iv - mean) / std


def iv_zscore_python(priors: Sequence[float], iv: float, eps: float = 1e-8) -> float:
    """Reference IV z-score matching IvZscoreWindowController."""
    if not priors:
        return 0.0
    mean = sum(priors) / len(priors)
    variance = sum((v - mean) ** 2 for v in priors) / len(priors)
    std = math.sqrt(max(0.0, variance))
    return (iv - mean) / std if std > eps else 0.0
