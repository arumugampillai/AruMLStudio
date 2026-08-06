"""EMA kernels (single-step and series)."""

from __future__ import annotations

import numpy as np

from ..numba_utils import njit


@njit(cache=True)
def ema_update_kernel(prev: float, price: float, alpha: float) -> float:
    """Single EMA step: price * α + prev * (1-α)."""
    return price * alpha + prev * (1.0 - alpha)


@njit(cache=True)
def ema_series_kernel(prices: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series; first sample seeds the EMA (matches EmaController)."""
    n = prices.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(period) + 1.0)
    out[0] = prices[0]
    for i in range(1, n):
        out[i] = prices[i] * alpha + out[i - 1] * (1.0 - alpha)
    return out


def ema_series_python(prices: np.ndarray, period: int) -> np.ndarray:
    """Pure-Python EMA series (reference)."""
    n = int(prices.shape[0])
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(period) + 1.0)
    out[0] = float(prices[0])
    for i in range(1, n):
        out[i] = float(prices[i]) * alpha + out[i - 1] * (1.0 - alpha)
    return out
