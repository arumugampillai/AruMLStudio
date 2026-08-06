"""Rolling window kernels (mean/std, max/min)."""

from __future__ import annotations

import math

import numpy as np

from ..numba_utils import njit


@njit(cache=True, parallel=False)
def rolling_mean_std_kernel(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing rolling mean and population std; NaN until window is full."""
    n = arr.shape[0]
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = np.nan
        stds[i] = np.nan
    if window <= 0 or n < window:
        return means, stds
    for i in range(window - 1, n):
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += arr[j]
        mean /= window
        var = 0.0
        for j in range(start, i + 1):
            d = arr[j] - mean
            var += d * d
        var /= window
        means[i] = mean
        stds[i] = math.sqrt(var) if var > 0.0 else 0.0
    return means, stds


@njit(cache=True)
def rolling_max_min_kernel(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing rolling max/min; NaN until window is full."""
    n = arr.shape[0]
    maxima = np.empty(n, dtype=np.float64)
    minima = np.empty(n, dtype=np.float64)
    for i in range(n):
        maxima[i] = np.nan
        minima[i] = np.nan
    if window <= 0 or n < window:
        return maxima, minima
    for i in range(window - 1, n):
        start = i - window + 1
        mx = arr[start]
        mn = arr[start]
        for j in range(start + 1, i + 1):
            v = arr[j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        maxima[i] = mx
        minima[i] = mn
    return maxima, minima
