"""Population statistics kernels (std)."""

from __future__ import annotations

import math

import numpy as np

from ..numba_utils import njit


@njit(cache=True)
def population_std_kernel(arr: np.ndarray) -> float:
    """Population std of a 1-d float64 array (same formula as np.std(..., ddof=0))."""
    n = arr.shape[0]
    if n <= 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n
    var = 0.0
    for i in range(n):
        d = arr[i] - mean
        var += d * d
    var /= n
    if var <= 0.0:
        return 0.0
    return math.sqrt(var)


def population_std_numpy(arr: np.ndarray) -> float:
    """Reference: NumPy population std."""
    if arr.size <= 0:
        return 0.0
    return float(np.std(arr, ddof=0))
