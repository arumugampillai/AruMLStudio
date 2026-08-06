"""Numeric kernels for feature-engine hotspots (Numba nopython when available).

Split by domain for maintainability. Public names are re-exported here so
existing imports keep working::

    from chain_replay_ml.performance.feature_kernels import population_std_kernel
"""

from __future__ import annotations

from .common import buffer_to_float64
from .ema import ema_series_kernel, ema_series_python, ema_update_kernel
from .normalization import distance_pct_kernel, safe_ratio_kernel
from .rolling import rolling_max_min_kernel, rolling_mean_std_kernel
from .statistics import population_std_kernel, population_std_numpy
from .volatility import iv_zscore_kernel, iv_zscore_python, pct_returns_kernel

__all__ = [
    "buffer_to_float64",
    "population_std_kernel",
    "population_std_numpy",
    "ema_update_kernel",
    "ema_series_kernel",
    "ema_series_python",
    "rolling_mean_std_kernel",
    "rolling_max_min_kernel",
    "pct_returns_kernel",
    "safe_ratio_kernel",
    "distance_pct_kernel",
    "iv_zscore_kernel",
    "iv_zscore_python",
]
