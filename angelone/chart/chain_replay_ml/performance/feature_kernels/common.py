"""Shared array helpers for feature kernels."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def buffer_to_float64(buffer: Sequence[float]) -> np.ndarray:
    """Copy a deque/list/array into a contiguous float64 ndarray."""
    if isinstance(buffer, np.ndarray):
        if buffer.dtype == np.float64 and buffer.flags.c_contiguous:
            return buffer
        return np.ascontiguousarray(buffer, dtype=np.float64)
    n = len(buffer)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return np.fromiter(buffer, dtype=np.float64, count=n)
