"""Correlation matrix engines — CPU (default) and optional RAPIDS GPU.

UI and Analysis Lab must import from this package (or engine), never
cudf / cupy directly.
"""

from __future__ import annotations

from .engine import CorrelationEngine, is_gpu_available, resolve_backend
from .types import (
    BackendPreference,
    BackendUsed,
    CorrelationComputeResult,
    CorrelationTiming,
)

__all__ = [
    "BackendPreference",
    "BackendUsed",
    "CorrelationComputeResult",
    "CorrelationEngine",
    "CorrelationTiming",
    "is_gpu_available",
    "resolve_backend",
]
