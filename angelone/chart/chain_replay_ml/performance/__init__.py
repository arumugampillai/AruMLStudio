"""Phase 6.0 — High Performance Feature Engine (Numba kernels + profiling).

Performance-only package. Feature logic stays in ``dataset_builder``;
this package holds numeric kernels, runtime dispatch, and benchmarks.

Enable/disable via env ``ARUNEO_FEATURE_NUMBA`` (default: on when Numba is installed).
When Numba is missing or JIT fails, Python fallback activates automatically.
"""

from __future__ import annotations

from .runtime import (
    begin_create_dataset_session,
    end_create_dataset_session,
    numba_available,
    numba_enabled,
    performance_stats,
    set_numba_enabled,
    warm_kernels,
    warmup_kernels,
)

__all__ = [
    "begin_create_dataset_session",
    "end_create_dataset_session",
    "numba_available",
    "numba_enabled",
    "performance_stats",
    "set_numba_enabled",
    "warm_kernels",
    "warmup_kernels",
]

__version__ = "0.1.0"

