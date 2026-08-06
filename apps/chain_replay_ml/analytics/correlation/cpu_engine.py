"""CPU Pearson correlation — wraps existing pandas ``DataFrame.corr`` behavior."""

from __future__ import annotations

import time
from typing import Any

from .types import CorrelationTiming


def pearson_corr_cpu(
    frame: Any,
    *,
    min_periods: int = 2,
) -> tuple[Any, CorrelationTiming]:
    """Compute Pearson correlation with pandas (unchanged semantics).

    ``frame`` must already be numeric-only with columns ordered as desired.
    Missing values use pairwise complete observations (pandas default).
    """
    t0 = time.perf_counter()
    corr = frame.corr(method="pearson", min_periods=int(min_periods))
    elapsed = max(time.perf_counter() - t0, 0.0)
    timing = CorrelationTiming(cpu_compute_sec=elapsed, total_sec=elapsed)
    return corr, timing
