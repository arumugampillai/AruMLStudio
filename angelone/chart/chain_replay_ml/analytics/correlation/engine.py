"""CorrelationEngine facade — selects CPU or GPU Pearson backend."""

from __future__ import annotations

import logging
import time
from typing import Any

from . import cpu_engine, gpu_engine
from .types import (
    BackendPreference,
    BackendUsed,
    CorrelationComputeResult,
    CorrelationTiming,
    normalize_preference,
)

logger = logging.getLogger(__name__)


def is_gpu_available() -> bool:
    """Safe probe — never raises; False on Windows without RAPIDS."""
    try:
        return bool(gpu_engine.is_gpu_available())
    except Exception:
        return False


def resolve_backend(preference: str | None = None) -> BackendUsed:
    """Apply selection priority without computing.

    1. User selected GPU → GPU if available else CPU
    2. Auto → GPU if RAPIDS+CUDA compatible else CPU
    3. User selected CPU → CPU
    """
    pref = normalize_preference(preference)
    gpu_ok = is_gpu_available()
    if pref == "cpu":
        return "cpu"
    if pref == "gpu":
        return "gpu" if gpu_ok else "cpu"
    # auto
    return "gpu" if gpu_ok else "cpu"


class CorrelationEngine:
    """Facade over CPU (default) and optional RAPIDS GPU Pearson engines.

    Existing Analysis Lab preprocessing stays on CPU; only the Pearson matrix
    step is delegated here.
    """

    def __init__(self, *, preference: str | None = None) -> None:
        self.preference: BackendPreference = normalize_preference(preference)

    def select_backend(self, preference: str | None = None) -> BackendUsed:
        pref = (
            normalize_preference(preference)
            if preference is not None
            else self.preference
        )
        return resolve_backend(pref)

    def compute(
        self,
        frame: Any,
        *,
        preference: str | None = None,
        min_periods: int = 2,
    ) -> CorrelationComputeResult:
        """Return Pearson matrix as pandas DataFrame + timing metadata.

        On GPU OOM / any GPU error, falls back to CPU (no crash).
        """
        pref = (
            normalize_preference(preference)
            if preference is not None
            else self.preference
        )
        gpu_ok = is_gpu_available()
        n_rows = int(getattr(frame, "shape", (0, 0))[0] or 0)
        n_features = int(getattr(frame, "shape", (0, 0))[1] or 0)
        target = resolve_backend(pref)

        if target == "gpu":
            try:
                matrix, timing = gpu_engine.pearson_corr_gpu(
                    frame, min_periods=min_periods
                )
                return CorrelationComputeResult(
                    matrix=matrix,
                    backend_used="gpu",
                    preference=pref,
                    timing=timing,
                    n_rows=n_rows,
                    n_features=n_features,
                    gpu_available=gpu_ok,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "GPU correlation failed; falling back to CPU (%s)", reason
                )
                matrix, timing = cpu_engine.pearson_corr_cpu(
                    frame, min_periods=min_periods
                )
                timing.fallback_reason = reason
                return CorrelationComputeResult(
                    matrix=matrix,
                    backend_used="cpu",
                    preference=pref,
                    timing=timing,
                    n_rows=n_rows,
                    n_features=n_features,
                    gpu_available=gpu_ok,
                )

        t0 = time.perf_counter()
        matrix, timing = cpu_engine.pearson_corr_cpu(frame, min_periods=min_periods)
        if timing.total_sec <= 0:
            timing.total_sec = max(time.perf_counter() - t0, 0.0)
        if pref == "gpu" and not gpu_ok:
            timing.fallback_reason = (
                gpu_engine.gpu_unavailable_reason() or "GPU unavailable"
            )
        return CorrelationComputeResult(
            matrix=matrix,
            backend_used="cpu",
            preference=pref,
            timing=timing,
            n_rows=n_rows,
            n_features=n_features,
            gpu_available=gpu_ok,
        )
