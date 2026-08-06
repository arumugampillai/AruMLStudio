"""Optional RAPIDS cuDF / CuPy Pearson correlation (Linux + CUDA typically).

On Windows and any environment without RAPIDS, ``is_gpu_available()`` is False
and this module never raises on import. Callers must not import ``cudf`` /
``cupy`` from UI code — use :mod:`engine` instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .types import CorrelationTiming

logger = logging.getLogger(__name__)

_GPU_PROBE: bool | None = None
_GPU_PROBE_ERROR: str | None = None


def _probe_gpu() -> tuple[bool, str | None]:
    """Return (available, error_message). Never raises."""
    global _GPU_PROBE, _GPU_PROBE_ERROR
    if _GPU_PROBE is not None:
        return _GPU_PROBE, _GPU_PROBE_ERROR
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except Exception as exc:  # ImportError, OSError, etc.
        _GPU_PROBE = False
        _GPU_PROBE_ERROR = f"{type(exc).__name__}: {exc}"
        return _GPU_PROBE, _GPU_PROBE_ERROR
    try:
        # Confirm a device is visible (avoids import-ok / runtime-fail on Windows).
        import cupy

        _ = cupy.cuda.runtime.getDeviceCount()
        if int(_) < 1:
            _GPU_PROBE = False
            _GPU_PROBE_ERROR = "No CUDA devices found"
            return _GPU_PROBE, _GPU_PROBE_ERROR
    except Exception as exc:
        _GPU_PROBE = False
        _GPU_PROBE_ERROR = f"{type(exc).__name__}: {exc}"
        return _GPU_PROBE, _GPU_PROBE_ERROR
    _GPU_PROBE = True
    _GPU_PROBE_ERROR = None
    return _GPU_PROBE, _GPU_PROBE_ERROR


def is_gpu_available() -> bool:
    """True only when cudf + cupy import and at least one CUDA device exists."""
    ok, _ = _probe_gpu()
    return bool(ok)


def gpu_unavailable_reason() -> str | None:
    """Human-readable reason when GPU is unavailable (else None)."""
    ok, err = _probe_gpu()
    if ok:
        return None
    return err or "RAPIDS cuDF/CuPy not available"


def pearson_corr_gpu(
    frame: Any,
    *,
    min_periods: int = 2,
) -> tuple[Any, CorrelationTiming]:
    """Pearson on GPU via cuDF; returns a pandas DataFrame.

    Raises on failure so the facade can fall back to CPU. Does not use pandas
    for the correlation itself (host→device transfer, then cuDF ``corr``).
    """
    if not is_gpu_available():
        raise RuntimeError(gpu_unavailable_reason() or "GPU unavailable")

    import cudf

    # min_periods kept for API parity; cuDF corr uses pairwise complete obs.
    _ = int(min_periods)

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    # Accept pandas (Analysis Lab prep) or already-cuDF frames.
    if type(frame).__module__.startswith("cudf"):
        gdf = frame
        transfer_in = 0.0
    else:
        gdf = cudf.DataFrame.from_pandas(frame)
        transfer_in = max(time.perf_counter() - t0, 0.0)

    t1 = time.perf_counter()
    gcorr = gdf.corr(method="pearson")
    compute_sec = max(time.perf_counter() - t1, 0.0)

    t2 = time.perf_counter()
    corr = gcorr.to_pandas()
    transfer_out = max(time.perf_counter() - t2, 0.0)

    timing = CorrelationTiming(
        gpu_transfer_sec=transfer_in + transfer_out,
        gpu_compute_sec=compute_sec,
        total_sec=max(time.perf_counter() - t_all, 0.0),
    )
    return corr, timing
