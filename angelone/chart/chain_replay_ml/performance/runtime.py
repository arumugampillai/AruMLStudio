"""Runtime dispatch for Numba kernels used by rolling controllers.

Feature flag: ``ARUNEO_FEATURE_NUMBA``
  - unset / on  → use Numba when installed and JIT healthy (default)
  - off         → force NumPy / pure-Python reference paths

When Numba is missing or JIT fails, Python paths are used automatically
(no need to set the env flag). Controllers call thin helpers here only.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import numpy as np

from . import feature_kernels as kernels
from .numba_utils import (
    activate_python_fallback,
    ensure_fallback_logged,
    env_numba_flag,
    has_numba,
    python_fallback_active,
    python_fallback_reason,
    timed_compile,
)

logger = logging.getLogger(__name__)

_ENABLED_OVERRIDE: bool | None = None
_WARMED = False
_COMPILE_SEC: dict[str, float] = {}
_KERNEL_HITS = 0
_PYTHON_PATH_HITS = 0
_CACHE_HITS = 0
_DISPATCH_SEC = 0.0
_SESSION_T0: float | None = None
_CACHE_HIT_THRESHOLD_SEC = 0.05

# Treat first successful warm as the Feature Engine warm-up entry point.
_WARM_MSG_PRINTED = False


def numba_available() -> bool:
    """True when the ``numba`` package imported (may still be in Python fallback)."""
    return has_numba()


def numba_enabled() -> bool:
    if python_fallback_active():
        ensure_fallback_logged()
        return False
    if _ENABLED_OVERRIDE is not None:
        return bool(_ENABLED_OVERRIDE) and has_numba()
    flag = env_numba_flag()
    if flag is False:
        return False
    # Default on when Numba is installed and healthy.
    return has_numba()


def set_numba_enabled(enabled: bool | None) -> None:
    """Override env flag for tests. Pass None to clear override."""
    global _ENABLED_OVERRIDE
    _ENABLED_OVERRIDE = enabled


def compile_timings() -> dict[str, float]:
    return dict(_COMPILE_SEC)


def performance_stats() -> dict[str, Any]:
    """Counters for the performance dashboard / Create Dataset logs."""
    flag = env_numba_flag()
    session_wall = None
    if _SESSION_T0 is not None:
        session_wall = round(time.perf_counter() - _SESSION_T0, 6)
    return {
        "numba_available": has_numba(),
        "numba_enabled": numba_enabled(),
        "numba_enabled_label": "YES" if numba_enabled() else "NO",
        "aruneo_feature_numba_env": (
            "unset" if flag is None else ("on" if flag else "off")
        ),
        "python_fallback": python_fallback_active(),
        "python_fallback_reason": python_fallback_reason(),
        "warmed": _WARMED,
        "compile_overhead_sec": compile_timings(),
        "compile_total_sec": round(sum(_COMPILE_SEC.values()), 6),
        "kernel_hits": _KERNEL_HITS,
        "python_fallback_hits": _PYTHON_PATH_HITS,
        "cache_hits": _CACHE_HITS,
        "feature_dispatch_sec": round(_DISPATCH_SEC, 6),
        "session_wall_sec": session_wall,
    }


def reset_perf_counters_for_tests() -> None:
    """Test helper."""
    global _WARMED, _KERNEL_HITS, _PYTHON_PATH_HITS, _CACHE_HITS
    global _WARM_MSG_PRINTED, _COMPILE_SEC, _DISPATCH_SEC, _SESSION_T0
    _WARMED = False
    _KERNEL_HITS = 0
    _PYTHON_PATH_HITS = 0
    _CACHE_HITS = 0
    _WARM_MSG_PRINTED = False
    _COMPILE_SEC = {}
    _DISPATCH_SEC = 0.0
    _SESSION_T0 = None


def _record_kernel_hit() -> None:
    global _KERNEL_HITS
    _KERNEL_HITS += 1


def _record_python_path_hit() -> None:
    global _PYTHON_PATH_HITS
    _PYTHON_PATH_HITS += 1


def _add_dispatch_sec(sec: float) -> None:
    global _DISPATCH_SEC
    _DISPATCH_SEC += float(sec)


def begin_create_dataset_session(*, verbose: bool = True) -> dict[str, Any]:
    """Reset counters, warm kernels, log Numba status for Create Dataset / master build."""
    global _KERNEL_HITS, _PYTHON_PATH_HITS, _DISPATCH_SEC, _SESSION_T0
    _KERNEL_HITS = 0
    _PYTHON_PATH_HITS = 0
    _DISPATCH_SEC = 0.0
    _SESSION_T0 = time.perf_counter()
    warm_kernels(verbose=verbose)
    stats = performance_stats()
    msg = (
        f"Feature Engine (Create Dataset): Numba enabled: {stats['numba_enabled_label']} "
        f"(ARUNEO_FEATURE_NUMBA={stats['aruneo_feature_numba_env']}, "
        f"available={stats['numba_available']})"
    )
    logger.info(msg)
    if verbose:
        print(msg, flush=True)
    return stats


def end_create_dataset_session(
    *,
    verbose: bool = True,
    create_dataset_wall_sec: float | None = None,
    feature_computation_sec: float | None = None,
) -> dict[str, Any]:
    """Log kernel / Python-path counters and return a stats snapshot."""
    stats = performance_stats()
    if create_dataset_wall_sec is not None:
        stats["create_dataset_wall_sec"] = round(float(create_dataset_wall_sec), 6)
    if feature_computation_sec is not None:
        stats["feature_computation_sec"] = round(float(feature_computation_sec), 6)
    else:
        stats["feature_computation_sec"] = stats.get("feature_dispatch_sec")
    msg = (
        "Feature Engine (Create Dataset) summary: "
        f"Numba enabled: {stats['numba_enabled_label']}; "
        f"Numba kernel calls: {stats['kernel_hits']:,}; "
        f"Python fallback calls: {stats['python_fallback_hits']:,}; "
        f"feature dispatch: {stats.get('feature_dispatch_sec', 0):.3f}s"
        + (
            f"; feature stage: {stats['feature_computation_sec']:.3f}s"
            if feature_computation_sec is not None
            else ""
        )
        + (
            f"; Create Dataset wall: {stats['create_dataset_wall_sec']:.3f}s"
            if create_dataset_wall_sec is not None
            else ""
        )
    )
    logger.info(msg)
    if verbose:
        print(msg, flush=True)
    return stats


def _call_numba(
    name: str,
    kernel_fn: Callable[..., Any],
    fallback_fn: Callable[[], Any],
    *args: Any,
) -> Any:
    """Invoke a Numba kernel; on JIT failure activate Python fallback once."""
    try:
        out = kernel_fn(*args)
        _record_kernel_hit()
        return out
    except Exception as exc:  # pragma: no cover - depends on Numba breakage
        activate_python_fallback(f"JIT failed for {name}: {type(exc).__name__}: {exc}")
        _record_python_path_hit()
        return fallback_fn()


def warm_kernels(*, verbose: bool = True) -> dict[str, float]:
    """Force-compile hot kernels once; return per-kernel compile seconds.

    Prints/logs warm-up messages so first-run compile cost is visible.
    On Numba missing or JIT failure, activates Python fallback automatically.
    """
    global _WARMED, _CACHE_HITS, _WARM_MSG_PRINTED

    if python_fallback_active() or not has_numba():
        if not has_numba():
            activate_python_fallback(
                f"Numba unavailable ({python_fallback_reason() or 'not installed'})"
            )
        else:
            ensure_fallback_logged()
        _WARMED = True
        return {}

    explicit_off = _ENABLED_OVERRIDE is False or (
        _ENABLED_OVERRIDE is None and env_numba_flag() is False
    )
    if explicit_off:
        # Explicit OFF — skip compile; do not treat as auto-fallback.
        _WARMED = True
        if verbose:
            msg = "Feature Engine: Numba disabled via ARUNEO_FEATURE_NUMBA=off (skipping warm-up)."
            logger.info(msg)
            print(msg, flush=True)
        return {}

    if verbose and not _WARM_MSG_PRINTED:
        _WARM_MSG_PRINTED = True
        for line in ("Feature Engine warming...", "Compiling kernels..."):
            logger.info(line)
            print(line, flush=True)

    prices = np.linspace(100.0, 110.0, 64, dtype=np.float64)
    priors = np.linspace(0.1, 0.2, 32, dtype=np.float64)
    specs: list[tuple[str, object, tuple]] = [
        ("population_std", kernels.population_std_kernel, (prices[:20],)),
        ("ema_update", kernels.ema_update_kernel, (100.0, 101.0, 0.2)),
        ("ema_series", kernels.ema_series_kernel, (prices, 20)),
        ("rolling_mean_std", kernels.rolling_mean_std_kernel, (prices, 20)),
        ("rolling_max_min", kernels.rolling_max_min_kernel, (prices, 20)),
        ("pct_returns", kernels.pct_returns_kernel, (prices,)),
        ("safe_ratio", kernels.safe_ratio_kernel, (1.0, 2.0)),
        ("distance_pct", kernels.distance_pct_kernel, (101.0, 100.0)),
        ("iv_zscore", kernels.iv_zscore_kernel, (priors, 0.15)),
    ]
    cache_hits = 0
    for name, fn, args in specs:
        try:
            _, sec = timed_compile(fn, *args)
        except Exception as exc:
            activate_python_fallback(f"JIT failed for {name}: {type(exc).__name__}: {exc}")
            _WARMED = True
            return {}
        _COMPILE_SEC[name] = float(sec)
        if sec < _CACHE_HIT_THRESHOLD_SEC:
            cache_hits += 1
    _CACHE_HITS += cache_hits
    _WARMED = True
    if verbose:
        total = sum(_COMPILE_SEC.values())
        done = (
            f"Feature Engine warm-up complete "
            f"({total:.3f}s compile, {cache_hits}/{len(specs)} cache hits)."
        )
        logger.info(done)
        print(done, flush=True)
    return compile_timings()


# Public alias (docs / Feature Engine warming path).
warmup_kernels = warm_kernels


def _ensure_warm() -> None:
    """Pay compile cost once on first Numba use (visible warm-up)."""
    global _WARMED
    if _WARMED:
        return
    if not numba_enabled():
        # Still mark warmed when disabled so we don't spam; log fallback if needed.
        if python_fallback_active() or not has_numba():
            warm_kernels(verbose=True)
        else:
            _WARMED = True
        return
    warm_kernels(verbose=True)


def population_std(buffer: Sequence[float]) -> float:
    """Population std (ddof=0) matching ``np.std(..., ddof=0)``."""
    arr = kernels.buffer_to_float64(buffer)
    if arr.size == 0:
        return 0.0
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            return kernels.population_std_numpy(arr)
        return float(
            _call_numba(
                "population_std",
                kernels.population_std_kernel,
                lambda: kernels.population_std_numpy(arr),
                arr,
            )
        )
    _record_python_path_hit()
    return kernels.population_std_numpy(arr)


def ema_update(prev: float, price: float, alpha: float) -> float:
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            return float(price) * float(alpha) + float(prev) * (1.0 - float(alpha))
        return float(
            _call_numba(
                "ema_update",
                kernels.ema_update_kernel,
                lambda: float(price) * float(alpha) + float(prev) * (1.0 - float(alpha)),
                float(prev),
                float(price),
                float(alpha),
            )
        )
    _record_python_path_hit()
    return float(price) * float(alpha) + float(prev) * (1.0 - float(alpha))


def ema_series(prices: np.ndarray | Sequence[float], period: int) -> np.ndarray:
    arr = kernels.buffer_to_float64(prices)
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            return kernels.ema_series_python(arr, int(period))
        return _call_numba(
            "ema_series",
            kernels.ema_series_kernel,
            lambda: kernels.ema_series_python(arr, int(period)),
            arr,
            int(period),
        )
    _record_python_path_hit()
    return kernels.ema_series_python(arr, int(period))


def iv_zscore(priors: Sequence[float], iv: float, *, eps: float = 1e-8) -> float:
    if not priors:
        return 0.0
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            return kernels.iv_zscore_python(priors, float(iv), eps=float(eps))
        arr = kernels.buffer_to_float64(priors)
        return float(
            _call_numba(
                "iv_zscore",
                kernels.iv_zscore_kernel,
                lambda: kernels.iv_zscore_python(priors, float(iv), eps=float(eps)),
                arr,
                float(iv),
                float(eps),
            )
        )
    _record_python_path_hit()
    return kernels.iv_zscore_python(priors, float(iv), eps=float(eps))


def safe_ratio(numer: float, denom: float) -> float | None:
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            if denom <= 0:
                return None
            return float(numer) / float(denom)
        out = float(
            _call_numba(
                "safe_ratio",
                kernels.safe_ratio_kernel,
                lambda: float("nan") if denom <= 0 else float(numer) / float(denom),
                float(numer),
                float(denom),
            )
        )
        return None if out != out else out  # NaN check
    _record_python_path_hit()
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def distance_pct(price: float, ref: float) -> float | None:
    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            if ref == 0.0:
                return None
            return (float(price) - float(ref)) / float(ref) * 100.0
        out = float(
            _call_numba(
                "distance_pct",
                kernels.distance_pct_kernel,
                lambda: float("nan")
                if ref == 0.0
                else (float(price) - float(ref)) / float(ref) * 100.0,
                float(price),
                float(ref),
            )
        )
        return None if out != out else out
    _record_python_path_hit()
    if ref == 0.0:
        return None
    return (float(price) - float(ref)) / float(ref) * 100.0


def rolling_mean_std(arr: np.ndarray | Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    a = kernels.buffer_to_float64(arr)

    def _numpy_ref() -> tuple[np.ndarray, np.ndarray]:
        n = a.shape[0]
        means = np.full(n, np.nan, dtype=np.float64)
        stds = np.full(n, np.nan, dtype=np.float64)
        w = int(window)
        if w <= 0 or n < w:
            return means, stds
        for i in range(w - 1, n):
            window_vals = a[i - w + 1 : i + 1]
            means[i] = float(np.mean(window_vals))
            stds[i] = float(np.std(window_vals, ddof=0))
        return means, stds

    if numba_enabled():
        _ensure_warm()
        if not numba_enabled():
            _record_python_path_hit()
            return _numpy_ref()
        return _call_numba(
            "rolling_mean_std",
            kernels.rolling_mean_std_kernel,
            _numpy_ref,
            a,
            int(window),
        )
    _record_python_path_hit()
    return _numpy_ref()
