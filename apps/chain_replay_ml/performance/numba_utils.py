"""Numba availability helpers and safe ``njit`` wrapper.

Auto-fallback: when Numba is missing or JIT compile/call fails, Python paths
are used automatically. Explicit ``ARUNEO_FEATURE_NUMBA=off`` is still honored.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_HAS_NUMBA = False
_NUMBA_IMPORT_ERROR: str | None = None

# Forced Python path after missing install or JIT failure (not env OFF).
_PYTHON_FALLBACK = False
_FALLBACK_REASON: str | None = None
_FALLBACK_LOGGED = False

try:
    from numba import njit as _numba_njit  # type: ignore

    _HAS_NUMBA = True
except Exception as exc:  # pragma: no cover - optional dependency
    _numba_njit = None  # type: ignore
    _NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    _PYTHON_FALLBACK = True
    _FALLBACK_REASON = f"Numba unavailable ({_NUMBA_IMPORT_ERROR})"


def has_numba() -> bool:
    """True when the ``numba`` package imported successfully."""
    return bool(_HAS_NUMBA)


def numba_import_error() -> str | None:
    return _NUMBA_IMPORT_ERROR


def env_numba_flag() -> bool | None:
    """Return True/False if ``ARUMLSTUDIO_FEATURE_NUMBA`` (or ``ARUNEO_FEATURE_NUMBA``) is set, else None (use default)."""
    raw = (
        os.environ.get("ARUMLSTUDIO_FEATURE_NUMBA")
        or os.environ.get("ARUNEO_FEATURE_NUMBA")
        or ""
    ).strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "on", "yes"}:
        return True
    if raw in {"0", "false", "off", "no"}:
        return False
    return None


def python_fallback_active() -> bool:
    """True when Numba is missing or JIT failed (auto Python path)."""
    return bool(_PYTHON_FALLBACK)


def python_fallback_reason() -> str | None:
    return _FALLBACK_REASON


def activate_python_fallback(reason: str) -> None:
    """Force Python implementations and log once."""
    global _PYTHON_FALLBACK, _FALLBACK_REASON
    _PYTHON_FALLBACK = True
    if not _FALLBACK_REASON:
        _FALLBACK_REASON = reason
    ensure_fallback_logged()


def ensure_fallback_logged() -> None:
    """Emit a single clear log/print that Python fallback is active."""
    global _FALLBACK_LOGGED
    if _FALLBACK_LOGGED or not _PYTHON_FALLBACK:
        return
    _FALLBACK_LOGGED = True
    msg = (
        "Feature Engine: Python fallback active"
        + (f" — {_FALLBACK_REASON}" if _FALLBACK_REASON else "")
        + ". Feature values unchanged; Numba speedup disabled."
    )
    logger.warning(msg)
    print(msg, flush=True)


def reset_python_fallback_for_tests() -> None:
    """Test helper: clear JIT-failure fallback (keeps import-time state)."""
    global _PYTHON_FALLBACK, _FALLBACK_REASON, _FALLBACK_LOGGED
    if _NUMBA_IMPORT_ERROR is not None:
        # Import failed — stay in fallback.
        _PYTHON_FALLBACK = True
        _FALLBACK_REASON = f"Numba unavailable ({_NUMBA_IMPORT_ERROR})"
        _FALLBACK_LOGGED = False
        return
    _PYTHON_FALLBACK = False
    _FALLBACK_REASON = None
    _FALLBACK_LOGGED = False


def njit(**kwargs: Any) -> Callable[[F], F]:
    """Decorator: Numba ``njit`` when available, otherwise identity (pure Python)."""

    def decorator(fn: F) -> F:
        if not _HAS_NUMBA or _numba_njit is None:
            activate_python_fallback(
                _FALLBACK_REASON or f"Numba unavailable ({_NUMBA_IMPORT_ERROR})"
            )
            return fn
        # Prefer cache=True for stable kernels; caller may override.
        opts = {"cache": True, "nogil": True}
        opts.update(kwargs)
        return _numba_njit(**opts)(fn)  # type: ignore[misc]

    return decorator


def ensure_compiled(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call once to trigger Numba compile; returns the call result."""
    return fn(*args, **kwargs)


def timed_compile(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Return ``(result, compile_or_call_seconds)`` for first invocation."""
    import time

    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
