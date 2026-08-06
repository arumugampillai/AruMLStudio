"""Transformation registry — discover and look up available transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FeatureTransformation

_REGISTRY: dict[str, type] = {}
_BUILTINS_LOADED = False


def register_transformation(cls: type) -> type:
    """Class decorator / registrar for ``FeatureTransformation`` subclasses."""
    tid = str(getattr(cls, "id", "") or "").strip()
    if not tid:
        raise ValueError(f"Transformation {cls!r} missing id")
    _REGISTRY[tid] = cls
    return cls


def ensure_builtin_transformations() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import anchor_return as _anchor_return  # noqa: F401
    from . import derived as _derived  # noqa: F401
    from . import difference as _difference  # noqa: F401
    from . import difference_clip as _difference_clip  # noqa: F401
    from . import exponential_rolling as _exponential_rolling  # noqa: F401
    from . import interaction as _interaction  # noqa: F401
    from . import lag as _lag  # noqa: F401
    from . import math_transform as _math_transform  # noqa: F401
    from . import normalization as _normalization  # noqa: F401
    from . import ohlc_aggregation as _ohlc_aggregation  # noqa: F401
    from . import regime as _regime  # noqa: F401
    from . import return_transform as _return_transform  # noqa: F401
    from . import rolling as _rolling  # noqa: F401
    from . import rolling_ohlc as _rolling_ohlc  # noqa: F401
    from . import rolling_statistics as _rolling_statistics  # noqa: F401

    _BUILTINS_LOADED = True


def get_transformation(transformation_id: str) -> FeatureTransformation:
    ensure_builtin_transformations()
    tid = str(transformation_id or "").strip()
    cls = _REGISTRY.get(tid)
    if cls is None:
        raise KeyError(f"No transformation registered for id={transformation_id!r}")
    return cls()  # type: ignore[call-arg]


def list_registered_transformations() -> list[FeatureTransformation]:
    ensure_builtin_transformations()
    instances = [cls() for cls in _REGISTRY.values()]  # type: ignore[misc]
    instances.sort(key=lambda t: (int(getattr(t, "order", 100)), str(t.id)))
    return instances


def registered_transformation_count() -> int:
    ensure_builtin_transformations()
    return len(_REGISTRY)


def registered_transformation_ids() -> list[str]:
    ensure_builtin_transformations()
    return sorted(_REGISTRY.keys())
