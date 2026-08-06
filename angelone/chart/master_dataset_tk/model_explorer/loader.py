"""Load trained XGBoost boosters from disk or in-memory objects."""

from __future__ import annotations

import os
import pickle
from typing import Any

SUPPORTED_EXTENSIONS = (".pkl", ".pickle", ".json", ".bst", ".ubj", ".model")


class LoadError(ValueError):
    """Raised when a path or object cannot be resolved to an xgboost.Booster."""


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise LoadError("xgboost is not installed") from exc
    return xgb


def _as_booster(obj: Any):
    """Normalize Booster / sklearn wrapper / dict payload to Booster."""
    xgb = _require_xgboost()
    if obj is None:
        raise LoadError("model object is None")
    if isinstance(obj, xgb.Booster):
        return obj
    get_booster = getattr(obj, "get_booster", None)
    if callable(get_booster):
        bst = get_booster()
        if isinstance(bst, xgb.Booster):
            return bst
    if isinstance(obj, dict):
        for key in ("booster", "model", "bst", "xgb_model"):
            if key in obj:
                return _as_booster(obj[key])
    raise LoadError(
        f"unsupported model type {type(obj).__name__!r}; "
        "expected xgboost.Booster or object with get_booster()"
    )


def load_booster(source: str | os.PathLike[str] | Any):
    """Load an xgboost.Booster from a path or return a normalized in-memory booster.

    Supported files: ``.pkl`` / ``.pickle`` (pickle of Booster or sklearn wrapper),
    ``.json`` / ``.bst`` / ``.ubj`` / ``.model`` (native ``Booster.load_model``).
    """
    xgb = _require_xgboost()
    if not isinstance(source, (str, os.PathLike)):
        return _as_booster(source)

    path = os.path.abspath(os.fspath(source))
    if not os.path.isfile(path):
        raise LoadError(f"model file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            raise LoadError(f"failed to unpickle {path}: {exc}") from exc
        return _as_booster(obj)

    # Native XGBoost formats (json / bst / ubj / model) and unknown extensions:
    # try load_model first; fall back to pickle for extension-less dumps.
    try:
        bst = xgb.Booster()
        bst.load_model(path)
        return bst
    except Exception as native_exc:
        if ext in (".json", ".bst", ".ubj", ".model"):
            raise LoadError(f"failed to load XGBoost model {path}: {native_exc}") from native_exc
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            return _as_booster(obj)
        except Exception as pickle_exc:
            raise LoadError(
                f"failed to load model {path} "
                f"(native: {native_exc}; pickle: {pickle_exc})"
            ) from pickle_exc
