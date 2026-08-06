"""Algorithm-native model filenames and inference helpers."""

from __future__ import annotations

import os
import time
from typing import Any

from .trainers import NATIVE_EXTENSIONS, normalize_algorithm_id

NATIVE_MODEL_EXTENSIONS: dict[str, str] = dict(NATIVE_EXTENSIONS)

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_MTIME: dict[str, float] = {}


def normalize_algorithm(algorithm: str | None) -> str:
    return normalize_algorithm_id(algorithm)


def native_model_extension(algorithm: str | None) -> str:
    return NATIVE_MODEL_EXTENSIONS.get(normalize_algorithm(algorithm), "ubj")


def native_model_basename(role: str, algorithm: str | None) -> str:
    """role: model | baseline_model | tuned_model"""
    ext = native_model_extension(algorithm)
    return f"{role}.{ext}"


def native_model_path(package_dir: str, role: str, algorithm: str | None) -> str:
    return os.path.join(package_dir, native_model_basename(role, algorithm))


def resolve_production_model_path(
    package_dir: str,
    *,
    algorithm: str | None,
    production_name: str | None = None,
) -> str:
    if production_name:
        candidate = os.path.join(package_dir, production_name)
        if os.path.isfile(candidate):
            return candidate
    algo = normalize_algorithm(algorithm)
    for role in ("model", "tuned_model", "baseline_model"):
        path = native_model_path(package_dir, role, algo)
        if os.path.isfile(path):
            return path
    # Legacy XGBoost JSON fallback
    legacy_json = os.path.join(package_dir, "model.json")
    if os.path.isfile(legacy_json):
        return legacy_json
    legacy_ubj = os.path.join(package_dir, "model.ubj")
    if os.path.isfile(legacy_ubj):
        return legacy_ubj
    legacy_joblib = os.path.join(package_dir, "model.joblib")
    if os.path.isfile(legacy_joblib):
        return legacy_joblib
    return ""


def _load_json_doc(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        import json

        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def resolve_prediction_model_package(data_dir: str, model_name: str) -> dict[str, Any]:
    """
    Resolve a named on-disk model package to its production model artifact plus
    training config metadata (algorithm, features, label_run_id).

    Reuses the same package-discovery pattern as prediction-package ladder
    members (config.json for algorithm/features, training_metadata.json for
    the champion "production_model" filename).

    Returns ``{"ok": True, "model_path", "algorithm", "features", "label_run_id"}``
    on success, or ``{"ok": False, "error": ...}`` otherwise.
    """
    from .paths import model_artifact_paths

    name = str(model_name or "").strip()
    if not name:
        return {"ok": False, "error": "Model name is empty"}

    paths = model_artifact_paths(data_dir, name)
    config = _load_json_doc(paths["config_json"])
    if not config:
        return {"ok": False, "error": f"Model config not found for '{name}'"}

    algorithm = config.get("algorithm")
    features = [
        str(f)
        for f in (config.get("features") or config.get("selected_features") or [])
        if str(f or "").strip()
    ]
    label_run_id = str(
        config.get("label_run_id") or config.get("label_strategy") or "triple_barrier"
    ).strip()

    production_name = str(
        _load_json_doc(paths["training_metadata_json"]).get("production_model") or ""
    ).strip()
    model_path = resolve_production_model_path(
        paths["package_dir"],
        algorithm=algorithm,
        production_name=production_name or None,
    )
    if not model_path or not os.path.isfile(model_path):
        return {
            "ok": False,
            "error": f"Model artifact not found for '{name}'",
            "algorithm": algorithm,
            "features": features,
            "label_run_id": label_run_id,
        }

    return {
        "ok": True,
        "model_path": model_path,
        "algorithm": algorithm,
        "features": features,
        "label_run_id": label_run_id,
    }


def load_prediction_model(model_path: str, algorithm: str | None) -> Any:
    """Load a trained regressor for batch predict on a feature matrix."""
    algo = normalize_algorithm(algorithm)
    path = str(model_path or "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if algo == "lightgbm":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=path)
        return _LgbPredictor(booster)

    if algo == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(path)
        return model

    if algo in ("random_forest", "extra_trees"):
        from .trainers.random_forest import load_rf_model

        return load_rf_model(path)

    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(path)
    return model


def load_prediction_model_cached(model_path: str, algorithm: str | None) -> tuple[Any, float, bool]:
    """
    Return (model, load_ms, loaded_from_disk).
    Cached models return load_ms=0 and loaded_from_disk=False.
    """
    path = str(model_path or "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    mtime = os.path.getmtime(path)
    cached = _MODEL_CACHE.get(path)
    if cached is not None and _MODEL_CACHE_MTIME.get(path) == mtime:
        return cached, 0.0, False

    t0 = time.perf_counter()
    model = load_prediction_model(path, algorithm)
    load_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    _MODEL_CACHE[path] = model
    _MODEL_CACHE_MTIME[path] = mtime
    return model, load_ms, True


def clear_prediction_model_cache() -> None:
    _MODEL_CACHE.clear()
    _MODEL_CACHE_MTIME.clear()


def prediction_model_cache_stats() -> dict[str, Any]:
    return {
        "cached_models": len(_MODEL_CACHE),
        "paths": list(_MODEL_CACHE.keys()),
    }


class _LgbPredictor:
    def __init__(self, booster: Any) -> None:
        self._booster = booster

    def predict(self, X_df) -> Any:
        import numpy as np

        return np.asarray(self._booster.predict(X_df))

    def save_model(self, path: str) -> None:
        self._booster.save_model(path)
