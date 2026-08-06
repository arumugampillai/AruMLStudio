"""Inference registry session cache — load regression specs once per models-dir revision."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from chain_replay_ml.replay_feature_scoring import resolve_replay_feature_config
from chain_replay_ml.training.model_runtime import normalize_algorithm, resolve_production_model_path
from chain_replay_ml.training.paths import model_artifact_paths, models_dir
from chain_replay_ml.training.prediction_packages import (
    probability_ladder_slot,
    target_horizon,
)
from chain_replay_ml.training.registry import safe_model_name

from .registry_specs import merge_replay_configs, model_tier, union_feature_names


@dataclass
class RegistryLoadMeta:
    cache_hit: bool = False
    scan_ms: float = 0.0
    config_read_ms: float = 0.0
    total_ms: float = 0.0
    model_count: int = 0
    models_dir_signature: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_hit": self.cache_hit,
            "scan_ms": round(self.scan_ms, 3),
            "config_read_ms": round(self.config_read_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "model_count": self.model_count,
            "models_dir_signature": self.models_dir_signature,
        }


@dataclass
class _RegistryBundle:
    signature: tuple[Any, ...]
    specs: list[dict[str, Any]]
    merged_config: dict[str, Any]
    union_features: list[str]


_registry_cache: dict[tuple[Any, ...], _RegistryBundle] = {}


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def models_dir_signature(data_dir: str) -> tuple[Any, ...]:
    base = models_dir(data_dir)
    if not os.path.isdir(base):
        return (0.0, 0, "")
    try:
        dir_mtime = os.path.getmtime(base)
    except OSError:
        dir_mtime = 0.0
    try:
        names = sorted(
            e for e in os.listdir(base)
            if not e.startswith(".") and os.path.isdir(os.path.join(base, e))
        )
    except OSError:
        names = []
    name_hash = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:16]
    return (round(dir_mtime, 3), len(names), name_hash)


def clear_inference_registry_cache() -> None:
    _registry_cache.clear()


def inference_registry_cache_stats() -> dict[str, Any]:
    return {
        "cached_bundles": len(_registry_cache),
        "keys": ["|".join(str(x) for x in k) for k in list(_registry_cache.keys())[:8]],
    }


def _metric_from_docs(*docs: dict[str, Any], key: str) -> float | None:
    for doc in docs:
        for src in (doc.get("metrics") or {}, doc.get("production_metrics") or {}, doc):
            val = src.get(key) if isinstance(src, dict) else None
            if val is None:
                continue
            try:
                return round(float(val), 4)
            except (TypeError, ValueError):
                continue
    return None


def _load_inference_package(
    data_dir: str,
    entry: str,
    *,
    read_ms: list[float],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Single-pass minimal read for inference — no directory walks or CSV parsing."""
    safe = safe_model_name(entry)
    paths = model_artifact_paths(data_dir, safe)
    if not os.path.isfile(paths["config_json"]):
        return None

    t0 = time.perf_counter()
    config = _load_json(paths["config_json"])
    registry_doc = _load_json(paths["registry_json"])
    metrics_doc = _load_json(paths["metrics_json"])
    production_name = ""
    if os.path.isfile(paths.get("training_metadata_json", "")):
        tmeta = _load_json(paths["training_metadata_json"])
        production_name = str(tmeta.get("production_model") or "").strip()
    read_ms[0] += (time.perf_counter() - t0) * 1000.0

    prediction_type = str(config.get("prediction_type") or "regression").strip().lower()
    tier = model_tier(config)
    if prediction_type in ("binary", "classification", "multiclass"):
        tier = "classification"
    else:
        tier = tier or "regression"
    target = str(config.get("target") or registry_doc.get("target") or "")
    if tier == "classification" and probability_ladder_slot(target) is None:
        # The package response currently supports the canonical six-rung ladder.
        return None

    features = list(config.get("features") or config.get("selected_features") or [])
    if not features or not target:
        return None

    algorithm = normalize_algorithm(config.get("algorithm"))
    model_path = resolve_production_model_path(
        paths["package_dir"],
        algorithm=algorithm,
        production_name=production_name or None,
    )
    if not model_path:
        return None

    replay_config, _source = resolve_replay_feature_config(data_dir, config)
    status = str(registry_doc.get("status") or config.get("status") or "ready").strip().lower()
    registry_row = {
        "model_name": safe,
        "status": status,
        "target": target,
        "dataset": config.get("dataset") or registry_doc.get("dataset"),
        "prediction_type": prediction_type,
        "algorithm": config.get("algorithm") or registry_doc.get("algorithm"),
        "mae": _metric_from_docs(metrics_doc, registry_doc, key="mae"),
        "rmse": _metric_from_docs(metrics_doc, registry_doc, key="rmse"),
        "metrics": {
            "mae": _metric_from_docs(metrics_doc, registry_doc, key="mae"),
            "rmse": _metric_from_docs(metrics_doc, registry_doc, key="rmse"),
        },
        "production_metrics": dict(registry_doc.get("production_metrics") or {}),
    }
    spec = {
        "registry": registry_row,
        "model_name": safe,
        "tier": tier,
        "features": features,
        "meta_features": list(config.get("meta_features") or []),
        "target": target,
        "dataset": config.get("dataset") or registry_doc.get("dataset"),
        "prediction_type": prediction_type,
        "probability_ladder_slot": probability_ladder_slot(target),
        "algorithm": algorithm,
        "model_path": model_path,
        "replay_config": replay_config,
        "mae": registry_row["mae"],
        "rmse": registry_row["rmse"],
        "feature_version": str(config.get("feature_version") or ""),
    }
    return registry_row, spec


def _cold_load_inference_registry(
    data_dir: str,
    *,
    status_filter: str,
    model_name: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], RegistryLoadMeta]:
    meta = RegistryLoadMeta()
    base = models_dir(data_dir)
    if not os.path.isdir(base):
        return [], {}, [], meta

    want_status = str(status_filter or "").strip().lower()
    want_name = str(model_name or "").strip()
    read_ms = [0.0]

    t_scan0 = time.perf_counter()
    try:
        entries = sorted(
            e for e in os.listdir(base)
            if not e.startswith(".") and os.path.isdir(os.path.join(base, e))
        )
    except OSError:
        entries = []
    meta.scan_ms = round((time.perf_counter() - t_scan0) * 1000.0, 3)

    loaded_specs: list[dict[str, Any]] = []
    for entry in entries:
        loaded = _load_inference_package(data_dir, entry, read_ms=read_ms)
        if loaded is None:
            continue
        _row, spec = loaded
        row_status = str(spec["registry"].get("status") or "").lower()
        if want_status and row_status != want_status:
            continue
        loaded_specs.append(spec)

    regression_specs = [
        spec for spec in loaded_specs if str(spec.get("tier") or "") == "regression"
    ]
    classification_specs = [
        spec for spec in loaded_specs if str(spec.get("tier") or "") == "classification"
    ]
    if want_name:
        anchors = [
            spec
            for spec in regression_specs
            if str(spec.get("model_name") or "") == want_name
        ]
    else:
        anchors = regression_specs

    identities = {
        (str(spec.get("dataset") or ""), target_horizon(spec.get("target")))
        for spec in anchors
    }
    members = [
        spec
        for spec in classification_specs
        if (str(spec.get("dataset") or ""), target_horizon(spec.get("target")))
        in identities
    ]
    specs = anchors + members

    meta.config_read_ms = round(read_ms[0], 3)
    meta.model_count = len(specs)
    merged = merge_replay_configs(specs)
    union = union_feature_names(specs)
    return specs, merged, union, meta


def acquire_inference_registry(
    data_dir: str,
    *,
    status_filter: str = "ready",
    model_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], RegistryLoadMeta]:
    """Return cached regression specs + merged config for inference."""
    t0 = time.perf_counter()
    signature = models_dir_signature(data_dir)
    cache_key = (os.path.abspath(data_dir), str(status_filter or "").lower(), str(model_name or ""))
    meta = RegistryLoadMeta(models_dir_signature="|".join(str(x) for x in signature))

    cached = _registry_cache.get(cache_key)
    if cached is not None and cached.signature == signature:
        meta.cache_hit = True
        meta.model_count = len(cached.specs)
        meta.total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return cached.specs, cached.merged_config, cached.union_features, meta

    specs, merged, union, cold_meta = _cold_load_inference_registry(
        data_dir,
        status_filter=status_filter,
        model_name=model_name,
    )
    _registry_cache[cache_key] = _RegistryBundle(
        signature=signature,
        specs=specs,
        merged_config=merged,
        union_features=union,
    )
    cold_meta.total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    cold_meta.models_dir_signature = meta.models_dir_signature
    return specs, merged, union, cold_meta
