"""Registry model specs for inference tiers."""

from __future__ import annotations

import copy
from typing import Any

from chain_replay_ml.replay_feature_scoring import load_model_inference_config
from chain_replay_ml.training.registry import list_trained_models


def _metric_from_registry(row: dict[str, Any], key: str) -> float | None:
    for src in (row.get("metrics") or {}, row.get("production_metrics") or {}):
        val = src.get(key)
        if val is None:
            continue
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            continue
    return None


def model_tier(config: dict[str, Any] | None) -> str:
    cfg = config or {}
    return str(cfg.get("tier") or "regression").strip().lower()


def load_regression_specs(
    data_dir: str,
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in registry:
        name = str(row.get("model_name") or "")
        if not name:
            continue
        loaded = load_model_inference_config(data_dir, name)
        if not loaded:
            continue
        config = loaded.get("config") or {}
        tier = model_tier(config)
        if tier not in ("regression", ""):
            continue
        specs.append({
            "registry": row,
            "model_name": name,
            "tier": tier or "regression",
            "features": list(loaded.get("features") or []),
            "meta_features": list(config.get("meta_features") or []),
            "target": str(loaded.get("target") or row.get("target") or ""),
            "algorithm": loaded.get("algorithm"),
            "model_path": loaded.get("model_path"),
            "replay_config": loaded.get("replay_config") or {},
            "mae": _metric_from_registry(row, "mae"),
            "rmse": _metric_from_registry(row, "rmse"),
            "feature_version": str(config.get("feature_version") or ""),
        })
    return specs


def filter_registry(
    data_dir: str,
    *,
    status_filter: str = "ready",
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    registry = list_trained_models(data_dir, lightweight=False)
    if status_filter:
        want = status_filter.lower()
        registry = [row for row in registry if str(row.get("status") or "").lower() == want]
    if model_name:
        want_name = str(model_name).strip()
        registry = [row for row in registry if str(row.get("model_name") or "") == want_name]
    return registry


def union_feature_names(specs: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for spec in specs:
        for feat in spec.get("features") or []:
            name = str(feat)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def merge_replay_configs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge feature groups from all models so the union superset can be built once."""
    if not specs:
        return {}
    anchor = max(specs, key=lambda s: len(s.get("features") or []))
    merged = copy.deepcopy(anchor.get("replay_config") or {})
    groups: set[str] = set()
    for key in ("feature_groups_implemented", "feature_groups"):
        for spec in specs:
            cfg = spec.get("replay_config") or {}
            for gid in cfg.get(key) or []:
                groups.add(str(gid))
        for gid in merged.get(key) or []:
            groups.add(str(gid))
    group_list = sorted(groups)
    merged["feature_groups_implemented"] = group_list
    merged["feature_groups"] = group_list
    return merged
