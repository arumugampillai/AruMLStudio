"""Stable hashes for prediction run provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(metadata: dict[str, Any] | None, dataset_name: str) -> str:
    meta = metadata or {}
    payload = {
        "dataset": dataset_name,
        "dataset_version": meta.get("dataset_version") or meta.get("builder_version"),
        "row_count": meta.get("row_count"),
        "feature_count": meta.get("feature_count"),
        "days": meta.get("days") or meta.get("trading_days"),
    }
    return _hash_obj(payload)


def feature_snapshot_hash(features: list[str]) -> str:
    return _hash_obj(sorted(features))


def walk_forward_config_hash(wf_cfg: dict[str, Any]) -> str:
    keys = (
        "n_folds", "window_mode", "fold_placement", "train_window_size", "validation_window_size",
        "test_holdout_rows", "feature_selection_method", "optimization_metric",
    )
    return _hash_obj({k: wf_cfg.get(k) for k in keys if k in wf_cfg})


def training_config_hash(config_dict: dict[str, Any]) -> str:
    payload = {
        "algorithm": config_dict.get("algorithm"),
        "target": config_dict.get("target"),
        "prediction_type": config_dict.get("prediction_type"),
        "split": config_dict.get("split"),
        "parameters": config_dict.get("parameters"),
    }
    return _hash_obj(payload)
