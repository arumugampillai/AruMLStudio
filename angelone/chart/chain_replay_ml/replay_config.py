"""Replay feature-build configuration (independent of training package imports)."""

from __future__ import annotations

import json
import os
from typing import Any

from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir


def load_dataset_metadata_json(data_dir: str, dataset_name: str) -> dict[str, Any]:
    safe = _safe_filename(dataset_name)
    meta_path = os.path.join(datasets_dir(data_dir), f"{safe}.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def build_replay_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Snapshot of dataset build settings needed to reproduce features on replay."""
    cfg = metadata.get("dataset_configuration") or {}
    return {
        "market": str(metadata.get("market") or "NIFTY"),
        "sampling": dict(metadata.get("sampling") or {}),
        "strike_selection": dict(metadata.get("strike_selection") or {}),
        "feature_groups_implemented": list(
            metadata.get("feature_groups_implemented") or metadata.get("feature_groups") or []
        ),
        "dataset_configuration": dict(cfg),
        "prediction_target_columns": list(metadata.get("prediction_target_columns") or []),
        "lookback_policy": dict(cfg.get("lookback_policy") or metadata.get("lookback_policy") or {}),
    }
