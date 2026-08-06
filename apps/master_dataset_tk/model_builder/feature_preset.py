"""Feature preset handoff — Tk parity with web ml_model_builder_feature_preset."""

from __future__ import annotations

import json
import os
import time
from typing import Any

PRESET_STORAGE = "ml_model_builder_feature_preset_tk.json"


def preset_storage_path(chart_dir: str) -> str:
    return os.path.join(chart_dir, "data", PRESET_STORAGE)


def save_feature_preset(
    chart_dir: str,
    *,
    features: list[str],
    dataset: str | None = None,
    source_model: str | None = None,
    analysis_feature_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    doc: dict[str, Any] = {
        "features": feats,
        "dataset": str(dataset).strip() if dataset else None,
        "source_model": str(source_model).strip() if source_model else None,
        "at": int(time.time() * 1000),
    }
    if isinstance(analysis_feature_selection, dict) and analysis_feature_selection:
        doc["analysis_feature_selection"] = dict(analysis_feature_selection)
    path = preset_storage_path(chart_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def load_feature_preset(chart_dir: str) -> dict[str, Any] | None:
    path = preset_storage_path(chart_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    feats = doc.get("features")
    if not isinstance(feats, list) or not feats:
        return None
    return doc


def clear_feature_preset(chart_dir: str) -> None:
    path = preset_storage_path(chart_dir)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def apply_feature_preset(
    preset: dict[str, Any] | None,
    *,
    dataset_name: str,
    dataset_feature_names: list[str],
) -> dict[str, Any]:
    """Intersect preset features with dataset columns; return apply result."""
    if not preset:
        return {"applied": False, "requested_count": 0, "applied_count": 0}
    requested = [str(f).strip() for f in (preset.get("features") or []) if str(f).strip()]
    allowed = set(dataset_feature_names) if dataset_feature_names else set(requested)
    applied = {f for f in requested if f in allowed}
    preset_ds = str(preset.get("dataset") or "").strip()
    return {
        "applied": bool(applied),
        "pending": bool(requested) and not applied,
        "requested_count": len(requested),
        "applied_count": len(applied),
        "features": sorted(applied),
        "dataset": preset_ds or None,
        "dataset_match": not preset_ds or preset_ds == dataset_name,
        "source_model": preset.get("source_model"),
        "analysis_feature_selection": (
            dict(preset["analysis_feature_selection"])
            if isinstance(preset.get("analysis_feature_selection"), dict)
            else None
        ),
    }
