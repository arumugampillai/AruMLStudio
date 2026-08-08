"""Registry feature export selection for Analysis Dataset builds (Phase 1A).

Unselected registry features remain on the Master table for Pipeline transforms;
only the final Analysis Dataset Parquet omits them.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from .feature_sources_catalog import registry_feature_names

STORAGE = "registry_features_export.json"
MODE_ALL = "all"
MODE_CUSTOM = "custom"


def storage_path(data_dir: str) -> str:
    root = str(data_dir or "").strip()
    if not root:
        return ""
    return os.path.join(root, STORAGE)


def _load_doc(data_dir: str) -> dict:
    path = storage_path(data_dir)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def load_registry_export_mode(data_dir: str) -> str:
    mode = str(_load_doc(data_dir).get("mode") or MODE_ALL).strip().lower()
    return mode if mode in (MODE_ALL, MODE_CUSTOM) else MODE_ALL


def load_registry_export_selected_names(data_dir: str) -> frozenset[str]:
    """Selected registry names when mode is custom; empty if unset."""
    doc = _load_doc(data_dir)
    raw = doc.get("selected") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        return frozenset()
    canonical = set(registry_feature_names(data_dir=data_dir))
    return frozenset(str(n).strip() for n in raw if str(n).strip() in canonical)


def save_registry_export_selection(
    data_dir: str,
    *,
    selected: Iterable[str],
    mode: str = MODE_CUSTOM,
) -> frozenset[str]:
    path = storage_path(data_dir)
    if not path:
        raise ValueError("data_dir is required")
    canonical = set(registry_feature_names(data_dir=data_dir))
    cleaned = sorted({str(n).strip() for n in selected if str(n).strip() in canonical})
    mode_norm = str(mode or MODE_CUSTOM).strip().lower()
    if mode_norm == MODE_ALL or len(cleaned) >= len(canonical):
        payload = {"version": 1, "mode": MODE_ALL, "selected": sorted(canonical)}
    else:
        payload = {"version": 1, "mode": MODE_CUSTOM, "selected": cleaned}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return frozenset(payload["selected"])


def resolve_registry_export_features(data_dir: str) -> frozenset[str]:
    """Registry feature names to include in the Analysis Dataset export."""
    all_names = frozenset(registry_feature_names(data_dir=data_dir))
    if load_registry_export_mode(data_dir) != MODE_CUSTOM:
        return all_names
    selected = load_registry_export_selected_names(data_dir)
    return selected if selected else all_names


def registry_export_selection_summary(data_dir: str) -> dict[str, int | str]:
    """Counts for UI: total, selected, mode."""
    total = len(registry_feature_names(data_dir=data_dir))
    selected = resolve_registry_export_features(data_dir)
    mode = load_registry_export_mode(data_dir)
    return {
        "mode": mode,
        "total": total,
        "selected": len(selected),
    }


__all__ = [
    "MODE_ALL",
    "MODE_CUSTOM",
    "STORAGE",
    "load_registry_export_mode",
    "load_registry_export_selected_names",
    "registry_export_selection_summary",
    "resolve_registry_export_features",
    "save_registry_export_selection",
    "storage_path",
]
