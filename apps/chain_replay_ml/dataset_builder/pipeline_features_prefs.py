"""Persist permanently deleted Pipeline Features (Auto Feature Transformation).

Deleted names are excluded from the Pipeline catalogue and from Analysis Dataset
transformation configs. Storage lives under the chart data directory.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Sequence

STORAGE = "pipeline_features_retired.json"


def storage_path(data_dir: str) -> str:
    root = str(data_dir or "").strip()
    if not root:
        return ""
    return os.path.join(root, STORAGE)


def load_retired_pipeline_features(data_dir: str) -> frozenset[str]:
    path = storage_path(data_dir)
    if not path or not os.path.isfile(path):
        return frozenset()
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return frozenset()
    raw = doc.get("retired") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(n).strip() for n in raw if str(n).strip())


def save_retired_pipeline_features(data_dir: str, names: Iterable[str]) -> frozenset[str]:
    path = storage_path(data_dir)
    if not path:
        raise ValueError("data_dir is required")
    cleaned = sorted({str(n).strip() for n in names if str(n).strip()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "retired": cleaned,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return frozenset(cleaned)


def retire_pipeline_features(data_dir: str, names: Sequence[str]) -> frozenset[str]:
    """Permanently remove ``names`` from the active Pipeline catalogue."""
    current = set(load_retired_pipeline_features(data_dir))
    current.update(str(n).strip() for n in names if str(n).strip())
    return save_retired_pipeline_features(data_dir, current)


def active_pipeline_feature_names(
    all_names: Sequence[str],
    *,
    data_dir: str | None = None,
    retired: frozenset[str] | None = None,
) -> list[str]:
    skip = retired if retired is not None else (
        load_retired_pipeline_features(data_dir) if data_dir else frozenset()
    )
    return [n for n in all_names if n not in skip]


__all__ = [
    "STORAGE",
    "active_pipeline_feature_names",
    "load_retired_pipeline_features",
    "retire_pipeline_features",
    "save_retired_pipeline_features",
    "storage_path",
]
