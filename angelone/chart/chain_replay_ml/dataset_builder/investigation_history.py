"""Persist audit investigation history per dataset."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .writer import _safe_filename, datasets_dir


def investigation_history_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.investigation-history.json")


def load_investigation_history(data_dir: str, dataset_name: str) -> dict[str, Any]:
    path = investigation_history_path(data_dir, _safe_filename(dataset_name))
    if not os.path.isfile(path):
        return {"dataset_name": _safe_filename(dataset_name), "investigations": []}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"dataset_name": _safe_filename(dataset_name), "investigations": []}
    doc.setdefault("investigations", [])
    return doc


def append_investigation(
    data_dir: str,
    dataset_name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Append investigation record; returns full history doc."""
    safe_name = _safe_filename(dataset_name)
    doc = load_investigation_history(data_dir, safe_name)
    investigations: list[dict[str, Any]] = list(doc.get("investigations") or [])
    next_id = max((int(r.get("id") or 0) for r in investigations), default=0) + 1
    entry = {
        "id": next_id,
        "investigated_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        **record,
    }
    investigations.append(entry)
    doc["dataset_name"] = safe_name
    doc["investigations"] = investigations
    doc["updated_at"] = entry["investigated_at"]
    path = investigation_history_path(data_dir, safe_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def update_investigation_status(
    data_dir: str,
    dataset_name: str,
    investigation_id: int,
    status: str,
) -> dict[str, Any]:
    doc = load_investigation_history(data_dir, dataset_name)
    for row in doc.get("investigations") or []:
        if int(row.get("id") or 0) == int(investigation_id):
            row["status"] = status
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            break
    path = investigation_history_path(data_dir, _safe_filename(dataset_name))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc
