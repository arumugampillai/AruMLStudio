"""Persist Strategy Simulator UI selections across relaunches."""

from __future__ import annotations

import json
import os
import time
from typing import Any

STORAGE = "ml_strategy_sim_prefs_tk.json"


def storage_path(chart_dir: str) -> str:
    return os.path.join(chart_dir, "data", STORAGE)


def load_strategy_sim_prefs(chart_dir: str) -> dict[str, Any]:
    path = storage_path(chart_dir)
    if not chart_dir or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_strategy_sim_prefs(chart_dir: str, patch: dict[str, Any]) -> dict[str, Any]:
    if not chart_dir:
        return {}
    existing = load_strategy_sim_prefs(chart_dir)
    doc = dict(existing)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(doc.get(key), dict):
            doc[key] = {**doc[key], **val}
        else:
            doc[key] = val
    doc["version"] = 1
    doc["at"] = int(time.time() * 1000)
    path = storage_path(chart_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc
