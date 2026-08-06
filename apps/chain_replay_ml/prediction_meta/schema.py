"""Prediction meta SQLite schema — samples columns for UI copy/export."""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from .store import PredictionMetaStore

_GROUP_ORDER = (
    "primary_key",
    "identity",
    "record",
    "context",
    "ensemble",
    "model_predictions",
    "model_meta",
    "outcomes",
    "other",
)

_GROUP_LABELS = {
    "primary_key": "Primary key",
    "identity": "Identity",
    "record": "Build / registry",
    "context": "Market context",
    "ensemble": "Ensemble",
    "model_predictions": "Model predictions",
    "model_meta": "Per-model meta",
    "outcomes": "Outcomes / quality",
    "other": "Other",
}


def _column_group(name: str) -> str:
    if name == "prediction_id":
        return "primary_key"
    if name in {
        "trading_day", "timestamp", "token", "strike", "option_type", "symbol", "market", "expiry",
    }:
        return "identity"
    if name in {
        "prediction_timestamp", "feature_version", "prediction_version",
        "model_registry_version", "prediction_time_ms",
    }:
        return "record"
    if name in {"current_ltp", "current_spot", "minutes_to_expiry"}:
        return "context"
    if name.startswith("ensemble_") or name in {
        "agreement", "models_ok", "models_failed",
        "prediction_min", "prediction_max", "prediction_range_pct",
        "mean_minus_current_ltp", "median_minus_current_ltp",
        "prediction_velocity", "prediction_acceleration", "prediction_trend",
    }:
        return "ensemble"
    if re.match(r"model_\d+_pred$", name):
        return "model_predictions"
    if re.match(r"model_\d+_(delta_from_mean|rank)$", name):
        return "model_meta"
    if name.startswith("actual_") or name in {
        "prediction_error", "direction_correct",
        "ticks_above_entry_5m", "ticks_below_entry_5m",
        "time_to_max_profit", "time_to_max_drawdown",
    }:
        return "outcomes"
    return "other"


def read_prediction_samples_schema(db_path: str) -> dict[str, Any]:
    if not os.path.isfile(db_path):
        return {"exists": False, "db_path": db_path}

    with PredictionMetaStore(db_path) as store:
        conn = store.conn
        tables = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        rows = conn.execute("PRAGMA table_info(samples)").fetchall()

    columns: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {g: [] for g in _GROUP_ORDER}
    for cid, name, col_type, notnull, _default, pk in rows:
        item = {
            "name": str(name),
            "type": str(col_type or "REAL"),
            "pk": bool(pk),
            "notnull": bool(notnull),
        }
        columns.append(item)
        groups[_column_group(str(name))].append(item)

    column_groups = [
        {"id": gid, "label": _GROUP_LABELS.get(gid, gid), "columns": groups.get(gid, [])}
        for gid in _GROUP_ORDER
        if groups.get(gid)
    ]
    names = [c["name"] for c in columns]
    return {
        "exists": True,
        "db_path": db_path,
        "tables": tables,
        "column_count": len(columns),
        "columns": columns,
        "column_groups": column_groups,
        "copy_text": "\n".join(names),
        "copy_csv": ", ".join(names),
    }
