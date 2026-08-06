"""Prediction model slot registry — maps prediction_version → model slots."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Sequence

from chain_replay_ml.training.paths import model_artifact_paths

from live_inference.versions import feature_version as live_feature_version


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slot_name(index: int) -> str:
    return f"model_{int(index)}"


def slot_pred_column(index: int) -> str:
    return f"model_{int(index)}_pred"


def slot_delta_column(index: int) -> str:
    return f"model_{int(index)}_delta_from_mean"


def slot_rank_column(index: int) -> str:
    return f"model_{int(index)}_rank"


def slot_columns_for_count(model_count: int) -> tuple[list[str], list[str], list[str]]:
    """Return (pred_cols, delta_cols, rank_cols) for N model slots."""
    preds = [slot_pred_column(i) for i in range(1, model_count + 1)]
    deltas = [slot_delta_column(i) for i in range(1, model_count + 1)]
    ranks = [slot_rank_column(i) for i in range(1, model_count + 1)]
    return preds, deltas, ranks


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _trained_on(data_dir: str, model_name: str) -> str | None:
    paths = model_artifact_paths(data_dir, model_name)
    for key in ("registry_json", "training_metadata_json", "config_json"):
        doc = _load_json(paths.get(key, ""))
        for field in ("trained_at", "trained_on", "created_at"):
            val = doc.get(field)
            if val:
                return str(val)
    return None


def _model_id(spec: dict[str, Any]) -> str:
    name = str(spec.get("model_name") or spec.get("registry", {}).get("model_name") or "").strip()
    return name or "unknown"


def registry_rows_from_specs(
    data_dir: str,
    specs: Sequence[dict[str, Any]],
    *,
    prediction_version: int,
    model_registry_version: str,
    default_feature_version: str | None = None,
) -> list[dict[str, Any]]:
    """Build prediction_model_registry rows — one per slot in loop order."""
    rows: list[dict[str, Any]] = []
    feat_default = default_feature_version or live_feature_version()
    for i, spec in enumerate(specs, start=1):
        model_name = _model_id(spec)
        reg = spec.get("registry") or {}
        rows.append({
            "prediction_version": int(prediction_version),
            "slot": slot_name(i),
            "model_id": model_name,
            "model_name": model_name,
            "target": str(spec.get("target") or reg.get("target") or ""),
            "feature_version": str(spec.get("feature_version") or reg.get("feature_version") or feat_default),
            "mae": _f(spec.get("mae") or reg.get("mae")),
            "rmse": _f(spec.get("rmse") or reg.get("rmse")),
            "trained_on": _trained_on(data_dir, model_name),
            "model_registry_version": str(model_registry_version or ""),
            "registered_at": _utc_now(),
        })
    return rows


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def slot_signature(specs: Sequence[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Stable slot assignment signature for comparing model sets."""
    out: list[tuple[str, str, str]] = []
    for i, spec in enumerate(specs, start=1):
        model_name = _model_id(spec)
        target = str(spec.get("target") or "")
        out.append((slot_name(i), model_name, target))
    return out


def ensure_registry_tables(conn: Any) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS prediction_version_index (
            prediction_version INTEGER PRIMARY KEY,
            model_registry_version TEXT NOT NULL UNIQUE,
            model_count INTEGER NOT NULL,
            registered_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_model_registry (
            prediction_version INTEGER NOT NULL,
            slot TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            target TEXT,
            feature_version TEXT,
            mae REAL,
            rmse REAL,
            trained_on TEXT,
            model_registry_version TEXT,
            registered_at TEXT,
            PRIMARY KEY (prediction_version, slot),
            FOREIGN KEY (prediction_version) REFERENCES prediction_version_index(prediction_version)
        );

        CREATE INDEX IF NOT EXISTS idx_prediction_model_registry_version
            ON prediction_model_registry(prediction_version);
        """
    )
    conn.commit()


def lookup_prediction_version(conn: Any, model_registry_version: str) -> int | None:
    row = conn.execute(
        "SELECT prediction_version FROM prediction_version_index WHERE model_registry_version = ?",
        (str(model_registry_version or ""),),
    ).fetchone()
    return int(row[0]) if row else None


def next_prediction_version(conn: Any) -> int:
    row = conn.execute("SELECT COALESCE(MAX(prediction_version), 0) FROM prediction_version_index").fetchone()
    return int(row[0] or 0) + 1


def register_prediction_version(
    conn: Any,
    *,
    prediction_version: int,
    model_registry_version: str,
    model_count: int,
    slot_rows: Sequence[dict[str, Any]],
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO prediction_version_index
            (prediction_version, model_registry_version, model_count, registered_at)
        VALUES (?, ?, ?, ?)
        """,
        (int(prediction_version), str(model_registry_version), int(model_count), _utc_now()),
    )
    insert_sql = """
        INSERT OR IGNORE INTO prediction_model_registry (
            prediction_version, slot, model_id, model_name, target,
            feature_version, mae, rmse, trained_on, model_registry_version, registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch = [
        (
            row["prediction_version"],
            row["slot"],
            row["model_id"],
            row["model_name"],
            row.get("target"),
            row.get("feature_version"),
            row.get("mae"),
            row.get("rmse"),
            row.get("trained_on"),
            row.get("model_registry_version"),
            row.get("registered_at"),
        )
        for row in slot_rows
    ]
    conn.executemany(insert_sql, batch)
    conn.commit()


def resolve_or_register_prediction_version(
    conn: Any,
    data_dir: str,
    specs: Sequence[dict[str, Any]],
    *,
    model_registry_version: str,
    default_feature_version: str | None = None,
) -> int:
    """Return existing or allocate new integer prediction_version for this model set."""
    ensure_registry_tables(conn)
    existing = lookup_prediction_version(conn, model_registry_version)
    if existing is not None:
        return existing

    version = next_prediction_version(conn)
    slot_rows = registry_rows_from_specs(
        data_dir,
        specs,
        prediction_version=version,
        model_registry_version=model_registry_version,
        default_feature_version=default_feature_version,
    )
    register_prediction_version(
        conn,
        prediction_version=version,
        model_registry_version=model_registry_version,
        model_count=len(specs),
        slot_rows=slot_rows,
    )
    return version


def read_model_registry(conn: Any, prediction_version: int | None = None) -> list[dict[str, Any]]:
    ensure_registry_tables(conn)
    if prediction_version is not None:
        rows = conn.execute(
            """
            SELECT prediction_version, slot, model_id, model_name, target,
                   feature_version, mae, rmse, trained_on, model_registry_version, registered_at
            FROM prediction_model_registry
            WHERE prediction_version = ?
            ORDER BY slot
            """,
            (int(prediction_version),),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT prediction_version, slot, model_id, model_name, target,
                   feature_version, mae, rmse, trained_on, model_registry_version, registered_at
            FROM prediction_model_registry
            ORDER BY prediction_version, slot
            """
        ).fetchall()
    cols = (
        "prediction_version", "slot", "model_id", "model_name", "target",
        "feature_version", "mae", "rmse", "trained_on", "model_registry_version", "registered_at",
    )
    return [dict(zip(cols, row)) for row in rows]


def read_prediction_versions(conn: Any) -> list[dict[str, Any]]:
    ensure_registry_tables(conn)
    rows = conn.execute(
        """
        SELECT prediction_version, model_registry_version, model_count, registered_at
        FROM prediction_version_index
        ORDER BY prediction_version
        """
    ).fetchall()
    return [
        {
            "prediction_version": row[0],
            "model_registry_version": row[1],
            "model_count": row[2],
            "registered_at": row[3],
        }
        for row in rows
    ]
