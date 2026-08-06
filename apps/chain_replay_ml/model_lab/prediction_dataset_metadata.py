"""Prediction Dataset Metadata — Research Lab Phase 1.5 (+ Phase 1.6 enhancements).

Gives the Metadata tab (Research Lab → Prediction Dataset) a fast, always-
accurate view of ``prediction_dataset``: row/column counts, per-column
populated/NULL/coverage (with owning pipeline stage), and per-stage build
status (Regression, Probability Ladder, Triple Barrier, Compute Outcomes,
Confidence, Identity/Other) driven entirely by the stage registry in
``prediction_metadata_stages``.

Every statistic comes from a single SQL aggregate pass — ``COUNT(col)``
counts non-NULL values natively in SQLite — so this never loads the table
into pandas and stays cheap even for multi-million row labs.

Stage "notes" explain *why* a stage isn't 100% covered. Ladder notes prefer an
authoritative ``package_members`` list (see
``prediction_packages.discover_prediction_package_members``) when the caller
supplies one; pure DB-only compute (no lab/data-dir context) instead infers
missing ladder members from column presence/population — a ladder output
column that exists in the schema but is 0% populated almost always means that
classifier was never trained/selected for the package.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from typing import Any

from .prediction_metadata_stages import (
    STAGE_REGISTRY,
    STATUS_LABEL,
    STATUS_NONE,
    STATUS_NOT_BUILT,
    STATUS_OK,
    STATUS_PARTIAL,
    READY_COVERAGE_PCT,
    StageContext,
    assign_column_stages,
    coverage_bucket,
    coverage_emoji,
    stage_expected_columns,
    stage_status,
)
from .prediction_schema import DAY_COMPLETED

STAGE_ORDER: tuple[str, ...] = tuple(spec.label for spec in STAGE_REGISTRY)

__all__ = [
    "STATUS_OK",
    "STATUS_PARTIAL",
    "STATUS_NONE",
    "STATUS_NOT_BUILT",
    "STAGE_ORDER",
    "compute_prediction_dataset_metadata",
    "prediction_dataset_metadata_cache_path",
    "read_cached_prediction_dataset_metadata",
    "refresh_prediction_dataset_metadata",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _actual_columns(conn: sqlite3.Connection) -> list[str]:
    """Physical column order (schema declaration order + any ALTER-added sf_*)."""
    rows = conn.execute("PRAGMA table_info(prediction_dataset)").fetchall()
    return [str(r[1]) for r in rows if str(r[1] or "") and str(r[1]) != "id"]


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row and row[0] is not None else 0


def _day_status_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """(completed, pending) trading days from prediction_day_metadata."""
    if not _table_exists(conn, "prediction_day_metadata"):
        return 0, 0
    completed = 0
    total = 0
    for status, cnt in conn.execute(
        "SELECT status, COUNT(*) FROM prediction_day_metadata GROUP BY status"
    ).fetchall():
        n = int(cnt or 0)
        total += n
        if str(status or "") == DAY_COMPLETED:
            completed += n
    return completed, max(0, total - completed)


def _column_populated_counts(
    conn: sqlite3.Connection, columns: list[str], total_rows: int
) -> list[dict[str, Any]]:
    """One aggregate query: ``COUNT(col)`` = non-NULL count per column."""
    if not columns:
        return []
    if total_rows <= 0:
        return [
            {
                "name": c,
                "populated": 0,
                "null": 0,
                "coverage_pct": 0.0,
                "coverage_bucket": coverage_bucket(0.0, populated=0),
                "coverage_emoji": coverage_emoji(0.0, populated=0),
            }
            for c in columns
        ]
    exprs = ", ".join(f'COUNT("{c}") AS "{c}"' for c in columns)
    row = conn.execute(f"SELECT {exprs} FROM prediction_dataset").fetchone()
    out: list[dict[str, Any]] = []
    for c in columns:
        populated = int(row[c] or 0)
        nulls = max(0, total_rows - populated)
        coverage = round((populated / total_rows) * 100.0, 2) if total_rows else 0.0
        out.append(
            {
                "name": c,
                "populated": populated,
                "null": nulls,
                "coverage_pct": coverage,
                "coverage_bucket": coverage_bucket(coverage, populated=populated),
                "coverage_emoji": coverage_emoji(coverage, populated=populated),
            }
        )
    return out


def _stage_entry(
    spec,
    all_columns: list[str],
    assigned: dict[str, str],
    stats_by_name: dict[str, dict[str, Any]],
    *,
    total_rows: int,
    package_members: tuple[dict[str, Any], ...] | None,
) -> dict[str, Any]:
    expected = [
        c for c in stage_expected_columns(spec, all_columns, assigned) if c in stats_by_name
    ]
    coverage_pct = 0.0
    if expected:
        coverage_pct = round(
            sum(stats_by_name[c]["coverage_pct"] for c in expected) / len(expected), 2
        )
    status = stage_status(coverage_pct, has_columns=bool(expected), total_rows=total_rows)
    ready = sum(
        1
        for c in expected
        if float(stats_by_name[c]["coverage_pct"]) > READY_COVERAGE_PCT
    )
    ctx = StageContext(
        status=status,
        coverage_pct=coverage_pct,
        expected_columns=tuple(expected),
        stats_by_name=stats_by_name,
        total_rows=total_rows,
        package_members=package_members if spec.key == "probability_ladder" else None,
    )
    notes = spec.notes(ctx) if spec.notes else ""
    return {
        "key": spec.key,
        "name": spec.label,
        "label": spec.label,
        "status": status,
        "status_label": STATUS_LABEL.get(status, status),
        "coverage_pct": coverage_pct,
        "coverage_bucket": coverage_bucket(coverage_pct),
        "coverage_emoji": coverage_emoji(coverage_pct),
        "expected": len(expected),
        "ready": ready,
        "columns": expected,
        "notes": notes,
    }


def _empty_metadata(*, error: str | None = None) -> dict[str, Any]:
    return {
        "row_count": 0,
        "column_count": 0,
        "trading_days": 0,
        "completed_days": 0,
        "pending_days": 0,
        "total_catalog_days": 0,
        "columns": [],
        "stages": [
            {
                "key": spec.key,
                "name": spec.label,
                "label": spec.label,
                "status": STATUS_NOT_BUILT,
                "status_label": STATUS_LABEL[STATUS_NOT_BUILT],
                "coverage_pct": 0.0,
                "coverage_bucket": coverage_bucket(0.0, populated=0),
                "coverage_emoji": coverage_emoji(0.0, populated=0),
                "expected": 0,
                "ready": 0,
                "columns": [],
                "notes": "",
            }
            for spec in STAGE_REGISTRY
        ],
        "generated_at": _utc_now(),
        "error": error,
    }


def compute_prediction_dataset_metadata(
    db_path: str,
    *,
    package_members: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute Dataset Summary + Column Coverage + Stage Coverage for one lab.

    Reads actual ``prediction_dataset`` columns (via ``PRAGMA table_info``) so
    dynamically-added ``sf_*`` embedded-feature columns are included, not just
    ``prediction_schema.CORE_COLUMN_NAMES``. Safe on an empty/never-built lab —
    returns zeroed-out stats instead of raising.

    ``package_members`` is an optional Probability Ladder member list (shape:
    ``prediction_packages.discover_prediction_package_members``) — pass it
    when the caller already has live package/model context (e.g. the Research
    Lab window after a build) to get authoritative "missing ladder model"
    notes instead of the DB-only column-population inference.
    """
    if not db_path or not os.path.isfile(db_path):
        return _empty_metadata()

    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "prediction_dataset"):
            return _empty_metadata()
        columns = _actual_columns(conn)
        if not columns:
            return _empty_metadata()

        total_rows = _scalar(conn, "SELECT COUNT(*) FROM prediction_dataset")
        trading_days = _scalar(
            conn,
            "SELECT COUNT(DISTINCT trading_day) FROM prediction_dataset "
            "WHERE trading_day IS NOT NULL AND trading_day != ''",
        )
        completed_days, pending_days = _day_status_counts(conn)
        column_stats = _column_populated_counts(conn, columns, total_rows)
        stats_by_name = {c["name"]: c for c in column_stats}

        assigned = assign_column_stages(columns)
        for col in column_stats:
            col["stage"] = assigned.get(col["name"], "Identity/Other")

        members_tuple = tuple(package_members) if package_members else None
        stages = [
            _stage_entry(
                spec,
                columns,
                assigned,
                stats_by_name,
                total_rows=total_rows,
                package_members=members_tuple,
            )
            for spec in STAGE_REGISTRY
        ]

        return {
            "row_count": total_rows,
            "column_count": len(columns),
            "trading_days": trading_days,
            "completed_days": completed_days,
            "pending_days": pending_days,
            "total_catalog_days": completed_days + pending_days,
            "columns": column_stats,
            "stages": stages,
            "generated_at": _utc_now(),
            "error": None,
        }
    finally:
        conn.close()


def prediction_dataset_metadata_cache_path(lab_db_path: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(lab_db_path))
    return f"{base}.prediction_dataset_metadata.json"


def read_cached_prediction_dataset_metadata(lab_db_path: str) -> dict[str, Any] | None:
    """Last computed metadata, for instant paint before a fresh recompute lands."""
    path = prediction_dataset_metadata_cache_path(lab_db_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_cache_atomic(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".pdm_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def refresh_prediction_dataset_metadata(
    lab_db_path: str,
    *,
    package_members: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute metadata and persist it to the sidecar cache next to the lab DB.

    Call this after Build day complete / Compute Outcomes day complete, and
    from the Metadata tab's Refresh button.
    """
    metadata = compute_prediction_dataset_metadata(lab_db_path, package_members=package_members)
    try:
        _write_cache_atomic(prediction_dataset_metadata_cache_path(lab_db_path), metadata)
    except OSError:
        pass
    return metadata
