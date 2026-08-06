"""Temporary / diagnostic helpers for Feature Explorer search (Sprint 9)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


# Exact SQL used by List All / match_all (ResearchStore.list_records).
MATCH_ALL_SQL = (
    "SELECT * FROM feature_research_record ORDER BY research_uuid ASC"
)


def count_table(db_path: Path, table: str) -> int | None:
    """Return COUNT(*) for ``table``, or None if the table/file is missing."""
    path = Path(db_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if row is None:
                return None
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def registry_and_frr_counts(db_path: Path) -> tuple[int | None, int | None]:
    """``(feature_registry count, feature_research_record count)``."""
    return (
        count_table(db_path, "feature_registry"),
        count_table(db_path, "feature_research_record"),
    )


def explorer_empty_hint(
    *,
    feature_count: int | None,
    frr_count: int | None,
) -> str | None:
    """Human-readable empty-state hint for Feature Explorer status label."""
    if feature_count is None or frr_count is None:
        return (
            "Database not ready (missing schema). "
            "Run: python -m feature_intelligence init-db"
        )
    if frr_count == 0 and feature_count > 0:
        return (
            f"No research records — features={feature_count}, FRR=0. "
            "Run: python -m feature_intelligence research sync"
        )
    if frr_count == 0 and feature_count == 0:
        return "Empty feature_registry and feature_research_record"
    return None
