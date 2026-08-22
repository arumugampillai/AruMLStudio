"""Connection Management & Database Initialization for analysis.db (Phase 4D.1).

Manages connection lifecycle, PRAGMA configuration, and idempotent schema creation
for `<data_dir>/analysis.db`.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from .schema import ANALYSIS_DB_TABLES_DDL, EXPECTED_INDICES, EXPECTED_TABLES


def analysis_db_path(data_dir: str | None = None) -> str:
    """Return the absolute path to analysis.db via DataRootService."""
    if data_dir is None:
        from chain_replay_ml.core.data_root import get_data_root_service
        return get_data_root_service().get_database_path("analysis")
    d_str = str(data_dir).strip()
    if d_str.endswith(".db"):
        return os.path.abspath(d_str)
    canonical_sub = os.path.join(d_str, "databases", "analysis.db")
    if os.path.isfile(canonical_sub):
        return os.path.abspath(canonical_sub)
    return os.path.abspath(os.path.join(d_str, "analysis.db"))


def connect_analysis_db(data_dir: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection to `<data_dir>/analysis.db` with enforced pragmas.
    
    Pragmas Enforced:
    - foreign_keys = ON
    - journal_mode = WAL
    - synchronous = NORMAL
    - cache_size = -4000 (4 MB cache for 16 GB workstation memory safety)
    """
    path = analysis_db_path(data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -4000;")
    return conn


def init_analysis_db(data_dir: str) -> str:
    """Idempotently initialize all tables and indices in `<data_dir>/analysis.db`.
    
    Safe to call repeatedly; uses additive CREATE TABLE/INDEX IF NOT EXISTS.
    Returns the absolute path to analysis.db.
    """
    path = analysis_db_path(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            conn.executescript(ANALYSIS_DB_TABLES_DDL)
    finally:
        conn.close()
    return path


def verify_analysis_db_schema(data_dir: str) -> dict[str, Any]:
    """Verify that all required tables and indices exist and return diagnostic telemetry."""
    path = analysis_db_path(data_dir)
    if not os.path.isfile(path):
        return {
            "exists": False,
            "path": path,
            "tables_found": [],
            "tables_missing": list(EXPECTED_TABLES),
            "indices_found": [],
            "indices_missing": list(EXPECTED_INDICES),
            "foreign_keys_enabled": False,
            "journal_mode": "unknown",
        }

    conn = connect_analysis_db(data_dir)
    try:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        indices = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        fk_on = bool(conn.execute("PRAGMA foreign_keys;").fetchone()[0])
        journal = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()

        missing_tables = [t for t in EXPECTED_TABLES if t not in tables]
        missing_indices = [idx for idx in EXPECTED_INDICES if idx not in indices]

        return {
            "exists": True,
            "path": path,
            "tables_found": tables,
            "tables_missing": missing_tables,
            "indices_found": indices,
            "indices_missing": missing_indices,
            "foreign_keys_enabled": fk_on,
            "journal_mode": journal,
            "is_valid": len(missing_tables) == 0 and len(missing_indices) == 0 and fk_on,
        }
    finally:
        conn.close()
