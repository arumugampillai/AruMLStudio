"""Baseline migration — infrastructure meta only (no FIC business schema)."""

from __future__ import annotations

import sqlite3

version = "0001"
description = "baseline infrastructure (migration framework smoke table)"


def upgrade(conn: sqlite3.Connection) -> None:
    # Intentionally tiny stub so migration runner can be tested.
    # No Feature Intelligence business tables.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fic_infra_ping (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            note TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO fic_infra_ping(id, note) VALUES (1, 'sprint0-ok')"
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS fic_infra_ping")
