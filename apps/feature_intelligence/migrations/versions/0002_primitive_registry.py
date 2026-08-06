"""Sprint 1 — Primitive Registry table + catalog 1.0 seed."""

from __future__ import annotations

import sqlite3

from feature_intelligence.registry.catalog import SEED_PRIMITIVES

version = "0002"
description = "primitive_registry table and catalog 1.0 seed"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS primitive_registry (
            primitive_id     TEXT PRIMARY KEY,
            name             TEXT UNIQUE NOT NULL,
            primitive_type   TEXT NOT NULL,
            description      TEXT,
            data_source      TEXT NOT NULL,
            units            TEXT NOT NULL,
            catalog_version  TEXT NOT NULL,
            created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_primitive_id ON primitive_registry(primitive_id);
        CREATE INDEX IF NOT EXISTS idx_primitive_name ON primitive_registry(name);
        CREATE INDEX IF NOT EXISTS idx_primitive_type ON primitive_registry(primitive_type);
        CREATE INDEX IF NOT EXISTS idx_primitive_catalog_version
            ON primitive_registry(catalog_version);
        """
    )
    for p in SEED_PRIMITIVES:
        conn.execute(
            """
            INSERT OR IGNORE INTO primitive_registry(
                primitive_id, name, primitive_type, description,
                data_source, units, catalog_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.primitive_id,
                p.name,
                p.primitive_type,
                p.description,
                p.data_source,
                p.units,
                p.catalog_version,
            ),
        )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_primitive_catalog_version")
    conn.execute("DROP INDEX IF EXISTS idx_primitive_type")
    conn.execute("DROP INDEX IF EXISTS idx_primitive_name")
    conn.execute("DROP INDEX IF EXISTS idx_primitive_id")
    conn.execute("DROP TABLE IF EXISTS primitive_registry")
