"""Sprint 2 — Feature Registry + feature_primitives junction."""

from __future__ import annotations

import sqlite3

version = "0003"
description = "feature_registry and feature_primitives tables"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feature_registry (
            feature_uuid           TEXT PRIMARY KEY,
            canonical_name         TEXT UNIQUE NOT NULL,
            display_name           TEXT NOT NULL,
            definition_version     TEXT NOT NULL,
            implementation_version TEXT NOT NULL,
            feature_version        TEXT,
            definition_hash        TEXT NOT NULL,
            transformation_uuid    TEXT,
            legacy_feature_id      TEXT,
            description            TEXT,
            created_by             TEXT NOT NULL,
            controller_owner       TEXT NOT NULL,
            warmup_periods         INTEGER NOT NULL,
            gap_policy             TEXT NOT NULL,
            memory_model           TEXT NOT NULL,
            research_state         TEXT NOT NULL DEFAULT 'EXPERIMENTAL',
            created_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (research_state IN ('EXPERIMENTAL', 'CANDIDATE', 'VALIDATED', 'DEPRECATED')),
            CHECK (
                transformation_uuid IS NULL
                OR transformation_uuid GLOB 'TR_*'
            ),
            CHECK (feature_uuid GLOB 'FEAT_*')
        );
        CREATE INDEX IF NOT EXISTS idx_feature_uuid ON feature_registry(feature_uuid);
        CREATE INDEX IF NOT EXISTS idx_canonical_name ON feature_registry(canonical_name);
        CREATE INDEX IF NOT EXISTS idx_research_state ON feature_registry(research_state);
        CREATE INDEX IF NOT EXISTS idx_controller_owner ON feature_registry(controller_owner);
        CREATE INDEX IF NOT EXISTS idx_transformation_uuid ON feature_registry(transformation_uuid);
        CREATE INDEX IF NOT EXISTS idx_definition_hash ON feature_registry(definition_hash);
        CREATE INDEX IF NOT EXISTS idx_legacy_feature_id ON feature_registry(legacy_feature_id);

        CREATE TABLE IF NOT EXISTS feature_primitives (
            feature_uuid  TEXT NOT NULL,
            primitive_id  TEXT NOT NULL,
            ordinal       INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (feature_uuid, primitive_id),
            FOREIGN KEY (feature_uuid) REFERENCES feature_registry(feature_uuid),
            FOREIGN KEY (primitive_id) REFERENCES primitive_registry(primitive_id)
        );
        CREATE INDEX IF NOT EXISTS idx_feature_primitives_feature
            ON feature_primitives(feature_uuid);
        CREATE INDEX IF NOT EXISTS idx_feature_primitives_primitive
            ON feature_primitives(primitive_id);
        """
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_feature_primitives_primitive")
    conn.execute("DROP INDEX IF EXISTS idx_feature_primitives_feature")
    conn.execute("DROP TABLE IF EXISTS feature_primitives")
    conn.execute("DROP INDEX IF EXISTS idx_legacy_feature_id")
    conn.execute("DROP INDEX IF EXISTS idx_definition_hash")
    conn.execute("DROP INDEX IF EXISTS idx_transformation_uuid")
    conn.execute("DROP INDEX IF EXISTS idx_controller_owner")
    conn.execute("DROP INDEX IF EXISTS idx_research_state")
    conn.execute("DROP INDEX IF EXISTS idx_canonical_name")
    conn.execute("DROP INDEX IF EXISTS idx_feature_uuid")
    conn.execute("DROP TABLE IF EXISTS feature_registry")
