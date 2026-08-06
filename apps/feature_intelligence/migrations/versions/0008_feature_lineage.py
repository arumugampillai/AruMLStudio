"""Sprint 7 — Feature Lineage tables + REL_* relationship seed pack 1.0.0."""

from __future__ import annotations

import hashlib
import sqlite3

from feature_intelligence.lineage.models import (
    GRAPH_SCHEMA_VERSION,
    LINEAGE_VERSION,
    RELATIONSHIP_PACK_VERSION,
)
from feature_intelligence.lineage.relationships import (
    EXPECTED_RELATIONSHIP_SEED_HASH,
    SEED_RELATIONSHIPS,
)

version = "0008"
description = (
    "lineage_registry, lineage_relationship_registry, lineage_edges, "
    "lineage_statistics, relationship_statistics + REL_* pack 1.0.0"
)


def _empty_graph_checksum() -> str:
    return hashlib.sha256(b"").hexdigest()


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lineage_relationship_registry (
            relationship_id      TEXT PRIMARY KEY,
            canonical_name       TEXT NOT NULL UNIQUE,
            display_name         TEXT NOT NULL,
            description          TEXT,
            lineage_version      TEXT NOT NULL,
            active               INTEGER NOT NULL DEFAULT 1,
            sort_order           INTEGER,
            created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (active IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_lineage_rel_active
            ON lineage_relationship_registry(active);
        CREATE INDEX IF NOT EXISTS idx_lineage_rel_sort
            ON lineage_relationship_registry(sort_order);

        CREATE TABLE IF NOT EXISTS lineage_registry (
            lineage_version              TEXT PRIMARY KEY,
            graph_schema_version         TEXT NOT NULL,
            relationship_pack_version    TEXT NOT NULL,
            relationship_seed_checksum   TEXT NOT NULL,
            graph_checksum               TEXT NOT NULL,
            description                  TEXT,
            created_at                   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at                   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS lineage_edges (
            lineage_uuid        TEXT PRIMARY KEY,
            parent_object       TEXT NOT NULL,
            child_object        TEXT NOT NULL,
            relationship_id     TEXT NOT NULL,
            edge_source         TEXT,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            FOREIGN KEY (relationship_id)
                REFERENCES lineage_relationship_registry(relationship_id),
            CHECK (parent_object != child_object),
            CHECK (
                edge_source IS NULL
                OR edge_source IN ('DERIVE', 'IMPORT', 'MIGRATION')
            ),
            UNIQUE (parent_object, child_object, relationship_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_parent
            ON lineage_edges(parent_object);
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_child
            ON lineage_edges(child_object);
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_rel
            ON lineage_edges(relationship_id);
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_parent_child
            ON lineage_edges(parent_object, child_object);

        CREATE TABLE IF NOT EXISTS lineage_statistics (
            stats_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            lineage_version       TEXT NOT NULL,
            graph_schema_version  TEXT NOT NULL,
            edges                 INTEGER NOT NULL,
            nodes                 INTEGER NOT NULL,
            root_primitives       INTEGER NOT NULL,
            orphans               INTEGER NOT NULL,
            depth                 INTEGER NOT NULL,
            components            INTEGER NOT NULL,
            created_at            TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lineage_statistics_created
            ON lineage_statistics(created_at);
        CREATE INDEX IF NOT EXISTS idx_lineage_statistics_version
            ON lineage_statistics(lineage_version);

        CREATE TABLE IF NOT EXISTS relationship_statistics (
            rel_stats_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stats_id              INTEGER NOT NULL,
            lineage_version       TEXT NOT NULL,
            relationship_id       TEXT NOT NULL,
            edge_count            INTEGER NOT NULL,
            created_at            TEXT NOT NULL,
            FOREIGN KEY (stats_id) REFERENCES lineage_statistics(stats_id),
            FOREIGN KEY (relationship_id)
                REFERENCES lineage_relationship_registry(relationship_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rel_statistics_stats
            ON relationship_statistics(stats_id);
        CREATE INDEX IF NOT EXISTS idx_rel_statistics_rel
            ON relationship_statistics(relationship_id);
        """
    )

    for r in SEED_RELATIONSHIPS:
        conn.execute(
            """
            INSERT OR IGNORE INTO lineage_relationship_registry(
                relationship_id, canonical_name, display_name, description,
                lineage_version, active, sort_order
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                r.relationship_id,
                r.canonical_name,
                r.display_name,
                r.description,
                r.lineage_version,
                r.active,
                r.sort_order,
            ),
        )

    conn.execute(
        """
        INSERT OR IGNORE INTO lineage_registry(
            lineage_version, graph_schema_version, relationship_pack_version,
            relationship_seed_checksum, graph_checksum, description
        ) VALUES (?,?,?,?,?,?)
        """,
        (
            LINEAGE_VERSION,
            GRAPH_SCHEMA_VERSION,
            RELATIONSHIP_PACK_VERSION,
            EXPECTED_RELATIONSHIP_SEED_HASH,
            _empty_graph_checksum(),
            "Feature Lineage pack 1.0.0 — relationships only (DAG)",
        ),
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_rel_statistics_rel;
        DROP INDEX IF EXISTS idx_rel_statistics_stats;
        DROP TABLE IF EXISTS relationship_statistics;

        DROP INDEX IF EXISTS idx_lineage_statistics_version;
        DROP INDEX IF EXISTS idx_lineage_statistics_created;
        DROP TABLE IF EXISTS lineage_statistics;

        DROP INDEX IF EXISTS idx_lineage_edges_parent_child;
        DROP INDEX IF EXISTS idx_lineage_edges_rel;
        DROP INDEX IF EXISTS idx_lineage_edges_child;
        DROP INDEX IF EXISTS idx_lineage_edges_parent;
        DROP TABLE IF EXISTS lineage_edges;

        DROP TABLE IF EXISTS lineage_registry;

        DROP INDEX IF EXISTS idx_lineage_rel_sort;
        DROP INDEX IF EXISTS idx_lineage_rel_active;
        DROP TABLE IF EXISTS lineage_relationship_registry;
        """
    )
