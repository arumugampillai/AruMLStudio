"""Sprint 8 — Feature Research Record tables + pack metadata 1.0.0.

Migration ≠ Business rows: creates schema + one research_registry pack row only.
Does NOT seed feature_research_record business rows.
"""

from __future__ import annotations

import hashlib
import sqlite3

from feature_intelligence.research.models import RESEARCH_VERSION, SCHEMA_VERSION

version = "0009"
description = (
    "research_registry, feature_research_record, research_statistics "
    "+ research pack 1.0.0 (schema only; no FRR business rows)"
)


def _empty_research_checksum() -> str:
    return hashlib.sha256(b"").hexdigest()


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_registry (
            research_version         TEXT PRIMARY KEY,
            schema_version           TEXT NOT NULL,
            checksum                 TEXT NOT NULL,
            description              TEXT,
            created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS feature_research_record (
            research_uuid          TEXT PRIMARY KEY,
            feature_uuid           TEXT NOT NULL UNIQUE,
            ontology_uuid          TEXT,
            transformation_uuid    TEXT,
            lineage_version        TEXT,
            compiler_version       TEXT,
            grammar_version        TEXT,
            research_status        TEXT NOT NULL DEFAULT 'EMPTY',
            validation_status      TEXT NOT NULL DEFAULT 'pending',
            evidence_json          TEXT,
            strengths_json         TEXT,
            weaknesses_json        TEXT,
            regimes_json           TEXT,
            failure_modes_json     TEXT,
            experiment_ids         TEXT,
            notes                  TEXT,
            record_source          TEXT,
            created_at             TEXT NOT NULL,
            updated_at             TEXT NOT NULL,
            FOREIGN KEY (feature_uuid)
                REFERENCES feature_registry(feature_uuid),
            CHECK (research_status IN ('EMPTY', 'ACTIVE', 'ARCHIVED')),
            CHECK (validation_status IN ('validated', 'pending', 'failed')),
            CHECK (
                record_source IS NULL
                OR record_source IN ('SYNC', 'IMPORT', 'MIGRATION')
            )
        );
        CREATE INDEX IF NOT EXISTS idx_frr_feature
            ON feature_research_record(feature_uuid);
        CREATE INDEX IF NOT EXISTS idx_frr_status
            ON feature_research_record(research_status);
        CREATE INDEX IF NOT EXISTS idx_frr_ontology
            ON feature_research_record(ontology_uuid);
        CREATE INDEX IF NOT EXISTS idx_frr_transformation
            ON feature_research_record(transformation_uuid);

        CREATE TABLE IF NOT EXISTS research_statistics (
            stats_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            research_version      TEXT NOT NULL,
            schema_version        TEXT NOT NULL,
            total_frr             INTEGER NOT NULL,
            expected_features     INTEGER NOT NULL,
            coverage_pct          REAL NOT NULL,
            status_empty          INTEGER NOT NULL,
            status_active         INTEGER NOT NULL,
            status_archived       INTEGER NOT NULL,
            last_sync_at          TEXT,
            created_at            TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_research_statistics_created
            ON research_statistics(created_at);
        CREATE INDEX IF NOT EXISTS idx_research_statistics_version
            ON research_statistics(research_version);
        """
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO research_registry(
            research_version, schema_version, checksum, description
        ) VALUES (?,?,?,?)
        """,
        (
            RESEARCH_VERSION,
            SCHEMA_VERSION,
            _empty_research_checksum(),
            "Feature Research Record pack 1.0.0 — metadata shell (no FRR business seed)",
        ),
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_research_statistics_version;
        DROP INDEX IF EXISTS idx_research_statistics_created;
        DROP TABLE IF EXISTS research_statistics;

        DROP INDEX IF EXISTS idx_frr_transformation;
        DROP INDEX IF EXISTS idx_frr_ontology;
        DROP INDEX IF EXISTS idx_frr_status;
        DROP INDEX IF EXISTS idx_frr_feature;
        DROP TABLE IF EXISTS feature_research_record;

        DROP TABLE IF EXISTS research_registry;
        """
    )
