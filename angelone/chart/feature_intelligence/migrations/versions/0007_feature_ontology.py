"""Sprint 6 — Feature Ontology tables + vocab/ontology seed pack 1.0.0."""

from __future__ import annotations

import sqlite3

from feature_intelligence.ontology.catalog import (
    EXPECTED_ONTOLOGY_SEED_HASH,
    EXPECTED_VOCAB_SEED_HASH,
    ONTOLOGY_VERSION,
    SEED_ONTOLOGY_ROWS,
    SEED_VOCABULARIES,
    VOCAB_PACK_VERSION,
    ontology_uuid_for_seed,
)
from feature_intelligence.ontology.models import OBJECT_TYPE_TABLE

version = "0007"
description = (
    "ontology_registry, vocabulary_registry, four *_ontology tables, "
    "ontology_statistics + pack 1.0.0 seed"
)

_SHARED_COLUMNS = """
    ontology_uuid              TEXT PRIMARY KEY,
    object_id                  TEXT NOT NULL UNIQUE,
    ontology_version           TEXT NOT NULL,
    domain                     TEXT NOT NULL REFERENCES vocabulary_registry(vocabulary_id),
    signal_type_json           TEXT NOT NULL,
    mathematical_family_json   TEXT NOT NULL,
    horizon                    TEXT NOT NULL REFERENCES vocabulary_registry(vocabulary_id),
    output_type                TEXT NOT NULL REFERENCES vocabulary_registry(vocabulary_id),
    frequency                  TEXT NOT NULL REFERENCES vocabulary_registry(vocabulary_id),
    stability                  TEXT NOT NULL REFERENCES vocabulary_registry(vocabulary_id),
    input_dependencies_json    TEXT NOT NULL DEFAULT '[]',
    meaning                    TEXT,
    confidence                 REAL,
    classification_source      TEXT,
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT NOT NULL,
    CHECK (ontology_uuid GLOB 'ONT_*'),
    CHECK (confidence IS NULL),
    CHECK (
        classification_source IS NULL
        OR classification_source IN ('SEED','IMPORT','MIGRATION')
    )
"""


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS vocabulary_registry (
            vocabulary_pk     INTEGER PRIMARY KEY AUTOINCREMENT,
            vocabulary_id     TEXT NOT NULL UNIQUE,
            vocabulary_type   TEXT NOT NULL,
            canonical_name    TEXT NOT NULL,
            display_name      TEXT NOT NULL,
            description       TEXT,
            ontology_version  TEXT NOT NULL,
            active            INTEGER NOT NULL DEFAULT 1,
            retired_reason    TEXT,
            sort_order        INTEGER,
            catalog_version   TEXT NOT NULL DEFAULT '1.0',
            created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (vocabulary_type IN (
                'DOMAIN','SIGNAL_TYPE','MATH_FAMILY','HORIZON',
                'OUTPUT_TYPE','FREQUENCY','STABILITY'
            )),
            CHECK (active IN (0, 1))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_vocabulary_type_canonical
            ON vocabulary_registry(vocabulary_type, canonical_name);
        CREATE INDEX IF NOT EXISTS idx_vocabulary_type
            ON vocabulary_registry(vocabulary_type);
        CREATE INDEX IF NOT EXISTS idx_vocabulary_sort
            ON vocabulary_registry(vocabulary_type, sort_order);
        CREATE INDEX IF NOT EXISTS idx_vocabulary_active
            ON vocabulary_registry(active);

        CREATE TABLE IF NOT EXISTS ontology_registry (
            ontology_version       TEXT PRIMARY KEY,
            vocab_pack_version     TEXT NOT NULL,
            vocab_seed_checksum    TEXT NOT NULL,
            ontology_seed_checksum TEXT NOT NULL,
            description            TEXT,
            created_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS primitive_ontology (
            {_SHARED_COLUMNS},
            CHECK (object_id GLOB 'PR_*'),
            FOREIGN KEY (object_id) REFERENCES primitive_registry(primitive_id)
        );

        CREATE TABLE IF NOT EXISTS operator_ontology (
            {_SHARED_COLUMNS},
            CHECK (object_id GLOB 'OP_*'),
            FOREIGN KEY (object_id) REFERENCES operator_registry(operator_id)
        );

        CREATE TABLE IF NOT EXISTS transformation_ontology (
            {_SHARED_COLUMNS},
            CHECK (object_id GLOB 'TR_*'),
            FOREIGN KEY (object_id) REFERENCES transformation_registry(transformation_uuid)
        );

        CREATE TABLE IF NOT EXISTS feature_ontology (
            {_SHARED_COLUMNS},
            CHECK (object_id GLOB 'FEAT_*'),
            FOREIGN KEY (object_id) REFERENCES feature_registry(feature_uuid)
        );

        CREATE INDEX IF NOT EXISTS idx_primitive_ontology_object
            ON primitive_ontology(object_id);
        CREATE INDEX IF NOT EXISTS idx_operator_ontology_object
            ON operator_ontology(object_id);
        CREATE INDEX IF NOT EXISTS idx_transformation_ontology_object
            ON transformation_ontology(object_id);
        CREATE INDEX IF NOT EXISTS idx_feature_ontology_object
            ON feature_ontology(object_id);
        CREATE INDEX IF NOT EXISTS idx_feature_ontology_domain
            ON feature_ontology(domain);
        CREATE INDEX IF NOT EXISTS idx_operator_ontology_domain
            ON operator_ontology(domain);

        CREATE TABLE IF NOT EXISTS ontology_statistics (
            stats_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ontology_version      TEXT NOT NULL,
            objects_total         INTEGER NOT NULL,
            objects_classified    INTEGER NOT NULL,
            objects_missing       INTEGER NOT NULL,
            coverage_pct          REAL NOT NULL,
            pr_expected           INTEGER NOT NULL,
            pr_classified         INTEGER NOT NULL,
            pr_missing            INTEGER NOT NULL,
            pr_coverage_pct       REAL NOT NULL,
            op_expected           INTEGER NOT NULL,
            op_classified         INTEGER NOT NULL,
            op_missing            INTEGER NOT NULL,
            op_coverage_pct       REAL NOT NULL,
            tr_expected           INTEGER NOT NULL,
            tr_classified         INTEGER NOT NULL,
            tr_missing            INTEGER NOT NULL,
            tr_coverage_pct       REAL NOT NULL,
            feat_expected         INTEGER NOT NULL,
            feat_classified       INTEGER NOT NULL,
            feat_missing          INTEGER NOT NULL,
            feat_coverage_pct     REAL NOT NULL,
            created_at            TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ontology_statistics_created
            ON ontology_statistics(created_at);
        CREATE INDEX IF NOT EXISTS idx_ontology_statistics_version
            ON ontology_statistics(ontology_version);
        """
    )

    # Seed vocabularies
    for v in SEED_VOCABULARIES:
        conn.execute(
            """
            INSERT OR IGNORE INTO vocabulary_registry(
                vocabulary_id, vocabulary_type, canonical_name, display_name,
                description, ontology_version, active, retired_reason,
                sort_order, catalog_version
            ) VALUES (?,?,?,?,?,?,?,NULL,?,?)
            """,
            (
                v.vocabulary_id,
                v.vocabulary_type,
                v.canonical_name,
                v.display_name,
                v.description,
                v.ontology_version,
                v.active,
                v.sort_order,
                v.catalog_version,
            ),
        )

    # Seed required ontology rows
    import json

    for row in SEED_ONTOLOGY_ROWS:
        table = OBJECT_TYPE_TABLE[row.object_type]
        uuid = ontology_uuid_for_seed(row)
        sig = json.dumps(list(row.signal_type), separators=(",", ":"))
        math = json.dumps(list(row.mathematical_family), separators=(",", ":"))
        deps = json.dumps(list(row.input_dependencies), separators=(",", ":"))
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {table} (
                ontology_uuid, object_id, ontology_version, domain,
                signal_type_json, mathematical_family_json, horizon,
                output_type, frequency, stability, input_dependencies_json,
                meaning, confidence, classification_source, created_at, updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,NULL,'SEED',
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
            """,
            (
                uuid,
                row.object_id,
                ONTOLOGY_VERSION,
                row.domain,
                sig,
                math,
                row.horizon,
                row.output_type,
                row.frequency,
                row.stability,
                deps,
                row.meaning,
            ),
        )

    conn.execute(
        """
        INSERT OR IGNORE INTO ontology_registry(
            ontology_version, vocab_pack_version,
            vocab_seed_checksum, ontology_seed_checksum, description
        ) VALUES (?,?,?,?,?)
        """,
        (
            ONTOLOGY_VERSION,
            VOCAB_PACK_VERSION,
            EXPECTED_VOCAB_SEED_HASH,
            EXPECTED_ONTOLOGY_SEED_HASH,
            "Feature Ontology pack 1.0.0 — controlled vocabulary classification",
        ),
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_ontology_statistics_version;
        DROP INDEX IF EXISTS idx_ontology_statistics_created;
        DROP TABLE IF EXISTS ontology_statistics;

        DROP INDEX IF EXISTS idx_operator_ontology_domain;
        DROP INDEX IF EXISTS idx_feature_ontology_domain;
        DROP INDEX IF EXISTS idx_feature_ontology_object;
        DROP INDEX IF EXISTS idx_transformation_ontology_object;
        DROP INDEX IF EXISTS idx_operator_ontology_object;
        DROP INDEX IF EXISTS idx_primitive_ontology_object;

        DROP TABLE IF EXISTS feature_ontology;
        DROP TABLE IF EXISTS transformation_ontology;
        DROP TABLE IF EXISTS operator_ontology;
        DROP TABLE IF EXISTS primitive_ontology;

        DROP TABLE IF EXISTS ontology_registry;

        DROP INDEX IF EXISTS idx_vocabulary_active;
        DROP INDEX IF EXISTS idx_vocabulary_sort;
        DROP INDEX IF EXISTS idx_vocabulary_type;
        DROP INDEX IF EXISTS uq_vocabulary_type_canonical;
        DROP TABLE IF EXISTS vocabulary_registry;
        """
    )
