"""Sprint 3 — Operator Registry table + pack 1.0.0 seed."""

from __future__ import annotations

import sqlite3

from feature_intelligence.operators.catalog import (
    OPERATOR_CATALOG_VERSION,
    OPERATOR_PACK_VERSION,
    OPERATOR_VERSION_DEFAULT,
    SEED_OPERATORS,
)

version = "0004"
description = "operator_registry table and pack 1.0.0 seed"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS operator_registry (
            operator_id            TEXT PRIMARY KEY,
            canonical_name         TEXT UNIQUE NOT NULL,
            display_name           TEXT NOT NULL,
            category               TEXT NOT NULL,
            description            TEXT,
            formula                TEXT NOT NULL,
            definition_text        TEXT NOT NULL,
            parameter_schema_json  TEXT NOT NULL,
            depends_on_operator_ids TEXT,
            input_arity_min        INTEGER NOT NULL,
            input_arity_max        INTEGER,
            output_count           INTEGER NOT NULL DEFAULT 1,
            warmup_policy          TEXT NOT NULL,
            missing_data_policy    TEXT NOT NULL,
            deterministic          INTEGER NOT NULL CHECK (deterministic IN (0, 1)),
            stateful               INTEGER NOT NULL CHECK (stateful IN (0, 1)),
            streaming_supported    INTEGER NOT NULL CHECK (streaming_supported IN (0, 1)),
            incremental_supported  INTEGER NOT NULL CHECK (incremental_supported IN (0, 1)),
            complexity_class       TEXT NOT NULL,
            extras_json            TEXT,
            operator_version       TEXT NOT NULL,
            catalog_version        TEXT NOT NULL,
            operator_pack_version  TEXT NOT NULL,
            created_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (operator_id GLOB 'OP_*'),
            CHECK (output_count >= 1),
            CHECK (input_arity_min >= 0),
            CHECK (input_arity_max IS NULL OR input_arity_max >= input_arity_min),
            CHECK (warmup_policy IN ('NONE', 'WINDOW', 'CUSTOM')),
            CHECK (missing_data_policy IN (
                'DROP', 'FORWARD_FILL', 'BACKWARD_FILL', 'ZERO', 'NAN', 'ERROR'
            )),
            CHECK (complexity_class IN ('O(1)', 'O(n)', 'O(window)', 'O(log n)')),
            CHECK (category IN (
                'TREND', 'ROLLING', 'STATISTICAL', 'ARITHMETIC', 'NORMALIZATION',
                'TIME', 'COMPARISON', 'TRANSFORMATION', 'INTERACTION', 'OTHER'
            ))
        );
        CREATE INDEX IF NOT EXISTS idx_operator_id ON operator_registry(operator_id);
        CREATE INDEX IF NOT EXISTS idx_operator_canonical_name ON operator_registry(canonical_name);
        CREATE INDEX IF NOT EXISTS idx_operator_category ON operator_registry(category);
        CREATE INDEX IF NOT EXISTS idx_operator_pack_version ON operator_registry(operator_pack_version);
        CREATE INDEX IF NOT EXISTS idx_operator_complexity ON operator_registry(complexity_class);
        CREATE INDEX IF NOT EXISTS idx_operator_deterministic ON operator_registry(deterministic);
        """
    )
    for o in SEED_OPERATORS:
        conn.execute(
            """
            INSERT OR IGNORE INTO operator_registry(
                operator_id, canonical_name, display_name, category, description,
                formula, definition_text, parameter_schema_json, depends_on_operator_ids,
                input_arity_min, input_arity_max, output_count,
                warmup_policy, missing_data_policy,
                deterministic, stateful, streaming_supported, incremental_supported,
                complexity_class, extras_json,
                operator_version, catalog_version, operator_pack_version
            ) VALUES (?,?,?,?,?,?,?,?,NULL,?,?,1,?,?,?,?,?,?,?,NULL,?,?,?)
            """,
            (
                o.operator_id,
                o.canonical_name,
                o.display_name,
                o.category,
                o.description or None,
                o.formula,
                o.definition_text,
                o.parameter_schema_json,
                o.input_arity_min,
                o.input_arity_max,
                o.warmup_policy,
                o.missing_data_policy,
                o.deterministic,
                o.stateful,
                o.streaming_supported,
                o.incremental_supported,
                o.complexity_class,
                OPERATOR_VERSION_DEFAULT,
                OPERATOR_CATALOG_VERSION,
                OPERATOR_PACK_VERSION,
            ),
        )


def downgrade(conn: sqlite3.Connection) -> None:
    for idx in (
        "idx_operator_deterministic",
        "idx_operator_complexity",
        "idx_operator_pack_version",
        "idx_operator_category",
        "idx_operator_canonical_name",
        "idx_operator_id",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    conn.execute("DROP TABLE IF EXISTS operator_registry")
