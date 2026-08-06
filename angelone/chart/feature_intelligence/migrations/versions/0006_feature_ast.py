"""Sprint 5 — Feature AST, transformation / compilation registries, statistics."""

from __future__ import annotations

import sqlite3

version = "0006"
description = (
    "transformation_registry, compilation_registry, feature_ast, "
    "ast_nodes, compiler_statistics (no dual AST JSON)"
)


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transformation_registry (
            transformation_uuid     TEXT PRIMARY KEY,
            expression_hash         TEXT NOT NULL UNIQUE,
            source_text             TEXT,
            canonical_text          TEXT NOT NULL UNIQUE,
            created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (transformation_uuid GLOB 'TR_*'),
            CHECK (length(expression_hash) = 64),
            CHECK (length(transformation_uuid) = 35)
        );
        CREATE INDEX IF NOT EXISTS idx_transformation_expression_hash
            ON transformation_registry(expression_hash);

        CREATE TABLE IF NOT EXISTS compilation_registry (
            compilation_uuid        TEXT PRIMARY KEY,
            transformation_uuid     TEXT NOT NULL,
            ast_schema_version      TEXT NOT NULL,
            grammar_version         TEXT NOT NULL,
            compiler_version        TEXT NOT NULL,
            operator_pack_version   TEXT NOT NULL,
            ast_hash                TEXT NOT NULL,
            root_node_id            TEXT NOT NULL,
            cache_hit               INTEGER NOT NULL DEFAULT 0,
            diagnostics_json        TEXT,
            warnings_json           TEXT,
            metrics_json            TEXT,
            status                  TEXT NOT NULL,
            compiled_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            FOREIGN KEY (transformation_uuid)
                REFERENCES transformation_registry(transformation_uuid),
            CHECK (compilation_uuid GLOB 'COMP_*'),
            CHECK (length(ast_hash) = 64),
            CHECK (cache_hit IN (0, 1)),
            CHECK (status IN ('success', 'failed'))
        );
        CREATE INDEX IF NOT EXISTS idx_compilation_transformation
            ON compilation_registry(transformation_uuid);
        CREATE INDEX IF NOT EXISTS idx_compilation_ast_hash
            ON compilation_registry(ast_hash);
        CREATE INDEX IF NOT EXISTS idx_compilation_compiler_version
            ON compilation_registry(compiler_version);
        CREATE INDEX IF NOT EXISTS idx_compilation_compiled_at
            ON compilation_registry(compiled_at);
        CREATE INDEX IF NOT EXISTS idx_compilation_cache_hit
            ON compilation_registry(cache_hit);

        CREATE TABLE IF NOT EXISTS ast_nodes (
            transformation_uuid TEXT NOT NULL,
            ast_hash            TEXT NOT NULL,
            node_id             TEXT NOT NULL,
            node_type           TEXT NOT NULL,
            parent_node_id      TEXT,
            param_name          TEXT,
            operator_id         TEXT,
            primitive_id        TEXT,
            feature_uuid        TEXT,
            literal_json        TEXT,
            child_node_ids_json TEXT NOT NULL,
            subtree_hash        TEXT NOT NULL,
            stable_node_hash    TEXT,
            ordinal             INTEGER NOT NULL,
            PRIMARY KEY (transformation_uuid, ast_hash, node_id),
            FOREIGN KEY (transformation_uuid)
                REFERENCES transformation_registry(transformation_uuid),
            CHECK (node_type IN ('operator', 'primitive', 'feature', 'literal', 'list')),
            CHECK (length(subtree_hash) = 64),
            CHECK (length(ast_hash) = 64)
        );
        CREATE INDEX IF NOT EXISTS idx_ast_nodes_subtree_hash ON ast_nodes(subtree_hash);
        CREATE INDEX IF NOT EXISTS idx_ast_nodes_operator_id ON ast_nodes(operator_id);
        CREATE INDEX IF NOT EXISTS idx_ast_nodes_parent
            ON ast_nodes(transformation_uuid, ast_hash, parent_node_id);
        CREATE INDEX IF NOT EXISTS idx_ast_nodes_ast_hash ON ast_nodes(ast_hash);

        CREATE TABLE IF NOT EXISTS feature_ast (
            feature_uuid          TEXT PRIMARY KEY,
            transformation_uuid   TEXT NOT NULL,
            compilation_uuid      TEXT NOT NULL,
            ast_schema_version    TEXT NOT NULL,
            ast_json              TEXT NOT NULL,
            ast_fingerprint       TEXT NOT NULL,
            subtree_hash          TEXT NOT NULL,
            created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            FOREIGN KEY (feature_uuid)
                REFERENCES feature_registry(feature_uuid),
            FOREIGN KEY (transformation_uuid)
                REFERENCES transformation_registry(transformation_uuid),
            FOREIGN KEY (compilation_uuid)
                REFERENCES compilation_registry(compilation_uuid),
            CHECK (length(ast_fingerprint) = 64),
            CHECK (length(subtree_hash) = 64)
        );
        CREATE INDEX IF NOT EXISTS idx_feature_ast_fingerprint ON feature_ast(ast_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_feature_ast_subtree_hash ON feature_ast(subtree_hash);
        CREATE INDEX IF NOT EXISTS idx_feature_ast_transformation
            ON feature_ast(transformation_uuid);
        CREATE INDEX IF NOT EXISTS idx_feature_ast_compilation
            ON feature_ast(compilation_uuid);

        CREATE TABLE IF NOT EXISTS compiler_statistics (
            id                   INTEGER PRIMARY KEY CHECK (id = 1),
            total_compiles       INTEGER NOT NULL DEFAULT 0,
            cache_hits           INTEGER NOT NULL DEFAULT 0,
            cache_misses         INTEGER NOT NULL DEFAULT 0,
            average_compile_ms   REAL,
            updated_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        INSERT OR IGNORE INTO compiler_statistics(id, total_compiles, cache_hits, cache_misses)
        VALUES (1, 0, 0, 0);
        """
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_feature_ast_compilation;
        DROP INDEX IF EXISTS idx_feature_ast_transformation;
        DROP INDEX IF EXISTS idx_feature_ast_subtree_hash;
        DROP INDEX IF EXISTS idx_feature_ast_fingerprint;
        DROP TABLE IF EXISTS feature_ast;

        DROP INDEX IF EXISTS idx_ast_nodes_ast_hash;
        DROP INDEX IF EXISTS idx_ast_nodes_parent;
        DROP INDEX IF EXISTS idx_ast_nodes_operator_id;
        DROP INDEX IF EXISTS idx_ast_nodes_subtree_hash;
        DROP TABLE IF EXISTS ast_nodes;

        DROP INDEX IF EXISTS idx_compilation_cache_hit;
        DROP INDEX IF EXISTS idx_compilation_compiled_at;
        DROP INDEX IF EXISTS idx_compilation_compiler_version;
        DROP INDEX IF EXISTS idx_compilation_ast_hash;
        DROP INDEX IF EXISTS idx_compilation_transformation;
        DROP TABLE IF EXISTS compilation_registry;

        DROP INDEX IF EXISTS idx_transformation_expression_hash;
        DROP TABLE IF EXISTS transformation_registry;

        DROP TABLE IF EXISTS compiler_statistics;
        """
    )
