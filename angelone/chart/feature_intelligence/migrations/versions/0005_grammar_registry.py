"""Sprint 4 — Grammar registry table + pack 1.0.0 seed."""

from __future__ import annotations

import sqlite3

from feature_intelligence.grammar.pack import (
    COMPAT_PATH,
    EXPECTED_GRAMMAR_CHECKSUM,
    FORMATTER_VERSION,
    GRAMMAR_PACK_VERSION,
    GRAMMAR_VERSION,
    TOKEN_PACK_VERSION,
    compute_grammar_pack_checksum,
)

version = "0005"
description = "grammar_registry table and pack 1.0.0 seed"

EBNF_REL_PATH = "grammar/ebnf/tl_v1.ebnf"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS grammar_registry (
            grammar_version        TEXT PRIMARY KEY,
            grammar_pack_version   TEXT NOT NULL,
            token_pack_version     TEXT NOT NULL,
            formatter_version      TEXT NOT NULL,
            created_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            checksum               TEXT NOT NULL,
            ebnf_path              TEXT NOT NULL,
            compatibility_json     TEXT NOT NULL,
            notes                  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_grammar_version ON grammar_registry(grammar_version);
        CREATE INDEX IF NOT EXISTS idx_grammar_pack_version ON grammar_registry(grammar_pack_version);
        """
    )
    checksum = EXPECTED_GRAMMAR_CHECKSUM or compute_grammar_pack_checksum()
    compatibility_json = COMPAT_PATH.read_text(encoding="utf-8")
    conn.execute(
        """
        INSERT OR IGNORE INTO grammar_registry(
            grammar_version, grammar_pack_version, token_pack_version,
            formatter_version, checksum, ebnf_path, compatibility_json, notes
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            GRAMMAR_VERSION,
            GRAMMAR_PACK_VERSION,
            TOKEN_PACK_VERSION,
            FORMATTER_VERSION,
            checksum,
            EBNF_REL_PATH,
            compatibility_json,
            "Sprint 4 TL Grammar 1.0 seed",
        ),
    )


def downgrade(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_grammar_pack_version")
    conn.execute("DROP INDEX IF EXISTS idx_grammar_version")
    conn.execute("DROP TABLE IF EXISTS grammar_registry")
