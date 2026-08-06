"""SQLite persistence for grammar_registry (Sprint 4)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feature_intelligence.grammar.models import GrammarRegistryRecord


class GrammarStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def table_exists(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='grammar_registry'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> GrammarRegistryRecord:
        return GrammarRegistryRecord(
            grammar_version=str(row["grammar_version"]),
            grammar_pack_version=str(row["grammar_pack_version"]),
            token_pack_version=str(row["token_pack_version"]),
            formatter_version=str(row["formatter_version"]),
            checksum=str(row["checksum"]),
            ebnf_path=str(row["ebnf_path"]),
            compatibility_json=str(row["compatibility_json"]),
            notes=None if row["notes"] is None else str(row["notes"]),
            created_at=str(row["created_at"]),
        )

    def get(self, grammar_version: str) -> GrammarRegistryRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM grammar_registry WHERE grammar_version = ?",
                (grammar_version,),
            ).fetchone()
            return None if row is None else self._row(row)
        finally:
            conn.close()

    def list_all(self) -> list[GrammarRegistryRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM grammar_registry ORDER BY grammar_version ASC"
            ).fetchall()
            return [self._row(r) for r in rows]
        finally:
            conn.close()

    def upsert(self, record: GrammarRegistryRecord) -> GrammarRegistryRecord:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO grammar_registry(
                    grammar_version, grammar_pack_version, token_pack_version,
                    formatter_version, checksum, ebnf_path, compatibility_json, notes
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(grammar_version) DO UPDATE SET
                    grammar_pack_version=excluded.grammar_pack_version,
                    token_pack_version=excluded.token_pack_version,
                    formatter_version=excluded.formatter_version,
                    checksum=excluded.checksum,
                    ebnf_path=excluded.ebnf_path,
                    compatibility_json=excluded.compatibility_json,
                    notes=excluded.notes
                """,
                (
                    record.grammar_version,
                    record.grammar_pack_version,
                    record.token_pack_version,
                    record.formatter_version,
                    record.checksum,
                    record.ebnf_path,
                    record.compatibility_json,
                    record.notes,
                ),
            )
            conn.commit()
            out = self.get(record.grammar_version)
            assert out is not None
            return out
        finally:
            conn.close()
