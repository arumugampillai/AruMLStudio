"""SQLite persistence for the Primitive Catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feature_intelligence.registry.models import PrimitiveRecord


class PrimitiveStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PrimitiveRecord:
        return PrimitiveRecord(
            primitive_id=str(row["primitive_id"]),
            name=str(row["name"]),
            primitive_type=str(row["primitive_type"]),
            description=None if row["description"] is None else str(row["description"]),
            data_source=str(row["data_source"]),
            units=str(row["units"]),
            catalog_version=str(row["catalog_version"]),
            created_at=str(row["created_at"]),
        )

    def table_exists(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'primitive_registry'
                """
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def index_exists(self, index_name: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'index' AND name = ?
                """,
                (index_name,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def list_all(self) -> list[PrimitiveRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT primitive_id, name, primitive_type, description,
                       data_source, units, catalog_version, created_at
                FROM primitive_registry
                ORDER BY primitive_id ASC
                """
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_by_id(self, primitive_id: str) -> PrimitiveRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT primitive_id, name, primitive_type, description,
                       data_source, units, catalog_version, created_at
                FROM primitive_registry
                WHERE primitive_id = ?
                """,
                (primitive_id,),
            ).fetchone()
            return None if row is None else self._row_to_record(row)
        finally:
            conn.close()

    def get_by_name(self, name: str) -> PrimitiveRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT primitive_id, name, primitive_type, description,
                       data_source, units, catalog_version, created_at
                FROM primitive_registry
                WHERE name = ?
                """,
                (name,),
            ).fetchone()
            return None if row is None else self._row_to_record(row)
        finally:
            conn.close()

    def update_metadata(
        self,
        primitive_id: str,
        *,
        description: str | None = None,
        data_source: str | None = None,
        units: str | None = None,
    ) -> PrimitiveRecord:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM primitive_registry WHERE primitive_id = ?",
                (primitive_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown primitive_id: {primitive_id}")

            if description is not None:
                conn.execute(
                    "UPDATE primitive_registry SET description = ? WHERE primitive_id = ?",
                    (description, primitive_id),
                )
            if data_source is not None:
                conn.execute(
                    "UPDATE primitive_registry SET data_source = ? WHERE primitive_id = ?",
                    (data_source, primitive_id),
                )
            if units is not None:
                conn.execute(
                    "UPDATE primitive_registry SET units = ? WHERE primitive_id = ?",
                    (units, primitive_id),
                )
            conn.commit()
            updated = conn.execute(
                """
                SELECT primitive_id, name, primitive_type, description,
                       data_source, units, catalog_version, created_at
                FROM primitive_registry
                WHERE primitive_id = ?
                """,
                (primitive_id,),
            ).fetchone()
            assert updated is not None
            return self._row_to_record(updated)
        finally:
            conn.close()
