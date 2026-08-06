"""SQLite persistence for Feature Registry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feature_intelligence.registry.feature_models import FeatureRecord


class FeatureStore:
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
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_registry'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def index_exists(self, name: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                (name,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _primitives_for(self, conn: sqlite3.Connection, feature_uuid: str) -> tuple[str, ...]:
        rows = conn.execute(
            """
            SELECT primitive_id FROM feature_primitives
            WHERE feature_uuid = ?
            ORDER BY ordinal ASC, primitive_id ASC
            """,
            (feature_uuid,),
        ).fetchall()
        return tuple(str(r["primitive_id"]) for r in rows)

    def _row_to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> FeatureRecord:
        fu = str(row["feature_uuid"])
        return FeatureRecord(
            feature_uuid=fu,
            canonical_name=str(row["canonical_name"]),
            display_name=str(row["display_name"]),
            definition_version=str(row["definition_version"]),
            implementation_version=str(row["implementation_version"]),
            feature_version=None if row["feature_version"] is None else str(row["feature_version"]),
            definition_hash=str(row["definition_hash"]),
            transformation_uuid=(
                None if row["transformation_uuid"] is None else str(row["transformation_uuid"])
            ),
            legacy_feature_id=(
                None if row["legacy_feature_id"] is None else str(row["legacy_feature_id"])
            ),
            description=None if row["description"] is None else str(row["description"]),
            created_by=str(row["created_by"]),
            controller_owner=str(row["controller_owner"]),
            warmup_periods=int(row["warmup_periods"]),
            gap_policy=str(row["gap_policy"]),
            memory_model=str(row["memory_model"]),
            research_state=str(row["research_state"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            primitive_ids=self._primitives_for(conn, fu),
        )

    def insert(
        self,
        record: FeatureRecord,
        *,
        primitive_ids: list[str],
    ) -> FeatureRecord:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO feature_registry(
                    feature_uuid, canonical_name, display_name,
                    definition_version, implementation_version, feature_version,
                    definition_hash, transformation_uuid, legacy_feature_id,
                    description, created_by, controller_owner,
                    warmup_periods, gap_policy, memory_model, research_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.feature_uuid,
                    record.canonical_name,
                    record.display_name,
                    record.definition_version,
                    record.implementation_version,
                    record.feature_version,
                    record.definition_hash,
                    record.transformation_uuid,
                    record.legacy_feature_id,
                    record.description,
                    record.created_by,
                    record.controller_owner,
                    record.warmup_periods,
                    record.gap_policy,
                    record.memory_model,
                    record.research_state,
                ),
            )
            for i, pid in enumerate(primitive_ids):
                conn.execute(
                    """
                    INSERT INTO feature_primitives(feature_uuid, primitive_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (record.feature_uuid, pid, i),
                )
            conn.commit()
            return self.get_by_uuid(record.feature_uuid)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_by_uuid(self, feature_uuid: str) -> FeatureRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM feature_registry WHERE feature_uuid = ?",
                (feature_uuid,),
            ).fetchone()
            return None if row is None else self._row_to_record(conn, row)
        finally:
            conn.close()

    def get_by_name(self, canonical_name: str) -> FeatureRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM feature_registry WHERE canonical_name = ?",
                (canonical_name,),
            ).fetchone()
            return None if row is None else self._row_to_record(conn, row)
        finally:
            conn.close()

    def list_all(
        self,
        *,
        research_state: str | None = None,
        controller_owner: str | None = None,
    ) -> list[FeatureRecord]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM feature_registry WHERE 1=1"
            params: list[object] = []
            if research_state is not None:
                sql += " AND research_state = ?"
                params.append(research_state)
            if controller_owner is not None:
                sql += " AND controller_owner = ?"
                params.append(controller_owner)
            sql += " ORDER BY canonical_name ASC"
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_record(conn, r) for r in rows]
        finally:
            conn.close()

    def find_by_primitive(self, primitive_id: str) -> list[FeatureRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT fr.* FROM feature_registry fr
                INNER JOIN feature_primitives fp ON fp.feature_uuid = fr.feature_uuid
                WHERE fp.primitive_id = ?
                ORDER BY fr.canonical_name ASC
                """,
                (primitive_id,),
            ).fetchall()
            return [self._row_to_record(conn, r) for r in rows]
        finally:
            conn.close()

    def update_metadata(
        self,
        feature_uuid: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        controller_owner: str | None = None,
        research_state: str | None = None,
        implementation_version: str | None = None,
    ) -> FeatureRecord:
        conn = self._connect()
        try:
            if self.get_by_uuid(feature_uuid) is None:
                raise KeyError(feature_uuid)
            fields: list[str] = []
            params: list[object] = []
            if display_name is not None:
                fields.append("display_name = ?")
                params.append(display_name)
            if description is not None:
                fields.append("description = ?")
                params.append(description)
            if controller_owner is not None:
                fields.append("controller_owner = ?")
                params.append(controller_owner)
            if research_state is not None:
                fields.append("research_state = ?")
                params.append(research_state)
            if implementation_version is not None:
                fields.append("implementation_version = ?")
                params.append(implementation_version)
            if not fields:
                raise ValueError("No metadata fields to update")
            fields.append("updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')")
            params.append(feature_uuid)
            conn.execute(
                f"UPDATE feature_registry SET {', '.join(fields)} WHERE feature_uuid = ?",
                params,
            )
            conn.commit()
            out = self.get_by_uuid(feature_uuid)
            assert out is not None
            return out
        finally:
            conn.close()

    def replace_definition(
        self,
        feature_uuid: str,
        *,
        definition_version: str,
        warmup_periods: int,
        gap_policy: str,
        memory_model: str,
        definition_hash: str,
        primitive_ids: list[str],
    ) -> FeatureRecord:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE feature_registry SET
                    definition_version = ?,
                    warmup_periods = ?,
                    gap_policy = ?,
                    memory_model = ?,
                    definition_hash = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE feature_uuid = ?
                """,
                (
                    definition_version,
                    warmup_periods,
                    gap_policy,
                    memory_model,
                    definition_hash,
                    feature_uuid,
                ),
            )
            conn.execute(
                "DELETE FROM feature_primitives WHERE feature_uuid = ?",
                (feature_uuid,),
            )
            for i, pid in enumerate(primitive_ids):
                conn.execute(
                    """
                    INSERT INTO feature_primitives(feature_uuid, primitive_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (feature_uuid, pid, i),
                )
            conn.commit()
            out = self.get_by_uuid(feature_uuid)
            assert out is not None
            return out
        finally:
            conn.close()
