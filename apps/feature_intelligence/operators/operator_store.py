"""SQLite persistence for Operator Registry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feature_intelligence.operators.operator_models import OperatorRecord


class OperatorStore:
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
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='operator_registry'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> OperatorRecord:
        return OperatorRecord(
            operator_id=str(row["operator_id"]),
            canonical_name=str(row["canonical_name"]),
            display_name=str(row["display_name"]),
            category=str(row["category"]),
            description=None if row["description"] is None else str(row["description"]),
            formula=str(row["formula"]),
            definition_text=str(row["definition_text"]),
            parameter_schema_json=str(row["parameter_schema_json"]),
            depends_on_operator_ids=(
                None
                if row["depends_on_operator_ids"] is None
                else str(row["depends_on_operator_ids"])
            ),
            input_arity_min=int(row["input_arity_min"]),
            input_arity_max=(
                None if row["input_arity_max"] is None else int(row["input_arity_max"])
            ),
            output_count=int(row["output_count"]),
            warmup_policy=str(row["warmup_policy"]),
            missing_data_policy=str(row["missing_data_policy"]),
            deterministic=bool(row["deterministic"]),
            stateful=bool(row["stateful"]),
            streaming_supported=bool(row["streaming_supported"]),
            incremental_supported=bool(row["incremental_supported"]),
            complexity_class=str(row["complexity_class"]),
            extras_json=None if row["extras_json"] is None else str(row["extras_json"]),
            operator_version=str(row["operator_version"]),
            catalog_version=str(row["catalog_version"]),
            operator_pack_version=str(row["operator_pack_version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list_all(self) -> list[OperatorRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM operator_registry ORDER BY operator_id ASC"
            ).fetchall()
            return [self._row(r) for r in rows]
        finally:
            conn.close()

    def get_by_id(self, operator_id: str) -> OperatorRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM operator_registry WHERE operator_id = ?",
                (operator_id,),
            ).fetchone()
            return None if row is None else self._row(row)
        finally:
            conn.close()

    def get_by_name(self, canonical_name: str) -> OperatorRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM operator_registry WHERE canonical_name = ?",
                (canonical_name,),
            ).fetchone()
            return None if row is None else self._row(row)
        finally:
            conn.close()

    def insert(self, record: OperatorRecord) -> OperatorRecord:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO operator_registry(
                    operator_id, canonical_name, display_name, category, description,
                    formula, definition_text, parameter_schema_json, depends_on_operator_ids,
                    input_arity_min, input_arity_max, output_count,
                    warmup_policy, missing_data_policy,
                    deterministic, stateful, streaming_supported, incremental_supported,
                    complexity_class, extras_json,
                    operator_version, catalog_version, operator_pack_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.operator_id,
                    record.canonical_name,
                    record.display_name,
                    record.category,
                    record.description,
                    record.formula,
                    record.definition_text,
                    record.parameter_schema_json,
                    record.depends_on_operator_ids,
                    record.input_arity_min,
                    record.input_arity_max,
                    record.output_count,
                    record.warmup_policy,
                    record.missing_data_policy,
                    1 if record.deterministic else 0,
                    1 if record.stateful else 0,
                    1 if record.streaming_supported else 0,
                    1 if record.incremental_supported else 0,
                    record.complexity_class,
                    record.extras_json,
                    record.operator_version,
                    record.catalog_version,
                    record.operator_pack_version,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        out = self.get_by_id(record.operator_id)
        assert out is not None
        return out

    def update_metadata(
        self,
        operator_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        extras_json: str | None = None,
    ) -> OperatorRecord:
        conn = self._connect()
        try:
            if self.get_by_id(operator_id) is None:
                raise KeyError(operator_id)
            fields: list[str] = []
            params: list[object] = []
            if display_name is not None:
                fields.append("display_name = ?")
                params.append(display_name)
            if description is not None:
                fields.append("description = ?")
                params.append(description)
            if extras_json is not None:
                fields.append("extras_json = ?")
                params.append(extras_json)
            if not fields:
                raise ValueError("No metadata fields to update")
            fields.append("updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')")
            params.append(operator_id)
            conn.execute(
                f"UPDATE operator_registry SET {', '.join(fields)} WHERE operator_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()
        out = self.get_by_id(operator_id)
        assert out is not None
        return out
