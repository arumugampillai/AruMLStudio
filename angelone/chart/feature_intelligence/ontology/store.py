"""Shared OntologyStore — four ontology tables + vocab + registry + stats (Sprint 6)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.models import (
    OBJECT_TYPE_TABLE,
    OBJECT_TYPES,
    OntologyRecord,
    VocabularyRecord,
    normalize_id_list,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _loads_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return [str(x) for x in data]


class OntologyStore:
    """One store parameterized by object_type → table map (freeze §11 / convention W)."""

    OBJECT_TYPE_TABLE = OBJECT_TYPE_TABLE

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _table(self, object_type: str) -> str:
        if object_type not in OBJECT_TYPE_TABLE:
            raise ValueError(f"Unknown object_type: {object_type!r}")
        return OBJECT_TYPE_TABLE[object_type]

    # ------------------------------------------------------------------ vocab
    def list_vocabularies(
        self, vocabulary_type: str | None = None
    ) -> list[VocabularyRecord]:
        conn = self._connect()
        try:
            if vocabulary_type:
                rows = conn.execute(
                    "SELECT * FROM vocabulary_registry WHERE vocabulary_type = ? "
                    "ORDER BY vocabulary_id ASC",
                    (vocabulary_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM vocabulary_registry ORDER BY vocabulary_id ASC"
                ).fetchall()
            return [self._vocab_row(r) for r in rows]
        finally:
            conn.close()

    def get_vocabulary(self, vocabulary_id: str) -> VocabularyRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM vocabulary_registry WHERE vocabulary_id = ?",
                (vocabulary_id,),
            ).fetchone()
            return None if row is None else self._vocab_row(row)
        finally:
            conn.close()

    def vocab_id_set(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT vocabulary_id FROM vocabulary_registry"
            ).fetchall()
            return {str(r[0]) for r in rows}
        finally:
            conn.close()

    def vocab_active_map(self) -> dict[str, bool]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT vocabulary_id, active FROM vocabulary_registry"
            ).fetchall()
            return {str(r[0]): bool(r[1]) for r in rows}
        finally:
            conn.close()

    def vocab_type_map(self) -> dict[str, str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT vocabulary_id, vocabulary_type FROM vocabulary_registry"
            ).fetchall()
            return {str(r[0]): str(r[1]) for r in rows}
        finally:
            conn.close()

    @staticmethod
    def _vocab_row(row: sqlite3.Row) -> VocabularyRecord:
        return VocabularyRecord(
            vocabulary_id=str(row["vocabulary_id"]),
            vocabulary_type=str(row["vocabulary_type"]),
            canonical_name=str(row["canonical_name"]),
            display_name=str(row["display_name"]),
            description=(
                None if row["description"] is None else str(row["description"])
            ),
            ontology_version=str(row["ontology_version"]),
            active=bool(row["active"]),
            retired_reason=(
                None
                if row["retired_reason"] is None
                else str(row["retired_reason"])
            ),
            sort_order=(
                None if row["sort_order"] is None else int(row["sort_order"])
            ),
            catalog_version=str(row["catalog_version"]),
            created_at=str(row["created_at"]),
        )

    # --------------------------------------------------------------- ontology
    def list_ontology(
        self, object_type: str | None = None
    ) -> list[OntologyRecord]:
        if object_type is not None:
            return self._list_table(object_type)
        out: list[OntologyRecord] = []
        for ot in sorted(OBJECT_TYPES):
            out.extend(self._list_table(ot))
        return out

    def _list_table(self, object_type: str) -> list[OntologyRecord]:
        table = self._table(object_type)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY object_id ASC"
            ).fetchall()
            return [self._ont_row(object_type, r) for r in rows]
        finally:
            conn.close()

    def get_ontology(
        self, object_type: str, object_id: str
    ) -> OntologyRecord | None:
        table = self._table(object_type)
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE object_id = ?",
                (object_id,),
            ).fetchone()
            return None if row is None else self._ont_row(object_type, row)
        finally:
            conn.close()

    def ontology_exists(self, object_type: str, object_id: str) -> bool:
        return self.get_ontology(object_type, object_id) is not None

    def count_ontology(self, object_type: str) -> int:
        table = self._table(object_type)
        conn = self._connect()
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def upsert_ontology(self, record: OntologyRecord) -> OntologyRecord:
        """In-place upsert; same ONT_* on re-class. Normalizes multi-fields."""
        rec = record.normalized()
        if not rec.ontology_uuid:
            rec.ontology_uuid = derive_ontology_uuid(rec.object_type, rec.object_id)
        table = self._table(rec.object_type)
        now = _utc_now()
        existing = self.get_ontology(rec.object_type, rec.object_id)
        created = existing.created_at if existing else (rec.created_at or now)
        updated = now
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT INTO {table} (
                    ontology_uuid, object_id, ontology_version, domain,
                    signal_type_json, mathematical_family_json, horizon,
                    output_type, frequency, stability, input_dependencies_json,
                    meaning, confidence, classification_source, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(object_id) DO UPDATE SET
                    ontology_version=excluded.ontology_version,
                    domain=excluded.domain,
                    signal_type_json=excluded.signal_type_json,
                    mathematical_family_json=excluded.mathematical_family_json,
                    horizon=excluded.horizon,
                    output_type=excluded.output_type,
                    frequency=excluded.frequency,
                    stability=excluded.stability,
                    input_dependencies_json=excluded.input_dependencies_json,
                    meaning=excluded.meaning,
                    confidence=excluded.confidence,
                    classification_source=excluded.classification_source,
                    updated_at=excluded.updated_at
                """,
                (
                    rec.ontology_uuid,
                    rec.object_id,
                    rec.ontology_version,
                    rec.domain,
                    rec.signal_type_json(),
                    rec.mathematical_family_json(),
                    rec.horizon,
                    rec.output_type,
                    rec.frequency,
                    rec.stability,
                    rec.input_dependencies_json(),
                    rec.meaning,
                    rec.confidence,
                    rec.classification_source,
                    created,
                    updated,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        out = self.get_ontology(rec.object_type, rec.object_id)
        assert out is not None
        return out

    @staticmethod
    def _ont_row(object_type: str, row: sqlite3.Row) -> OntologyRecord:
        return OntologyRecord(
            ontology_uuid=str(row["ontology_uuid"]),
            object_type=object_type,
            object_id=str(row["object_id"]),
            ontology_version=str(row["ontology_version"]),
            domain=str(row["domain"]),
            signal_type=_loads_list(str(row["signal_type_json"])),
            mathematical_family=_loads_list(str(row["mathematical_family_json"])),
            horizon=str(row["horizon"]),
            output_type=str(row["output_type"]),
            frequency=str(row["frequency"]),
            stability=str(row["stability"]),
            input_dependencies=_loads_list(str(row["input_dependencies_json"])),
            meaning=None if row["meaning"] is None else str(row["meaning"]),
            confidence=(
                None if row["confidence"] is None else float(row["confidence"])
            ),
            classification_source=(
                None
                if row["classification_source"] is None
                else str(row["classification_source"])
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    # --------------------------------------------------------------- registry
    def get_ontology_pack(self, version: str | None = None) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            if version:
                row = conn.execute(
                    "SELECT * FROM ontology_registry WHERE ontology_version = ?",
                    (version,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM ontology_registry ORDER BY ontology_version DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def ontology_versions(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT ontology_version FROM ontology_registry"
            ).fetchall()
            return {str(r[0]) for r in rows}
        finally:
            conn.close()

    # ------------------------------------------------------------- registry counts
    def count_registry_objects(self, object_type: str) -> int:
        """Count objects in the owning registry for coverage denominators."""
        queries = {
            "PRIMITIVE": "SELECT COUNT(*) FROM primitive_registry",
            "OPERATOR": "SELECT COUNT(*) FROM operator_registry",
            "TRANSFORMATION": "SELECT COUNT(*) FROM transformation_registry",
            "FEATURE": "SELECT COUNT(*) FROM feature_registry",
        }
        sql = queries.get(object_type)
        if sql is None:
            raise ValueError(f"Unknown object_type: {object_type!r}")
        conn = self._connect()
        try:
            # Tables may not exist in partial DBs — treat as 0
            try:
                row = conn.execute(sql).fetchone()
            except sqlite3.OperationalError:
                return 0
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def registry_object_ids(self, object_type: str) -> list[str]:
        col_table = {
            "PRIMITIVE": ("primitive_id", "primitive_registry"),
            "OPERATOR": ("operator_id", "operator_registry"),
            "TRANSFORMATION": ("transformation_uuid", "transformation_registry"),
            "FEATURE": ("feature_uuid", "feature_registry"),
        }
        col, table = col_table[object_type]
        conn = self._connect()
        try:
            try:
                rows = conn.execute(
                    f"SELECT {col} FROM {table} ORDER BY {col} ASC"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [str(r[0]) for r in rows]
        finally:
            conn.close()

    def object_exists_in_registry(self, object_type: str, object_id: str) -> bool:
        return object_id in set(self.registry_object_ids(object_type))

    # ----------------------------------------------------------------- stats
    def latest_statistics(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM ontology_statistics ORDER BY stats_id DESC LIMIT 1"
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            conn.close()

    def count_statistics(self) -> int:
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM ontology_statistics"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def insert_statistics(self, payload: dict[str, Any]) -> None:
        """Append a coverage snapshot (single writer path — call via service helper)."""
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO ontology_statistics (
                    ontology_version, objects_total, objects_classified,
                    objects_missing, coverage_pct,
                    pr_expected, pr_classified, pr_missing, pr_coverage_pct,
                    op_expected, op_classified, op_missing, op_coverage_pct,
                    tr_expected, tr_classified, tr_missing, tr_coverage_pct,
                    feat_expected, feat_classified, feat_missing, feat_coverage_pct,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    payload["ontology_version"],
                    payload["objects_total"],
                    payload["objects_classified"],
                    payload["objects_missing"],
                    payload["coverage_pct"],
                    payload["pr_expected"],
                    payload["pr_classified"],
                    payload["pr_missing"],
                    payload["pr_coverage_pct"],
                    payload["op_expected"],
                    payload["op_classified"],
                    payload["op_missing"],
                    payload["op_coverage_pct"],
                    payload["tr_expected"],
                    payload["tr_classified"],
                    payload["tr_missing"],
                    payload["tr_coverage_pct"],
                    payload["feat_expected"],
                    payload["feat_classified"],
                    payload["feat_missing"],
                    payload["feat_coverage_pct"],
                    payload["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()


# Re-export normalize for callers that import from store
normalize_json_id_list = normalize_id_list
