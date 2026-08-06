"""ResearchStore — pack + FRR rows + statistics (Sprint 8)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.research.identity import (
    derive_research_uuid,
    normalize_feature_uuid,
)
from feature_intelligence.research.models import (
    RESEARCH_VERSION,
    SCHEMA_VERSION,
    FeatureResearchRecord,
    normalize_experiment_ids,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def empty_research_checksum() -> str:
    return hashlib.sha256(b"").hexdigest()


def compute_research_checksum(rows: list[FeatureResearchRecord]) -> str:
    """
    Canonical checksum: sort by research_uuid ASCII ascending;
    UTF-8 lines research_uuid\\tfeature_uuid\\tontology\\ttransformation\\tstatus\\tvalidation\\n
    """
    ordered = sorted(rows, key=lambda r: r.research_uuid)
    lines: list[str] = []
    for r in ordered:
        ont = r.ontology_uuid or ""
        tr = r.transformation_uuid or ""
        lines.append(
            f"{r.research_uuid}\t{r.feature_uuid}\t{ont}\t{tr}\t"
            f"{r.research_status}\t{r.validation_status}\n"
        )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


class ResearchStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # -------------------------------------------------------------- pack meta
    def get_pack(
        self, research_version: str = RESEARCH_VERSION
    ) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM research_registry WHERE research_version = ?",
                (research_version,),
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            conn.close()

    def list_pack_versions(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT research_version FROM research_registry"
            ).fetchall()
            return {str(r[0]) for r in rows}
        finally:
            conn.close()

    def update_checksum(
        self,
        checksum: str,
        *,
        research_version: str = RESEARCH_VERSION,
    ) -> None:
        now = _utc_now()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE research_registry
                SET checksum = ?, updated_at = ?
                WHERE research_version = ?
                """,
                (checksum, now, research_version),
            )
            conn.commit()
        finally:
            conn.close()

    def recompute_and_store_checksum(
        self, *, research_version: str = RESEARCH_VERSION
    ) -> str:
        """Shared checksum writer used by validate / sync / import."""
        checksum = compute_research_checksum(self.list_records())
        self.update_checksum(checksum, research_version=research_version)
        return checksum

    # ---------------------------------------------------------------- records
    def list_records(
        self, status: str | None = None
    ) -> list[FeatureResearchRecord]:
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM feature_research_record
                    WHERE research_status = ?
                    ORDER BY research_uuid ASC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM feature_research_record
                    ORDER BY research_uuid ASC
                    """
                ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_by_uuid(self, research_uuid: str) -> FeatureResearchRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM feature_research_record WHERE research_uuid = ?",
                (research_uuid,),
            ).fetchone()
            return None if row is None else self._row_to_record(row)
        finally:
            conn.close()

    def get_by_feature(
        self, feature_uuid: str
    ) -> FeatureResearchRecord | None:
        feat = normalize_feature_uuid(feature_uuid)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM feature_research_record WHERE feature_uuid = ?",
                (feat,),
            ).fetchone()
            return None if row is None else self._row_to_record(row)
        finally:
            conn.close()

    def count_records(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM feature_research_record"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def upsert_record(self, rec: FeatureResearchRecord) -> FeatureResearchRecord:
        """Idempotent upsert by research_uuid / feature_uuid."""
        feat = normalize_feature_uuid(rec.feature_uuid)
        uuid = rec.research_uuid or derive_research_uuid(feat)
        now = _utc_now()
        existing = self.get_by_uuid(uuid)
        if existing is None:
            existing = self.get_by_feature(feat)
        created = existing.created_at if existing else (rec.created_at or now)
        exp_json = rec.experiment_ids_json()
        if exp_json is None and existing is not None and rec.experiment_ids is None:
            # Preserve existing experiment_ids when caller leaves None
            exp_json = existing.experiment_ids_json()

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO feature_research_record (
                    research_uuid, feature_uuid, ontology_uuid, transformation_uuid,
                    lineage_version, compiler_version, grammar_version,
                    research_status, validation_status,
                    evidence_json, strengths_json, weaknesses_json,
                    regimes_json, failure_modes_json, experiment_ids, notes,
                    record_source, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(research_uuid) DO UPDATE SET
                    feature_uuid = excluded.feature_uuid,
                    ontology_uuid = excluded.ontology_uuid,
                    transformation_uuid = excluded.transformation_uuid,
                    lineage_version = excluded.lineage_version,
                    compiler_version = excluded.compiler_version,
                    grammar_version = excluded.grammar_version,
                    research_status = excluded.research_status,
                    validation_status = excluded.validation_status,
                    evidence_json = excluded.evidence_json,
                    strengths_json = excluded.strengths_json,
                    weaknesses_json = excluded.weaknesses_json,
                    regimes_json = excluded.regimes_json,
                    failure_modes_json = excluded.failure_modes_json,
                    experiment_ids = excluded.experiment_ids,
                    notes = excluded.notes,
                    record_source = COALESCE(excluded.record_source,
                                             feature_research_record.record_source),
                    updated_at = excluded.updated_at
                """,
                (
                    uuid,
                    feat,
                    rec.ontology_uuid,
                    rec.transformation_uuid,
                    rec.lineage_version,
                    rec.compiler_version,
                    rec.grammar_version,
                    rec.research_status,
                    rec.validation_status,
                    rec.evidence_json,
                    rec.strengths_json,
                    rec.weaknesses_json,
                    rec.regimes_json,
                    rec.failure_modes_json,
                    exp_json,
                    rec.notes,
                    rec.record_source,
                    created,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        out = self.get_by_uuid(uuid)
        assert out is not None
        return out

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeatureResearchRecord:
        exp_raw = row["experiment_ids"]
        experiment_ids: list[str] | None = None
        if exp_raw is not None:
            try:
                parsed = json.loads(str(exp_raw))
                if isinstance(parsed, list):
                    experiment_ids = normalize_experiment_ids(
                        [str(x) for x in parsed]
                    )
            except (json.JSONDecodeError, TypeError):
                experiment_ids = None
        return FeatureResearchRecord(
            research_uuid=str(row["research_uuid"]),
            feature_uuid=str(row["feature_uuid"]),
            ontology_uuid=(
                None if row["ontology_uuid"] is None else str(row["ontology_uuid"])
            ),
            transformation_uuid=(
                None
                if row["transformation_uuid"] is None
                else str(row["transformation_uuid"])
            ),
            lineage_version=(
                None if row["lineage_version"] is None else str(row["lineage_version"])
            ),
            compiler_version=(
                None
                if row["compiler_version"] is None
                else str(row["compiler_version"])
            ),
            grammar_version=(
                None if row["grammar_version"] is None else str(row["grammar_version"])
            ),
            research_status=str(row["research_status"]),
            validation_status=str(row["validation_status"]),
            evidence_json=(
                None if row["evidence_json"] is None else str(row["evidence_json"])
            ),
            strengths_json=(
                None if row["strengths_json"] is None else str(row["strengths_json"])
            ),
            weaknesses_json=(
                None if row["weaknesses_json"] is None else str(row["weaknesses_json"])
            ),
            regimes_json=(
                None if row["regimes_json"] is None else str(row["regimes_json"])
            ),
            failure_modes_json=(
                None
                if row["failure_modes_json"] is None
                else str(row["failure_modes_json"])
            ),
            experiment_ids=experiment_ids,
            notes=None if row["notes"] is None else str(row["notes"]),
            record_source=(
                None if row["record_source"] is None else str(row["record_source"])
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    # ------------------------------------------------------------- registries
    def list_feature_uuids(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT feature_uuid FROM feature_registry ORDER BY feature_uuid ASC"
            ).fetchall()
            return [normalize_feature_uuid(str(r[0])) for r in rows]
        finally:
            conn.close()

    def feature_exists(self, feature_uuid: str) -> bool:
        feat = normalize_feature_uuid(feature_uuid)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM feature_registry WHERE feature_uuid = ?",
                (feat,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def resolve_ontology_uuid(self, feature_uuid: str) -> str | None:
        """Lookup ONT_* for FEATURE object_id when present."""
        feat = normalize_feature_uuid(feature_uuid)
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    """
                    SELECT ontology_uuid FROM feature_ontology
                    WHERE object_id = ?
                    """,
                    (feat,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
            return None if row is None else str(row[0])
        finally:
            conn.close()

    def resolve_transformation_uuid(self, feature_uuid: str) -> str | None:
        feat = normalize_feature_uuid(feature_uuid)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT transformation_uuid FROM feature_registry WHERE feature_uuid = ?",
                (feat,),
            ).fetchone()
            if row is not None and row[0] is not None:
                return str(row[0])
            try:
                row2 = conn.execute(
                    "SELECT transformation_uuid FROM feature_ast WHERE feature_uuid = ?",
                    (feat,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
            return None if row2 is None else str(row2[0])
        finally:
            conn.close()

    def ontology_exists(self, ontology_uuid: str) -> bool:
        conn = self._connect()
        try:
            for table in (
                "feature_ontology",
                "primitive_ontology",
                "operator_ontology",
                "transformation_ontology",
            ):
                try:
                    row = conn.execute(
                        f"SELECT 1 FROM {table} WHERE ontology_uuid = ?",
                        (ontology_uuid,),
                    ).fetchone()
                except sqlite3.OperationalError:
                    continue
                if row is not None:
                    return True
            return False
        finally:
            conn.close()

    def transformation_exists(self, transformation_uuid: str) -> bool:
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT 1 FROM transformation_registry WHERE transformation_uuid = ?",
                    (transformation_uuid,),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
            return row is not None
        finally:
            conn.close()

    def known_lineage_versions(self) -> set[str]:
        conn = self._connect()
        try:
            try:
                rows = conn.execute(
                    "SELECT lineage_version FROM lineage_registry"
                ).fetchall()
            except sqlite3.OperationalError:
                return set()
            return {str(r[0]) for r in rows}
        finally:
            conn.close()

    def known_compiler_versions(self) -> set[str]:
        from feature_intelligence.compiler.models import COMPILER_VERSION

        return {COMPILER_VERSION}

    def known_grammar_versions(self) -> set[str]:
        from feature_intelligence.grammar.pack import GRAMMAR_VERSION

        return {GRAMMAR_VERSION}

    # ----------------------------------------------------------------- stats
    def latest_statistics(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM research_statistics ORDER BY stats_id DESC LIMIT 1"
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            conn.close()

    def count_statistics(self) -> int:
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM research_statistics"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def insert_statistics(self, payload: dict[str, Any]) -> int:
        """Append research_statistics row. Single writer path via statistics helper."""
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO research_statistics (
                    research_version, schema_version, total_frr, expected_features,
                    coverage_pct, status_empty, status_active, status_archived,
                    last_sync_at, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    payload["research_version"],
                    payload["schema_version"],
                    payload["total_frr"],
                    payload["expected_features"],
                    payload["coverage_pct"],
                    payload["status_empty"],
                    payload["status_active"],
                    payload["status_archived"],
                    payload.get("last_sync_at"),
                    payload["created_at"],
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()
