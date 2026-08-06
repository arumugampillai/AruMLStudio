"""Ontology service façade (Sprint 6) — coverage + single stats writer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.ontology.catalog import ONTOLOGY_VERSION
from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.models import (
    OBJECT_TYPE_FEATURE,
    OBJECT_TYPE_OPERATOR,
    OBJECT_TYPE_PRIMITIVE,
    OBJECT_TYPE_TRANSFORMATION,
    CoverageReport,
    CoverageTypeMetrics,
    OntologyRecord,
)
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.ontology.validation import validate_ontology
from feature_intelligence.registry.models import ValidationReport

_REQUIRED = frozenset({OBJECT_TYPE_PRIMITIVE, OBJECT_TYPE_OPERATOR})
_ALL_TYPES = (
    OBJECT_TYPE_PRIMITIVE,
    OBJECT_TYPE_OPERATOR,
    OBJECT_TYPE_TRANSFORMATION,
    OBJECT_TYPE_FEATURE,
)
_TYPE_PREFIX = {
    OBJECT_TYPE_PRIMITIVE: "pr",
    OBJECT_TYPE_OPERATOR: "op",
    OBJECT_TYPE_TRANSFORMATION: "tr",
    OBJECT_TYPE_FEATURE: "feat",
}


class OntologyNotFoundError(LookupError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _pct(classified: int, expected: int) -> float:
    if expected > 0:
        return 100.0 * classified / expected
    return 100.0


class OntologyService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.store = OntologyStore(self.db_path)

    def list_ontology(
        self, object_type: str | None = None
    ) -> list[OntologyRecord]:
        return self.store.list_ontology(object_type)

    def get_ontology(self, object_type: str, object_id: str) -> OntologyRecord:
        row = self.store.get_ontology(object_type, object_id)
        if row is None:
            raise OntologyNotFoundError(f"{object_type}:{object_id}")
        return row

    def ontology_exists(self, object_type: str, object_id: str) -> bool:
        return self.store.ontology_exists(object_type, object_id)

    @staticmethod
    def derive_ontology_uuid(object_type: str, object_id: str) -> str:
        return derive_ontology_uuid(object_type, object_id)

    def upsert_ontology(self, record: OntologyRecord) -> OntologyRecord:
        """In-process upsert (import/seed). Not exposed as CLI edit."""
        return self.store.upsert_ontology(record)

    # ------------------------------------------------------------------ stats
    def compute_coverage_metrics(self) -> CoverageReport:
        """Compute live coverage (does not write)."""
        by_type: dict[str, CoverageTypeMetrics] = {}
        total_exp = 0
        total_cls = 0
        for ot in _ALL_TYPES:
            expected = self.store.count_registry_objects(ot)
            classified = self.store.count_ontology(ot)
            missing = max(0, expected - classified)
            by_type[ot] = CoverageTypeMetrics(
                object_type=ot,
                expected=expected,
                classified=classified,
                missing=missing,
                coverage_pct=_pct(classified, expected),
            )
            total_exp += expected
            total_cls += classified
        missing_all = max(0, total_exp - total_cls)
        return CoverageReport(
            ontology_version=ONTOLOGY_VERSION,
            objects_total=total_exp,
            objects_classified=total_cls,
            objects_missing=missing_all,
            coverage_pct=_pct(total_cls, total_exp),
            by_type=by_type,
            from_snapshot=False,
            snapshot_created_at=None,
        )

    def write_coverage_snapshot(
        self, report: CoverageReport | None = None
    ) -> CoverageReport:
        """
        Single shared stats writer (freeze §9.4.1).

        Appends one ontology_statistics row. Only validate (always) and
        coverage (on miss) should call this — never import.
        """
        live = report or self.compute_coverage_metrics()
        created = _utc_now()
        payload = {
            "ontology_version": live.ontology_version,
            "objects_total": live.objects_total,
            "objects_classified": live.objects_classified,
            "objects_missing": live.objects_missing,
            "coverage_pct": live.coverage_pct,
            "created_at": created,
        }
        for ot, prefix in _TYPE_PREFIX.items():
            m = live.by_type[ot]
            payload[f"{prefix}_expected"] = m.expected
            payload[f"{prefix}_classified"] = m.classified
            payload[f"{prefix}_missing"] = m.missing
            payload[f"{prefix}_coverage_pct"] = m.coverage_pct
        self.store.insert_statistics(payload)
        return CoverageReport(
            ontology_version=live.ontology_version,
            objects_total=live.objects_total,
            objects_classified=live.objects_classified,
            objects_missing=live.objects_missing,
            coverage_pct=live.coverage_pct,
            by_type=live.by_type,
            from_snapshot=True,
            snapshot_created_at=created,
        )

    def _report_from_snapshot(self, snap: dict) -> CoverageReport:
        by_type: dict[str, CoverageTypeMetrics] = {}
        for ot, prefix in _TYPE_PREFIX.items():
            by_type[ot] = CoverageTypeMetrics(
                object_type=ot,
                expected=int(snap[f"{prefix}_expected"]),
                classified=int(snap[f"{prefix}_classified"]),
                missing=int(snap[f"{prefix}_missing"]),
                coverage_pct=float(snap[f"{prefix}_coverage_pct"]),
            )
        return CoverageReport(
            ontology_version=str(snap["ontology_version"]),
            objects_total=int(snap["objects_total"]),
            objects_classified=int(snap["objects_classified"]),
            objects_missing=int(snap["objects_missing"]),
            coverage_pct=float(snap["coverage_pct"]),
            by_type=by_type,
            from_snapshot=True,
            snapshot_created_at=str(snap["created_at"]),
        )

    def coverage_ontology(self) -> CoverageReport:
        """
        Read latest snapshot if present; regenerate + write only if none exists.
        """
        snap = self.store.latest_statistics()
        if snap is not None:
            return self._report_from_snapshot(snap)
        return self.write_coverage_snapshot()

    def validate_ontology(
        self,
        *,
        mode: str = "strict",
        strict_refs: bool = False,
    ) -> ValidationReport:
        """Always writes a coverage snapshot at end (pass or fail)."""
        report = validate_ontology(
            self.store, mode=mode, strict_refs=strict_refs
        )
        cov = self.compute_coverage_metrics()
        self.write_coverage_snapshot(cov)
        # Attach coverage summary as warning line for CLI visibility
        summary = (
            f"coverage: overall={cov.coverage_pct:.1f}% "
            f"PR={cov.by_type[OBJECT_TYPE_PRIMITIVE].coverage_pct:.1f}% "
            f"OP={cov.by_type[OBJECT_TYPE_OPERATOR].coverage_pct:.1f}% "
            f"TR={cov.by_type[OBJECT_TYPE_TRANSFORMATION].coverage_pct:.1f}% "
            f"FEAT={cov.by_type[OBJECT_TYPE_FEATURE].coverage_pct:.1f}%"
        )
        report.warnings.append(summary)
        return report
