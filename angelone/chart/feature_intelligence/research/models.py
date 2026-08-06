"""Feature Research Record models (Sprint 8) — metadata shell only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

RESEARCH_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
RESEARCH_EXPORT_VERSION = "1.0"

RESEARCH_STATUSES: frozenset[str] = frozenset({"EMPTY", "ACTIVE", "ARCHIVED"})
VALIDATION_STATUSES: frozenset[str] = frozenset({"validated", "pending", "failed"})
RECORD_SOURCES: frozenset[str] = frozenset({"SYNC", "IMPORT", "MIGRATION"})

STATUS_EMPTY = "EMPTY"
STATUS_ACTIVE = "ACTIVE"
STATUS_ARCHIVED = "ARCHIVED"

VALIDATION_PENDING = "pending"
VALIDATION_VALIDATED = "validated"
VALIDATION_FAILED = "failed"

SOURCE_SYNC = "SYNC"
SOURCE_IMPORT = "IMPORT"
SOURCE_MIGRATION = "MIGRATION"


def normalize_experiment_ids(
    values: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Dedupe + ASCII ascending sort; None stays None (prefer NULL on sync create)."""
    if values is None:
        return None
    if not values:
        return []
    return sorted(set(str(v) for v in values), key=lambda s: s.encode("ascii", "replace"))


@dataclass(frozen=True)
class ResearchSyncSummary:
    created: int
    updated: int
    unchanged: int
    skipped: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class FeatureResearchRecord:
    research_uuid: str
    feature_uuid: str
    ontology_uuid: str | None = None
    transformation_uuid: str | None = None
    lineage_version: str | None = None
    compiler_version: str | None = None
    grammar_version: str | None = None
    research_status: str = STATUS_EMPTY
    validation_status: str = VALIDATION_PENDING
    evidence_json: str | None = None
    strengths_json: str | None = None
    weaknesses_json: str | None = None
    regimes_json: str | None = None
    failure_modes_json: str | None = None
    experiment_ids: list[str] | None = None
    notes: str | None = None
    record_source: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def normalized(self) -> FeatureResearchRecord:
        return FeatureResearchRecord(
            research_uuid=self.research_uuid,
            feature_uuid=self.feature_uuid,
            ontology_uuid=self.ontology_uuid,
            transformation_uuid=self.transformation_uuid,
            lineage_version=self.lineage_version,
            compiler_version=self.compiler_version,
            grammar_version=self.grammar_version,
            research_status=self.research_status,
            validation_status=self.validation_status,
            evidence_json=self.evidence_json,
            strengths_json=self.strengths_json,
            weaknesses_json=self.weaknesses_json,
            regimes_json=self.regimes_json,
            failure_modes_json=self.failure_modes_json,
            experiment_ids=normalize_experiment_ids(self.experiment_ids),
            notes=self.notes,
            record_source=self.record_source,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_dict(self, *, include_timestamps: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_timestamps:
            d.pop("created_at", None)
            d.pop("updated_at", None)
        return d

    def experiment_ids_json(self) -> str | None:
        ids = normalize_experiment_ids(self.experiment_ids)
        if ids is None:
            return None
        return json.dumps(ids, separators=(",", ":"))


@dataclass
class ResearchStatsReport:
    research_version: str
    schema_version: str
    total_frr: int
    expected_features: int
    coverage_pct: float
    status_empty: int
    status_active: int
    status_archived: int
    last_sync_at: str | None
    from_snapshot: bool
    snapshot_created_at: str | None = None
    stats_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompletenessGap:
    research_uuid: str
    feature_uuid: str
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompletenessReport:
    total_frr: int
    complete: int
    incomplete: int
    gaps: list[CompletenessGap] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frr": self.total_frr,
            "complete": self.complete,
            "incomplete": self.incomplete,
            "gaps": [g.to_dict() for g in self.gaps],
        }
