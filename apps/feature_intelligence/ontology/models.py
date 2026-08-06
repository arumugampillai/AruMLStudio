"""Ontology data models (Sprint 6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Object types (canonical hash / import envelope literals)
OBJECT_TYPE_PRIMITIVE = "PRIMITIVE"
OBJECT_TYPE_OPERATOR = "OPERATOR"
OBJECT_TYPE_TRANSFORMATION = "TRANSFORMATION"
OBJECT_TYPE_FEATURE = "FEATURE"

OBJECT_TYPES: frozenset[str] = frozenset(
    {
        OBJECT_TYPE_PRIMITIVE,
        OBJECT_TYPE_OPERATOR,
        OBJECT_TYPE_TRANSFORMATION,
        OBJECT_TYPE_FEATURE,
    }
)

# Single mapping: object_type → physical table (no scattered if/elif)
OBJECT_TYPE_TABLE: dict[str, str] = {
    OBJECT_TYPE_PRIMITIVE: "primitive_ontology",
    OBJECT_TYPE_OPERATOR: "operator_ontology",
    OBJECT_TYPE_TRANSFORMATION: "transformation_ontology",
    OBJECT_TYPE_FEATURE: "feature_ontology",
}

TABLE_OBJECT_TYPE: dict[str, str] = {v: k for k, v in OBJECT_TYPE_TABLE.items()}

CLASSIFICATION_SOURCES: frozenset[str] = frozenset({"SEED", "IMPORT", "MIGRATION"})

VOCABULARY_TYPES: frozenset[str] = frozenset(
    {
        "DOMAIN",
        "SIGNAL_TYPE",
        "MATH_FAMILY",
        "HORIZON",
        "OUTPUT_TYPE",
        "FREQUENCY",
        "STABILITY",
    }
)

ONTOLOGY_VERSION = "1.0.0"
VOCAB_PACK_VERSION = "1.0.0"
VOCAB_CATALOG_VERSION = "1.0"


def normalize_id_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Single shared helper: dedupe + ASCII ascending sort (freeze §11 convention 2)."""
    if not values:
        return []
    return sorted(set(str(v) for v in values), key=lambda s: s.encode("ascii", "replace"))


@dataclass(frozen=True)
class VocabularyRecord:
    vocabulary_id: str
    vocabulary_type: str
    canonical_name: str
    display_name: str
    description: str | None
    ontology_version: str
    active: bool
    retired_reason: str | None = None
    sort_order: int | None = None
    catalog_version: str = VOCAB_CATALOG_VERSION
    created_at: str = ""
    # vocabulary_pk is NEVER part of this public record


@dataclass
class OntologyRecord:
    ontology_uuid: str
    object_type: str
    object_id: str
    ontology_version: str
    domain: str
    signal_type: list[str]
    mathematical_family: list[str]
    horizon: str
    output_type: str
    frequency: str
    stability: str
    input_dependencies: list[str] = field(default_factory=list)
    meaning: str | None = None
    confidence: float | None = None
    classification_source: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def normalized(self) -> OntologyRecord:
        """Return a copy with multi-fields normalized."""
        return OntologyRecord(
            ontology_uuid=self.ontology_uuid,
            object_type=self.object_type,
            object_id=self.object_id,
            ontology_version=self.ontology_version,
            domain=self.domain,
            signal_type=normalize_id_list(self.signal_type),
            mathematical_family=normalize_id_list(self.mathematical_family),
            horizon=self.horizon,
            output_type=self.output_type,
            frequency=self.frequency,
            stability=self.stability,
            input_dependencies=normalize_id_list(self.input_dependencies),
            meaning=self.meaning,
            confidence=self.confidence,
            classification_source=self.classification_source,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_dict(self, *, include_timestamps: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_timestamps:
            d.pop("created_at", None)
            d.pop("updated_at", None)
        return d

    def signal_type_json(self) -> str:
        return json.dumps(normalize_id_list(self.signal_type), separators=(",", ":"))

    def mathematical_family_json(self) -> str:
        return json.dumps(
            normalize_id_list(self.mathematical_family), separators=(",", ":")
        )

    def input_dependencies_json(self) -> str:
        return json.dumps(
            normalize_id_list(self.input_dependencies), separators=(",", ":")
        )


@dataclass
class CoverageTypeMetrics:
    object_type: str
    expected: int
    classified: int
    missing: int
    coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageReport:
    ontology_version: str
    objects_total: int
    objects_classified: int
    objects_missing: int
    coverage_pct: float
    by_type: dict[str, CoverageTypeMetrics]
    from_snapshot: bool
    snapshot_created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "objects_total": self.objects_total,
            "objects_classified": self.objects_classified,
            "objects_missing": self.objects_missing,
            "coverage_pct": self.coverage_pct,
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "from_snapshot": self.from_snapshot,
            "snapshot_created_at": self.snapshot_created_at,
        }
