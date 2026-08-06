"""Semantic Query models and versions (Sprint 9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

QUERY_ENGINE_VERSION = "1.0.0"
QUERY_LANGUAGE_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
QUERY_EXPORT_VERSION = "1.0"

QUERY_FIELDS: tuple[str, ...] = (
    "feature",
    "domain",
    "signal",
    "operator",
    "primitive",
    "transformation",
    "status",
    "validation",
    "grammar",
    "compiler",
    "ontology_version",
)

SUPPORTED_MODES: tuple[str, ...] = (
    "By Feature",
    "By Primitive",
    "By Operator",
    "By Ontology",
    "By Transformation",
    "By Research Status",
)

SUPPORTED_EXPORTS: tuple[str, ...] = ("json", "yaml", "csv")

STATUS_VALUES: frozenset[str] = frozenset({"EMPTY", "ACTIVE", "ARCHIVED"})
VALIDATION_VALUES: frozenset[str] = frozenset({"validated", "pending", "failed"})

INSPECTOR_SECTIONS: tuple[str, ...] = (
    "identity",
    "compiler",
    "ast",
    "ontology",
    "lineage",
    "research",
    "references",
)


@dataclass(frozen=True)
class QueryToken:
    field: str
    value: str
    raw: str


@dataclass(frozen=True)
class QuerySpec:
    tokens: tuple[QueryToken, ...]
    match_all: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_all": self.match_all,
            "tokens": [
                {"field": t.field, "value": t.value, "raw": t.raw} for t in self.tokens
            ],
        }


@dataclass
class QueryValidationReport:
    ok: bool
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spec: QuerySpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "spec": None if self.spec is None else self.spec.to_dict(),
        }


@dataclass
class SearchHit:
    research_uuid: str
    feature_uuid: str
    canonical_name: str | None = None
    research_status: str | None = None
    validation_status: str | None = None
    ontology_uuid: str | None = None
    transformation_uuid: str | None = None
    # Enriched grid fields (read-only joins via FRR → ontology / lineage / feature)
    domain: str | None = None
    primary_operator: str | None = None
    primary_primitive: str | None = None
    compiler_version: str | None = None
    grammar_version: str | None = None
    ontology_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApiEnvelope:
    ok: bool
    data: Any = None
    error: dict[str, str] | None = None
    execution_ms: float | None = None
    schema_version: str = SCHEMA_VERSION
    query_engine_version: str = QUERY_ENGINE_VERSION
    query_language_version: str = QUERY_LANGUAGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_engine_version": self.query_engine_version,
            "query_language_version": self.query_language_version,
            "ok": self.ok,
            "error": self.error,
            "execution_ms": self.execution_ms,
            "data": self.data,
        }


def capabilities_payload() -> dict[str, Any]:
    return {
        "query_language_version": QUERY_LANGUAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "query_engine_version": QUERY_ENGINE_VERSION,
        "supported_filters": list(QUERY_FIELDS),
        "supported_modes": list(SUPPORTED_MODES),
        "supported_exports": list(SUPPORTED_EXPORTS),
        "read_only": True,
        "frr_mandatory": True,
        # SavedQuery reserved — not part of Phase 1 capabilities
    }
