"""api package — thin public read-only surface (Sprint 9)."""

from __future__ import annotations

from feature_intelligence.api.public import (
    PUBLIC_CALLABLES,
    get_capabilities,
    get_feature,
    get_lineage,
    get_ontology,
    get_platform_summary,
    get_references,
    get_research,
    inspect_feature,
    search_features,
)
from feature_intelligence.query.models import (
    QUERY_ENGINE_VERSION,
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
    ApiEnvelope,
)

__all__ = [
    "ApiEnvelope",
    "PUBLIC_CALLABLES",
    "QUERY_ENGINE_VERSION",
    "QUERY_LANGUAGE_VERSION",
    "SCHEMA_VERSION",
    "get_capabilities",
    "get_feature",
    "get_lineage",
    "get_ontology",
    "get_platform_summary",
    "get_references",
    "get_research",
    "inspect_feature",
    "search_features",
]

