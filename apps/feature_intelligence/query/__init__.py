"""query package — Semantic Query Engine (Sprint 9)."""

from __future__ import annotations

from feature_intelligence.query.engine import QueryEngine, coverage_gate
from feature_intelligence.query.models import (
    QUERY_ENGINE_VERSION,
    QUERY_EXPORT_VERSION,
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
    ApiEnvelope,
    capabilities_payload,
)
from feature_intelligence.query.service import QueryService

__all__ = [
    "QUERY_ENGINE_VERSION",
    "QUERY_EXPORT_VERSION",
    "QUERY_LANGUAGE_VERSION",
    "SCHEMA_VERSION",
    "ApiEnvelope",
    "QueryEngine",
    "QueryService",
    "capabilities_payload",
    "coverage_gate",
]
