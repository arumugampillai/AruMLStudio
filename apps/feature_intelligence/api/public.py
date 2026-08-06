"""Thin public API surface (Sprint 9) — re-exports query engine callables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feature_intelligence.core.config import load_config
from feature_intelligence.query.engine import QueryEngine
from feature_intelligence.query.models import ApiEnvelope


def _db(db_path: Path | None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return load_config().database.path


def _engine(db_path: Path | None = None) -> QueryEngine:
    return QueryEngine(_db(db_path))


def get_feature(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    canonical_name: str | None = None,
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).get_feature(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
        canonical_name=canonical_name,
    )


def search_features(
    *,
    query: str | None = None,
    match_all: bool = False,
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).search_features(query=query, match_all=match_all)


def get_research(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).get_research(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
    )


def get_lineage(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    direction: str = "both",
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).get_lineage(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
        direction=direction,
    )


def get_ontology(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    ontology_uuid: str | None = None,
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).get_ontology(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
        ontology_uuid=ontology_uuid,
    )


def inspect_feature(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    canonical_name: str | None = None,
    db_path: Path | None = None,
) -> ApiEnvelope:
    return _engine(db_path).inspect_feature(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
        canonical_name=canonical_name,
    )


def get_capabilities(*, db_path: Path | None = None) -> ApiEnvelope:
    # db_path unused (static snapshot) — accepted for API symmetry
    _ = db_path
    return _engine(None if db_path is None else db_path).get_capabilities()


def get_platform_summary(*, db_path: Path | None = None) -> ApiEnvelope:
    """Read-only dashboard counts + pack versions for Feature Explorer empty state."""
    return _engine(db_path).get_platform_summary()


def get_references(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    db_path: Path | None = None,
) -> ApiEnvelope:
    """Phase 1 stub: empty models/datasets/programs linkage lists."""
    return _engine(db_path).get_references(
        feature_uuid=feature_uuid,
        research_uuid=research_uuid,
    )


# Read-only surface — no create/update/delete helpers here.
PUBLIC_CALLABLES: tuple[str, ...] = (
    "get_feature",
    "search_features",
    "get_research",
    "get_lineage",
    "get_ontology",
    "inspect_feature",
    "get_capabilities",
    "get_platform_summary",
    "get_references",
)

__all__ = [
    "ApiEnvelope",
    "PUBLIC_CALLABLES",
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
