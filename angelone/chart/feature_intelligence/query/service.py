"""CLI-facing query service façade (Sprint 9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feature_intelligence.query.engine import QueryEngine, coverage_gate
from feature_intelligence.query.export import export_payload
from feature_intelligence.query.models import ApiEnvelope


class QueryService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.engine = QueryEngine(self.db_path)

    def search(
        self, query: str | None = None, *, match_all: bool = False
    ) -> ApiEnvelope:
        return self.engine.search_features(query=query, match_all=match_all)

    def inspect(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        canonical_name: str | None = None,
    ) -> ApiEnvelope:
        return self.engine.inspect_feature(
            feature_uuid=feature_uuid,
            research_uuid=research_uuid,
            canonical_name=canonical_name,
        )

    def validate(self, query: str | None) -> ApiEnvelope:
        return self.engine.validate(query)

    def capabilities(self) -> ApiEnvelope:
        return self.engine.get_capabilities()

    def platform_summary(self) -> ApiEnvelope:
        return self.engine.get_platform_summary()

    def references(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
    ) -> ApiEnvelope:
        return self.engine.get_references(
            feature_uuid=feature_uuid,
            research_uuid=research_uuid,
        )

    def coverage(self) -> dict[str, Any]:
        return coverage_gate(self.db_path)

    def export_search(
        self,
        out: Path,
        *,
        query: str | None = None,
        match_all: bool = False,
        fmt: str = "json",
    ) -> Path:
        env = self.search(query, match_all=match_all)
        if not env.ok:
            raise ValueError(env.error or {"code": "EXPORT_FAILED", "message": "search failed"})
        return export_payload(env.data, out, fmt=fmt, kind="search")

    def export_inspect(
        self,
        out: Path,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        canonical_name: str | None = None,
        fmt: str = "json",
    ) -> Path:
        env = self.inspect(
            feature_uuid=feature_uuid,
            research_uuid=research_uuid,
            canonical_name=canonical_name,
        )
        if not env.ok:
            raise ValueError(env.error or {"code": "EXPORT_FAILED", "message": "inspect failed"})
        return export_payload(env.data, out, fmt=fmt, kind="inspect")
