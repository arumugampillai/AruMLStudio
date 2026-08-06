"""Feature Research Record package — Sprint 8 (metadata shell; no AI/evidence compute)."""

from __future__ import annotations

from feature_intelligence.research.identity import derive_research_uuid
from feature_intelligence.research.import_export import export_research, import_research
from feature_intelligence.research.models import (
    RESEARCH_EXPORT_VERSION,
    RESEARCH_VERSION,
    SCHEMA_VERSION,
    CompletenessReport,
    FeatureResearchRecord,
    ResearchStatsReport,
    ResearchSyncSummary,
)
from feature_intelligence.research.service import (
    ResearchNotFoundError,
    ResearchService,
)
from feature_intelligence.research.store import ResearchStore

__all__ = [
    "RESEARCH_VERSION",
    "SCHEMA_VERSION",
    "RESEARCH_EXPORT_VERSION",
    "FeatureResearchRecord",
    "ResearchSyncSummary",
    "ResearchStatsReport",
    "CompletenessReport",
    "ResearchStore",
    "ResearchService",
    "ResearchNotFoundError",
    "derive_research_uuid",
    "export_research",
    "import_research",
]
