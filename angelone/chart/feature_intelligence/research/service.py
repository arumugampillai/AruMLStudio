"""Research service façade (Sprint 8) — stats + completeness + sync."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.research.completeness import research_completeness
from feature_intelligence.research.identity import derive_research_uuid
from feature_intelligence.research.models import (
    CompletenessReport,
    FeatureResearchRecord,
    ResearchStatsReport,
    ResearchSyncSummary,
)
from feature_intelligence.research.statistics import (
    compute_research_metrics,
    refresh_research_statistics,
    report_from_snapshot,
)
from feature_intelligence.research.store import ResearchStore
from feature_intelligence.research.sync import sync_research as _sync_research
from feature_intelligence.research.validation import validate_research as _validate_research


class ResearchNotFoundError(LookupError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ResearchService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.store = ResearchStore(self.db_path)

    def list_research(
        self, status: str | None = None
    ) -> list[FeatureResearchRecord]:
        return self.store.list_records(status)

    def get_research(self, research_uuid: str) -> FeatureResearchRecord:
        row = self.store.get_by_uuid(research_uuid)
        if row is None:
            raise ResearchNotFoundError(research_uuid)
        return row

    def get_research_by_feature(
        self, feature_uuid: str
    ) -> FeatureResearchRecord:
        row = self.store.get_by_feature(feature_uuid)
        if row is None:
            raise ResearchNotFoundError(feature_uuid)
        return row

    @staticmethod
    def derive_research_uuid(feature_uuid: str) -> str:
        return derive_research_uuid(feature_uuid)

    def sync_research(
        self, feature_uuid: str | None = None
    ) -> ResearchSyncSummary:
        """
        Create missing EMPTY shells; optional link fill; checksum writer;
        after success refresh stats with last_sync_at.
        """
        summary = _sync_research(
            self.store,
            feature_uuid=feature_uuid,
            refresh_checksum=True,
        )
        sync_at = _utc_now()
        refresh_research_statistics(self.store, last_sync_at=sync_at)
        return summary

    def validate_research(
        self,
        *,
        mode: str = "strict",
        strict_refs: bool = False,
        strict_coverage: bool = False,
    ) -> ValidationReport:
        """Always refreshes checksum and writes stats."""
        report = _validate_research(
            self.store,
            mode=mode,
            strict_refs=strict_refs,
            strict_coverage=strict_coverage,
        )
        stats = refresh_research_statistics(self.store)
        report.warnings.append(
            f"stats: total_frr={stats.total_frr};"
            f"expected={stats.expected_features};"
            f"coverage_pct={stats.coverage_pct:.1f}"
        )
        return report

    def research_stats(self) -> ResearchStatsReport:
        """Read latest snapshot if present; regenerate + write only if none."""
        snap = self.store.latest_statistics()
        if snap is not None:
            return report_from_snapshot(snap)
        return refresh_research_statistics(
            self.store, report=compute_research_metrics(self.store)
        )

    def research_completeness(self) -> CompletenessReport:
        """Read-only link-gap report; does not affect validity."""
        return research_completeness(self.store)
