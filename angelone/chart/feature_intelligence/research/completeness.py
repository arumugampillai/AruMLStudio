"""Research completeness — read-only link-gap report (Sprint 8).

Not a validity gate. Missing optional links are allowed.
"""

from __future__ import annotations

from feature_intelligence.research.models import CompletenessGap, CompletenessReport
from feature_intelligence.research.store import ResearchStore

_OPTIONAL_LINK_FIELDS = (
    "ontology_uuid",
    "transformation_uuid",
    "compiler_version",
    "grammar_version",
    "lineage_version",
)


def research_completeness(store: ResearchStore) -> CompletenessReport:
    """Report FRRs lacking optional cross-registry / version links."""
    records = store.list_records()
    gaps: list[CompletenessGap] = []
    for rec in records:
        missing: list[str] = []
        for field in _OPTIONAL_LINK_FIELDS:
            if getattr(rec, field) is None:
                missing.append(field)
        if missing:
            gaps.append(
                CompletenessGap(
                    research_uuid=rec.research_uuid,
                    feature_uuid=rec.feature_uuid,
                    missing_fields=missing,
                )
            )
    incomplete = len(gaps)
    return CompletenessReport(
        total_frr=len(records),
        complete=len(records) - incomplete,
        incomplete=incomplete,
        gaps=gaps,
    )
