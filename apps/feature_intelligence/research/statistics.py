"""Research statistics snapshot writer (Sprint 8) — reporting only; no AI."""

from __future__ import annotations

from datetime import datetime, timezone

from feature_intelligence.research.models import (
    RESEARCH_VERSION,
    SCHEMA_VERSION,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_EMPTY,
    ResearchStatsReport,
)
from feature_intelligence.research.store import ResearchStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_research_metrics(
    store: ResearchStore,
    *,
    last_sync_at: str | None = None,
) -> ResearchStatsReport:
    """Live metrics (does not write)."""
    records = store.list_records()
    expected = len(store.list_feature_uuids())
    total = len(records)
    coverage = (100.0 * total / expected) if expected > 0 else 0.0
    empty = sum(1 for r in records if r.research_status == STATUS_EMPTY)
    active = sum(1 for r in records if r.research_status == STATUS_ACTIVE)
    archived = sum(1 for r in records if r.research_status == STATUS_ARCHIVED)

    # Preserve prior last_sync_at when caller does not supply a new sync time
    if last_sync_at is None:
        snap = store.latest_statistics()
        if snap is not None and snap.get("last_sync_at") is not None:
            last_sync_at = str(snap["last_sync_at"])

    return ResearchStatsReport(
        research_version=RESEARCH_VERSION,
        schema_version=SCHEMA_VERSION,
        total_frr=total,
        expected_features=expected,
        coverage_pct=coverage,
        status_empty=empty,
        status_active=active,
        status_archived=archived,
        last_sync_at=last_sync_at,
        from_snapshot=False,
        snapshot_created_at=None,
    )


def refresh_research_statistics(
    store: ResearchStore,
    *,
    last_sync_at: str | None = None,
    report: ResearchStatsReport | None = None,
) -> ResearchStatsReport:
    """
    Single shared stats writer (freeze §8.4.1).

    Appends a research_statistics row. Callers:
    - validate: always
    - sync: after successful sync (with last_sync_at)
    - stats: only on miss
    - import: never
    """
    live = report or compute_research_metrics(store, last_sync_at=last_sync_at)
    if last_sync_at is not None:
        live.last_sync_at = last_sync_at
    created = _utc_now()
    payload = {
        "research_version": live.research_version,
        "schema_version": live.schema_version,
        "total_frr": live.total_frr,
        "expected_features": live.expected_features,
        "coverage_pct": live.coverage_pct,
        "status_empty": live.status_empty,
        "status_active": live.status_active,
        "status_archived": live.status_archived,
        "last_sync_at": live.last_sync_at,
        "created_at": created,
    }
    stats_id = store.insert_statistics(payload)
    return ResearchStatsReport(
        research_version=live.research_version,
        schema_version=live.schema_version,
        total_frr=live.total_frr,
        expected_features=live.expected_features,
        coverage_pct=live.coverage_pct,
        status_empty=live.status_empty,
        status_active=live.status_active,
        status_archived=live.status_archived,
        last_sync_at=live.last_sync_at,
        from_snapshot=True,
        snapshot_created_at=created,
        stats_id=stats_id,
    )


def report_from_snapshot(snap: dict) -> ResearchStatsReport:
    return ResearchStatsReport(
        research_version=str(snap["research_version"]),
        schema_version=str(snap["schema_version"]),
        total_frr=int(snap["total_frr"]),
        expected_features=int(snap["expected_features"]),
        coverage_pct=float(snap["coverage_pct"]),
        status_empty=int(snap["status_empty"]),
        status_active=int(snap["status_active"]),
        status_archived=int(snap["status_archived"]),
        last_sync_at=(
            None if snap.get("last_sync_at") is None else str(snap["last_sync_at"])
        ),
        from_snapshot=True,
        snapshot_created_at=str(snap["created_at"]),
        stats_id=int(snap["stats_id"]),
    )
