"""Feature Registry models (Sprint 2)."""

from __future__ import annotations

from dataclasses import dataclass, field


RESEARCH_STATES = frozenset(
    {"EXPERIMENTAL", "CANDIDATE", "VALIDATED", "DEPRECATED"}
)
GAP_POLICIES = frozenset({"CONTINUOUS", "GAP_AWARE", "RESET_ON_GAP", "OTHER"})
MEMORY_MODELS = frozenset({"TICK", "SESSION", "SLIDING_WINDOW", "DAY", "OTHER"})

CANONICAL_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


@dataclass(frozen=True)
class FeatureRecord:
    feature_uuid: str
    canonical_name: str
    display_name: str
    definition_version: str
    implementation_version: str
    definition_hash: str
    created_by: str
    controller_owner: str
    warmup_periods: int
    gap_policy: str
    memory_model: str
    research_state: str
    primitive_ids: tuple[str, ...]
    feature_version: str | None = None
    transformation_uuid: str | None = None
    legacy_feature_id: str | None = None
    description: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SyncReport:
    processed: int = 0
    registered: int = 0
    updated: int = 0
    skipped: int = 0
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "processed": self.processed,
            "registered": self.registered,
            "updated": self.updated,
            "skipped": self.skipped,
            "conflicts": list(self.conflicts),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SyncFailure:
    """One source feature that could not be imported."""

    name: str
    reason: str
    legacy_feature_id: str | None = None
    feature_uuid: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reason": self.reason,
            "legacy_feature_id": self.legacy_feature_id,
            "feature_uuid": self.feature_uuid,
        }


@dataclass
class SyncSummary:
    """User-facing Feature Registry Synchronizer result."""

    total_source: int = 0
    already_registered: int = 0
    newly_imported: int = 0
    failed: int = 0
    failures: list[SyncFailure] = field(default_factory=list)
    updated: int = 0
    research_created: int = 0
    research_updated: int = 0
    duration_ms: int = 0
    timestamp: str = ""
    mode: str = "lenient"

    def to_dict(self) -> dict[str, object]:
        return {
            "total_source": self.total_source,
            "already_registered": self.already_registered,
            "newly_imported": self.newly_imported,
            "failed": self.failed,
            "failures": [f.to_dict() for f in self.failures],
            "updated": self.updated,
            "research_created": self.research_created,
            "research_updated": self.research_updated,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "mode": self.mode,
        }

    def to_sync_report(self) -> SyncReport:
        """Backward-compatible adapter report shape."""
        errors = [
            f"{f.reason}:{f.name}" if f.name else f.reason for f in self.failures
        ]
        return SyncReport(
            processed=self.total_source,
            registered=self.newly_imported,
            updated=self.updated,
            skipped=self.already_registered + self.failed,
            conflicts=[
                f"{f.reason}:{f.name}"
                for f in self.failures
                if f.reason.startswith("CONFLICT_")
            ],
            errors=errors,
            warnings=[],
            duration_ms=self.duration_ms,
            timestamp=self.timestamp,
        )
