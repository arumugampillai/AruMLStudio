"""Lineage data models (Sprint 7) — relationships only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LINEAGE_VERSION = "1.0.0"
GRAPH_SCHEMA_VERSION = "1.0"
RELATIONSHIP_PACK_VERSION = "1.0.0"
GRAPH_EXPORT_VERSION = "1.0"

EDGE_SOURCES: frozenset[str] = frozenset({"DERIVE", "IMPORT", "MIGRATION"})

REL_USES = "REL_USES"
REL_GENERATED_BY = "REL_GENERATED_BY"
REL_DEPENDS_ON = "REL_DEPENDS_ON"
REL_DERIVED_FROM = "REL_DERIVED_FROM"
REL_INPUT_TO = "REL_INPUT_TO"

FROZEN_RELATIONSHIP_IDS: frozenset[str] = frozenset(
    {
        REL_USES,
        REL_GENERATED_BY,
        REL_DEPENDS_ON,
        REL_DERIVED_FROM,
        REL_INPUT_TO,
    }
)


@dataclass(frozen=True)
class RelationshipRecord:
    relationship_id: str
    canonical_name: str
    display_name: str
    description: str | None
    lineage_version: str
    active: bool
    sort_order: int | None = None
    created_at: str = ""


@dataclass
class LineageEdge:
    lineage_uuid: str
    parent_object: str
    child_object: str
    relationship_id: str
    edge_source: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self, *, include_timestamps: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_timestamps:
            d.pop("created_at", None)
            d.pop("updated_at", None)
        return d

    def triple(self) -> tuple[str, str, str]:
        return (self.parent_object, self.child_object, self.relationship_id)


@dataclass
class RelStatsRow:
    relationship_id: str
    edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LineageStatsReport:
    lineage_version: str
    graph_schema_version: str
    edges: int
    nodes: int
    root_primitives: int
    orphans: int
    depth: int
    components: int
    relationship_counts: list[RelStatsRow]
    from_snapshot: bool
    snapshot_created_at: str | None = None
    stats_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_version": self.lineage_version,
            "graph_schema_version": self.graph_schema_version,
            "edges": self.edges,
            "nodes": self.nodes,
            "root_primitives": self.root_primitives,
            "orphans": self.orphans,
            "depth": self.depth,
            "components": self.components,
            "relationship_counts": [r.to_dict() for r in self.relationship_counts],
            "from_snapshot": self.from_snapshot,
            "snapshot_created_at": self.snapshot_created_at,
            "stats_id": self.stats_id,
        }


@dataclass
class DeriveResult:
    upserted: int
    skipped: int
    warnings: list[str]
    edges: list[LineageEdge]
