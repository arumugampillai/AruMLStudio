"""Lineage service façade (Sprint 7) — stats + single writers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.lineage.derive import derive_lineage as _derive_lineage
from feature_intelligence.lineage.graph import (
    ancestors_of,
    children_of,
    descendants_of,
    max_dag_depth,
    parents_of,
    weakly_connected_components,
)
from feature_intelligence.lineage.identity import derive_lineage_uuid
from feature_intelligence.lineage.models import (
    GRAPH_SCHEMA_VERSION,
    LINEAGE_VERSION,
    DeriveResult,
    LineageEdge,
    LineageStatsReport,
    RelStatsRow,
)
from feature_intelligence.lineage.relationships import (
    EXPECTED_RELATIONSHIP_SEED_HASH,
    SEED_RELATIONSHIPS,
)
from feature_intelligence.lineage.store import LineageStore
from feature_intelligence.lineage.validation import validate_lineage as _validate_lineage
from feature_intelligence.registry.models import ValidationReport


class LineageNotFoundError(LookupError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class LineageService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.store = LineageStore(self.db_path)

    # ---------------------------------------------------------------- edges
    def list_edges(
        self, relationship_id: str | None = None
    ) -> list[LineageEdge]:
        return self.store.list_edges(relationship_id)

    def get_edge(self, lineage_uuid: str) -> LineageEdge:
        row = self.store.get_edge(lineage_uuid)
        if row is None:
            raise LineageNotFoundError(lineage_uuid)
        return row

    def get_edge_by_triple(
        self, parent_object: str, child_object: str, relationship_id: str
    ) -> LineageEdge:
        row = self.store.get_edge_by_triple(
            parent_object, child_object, relationship_id
        )
        if row is None:
            raise LineageNotFoundError(
                f"{parent_object}|{child_object}|{relationship_id}"
            )
        return row

    @staticmethod
    def derive_lineage_uuid(
        parent_object: str, child_object: str, relationship_id: str
    ) -> str:
        return derive_lineage_uuid(parent_object, child_object, relationship_id)

    # ------------------------------------------------------------ navigation
    def parents(self, object_id: str) -> list[str]:
        return parents_of(object_id, self.store.edge_pairs())

    def children(self, object_id: str) -> list[str]:
        return children_of(object_id, self.store.edge_pairs())

    def ancestors(self, object_id: str) -> list[str]:
        return ancestors_of(object_id, self.store.edge_pairs())

    def descendants(self, object_id: str) -> list[str]:
        return descendants_of(object_id, self.store.edge_pairs())

    # --------------------------------------------------------------- derive
    def derive_lineage(
        self,
        *,
        transformation_uuid: str | None = None,
        feature_uuid: str | None = None,
        include_closure: bool = True,
        strict_refs: bool = False,
    ) -> DeriveResult:
        """Sprint 5 extractor; idempotent; does not write stats."""
        return _derive_lineage(
            self.store,
            transformation_uuid=transformation_uuid,
            feature_uuid=feature_uuid,
            include_closure=include_closure,
            strict_refs=strict_refs,
            refresh_checksum=True,
        )

    # --------------------------------------------------------------- stats
    def compute_lineage_metrics(self) -> LineageStatsReport:
        """Compute live metrics (does not write)."""
        edges = self.store.list_edges()
        pairs = [(e.parent_object, e.child_object) for e in edges]
        nodes: set[str] = set()
        for e in edges:
            nodes.add(e.parent_object)
            nodes.add(e.child_object)

        inbound = {c for _, c in pairs}
        root_primitives = {
            n for n in nodes if n.startswith("PR_") and n not in inbound
        }

        orphan_count = 0
        for feat, _tr in self.store.list_feature_ast_links():
            ancs = ancestors_of(feat, pairs)
            if not any(a.startswith("PR_") for a in ancs):
                orphan_count += 1

        rel_count_map = {r.relationship_id: 0 for r in SEED_RELATIONSHIPS}
        for e in edges:
            if e.relationship_id in rel_count_map:
                rel_count_map[e.relationship_id] += 1
            else:
                rel_count_map[e.relationship_id] = (
                    rel_count_map.get(e.relationship_id, 0) + 1
                )

        # Exactly one row per frozen REL_*
        rel_rows = [
            RelStatsRow(relationship_id=r.relationship_id, edge_count=rel_count_map.get(r.relationship_id, 0))
            for r in sorted(SEED_RELATIONSHIPS, key=lambda x: x.relationship_id)
        ]

        return LineageStatsReport(
            lineage_version=LINEAGE_VERSION,
            graph_schema_version=GRAPH_SCHEMA_VERSION,
            edges=len(edges),
            nodes=len(nodes),
            root_primitives=len(root_primitives),
            orphans=orphan_count,
            depth=max_dag_depth(pairs),
            components=weakly_connected_components(pairs),
            relationship_counts=rel_rows,
            from_snapshot=False,
            snapshot_created_at=None,
        )

    def write_lineage_statistics(
        self, report: LineageStatsReport | None = None
    ) -> LineageStatsReport:
        """
        Single shared stats writer (freeze §7.4.1).

        Appends lineage_statistics + relationship_statistics.
        Only validate (always) and stats (on miss) should call this —
        never import or derive.
        """
        live = report or self.compute_lineage_metrics()
        created = _utc_now()
        payload = {
            "lineage_version": live.lineage_version,
            "graph_schema_version": live.graph_schema_version,
            "edges": live.edges,
            "nodes": live.nodes,
            "root_primitives": live.root_primitives,
            "orphans": live.orphans,
            "depth": live.depth,
            "components": live.components,
            "created_at": created,
        }
        rel_counts = [
            (r.relationship_id, r.edge_count) for r in live.relationship_counts
        ]
        stats_id = self.store.insert_statistics(payload, rel_counts)
        return LineageStatsReport(
            lineage_version=live.lineage_version,
            graph_schema_version=live.graph_schema_version,
            edges=live.edges,
            nodes=live.nodes,
            root_primitives=live.root_primitives,
            orphans=live.orphans,
            depth=live.depth,
            components=live.components,
            relationship_counts=live.relationship_counts,
            from_snapshot=True,
            snapshot_created_at=created,
            stats_id=stats_id,
        )

    def _report_from_snapshot(self, snap: dict) -> LineageStatsReport:
        stats_id = int(snap["stats_id"])
        rel_rows_raw = self.store.list_relationship_statistics(stats_id)
        # Ensure all frozen REL_* present
        by_id = {
            str(r["relationship_id"]): int(r["edge_count"]) for r in rel_rows_raw
        }
        rel_rows = [
            RelStatsRow(
                relationship_id=r.relationship_id,
                edge_count=by_id.get(r.relationship_id, 0),
            )
            for r in sorted(SEED_RELATIONSHIPS, key=lambda x: x.relationship_id)
        ]
        return LineageStatsReport(
            lineage_version=str(snap["lineage_version"]),
            graph_schema_version=str(snap["graph_schema_version"]),
            edges=int(snap["edges"]),
            nodes=int(snap["nodes"]),
            root_primitives=int(snap["root_primitives"]),
            orphans=int(snap["orphans"]),
            depth=int(snap["depth"]),
            components=int(snap["components"]),
            relationship_counts=rel_rows,
            from_snapshot=True,
            snapshot_created_at=str(snap["created_at"]),
            stats_id=stats_id,
        )

    def lineage_stats(self) -> LineageStatsReport:
        """Read latest snapshot if present; regenerate + write only if none."""
        snap = self.store.latest_statistics()
        if snap is not None:
            return self._report_from_snapshot(snap)
        return self.write_lineage_statistics()

    def validate_lineage(
        self,
        *,
        mode: str = "strict",
        strict_refs: bool = False,
    ) -> ValidationReport:
        """Always refreshes checksum (inside validate) and writes stats."""
        report = _validate_lineage(
            self.store, mode=mode, strict_refs=strict_refs
        )
        stats = self.compute_lineage_metrics()
        written = self.write_lineage_statistics(stats)
        report.warnings.append(
            f"stats: edges={written.edges};nodes={written.nodes};"
            f"roots={written.root_primitives};orphans={written.orphans};"
            f"depth={written.depth};components={written.components}"
        )
        if not report.seed_hash:
            report.seed_hash = EXPECTED_RELATIONSHIP_SEED_HASH
        if not report.expected_seed_hash:
            report.expected_seed_hash = EXPECTED_RELATIONSHIP_SEED_HASH
        return report
