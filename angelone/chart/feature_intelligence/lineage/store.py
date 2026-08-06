"""LineageStore — registry + edges + relationships + stats (Sprint 7)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.lineage.identity import (
    derive_lineage_uuid,
    normalize_object_id,
)
from feature_intelligence.lineage.models import (
    GRAPH_SCHEMA_VERSION,
    LINEAGE_VERSION,
    LineageEdge,
    RelationshipRecord,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def empty_graph_checksum() -> str:
    return hashlib.sha256(b"").hexdigest()


def compute_graph_checksum(triples: list[tuple[str, str, str]]) -> str:
    """
    Canonical checksum: sort (parent, child, relationship_id) ASCII ascending;
    UTF-8 lines parent\\tchild\\trel_id\\n; SHA-256 hex.
    """
    ordered = sorted(triples, key=lambda t: (t[0], t[1], t[2]))
    payload = "".join(f"{p}\t{c}\t{r}\n" for p, c, r in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LineageStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ----------------------------------------------------------- relationships
    def list_relationships(self, *, active_only: bool = False) -> list[RelationshipRecord]:
        conn = self._connect()
        try:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM lineage_relationship_registry "
                    "WHERE active = 1 ORDER BY relationship_id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM lineage_relationship_registry "
                    "ORDER BY relationship_id ASC"
                ).fetchall()
            return [self._rel_row(r) for r in rows]
        finally:
            conn.close()

    def get_relationship(self, relationship_id: str) -> RelationshipRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM lineage_relationship_registry WHERE relationship_id = ?",
                (relationship_id,),
            ).fetchone()
            return None if row is None else self._rel_row(row)
        finally:
            conn.close()

    def relationship_id_set(self) -> set[str]:
        return {r.relationship_id for r in self.list_relationships()}

    def relationship_active_map(self) -> dict[str, bool]:
        return {r.relationship_id: r.active for r in self.list_relationships()}

    @staticmethod
    def _rel_row(row: sqlite3.Row) -> RelationshipRecord:
        return RelationshipRecord(
            relationship_id=str(row["relationship_id"]),
            canonical_name=str(row["canonical_name"]),
            display_name=str(row["display_name"]),
            description=(
                None if row["description"] is None else str(row["description"])
            ),
            lineage_version=str(row["lineage_version"]),
            active=bool(row["active"]),
            sort_order=(
                None if row["sort_order"] is None else int(row["sort_order"])
            ),
            created_at=str(row["created_at"]),
        )

    # -------------------------------------------------------------- pack meta
    def get_pack(self, lineage_version: str = LINEAGE_VERSION) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM lineage_registry WHERE lineage_version = ?",
                (lineage_version,),
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            conn.close()

    def list_pack_versions(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT lineage_version FROM lineage_registry"
            ).fetchall()
            return {str(r[0]) for r in rows}
        finally:
            conn.close()

    def update_graph_checksum(
        self,
        checksum: str,
        *,
        lineage_version: str = LINEAGE_VERSION,
    ) -> None:
        now = _utc_now()
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE lineage_registry
                SET graph_checksum = ?, updated_at = ?
                WHERE lineage_version = ?
                """,
                (checksum, now, lineage_version),
            )
            conn.commit()
        finally:
            conn.close()

    def recompute_and_store_graph_checksum(
        self, *, lineage_version: str = LINEAGE_VERSION
    ) -> str:
        """Shared checksum writer used by validate / import / derive."""
        triples = [
            (e.parent_object, e.child_object, e.relationship_id)
            for e in self.list_edges()
        ]
        checksum = compute_graph_checksum(triples)
        self.update_graph_checksum(checksum, lineage_version=lineage_version)
        return checksum

    # ------------------------------------------------------------------ edges
    def list_edges(
        self, relationship_id: str | None = None
    ) -> list[LineageEdge]:
        conn = self._connect()
        try:
            if relationship_id:
                rows = conn.execute(
                    """
                    SELECT * FROM lineage_edges
                    WHERE relationship_id = ?
                    ORDER BY parent_object ASC, child_object ASC, relationship_id ASC
                    """,
                    (relationship_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM lineage_edges
                    ORDER BY parent_object ASC, child_object ASC, relationship_id ASC
                    """
                ).fetchall()
            return [self._edge_row(r) for r in rows]
        finally:
            conn.close()

    def get_edge(self, lineage_uuid: str) -> LineageEdge | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM lineage_edges WHERE lineage_uuid = ?",
                (lineage_uuid,),
            ).fetchone()
            return None if row is None else self._edge_row(row)
        finally:
            conn.close()

    def get_edge_by_triple(
        self, parent_object: str, child_object: str, relationship_id: str
    ) -> LineageEdge | None:
        parent = normalize_object_id(parent_object)
        child = normalize_object_id(child_object)
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM lineage_edges
                WHERE parent_object = ? AND child_object = ? AND relationship_id = ?
                """,
                (parent, child, relationship_id),
            ).fetchone()
            return None if row is None else self._edge_row(row)
        finally:
            conn.close()

    def count_edges(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM lineage_edges").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def edge_pairs(self) -> list[tuple[str, str]]:
        return [(e.parent_object, e.child_object) for e in self.list_edges()]

    def upsert_edge(self, edge: LineageEdge) -> LineageEdge:
        """Idempotent upsert by lineage_uuid / triple."""
        parent = normalize_object_id(edge.parent_object)
        child = normalize_object_id(edge.child_object)
        uuid = edge.lineage_uuid or derive_lineage_uuid(
            parent, child, edge.relationship_id
        )
        now = _utc_now()
        existing = self.get_edge(uuid)
        if existing is None:
            existing = self.get_edge_by_triple(parent, child, edge.relationship_id)
        created = existing.created_at if existing else (edge.created_at or now)
        # Preserve DERIVE source if re-derived; import may set IMPORT
        edge_source = edge.edge_source
        if existing is not None and edge_source is None:
            edge_source = existing.edge_source
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO lineage_edges (
                    lineage_uuid, parent_object, child_object, relationship_id,
                    edge_source, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(lineage_uuid) DO UPDATE SET
                    parent_object = excluded.parent_object,
                    child_object = excluded.child_object,
                    relationship_id = excluded.relationship_id,
                    edge_source = COALESCE(excluded.edge_source, lineage_edges.edge_source),
                    updated_at = excluded.updated_at
                """,
                (
                    uuid,
                    parent,
                    child,
                    edge.relationship_id,
                    edge_source,
                    created,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        out = self.get_edge(uuid)
        assert out is not None
        return out

    @staticmethod
    def _edge_row(row: sqlite3.Row) -> LineageEdge:
        return LineageEdge(
            lineage_uuid=str(row["lineage_uuid"]),
            parent_object=str(row["parent_object"]),
            child_object=str(row["child_object"]),
            relationship_id=str(row["relationship_id"]),
            edge_source=(
                None if row["edge_source"] is None else str(row["edge_source"])
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    # ------------------------------------------------------------- registries
    def object_exists_in_registry(self, object_id: str) -> bool:
        from feature_intelligence.lineage.identity import infer_object_type

        ot = infer_object_type(object_id)
        if ot is None:
            return False
        oid = normalize_object_id(object_id)
        queries = {
            "PRIMITIVE": (
                "SELECT 1 FROM primitive_registry WHERE primitive_id = ?"
            ),
            "OPERATOR": (
                "SELECT 1 FROM operator_registry WHERE operator_id = ?"
            ),
            "TRANSFORMATION": (
                "SELECT 1 FROM transformation_registry WHERE transformation_uuid = ?"
            ),
            "FEATURE": (
                "SELECT 1 FROM feature_registry WHERE feature_uuid = ?"
            ),
        }
        sql = queries[ot]
        conn = self._connect()
        try:
            try:
                row = conn.execute(sql, (oid,)).fetchone()
            except sqlite3.OperationalError:
                return False
            return row is not None
        finally:
            conn.close()

    def list_feature_ast_links(self) -> list[tuple[str, str]]:
        """Return (feature_uuid, transformation_uuid) from feature_ast."""
        conn = self._connect()
        try:
            try:
                rows = conn.execute(
                    "SELECT feature_uuid, transformation_uuid FROM feature_ast "
                    "ORDER BY feature_uuid ASC"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [
                (normalize_object_id(str(r[0])), normalize_object_id(str(r[1])))
                for r in rows
            ]
        finally:
            conn.close()

    def list_transformation_uuids_with_ast(self) -> list[str]:
        conn = self._connect()
        try:
            try:
                rows = conn.execute(
                    "SELECT DISTINCT transformation_uuid FROM ast_nodes "
                    "ORDER BY transformation_uuid ASC"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [normalize_object_id(str(r[0])) for r in rows]
        finally:
            conn.close()

    def list_ast_node_refs(
        self, transformation_uuid: str | None = None
    ) -> list[dict[str, Any]]:
        """Read-only AST node refs for derive (relationships only — no AST copy)."""
        conn = self._connect()
        try:
            try:
                if transformation_uuid:
                    rows = conn.execute(
                        """
                        SELECT transformation_uuid, node_type, operator_id,
                               primitive_id, feature_uuid
                        FROM ast_nodes
                        WHERE transformation_uuid = ?
                        """,
                        (normalize_object_id(transformation_uuid),),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT transformation_uuid, node_type, operator_id,
                               primitive_id, feature_uuid
                        FROM ast_nodes
                        """
                    ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ----------------------------------------------------------------- stats
    def latest_statistics(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM lineage_statistics ORDER BY stats_id DESC LIMIT 1"
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            conn.close()

    def count_statistics(self) -> int:
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM lineage_statistics"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def list_relationship_statistics(self, stats_id: int) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM relationship_statistics
                WHERE stats_id = ?
                ORDER BY relationship_id ASC
                """,
                (stats_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def insert_statistics(
        self,
        lineage_payload: dict[str, Any],
        rel_counts: list[tuple[str, int]],
    ) -> int:
        """
        Append lineage_statistics + relationship_statistics (same snapshot).
        Single writer path — call via service helper only.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO lineage_statistics (
                    lineage_version, graph_schema_version, edges, nodes,
                    root_primitives, orphans, depth, components, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    lineage_payload["lineage_version"],
                    lineage_payload["graph_schema_version"],
                    lineage_payload["edges"],
                    lineage_payload["nodes"],
                    lineage_payload["root_primitives"],
                    lineage_payload["orphans"],
                    lineage_payload["depth"],
                    lineage_payload["components"],
                    lineage_payload["created_at"],
                ),
            )
            stats_id = int(cur.lastrowid)
            for rid, count in rel_counts:
                conn.execute(
                    """
                    INSERT INTO relationship_statistics (
                        stats_id, lineage_version, relationship_id,
                        edge_count, created_at
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        stats_id,
                        lineage_payload["lineage_version"],
                        rid,
                        count,
                        lineage_payload["created_at"],
                    ),
                )
            conn.commit()
            return stats_id
        finally:
            conn.close()
