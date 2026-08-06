"""Persistence for transformation / compilation / AST / statistics (Sprint 5)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from feature_intelligence.ast.nodes import AstNode, flatten_nodes
from feature_intelligence.ast.serialize import ast_document, dumps_canonical
from feature_intelligence.compiler.models import CompileMetrics, CompilerManifest


class CompilerStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def tables_exist(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='transformation_registry'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_transformation(self, transformation_uuid: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM transformation_registry WHERE transformation_uuid = ?",
                (transformation_uuid,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def count_transformations(self) -> int:
        """Read-only count of transformation_registry rows."""
        if not self.tables_exist():
            return 0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM transformation_registry"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def get_by_expression_hash(self, expr_hash: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM transformation_registry WHERE expression_hash = ?",
                (expr_hash,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def has_ast_nodes(self, transformation_uuid: str, ast_hash: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM ast_nodes WHERE transformation_uuid = ? AND ast_hash = ? LIMIT 1",
                (transformation_uuid, ast_hash),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def insert_transformation(
        self,
        *,
        transformation_uuid: str,
        expression_hash: str,
        canonical_text: str,
        source_text: str | None,
    ) -> bool:
        """Insert if novel. Returns True if inserted, False if already present."""
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO transformation_registry(
                    transformation_uuid, expression_hash, source_text, canonical_text
                ) VALUES (?,?,?,?)
                """,
                (transformation_uuid, expression_hash, source_text, canonical_text),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def insert_compilation(
        self,
        *,
        compilation_uuid: str,
        transformation_uuid: str,
        ast_schema_version: str,
        grammar_version: str,
        compiler_version: str,
        operator_pack_version: str,
        ast_hash: str,
        root_node_id: str,
        cache_hit: bool,
        status: str,
        diagnostics_json: str | None = None,
        warnings_json: str | None = None,
        metrics: CompileMetrics | None = None,
    ) -> None:
        metrics_json = None if metrics is None else dumps_canonical(metrics.to_dict())
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO compilation_registry(
                    compilation_uuid, transformation_uuid, ast_schema_version,
                    grammar_version, compiler_version, operator_pack_version,
                    ast_hash, root_node_id, cache_hit, diagnostics_json,
                    warnings_json, metrics_json, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    compilation_uuid,
                    transformation_uuid,
                    ast_schema_version,
                    grammar_version,
                    compiler_version,
                    operator_pack_version,
                    ast_hash,
                    root_node_id,
                    1 if cache_hit else 0,
                    diagnostics_json,
                    warnings_json,
                    metrics_json,
                    status,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def persist_ast_nodes(
        self,
        *,
        transformation_uuid: str,
        ast_hash: str,
        root: AstNode,
    ) -> None:
        """Write normalized nodes once; no-op if already present for this hash."""
        if self.has_ast_nodes(transformation_uuid, ast_hash):
            return
        nodes = flatten_nodes(root)
        conn = self._connect()
        try:
            for node in nodes:
                ordinal = int(node.node_id[1:]) if node.node_id.startswith("N") else 0
                conn.execute(
                    """
                    INSERT OR IGNORE INTO ast_nodes(
                        transformation_uuid, ast_hash, node_id, node_type,
                        parent_node_id, param_name, operator_id, primitive_id,
                        feature_uuid, literal_json, child_node_ids_json,
                        subtree_hash, stable_node_hash, ordinal
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        transformation_uuid,
                        ast_hash,
                        node.node_id,
                        node.node_type,
                        node.parent,
                        node.param_name,
                        node.operator_id,
                        node.primitive_id,
                        node.feature_uuid,
                        None
                        if node.literal is None
                        else dumps_canonical(node.literal.to_dict()),
                        dumps_canonical([c.node_id for c in node.children]),
                        node.subtree_hash,
                        None,  # reserved
                        ordinal,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def upsert_feature_ast(
        self,
        *,
        feature_uuid: str,
        transformation_uuid: str,
        compilation_uuid: str,
        ast_schema_version: str,
        root: AstNode,
        ast_hash: str,
        expression_hash: str,
        grammar_version: str,
        compiler_version: str,
        operator_pack_version: str,
        rewrite_body: bool = True,
    ) -> None:
        doc = ast_document(
            root,
            transformation_uuid=transformation_uuid,
            expression_hash=expression_hash,
            ast_hash=ast_hash,
            grammar_version=grammar_version,
            compiler_version=compiler_version,
            operator_pack_version=operator_pack_version,
            ast_schema_version=ast_schema_version,
        )
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT feature_uuid, ast_fingerprint FROM feature_ast WHERE feature_uuid = ?",
                (feature_uuid,),
            ).fetchone()
            if existing is not None and not rewrite_body:
                # Cache-hit verify: update compilation_uuid pointer only
                conn.execute(
                    "UPDATE feature_ast SET compilation_uuid = ? WHERE feature_uuid = ?",
                    (compilation_uuid, feature_uuid),
                )
            elif existing is not None:
                conn.execute(
                    """
                    UPDATE feature_ast SET
                        transformation_uuid = ?, compilation_uuid = ?,
                        ast_schema_version = ?, ast_json = ?,
                        ast_fingerprint = ?, subtree_hash = ?
                    WHERE feature_uuid = ?
                    """,
                    (
                        transformation_uuid,
                        compilation_uuid,
                        ast_schema_version,
                        dumps_canonical(doc),
                        ast_hash,
                        root.subtree_hash,
                        feature_uuid,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO feature_ast(
                        feature_uuid, transformation_uuid, compilation_uuid,
                        ast_schema_version, ast_json, ast_fingerprint, subtree_hash
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        feature_uuid,
                        transformation_uuid,
                        compilation_uuid,
                        ast_schema_version,
                        dumps_canonical(doc),
                        ast_hash,
                        root.subtree_hash,
                    ),
                )
            # Keep feature_registry.transformation_uuid consistent when column exists
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(feature_registry)").fetchall()
            }
            if "transformation_uuid" in cols:
                conn.execute(
                    "UPDATE feature_registry SET transformation_uuid = ? WHERE feature_uuid = ?",
                    (transformation_uuid, feature_uuid),
                )
            conn.commit()
        finally:
            conn.close()

    def bump_statistics(
        self,
        *,
        cache_hit: bool,
        total_ms: float | None = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO compiler_statistics(id, total_compiles, cache_hits, cache_misses)
                VALUES (1, 0, 0, 0)
                """
            )
            if cache_hit:
                conn.execute(
                    """
                    UPDATE compiler_statistics SET
                        total_compiles = total_compiles + 1,
                        cache_hits = cache_hits + 1,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = 1
                    """
                )
            else:
                conn.execute(
                    """
                    UPDATE compiler_statistics SET
                        total_compiles = total_compiles + 1,
                        cache_misses = cache_misses + 1,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = 1
                    """
                )
            if total_ms is not None:
                row = conn.execute(
                    "SELECT total_compiles, average_compile_ms FROM compiler_statistics WHERE id = 1"
                ).fetchone()
                n = int(row["total_compiles"]) if row else 1
                prev = row["average_compile_ms"] if row else None
                if prev is None:
                    avg = float(total_ms)
                else:
                    # running average over all compiles that sampled metrics
                    avg = ((float(prev) * (n - 1)) + float(total_ms)) / max(n, 1)
                conn.execute(
                    "UPDATE compiler_statistics SET average_compile_ms = ? WHERE id = 1",
                    (avg,),
                )
            conn.commit()
        finally:
            conn.close()

    def compilation_has_compiled_json_column(self) -> bool:
        """Guard for dual-blob regression tests."""
        conn = self._connect()
        try:
            cols = {
                r[1]
                for r in conn.execute(
                    "PRAGMA table_info(compilation_registry)"
                ).fetchall()
            }
            return "compiled_json" in cols
        finally:
            conn.close()

    def get_compilation(self, compilation_uuid: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM compilation_registry WHERE compilation_uuid = ?",
                (compilation_uuid,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def load_ast_document_for_transformation(
        self, transformation_uuid: str, ast_hash: str
    ) -> dict[str, Any] | None:
        """Prefer feature_ast.ast_json; else reconstruct from ast_nodes."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT ast_json FROM feature_ast
                WHERE transformation_uuid = ? AND ast_fingerprint = ?
                LIMIT 1
                """,
                (transformation_uuid, ast_hash),
            ).fetchone()
            if row:
                return json.loads(row["ast_json"])
            nodes = conn.execute(
                """
                SELECT * FROM ast_nodes
                WHERE transformation_uuid = ? AND ast_hash = ?
                ORDER BY ordinal ASC
                """,
                (transformation_uuid, ast_hash),
            ).fetchall()
            if not nodes:
                return None
            flat = []
            for n in nodes:
                item: dict[str, Any] = {
                    "node_id": n["node_id"],
                    "node_type": n["node_type"],
                    "parent": n["parent_node_id"],
                    "subtree_hash": n["subtree_hash"],
                    "stable_node_hash": n["stable_node_hash"],
                    "child_node_ids": json.loads(n["child_node_ids_json"]),
                }
                if n["param_name"] is not None:
                    item["param_name"] = n["param_name"]
                if n["operator_id"] is not None:
                    item["operator_id"] = n["operator_id"]
                if n["primitive_id"] is not None:
                    item["primitive_id"] = n["primitive_id"]
                if n["feature_uuid"] is not None:
                    item["feature_uuid"] = n["feature_uuid"]
                if n["literal_json"] is not None:
                    item["literal"] = json.loads(n["literal_json"])
                flat.append(item)
            tr = conn.execute(
                "SELECT expression_hash, canonical_text FROM transformation_registry "
                "WHERE transformation_uuid = ?",
                (transformation_uuid,),
            ).fetchone()
            return {
                "ast_schema_version": "1.0",
                "transformation_uuid": transformation_uuid,
                "expression_hash": tr["expression_hash"] if tr else "",
                "ast_hash": ast_hash,
                "root_node_id": flat[0]["node_id"],
                "nodes": flat,
            }
        finally:
            conn.close()
