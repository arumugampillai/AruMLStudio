"""Compiler Manifest emit helpers (Sprint 5)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from feature_intelligence.ast.nodes import AST_SCHEMA_VERSION, AstNode, flatten_nodes
from feature_intelligence.compiler.models import COMPILER_VERSION, CompilerManifest
from feature_intelligence.grammar.pack import (
    FORMATTER_VERSION,
    GRAMMAR_PACK_VERSION,
    GRAMMAR_VERSION,
)
from feature_intelligence.operators.catalog import OPERATOR_PACK_VERSION


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def collect_operator_ids(root: AstNode) -> list[str]:
    ids: list[str] = []
    for node in flatten_nodes(root):
        if node.node_type == "operator" and node.operator_id:
            ids.append(node.operator_id)
    # stable unique order: first appearance
    seen: set[str] = set()
    out: list[str] = []
    for oid in ids:
        if oid not in seen:
            seen.add(oid)
            out.append(oid)
    return out


def emit_manifest(
    *,
    transformation_uuid: str,
    compilation_uuid: str,
    expression_hash: str,
    ast_hash: str,
    root: AstNode,
    canonical_text: str,
    operator_versions: dict[str, str] | None = None,
    cache_hit: bool = False,
    definition_version: str | None = None,
    implementation_version: str | None = None,
    created_at: str | None = None,
    ast_schema_version: str = AST_SCHEMA_VERSION,
    grammar_version: str = GRAMMAR_VERSION,
    grammar_pack_version: str = GRAMMAR_PACK_VERSION,
    formatter_version: str = FORMATTER_VERSION,
    compiler_version: str = COMPILER_VERSION,
    operator_pack_version: str = OPERATOR_PACK_VERSION,
) -> CompilerManifest:
    return CompilerManifest(
        transformation_uuid=transformation_uuid,
        compilation_uuid=compilation_uuid,
        expression_hash=expression_hash,
        ast_schema_version=ast_schema_version,
        grammar_version=grammar_version,
        grammar_pack_version=grammar_pack_version,
        formatter_version=formatter_version,
        compiler_version=compiler_version,
        operator_pack_version=operator_pack_version,
        definition_version=definition_version,
        implementation_version=implementation_version,
        ast_hash=ast_hash,
        root_subtree_hash=root.subtree_hash,
        operator_versions=dict(operator_versions or {}),
        canonical_text=canonical_text,
        cache_hit=cache_hit,
        created_at=created_at or _utc_now(),
    )


def manifests_equal_ignoring_ephemeral(
    a: CompilerManifest | dict[str, Any],
    b: CompilerManifest | dict[str, Any],
) -> bool:
    """Determinism equality: ignore created_at, compilation_uuid, cache_hit."""

    def _body(m: CompilerManifest | dict[str, Any]) -> dict[str, Any]:
        if isinstance(m, CompilerManifest):
            d = m.to_dict()["transformation_manifest"]
        elif "transformation_manifest" in m:
            d = dict(m["transformation_manifest"])
        else:
            d = dict(m)
        for k in ("created_at", "compilation_uuid", "cache_hit"):
            d.pop(k, None)
        return d

    return _body(a) == _body(b)
