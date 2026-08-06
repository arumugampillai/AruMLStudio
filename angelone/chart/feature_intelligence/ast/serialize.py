"""Canonical JSON serialization for AST hashing and storage (Sprint 5)."""

from __future__ import annotations

import json
from typing import Any

from feature_intelligence.ast.nodes import (
    AST_SCHEMA_VERSION,
    AstNode,
    LiteralPayload,
    flatten_nodes,
)


def dumps_canonical(obj: Any) -> str:
    """Compact JSON with sorted keys (hash / storage canonical form)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_subtree_payload(node: AstNode) -> dict[str, Any]:
    """Build the §9.1 payload object for ``subtree_hash`` (excludes bookkeeping)."""
    if node.node_type == "operator":
        param_names = [c.param_name or "" for c in node.children]
        payload: dict[str, Any] = {
            "operator_id": node.operator_id,
            "param_names": param_names,
        }
    elif node.node_type == "primitive":
        payload = {"primitive_id": node.primitive_id}
    elif node.node_type == "feature":
        payload = {"feature_uuid": node.feature_uuid}
    elif node.node_type == "literal":
        assert node.literal is not None
        payload = {"literal": node.literal.to_dict()}
    elif node.node_type == "list":
        payload = {}
    else:
        raise ValueError(f"unknown node_type: {node.node_type}")

    return {
        "child_subtree_hashes": [c.subtree_hash for c in node.children],
        "node_type": node.node_type,
        "payload": payload,
    }


def node_to_flat_dict(node: AstNode) -> dict[str, Any]:
    """Serialize one node for the document ``nodes`` array (children as ids)."""
    d: dict[str, Any] = {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "parent": node.parent,
        "subtree_hash": node.subtree_hash,
        "stable_node_hash": node.stable_node_hash,
        "child_node_ids": [c.node_id for c in node.children],
    }
    if node.param_name is not None:
        d["param_name"] = node.param_name
    if node.operator_id is not None:
        d["operator_id"] = node.operator_id
    if node.primitive_id is not None:
        d["primitive_id"] = node.primitive_id
    if node.feature_uuid is not None:
        d["feature_uuid"] = node.feature_uuid
    if node.literal is not None:
        d["literal"] = node.literal.to_dict()
    return d


def ast_document(
    root: AstNode,
    *,
    transformation_uuid: str,
    expression_hash: str,
    ast_hash: str,
    grammar_version: str,
    compiler_version: str,
    operator_pack_version: str,
    ast_schema_version: str = AST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Sole full AST document shape (§10.5)."""
    nodes = flatten_nodes(root)
    return {
        "ast_schema_version": ast_schema_version,
        "grammar_version": grammar_version,
        "compiler_version": compiler_version,
        "operator_pack_version": operator_pack_version,
        "transformation_uuid": transformation_uuid,
        "expression_hash": expression_hash,
        "ast_hash": ast_hash,
        "root_node_id": root.node_id,
        "nodes": [node_to_flat_dict(n) for n in nodes],
    }


def _node_from_flat(data: dict[str, Any]) -> AstNode:
    lit = data.get("literal")
    return AstNode(
        node_id=str(data["node_id"]),
        node_type=str(data["node_type"]),  # type: ignore[arg-type]
        children=[],
        parent=data.get("parent"),
        subtree_hash=str(data.get("subtree_hash") or ""),
        stable_node_hash=data.get("stable_node_hash"),
        param_name=data.get("param_name"),
        operator_id=data.get("operator_id"),
        primitive_id=data.get("primitive_id"),
        feature_uuid=data.get("feature_uuid"),
        literal=LiteralPayload.from_dict(lit) if isinstance(lit, dict) else None,
    )


def document_to_root(doc: dict[str, Any]) -> AstNode:
    """Rebuild tree from a §10.5 document."""
    if "root" in doc and isinstance(doc["root"], dict):
        return AstNode.from_dict(doc["root"])

    nodes_raw = doc.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("AST document missing nodes")

    first = nodes_raw[0]
    if isinstance(first.get("children"), list) and first["children"] and isinstance(
        first["children"][0], dict
    ):
        root_id = str(doc.get("root_node_id") or first["node_id"])
        for n in nodes_raw:
            if str(n["node_id"]) == root_id:
                return AstNode.from_dict(n)
        return AstNode.from_dict(first)

    by_id = {str(n["node_id"]): _node_from_flat(n) for n in nodes_raw}
    for raw in nodes_raw:
        nid = str(raw["node_id"])
        node = by_id[nid]
        child_ids = list(raw.get("child_node_ids") or [])
        if not child_ids:
            child_ids = [
                str(other["node_id"])
                for other in nodes_raw
                if other.get("parent") == nid
            ]
            child_ids.sort(key=lambda x: int(x[1:]) if x.startswith("N") else x)
        node.children = [by_id[cid] for cid in child_ids if cid in by_id]
    root_id = str(doc.get("root_node_id") or "N0")
    return by_id[root_id]
