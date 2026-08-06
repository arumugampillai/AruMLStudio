"""ast package — Feature AST node models + hashing (Sprint 5)."""

from __future__ import annotations

from feature_intelligence.ast.hashing import (
    compute_ast_hash,
    compute_subtree_hash,
    hash_tree,
    sha256_hex,
)
from feature_intelligence.ast.nodes import (
    AST_SCHEMA_VERSION,
    AstNode,
    LiteralPayload,
    assign_node_ids,
    flatten_nodes,
)
from feature_intelligence.ast.serialize import (
    ast_document,
    canonical_subtree_payload,
    document_to_root,
    dumps_canonical,
    node_to_flat_dict,
)

__all__ = [
    "AST_SCHEMA_VERSION",
    "AstNode",
    "LiteralPayload",
    "assign_node_ids",
    "ast_document",
    "canonical_subtree_payload",
    "compute_ast_hash",
    "compute_subtree_hash",
    "document_to_root",
    "dumps_canonical",
    "flatten_nodes",
    "hash_tree",
    "node_to_flat_dict",
    "sha256_hex",
]
