"""AST hashing: subtree_hash + versioned ast_hash envelope (Sprint 5)."""

from __future__ import annotations

import hashlib

from feature_intelligence.ast.nodes import AstNode
from feature_intelligence.ast.serialize import canonical_subtree_payload, dumps_canonical


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_subtree_hash(node: AstNode) -> str:
    """Bottom-up: hash children first, then this node's canonical payload."""
    for child in node.children:
        compute_subtree_hash(child)
    payload = canonical_subtree_payload(node)
    node.subtree_hash = sha256_hex(dumps_canonical(payload))
    return node.subtree_hash


def compute_ast_hash(
    root: AstNode,
    *,
    grammar_version: str,
    compiler_version: str,
    operator_pack_version: str,
) -> str:
    """Versioned envelope fingerprint (integrity only — not used for TR_*)."""
    if not root.subtree_hash:
        compute_subtree_hash(root)
    material = (
        f"{grammar_version}|{compiler_version}|{operator_pack_version}|{root.subtree_hash}"
    )
    return sha256_hex(material)


def hash_tree(
    root: AstNode,
    *,
    grammar_version: str,
    compiler_version: str,
    operator_pack_version: str,
) -> str:
    """Compute all subtree hashes and return ``ast_hash``."""
    compute_subtree_hash(root)
    return compute_ast_hash(
        root,
        grammar_version=grammar_version,
        compiler_version=compiler_version,
        operator_pack_version=operator_pack_version,
    )
