"""Feature AST node models (Sprint 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["operator", "primitive", "feature", "literal", "list"]
LiteralKind = Literal["int", "float", "bool", "string"]

AST_SCHEMA_VERSION = "1.0"


@dataclass
class LiteralPayload:
    kind: LiteralKind
    value: int | float | bool | str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiteralPayload:
        return cls(kind=str(data["kind"]), value=data["value"])  # type: ignore[arg-type]


@dataclass
class AstNode:
    """In-memory AST node. ``stable_node_hash`` is reserved (null / omit)."""

    node_id: str
    node_type: NodeType
    children: list[AstNode] = field(default_factory=list)
    parent: str | None = None
    subtree_hash: str = ""
    stable_node_hash: str | None = None
    param_name: str | None = None
    operator_id: str | None = None
    primitive_id: str | None = None
    feature_uuid: str | None = None
    literal: LiteralPayload | None = None

    def to_dict(self, *, include_stable: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "children": [c.to_dict(include_stable=include_stable) for c in self.children],
            "parent": self.parent,
            "subtree_hash": self.subtree_hash,
        }
        if include_stable:
            d["stable_node_hash"] = self.stable_node_hash
        if self.param_name is not None:
            d["param_name"] = self.param_name
        if self.operator_id is not None:
            d["operator_id"] = self.operator_id
        if self.primitive_id is not None:
            d["primitive_id"] = self.primitive_id
        if self.feature_uuid is not None:
            d["feature_uuid"] = self.feature_uuid
        if self.literal is not None:
            d["literal"] = self.literal.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AstNode:
        lit = data.get("literal")
        children_raw = data.get("children") or []
        node = cls(
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
        node.children = [cls.from_dict(c) for c in children_raw]
        return node


def assign_node_ids(root: AstNode) -> AstNode:
    """Assign DFS preorder ``N0``, ``N1``, … and parent pointers."""
    counter = 0

    def walk(node: AstNode, parent_id: str | None) -> None:
        nonlocal counter
        node.node_id = f"N{counter}"
        node.parent = parent_id
        counter += 1
        for child in node.children:
            walk(child, node.node_id)

    walk(root, None)
    return root


def flatten_nodes(root: AstNode) -> list[AstNode]:
    """Return all nodes in DFS preorder (ordinal ascending)."""
    out: list[AstNode] = []

    def walk(node: AstNode) -> None:
        out.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return out
