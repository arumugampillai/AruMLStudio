"""AST → TL text decompiler (Sprint 5). No execution."""

from __future__ import annotations

from feature_intelligence.ast.nodes import AstNode


def decompile(root: AstNode, *, indent: int = 0) -> str:
    """Emit TL text from an AST (pre-format layout matching Sprint 4 formatter)."""
    if root.node_type == "operator":
        return _decompile_operator(root, indent)
    return _decompile_value(root, indent)


def _decompile_value(node: AstNode, indent: int) -> str:
    if node.node_type == "operator":
        return _decompile_operator(node, indent)
    if node.node_type == "primitive":
        return node.primitive_id or ""
    if node.node_type == "feature":
        return node.feature_uuid or ""
    if node.node_type == "literal":
        assert node.literal is not None
        kind = node.literal.kind
        val = node.literal.value
        if kind == "bool":
            return "true" if val else "false"
        if kind == "string":
            escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if kind == "float":
            # Prefer a stable float text
            text = repr(float(val)) if isinstance(val, (int, float)) else str(val)
            if text.endswith(".0") and "." in text:
                return text
            return str(val)
        return str(val)
    if node.node_type == "list":
        if not node.children:
            return "[]"
        inner = ", ".join(_decompile_value(c, indent) for c in node.children)
        return f"[{inner}]"
    raise ValueError(f"unknown node_type: {node.node_type}")


def _decompile_operator(node: AstNode, indent: int) -> str:
    pad = " " * indent
    inner_pad = " " * (indent + 4)
    oid = node.operator_id or ""
    if not node.children:
        return f"{oid}()"
    lines = [f"{oid}("]
    for i, child in enumerate(node.children):
        name = child.param_name or ""
        val_text = _decompile_value(child, indent + 4)
        comma = "," if i < len(node.children) - 1 else ""
        val_lines = val_text.split("\n")
        if len(val_lines) == 1:
            lines.append(f"{inner_pad}{name} = {val_lines[0]}{comma}")
        else:
            lines.append(f"{inner_pad}{name} = {val_lines[0]}")
            for vl in val_lines[1:-1]:
                lines.append(vl)
            lines.append(f"{val_lines[-1]}{comma}")
    lines.append(f"{pad})")
    return "\n".join(lines)
