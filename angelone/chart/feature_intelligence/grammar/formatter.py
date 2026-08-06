"""Canonical TL formatter (Sprint 4, formatter_version 1.0.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_intelligence.grammar.models import (
    _Arg,
    _BoolLit,
    _Call,
    _FeatureRef,
    _FloatLit,
    _IntLit,
    _ListLit,
    _PrimitiveRef,
    _StringLit,
    _Value,
)
from feature_intelligence.grammar.validator import parse_internal


def _format_value(value: _Value, indent: int) -> str:
    if isinstance(value, _Call):
        return _format_call(value, indent)
    if isinstance(value, _PrimitiveRef):
        return value.name
    if isinstance(value, _FeatureRef):
        return value.name
    if isinstance(value, _IntLit):
        return value.text
    if isinstance(value, _FloatLit):
        return value.text
    if isinstance(value, _BoolLit):
        return "true" if value.value else "false"
    if isinstance(value, _StringLit):
        escaped = value.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, _ListLit):
        if not value.items:
            return "[]"
        inner = ", ".join(_format_value(v, indent) for v in value.items)
        return f"[{inner}]"
    raise TypeError(f"unknown value type: {type(value)}")


def _ordered_args(
    args: list[_Arg],
    schema: dict[str, Any] | None,
) -> list[_Arg]:
    by_name = {a.name: a for a in args}
    ordered: list[_Arg] = []
    seen: set[str] = set()
    required = list((schema or {}).get("required") or [])
    for name in required:
        if name in by_name:
            ordered.append(by_name[name])
            seen.add(name)
    rest = sorted(n for n in by_name if n not in seen)
    for name in rest:
        ordered.append(by_name[name])
    return ordered


def _format_call(
    call: _Call,
    indent: int,
    *,
    schema: dict[str, Any] | None = None,
    schema_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    pad = " " * indent
    inner_pad = " " * (indent + 4)
    sch = schema
    if sch is None and schema_lookup is not None:
        sch = schema_lookup.get(call.operator)
    args = _ordered_args(call.args, sch)
    if not args:
        return f"{call.operator}()"
    lines = [f"{call.operator}("]
    for i, arg in enumerate(args):
        child_schema = None
        if isinstance(arg.value, _Call) and schema_lookup is not None:
            child_schema = schema_lookup.get(arg.value.operator)
        if isinstance(arg.value, _Call):
            val_text = _format_call(
                arg.value,
                indent + 4,
                schema=child_schema,
                schema_lookup=schema_lookup,
            )
        else:
            val_text = _format_value(arg.value, indent + 4)
        comma = "," if i < len(args) - 1 else ""
        # Multi-line nested call: first line joins with name =
        val_lines = val_text.split("\n")
        if len(val_lines) == 1:
            lines.append(f"{inner_pad}{arg.name} = {val_lines[0]}{comma}")
        else:
            lines.append(f"{inner_pad}{arg.name} = {val_lines[0]}")
            for vl in val_lines[1:-1]:
                lines.append(vl)
            lines.append(f"{val_lines[-1]}{comma}")
    lines.append(f"{pad})")
    return "\n".join(lines)


def format_tree(
    call: _Call,
    *,
    schema_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    return _format_call(call, 0, schema_lookup=schema_lookup)


def format_expression(
    text: str,
    *,
    db_path: Path | None = None,
) -> str:
    """Parse and emit canonical TL text. Raises ValueError on syntax errors."""
    try:
        tree = parse_internal(text)
    except Exception as exc:  # noqa: BLE001 — surface parse codes
        code = getattr(exc, "code", None)
        msg = getattr(exc, "message", str(exc))
        raise ValueError(f"{code or 'PARSE'}:{msg}") from exc

    schema_lookup: dict[str, dict[str, Any]] | None = None
    if db_path is not None:
        from feature_intelligence.operators.operator_store import OperatorStore

        store = OperatorStore(db_path)
        if store.table_exists():
            schema_lookup = {}
            for row in store.list_all():
                schema_lookup[row.operator_id] = json.loads(row.parameter_schema_json)
    else:
        # Offline: use seed catalog schemas for stable param order
        from feature_intelligence.operators.catalog import SEED_OPERATORS

        schema_lookup = {
            o.operator_id: json.loads(o.parameter_schema_json) for o in SEED_OPERATORS
        }

    result = format_tree(tree, schema_lookup=schema_lookup)
    if not result.endswith("\n"):
        # Spec uses \n line endings inside; no trailing blank line required
        pass
    return result


def format_file(path: Path, *, db_path: Path | None = None) -> str:
    return format_expression(Path(path).read_text(encoding="utf-8"), db_path=db_path)
