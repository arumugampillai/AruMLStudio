"""TL expression import/export envelopes (Sprint 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_intelligence.core import _yaml_lite
from feature_intelligence.grammar.formatter import format_expression
from feature_intelligence.grammar.pack import (
    FORMATTER_VERSION,
    GRAMMAR_PACK_VERSION,
    GRAMMAR_VERSION,
    TOKEN_PACK_VERSION,
)

SCHEMA_VERSION = "1.0"


def _envelope(expression: str, *, mode: str = "syntax_only") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "grammar_version": GRAMMAR_VERSION,
        "grammar_pack_version": GRAMMAR_PACK_VERSION,
        "token_pack_version": TOKEN_PACK_VERSION,
        "formatter_version": FORMATTER_VERSION,
        "expression": expression,
        "mode": mode,
    }


def export_expression(
    text: str,
    path: Path,
    *,
    fmt: str = "json",
    mode: str = "syntax_only",
    db_path: Path | None = None,
) -> Path:
    """Export always writes canonical expression text."""
    canonical = format_expression(text, db_path=db_path)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "text":
        out.write_text(canonical + ("\n" if not canonical.endswith("\n") else ""), encoding="utf-8")
        return out
    envelope = _envelope(canonical, mode=mode)
    if fmt == "json":
        out.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    elif fmt == "yaml":
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(envelope, sort_keys=False), encoding="utf-8")
        except ImportError:
            # Match operator_import_export: JSON body when PyYAML is absent.
            out.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out


def _load_envelope(path: Path, fmt: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if fmt == "text":
        return _envelope(text, mode="syntax_only")
    if fmt == "json":
        data = json.loads(text)
    elif fmt == "yaml":
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            stripped = text.lstrip()
            if stripped.startswith("{"):
                data = json.loads(text)
            else:
                data = _yaml_lite.loads(text)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if not isinstance(data, dict):
        raise ValueError("Envelope must be a mapping")
    return data


def import_expression(
    path: Path,
    out_path: Path,
    *,
    fmt: str = "json",
    db_path: Path | None = None,
) -> str:
    """Import any valid text; re-format to canonical; write to out_path as .tl text."""
    envelope = _load_envelope(Path(path), fmt)
    if fmt != "text":
        ver = str(envelope.get("schema_version") or "")
        if ver and ver != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {ver!r}")
        gver = str(envelope.get("grammar_version") or GRAMMAR_VERSION)
        if gver != GRAMMAR_VERSION:
            raise ValueError(f"Unsupported grammar_version: {gver!r}")
    expr = str(envelope.get("expression") or "")
    if not expr.strip():
        # Plain text path already put body into expression
        raise ValueError("Empty expression")
    canonical = format_expression(expr, db_path=db_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical + ("\n" if not canonical.endswith("\n") else ""), encoding="utf-8")
    return canonical
