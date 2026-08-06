"""Transformation envelope import / export (Sprint 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_intelligence.ast.nodes import AST_SCHEMA_VERSION
from feature_intelligence.ast.serialize import ast_document
from feature_intelligence.compiler.identity import (
    derive_transformation_uuid,
    expression_hash,
)
from feature_intelligence.compiler.models import COMPILER_VERSION
from feature_intelligence.compiler.pipeline import compile
from feature_intelligence.grammar.pack import FORMATTER_VERSION, GRAMMAR_VERSION
from feature_intelligence.operators.catalog import OPERATOR_PACK_VERSION


def _load(path: Path, fmt: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            from feature_intelligence.core import _yaml_lite

            stripped = text.lstrip()
            if stripped.startswith("{"):
                data = json.loads(text)
            else:
                data = _yaml_lite.loads(text)
        if not isinstance(data, dict):
            raise ValueError("YAML envelope must be a mapping")
        return data
    raise ValueError(f"Unsupported format: {fmt}")


def _dump(data: dict[str, Any], path: Path, fmt: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return
    if fmt == "yaml":
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        except ImportError:
            out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return
    raise ValueError(f"Unsupported format: {fmt}")


def export_transformation(
    text: str,
    path: Path,
    *,
    fmt: str = "json",
    db: Path | None = None,
    persist: bool = False,
) -> Path:
    result = compile(text, mode="bound", db=db, persist=persist)
    if not result.ok or result.transformation is None:
        raise ValueError(f"compile failed: {result.report.failed_rules}")
    obj = result.transformation
    doc = ast_document(
        obj.root,
        transformation_uuid=obj.transformation_uuid,
        expression_hash=obj.expression_hash,
        ast_hash=obj.ast_hash,
        grammar_version=obj.grammar_version,
        compiler_version=obj.compiler_version,
        operator_pack_version=obj.operator_pack_version,
        ast_schema_version=obj.ast_schema_version,
    )
    envelope: dict[str, Any] = {
        "ast_schema_version": obj.ast_schema_version,
        "grammar_version": obj.grammar_version,
        "compiler_version": obj.compiler_version,
        "operator_pack_version": obj.operator_pack_version,
        "formatter_version": FORMATTER_VERSION,
        "transformation_uuid": obj.transformation_uuid,
        "expression_hash": obj.expression_hash,
        "compilation_uuid": obj.compilation_uuid,
        "ast_hash": obj.ast_hash,
        "canonical_text": obj.canonical_text,
        "source_text": obj.source_text,
        "manifest": None if obj.manifest is None else obj.manifest.to_dict(),
        "ast": doc,
    }
    _dump(envelope, path, fmt)
    return Path(path)


def import_transformation(
    path: Path,
    *,
    fmt: str = "json",
    db: Path | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """Re-validate + re-hash; re-derive TR_*; conflict if envelope id ≠ derived."""
    envelope = _load(path, fmt)
    canonical = str(envelope.get("canonical_text") or "")
    if not canonical:
        raise ValueError("envelope missing canonical_text")
    derived = derive_transformation_uuid(canonical)
    env_tr = envelope.get("transformation_uuid")
    if env_tr is not None and str(env_tr) != derived:
        raise ValueError(f"REGISTRY_CONFLICT: envelope TR_ {env_tr} != derived {derived}")
    env_hash = envelope.get("expression_hash")
    if env_hash is not None and str(env_hash) != expression_hash(canonical):
        raise ValueError("HASH_MISMATCH: expression_hash")

    result = compile(
        canonical,
        mode="bound",
        db=db,
        persist=persist,
        record_cache_hit_event=True,
    )
    if not result.ok or result.transformation is None:
        raise ValueError(f"import compile failed: {result.report.failed_rules}")

    # Optional: verify envelope AST hash when present and versions match
    env_ast = envelope.get("ast_hash")
    if (
        env_ast
        and str(envelope.get("compiler_version") or "") == COMPILER_VERSION
        and str(envelope.get("grammar_version") or "") == GRAMMAR_VERSION
        and str(envelope.get("operator_pack_version") or "") == OPERATOR_PACK_VERSION
        and str(env_ast) != result.transformation.ast_hash
    ):
        raise ValueError("HASH_MISMATCH: ast_hash")

    return {
        "transformation_uuid": result.transformation.transformation_uuid,
        "expression_hash": result.transformation.expression_hash,
        "compilation_uuid": result.transformation.compilation_uuid,
        "ast_hash": result.transformation.ast_hash,
        "canonical_text": result.transformation.canonical_text,
        "ast_schema_version": AST_SCHEMA_VERSION,
    }
