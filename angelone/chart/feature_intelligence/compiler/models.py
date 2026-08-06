"""Compiler version constants and public result models (Sprint 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from feature_intelligence.ast.nodes import AST_SCHEMA_VERSION, AstNode
from feature_intelligence.registry.models import ValidationReport

COMPILER_VERSION = "1.0.0"


@dataclass
class CompileMetrics:
    parse_ms: float | None = None
    compile_ms: float | None = None
    hash_ms: float | None = None
    persist_ms: float | None = None
    total_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompilerManifest:
    transformation_uuid: str
    compilation_uuid: str
    expression_hash: str
    ast_schema_version: str
    grammar_version: str
    grammar_pack_version: str
    formatter_version: str
    compiler_version: str
    operator_pack_version: str
    ast_hash: str
    root_subtree_hash: str
    canonical_text: str
    created_at: str
    definition_version: str | None = None
    implementation_version: str | None = None
    operator_versions: dict[str, str] = field(default_factory=dict)
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_manifest": {
                "transformation_uuid": self.transformation_uuid,
                "compilation_uuid": self.compilation_uuid,
                "expression_hash": self.expression_hash,
                "ast_schema_version": self.ast_schema_version,
                "grammar_version": self.grammar_version,
                "grammar_pack_version": self.grammar_pack_version,
                "formatter_version": self.formatter_version,
                "compiler_version": self.compiler_version,
                "operator_pack_version": self.operator_pack_version,
                "definition_version": self.definition_version,
                "implementation_version": self.implementation_version,
                "ast_hash": self.ast_hash,
                "root_subtree_hash": self.root_subtree_hash,
                "operator_versions": dict(self.operator_versions),
                "canonical_text": self.canonical_text,
                "cache_hit": self.cache_hit,
                "created_at": self.created_at,
            }
        }


@dataclass
class TransformationObject:
    transformation_uuid: str
    expression_hash: str
    ast_hash: str
    ast_schema_version: str
    grammar_version: str
    compiler_version: str
    operator_pack_version: str
    canonical_text: str
    root: AstNode
    compilation_uuid: str | None = None
    source_text: str | None = None
    manifest: CompilerManifest | None = None
    cache_hit: bool | None = None
    metrics: CompileMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_uuid": self.transformation_uuid,
            "expression_hash": self.expression_hash,
            "compilation_uuid": self.compilation_uuid,
            "ast_hash": self.ast_hash,
            "ast_schema_version": self.ast_schema_version,
            "grammar_version": self.grammar_version,
            "compiler_version": self.compiler_version,
            "operator_pack_version": self.operator_pack_version,
            "source_text": self.source_text,
            "canonical_text": self.canonical_text,
            "cache_hit": self.cache_hit,
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "root_node_id": self.root.node_id,
            "root_subtree_hash": self.root.subtree_hash,
        }


@dataclass
class ParseResult:
    ok: bool
    transformation: TransformationObject | None
    report: ValidationReport


@dataclass
class CompileResult:
    ok: bool
    transformation: TransformationObject | None
    report: ValidationReport


# Re-export for convenience
__all__ = [
    "AST_SCHEMA_VERSION",
    "COMPILER_VERSION",
    "CompileMetrics",
    "CompileResult",
    "CompilerManifest",
    "ParseResult",
    "TransformationObject",
]
