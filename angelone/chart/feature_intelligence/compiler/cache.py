"""In-memory compiler cache keyed by expression_hash (Sprint 5)."""

from __future__ import annotations

from dataclasses import dataclass

from feature_intelligence.ast.nodes import AstNode
from feature_intelligence.compiler.models import CompilerManifest, TransformationObject


@dataclass
class CacheEntry:
    transformation_uuid: str
    expression_hash: str
    ast_hash: str
    canonical_text: str
    root: AstNode
    # Snapshot of last successful non-cache-hit manifest fields (sans event ids)
    grammar_version: str
    compiler_version: str
    operator_pack_version: str
    ast_schema_version: str
    root_subtree_hash: str
    operator_versions: dict[str, str]


class CompilerCache:
    """Process-local cache: expression_hash → AST + transformation identity."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def get(self, expr_hash: str) -> CacheEntry | None:
        return self._entries.get(expr_hash)

    def put(self, entry: CacheEntry) -> None:
        self._entries[entry.expression_hash] = entry

    def put_from_object(self, obj: TransformationObject) -> None:
        op_versions: dict[str, str] = {}
        if obj.manifest is not None:
            op_versions = dict(obj.manifest.operator_versions)
        self.put(
            CacheEntry(
                transformation_uuid=obj.transformation_uuid,
                expression_hash=obj.expression_hash,
                ast_hash=obj.ast_hash,
                canonical_text=obj.canonical_text,
                root=obj.root,
                grammar_version=obj.grammar_version,
                compiler_version=obj.compiler_version,
                operator_pack_version=obj.operator_pack_version,
                ast_schema_version=obj.ast_schema_version,
                root_subtree_hash=obj.root.subtree_hash,
                operator_versions=op_versions,
            )
        )

    def clear(self) -> None:
        self._entries.clear()


# Module-level shared cache for library callers (tests may clear).
_DEFAULT_CACHE = CompilerCache()


def get_default_cache() -> CompilerCache:
    return _DEFAULT_CACHE
