"""compiler package — Transformation Compiler (Sprint 5).

Public surface: parse / compile / decompile. No execution.
"""

from __future__ import annotations

from feature_intelligence.compiler.compatibility import (
    CompatibilityError,
    is_supported,
    load_compatibility,
    require_supported,
)
from feature_intelligence.compiler.identity import (
    derive_transformation_uuid,
    expression_hash,
    generate_compilation_uuid,
)
from feature_intelligence.compiler.import_export import (
    export_transformation,
    import_transformation,
)
from feature_intelligence.compiler.models import (
    AST_SCHEMA_VERSION,
    COMPILER_VERSION,
    CompileMetrics,
    CompileResult,
    CompilerManifest,
    ParseResult,
    TransformationObject,
)
from feature_intelligence.compiler.pipeline import (
    compile,
    decompile,
    parse,
    validate_roundtrip,
)

__all__ = [
    "AST_SCHEMA_VERSION",
    "COMPILER_VERSION",
    "CompatibilityError",
    "CompileMetrics",
    "CompileResult",
    "CompilerManifest",
    "ParseResult",
    "TransformationObject",
    "compile",
    "decompile",
    "derive_transformation_uuid",
    "export_transformation",
    "expression_hash",
    "generate_compilation_uuid",
    "import_transformation",
    "is_supported",
    "load_compatibility",
    "parse",
    "require_supported",
    "validate_roundtrip",
]
