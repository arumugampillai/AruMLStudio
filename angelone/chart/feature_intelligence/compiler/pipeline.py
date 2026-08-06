"""Transformation compiler pipeline (Sprint 5).

Format → validate → derive TR_* → cache → build AST → hash → COMP_* / Manifest.
No execution, optimization, folding, or bytecode.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.ast.hashing import hash_tree
from feature_intelligence.ast.nodes import (
    AST_SCHEMA_VERSION,
    AstNode,
    LiteralPayload,
    assign_node_ids,
)
from feature_intelligence.compiler.cache import CompilerCache, get_default_cache
from feature_intelligence.compiler.compatibility import (
    CompatibilityError,
    require_supported,
)
from feature_intelligence.compiler.decompiler import decompile
from feature_intelligence.compiler.identity import (
    derive_transformation_uuid,
    expression_hash,
    generate_compilation_uuid,
)
from feature_intelligence.compiler.manifest import collect_operator_ids, emit_manifest
from feature_intelligence.compiler.models import (
    COMPILER_VERSION,
    CompileMetrics,
    CompileResult,
    ParseResult,
    TransformationObject,
)
from feature_intelligence.compiler.store import CompilerStore
from feature_intelligence.grammar.formatter import format_expression
from feature_intelligence.grammar.models import (
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
from feature_intelligence.grammar.pack import (
    FORMATTER_VERSION,
    GRAMMAR_PACK_VERSION,
    GRAMMAR_VERSION,
)
from feature_intelligence.grammar.validator import validate_expression
from feature_intelligence.operators.catalog import OPERATOR_PACK_VERSION, SEED_OPERATORS
from feature_intelligence.registry.models import ValidationReport

# Re-export decompile at pipeline module for convenience
__all__ = [
    "build_ast_from_call",
    "compile",
    "decompile",
    "parse",
    "validate_roundtrip",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _fail(code: str, detail: str = "") -> str:
    return f"{code}:{detail}" if detail else code


def _report(
    *,
    passed: bool,
    failed: list[str],
    warnings: list[str] | None = None,
    validated_objects: str = "",
) -> ValidationReport:
    return ValidationReport(
        passed=passed,
        failed_rules=failed,
        warnings=list(warnings or []),
        seed_hash="",
        expected_seed_hash="",
        validated_objects=validated_objects,
        timestamp=_utc_now(),
    )


def _schema_lookup(db_path: Path | None) -> dict[str, dict[str, Any]]:
    if db_path is not None:
        from feature_intelligence.operators.operator_store import OperatorStore

        store = OperatorStore(db_path)
        if store.table_exists():
            out: dict[str, dict[str, Any]] = {}
            for row in store.list_all():
                out[row.operator_id] = json.loads(row.parameter_schema_json)
            return out
    return {
        o.operator_id: json.loads(o.parameter_schema_json) for o in SEED_OPERATORS
    }


def _ordered_arg_names(
    call: _Call,
    schema: dict[str, Any] | None,
) -> list[str]:
    """Sprint 4 formatter order: required (schema array order), then alpha rest."""
    present = {a.name: a for a in call.args}
    ordered: list[str] = []
    seen: set[str] = set()
    required = list((schema or {}).get("required") or [])
    for name in required:
        if name in present:
            ordered.append(name)
            seen.add(name)
    for name in sorted(n for n in present if n not in seen):
        ordered.append(name)
    return ordered


def _reject_undeclared_params(
    call: _Call,
    schemas: dict[str, dict[str, Any]],
    failures: list[str],
) -> None:
    """Sprint 5: only schema-declared params may appear (no invented series slots)."""
    schema = schemas.get(call.operator) or {}
    props = dict(schema.get("properties") or {})
    for arg in call.args:
        if arg.name not in props:
            failures.append(_fail("UNKNOWN_PARAM", f"{call.operator}.{arg.name}"))
        val = arg.value
        if isinstance(val, _Call):
            _reject_undeclared_params(val, schemas, failures)
        elif isinstance(val, _ListLit):
            for item in val.items:
                if isinstance(item, _Call):
                    _reject_undeclared_params(item, schemas, failures)


def _value_to_node(
    value: _Value,
    *,
    schemas: dict[str, dict[str, Any]],
) -> AstNode:
    if isinstance(value, _Call):
        raise TypeError("use _call_to_node for calls")
    if isinstance(value, _PrimitiveRef):
        return AstNode(
            node_id="",
            node_type="primitive",
            primitive_id=value.name,
            stable_node_hash=None,
        )
    if isinstance(value, _FeatureRef):
        return AstNode(
            node_id="",
            node_type="feature",
            feature_uuid=value.name,
            stable_node_hash=None,
        )
    if isinstance(value, _IntLit):
        return AstNode(
            node_id="",
            node_type="literal",
            literal=LiteralPayload(kind="int", value=value.value),
            stable_node_hash=None,
        )
    if isinstance(value, _FloatLit):
        return AstNode(
            node_id="",
            node_type="literal",
            literal=LiteralPayload(kind="float", value=value.value),
            stable_node_hash=None,
        )
    if isinstance(value, _BoolLit):
        return AstNode(
            node_id="",
            node_type="literal",
            literal=LiteralPayload(kind="bool", value=value.value),
            stable_node_hash=None,
        )
    if isinstance(value, _StringLit):
        return AstNode(
            node_id="",
            node_type="literal",
            literal=LiteralPayload(kind="string", value=value.value),
            stable_node_hash=None,
        )
    if isinstance(value, _ListLit):
        children = []
        for item in value.items:
            if isinstance(item, _Call):
                children.append(_call_to_node(item, schemas=schemas))
            else:
                children.append(_value_to_node(item, schemas=schemas))
        return AstNode(
            node_id="",
            node_type="list",
            children=children,
            stable_node_hash=None,
        )
    raise TypeError(f"unsupported value type: {type(value)}")


def _call_to_node(
    call: _Call,
    *,
    schemas: dict[str, dict[str, Any]],
) -> AstNode:
    schema = schemas.get(call.operator)
    names = _ordered_arg_names(call, schema)
    by_name = {a.name: a for a in call.args}
    children: list[AstNode] = []
    for name in names:
        arg = by_name[name]
        if isinstance(arg.value, _Call):
            child = _call_to_node(arg.value, schemas=schemas)
        else:
            child = _value_to_node(arg.value, schemas=schemas)
        child.param_name = name
        children.append(child)
    return AstNode(
        node_id="",
        node_type="operator",
        operator_id=call.operator,
        children=children,
        stable_node_hash=None,
    )


def build_ast_from_call(
    call: _Call,
    *,
    schemas: dict[str, dict[str, Any]] | None = None,
) -> AstNode:
    schemas = schemas or {}
    root = _call_to_node(call, schemas=schemas)
    assign_node_ids(root)
    return root


def _operator_versions(
    root: AstNode,
    db_path: Path | None,
) -> dict[str, str]:
    ids = collect_operator_ids(root)
    versions: dict[str, str] = {}
    if db_path is not None:
        from feature_intelligence.operators.operator_store import OperatorStore

        store = OperatorStore(db_path)
        if store.table_exists():
            for oid in ids:
                row = store.get_by_id(oid)
                if row is not None:
                    versions[oid] = row.operator_version
                    continue
                versions[oid] = "1.0"
            return versions
    for oid in ids:
        versions[oid] = "1.0"
    return versions


def _build_object(
    *,
    source_text: str | None,
    canonical_text: str,
    root: AstNode,
    ast_hash: str,
    compilation_uuid: str | None,
    manifest: Any,
    cache_hit: bool | None,
    metrics: CompileMetrics | None,
) -> TransformationObject:
    return TransformationObject(
        transformation_uuid=derive_transformation_uuid(canonical_text),
        expression_hash=expression_hash(canonical_text),
        compilation_uuid=compilation_uuid,
        ast_hash=ast_hash,
        ast_schema_version=AST_SCHEMA_VERSION,
        grammar_version=GRAMMAR_VERSION,
        compiler_version=COMPILER_VERSION,
        operator_pack_version=OPERATOR_PACK_VERSION,
        source_text=source_text,
        canonical_text=canonical_text,
        root=root,
        manifest=manifest,
        cache_hit=cache_hit,
        metrics=metrics,
    )


def parse(
    text: str,
    *,
    mode: str = "bound",
    db: Path | str | None = None,
) -> ParseResult:
    """Format → validate → derive TR_* → build AST. No persist / no COMP_*."""
    db_path = Path(db) if db is not None else None
    source_text = text
    failed: list[str] = []

    try:
        require_supported()
    except CompatibilityError as exc:
        return ParseResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[_fail(exc.code, str(exc))],
                validated_objects="0 transformations",
            ),
        )

    try:
        canonical = format_expression(text, db_path=db_path)
    except ValueError as exc:
        return ParseResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[str(exc)],
                validated_objects="0 transformations",
            ),
        )

    report, tree = validate_expression(canonical, mode=mode, db_path=db_path)
    if not report.passed or tree is None:
        return ParseResult(ok=False, transformation=None, report=report)

    schemas = _schema_lookup(db_path)
    _reject_undeclared_params(tree, schemas, failed)
    if failed:
        return ParseResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=failed,
                validated_objects="0 transformations",
            ),
        )

    root = build_ast_from_call(tree, schemas=schemas)
    ast_h = hash_tree(
        root,
        grammar_version=GRAMMAR_VERSION,
        compiler_version=COMPILER_VERSION,
        operator_pack_version=OPERATOR_PACK_VERSION,
    )
    obj = _build_object(
        source_text=source_text,
        canonical_text=canonical,
        root=root,
        ast_hash=ast_h,
        compilation_uuid=None,
        manifest=None,
        cache_hit=None,
        metrics=None,
    )
    return ParseResult(
        ok=True,
        transformation=obj,
        report=_report(
            passed=True,
            failed=[],
            validated_objects="1 transformation",
        ),
    )


def compile(  # noqa: A001 — public API name per freeze
    text: str,
    *,
    mode: str = "bound",
    db: Path | str | None = None,
    persist: bool = False,
    feature_uuid: str | None = None,
    metrics: bool = False,
    record_cache_hit_event: bool = True,
    cache: CompilerCache | None = None,
) -> CompileResult:
    """Full pipeline with optional persist and COMP_* mint."""
    t0 = time.perf_counter()
    db_path = Path(db) if db is not None else None
    cache = cache if cache is not None else get_default_cache()
    source_text = text
    m_parse = m_compile = m_hash = m_persist = None

    try:
        require_supported()
    except CompatibilityError as exc:
        return CompileResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[_fail(exc.code, str(exc))],
                validated_objects="0 transformations",
            ),
        )

    t_parse0 = time.perf_counter()
    try:
        canonical = format_expression(text, db_path=db_path)
    except ValueError as exc:
        return CompileResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[str(exc)],
                validated_objects="0 transformations",
            ),
        )

    report, tree = validate_expression(canonical, mode=mode, db_path=db_path)
    if metrics:
        m_parse = (time.perf_counter() - t_parse0) * 1000.0

    if not report.passed or tree is None:
        return CompileResult(ok=False, transformation=None, report=report)

    schemas = _schema_lookup(db_path)
    failed: list[str] = []
    _reject_undeclared_params(tree, schemas, failed)
    if failed:
        return CompileResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=failed,
                validated_objects="0 transformations",
            ),
        )

    expr_h = expression_hash(canonical)
    tr_uuid = derive_transformation_uuid(canonical)

    # Cache lookup
    hit = cache.get(expr_h)
    cache_hit = hit is not None

    t_compile0 = time.perf_counter()
    if cache_hit and hit is not None:
        root = hit.root
        ast_h = hit.ast_hash
        op_versions = dict(hit.operator_versions)
    else:
        root = build_ast_from_call(tree, schemas=schemas)
        t_hash0 = time.perf_counter()
        ast_h = hash_tree(
            root,
            grammar_version=GRAMMAR_VERSION,
            compiler_version=COMPILER_VERSION,
            operator_pack_version=OPERATOR_PACK_VERSION,
        )
        if metrics:
            m_hash = (time.perf_counter() - t_hash0) * 1000.0
        op_versions = _operator_versions(root, db_path)

    if metrics:
        m_compile = (time.perf_counter() - t_compile0) * 1000.0

    # Round-trip check on miss (and always cheap verify)
    decompiled = decompile(root)
    try:
        roundtrip = format_expression(decompiled, db_path=db_path)
    except ValueError as exc:
        return CompileResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[_fail("ROUNDTRIP_MISMATCH", str(exc))],
                validated_objects="0 transformations",
            ),
        )
    if roundtrip != canonical:
        return CompileResult(
            ok=False,
            transformation=None,
            report=_report(
                passed=False,
                failed=[_fail("ROUNDTRIP_MISMATCH", "canonical inequality")],
                validated_objects="0 transformations",
            ),
        )

    # Successful compile always emits an in-memory Manifest + COMP_* except when
    # a cache-hit opts out of the lightweight event entirely.
    skip_cache_hit_event = cache_hit and not record_cache_hit_event
    compilation_uuid: str | None = None
    manifest = None
    if not skip_cache_hit_event:
        compilation_uuid = generate_compilation_uuid()
        manifest = emit_manifest(
            transformation_uuid=tr_uuid,
            compilation_uuid=compilation_uuid,
            expression_hash=expr_h,
            ast_hash=ast_h,
            root=root,
            canonical_text=canonical,
            operator_versions=op_versions,
            cache_hit=cache_hit,
        )

    metrics_bag = None
    if metrics:
        total = (time.perf_counter() - t0) * 1000.0
        metrics_bag = CompileMetrics(
            parse_ms=m_parse,
            compile_ms=m_compile,
            hash_ms=m_hash,
            persist_ms=m_persist,
            total_ms=total,
        )

    obj = _build_object(
        source_text=source_text,
        canonical_text=canonical,
        root=root,
        ast_hash=ast_h,
        compilation_uuid=compilation_uuid,
        manifest=manifest,
        cache_hit=cache_hit,
        metrics=metrics_bag,
    )

    if not cache_hit:
        cache.put_from_object(obj)

    if persist:
        if db_path is None:
            return CompileResult(
                ok=False,
                transformation=obj,
                report=_report(
                    passed=False,
                    failed=[_fail("PERSIST_FAILED", "persist requires db")],
                    validated_objects="0 transformations",
                ),
            )
        t_p0 = time.perf_counter()
        try:
            store = CompilerStore(db_path)
            store.insert_transformation(
                transformation_uuid=tr_uuid,
                expression_hash=expr_h,
                canonical_text=canonical,
                source_text=source_text,
            )
            write_ast = not cache_hit or not store.has_ast_nodes(tr_uuid, ast_h)
            if write_ast:
                store.persist_ast_nodes(
                    transformation_uuid=tr_uuid,
                    ast_hash=ast_h,
                    root=root,
                )
            should_record_comp = compilation_uuid is not None and (
                not cache_hit or record_cache_hit_event
            )
            if should_record_comp and compilation_uuid is not None:
                store.insert_compilation(
                    compilation_uuid=compilation_uuid,
                    transformation_uuid=tr_uuid,
                    ast_schema_version=AST_SCHEMA_VERSION,
                    grammar_version=GRAMMAR_VERSION,
                    compiler_version=COMPILER_VERSION,
                    operator_pack_version=OPERATOR_PACK_VERSION,
                    ast_hash=ast_h,
                    root_node_id=root.node_id,
                    cache_hit=cache_hit,
                    status="success",
                    metrics=metrics_bag,
                )
            if feature_uuid:
                from feature_intelligence.registry.feature_ids import (
                    normalize_feature_uuid,
                )

                try:
                    fu = normalize_feature_uuid(feature_uuid)
                except ValueError:
                    return CompileResult(
                        ok=False,
                        transformation=obj,
                        report=_report(
                            passed=False,
                            failed=[_fail("FEATURE_LINK", feature_uuid)],
                            validated_objects="0 transformations",
                        ),
                    )
                if compilation_uuid is None:
                    # Need a verifying compile id for feature_ast FK
                    compilation_uuid = generate_compilation_uuid()
                    store.insert_compilation(
                        compilation_uuid=compilation_uuid,
                        transformation_uuid=tr_uuid,
                        ast_schema_version=AST_SCHEMA_VERSION,
                        grammar_version=GRAMMAR_VERSION,
                        compiler_version=COMPILER_VERSION,
                        operator_pack_version=OPERATOR_PACK_VERSION,
                        ast_hash=ast_h,
                        root_node_id=root.node_id,
                        cache_hit=cache_hit,
                        status="success",
                        metrics=metrics_bag,
                    )
                    obj.compilation_uuid = compilation_uuid
                store.upsert_feature_ast(
                    feature_uuid=fu,
                    transformation_uuid=tr_uuid,
                    compilation_uuid=compilation_uuid,
                    ast_schema_version=AST_SCHEMA_VERSION,
                    root=root,
                    ast_hash=ast_h,
                    expression_hash=expr_h,
                    grammar_version=GRAMMAR_VERSION,
                    compiler_version=COMPILER_VERSION,
                    operator_pack_version=OPERATOR_PACK_VERSION,
                    rewrite_body=write_ast,
                )
            store.bump_statistics(
                cache_hit=cache_hit,
                total_ms=metrics_bag.total_ms if metrics_bag else None,
            )
        except Exception as exc:  # noqa: BLE001
            return CompileResult(
                ok=False,
                transformation=obj,
                report=_report(
                    passed=False,
                    failed=[_fail("PERSIST_FAILED", str(exc))],
                    validated_objects="0 transformations",
                ),
            )
        if metrics and metrics_bag is not None:
            metrics_bag.persist_ms = (time.perf_counter() - t_p0) * 1000.0
            metrics_bag.total_ms = (time.perf_counter() - t0) * 1000.0
            obj.metrics = metrics_bag

    return CompileResult(
        ok=True,
        transformation=obj,
        report=_report(
            passed=True,
            failed=[],
            validated_objects="1 transformation",
        ),
    )


def validate_roundtrip(
    text: str,
    *,
    mode: str = "bound",
    db: Path | str | None = None,
) -> ValidationReport:
    """Compile → decompile → format; compare to canonical."""
    result = compile(text, mode=mode, db=db, persist=False, record_cache_hit_event=False)
    if not result.ok or result.transformation is None:
        return result.report
    # Round-trip already enforced inside compile; re-assert for report clarity
    obj = result.transformation
    again = format_expression(decompile(obj.root), db_path=Path(db) if db else None)
    if again != obj.canonical_text:
        return _report(
            passed=False,
            failed=[_fail("ROUNDTRIP_MISMATCH")],
            validated_objects="1 transformation",
        )
    return _report(
        passed=True,
        failed=[],
        validated_objects="1 transformation",
    )
