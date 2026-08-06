"""Feature Inspector aggregate (Sprint 9) — read-only metadata sections."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from feature_intelligence.compiler.store import CompilerStore
from feature_intelligence.lineage.service import LineageService
from feature_intelligence.ontology.models import OBJECT_TYPE_FEATURE
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.query.models import INSPECTOR_SECTIONS
from feature_intelligence.query.resolve import ResolvedSubject
from feature_intelligence.registry.feature_service import (
    FeatureNotFoundError,
    FeatureRegistryService,
)
from feature_intelligence.research.models import FeatureResearchRecord


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _feature_ast_row(db_path: Path, feature_uuid: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM feature_ast WHERE feature_uuid = ?",
            (feature_uuid,),
        ).fetchone()
        return None if row is None else dict(row)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _ontology_by_uuid(
    store: OntologyStore, ontology_uuid: str
) -> Any | None:
    for ot in (
        OBJECT_TYPE_FEATURE,
        "PRIMITIVE",
        "OPERATOR",
        "TRANSFORMATION",
    ):
        for rec in store.list_ontology(ot):
            if rec.ontology_uuid == ontology_uuid:
                return rec
    return None


def _resolve_display(store: OntologyStore, vocabulary_id: str | None) -> str | None:
    if not vocabulary_id:
        return None
    v = store.get_vocabulary(vocabulary_id)
    return None if v is None else v.display_name


def _identity_section(
    feat_svc: FeatureRegistryService, feature_uuid: str
) -> dict[str, Any] | None:
    try:
        feat = feat_svc.get_by_uuid(feature_uuid)
    except FeatureNotFoundError:
        return None
    return {
        "feature_uuid": feat.feature_uuid,
        "canonical_name": feat.canonical_name,
        "display_name": feat.display_name,
        "definition_version": feat.definition_version,
        "implementation_version": feat.implementation_version,
        "definition_hash": feat.definition_hash,
        "research_state": feat.research_state,
        "controller_owner": feat.controller_owner,
        "created_by": feat.created_by,
        "warmup_periods": feat.warmup_periods,
        "gap_policy": feat.gap_policy,
        "memory_model": feat.memory_model,
        "primitive_ids": list(feat.primitive_ids),
        "transformation_uuid": feat.transformation_uuid,
        "description": feat.description,
        "created_at": feat.created_at or None,
        "updated_at": feat.updated_at or None,
    }


def _compiler_and_ast(
    db_path: Path, frr: FeatureResearchRecord
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    tr_uuid = frr.transformation_uuid
    ast_row = _feature_ast_row(db_path, frr.feature_uuid)
    if tr_uuid is None and ast_row is not None:
        tr_uuid = ast_row.get("transformation_uuid")

    if tr_uuid is None and ast_row is None:
        return None, None

    compiler_store = CompilerStore(db_path)
    tr = compiler_store.get_transformation(tr_uuid) if tr_uuid else None

    compiler: dict[str, Any] | None = None
    if tr is not None or frr.compiler_version or frr.grammar_version:
        compiler = {
            "transformation_uuid": tr_uuid,
            "grammar_version": frr.grammar_version,
            "compiler_version": frr.compiler_version,
            "expression_hash": None if tr is None else tr.get("expression_hash"),
            "canonical_text": None if tr is None else tr.get("canonical_text"),
            "compilation_uuid": None if ast_row is None else ast_row.get("compilation_uuid"),
        }
        if ast_row is not None and ast_row.get("compilation_uuid"):
            comp = compiler_store.get_compilation(str(ast_row["compilation_uuid"]))
            if comp is not None:
                compiler["operator_pack_version"] = comp.get("operator_pack_version")
                compiler["ast_schema_version"] = comp.get("ast_schema_version")
                compiler["manifest_summary"] = {
                    "compilation_uuid": comp.get("compilation_uuid"),
                    "ast_hash": comp.get("ast_hash"),
                    "root_node_id": comp.get("root_node_id"),
                    "cache_hit": comp.get("cache_hit"),
                }

    ast_summary: dict[str, Any] | None = None
    if ast_row is not None:
        node_count = 0
        root_operator = None
        ast_hash = ast_row.get("ast_fingerprint") or ast_row.get("ast_hash")
        ast_json = ast_row.get("ast_json")
        if ast_json:
            try:
                doc = json.loads(ast_json) if isinstance(ast_json, str) else ast_json
                nodes = doc.get("nodes") if isinstance(doc, dict) else None
                if isinstance(nodes, list):
                    node_count = len(nodes)
                    for n in nodes:
                        if isinstance(n, dict) and n.get("operator_id"):
                            root_operator = n.get("operator_id")
                            break
            except (TypeError, json.JSONDecodeError):
                pass
        if node_count == 0 and tr_uuid and ast_hash:
            doc2 = compiler_store.load_ast_document_for_transformation(
                str(tr_uuid), str(ast_hash)
            )
            if doc2 and isinstance(doc2.get("nodes"), list):
                nodes = doc2["nodes"]
                node_count = len(nodes)
                for n in nodes:
                    if n.get("operator_id"):
                        root_operator = n.get("operator_id")
                        break
        ast_summary = {
            "ast_hash": ast_hash,
            "node_count": node_count,
            "root_operator": root_operator,
            "subtree_hash": ast_row.get("subtree_hash"),
            "ast_schema_version": ast_row.get("ast_schema_version"),
        }

    return compiler, ast_summary


def _ontology_section(
    store: OntologyStore, frr: FeatureResearchRecord
) -> dict[str, Any] | None:
    rec = None
    if frr.ontology_uuid:
        rec = _ontology_by_uuid(store, frr.ontology_uuid)
    if rec is None:
        rec = store.get_ontology(OBJECT_TYPE_FEATURE, frr.feature_uuid)
    if rec is None:
        return None
    signal_display = [
        {"vocabulary_id": s, "display_name": _resolve_display(store, s)}
        for s in (rec.signal_type or [])
    ]
    return {
        "ontology_uuid": rec.ontology_uuid,
        "object_type": rec.object_type,
        "object_id": rec.object_id,
        "ontology_version": rec.ontology_version,
        "domain": rec.domain,
        "domain_display": _resolve_display(store, rec.domain),
        "signal_type": list(rec.signal_type or []),
        "signal_type_display": signal_display,
        "mathematical_family": list(rec.mathematical_family or []),
        "horizon": rec.horizon,
        "output_type": rec.output_type,
        "frequency": rec.frequency,
        "stability": rec.stability,
        "meaning": rec.meaning,
    }


def _lineage_section(
    db_path: Path,
    feature_uuid: str,
    *,
    identity: dict[str, Any] | None = None,
    transformation_uuid: str | None = None,
) -> dict[str, Any] | None:
    svc = LineageService(db_path)
    parents = svc.parents(feature_uuid)
    children = svc.children(feature_uuid)
    ancestors = svc.ancestors(feature_uuid)
    if not parents and not children and not ancestors:
        # still "present" if any edges mention the feature? empty summary OK as absent
        pairs = svc.store.edge_pairs()
        if not any(feature_uuid in (a, b) for a, b in pairs):
            return None
    sample_parents = parents[:10]
    sample_children = children[:10]
    prim_anc = [a for a in ancestors if a.startswith("PR_")]
    op_anc = [a for a in ancestors if a.startswith("OP_")]
    prim_inputs = list((identity or {}).get("primitive_ids") or []) or prim_anc
    # Simple transformation chain: parents → subject TR → children
    chain: list[str] = []
    for p in sample_parents[:5]:
        chain.append(str(p))
    if transformation_uuid:
        chain.append(str(transformation_uuid))
    else:
        chain.append(feature_uuid)
    for c in sample_children[:5]:
        chain.append(str(c))
    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "ancestor_count": len(ancestors),
        "sample_parents": sample_parents,
        "sample_children": sample_children,
        "primitive_ancestors": prim_anc,
        "operator_ancestors": op_anc,
        "primitive_inputs": list(prim_inputs),
        "operators_used": list(op_anc),
        "transformation_chain": chain,
    }


def _research_section(frr: FeatureResearchRecord) -> dict[str, Any]:
    return {
        "research_uuid": frr.research_uuid,
        "feature_uuid": frr.feature_uuid,
        "research_status": frr.research_status,
        "validation_status": frr.validation_status,
        "ontology_uuid": frr.ontology_uuid,
        "transformation_uuid": frr.transformation_uuid,
        "lineage_version": frr.lineage_version,
        "compiler_version": frr.compiler_version,
        "grammar_version": frr.grammar_version,
        "evidence_json": frr.evidence_json,
        "strengths_json": frr.strengths_json,
        "weaknesses_json": frr.weaknesses_json,
        "regimes_json": frr.regimes_json,
        "failure_modes_json": frr.failure_modes_json,
        "experiment_ids": frr.experiment_ids,
        "notes": frr.notes,
        "record_source": frr.record_source,
        "created_at": frr.created_at or None,
        "updated_at": frr.updated_at or None,
    }


def _references_section(
    frr: FeatureResearchRecord,
    *,
    canonical_name: str | None,
    identity: dict[str, Any] | None,
    ontology: dict[str, Any] | None,
    compiler: dict[str, Any] | None,
) -> dict[str, Any]:
    exp = list(frr.experiment_ids or [])
    return {
        "research_uuid": frr.research_uuid,
        "feature_uuid": frr.feature_uuid,
        "canonical_name": canonical_name
        or (None if identity is None else identity.get("canonical_name")),
        "ontology_uuid": frr.ontology_uuid
        or (None if ontology is None else ontology.get("ontology_uuid")),
        "transformation_uuid": frr.transformation_uuid
        or (None if compiler is None else compiler.get("transformation_uuid")),
        "lineage_version": frr.lineage_version,
        "compiler_version": frr.compiler_version,
        "grammar_version": frr.grammar_version,
        # Phase 1: no linkage tables — always empty lists (do not invent)
        "models": [],
        "datasets": [],
        "experiments": exp,
        "research_programs": [],
        # Deep links to OP_/TR_/ONT_/FRR_ registries: reserved post Phase 1
        "deep_links_reserved": True,
    }


def _overview_summary(
    *,
    identity: dict[str, Any] | None,
    research: dict[str, Any],
    ontology: dict[str, Any] | None,
    lineage: dict[str, Any] | None,
    feature_uuid: str,
    canonical_name: str | None,
) -> str:
    """Templated metadata paragraph — not AI-generated."""
    name = (
        (identity or {}).get("display_name")
        or (identity or {}).get("canonical_name")
        or canonical_name
        or feature_uuid
    )
    status = research.get("research_status") or "UNKNOWN"
    domain = None
    if ontology:
        domain = ontology.get("domain_display") or ontology.get("domain")
    domain_bit = f" in domain {domain}" if domain else ""
    ops = (lineage or {}).get("operators_used") or (lineage or {}).get(
        "operator_ancestors"
    ) or []
    op_bit = f" Primary operator: {ops[0]}." if ops else ""
    prims = (identity or {}).get("primitive_ids") or (lineage or {}).get(
        "primitive_inputs"
    ) or []
    prim_bit = f" Primary primitive: {prims[0]}." if prims else ""
    return (
        f"{name} ({feature_uuid}) is a research feature with status {status}"
        f"{domain_bit}.{op_bit}{prim_bit}"
    )


def build_inspect_model(
    db_path: Path,
    subject: ResolvedSubject,
) -> dict[str, Any]:
    """
    Assemble Feature Inspector payload including sections_present (booleans).
    Incomplete sections → false; does not fail the call.
    """
    path = Path(db_path)
    frr = subject.research
    feat_svc = FeatureRegistryService(path)
    ont_store = OntologyStore(path)

    identity = _identity_section(feat_svc, frr.feature_uuid)
    compiler, ast_summary = _compiler_and_ast(path, frr)
    ontology = _ontology_section(ont_store, frr)
    research = _research_section(frr)
    lineage = _lineage_section(
        path,
        frr.feature_uuid,
        identity=identity,
        transformation_uuid=frr.transformation_uuid
        or (None if compiler is None else compiler.get("transformation_uuid")),
    )
    references = _references_section(
        frr,
        canonical_name=subject.canonical_name,
        identity=identity,
        ontology=ontology,
        compiler=compiler,
    )
    overview_summary = _overview_summary(
        identity=identity,
        research=research,
        ontology=ontology,
        lineage=lineage,
        feature_uuid=frr.feature_uuid,
        canonical_name=subject.canonical_name,
    )

    sections = {
        "identity": identity is not None,
        "compiler": compiler is not None,
        "ast": ast_summary is not None,
        "ontology": ontology is not None,
        "lineage": lineage is not None,
        "research": True,  # inspect always starts from FRR
        "references": True,
    }
    # ensure stable key set
    sections_present = {k: bool(sections.get(k, False)) for k in INSPECTOR_SECTIONS}

    return {
        "research_uuid": frr.research_uuid,
        "feature_uuid": frr.feature_uuid,
        "canonical_name": subject.canonical_name
        or (None if identity is None else identity.get("canonical_name")),
        "overview_summary": overview_summary,
        "sections_present": sections_present,
        "identity": identity,
        "compiler": compiler,
        "ast": ast_summary,
        "ontology": ontology,
        "lineage": lineage,
        "research": research,
        "references": references,
    }
