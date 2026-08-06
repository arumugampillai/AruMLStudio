"""Platform summary + search-hit enrichment (read-only, Sprint 9+)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feature_intelligence.compiler.models import COMPILER_VERSION
from feature_intelligence.compiler.store import CompilerStore
from feature_intelligence.grammar.pack import GRAMMAR_PACK_VERSION, GRAMMAR_VERSION
from feature_intelligence.lineage.graph import ancestors_of
from feature_intelligence.ontology.models import OBJECT_TYPES, ONTOLOGY_VERSION
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.operators.operator_service import OperatorRegistryService
from feature_intelligence.query.filters import FilterContext, _feature_ontology
from feature_intelligence.query.models import SearchHit
from feature_intelligence.registry.feature_service import FeatureRegistryService
from feature_intelligence.registry.service import PrimitiveCatalogService
from feature_intelligence.research.models import FeatureResearchRecord
from feature_intelligence.research.store import ResearchStore


def _domain_label(ctx: FilterContext, frr: FeatureResearchRecord) -> str | None:
    ont = _feature_ontology(ctx, frr)
    if ont is None:
        return None
    domain = getattr(ont, "domain", None)
    if not domain:
        return None
    vocab = ctx.vocab_by_id.get(str(domain))
    if vocab is not None and vocab.display_name:
        return str(vocab.display_name)
    return str(domain)


def _ontology_version(ctx: FilterContext, frr: FeatureResearchRecord) -> str | None:
    ont = _feature_ontology(ctx, frr)
    if ont is None:
        return None
    ver = getattr(ont, "ontology_version", None)
    return None if ver is None else str(ver)


def _primary_primitive(
    feat_prims: dict[str, tuple[str, ...]], feature_uuid: str
) -> str | None:
    prims = feat_prims.get(feature_uuid) or ()
    return prims[0] if prims else None


def _primary_operator(
    ctx: FilterContext, feature_uuid: str
) -> str | None:
    """First OP_* ancestor in lineage, if any."""
    try:
        ancestors = ancestors_of(feature_uuid, ctx.edge_pairs)
    except Exception:
        return None
    for a in ancestors:
        if str(a).startswith("OP_"):
            return str(a)
    return None


def enrich_search_hit(
    ctx: FilterContext,
    frr: FeatureResearchRecord,
    *,
    feat_prims: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Build an enriched SearchHit dict from an FRR + filter context."""
    prims_map = feat_prims if feat_prims is not None else {}
    return SearchHit(
        research_uuid=frr.research_uuid,
        feature_uuid=frr.feature_uuid,
        canonical_name=ctx.feat_to_name.get(frr.feature_uuid),
        research_status=frr.research_status,
        validation_status=frr.validation_status,
        ontology_uuid=frr.ontology_uuid,
        transformation_uuid=frr.transformation_uuid,
        domain=_domain_label(ctx, frr),
        primary_operator=_primary_operator(ctx, frr.feature_uuid),
        primary_primitive=_primary_primitive(prims_map, frr.feature_uuid),
        compiler_version=frr.compiler_version,
        grammar_version=frr.grammar_version,
        ontology_version=_ontology_version(ctx, frr),
    ).to_dict()


def build_feat_prims_map(ctx: FilterContext) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for feat in ctx.features.list_features():
        out[feat.feature_uuid] = tuple(feat.primitive_ids)
    return out


def enrich_search_hits(
    ctx: FilterContext, rows: list[FeatureResearchRecord]
) -> list[dict[str, Any]]:
    prims = build_feat_prims_map(ctx)
    return [enrich_search_hit(ctx, r, feat_prims=prims) for r in rows]


def empty_references_payload(
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
) -> dict[str, Any]:
    """Phase 1 stub: no model/dataset/experiment/program linkage tables."""
    return {
        "feature_uuid": feature_uuid,
        "research_uuid": research_uuid,
        "models": [],
        "datasets": [],
        "experiments": [],
        "research_programs": [],
        "deep_links_reserved": True,
        "note": "Phase 1: linkage tables not populated",
    }


def build_platform_summary(db_path: Path) -> dict[str, Any]:
    """
    Read-only dashboard counts + registry pack versions.

    Uses existing stores/services only — no writes.
    """
    path = Path(db_path)
    primitives = PrimitiveCatalogService(path)
    features = FeatureRegistryService(path)
    operators = OperatorRegistryService(path)
    compiler = CompilerStore(path)
    ontology = OntologyStore(path)
    research = ResearchStore(path)

    ontology_total = 0
    for ot in sorted(OBJECT_TYPES):
        ontology_total += ontology.count_ontology(ot)

    # Prefer live registry pack constants; fall back to DB when present
    ont_versions = sorted(ontology.ontology_versions())
    ontology_version = ont_versions[-1] if ont_versions else ONTOLOGY_VERSION

    grammar_versions = sorted(research.known_grammar_versions())
    compiler_versions = sorted(research.known_compiler_versions())

    return {
        "counts": {
            "primitives": len(primitives.list_primitives()),
            "features": len(features.list_features()),
            "operators": len(operators.list_operators()),
            "transformations": compiler.count_transformations(),
            "ontology_records": ontology_total,
            "research_records": research.count_records(),
        },
        "versions": {
            "compiler_version": (
                compiler_versions[-1] if compiler_versions else COMPILER_VERSION
            ),
            "grammar_version": (
                grammar_versions[-1] if grammar_versions else GRAMMAR_VERSION
            ),
            "grammar_pack_version": GRAMMAR_PACK_VERSION,
            "ontology_version": ontology_version,
        },
        "read_only": True,
    }
