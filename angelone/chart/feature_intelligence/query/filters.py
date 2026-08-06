"""Field → predicate resolution and FRR filtering (Sprint 9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from feature_intelligence.lineage.graph import ancestors_of
from feature_intelligence.lineage.store import LineageStore
from feature_intelligence.ontology.models import OBJECT_TYPE_FEATURE, VocabularyRecord
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.operators.operator_service import (
    OperatorNotFoundError,
    OperatorRegistryService,
)
from feature_intelligence.query import error_codes as ec
from feature_intelligence.query.models import (
    STATUS_VALUES,
    VALIDATION_VALUES,
    QuerySpec,
    QueryToken,
)
from feature_intelligence.registry.feature_service import (
    FeatureNotFoundError,
    FeatureRegistryService,
)
from feature_intelligence.registry.service import (
    PrimitiveCatalogService,
    PrimitiveNotFoundError,
)
from feature_intelligence.research.identity import is_feat_uuid, normalize_feature_uuid
from feature_intelligence.research.models import FeatureResearchRecord
from feature_intelligence.research.store import ResearchStore


class FilterResolveError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


Predicate = Callable[[FeatureResearchRecord], bool]


@dataclass
class FilterContext:
    db_path: Path
    research: ResearchStore
    features: FeatureRegistryService
    ontology: OntologyStore
    lineage: LineageStore
    operators: OperatorRegistryService
    primitives: PrimitiveCatalogService
    edge_pairs: list[tuple[str, str]]
    # feature_uuid → OntologyRecord (FEATURE) when present
    feature_ontology: dict[str, object]
    # ontology_uuid → OntologyRecord
    ontology_by_uuid: dict[str, object]
    # canonical_name → feature_uuid
    name_to_feat: dict[str, str]
    feat_to_name: dict[str, str]
    vocab_by_id: dict[str, VocabularyRecord]
    # (vocab_type, canonical_name) → vocabulary_id for active only
    vocab_alias: dict[tuple[str, str], str]


def build_filter_context(db_path: Path) -> FilterContext:
    path = Path(db_path)
    research = ResearchStore(path)
    features = FeatureRegistryService(path)
    ontology = OntologyStore(path)
    lineage = LineageStore(path)
    operators = OperatorRegistryService(path)
    primitives = PrimitiveCatalogService(path)

    feat_ont: dict[str, object] = {}
    ont_by_uuid: dict[str, object] = {}
    for rec in ontology.list_ontology(OBJECT_TYPE_FEATURE):
        feat_ont[rec.object_id] = rec
        ont_by_uuid[rec.ontology_uuid] = rec

    name_to_feat: dict[str, str] = {}
    feat_to_name: dict[str, str] = {}
    for f in features.list_features():
        name_to_feat[f.canonical_name] = f.feature_uuid
        feat_to_name[f.feature_uuid] = f.canonical_name

    vocab_by_id: dict[str, VocabularyRecord] = {}
    vocab_alias: dict[tuple[str, str], str] = {}
    for v in ontology.list_vocabularies():
        vocab_by_id[v.vocabulary_id] = v
        if v.active:
            vocab_alias[(v.vocabulary_type, v.canonical_name)] = v.vocabulary_id

    return FilterContext(
        db_path=path,
        research=research,
        features=features,
        ontology=ontology,
        lineage=lineage,
        operators=operators,
        primitives=primitives,
        edge_pairs=lineage.edge_pairs(),
        feature_ontology=feat_ont,
        ontology_by_uuid=ont_by_uuid,
        name_to_feat=name_to_feat,
        feat_to_name=feat_to_name,
        vocab_by_id=vocab_by_id,
        vocab_alias=vocab_alias,
    )


def resolve_vocab_id(
    ctx: FilterContext,
    *,
    vocabulary_type: str,
    value: str,
) -> str:
    """Prefer vocabulary_id; accept active canonical_name alias."""
    raw = str(value)
    if raw in ctx.vocab_by_id:
        rec = ctx.vocab_by_id[raw]
        if rec.vocabulary_type != vocabulary_type:
            raise FilterResolveError(
                ec.QUERY_VOCAB_UNRESOLVED,
                f"Vocabulary {raw} is type {rec.vocabulary_type}, expected {vocabulary_type}",
            )
        if not rec.active:
            raise FilterResolveError(
                ec.QUERY_VOCAB_INACTIVE,
                f"Vocabulary inactive: {raw}",
            )
        return raw
    alias = ctx.vocab_alias.get((vocabulary_type, raw))
    if alias is None:
        raise FilterResolveError(
            ec.QUERY_VOCAB_UNRESOLVED,
            f"Cannot resolve {vocabulary_type} value: {raw!r}",
        )
    return alias


def resolve_operator_id(ctx: FilterContext, value: str) -> str:
    raw = str(value)
    if raw.startswith("OP_"):
        try:
            ctx.operators.get_by_id(raw)
            return raw
        except OperatorNotFoundError as exc:
            raise FilterResolveError(
                ec.QUERY_OPERATOR_UNRESOLVED, f"Unknown operator: {raw}"
            ) from exc

    # exact canonical_name
    try:
        return ctx.operators.get_by_name(raw).operator_id
    except OperatorNotFoundError:
        pass

    # case-insensitive / OP_{UPPER} short token
    candidates: list[str] = []
    lower = raw.lower()
    for op in ctx.operators.list_operators():
        if op.canonical_name.lower() == lower:
            candidates.append(op.operator_id)
        elif op.operator_id == f"OP_{raw.upper()}":
            candidates.append(op.operator_id)
        elif op.operator_id.upper() == f"OP_{raw.upper()}":
            candidates.append(op.operator_id)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FilterResolveError(
            ec.QUERY_OPERATOR_AMBIGUOUS,
            f"Ambiguous operator token {raw!r}: {candidates}",
        )
    raise FilterResolveError(
        ec.QUERY_OPERATOR_UNRESOLVED, f"Cannot resolve operator: {raw!r}"
    )


def resolve_primitive_id(ctx: FilterContext, value: str) -> str:
    raw = str(value)
    if raw.startswith("PR_"):
        try:
            ctx.primitives.get_primitive(raw)
            return raw
        except PrimitiveNotFoundError as exc:
            raise FilterResolveError(
                ec.QUERY_PRIMITIVE_UNRESOLVED, f"Unknown primitive: {raw}"
            ) from exc

    try:
        return ctx.primitives.get_primitive_by_name(raw).primitive_id
    except PrimitiveNotFoundError:
        pass

    candidates: list[str] = []
    lower = raw.lower()
    for p in ctx.primitives.list_primitives():
        if p.name.lower() == lower:
            candidates.append(p.primitive_id)
        elif p.primitive_id == f"PR_{raw.upper()}":
            candidates.append(p.primitive_id)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FilterResolveError(
            ec.QUERY_PRIMITIVE_AMBIGUOUS,
            f"Ambiguous primitive token {raw!r}: {candidates}",
        )
    raise FilterResolveError(
        ec.QUERY_PRIMITIVE_UNRESOLVED, f"Cannot resolve primitive: {raw!r}"
    )


def _feature_ontology(ctx: FilterContext, frr: FeatureResearchRecord):
    if frr.ontology_uuid and frr.ontology_uuid in ctx.ontology_by_uuid:
        return ctx.ontology_by_uuid[frr.ontology_uuid]
    return ctx.feature_ontology.get(frr.feature_uuid)


def _token_to_predicate(ctx: FilterContext, token: QueryToken) -> Predicate:
    field, value = token.field, token.value

    if field == "feature":
        if is_feat_uuid(value):
            feat = normalize_feature_uuid(value)
        elif value in ctx.name_to_feat:
            feat = ctx.name_to_feat[value]
        else:
            # Parse succeeds; search returns empty (locked) — use impossible id
            feat = "__missing_feature__"

        def pred_feature(frr: FeatureResearchRecord, _f: str = feat) -> bool:
            return frr.feature_uuid == _f

        return pred_feature

    if field == "domain":
        vid = resolve_vocab_id(ctx, vocabulary_type="DOMAIN", value=value)

        def pred_domain(frr: FeatureResearchRecord, _v: str = vid) -> bool:
            ont = _feature_ontology(ctx, frr)
            return ont is not None and getattr(ont, "domain", None) == _v

        return pred_domain

    if field == "signal":
        vid = resolve_vocab_id(ctx, vocabulary_type="SIGNAL_TYPE", value=value)

        def pred_signal(frr: FeatureResearchRecord, _v: str = vid) -> bool:
            ont = _feature_ontology(ctx, frr)
            if ont is None:
                return False
            signals = getattr(ont, "signal_type", None) or []
            return _v in signals

        return pred_signal

    if field == "operator":
        oid = resolve_operator_id(ctx, value)

        def pred_op(frr: FeatureResearchRecord, _o: str = oid) -> bool:
            ancs = ancestors_of(frr.feature_uuid, ctx.edge_pairs)
            if _o in ancs:
                return True
            # also check transformation path via TR node if linked
            if frr.transformation_uuid:
                tr_ancs = ancestors_of(frr.transformation_uuid, ctx.edge_pairs)
                if _o in tr_ancs:
                    return True
            return False

        return pred_op

    if field == "primitive":
        pid = resolve_primitive_id(ctx, value)

        def pred_prim(frr: FeatureResearchRecord, _p: str = pid) -> bool:
            ancs = ancestors_of(frr.feature_uuid, ctx.edge_pairs)
            return _p in ancs

        return pred_prim

    if field == "transformation":
        tr = str(value)

        def pred_tr(frr: FeatureResearchRecord, _t: str = tr) -> bool:
            return frr.transformation_uuid == _t

        return pred_tr

    if field == "status":
        status = value.upper()
        if status not in STATUS_VALUES:
            raise FilterResolveError(
                ec.QUERY_INVALID_STATUS,
                f"Invalid status {value!r}; expected EMPTY|ACTIVE|ARCHIVED",
            )

        def pred_status(frr: FeatureResearchRecord, _s: str = status) -> bool:
            return frr.research_status == _s

        return pred_status

    if field == "validation":
        val = value.lower()
        if val not in VALIDATION_VALUES:
            raise FilterResolveError(
                ec.QUERY_INVALID_VALIDATION,
                f"Invalid validation {value!r}; expected validated|pending|failed",
            )

        def pred_val(frr: FeatureResearchRecord, _v: str = val) -> bool:
            return frr.validation_status == _v

        return pred_val

    if field == "grammar":

        def pred_gram(frr: FeatureResearchRecord, _g: str = value) -> bool:
            return frr.grammar_version == _g

        return pred_gram

    if field == "compiler":

        def pred_comp(frr: FeatureResearchRecord, _c: str = value) -> bool:
            return frr.compiler_version == _c

        return pred_comp

    if field == "ontology_version":

        def pred_ov(frr: FeatureResearchRecord, _v: str = value) -> bool:
            ont = _feature_ontology(ctx, frr)
            return ont is not None and getattr(ont, "ontology_version", None) == _v

        return pred_ov

    raise FilterResolveError(ec.QUERY_UNKNOWN_FIELD, f"Unknown field: {field}")


def compile_predicates(ctx: FilterContext, spec: QuerySpec) -> list[Predicate]:
    if spec.match_all:
        return []
    return [_token_to_predicate(ctx, t) for t in spec.tokens]


def apply_filters(
    rows: list[FeatureResearchRecord],
    predicates: list[Predicate],
) -> list[FeatureResearchRecord]:
    if not predicates:
        return list(rows)
    out: list[FeatureResearchRecord] = []
    for row in rows:
        if all(p(row) for p in predicates):
            out.append(row)
    return out


def sort_frrs(rows: list[FeatureResearchRecord]) -> list[FeatureResearchRecord]:
    return sorted(
        rows,
        key=lambda r: (r.research_uuid, r.feature_uuid),
    )


def resolve_feature_token_existence(ctx: FilterContext, value: str) -> bool:
    """True if feature: value exists (for inspect/get validation)."""
    if is_feat_uuid(value):
        try:
            ctx.features.get_by_uuid(normalize_feature_uuid(value))
            return True
        except FeatureNotFoundError:
            return False
    return value in ctx.name_to_feat
