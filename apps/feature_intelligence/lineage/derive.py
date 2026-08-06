"""Derive lineage edges from Sprint 5 AST / transformation / feature_ast (Sprint 7).

Read-only w.r.t. compiler tables. Stores relationship edges only — never copies AST.
"""

from __future__ import annotations

from feature_intelligence.lineage.graph import has_cycle, would_introduce_cycle
from feature_intelligence.lineage.identity import (
    derive_lineage_uuid,
    normalize_object_id,
)
from feature_intelligence.lineage.models import (
    REL_DEPENDS_ON,
    REL_DERIVED_FROM,
    REL_GENERATED_BY,
    REL_INPUT_TO,
    REL_USES,
    DeriveResult,
    LineageEdge,
)
from feature_intelligence.lineage.store import LineageStore


def collect_derive_triples(
    store: LineageStore,
    *,
    transformation_uuid: str | None = None,
    feature_uuid: str | None = None,
    include_closure: bool = True,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Extract unique (parent, child, relationship_id) triples from Sprint 5 sources.

    D1 primitive → TR REL_INPUT_TO
    D2 operator → TR REL_USES
    D3 feature → TR REL_DEPENDS_ON
    D4 TR → FEAT REL_GENERATED_BY (feature_ast)
    D5 PR → FEAT REL_DERIVED_FROM (closure, default on)
    """
    warnings: list[str] = []
    triples: set[tuple[str, str, str]] = set()

    feat_filter = (
        normalize_object_id(feature_uuid) if feature_uuid else None
    )
    tr_filter = (
        normalize_object_id(transformation_uuid) if transformation_uuid else None
    )

    # Restrict transformations when filtering by feature
    feature_links = store.list_feature_ast_links()
    if feat_filter:
        feature_links = [(f, t) for f, t in feature_links if f == feat_filter]
        if tr_filter is None:
            allowed_tr = {t for _, t in feature_links}
        else:
            allowed_tr = {tr_filter} & {t for _, t in feature_links}
    elif tr_filter:
        allowed_tr = {tr_filter}
        feature_links = [(f, t) for f, t in feature_links if t == tr_filter]
    else:
        allowed_tr = None  # all

    nodes = store.list_ast_node_refs(tr_filter)
    # D1–D3
    tr_primitives: dict[str, set[str]] = {}
    for n in nodes:
        tr = normalize_object_id(str(n["transformation_uuid"]))
        if allowed_tr is not None and tr not in allowed_tr:
            continue
        ntype = str(n["node_type"])
        if ntype == "primitive" and n["primitive_id"]:
            pr = normalize_object_id(str(n["primitive_id"]))
            triples.add((pr, tr, REL_INPUT_TO))
            tr_primitives.setdefault(tr, set()).add(pr)
        elif ntype == "operator" and n["operator_id"]:
            op = normalize_object_id(str(n["operator_id"]))
            triples.add((op, tr, REL_USES))
        elif ntype == "feature" and n["feature_uuid"]:
            feat = normalize_object_id(str(n["feature_uuid"]))
            triples.add((feat, tr, REL_DEPENDS_ON))
        # literals / list → no edges

    # D4
    tr_features: dict[str, set[str]] = {}
    for feat, tr in feature_links:
        if allowed_tr is not None and tr not in allowed_tr:
            continue
        triples.add((tr, feat, REL_GENERATED_BY))
        tr_features.setdefault(tr, set()).add(feat)

    # D5 closure
    if include_closure:
        for tr, feats in tr_features.items():
            prims = tr_primitives.get(tr)
            if not prims:
                # primitives may come from AST even if we already scanned;
                # ensure we have them for this TR
                if tr not in tr_primitives:
                    for n in store.list_ast_node_refs(tr):
                        if n["node_type"] == "primitive" and n["primitive_id"]:
                            tr_primitives.setdefault(tr, set()).add(
                                normalize_object_id(str(n["primitive_id"]))
                            )
                prims = tr_primitives.get(tr, set())
            for pr in prims:
                for feat in feats:
                    triples.add((pr, feat, REL_DERIVED_FROM))

    ordered = sorted(triples, key=lambda t: (t[0], t[1], t[2]))
    return ordered, warnings


def derive_lineage(
    store: LineageStore,
    *,
    transformation_uuid: str | None = None,
    feature_uuid: str | None = None,
    include_closure: bool = True,
    strict_refs: bool = False,
    refresh_checksum: bool = True,
) -> DeriveResult:
    """Idempotent upsert of derived edges; edge_source=DERIVE."""
    triples, warnings = collect_derive_triples(
        store,
        transformation_uuid=transformation_uuid,
        feature_uuid=feature_uuid,
        include_closure=include_closure,
    )

    # Existing pairs for cycle check (include current + pending)
    existing_pairs = store.edge_pairs()
    upserted = 0
    skipped = 0
    edges_out: list[LineageEdge] = []

    # Apply in stable order; reject cycle-introducing edges
    applied_pairs = list(existing_pairs)
    for parent, child, rel in triples:
        if strict_refs:
            if not store.object_exists_in_registry(parent):
                warnings.append(f"missing_ref:{parent}")
                skipped += 1
                continue
            if not store.object_exists_in_registry(child):
                warnings.append(f"missing_ref:{child}")
                skipped += 1
                continue
        if would_introduce_cycle(applied_pairs, parent, child):
            # Same edge already present is fine; new cycle is not
            existing = store.get_edge_by_triple(parent, child, rel)
            if existing is not None:
                edges_out.append(existing)
                skipped += 1
                continue
            warnings.append(f"cycle_skip:{parent}->{child}:{rel}")
            skipped += 1
            continue

        edge = LineageEdge(
            lineage_uuid=derive_lineage_uuid(parent, child, rel),
            parent_object=parent,
            child_object=child,
            relationship_id=rel,
            edge_source="DERIVE",
        )
        stored = store.upsert_edge(edge)
        edges_out.append(stored)
        upserted += 1
        if (parent, child) not in applied_pairs:
            applied_pairs.append((parent, child))

    # Safety: never leave a cyclic graph after derive
    if has_cycle(store.edge_pairs()):
        warnings.append("cycle_detected_after_derive")

    if refresh_checksum:
        store.recompute_and_store_graph_checksum()

    return DeriveResult(
        upserted=upserted,
        skipped=skipped,
        warnings=warnings,
        edges=edges_out,
    )
