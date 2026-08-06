"""Lineage validation — cycle detection, checksum + stats refresh (Sprint 7)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from feature_intelligence.lineage import error_codes as ec
from feature_intelligence.lineage.graph import (
    ancestors_of,
    has_cycle,
)
from feature_intelligence.lineage.identity import (
    RELATIONSHIP_ID_PATTERN,
    derive_lineage_uuid,
    is_valid_lineage_node,
    normalize_object_id,
)
from feature_intelligence.lineage.models import (
    EDGE_SOURCES,
    FROZEN_RELATIONSHIP_IDS,
    LINEAGE_VERSION,
    LineageEdge,
)
from feature_intelligence.lineage.relationships import (
    EXPECTED_RELATIONSHIP_SEED_HASH,
    SEED_RELATIONSHIPS,
    compute_relationship_seed_hash,
)
from feature_intelligence.lineage.store import (
    LineageStore,
    compute_graph_checksum,
)
from feature_intelligence.registry.models import ValidationReport

_FREE_TEXT_HINT = re.compile(r"^[a-z][a-z0-9_ ]*$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def validate_edge(
    edge: LineageEdge,
    *,
    rel_ids: set[str],
    rel_active: dict[str, bool],
    pack_versions: set[str],
    strict_refs: bool = False,
    store: LineageStore | None = None,
    for_new_assignment: bool = True,
) -> tuple[list[str], list[str]]:
    failed: list[str] = []
    warnings: list[str] = []

    parent = normalize_object_id(edge.parent_object)
    child = normalize_object_id(edge.child_object)

    expected = derive_lineage_uuid(parent, child, edge.relationship_id)
    if edge.lineage_uuid != expected:
        _fail(failed, ec.LINEAGE_ID_MISMATCH)

    if parent == child:
        _fail(failed, ec.SELF_EDGE)

    if not is_valid_lineage_node(parent) or not is_valid_lineage_node(child):
        _fail(failed, ec.INVALID_OBJECT_REF)

    rid = edge.relationship_id
    if not RELATIONSHIP_ID_PATTERN.match(rid) or _FREE_TEXT_HINT.match(rid):
        if not rid.startswith("REL_"):
            _fail(failed, ec.FREE_TEXT_RELATIONSHIP)
        else:
            _fail(failed, ec.UNKNOWN_RELATIONSHIP)
    elif rid not in rel_ids:
        _fail(failed, ec.UNKNOWN_RELATIONSHIP)
    else:
        active = rel_active.get(rid, False)
        if not active and for_new_assignment:
            _fail(failed, ec.REL_INACTIVE)
        elif not active:
            warnings.append(f"inactive_rel:{rid}")

    if edge.edge_source is not None and edge.edge_source not in EDGE_SOURCES:
        _fail(failed, ec.INVALID_EDGE_SOURCE)

    if store is not None and strict_refs:
        if not store.object_exists_in_registry(parent):
            _fail(failed, ec.OBJECT_NOT_FOUND)
        if not store.object_exists_in_registry(child):
            _fail(failed, ec.OBJECT_NOT_FOUND)

    return failed, warnings


def validate_lineage(
    store: LineageStore,
    *,
    mode: str = "strict",
    strict_refs: bool = False,
) -> ValidationReport:
    """
    Validate relationship seed + edges. Always recomputes graph_checksum after
    reading edges (caller / service also writes stats).
    """
    failed: list[str] = []
    warnings: list[str] = []

    seed_hash = compute_relationship_seed_hash()
    if seed_hash != EXPECTED_RELATIONSHIP_SEED_HASH:
        _fail(failed, ec.REL_SEED_HASH_MISMATCH)

    db_rels = store.list_relationships()
    db_ids = {r.relationship_id for r in db_rels}
    expected_ids = {r.relationship_id for r in SEED_RELATIONSHIPS}
    if not expected_ids.issubset(db_ids):
        _fail(failed, ec.REL_INCOMPLETE)
    extras = db_ids - FROZEN_RELATIONSHIP_IDS
    if extras:
        _fail(failed, ec.REL_EXTRA)

    pack_versions = store.list_pack_versions()
    if LINEAGE_VERSION not in pack_versions:
        _fail(failed, ec.LINEAGE_VERSION_MISMATCH)

    rel_active = store.relationship_active_map()
    edges = store.list_edges()

    # Duplicate triple check (DB unique should prevent; still report)
    seen_triples: set[tuple[str, str, str]] = set()
    for edge in edges:
        triple = (
            edge.parent_object,
            edge.child_object,
            edge.relationship_id,
        )
        if triple in seen_triples:
            _fail(failed, ec.DUPLICATE_EDGE)
        seen_triples.add(triple)
        e_failed, e_warn = validate_edge(
            edge,
            rel_ids=db_ids,
            rel_active=rel_active,
            pack_versions=pack_versions,
            strict_refs=strict_refs or mode == "strict",
            store=store,
            for_new_assignment=False,
        )
        for code in e_failed:
            _fail(failed, code)
        warnings.extend(e_warn)

    pairs = [(e.parent_object, e.child_object) for e in edges]
    if has_cycle(pairs):
        _fail(failed, ec.CYCLE_DETECTED)

    # Orphan FEAT_* with feature_ast but no path to PR_*
    orphan_count = 0
    for feat, _tr in store.list_feature_ast_links():
        ancs = ancestors_of(feat, pairs)
        if not any(a.startswith("PR_") for a in ancs):
            orphan_count += 1
            if mode == "strict":
                _fail(failed, ec.ORPHAN_NODE)
            else:
                warnings.append(f"orphan:{feat}")

    # Checksum: compare then always refresh
    triples = [
        (e.parent_object, e.child_object, e.relationship_id) for e in edges
    ]
    recomputed = compute_graph_checksum(triples)
    pack = store.get_pack(LINEAGE_VERSION)
    stored_checksum = str(pack["graph_checksum"]) if pack else ""
    if stored_checksum and stored_checksum != recomputed:
        _fail(failed, ec.CHECKSUM_MISMATCH)
        warnings.append(
            f"checksum_was={stored_checksum};checksum_now={recomputed}"
        )

    store.update_graph_checksum(recomputed)

    validated_objects = (
        f"relationships={len(db_ids)};edges={len(edges)};"
        f"checksum={recomputed};orphans={orphan_count}"
    )

    return ValidationReport(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warnings,
        seed_hash=seed_hash,
        expected_seed_hash=EXPECTED_RELATIONSHIP_SEED_HASH,
        validated_objects=validated_objects,
        timestamp=_now(),
    )
