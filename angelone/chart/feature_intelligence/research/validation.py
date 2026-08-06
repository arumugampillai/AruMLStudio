"""Research validation — coverage, identity, FKs; checksum refresh (Sprint 8)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.research import error_codes as ec
from feature_intelligence.research.identity import (
    ONT_PATTERN,
    TR_PATTERN,
    derive_research_uuid,
    is_feat_uuid,
    normalize_feature_uuid,
)
from feature_intelligence.research.models import (
    RECORD_SOURCES,
    RESEARCH_STATUSES,
    RESEARCH_VERSION,
    VALIDATION_STATUSES,
    FeatureResearchRecord,
)
from feature_intelligence.research.store import (
    ResearchStore,
    compute_research_checksum,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def validate_record(
    rec: FeatureResearchRecord,
    *,
    store: ResearchStore,
    strict_refs: bool = False,
) -> tuple[list[str], list[str]]:
    failed: list[str] = []
    warnings: list[str] = []

    feat = normalize_feature_uuid(rec.feature_uuid)
    if not is_feat_uuid(feat):
        _fail(failed, ec.INVALID_FEATURE_REF)
        if feat.startswith(("PR_", "OP_", "TR_")):
            _fail(failed, ec.PR_OP_TR_FRR_FORBIDDEN)

    expected = derive_research_uuid(feat)
    if rec.research_uuid != expected:
        _fail(failed, ec.FRR_ID_MISMATCH)

    if not store.feature_exists(feat):
        _fail(failed, ec.ORPHAN_FRR)

    if rec.research_status not in RESEARCH_STATUSES:
        _fail(failed, ec.INVALID_RESEARCH_STATUS)

    if rec.validation_status not in VALIDATION_STATUSES:
        _fail(failed, ec.INVALID_VALIDATION_STATUS)

    if rec.record_source is not None and rec.record_source not in RECORD_SOURCES:
        _fail(failed, ec.INVALID_RECORD_SOURCE)

    if rec.ontology_uuid is not None:
        if not ONT_PATTERN.match(rec.ontology_uuid):
            _fail(failed, ec.INVALID_ONTOLOGY_REF)
        elif not store.ontology_exists(rec.ontology_uuid):
            _fail(failed, ec.INVALID_ONTOLOGY_REF)

    if rec.transformation_uuid is not None:
        if not TR_PATTERN.match(rec.transformation_uuid):
            _fail(failed, ec.INVALID_TRANSFORMATION_REF)
        elif not store.transformation_exists(rec.transformation_uuid):
            _fail(failed, ec.INVALID_TRANSFORMATION_REF)

    if rec.experiment_ids is not None:
        if not isinstance(rec.experiment_ids, list):
            _fail(failed, ec.INVALID_EXPERIMENT_IDS)
        else:
            try:
                json.dumps(rec.experiment_ids)
            except (TypeError, ValueError):
                _fail(failed, ec.INVALID_EXPERIMENT_IDS)

    if strict_refs:
        if rec.lineage_version is not None:
            known = store.known_lineage_versions()
            if rec.lineage_version not in known:
                warnings.append(f"unknown_lineage_version:{rec.lineage_version}")
        if rec.compiler_version is not None:
            if rec.compiler_version not in store.known_compiler_versions():
                warnings.append(f"unknown_compiler_version:{rec.compiler_version}")
        if rec.grammar_version is not None:
            if rec.grammar_version not in store.known_grammar_versions():
                warnings.append(f"unknown_grammar_version:{rec.grammar_version}")

    return failed, warnings


def validate_research(
    store: ResearchStore,
    *,
    mode: str = "strict",
    strict_refs: bool = False,
    strict_coverage: bool = False,
) -> ValidationReport:
    """
    Validate FRR coverage / identity / FKs.

    Always recomputes and writes research_registry.checksum.
    Does NOT create missing FRRs. Does NOT write statistics (caller does).
    """
    failed: list[str] = []
    warnings: list[str] = []

    pack_versions = store.list_pack_versions()
    if RESEARCH_VERSION not in pack_versions:
        _fail(failed, ec.RESEARCH_VERSION_MISMATCH)

    records = store.list_records()
    features = store.list_feature_uuids()
    feat_set = set(features)
    frr_by_feat: dict[str, list[FeatureResearchRecord]] = {}
    for rec in records:
        frr_by_feat.setdefault(rec.feature_uuid, []).append(rec)

    for feat, group in frr_by_feat.items():
        if len(group) > 1:
            _fail(failed, ec.DUPLICATE_FRR)

    missing = [f for f in features if f not in frr_by_feat]
    if mode == "strict" or strict_coverage:
        for _ in missing:
            _fail(failed, ec.MISSING_FRR)
            break  # one code is enough; count in validated_objects
        if missing and ec.MISSING_FRR not in failed:
            _fail(failed, ec.MISSING_FRR)
    else:
        if missing:
            warnings.append(f"missing_frr_count={len(missing)}")

    for rec in records:
        e_fail, e_warn = validate_record(
            rec, store=store, strict_refs=strict_refs or mode == "strict"
        )
        for code in e_fail:
            _fail(failed, code)
        warnings.extend(e_warn)

    prior = store.get_pack()
    prior_checksum = str(prior["checksum"]) if prior else ""
    recomputed = compute_research_checksum(records)
    if prior_checksum and prior_checksum != recomputed:
        # Diagnostic before rewrite — not a hard fail once we refresh
        warnings.append(f"{ec.CHECKSUM_MISMATCH}:before_refresh")

    new_checksum = store.recompute_and_store_checksum()

    return ValidationReport(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warnings,
        seed_hash=new_checksum,
        expected_seed_hash=new_checksum,
        validated_objects=(
            f"features={len(features)};frr={len(records)};"
            f"missing={len(missing)}"
        ),
        timestamp=_now(),
    )
