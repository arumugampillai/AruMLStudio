"""Research sync — materialize EMPTY FRR shells from feature_registry (Sprint 8)."""

from __future__ import annotations

from feature_intelligence.compiler.models import COMPILER_VERSION
from feature_intelligence.grammar.pack import GRAMMAR_VERSION
from feature_intelligence.lineage.models import LINEAGE_VERSION
from feature_intelligence.research.identity import (
    derive_research_uuid,
    is_feat_uuid,
    normalize_feature_uuid,
)
from feature_intelligence.research.models import (
    SOURCE_SYNC,
    STATUS_EMPTY,
    VALIDATION_PENDING,
    FeatureResearchRecord,
    ResearchSyncSummary,
)
from feature_intelligence.research.store import ResearchStore


def _link_fields(
    store: ResearchStore, feature_uuid: str
) -> dict[str, str | None]:
    """Resolve optional cross-registry links when available; never invent ids."""
    ont = store.resolve_ontology_uuid(feature_uuid)
    tr = store.resolve_transformation_uuid(feature_uuid)
    lineage_ver: str | None = None
    known_lin = store.known_lineage_versions()
    if LINEAGE_VERSION in known_lin:
        lineage_ver = LINEAGE_VERSION
    compiler_ver: str | None = COMPILER_VERSION if tr is not None else None
    grammar_ver: str | None = GRAMMAR_VERSION if tr is not None else None
    return {
        "ontology_uuid": ont,
        "transformation_uuid": tr,
        "lineage_version": lineage_ver,
        "compiler_version": compiler_ver,
        "grammar_version": grammar_ver,
    }


def _merged_links(
    existing: FeatureResearchRecord, links: dict[str, str | None]
) -> dict[str, str | None]:
    """Prefer newly resolved non-null links; keep existing otherwise."""
    return {
        "ontology_uuid": links["ontology_uuid"] or existing.ontology_uuid,
        "transformation_uuid": (
            links["transformation_uuid"] or existing.transformation_uuid
        ),
        "lineage_version": links["lineage_version"] or existing.lineage_version,
        "compiler_version": links["compiler_version"] or existing.compiler_version,
        "grammar_version": links["grammar_version"] or existing.grammar_version,
    }


def _same_links(
    existing: FeatureResearchRecord, merged: dict[str, str | None]
) -> bool:
    return (
        existing.ontology_uuid == merged["ontology_uuid"]
        and existing.transformation_uuid == merged["transformation_uuid"]
        and existing.lineage_version == merged["lineage_version"]
        and existing.compiler_version == merged["compiler_version"]
        and existing.grammar_version == merged["grammar_version"]
    )


def sync_research(
    store: ResearchStore,
    *,
    feature_uuid: str | None = None,
    refresh_checksum: bool = True,
) -> ResearchSyncSummary:
    """
    Create missing EMPTY/pending FRR shells; optionally refresh resolvable links.

    Does not compute evidence. Does not write statistics (caller does after success).
    """
    created = updated = unchanged = skipped = 0

    if feature_uuid is not None:
        feat = normalize_feature_uuid(feature_uuid)
        if not is_feat_uuid(feat) or not store.feature_exists(feat):
            return ResearchSyncSummary(
                created=0, updated=0, unchanged=0, skipped=1
            )
        targets = [feat]
    else:
        targets = store.list_feature_uuids()

    for feat in targets:
        if not is_feat_uuid(feat):
            skipped += 1
            continue
        links = _link_fields(store, feat)
        existing = store.get_by_feature(feat)
        if existing is None:
            store.upsert_record(
                FeatureResearchRecord(
                    research_uuid=derive_research_uuid(feat),
                    feature_uuid=feat,
                    ontology_uuid=links["ontology_uuid"],
                    transformation_uuid=links["transformation_uuid"],
                    lineage_version=links["lineage_version"],
                    compiler_version=links["compiler_version"],
                    grammar_version=links["grammar_version"],
                    research_status=STATUS_EMPTY,
                    validation_status=VALIDATION_PENDING,
                    evidence_json=None,
                    strengths_json=None,
                    weaknesses_json=None,
                    regimes_json=None,
                    failure_modes_json=None,
                    experiment_ids=None,
                    notes=None,
                    record_source=SOURCE_SYNC,
                )
            )
            created += 1
            continue

        merged = _merged_links(existing, links)
        if _same_links(existing, merged):
            unchanged += 1
            continue

        store.upsert_record(
            FeatureResearchRecord(
                research_uuid=existing.research_uuid,
                feature_uuid=existing.feature_uuid,
                ontology_uuid=merged["ontology_uuid"],
                transformation_uuid=merged["transformation_uuid"],
                lineage_version=merged["lineage_version"],
                compiler_version=merged["compiler_version"],
                grammar_version=merged["grammar_version"],
                research_status=existing.research_status,
                validation_status=existing.validation_status,
                evidence_json=existing.evidence_json,
                strengths_json=existing.strengths_json,
                weaknesses_json=existing.weaknesses_json,
                regimes_json=existing.regimes_json,
                failure_modes_json=existing.failure_modes_json,
                experiment_ids=existing.experiment_ids,
                notes=existing.notes,
                record_source=existing.record_source or SOURCE_SYNC,
                created_at=existing.created_at,
            )
        )
        updated += 1

    if refresh_checksum:
        store.recompute_and_store_checksum()

    return ResearchSyncSummary(
        created=created,
        updated=updated,
        unchanged=unchanged,
        skipped=skipped,
    )
