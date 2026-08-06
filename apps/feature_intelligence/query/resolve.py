"""FRR-mandatory subject resolution (Sprint 9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feature_intelligence.query import error_codes as ec
from feature_intelligence.registry.feature_service import (
    FeatureNotFoundError,
    FeatureRegistryService,
)
from feature_intelligence.research.identity import is_feat_uuid, normalize_feature_uuid
from feature_intelligence.research.models import FeatureResearchRecord
from feature_intelligence.research.store import ResearchStore


class ResolveError(LookupError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResolvedSubject:
    research: FeatureResearchRecord
    canonical_name: str | None = None


def is_frr_uuid(value: str) -> bool:
    v = str(value or "")
    return v.startswith("FRR_") and len(v) == 36


def resolve_to_frr(
    db_path: Path,
    *,
    feature_uuid: str | None = None,
    research_uuid: str | None = None,
    canonical_name: str | None = None,
) -> ResolvedSubject:
    """
    Resolve exactly one subject selector to an existing FRR.

    Missing FRR → error (never sync-create).
    """
    selectors = [
        x
        for x in (feature_uuid, research_uuid, canonical_name)
        if x is not None and str(x).strip()
    ]
    if len(selectors) == 0:
        raise ResolveError(ec.QUERY_SUBJECT_REQUIRED, "Exactly one subject selector required")
    if len(selectors) > 1:
        raise ResolveError(
            ec.QUERY_SUBJECT_AMBIGUOUS,
            "Provide exactly one of feature_uuid, research_uuid, canonical_name",
        )

    store = ResearchStore(Path(db_path))
    feat_svc = FeatureRegistryService(Path(db_path))
    name: str | None = None

    if research_uuid is not None and str(research_uuid).strip():
        ru = str(research_uuid).strip()
        if not is_frr_uuid(ru):
            raise ResolveError(ec.QUERY_NOT_FOUND, f"Invalid research_uuid: {ru}")
        row = store.get_by_uuid(ru)
        if row is None:
            raise ResolveError(ec.QUERY_FRR_MISSING, f"FRR not found: {ru}")
        try:
            feat = feat_svc.get_by_uuid(row.feature_uuid)
            name = feat.canonical_name
        except FeatureNotFoundError:
            name = None
        return ResolvedSubject(research=row, canonical_name=name)

    if feature_uuid is not None and str(feature_uuid).strip():
        fu = str(feature_uuid).strip()
        if not is_feat_uuid(fu):
            raise ResolveError(ec.QUERY_FEATURE_MISSING, f"Invalid feature_uuid: {fu}")
        fu = normalize_feature_uuid(fu)
        try:
            feat = feat_svc.get_by_uuid(fu)
            name = feat.canonical_name
        except FeatureNotFoundError as exc:
            raise ResolveError(ec.QUERY_FEATURE_MISSING, f"Feature not found: {fu}") from exc
        row = store.get_by_feature(fu)
        if row is None:
            raise ResolveError(
                ec.QUERY_FRR_MISSING,
                f"No FRR for feature {fu}; sync via research CLI first",
            )
        return ResolvedSubject(research=row, canonical_name=name)

    # canonical_name
    cn = str(canonical_name).strip()
    try:
        feat = feat_svc.get_by_name(cn)
    except FeatureNotFoundError as exc:
        raise ResolveError(ec.QUERY_FEATURE_MISSING, f"Feature not found: {cn}") from exc
    row = store.get_by_feature(feat.feature_uuid)
    if row is None:
        raise ResolveError(
            ec.QUERY_FRR_MISSING,
            f"No FRR for feature {feat.feature_uuid}; sync via research CLI first",
        )
    return ResolvedSubject(research=row, canonical_name=feat.canonical_name)
