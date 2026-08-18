"""Feature sources catalogues for Auto Feature Transformation (Phase 1A).

Terminology: everything is a *feature source* (Registry Features, Pipeline Features).
Do not label these as "libraries".
"""

from __future__ import annotations

from typing import Any

from .feature_domains import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    features_by_domain,
    validate_domain_coverage,
)
from .feature_migration import PIPELINE_OWNED_FEATURES, PIPELINE_OWNED_GENERATORS
from .feature_ownership import canonical_registry_features

FEATURE_SOURCE_REGISTRY = "registry"
FEATURE_SOURCE_PIPELINE = "pipeline"

GENERATOR_FAMILY_ORDER: tuple[str, ...] = (
    "interaction",
    "difference",
    "return",
    "lag",
    "rolling",
    "derived",
)

GENERATOR_FAMILY_LABELS: dict[str, str] = {
    "interaction": "Interaction",
    "difference": "Difference",
    "return": "Return",
    "lag": "Lag",
    "rolling": "Rolling",
    "derived": "Derived",
}

_GENERATOR_TO_FAMILY: dict[str, str] = {
    "interaction": "interaction",
    "difference": "difference",
    "difference_clip": "difference",
    "return": "return",
    "lag": "lag",
    "rolling_zscore": "rolling",
    "rolling_ohlc": "rolling",
    "rolling_statistics": "rolling",
    "derived": "derived",
}


def registry_retired_feature_names(data_dir: str | None = None) -> frozenset[str]:
    if not data_dir:
        return frozenset()
    from .feature_registry_store import disabled_registry_feature_names, load_store

    return frozenset(disabled_registry_feature_names(load_store(data_dir)))


def registry_feature_names(*, data_dir: str | None = None) -> list[str]:
    names = sorted(canonical_registry_features())
    retired = registry_retired_feature_names(data_dir)
    if retired:
        names = [n for n in names if n not in retired]
    return names


def get_active_feature_names(
    data_dir: str,
    *,
    feature_project_id: str | None = None,
) -> list[str]:
    """Authoritative active (non-retired) Feature Registry names for transformations."""
    if feature_project_id:
        from .feature_project_organization import project_registry_feature_source

        src = project_registry_feature_source(
            data_dir=data_dir,
            project_id=str(feature_project_id).strip().lower(),
        )
        return list(src.get("features") or [])
    return registry_feature_names(data_dir=data_dir)


def transformation_forbidden_feature_names(data_dir: str | None = None) -> frozenset[str]:
    """Registry-retired, pipeline-retired, and static retired — never transform sources or outputs."""
    from .feature_migration import RETIRED_FEATURES

    skip: set[str] = set(RETIRED_FEATURES)
    if data_dir:
        from .pipeline_features_prefs import load_retired_pipeline_features

        skip |= set(load_retired_pipeline_features(data_dir))
        skip |= set(registry_retired_feature_names(data_dir))
    return frozenset(skip)


def pipeline_feature_names(
    *,
    data_dir: str | None = None,
    retired: frozenset[str] | None = None,
) -> list[str]:
    from .pipeline_features_prefs import active_pipeline_feature_names

    return active_pipeline_feature_names(
        sorted(PIPELINE_OWNED_FEATURES),
        data_dir=data_dir,
        retired=retired,
    )


def pipeline_family_of(feature: str) -> str:
    gen = PIPELINE_OWNED_GENERATORS.get(str(feature or "").strip(), "derived")
    return _GENERATOR_TO_FAMILY.get(gen, "derived")


def pipeline_features_by_family(
    *,
    data_dir: str | None = None,
    retired: frozenset[str] | None = None,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {fid: [] for fid in GENERATOR_FAMILY_ORDER}
    for name in pipeline_feature_names(data_dir=data_dir, retired=retired):
        fam = pipeline_family_of(name)
        out.setdefault(fam, []).append(name)
    for fam in out:
        out[fam].sort()
    return out


def registry_feature_source(*, data_dir: str | None = None) -> dict[str, Any]:
    retired = registry_retired_feature_names(data_dir)
    active = set(registry_feature_names(data_dir=data_dir))
    by_domain = features_by_domain()
    coverage = validate_domain_coverage(expected_total=206)
    groups = [
        {
            "id": domain_id,
            "label": DOMAIN_LABELS[domain_id],
            "count": len([f for f in by_domain.get(domain_id, []) if f in active]),
            "features": [f for f in by_domain.get(domain_id, []) if f in active],
        }
        for domain_id in DOMAIN_ORDER
        if any(f in active for f in (by_domain.get(domain_id) or []))
    ]
    names = sorted(active)
    canonical_total = len(canonical_registry_features())
    return {
        "id": FEATURE_SOURCE_REGISTRY,
        "label": "Registry Features",
        "description": "Canonical Feature Registry",
        "total": len(names),
        "expected_total": 206,
        "retired_count": len(retired),
        "ready": bool(coverage.get("ok")) and canonical_total == 206,
        "groups": groups,
        "features": names,
    }


def pipeline_feature_source(
    *,
    data_dir: str | None = None,
    retired: frozenset[str] | None = None,
) -> dict[str, Any]:
    by_family = pipeline_features_by_family(data_dir=data_dir, retired=retired)
    names = pipeline_feature_names(data_dir=data_dir, retired=retired)
    groups = [
        {
            "id": fam_id,
            "label": GENERATOR_FAMILY_LABELS.get(fam_id, fam_id.title()),
            "count": len(by_family.get(fam_id, [])),
            "features": list(by_family.get(fam_id, [])),
        }
        for fam_id in GENERATOR_FAMILY_ORDER
        if by_family.get(fam_id)
    ]
    # Any unexpected generator families last.
    for fam_id, feats in by_family.items():
        if fam_id in GENERATOR_FAMILY_ORDER or not feats:
            continue
        groups.append({
            "id": fam_id,
            "label": GENERATOR_FAMILY_LABELS.get(fam_id, fam_id.replace("_", " ").title()),
            "count": len(feats),
            "features": list(feats),
        })
    retired_n = 0
    if retired is not None:
        retired_n = len(retired)
    elif data_dir:
        from .pipeline_features_prefs import load_retired_pipeline_features

        retired_n = len(load_retired_pipeline_features(data_dir))
    return {
        "id": FEATURE_SOURCE_PIPELINE,
        "label": "Pipeline Features",
        "description": "Pipeline-owned / migrated features",
        "total": len(names),
        "expected_total": 212,
        "retired_count": retired_n,
        "ready": len(names) > 0,
        "groups": groups,
        "features": names,
    }


def feature_sources_catalog(
    *,
    data_dir: str | None = None,
    retired: frozenset[str] | None = None,
    feature_project_id: str | None = None,
) -> dict[str, Any]:
    """Phase 1A catalogue payload for UI + analysis builds."""
    if data_dir and feature_project_id:
        from .feature_project_organization import project_registry_feature_source

        registry = project_registry_feature_source(
            data_dir=data_dir,
            project_id=str(feature_project_id).strip().lower(),
        )
    else:
        registry = registry_feature_source(data_dir=data_dir)
    pipeline = pipeline_feature_source(data_dir=data_dir, retired=retired)
    return {
        "version": 1,
        "phase": "1A",
        "sources": [registry, pipeline],
        "totals": {
            FEATURE_SOURCE_REGISTRY: int(registry["total"]),
            FEATURE_SOURCE_PIPELINE: int(pipeline["total"]),
            "union": int(registry["total"]) + int(pipeline["total"]),
        },
    }


DATASET_SOURCE_FEATURE_REGISTRY = "feature_registry"
DATASET_SOURCE_BASE_PIPELINE = "base_pipeline"
DATASET_SOURCE_OTHER_PIPELINE = "other_pipeline"

_DATASET_SOURCE_LABELS: dict[str, str] = {
    DATASET_SOURCE_FEATURE_REGISTRY: "Feature Registry",
    DATASET_SOURCE_BASE_PIPELINE: "Base Pipeline",
    DATASET_SOURCE_OTHER_PIPELINE: "Other Pipeline",
}


def dataset_feature_source_label(bucket: str) -> str:
    return _DATASET_SOURCE_LABELS.get(str(bucket or "").strip(), "Other Pipeline")


def _matches_pipeline_catalogue_feature(name: str, catalogue: frozenset[str]) -> bool:
    """True when ``name`` is a catalogue column or a transform output of one."""
    n = str(name or "").strip()
    if not n or not catalogue:
        return False
    if n in catalogue:
        return True
    for base in catalogue:
        b = str(base).strip()
        if b and n.startswith(f"{b}_"):
            return True
    return False


def base_pipeline_feature_names(data_dir: str) -> frozenset[str]:
    """Active Pipeline Features catalogue (approved pool / PIPELINE_OWNED minus retired)."""
    names = pipeline_feature_names(data_dir=data_dir)
    if names:
        return frozenset(names)
    from .pipeline_registry_store import ensure_default_existing_pipeline, is_base_pipeline_record

    doc = ensure_default_existing_pipeline(data_dir)
    stored: set[str] = set()
    for rec in (doc.get("pipelines") or {}).values():
        if not isinstance(rec, dict) or not is_base_pipeline_record(rec):
            continue
        stored.update(
            str(n).strip() for n in (rec.get("candidate_features") or []) if str(n).strip()
        )
        break
    return frozenset(stored)


def dataset_registry_export_feature_names(
    metadata: dict[str, Any] | None,
    *,
    data_dir: str,
) -> frozenset[str]:
    """Selected registry export names for a dataset (snapshot at build, else current prefs)."""
    if isinstance(metadata, dict):
        snap = metadata.get("registry_export_features") or metadata.get("registry_features")
        if isinstance(snap, list) and snap:
            return frozenset(str(n).strip() for n in snap if str(n).strip())
    from .registry_features_prefs import resolve_registry_export_features

    return resolve_registry_export_features(data_dir)


def dataset_base_pipeline_export_feature_names(
    metadata: dict[str, Any] | None,
    *,
    data_dir: str,
) -> frozenset[str]:
    """Base pipeline catalogue snapshot at build (else current active catalogue)."""
    if isinstance(metadata, dict):
        snap = metadata.get("base_pipeline_export_features") or metadata.get("base_pipeline_features")
        if isinstance(snap, list) and snap:
            return frozenset(str(n).strip() for n in snap if str(n).strip())
    return base_pipeline_feature_names(data_dir)


def other_pipeline_feature_names_from_metadata(
    metadata: dict[str, Any] | None,
) -> frozenset[str]:
    """Experimental pipeline candidate snapshot stored on a built dataset."""
    if not isinstance(metadata, dict):
        return frozenset()
    prov = metadata.get("pipeline_provenance")
    if isinstance(prov, dict):
        cand = prov.get("candidate_features")
        if isinstance(cand, list) and cand:
            return frozenset(str(n).strip() for n in cand if str(n).strip())
    other = metadata.get("other_pipeline_features") or metadata.get("experimental_features")
    if isinstance(other, list) and other:
        return frozenset(str(n).strip() for n in other if str(n).strip())
    return frozenset()


def classify_dataset_feature_source(
    feature: str,
    *,
    data_dir: str,
    registry_names: frozenset[str] | None = None,
    base_pipeline_names: frozenset[str] | None = None,
) -> str:
    """Partition one dataset column into Registry / Base Pipeline / Other Pipeline.

    Order (each step only when not already classified):
    1. Feature Registry — exact name in the *selected* registry export set
    2. Base Pipeline — approved Pipeline Features catalogue name or derived output
    3. Other Pipeline — experimental pipeline candidates and remainder
    """
    name = str(feature or "").strip()
    if not name:
        return DATASET_SOURCE_OTHER_PIPELINE
    registry = (
        registry_names
        if registry_names is not None
        else dataset_registry_export_feature_names(None, data_dir=data_dir)
    )
    if name in registry:
        return DATASET_SOURCE_FEATURE_REGISTRY
    base = (
        base_pipeline_names
        if base_pipeline_names is not None
        else base_pipeline_feature_names(data_dir)
    )
    from .feature_migration import is_pipeline_owned

    if is_pipeline_owned(name) or _matches_pipeline_catalogue_feature(name, base):
        return DATASET_SOURCE_BASE_PIPELINE
    return DATASET_SOURCE_OTHER_PIPELINE


__all__ = [
    "DATASET_SOURCE_BASE_PIPELINE",
    "DATASET_SOURCE_FEATURE_REGISTRY",
    "DATASET_SOURCE_OTHER_PIPELINE",
    "FEATURE_SOURCE_PIPELINE",
    "FEATURE_SOURCE_REGISTRY",
    "GENERATOR_FAMILY_LABELS",
    "GENERATOR_FAMILY_ORDER",
    "base_pipeline_feature_names",
    "classify_dataset_feature_source",
    "dataset_feature_source_label",
    "dataset_registry_export_feature_names",
    "dataset_base_pipeline_export_feature_names",
    "other_pipeline_feature_names_from_metadata",
    "feature_sources_catalog",
    "pipeline_feature_names",
    "pipeline_feature_source",
    "pipeline_features_by_family",
    "pipeline_family_of",
    "get_active_feature_names",
    "registry_feature_names",
    "registry_feature_source",
    "registry_retired_feature_names",
    "transformation_forbidden_feature_names",
]
