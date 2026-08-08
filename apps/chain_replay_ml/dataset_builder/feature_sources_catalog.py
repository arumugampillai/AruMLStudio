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
) -> dict[str, Any]:
    """Phase 1A catalogue payload for UI + analysis builds."""
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


__all__ = [
    "FEATURE_SOURCE_PIPELINE",
    "FEATURE_SOURCE_REGISTRY",
    "GENERATOR_FAMILY_LABELS",
    "GENERATOR_FAMILY_ORDER",
    "feature_sources_catalog",
    "pipeline_feature_names",
    "pipeline_feature_source",
    "pipeline_features_by_family",
    "pipeline_family_of",
    "registry_feature_names",
    "registry_feature_source",
    "registry_retired_feature_names",
]
