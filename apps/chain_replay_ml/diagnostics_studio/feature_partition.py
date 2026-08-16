"""Partition diagnostic rows into Feature Registry, Base Pipeline, and Experimental Pipeline with strict ownership invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    DATASET_SOURCE_BASE_PIPELINE,
    DATASET_SOURCE_FEATURE_REGISTRY,
    DATASET_SOURCE_OTHER_PIPELINE,
    classify_dataset_feature_source,
    dataset_base_pipeline_export_feature_names,
    dataset_registry_export_feature_names,
    other_pipeline_feature_names_from_metadata,
)

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticFeaturePartition:
    registry_rows: list[dict[str, Any]] = field(default_factory=list)
    base_pipeline_rows: list[dict[str, Any]] = field(default_factory=list)
    experimental_rows: list[dict[str, Any]] = field(default_factory=list)
    unclassified_rows: list[dict[str, Any]] = field(default_factory=list)
    unclassified_feature_names: list[str] = field(default_factory=list)
    total_count: int = 0
    is_valid: bool = True
    error_message: str | None = None

    @property
    def registry_count(self) -> int:
        return len(self.registry_rows)

    @property
    def base_pipeline_count(self) -> int:
        return len(self.base_pipeline_rows)

    @property
    def experimental_count(self) -> int:
        return len(self.experimental_rows)


def partition_diagnostic_rows(
    rows: list[dict[str, Any]],
    *,
    data_dir: str,
    dataset_metadata: dict[str, Any] | None,
) -> DiagnosticFeaturePartition:
    """Partition diagnostic comparison rows into 3 disjoint feature-source subsets.

    Enforces ownership invariants:
    1. Every selected model feature belongs to exactly one feature-source category.
    2. No feature appears in multiple tabs.
    3. No selected feature is missing from all tabs.
    4. registry_count + base_pipeline_count + experimental_count == total_model_selected_features.
    5. If a feature cannot be classified, it is recorded under unclassified_features with an error.
    """
    reg_set = dataset_registry_export_feature_names(dataset_metadata, data_dir=data_dir)
    base_set = dataset_base_pipeline_export_feature_names(dataset_metadata, data_dir=data_dir)
    exp_set = other_pipeline_feature_names_from_metadata(dataset_metadata)

    reg_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    exp_rows: list[dict[str, Any]] = []
    unclassified_rows: list[dict[str, Any]] = []
    unclassified_names: list[str] = []

    seen_features: set[str] = set()
    duplicate_features: set[str] = set()

    for r in rows:
        if not isinstance(r, dict):
            continue
        feat = str(r.get("feature") or "").strip()
        if not feat:
            continue

        if feat in seen_features:
            duplicate_features.add(feat)
        seen_features.add(feat)

        category = classify_dataset_feature_source(
            feat,
            data_dir=data_dir,
            registry_names=reg_set,
            base_pipeline_names=base_set,
        )

        row_copy = dict(r)
        if category == DATASET_SOURCE_FEATURE_REGISTRY:
            row_copy["feature_source"] = DATASET_SOURCE_FEATURE_REGISTRY
            reg_rows.append(row_copy)
        elif category == DATASET_SOURCE_BASE_PIPELINE:
            row_copy["feature_source"] = DATASET_SOURCE_BASE_PIPELINE
            base_rows.append(row_copy)
        elif category == DATASET_SOURCE_OTHER_PIPELINE:
            # Must either be in experimental candidate snapshot or pipeline features enabled
            row_copy["feature_source"] = DATASET_SOURCE_OTHER_PIPELINE
            exp_rows.append(row_copy)
        else:
            unclassified_rows.append(row_copy)
            unclassified_names.append(feat)

    total = len(rows)
    partition_sum = len(reg_rows) + len(base_rows) + len(exp_rows)
    errors: list[str] = []

    if duplicate_features:
        errors.append(f"Duplicate features found in model diagnostics: {sorted(duplicate_features)}")

    if unclassified_names:
        errors.append(
            f"{len(unclassified_names)} feature(s) could not be classified: {unclassified_names[:10]}"
        )

    if partition_sum != total:
        errors.append(
            f"Partition sum mismatch: registry({len(reg_rows)}) + base({len(base_rows)}) + "
            f"experimental({len(exp_rows)}) = {partition_sum} != total({total})"
        )

    # Disjointness check
    set_r = {str(r.get("feature")) for r in reg_rows}
    set_b = {str(r.get("feature")) for r in base_rows}
    set_e = {str(r.get("feature")) for r in exp_rows}

    overlap_rb = set_r & set_b
    overlap_re = set_r & set_e
    overlap_be = set_b & set_e

    if overlap_rb:
        errors.append(f"Overlap between Registry and Base Pipeline: {sorted(overlap_rb)}")
    if overlap_re:
        errors.append(f"Overlap between Registry and Experimental: {sorted(overlap_re)}")
    if overlap_be:
        errors.append(f"Overlap between Base Pipeline and Experimental: {sorted(overlap_be)}")

    is_valid = len(errors) == 0
    err_msg = "; ".join(errors) if errors else None
    if not is_valid:
        logger.warning("Diagnostic feature partition invariant violation: %s", err_msg)

    return DiagnosticFeaturePartition(
        registry_rows=reg_rows,
        base_pipeline_rows=base_rows,
        experimental_rows=exp_rows,
        unclassified_rows=unclassified_rows,
        unclassified_feature_names=unclassified_names,
        total_count=total,
        is_valid=is_valid,
        error_message=err_msg,
    )
