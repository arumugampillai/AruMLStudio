"""Dataset manifest — frozen policy snapshot at build time."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.schema_registry import schema_registry_hash

from .registry import FeaturePolicyRegistry, registry_version_info
from .types import DEFAULT_GAP_MAX_SEC, FEATURE_POLICY_VERSION, WarmupMode


def build_dataset_policy_manifest(
    registry: FeaturePolicyRegistry,
    *,
    sampling_interval_sec: float,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    warmup_mode: WarmupMode = WarmupMode.SAMPLE_COUNT,
    selected_features: list[str] | None = None,
    build_stats: dict[str, Any] | None = None,
    health_summary: list[dict[str, Any]] | None = None,
    readiness_enforcement: bool = True,
) -> dict[str, Any]:
    preview = registry.validation_preview(
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    from .build_readiness import build_feature_readiness_manifest

    readiness = build_feature_readiness_manifest(gap_max_sec=gap_max_sec)
    if build_stats:
        readiness["enforcement_stats"] = {
            k: build_stats.get(k)
            for k in (
                "nulled_cells", "rolling_nulled", "derived_nulled",
                "gap_resets", "enforced_rows",
            )
            if k in build_stats
        }
        compliance = build_stats.get("readiness_compliance")
        if compliance:
            readiness["compliance"] = compliance
    return {
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "feature_registry_version": schema_registry_hash(),
        "rolling_policy": {
            "gap_max_sec": gap_max_sec,
            "warmup_mode": warmup_mode.value,
            "reset_on_gap": True,
            "sampling_interval_sec": sampling_interval_sec,
        },
        "feature_readiness": readiness,
        "readiness_enforcement": readiness_enforcement,
        "classification": preview.get("classification"),
        "selected_features": selected_features,
        "build_stats": build_stats or {},
        "feature_health": health_summary or [],
        "registry_versions": registry_version_info(),
    }


def build_report_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    stats = manifest.get("build_stats") or {}
    return {
        "rows": stats.get("rows"),
        "warmup_rows_skipped": stats.get("warmup_rows_skipped"),
        "gap_resets": stats.get("gap_resets"),
        "largest_gap_sec": stats.get("largest_gap_sec"),
        "rolling_resets": stats.get("gap_resets"),
        "rows_with_null_rolling": stats.get("rolling_not_ready_outputs"),
        "derived_null_propagations": stats.get("derived_null_propagations"),
        "policy": manifest.get("rolling_policy"),
        "feature_policy_version": manifest.get("feature_policy_version"),
        "feature_registry_version": manifest.get("feature_registry_version"),
    }
