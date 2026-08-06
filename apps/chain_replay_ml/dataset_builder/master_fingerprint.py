"""Compact dataset fingerprint for master metadata APIs."""

from __future__ import annotations

import json
from typing import Any

from .pipeline_identity import BUILDER_VERSION, build_pipeline_fingerprint, compute_content_hash


METADATA_STATUS_VALID = "VALID"
METADATA_STATUS_BUILDING = "BUILDING"
METADATA_STATUS_REPAIRING = "REPAIRING"
METADATA_STATUS_STALE = "STALE"
METADATA_STATUS_ERROR = "ERROR"

# Legacy value from Phase 1
METADATA_STATUS_OK = "OK"


def normalize_metadata_status(status: str | None) -> str:
    raw = str(status or METADATA_STATUS_VALID).upper()
    if raw == METADATA_STATUS_OK:
        return METADATA_STATUS_VALID
    return raw


def target_hash_from_columns(target_columns: list[str]) -> str:
    return compute_content_hash(sorted(str(c) for c in target_columns))


def build_dataset_fingerprint_blob(
    *,
    sampling_interval_sec: int | None,
    feature_registry_version: str | None,
    feature_count: int | None,
    target_count: int | None,
    schema_hash: str | None,
    feature_hash: str | None,
    target_hash: str | None,
    builder_version: str | None,
    market: str | None = None,
) -> dict[str, Any]:
    return {
        "sampling_interval": int(sampling_interval_sec) if sampling_interval_sec is not None else None,
        "feature_registry": feature_registry_version,
        "features": int(feature_count) if feature_count is not None else None,
        "targets": int(target_count) if target_count is not None else None,
        "schema_hash": schema_hash,
        "feature_hash": feature_hash,
        "target_hash": target_hash,
        "builder": str(builder_version or BUILDER_VERSION),
        "market": str(market).upper() if market else None,
    }


def build_identity_from_build(
    *,
    market: str,
    sampling_interval_sec: int,
    atm_band: int,
    feature_count: int,
    target_horizons_sec: list[int],
    lookback_policy: str,
    registry: dict[str, Any] | None,
    target_columns: list[str],
    created_from: str = "master_build",
) -> dict[str, Any]:
    """Fields to persist on master_dataset_meta after a build."""
    fp = build_pipeline_fingerprint(
        sampling_interval_sec=sampling_interval_sec,
        atm_band=atm_band,
        feature_count=feature_count,
        target_horizons_sec=target_horizons_sec,
        lookback_policy=lookback_policy,
        registry=registry,
    )
    schema_hash = fp.get("schema_registry_hash") or fp.get("feature_registry_hash")
    feature_hash = fp.get("feature_registry_hash")
    target_hash = target_hash_from_columns(target_columns)
    reg_version = fp.get("feature_registry_version")
    fingerprint = build_dataset_fingerprint_blob(
        sampling_interval_sec=sampling_interval_sec,
        feature_registry_version=str(reg_version) if reg_version is not None else None,
        feature_count=feature_count,
        target_count=len(target_columns),
        schema_hash=str(schema_hash) if schema_hash else None,
        feature_hash=str(feature_hash) if feature_hash else None,
        target_hash=target_hash,
        builder_version=str(fp.get("builder_version") or BUILDER_VERSION),
        market=market,
    )
    return {
        "market": str(market).upper(),
        "sampling_interval_sec": int(sampling_interval_sec),
        "builder_version": str(fp.get("builder_version") or BUILDER_VERSION),
        "created_from": created_from,
        "feature_registry_version": reg_version,
        "feature_hash": feature_hash,
        "target_hash": target_hash,
        "schema_hash": schema_hash,
        "feature_count": int(feature_count),
        "target_count": len(target_columns),
        "dataset_fingerprint": fingerprint,
        "pipeline_fingerprint": fp,
    }


def fingerprint_json_blob(fingerprint: dict[str, Any]) -> str:
    return json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
