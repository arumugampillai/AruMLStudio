"""Dataset pipeline version stamps and identity fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

METADATA_VERSION = 2
BUILDER_VERSION = "1.4.2"
VALIDATOR_VERSION = "1.4.3"

_REGISTRY_FILES = (
    "ml_schema_registry.json",
    "ml_feature_registry.json",
)


def compute_content_hash(material: Any) -> str:
    """Short uppercase SHA256 fingerprint (8 chars) of canonical JSON material."""
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def load_feature_registry(path: str | None = None) -> dict[str, Any]:
    """Backward-compatible feature-group API — reads ml_schema_registry.json when present."""
    from .schema_registry import load_feature_registry as _load

    return _load(path)


def feature_registry_identity_material(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Legacy combined material — prefer split hashes for new code."""
    from .schema_registry import (
        implementation_identity_material,
        schema_registry_identity_material,
    )
    from .validation_rules import validation_identity_material

    reg = registry or load_feature_registry()
    return {
        "schema": schema_registry_identity_material(),
        "implementation": implementation_identity_material(),
        "validation": validation_identity_material(),
        "registry_legacy": reg,
    }


def feature_registry_hash(registry: dict[str, Any] | None = None) -> str:
    """Schema registry hash (backward-compatible name)."""
    from .schema_registry import schema_registry_hash

    return schema_registry_hash()


def schema_registry_hash() -> str:
    from .schema_registry import schema_registry_hash as _hash

    return _hash()


def validation_rules_hash() -> str:
    from .validation_rules import validation_rules_hash as _hash

    return _hash()


def implementation_hash() -> str:
    from .schema_registry import implementation_hash as _hash

    return _hash()


def lookback_policy_hash() -> str:
    from .lookback_policy import lookback_policy_identity_material

    return compute_content_hash(lookback_policy_identity_material())


def build_pipeline_stage_hashes(
    *,
    sampling_interval_sec: int,
    atm_band: int,
    target_horizons_sec: list[int],
    lookback_policy: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Per-stage pipeline hashes for regression (sampling · features · targets · validation)."""
    from .validation_rules import validation_identity_material

    schema_hash = schema_registry_hash()
    impl_hash = implementation_hash()
    lb_hash = lookback_policy_hash()
    sampling_hash = compute_content_hash({
        "sampling_interval_sec": int(sampling_interval_sec),
        "atm_band": int(atm_band),
        "lookback_policy": str(lookback_policy),
    })
    target_hash = compute_content_hash({
        "horizons_sec": [int(h) for h in target_horizons_sec],
    })
    validation_hash = compute_content_hash({
        "validator_version": VALIDATOR_VERSION,
        **validation_identity_material(),
    })
    return {
        "sampling": sampling_hash,
        "feature": schema_hash,
        "target": target_hash,
        "validation": validation_hash,
        "schema_registry": schema_hash,
        "validation_rules": validation_rules_hash(),
        "implementation": impl_hash,
        "lookback_policy": lb_hash,
        # Legacy alias
        "feature_legacy": compute_content_hash({
            "schema": schema_hash,
            "implementation": impl_hash,
            "validation": validation_hash,
        }),
    }


def pipeline_stage_hashes_from_fingerprint(
    pipeline_fingerprint: dict[str, Any] | None,
    *,
    meta_doc: dict[str, Any] | None = None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Read frozen stage hashes or recompute for legacy datasets."""
    fp = pipeline_fingerprint or {}
    stored = fp.get("pipeline_stage_hashes")
    if isinstance(stored, dict) and stored:
        return {str(k): str(v).upper() for k, v in stored.items()}

    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    sampling = meta_doc.get("sampling") or expected_doc.get("sampling") or {}
    strike = meta_doc.get("strike_selection") or expected_doc.get("strike_selection") or {}
    config = meta_doc.get("dataset_configuration") or expected_doc.get("dataset_configuration") or {}
    pol = config.get("lookback_policy") or {}

    interval_raw = (
        sampling.get("interval_sec")
        or sampling.get("trainingIntervalSec")
        or fp.get("sampling")
        or "10s"
    )
    if isinstance(interval_raw, str):
        interval_sec = int(str(interval_raw).rstrip("s") or "10")
    else:
        interval_sec = int(interval_raw or 10)

    horizons = fp.get("targets")
    if not horizons:
        horizons = config.get("future_targets_sec") or []
    horizons = [int(h) for h in horizons]

    lookback = str(fp.get("lookback_policy") or pol.get("method") or "nearest_snapshot")
    return build_pipeline_stage_hashes(
        sampling_interval_sec=interval_sec,
        atm_band=int(fp.get("atm_band") or strike.get("band") or strike.get("atmBand") or 10),
        target_horizons_sec=horizons,
        lookback_policy=lookback,
    )


def format_version(version: str | None) -> str:
    """Prefix bare semver with ``v`` for display."""
    v = str(version or "").strip()
    if not v:
        return "—"
    return v if v.startswith("v") else f"v{v}"


def git_commit(*, short: int = 7) -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        commit = out.strip()
        return commit[:short] if commit else None
    except (OSError, subprocess.SubprocessError):
        return None


def registry_feature_count(registry: dict[str, Any] | None = None) -> int:
    reg = registry or load_feature_registry()
    groups = reg.get("groups") or {}
    return sum(len((g or {}).get("features") or []) for g in groups.values())


def feature_registry_version_label(
    *,
    feature_count: int | None = None,
    registry: dict[str, Any] | None = None,
) -> str:
    """Human label like ``124-v1`` — feature count + registry schema version."""
    from .schema_registry import load_schema_registry

    schema = load_schema_registry()
    reg = registry or load_feature_registry()
    count = int(feature_count if feature_count is not None else registry_feature_count(reg))
    schema_v = int(schema.get("version") or reg.get("version") or 1)
    return f"{count}-v{schema_v}"


def build_pipeline_fingerprint(
    *,
    sampling_interval_sec: int,
    atm_band: int,
    feature_count: int,
    target_horizons_sec: list[int],
    lookback_policy: str,
    registry: dict[str, Any] | None = None,
    builder_version: str | None = None,
) -> dict[str, Any]:
    """Canonical dataset identity — persisted in expected spec and metadata."""
    reg = registry or load_feature_registry()
    schema_hash = schema_registry_hash()
    stage_hashes = build_pipeline_stage_hashes(
        sampling_interval_sec=sampling_interval_sec,
        atm_band=atm_band,
        target_horizons_sec=target_horizons_sec,
        lookback_policy=lookback_policy,
        registry=reg,
    )
    return {
        "sampling": f"{int(sampling_interval_sec)}s",
        "atm_band": int(atm_band),
        "features": int(feature_count),
        "targets": [int(h) for h in target_horizons_sec],
        "lookback_policy": str(lookback_policy),
        "feature_registry_version": feature_registry_version_label(
            feature_count=feature_count,
            registry=reg,
        ),
        "feature_registry_hash": schema_hash,
        "schema_registry_hash": schema_hash,
        "validation_rules_hash": validation_rules_hash(),
        "implementation_hash": implementation_hash(),
        "lookback_policy_hash": lookback_policy_hash(),
        "pipeline_stage_hashes": stage_hashes,
        "builder_version": str(builder_version or BUILDER_VERSION),
    }


def build_version_metadata_fields(
    pipeline_fingerprint: dict[str, Any],
    *,
    git_commit_hash: str | None = None,
) -> dict[str, Any]:
    """Fields frozen into dataset metadata at build time."""
    bv = str(pipeline_fingerprint.get("builder_version") or BUILDER_VERSION)
    return {
        "dataset_version": bv,
        "builder_version": bv,
        "pipeline_fingerprint": dict(pipeline_fingerprint),
        "git_commit": git_commit_hash if git_commit_hash is not None else git_commit(),
        "feature_registry_version": pipeline_fingerprint.get("feature_registry_version"),
        "feature_registry_hash": pipeline_fingerprint.get("feature_registry_hash"),
        "schema_registry_hash": pipeline_fingerprint.get("schema_registry_hash"),
        "validation_rules_hash": pipeline_fingerprint.get("validation_rules_hash"),
        "implementation_hash": pipeline_fingerprint.get("implementation_hash"),
        "lookback_policy_hash": pipeline_fingerprint.get("lookback_policy_hash"),
        "feature_registry_count": pipeline_fingerprint.get("features"),
        "pipeline_stage_hashes": pipeline_fingerprint.get("pipeline_stage_hashes"),
    }


def _parse_created_label(iso_ts: str | None) -> str | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso_ts)


def dataset_version_view(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen dataset identity for audit / metadata UI."""
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    fp = meta_doc.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint") or {}
    bv = meta_doc.get("dataset_version") or meta_doc.get("builder_version") or fp.get("builder_version")
    created = meta_doc.get("created_at") or expected_doc.get("created_at")
    return {
        "dataset_version": format_version(str(bv) if bv else None),
        "dataset_version_raw": bv,
        "builder_version": format_version(str(meta_doc.get("builder_version") or bv or None)),
        "feature_registry": (
            meta_doc.get("feature_registry_version")
            or fp.get("feature_registry_version")
        ),
        "feature_registry_hash": (
            meta_doc.get("feature_registry_hash")
            or fp.get("feature_registry_hash")
            or fp.get("schema_registry_hash")
        ),
        "schema_registry_hash": (
            meta_doc.get("schema_registry_hash")
            or fp.get("schema_registry_hash")
            or fp.get("feature_registry_hash")
        ),
        "validation_rules_hash": (
            meta_doc.get("validation_rules_hash")
            or fp.get("validation_rules_hash")
        ),
        "implementation_hash": (
            meta_doc.get("implementation_hash")
            or fp.get("implementation_hash")
        ),
        "lookback_policy_hash": (
            meta_doc.get("lookback_policy_hash")
            or fp.get("lookback_policy_hash")
        ),
        "pipeline_stage_hashes": (
            meta_doc.get("pipeline_stage_hashes")
            or fp.get("pipeline_stage_hashes")
            or {}
        ),
        "feature_count": meta_doc.get("feature_registry_count") or fp.get("features") or meta_doc.get("feature_count"),
        "git_commit": meta_doc.get("git_commit") or expected_doc.get("git_commit"),
        "created_at": created,
        "created_label": _parse_created_label(created),
        "pipeline_fingerprint": fp,
    }


def investigation_version_context(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Version context stamped on every RCA investigation."""
    ds = dataset_version_view(meta_doc, expected_doc)
    validator_v = format_version(VALIDATOR_VERSION)
    dataset_v = ds.get("dataset_version") or "—"
    raw_builder = str(ds.get("dataset_version_raw") or "").lstrip("v")
    raw_validator = str(VALIDATOR_VERSION).lstrip("v")
    aligned = raw_builder == raw_validator if raw_builder else None
    note = None
    if aligned is False:
        note = (
            "Validator is newer than the dataset builder version recorded at build time. "
            "Some failures may reflect validator improvements, not dataset bugs."
        )
    elif aligned is True:
        note = "Validator and dataset were built with the same pipeline version."
    return {
        "title": "Investigated using",
        "validator_version": validator_v,
        "dataset_version": dataset_v,
        "builder_version": ds.get("builder_version"),
        "validator_version_raw": VALIDATOR_VERSION,
        "dataset_version_raw": ds.get("dataset_version_raw"),
        "validator_git_commit": git_commit(),
        "dataset_git_commit": ds.get("git_commit"),
        "versions_aligned": aligned,
        "note": note,
    }
