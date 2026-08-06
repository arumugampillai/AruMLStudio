"""Dataset specification identity — canonical hash and audit summary."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .feature_plugins import horizon_label
from .lookback_policy import (
    lookback_policy,
    policy_alignment_view,
    policy_label,
    read_builder_policy,
    read_dataset_configuration,
    read_validator_policy,
)
from .pipeline_identity import build_pipeline_fingerprint


def canonical_spec_material(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable dict hashed to identify the validation specification."""
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    config = read_dataset_configuration(meta_doc, expected_doc)
    pol = lookback_policy(config)
    fp = meta_doc.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint")
    if fp:
        fp = dict(fp)
    else:
        sampling = meta_doc.get("sampling") or expected_doc.get("sampling") or {}
        strike = meta_doc.get("strike_selection") or expected_doc.get("strike_selection") or {}
        horizons = [int(h) for h in (config.get("future_targets_sec") or [])]
        expected_block = expected_doc.get("expected") or {}
        feature_count = int(
            expected_block.get("expected_feature_columns")
            or meta_doc.get("feature_count")
            or meta_doc.get("feature_registry_count")
            or 0
        )
        fp = build_pipeline_fingerprint(
            sampling_interval_sec=int(
                sampling.get("interval_sec") or sampling.get("trainingIntervalSec") or config.get("sampling_interval_sec") or 10
            ),
            atm_band=int(strike.get("band") or strike.get("atmBand") or 10),
            feature_count=feature_count,
            target_horizons_sec=horizons,
            lookback_policy=str(pol.get("method") or "nearest_snapshot"),
        )
    return {
        "pipeline_fingerprint": fp,
        "lookback_policy": {
            "method": pol.get("method"),
            "tolerance_min_pct": int(pol.get("tolerance_min_pct") or 50),
            "tolerance_max_pct": int(pol.get("tolerance_max_pct") or 150),
        },
    }


def compute_spec_hash(material: dict[str, Any]) -> str:
    """Short uppercase hex fingerprint (8 chars) of the canonical specification."""
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def compute_spec_hash_from_fingerprint(
    pipeline_fingerprint: dict[str, Any],
    dataset_configuration: dict[str, Any],
) -> str:
    """Hash at build / expected-spec write time."""
    pol = lookback_policy(dataset_configuration)
    return compute_spec_hash({
        "pipeline_fingerprint": dict(pipeline_fingerprint),
        "lookback_policy": {
            "method": pol.get("method"),
            "tolerance_min_pct": int(pol.get("tolerance_min_pct") or 50),
            "tolerance_max_pct": int(pol.get("tolerance_max_pct") or 150),
        },
    })


def dataset_spec_hash(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> str:
    """Frozen dataset spec hash — stored at build time, recomputed for legacy datasets."""
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    stored = meta_doc.get("dataset_spec_hash") or expected_doc.get("dataset_spec_hash")
    if stored:
        return str(stored).upper()
    return compute_spec_hash(canonical_spec_material(meta_doc, expected_doc))


def validator_spec_hash(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> str:
    """Spec hash the validator derives when reading dataset metadata / expected spec."""
    return compute_spec_hash(canonical_spec_material(meta_doc, expected_doc))


def spec_hash_fields(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dataset vs validator spec hash comparison."""
    ds_hash = dataset_spec_hash(meta_doc, expected_doc)
    val_hash = validator_spec_hash(meta_doc, expected_doc)
    match = ds_hash == val_hash
    return {
        "dataset_spec_hash": ds_hash,
        "validator_spec_hash": val_hash,
        "spec_hash_match": match,
        "spec_hash_status": "✓ Match" if match else "❌ Different Specification",
    }


def build_audit_specification_summary(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dataset specification + validator alignment block for audit UI."""
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    config = read_dataset_configuration(meta_doc, expected_doc)
    builder_policy = read_builder_policy(meta_doc, expected_doc)
    validator_policy = read_validator_policy(meta_doc, expected_doc)
    alignment = policy_alignment_view(meta_doc, expected_doc, validator_policy=validator_policy)
    hashes = spec_hash_fields(meta_doc, expected_doc)

    sampling = meta_doc.get("sampling") or expected_doc.get("sampling") or {}
    interval = int(sampling.get("interval_sec") or config.get("sampling_interval_sec") or 10)

    targets = list(expected_doc.get("prediction_targets") or config.get("future_targets_labels") or [])
    if not targets and config.get("future_targets_sec"):
        targets = [horizon_label(int(h)) for h in config["future_targets_sec"]]

    policies_match = bool(alignment.get("policies_match", True))
    hash_match = bool(hashes["spec_hash_match"])
    if policies_match and hash_match:
        validator_status = "✓ Matches Dataset Specification"
    elif policies_match:
        validator_status = "⚠ Policy matches — spec hash differs"
    else:
        validator_status = "❌ Does not match dataset specification"

    return {
        "dataset_specification": {
            "title": "Dataset Specification",
            "sampling": f"{interval} sec",
            "lookback_policy": policy_label(builder_policy),
            "prediction_targets": targets,
            "spec_hash": hashes["dataset_spec_hash"],
        },
        "validator": {
            "title": "Validator",
            "policy": alignment.get("validator_policy") or policy_label(validator_policy),
            "policies_match": policies_match,
            "status": validator_status,
            "spec_hash": hashes["validator_spec_hash"],
            "spec_hash_match": hash_match,
            "spec_hash_status": hashes["spec_hash_status"],
        },
        "policy_alignment": alignment,
        **hashes,
    }
