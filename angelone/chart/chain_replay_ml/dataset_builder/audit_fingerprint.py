"""Dataset fingerprint and quality score for audit reports."""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from typing import Any

from .pipeline_identity import (
    VALIDATOR_VERSION,
    dataset_version_view,
    format_version,
    git_commit,
    load_feature_registry,
)
from .writer import BUILDER_VERSION, METADATA_VERSION


def _file_sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_schema_version() -> int | None:
    try:
        return int(load_feature_registry().get("version") or 0) or None
    except (TypeError, ValueError):
        return None


def build_dataset_fingerprint(
    *,
    parquet_path: str | None,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dataset_version_view(meta_doc, expected_doc)
    fp = identity.get("pipeline_fingerprint") or {}
    from .spec_identity import dataset_spec_hash

    created = meta_doc.get("created_at") or (expected_doc or {}).get("created_at")
    return {
        "dataset_hash_sha256": _file_sha256(parquet_path or ""),
        "dataset_version": identity.get("dataset_version"),
        "dataset_spec_hash": dataset_spec_hash(meta_doc, expected_doc),
        "builder_version": format_version(str(meta_doc.get("builder_version") or BUILDER_VERSION)),
        "validator_version": format_version(VALIDATOR_VERSION),
        "metadata_version": int(meta_doc.get("version") or METADATA_VERSION),
        "registry_version": _registry_schema_version(),
        "feature_registry": identity.get("feature_registry"),
        "feature_registry_hash": identity.get("feature_registry_hash") or fp.get("feature_registry_hash"),
        "pipeline_stage_hashes": identity.get("pipeline_stage_hashes") or fp.get("pipeline_stage_hashes"),
        "feature_count": identity.get("feature_count"),
        "git_commit": identity.get("git_commit") or git_commit(),
        "python_version": sys.version.split()[0],
        "created_at": created,
        "created_label": identity.get("created_label"),
        "pipeline_fingerprint": fp,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }


def build_quality_score(
    *,
    errors: list[str],
    warnings: list[str],
    formula_registry: list[dict[str, Any]],
    formula_recalc: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    strike_audit: dict[str, Any] | None,
    targets_audit: dict[str, Any] | None,
    integrity_audit: dict[str, Any] | None,
    rows_audit: dict[str, Any] | None,
    files: dict[str, Any],
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    extended_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Section-level PASS/WARN/FAIL and overall confidence."""
    sa = sampling_audit or {}
    sta = strike_audit or {}
    ta = targets_audit or {}
    ia = integrity_audit or {}
    ra = rows_audit or {}
    fr = formula_recalc or {}
    ext = extended_audit or {}
    ind = ext.get("independent_formulas") or {}
    dist = ext.get("feature_distributions") or {}
    corr = ext.get("correlation_checks") or {}
    missing_vals = int(ia.get("missing_feature_values") or 0)

    unexpected = int(sa.get("unexpected_missing") or 0)
    registry_ok = all(r.get("status") == "pass" for r in formula_registry) if formula_registry else True
    recalc_ok = fr.get("status") == "pass" if fr else True
    independent_ok = ind.get("status") in ("pass", "configuration_mismatch") if ind else True
    distribution_ok = dist.get("status") == "pass" if dist else True
    correlation_ok = corr.get("status") in ("pass", "warn") if corr else True

    def _section(status: str) -> dict[str, Any]:
        return {"status": status, "pass": status == "pass"}

    sections = [
        ("configuration", "Configuration", _section(
            "pass" if files.get("specification", {}).get("exists") and files.get("metadata", {}).get("exists") else "fail"
        )),
        ("sampling", "Sampling", _section(
            "pass" if unexpected == 0 else "fail"
        )),
        ("strike_selection", "Strike Selection", _section(
            "pass" if int(sta.get("failures") or 0) == 0 else "warn"
        )),
        ("prediction_targets", "Prediction Targets", _section(
            ta.get("status", "fail")
        )),
        ("feature_generation", "Feature Generation", _section(
            "pass" if registry_ok else "warn"
        )),
        ("feature_validation", "Feature Validation", _section(
            "pass" if recalc_ok and independent_ok else ("warn" if fr.get("status") == "warn" else "fail")
        )),
        ("distribution", "Feature Distribution", _section(
            "pass" if distribution_ok else "fail"
        )),
        ("correlation", "Correlation Checks", _section(
            corr.get("status", "pass") if corr else "pass"
        )),
        ("integrity", "Integrity", _section(
            "fail" if int(ia.get("all_null_count") or 0) > 0 else (
                "pass" if int(ia.get("duplicate_rows") or 0) == 0
                and int(ia.get("invalid_timestamps") or 0) == 0
                and missing_vals == 0
                else "warn"
            )
        )),
        ("performance", "Performance", _section("pass")),
    ]

    weights = {
        "configuration": 12,
        "sampling": 12,
        "strike_selection": 8,
        "prediction_targets": 12,
        "feature_generation": 8,
        "feature_validation": 15,
        "distribution": 10,
        "correlation": 8,
        "integrity": 10,
        "performance": 5,
    }
    total_w = sum(weights.values())
    score = 0.0
    for key, _label, sec in sections:
        st = sec["status"]
        w = weights.get(key, 10)
        if st == "pass":
            score += w
        elif st == "warn":
            score += w * 0.85

    if ra.get("status") == "pass":
        score += 0
    confidence = round(min(100.0, score / total_w * 100.0), 1)

    from .audit_investigation_engine import build_training_readiness

    readiness = build_training_readiness({
        "errors": errors,
        "integrity_audit": ia,
        "feature_audit": {"formula_recalc": fr},
        "extended_audit": ext,
        "targets_audit": ta,
        "files": files,
    })
    ready = readiness["ready"]
    ready_label = readiness["recommendation"]
    if ready and missing_vals > 0 and "missing values" not in ready_label.lower():
        ready_label = (
            "🟡 READY WITH WARNINGS — missing values exist in conditional features. "
            "See Integrity section."
        )

    return {
        "sections": [{"id": k, "label": lbl, "status": s["status"]} for k, lbl, s in sections],
        "confidence_pct": confidence,
        "ready_for_training": ready,
        "ready_label": ready_label,
        "missing_feature_values": missing_vals,
        "has_missing_warnings": missing_vals > 0,
    }
