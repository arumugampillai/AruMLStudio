"""Standalone feature validation — recalculate from replay DB and compare to dataset."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from .expected_spec import expected_spec_path
from .formula_recalc import validate_dataset_features
from .writer import _safe_filename, datasets_dir

_CHART_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pipeline_identity import load_feature_registry


def _load_registry() -> dict[str, Any]:
    return load_feature_registry()


def validation_cache_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.validation.json")


def audit_cache_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.audit-cache.json")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_validation_cache(data_dir: str, dataset_name: str) -> dict[str, Any] | None:
    path = validation_cache_path(data_dir, _safe_filename(dataset_name))
    if not os.path.isfile(path):
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def load_audit_cache(data_dir: str, dataset_name: str) -> dict[str, Any] | None:
    path = audit_cache_path(data_dir, _safe_filename(dataset_name))
    if not os.path.isfile(path):
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def save_validation_cache(data_dir: str, dataset_name: str, report: dict[str, Any]) -> str:
    safe_name = _safe_filename(dataset_name)
    path = validation_cache_path(data_dir, safe_name)
    doc = {
        **report,
        "dataset_name": safe_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return path


def save_audit_cache(data_dir: str, dataset_name: str, report: dict[str, Any]) -> str:
    safe_name = _safe_filename(dataset_name)
    path = audit_cache_path(data_dir, safe_name)
    overall = report.get("overall") or {}
    result = report.get("result") or {}
    spec_summary = report.get("specification_summary") or {}
    val_block = spec_summary.get("validator") or {}
    from .dataset_summary import extract_summary_snapshot

    doc = {
        "dataset_name": safe_name,
        "status": overall.get("status") or ("pass" if report.get("passed") else "fail"),
        "label": overall.get("label") or "—",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "safe_to_train": bool(result.get("safe_to_train")),
        "ready_for_training": bool(result.get("ready_for_training")),
        "training_readiness_label": result.get("training_readiness_label"),
        "errors": int(result.get("errors") or 0),
        "warnings": int(result.get("warnings") or 0),
        "spec_hash_match": spec_summary.get("spec_hash_match"),
        "policies_match": val_block.get("policies_match"),
        "investigations_count": len(report.get("investigations") or []),
        "audit_conclusion": report.get("audit_conclusion"),
        "audit_decision": report.get("audit_decision"),
        "blocking_issues": (report.get("training_readiness") or {}).get("blocking_issues")
            or (report.get("result") or {}).get("blocking_issues")
            or [],
        "training_recommendation": report.get("training_recommendation")
            or (report.get("training_readiness") or {}).get("training_recommendation"),
        "training_recommendation_display": (report.get("training_readiness") or {}).get("recommendation"),
        "merged_root_causes": report.get("merged_root_causes"),
        "investigations": [
            {
                "key": inv.get("key"),
                "feature": inv.get("feature"),
                "check": inv.get("check"),
                "category": inv.get("category"),
                "status": inv.get("status"),
                "root_cause": inv.get("root_cause_category"),
                "confidence": inv.get("confidence"),
                "severity": inv.get("severity"),
                "recommendation": inv.get("recommended_action"),
                "affected_rows": inv.get("affected_rows"),
                "affected_features": inv.get("affected_features"),
                "timeline": inv.get("timeline"),
                "classification": inv.get("classification"),
            }
            for inv in (report.get("investigations") or [])
        ],
        "investigations_by_key": report.get("investigations_by_key"),
        "execution_tree": report.get("execution_tree"),
        "training_readiness": report.get("training_readiness"),
        "summary_snapshot": extract_summary_snapshot(report),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return path


def run_dataset_validation(
    data_dir: str,
    dataset_name: str,
    *,
    n_sample: int = 100,
    tolerance: float = 1e-6,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    save_cache: bool = True,
) -> dict[str, Any]:
    """Recalculate features on sample rows and compare against the parquet dataset."""
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    metadata_path = os.path.join(out_dir, f"{safe_name}.json")
    expected_path = expected_spec_path(data_dir, safe_name)

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not os.path.isfile(expected_path):
        raise FileNotFoundError(f"Expected spec not found: {expected_path}")

    meta_doc = _load_json(metadata_path)
    expected_doc = _load_json(expected_path)
    expected_groups = list(expected_doc.get("feature_groups") or [])
    target_cols = list(
        expected_doc.get("prediction_target_columns")
        or (expected_doc.get("expected") or {}).get("target_column_names")
        or []
    )
    registry = _load_registry()
    df = pd.read_parquet(parquet_path)

    report = validate_dataset_features(
        df,
        meta_doc,
        chart_dir=_CHART_DIR,
        registry=registry,
        enabled_groups=expected_groups,
        target_columns=target_cols,
        n_sample=n_sample,
        tolerance=tolerance,
        on_progress=on_progress,
    )
    report["dataset_name"] = safe_name
    report["n_sample_requested"] = n_sample
    report["tolerance"] = tolerance

    if save_cache:
        save_validation_cache(data_dir, safe_name, report)
    return report


def _lifecycle_step(state: str, icon: str, label: str) -> dict[str, str]:
    return {"state": state, "icon": icon, "label": label}


def build_registry_status(
    *,
    has_parquet: bool,
    has_expected: bool,
    meta: dict[str, Any],
    audit_cache: dict[str, Any] | None,
    validation_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    """Lifecycle + readiness for the dataset registry table."""
    from .audit_investigation_engine import (
        TRAINING_RECOMMENDATION_NOT_READY,
        TRAINING_RECOMMENDATION_READY,
        is_training_allowed,
        normalize_audit_status,
        normalize_training_recommendation,
        training_recommendation_display,
    )
    from .audit_options import audit_validation_required_for_dataset

    row_count = int(meta.get("row_count") or 0)
    built = has_parquet and bool(meta) and row_count > 0
    is_draft = bool(meta) and not built
    gates_required = audit_validation_required_for_dataset(meta)

    audit_status = (audit_cache or {}).get("status")
    validation_status = (validation_cache or {}).get("status")
    spec_hash_match = (audit_cache or {}).get("spec_hash_match")
    policies_match = (audit_cache or {}).get("policies_match")
    training_recommendation = TRAINING_RECOMMENDATION_NOT_READY
    train_allowed = False
    summary = "not_built"
    lifecycle: dict[str, dict[str, str]] = {
        "build": _lifecycle_step("fail", "❌", "Build"),
        "audit": _lifecycle_step("locked", "🔒", "Audit"),
        "validation": _lifecycle_step("locked", "🔒", "Validation"),
        "spec": _lifecycle_step("locked", "🔒", "Spec"),
        "training": _lifecycle_step("locked", "🔒", "Training"),
    }

    if audit_cache and built:
        training_recommendation = normalize_training_recommendation(audit_cache)
        audit_status = normalize_audit_status(audit_cache, training_recommendation) or audit_status
        train_allowed = is_training_allowed(training_recommendation)

    if built and not gates_required:
        train_allowed = True
        training_recommendation = TRAINING_RECOMMENDATION_READY

    if spec_hash_match is False or policies_match is False:
        spec_state = "fail"
    elif spec_hash_match is True and policies_match is not False:
        spec_state = "pass"
    elif audit_status == "pass":
        spec_state = "pending"
    else:
        spec_state = "locked" if not built else "pending"

    if not built:
        lifecycle = {
            "build": _lifecycle_step("draft" if is_draft else "fail", "📝" if is_draft else "❌", "Draft" if is_draft else "Build"),
            "audit": _lifecycle_step("locked", "🔒", "Audit"),
            "validation": _lifecycle_step("locked", "🔒", "Validation"),
            "spec": _lifecycle_step("locked", "🔒", "Spec"),
            "training": _lifecycle_step("locked", "🔒", "Training"),
        }
        train_allowed = False
        summary = "draft" if is_draft else "not_built"
    else:
        if audit_status == "pass":
            audit_step = _lifecycle_step("pass", "✅", "Audit")
        elif audit_status == "warn":
            audit_step = _lifecycle_step("warn", "⚠", "Audit")
        elif audit_status == "fail":
            audit_step = _lifecycle_step("fail", "❌", "Audit")
        else:
            audit_step = _lifecycle_step("pending", "⏳", "Audit")

        if validation_status == "pass":
            val_step = _lifecycle_step("pass", "✅", "Validation")
        elif validation_status == "fail":
            val_step = _lifecycle_step("fail", "❌", "Validation")
        elif validation_status == "warn":
            val_step = _lifecycle_step("warn", "⚠", "Validation")
        else:
            val_step = _lifecycle_step("pending", "⏳", "Validation")

        if spec_state == "pass":
            spec_step = _lifecycle_step("pass", "✅", "Spec")
        elif spec_state == "fail":
            spec_step = _lifecycle_step("fail", "❌", "Spec")
        else:
            spec_step = _lifecycle_step("pending", "⏳", "Spec")

        spec_ok = spec_state == "pass"
        training_step = (
            _lifecycle_step("pass", "✅", "Training")
            if train_allowed
            else _lifecycle_step("locked", "🔒", "Training")
        )
        lifecycle = {
            "build": _lifecycle_step("pass", "✅", "Build"),
            "audit": audit_step,
            "validation": val_step,
            "spec": spec_step,
            "training": training_step,
        }
        if train_allowed:
            summary = "ready_for_training"
        elif validation_status == "fail":
            summary = "validation_failed"
        elif audit_status in (None, "pending"):
            summary = "audit_pending"
        else:
            summary = "not_ready_for_training"

    audit_label = "—"
    audit_display = "—"
    if audit_cache:
        st = audit_status or audit_cache.get("status")
        if st == "pass":
            audit_label = "PASS"
            audit_display = "Passed"
        elif st == "warn":
            audit_label = "WARN"
            audit_display = "Warnings"
        elif st == "fail":
            audit_label = "FAIL"
            audit_display = "Failed"
    elif built and has_expected:
        audit_label = "PENDING"
        audit_display = "Pending"

    validation_label = (validation_cache or {}).get("label") or (
        "PASS" if validation_status == "pass" else (
            "FAIL" if validation_status == "fail" else (
                "WARN" if validation_status == "warn" else "—"
            )
        )
    )

    display = training_recommendation_display(training_recommendation)
    readiness_title = display
    if not train_allowed and training_recommendation == TRAINING_RECOMMENDATION_NOT_READY:
        blocking = (audit_cache or {}).get("blocking_issues") or []
        if blocking:
            readiness_title = f"{display} — {blocking[0]}"
        elif not audit_cache:
            readiness_title = "Run Audit to compute training recommendation."
    elif built and not gates_required:
        readiness_title = "Audit & validation optional — training allowed"

    return {
        "summary": summary,
        "is_draft": is_draft,
        "lifecycle": lifecycle,
        "training_recommendation": training_recommendation,
        "audit_validation_required": gates_required,
        "readiness": {
            "ready": train_allowed,
            "training_recommendation": training_recommendation,
            "display": display,
            "label": display.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""),
            "title": readiness_title,
        },
        "ready_for_training": train_allowed,
        "audit_status": audit_status,
        "audit_label": audit_label,
        "audit_display": audit_display,
        "validation_status": validation_status,
        "validation_label": validation_label,
        "spec_hash_match": spec_hash_match,
        "policies_match": policies_match,
        # Legacy badges for any older UI paths
        "badges": [],
    }
