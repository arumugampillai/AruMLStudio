"""In-process dataset registry operations (no HTTP / chart server)."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .build_service import chart_data_dir


def data_dir_for(chart_dir: str) -> str:
    return chart_data_dir(chart_dir)


def list_registry_datasets(chart_dir: str) -> list[dict[str, Any]]:
    from .selection_lists import get_sorted_datasets

    return get_sorted_datasets(data_dir_for(chart_dir))


def delete_registry_dataset(chart_dir: str, dataset_name: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder import delete_dataset

    return delete_dataset(data_dir=data_dir_for(chart_dir), dataset_name=dataset_name)


def load_dataset_summary(chart_dir: str, dataset_name: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder import build_dataset_summary

    return build_dataset_summary(data_dir_for(chart_dir), dataset_name)


def load_dataset_metadata(chart_dir: str, dataset_name: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.dataset_validator import load_audit_cache, load_validation_cache
    from chain_replay_ml.dataset_builder.dataset_csv_export import build_csv_export_metadata
    from chain_replay_ml.dataset_builder.expected_spec import expected_spec_path
    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

    data_dir = data_dir_for(chart_dir)
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    meta_path = os.path.join(out_dir, f"{safe_name}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found for {safe_name}")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    expected_doc: dict[str, Any] | None = None
    expected_path = expected_spec_path(data_dir, safe_name)
    if os.path.isfile(expected_path):
        with open(expected_path, encoding="utf-8") as fh:
            expected_doc = json.load(fh)
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    return {
        "dataset_name": safe_name,
        "metadata": meta,
        "expected_spec": expected_doc,
        "audit_cache": load_audit_cache(data_dir, safe_name),
        "validation_cache": load_validation_cache(data_dir, safe_name),
        "source_dataset": {
            "kind": "parquet",
            "dataset_name": safe_name,
            "dataset_id": str(meta.get("dataset_id") or safe_name),
            "dataset_version": str(
                meta.get("dataset_version") or meta.get("builder_version") or ""
            ),
            "path": parquet_path,
        },
        "csv_export": build_csv_export_metadata(data_dir, safe_name),
    }


def load_schema_viewer() -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.lookback_policy import DEFAULT_LOOKBACK_POLICY, lookback_policy_hash
    from chain_replay_ml.dataset_builder.pipeline_identity import implementation_hash
    from chain_replay_ml.dataset_builder.schema_registry import (
        load_schema_registry,
        metadata_column_names,
        schema_registry_hash,
        targets_map,
    )
    from chain_replay_ml.dataset_builder.validation_rules import load_validation_rules, validation_rules_hash

    schema = load_schema_registry()
    columns = schema.get("columns") or {}
    feature_count = sum(1 for c in columns.values() if str(c.get("type") or "").lower() == "feature")
    target_count = sum(1 for c in columns.values() if str(c.get("type") or "").lower() == "target")
    rules = load_validation_rules()
    return {
        "overview": {
            "schema_version": schema.get("version"),
            "schema_registry_hash": schema_registry_hash(schema),
            "column_count": len(columns),
            "feature_count": feature_count,
            "target_count": target_count,
            "metadata_count": len(metadata_column_names(schema)),
            "validation_rules_version": rules.get("version"),
            "validation_rules_hash": validation_rules_hash(rules),
            "implementation_hash": implementation_hash(),
            "lookback_policy": (DEFAULT_LOOKBACK_POLICY or {}).get("method") or "nearest_snapshot",
            "lookback_policy_hash": lookback_policy_hash(),
        },
        "schema": schema,
        "validation_rules": rules,
    }


def run_audit(
    chart_dir: str,
    dataset_name: str,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder import audit_dataset
    from chain_replay_ml.dataset_builder.dataset_validator import save_audit_cache

    data_dir = data_dir_for(chart_dir)
    report = audit_dataset(data_dir=data_dir, dataset_name=dataset_name, on_progress=on_progress)
    try:
        save_audit_cache(data_dir, dataset_name, report)
    except OSError:
        pass
    return report


def run_validation(
    chart_dir: str,
    dataset_name: str,
    *,
    n_sample: int = 100,
    tolerance: float = 1e-6,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.dataset_validator import run_dataset_validation

    return run_dataset_validation(
        data_dir_for(chart_dir),
        dataset_name,
        n_sample=n_sample,
        tolerance=tolerance,
        on_progress=on_progress,
        save_cache=True,
    )


def compare_registry_datasets(chart_dir: str, dataset_a: str, dataset_b: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder import compare_datasets

    return compare_datasets(
        data_dir=data_dir_for(chart_dir),
        dataset_a=dataset_a,
        dataset_b=dataset_b,
    )


def merge_plan(chart_dir: str, dataset_name: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_merge_ops import plan_feature_merge

    return plan_feature_merge(data_dir_for(chart_dir), dataset_name)


def start_merge(chart_dir: str, dataset_name: str, features: list[str]) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_merge_ops import start_feature_merge_job

    return start_feature_merge_job(data_dir_for(chart_dir), dataset_name, features)


def merge_job_status(job_id: str) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.feature_merge_ops import get_feature_merge_job

    return get_feature_merge_job(job_id)


def golden_status(chart_dir: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.golden_regression import golden_regression_status

    return golden_regression_status(data_dir_for(chart_dir))


def run_golden(chart_dir: str, mode: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.golden_regression import run_golden_regression

    return run_golden_regression(data_dir_for(chart_dir), mode=mode)


def update_golden_manifest(chart_dir: str, dataset_name: str = "golden") -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.golden_regression import build_manifest_from_dataset

    return build_manifest_from_dataset(data_dir_for(chart_dir), dataset_name)


def save_summary_pdf(chart_dir: str, dataset_name: str, out_path: str) -> None:
    from chain_replay_ml.dataset_builder.dataset_summary_pdf import build_summary_pdf

    summary = load_dataset_summary(chart_dir, dataset_name)
    pdf_bytes = build_summary_pdf(summary)
    with open(out_path, "wb") as fh:
        fh.write(pdf_bytes)


def training_allowed(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    from chain_replay_ml.dataset_builder.audit_investigation_engine import is_training_allowed, normalize_training_recommendation

    if row.get("audit_validation_required") is False:
        return True
    readiness = row.get("readiness") or (row.get("status") or {}).get("readiness") or {}
    rec = normalize_training_recommendation(
        row.get("training_recommendation")
        or readiness.get("training_recommendation")
        or row.get("status", {}).get("training_recommendation"),
    )
    return is_training_allowed(rec)


def list_rr_enrichment_labs(chart_dir: str, dataset_name: str | None = None) -> list[dict[str, Any]]:
    from chain_replay_ml.model_lab.rr_dataset_enrich import list_labs_with_rr_labels

    return list_labs_with_rr_labels(parent_dataset=dataset_name)


def enrich_dataset_with_rr_labels(
    chart_dir: str,
    dataset_name: str,
    lab_db_path: str,
) -> dict[str, Any]:
    from chain_replay_ml.model_lab.rr_dataset_enrich import enrich_training_dataset_with_rr_labels

    return enrich_training_dataset_with_rr_labels(
        data_dir_for(chart_dir),
        dataset_name,
        lab_db_path,
    )


def generate_registry_csv(
    chart_dir: str,
    dataset_name: str,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.dataset_csv_export import generate_dataset_csv_export

    return generate_dataset_csv_export(
        data_dir_for(chart_dir),
        dataset_name,
        replace=replace,
    )


def delete_registry_csv(chart_dir: str, dataset_name: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.dataset_csv_export import delete_dataset_csv_export

    return delete_dataset_csv_export(data_dir_for(chart_dir), dataset_name)
