"""Model Builder backend — direct Python calls (standalone Tk, no HTTP)."""

from __future__ import annotations

import json
import os
from typing import Any

from chain_replay_ml.dataset_builder.audit_investigation_engine import (
    normalize_training_recommendation,
    training_recommendation_display,
)
from chain_replay_ml.dataset_builder.audit_options import audit_validation_required_for_dataset
from chain_replay_ml.dataset_builder.dataset_validator import load_audit_cache
from chain_replay_ml.dataset_builder.expected_spec import expected_spec_path
from chain_replay_ml.dataset_builder.registry_auto_features import build_registry_auto_features
from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry as _load_schema_registry
from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.training.config_validator import validate_training_config
from chain_replay_ml.training.lifecycle_store import get_model_lifecycle_view
from chain_replay_ml.training.model_lifecycle import build_model_builder_preset
from chain_replay_ml.training.registry import get_model_summary
from chain_replay_ml.training.retrain_compatibility import (
    evaluate_retrain_dataset_choice,
    list_retrain_compatible_datasets,
)


def list_builder_datasets(data_dir: str) -> list[dict[str, Any]]:
    """Datasets selectable in Create Model (registry rows with rows > 0)."""
    from ..selection_lists import get_sorted_datasets

    rows = get_sorted_datasets(data_dir)
    out: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("row_count") or 0) <= 0:
            continue
        item = dict(row)
        if row.get("has_parquet") and not row.get("is_draft"):
            item["needs_parquet"] = False
            out.append(item)
            continue
        # Metadata-only registry export (master DB) — show in picker; training needs parquet.
        meta_path = str(row.get("metadata_path") or "")
        if meta_path and os.path.isfile(meta_path):
            item["needs_parquet"] = True
            out.append(item)
    return out


def load_dataset_metadata_doc(data_dir: str, dataset_name: str) -> dict[str, Any]:
    name = str(dataset_name or "").strip()
    if not name:
        raise ValueError("dataset_name is required")
    safe = _safe_filename(name)
    out_dir = datasets_dir(data_dir)
    meta_path = os.path.join(out_dir, f"{safe}.json")
    expected_path = expected_spec_path(data_dir, safe)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found for {safe}")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        if not raw:
            raise ValueError(
                f"Dataset metadata is empty: {safe}.json\n"
                "Re-export or rebuild this dataset, then try again."
            )
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Dataset metadata is not valid JSON: {safe}.json\n{exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {safe}.json")
    expected_doc: dict[str, Any] | None = None
    if os.path.isfile(expected_path):
        try:
            with open(expected_path, encoding="utf-8") as fh:
                expected_doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            expected_doc = None
    audit_cache = load_audit_cache(data_dir, safe)
    training_recommendation = normalize_training_recommendation(audit_cache)
    gates_required = audit_validation_required_for_dataset(meta if isinstance(meta, dict) else {})
    return {
        "dataset_name": safe,
        "metadata_path": meta_path,
        "expected_path": expected_path if os.path.isfile(expected_path) else None,
        "metadata": meta,
        "expected_spec": expected_doc,
        "training_recommendation": training_recommendation,
        "training_recommendation_display": training_recommendation_display(training_recommendation),
        "audit_cache": audit_cache,
        "audit_validation_required": gates_required,
    }


def load_schema_registry() -> dict[str, Any]:
    return _load_schema_registry()


def load_validation_rules() -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.validation_rules import load_validation_rules as _load

    return _load()


def validate_config(data_dir: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    return validate_training_config(data_dir, raw_config)


def list_models_light(data_dir: str) -> list[dict[str, Any]]:
    from ..selection_lists import get_sorted_models

    return get_sorted_models(data_dir, lightweight=True)


def model_summary(data_dir: str, model_name: str) -> dict[str, Any]:
    return get_model_summary(data_dir, model_name)


def registry_auto_features(data_dir: str, *, top: str | int = 75, underlying: str | None = None) -> dict[str, Any]:
    return build_registry_auto_features(data_dir, top=top, underlying=underlying)


def lifecycle_preset(data_dir: str, model_name: str, mode: str) -> dict[str, Any]:
    return build_model_builder_preset(data_dir, model_name, mode)


def retrain_compatible(data_dir: str, source_model: str) -> dict[str, Any]:
    return list_retrain_compatible_datasets(data_dir, source_model)


def retrain_compatibility(data_dir: str, source_model: str, dataset_name: str) -> dict[str, Any]:
    return evaluate_retrain_dataset_choice(
        data_dir,
        source_model=source_model,
        dataset_name=dataset_name,
    )


def model_lifecycle_view(data_dir: str, model_name: str) -> dict[str, Any]:
    return get_model_lifecycle_view(data_dir, model_name=model_name)
