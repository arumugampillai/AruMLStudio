"""Filesystem paths for trained model packages."""

from __future__ import annotations

import os
import re

from chain_replay_ml.dataset_builder.writer import _safe_filename


def models_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "models")
    os.makedirs(path, exist_ok=True)
    return path


def safe_model_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(name or "").strip())
    return cleaned or "unnamed_model"


def model_package_dir(data_dir: str, model_name: str) -> str:
    return os.path.join(models_dir(data_dir), safe_model_name(model_name))


def model_artifact_paths(data_dir: str, model_name: str) -> dict[str, str]:
    base = model_package_dir(data_dir, model_name)
    return {
        "package_dir": base,
        "model_json": os.path.join(base, "model.json"),
        "model_ubj": os.path.join(base, "model.ubj"),
        "baseline_model_ubj": os.path.join(base, "baseline_model.ubj"),
        "tuned_model_ubj": os.path.join(base, "tuned_model.ubj"),
        "config_json": os.path.join(base, "config.json"),
        "metrics_json": os.path.join(base, "metrics.json"),
        "feature_importance_csv": os.path.join(base, "feature_importance.csv"),
        "training_summary_json": os.path.join(base, "training_summary.json"),
        "training_metadata_json": os.path.join(base, "training_metadata.json"),
        "training_monitor_csv": os.path.join(base, "training_monitor.csv"),
        "training_log_txt": os.path.join(base, "training_log.txt"),
        "training_report_html": os.path.join(base, "training_report.html"),
        "pipeline_fingerprint_json": os.path.join(base, "pipeline_fingerprint.json"),
        "dataset_build_snapshot_json": os.path.join(base, "dataset_build_snapshot.json"),
        "schema_registry_json": os.path.join(base, "schema_registry.json"),
        "validation_rules_json": os.path.join(base, "validation_rules.json"),
        "registry_json": os.path.join(base, "registry.json"),
        "model_note_json": os.path.join(base, "model_note.json"),
        "feature_studio_status_json": os.path.join(base, "feature_studio_status.json"),
        # legacy hash sidecars (optional)
        "schema_hash_txt": os.path.join(base, "schema_hash.txt"),
        "validation_hash_txt": os.path.join(base, "validation_hash.txt"),
    }


def dataset_safe_name(dataset_name: str) -> str:
    return _safe_filename(dataset_name)
