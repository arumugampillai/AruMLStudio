"""Persist trained model package under data/models/{model_name}/."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from chain_replay_ml.replay_config import build_replay_config_from_metadata
from chain_replay_ml.dataset_builder.pipeline_identity import (
    implementation_hash,
    schema_registry_hash,
    validation_rules_hash,
)
from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry, schema_registry_path
from chain_replay_ml.dataset_builder.validation_rules import load_validation_rules, validation_rules_path

from .config import TrainingConfig
from .naming import suggest_model_name_from_split
from .model_runtime import native_model_path, normalize_algorithm
from .paths import model_artifact_paths, safe_model_name
from .training_report import build_training_report_html
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

_ALGO_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "linear": "Linear",
    "neural": "Neural",
}

_FEATURE_SELECTION_LABELS = {
    "none": "None (all features)",
    "shap": "SHAP Importance",
    "rfe": "Recursive Feature Elimination",
    "permutation": "Permutation Importance",
}


def resolve_training_row_count(
    *,
    metadata: dict[str, Any] | None = None,
    matrix_report: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> int:
    """Dataset row count for registry display — not filtered training-matrix rows."""
    for block in (metadata, dataset_metadata):
        if isinstance(block, dict) and block.get("row_count") is not None:
            try:
                return int(block["row_count"])
            except (TypeError, ValueError):
                continue
    nested = (metadata or {}).get("dataset_metadata")
    if isinstance(nested, dict) and nested.get("row_count") is not None:
        try:
            return int(nested["row_count"])
        except (TypeError, ValueError):
            pass
    shape = (matrix_report or {}).get("x_shape")
    if isinstance(shape, (list, tuple)) and shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            pass
    return 0


def feature_selection_method_label(method: str | None) -> str:
    key = str(method or "rfe").strip().lower()
    return _FEATURE_SELECTION_LABELS.get(key, key)


def build_feature_elimination_doc(
    *,
    config: TrainingConfig,
    walk_forward_summary: dict[str, Any] | None = None,
    stored_elimination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return resolve_feature_elimination_doc(
        config=config,
        config_features=list(config.features or []),
        walk_forward_summary=walk_forward_summary,
        stored_elimination=stored_elimination,
    )


def resolve_feature_elimination_doc(
    *,
    config: TrainingConfig | None = None,
    config_features: list[str] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    stored_elimination: dict[str, Any] | None = None,
    csv_selected_count: int | None = None,
) -> dict[str, Any]:
    """Reconcile elimination counts — final model config beats walk-forward summary list."""
    from .objective_scoring import format_stopped_reason, metric_display_label

    wf_summary = dict(walk_forward_summary or {})
    fs = dict(wf_summary.get("feature_selection") or {})
    stored = dict(stored_elimination or {})
    wf_cfg = dict((config.split or {}).get("walk_forward") or {}) if config else {}
    method = str(stored.get("method") or fs.get("method") or wf_cfg.get("feature_selection_method") or "rfe").strip().lower()

    finished = 0
    if config_features:
        finished = len(config_features)
    elif config is not None and config.features:
        finished = len(config.features)
    elif isinstance(csv_selected_count, int) and csv_selected_count > 0:
        finished = int(csv_selected_count)
    else:
        for src in (stored, fs):
            for key in ("finished_features", "best_features_count"):
                val = src.get(key)
                if isinstance(val, int) and val > 0:
                    finished = int(val)
                    break
            if finished:
                break
        if not finished:
            val = fs.get("selected_features")
            if isinstance(val, int) and val > 0:
                finished = int(val)
        if not finished:
            wf_list = wf_summary.get("selected_features")
            if isinstance(wf_list, list) and wf_list:
                finished = len(wf_list)

    started = 0
    for key in ("started_features",):
        val = fs.get(key) or stored.get(key)
        if isinstance(val, int) and val > 0:
            started = int(val)
            break
    if not started:
        history = fs.get("history") if isinstance(fs.get("history"), list) else []
        if history:
            try:
                started = max(int(h.get("features_remaining") or 0) for h in history if isinstance(h, dict))
            except (TypeError, ValueError):
                started = 0
    if not started:
        started = finished

    eliminated = max(0, started - finished)
    iterations = int(fs.get("iterations") or stored.get("iterations") or 0)
    opt_metric = fs.get("optimization_metric") or stored.get("optimization_metric") or wf_cfg.get("optimization_metric")
    stopped_reason = str(fs.get("stopped_reason") or stored.get("stopped_reason") or "")
    stopped_reason_text = str(fs.get("stopped_reason_text") or stored.get("stopped_reason_text") or "")

    if eliminated > 0 and "kept starting set" in stopped_reason_text.lower():
        metric_label = metric_display_label(str(opt_metric or "composite"))
        best_display_raw = fs.get("best_display_score") if fs.get("best_display_score") is not None else stored.get("best_display_score")
        try:
            best_display = float(best_display_raw) if best_display_raw is not None else None
        except (TypeError, ValueError):
            best_display = None
        stopped_reason_text = format_stopped_reason(
            stopped_reason or "no_improvement",
            best_features=finished,
            started_features=started,
            metric_label=metric_label,
            best_display=best_display,
        )
    elif not stopped_reason_text:
        stopped_reason_text = format_stopped_reason(
            stopped_reason or ("complete" if iterations else "none"),
            best_features=finished,
            started_features=started,
        )

    strategy = (config.split or {}).get("strategy") if config else None
    ran = method != "none" and bool(strategy == "walk_forward" or wf_summary)
    return {
        "method": method,
        "method_label": feature_selection_method_label(method),
        "optimization_metric": opt_metric,
        "started_features": started,
        "finished_features": finished,
        "eliminated_features": eliminated,
        "iterations": iterations,
        "stopped_reason": stopped_reason or None,
        "stopped_reason_text": stopped_reason_text,
        "stability_trim_applied": bool(fs.get("stability_trim_applied") or stored.get("stability_trim_applied")),
        "ran": ran,
        "ran_during_training": ran and method != "none",
    }


def algorithm_label(algorithm: str) -> str:
    key = str(algorithm or "").strip().lower()
    return _ALGO_LABELS.get(key, key.title() or "Unknown")


def _test_metrics_view(metrics: dict[str, Any]) -> dict[str, Any]:
    test = dict(metrics.get("test") or {})
    if "directional_accuracy_pct" in test and "directional_accuracy" not in test:
        test["directional_accuracy"] = test["directional_accuracy_pct"]
    return test


def _importance_csv_df(feature_importance: pd.DataFrame) -> pd.DataFrame:
    out = feature_importance.copy()
    if "importance_pct" in out.columns:
        out = out.rename(columns={"feature": "Feature", "importance_pct": "Importance"})
    elif "feature" in out.columns and "importance" in out.columns:
        total = float(out["importance"].sum()) or 1.0
        out = out.assign(Importance=(out["importance"] / total * 100).round(2))
        out = out.rename(columns={"feature": "Feature"})[["Feature", "Importance"]]
    return out[["Feature", "Importance"]]


def build_training_summary(
    *,
    model_name: str,
    config: TrainingConfig,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    matrix_report: dict[str, Any],
    trees_trained: int,
    early_stopped: bool,
    training_time_sec: float,
    total_elapsed_sec: float,
    training_meta: dict[str, Any] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    val = metrics.get("validation") or {}
    meta = training_meta or {}
    wf = metrics.get("walk_forward") or {}
    elimination = build_feature_elimination_doc(config=config, walk_forward_summary=walk_forward_summary)
    wf_summary = dict(walk_forward_summary or {})
    wf_meta = dict(
        wf_summary.get("meta")
        or wf.get("walk_forward_meta")
        or {}
    )
    if not wf_meta and config.split.get("strategy") == "walk_forward":
        from .split import walk_forward_meta_from_config

        wf_meta = walk_forward_meta_from_config(
            dict(wf_summary.get("config") or (config.split or {}).get("walk_forward") or {}),
            split_cfg=config.split,
        )
    from .split import validation_strategy_fields_from_split

    strat_fields = validation_strategy_fields_from_split(
        config.split,
        wf_meta or (config.split or {}).get("walk_forward"),
    )
    if wf_meta and "validation_strategy_ui" not in wf_meta:
        wf_meta.update(strat_fields)
    out = {
        "model_name": model_name,
        "dataset": config.dataset,
        "algorithm": algorithm_label(config.algorithm),
        "target": config.target,
        "rows": resolve_training_row_count(metadata=metadata, matrix_report=matrix_report),
        "features": len(config.features),
        "trees_trained": int(trees_trained),
        "early_stopped": bool(early_stopped),
        "training_time_sec": round(float(training_time_sec), 2),
        "total_elapsed_sec": round(float(total_elapsed_sec), 2),
        "implementation": meta.get("implementation"),
        "device": meta.get("device"),
        "device_label": meta.get("device_label"),
        "gpu_name": meta.get("gpu_name"),
        "fallback_reason": meta.get("fallback_reason"),
        "algorithm_parameters": dict(meta.get("algorithm_parameters") or {}),
        "algorithm_runtime": dict(meta.get("algorithm_runtime") or {}),
        "prediction_time_ms": meta.get("prediction_time_ms"),
        "validation_rmse": meta.get("best_validation_rmse") or val.get("rmse"),
        "validation_mae": val.get("mae"),
        "validation_r2": val.get("r2"),
        "validation_mape": val.get("mape"),
        "validation_directional_accuracy_pct": val.get("directional_accuracy_pct"),
        "best_iteration": meta.get("best_iteration"),
        "early_stopping_rounds": meta.get("early_stopping_rounds"),
        "train_rmse": meta.get("train_rmse"),
        "walk_forward_mean_rmse": wf.get("mean_rmse"),
        "walk_forward_mean_mae": wf.get("mean_mae"),
        "walk_forward_mean_r2": wf.get("mean_r2"),
        "walk_forward_mean_mape": wf.get("mean_mape"),
        "walk_forward_mean_directional_accuracy_pct": wf.get("mean_directional_accuracy_pct"),
        "walk_forward_std_rmse": wf.get("std_rmse"),
        "walk_forward_std_mae": wf.get("std_mae"),
        "walk_forward_std_r2": wf.get("std_r2"),
        "walk_forward_std_mape": wf.get("std_mape"),
        "validation_std_rmse": val.get("std_rmse"),
        "validation_std_mae": val.get("std_mae"),
        "validation_std_r2": val.get("std_r2"),
        "validation_std_mape": val.get("std_mape"),
        "selected_feature_count": len(wf.get("selected_features") or []),
        "feature_selection_method": elimination.get("method"),
        "feature_selection_method_label": elimination.get("method_label"),
        "feature_elimination": elimination,
        "test_metrics": _test_metrics_view(metrics),
        "model_version": config.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_strategy_ui": wf_meta.get("validation_strategy_ui") or strat_fields["validation_strategy_ui"],
        "validation_strategy_label": wf_meta.get("validation_strategy_label") or strat_fields["validation_strategy_label"],
    }
    if wf_meta:
        out["walk_forward_meta"] = wf_meta
        out["fold_placement"] = wf_meta.get("fold_placement")
        out["fold_placement_label"] = wf_meta.get("fold_placement_label")
        out["walk_forward_window_mode"] = wf_meta.get("window_mode")
        out["walk_forward_n_folds"] = wf_meta.get("n_folds")
        out["walk_forward_train_window"] = wf_meta.get("train_window_size")
        out["walk_forward_validation_window"] = wf_meta.get("validation_window_size")
    return out


def _copy_json_snapshot(src_path: str, dest_path: str, fallback_loader) -> None:
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dest_path)
        return
    with open(dest_path, "w", encoding="utf-8") as fh:
        json.dump(fallback_loader(), fh, indent=2)


def save_model_package(
    *,
    data_dir: str,
    config: TrainingConfig,
    model: Any,
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    metadata: dict[str, Any],
    matrix_report: dict[str, Any],
    split_info: dict[str, Any],
    training_log_text: str = "",
    trees_trained: int = 0,
    early_stopped: bool = False,
    training_time_sec: float = 0.0,
    total_elapsed_sec: float = 0.0,
    training_meta: dict[str, Any] | None = None,
    validation_loss_curve: list[dict[str, Any]] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    shap_importance: list[dict[str, Any]] | None = None,
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_name = safe_model_name(
        config.model_name or suggest_model_name_from_split(
            config.target, config.algorithm, config.split, feature_count=len(config.features)
        )
    )
    paths = model_artifact_paths(data_dir, model_name)
    os.makedirs(paths["package_dir"], exist_ok=True)

    trained_at = datetime.now(timezone.utc).isoformat()
    config_doc = config.to_dict()
    config_doc["trained_at"] = trained_at
    config_doc["algorithm_label"] = algorithm_label(config.algorithm)
    from chain_replay_ml.dataset_builder.dataset_summary import build_dataset_build_snapshot

    dataset_build_snapshot = build_dataset_build_snapshot(
        metadata if isinstance(metadata, dict) else {},
        dataset_name=config.dataset,
        snapshotted_at=trained_at,
    )
    config_doc["dataset_build_snapshot"] = dataset_build_snapshot
    config_doc["dataset_metadata"] = {
        "row_count": dataset_build_snapshot.get("row_count") or metadata.get("row_count"),
        "feature_count": dataset_build_snapshot.get("feature_count") or metadata.get("feature_count"),
        "target_count": dataset_build_snapshot.get("target_count"),
        "dataset_version": dataset_build_snapshot.get("dataset_version") or metadata.get("dataset_version") or metadata.get("builder_version"),
        "dataset_name": config.dataset,
        "trading_days": dataset_build_snapshot.get("trading_days"),
        "trading_day_labels": dataset_build_snapshot.get("trading_day_labels"),
        "market": dataset_build_snapshot.get("market"),
        "sampling_label": dataset_build_snapshot.get("sampling_label"),
        "strike_selection_label": dataset_build_snapshot.get("strike_selection_label"),
        "filter_summary": dataset_build_snapshot.get("filter_summary"),
        "selection_method": dataset_build_snapshot.get("selection_method"),
        "master_filter": dataset_build_snapshot.get("master_filter"),
        "export_source": dataset_build_snapshot.get("export_source"),
        "created_at": dataset_build_snapshot.get("created_at"),
        "snapshotted_at": dataset_build_snapshot.get("snapshotted_at"),
    }
    config_doc["matrix_report"] = matrix_report
    _tm = training_meta or {}
    _rt = dict(_tm.get("algorithm_runtime") or {})
    if _rt or _tm.get("algorithm_parameters"):
        config_doc["algorithm_parameters"] = dict(_rt.get("algorithm_parameters") or _tm.get("algorithm_parameters") or {})
        config_doc["algorithm_runtime"] = _rt or {
            "implementation": _tm.get("implementation"),
            "device": _tm.get("device"),
            "device_label": _tm.get("device_label"),
            "gpu_name": _tm.get("gpu_name"),
            "fallback_reason": _tm.get("fallback_reason"),
            "algorithm_parameters": dict(_tm.get("algorithm_parameters") or {}),
            "prediction_time_ms": _tm.get("prediction_time_ms"),
        }
    from .split import validation_strategy_fields_from_split

    strat_fields = validation_strategy_fields_from_split(
        config.split,
        split_info.get("walk_forward") or (config.split or {}).get("walk_forward"),
    )
    config_doc["split_info"] = {
        "strategy": split_info.get("strategy"),
        "slices": split_info.get("slices"),
        "folds": split_info.get("folds"),
        "walk_forward": split_info.get("walk_forward"),
        "train": config.split.get("train"),
        "validation": config.split.get("validation"),
        "test": config.split.get("test"),
        "validation_strategy_ui": strat_fields["validation_strategy_ui"],
        "validation_strategy_label": strat_fields["validation_strategy_label"],
    }
    replay_config = build_replay_config_from_metadata(metadata)
    if replay_config:
        config_doc["replay_config"] = replay_config

    summary = build_training_summary(
        model_name=model_name,
        config=config,
        metrics=metrics,
        metadata=metadata,
        matrix_report=matrix_report,
        trees_trained=trees_trained,
        early_stopped=early_stopped,
        training_time_sec=training_time_sec,
        total_elapsed_sec=total_elapsed_sec,
        training_meta=training_meta,
        walk_forward_summary=walk_forward_summary,
    )

    elimination = build_feature_elimination_doc(config=config, walk_forward_summary=walk_forward_summary)

    with open(paths["config_json"], "w", encoding="utf-8") as fh:
        json.dump(config_doc, fh, indent=2)

    runtime = dict((training_meta or {}).get("algorithm_runtime") or {})
    metadata_doc = {
        "model_name": model_name,
        "trained_at": trained_at,
        "dataset": config.dataset,
        "target": config.target,
        "algorithm": config.algorithm,
        "algorithm_label": algorithm_label(config.algorithm),
        "prediction_type": config.prediction_type,
        "model_version": config.model_version,
        "model_description": config.model_description,
        "feature_count": len(config.features),
        "feature_selection_method": elimination.get("method"),
        "feature_selection_method_label": elimination.get("method_label"),
        "feature_elimination": elimination,
        "row_count": metadata.get("row_count"),
        "dataset_version": metadata.get("dataset_version") or metadata.get("builder_version"),
        "dataset_metadata": config_doc.get("dataset_metadata"),
        "dataset_build_snapshot": dataset_build_snapshot,
        "matrix_report": matrix_report,
        "pipeline_fingerprint": metadata.get("pipeline_fingerprint") or dataset_build_snapshot.get("pipeline_fingerprint"),
        "implementation": runtime.get("implementation") or (training_meta or {}).get("implementation"),
        "device": runtime.get("device") or (training_meta or {}).get("device"),
        "device_label": runtime.get("device_label") or (training_meta or {}).get("device_label"),
        "gpu_name": runtime.get("gpu_name") or (training_meta or {}).get("gpu_name"),
        "fallback_reason": runtime.get("fallback_reason") or (training_meta or {}).get("fallback_reason"),
        "algorithm_parameters": dict(
            runtime.get("algorithm_parameters")
            or (training_meta or {}).get("algorithm_parameters")
            or {}
        ),
        "algorithm_runtime": runtime,
        "training_time_sec": round(float(training_time_sec), 2),
        "prediction_time_ms": (training_meta or {}).get("prediction_time_ms") or runtime.get("prediction_time_ms"),
    }
    # Dataset Engine / pandas load telemetry (Model Builder observation gate).
    if isinstance(metadata.get("dataset_load"), dict) and metadata["dataset_load"]:
        metadata_doc["dataset_load"] = dict(metadata["dataset_load"])
    if training_metadata:
        metadata_doc["training_metadata"] = training_metadata
    lc = config.lifecycle if isinstance(getattr(config, "lifecycle", None), dict) else None
    if lc:
        from .model_lifecycle import build_lineage_record, feature_snapshot_hash

        lineage = build_lineage_record(lc, child_model_name=model_name)
        if lineage:
            metadata_doc["lineage"] = lineage
        snap = lc.get("feature_snapshot") or list(config.features or [])
        if snap:
            metadata_doc["feature_snapshot"] = {
                "hash": lc.get("feature_snapshot_hash") or feature_snapshot_hash(list(snap)),
                "feature_count": len(snap),
                "snapshot_date": lc.get("snapshot_date"),
                "registry_version": lc.get("registry_version"),
            }
    # Phase 3C: Recommendation Decision Provenance Stamping
    rdb = getattr(config, "recommendation_decision_bundle", None)
    if isinstance(rdb, dict) and rdb:
        prov_map = rdb.get("feature_provenance") or {}
        trained_candidates = list(config.features or [])
        feature_snapshots = {}
        for feat in trained_candidates:
            p = prov_map.get(feat) or {}
            if p:
                feature_snapshots[feat] = {
                    "source": p.get("feature_source"),
                    "decision": p.get("decision"),
                    "primary_reason": p.get("primary_reason"),
                    "badges": p.get("reason_badges") or [],
                    "pre_train_evidence_score": p.get("evidence_score"),
                    "pre_train_confidence": p.get("evidence_confidence"),
                }
            else:
                feature_snapshots[feat] = {
                    "source": "unknown",
                    "decision": "TRAIN_CANDIDATE",
                    "primary_reason": "MANUALLY_INCLUDED",
                    "badges": [],
                    "pre_train_evidence_score": 0.0,
                    "pre_train_confidence": 0.0,
                }

        recommendation_provenance = {
            "has_recommendation_lineage": True,
            "context_id": rdb.get("context_id"),
            "market": rdb.get("market"),
            "sampling_interval_sec": rdb.get("sampling_interval_sec"),
            "sliding_window": rdb.get("sliding_window"),
            "feature_project_id": rdb.get("feature_project_id"),
            "originating_policy_id": rdb.get("policy_id"),
            "originating_policy_version": rdb.get("policy_version"),
            "decision_engine_version": rdb.get("decision_engine_version") or "3B.1",
            "handoff_timestamp": rdb.get("generated_at_ms"),
            "trained_candidates": trained_candidates,
            "selection_summary": {
                "eligible_candidates_count": rdb.get("eligible_candidates_count", len(trained_candidates)),
                "selected_candidates_count": len(trained_candidates),
                "review_count": rdb.get("review_count", 0),
                "unseen_count": rdb.get("unseen_count", 0),
                "excluded_count": rdb.get("excluded_count", 0),
            },
            "feature_decision_snapshots": feature_snapshots,
        }
        metadata_doc["recommendation_provenance"] = recommendation_provenance
        config_doc["recommendation_provenance"] = recommendation_provenance

    with open(os.path.join(paths["package_dir"], "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata_doc, fh, indent=2)
    with open(os.path.join(paths["package_dir"], "training_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_doc, fh, indent=2)

    manifest_doc = {
        "model_name": model_name,
        "trained_at": trained_at,
        "dataset": config.dataset,
        "target": config.target,
        "algorithm": config.algorithm,
        "feature_count": len(config.features),
        "features": list(config.features),
        "recommendation_provenance": metadata_doc.get("recommendation_provenance"),
    }
    with open(os.path.join(paths["package_dir"], "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest_doc, fh, indent=2)

    algo = normalize_algorithm(config.algorithm)
    native_path = native_model_path(paths["package_dir"], "model", algo)
    try:
        model.save_model(native_path)
    except Exception:
        pass
    if algo == "xgboost":
        model.save_model(paths["model_json"])

    with open(paths["metrics_json"], "w", encoding="utf-8") as fh:
        metrics_doc = dict(metrics)
        if training_meta:
            metrics_doc["training_meta"] = training_meta
        if validation_loss_curve:
            metrics_doc["validation_loss_curve"] = validation_loss_curve
        if walk_forward_summary:
            metrics_doc["walk_forward_summary"] = walk_forward_summary
        if shap_importance:
            metrics_doc["shap_importance"] = shap_importance
        json.dump(metrics_doc, fh, indent=2)

    _importance_csv_df(feature_importance).to_csv(paths["feature_importance_csv"], index=False)
    if shap_importance:
        with open(os.path.join(paths["package_dir"], "shap_importance.json"), "w", encoding="utf-8") as fh:
            json.dump(shap_importance, fh, indent=2)

    with open(paths["training_summary_json"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if training_metadata:
        with open(paths["training_metadata_json"], "w", encoding="utf-8") as fh:
            json.dump(training_metadata, fh, indent=2)

    log_text = (training_log_text or "").rstrip("\n")
    if log_text:
        log_text += "\n"
    log_text += f"{datetime.now(_IST).strftime('%H:%M:%S')} Model saved\n"
    with open(paths["training_log_txt"], "w", encoding="utf-8") as fh:
        fh.write(log_text)

    schema_hash = schema_registry_hash()
    val_hash = validation_rules_hash()
    with open(paths["schema_hash_txt"], "w", encoding="utf-8") as fh:
        fh.write(schema_hash)
    with open(paths["validation_hash_txt"], "w", encoding="utf-8") as fh:
        fh.write(val_hash)

    _copy_json_snapshot(schema_registry_path(), paths["schema_registry_json"], load_schema_registry)
    _copy_json_snapshot(validation_rules_path(), paths["validation_rules_json"], load_validation_rules)

    fingerprint = metadata.get("pipeline_fingerprint") or dataset_build_snapshot.get("pipeline_fingerprint")
    if not fingerprint:
        fingerprint = {
            "schema_registry_hash": schema_hash,
            "validation_rules_hash": val_hash,
            "implementation_hash": implementation_hash(),
            "training_pipeline": "chain_replay_ml.training",
        }
    with open(paths["pipeline_fingerprint_json"], "w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2)
    with open(paths["dataset_build_snapshot_json"], "w", encoding="utf-8") as fh:
        json.dump(dataset_build_snapshot, fh, indent=2)

    fi_records = feature_importance.to_dict(orient="records")
    report_html = build_training_report_html(
        summary=summary,
        config=config_doc,
        metrics=metrics,
        feature_importance=fi_records,
        pipeline_fingerprint=fingerprint,
        training_log=log_text,
    )
    with open(paths["training_report_html"], "w", encoding="utf-8") as fh:
        fh.write(report_html)

    test_m = metrics.get("test") or {}
    registry_doc = {
        "model_name": model_name,
        "dataset": config.dataset,
        "algorithm": algorithm_label(config.algorithm),
        "target": config.target,
        "status": "ready",
        "trained_at": trained_at,
        "validation_strategy_ui": strat_fields["validation_strategy_ui"],
        "validation_strategy_label": strat_fields["validation_strategy_label"],
        "device": metadata_doc.get("device_label") or metadata_doc.get("device"),
        "implementation": metadata_doc.get("implementation"),
        "training_time_sec": metadata_doc.get("training_time_sec"),
        "prediction_time_ms": metadata_doc.get("prediction_time_ms"),
        "metrics": {
            "rmse": test_m.get("rmse") or (metrics.get("validation") or {}).get("rmse"),
            "mae": test_m.get("mae") or (metrics.get("validation") or {}).get("mae"),
        },
        "feature_selection_method": elimination.get("method"),
        "feature_elimination": elimination,
        "training_metadata_available": bool(training_metadata),
    }
    with open(paths["registry_json"], "w", encoding="utf-8") as fh:
        json.dump(registry_doc, fh, indent=2)

    lifecycle_result: dict[str, Any] | None = None
    try:
        from .lifecycle_store import record_training_history

        lineage_doc = metadata_doc.get("lineage") if isinstance(metadata_doc.get("lineage"), dict) else None
        hpo_trials: int | None = None
        hpo_meta = metrics.get("hyperparameter_optimization")
        if isinstance(hpo_meta, dict):
            hpo_trials = int(hpo_meta.get("n_trials") or hpo_meta.get("n_trials_completed") or 0) or None
        opt = metrics.get("optimization_result")
        params_changed: int | None = None
        if isinstance(opt, dict) and isinstance(opt.get("parameter_changes"), list):
            params_changed = len(opt["parameter_changes"])
        lifecycle_result = record_training_history(
            data_dir=data_dir,
            model_name=model_name,
            trained_at=trained_at,
            config=config,
            metrics=metrics,
            metadata=metadata,
            matrix_report=matrix_report,
            lineage=lineage_doc,
            validation_strategy=str(split_info.get("strategy") or config.split.get("strategy") or ""),
            hpo_trials=hpo_trials,
            parameters_changed=params_changed,
        )
    except Exception:
        lifecycle_result = None

    return {
        "model_name": model_name,
        "package_dir": paths["package_dir"],
        "paths": paths,
        "registry": registry_doc,
        "training_summary": summary,
        "feature_importance": fi_records,
        "training_report_path": paths["training_report_html"],
        "lifecycle": lifecycle_result,
    }
