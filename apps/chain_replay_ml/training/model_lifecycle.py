"""Model lifecycle — reopen registry models in Model Builder for retrain / HPO / feature selection."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from .artifacts import feature_selection_method_label
from .registry import load_model_detail


LIFECYCLE_MODES = frozenset({
    "retrain",
    "complete_optimization",
    "feature_optimization",
    "calibration_only",
})


def _hpo_was_performed(detail: dict[str, Any]) -> bool:
    config = detail.get("config") if isinstance(detail.get("config"), dict) else {}
    split = config.get("split") if isinstance(config.get("split"), dict) else {}
    wf_cfg = split.get("walk_forward") if isinstance(split.get("walk_forward"), dict) else {}
    hpo_cfg = dict(split.get("hyperparameter_optimization") or wf_cfg.get("hyperparameter_optimization") or {})
    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), dict) else {}
    opt = metrics.get("optimization_result") if isinstance(metrics.get("optimization_result"), dict) else {}
    if hpo_cfg.get("enabled") is False and opt.get("enabled") is False:
        return False
    if hpo_cfg.get("enabled") is True or opt.get("enabled") is True:
        return True
    wf = detail.get("walk_forward") if isinstance(detail.get("walk_forward"), dict) else {}
    bp_art = wf.get("best_parameters") or {}
    bp = bp_art.get("data") if isinstance(bp_art.get("data"), dict) else {}
    trials = int(bp.get("n_trials_completed") or bp.get("n_trials_target") or 0)
    if trials > 0:
        return True
    hpo_meta = metrics.get("hyperparameter_optimization") if isinstance(metrics.get("hyperparameter_optimization"), dict) else {}
    return int(hpo_meta.get("n_trials") or 0) > 0


def hpo_status_summary(detail: dict[str, Any]) -> dict[str, Any]:
    """Registry UI summary for hyperparameter optimization state."""
    performed = _hpo_was_performed(detail)
    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), dict) else {}
    prod = detail.get("production_metrics") if isinstance(detail.get("production_metrics"), dict) else {}
    composite = prod.get("composite_score")
    if composite is None:
        composite = (metrics.get("composite_scores") or {}).get("production_composite")
    wf = detail.get("walk_forward") if isinstance(detail.get("walk_forward"), dict) else {}
    bp_art = wf.get("best_parameters") or {}
    bp = bp_art.get("data") if isinstance(bp_art.get("data"), dict) else {}
    best_trial = bp.get("trial_summary", {}).get("best_trial") if isinstance(bp.get("trial_summary"), dict) else None
    if best_trial is None:
        best_trial = bp.get("best_trial_number")
    return {
        "performed": performed,
        "status_label": "Completed" if performed else "Not Performed",
        "best_composite": composite,
        "best_trial": best_trial,
        "n_trials_completed": bp.get("n_trials_completed"),
        "baseline_parameters": _champion_parameters(detail),
    }


def _champion_parameters(detail: dict[str, Any]) -> dict[str, Any]:
    config = detail.get("config") if isinstance(detail.get("config"), dict) else {}
    params = dict(config.get("parameters") or {})
    wf = detail.get("walk_forward") if isinstance(detail.get("walk_forward"), dict) else {}
    bp_art = wf.get("best_parameters") or {}
    bp = bp_art.get("data") if isinstance(bp_art.get("data"), dict) else {}
    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), dict) else {}
    opt = metrics.get("optimization_result") if isinstance(metrics.get("optimization_result"), dict) else {}
    winner = str(opt.get("winner") or "").strip().lower()
    if winner == "tuned":
        tuned = bp.get("best_parameters") or bp.get("full_parameters")
        if isinstance(tuned, dict) and tuned:
            params.update(tuned)
    elif isinstance(bp.get("base_parameters"), dict):
        params.update(bp["base_parameters"])
    return params


def _validation_strategy_ui(detail: dict[str, Any]) -> str:
    strategy = detail.get("validation_strategy") or {}
    key = str(strategy.get("key") or "time_series_split")
    config = detail.get("config") if isinstance(detail.get("config"), dict) else {}
    split = config.get("split") if isinstance(config.get("split"), dict) else {}
    ui = str(split.get("validation_strategy_ui") or "").strip()
    if ui:
        return ui
    if key == "rolling_window":
        return "rolling_window"
    if key == "walk_forward":
        return "walk_forward"
    return "time_series_split"


def _bump_model_version(version: str) -> str:
    raw = str(version or "1.0").strip()
    m = re.match(r"^(\d+)(?:\.(\d+))?$", raw)
    if not m:
        return raw
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return f"{major}.{minor + 1}"


def resolve_family_model_name(detail: dict[str, Any], source_model: str) -> str:
    """Stable model family id — stays fixed across lifecycle versions."""
    meta_art = detail.get("metadata") if isinstance(detail.get("metadata"), dict) else {}
    meta = meta_art.get("data") if isinstance(meta_art.get("data"), dict) else {}
    if not meta:
        tm = detail.get("training_metadata") if isinstance(detail.get("training_metadata"), dict) else {}
        meta = tm
    lineage = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
    return str(lineage.get("ancestor_model_id") or source_model).strip()


def resolve_next_lifecycle_version(data_dir: str, source_model: str) -> tuple[int, str]:
    """Next version number/label for a lifecycle child of source_model."""
    from .lifecycle_store import get_history_by_model_name

    hist = get_history_by_model_name(data_dir, source_model)
    if hist and hist.get("version_number") is not None:
        n = int(hist["version_number"]) + 1
    else:
        n = 2
    return n, f"v{n}"


def _source_feature_selection_label(detail: dict[str, Any]) -> str:
    elim = detail.get("feature_elimination") if isinstance(detail.get("feature_elimination"), dict) else {}
    label = str(elim.get("method_label") or "").strip()
    if label:
        return label
    config = detail.get("config") if isinstance(detail.get("config"), dict) else {}
    split = config.get("split") if isinstance(config.get("split"), dict) else {}
    wf = split.get("walk_forward") if isinstance(split.get("walk_forward"), dict) else {}
    method = str(wf.get("feature_selection_method") or "rfe").strip().lower()
    if method == "none":
        return "Manual / Fixed Feature Set"
    return feature_selection_method_label(method)


def feature_snapshot_hash(features: list[str]) -> str:
    names = sorted({str(f).strip() for f in features if str(f).strip()})
    payload = "\n".join(names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8].upper()


def feature_group_breakdown(features: list[str]) -> list[dict[str, Any]]:
    """Count selected features per schema registry group."""
    from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry

    feature_set = {str(f).strip() for f in features if str(f).strip()}
    if not feature_set:
        return []
    schema = load_schema_registry()
    groups = schema.get("groups") or {}
    order = list(schema.get("groupOrder") or groups.keys())
    assigned: set[str] = set()
    rows: list[dict[str, Any]] = []
    for gid in order:
        block = groups.get(gid) or {}
        feats = [f for f in (block.get("features") or []) if f in feature_set]
        if not feats:
            continue
        assigned.update(feats)
        rows.append({
            "group_id": gid,
            "label": str(block.get("label") or gid),
            "count": len(feats),
        })
    for gid, block in groups.items():
        if gid in order:
            continue
        feats = [f for f in (block.get("features") or []) if f in feature_set and f not in assigned]
        if feats:
            assigned.update(feats)
            rows.append({
                "group_id": gid,
                "label": str(block.get("label") or gid),
                "count": len(feats),
            })
    other = sorted(feature_set - assigned)
    if other:
        rows.append({"group_id": "other", "label": "Other", "count": len(other)})
    rows.sort(key=lambda r: (-int(r["count"]), str(r["label"])))
    return rows


def _source_production_metrics(detail: dict[str, Any]) -> dict[str, Any]:
    prod = detail.get("production_metrics") if isinstance(detail.get("production_metrics"), dict) else {}
    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), dict) else {}
    block = prod if prod else (metrics.get("production_walk_forward") or metrics.get("validation") or {})
    if not isinstance(block, dict):
        block = {}
    composite = prod.get("composite_score")
    if composite is None:
        composite = (metrics.get("composite_scores") or {}).get("production_composite")
    return {
        "mae": block.get("mae"),
        "directional_accuracy_pct": block.get("directional_accuracy_pct"),
        "composite_score": composite,
        "rmse": block.get("rmse"),
    }


def _source_lineage(detail: dict[str, Any], source_model: str) -> dict[str, Any]:
    meta_art = detail.get("metadata") if isinstance(detail.get("metadata"), dict) else {}
    meta = meta_art.get("data") if isinstance(meta_art.get("data"), dict) else {}
    if not meta:
        tm = detail.get("training_metadata") if isinstance(detail.get("training_metadata"), dict) else {}
        meta = tm
    lineage = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
    ancestor = str(lineage.get("ancestor_model_id") or source_model).strip()
    generation = int(lineage.get("generation") or 0) + 1
    return {
        "parent_model_id": source_model,
        "ancestor_model_id": ancestor,
        "generation": generation,
    }


def build_lineage_record(lifecycle: dict[str, Any] | None, *, child_model_name: str = "") -> dict[str, Any] | None:
    """Persistable lineage for a newly trained lifecycle child model."""
    if not isinstance(lifecycle, dict) or not lifecycle.get("source_model"):
        return None
    parent = str(lifecycle.get("source_model") or "").strip()
    if not parent:
        return None
    ancestor = str(lifecycle.get("ancestor_model_id") or parent).strip()
    generation = int(lifecycle.get("generation") or 1)
    mode = str(lifecycle.get("mode") or "").strip().lower()
    return {
        "parent_model_id": parent,
        "ancestor_model_id": ancestor,
        "generation": generation,
        "lifecycle_mode": mode,
        "child_model_id": str(child_model_name or "").strip() or None,
    }


def lifecycle_diff_summary(mode: str, *, feature_count: int = 0) -> dict[str, list[str]]:
    """UI helper — what stays fixed vs changes for each lifecycle mode."""
    mode = str(mode or "").strip().lower()
    feat_label = f"Feature Set ({feature_count})" if feature_count else "Feature Set"
    if mode == "retrain":
        return {
            "same": [
                "Target",
                "Strike Selection",
                "Sampling Interval",
                "Prediction Type",
                "Algorithm",
                "Validation Strategy",
                "Feature Snapshot",
                "Feature Elimination (None)",
                "Hyperparameters",
                "Walk Forward",
            ],
            "changes": ["Dataset (compatible only)", "Model Version", "Training Timestamp"],
        }
    if mode == "complete_optimization":
        return {
            "same": ["Dataset", "Target", "Validation", "Algorithm", feat_label],
            "changes": ["Hyperparameters", "Model Version", "Training Timestamp"],
        }
    if mode == "feature_optimization":
        return {
            "same": ["Dataset", "Target", "Validation", "Algorithm", "Hyperparameters"],
            "changes": ["Prediction Type", "Feature Set", "Walk-Forward Feature Elimination", "Model Version", "Training Timestamp"],
        }
    return {"same": [], "changes": []}


def build_model_builder_preset(
    data_dir: str,
    model_name: str,
    mode: str = "complete_optimization",
) -> dict[str, Any]:
    """Build Model Builder preload payload from a saved training package."""
    mode = str(mode or "complete_optimization").strip().lower()
    if mode not in LIFECYCLE_MODES:
        raise ValueError(f"Unknown lifecycle mode: {mode}")
    if mode == "calibration_only":
        raise ValueError("Calibration-only lifecycle is not implemented yet.")

    detail = load_model_detail(data_dir, model_name)
    config = dict(detail.get("config") or {})
    split = dict(config.get("split") or {})
    wf_in = dict(split.get("walk_forward") or {})
    hpo_in = dict(split.get("hyperparameter_optimization") or wf_in.get("hyperparameter_optimization") or {})
    features = list(config.get("features") or [])
    if not features:
        elim = detail.get("feature_elimination") or {}
        features = list(elim.get("selected_features") or elim.get("final_features") or [])

    baseline_params = _champion_parameters(detail)
    hpo_status = hpo_status_summary(detail)
    selection_method = _source_feature_selection_label(detail)
    source_metrics = _source_production_metrics(detail)
    lineage_preview = _source_lineage(detail, model_name)
    group_breakdown = feature_group_breakdown(features)
    snap_hash = feature_snapshot_hash(features)
    snap_date = datetime.now(timezone.utc).date().isoformat()
    registry_version = (detail.get("registry") or {}).get("version")
    if registry_version is not None:
        registry_version = str(registry_version)
    source_version = str(config.get("model_version") or "1.0")
    family_name = resolve_family_model_name(detail, model_name)
    next_version_number, next_version_label = resolve_next_lifecycle_version(data_dir, model_name)
    source_feature_count = len(features)
    diff = lifecycle_diff_summary(mode, feature_count=len(features))

    training_config = {
        "dataset": str(config.get("dataset") or "").strip(),
        "target": str(config.get("target") or "").strip(),
        "algorithm": str(config.get("algorithm") or "xgboost").strip().lower(),
        "prediction_type": str(config.get("prediction_type") or "regression"),
        "features": features,
        "parameters": baseline_params,
        "model_name": family_name,
        "model_version": next_version_label,
        "model_description": (
            f"Lifecycle {mode.replace('_', ' ')} · {family_name} {next_version_label} "
            f"(from {model_name} {source_version})"
        ),
        "split": {
            "train": int(split.get("train") or 70),
            "validation": int(split.get("validation") or split.get("val") or 15),
            "test": int(split.get("test") or 15),
            "strategy": str(split.get("strategy") or "time_series"),
            "validation_strategy_ui": _validation_strategy_ui(detail),
            "hyperparameter_optimization": dict(hpo_in),
            "walk_forward": dict(wf_in),
        },
        "lifecycle": {
            "mode": mode,
            "source_model": model_name,
            "source_model_version": source_version,
            "family_model_name": family_name,
            "next_version_number": next_version_number,
            "next_version_label": next_version_label,
            "source_feature_count": source_feature_count,
            "baseline_parameters": baseline_params,
            "center_on_baseline": mode == "complete_optimization",
            "hpo_status": hpo_status,
            "selection_method": selection_method,
            "feature_snapshot": list(features),
            "feature_snapshot_hash": snap_hash,
            "snapshot_date": snap_date,
            "registry_version": registry_version,
            "source_metrics": source_metrics,
            "ancestor_model_id": lineage_preview["ancestor_model_id"],
            "generation": lineage_preview["generation"],
            "lifecycle_diff": diff,
            "feature_group_breakdown": group_breakdown,
        },
    }

    if mode == "retrain":
        training_config["split"]["hyperparameter_optimization"] = {
            **dict(hpo_in),
            "enabled": False,
        }
        if isinstance(training_config["split"].get("walk_forward"), dict):
            wf_copy = dict(training_config["split"]["walk_forward"])
            wf_copy["feature_selection_method"] = "none"
            wf_copy["hyperparameter_optimization"] = {
                **dict(hpo_in),
                "enabled": False,
            }
            training_config["split"]["walk_forward"] = wf_copy
        from .retrain_compatibility import build_retrain_profile_from_model

        retrain_profile = build_retrain_profile_from_model(data_dir, model_name)
        training_config["lifecycle"]["retrain_profile"] = retrain_profile
    elif mode == "complete_optimization":
        training_config["split"]["hyperparameter_optimization"] = {
            **dict(hpo_in),
            "enabled": True,
            "resume": False,
        }
        if isinstance(training_config["split"].get("walk_forward"), dict):
            wf_copy = dict(training_config["split"]["walk_forward"])
            wf_copy["feature_selection_method"] = "none"
            wf_copy["hyperparameter_optimization"] = {
                **dict(hpo_in),
                "enabled": True,
                "resume": False,
            }
            training_config["split"]["walk_forward"] = wf_copy
    elif mode == "feature_optimization":
        training_config["split"]["hyperparameter_optimization"] = {
            **dict(hpo_in),
            "enabled": False,
        }
        if isinstance(training_config["split"].get("walk_forward"), dict):
            wf_copy = dict(training_config["split"]["walk_forward"])
            wf_copy["hyperparameter_optimization"] = {
                **dict(hpo_in),
                "enabled": False,
            }
            if not str(wf_copy.get("feature_selection_method") or "").strip():
                wf_copy["feature_selection_method"] = "rfe"
            training_config["split"]["walk_forward"] = wf_copy

    return {
        "ok": True,
        "mode": mode,
        "source_model": model_name,
        "hpo_status": hpo_status,
        "source_metrics": source_metrics,
        "feature_count": len(features),
        "feature_group_breakdown": group_breakdown,
        "feature_snapshot_hash": snap_hash,
        "snapshot_date": snap_date,
        "lifecycle_diff": diff,
        "lineage_preview": lineage_preview,
        "training_config": training_config,
        "feature_importance": detail.get("feature_importance") or [],
        "registry_version": registry_version,
    }
