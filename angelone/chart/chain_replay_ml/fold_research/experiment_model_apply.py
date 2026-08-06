"""Apply experiment model / feature changes — Phase C training path."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from chain_replay_ml.training.model_lifecycle import build_model_builder_preset
from chain_replay_ml.training.orchestrator import train_model
from chain_replay_ml.training.paths import model_package_dir

HINT_FEATURE_CANDIDATES: dict[str, list[str]] = {
    "theta": ["theta", "theta_per_min", "abs_theta"],
    "iv": ["current_iv", "iv", "implied_vol", "iv_expansion"],
}


def infer_lifecycle_mode(routing: dict[str, Any]) -> str:
    if routing.get("feature_changes"):
        return "feature_optimization"
    if routing.get("optimization_changes") and not routing.get("model_changes"):
        return "complete_optimization"
    if routing.get("model_changes"):
        return "retrain"
    if routing.get("optimization_changes"):
        return "complete_optimization"
    return "retrain"


def _collect_feature_hints(feature_changes: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for change in feature_changes:
        for hint in change.get("feature_hints") or []:
            h = str(hint or "").strip().lower()
            if h and h not in hints:
                hints.append(h)
    return hints


def _match_features_for_hint(hint: str, candidates: list[str]) -> list[str]:
    hint = hint.lower()
    patterns = HINT_FEATURE_CANDIDATES.get(hint, [hint])
    matched: list[str] = []
    for feat in candidates:
        low = feat.lower()
        if any(p in low for p in patterns):
            matched.append(feat)
    return matched


def resolve_feature_hints(
    data_dir: str,
    dataset_name: str,
    hints: list[str],
    *,
    existing_features: list[str],
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_merge_ops import plan_feature_merge

    notes: list[str] = []
    if not hints:
        return {
            "features_to_merge": [],
            "features": list(existing_features),
            "notes": notes,
        }

    try:
        plan = plan_feature_merge(data_dir, dataset_name)
    except Exception as exc:
        notes.append(f"Feature merge planning skipped: {exc}")
        return {"features_to_merge": [], "features": list(existing_features), "notes": notes}

    merge_candidates = [str(c.get("name") or c.get("feature") or "") for c in (plan.get("merge_candidates") or [])]
    merge_candidates = [c for c in merge_candidates if c]
    all_names = sorted(set(existing_features) | set(merge_candidates))

    to_merge: list[str] = []
    updated = list(existing_features)
    for hint in hints:
        matched = _match_features_for_hint(hint, all_names)
        if not matched:
            notes.append(f"No registry feature matched hint '{hint}'.")
            continue
        for feat in matched:
            if feat not in updated:
                updated.append(feat)
            if feat in merge_candidates and feat not in existing_features and feat not in to_merge:
                to_merge.append(feat)

    return {
        "features_to_merge": to_merge,
        "features": updated,
        "notes": notes,
    }


def apply_feature_changes_for_template(
    data_dir: str,
    template: dict[str, Any],
    *,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    routing = template.get("routing") or {}
    feature_changes = routing.get("feature_changes") or []
    hints = _collect_feature_hints(feature_changes)
    dataset_name = str(training_config.get("dataset") or "")
    existing = list(training_config.get("features") or [])
    resolved = resolve_feature_hints(
        data_dir,
        dataset_name,
        hints,
        existing_features=existing,
    )
    notes = list(resolved.get("notes") or [])
    features = list(resolved.get("features") or existing)
    to_merge = list(resolved.get("features_to_merge") or [])

    if to_merge and dataset_name:
        from chain_replay_ml.dataset_builder.feature_merge_ops import merge_features_into_dataset

        try:
            merge_result = merge_features_into_dataset(data_dir, dataset_name, to_merge)
            notes.append(f"Merged {len(to_merge)} feature column(s) into {dataset_name}.")
            if not merge_result.get("ok", True):
                notes.append(str(merge_result.get("error") or "Feature merge reported issues."))
        except Exception as exc:
            notes.append(f"Feature merge failed: {exc}")

    training_config = dict(training_config)
    training_config["features"] = features
    return {
        "training_config": training_config,
        "merged_features": to_merge,
        "notes": notes,
    }


def clone_training_config_for_template(
    data_dir: str,
    template: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_id = str(template.get("model_id") or "")
    if not model_id:
        raise ValueError("template missing baseline model_id")

    routing = template.get("routing") or {}
    mode = infer_lifecycle_mode(routing)
    preset = build_model_builder_preset(data_dir, model_id, mode)
    if not preset.get("ok"):
        raise ValueError(preset.get("error") or f"failed to build preset for {model_id}")

    training_config = dict(preset.get("training_config") or {})
    if overrides:
        training_config.update({k: v for k, v in overrides.items() if k != "split"})
        if overrides.get("split"):
            training_config["split"] = {
                **dict(training_config.get("split") or {}),
                **dict(overrides["split"]),
            }

    feature_prep = apply_feature_changes_for_template(data_dir, template, training_config=training_config)
    training_config = feature_prep["training_config"]
    notes = list(feature_prep.get("notes") or [])
    if routing.get("dataset_changes"):
        notes.append("Dataset migration deferred — rebuild manually or wait for Phase D.")

    return {
        "mode": mode,
        "source_model": model_id,
        "training_config": training_config,
        "merged_features": feature_prep.get("merged_features") or [],
        "notes": notes,
    }


def read_prediction_run_id(package_dir: str) -> str | None:
    path = os.path.join(package_dir, "prediction_run.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        rid = doc.get("run_id")
        return str(rid) if rid else None

    agg_path = os.path.join(package_dir, "walk_forward", "champion_aggregate.json")
    if os.path.isfile(agg_path):
        with open(agg_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        pr = doc.get("prediction_run") or {}
        rid = pr.get("run_id")
        return str(rid) if rid else None
    return None


def run_training_for_template(
    data_dir: str,
    template: dict[str, Any],
    training_config: dict[str, Any],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    result = train_model(
        data_dir=data_dir,
        raw_config=training_config,
        on_progress=on_progress,
        cancel_check=cancel_check,
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "blocked": result.get("blocked"),
            "validation": result.get("validation"),
            "error": result.get("error") or "training failed",
            "training_result": result,
        }

    model_name = str(result.get("model_name") or training_config.get("model_name") or "")
    package_dir = str(result.get("package_dir") or model_package_dir(data_dir, model_name))
    prediction_run_id = read_prediction_run_id(package_dir)
    if not prediction_run_id:
        return {
            "ok": False,
            "error": "training completed but prediction_run_id was not found",
            "training_result": result,
            "model_name": model_name,
            "package_dir": package_dir,
        }

    return {
        "ok": True,
        "model_name": model_name,
        "package_dir": package_dir,
        "prediction_run_id": prediction_run_id,
        "training_result": result,
    }


def prepare_model_training_for_template(
    data_dir: str,
    template: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return clone_training_config_for_template(data_dir, template, overrides=overrides)
