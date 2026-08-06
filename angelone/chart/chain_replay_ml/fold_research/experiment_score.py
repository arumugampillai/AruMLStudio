"""Experiment score — rank proposals and templates before Run."""

from __future__ import annotations

from typing import Any


def _score_from_label(label: str) -> int:
    mapping = {
        "High": 85,
        "Medium-High": 72,
        "Medium": 58,
        "Medium-Low": 45,
        "Low": 30,
    }
    return mapping.get(label, 50)


def _time_estimate_minutes(accepted_items: list[dict[str, Any]]) -> dict[str, Any]:
    targets = {str(i.get("target") or "") for i in accepted_items}
    needs_train = "model_builder" in targets or "feature_registry" in targets
    needs_dataset = "master_dataset" in targets or "dataset_migration" in targets
    needs_hpo = "hyperparameter_optimization" in targets
    gpu_min = 0
    cpu_min = 2
    if needs_dataset:
        cpu_min += 15
    if needs_train:
        gpu_min += 35
        cpu_min += 5
    if needs_hpo:
        gpu_min += 60
    if not needs_train and not needs_dataset:
        gpu_min = 0
        cpu_min = max(cpu_min, 4)
    wf_folds = 10 if needs_train else 0
    total = gpu_min + cpu_min
    if total >= 60:
        cost = "High"
    elif total >= 25:
        cost = "Medium"
    else:
        cost = "Low"
    return {
        "estimated_minutes": total,
        "gpu_minutes": gpu_min,
        "cpu_minutes": cpu_min,
        "dataset_build": needs_dataset,
        "training": needs_train,
        "walk_forward_folds": wf_folds,
        "gpu_cost": cost,
    }


def derive_experiment_tags(accepted_items: list[dict[str, Any]], *, goal: str = "") -> list[str]:
    tags: set[str] = set()
    blob = goal.lower()
    for item in accepted_items:
        text = str(item.get("text") or "").lower()
        target = str(item.get("target") or "")
        blob += " " + text
        if "premium" in text:
            tags.add("premium")
        if "confidence" in text:
            tags.add("confidence")
        if "theta" in text:
            tags.add("theta")
        if "iv" in text:
            tags.add("iv")
        if "range" in text or "stop" in text:
            tags.add("risk")
        if target == "strategy_registry":
            tags.add("strategy")
        if target in ("feature_registry", "master_dataset", "dataset_migration"):
            tags.add("feature")
        if target == "model_builder":
            tags.add("retrain")
        if target == "hyperparameter_optimization":
            tags.add("optimization")
    if "theta" in blob:
        tags.add("theta")
    if "premium" in blob:
        tags.add("premium")
    return sorted(tags)


def compute_experiment_score(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    from .knowledge_retrieval import score_experiment_proposal

    scored = score_experiment_proposal(
        data_dir,
        report,
        accepted_items=accepted_items,
        goal=goal,
    )
    novelty = int(float(scored.get("novelty_score") or 50))
    evidence_quality = str(scored.get("evidence_quality") or "weak")
    evidence_map = {"strong": 85, "moderate": 65, "weak": 40}
    evidence_strength = evidence_map.get(evidence_quality, 45)
    expected_gain = _score_from_label(str(scored.get("improvement_probability") or "Medium"))
    timing = _time_estimate_minutes(accepted_items)
    time_penalty = min(15, timing["estimated_minutes"] // 4)
    overall = int(max(
        0,
        min(100, novelty * 0.28 + evidence_strength * 0.28 + expected_gain * 0.34 - time_penalty * 0.1),
    ))
    stars = max(1, min(5, round(overall / 20)))
    duplicate = scored.get("verdict") == "very_similar" and scored.get("should_warn")
    recommendation = "Recommended" if overall >= 75 and not duplicate else (
        "Likely Duplicate" if duplicate else "Review"
    )
    return {
        "novelty": novelty,
        "evidence_strength": evidence_strength,
        "expected_gain": expected_gain,
        "estimated_minutes": timing["estimated_minutes"],
        "gpu_minutes": timing["gpu_minutes"],
        "cpu_minutes": timing["cpu_minutes"],
        "dataset_build": timing["dataset_build"],
        "training": timing["training"],
        "walk_forward_folds": timing["walk_forward_folds"],
        "gpu_cost": timing["gpu_cost"],
        "overall": overall,
        "stars": stars,
        "recommendation": recommendation,
        "duplicate_check": scored,
        "tags": derive_experiment_tags(accepted_items, goal=goal or ""),
    }
