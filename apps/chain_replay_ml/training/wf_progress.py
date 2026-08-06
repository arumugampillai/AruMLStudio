"""Walk-forward / HPO progress helpers for WebSocket UI."""

from __future__ import annotations

from typing import Any

FOLD_STAGES = ("train_model", "validation", "shap_importance")
FS_STAGE = "feature_selection"
HPO_STAGE = "hpo_trial"
WF_PHASE_FOLDS = "fold_validation"
WF_PHASE_FEATURE_SELECTION = "feature_selection"
WF_PHASE_HPO = "hyperparameter_optimization"


def fold_stage_index(stage: str) -> int:
    try:
        return FOLD_STAGES.index(stage)
    except ValueError:
        return 0


def compute_wf_overall_pct(
    *,
    fold: int,
    n_folds: int,
    stage: str,
    fs_iteration: int = 0,
    fs_total_iterations: int = 0,
    in_feature_selection: bool = False,
) -> int:
    """0–100 overall progress through walk-forward validation + feature selection."""
    n_folds = max(1, int(n_folds))
    if in_feature_selection:
        fs_total = max(1, int(fs_total_iterations) or 1)
        fs_frac = min(1.0, int(fs_iteration) / fs_total)
        return int(round(90 + fs_frac * 10))
    stage_i = fold_stage_index(stage) + 1
    completed = max(0, (int(fold) - 1) * len(FOLD_STAGES) + stage_i)
    total = n_folds * len(FOLD_STAGES)
    return int(round(min(89, (completed / total) * 90)))


def build_wf_progress_payload(
    *,
    fold: int | None = None,
    n_folds: int | None = None,
    wf_stage: str,
    feature_count: int | None = None,
    fs_iteration: int | None = None,
    fs_total_iterations: int | None = None,
    in_feature_selection: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    phase = WF_PHASE_FEATURE_SELECTION if in_feature_selection else WF_PHASE_FOLDS
    if wf_stage == HPO_STAGE:
        phase = WF_PHASE_HPO
    payload: dict[str, Any] = {"wf_stage": wf_stage, "wf_phase": phase, **extra}
    if fold is not None:
        payload["fold"] = fold
    if n_folds is not None:
        payload["n_folds"] = n_folds
    if feature_count is not None:
        payload["feature_count"] = feature_count
        payload["current_features"] = feature_count
    if fs_iteration is not None:
        payload["fs_iteration"] = fs_iteration
        payload["current_iteration"] = fs_iteration
    if fs_total_iterations is not None:
        payload["fs_total_iterations"] = fs_total_iterations
    if fold is not None and n_folds is not None:
        payload["wf_overall_pct"] = compute_wf_overall_pct(
            fold=fold,
            n_folds=n_folds,
            stage=wf_stage,
            fs_iteration=fs_iteration or 0,
            fs_total_iterations=fs_total_iterations or 0,
            in_feature_selection=in_feature_selection,
        )
    elif in_feature_selection:
        payload["wf_overall_pct"] = compute_wf_overall_pct(
            fold=1,
            n_folds=1,
            stage=FS_STAGE,
            fs_iteration=fs_iteration or 0,
            fs_total_iterations=fs_total_iterations or 1,
            in_feature_selection=True,
        )
    return payload


def build_hpo_progress_payload(
    *,
    trial: int,
    n_trials: int,
    seed: int | None = None,
    seed_index: int | None = None,
    n_seeds: int | None = None,
    feature_count: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Progress payload for hyperparameter optimization step."""
    n_trials = max(1, int(n_trials))
    n_seeds = max(1, int(n_seeds or 1))
    trial_i = max(0, int(trial) - 1)
    seed_i = max(0, int(seed_index or 1) - 1)
    units_done = trial_i * n_seeds + seed_i
    units_total = n_trials * n_seeds
    overall = int(round((units_done / units_total) * 100))
    payload: dict[str, Any] = {
        "wf_stage": HPO_STAGE,
        "wf_phase": WF_PHASE_HPO,
        "trial": trial,
        "n_trials": n_trials,
        "current_trial": trial,
        "hpo_overall_pct": overall,
        **extra,
    }
    if seed is not None:
        payload["current_seed"] = seed
    if seed_index is not None:
        payload["seed_index"] = seed_index
    if n_seeds is not None:
        payload["n_seeds"] = n_seeds
    if feature_count is not None:
        payload["feature_count"] = feature_count
        payload["current_features"] = feature_count
    return payload
