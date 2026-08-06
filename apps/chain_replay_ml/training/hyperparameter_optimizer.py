"""Optuna hyperparameter search using walk-forward validation objective."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .objective_scoring import (
    DEFAULT_HPO_SEEDS,
    composite_score_from_aggregated,
    compute_validation_reference_stats,
    display_score_for_ui,
    metric_display_label,
    objective_to_minimize,
    resolve_optimization_metric,
)
from .walk_forward_runner import evaluate_walk_forward_folds
from .wf_progress import build_hpo_progress_payload
from .xgb_trainer import TrainingCancelled

_HPO_SEARCH_KEYS = (
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "max_delta_step",
)

_STUDY_DB_NAME = "hpo_study.db"
_STUDY_NAME = "xgboost_hpo"


def _higher_better(metric_key: str) -> bool:
    return metric_key in ("composite", "directional_accuracy", "accuracy", "f1")


def _build_parameter_changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(_HPO_SEARCH_KEYS)
    rows: list[dict[str, Any]] = []
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        if b is None and a is None:
            continue
        rows.append({"parameter": key, "before": b, "after": a})
    return rows


def _build_trial_summary(
    history_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    best_trial_number: int,
) -> dict[str, Any]:
    if not history_rows:
        return {
            "completed": 0,
            "best_trial": None,
            "improved": 0,
            "no_change": 0,
            "worse": 0,
            "best_found_at_trial": None,
            "search_efficiency_pct": None,
        }
    higher = _higher_better(metric_key)
    improved = 0
    no_change = 0
    worse = 0
    running_best = None
    ordered = sorted(history_rows, key=lambda r: int(r.get("trial") or 0))
    for row in ordered:
        score = row.get("display_score")
        if score is None:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        if running_best is None:
            running_best = s
            improved += 1
            continue
        if (higher and s > running_best) or ((not higher) and s < running_best):
            improved += 1
            running_best = s
        elif s == running_best:
            no_change += 1
        else:
            worse += 1
    completed = len(ordered)
    best_at = int(best_trial_number) + 1
    eff = round((best_at / completed) * 100.0, 1) if completed > 0 else None
    return {
        "completed": completed,
        "best_trial": int(best_trial_number),
        "best_found_at_trial": best_at,
        "improved": improved,
        "no_change": no_change,
        "worse": worse,
        "search_efficiency_pct": eff,
    }


def _build_best_trial_reasons(
    history_rows: list[dict[str, Any]],
    *,
    best_trial_number: int,
    fold_count: int,
) -> list[str]:
    if not history_rows:
        return []
    by_trial = {int(r.get("trial") or 0): r for r in history_rows}
    best = by_trial.get(int(best_trial_number))
    if not best:
        return []
    reasons: list[str] = []
    try:
        rmse_vals = [float(r.get("mean_rmse")) for r in history_rows if r.get("mean_rmse") is not None]
        if rmse_vals and float(best.get("mean_rmse")) <= min(rmse_vals):
            reasons.append("Lowest RMSE")
    except Exception:
        pass
    try:
        mae_vals = [float(r.get("mean_mae")) for r in history_rows if r.get("mean_mae") is not None]
        if mae_vals and float(best.get("mean_mae")) <= min(mae_vals):
            reasons.append("Lowest MAE")
    except Exception:
        pass
    try:
        comp_vals = [float(r.get("composite_score")) for r in history_rows if r.get("composite_score") is not None]
        if comp_vals and float(best.get("composite_score")) >= max(comp_vals):
            reasons.append("Highest Composite Score")
    except Exception:
        pass
    if fold_count >= 3:
        reasons.append(f"Stable across {fold_count} folds")
    return reasons


def _trial_parameters(base: dict[str, Any], trial_params: dict[str, Any], *, fast: bool = True) -> dict[str, Any]:
    merged = dict(base)
    merged.update(trial_params)
    if fast:
        merged["n_estimators"] = int(base.get("hpo_n_estimators") or base.get("n_estimators_fast") or 400)
        merged["early_stopping_rounds"] = min(
            int(merged.get("early_stopping_rounds") or 100),
            50,
        )
    return merged


def _suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
    }


def _suggest_params_around_baseline(trial: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    """Narrow Optuna search around champion parameters (lifecycle complete optimization)."""
    lr = float(baseline.get("learning_rate") or 0.05)
    depth = int(baseline.get("max_depth") or 6)
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            max(0.005, lr * 0.5),
            min(0.35, lr * 2.0),
            log=True,
        ),
        "max_depth": trial.suggest_int("max_depth", max(3, depth - 2), min(12, depth + 2)),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            max(1.0, float(baseline.get("min_child_weight") or 1.0) * 0.5),
            min(20.0, float(baseline.get("min_child_weight") or 1.0) * 2.0),
        ),
        "subsample": trial.suggest_float(
            "subsample",
            max(0.5, float(baseline.get("subsample") or 0.8) - 0.15),
            min(1.0, float(baseline.get("subsample") or 0.8) + 0.15),
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            max(0.5, float(baseline.get("colsample_bytree") or 0.8) - 0.15),
            min(1.0, float(baseline.get("colsample_bytree") or 0.8) + 0.15),
        ),
        "gamma": trial.suggest_float(
            "gamma",
            max(0.0, float(baseline.get("gamma") or 0.0) - 1.0),
            min(5.0, float(baseline.get("gamma") or 0.0) + 1.0),
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            max(1e-8, float(baseline.get("reg_alpha") or 0.0) * 0.25 or 1e-8),
            min(10.0, max(1e-6, float(baseline.get("reg_alpha") or 0.0) * 4.0 or 1e-3)),
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            max(1e-3, float(baseline.get("reg_lambda") or 1.0) * 0.25),
            min(10.0, float(baseline.get("reg_lambda") or 1.0) * 4.0),
            log=True,
        ),
        "max_delta_step": trial.suggest_float(
            "max_delta_step",
            max(0.0, float(baseline.get("max_delta_step") or 0.0) - 2.0),
            min(10.0, float(baseline.get("max_delta_step") or 0.0) + 2.0),
        ),
    }


def _avg_aggregated(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        return {}
    out: dict[str, Any] = {"n_folds": aggregates[0].get("n_folds")}
    for key in (
        "mean_rmse", "mean_mae", "mean_directional_accuracy_pct",
        "mean_accuracy_pct", "mean_f1_pct",
        "std_rmse", "std_mae", "std_directional_accuracy_pct",
    ):
        vals = [float(a[key]) for a in aggregates if a.get(key) is not None]
        if vals:
            out[key] = round(float(np.mean(vals)), 6)
    return out


def _evaluate_multi_seed(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    fold_defs: list[dict[str, Any]],
    base_parameters: dict[str, Any],
    trial_params: dict[str, Any],
    seeds: list[int],
    metric_key: str,
    prediction_type: str,
    refs: dict[str, float],
    cancel_check: Callable[[], bool] | None,
    on_seed_progress: Callable[[int, int, int], None] | None = None,
    context_df: pd.DataFrame | None = None,
) -> tuple[float, dict[str, Any], list[float], list[int], dict[str, Any]]:
    seed_objectives: list[float] = []
    seed_aggregates: list[dict[str, Any]] = []
    seed_trees: list[float] = []
    seed_eval_sec: list[float] = []
    for seed_idx, seed in enumerate(seeds, start=1):
        if cancel_check and cancel_check():
            raise TrainingCancelled("Training cancelled")
        if on_seed_progress:
            on_seed_progress(seed_idx, int(seed), len(seeds))
        params = _trial_parameters(base_parameters, trial_params, fast=True)
        params["random_seed"] = int(seed)
        started = time.monotonic()
        wf_out = evaluate_walk_forward_folds(
            X=X,
            y=y,
            features=features,
            parameters=params,
            fold_defs=fold_defs,
            compute_shap=False,
            cancel_check=cancel_check,
            prediction_type=prediction_type,
            optimization_metric=metric_key,
            score_refs=refs,
            context_df=context_df,
        )
        aggregated = wf_out["aggregated"]
        fold_rows = wf_out.get("fold_results") or []
        tree_vals = [float(r.get("trees_trained")) for r in fold_rows if r.get("trees_trained") is not None]
        if tree_vals:
            seed_trees.append(float(np.mean(tree_vals)))
        seed_eval_sec.append(max(0.0, time.monotonic() - started))
        seed_aggregates.append(aggregated)
        seed_objectives.append(
            objective_to_minimize(
                aggregated,
                metric_key,
                prediction_type=prediction_type,
                refs=refs,
            )
        )
    avg_objective = float(np.mean(seed_objectives))
    extra = {
        "mean_trees_trained": round(float(np.mean(seed_trees)), 2) if seed_trees else None,
        "mean_eval_time_sec": round(float(np.mean(seed_eval_sec)), 3) if seed_eval_sec else None,
    }
    return avg_objective, _avg_aggregated(seed_aggregates), seed_objectives, seeds, extra


def _metric_before_after(before: dict[str, Any], after: dict[str, Any], metric_key: str) -> dict[str, Any]:
    b_rmse = before.get("mean_rmse")
    a_rmse = after.get("mean_rmse")
    b_mae = before.get("mean_mae")
    a_mae = after.get("mean_mae")
    b_dir = before.get("mean_directional_accuracy_pct")
    a_dir = after.get("mean_directional_accuracy_pct")
    b_comp = before.get("composite_score")
    a_comp = after.get("composite_score")
    b_trees = before.get("mean_trees_trained")
    a_trees = after.get("mean_trees_trained")
    b_time = before.get("mean_eval_time_sec")
    a_time = after.get("mean_eval_time_sec")

    def _pct_delta(old: Any, new: Any) -> float | None:
        try:
            o = float(old)
            n = float(new)
            if o == 0:
                return None
            return round(((n - o) / abs(o)) * 100.0, 2)
        except (TypeError, ValueError):
            return None

    return {
        "metric_key": metric_key,
        "before": {
            "rmse": b_rmse,
            "mae": b_mae,
            "directional_accuracy_pct": b_dir,
            "composite_score": b_comp,
            "mean_trees_trained": b_trees,
            "mean_eval_time_sec": b_time,
        },
        "after": {
            "rmse": a_rmse,
            "mae": a_mae,
            "directional_accuracy_pct": a_dir,
            "composite_score": a_comp,
            "mean_trees_trained": a_trees,
            "mean_eval_time_sec": a_time,
        },
        "change_pct": {
            "rmse": _pct_delta(b_rmse, a_rmse),
            "mae": _pct_delta(b_mae, a_mae),
            "directional_accuracy_pct": _pct_delta(b_dir, a_dir),
            "composite_score": _pct_delta(b_comp, a_comp),
            "mean_trees_trained": _pct_delta(b_trees, a_trees),
            "mean_eval_time_sec": _pct_delta(b_time, a_time),
        },
    }


def _display_score(objective: float, metric_key: str, aggregated: dict[str, Any], *, prediction_type: str) -> float:
    if metric_key == "composite":
        return round(composite_score_from_aggregated(aggregated, prediction_type=prediction_type), 6)
    if metric_key in ("directional_accuracy", "accuracy", "f1"):
        return round(-objective, 6)
    return round(objective, 6)


def _history_from_study(study: Any, metric_key: str, *, prediction_type: str) -> list[dict[str, Any]]:
    import optuna

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        attrs = dict(trial.user_attrs)
        aggregated = {
            "mean_rmse": attrs.get("mean_rmse"),
            "mean_mae": attrs.get("mean_mae"),
            "mean_directional_accuracy_pct": attrs.get("mean_directional_accuracy_pct"),
            "mean_accuracy_pct": attrs.get("mean_accuracy_pct"),
            "mean_f1_pct": attrs.get("mean_f1_pct"),
        }
        row: dict[str, Any] = {
            "trial": trial.number,
            "objective": round(float(trial.value), 6),
            "display_score": attrs.get("display_score"),
            "composite_score": attrs.get("composite_score"),
            "mean_rmse": attrs.get("mean_rmse"),
            "mean_mae": attrs.get("mean_mae"),
            "mean_directional_accuracy_pct": attrs.get("mean_directional_accuracy_pct"),
            "mean_accuracy_pct": attrs.get("mean_accuracy_pct"),
            "mean_f1_pct": attrs.get("mean_f1_pct"),
            "seed_scores": attrs.get("seed_scores"),
            "validation_seeds": attrs.get("validation_seeds"),
            "mean_trees_trained": attrs.get("mean_trees_trained"),
            "mean_eval_time_sec": attrs.get("mean_eval_time_sec"),
        }
        for k in _HPO_SEARCH_KEYS:
            if k in trial.params:
                row[k] = trial.params[k]
        if row.get("display_score") is None:
            row["display_score"] = _display_score(float(trial.value), metric_key, aggregated, prediction_type=prediction_type)
        rows.append(row)
    return rows


def _top_trials_rows(study: Any, metric_key: str, *, prediction_type: str, top_k: int = 10) -> list[dict[str, Any]]:
    import optuna

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    complete.sort(key=lambda t: float(t.value))
    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(complete[:top_k], start=1):
        attrs = dict(trial.user_attrs)
        aggregated = {
            "mean_rmse": attrs.get("mean_rmse"),
            "mean_mae": attrs.get("mean_mae"),
            "mean_directional_accuracy_pct": attrs.get("mean_directional_accuracy_pct"),
        }
        row: dict[str, Any] = {
            "rank": rank,
            "trial": trial.number,
            "objective": round(float(trial.value), 6),
            "score": attrs.get("display_score") or _display_score(float(trial.value), metric_key, aggregated, prediction_type=prediction_type),
            "mean_rmse": attrs.get("mean_rmse"),
            "mean_mae": attrs.get("mean_mae"),
            "mean_directional_accuracy_pct": attrs.get("mean_directional_accuracy_pct"),
            "composite_score": attrs.get("composite_score"),
            "mean_trees_trained": attrs.get("mean_trees_trained"),
            "mean_eval_time_sec": attrs.get("mean_eval_time_sec"),
        }
        for k in _HPO_SEARCH_KEYS:
            if k in trial.params:
                row[k] = trial.params[k]
        rows.append(row)
    return rows


def _save_hpo_artifacts(
    artifacts_dir: str,
    *,
    best_search_params: dict[str, Any],
    full_parameters: dict[str, Any],
    history_rows: list[dict[str, Any]],
    top_trials: list[dict[str, Any]],
    param_importance: dict[str, float],
    meta: dict[str, Any],
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "best_parameters.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_parameters": best_search_params,
                "full_parameters": full_parameters,
                **meta,
            },
            fh,
            indent=2,
        )
    pd.DataFrame(history_rows).to_csv(
        os.path.join(artifacts_dir, "optimization_history.csv"), index=False
    )
    pd.DataFrame(top_trials).to_csv(
        os.path.join(artifacts_dir, "top_trials.csv"), index=False
    )
    pd.DataFrame(
        [{"parameter": k, "importance": v} for k, v in sorted(param_importance.items(), key=lambda x: -x[1])]
    ).to_csv(os.path.join(artifacts_dir, "parameter_importance.csv"), index=False)


def optimize_xgboost_hyperparameters(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    fold_defs: list[dict[str, Any]],
    base_parameters: dict[str, Any],
    optimization_metric: str = "auto",
    prediction_type: str = "regression",
    n_trials: int = 25,
    validation_seeds: list[int] | None = None,
    resume: bool = True,
    artifacts_dir: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_trial: Callable[[dict[str, Any]], None] | None = None,
    center_on_baseline: bool = False,
    context_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Search XGBoost hyperparameters; objective = walk-forward validation performance."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for hyperparameter optimization. pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric_key = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)
    seeds = [int(s) for s in (validation_seeds or DEFAULT_HPO_SEEDS)]
    refs = compute_validation_reference_stats(y, fold_defs)

    db_path = os.path.join(artifacts_dir, _STUDY_DB_NAME) if artifacts_dir else None
    if db_path and not resume and os.path.exists(db_path):
        os.remove(db_path)

    storage = f"sqlite:///{db_path}" if db_path else None
    study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=storage,
        load_if_exists=bool(resume and storage),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(base_parameters.get("random_seed", 42))),
    )

    completed_before = len([
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ])
    target_trials = max(1, int(n_trials))
    remaining = max(0, target_trials - completed_before)
    hpo_best_display: dict[str, Any] = {"value": None, "trial": None, "mae": None, "direction_pct": None}
    baseline_objective, baseline_aggregated, _baseline_seed_scores, _baseline_seeds, baseline_extra = _evaluate_multi_seed(
        X=X,
        y=y,
        features=features,
        fold_defs=fold_defs,
        base_parameters=base_parameters,
        trial_params={},
        seeds=seeds,
        metric_key=metric_key,
        prediction_type=prediction_type,
        refs=refs,
        cancel_check=cancel_check,
        on_seed_progress=None,
        context_df=context_df,
    )
    baseline_composite = composite_score_from_aggregated(baseline_aggregated, prediction_type=prediction_type, refs=refs)
    baseline_display = _display_score(baseline_objective, metric_key, baseline_aggregated, prediction_type=prediction_type)
    baseline_eval = {
        "objective": round(float(baseline_objective), 6),
        "display_score": baseline_display,
        "composite_score": baseline_composite if metric_key == "composite" else None,
        **baseline_aggregated,
        **baseline_extra,
    }

    def objective(trial: optuna.Trial) -> float:
        if cancel_check and cancel_check():
            raise TrainingCancelled("Training cancelled")
        if center_on_baseline:
            trial_params = _suggest_params_around_baseline(trial, base_parameters)
        else:
            trial_params = _suggest_params(trial)

        def _seed_progress(seed_idx: int, seed: int, n_seeds: int) -> None:
            if on_trial:
                on_trial(build_hpo_progress_payload(
                    trial=trial.number + 1,
                    n_trials=target_trials,
                    seed=seed,
                    seed_index=seed_idx,
                    n_seeds=n_seeds,
                    feature_count=len(features),
                    resumed=completed_before > 0,
                ))

        avg_objective, aggregated, seed_scores, used_seeds, extra = _evaluate_multi_seed(
            X=X,
            y=y,
            features=features,
            fold_defs=fold_defs,
            base_parameters=base_parameters,
            trial_params=trial_params,
            seeds=seeds,
            metric_key=metric_key,
            prediction_type=prediction_type,
            refs=refs,
            cancel_check=cancel_check,
            on_seed_progress=_seed_progress,
            context_df=context_df,
        )
        composite = composite_score_from_aggregated(aggregated, prediction_type=prediction_type, refs=refs)
        display = _display_score(avg_objective, metric_key, aggregated, prediction_type=prediction_type)
        trial.set_user_attr("mean_rmse", aggregated.get("mean_rmse"))
        trial.set_user_attr("mean_mae", aggregated.get("mean_mae"))
        trial.set_user_attr("mean_directional_accuracy_pct", aggregated.get("mean_directional_accuracy_pct"))
        trial.set_user_attr("mean_accuracy_pct", aggregated.get("mean_accuracy_pct"))
        trial.set_user_attr("mean_f1_pct", aggregated.get("mean_f1_pct"))
        trial.set_user_attr("composite_score", composite if metric_key == "composite" else None)
        trial.set_user_attr("display_score", display)
        trial.set_user_attr("seed_scores", seed_scores)
        trial.set_user_attr("validation_seeds", used_seeds)
        trial.set_user_attr("mean_trees_trained", extra.get("mean_trees_trained"))
        trial.set_user_attr("mean_eval_time_sec", extra.get("mean_eval_time_sec"))
        higher_better = metric_key in ("composite", "directional_accuracy", "accuracy", "f1")
        prev_best = hpo_best_display["value"]
        improved = False
        if prev_best is None:
            hpo_best_display["value"] = display
            improved = True
        elif higher_better and display is not None and display > prev_best:
            hpo_best_display["value"] = display
            improved = True
        elif not higher_better and display is not None and display < prev_best:
            hpo_best_display["value"] = display
            improved = True
        if improved:
            hpo_best_display["trial"] = trial.number + 1
            hpo_best_display["mae"] = aggregated.get("mean_mae")
            hpo_best_display["direction_pct"] = aggregated.get("mean_directional_accuracy_pct")
        if on_trial:
            on_trial({
                "trial": trial.number + 1,
                "n_trials": target_trials,
                "objective": avg_objective,
                "display_score": display,
                "current_display_score": display,
                "best_display_score": hpo_best_display["value"],
                "best_trial": hpo_best_display.get("trial"),
                "baseline_display_score": baseline_display,
                "baseline_composite_score": baseline_composite if metric_key == "composite" else baseline_display,
                "best_mean_mae": hpo_best_display.get("mae"),
                "best_mean_directional_accuracy_pct": hpo_best_display.get("direction_pct"),
                "metric_label": metric_display_label(metric_key),
                "mean_rmse": aggregated.get("mean_rmse"),
                "mean_mae": aggregated.get("mean_mae"),
                "mean_directional_accuracy_pct": aggregated.get("mean_directional_accuracy_pct"),
                "resumed": completed_before > 0,
            })
        return avg_objective

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_trial = study.best_trial
    best_eval = {
        "objective": round(float(study.best_value), 6),
        "display_score": best_trial.user_attrs.get("display_score"),
        "composite_score": best_trial.user_attrs.get("composite_score"),
        "mean_rmse": best_trial.user_attrs.get("mean_rmse"),
        "mean_mae": best_trial.user_attrs.get("mean_mae"),
        "mean_directional_accuracy_pct": best_trial.user_attrs.get("mean_directional_accuracy_pct"),
        "mean_accuracy_pct": best_trial.user_attrs.get("mean_accuracy_pct"),
        "mean_f1_pct": best_trial.user_attrs.get("mean_f1_pct"),
        "mean_trees_trained": best_trial.user_attrs.get("mean_trees_trained"),
        "mean_eval_time_sec": best_trial.user_attrs.get("mean_eval_time_sec"),
    }
    before_after = _metric_before_after(baseline_eval, best_eval, metric_key)
    best_search = {k: best_trial.params[k] for k in _HPO_SEARCH_KEYS if k in best_trial.params}
    best_parameters = _trial_parameters(base_parameters, best_search, fast=False)

    param_importance: dict[str, float] = {}
    try:
        from optuna.importance import get_param_importances

        param_importance = get_param_importances(study)
    except Exception:
        param_importance = {}

    history_rows = _history_from_study(study, metric_key, prediction_type=prediction_type)
    top_trials = _top_trials_rows(study, metric_key, prediction_type=prediction_type, top_k=10)
    parameter_changes = _build_parameter_changes(base_parameters, best_parameters)
    trial_summary = _build_trial_summary(
        history_rows,
        metric_key=metric_key,
        best_trial_number=best_trial.number,
    )
    best_trial_reasons = _build_best_trial_reasons(
        history_rows,
        best_trial_number=best_trial.number,
        fold_count=len(fold_defs),
    )

    meta = {
        "optimizer": "optuna_tpe",
        "n_trials_target": target_trials,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_resumed_from": completed_before,
        "optimization_metric": metric_key,
        "optimization_metric_requested": optimization_metric,
        "best_objective": study.best_value,
        "best_display_score": best_trial.user_attrs.get("display_score"),
        "best_trial": best_trial.number,
        "validation_seeds": seeds,
        "multi_seed_averaging": True,
        "resume_enabled": bool(resume and storage),
        "composite_weights_regression": {"directional_accuracy": 0.60, "rmse": -0.25, "mae": -0.15},
        "composite_weights_classification": {"accuracy": 0.40, "f1": 0.35, "directional_accuracy": 0.25},
        "reference_stats": refs,
        "baseline_evaluation": baseline_eval,
        "best_evaluation": best_eval,
        "before_after_comparison": before_after,
        "parameter_changes": parameter_changes,
        "trial_summary": trial_summary,
        "best_trial_reasons": best_trial_reasons,
        "base_parameters": base_parameters,
    }

    if artifacts_dir:
        _save_hpo_artifacts(
            artifacts_dir,
            best_search_params=best_search,
            full_parameters=best_parameters,
            history_rows=history_rows,
            top_trials=top_trials,
            param_importance=param_importance,
            meta=meta,
        )

    return {
        "best_parameters": best_parameters,
        "best_search_params": best_search,
        "optimization_metric": metric_key,
        "best_objective": study.best_value,
        "best_display_score": best_trial.user_attrs.get("display_score"),
        "n_trials": meta["n_trials_completed"],
        "n_trials_target": target_trials,
        "n_trials_resumed_from": completed_before,
        "history": history_rows,
        "top_trials": top_trials,
        "parameter_importance": param_importance,
        "validation_seeds": seeds,
        "optimizer": meta["optimizer"],
        "best_trial": best_trial.number,
        "baseline_evaluation": baseline_eval,
        "best_evaluation": best_eval,
        "before_after_comparison": before_after,
        "parameter_changes": parameter_changes,
        "trial_summary": trial_summary,
        "best_trial_reasons": best_trial_reasons,
        "base_parameters": base_parameters,
    }
