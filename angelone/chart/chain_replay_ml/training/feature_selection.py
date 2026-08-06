"""Production-grade validation-driven feature selection after walk-forward folds."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .evaluator import evaluate_predictions, resolve_ltp_baseline_from_frames
from .objective_scoring import (
    OPTIMIZATION_METRICS,
    display_score_for_ui,
    format_stopped_reason,
    metric_display_label,
    metric_score_for_selection as _metric_score,
    resolve_optimization_metric,
)
from .shap_importance import compute_shap_importance
from .wf_progress import FS_STAGE, build_wf_progress_payload
from .boost_trainer import train_regressor

FEATURE_SELECTION_METHODS = ("none", "shap", "rfe", "permutation")

_PROBE_PARAMS = {"n_estimators": 250, "early_stopping_rounds": 30}
_NO_IMPROVE_LIMIT = 5


def _refs_from_validation_y(val_y: pd.Series) -> dict[str, float]:
    vals = val_y.astype(float)
    return {
        "rmse_ref": max(float(np.std(vals)), float(np.mean(np.abs(vals))) * 0.25, 1e-6),
        "mae_ref": max(float(np.mean(np.abs(vals))), 1e-6),
    }


def _is_better(score: float, best_score: float) -> bool:
    return score < best_score


def adaptive_remove_count(n_features: int) -> int:
    if n_features > 100:
        return max(1, int(round(n_features * 0.10)))
    if n_features >= 50:
        return max(1, int(round(n_features * 0.05)))
    if n_features >= 20:
        return 2
    return 1


def compute_fold_stability(
    fold_importances: list[list[dict[str, Any]]],
    features: list[str],
    *,
    top_fraction: float = 0.5,
) -> tuple[dict[str, int], dict[str, str], int]:
    n_folds = max(len(fold_importances), 1)
    counts = {f: 0 for f in features}
    for rows in fold_importances:
        ranked = sorted(rows, key=lambda r: float(r.get("importance_pct") or 0), reverse=True)
        k = max(1, int(len(ranked) * top_fraction))
        top = {str(r.get("feature")) for r in ranked[:k]}
        for feat in top:
            if feat in counts:
                counts[feat] += 1
    labels = {f: f"{counts[f]}/{n_folds}" for f in features}
    return counts, labels, n_folds


def _aggregate_shap_map(fold_shap: list[list[dict[str, Any]]], features: list[str]) -> dict[str, float]:
    totals = {f: 0.0 for f in features}
    n = max(len(fold_shap), 1)
    for rows in fold_shap:
        for row in rows:
            feat = row.get("feature")
            if feat in totals:
                totals[str(feat)] += float(row.get("importance_pct") or 0.0)
    return {f: totals[f] / n for f in features}


def _aggregate_gain_map(fold_importances: list[list[dict[str, Any]]], features: list[str]) -> dict[str, float]:
    totals = {f: 0.0 for f in features}
    n = max(len(fold_importances), 1)
    for rows in fold_importances:
        for row in rows:
            feat = row.get("feature")
            if feat in totals:
                totals[str(feat)] += float(row.get("importance_pct") or 0.0)
    return {f: totals[f] / n for f in features}


def _importance_ranking(
    features: list[str],
    *,
    algorithm: str,
    method: str,
    fold_shap: list[list[dict[str, Any]]],
    fold_importances: list[list[dict[str, Any]]],
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    parameters: dict[str, Any],
) -> list[str]:
    if method == "shap":
        shap_map = _aggregate_shap_map(fold_shap, features)
        if any(v > 0 for v in shap_map.values()):
            return sorted(features, key=lambda f: shap_map.get(f, 0.0), reverse=True)
        probe = train_regressor(
            algorithm=algorithm,
            train_X=train_X[features],
            train_y=train_y,
            val_X=val_X[features],
            val_y=val_y,
            features=features,
            parameters={**parameters, **_PROBE_PARAMS},
        )
        shap_rows = compute_shap_importance(probe.get("booster") or probe["model"], val_X, features)
        return [r["feature"] for r in shap_rows] or list(features)

    if method == "permutation":
        probe_params = {**parameters, **_PROBE_PARAMS}
        trained = train_regressor(
            algorithm=algorithm,
            train_X=train_X[features],
            train_y=train_y,
            val_X=val_X[features],
            val_y=val_y,
            features=features,
            parameters=probe_params,
        )
        try:
            from sklearn.inspection import permutation_importance

            imp = permutation_importance(
                _SklearnPredictAdapter(trained["model"], features),
                val_X[features],
                val_y,
                n_repeats=2,
                random_state=int(parameters.get("random_seed", 42)),
                n_jobs=1,
            )
            return sorted(
                features,
                key=lambda f: float(imp.importances_mean[features.index(f)]),
                reverse=True,
            )
        except Exception:
            pass

    gain_map = _aggregate_gain_map(fold_importances, features)
    return sorted(features, key=lambda f: gain_map.get(f, 0.0), reverse=True)


class _SklearnPredictAdapter:
    def __init__(self, model: Any, features: list[str]) -> None:
        self._model = model
        self._features = features

    def fit(self, X, y):  # noqa: ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self._features if c in X.columns]
            return self._model.predict(X[cols])
        return self._model.predict(X)


def _evaluate_subset(
    *,
    algorithm: str,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    subset: list[str],
    parameters: dict[str, Any],
    baseline_val: pd.Series | None,
) -> dict[str, Any]:
    if not subset:
        return {}
    probe_params = {**parameters, **_PROBE_PARAMS}
    inner_val_n = max(20, min(len(train_X) // 5, 500))
    if len(train_X) <= inner_val_n + 30:
        fit_X, fit_y = train_X[subset], train_y
        hold_X, hold_y = val_X[subset], val_y
    else:
        split_at = len(train_X) - inner_val_n
        fit_X = train_X.iloc[:split_at][subset]
        fit_y = train_y.iloc[:split_at]
        hold_X = train_X.iloc[split_at:][subset]
        hold_y = train_y.iloc[split_at:]

    trained = train_regressor(
        algorithm=algorithm,
        train_X=fit_X,
        train_y=fit_y,
        val_X=hold_X,
        val_y=hold_y,
        features=subset,
        parameters=probe_params,
        prediction_type=str(parameters.get("prediction_type") or "regression"),
    )
    pred = trained["model"].predict(val_X[subset])
    return evaluate_predictions(
        val_y,
        pred,
        prediction_type=str(parameters.get("prediction_type") or "regression"),
        baseline=baseline_val,
        target=str(parameters.get("target") or ""),
    )


def _subset_stability_score(subset: list[str], fold_counts: dict[str, int], n_folds: int) -> float:
    if not subset or n_folds <= 0:
        return 0.0
    return float(np.mean([fold_counts.get(f, 0) / n_folds for f in subset]))


def _adaptive_elimination_search(
    *,
    algorithm: str,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    baseline_val: pd.Series | None,
    optimization_metric: str,
    importance_rank: list[str],
    min_features: int,
    fold_counts: dict[str, int],
    n_folds: int,
    cancel_check: Callable[[], bool] | None,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Importance ranks removals; validation score alone picks the winner."""
    opt_key = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)
    refs = refs or {}
    current = list(features)
    rank_index = {f: i for i, f in enumerate(importance_rank)}
    best_subset = list(current)
    best_metrics: dict[str, Any] = {}
    best_score = float("inf")
    best_stability = 0.0
    history: list[dict[str, Any]] = []
    no_improve_streak = 0
    iteration = 0

    def _leaderboard_rows(limit: int = 5) -> list[dict[str, Any]]:
        ranked = sorted(history, key=lambda h: float(h.get("score") or float("inf")))
        rows: list[dict[str, Any]] = []
        for rank, h in enumerate(ranked[:limit], start=1):
            disp = h.get("display_score")
            rows.append({
                "rank": rank,
                "features": h.get("features_remaining"),
                "score": disp,
                "validation_rmse": h.get("validation_rmse"),
            })
        return rows

    def _emit_fs_detail(
        *,
        subset: list[str],
        removed: list[str],
        metrics: dict[str, Any],
        improved: bool,
        previous_best_display: float | None,
    ) -> None:
        if not on_progress:
            return
        current_display = display_score_for_ui(metrics, opt_key, prediction_type=prediction_type, refs=refs)
        best_display = display_score_for_ui(best_metrics, opt_key, prediction_type=prediction_type, refs=refs)
        est_total = max(iteration + 5, len(features))
        on_progress(build_wf_progress_payload(
            fold=n_folds,
            n_folds=n_folds,
            wf_stage=FS_STAGE,
            feature_count=len(subset),
            best_features_count=len(best_subset),
            fs_iteration=iteration,
            fs_total_iterations=est_total,
            in_feature_selection=True,
            removed_features=removed,
            removing_count=len(removed),
            validation_rmse=metrics.get("rmse"),
            validation_mae=metrics.get("mae"),
            validation_directional_accuracy_pct=metrics.get("directional_accuracy_pct"),
            current_display_score=current_display,
            best_display_score=best_display,
            previous_best_display=previous_best_display,
            improved=improved,
            metric_label=metric_display_label(opt_key),
            optimization_metric=opt_key,
            fs_leaderboard=_leaderboard_rows(),
        ))

    def _record(subset: list[str], removed: list[str], metrics: dict[str, Any], score: float) -> None:
        nonlocal iteration, best_score, best_subset, best_metrics, best_stability, no_improve_streak, current
        iteration += 1
        previous_best_display = display_score_for_ui(
            best_metrics, opt_key, prediction_type=prediction_type, refs=refs,
        ) if best_metrics else None
        stability = _subset_stability_score(subset, fold_counts, n_folds)
        improved = _is_better(score, best_score) or (
            abs(score - best_score) < 1e-9 and stability > best_stability
        )
        if improved:
            best_score = score
            best_subset = list(subset)
            best_metrics = dict(metrics)
            best_stability = stability
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        disp = display_score_for_ui(metrics, opt_key, prediction_type=prediction_type, refs=refs)
        history.append({
            "iteration": iteration,
            "features_remaining": len(subset),
            "removed_features": ", ".join(removed) if removed else "",
            "removed_features_list": list(removed),
            "validation_rmse": metrics.get("rmse"),
            "validation_mae": metrics.get("mae"),
            "validation_directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
            "display_score": disp,
            "best_score": round(best_score, 6) if np.isfinite(best_score) else None,
            "score": round(score, 6) if np.isfinite(score) else None,
            "selected": "Yes" if improved else "",
            "fold_stability": round(stability, 4),
            "improved": improved,
        })
        current = list(subset)
        _emit_fs_detail(
            subset=subset,
            removed=removed,
            metrics=metrics,
            improved=improved,
            previous_best_display=previous_best_display,
        )

    metrics = _evaluate_subset(
        algorithm=algorithm,
        train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
        subset=current, parameters=parameters, baseline_val=baseline_val,
    )
    score = _metric_score(metrics, optimization_metric, prediction_type=prediction_type, refs=refs)
    _record(current, [], metrics, score)

    while len(current) > max(1, min_features):
        if cancel_check and cancel_check():
            break
        if no_improve_streak >= _NO_IMPROVE_LIMIT:
            break

        n_remove = adaptive_remove_count(len(current))
        n_remove = min(n_remove, len(current) - max(1, min_features))
        if n_remove <= 0:
            break

        weakest = sorted(current, key=lambda f: rank_index.get(f, 9999), reverse=True)[:n_remove]
        candidate = [f for f in current if f not in weakest]
        metrics = _evaluate_subset(
            algorithm=algorithm,
            train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
            subset=candidate, parameters=parameters, baseline_val=baseline_val,
        )
        score = _metric_score(metrics, optimization_metric, prediction_type=prediction_type, refs=refs)
        _record(candidate, weakest, metrics, score)

    for row in history:
        row["selected"] = "⭐ Best" if row["features_remaining"] == len(best_subset) else ""

    last_evaluated = int(history[-1]["features_remaining"]) if history else len(best_subset)
    metric_label = metric_display_label(opt_key)
    best_display = display_score_for_ui(best_metrics, opt_key, prediction_type=prediction_type, refs=refs)
    stopped = (
        "min_features" if len(best_subset) <= max(1, min_features)
        else "no_improvement" if no_improve_streak >= _NO_IMPROVE_LIMIT
        else "complete"
    )
    meta = {
        "optimization_metric": opt_key,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_display_score": best_display,
        "iterations": len(history),
        "started_features": len(features),
        "finished_features": len(best_subset),
        "selected_features": len(best_subset),
        "best_features_count": len(best_subset),
        "last_evaluated_features": last_evaluated,
        "stopped_reason": stopped,
        "stopped_reason_text": format_stopped_reason(
            stopped,
            best_features=len(best_subset),
            started_features=len(features),
            last_evaluated_features=last_evaluated,
            metric_label=metric_label,
            best_display=best_display,
        ),
        "leaderboard": [
            {
                "rank": i + 1,
                "features": h["features_remaining"],
                "validation_rmse": h["validation_rmse"],
                "validation_mae": h["validation_mae"],
                "validation_directional_accuracy_pct": h["validation_directional_accuracy_pct"],
                "status": h["selected"],
                "score": h.get("display_score", h["score"]),
            }
            for i, h in enumerate(sorted(history, key=lambda x: float(x.get("score") or float("inf")))[:10])
        ],
        "chart": [
            {"features_remaining": h["features_remaining"], "score": h["score"]}
            for h in history
        ],
    }
    return best_subset, history, meta


def _build_selected_features_csv(
    features: list[str],
    selected: list[str],
    shap_map: dict[str, float],
    gain_map: dict[str, float],
    fold_labels: dict[str, str],
) -> pd.DataFrame:
    selected_set = set(selected)
    ranked = sorted(
        features,
        key=lambda f: (f not in selected_set, -gain_map.get(f, 0.0)),
    )
    rows = []
    for rank, feat in enumerate(ranked, start=1):
        rows.append({
            "feature": feat,
            "final_rank": rank,
            "shap_importance_pct": round(shap_map.get(feat, 0.0), 4),
            "gain_importance_pct": round(gain_map.get(feat, 0.0), 4),
            "selected_in_folds": fold_labels.get(feat, "0/0"),
            "selected": "Yes" if feat in selected_set else "No",
        })
    return pd.DataFrame(rows)


def _save_selection_artifacts(
    artifacts_dir: str,
    history: list[dict[str, Any]],
    selected_df: pd.DataFrame,
    stability_rows: list[dict[str, Any]],
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    hist_df = pd.DataFrame(history)
    cols = [
        "iteration", "features_remaining", "removed_features",
        "validation_rmse", "validation_mae", "validation_directional_accuracy_pct",
        "best_score", "selected",
    ]
    for c in cols:
        if c not in hist_df.columns:
            hist_df[c] = None
    hist_df[cols].to_csv(os.path.join(artifacts_dir, "feature_selection_history.csv"), index=False)
    selected_df.to_csv(os.path.join(artifacts_dir, "selected_features.csv"), index=False)
    pd.DataFrame(stability_rows).to_csv(
        os.path.join(artifacts_dir, "feature_stability.csv"), index=False
    )


def select_features_after_walk_forward(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    algorithm: str = "xgboost",
    test_slice_start: int,
    val_window_size: int,
    method: str,
    optimization_metric: str,
    prediction_type: str = "regression",
    fold_shap: list[list[dict[str, Any]]],
    fold_importances: list[list[dict[str, Any]]],
    min_features: int = 3,
    artifacts_dir: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    n_folds: int = 1,
    context_df: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, Any]]:
    method_key = str(method or "rfe").strip().lower()
    if method_key not in FEATURE_SELECTION_METHODS:
        method_key = "rfe"
    opt_metric = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)

    fold_counts, fold_labels, n_folds = compute_fold_stability(fold_importances, features)
    stability_rows = [
        {"feature": f, "selected_in_folds": fold_labels[f], "fold_count": fold_counts[f]}
        for f in sorted(features, key=lambda x: (-fold_counts[x], x))
    ]

    shap_map = _aggregate_shap_map(fold_shap, features)
    gain_map = _aggregate_gain_map(fold_importances, features)

    if method_key == "none":
        selected_df = _build_selected_features_csv(features, features, shap_map, gain_map, fold_labels)
        meta = {
            "method": "none",
            "optimization_metric": opt_metric,
            "optimization_metric_requested": optimization_metric,
            "selected_count": len(features),
            "fold_stability": stability_rows,
        }
        if artifacts_dir:
            _save_selection_artifacts(artifacts_dir, [], selected_df, stability_rows)
        return list(features), meta

    n = int(test_slice_start)
    val_n = min(max(20, int(val_window_size)), max(1, n // 5))
    if n <= val_n + 50:
        val_n = max(1, n // 5)
    split_at = n - val_n

    train_X = X.iloc[:split_at]
    train_y = y.iloc[:split_at]
    val_X = X.iloc[split_at:n]
    val_y = y.iloc[split_at:n]
    val_ctx = (
        context_df.iloc[split_at:n]
        if context_df is not None and len(context_df) == len(X)
        else None
    )
    baseline_val = resolve_ltp_baseline_from_frames(val_X, val_ctx)
    refs = _refs_from_validation_y(val_y)

    rank_method = "shap" if method_key == "shap" else ("permutation" if method_key == "permutation" else "rfe")
    importance_rank = _importance_ranking(
        features,
        algorithm=algorithm,
        method=rank_method,
        fold_shap=fold_shap,
        fold_importances=fold_importances,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        parameters=parameters,
    )

    selected, history, search_meta = _adaptive_elimination_search(
        algorithm=algorithm,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        features=features,
        parameters=parameters,
        baseline_val=baseline_val,
        optimization_metric=opt_metric,
        importance_rank=importance_rank,
        min_features=min_features,
        fold_counts=fold_counts,
        n_folds=n_folds,
        cancel_check=cancel_check,
        prediction_type=prediction_type,
        refs=refs,
        on_progress=on_progress,
    )

    # Prefer stable features: drop fold-unstable features if validation stays within tolerance
    stable_min = max(1, (n_folds + 1) // 2)
    stable_subset = [f for f in selected if fold_counts.get(f, 0) >= stable_min]
    if len(stable_subset) >= max(1, min_features):
        stable_metrics = _evaluate_subset(
            algorithm=algorithm,
            train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
            subset=stable_subset, parameters=parameters, baseline_val=baseline_val,
        )
        stable_score = _metric_score(stable_metrics, opt_metric, prediction_type=prediction_type, refs=refs)
        if _is_better(stable_score, search_meta["best_score"]) or abs(stable_score - search_meta["best_score"]) < 0.02:
            selected = stable_subset
            search_meta["stability_trim_applied"] = True
            search_meta["best_metrics"] = stable_metrics
            search_meta["best_score"] = stable_score

    search_meta["finished_features"] = len(selected)
    search_meta["selected_features"] = len(selected)
    search_meta["best_features_count"] = len(selected)
    metric_label = metric_display_label(opt_metric)
    best_display = display_score_for_ui(
        search_meta.get("best_metrics") or {},
        opt_metric,
        prediction_type=prediction_type,
        refs=refs,
    )
    search_meta["best_display_score"] = best_display
    search_meta["stopped_reason_text"] = format_stopped_reason(
        str(search_meta.get("stopped_reason") or "complete"),
        best_features=len(selected),
        started_features=int(search_meta.get("started_features") or len(features)),
        last_evaluated_features=int(search_meta.get("last_evaluated_features") or len(selected)),
        metric_label=metric_label,
        best_display=best_display,
    )

    if on_progress:
        on_progress({
            "wf_phase": "feature_selection",
            "wf_stage": "feature_selection_complete",
            "fs_complete": True,
            "started_features": len(features),
            "finished_features": len(selected),
            "selected_features": len(selected),
            "best_features_count": len(selected),
            "last_evaluated_features": search_meta.get("last_evaluated_features"),
            "stopped_reason": search_meta.get("stopped_reason"),
            "stopped_reason_text": search_meta.get("stopped_reason_text"),
            "best_display_score": best_display,
            "metric_label": metric_label,
            "fs_leaderboard": search_meta.get("leaderboard", [])[:10],
            "wf_overall_pct": 100,
        })

    selected_df = _build_selected_features_csv(features, selected, shap_map, gain_map, fold_labels)

    meta: dict[str, Any] = {
        "method": method_key,
        "optimization_metric": opt_metric,
        "optimization_metric_requested": optimization_metric,
        "selection_rows": {"train": split_at, "validation": val_n},
        "importance_used_for": "candidate_generation_only",
        "history": history,
        "fold_stability": stability_rows,
        **search_meta,
    }

    if artifacts_dir:
        _save_selection_artifacts(artifacts_dir, history, selected_df, stability_rows)

    return selected, meta
