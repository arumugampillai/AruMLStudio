"""Walk-forward validation — multi-fold training, aggregation, final model."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import (
    aggregate_premium_band_performance,
    aggregate_threshold_analysis,
    evaluate_predictions,
    resolve_ltp_baseline_from_frames,
)
from .feature_selection import select_features_after_walk_forward
from .objective_scoring import (
    composite_score_from_aggregated,
    composite_score_from_metrics,
    compute_validation_reference_stats,
    display_score_for_ui,
    resolve_optimization_metric,
)
from .shap_importance import compute_shap_importance
from .split import (
    WalkForwardSplitError,
    normalize_walk_forward_config,
    walk_forward_fold_slices,
    walk_forward_meta_from_config,
)
from .wf_progress import FS_STAGE, build_wf_progress_payload
from .boost_trainer import TrainingCancelled, feature_importance_df, select_feature_columns, train_regressor


def _fold_metrics_view(
    metrics: dict[str, Any],
    *,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> dict[str, Any]:
    view = {
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
        "accuracy_pct": metrics.get("accuracy_pct"),
        "f1_pct": metrics.get("f1_pct"),
        "precision_pct": metrics.get("precision_pct"),
        "recall_pct": metrics.get("recall_pct"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "brier_score": metrics.get("brier_score"),
        "specificity_pct": metrics.get("specificity_pct"),
        "mape": metrics.get("mape"),
        "r2": metrics.get("r2"),
        "premium_mae_pct": metrics.get("premium_mae_pct"),
        "premium_rmse_pct": metrics.get("premium_rmse_pct"),
        "medae": metrics.get("medae", metrics.get("median_error")),
        "median_error": metrics.get("median_error", metrics.get("medae")),
        "p95_error": metrics.get("p95_error"),
        "prediction_bias": metrics.get("prediction_bias"),
        "prediction_bias_pct": metrics.get("prediction_bias_pct"),
        "endpoint_hit_pct": metrics.get("endpoint_hit_pct"),
        "hit_rate_pct": metrics.get("hit_rate_pct") or metrics.get("endpoint_hit_pct"),
        "hit_rate_tolerance_pct": metrics.get("hit_rate_tolerance_pct"),
        "threshold": metrics.get("threshold"),
        "positive_rate_pct": metrics.get("positive_rate_pct"),
        "predicted_positive_rate_pct": metrics.get("predicted_positive_rate_pct"),
        "premium_band_performance": list(metrics.get("premium_band_performance") or []),
        "confusion": dict(metrics.get("confusion") or {}) if isinstance(metrics.get("confusion"), dict) else None,
        "threshold_analysis": (
            list(metrics.get("threshold_analysis") or [])
            if isinstance(metrics.get("threshold_analysis"), list)
            else []
        ),
    }
    stored = metrics.get("composite_score")
    if stored is not None:
        view["composite_score"] = stored
    else:
        try:
            view["composite_score"] = round(
                float(
                    composite_score_from_metrics(
                        view,
                        prediction_type=prediction_type,
                        refs=score_refs,
                    )
                ),
                6,
            )
        except (TypeError, ValueError):
            view["composite_score"] = None
    return view


def enrich_fold_results_composite(
    fold_results: list[dict[str, Any]] | None,
    *,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Ensure each fold's metrics dict includes composite_score (backfill for older models)."""
    enriched: list[dict[str, Any]] = []
    refs = score_refs or {}
    for row in list(fold_results or []):
        fr = dict(row)
        metrics = dict(fr.get("metrics") or {})
        if metrics.get("composite_score") is None:
            try:
                metrics["composite_score"] = round(
                    float(
                        composite_score_from_metrics(
                            metrics,
                            prediction_type=prediction_type,
                            refs=refs,
                        )
                    ),
                    6,
                )
            except (TypeError, ValueError):
                metrics["composite_score"] = None
        fr["metrics"] = metrics
        enriched.append(fr)
    return enriched


WF_CORE_VALIDATION_KEYS = (
    "rmse",
    "mae",
    "r2",
    "mape",
    "premium_mae_pct",
    "premium_rmse_pct",
    "medae",
    "p95_error",
    "prediction_bias",
    "prediction_bias_pct",
)


def enrich_aggregated_from_fold_results(
    aggregated: dict[str, Any],
    fold_results: list[dict[str, Any]] | None,
    *,
    keys: tuple[str, ...] = WF_CORE_VALIDATION_KEYS,
) -> dict[str, Any]:
    """Fill missing mean/std aggregate keys from per-fold metrics (backfill + persist safety)."""
    out = dict(aggregated or {})
    folds = list(fold_results or out.get("folds") or [])
    if not folds:
        return out
    for key in keys:
        mean_key = f"mean_{key}"
        std_key = f"std_{key}"
        vals: list[float] = []
        for row in folds:
            raw = (row.get("metrics") or {}).get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val == val:
                vals.append(val)
        if not vals:
            continue
        if out.get(mean_key) is None:
            out[mean_key] = round(float(np.mean(vals)), 6)
        if out.get(std_key) is None:
            out[std_key] = round(float(np.std(vals)), 6)
    return out


def aggregate_fold_metrics(
    fold_results: list[dict[str, Any]],
    *,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> dict[str, Any]:
    # Prefer explicit medae; fall back to median_error for older fold metrics.
    for row in fold_results:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else None
        if not metrics:
            continue
        if metrics.get("medae") is None and metrics.get("median_error") is not None:
            metrics["medae"] = metrics.get("median_error")

    keys = (
        "rmse",
        "mae",
        "directional_accuracy_pct",
        "accuracy_pct",
        "f1_pct",
        "precision_pct",
        "recall_pct",
        "specificity_pct",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "r2",
        "mape",
        "premium_mae_pct",
        "premium_rmse_pct",
        "medae",
        "p95_error",
        "prediction_bias",
        "prediction_bias_pct",
        "positive_rate_pct",
        "predicted_positive_rate_pct",
    )
    out: dict[str, Any] = {"n_folds": len(fold_results), "folds": fold_results}
    for key in keys:
        vals = [
            float(r["metrics"][key])
            for r in fold_results
            if r.get("metrics", {}).get(key) is not None and np.isfinite(float(r["metrics"][key]))
        ]
        if vals:
            out[f"mean_{key}"] = round(float(np.mean(vals)), 6)
            out[f"std_{key}"] = round(float(np.std(vals)), 6)
    band_lists = [
        list((r.get("metrics") or {}).get("premium_band_performance") or [])
        for r in fold_results
    ]
    if any(band_lists):
        out["premium_band_performance"] = aggregate_premium_band_performance(band_lists)
    confusion = _sum_confusion_matrices(fold_results)
    if confusion is not None:
        out["confusion"] = confusion
    thr_blocks = [
        list((r.get("metrics") or {}).get("threshold_analysis") or [])
        for r in fold_results
    ]
    if any(thr_blocks):
        out["threshold_analysis"] = aggregate_threshold_analysis(thr_blocks)
    if any(out.get(f"mean_{k}") is not None for k in keys):
        out["mean_composite_score"] = round(
            float(
                composite_score_from_aggregated(
                    out,
                    prediction_type=prediction_type,
                    refs=score_refs,
                )
            ),
            6,
        )
    return out


def _sum_confusion_matrices(fold_results: list[dict[str, Any]]) -> dict[str, int] | None:
    totals = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    found = False
    for row in fold_results:
        conf = (row.get("metrics") or {}).get("confusion")
        if not isinstance(conf, dict):
            continue
        found = True
        for key in totals:
            try:
                totals[key] += int(conf.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return totals if found else None


def validation_metrics_from_wf_aggregate(
    aggregated: dict[str, Any],
    *,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Flatten walk-forward aggregate dict into validation-style metrics for persistence/UI."""
    composite = aggregated.get("mean_composite_score")
    if composite is None:
        composite = composite_score_from_aggregated(
            aggregated,
            prediction_type=prediction_type,
            refs=score_refs,
        )
    out: dict[str, Any] = {
        "rmse": aggregated.get("mean_rmse"),
        "mae": aggregated.get("mean_mae"),
        "directional_accuracy_pct": aggregated.get("mean_directional_accuracy_pct"),
        "r2": aggregated.get("mean_r2"),
        "mape": aggregated.get("mean_mape"),
        "premium_mae_pct": aggregated.get("mean_premium_mae_pct"),
        "premium_rmse_pct": aggregated.get("mean_premium_rmse_pct"),
        "medae": aggregated.get("mean_medae"),
        "p95_error": aggregated.get("mean_p95_error"),
        "prediction_bias": aggregated.get("mean_prediction_bias"),
        "prediction_bias_pct": aggregated.get("mean_prediction_bias_pct"),
        "premium_band_performance": list(aggregated.get("premium_band_performance") or []),
        "confusion": dict(aggregated.get("confusion") or {}) if isinstance(aggregated.get("confusion"), dict) else None,
        "std_rmse": aggregated.get("std_rmse"),
        "std_mae": aggregated.get("std_mae"),
        "std_directional_accuracy_pct": aggregated.get("std_directional_accuracy_pct"),
        "std_r2": aggregated.get("std_r2"),
        "std_mape": aggregated.get("std_mape"),
        "std_premium_mae_pct": aggregated.get("std_premium_mae_pct"),
        "std_premium_rmse_pct": aggregated.get("std_premium_rmse_pct"),
        "std_medae": aggregated.get("std_medae"),
        "std_p95_error": aggregated.get("std_p95_error"),
        "std_prediction_bias": aggregated.get("std_prediction_bias"),
        "std_prediction_bias_pct": aggregated.get("std_prediction_bias_pct"),
        "composite_score": round(float(composite), 6) if composite is not None else None,
        "n_folds": aggregated.get("n_folds"),
        "accuracy_pct": aggregated.get("mean_accuracy_pct"),
        "precision_pct": aggregated.get("mean_precision_pct"),
        "recall_pct": aggregated.get("mean_recall_pct"),
        "specificity_pct": aggregated.get("mean_specificity_pct"),
        "f1_pct": aggregated.get("mean_f1_pct"),
        "roc_auc": aggregated.get("mean_roc_auc"),
        "pr_auc": aggregated.get("mean_pr_auc"),
        "brier_score": aggregated.get("mean_brier_score"),
        "positive_rate_pct": aggregated.get("mean_positive_rate_pct"),
        "predicted_positive_rate_pct": aggregated.get("mean_predicted_positive_rate_pct"),
        "threshold": 0.5,
        "threshold_analysis": (
            list(aggregated.get("threshold_analysis") or [])
            if isinstance(aggregated.get("threshold_analysis"), list)
            else []
        ),
    }
    conf = out.get("confusion")
    if isinstance(conf, dict):
        try:
            tn = int(conf.get("tn") or 0)
            fp = int(conf.get("fp") or 0)
            fn = int(conf.get("fn") or 0)
            tp = int(conf.get("tp") or 0)
            total = tn + fp + fn + tp
            if total > 0:
                if out.get("positive_rate_pct") is None:
                    out["positive_rate_pct"] = round(100.0 * (tp + fn) / total, 2)
                if out.get("predicted_positive_rate_pct") is None:
                    out["predicted_positive_rate_pct"] = round(100.0 * (tp + fp) / total, 2)
        except (TypeError, ValueError):
            pass
    return out



def evaluate_hyperparameters_on_walk_forward(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    fold_defs: list[dict[str, Any]],
    algorithm: str = "xgboost",
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
    cancel_check=None,
    context_df: pd.DataFrame | None = None,
    prediction_writer: Any | None = None,
) -> dict[str, Any]:
    """Full walk-forward re-evaluation for production hyperparameters (post-retrain)."""
    wf_eval = evaluate_walk_forward_folds(
        X=X,
        y=y,
        features=features,
        parameters=parameters,
        fold_defs=fold_defs,
        algorithm=algorithm,
        artifacts_dir=None,
        compute_shap=False,
        cancel_check=cancel_check,
        prediction_type=prediction_type,
        score_refs=score_refs,
        context_df=context_df,
        prediction_writer=prediction_writer,
    )
    aggregated = dict(wf_eval["aggregated"])
    fold_results = list(wf_eval.get("fold_results") or [])
    aggregated = enrich_aggregated_from_fold_results(aggregated, fold_results)
    return {
        "aggregated": aggregated,
        "fold_results": fold_results,
        "validation_metrics": validation_metrics_from_wf_aggregate(
            aggregated,
            prediction_type=prediction_type,
            score_refs=score_refs,
        ),
    }


def evaluate_walk_forward_folds(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    fold_defs: list[dict[str, Any]],
    algorithm: str = "xgboost",
    artifacts_dir: str | None = None,
    compute_shap: bool = True,
    cancel_check=None,
    on_fold_progress=None,
    prediction_type: str = "regression",
    optimization_metric: str = "auto",
    score_refs: dict[str, float] | None = None,
    context_df: pd.DataFrame | None = None,
    prediction_writer: Any | None = None,
) -> dict[str, Any]:
    """Run walk-forward folds and return aggregated validation metrics."""
    fold_results: list[dict[str, Any]] = []
    fold_importances: list[list[dict[str, Any]]] = []
    fold_shap: list[list[dict[str, Any]]] = []

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)

    n_folds = len(fold_defs)
    feature_count = len(features)
    metric_key = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)
    refs = score_refs or {}

    def _emit(**kwargs: Any) -> None:
        if on_fold_progress:
            on_fold_progress(kwargs)

    for fold_def in fold_defs:
        if cancel_check and cancel_check():
            raise TrainingCancelled("Training cancelled")

        fold_idx = int(fold_def["fold"])
        tr = fold_def["train"]
        va = fold_def["validation"]
        train_X = X.iloc[tr["start"]:tr["stop"]]
        train_y = y.iloc[tr["start"]:tr["stop"]]
        val_X = X.iloc[va["start"]:va["stop"]]
        val_y = y.iloc[va["start"]:va["stop"]]

        _emit(**build_wf_progress_payload(
            fold=fold_idx,
            n_folds=n_folds,
            wf_stage="train_model",
            feature_count=feature_count,
            fold_status="training",
        ))

        trained = train_regressor(
            algorithm=algorithm,
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            features=features,
            parameters={**parameters, "prediction_type": prediction_type},
            cancel_check=cancel_check,
            prediction_type=prediction_type,
        )
        model = trained["model"]
        booster = trained.get("booster") or model
        fold_features = trained.get("features") or features

        _emit(**build_wf_progress_payload(
            fold=fold_idx,
            n_folds=n_folds,
            wf_stage="validation",
            feature_count=len(fold_features),
            fold_status="training",
        ))

        val_X_feat, _ = select_feature_columns(val_X, fold_features)
        val_pred = model.predict(val_X_feat)
        val_context_slice = (
            context_df.iloc[va["start"]:va["stop"]]
            if context_df is not None and len(context_df) == len(X)
            else None
        )
        val_baseline = resolve_ltp_baseline_from_frames(val_X, val_context_slice)
        metrics = evaluate_predictions(
            val_y,
            val_pred,
            prediction_type=prediction_type,
            baseline=val_baseline,
            target=str(parameters.get("target") or ""),
        )
        importance = feature_importance_df(trained["model"], fold_features).to_dict(orient="records")

        if compute_shap:
            _emit(**build_wf_progress_payload(
                fold=fold_idx,
                n_folds=n_folds,
                wf_stage="shap_importance",
                feature_count=len(fold_features),
                fold_status="training",
            ))
        shap_rows = compute_shap_importance(booster, val_X_feat, fold_features) if compute_shap else []
        fold_importances.append(importance)
        if shap_rows:
            fold_shap.append(shap_rows)

        fold_payload = {
            "fold": fold_idx,
            "fold_def": fold_def,
            "metrics": _fold_metrics_view(metrics, prediction_type=prediction_type, score_refs=refs),
            "training_meta": trained["training_meta"],
            "feature_importance": importance,
            "shap_importance": shap_rows,
            "trees_trained": trained["trees_trained"],
        }
        if prediction_writer is not None and context_df is not None and len(context_df) == len(X):
            val_context = context_df.iloc[va["start"]:va["stop"]]
            baseline_ltp = resolve_ltp_baseline_from_frames(val_context, val_X)
            prediction_writer.write_fold_predictions(
                fold_number=fold_idx,
                fold_def=fold_def,
                metrics=fold_payload["metrics"],
                val_context=val_context,
                val_pred=val_pred,
                val_y=val_y,
                baseline_ltp=baseline_ltp,
            )
        fold_results.append(fold_payload)
        if artifacts_dir:
            _save_fold_artifacts(artifacts_dir, fold_idx, fold_payload)

        _emit(**build_wf_progress_payload(
            fold=fold_idx,
            n_folds=n_folds,
            wf_stage="next_fold",
            feature_count=len(fold_features),
            fold_status="done",
            fold_complete=True,
            fold_result={
                "fold": fold_idx,
                "mae": fold_payload["metrics"].get("mae"),
                "rmse": fold_payload["metrics"].get("rmse"),
                "directional_accuracy_pct": fold_payload["metrics"].get("directional_accuracy_pct"),
                "accuracy_pct": fold_payload["metrics"].get("accuracy_pct"),
                "precision_pct": fold_payload["metrics"].get("precision_pct"),
                "recall_pct": fold_payload["metrics"].get("recall_pct"),
                "f1_pct": fold_payload["metrics"].get("f1_pct"),
                "roc_auc": fold_payload["metrics"].get("roc_auc"),
                "composite_score": round(composite_score_from_metrics(
                    fold_payload["metrics"], prediction_type=prediction_type, refs=refs,
                ), 4),
                "display_score": display_score_for_ui(
                    fold_payload["metrics"], metric_key, prediction_type=prediction_type, refs=refs,
                ),
                "metric_label": metric_key,
                "feature_count": len(fold_features),
                "prediction_type": prediction_type,
            },
            metrics=fold_payload["metrics"],
        ))

    aggregated = aggregate_fold_metrics(
        fold_results,
        prediction_type=prediction_type,
        score_refs=refs,
    )
    return {
        "aggregated": aggregated,
        "fold_results": fold_results,
        "fold_importances": fold_importances,
        "fold_shap": fold_shap,
    }


def _save_fold_artifacts(base_dir: str, fold_idx: int, payload: dict[str, Any]) -> None:
    fold_dir = os.path.join(base_dir, f"fold_{fold_idx:02d}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload.get("metrics") or {}, fh, indent=2)
    with open(os.path.join(fold_dir, "training_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(payload.get("training_meta") or {}, fh, indent=2)
    imp = payload.get("feature_importance") or []
    pd.DataFrame(imp).to_csv(os.path.join(fold_dir, "feature_importance.csv"), index=False)
    with open(os.path.join(fold_dir, "shap_importance.json"), "w", encoding="utf-8") as fh:
        json.dump(payload.get("shap_importance") or [], fh, indent=2)
    with open(os.path.join(fold_dir, "fold.json"), "w", encoding="utf-8") as fh:
        json.dump(payload.get("fold_def") or {}, fh, indent=2)


def run_walk_forward_validation(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    walk_forward_cfg: dict[str, Any],
    split_cfg: dict[str, Any] | None = None,
    algorithm: str = "xgboost",
    prediction_type: str = "regression",
    artifacts_dir: str | None = None,
    cancel_check=None,
    on_fold_progress=None,
    context_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    wf_cfg = normalize_walk_forward_config({"walk_forward": walk_forward_cfg, "test": walk_forward_cfg.get("test_holdout_pct", 15)}, len(X))
    fold_defs, test_sl = walk_forward_fold_slices(len(X), wf_cfg)
    score_refs = compute_validation_reference_stats(y, fold_defs)

    wf_base = artifacts_dir
    wf_eval = evaluate_walk_forward_folds(
        X=X,
        y=y,
        features=features,
        parameters=parameters,
        fold_defs=fold_defs,
        algorithm=algorithm,
        artifacts_dir=wf_base,
        compute_shap=True,
        cancel_check=cancel_check,
        on_fold_progress=on_fold_progress,
        prediction_type=prediction_type,
        optimization_metric=str(wf_cfg.get("optimization_metric") or "auto"),
        score_refs=score_refs,
        context_df=context_df,
    )
    fold_results = wf_eval["fold_results"]
    fold_importances = wf_eval["fold_importances"]
    fold_shap = wf_eval["fold_shap"]

    selected_features, selection_meta = select_features_after_walk_forward(
        X=X,
        y=y,
        features=features,
        parameters=parameters,
        algorithm=algorithm,
        test_slice_start=int(test_sl.start),
        val_window_size=int(wf_cfg["validation_window_size"]),
        method=str(wf_cfg.get("feature_selection_method") or "rfe"),
        optimization_metric=str(wf_cfg.get("optimization_metric") or "auto"),
        prediction_type=prediction_type,
        fold_shap=fold_shap,
        fold_importances=fold_importances,
        min_features=int(wf_cfg.get("min_selected_features") or 3),
        artifacts_dir=wf_base,
        cancel_check=cancel_check,
        on_progress=on_fold_progress,
        n_folds=len(fold_defs),
        context_df=context_df,
    )
    aggregated = aggregate_fold_metrics(
        fold_results,
        prediction_type=prediction_type,
        score_refs=score_refs,
    )
    aggregated = enrich_aggregated_from_fold_results(aggregated, fold_results)
    aggregated["selected_features"] = selected_features
    aggregated["feature_selection"] = selection_meta
    wf_meta = walk_forward_meta_from_config(wf_cfg, split_cfg=split_cfg)
    for key, val in wf_meta.items():
        if val is not None and key not in aggregated:
            aggregated[key] = val
    aggregated["walk_forward_meta"] = wf_meta

    summary = {
        "config": wf_cfg,
        "meta": wf_meta,
        "folds": fold_defs,
        "test_holdout": {"start": test_sl.start, "stop": test_sl.stop, "rows": test_sl.stop - test_sl.start},
        "fold_results": [{k: v for k, v in r.items() if k != "feature_importance"} for r in fold_results],
        "aggregated": aggregated,
        "selected_features": selected_features,
        "feature_selection": selection_meta,
        "reference_stats": score_refs,
    }
    if wf_base:
        with open(os.path.join(wf_base, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        with open(os.path.join(wf_base, "aggregated_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2)

    return {
        "summary": summary,
        "aggregated": aggregated,
        "selected_features": selected_features,
        "test_slice": test_sl,
        "fold_results": fold_results,
    }


def train_final_model_after_walk_forward(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    selected_features: list[str],
    parameters: dict[str, Any],
    algorithm: str = "xgboost",
    test_slice: slice,
    val_window_size: int,
    cancel_check=None,
    on_iteration=None,
) -> dict[str, Any]:
    """Train final model on all pre-test data using selected features."""
    use_features = [f for f in selected_features if f in features]
    if not use_features:
        use_features = list(features)

    trainval_X = X.iloc[:test_slice.start][use_features]
    trainval_y = y.iloc[:test_slice.start]
    n = len(trainval_X)
    val_n = min(max(20, int(val_window_size)), max(1, n // 5))
    if n <= val_n + 50:
        val_n = max(1, n // 5)
    split_at = n - val_n
    fit_X = trainval_X.iloc[:split_at]
    fit_y = trainval_y.iloc[:split_at]
    hold_X = trainval_X.iloc[split_at:]
    hold_y = trainval_y.iloc[split_at:]

    return train_regressor(
        algorithm=algorithm,
        train_X=fit_X,
        train_y=fit_y,
        val_X=hold_X,
        val_y=hold_y,
        features=use_features,
        parameters=parameters,
        cancel_check=cancel_check,
        on_iteration=on_iteration,
        prediction_type=str(parameters.get("prediction_type") or "regression"),
    )
