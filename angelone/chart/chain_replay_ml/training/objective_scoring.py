"""Optimization objectives — single metrics, composite scores, normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HPO_SEEDS = (42, 123, 999)

REGRESSION_COMPOSITE_WEIGHTS = {
    "directional_accuracy": 0.60,
    "rmse": -0.25,
    "mae": -0.15,
}

CLASSIFICATION_COMPOSITE_WEIGHTS = {
    # Trader-facing: when model says BUY, how often correct (precision) first.
    "precision": 0.40,
    "f1": 0.30,
    "roc_auc": 0.20,
    "recall": 0.10,
}

OPTIMIZATION_METRICS = (
    "auto",
    "rmse",
    "mae",
    "directional_accuracy",
    "accuracy",
    "f1",
    "composite",
    "custom",
)


def is_classification_prediction(prediction_type: str) -> bool:
    return str(prediction_type or "").strip().lower() in ("binary", "multiclass", "classification")


def resolve_optimization_metric(
    optimization_metric: str,
    *,
    prediction_type: str = "regression",
) -> str:
    key = str(optimization_metric or "auto").strip().lower()
    if key == "auto":
        return "composite"
    if key == "custom":
        return "composite"
    if key in ("rmse", "mae", "directional_accuracy", "accuracy", "f1", "composite"):
        return key
    return "composite"


def compute_validation_reference_stats(
    y: pd.Series,
    fold_defs: list[dict[str, Any]],
) -> dict[str, float]:
    """Reference scales for normalizing RMSE/MAE from walk-forward validation windows."""
    chunks: list[pd.Series] = []
    for fold_def in fold_defs:
        va = fold_def["validation"]
        chunks.append(y.iloc[int(va["start"]): int(va["stop"])])
    if not chunks:
        return {"rmse_ref": 1.0, "mae_ref": 1.0}
    all_y = pd.concat(chunks).astype(float)
    std_y = float(np.std(all_y))
    mean_abs = float(np.mean(np.abs(all_y)))
    return {
        "rmse_ref": max(std_y, mean_abs * 0.25, 1e-6),
        "mae_ref": max(mean_abs, 1e-6),
    }


def _scale_pct(value: float | None, *, default: float = 0.5) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value) / 100.0


def regression_composite_score(
    metrics: dict[str, Any],
    *,
    refs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Higher is better. Uses directional accuracy minus normalized error terms."""
    w = weights or REGRESSION_COMPOSITE_WEIGHTS
    refs = refs or {"rmse_ref": 1.0, "mae_ref": 1.0}
    da = _scale_pct(metrics.get("directional_accuracy_pct"))
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    rmse_norm = float(rmse) / float(refs["rmse_ref"]) if rmse is not None else 1.0
    mae_norm = float(mae) / float(refs["mae_ref"]) if mae is not None else 1.0
    return (
        w["directional_accuracy"] * da
        + w["rmse"] * rmse_norm
        + w["mae"] * mae_norm
    )


def classification_composite_score(
    metrics: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Higher is better. Default: 40% Precision + 30% F1 + 20% ROC-AUC + 10% Recall."""
    w = weights or CLASSIFICATION_COMPOSITE_WEIGHTS
    precision = _scale_pct(metrics.get("precision_pct"))
    f1 = _scale_pct(metrics.get("f1_pct"))
    recall = _scale_pct(metrics.get("recall_pct"))
    roc = metrics.get("roc_auc")
    roc_v = float(roc) if roc is not None and np.isfinite(roc) else 0.5
    return (
        w.get("precision", 0.40) * precision
        + w.get("f1", 0.30) * f1
        + w.get("roc_auc", 0.20) * roc_v
        + w.get("recall", 0.10) * recall
    )


def classification_composite_breakdown(
    metrics: dict[str, Any] | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Human-readable weight recipe for Overview / Composite audit."""
    w = dict(weights or CLASSIFICATION_COMPOSITE_WEIGHTS)
    order = ("precision", "f1", "roc_auc", "recall")
    labels = {
        "precision": "Precision",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "pr_auc": "PR-AUC",
    }
    rows: list[dict[str, Any]] = []
    for key in order:
        if key not in w:
            continue
        weight = float(w[key])
        row: dict[str, Any] = {
            "key": key,
            "label": labels.get(key, key),
            "weight": weight,
            "weight_pct": round(weight * 100.0, 1),
        }
        if isinstance(metrics, dict):
            if key == "roc_auc":
                raw = metrics.get("roc_auc")
                row["value"] = float(raw) if raw is not None and np.isfinite(float(raw)) else None
                row["contribution"] = (
                    weight * float(row["value"]) if row["value"] is not None else None
                )
            else:
                pct_key = f"{key}_pct" if not key.endswith("_pct") else key
                raw = metrics.get(pct_key)
                row["value_pct"] = float(raw) if raw is not None and np.isfinite(float(raw)) else None
                row["contribution"] = (
                    weight * (float(row["value_pct"]) / 100.0)
                    if row["value_pct"] is not None
                    else None
                )
        rows.append(row)
    return rows


def composite_score_from_metrics(
    metrics: dict[str, Any],
    *,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    if is_classification_prediction(prediction_type):
        return classification_composite_score(metrics, weights=weights)
    return regression_composite_score(metrics, refs=refs, weights=weights)


def composite_score_from_aggregated(
    aggregated: dict[str, Any],
    *,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    metrics = {
        "rmse": aggregated.get("mean_rmse"),
        "mae": aggregated.get("mean_mae"),
        "directional_accuracy_pct": aggregated.get("mean_directional_accuracy_pct"),
        "accuracy_pct": aggregated.get("mean_accuracy_pct"),
        "f1_pct": aggregated.get("mean_f1_pct"),
        "roc_auc": aggregated.get("mean_roc_auc"),
        "pr_auc": aggregated.get("mean_pr_auc"),
        "precision_pct": aggregated.get("mean_precision_pct"),
        "recall_pct": aggregated.get("mean_recall_pct"),
    }
    return composite_score_from_metrics(
        metrics,
        prediction_type=prediction_type,
        refs=refs,
        weights=weights,
    )


def objective_to_minimize(
    aggregated: dict[str, Any],
    optimization_metric: str,
    *,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
    composite_weights: dict[str, float] | None = None,
) -> float:
    """Convert validation aggregate metrics to an Optuna objective (lower is better)."""
    key = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)
    if key == "composite":
        composite = composite_score_from_aggregated(
            aggregated,
            prediction_type=prediction_type,
            refs=refs,
            weights=composite_weights,
        )
        return -float(composite)
    if key == "mae":
        val = aggregated.get("mean_mae")
        return float(val) if val is not None else float("inf")
    if key == "directional_accuracy":
        val = aggregated.get("mean_directional_accuracy_pct")
        return -float(val) if val is not None else float("inf")
    if key == "accuracy":
        val = aggregated.get("mean_accuracy_pct")
        return -float(val) if val is not None else float("inf")
    if key == "f1":
        val = aggregated.get("mean_f1_pct")
        return -float(val) if val is not None else float("inf")
    val = aggregated.get("mean_rmse")
    return float(val) if val is not None else float("inf")


def metric_score_for_selection(
    metrics: dict[str, Any],
    optimization_metric: str,
    *,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
    composite_weights: dict[str, float] | None = None,
) -> float:
    """Lower is better — used by feature-selection leaderboard comparisons."""
    key = resolve_optimization_metric(optimization_metric, prediction_type=prediction_type)
    if key == "composite":
        composite = composite_score_from_metrics(
            metrics,
            prediction_type=prediction_type,
            refs=refs,
            weights=composite_weights,
        )
        return -float(composite)
    if key == "mae":
        return float(metrics.get("mae") if metrics.get("mae") is not None else float("inf"))
    if key == "directional_accuracy":
        da = metrics.get("directional_accuracy_pct")
        return float("inf") if da is None else -float(da)
    if key == "accuracy":
        acc = metrics.get("accuracy_pct")
        return float("inf") if acc is None else -float(acc)
    if key == "f1":
        f1 = metrics.get("f1_pct")
        return float("inf") if f1 is None else -float(f1)
    return float(metrics.get("rmse") if metrics.get("rmse") is not None else float("inf"))


def metric_display_label(metric_key: str) -> str:
    labels = {
        "composite": "Composite Score",
        "rmse": "RMSE",
        "mae": "MAE",
        "directional_accuracy": "Directional Accuracy",
        "accuracy": "Accuracy",
        "f1": "F1",
    }
    return labels.get(str(metric_key or "").strip().lower(), str(metric_key))


def display_score_for_ui(
    metrics: dict[str, Any],
    metric_key: str,
    *,
    prediction_type: str = "regression",
    refs: dict[str, float] | None = None,
) -> float | None:
    """User-facing validation score for live UI (composite/accuracy: higher better)."""
    key = resolve_optimization_metric(metric_key, prediction_type=prediction_type)
    if key == "composite":
        val = composite_score_from_metrics(metrics, prediction_type=prediction_type, refs=refs)
        return round(float(val), 4) if val is not None else None
    if key == "mae":
        v = metrics.get("mae")
        return round(float(v), 4) if v is not None else None
    if key == "directional_accuracy":
        v = metrics.get("directional_accuracy_pct")
        return round(float(v), 2) if v is not None else None
    if key == "accuracy":
        v = metrics.get("accuracy_pct")
        return round(float(v), 2) if v is not None else None
    if key == "f1":
        v = metrics.get("f1_pct")
        return round(float(v), 2) if v is not None else None
    v = metrics.get("rmse")
    return round(float(v), 4) if v is not None else None


def is_higher_better_metric(metric_key: str) -> bool:
    return resolve_optimization_metric(metric_key) in (
        "composite", "directional_accuracy", "accuracy", "f1",
    )


def format_stopped_reason(
    reason: str,
    *,
    best_features: int | None = None,
    started_features: int | None = None,
    last_evaluated_features: int | None = None,
    metric_label: str = "",
    best_display: float | None = None,
) -> str:
    key = str(reason or "").strip().lower()
    if key == "no_improvement":
        base = "No improvement for 5 iterations"
        if best_features is not None:
            score_txt = f", best {metric_label} {best_display:.3f}" if best_display is not None and metric_label else ""
            if started_features is not None and best_features == started_features:
                return f"{base} — kept starting set ({best_features} features{score_txt})"
            return f"{base} — selected best subset ({best_features} features{score_txt})"
        if last_evaluated_features is not None:
            return f"{base} (last evaluated {last_evaluated_features} features)"
        return "No improvement for 5 iterations"
    mapping = {
        "min_features": "Minimum feature count reached",
        "complete": "Search completed",
        "cancelled": "Training cancelled",
    }
    return mapping.get(key, str(reason or "complete"))
