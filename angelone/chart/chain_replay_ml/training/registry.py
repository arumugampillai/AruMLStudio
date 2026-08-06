"""Model registry — list, load, delete, and protect trained model packages."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Any

from .artifacts import resolve_feature_elimination_doc
from .config import TrainingConfig, normalize_training_config
from .objective_scoring import composite_score_from_aggregated, composite_score_from_metrics
from .paths import model_artifact_paths, model_package_dir, models_dir, safe_model_name
from .walk_forward_runner import enrich_aggregated_from_fold_results, enrich_fold_results_composite

_PROTECTED_STATUSES = frozenset({"deployed", "production", "live"})
_RESEARCH_SCOPE_MARKERS = frozenset(
    {"experiment", "research_experiment", "research", "analysis_experiment"}
)
# Analysis Lab packages: Exp_Exp_001 / Exp_001 (not Create Model Builder production names).
_RESEARCH_MODEL_NAME_RE = re.compile(r"^Exp_(?:Exp_)?\d+", re.IGNORECASE)


def is_research_experiment_model(
    model_name: str,
    *docs: dict[str, Any] | None,
) -> bool:
    """True for Analysis Lab experiment packages (not Model Builder production)."""
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        scope = str(
            doc.get("registry_scope") or doc.get("origin") or doc.get("model_origin") or ""
        ).strip().lower()
        if scope in _RESEARCH_SCOPE_MARKERS:
            return True
        if str(doc.get("experiment_id") or "").strip():
            return True
    name = str(model_name or "").strip()
    if _RESEARCH_MODEL_NAME_RE.match(name):
        return True
    return False


class ModelDeleteBlockedError(Exception):
    """Raised when a model cannot be deleted (in use or deployed)."""

    def __init__(self, message: str = "This model is currently in use.") -> None:
        super().__init__(message)
        self.message = message


def _active_model_path(data_dir: str) -> str:
    return os.path.join(models_dir(data_dir), ".active_model.json")


def get_active_model(data_dir: str) -> str | None:
    path = _active_model_path(data_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        name = str(doc.get("model_name") or "").strip()
        return safe_model_name(name) if name else None
    except (OSError, json.JSONDecodeError):
        return None


def set_active_model(data_dir: str, model_name: str) -> dict[str, Any]:
    safe = safe_model_name(model_name)
    path = _active_model_path(data_dir)
    doc = {
        "model_name": safe,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def clear_active_model_if(data_dir: str, model_name: str) -> None:
    if get_active_model(data_dir) == safe_model_name(model_name):
        path = _active_model_path(data_dir)
        if os.path.isfile(path):
            os.remove(path)


def _load_json(path: str) -> dict[str, Any] | list[Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        if not str(raw).strip():
            return None
        return json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_text(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _load_json_artifact(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"available": False, "path": os.path.basename(path), "data": None}
    data = _load_json(path)
    return {"available": True, "path": os.path.basename(path), "data": data}


def _load_csv_artifact(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"available": False, "path": os.path.basename(path), "rows": []}
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return {"available": True, "path": os.path.basename(path), "rows": rows}


def _folder_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _algorithm_label(raw: str | None) -> str:
    labels = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "linear": "Linear",
        "neural": "Neural",
    }
    key = str(raw or "").strip().lower()
    return labels.get(key, str(raw or "—").title())


def _detect_validation_strategy(config: dict[str, Any]) -> dict[str, str]:
    split_info = dict(config.get("split_info") or {})
    split = dict(config.get("split") or {})
    wf = dict(split.get("walk_forward") or split_info.get("walk_forward") or {})
    window_mode = str(wf.get("window_mode") or "expanding").strip().lower()

    ui = str(
        split_info.get("validation_strategy_ui")
        or split.get("validation_strategy_ui")
        or ""
    ).strip().lower()
    label = str(
        split_info.get("validation_strategy_label")
        or split.get("validation_strategy_label")
        or ""
    ).strip()
    if ui:
        from .split import validation_strategy_label_from_ui

        if not label:
            label = validation_strategy_label_from_ui(ui, window_mode=window_mode)
        if ui == "rolling_window" or (ui == "walk_forward" and window_mode == "rolling"):
            key = "rolling_window"
        elif ui == "walk_forward":
            key = "walk_forward"
        else:
            key = "time_series"
        return {"key": key, "label": label}

    strategy = str(split_info.get("strategy") or split.get("strategy") or "time_series").strip().lower()
    if strategy == "walk_forward" and window_mode == "rolling":
        return {"key": "rolling_window", "label": "Rolling Window"}
    if strategy == "walk_forward":
        return {"key": "walk_forward", "label": "Walk Forward"}
    return {"key": "time_series", "label": "Time Series Split"}


def _num_or_none(v: Any) -> float | None:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n == n else None


def _first_num(*vals: Any) -> float | None:
    for v in vals:
        n = _num_or_none(v)
        if n is not None:
            return n
    return None


def _first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if v not in (None, "", {}):
            return v
    return None


def _has_any_metric(doc: dict[str, Any] | None) -> bool:
    if not isinstance(doc, dict):
        return False
    keys = (
        "rmse",
        "mae",
        "directional_accuracy_pct",
        "directional_accuracy",
        "composite_score",
        "accuracy_pct",
        "f1_pct",
        "precision_pct",
        "recall_pct",
        "specificity_pct",
        "roc_auc",
        "pr_auc",
        "brier_score",
    )
    return any(_num_or_none(doc.get(k)) is not None for k in keys)


def _classification_fields_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull classification metrics from a metrics blob or WF mean_* aggregate."""
    out: dict[str, Any] = {
        "accuracy_pct": _first_num(raw.get("accuracy_pct"), raw.get("mean_accuracy_pct")),
        "precision_pct": _first_num(raw.get("precision_pct"), raw.get("mean_precision_pct")),
        "recall_pct": _first_num(raw.get("recall_pct"), raw.get("mean_recall_pct")),
        "specificity_pct": _first_num(raw.get("specificity_pct"), raw.get("mean_specificity_pct")),
        "f1_pct": _first_num(raw.get("f1_pct"), raw.get("mean_f1_pct")),
        "roc_auc": _first_num(raw.get("roc_auc"), raw.get("mean_roc_auc")),
        "pr_auc": _first_num(raw.get("pr_auc"), raw.get("mean_pr_auc")),
        "brier_score": _first_num(raw.get("brier_score"), raw.get("mean_brier_score")),
        "threshold": _first_num(raw.get("threshold"), raw.get("decision_threshold")),
        "positive_rate_pct": _first_num(
            raw.get("positive_rate_pct"), raw.get("mean_positive_rate_pct")
        ),
        "predicted_positive_rate_pct": _first_num(
            raw.get("predicted_positive_rate_pct"),
            raw.get("mean_predicted_positive_rate_pct"),
        ),
    }
    conf = raw.get("confusion")
    if isinstance(conf, dict) and any(k in conf for k in ("tp", "tn", "fp", "fn")):
        cleaned: dict[str, int] = {}
        for key in ("tn", "fp", "fn", "tp"):
            try:
                cleaned[key] = int(conf.get(key) or 0)
            except (TypeError, ValueError):
                cleaned[key] = 0
        out["confusion"] = cleaned
        total = cleaned["tn"] + cleaned["fp"] + cleaned["fn"] + cleaned["tp"]
        # Derive Specificity from stored confusion when the metric itself is missing.
        if out.get("specificity_pct") is None:
            denom = cleaned["tn"] + cleaned["fp"]
            if denom > 0:
                out["specificity_pct"] = round(100.0 * cleaned["tn"] / denom, 2)
        if total > 0:
            if out.get("positive_rate_pct") is None:
                out["positive_rate_pct"] = round(
                    100.0 * (cleaned["tp"] + cleaned["fn"]) / total, 2
                )
            if out.get("predicted_positive_rate_pct") is None:
                out["predicted_positive_rate_pct"] = round(
                    100.0 * (cleaned["tp"] + cleaned["fp"]) / total, 2
                )
    if out.get("threshold") is None:
        out["threshold"] = 0.5
    ta = raw.get("threshold_analysis")
    if isinstance(ta, list) and ta:
        out["threshold_analysis"] = [dict(row) for row in ta if isinstance(row, dict)]
    return out


def _build_production_metrics(
    *,
    stage_key: str,
    stage_label: str,
    source_file: str,
    source_path: str,
    raw_metrics: dict[str, Any] | None,
    n_folds: int | None = None,
    prediction_type: str = "regression",
) -> dict[str, Any]:
    raw = dict(raw_metrics or {})
    directional = _first_num(raw.get("directional_accuracy_pct"), raw.get("directional_accuracy"))
    bands = raw.get("premium_band_performance")
    if not isinstance(bands, list):
        bands = []
    pred = str(prediction_type or "regression").strip().lower()
    out: dict[str, Any] = {
        "stage_key": stage_key,
        "stage_label": stage_label,
        "source_file": source_file,
        "source_path": source_path,
        "prediction_type": pred,
        "rmse": _num_or_none(raw.get("rmse")),
        "mae": _num_or_none(raw.get("mae")),
        "directional_accuracy_pct": directional,
        "composite_score": _num_or_none(raw.get("composite_score")),
        "premium_mae_pct": _num_or_none(raw.get("premium_mae_pct")),
        "premium_rmse_pct": _num_or_none(raw.get("premium_rmse_pct")),
        "medae": _first_num(raw.get("medae"), raw.get("median_error")),
        "p95_error": _num_or_none(raw.get("p95_error")),
        "prediction_bias": _num_or_none(raw.get("prediction_bias")),
        "prediction_bias_pct": _num_or_none(raw.get("prediction_bias_pct")),
        "premium_band_performance": bands,
        "n_folds": n_folds,
    }
    out.update(_classification_fields_from_raw(raw))
    return out


def _derive_composite_score(
    raw_metrics: dict[str, Any],
    *,
    aggregated: dict[str, Any] | None = None,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> float | None:
    stored = _first_num(raw_metrics.get("composite_score"), raw_metrics.get("mean_composite_score"))
    if stored is not None:
        return stored
    if aggregated:
        try:
            return round(
                float(
                    composite_score_from_aggregated(
                        aggregated,
                        prediction_type=prediction_type,
                        refs=score_refs,
                    )
                ),
                6,
            )
        except (TypeError, ValueError):
            pass
    if _has_any_metric(raw_metrics):
        try:
            return round(
                float(
                    composite_score_from_metrics(
                        {
                            "rmse": raw_metrics.get("rmse"),
                            "mae": raw_metrics.get("mae"),
                            "directional_accuracy_pct": raw_metrics.get("directional_accuracy_pct"),
                            "accuracy_pct": raw_metrics.get("accuracy_pct"),
                            "precision_pct": raw_metrics.get("precision_pct"),
                            "recall_pct": raw_metrics.get("recall_pct"),
                            "f1_pct": raw_metrics.get("f1_pct"),
                            "roc_auc": raw_metrics.get("roc_auc"),
                            "pr_auc": raw_metrics.get("pr_auc"),
                        },
                        prediction_type=prediction_type,
                        refs=score_refs,
                    )
                ),
                6,
            )
        except (TypeError, ValueError):
            pass
    return None


def _resolve_authoritative_metrics(
    *,
    strategy: dict[str, str],
    metrics_doc: dict[str, Any],
    summary_doc: dict[str, Any],
    wf_summary_doc: dict[str, Any] | None,
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
) -> dict[str, Any]:
    strategy_key = str(strategy.get("key") or "")
    if strategy_key in ("walk_forward", "rolling_window"):
        composite_doc = dict(metrics_doc.get("composite_scores") or {})
        prod_wf = dict(metrics_doc.get("production_walk_forward") or {})
        if prod_wf:
            n_folds_raw = _first_nonempty(prod_wf.get("n_folds"))
            try:
                n_folds = int(n_folds_raw) if n_folds_raw is not None else None
            except (TypeError, ValueError):
                n_folds = None
            refs = score_refs or dict((wf_summary_doc or {}).get("reference_stats") or {})
            prod_composite_block = dict(composite_doc.get("production_composite") or {})
            merged = {
                "rmse": _first_num(prod_wf.get("mean_rmse")),
                "mae": _first_num(prod_wf.get("mean_mae")),
                "directional_accuracy_pct": _first_num(prod_wf.get("mean_directional_accuracy_pct")),
                "premium_mae_pct": _first_num(prod_wf.get("mean_premium_mae_pct"), prod_wf.get("premium_mae_pct")),
                "premium_rmse_pct": _first_num(prod_wf.get("mean_premium_rmse_pct"), prod_wf.get("premium_rmse_pct")),
                "medae": _first_num(
                    prod_wf.get("mean_medae"),
                    prod_wf.get("medae"),
                    prod_wf.get("mean_median_error"),
                    prod_wf.get("median_error"),
                ),
                "p95_error": _first_num(prod_wf.get("mean_p95_error"), prod_wf.get("p95_error")),
                "prediction_bias": _first_num(
                    prod_wf.get("mean_prediction_bias"), prod_wf.get("prediction_bias"),
                ),
                "prediction_bias_pct": _first_num(
                    prod_wf.get("mean_prediction_bias_pct"), prod_wf.get("prediction_bias_pct"),
                ),
                "premium_band_performance": list(prod_wf.get("premium_band_performance") or []),
                "composite_score": _first_num(
                    prod_composite_block.get("score"),
                    prod_wf.get("mean_composite_score"),
                    prod_wf.get("composite_score"),
                ),
            }
            merged.update(_classification_fields_from_raw(prod_wf))
            if merged["composite_score"] is None:
                merged["composite_score"] = _derive_composite_score(
                    merged,
                    aggregated=prod_wf,
                    prediction_type=prediction_type,
                    score_refs=refs or None,
                )
            return _build_production_metrics(
                stage_key="production_retrained_wf",
                stage_label="Production (Retrained Champion WF)",
                source_file=str(prod_composite_block.get("source_file") or "metrics.json"),
                source_path=str(prod_composite_block.get("source_path") or "$.production_walk_forward"),
                raw_metrics=merged,
                n_folds=n_folds,
                prediction_type=prediction_type,
            )

        summary_agg = dict((wf_summary_doc or {}).get("aggregated") or {})
        wf_doc = dict(metrics_doc.get("walk_forward") or {})
        n_folds_raw = _first_nonempty(summary_agg.get("n_folds"), wf_doc.get("n_folds"))
        try:
            n_folds = int(n_folds_raw) if n_folds_raw is not None else None
        except (TypeError, ValueError):
            n_folds = None
        source_file = "walk_forward/summary.json" if summary_agg else "metrics.json"
        source_path = "$.aggregated" if summary_agg else "$.walk_forward"
        agg_for_composite = summary_agg if summary_agg else wf_doc
        refs = score_refs or dict((wf_summary_doc or {}).get("reference_stats") or {})
        merged = {
            "rmse": _first_num(summary_agg.get("mean_rmse"), wf_doc.get("mean_rmse")),
            "mae": _first_num(summary_agg.get("mean_mae"), wf_doc.get("mean_mae")),
            "directional_accuracy_pct": _first_num(
                summary_agg.get("mean_directional_accuracy_pct"),
                wf_doc.get("mean_directional_accuracy_pct"),
            ),
            "premium_mae_pct": _first_num(
                summary_agg.get("mean_premium_mae_pct"),
                summary_agg.get("premium_mae_pct"),
                wf_doc.get("mean_premium_mae_pct"),
                wf_doc.get("premium_mae_pct"),
            ),
            "premium_rmse_pct": _first_num(
                summary_agg.get("mean_premium_rmse_pct"),
                summary_agg.get("premium_rmse_pct"),
                wf_doc.get("mean_premium_rmse_pct"),
                wf_doc.get("premium_rmse_pct"),
            ),
            "medae": _first_num(
                summary_agg.get("mean_medae"),
                summary_agg.get("medae"),
                summary_agg.get("mean_median_error"),
                wf_doc.get("mean_medae"),
                wf_doc.get("medae"),
            ),
            "p95_error": _first_num(
                summary_agg.get("mean_p95_error"),
                summary_agg.get("p95_error"),
                wf_doc.get("mean_p95_error"),
                wf_doc.get("p95_error"),
            ),
            "prediction_bias": _first_num(
                summary_agg.get("mean_prediction_bias"),
                summary_agg.get("prediction_bias"),
                wf_doc.get("mean_prediction_bias"),
                wf_doc.get("prediction_bias"),
            ),
            "prediction_bias_pct": _first_num(
                summary_agg.get("mean_prediction_bias_pct"),
                summary_agg.get("prediction_bias_pct"),
                wf_doc.get("mean_prediction_bias_pct"),
                wf_doc.get("prediction_bias_pct"),
            ),
            "premium_band_performance": list(
                summary_agg.get("premium_band_performance")
                or wf_doc.get("premium_band_performance")
                or []
            ),
            "composite_score": _first_num(
                summary_agg.get("mean_composite_score"),
                summary_agg.get("composite_score"),
                wf_doc.get("mean_composite_score"),
                wf_doc.get("composite_score"),
            ),
        }
        clf = _classification_fields_from_raw(summary_agg)
        if all(v is None for v in clf.values()):
            clf = _classification_fields_from_raw(wf_doc)
        merged.update(clf)
        if merged["composite_score"] is None:
            merged["composite_score"] = _derive_composite_score(
                merged,
                aggregated=agg_for_composite,
                prediction_type=prediction_type,
                score_refs=refs or None,
            )
        return _build_production_metrics(
            stage_key="walk_forward_aggregate",
            stage_label="Walk-Forward Aggregate",
            source_file=source_file,
            source_path=source_path,
            raw_metrics=merged,
            n_folds=n_folds,
            prediction_type=prediction_type,
        )

    test_doc = dict(metrics_doc.get("test") or {})
    if _has_any_metric(test_doc):
        if test_doc.get("composite_score") is None:
            test_doc["composite_score"] = _derive_composite_score(
                test_doc,
                prediction_type=prediction_type,
                score_refs=score_refs,
            )
        return _build_production_metrics(
            stage_key="test",
            stage_label="Test Metrics",
            source_file="metrics.json",
            source_path="$.test",
            raw_metrics=test_doc,
            prediction_type=prediction_type,
        )

    val_doc = dict(metrics_doc.get("validation") or {})
    if _has_any_metric(val_doc):
        if val_doc.get("composite_score") is None:
            val_doc["composite_score"] = _derive_composite_score(
                val_doc,
                prediction_type=prediction_type,
                score_refs=score_refs,
            )
        return _build_production_metrics(
            stage_key="validation",
            stage_label="Validation Metrics",
            source_file="metrics.json",
            source_path="$.validation",
            raw_metrics=val_doc,
            prediction_type=prediction_type,
        )

    summary_test = dict(summary_doc.get("test_metrics") or {})
    if _has_any_metric(summary_test):
        if summary_test.get("composite_score") is None:
            summary_test["composite_score"] = _derive_composite_score(
                summary_test,
                prediction_type=prediction_type,
                score_refs=score_refs,
            )
        return _build_production_metrics(
            stage_key="test",
            stage_label="Test Metrics",
            source_file="training_summary.json",
            source_path="$.test_metrics",
            raw_metrics=summary_test,
            prediction_type=prediction_type,
        )

    return _build_production_metrics(
        stage_key="unknown",
        stage_label="Unknown",
        source_file="—",
        source_path="—",
        raw_metrics={},
        prediction_type=prediction_type,
    )


def _aggregate_fold_metric_stats(fold_results: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    vals: list[float] = []
    for row in fold_results:
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
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return round(mean, 6), 0.0
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(mean, 6), round(variance ** 0.5, 6)


def _enrich_validation_metrics_for_walk_forward(
    metrics: dict[str, Any],
    *,
    wf_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Backfill validation R² / MAPE for walk-forward models from fold aggregates."""
    val = dict(metrics.get("validation") or {})
    needs_r2 = val.get("r2") is None
    needs_mape = val.get("mape") is None
    if not needs_r2 and not needs_mape:
        return metrics

    wf = dict(metrics.get("walk_forward") or {})
    summary = dict(wf_summary or {})
    agg = dict(summary.get("aggregated") or wf or {})
    fold_results = summary.get("fold_results") if isinstance(summary.get("fold_results"), list) else None

    if needs_r2:
        r2 = _first_num(agg.get("mean_r2"), wf.get("mean_r2"))
        if r2 is None and fold_results:
            r2, std_r2 = _aggregate_fold_metric_stats(fold_results, "r2")
            if val.get("std_r2") is None and std_r2 is not None:
                val["std_r2"] = std_r2
        if r2 is not None:
            val["r2"] = r2

    if needs_mape:
        mape = _first_num(agg.get("mean_mape"), wf.get("mean_mape"))
        if mape is None and fold_results:
            mape, std_mape = _aggregate_fold_metric_stats(fold_results, "mape")
            if val.get("std_mape") is None and std_mape is not None:
                val["std_mape"] = std_mape
        if mape is not None:
            val["mape"] = mape

    if val.get("std_r2") is None:
        std_r2 = _first_num(agg.get("std_r2"), wf.get("std_r2"))
        if std_r2 is None and fold_results:
            _, std_r2 = _aggregate_fold_metric_stats(fold_results, "r2")
        if std_r2 is not None:
            val["std_r2"] = std_r2
    if val.get("std_mape") is None:
        std_mape = _first_num(agg.get("std_mape"), wf.get("std_mape"))
        if std_mape is None and fold_results:
            _, std_mape = _aggregate_fold_metric_stats(fold_results, "mape")
        if std_mape is not None:
            val["std_mape"] = std_mape

    out = dict(metrics)
    out["validation"] = val
    return out


def _count_csv_selected_features(sel_rows: dict[str, Any]) -> int:
    if not sel_rows.get("available"):
        return 0
    rows = list(sel_rows.get("rows") or [])
    if not rows:
        return 0
    has_selected_col = any(isinstance(r, dict) and "selected" in r for r in rows)
    if not has_selected_col:
        return len(rows)
    count = 0
    for row in rows:
        if isinstance(row, dict) and _csv_row_is_selected(row):
            count += 1
    return count


def _load_walk_forward_artifacts(pkg: str, *, prediction_type: str = "regression") -> dict[str, Any]:
    wf_dir = os.path.join(pkg, "walk_forward")
    summary_path = os.path.join(wf_dir, "summary.json")
    has_wf_dir = os.path.isdir(wf_dir)
    summary_art = _load_json_artifact(summary_path)
    summary = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
    wf_cfg = dict(summary.get("config") or {})
    wf_meta = dict(summary.get("meta") or {})
    champion_agg_art = _load_json_artifact(os.path.join(wf_dir, "champion_aggregate.json"))
    champion_agg = champion_agg_art.get("data") if isinstance(champion_agg_art.get("data"), dict) else {}
    if not wf_meta and wf_cfg:
        from .split import walk_forward_meta_from_config

        wf_meta = walk_forward_meta_from_config(wf_cfg)
    champ_meta = dict(champion_agg.get("meta") or {}) if isinstance(champion_agg, dict) else {}
    if not wf_meta and champ_meta:
        wf_meta = champ_meta
    if not wf_cfg and isinstance(champion_agg, dict):
        wf_cfg = dict(champion_agg.get("config") or {})
    hpo_cfg = dict(wf_cfg.get("hyperparameter_optimization") or {})
    best_params_art = _load_json_artifact(os.path.join(wf_dir, "best_parameters.json"))
    best_params = best_params_art.get("data") if isinstance(best_params_art.get("data"), dict) else {}
    score_refs = dict(summary.get("reference_stats") or {})
    if not score_refs and isinstance(best_params, dict):
        score_refs = dict(best_params.get("reference_stats") or {})
    fold_results = summary.get("fold_results") if isinstance(summary.get("fold_results"), list) else []
    fold_results = enrich_fold_results_composite(
        fold_results,
        prediction_type=prediction_type,
        score_refs=score_refs or None,
    )
    if summary_art.get("available") and isinstance(summary_art.get("data"), dict):
        summary_art["data"]["fold_results"] = fold_results
        summary = summary_art["data"]
    summary_agg = enrich_aggregated_from_fold_results(
        dict(summary.get("aggregated") or {}),
        fold_results,
    )

    champ_folds = champion_agg.get("fold_results") if isinstance(champion_agg.get("fold_results"), list) else []
    if champ_folds:
        champ_folds = enrich_fold_results_composite(
            champ_folds,
            prediction_type=prediction_type,
            score_refs=score_refs or None,
        )
        champion_agg["fold_results"] = champ_folds
        if champion_agg_art.get("available") and isinstance(champion_agg_art.get("data"), dict):
            champion_agg_art["data"]["fold_results"] = champ_folds

    hpo_enabled = bool(hpo_cfg.get("enabled", True)) if hpo_cfg else bool(best_params_art.get("available"))

    sel_rows = _load_csv_artifact(os.path.join(wf_dir, "selected_features.csv"))
    csv_selected_count = _count_csv_selected_features(sel_rows)
    sel_count = csv_selected_count
    if not sel_count:
        wf_sel = summary.get("selected_features")
        if isinstance(wf_sel, list) and wf_sel:
            sel_count = len(wf_sel)
        elif isinstance(wf_sel, int) and wf_sel > 0:
            sel_count = int(wf_sel)

    # Heavy CSVs (history / HPO dumps) are deferred — Overview only needs counts/JSON.
    empty_csv = {"available": False, "path": "", "rows": []}
    top_trials_art = empty_csv
    hpo_trials = best_params.get("n_trials_completed") or best_params.get("n_trials_target")

    return {
        "available": has_wf_dir and summary_art.get("available"),
        "directory": "walk_forward/",
        "summary": summary_art,
        "selected_features": sel_rows,
        "feature_selection_history": empty_csv,
        "feature_stability": empty_csv,
        "best_parameters": best_params_art,
        "champion_aggregate": champion_agg_art,
        "top_trials": top_trials_art,
        "optimization_history": empty_csv,
        "parameter_importance": empty_csv,
        "_heavy_deferred": True,
        "display": {
            "validation_strategy": wf_meta.get("validation_strategy_label") or "Walk Forward",
            "validation_strategy_ui": wf_meta.get("validation_strategy_ui"),
            "validation_strategy_label": wf_meta.get("validation_strategy_label"),
            "n_folds": wf_meta.get("n_folds") or wf_cfg.get("n_folds") or (summary.get("aggregated") or {}).get("n_folds"),
            "window_mode": wf_meta.get("window_mode") or wf_cfg.get("window_mode"),
            "fold_placement": wf_meta.get("fold_placement") or wf_cfg.get("fold_placement"),
            "fold_placement_label": wf_meta.get("fold_placement_label"),
            "train_window_size": wf_meta.get("train_window_size") or wf_cfg.get("train_window_size"),
            "validation_window_size": wf_meta.get("validation_window_size") or wf_cfg.get("validation_window_size"),
            "test_holdout_pct": wf_cfg.get("test_holdout_pct") or summary.get("test_holdout"),
            "mean_validation_rmse": summary_agg.get("mean_rmse"),
            "std_validation_rmse": summary_agg.get("std_rmse"),
            "mean_validation_mae": summary_agg.get("mean_mae"),
            "std_validation_mae": summary_agg.get("std_mae"),
            "mean_validation_r2": summary_agg.get("mean_r2"),
            "std_validation_r2": summary_agg.get("std_r2"),
            "mean_validation_mape": summary_agg.get("mean_mape"),
            "std_validation_mape": summary_agg.get("std_mape"),
            "mean_directional_accuracy_pct": summary_agg.get("mean_directional_accuracy_pct"),
            "std_directional_accuracy_pct": summary_agg.get("std_directional_accuracy_pct"),
            "optimization_metric": (
                (summary.get("feature_selection") or {}).get("optimization_metric")
                or wf_cfg.get("optimization_metric")
            ),
            "feature_selection_method": wf_cfg.get("feature_selection_method"),
            "selected_feature_count": sel_count,
            "hyperparameter_optimization_enabled": hpo_enabled,
            "hpo_n_trials": hpo_trials,
            "best_composite_score": best_params.get("best_display_score") or best_params.get("best_objective"),
            "best_parameters": best_params.get("best_parameters") or best_params.get("full_parameters"),
            "production_composite_score": (
                champion_agg.get("composite_score")
                or (champion_agg.get("validation_metrics") or {}).get("composite_score")
            ),
        },
    }


def load_deferred_walk_forward_csvs(pkg: str, walk_forward: dict[str, Any]) -> dict[str, Any]:
    """Fill HPO / history CSVs that were skipped in the fast model-detail path."""
    if not isinstance(walk_forward, dict):
        return {}
    if not walk_forward.get("_heavy_deferred"):
        return walk_forward
    wf_dir = os.path.join(pkg, "walk_forward")
    walk_forward["feature_selection_history"] = _load_csv_artifact(
        os.path.join(wf_dir, "feature_selection_history.csv")
    )
    walk_forward["feature_stability"] = _load_csv_artifact(
        os.path.join(wf_dir, "feature_stability.csv")
    )
    top_trials_art = _load_csv_artifact(os.path.join(wf_dir, "top_trials.csv"))
    walk_forward["top_trials"] = top_trials_art
    walk_forward["optimization_history"] = _load_csv_artifact(
        os.path.join(wf_dir, "optimization_history.csv")
    )
    walk_forward["parameter_importance"] = _load_csv_artifact(
        os.path.join(wf_dir, "parameter_importance.csv")
    )
    disp = walk_forward.get("display") if isinstance(walk_forward.get("display"), dict) else {}
    if disp.get("hpo_n_trials") is None and top_trials_art.get("available"):
        disp["hpo_n_trials"] = len(top_trials_art.get("rows") or [])
        walk_forward["display"] = disp
    walk_forward["_heavy_deferred"] = False
    return walk_forward


def _list_package_files(pkg: str) -> list[str]:
    out: list[str] = []
    if not os.path.isdir(pkg):
        return out
    for root, _, files in os.walk(pkg):
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(root, name), pkg).replace("\\", "/")
            out.append(rel)
    return out


def _load_package_doc(entry: str, pkg: str) -> dict[str, Any]:
    registry_path = os.path.join(pkg, "registry.json")
    summary_path = os.path.join(pkg, "training_summary.json")
    config_path = os.path.join(pkg, "config.json")
    metrics_path = os.path.join(pkg, "metrics.json")

    doc: dict[str, Any] = {"model_name": entry, "status": "unknown"}
    summary: dict[str, Any] = {}
    config: dict[str, Any] = {}
    metrics_doc: dict[str, Any] = {}
    strategy: dict[str, str] = {"key": "time_series", "label": "Time Series Split"}
    wf_summary_doc: dict[str, Any] = {}

    if os.path.isfile(registry_path):
        reg = _load_json(registry_path)
        if isinstance(reg, dict):
            doc.update(reg)

    if os.path.isfile(summary_path):
        loaded = _load_json(summary_path)
        if isinstance(loaded, dict):
            summary = loaded
            doc.setdefault("dataset", summary.get("dataset"))
            doc.setdefault("algorithm", summary.get("algorithm"))
            doc.setdefault("target", summary.get("target"))
            doc.setdefault("trained_at", summary.get("trained_at"))
            doc.setdefault("rows", summary.get("rows"))
    if os.path.isfile(config_path):
        loaded = _load_json(config_path)
        if isinstance(loaded, dict):
            config = loaded
            doc.setdefault("algorithm", config.get("algorithm_label") or _algorithm_label(config.get("algorithm")))
            doc.setdefault("dataset", config.get("dataset"))
            doc.setdefault("target", config.get("target"))
            doc.setdefault("trained_at", config.get("trained_at"))
            if config.get("package_anchor") or config.get("prediction_package_id"):
                doc.setdefault(
                    "package_anchor",
                    config.get("package_anchor") or config.get("prediction_package_id"),
                )
            matrix = config.get("matrix_report") or {}
            shape = matrix.get("x_shape")
            if doc.get("rows") is None and isinstance(shape, list) and shape:
                doc["rows"] = int(shape[0])
            ds_meta = config.get("dataset_metadata") or {}
            if doc.get("rows") is None and ds_meta.get("row_count") is not None:
                doc["rows"] = int(ds_meta["row_count"])
    if summary.get("validation_strategy_ui"):
        split_info = dict(config.get("split_info") or {})
        split_info["validation_strategy_ui"] = summary["validation_strategy_ui"]
        if summary.get("validation_strategy_label"):
            split_info["validation_strategy_label"] = summary["validation_strategy_label"]
        config["split_info"] = split_info
    if config:
        try:
            strategy = _detect_validation_strategy(config)
            doc.setdefault("validation_strategy", strategy.get("label"))
        except Exception:
            pass

    if os.path.isfile(metrics_path):
        metrics = _load_json(metrics_path)
        if isinstance(metrics, dict):
            metrics_doc = metrics

    wf_summary_path = os.path.join(pkg, "walk_forward", "summary.json")
    wf_loaded = _load_json(wf_summary_path)
    if isinstance(wf_loaded, dict):
        wf_summary_doc = wf_loaded

    score_refs: dict[str, float] | None = None
    bp_path = os.path.join(pkg, "walk_forward", "best_parameters.json")
    if os.path.isfile(bp_path):
        bp_loaded = _load_json(bp_path)
        if isinstance(bp_loaded, dict):
            refs = bp_loaded.get("reference_stats")
            if isinstance(refs, dict) and refs:
                score_refs = refs

    prediction_type = str(config.get("prediction_type") or summary.get("prediction_type") or "regression")
    doc["prediction_type"] = prediction_type

    production = _resolve_authoritative_metrics(
        strategy=strategy,
        metrics_doc=metrics_doc,
        summary_doc=summary,
        wf_summary_doc=wf_summary_doc,
        prediction_type=prediction_type,
        score_refs=score_refs,
    )
    doc["production_metrics"] = production
    doc["metrics"] = {
        "rmse": production.get("rmse"),
        "mae": production.get("mae"),
        "directional_accuracy_pct": production.get("directional_accuracy_pct"),
        "composite_score": production.get("composite_score"),
    }
    if strategy.get("key") in ("walk_forward", "rolling_window") and production.get("n_folds"):
        doc["validation_strategy"] = f"{strategy.get('label')} ({int(production['n_folds'])} folds)"
    else:
        doc.setdefault("validation_strategy", strategy.get("label"))

    if not doc.get("algorithm") and config.get("algorithm"):
        doc["algorithm"] = _algorithm_label(config.get("algorithm"))

    meta_path = os.path.join(pkg, "metadata.json")
    pkg_meta = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    if not isinstance(pkg_meta, dict):
        pkg_meta = {}
    from .artifacts import resolve_training_row_count

    doc["rows"] = resolve_training_row_count(
        metadata=pkg_meta,
        dataset_metadata=config.get("dataset_metadata") if isinstance(config, dict) else None,
        matrix_report=config.get("matrix_report") if isinstance(config, dict) else None,
    ) or doc.get("rows")

    return doc


def _protection_info(data_dir: str, doc: dict[str, Any]) -> tuple[bool, str | None]:
    """Only hard-block delete for true deployment statuses.

    Last-selected / "active" models must stay deletable — selection is just
    registry navigation, not a deploy lock. ``delete_model`` clears active.
    """
    status = str(doc.get("status") or "").strip().lower()
    if status in _PROTECTED_STATUSES:
        return True, "deployed"
    return False, None


def _is_active_model(data_dir: str, model_name: str) -> bool:
    active = get_active_model(data_dir)
    if not active:
        return False
    return active == safe_model_name(str(model_name or ""))


def _feature_count_from_config_path(config_path: str) -> int | None:
    cfg = _load_json(config_path)
    if not isinstance(cfg, dict):
        return None
    fc = cfg.get("feature_count")
    if fc is None:
        sel = cfg.get("selected_features") or cfg.get("features")
        if isinstance(sel, list):
            fc = len(sel)
    if fc is None:
        return None
    try:
        return int(fc)
    except (TypeError, ValueError):
        return None


def _feature_count_for_package(data_dir: str, entry: str) -> int | None:
    paths = model_artifact_paths(data_dir, entry)
    fc = _feature_count_from_config_path(paths["config_json"])
    if fc is not None:
        return fc
    selected = _selected_feature_names(data_dir, entry, paths)
    if selected:
        return len(selected)
    return None


def _note_preview(text: str, *, limit: int = 60) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _load_model_note_doc(pkg: str) -> dict[str, Any]:
    path = os.path.join(pkg, "model_note.json")
    if not os.path.isfile(path):
        return {"note": "", "updated_at": None}
    doc = _load_json(path)
    if not isinstance(doc, dict):
        return {"note": "", "updated_at": None}
    return {
        "note": str(doc.get("note") or ""),
        "updated_at": doc.get("updated_at"),
    }


def get_model_note(data_dir: str, model_name: str) -> dict[str, Any]:
    safe = safe_model_name(model_name)
    pkg = model_artifact_paths(data_dir, safe)["package_dir"]
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"Model package not found: {safe}")
    doc = _load_model_note_doc(pkg)
    note = str(doc.get("note") or "")
    return {
        "model_name": safe,
        "note": note,
        "note_preview": _note_preview(note),
        "has_note": bool(note.strip()),
        "updated_at": doc.get("updated_at"),
    }


def set_model_note(data_dir: str, model_name: str, note: str) -> dict[str, Any]:
    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    pkg = paths["package_dir"]
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"Model package not found: {safe}")
    text = str(note or "")
    payload = {
        "note": text,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    note_path = paths.get("model_note_json") or os.path.join(pkg, "model_note.json")
    with open(note_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    stripped = text.strip()
    return {
        "model_name": safe,
        "note": text,
        "note_preview": _note_preview(stripped),
        "has_note": bool(stripped),
        "updated_at": payload["updated_at"],
    }


def _interval_from_dataset_meta(data_dir: str, dataset_name: str) -> int | None:
    safe = str(dataset_name or "").strip()
    if not safe:
        return None
    meta_path = os.path.join(data_dir, "datasets", f"{safe}.json")
    if not os.path.isfile(meta_path):
        return None
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        return None
    for key in ("sampling_interval_sec", "interval_sec"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    sampling = meta.get("sampling")
    if isinstance(sampling, dict) and sampling.get("interval_sec") is not None:
        try:
            return int(sampling["interval_sec"])
        except (TypeError, ValueError):
            pass
    expected = meta.get("expected_spec") or meta.get("dataset_configuration") or {}
    if isinstance(expected, dict):
        for key in ("sampling_interval_sec", "interval_sec"):
            val = expected.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
    return None


def _interval_from_dataset_name(dataset_name: str) -> int | None:
    name = str(dataset_name or "")
    match = re.search(r"_(\d+)s_", name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _resolve_sampling_interval_sec(
    data_dir: str,
    *,
    config: dict[str, Any] | None = None,
    dataset_name: str | None = None,
) -> int | None:
    cfg = config if isinstance(config, dict) else {}
    ds_name = str(dataset_name or cfg.get("dataset") or "").strip()

    replay = cfg.get("replay_config") or {}
    if isinstance(replay, dict):
        sampling = replay.get("sampling") or {}
        if isinstance(sampling, dict) and sampling.get("interval_sec") is not None:
            try:
                return int(sampling["interval_sec"])
            except (TypeError, ValueError):
                pass
        ds_cfg = replay.get("dataset_configuration") or {}
        if isinstance(ds_cfg, dict) and ds_cfg.get("sampling_interval_sec") is not None:
            try:
                return int(ds_cfg["sampling_interval_sec"])
            except (TypeError, ValueError):
                pass

    ds_meta = cfg.get("dataset_metadata") or {}
    if isinstance(ds_meta, dict) and ds_meta.get("sampling_interval_sec") is not None:
        try:
            return int(ds_meta["sampling_interval_sec"])
        except (TypeError, ValueError):
            pass

    from_meta = _interval_from_dataset_meta(data_dir, ds_name)
    if from_meta is not None:
        return from_meta
    return _interval_from_dataset_name(ds_name)


def resolve_model_registry_family(row: dict[str, Any] | None) -> str:
    """Map a registry row to Model Registry list tab: ``regression`` | ``triple_barrier``.

    Prefer persisted ``label_strategy``; fall back to OLE primary target ``label_id``.
    """
    data = row if isinstance(row, dict) else {}
    strat = str(
        data.get("label_strategy")
        or data.get("label_strategy_id")
        or ""
    ).strip().lower()
    if strat == "triple_barrier":
        return "triple_barrier"
    target = str(data.get("target") or "").strip().lower()
    if target == "label_id":
        return "triple_barrier"
    return "regression"


def _table_row(data_dir: str, entry: str, pkg: str) -> dict[str, Any]:
    doc = _load_package_doc(entry, pkg)
    config_path = os.path.join(pkg, "config.json")
    config_doc = _load_json(config_path)
    config = config_doc if isinstance(config_doc, dict) else {}
    training_config_path = os.path.join(pkg, "training_config.json")
    training_config = _load_json(training_config_path) if os.path.isfile(training_config_path) else {}
    if not isinstance(training_config, dict):
        training_config = {}
    sampling_interval_sec = _resolve_sampling_interval_sec(
        data_dir,
        config=config,
        dataset_name=str(doc.get("dataset") or ""),
    )
    metrics = doc.get("metrics") or {}
    production = doc.get("production_metrics") or {}
    protected, reason = _protection_info(data_dir, doc)
    size_bytes = _folder_size_bytes(pkg)
    feature_count = _feature_count_for_package(data_dir, entry)
    note_doc = _load_model_note_doc(pkg)
    note_text = str(note_doc.get("note") or "")
    meta_path = os.path.join(pkg, "metadata.json")
    meta_raw = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    lineage = meta_raw.get("lineage") if isinstance(meta_raw.get("lineage"), dict) else {}
    lifecycle_mode = str(lineage.get("lifecycle_mode") or "").strip().lower()
    model_name = doc.get("model_name") or entry
    research = is_research_experiment_model(str(model_name), doc, config, meta_raw)
    label_strategy = str(
        config.get("label_strategy")
        or training_config.get("label_strategy")
        or doc.get("label_strategy")
        or "fixed_horizon"
    ).strip().lower() or "fixed_horizon"
    label_strategy_params = (
        config.get("label_strategy_params")
        if isinstance(config.get("label_strategy_params"), dict)
        else (
            training_config.get("label_strategy_params")
            if isinstance(training_config.get("label_strategy_params"), dict)
            else {}
        )
    )
    row = {
        "model_name": model_name,
        "algorithm": doc.get("algorithm") or "—",
        "dataset": doc.get("dataset") or "—",
        "sampling_interval_sec": sampling_interval_sec,
        "validation_strategy": doc.get("validation_strategy") or "—",
        "target": doc.get("target") or "—",
        "prediction_type": doc.get("prediction_type") or "regression",
        "label_strategy": label_strategy,
        "label_strategy_params": dict(label_strategy_params or {}),
        "label_run_id": str(
            config.get("label_run_id")
            or training_config.get("label_run_id")
            or doc.get("label_run_id")
            or ""
        ).strip()
        or None,
        "package_anchor": doc.get("package_anchor") or doc.get("prediction_package_id"),
        "rows": doc.get("rows"),
        "feature_count": feature_count,
        "lifecycle_mode": lifecycle_mode or None,
        "lineage": lineage or None,
        "generation": lineage.get("generation"),
        "parent_model_id": lineage.get("parent_model_id"),
        "note": note_text,
        "note_preview": _note_preview(note_text),
        "has_note": bool(note_text.strip()),
        "note_updated_at": note_doc.get("updated_at"),
        "trained_at": doc.get("trained_at"),
        "metrics": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
            "composite_score": metrics.get("composite_score"),
        },
        "production_metrics": production,
        "status": doc.get("status") or "ready",
        "size_bytes": size_bytes,
        "protected": protected,
        "protected_reason": reason,
        "is_active": _is_active_model(data_dir, str(model_name)),
        "report_url": "",
        "is_research_experiment": research,
        "registry_scope": "experiment" if research else "production",
        "experiment_id": doc.get("experiment_id") or config.get("experiment_id"),
    }
    row["registry_family"] = resolve_model_registry_family(row)
    # Phase X — live disk status (models remain listed if sources deleted).
    ds_name = str(row.get("dataset") or "").strip()
    run_id = str(row.get("label_run_id") or "").strip()
    try:
        from chain_replay_ml.training.dataset_loader import dataset_parquet_exists

        row["dataset_status"] = (
            "available" if ds_name and dataset_parquet_exists(data_dir, ds_name) else "deleted"
        )
    except Exception:
        row["dataset_status"] = "unknown"
    if run_id:
        try:
            from chain_replay_ml.label_runs import label_run_exists

            row["label_run_status"] = "available" if label_run_exists(data_dir, run_id) else "deleted"
        except Exception:
            row["label_run_status"] = "unknown"
    else:
        row["label_run_status"] = "n/a"
    return row


def list_trained_models(
    data_dir: str,
    *,
    lightweight: bool = False,
    include_experiments: bool = False,
) -> list[dict[str, Any]]:
    """List model packages for the production Model Registry.

    Analysis Lab experiment packages (``Exp_*``) are excluded by default so the
    Model Registry only shows Create Model Builder / production candidates.
    Pass ``include_experiments=True`` for research lookups (SHAP, get-by-name).
    """
    base = models_dir(data_dir)
    if not os.path.isdir(base):
        return []

    rows: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(base)):
        if entry.startswith("."):
            continue
        pkg = os.path.join(base, entry)
        if not os.path.isdir(pkg):
            continue
        try:
            row = _table_row(data_dir, entry, pkg)
        except Exception:
            # Corrupt / truncated package JSON must not block Model Builder catalog load.
            research = is_research_experiment_model(entry)
            row = {
                "model_name": entry,
                "metrics": {
                    "rmse": None,
                    "mae": None,
                    "directional_accuracy_pct": None,
                    "composite_score": None,
                },
                "status": "corrupt",
                "trained_at": None,
                "is_research_experiment": research,
                "registry_scope": "experiment" if research else "production",
            }
        if not include_experiments and row.get("is_research_experiment"):
            continue
        if lightweight:
            row = {
                "model_name": row["model_name"],
                "metrics": row["metrics"],
                "status": row["status"],
                "trained_at": row["trained_at"],
                "is_research_experiment": bool(row.get("is_research_experiment")),
                "registry_scope": row.get("registry_scope") or "production",
            }
        rows.append(row)

    rows.sort(key=lambda r: str(r.get("trained_at") or r.get("model_name") or ""), reverse=True)
    if not lightweight:
        _enrich_rows_with_lifecycle(data_dir, rows)
    return rows


def _enrich_rows_with_lifecycle(data_dir: str, rows: list[dict[str, Any]]) -> None:
    try:
        from .lifecycle_store import (
            get_history_by_model_name,
            list_model_champions,
            rebuild_lifecycle_index,
        )

        rebuild_lifecycle_index(data_dir)
        champions = {c["current_model_name"]: c for c in list_model_champions(data_dir)}
    except Exception:
        champions = {}
    for row in rows:
        name = str(row.get("model_name") or "")
        hist = None
        try:
            hist = get_history_by_model_name(data_dir, name)
        except Exception:
            hist = None
        if hist:
            row["model_id"] = hist.get("model_id")
            row["version_label"] = hist.get("version_label")
            row["version_number"] = hist.get("version_number")
            row["history_id"] = hist.get("history_id")
            row["lifecycle_event"] = hist.get("lifecycle")
        champ = champions.get(name)
        row["is_champion"] = bool(champ)
        if champ:
            row["model_id"] = champ.get("model_id")
            row["current_version"] = champ.get("current_version")
            row["version_count"] = champ.get("current_version_number")


def list_champion_models(data_dir: str) -> list[dict[str, Any]]:
    """One row per model family — current champion only."""
    from .lifecycle_store import list_history_for_model, list_model_champions, rebuild_lifecycle_index

    rebuild_lifecycle_index(data_dir)
    champions = list_model_champions(data_dir)
    rows: list[dict[str, Any]] = []
    for ch in champions:
        name = str(ch.get("current_model_name") or "")
        if not name:
            continue
        pkg = model_package_dir(data_dir, name)
        if not os.path.isdir(pkg):
            continue
        row = _table_row(data_dir, name, pkg)
        family = str(ch.get("display_name") or ch.get("model_id") or name).strip()
        row["family_name"] = family
        row["display_name"] = family
        row["package_name"] = name
        row["model_id"] = ch.get("model_id")
        row["current_version"] = ch.get("current_version")
        row["version_number"] = ch.get("current_version_number")
        row["version_label"] = ch.get("current_version")
        row["is_champion"] = True
        row["created_on"] = ch.get("created_on")
        row["registry_updated_on"] = ch.get("updated_on")
        history = list_history_for_model(data_dir, model_id=str(ch.get("model_id") or ""))
        row["version_count"] = len(history) or ch.get("current_version_number")
        # Do not overlay deprecated lifecycle current_metrics — package production_metrics win.
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("trained_at") or ""), reverse=True)
    return rows


def get_trained_model(data_dir: str, model_name: str) -> dict[str, Any] | None:
    safe = safe_model_name(model_name)
    for row in list_trained_models(
        data_dir, lightweight=False, include_experiments=True
    ):
        if row.get("model_name") == safe:
            return row
    return None


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _csv_row_is_selected(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    sel = row.get("selected")
    if sel is None:
        return True
    s = str(sel).strip().lower()
    if s in ("no", "false", "0", "n"):
        return False
    return s in ("", "yes", "true", "1", "y")


def _feature_name_from_csv_row(row: dict[str, Any]) -> str | None:
    feat = row.get("feature") or row.get("Feature")
    if not feat and row:
        feat = next(iter(row.values()), None)
    name = str(feat or "").strip()
    return name or None


def _selected_feature_names(data_dir: str, safe: str, paths: dict[str, str]) -> list[str]:
    """Final model feature set — not the full walk-forward candidate pool."""
    cfg = _load_json(paths["config_json"])
    if isinstance(cfg, dict):
        for key in ("selected_features", "features"):
            sel = cfg.get(key)
            if isinstance(sel, list) and sel:
                names = [str(f).strip() for f in sel if str(f).strip()]
                if names:
                    return _dedupe_names(names)

    summary = _load_json(os.path.join(paths["package_dir"], "walk_forward", "summary.json"))
    if isinstance(summary, dict):
        sf = summary.get("selected_features")
        if isinstance(sf, list) and sf:
            names = [str(f).strip() for f in sf if str(f).strip()]
            if names:
                return _dedupe_names(names)

    names: list[str] = []
    sel_csv = _load_csv_artifact(os.path.join(paths["package_dir"], "walk_forward", "selected_features.csv"))
    if sel_csv.get("available"):
        rows = list(sel_csv.get("rows") or [])
        has_selected_col = any(isinstance(r, dict) and "selected" in r for r in rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            if has_selected_col and not _csv_row_is_selected(row):
                continue
            name = _feature_name_from_csv_row(row)
            if name:
                names.append(name)
    return _dedupe_names(names)


def get_model_summary(data_dir: str, model_name: str) -> dict[str, Any]:
    """Lightweight registry summary for UI dialogs (JSON metadata only)."""
    safe = safe_model_name(model_name)
    pkg = os.path.join(models_dir(data_dir), safe)
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"Model package not found: {safe}")
    doc = _load_package_doc(safe, pkg)
    protected, reason = _protection_info(data_dir, doc)
    prod = dict(doc.get("production_metrics") or {})
    metrics = dict(doc.get("metrics") or {})
    paths = model_artifact_paths(data_dir, safe)
    feature_count = _feature_count_for_package(data_dir, safe)
    top_features: list[dict[str, Any]] = []
    fi_csv = _load_csv_artifact(paths["feature_importance_csv"])
    if fi_csv.get("available"):
        for row in fi_csv.get("rows") or []:
            feat = row.get("Feature") or row.get("feature")
            imp = row.get("Importance") or row.get("importance_pct") or row.get("importance")
            if not feat:
                continue
            try:
                pct = round(float(imp), 4) if imp not in (None, "") else 0.0
            except (TypeError, ValueError):
                pct = 0.0
            top_features.append({"feature": str(feat), "importance_pct": pct})
        top_features.sort(key=lambda x: x.get("importance_pct") or 0.0, reverse=True)
        top_features = top_features[:20]
    selected_features = _selected_feature_names(data_dir, safe, paths)
    config_doc = _load_json(paths["config_json"])
    config = config_doc if isinstance(config_doc, dict) else {}
    sampling_interval_sec = _resolve_sampling_interval_sec(
        data_dir,
        config=config,
        dataset_name=str(doc.get("dataset") or ""),
    )
    return {
        "model_name": safe,
        "status": doc.get("status"),
        "algorithm": doc.get("algorithm"),
        "dataset": doc.get("dataset"),
        "sampling_interval_sec": sampling_interval_sec,
        "target": doc.get("target"),
        "trained_at": doc.get("trained_at"),
        "validation_strategy": doc.get("validation_strategy"),
        "rows": doc.get("rows"),
        "feature_count": feature_count,
        "production_metrics": prod,
        "metrics": metrics,
        "composite_score": prod.get("composite_score") or metrics.get("composite_score"),
        "rmse": prod.get("rmse") or metrics.get("rmse"),
        "mae": prod.get("mae") or metrics.get("mae"),
        "directional_accuracy_pct": prod.get("directional_accuracy_pct") or metrics.get("directional_accuracy_pct"),
        "protected": protected,
        "protection_reason": reason,
        "stage_label": prod.get("stage_label"),
        "top_features": top_features,
        "selected_features": selected_features,
    }


def assert_model_deletable(data_dir: str, model_name: str) -> None:
    row = get_trained_model(data_dir, model_name)
    if not row:
        raise FileNotFoundError(f"Model package not found: {safe_model_name(model_name)}")
    if row.get("protected"):
        raise ModelDeleteBlockedError("This model is currently in use.")


def load_model_detail(data_dir: str, model_name: str) -> dict[str, Any]:
    """Load full saved training package for detail view — disk artifacts only."""
    t_all = time.perf_counter()
    stages: list[dict[str, Any]] = []

    def _mark(label: str, started: float) -> None:
        stages.append({"label": label, "ms": round((time.perf_counter() - started) * 1000, 1)})

    t0 = time.perf_counter()
    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    pkg = paths["package_dir"]
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"Model package not found: {safe}")
    _mark("resolve_package", t0)

    t0 = time.perf_counter()
    metadata_art = _load_json_artifact(os.path.join(pkg, "metadata.json"))
    training_config_art = _load_json_artifact(os.path.join(pkg, "training_config.json"))
    if not training_config_art.get("available"):
        training_config_art = _load_json_artifact(paths["config_json"])

    config_data = training_config_art.get("data") if isinstance(training_config_art.get("data"), dict) else {}
    prediction_type = str(config_data.get("prediction_type") or "regression")

    metrics_art = _load_json_artifact(paths["metrics_json"])
    metrics = metrics_art.get("data") if isinstance(metrics_art.get("data"), dict) else {}

    summary_art = _load_json_artifact(paths["training_summary_json"])
    summary = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
    merged_config = dict(config_data)
    if summary.get("validation_strategy_ui"):
        split_info = dict(merged_config.get("split_info") or {})
        split_info["validation_strategy_ui"] = summary["validation_strategy_ui"]
        if summary.get("validation_strategy_label"):
            split_info["validation_strategy_label"] = summary["validation_strategy_label"]
        merged_config["split_info"] = split_info
    strategy = _detect_validation_strategy(merged_config)
    training_metadata_art = _load_json_artifact(paths["training_metadata_json"])
    training_monitor_csv_art = _load_csv_artifact(paths["training_monitor_csv"])

    registry_art = _load_json_artifact(paths["registry_json"])
    fingerprint_art = _load_json_artifact(paths["pipeline_fingerprint_json"])
    snapshot_art = _load_json_artifact(paths["dataset_build_snapshot_json"])
    _mark("load_core_json", t0)

    t0 = time.perf_counter()
    feature_importance: list[dict[str, Any]] = []
    fi_csv = _load_csv_artifact(paths["feature_importance_csv"])
    if fi_csv.get("available"):
        for row in fi_csv.get("rows") or []:
            feat = row.get("Feature") or row.get("feature")
            imp = row.get("Importance") or row.get("importance_pct") or row.get("importance")
            if feat:
                feature_importance.append({
                    "feature": feat,
                    "importance_pct": float(imp) if imp not in (None, "") else 0.0,
                })
        feature_importance.sort(key=lambda x: x.get("importance_pct") or 0.0, reverse=True)
    _mark("feature_importance", t0)

    t0 = time.perf_counter()
    walk_forward = _load_walk_forward_artifacts(pkg, prediction_type=prediction_type)
    if strategy["key"] == "rolling_window" and walk_forward.get("display"):
        walk_forward["display"]["validation_strategy"] = (
            walk_forward["display"].get("validation_strategy_label")
            or strategy.get("label")
            or "Rolling Window"
        )
        walk_forward["display"]["window_mode"] = "rolling"

    is_wf = strategy["key"] in ("walk_forward", "rolling_window")
    best_params_data = walk_forward.get("best_parameters") or {}
    if isinstance(best_params_data.get("data"), dict):
        best_params_data = best_params_data["data"]
    score_refs = best_params_data.get("reference_stats") if isinstance(best_params_data, dict) else None
    wf_summary_data = walk_forward.get("summary") or {}
    if isinstance(wf_summary_data.get("data"), dict):
        wf_summary_data = wf_summary_data["data"]
    if not metrics.get("production_walk_forward"):
        champ_art = walk_forward.get("champion_aggregate") or {}
        champ_data = champ_art.get("data") if isinstance(champ_art.get("data"), dict) else {}
        if champ_data.get("aggregated"):
            metrics = dict(metrics)
            metrics["production_walk_forward"] = champ_data["aggregated"]
    production_metrics = _resolve_authoritative_metrics(
        strategy=strategy,
        metrics_doc=metrics,
        summary_doc=summary,
        wf_summary_doc=wf_summary_data if isinstance(wf_summary_data, dict) else None,
        prediction_type=prediction_type,
        score_refs=score_refs if isinstance(score_refs, dict) else None,
    )
    if is_wf:
        metrics = _enrich_validation_metrics_for_walk_forward(
            metrics,
            wf_summary=wf_summary_data if isinstance(wf_summary_data, dict) else None,
        )
    _mark("walk_forward_artifacts", t0)

    t0 = time.perf_counter()
    meta_data = metadata_art.get("data") if isinstance(metadata_art.get("data"), dict) else {}
    stored_elim = meta_data.get("feature_elimination") if isinstance(meta_data.get("feature_elimination"), dict) else {}
    summary_elim = summary.get("feature_elimination") if isinstance(summary.get("feature_elimination"), dict) else {}
    try:
        training_cfg = normalize_training_config(config_data) if config_data else None
    except Exception:
        training_cfg = None
    final_feature_names = _selected_feature_names(data_dir, safe, paths)
    wf_sel_rows = walk_forward.get("selected_features") or {}
    feature_elimination = resolve_feature_elimination_doc(
        config=training_cfg,
        config_features=final_feature_names or (list(config_data.get("features") or []) if isinstance(config_data, dict) else None),
        walk_forward_summary=wf_summary_data if isinstance(wf_summary_data, dict) else None,
        stored_elimination=stored_elim or summary_elim,
        csv_selected_count=_count_csv_selected_features(wf_sel_rows),
    )
    _mark("feature_elimination", t0)

    t0 = time.perf_counter()
    package_files: list[str] = []
    artifact_inventory = {
        "metadata.json": metadata_art,
        "metrics.json": metrics_art,
        "training_config.json": training_config_art,
        "config.json": _load_json_artifact(paths["config_json"]),
        "training_summary.json": summary_art,
        "training_metadata.json": training_metadata_art,
        "training_monitor.csv": training_monitor_csv_art,
        "feature_importance.csv": fi_csv,
        "model.ubj": {"available": os.path.isfile(paths["model_ubj"]), "path": os.path.basename(paths["model_ubj"])},
        "baseline_model.ubj": {"available": os.path.isfile(paths["baseline_model_ubj"]), "path": os.path.basename(paths["baseline_model_ubj"])},
        "tuned_model.ubj": {"available": os.path.isfile(paths["tuned_model_ubj"]), "path": os.path.basename(paths["tuned_model_ubj"])},
        "walk_forward/summary.json": walk_forward.get("summary") or {"available": False, "data": None},
        "walk_forward/selected_features.csv": walk_forward.get("selected_features"),
        "walk_forward/feature_selection_history.csv": walk_forward.get("feature_selection_history"),
        "walk_forward/feature_stability.csv": walk_forward.get("feature_stability"),
        "walk_forward/best_parameters.json": walk_forward.get("best_parameters"),
        "walk_forward/champion_aggregate.json": walk_forward.get("champion_aggregate"),
        "walk_forward/top_trials.csv": walk_forward.get("top_trials"),
        "walk_forward/optimization_history.csv": walk_forward.get("optimization_history"),
        "walk_forward/parameter_importance.csv": walk_forward.get("parameter_importance"),
    }
    _mark("artifact_inventory", t0)

    t0 = time.perf_counter()
    table_row = _table_row(data_dir, safe, pkg)
    if strategy.get("key") in ("walk_forward", "rolling_window") and production_metrics.get("n_folds"):
        table_row["validation_strategy"] = f"{strategy.get('label')} ({int(production_metrics['n_folds'])} folds)"
    _mark("table_row", t0)

    strike_selection_label: str | None = None
    sampling_interval_label: str | None = None
    dataset_build_snapshot: dict[str, Any] = {}
    try:
        from chain_replay_ml.replay_config import load_dataset_metadata_json
        from chain_replay_ml.dataset_builder.expected_spec import (
            strike_selection_display_label,
            sampling_interval_display_label,
        )
        from .dataset_build_snapshot import resolve_dataset_build_snapshot

        interim_doc = {
            "config": config_data,
            "metadata": metadata_art,
            "dataset_build_snapshot": snapshot_art.get("data") if isinstance(snapshot_art.get("data"), dict) else {},
        }
        dataset_build_snapshot = resolve_dataset_build_snapshot(interim_doc)
        if dataset_build_snapshot.get("strike_selection_label"):
            strike_selection_label = str(dataset_build_snapshot["strike_selection_label"])
        if dataset_build_snapshot.get("sampling_label"):
            sampling_interval_label = str(dataset_build_snapshot["sampling_label"])

        dataset_name = str(config_data.get("dataset") or meta_data.get("dataset") or "").strip()
        ds_meta = load_dataset_metadata_json(data_dir, dataset_name) if dataset_name else {}
        if not strike_selection_label:
            strike_selection_label = strike_selection_display_label(ds_meta)
        if not sampling_interval_label:
            sampling_interval_label = sampling_interval_display_label(ds_meta)
        if not strike_selection_label:
            fp_data = fingerprint_art.get("data") if isinstance(fingerprint_art.get("data"), dict) else {}
            fp_band = fp_data.get("atm_band")
            if fp_band is not None:
                from chain_replay_ml.dataset_builder.expected_spec import format_strike_selection_label

                strike_selection_label = format_strike_selection_label(
                    {"mode": "atm_band", "band": fp_band},
                )
        if not sampling_interval_label:
            fp_data = fingerprint_art.get("data") if isinstance(fingerprint_art.get("data"), dict) else {}
            if fp_data.get("sampling"):
                sampling_interval_label = str(fp_data["sampling"])
            elif fp_data.get("sampling_interval_sec") is not None:
                from chain_replay_ml.dataset_builder.expected_spec import format_sampling_interval_label

                sampling_interval_label = format_sampling_interval_label(fp_data["sampling_interval_sec"])
    except Exception:
        strike_selection_label = None
        sampling_interval_label = None

    t0 = time.perf_counter()
    # Lifecycle + training log are deferred until those tabs open (can be large / SQLite).
    lifecycle_view: dict[str, Any] = {}
    _mark("lifecycle_view", t0)

    lifecycle_timing = None

    t0 = time.perf_counter()
    training_log = ""
    _mark("training_log", t0)

    return {
        "model_name": safe,
        "prediction_type": prediction_type,
        "validation_strategy": strategy,
        "is_walk_forward": is_wf,
        "strike_selection_label": strike_selection_label,
        "sampling_interval_label": sampling_interval_label,
        "dataset_build_snapshot": dataset_build_snapshot,
        "registry": registry_art.get("data") or {},
        "metadata": metadata_art,
        "training_config": training_config_art,
        "training_summary": summary,
        "training_metadata": training_metadata_art.get("data") or {},
        "training_monitor": training_monitor_csv_art,
        "config": config_data,
        "metrics": metrics,
        "production_metrics": production_metrics,
        "feature_importance": feature_importance,
        "pipeline_fingerprint": fingerprint_art.get("data") or {},
        "training_log": training_log,
        "table_row": table_row,
        "walk_forward": walk_forward,
        "feature_elimination": feature_elimination,
        "artifact_inventory": artifact_inventory,
        "package_files": package_files,
        "artifacts": {
            "training_report": os.path.basename(paths["training_report_html"]),
            "feature_importance_csv": os.path.basename(paths["feature_importance_csv"]),
            "model_json": os.path.basename(paths["model_json"]),
            "model_ubj": os.path.basename(paths["model_ubj"]),
            "baseline_model_ubj": os.path.basename(paths["baseline_model_ubj"]),
            "tuned_model_ubj": os.path.basename(paths["tuned_model_ubj"]),
        },
        "report_url": table_row.get("report_url") or "",
        "model_lifecycle": lifecycle_view,
        "_deferred_heavy": True,
        "_timing": {
            "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
            "stages": stages,
            "lifecycle": lifecycle_timing,
        },
    }


def enrich_model_detail_heavy(data_dir: str, doc: dict[str, Any], *, need: str) -> dict[str, Any]:
    """Load deferred heavy artifacts when a detail tab first needs them."""
    if not isinstance(doc, dict):
        return doc
    safe = str(doc.get("model_name") or "").strip()
    if not safe:
        return doc
    paths = model_artifact_paths(data_dir, safe)
    pkg = paths["package_dir"]
    need_key = str(need or "").strip().lower()

    if need_key in ("walk_forward", "artifacts", "features"):
        wf = doc.get("walk_forward") if isinstance(doc.get("walk_forward"), dict) else {}
        if wf.get("_heavy_deferred"):
            doc["walk_forward"] = load_deferred_walk_forward_csvs(pkg, wf)
            inv = doc.get("artifact_inventory") if isinstance(doc.get("artifact_inventory"), dict) else {}
            wf2 = doc["walk_forward"]
            inv["walk_forward/feature_selection_history.csv"] = wf2.get("feature_selection_history")
            inv["walk_forward/feature_stability.csv"] = wf2.get("feature_stability")
            inv["walk_forward/top_trials.csv"] = wf2.get("top_trials")
            inv["walk_forward/optimization_history.csv"] = wf2.get("optimization_history")
            inv["walk_forward/parameter_importance.csv"] = wf2.get("parameter_importance")
            doc["artifact_inventory"] = inv

    if need_key in ("artifacts",):
        if not doc.get("package_files"):
            doc["package_files"] = _list_package_files(pkg)
        if not doc.get("training_log"):
            doc["training_log"] = _load_text(paths["training_log_txt"])

    if need_key in ("lifecycle",) and not doc.get("model_lifecycle"):
        try:
            from .lifecycle_store import get_model_lifecycle_view

            doc["model_lifecycle"] = get_model_lifecycle_view(data_dir, model_name=safe)
        except Exception:
            doc["model_lifecycle"] = {}

    return doc


def delete_model(data_dir: str, model_name: str) -> dict[str, Any]:
    assert_model_deletable(data_dir, model_name)
    safe = safe_model_name(model_name)
    pkg = model_artifact_paths(data_dir, model_name)["package_dir"]
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"Model package not found: {safe}")
    from chain_replay_ml.replay_session_store import delete_replay_sessions_for_model

    # Clear selection before rmtree so a failed wipe still leaves no stale active pointer.
    clear_active_model_if(data_dir, model_name)
    replay_cleanup = delete_replay_sessions_for_model(data_dir, safe)
    shutil.rmtree(pkg)
    try:
        from .lifecycle_store import delete_history_for_model

        delete_history_for_model(data_dir, safe)
    except Exception:
        pass
    return {
        "deleted": True,
        "model_name": safe,
        "replay_sessions_deleted": int(replay_cleanup.get("deleted_sessions") or 0),
    }


from .default_model import resolve_default_model_name  # re-export for callers
