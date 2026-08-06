"""Build training dashboard summary from model builder config."""

from __future__ import annotations

from typing import Any

_VAL_STRATEGY_LABELS = {
    "time_series_split": "Time Series Split",
    "walk_forward": "Walk Forward",
    "rolling_window": "Rolling Window",
}
_WF_FEATURE_SEL_LABELS = {
    "none": "None",
    "shap": "SHAP Importance",
    "rfe": "Recursive Feature Elimination",
    "permutation": "Permutation Importance",
}
_WF_OPT_METRIC_LABELS = {
    "composite": "Composite",
    "rmse": "Lowest RMSE",
    "mae": "Lowest MAE",
    "directional_accuracy": "Highest Direction %",
    "auto": "Auto",
}
_LIFECYCLE_LABELS = {
    "retrain": "Retrain",
    "complete_optimization": "Complete Optimization",
    "feature_optimization": "Feature Optimization",
}
_ALGO_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def _fmt_rows(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _target_label(name: str) -> str:
    n = str(name or "")
    if n.startswith("future_ltp_"):
        return f"Future LTP {n[len('future_ltp_'):]}"
    return n.replace("_", " ").title()


def _label(mapping: dict[str, str], key: Any, default: str = "—") -> str:
    if key is None or key == "":
        return default
    return mapping.get(str(key), str(key).replace("_", " ").title())


def _premium_filter_label(config: dict[str, Any]) -> str:
    """Label for Create Model Premium Selection (train-time LTP band)."""
    prem = config.get("premium_selection") or config.get("premiumSelection")
    if not isinstance(prem, dict):
        if (
            config.get("premium_selection_enabled")
            and config.get("premium_min") is not None
            and config.get("premium_max") is not None
        ):
            prem = {
                "enabled": True,
                "premium_min": config.get("premium_min"),
                "premium_max": config.get("premium_max"),
            }
        else:
            return "Off"
    if not prem.get("enabled"):
        return "Off"
    try:
        lo = float(prem.get("premium_min"))
        hi = float(prem.get("premium_max"))
    except (TypeError, ValueError):
        return "On"
    if lo > hi:
        lo, hi = hi, lo
    return f"LTP {lo:g}–{hi:g}"


def build_config_summary_rows(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Static configuration summary shown for the full training run."""
    split = config.get("split") or {}
    params = config.get("parameters") or {}
    lc = config.get("lifecycle") or {}
    stats = config.get("dataset_stats") or {}
    features = list(config.get("features") or [])
    strategy_ui = split.get("validation_strategy_ui") or split.get("strategy")
    build_snap = config.get("dataset_build_snapshot") if isinstance(config.get("dataset_build_snapshot"), dict) else {}
    rows: list[tuple[str, str]] = [
        ("Model", str(config.get("model_name") or "—")),
        ("Version", str(config.get("model_version") or "—")),
    ]
    desc = str(config.get("model_description") or "").strip()
    if desc:
        rows.append(("Description", desc))
    rows.extend([
        ("Dataset", str(config.get("dataset") or "—")),
        ("Dataset rows", _fmt_rows(stats.get("row_count") or build_snap.get("row_count")) if (stats.get("row_count") is not None or build_snap.get("row_count") is not None) else "—"),
        ("Trading days", _fmt_rows(build_snap.get("trading_days")) if build_snap.get("trading_days") is not None else "—"),
        ("Registry features", _fmt_rows(stats.get("feature_count") or build_snap.get("feature_count")) if (stats.get("feature_count") is not None or build_snap.get("feature_count") is not None) else "—"),
        ("Sampling", build_snap.get("sampling_label") or (f"{stats.get('sampling_interval_sec')}s" if stats.get("sampling_interval_sec") else "—")),
        ("Market", str(build_snap.get("market") or "—")),
        ("Premium filter", _premium_filter_label(config)),
        ("Target", _target_label(str(config.get("target") or ""))),
        ("Algorithm", _label(_ALGO_LABELS, config.get("algorithm"))),
        ("Prediction type", str(config.get("prediction_type") or "—")),
        ("Selected features", str(len(features))),
    ])
    for item in build_snap.get("filter_summary") or []:
        if isinstance(item, dict) and item.get("label"):
            label = str(item["label"])
            if label == "Trading dates":
                continue
            # Prefer Create Model Premium Selection over dataset-build premium row.
            if label.lower() in ("premium filter", "premium", "ltp"):
                continue
            rows.append((label, str(item.get("value") or "—")))
    rows.extend([
        ("Validation strategy", _label(_VAL_STRATEGY_LABELS, strategy_ui)),
    ])
    mode = lc.get("mode")
    if mode:
        rows.append(("Lifecycle mode", _label(_LIFECYCLE_LABELS, mode)))
        source = lc.get("source_model")
        if source:
            rows.append(("Source model", str(source)))

    if split.get("strategy") == "walk_forward":
        wf = split.get("walk_forward") or {}
        wf_hpo = wf.get("hyperparameter_optimization") or {}
        rows.extend([
            ("WF folds", str(wf.get("n_folds") or "—")),
            ("WF window mode", str(wf.get("window_mode") or "—")),
            ("WF fold placement", str(wf.get("fold_placement") or "—")),
            ("WF train window", _fmt_rows(wf.get("train_window_size"))),
            ("WF val window", _fmt_rows(wf.get("validation_window_size"))),
            ("WF feature selection", _label(_WF_FEATURE_SEL_LABELS, wf.get("feature_selection_method"))),
            ("WF optimization metric", _label(_WF_OPT_METRIC_LABELS, wf.get("optimization_metric"))),
            (
                "WF HPO",
                f"{'On' if wf_hpo.get('enabled') else 'Off'}"
                + (f" · {wf_hpo.get('n_trials')} trials" if wf_hpo.get("enabled") else ""),
            ),
        ])
        preview = config.get("walk_forward_preview") or {}
        summary = preview.get("summary") if isinstance(preview.get("summary"), dict) else {}
        if summary:
            rows.extend([
                ("WF region rows", _fmt_rows(summary.get("walk_forward_region_rows"))),
                (
                    "Test holdout",
                    f"{_fmt_rows(summary.get('test_holdout_rows'))} rows ({summary.get('test_holdout_pct')}%)",
                ),
                (
                    "WF row range",
                    f"{summary.get('walk_forward_region_start')}–{summary.get('walk_forward_region_end')}",
                ),
            ])
    else:
        rows.append(
            (
                "Split",
                f"Train {split.get('train', '—')}% · Val {split.get('validation', '—')}% · "
                f"Test {split.get('test', '—')}%",
            ),
        )

    global_hpo = split.get("hyperparameter_optimization") or {}
    rows.append(
        (
            "Global HPO",
            f"{'On' if global_hpo.get('enabled') else 'Off'}"
            + (f" · {global_hpo.get('n_trials')} trials" if global_hpo.get("enabled") else ""),
        ),
    )
    rows.extend([
        ("Trees (n_estimators)", str(params.get("n_estimators") or "—")),
        ("Learning rate", str(params.get("learning_rate") or "—")),
        ("Max depth", str(params.get("max_depth") or "—")),
        ("Early stopping", str(params.get("early_stopping_rounds") or "—")),
        ("Random seed", str(params.get("random_seed") or "—")),
    ])
    return rows


def format_config_summary(config: dict[str, Any]) -> str:
    rows = build_config_summary_rows(config)
    lines = ["Training configuration", "─" * 28]
    lines.extend(f"{k}: {v}" for k, v in rows)
    return "\n".join(lines)


def format_live_dashboard_section(
    live_rows: list[tuple[str, str]],
    *,
    heading: str = "Live status",
) -> str:
    lines = ["", heading, "─" * 28]
    lines.extend(f"{k}: {v}" for k, v in live_rows)
    return "\n".join(lines)
