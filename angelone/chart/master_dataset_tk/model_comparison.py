"""Model comparison logic for Tk Model Comparison panel."""

from __future__ import annotations

from typing import Any

from .model_registry_detail import (
    _cfg,
    _dataset_build_snapshot,
    _fold_placement_display,
    _meta,
    _metrics,
    _prod,
    _resolve_premium_bands,
    _strat_label,
    _summary,
    _wf,
    _wf_display,
    _wf_summary_rows,
)
from .model_registry_widgets import fmt_num, fmt_pct, fmt_rows

SummaryRow = tuple[str, Any, Any]
SummaryGroup = tuple[str, list[SummaryRow]]
MetricRow = tuple[str, Any, Any, str | None, str | None]
PremiumBandRow = tuple[str, str, str, str, str, str, str]

_LOWER_BETTER_KEYS = frozenset({
    "mae", "rmse", "premium_mae_pct", "premium_rmse_pct", "medae", "median_error",
    "p95_error", "mape", "std_rmse", "std_mae",
})
_HIGHER_BETTER_KEYS = frozenset({
    "directional_accuracy_pct", "composite_score", "r2",
})


def model_display_label(doc: dict[str, Any]) -> str:
    return str(doc.get("model_name") or "—")


def _numeric_delta(
    a: Any,
    b: Any,
    *,
    higher_better: bool = False,
) -> tuple[str | None, str | None]:
    """Return (delta_str, winner) where winner is 'A', 'B', 'Tie', or None."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return None, None
    delta = fb - fa
    if abs(delta) < 1e-12:
        return "0", "Tie"
    sign = "+" if delta > 0 else "−"
    delta_str = f"{sign}{abs(delta):,.4f}"
    if higher_better:
        winner = "B" if fb > fa else "A"
    else:
        winner = "B" if fb < fa else "A"
    return delta_str, winner


def _higher_better_for_key(key: str) -> bool:
    key_l = key.lower()
    if key_l in _HIGHER_BETTER_KEYS:
        return True
    if key_l in _LOWER_BETTER_KEYS:
        return False
    return False


def _metric_row(
    label: str,
    val_a: Any,
    val_b: Any,
    *,
    higher_better: bool | None = None,
    metric_key: str | None = None,
) -> MetricRow:
    hb = higher_better if higher_better is not None else _higher_better_for_key(metric_key or label)
    delta, winner = _numeric_delta(val_a, val_b, higher_better=hb)
    return (label, val_a, val_b, delta, winner)


def _summary_value(doc: dict[str, Any], key: str) -> Any:
    meta = _meta(doc)
    cfg = _cfg(doc)
    summary = _summary(doc)
    row = doc.get("table_row") or {}
    runtime = (
        summary.get("algorithm_runtime")
        or meta.get("algorithm_runtime")
        or cfg.get("algorithm_runtime")
        or {}
    )
    if not isinstance(runtime, dict):
        runtime = {}
    from chain_replay_ml.training.algorithm_runtime import device_display, format_duration

    device_raw = (
        summary.get("device_label")
        or summary.get("device")
        or runtime.get("device_label")
        or runtime.get("device")
        or meta.get("device_label")
        or meta.get("device")
        or row.get("device")
    )
    train_sec = summary.get("training_time_sec")
    pred_ms = (
        summary.get("prediction_time_ms")
        or runtime.get("prediction_time_ms")
        or meta.get("prediction_time_ms")
        or row.get("prediction_time_ms")
    )
    mapping: dict[str, Any] = {
        "validation_strategy": _strat_label(doc),
        "algorithm": meta.get("algorithm") or cfg.get("algorithm_label") or cfg.get("algorithm") or summary.get("algorithm"),
        "dataset": meta.get("dataset") or cfg.get("dataset") or row.get("dataset"),
        "target": meta.get("target") or cfg.get("target") or row.get("target"),
        "rows": meta.get("row_count") or summary.get("rows") or row.get("rows"),
        "selected_features": meta.get("feature_count") or summary.get("features") or row.get("feature_count"),
        "trees_trained": summary.get("trees_trained"),
        "early_stopped": "Yes" if summary.get("early_stopped") else ("No" if summary.get("early_stopped") is False else None),
        "training_time_sec": train_sec,
        "training_time": format_duration(train_sec) if train_sec is not None else None,
        "total_elapsed_sec": summary.get("total_elapsed_sec"),
        "device": device_display(device_raw) if device_raw else None,
        "implementation": summary.get("implementation") or runtime.get("implementation") or meta.get("implementation"),
        "gpu_name": summary.get("gpu_name") or runtime.get("gpu_name") or meta.get("gpu_name"),
        "fallback_reason": summary.get("fallback_reason") or runtime.get("fallback_reason") or meta.get("fallback_reason"),
        "prediction_time_ms": f"{pred_ms} ms" if pred_ms is not None else None,
        "trained_at": meta.get("trained_at") or row.get("trained_at"),
        "status": row.get("status") or "ready",
        "model_version": meta.get("model_version") or cfg.get("model_version"),
        "description": meta.get("model_description") or cfg.get("model_description"),
        "strike_selection": doc.get("strike_selection_label"),
        "sampling": doc.get("sampling_interval_label"),
        "is_walk_forward": "Yes" if doc.get("is_walk_forward") else "No",
    }
    return mapping.get(key)


def _field_label(field: str) -> str:
    if field == "training_time_sec":
        return "Training time (s)"
    if field == "training_time":
        return "Train Time"
    if field == "prediction_time_ms":
        return "Prediction Time"
    if field == "total_elapsed_sec":
        return "Total elapsed (s)"
    if field == "is_walk_forward":
        return "Is walk forward"
    if field == "validation_strategy":
        return "Validation strategy"
    if field == "gpu_name":
        return "GPU"
    if field == "fallback_reason":
        return "Fallback reason"
    return field.replace("_", " ").title()


_DATASET_COMPARISON_FIELDS = (
    "Dataset name",
    "Rows count",
    "Sampling",
    "Strike selection",
    "LTP / Premium",
    "Trading days count",
    "Trading day filter",
    "Excluded expiry dates",
)


def _filter_summary_value(snap: dict[str, Any], label: str) -> Any:
    for item in snap.get("filter_summary") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "").strip().casefold() == label.casefold():
            return item.get("value") or "—"
    return None


def _trading_day_filter_display(snap: dict[str, Any] | None) -> dict[str, Any]:
    """Trading-day filter rows for Dataset comparison (from filter_summary or trading_day_filter)."""
    empty = {
        "Trading day filter": None,
        "Excluded expiry dates": None,
    }
    if not snap:
        return empty
    mode_label = _filter_summary_value(snap, "Trading day filter")
    excluded = _filter_summary_value(snap, "Excluded expiry dates")
    if excluded is None:
        excluded = _filter_summary_value(snap, "Expiry dates included")
    if mode_label and mode_label != "—":
        return {
            "Trading day filter": mode_label,
            "Excluded expiry dates": excluded,
        }

    tdf = snap.get("trading_day_filter")
    if not isinstance(tdf, dict) or not tdf:
        return empty
    from chain_replay_ml.dataset_builder.trading_day_filter import trading_day_filter_summary_rows

    labels = str(snap.get("trading_day_labels") or "").strip()
    exported = [p.strip() for p in labels.split(",") if p.strip()] if labels and labels != "—" else None
    by_label = {
        str(item.get("label") or "").strip(): item.get("value")
        for item in trading_day_filter_summary_rows(tdf, exported_dates=exported)
        if isinstance(item, dict) and item.get("label")
    }
    return {
        "Trading day filter": by_label.get("Trading day filter"),
        "Excluded expiry dates": (
            by_label.get("Excluded expiry dates")
            or by_label.get("Expiry dates included")
        ),
    }


def _dataset_row_map(doc: dict[str, Any]) -> dict[str, Any]:
    from chain_replay_ml.training.dataset_build_snapshot import enrich_snapshot_trading_day_filter

    snap = _dataset_build_snapshot(doc)
    meta = _meta(doc)
    cfg = _cfg(doc)
    summary = _summary(doc)
    row = doc.get("table_row") or {}

    dataset_name = meta.get("dataset") or cfg.get("dataset") or row.get("dataset") or snap.get("dataset_name")
    rows_count = meta.get("row_count") or summary.get("rows") or row.get("rows")
    data_dir = str(doc.get("_data_dir") or doc.get("data_dir") or "").strip() or None
    if isinstance(snap, dict):
        lookup_name = str(
            snap.get("dataset_name")
            or dataset_name
            or "",
        ).strip() or None
        snap = enrich_snapshot_trading_day_filter(
            snap,
            data_dir,
            dataset_name=lookup_name,
        )

    sampling = snap.get("sampling_label") if snap else None
    strike = snap.get("strike_selection_label") if snap else None
    trading_days = snap.get("trading_days") if snap else None
    if snap:
        ltp_premium = _filter_summary_value(snap, "LTP / Premium")
    else:
        ltp_premium = None
    tdf_display = _trading_day_filter_display(snap if isinstance(snap, dict) else None)

    if not sampling or str(sampling) == "—":
        sampling = doc.get("sampling_interval_label")
    if not strike or str(strike) == "—":
        strike = doc.get("strike_selection_label")

    return {
        "Dataset name": dataset_name,
        "Rows count": fmt_rows(rows_count) if rows_count is not None else None,
        "Sampling": sampling,
        "Strike selection": strike,
        "LTP / Premium": ltp_premium,
        "Trading days count": fmt_rows(trading_days) if trading_days is not None else None,
        "Trading day filter": tdf_display.get("Trading day filter"),
        "Excluded expiry dates": tdf_display.get("Excluded expiry dates"),
    }


def _dataset_comparison_rows(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[SummaryRow]:
    map_a = _dataset_row_map(doc_a)
    map_b = _dataset_row_map(doc_b)
    return [
        (label, map_a.get(label) or "—", map_b.get(label) or "—")
        for label in _DATASET_COMPARISON_FIELDS
    ]


def build_summary_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[SummaryGroup]:
    validation_rows: list[SummaryRow] = [
        (_field_label("validation_strategy"), _summary_value(doc_a, "validation_strategy"), _summary_value(doc_b, "validation_strategy")),
        (_field_label("is_walk_forward"), _summary_value(doc_a, "is_walk_forward"), _summary_value(doc_b, "is_walk_forward")),
    ]
    validation_rows.extend(_zip_wf_summary_rows(doc_a, doc_b))

    dataset_rows = _dataset_comparison_rows(doc_a, doc_b)

    general_fields = [
        "algorithm",
        "device",
        "implementation",
        "gpu_name",
        "training_time",
        "prediction_time_ms",
        "target",
        "selected_features",
        "trees_trained",
        "early_stopped",
        "training_time_sec",
        "total_elapsed_sec",
        "fallback_reason",
    ]
    general_rows: list[SummaryRow] = []
    for field in general_fields:
        general_rows.append((_field_label(field), _summary_value(doc_a, field), _summary_value(doc_b, field)))

    # Compact operational performance table (accuracy-adjacent ops metrics)
    ops_fields = ["algorithm", "device", "training_time", "prediction_time_ms"]
    ops_rows: list[SummaryRow] = [
        (_field_label(field), _summary_value(doc_a, field), _summary_value(doc_b, field))
        for field in ops_fields
    ]

    return [
        ("Validation Strategy", validation_rows),
        ("Dataset", dataset_rows),
        ("Operational Performance", ops_rows),
        ("Model Details", general_rows),
    ]


def _zip_wf_summary_rows(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[SummaryRow]:
    map_a = {k: v for k, v in _wf_summary_rows(doc_a)}
    map_b = {k: v for k, v in _wf_summary_rows(doc_b)}
    keys = list(dict.fromkeys([*map_a.keys(), *map_b.keys()]))
    return [(k, map_a.get(k), map_b.get(k)) for k in keys]


def build_metric_comparison(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    section: str,
) -> list[MetricRow]:
    prod_a = _prod(doc_a)
    prod_b = _prod(doc_b)
    key = section.strip().lower()
    if key in ("production", "model", "model_metrics"):
        return [
            _metric_row("MAE (₹)", prod_a.get("mae"), prod_b.get("mae"), metric_key="mae"),
            _metric_row("RMSE (₹)", prod_a.get("rmse"), prod_b.get("rmse"), metric_key="rmse"),
            _metric_row("Premium MAE (%)", prod_a.get("premium_mae_pct"), prod_b.get("premium_mae_pct"), metric_key="premium_mae_pct"),
            _metric_row("Premium RMSE (%)", prod_a.get("premium_rmse_pct"), prod_b.get("premium_rmse_pct"), metric_key="premium_rmse_pct"),
            _metric_row("Median Absolute Error", prod_a.get("medae") or prod_a.get("median_error"), prod_b.get("medae") or prod_b.get("median_error"), metric_key="medae"),
            _metric_row("95th Percentile Error", prod_a.get("p95_error"), prod_b.get("p95_error"), metric_key="p95_error"),
            _metric_row("Prediction Bias", prod_a.get("prediction_bias"), prod_b.get("prediction_bias")),
            _metric_row("Prediction Bias %", prod_a.get("prediction_bias_pct"), prod_b.get("prediction_bias_pct")),
            _metric_row("Direction Accuracy", prod_a.get("directional_accuracy_pct"), prod_b.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
            _metric_row("Composite Score", prod_a.get("composite_score"), prod_b.get("composite_score"), metric_key="composite_score"),
        ]
    return []


def build_core_model_metrics_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[MetricRow]:
    prod_a = _comparison_production_block(doc_a)
    prod_b = _comparison_production_block(doc_b)
    return [
        _metric_row("MAE", prod_a.get("mae"), prod_b.get("mae"), metric_key="mae"),
        _metric_row("RMSE", prod_a.get("rmse"), prod_b.get("rmse"), metric_key="rmse"),
        _metric_row("Direction Accuracy", prod_a.get("directional_accuracy_pct"), prod_b.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
        _metric_row("Composite Score", prod_a.get("composite_score"), prod_b.get("composite_score"), metric_key="composite_score"),
    ]


def _pick_metric_value(sources: list[dict[str, Any]], *keys: str) -> Any:
    for src in sources:
        for key in keys:
            val = src.get(key)
            if val is not None and val != "":
                return val
    return None


def _comparison_metric_sources(doc: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = _metrics(doc)
    wf_art = _wf(doc)
    champ_data = (wf_art.get("champion_aggregate") or {}).get("data")
    champ_data = champ_data if isinstance(champ_data, dict) else {}
    agg = champ_data.get("aggregated") if isinstance(champ_data.get("aggregated"), dict) else {}
    champ_val = champ_data.get("validation_metrics") if isinstance(champ_data.get("validation_metrics"), dict) else {}
    return [
        dict(_prod(doc) or {}),
        metrics.get("production_walk_forward") if isinstance(metrics.get("production_walk_forward"), dict) else {},
        metrics.get("walk_forward") if isinstance(metrics.get("walk_forward"), dict) else {},
        agg,
        metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {},
        metrics.get("test") if isinstance(metrics.get("test"), dict) else {},
        champ_val,
    ]


def _comparison_production_block(doc: dict[str, Any]) -> dict[str, Any]:
    sources = _comparison_metric_sources(doc)
    block = dict(sources[0]) if sources else {}
    for field, keys in (
        ("mae", ("mae", "mean_mae")),
        ("rmse", ("rmse", "mean_rmse")),
        ("directional_accuracy_pct", ("directional_accuracy_pct", "mean_directional_accuracy_pct", "directional_accuracy")),
        ("composite_score", ("composite_score", "mean_composite_score")),
        ("premium_mae_pct", ("premium_mae_pct", "mean_premium_mae_pct")),
        ("premium_rmse_pct", ("premium_rmse_pct", "mean_premium_rmse_pct")),
        ("medae", ("medae", "median_error", "mean_medae", "mean_median_error")),
        ("p95_error", ("p95_error", "mean_p95_error")),
        ("prediction_bias", ("prediction_bias", "mean_prediction_bias")),
        ("prediction_bias_pct", ("prediction_bias_pct", "mean_prediction_bias_pct")),
    ):
        if block.get(field) is None:
            block[field] = _pick_metric_value(sources, *keys)
    if not block.get("premium_band_performance"):
        bands = _resolve_premium_bands(doc)
        if bands:
            block["premium_band_performance"] = bands
    return block


def premium_metrics_available(doc: dict[str, Any]) -> bool:
    block = _comparison_production_block(doc)
    if block.get("premium_band_performance"):
        return True
    return any(
        block.get(key) is not None
        for key in (
            "premium_mae_pct",
            "premium_rmse_pct",
            "medae",
            "p95_error",
            "prediction_bias",
            "prediction_bias_pct",
        )
    )


def build_premium_model_metrics_rows(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[SummaryRow]:
    prod_a = _comparison_production_block(doc_a)
    prod_b = _comparison_production_block(doc_b)
    return [
        ("Premium MAE", prod_a.get("premium_mae_pct"), prod_b.get("premium_mae_pct")),
        ("Premium RMSE", prod_a.get("premium_rmse_pct"), prod_b.get("premium_rmse_pct")),
        ("Median Absolute Error", prod_a.get("medae") or prod_a.get("median_error"), prod_b.get("medae") or prod_b.get("median_error")),
        ("95th Percentile Error", prod_a.get("p95_error"), prod_b.get("p95_error")),
        ("Prediction Bias", prod_a.get("prediction_bias"), prod_b.get("prediction_bias")),
        ("Prediction Bias %", prod_a.get("prediction_bias_pct"), prod_b.get("prediction_bias_pct")),
    ]


def _band_label(row: dict[str, Any]) -> str:
    label = row.get("band_label")
    if label:
        return str(label)
    band = row.get("band")
    if band is not None and str(band).strip():
        return f"₹{band}"
    return "—"


def _pair_display(val_a: Any, val_b: Any, *, digits: int = 2, as_pct: bool = False) -> str:
    def _one(v: Any) -> str:
        if v is None or v == "":
            return "—"
        if as_pct:
            return fmt_pct(v, digits)
        return fmt_num(v, digits)

    return f"{_one(val_a)} / {_one(val_b)}"


def _band_overall_winner(ra: dict[str, Any], rb: dict[str, Any]) -> str:
    wins_a = 0
    wins_b = 0
    for key, higher_better in (
        ("mae", False),
        ("rmse", False),
        ("premium_mae_pct", False),
        ("premium_rmse_pct", False),
        ("directional_accuracy_pct", True),
    ):
        _, winner = _numeric_delta(ra.get(key), rb.get(key), higher_better=higher_better)
        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
    if wins_a > wins_b:
        return "A"
    if wins_b > wins_a:
        return "B"
    return "Tie"


def build_premium_band_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[PremiumBandRow]:
    bands_a = {_band_label(r): r for r in _resolve_premium_bands(doc_a) if isinstance(r, dict)}
    bands_b = {_band_label(r): r for r in _resolve_premium_bands(doc_b) if isinstance(r, dict)}
    band_ids = [k for k in dict.fromkeys([*bands_a.keys(), *bands_b.keys()]) if k and k != "—"]
    rows: list[PremiumBandRow] = []
    for band_id in band_ids:
        ra = bands_a.get(band_id, {})
        rb = bands_b.get(band_id, {})
        rows.append((
            band_id,
            _pair_display(ra.get("mae"), rb.get("mae"), digits=2),
            _pair_display(ra.get("rmse"), rb.get("rmse"), digits=2),
            _pair_display(ra.get("premium_mae_pct"), rb.get("premium_mae_pct"), digits=1, as_pct=True),
            _pair_display(ra.get("premium_rmse_pct"), rb.get("premium_rmse_pct"), digits=1, as_pct=True),
            _pair_display(ra.get("directional_accuracy_pct"), rb.get("directional_accuracy_pct"), digits=1, as_pct=True),
            _band_overall_winner(ra, rb),
        ))
    return rows


def build_validation_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> dict[str, list[MetricRow]]:
    metrics_a = _metrics(doc_a)
    metrics_b = _metrics(doc_b)
    prod_a = _prod(doc_a)
    prod_b = _prod(doc_b)
    val_a = metrics_a.get("validation") if isinstance(metrics_a.get("validation"), dict) else {}
    val_b = metrics_b.get("validation") if isinstance(metrics_b.get("validation"), dict) else {}
    test_a = metrics_a.get("test") if isinstance(metrics_a.get("test"), dict) else {}
    test_b = metrics_b.get("test") if isinstance(metrics_b.get("test"), dict) else {}
    train_a = doc_a.get("training_metadata") if isinstance(doc_a.get("training_metadata"), dict) else {}
    train_b = doc_b.get("training_metadata") if isinstance(doc_b.get("training_metadata"), dict) else {}
    summary_a = _summary(doc_a)
    summary_b = _summary(doc_b)

    training_validation = [
        _metric_row("MAE (₹)", val_a.get("mae"), val_b.get("mae"), metric_key="mae"),
        _metric_row("RMSE (₹)", val_a.get("rmse"), val_b.get("rmse"), metric_key="rmse"),
        _metric_row("Premium MAE (%)", val_a.get("premium_mae_pct"), val_b.get("premium_mae_pct"), metric_key="premium_mae_pct"),
        _metric_row("Premium RMSE (%)", val_a.get("premium_rmse_pct"), val_b.get("premium_rmse_pct"), metric_key="premium_rmse_pct"),
        _metric_row("Validation R²", val_a.get("r2"), val_b.get("r2"), metric_key="r2"),
        _metric_row("Validation MAPE", val_a.get("mape"), val_b.get("mape"), metric_key="mape"),
        _metric_row("Val Directional Acc.", val_a.get("directional_accuracy_pct"), val_b.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
        _metric_row("Best Iteration", train_a.get("best_iteration") or summary_a.get("best_iteration"), train_b.get("best_iteration") or summary_b.get("best_iteration")),
        _metric_row("Early Stopping Round", train_a.get("early_stopping_rounds") or summary_a.get("early_stopping_rounds"), train_b.get("early_stopping_rounds") or summary_b.get("early_stopping_rounds")),
        _metric_row("Training RMSE", train_a.get("train_rmse") or summary_a.get("train_rmse"), train_b.get("train_rmse") or summary_b.get("train_rmse"), metric_key="rmse"),
    ]

    production = [
        _metric_row("MAE (₹)", prod_a.get("mae"), prod_b.get("mae"), metric_key="mae"),
        _metric_row("RMSE (₹)", prod_a.get("rmse"), prod_b.get("rmse"), metric_key="rmse"),
        _metric_row("Premium MAE (%)", prod_a.get("premium_mae_pct"), prod_b.get("premium_mae_pct"), metric_key="premium_mae_pct"),
        _metric_row("Premium RMSE (%)", prod_a.get("premium_rmse_pct"), prod_b.get("premium_rmse_pct"), metric_key="premium_rmse_pct"),
        _metric_row("Direction Accuracy", prod_a.get("directional_accuracy_pct"), prod_b.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
        _metric_row("Composite Score", prod_a.get("composite_score"), prod_b.get("composite_score"), metric_key="composite_score"),
    ]

    holdout_test = [
        _metric_row("MAE (₹)", test_a.get("mae"), test_b.get("mae"), metric_key="mae"),
        _metric_row("RMSE (₹)", test_a.get("rmse"), test_b.get("rmse"), metric_key="rmse"),
        _metric_row("Premium MAE (%)", test_a.get("premium_mae_pct"), test_b.get("premium_mae_pct"), metric_key="premium_mae_pct"),
        _metric_row("Premium RMSE (%)", test_a.get("premium_rmse_pct"), test_b.get("premium_rmse_pct"), metric_key="premium_rmse_pct"),
        _metric_row("Test R²", test_a.get("r2"), test_b.get("r2"), metric_key="r2"),
        _metric_row("Test MAPE", test_a.get("mape"), test_b.get("mape"), metric_key="mape"),
        _metric_row("Test Directional Acc.", test_a.get("directional_accuracy_pct"), test_b.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
    ]

    return {
        "training_validation": training_validation,
        "production": production,
        "holdout_test": holdout_test,
    }


def _fold_results(doc: dict[str, Any]) -> list[dict[str, Any]]:
    wf = _wf(doc)
    wf_summary_art = wf.get("summary") or {}
    wf_summary = wf_summary_art.get("data") if isinstance(wf_summary_art.get("data"), dict) else {}
    champ = wf.get("champion_aggregate") or {}
    champ_data = champ.get("data") if isinstance(champ.get("data"), dict) else {}
    folds = champ_data.get("fold_results") if isinstance(champ_data.get("fold_results"), list) else []
    if not folds and isinstance(wf_summary.get("fold_results"), list):
        folds = wf_summary["fold_results"]
    return [fr for fr in folds if isinstance(fr, dict)]


def _fold_metrics(fr: dict[str, Any]) -> dict[str, Any]:
    m = fr.get("metrics") if isinstance(fr.get("metrics"), dict) else {}
    return m


def build_walk_forward_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> dict[str, Any]:
    disp_a = _wf_display(doc_a)
    disp_b = _wf_display(doc_b)
    metrics_a = _metrics(doc_a)
    metrics_b = _metrics(doc_b)
    val_a = metrics_a.get("validation") if isinstance(metrics_a.get("validation"), dict) else {}
    val_b = metrics_b.get("validation") if isinstance(metrics_b.get("validation"), dict) else {}
    elim_a = doc_a.get("feature_elimination") if isinstance(doc_a.get("feature_elimination"), dict) else {}
    elim_b = doc_b.get("feature_elimination") if isinstance(doc_b.get("feature_elimination"), dict) else {}
    wf_a = _wf(doc_a)
    wf_b = _wf(doc_b)
    wf_summary_a = (wf_a.get("summary") or {}).get("data") if isinstance((wf_a.get("summary") or {}).get("data"), dict) else {}
    wf_summary_b = (wf_b.get("summary") or {}).get("data") if isinstance((wf_b.get("summary") or {}).get("data"), dict) else {}
    fs_a = wf_summary_a.get("feature_selection") if isinstance(wf_summary_a.get("feature_selection"), dict) else {}
    fs_b = wf_summary_b.get("feature_selection") if isinstance(wf_summary_b.get("feature_selection"), dict) else {}

    config: list[SummaryRow] = [
        ("Validation strategy", _strat_label(doc_a), _strat_label(doc_b)),
        ("Number of folds", disp_a.get("n_folds"), disp_b.get("n_folds")),
        ("Window mode", disp_a.get("window_mode"), disp_b.get("window_mode")),
        ("Fold placement", _fold_placement_display(doc_a) or disp_a.get("fold_placement"), _fold_placement_display(doc_b) or disp_b.get("fold_placement")),
        ("Train window", disp_a.get("train_window_size"), disp_b.get("train_window_size")),
        ("Validation window", disp_a.get("validation_window_size"), disp_b.get("validation_window_size")),
        ("Mean Validation RMSE", val_a.get("rmse") or disp_a.get("mean_validation_rmse"), val_b.get("rmse") or disp_b.get("mean_validation_rmse")),
        ("Std Validation RMSE", val_a.get("std_rmse") or disp_a.get("std_validation_rmse"), val_b.get("std_rmse") or disp_b.get("std_validation_rmse")),
        ("Mean Validation MAE", val_a.get("mae") or disp_a.get("mean_validation_mae"), val_b.get("mae") or disp_b.get("mean_validation_mae")),
        ("Std Validation MAE", val_a.get("std_mae") or disp_a.get("std_validation_mae"), val_b.get("std_mae") or disp_b.get("std_validation_mae")),
        ("Mean Directional Accuracy", val_a.get("directional_accuracy_pct") or disp_a.get("mean_directional_accuracy_pct"), val_b.get("directional_accuracy_pct") or disp_b.get("mean_directional_accuracy_pct")),
        ("Optimization Metric", disp_a.get("optimization_metric"), disp_b.get("optimization_metric")),
        ("Feature Elimination Method", elim_a.get("method_label") or elim_a.get("method") or fs_a.get("method"), elim_b.get("method_label") or elim_b.get("method") or fs_b.get("method")),
        ("Features Before Elimination", elim_a.get("started_features") or fs_a.get("started_features"), elim_b.get("started_features") or fs_b.get("started_features")),
        ("Features After Elimination", elim_a.get("finished_features") or fs_a.get("finished_features"), elim_b.get("finished_features") or fs_b.get("finished_features")),
        ("HPO enabled", "Yes" if disp_a.get("hyperparameter_optimization_enabled") else ("No" if disp_a.get("hyperparameter_optimization_enabled") is False else None),
         "Yes" if disp_b.get("hyperparameter_optimization_enabled") else ("No" if disp_b.get("hyperparameter_optimization_enabled") is False else None)),
        ("Optuna trials", disp_a.get("hpo_n_trials"), disp_b.get("hpo_n_trials")),
        ("Best composite score", disp_a.get("best_composite_score"), disp_b.get("best_composite_score")),
    ]

    folds_a = {str(fr.get("fold", i + 1)): _fold_metrics(fr) for i, fr in enumerate(_fold_results(doc_a))}
    folds_b = {str(fr.get("fold", i + 1)): _fold_metrics(fr) for i, fr in enumerate(_fold_results(doc_b))}
    fold_ids = list(dict.fromkeys([*folds_a.keys(), *folds_b.keys()]))

    fold_comparisons: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        ma = folds_a.get(fold_id, {})
        mb = folds_b.get(fold_id, {})
        fold_comparisons.append({
            "fold": fold_id,
            "metrics": [
                _metric_row("RMSE", ma.get("rmse"), mb.get("rmse"), metric_key="rmse"),
                _metric_row("MAE", ma.get("mae"), mb.get("mae"), metric_key="mae"),
                _metric_row("R²", ma.get("r2"), mb.get("r2"), metric_key="r2"),
                _metric_row("MAPE", ma.get("mape"), mb.get("mape"), metric_key="mape"),
                _metric_row("Direction Accuracy", ma.get("directional_accuracy_pct"), mb.get("directional_accuracy_pct"), metric_key="directional_accuracy_pct"),
                _metric_row("Composite Score", ma.get("composite_score"), mb.get("composite_score"), metric_key="composite_score"),
            ],
        })

    return {"config": config, "folds": fold_comparisons}


def _feature_names_from_doc(doc: dict[str, Any]) -> list[str]:
    """Resolved feature list for a model package (selected / config / WF CSV)."""
    from .model_registry_detail import model_feature_preset

    names, _ds = model_feature_preset(doc)
    if names:
        return list(dict.fromkeys(names))
    cfg = _cfg(doc)
    for key in ("selected_features", "features"):
        raw = cfg.get(key)
        if isinstance(raw, list) and raw:
            return list(dict.fromkeys(str(f).strip() for f in raw if str(f).strip()))
    return []


def build_feature_set_comparison(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare feature sets: common, A-only, B-only."""
    feats_a = _feature_names_from_doc(doc_a)
    feats_b = _feature_names_from_doc(doc_b)
    set_a = set(feats_a)
    set_b = set(feats_b)
    common = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    return {
        "features_a": feats_a,
        "features_b": feats_b,
        "count_a": len(feats_a),
        "count_b": len(feats_b),
        "common": common,
        "only_a": only_a,
        "only_b": only_b,
        "common_count": len(common),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "overlap_pct": (
            round(100.0 * len(common) / max(len(set_a | set_b), 1), 1)
            if (set_a or set_b)
            else 0.0
        ),
    }
