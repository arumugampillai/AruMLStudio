"""Compare two walk-forward folds of the same trained model."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from .dataset_loader import DatasetLoaderError, load_dataset_frame
from .paths import model_package_dir

# Absolute prediction error buckets (rupees): |pred - actual|
ERROR_HISTOGRAM_BUCKETS: list[tuple[str, float, float | None]] = [
    ("0-1", 0.0, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-5", 3.0, 5.0),
    (">5", 5.0, None),
]


_LOWER_BETTER = frozenset({
    "mae", "rmse", "premium_mae_pct", "premium_rmse_pct",
    "medae", "median_error", "p95_error", "mape", "prediction_bias",
})
_HIGHER_BETTER = frozenset({
    "directional_accuracy_pct", "composite_score", "r2", "hit_rate",
    "hit_rate_pct", "endpoint_hit_pct", "target_hit_pct",
})


class FoldComparisonError(Exception):
    pass


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _higher_better(metric_key: str) -> bool:
    key = metric_key.lower()
    if key in _HIGHER_BETTER:
        return True
    if key in _LOWER_BETTER:
        return False
    return False


def _winner_label(
    val_a: Any,
    val_b: Any,
    *,
    higher_better: bool,
    label_a: str,
    label_b: str,
) -> str | None:
    fa, fb = _num(val_a), _num(val_b)
    if fa is None or fb is None:
        return None
    if abs(fa - fb) < 1e-12:
        return "Tie"
    if higher_better:
        return label_b if fb > fa else label_a
    return label_b if fb < fa else label_a


def _delta_str(val_a: Any, val_b: Any, *, as_pct_pts: bool = False, rupee: bool = False) -> str | None:
    fa, fb = _num(val_a), _num(val_b)
    if fa is None or fb is None:
        return None
    delta = fb - fa
    if abs(delta) < 1e-12:
        return "0"
    sign = "+" if delta > 0 else "−"
    if as_pct_pts:
        return f"{sign}{abs(delta):.2f} pts"
    if rupee:
        return f"{sign}₹{abs(delta):,.2f}"
    return f"{sign}{abs(delta):,.4f}"


def _metric_row(
    label: str,
    val_a: Any,
    val_b: Any,
    *,
    metric_key: str,
    label_a: str,
    label_b: str,
) -> tuple[str, Any, Any, str | None, str | None]:
    hb = _higher_better(metric_key)
    return (
        label,
        val_a,
        val_b,
        _delta_str(
            val_a,
            val_b,
            as_pct_pts=metric_key in ("directional_accuracy_pct", "hit_rate_pct"),
            rupee=metric_key in ("mae", "rmse", "medae", "p95_error", "prediction_bias"),
        ),
        _winner_label(val_a, val_b, higher_better=hb, label_a=label_a, label_b=label_b),
    )


def _meta_data(doc: dict[str, Any]) -> dict[str, Any]:
    art = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    data = art.get("data") if isinstance(art.get("data"), dict) else {}
    return data if data else art


def _dataset_name(doc: dict[str, Any]) -> str:
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    meta = _meta_data(doc)
    row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
    return str(
        cfg.get("dataset")
        or meta.get("dataset")
        or row.get("dataset")
        or ""
    ).strip()


def list_fold_ids(doc: dict[str, Any]) -> list[int]:
    """Return sorted fold numbers available for comparison."""
    folds = _fold_result_list(doc)
    ids: list[int] = []
    for fr in folds:
        try:
            ids.append(int(fr.get("fold")))
        except (TypeError, ValueError):
            continue
    if ids:
        return sorted(set(ids))
    # Fall back to on-disk fold_* dirs when registry summary is empty
    return []


def _pick_stored_endpoint_hit_pct(metrics: dict[str, Any]) -> float | None:
    """Model Quality Endpoint Hit % from the same fold metrics blob (fold model).

    Prefer canonical ``endpoint_hit_pct``; accept legacy ``hit_rate_pct`` /
    ``target_hit_pct``. Classification models may store accuracy as a hit proxy.
    Never re-score with the production champion model.
    """
    for key in ("endpoint_hit_pct", "hit_rate_pct", "target_hit_pct", "accuracy_pct"):
        val = _num(metrics.get(key))
        if val is not None:
            return round(val, 2)
    return None


def _get_fold_scoring_bundle(
    data_dir: str,
    doc: dict[str, Any],
) -> dict[str, Any] | None:
    """Load dataset, production model, and feature list for fold validation scoring."""
    from .config import normalize_training_config
    from .model_runtime import load_prediction_model, resolve_production_model_path
    from .registry import _selected_feature_names

    model_name = str(doc.get("model_name") or "").strip()
    dataset = _dataset_name(doc)
    if not model_name or not dataset:
        return None

    cfg_raw = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    try:
        training_cfg = normalize_training_config(cfg_raw)
    except Exception:
        return None

    target = training_cfg.target
    paths = None
    try:
        from .paths import model_artifact_paths
        paths = model_artifact_paths(data_dir, model_name)
    except Exception:
        paths = None
    features = _selected_feature_names(data_dir, model_name, paths) if paths else []
    if not features:
        features = [str(f) for f in (cfg_raw.get("selected_features") or cfg_raw.get("features") or []) if f]
    if not features:
        return None

    wanted = list(dict.fromkeys([target, *features]))
    try:
        df, _, _ = load_dataset_frame(data_dir, dataset, columns=wanted)
    except DatasetLoaderError:
        return None

    use_features = [f for f in features if f in df.columns]
    if not use_features or target not in df.columns:
        return None

    pkg = model_package_dir(data_dir, model_name)
    model_path = resolve_production_model_path(pkg, algorithm=training_cfg.algorithm)
    if not model_path:
        return None
    try:
        model = load_prediction_model(model_path, training_cfg.algorithm)
    except Exception:
        return None

    pred_type = str(
        (doc.get("production_metrics") or {}).get("prediction_type")
        or cfg_raw.get("prediction_type")
        or training_cfg.prediction_type
        or "regression"
    ).strip().lower()

    return {
        "df": df,
        "model": model,
        "target": target,
        "features": use_features,
        "prediction_type": pred_type,
    }


def _fold_validation_abs_errors(
    bundle: dict[str, Any],
    fold_def: dict[str, Any],
) -> list[float] | None:
    """Absolute errors on a fold validation window."""
    if bundle.get("prediction_type") in ("binary", "classification", "multiclass"):
        return None
    df = bundle["df"]
    target = bundle["target"]
    start, stop, _ = _slice_bounds(fold_def, "validation")
    if start is None or stop is None or not (0 <= start < stop <= len(df)):
        return None
    y = np.asarray(df.iloc[start:stop][target], dtype=float)
    X = df.iloc[start:stop][bundle["features"]]
    try:
        pred = np.asarray(bundle["model"].predict(X), dtype=float)
    except Exception:
        return None
    mask = np.isfinite(y) & np.isfinite(pred)
    if not mask.any():
        return None
    return np.abs(pred[mask] - y[mask]).tolist()


def build_absolute_error_histogram(abs_errors: list[float]) -> dict[str, Any]:
    """Bucket absolute errors into fixed rupee ranges."""
    n = len(abs_errors)
    if n <= 0:
        return {"rows": [], "total": 0, "tail_gt5_pct": None}

    rows: list[dict[str, Any]] = []
    for label, lo, hi in ERROR_HISTOGRAM_BUCKETS:
        if hi is None:
            count = sum(1 for e in abs_errors if float(e) >= lo)
        else:
            count = sum(1 for e in abs_errors if lo <= float(e) < hi)
        rows.append({
            "bucket": label,
            "count": count,
            "pct": round(count / n * 100.0, 1),
        })
    tail = next((r["pct"] for r in rows if r["bucket"] == ">5"), None)
    return {
        "rows": rows,
        "total": n,
        "tail_gt5_pct": tail,
        "mean_abs_error": round(float(np.mean(abs_errors)), 4),
        "median_abs_error": round(float(np.median(abs_errors)), 4),
        "max_abs_error": round(float(np.max(abs_errors)), 4),
    }


def build_fold_error_histogram_comparison(
    data_dir: str,
    doc: dict[str, Any],
    *,
    fold_a: int,
    fold_b: int,
    fold_def_a: dict[str, Any],
    fold_def_b: dict[str, Any],
    label_a: str,
    label_b: str,
    days_a: list[str],
    days_b: list[str],
) -> dict[str, Any]:
    """Absolute-error histograms for fold comparison.

    Fold boosters are not persisted on disk, so we refuse to re-score with the
    production champion (that mixed models). Histograms require saved fold
    prediction rows — unavailable in the default WF artifact layout.
    """
    _ = (data_dir, doc, fold_a, fold_b, fold_def_a, fold_def_b, label_a, label_b, days_a, days_b)
    return {
        "available": False,
        "message": (
            "Error histogram requires per-fold predictions from the fold model. "
            "Fold boosters are not saved; production-champion re-scoring is disabled "
            "to keep Fold Comparison metrics consistent."
        ),
    }


def _compute_fold_target_hit_map(
    data_dir: str,
    doc: dict[str, Any],
    fold_defs: list[tuple[int, dict[str, Any]]],
) -> dict[int, float | None]:
    """Deprecated: do not re-score folds with the production model.

    Kept as a no-op stub so older imports do not break. Endpoint Hit % must come
    from fold metrics written by the fold model during walk-forward.
    """
    _ = (data_dir, doc)
    return {fid: None for fid, _ in fold_defs}

def model_fold_metrics_table(
    doc: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Fold Metrics rows (Registry Walk Forward style) + validation rows/days + Target Hit %."""
    folds = _fold_result_list(doc)
    wf = _wf_data(doc)
    champ = wf.get("champion_aggregate") if isinstance(wf.get("champion_aggregate"), dict) else {}
    champ_data = champ.get("data") if isinstance(champ.get("data"), dict) else {}
    if champ_data.get("fold_results"):
        source_label = "Production champion re-evaluation folds"
    elif folds:
        source_label = "Initial walk-forward folds (summary.json)"
    else:
        source_label = "Not available"

    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    meta = _meta_data(doc)
    pred_type = str(
        cfg.get("prediction_type")
        or meta.get("prediction_type")
        or doc.get("prediction_type")
        or "regression"
    ).strip().lower()
    is_classification = pred_type in ("binary", "classification", "multiclass")

    model_name = str(doc.get("model_name") or "").strip()
    dataset = _dataset_name(doc)
    resolve_days = bool(data_dir and dataset)

    day_series = None
    if resolve_days and data_dir:
        try:
            df, _, _ = load_dataset_frame(data_dir, dataset, columns=["trading_day"])
            day_series = df["trading_day"]
        except DatasetLoaderError:
            day_series = None

    rows: list[dict[str, Any]] = []

    for fr in folds:
        if not isinstance(fr, dict):
            continue
        m = fr.get("metrics") if isinstance(fr.get("metrics"), dict) else {}
        try:
            fold_id = int(fr.get("fold"))
        except (TypeError, ValueError):
            fold_id = fr.get("fold")

        fold_def = fr.get("fold_def") if isinstance(fr.get("fold_def"), dict) else {}
        if not fold_def and data_dir and model_name and isinstance(fold_id, int):
            fold_def = _resolve_fold_def(data_dir, model_name, fold_id, fr)

        # Prefer on-disk fold metrics (written by that fold's model during WF)
        if data_dir and model_name and isinstance(fold_id, int):
            disk_m = _resolve_fold_metrics(data_dir, model_name, fold_id, fr)
            if disk_m:
                m = {**m, **disk_m}

        _, _, val_rows = _slice_bounds(fold_def, "validation")
        if val_rows is None:
            val_rows = m.get("rows") or m.get("n") or m.get("samples")

        trading_days: list[str] = []
        if day_series is not None and fold_def:
            start, stop, _ = _slice_bounds(fold_def, "validation")
            if start is not None and stop is not None and 0 <= start < stop <= len(day_series):
                trading_days = sorted(
                    day_series.iloc[start:stop].dropna().astype(str).unique().tolist()
                )
        elif data_dir and dataset and fold_def:
            trading_days = resolve_fold_trading_days(data_dir, dataset, fold_def)

        endpoint_hit = _pick_stored_endpoint_hit_pct(m)

        rows.append({
            "fold": fold_id,
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "r2": m.get("r2"),
            "mape": m.get("mape"),
            "directional_accuracy_pct": m.get("directional_accuracy_pct"),
            "accuracy_pct": m.get("accuracy_pct"),
            "precision_pct": m.get("precision_pct"),
            "recall_pct": m.get("recall_pct"),
            "f1_pct": m.get("f1_pct"),
            "roc_auc": m.get("roc_auc"),
            "composite_score": m.get("composite_score"),
            "trees_trained": fr.get("trees_trained"),
            "feature_count": fr.get("feature_count") or m.get("feature_count"),
            "validation_rows": val_rows,
            "validation_days": trading_days,
            "validation_days_count": len(trading_days) if trading_days else None,
            "validation_days_label": _format_validation_days(trading_days),
            "endpoint_hit_pct": endpoint_hit,
            "target_hit_pct": endpoint_hit,  # legacy key for older UI bindings
        })

    hit_note = None
    if is_classification:
        hit_note = (
            "Classification folds — Accuracy / Precision / Recall / F1 / AUC from the fold model."
        )
    elif any(r.get("endpoint_hit_pct") is not None for r in rows):
        hit_note = (
            "Endpoint Hit % = share of validation rows with |pred−actual|/|actual| ≤ 5% "
            "(same fold model as Direction / RMSE / MAE)."
        )
    elif rows:
        hit_note = (
            "Endpoint Hit % not stored on these folds. Retrain to persist it from the "
            "fold model (champion re-scoring is disabled)."
        )

    rows.sort(key=lambda r: int(r["fold"]) if isinstance(r.get("fold"), int) else 9999)
    return {
        "source_label": source_label,
        "hit_note": hit_note,
        "prediction_type": pred_type,
        "is_classification": is_classification,
        "rows": rows,
        "fold_ids": [
            int(r["fold"]) for r in rows if isinstance(r.get("fold"), int)
        ],
    }


def _format_validation_days(days: list[str]) -> str:
    if not days:
        return "—"
    if len(days) <= 3:
        return ", ".join(days)
    return f"{len(days)} days ({days[0]}…{days[-1]})"


def list_fold_ids_on_disk(data_dir: str, model_name: str) -> list[int]:
    pkg = model_package_dir(data_dir, model_name)
    wf = os.path.join(pkg, "walk_forward")
    if not os.path.isdir(wf):
        return []
    ids: list[int] = []
    for name in os.listdir(wf):
        if not name.startswith("fold_"):
            continue
        try:
            ids.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(ids))


def _wf_data(doc: dict[str, Any]) -> dict[str, Any]:
    wf = doc.get("walk_forward") if isinstance(doc.get("walk_forward"), dict) else {}
    return wf


def _fold_result_list(doc: dict[str, Any]) -> list[dict[str, Any]]:
    wf = _wf_data(doc)
    champ = wf.get("champion_aggregate") if isinstance(wf.get("champion_aggregate"), dict) else {}
    champ_data = champ.get("data") if isinstance(champ.get("data"), dict) else {}
    folds = champ_data.get("fold_results") if isinstance(champ_data.get("fold_results"), list) else []
    if not folds:
        summary_art = wf.get("summary") if isinstance(wf.get("summary"), dict) else {}
        summary = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
        folds = summary.get("fold_results") if isinstance(summary.get("fold_results"), list) else []
    return [fr for fr in folds if isinstance(fr, dict)]


def _find_fold_result(doc: dict[str, Any], fold: int) -> dict[str, Any] | None:
    for fr in _fold_result_list(doc):
        try:
            if int(fr.get("fold")) == int(fold):
                return fr
        except (TypeError, ValueError):
            continue
    return None


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _fold_dir(data_dir: str, model_name: str, fold: int) -> str:
    return os.path.join(model_package_dir(data_dir, model_name), "walk_forward", f"fold_{int(fold):02d}")


def _resolve_fold_def(
    data_dir: str,
    model_name: str,
    fold: int,
    fr: dict[str, Any] | None,
) -> dict[str, Any]:
    if fr and isinstance(fr.get("fold_def"), dict):
        return dict(fr["fold_def"])
    disk = _load_json(os.path.join(_fold_dir(data_dir, model_name, fold), "fold.json"))
    return disk or {}


def _resolve_fold_metrics(
    data_dir: str,
    model_name: str,
    fold: int,
    fr: dict[str, Any] | None,
) -> dict[str, Any]:
    if fr and isinstance(fr.get("metrics"), dict) and fr["metrics"]:
        return dict(fr["metrics"])
    disk = _load_json(os.path.join(_fold_dir(data_dir, model_name, fold), "metrics.json"))
    return disk or {}


def _slice_bounds(fold_def: dict[str, Any], key: str) -> tuple[int | None, int | None, int | None]:
    block = fold_def.get(key) if isinstance(fold_def.get(key), dict) else {}
    start = block.get("start")
    stop = block.get("stop")
    rows = block.get("rows")
    try:
        s = int(start) if start is not None else None
        e = int(stop) if stop is not None else None
        n = int(rows) if rows is not None else ((e - s) if s is not None and e is not None else None)
    except (TypeError, ValueError):
        return None, None, None
    return s, e, n


def resolve_fold_trading_days(
    data_dir: str,
    dataset_name: str,
    fold_def: dict[str, Any],
) -> list[str]:
    """Unique trading_day values in the fold validation window (sorted)."""
    start, stop, _ = _slice_bounds(fold_def, "validation")
    if start is None or stop is None or stop <= start or not dataset_name:
        return []
    try:
        df, _, _ = load_dataset_frame(data_dir, dataset_name, columns=["trading_day"])
    except DatasetLoaderError:
        return []
    if stop > len(df):
        stop = len(df)
    if start >= len(df) or start < 0:
        return []
    days = df.iloc[start:stop]["trading_day"].dropna().astype(str).unique().tolist()
    return sorted(days)


def load_fold_feature_importance(
    data_dir: str,
    model_name: str,
    fold: int,
) -> dict[str, Any]:
    path = os.path.join(_fold_dir(data_dir, model_name, fold), "feature_importance.csv")
    if not os.path.isfile(path):
        return {"available": False, "rows": [], "message": "Not saved for this fold"}
    try:
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                feat = str(raw.get("feature") or "").strip()
                if not feat:
                    continue
                imp = _num(raw.get("importance_pct"))
                if imp is None:
                    imp = _num(raw.get("importance"))
                rows.append({"feature": feat, "importance_pct": imp})
        if not rows:
            return {"available": False, "rows": [], "message": "Not saved for this fold"}
        return {"available": True, "rows": rows, "message": None, "path": path}
    except OSError:
        return {"available": False, "rows": [], "message": "Not saved for this fold"}


def _importance_map(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        val = _num(row.get("importance_pct"))
        if feat and val is not None:
            out[feat] = val
    return out


def build_feature_importance_delta(
    imp_a: dict[str, Any],
    imp_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
    limit: int | None = None,
    near_identical_pct: float = 1.0,
) -> dict[str, Any]:
    if not imp_a.get("available") and not imp_b.get("available"):
        return {
            "available": False,
            "message": "Not saved for either fold",
            "rows": [],
            "largest_shifts": [],
        }
    if not imp_a.get("available") or not imp_b.get("available"):
        missing = label_a if not imp_a.get("available") else label_b
        return {
            "available": False,
            "message": f"Not saved for {missing}",
            "rows": [],
            "largest_shifts": [],
        }
    map_a = _importance_map(imp_a)
    map_b = _importance_map(imp_b)
    features = sorted(
        set(map_a) | set(map_b),
        key=lambda f: abs(map_b.get(f, 0.0) - map_a.get(f, 0.0)),
        reverse=True,
    )
    if limit is not None:
        features = features[:limit]
    rows = []
    for feat in features:
        va = map_a.get(feat)
        vb = map_b.get(feat)
        delta = None
        if va is not None and vb is not None:
            delta = round(vb - va, 4)
        signal = _importance_signal(
            delta,
            near_identical_pct=near_identical_pct,
            label_a=label_a,
            label_b=label_b,
        )
        rows.append({
            "feature": feat,
            "fold_a": va,
            "fold_b": vb,
            "delta": delta,
            "delta_display": (
                f"{delta:+.2f}%" if delta is not None else "—"
            ),
            "abs_delta": abs(delta) if delta is not None else None,
            "signal": signal["code"],
            "signal_label": signal["label"],
            "signal_emoji": signal["emoji"],
            "arrow": signal["arrow"],
        })
    largest = []
    for row in rows[:5]:
        if row.get("delta") is None:
            continue
        largest.append({
            "feature": row["feature"],
            "delta": row["delta"],
            "delta_display": row["delta_display"],
            "arrow": row["arrow"],
            "signal_emoji": row["signal_emoji"],
            "signal_label": row["signal_label"],
        })
    return {
        "available": True,
        "message": None,
        "rows": rows,
        "largest_shifts": largest,
        "label_a": label_a,
        "label_b": label_b,
        "near_identical_pct": near_identical_pct,
    }


def _importance_signal(
    delta: float | None,
    *,
    near_identical_pct: float = 1.0,
    label_a: str = "Fold A",
    label_b: str = "Fold B",
) -> dict[str, str]:
    """Classify which fold relies more on the feature (Δ = Fold B − Fold A)."""
    if delta is None or abs(float(delta)) <= float(near_identical_pct):
        return {
            "code": "similar",
            "emoji": "⚪",
            "label": "Nearly identical",
            "arrow": "·",
        }
    if float(delta) < 0:
        return {
            "code": "fold_a",
            "emoji": "🟢",
            "label": f"{label_a} relies much more",
            "arrow": "↓",
        }
    return {
        "code": "fold_b",
        "emoji": "🔴",
        "label": f"{label_b} relies much more",
        "arrow": "↑",
    }



def _premium_band_rows(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
) -> list[dict[str, Any]]:
    bands_a = {
        str(b.get("band") or b.get("band_label") or ""): b
        for b in (metrics_a.get("premium_band_performance") or [])
        if isinstance(b, dict)
    }
    bands_b = {
        str(b.get("band") or b.get("band_label") or ""): b
        for b in (metrics_b.get("premium_band_performance") or [])
        if isinstance(b, dict)
    }
    keys = list(dict.fromkeys([*bands_a.keys(), *bands_b.keys()]))
    out: list[dict[str, Any]] = []
    for key in keys:
        if not key:
            continue
        a = bands_a.get(key) or {}
        b = bands_b.get(key) or {}
        label = a.get("band_label") or b.get("band_label") or f"₹{key}"
        mae_a, mae_b = a.get("mae"), b.get("mae")
        dir_a, dir_b = a.get("directional_accuracy_pct"), b.get("directional_accuracy_pct")
        out.append({
            "band": key,
            "band_label": label,
            "samples_a": a.get("samples"),
            "samples_b": b.get("samples"),
            "mae_a": mae_a,
            "mae_b": mae_b,
            "dir_a": dir_a,
            "dir_b": dir_b,
            "mae_winner": _winner_label(mae_a, mae_b, higher_better=False, label_a=label_a, label_b=label_b),
            "dir_winner": _winner_label(dir_a, dir_b, higher_better=True, label_a=label_a, label_b=label_b),
        })
    return out


def _overall_winner(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
) -> str:
    for key, hb in (
        ("composite_score", True),
        ("directional_accuracy_pct", True),
        ("mae", False),
        ("rmse", False),
    ):
        w = _winner_label(
            metrics_a.get(key),
            metrics_b.get(key),
            higher_better=hb,
            label_a=label_a,
            label_b=label_b,
        )
        if w and w != "Tie":
            return w
    return "Tie"


def build_fold_diagnosis(
    *,
    label_a: str,
    label_b: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    days_a: list[str],
    days_b: list[str],
    premium_bands: list[dict[str, Any]],
    importance: dict[str, Any],
) -> dict[str, Any]:
    winner = _overall_winner(metrics_a, metrics_b, label_a=label_a, label_b=label_b)
    if winner == "Tie":
        headline = f"{label_a} and {label_b} are closely matched."
    else:
        loser = label_b if winner == label_a else label_a
        headline = f"{winner} outperformed {loser}."

    reasons: list[str] = []

    da, db = _num(metrics_a.get("directional_accuracy_pct")), _num(metrics_b.get("directional_accuracy_pct"))
    if da is not None and db is not None:
        d = db - da
        if abs(d) >= 1.0:
            # Express as the winner's advantage in percentage points
            if winner == label_a:
                reasons.append(f"Direction {abs(d):+.0f}%")
            elif winner == label_b:
                reasons.append(f"Direction {abs(d):+.0f}%")
            else:
                reasons.append(f"Direction Δ {d:+.0f} pts")

    mae_a, mae_b = _num(metrics_a.get("mae")), _num(metrics_b.get("mae"))
    if mae_a is not None and mae_b is not None:
        d = mae_b - mae_a
        if abs(d) >= 0.05:
            reasons.append(f"MAE lower by Rs {abs(d):.2f}")

    # Strongest premium-band MAE winner for winner fold
    best_band = None
    best_ratio = 0.0
    for band in premium_bands:
        a, b = _num(band.get("mae_a")), _num(band.get("mae_b"))
        if a is None or b is None or min(a, b) < 1e-9:
            continue
        # Prefer band where winner fold is clearly better on MAE
        if winner == label_a and a < b:
            ratio = b / a
            if ratio > best_ratio:
                best_ratio = ratio
                best_band = band
        elif winner == label_b and b < a:
            ratio = a / b
            if ratio > best_ratio:
                best_ratio = ratio
                best_band = band
    if best_band and best_ratio >= 1.15:
        reasons.append(
            f"Premium {best_band.get('band_label') or best_band.get('band')} much stronger on {winner}"
        )

    if importance.get("available"):
        top = None
        for row in importance.get("rows") or []:
            delta = _num(row.get("delta"))
            if delta is None:
                continue
            if top is None or abs(delta) > abs(_num(top.get("delta")) or 0.0):
                top = row
        if top and abs(_num(top.get("delta")) or 0) >= 1.0:
            feat = top.get("feature")
            delta = _num(top.get("delta")) or 0.0
            side = label_b if delta > 0 else label_a
            reasons.append(
                f"Feature \"{feat}\" importance {delta:+.0f}% pts (higher on {side})"
            )

    if days_a or days_b:
        if set(days_a) != set(days_b):
            a_txt = ", ".join(days_a) if days_a else "—"
            b_txt = ", ".join(days_b) if days_b else "—"
            reasons.append(f"Validation days differ ({a_txt} vs {b_txt})")
        else:
            reasons.append(f"Same validation day(s): {', '.join(days_a) or '—'}")

    if not reasons:
        reasons.append("No dominant single-metric gap; inspect tables for details.")

    return {
        "headline": headline,
        "winner": winner,
        "reasons": reasons,
    }


def try_fold_prediction_metrics(
    data_dir: str,
    model_name: str,
    fold_a: int,
    fold_b: int,
) -> dict[str, Any]:
    """Optional Hit Rate / DD / Time-to-Target — only when prediction DB has fold rows."""
    # V1: fold ML artifacts do not store these; Prediction Runs may in future.
    _ = (data_dir, model_name, fold_a, fold_b)
    return {
        "available": False,
        "message": "Prediction data not available for this fold.",
        "rows": [],
    }


def build_fold_comparison(
    data_dir: str,
    doc: dict[str, Any],
    fold_a: int,
    fold_b: int,
) -> dict[str, Any]:
    model_name = str(doc.get("model_name") or "").strip()
    if not model_name:
        return {"ok": False, "error": "Model name missing"}
    if not doc.get("is_walk_forward"):
        return {"ok": False, "error": "Fold comparison requires a walk-forward model"}
    try:
        fa, fb = int(fold_a), int(fold_b)
    except (TypeError, ValueError):
        return {"ok": False, "error": "Invalid fold numbers"}
    if fa == fb:
        return {"ok": False, "error": "Choose two different folds"}

    label_a = f"Fold {fa}"
    label_b = f"Fold {fb}"

    fr_a = _find_fold_result(doc, fa)
    fr_b = _find_fold_result(doc, fb)
    # Allow disk-only folds
    fold_def_a = _resolve_fold_def(data_dir, model_name, fa, fr_a)
    fold_def_b = _resolve_fold_def(data_dir, model_name, fb, fr_b)
    if not fold_def_a and not fr_a:
        return {"ok": False, "error": f"{label_a} not found for this model"}
    if not fold_def_b and not fr_b:
        return {"ok": False, "error": f"{label_b} not found for this model"}

    metrics_a = _resolve_fold_metrics(data_dir, model_name, fa, fr_a)
    metrics_b = _resolve_fold_metrics(data_dir, model_name, fb, fr_b)
    if not metrics_a:
        return {"ok": False, "error": f"No metrics found for {label_a}"}
    if not metrics_b:
        return {"ok": False, "error": f"No metrics found for {label_b}"}

    dataset = _dataset_name(doc)
    days_a = resolve_fold_trading_days(data_dir, dataset, fold_def_a)
    days_b = resolve_fold_trading_days(data_dir, dataset, fold_def_b)

    _, _, val_rows_a = _slice_bounds(fold_def_a, "validation")
    _, _, val_rows_b = _slice_bounds(fold_def_b, "validation")
    _, _, train_rows_a = _slice_bounds(fold_def_a, "train")
    _, _, train_rows_b = _slice_bounds(fold_def_b, "train")
    val_start_a, val_stop_a, _ = _slice_bounds(fold_def_a, "validation")
    val_start_b, val_stop_b, _ = _slice_bounds(fold_def_b, "validation")

    summary_metrics = [
        _metric_row("MAE", metrics_a.get("mae"), metrics_b.get("mae"), metric_key="mae", label_a=label_a, label_b=label_b),
        _metric_row("RMSE", metrics_a.get("rmse"), metrics_b.get("rmse"), metric_key="rmse", label_a=label_a, label_b=label_b),
        _metric_row(
            "Direction",
            metrics_a.get("directional_accuracy_pct"),
            metrics_b.get("directional_accuracy_pct"),
            metric_key="directional_accuracy_pct",
            label_a=label_a,
            label_b=label_b,
        ),
        _metric_row(
            "Composite",
            metrics_a.get("composite_score"),
            metrics_b.get("composite_score"),
            metric_key="composite_score",
            label_a=label_a,
            label_b=label_b,
        ),
    ]

    error_metrics = [
        _metric_row("MAE", metrics_a.get("mae"), metrics_b.get("mae"), metric_key="mae", label_a=label_a, label_b=label_b),
        _metric_row("RMSE", metrics_a.get("rmse"), metrics_b.get("rmse"), metric_key="rmse", label_a=label_a, label_b=label_b),
        _metric_row(
            "Bias",
            metrics_a.get("prediction_bias"),
            metrics_b.get("prediction_bias"),
            metric_key="prediction_bias",
            label_a=label_a,
            label_b=label_b,
        ),
        _metric_row(
            "P95",
            metrics_a.get("p95_error"),
            metrics_b.get("p95_error"),
            metric_key="p95_error",
            label_a=label_a,
            label_b=label_b,
        ),
    ]

    premium_bands = _premium_band_rows(metrics_a, metrics_b, label_a=label_a, label_b=label_b)

    imp_a = load_fold_feature_importance(data_dir, model_name, fa)
    imp_b = load_fold_feature_importance(data_dir, model_name, fb)
    importance = build_feature_importance_delta(imp_a, imp_b, label_a=label_a, label_b=label_b)

    diagnosis = build_fold_diagnosis(
        label_a=label_a,
        label_b=label_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        days_a=days_a,
        days_b=days_b,
        premium_bands=premium_bands,
        importance=importance,
    )

    prediction_metrics = try_fold_prediction_metrics(data_dir, model_name, fa, fb)

    error_histograms = build_fold_error_histogram_comparison(
        data_dir,
        doc,
        fold_a=fa,
        fold_b=fb,
        fold_def_a=fold_def_a,
        fold_def_b=fold_def_b,
        label_a=label_a,
        label_b=label_b,
        days_a=days_a,
        days_b=days_b,
    )
    if error_histograms.get("available") and error_histograms.get("insight"):
        diagnosis["reasons"] = list(diagnosis.get("reasons") or []) + [error_histograms["insight"]]

    model_features: list[str] = []
    scoring = _get_fold_scoring_bundle(data_dir, doc)
    if scoring:
        model_features = list(scoring.get("features") or [])

    from .fold_diagnostics import build_fold_pair_diagnostics

    diagnostics = build_fold_pair_diagnostics(
        data_dir,
        dataset=dataset,
        fold_a=fa,
        fold_b=fb,
        fold_def_a=fold_def_a or {},
        fold_def_b=fold_def_b or {},
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        label_a=label_a,
        label_b=label_b,
        days_a=days_a,
        days_b=days_b,
        error_histograms=error_histograms,
        model_features=model_features,
    )
    why = diagnostics.get("why") if isinstance(diagnostics.get("why"), dict) else {}
    if why.get("available") and why.get("bullets"):
        # Prefer Why? bullets as the primary diagnosis reasons
        diagnosis = {
            **diagnosis,
            "headline": why.get("headline") or diagnosis.get("headline"),
            "reasons": list(why.get("bullets") or diagnosis.get("reasons") or []),
            "worse_label": why.get("worse_label"),
            "better_label": why.get("better_label"),
            "metric_cards": why.get("metric_cards") or [],
            "what_is_unique": why.get("what_is_unique") or {},
        }

    source = "champion" if fr_a and fr_b else "disk"

    return {
        "ok": True,
        "model_name": model_name,
        "dataset": dataset,
        "fold_a": fa,
        "fold_b": fb,
        "label_a": label_a,
        "label_b": label_b,
        "source": source,
        "diagnosis": diagnosis,
        "summary_metrics": summary_metrics,
        "error_metrics": error_metrics,
        "error_histograms": error_histograms,
        "premium_bands": premium_bands,
        "feature_importance": importance,
        "prediction_metrics": prediction_metrics,
        "diagnostics": diagnostics,
        "validation_window": {
            "fold_a": {
                "fold": fa,
                "trading_days": days_a,
                "validation_rows": val_rows_a,
                "train_rows": train_rows_a,
                "validation_start": val_start_a,
                "validation_stop": val_stop_a,
            },
            "fold_b": {
                "fold": fb,
                "trading_days": days_b,
                "validation_rows": val_rows_b,
                "train_rows": train_rows_b,
                "validation_start": val_start_b,
                "validation_stop": val_stop_b,
            },
        },
    }


def build_fold_comparison_csv(report: dict[str, Any]) -> str:
    """Serialize every Fold Comparison tab into one multi-section CSV."""
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)

    def _section(title: str) -> None:
        writer.writerow([])
        writer.writerow([title])

    def _cell(v: Any) -> Any:
        if v is None:
            return ""
        if isinstance(v, float):
            return round(v, 6)
        return v

    if not report.get("ok"):
        writer.writerow(["Error", report.get("error") or "Comparison failed"])
        return buf.getvalue().lstrip("\n")

    label_a = str(report.get("label_a") or "Fold A")
    label_b = str(report.get("label_b") or "Fold B")

    _section("Fold Comparison")
    writer.writerow(["Field", "Value"])
    writer.writerow(["Model", report.get("model_name") or ""])
    writer.writerow(["Dataset", report.get("dataset") or ""])
    writer.writerow(["Fold A", report.get("fold_a")])
    writer.writerow(["Fold B", report.get("fold_b")])
    writer.writerow(["Label A", label_a])
    writer.writerow(["Label B", label_b])
    writer.writerow(["Metrics source", report.get("source") or ""])

    diagnosis = report.get("diagnosis") if isinstance(report.get("diagnosis"), dict) else {}
    _section("Summary — Why?")
    writer.writerow(["Field", "Value"])
    writer.writerow(["Headline", diagnosis.get("headline") or ""])
    writer.writerow(["Worse fold", diagnosis.get("worse_label") or ""])
    writer.writerow(["Better fold", diagnosis.get("better_label") or ""])
    writer.writerow(["Winner (legacy)", diagnosis.get("winner") or ""])
    for bullet in diagnosis.get("reasons") or []:
        writer.writerow(["Why", bullet])

    for card in diagnosis.get("metric_cards") or []:
        if not isinstance(card, dict):
            continue
        _section(f"Summary — Metric Card: {card.get('title') or card.get('key') or ''}")
        writer.writerow(["Field", "Value"])
        writer.writerow(["Fold", card.get("fold_label") or ""])
        writer.writerow(["Value", card.get("value_display") or ""])
        writer.writerow(["Better value", _cell(card.get("better_value"))])
        writer.writerow(["Worse value", _cell(card.get("worse_value"))])
        for w in card.get("why") or []:
            writer.writerow(["Why", w])

    unique = diagnosis.get("what_is_unique") if isinstance(diagnosis.get("what_is_unique"), dict) else {}
    if unique.get("available"):
        _section(f"Summary — What is unique about {unique.get('fold_label') or ''}")
        writer.writerow(["Rank", "Feature", "Z-score display", "Z-score"])
        for i, row in enumerate(unique.get("rows") or [], start=1):
            if not isinstance(row, dict):
                continue
            writer.writerow([
                i,
                row.get("feature") or "",
                row.get("display") or "",
                _cell(row.get("z_score")),
            ])

    diag = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    for side, key in ((label_a, "context_a"), (label_b, "context_b")):
        ctx = diag.get(key) if isinstance(diag.get(key), dict) else {}
        _section(f"Summary — Fold Context: {side}")
        writer.writerow(["Field", "Value"])
        if not ctx.get("available"):
            writer.writerow(["Status", ctx.get("message") or "Unavailable"])
        else:
            writer.writerow(["Market regime", ctx.get("market_regime") or ""])
            for row in ctx.get("rows") or []:
                if isinstance(row, dict):
                    writer.writerow([row.get("label") or row.get("key") or "", row.get("value") or ""])

    _section("Summary — Summary Metrics")
    writer.writerow(["Metric", label_a, label_b, "Delta", "Winner"])
    for row in report.get("summary_metrics") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            writer.writerow([
                row[0],
                _cell(row[1]),
                _cell(row[2]),
                row[3] if len(row) > 3 else "",
                row[4] if len(row) > 4 else "",
            ])

    vw = report.get("validation_window") if isinstance(report.get("validation_window"), dict) else {}
    wa = vw.get("fold_a") if isinstance(vw.get("fold_a"), dict) else {}
    wb = vw.get("fold_b") if isinstance(vw.get("fold_b"), dict) else {}
    _section("Summary — Validation Window")
    writer.writerow(["Field", label_a, label_b])
    writer.writerow(["Fold", wa.get("fold"), wb.get("fold")])
    writer.writerow(["Validation rows", _cell(wa.get("validation_rows")), _cell(wb.get("validation_rows"))])
    writer.writerow(["Training rows", _cell(wa.get("train_rows")), _cell(wb.get("train_rows"))])
    writer.writerow([
        "Validation indices",
        f"{wa.get('validation_start')} – {wa.get('validation_stop')}"
        if wa.get("validation_start") is not None else "",
        f"{wb.get('validation_start')} – {wb.get('validation_stop')}"
        if wb.get("validation_start") is not None else "",
    ])
    writer.writerow([
        "Trading days",
        ", ".join(wa.get("trading_days") or []),
        ", ".join(wb.get("trading_days") or []),
    ])

    dist = diag.get("distribution_shift") if isinstance(diag.get("distribution_shift"), dict) else {}
    _section("Distributions — Feature Distribution Shift")
    if not dist.get("available"):
        writer.writerow(["Status", dist.get("message") or diag.get("message") or "Unavailable"])
    else:
        writer.writerow([
            "Feature", label_a, label_b, "Delta", "Pct change", "Z diff", "Severity",
        ])
        for row in dist.get("rows") or []:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("feature") or "",
                _cell(row.get("fold_a")),
                _cell(row.get("fold_b")),
                row.get("display_delta") or _cell(row.get("delta")),
                _cell(row.get("pct_change")),
                _cell(row.get("z_diff")),
                row.get("severity") or "",
            ])

    for side, key in ((label_a, "outliers_a"), (label_b, "outliers_b")):
        out = diag.get(key) if isinstance(diag.get(key), dict) else {}
        _section(f"Distributions — Unusual vs Training: {side}")
        if not out.get("available"):
            writer.writerow(["Status", out.get("message") or "Unavailable"])
        else:
            writer.writerow([
                "Feature", "Fold mean", "Train mean", "Difference", "Z-score", "Percentile",
            ])
            for row in out.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                writer.writerow([
                    row.get("feature") or "",
                    _cell(row.get("fold_mean")),
                    _cell(row.get("train_mean")),
                    _cell(row.get("difference")),
                    row.get("display") or _cell(row.get("z_score")),
                    _cell(row.get("percentile")),
                ])

    hist = report.get("error_histograms") if isinstance(report.get("error_histograms"), dict) else {}
    _section("Prediction Errors — Error Histogram")
    if not hist.get("available"):
        writer.writerow(["Status", hist.get("message") or "Unavailable"])
    else:
        writer.writerow(["Insight", hist.get("insight") or ""])
        writer.writerow(["Unit", hist.get("unit") or ""])
        for side_key, side_label in (("fold_a", label_a), ("fold_b", label_b)):
            block = hist.get(side_key) if isinstance(hist.get(side_key), dict) else {}
            _section(f"Prediction Errors — {side_label}")
            writer.writerow(["Field", "Value"])
            writer.writerow(["Trading days", ", ".join(str(d) for d in (block.get("trading_days") or []))])
            writer.writerow(["Total rows", _cell(block.get("total"))])
            writer.writerow(["Mean abs error", _cell(block.get("mean_abs_error"))])
            writer.writerow(["Median abs error", _cell(block.get("median_abs_error"))])
            writer.writerow(["Max abs error", _cell(block.get("max_abs_error"))])
            writer.writerow(["Tail >5 %", _cell(block.get("tail_gt5_pct"))])
            writer.writerow(["Bucket", "Count", "Share %"])
            for row in block.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                writer.writerow([
                    row.get("bucket") or "",
                    _cell(row.get("count")),
                    _cell(row.get("pct")),
                ])

    _section("Premium Bands")
    bands = report.get("premium_bands") or []
    if not bands:
        writer.writerow(["Status", "Unavailable"])
    else:
        writer.writerow([
            "Band",
            f"Rows {label_a}",
            f"Rows {label_b}",
            f"MAE {label_a}",
            f"MAE {label_b}",
            "MAE Winner",
            f"Dir {label_a}",
            f"Dir {label_b}",
            "Dir Winner",
        ])
        for b in bands:
            if not isinstance(b, dict):
                continue
            writer.writerow([
                b.get("band_label") or b.get("band") or "",
                _cell(b.get("samples_a")),
                _cell(b.get("samples_b")),
                _cell(b.get("mae_a")),
                _cell(b.get("mae_b")),
                b.get("mae_winner") or "",
                _cell(b.get("dir_a")),
                _cell(b.get("dir_b")),
                b.get("dir_winner") or "",
            ])

    imp = report.get("feature_importance") if isinstance(report.get("feature_importance"), dict) else {}
    _section("Feature Importance Δ")
    if not imp.get("available"):
        writer.writerow(["Status", imp.get("message") or "Unavailable"])
    else:
        writer.writerow([
            "Feature",
            f"{label_a} %",
            f"{label_b} %",
            "Delta",
            "Signal",
            "Signal label",
        ])
        for row in imp.get("rows") or []:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("feature") or "",
                _cell(row.get("fold_a")),
                _cell(row.get("fold_b")),
                row.get("delta_display") or _cell(row.get("delta")),
                row.get("signal") or "",
                row.get("signal_label") or "",
            ])
        if imp.get("largest_shifts"):
            _section("Feature Importance — Largest Shifts")
            writer.writerow(["Feature", "Delta", "Arrow", "Signal"])
            for row in imp.get("largest_shifts") or []:
                if not isinstance(row, dict):
                    continue
                writer.writerow([
                    row.get("feature") or "",
                    row.get("delta_display") or _cell(row.get("delta")),
                    row.get("arrow") or "",
                    row.get("signal_label") or "",
                ])

    _section("Error Metrics")
    writer.writerow(["Metric", label_a, label_b, "Delta", "Winner"])
    for row in report.get("error_metrics") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            writer.writerow([
                row[0],
                _cell(row[1]),
                _cell(row[2]),
                row[3] if len(row) > 3 else "",
                row[4] if len(row) > 4 else "",
            ])

    pred = report.get("prediction_metrics") if isinstance(report.get("prediction_metrics"), dict) else {}
    _section("Prediction Metrics")
    if not pred.get("available"):
        writer.writerow(["Status", pred.get("message") or "Unavailable"])
    else:
        writer.writerow(["Metric", label_a, label_b, "Delta", "Winner"])
        for row in pred.get("rows") or []:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                writer.writerow([
                    row[0],
                    _cell(row[1]),
                    _cell(row[2]),
                    row[3] if len(row) > 3 else "",
                    row[4] if len(row) > 4 else "",
                ])
            elif isinstance(row, dict):
                writer.writerow([
                    row.get("metric") or row.get("label") or "",
                    _cell(row.get("fold_a") or row.get("a")),
                    _cell(row.get("fold_b") or row.get("b")),
                    row.get("delta") or "",
                    row.get("winner") or "",
                ])

    return buf.getvalue().lstrip("\n")
