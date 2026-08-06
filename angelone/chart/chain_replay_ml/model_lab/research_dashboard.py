"""Research Dashboard analytics on Model Lab prediction datasets (Phase 3)."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any

from .prediction_schema import DATASET_TYPE_SEEN, DATASET_TYPE_UNSEEN, normalize_dataset_type
from .store import ModelLabStore

# Bump when summary table layout / metric definitions change
DASHBOARD_CACHE_SCHEMA_VERSION = 1

# Entry premium (current_ltp) bands for Phase 3.3
PREMIUM_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("₹15–30", 15.0, 30.0),
    ("₹30–50", 30.0, 50.0),
    ("₹50–100", 50.0, 100.0),
    ("₹100–200", 100.0, 200.0),
    ("₹200+", 200.0, None),
)

# Evaluation Set control values
EVAL_SET_ALL = "all"
EVAL_SET_SEEN = "seen"
EVAL_SET_UNSEEN = "unseen"
EVAL_SET_LABELS = {
    EVAL_SET_ALL: "Seen + Unseen",
    EVAL_SET_SEEN: "Seen Only",
    EVAL_SET_UNSEEN: "Unseen Only",
}


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or math.isinf(x):
        return None
    return x


def _rate(hits: float | None, n: float | None) -> float | None:
    if hits is None or n is None or n <= 0:
        return None
    return float(hits) / float(n)


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """Inclusive linear percentile; *sorted_vals* must be sorted ascending."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    p = max(0.0, min(100.0, float(p)))
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    w = k - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def _mean(vals: list[float]) -> float | None:
    return float(statistics.fmean(vals)) if vals else None


def _median(vals: list[float]) -> float | None:
    return float(statistics.median(vals)) if vals else None


def _rmse(vals: list[float]) -> float | None:
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hit / dir / MAE / DD averages for a subset of prediction rows."""
    n = len(rows)
    n_hit = sum(1 for r in rows if r.get("target_reached") is not None)
    hit = sum(1 for r in rows if r.get("target_reached") == 1)
    n_dir = sum(1 for r in rows if r.get("direction_correct") is not None)
    dir_h = sum(1 for r in rows if r.get("direction_correct") == 1)
    abs_errs = [float(r["absolute_error"]) for r in rows if r.get("absolute_error") is not None]
    dds = [float(r["dd_before_target"]) for r in rows if r.get("dd_before_target") is not None]
    ttts = [
        float(r["time_to_target"])
        for r in rows
        if r.get("time_to_target") is not None and float(r["time_to_target"]) >= 0
    ]
    prems = [float(r["premium_error_pct"]) for r in rows if r.get("premium_error_pct") is not None]
    return {
        "rows": n,
        "hit_rate": _rate(float(hit), float(n_hit)),
        "direction_accuracy": _rate(float(dir_h), float(n_dir)),
        "mae": _mean(abs_errs),
        "avg_dd_before_target": _mean(dds),
        "avg_time_to_target": _mean(ttts),
        "premium_mae": _mean(prems),
    }


def _trend_is_up(trend: Any, move: Any) -> bool | None:
    if trend is not None and str(trend).strip():
        t = str(trend).strip().upper()
        if t == "UP":
            return True
        if t == "DOWN":
            return False
        if t == "FLAT":
            return None
    m = _f(move)
    if m is None:
        return None
    if m > 0:
        return True
    if m < 0:
        return False
    return None


def _sql_core_kpis(
    store: ModelLabStore,
    *,
    where_sql: str = "",
    where_args: list[Any] | None = None,
) -> dict[str, Any]:
    """Fast SQL aggregates for Confidence Filter Effect (hit / DD / max profit)."""
    cols = set(store._prediction_table_columns())
    need = {"target_reached", "dd_before_target", "maximum_profit"}
    if not need.issubset(cols):
        return {"rows": 0, "target_hit_rate": None, "avg_dd": None, "avg_max_profit": None}
    sql = f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN "target_reached" IS NOT NULL THEN 1 ELSE 0 END) AS n_hit,
            SUM(CASE WHEN "target_reached" = 1 THEN 1 ELSE 0 END) AS hits,
            AVG("dd_before_target") AS avg_dd,
            AVG("maximum_profit") AS avg_mfe
        FROM prediction_dataset
        {where_sql}
    """
    row = store.conn.execute(sql, list(where_args or [])).fetchone()
    if not row:
        return {"rows": 0, "target_hit_rate": None, "avg_dd": None, "avg_max_profit": None}
    n = int(row[0] or 0)
    n_hit = float(row[1] or 0)
    hits = float(row[2] or 0)
    return {
        "rows": n,
        "target_hit_rate": _rate(hits, n_hit),
        "avg_dd": float(row[3]) if row[3] is not None else None,
        "avg_max_profit": float(row[4]) if row[4] is not None else None,
    }


def _pct_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before == 0:
        return None
    return 100.0 * (after - before) / before


def _pp_change(before: float | None, after: float | None) -> float | None:
    """Percentage-point change for rates in [0, 1]."""
    if before is None or after is None:
        return None
    return 100.0 * (after - before)


def normalize_evaluation_set(value: str | None) -> str:
    raw = str(value or EVAL_SET_ALL).strip().lower().replace(" ", "_").replace("+", "_")
    if raw in ("all", "both", "seen_unseen", "seen_+_unseen", "seen_and_unseen", ""):
        return EVAL_SET_ALL
    if raw in ("seen", "seen_only", "training"):
        return EVAL_SET_SEEN
    if raw in ("unseen", "unseen_only", "holdout"):
        return EVAL_SET_UNSEEN
    return EVAL_SET_ALL


def resolve_confidence_filter_spec(
    confidence_filter: str | None = None,
    *,
    confidence_classifier: str | None = None,
    confidence_prediction: int | str | None = None,
) -> dict[str, Any]:
    """
    Resolve Classifier + Prediction into a filter spec.

    Returns ``{active, model_key, value, mode, pred_col, label}``.
    Legacy *confidence_filter* strings like ``target_hit_1`` / ``rr_1_1_0`` still work.
    """
    from .confidence_inference import INFERENCE_COLUMNS
    from .confidence_manifest import TARGET_BY_KEY

    inactive = {
        "active": False,
        "model_key": None,
        "value": None,
        "mode": "disabled",
        "pred_col": None,
        "label": None,
    }

    pred_raw = confidence_prediction
    if pred_raw is not None and str(pred_raw).strip().lower() not in (
        "",
        "disabled",
        "none",
        "off",
    ):
        try:
            want = int(pred_raw)
        except (TypeError, ValueError):
            return inactive
        if want not in (0, 1):
            return inactive
        key = str(confidence_classifier or "target_hit").strip().lower()
        if key not in INFERENCE_COLUMNS:
            return inactive
        cols = INFERENCE_COLUMNS[key]
        return {
            "active": True,
            "model_key": key,
            "value": want,
            "mode": f"{key}_{want}",
            "pred_col": cols["pred"],
            "label": (TARGET_BY_KEY.get(key) or {}).get("label") or key,
        }

    mode = str(confidence_filter or "").strip().lower()
    if mode in ("", "disabled", "none", "off"):
        return inactive
    for key in sorted(INFERENCE_COLUMNS.keys(), key=len, reverse=True):
        if mode == f"{key}_1":
            cols = INFERENCE_COLUMNS[key]
            return {
                "active": True,
                "model_key": key,
                "value": 1,
                "mode": mode,
                "pred_col": cols["pred"],
                "label": (TARGET_BY_KEY.get(key) or {}).get("label") or key,
            }
        if mode == f"{key}_0":
            cols = INFERENCE_COLUMNS[key]
            return {
                "active": True,
                "model_key": key,
                "value": 0,
                "mode": mode,
                "pred_col": cols["pred"],
                "label": (TARGET_BY_KEY.get(key) or {}).get("label") or key,
            }
    return inactive


def _evaluation_set_day_clause(
    store: ModelLabStore,
    evaluation_set: str,
    *,
    cols: set[str],
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """
    SQL fragments restricting prediction rows to Seen / Unseen days via build catalog.
    """
    eval_set = normalize_evaluation_set(evaluation_set)
    meta: dict[str, Any] = {
        "evaluation_set": eval_set,
        "evaluation_set_label": EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
    }
    if eval_set == EVAL_SET_ALL or "trading_day" not in cols:
        return [], [], meta

    summary = store.read_prediction_summary() or {}
    lab_uuid = str(summary.get("lab_uuid") or "").strip()
    days: list[str] = []
    if lab_uuid:
        for d in store.list_build_days(lab_uuid) or []:
            dtype = normalize_dataset_type(d.get("dataset_type"))
            want = DATASET_TYPE_SEEN if eval_set == EVAL_SET_SEEN else DATASET_TYPE_UNSEEN
            if dtype == want:
                td = str(d.get("trading_day") or "").strip()
                if td:
                    days.append(td)
    meta["day_count"] = len(days)
    if not days:
        # No matching days → empty result set
        return ["0 = 1"], [], meta
    placeholders = ", ".join("?" for _ in days)
    return [f'"trading_day" IN ({placeholders})'], list(days), meta


def compute_confidence_filter_effect(
    store: ModelLabStore,
    *,
    conf_spec: dict[str, Any] | None = None,
    conf_mode: str | None = None,
    day_filter: str = "",
    evaluation_set: str = EVAL_SET_ALL,
) -> dict[str, Any]:
    """
    Before → after summary when a Confidence Filter is active.

    Baseline = same Evaluation Set / trading-day scope without confidence filter.
    Filtered = confidence_*_pred = 0|1 for the selected classifier.
    """
    empty = {"available": False}
    spec = conf_spec
    if spec is None:
        spec = resolve_confidence_filter_spec(conf_mode)
    if not spec.get("active"):
        return empty
    pred_col = str(spec.get("pred_col") or "")
    cols = set(store._prediction_table_columns())
    if not pred_col or pred_col not in cols:
        return {
            **empty,
            "error": f"Confidence column missing for {spec.get('label') or 'classifier'}.",
        }

    base_parts: list[str] = []
    base_args: list[Any] = []
    if day_filter and "trading_day" in cols:
        base_parts.append('"trading_day" = ?')
        base_args.append(day_filter)
    eval_parts, eval_args, _eval_meta = _evaluation_set_day_clause(
        store, evaluation_set, cols=cols
    )
    base_parts.extend(eval_parts)
    base_args.extend(eval_args)
    base_where = (" WHERE " + " AND ".join(base_parts)) if base_parts else ""

    want = int(spec["value"])
    filt_parts = list(base_parts) + [f'"{pred_col}" = ?']
    filt_args = list(base_args) + [want]
    filt_where = " WHERE " + " AND ".join(filt_parts)

    before = _sql_core_kpis(store, where_sql=base_where, where_args=base_args)
    after = _sql_core_kpis(store, where_sql=filt_where, where_args=filt_args)
    n0 = int(before.get("rows") or 0)
    n1 = int(after.get("rows") or 0)
    removed = max(n0 - n1, 0)
    removed_pct = (100.0 * removed / n0) if n0 > 0 else None
    return {
        "available": True,
        "mode": spec.get("mode"),
        "model_key": spec.get("model_key"),
        "label": spec.get("label"),
        "value": want,
        "rows_before": n0,
        "rows_after": n1,
        "rows_removed": removed,
        "rows_removed_pct": removed_pct,
        "target_hit_before": before.get("target_hit_rate"),
        "target_hit_after": after.get("target_hit_rate"),
        "target_hit_delta_pp": _pp_change(
            before.get("target_hit_rate"), after.get("target_hit_rate")
        ),
        "avg_dd_before": before.get("avg_dd"),
        "avg_dd_after": after.get("avg_dd"),
        "avg_dd_delta_pct": _pct_change(before.get("avg_dd"), after.get("avg_dd")),
        "avg_max_profit_before": before.get("avg_max_profit"),
        "avg_max_profit_after": after.get("avg_max_profit"),
        "avg_max_profit_delta_pct": _pct_change(
            before.get("avg_max_profit"), after.get("avg_max_profit")
        ),
    }


def _load_feature_map(store: ModelLabStore, cols: set[str]) -> list[tuple[str, str]]:
    """Return [(feature_name, physical_column), ...] (embedded sf_* or master col)."""
    from .prediction_feature_store import PredictionFeatureStore

    access = PredictionFeatureStore.from_store(store)
    pairs = access.feature_map()
    if pairs:
        return pairs
    # Fallback for labs that never wrote feature_columns_json
    out: list[tuple[str, str]] = []
    for col in sorted(c for c in cols if str(c).startswith("sf_")):
        out.append((col.removeprefix("sf_"), col))
    return out


def _compute_research_dashboard_fresh(
    db_path: str,
    *,
    data_dir: str | None = None,
    trading_day: str | None = None,
    confidence_filter: str | None = None,
    confidence_classifier: str | None = None,
    confidence_prediction: int | str | None = None,
    evaluation_set: str | None = None,
) -> dict[str, Any]:
    """
    Scan prediction_dataset outcomes and compute dashboard payload.

    Feature Research is intentionally excluded — it is a separate workload.
    Rates are fractions in [0, 1]. Time metrics in seconds. Premium errors in %.

    Confidence Filter: Classifier (model key) + Prediction (=1 / =0).
    Evaluation Set: all | seen | unseen (Seen/Unseen via build-day catalog).
    Legacy *confidence_filter* ``target_hit_1`` / ``rr_1_1_0`` still accepted.
    """
    _ = data_dir  # reserved for callers that still pass lab data_dir
    day_filter = str(trading_day or "").strip()
    conf_spec = resolve_confidence_filter_spec(
        confidence_filter,
        confidence_classifier=confidence_classifier,
        confidence_prediction=confidence_prediction,
    )
    conf_mode = str(conf_spec.get("mode") or "disabled")
    eval_set = normalize_evaluation_set(evaluation_set)
    empty: dict[str, Any] = {
        "available": False,
        "error": None,
        "cached": False,
        "total_predictions": 0,
        "kpi": {},
        "quality": {},
        "risk": {},
        "error_metrics": {},
        "distribution": {},
        "premium_bands": [],
        "trading_days": [],
        "features": [],
        "confidence_filter": conf_mode,
        "evaluation_set": eval_set,
        "evaluation_set_label": EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
    }
    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = store._prediction_table_columns()
            tables = {
                str(r[0])
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "prediction_dataset" not in tables:
                return empty

            base_cols = [
                c
                for c in (
                    "trading_day",
                    "current_ltp",
                    "expected_move",
                    "actual_move",
                    "predicted_trend",
                    "actual_trend",
                    "direction_correct",
                    "target_reached",
                    "time_to_target",
                    "dd_before_target",
                    "maximum_profit",
                    "maximum_drawdown",
                    "absolute_error",
                    "prediction_error",
                    "premium_error_pct",
                )
                if c in cols
            ]
            if not base_cols:
                return empty

            where_parts: list[str] = []
            where_args: list[Any] = []
            if day_filter and "trading_day" in base_cols:
                where_parts.append('"trading_day" = ?')
                where_args.append(day_filter)

            eval_parts, eval_args, eval_meta = _evaluation_set_day_clause(
                store, eval_set, cols=set(cols)
            )
            where_parts.extend(eval_parts)
            where_args.extend(eval_args)

            conf_meta: dict[str, Any] = {
                "mode": conf_mode,
                "evaluation_set": eval_set,
                "evaluation_set_label": eval_meta.get("evaluation_set_label"),
            }
            if conf_spec.get("active"):
                pred_col = str(conf_spec.get("pred_col") or "")
                if pred_col not in cols:
                    return {
                        **empty,
                        "error": (
                            f"Confidence column missing for {conf_spec.get('label')} — "
                            "run Confidence Inference first."
                        ),
                    }
                want = int(conf_spec["value"])
                where_parts.append(f'"{pred_col}" = ?')
                where_args.append(want)
                thr_col = None
                mid_col = None
                from .confidence_inference import INFERENCE_COLUMNS

                icols = INFERENCE_COLUMNS.get(str(conf_spec["model_key"])) or {}
                thr_col = icols.get("threshold")
                mid_col = icols.get("model_id")
                thr_val = None
                mid_val = None
                if thr_col and thr_col in cols:
                    thr_row = store.conn.execute(
                        f"""
                        SELECT "{thr_col}"{f', "{mid_col}"' if mid_col and mid_col in cols else ''}
                        FROM prediction_dataset
                        WHERE "{pred_col}" IS NOT NULL
                        LIMIT 1
                        """
                    ).fetchone()
                    if thr_row:
                        thr_val = float(thr_row[0]) if thr_row[0] is not None else None
                        if mid_col and mid_col in cols and len(thr_row) > 1:
                            mid_val = thr_row[1]
                conf_meta.update(
                    {
                        "label": conf_spec.get("label"),
                        "model_key": conf_spec.get("model_key"),
                        "value": want,
                        "model": conf_spec.get("label"),
                        "threshold": thr_val,
                        "model_id": mid_val,
                    }
                )

            col_sql = ", ".join(f'"{c}"' for c in base_cols)
            where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
            raw_rows = store.conn.execute(
                f"SELECT {col_sql} FROM prediction_dataset{where_sql}",
                where_args,
            ).fetchall()
            idx = {c: i for i, c in enumerate(base_cols)}
            records: list[dict[str, Any]] = []
            for row in raw_rows:
                rec = {c: row[idx[c]] for c in base_cols}
                for k in ("direction_correct", "target_reached"):
                    if rec.get(k) is not None:
                        try:
                            rec[k] = int(rec[k])
                        except (TypeError, ValueError):
                            rec[k] = None
                records.append(rec)

            _n_src, fingerprint = store.research_dashboard_source_fingerprint()

            filter_effect: dict[str, Any] = {"available": False}
            if conf_spec.get("active"):
                filter_effect = compute_confidence_filter_effect(
                    store,
                    conf_spec=conf_spec,
                    day_filter=day_filter,
                    evaluation_set=eval_set,
                )
                if filter_effect.get("available"):
                    conf_meta["total_rows"] = int(filter_effect.get("rows_before") or 0)
                    conf_meta["rows_remaining"] = int(filter_effect.get("rows_after") or 0)
                    conf_meta["rows_filtered"] = int(filter_effect.get("rows_removed") or 0)

        if not records:
            out = {
                **empty,
                "source_fingerprint": fingerprint,
                "confidence_filter": conf_mode,
                "confidence_filter_meta": conf_meta,
                "confidence_filter_effect": filter_effect,
                "evaluation_set": eval_set,
                "evaluation_set_label": EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
            }
            if day_filter:
                out["filter_trading_day"] = day_filter
            return out

        n = len(records)
        n_dir = sum(1 for r in records if r.get("direction_correct") is not None)
        dir_hits = sum(1 for r in records if r.get("direction_correct") == 1)
        n_hit = sum(1 for r in records if r.get("target_reached") is not None)
        hits = sum(1 for r in records if r.get("target_reached") == 1)
        misses = sum(1 for r in records if r.get("target_reached") == 0)

        ttt_hit = sorted(
            float(r["time_to_target"])
            for r in records
            if r.get("time_to_target") is not None and float(r["time_to_target"]) >= 0
        )
        dd_vals = sorted(
            float(r["dd_before_target"])
            for r in records
            if r.get("dd_before_target") is not None
        )
        mfe_vals = [
            float(r["maximum_profit"])
            for r in records
            if r.get("maximum_profit") is not None
        ]
        mae_path = [
            float(r["maximum_drawdown"])
            for r in records
            if r.get("maximum_drawdown") is not None
        ]
        abs_errs = [
            float(r["absolute_error"])
            for r in records
            if r.get("absolute_error") is not None
        ]
        signed_errs = [
            float(r["prediction_error"])
            for r in records
            if r.get("prediction_error") is not None
        ]
        prem_errs = [
            float(r["premium_error_pct"])
            for r in records
            if r.get("premium_error_pct") is not None
        ]

        dir_acc = _rate(float(dir_hits), float(n_dir))
        hit_rate = _rate(float(hits), float(n_hit))
        miss_rate = _rate(float(misses), float(n_hit))
        mae = _mean(abs_errs)
        rmse = _rmse(signed_errs) if signed_errs else _rmse(abs_errs)
        prem_mae = _mean(prem_errs)
        prem_rmse = _rmse(prem_errs)

        quality = {
            "total_predictions": n,
            "direction_accuracy": dir_acc,
            "target_hit_rate": hit_rate,
            "target_miss_rate": miss_rate,
            "average_time_to_target": _mean(ttt_hit),
            "median_time_to_target": _median(ttt_hit),
            "p95_time_to_target": _percentile(ttt_hit, 95.0),
        }
        risk = {
            "average_dd_before_target": _mean(dd_vals),
            "median_dd_before_target": _median(dd_vals),
            "p95_dd_before_target": _percentile(dd_vals, 95.0),
            "average_max_dd": _mean(mae_path),
            "average_max_profit": _mean(mfe_vals),
        }
        error_metrics = {
            "mae": mae,
            "rmse": rmse,
            "premium_mae": prem_mae,
            "premium_rmse": prem_rmse,
            "mean_prediction_error": _mean(signed_errs),
            "median_absolute_error": _median(abs_errs),
        }

        pred_up = pred_dn = act_up = act_dn = 0
        for r in records:
            pu = _trend_is_up(r.get("predicted_trend"), r.get("expected_move"))
            au = _trend_is_up(r.get("actual_trend"), r.get("actual_move"))
            if pu is True:
                pred_up += 1
            elif pu is False:
                pred_dn += 1
            if au is True:
                act_up += 1
            elif au is False:
                act_dn += 1
        pred_n = pred_up + pred_dn
        act_n = act_up + act_dn
        distribution = {
            "predicted_up_rate": _rate(float(pred_up), float(pred_n)),
            "predicted_down_rate": _rate(float(pred_dn), float(pred_n)),
            "actual_up_rate": _rate(float(act_up), float(act_n)),
            "actual_down_rate": _rate(float(act_dn), float(act_n)),
            "target_hits": hits,
            "target_misses": misses,
            "predicted_up": pred_up,
            "predicted_down": pred_dn,
            "actual_up": act_up,
            "actual_down": act_dn,
        }

        premium_bands: list[dict[str, Any]] = []
        for label, lo, hi in PREMIUM_BANDS:
            band_rows: list[dict[str, Any]] = []
            for r in records:
                ltp = _f(r.get("current_ltp"))
                if ltp is None:
                    continue
                if lo is not None and ltp < lo:
                    continue
                if hi is not None and ltp >= hi:
                    continue
                band_rows.append(r)
            m = _bucket_metrics(band_rows)
            premium_bands.append({"band": label, **m})

        by_day: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            day = str(r.get("trading_day") or "").strip() or "—"
            by_day.setdefault(day, []).append(r)
        trading_days: list[dict[str, Any]] = []
        for day in sorted(by_day.keys()):
            m = _bucket_metrics(by_day[day])
            trading_days.append({"trading_day": day, **m})

        kpi = {
            "target_hit_rate": hit_rate,
            "direction_accuracy": dir_acc,
            "mae": mae,
            "premium_rmse": prem_rmse,
        }
        out = {
            "available": True,
            "error": None,
            "cached": False,
            "source_fingerprint": fingerprint,
            "total_predictions": n,
            "kpi": kpi,
            "quality": quality,
            "risk": risk,
            "error_metrics": error_metrics,
            "distribution": distribution,
            "premium_bands": premium_bands,
            "trading_days": trading_days,
            "features": [],
            "confidence_filter": conf_mode,
            "confidence_filter_meta": conf_meta,
            "confidence_filter_effect": filter_effect,
            "evaluation_set": eval_set,
            "evaluation_set_label": EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
        }
        if day_filter:
            out["filter_trading_day"] = day_filter
        if conf_spec.get("active") and conf_meta.get("total_rows") is not None:
            conf_meta["rows_remaining"] = n
            conf_meta["rows_filtered"] = int(conf_meta["total_rows"]) - n
            out["confidence_filter_meta"] = conf_meta
        return out
    except Exception as exc:
        return {**empty, "error": str(exc)}


def research_dashboard_cache_is_fresh(db_path: str) -> bool:
    """True when summary tables match current prediction row fingerprint."""
    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            n, fingerprint = store.research_dashboard_source_fingerprint()
            meta = store.read_research_dashboard_meta()
            if meta is None:
                return n == 0
            if int(meta.get("schema_version") or 0) != DASHBOARD_CACHE_SCHEMA_VERSION:
                return False
            if n == 0:
                return int(meta.get("source_row_count") or 0) == 0
            return str(meta.get("source_fingerprint") or "") == fingerprint
    except Exception:
        return False


def refresh_research_dashboard_cache(
    db_path: str,
    *,
    force: bool = False,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Recompute dashboard statistics and write summary tables.

    Skips work when the cache is already fresh unless *force* is True.
    Call this after prediction rows are added/updated.
    """
    if not force and research_dashboard_cache_is_fresh(db_path):
        with ModelLabStore(db_path) as store:
            cached = store.read_research_dashboard_cache()
            if cached is not None:
                return {**cached, "rebuilt": False}
            # Empty dataset, no meta yet
            n, fingerprint = store.research_dashboard_source_fingerprint()
            if n == 0:
                return {
                    "available": False,
                    "error": None,
                    "cached": True,
                    "rebuilt": False,
                    "total_predictions": 0,
                    "kpi": {},
                    "quality": {},
                    "risk": {},
                    "error_metrics": {},
                    "distribution": {},
                    "premium_bands": [],
                    "trading_days": [],
                    "features": [],
                    "source_fingerprint": fingerprint,
                }

    dash = _compute_research_dashboard_fresh(db_path, data_dir=data_dir)
    if dash.get("error"):
        return {**dash, "rebuilt": False}

    computed_at = datetime.now(timezone.utc).isoformat()
    with ModelLabStore(db_path) as store:
        n, fingerprint = store.research_dashboard_source_fingerprint()
        if n <= 0:
            store.clear_research_dashboard_cache()
            return {
                "available": False,
                "error": None,
                "cached": True,
                "rebuilt": True,
                "computed_at": computed_at,
                "total_predictions": 0,
                "kpi": {},
                "quality": {},
                "risk": {},
                "error_metrics": {},
                "distribution": {},
                "premium_bands": [],
                "trading_days": [],
                "features": [],
                "source_fingerprint": fingerprint,
            }
        store.write_research_dashboard_cache(
            schema_version=DASHBOARD_CACHE_SCHEMA_VERSION,
            source_row_count=n,
            source_fingerprint=fingerprint,
            computed_at=computed_at,
            kpi=dash.get("kpi") or {},
            quality=dash.get("quality") or {},
            risk=dash.get("risk") or {},
            error_metrics=dash.get("error_metrics") or {},
            distribution=dash.get("distribution") or {},
            premium_bands=list(dash.get("premium_bands") or []),
            trading_days=list(dash.get("trading_days") or []),
        )
        cached = store.read_research_dashboard_cache() or {}
    out = {**cached, "rebuilt": True, "computed_at": computed_at}
    return out


def compute_research_dashboard_for_day(
    db_path: str,
    trading_day: str,
    *,
    data_dir: str | None = None,
    confidence_filter: str | None = None,
    confidence_classifier: str | None = None,
    confidence_prediction: int | str | None = None,
    evaluation_set: str | None = None,
) -> dict[str, Any]:
    """Research Dashboard scoped to one trading day (always computed live)."""
    day = str(trading_day or "").strip()
    if not day:
        return {
            "available": False,
            "error": "trading_day is required",
            "cached": False,
            "total_predictions": 0,
            "kpi": {},
            "quality": {},
            "risk": {},
            "error_metrics": {},
            "distribution": {},
            "premium_bands": [],
            "trading_days": [],
            "features": [],
            "confidence_filter": "disabled",
            "evaluation_set": normalize_evaluation_set(evaluation_set),
        }
    return _compute_research_dashboard_fresh(
        db_path,
        data_dir=data_dir,
        trading_day=day,
        confidence_filter=confidence_filter,
        confidence_classifier=confidence_classifier,
        confidence_prediction=confidence_prediction,
        evaluation_set=evaluation_set,
    )


def compute_research_dashboard(
    db_path: str,
    *,
    force_recompute: bool = False,
    data_dir: str | None = None,
    trading_day: str | None = None,
    confidence_filter: str | None = None,
    confidence_classifier: str | None = None,
    confidence_prediction: int | str | None = None,
    evaluation_set: str | None = None,
) -> dict[str, Any]:
    """
    Load Research Dashboard from summary tables (instant).

    Rebuilds cache only when prediction rows changed (or *force_recompute*).

    When *trading_day*, Confidence Filter, or Evaluation Set ≠ all is set,
    returns live stats (no unfiltered cache) so SQL WHERE filters apply.
    """
    day_filter = str(trading_day or "").strip()
    conf_spec = resolve_confidence_filter_spec(
        confidence_filter,
        confidence_classifier=confidence_classifier,
        confidence_prediction=confidence_prediction,
    )
    conf = str(conf_spec.get("mode") or "disabled")
    if conf in ("", "disabled"):
        conf = ""
    eval_set = normalize_evaluation_set(evaluation_set)
    live = bool(day_filter or conf or eval_set != EVAL_SET_ALL)
    if live:
        return _compute_research_dashboard_fresh(
            db_path,
            data_dir=data_dir,
            trading_day=day_filter or None,
            confidence_filter=confidence_filter,
            confidence_classifier=confidence_classifier,
            confidence_prediction=confidence_prediction,
            evaluation_set=eval_set,
        )
    if force_recompute or not research_dashboard_cache_is_fresh(db_path):
        return refresh_research_dashboard_cache(
            db_path, force=force_recompute, data_dir=data_dir
        )
    try:
        with ModelLabStore(db_path) as store:
            cached = store.read_research_dashboard_cache()
            if cached is not None:
                return {
                    **cached,
                    "rebuilt": False,
                    "confidence_filter": "disabled",
                    "evaluation_set": EVAL_SET_ALL,
                    "evaluation_set_label": EVAL_SET_LABELS[EVAL_SET_ALL],
                }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "cached": False,
            "total_predictions": 0,
            "kpi": {},
            "quality": {},
            "risk": {},
            "error_metrics": {},
            "distribution": {},
            "premium_bands": [],
            "trading_days": [],
            "features": [],
            "confidence_filter": "disabled",
            "evaluation_set": EVAL_SET_ALL,
        }
    return refresh_research_dashboard_cache(db_path, force=True, data_dir=data_dir)


def compute_overall_statistics(db_path: str) -> dict[str, Any]:
    """Backward-compatible flat overall stats (maps into quality/risk/error)."""
    dash = compute_research_dashboard(db_path)
    if dash.get("error"):
        return {
            "total_predictions": 0,
            "available": False,
            "error": dash.get("error"),
        }
    q = dash.get("quality") or {}
    r = dash.get("risk") or {}
    e = dash.get("error_metrics") or {}
    return {
        "total_predictions": int(dash.get("total_predictions") or 0),
        "direction_accuracy": q.get("direction_accuracy"),
        "target_hit_rate": q.get("target_hit_rate"),
        "target_miss_rate": q.get("target_miss_rate"),
        "average_time_to_target": q.get("average_time_to_target"),
        "median_time_to_target": q.get("median_time_to_target"),
        "p95_time_to_target": q.get("p95_time_to_target"),
        "average_dd_before_target": r.get("average_dd_before_target"),
        "median_dd_before_target": r.get("median_dd_before_target"),
        "p95_dd_before_target": r.get("p95_dd_before_target"),
        "average_max_profit": r.get("average_max_profit"),
        "average_max_dd": r.get("average_max_dd"),
        "mae": e.get("mae"),
        "rmse": e.get("rmse"),
        "premium_mae": e.get("premium_mae"),
        "premium_rmse": e.get("premium_rmse"),
        "mean_prediction_error": e.get("mean_prediction_error"),
        "median_absolute_error": e.get("median_absolute_error"),
        "available": bool(dash.get("available")),
        "error": None,
        "cached": bool(dash.get("cached")),
        "computed_at": dash.get("computed_at"),
    }


QUALITY_ROWS: tuple[tuple[str, str], ...] = (
    ("total_predictions", "Total Predictions"),
    ("direction_accuracy", "Direction Accuracy"),
    ("target_hit_rate", "Path Touch Rate"),
    ("target_miss_rate", "Path Miss Rate"),
    ("average_time_to_target", "Average Time to Target (s)"),
    ("median_time_to_target", "Median Time to Target (s)"),
    ("p95_time_to_target", "95th Percentile Time to Target (s)"),
)

RISK_ROWS: tuple[tuple[str, str], ...] = (
    ("average_dd_before_target", "Average DD Before Target"),
    ("median_dd_before_target", "Median DD Before Target"),
    ("p95_dd_before_target", "95th Percentile DD Before Target"),
    ("average_max_dd", "Average Max DD"),
    ("average_max_profit", "Average Max Profit"),
)

ERROR_ROWS: tuple[tuple[str, str], ...] = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("premium_mae", "Premium MAE"),
    ("premium_rmse", "Premium RMSE"),
    ("mean_prediction_error", "Mean Prediction Error (signed)"),
    ("median_absolute_error", "Median Absolute Error"),
)

DISTRIBUTION_ROWS: tuple[tuple[str, str], ...] = (
    ("predicted_up_rate", "Predicted Up"),
    ("predicted_down_rate", "Predicted Down"),
    ("actual_up_rate", "Actual Up"),
    ("actual_down_rate", "Actual Down"),
    ("target_hits", "Target Hits"),
    ("target_misses", "Target Misses"),
)

OVERALL_STAT_ROWS = QUALITY_ROWS + RISK_ROWS + ERROR_ROWS


def _sql_comparison_metrics(
    store: ModelLabStore,
    *,
    where_sql: str = "",
    where_args: list[Any] | None = None,
) -> dict[str, Any]:
    """
    SQL aggregates for Confidence Filter Comparison rows.

    Avg DD uses ``maximum_drawdown`` (path risk), not dd_before_target.
    """
    cols = set(store._prediction_table_columns())
    empty = {
        "rows": 0,
        "hit_rate": None,
        "avg_dd": None,
        "avg_max_profit": None,
        "profit_dd": None,
        "mae": None,
        "premium_rmse": None,
    }
    if "target_reached" not in cols:
        return empty

    avg_dd_expr = (
        'AVG("maximum_drawdown")' if "maximum_drawdown" in cols else "NULL"
    )
    avg_mfe_expr = (
        'AVG("maximum_profit")' if "maximum_profit" in cols else "NULL"
    )
    mae_expr = 'AVG("absolute_error")' if "absolute_error" in cols else "NULL"
    if "premium_error_pct" in cols:
        prem_rmse_expr = (
            'SQRT(AVG(CASE WHEN "premium_error_pct" IS NOT NULL '
            'THEN "premium_error_pct" * "premium_error_pct" END))'
        )
    else:
        prem_rmse_expr = "NULL"

    sql = f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN "target_reached" IS NOT NULL THEN 1 ELSE 0 END) AS n_hit,
            SUM(CASE WHEN "target_reached" = 1 THEN 1 ELSE 0 END) AS hits,
            {avg_dd_expr} AS avg_dd,
            {avg_mfe_expr} AS avg_mfe,
            {mae_expr} AS mae,
            {prem_rmse_expr} AS prem_rmse
        FROM prediction_dataset
        {where_sql}
    """
    row = store.conn.execute(sql, list(where_args or [])).fetchone()
    if not row:
        return empty
    n = int(row[0] or 0)
    n_hit = float(row[1] or 0)
    hits = float(row[2] or 0)
    avg_dd = float(row[3]) if row[3] is not None else None
    avg_mfe = float(row[4]) if row[4] is not None else None
    profit_dd = None
    if avg_dd is not None and avg_mfe is not None and abs(avg_dd) > 1e-12:
        profit_dd = avg_mfe / avg_dd
    return {
        "rows": n,
        "hit_rate": _rate(hits, n_hit),
        "avg_dd": avg_dd,
        "avg_max_profit": avg_mfe,
        "profit_dd": profit_dd,
        "mae": float(row[5]) if row[5] is not None else None,
        "premium_rmse": float(row[6]) if row[6] is not None else None,
    }


def list_comparable_confidence_models(
    store: ModelLabStore,
) -> list[dict[str, Any]]:
    """
    Confidence classifiers that have inference columns with at least one scored row.

    Enumerated from CONFIDENCE_TARGETS — not hard-coded in the UI.
    """
    from .confidence_inference import INFERENCE_COLUMNS
    from .confidence_manifest import CONFIDENCE_TARGETS

    cols = set(store._prediction_table_columns())
    out: list[dict[str, Any]] = []
    for spec in CONFIDENCE_TARGETS:
        key = spec["key"]
        icols = INFERENCE_COLUMNS.get(key) or {}
        pred_col = icols.get("pred")
        if not pred_col or pred_col not in cols:
            continue
        scored = store.conn.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                SUM(CASE WHEN "{pred_col}" = 1 THEN 1 ELSE 0 END) AS pos
            FROM prediction_dataset
            WHERE "{pred_col}" IS NOT NULL
            """
        ).fetchone()
        n_scored = int((scored[0] if scored else 0) or 0)
        if n_scored <= 0:
            continue
        out.append(
            {
                "model_key": key,
                "label": spec["label"],
                "pred_col": pred_col,
                "scored_rows": n_scored,
                "positive_rows": int((scored[1] if scored else 0) or 0),
            }
        )
    return out


def compute_confidence_filter_comparison(
    db_path: str,
    *,
    evaluation_set: str | None = None,
    trading_day: str | None = None,
) -> dict[str, Any]:
    """
    Side-by-side trading impact of every inferred confidence classifier.

    All rows share the same Evaluation Set. Baseline (``None``) has no
    confidence filter; each model row uses ``confidence_*_pred = 1``.
    SQL aggregates only — never runs XGBoost.
    """
    eval_set = normalize_evaluation_set(evaluation_set)
    day_filter = str(trading_day or "").strip()
    empty = {
        "ok": True,
        "available": False,
        "evaluation_set": eval_set,
        "evaluation_set_label": EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
        "baseline_rows": 0,
        "rows": [],
        "models_compared": 0,
    }
    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = set(store._prediction_table_columns())
            if "prediction_dataset" not in {
                str(r[0])
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }:
                return {**empty, "ok": False, "error": "No prediction_dataset table."}

            base_parts: list[str] = []
            base_args: list[Any] = []
            if day_filter and "trading_day" in cols:
                base_parts.append('"trading_day" = ?')
                base_args.append(day_filter)
            eval_parts, eval_args, eval_meta = _evaluation_set_day_clause(
                store, eval_set, cols=cols
            )
            base_parts.extend(eval_parts)
            base_args.extend(eval_args)
            base_where = (" WHERE " + " AND ".join(base_parts)) if base_parts else ""

            baseline = _sql_comparison_metrics(
                store, where_sql=base_where, where_args=base_args
            )
            n0 = int(baseline.get("rows") or 0)

            def _row_payload(
                *,
                filter_key: str,
                label: str,
                metrics: dict[str, Any],
                is_baseline: bool = False,
            ) -> dict[str, Any]:
                n1 = int(metrics.get("rows") or 0)
                removed = max(n0 - n1, 0) if not is_baseline else 0
                hit = metrics.get("hit_rate")
                avg_dd = metrics.get("avg_dd")
                avg_mfe = metrics.get("avg_max_profit")
                profit_dd = metrics.get("profit_dd")
                payload: dict[str, Any] = {
                    "filter_key": filter_key,
                    "label": label,
                    "is_baseline": is_baseline,
                    "rows_left": n1,
                    "rows_removed": removed,
                    "rows_removed_pct": (100.0 * removed / n0) if n0 > 0 and not is_baseline else (
                        0.0 if is_baseline else None
                    ),
                    "hit_rate": hit,
                    "avg_dd": avg_dd,
                    "avg_max_profit": avg_mfe,
                    "profit_dd": profit_dd,
                    "mae": metrics.get("mae"),
                    "premium_rmse": metrics.get("premium_rmse"),
                    "delta_hit_pp": None,
                    "delta_avg_dd_pct": None,
                    "delta_avg_max_profit_pct": None,
                    "delta_profit_dd_pct": None,
                    "improved_hit": None,
                    "improved_dd": None,
                    "improved_profit": None,
                    "improved_profit_dd": None,
                }
                if not is_baseline:
                    # Hit %: higher is better (pp change)
                    d_hit = _pp_change(baseline.get("hit_rate"), hit)
                    payload["delta_hit_pp"] = d_hit
                    payload["improved_hit"] = (
                        None if d_hit is None else bool(d_hit > 0)
                    )
                    # Avg DD: lower is better
                    d_dd = _pct_change(baseline.get("avg_dd"), avg_dd)
                    payload["delta_avg_dd_pct"] = d_dd
                    payload["improved_dd"] = (
                        None if d_dd is None else bool(d_dd < 0)
                    )
                    # Avg Max Profit: higher is better
                    d_mfe = _pct_change(baseline.get("avg_max_profit"), avg_mfe)
                    payload["delta_avg_max_profit_pct"] = d_mfe
                    payload["improved_profit"] = (
                        None if d_mfe is None else bool(d_mfe > 0)
                    )
                    # Profit/DD: higher is better
                    d_pd = _pct_change(baseline.get("profit_dd"), profit_dd)
                    payload["delta_profit_dd_pct"] = d_pd
                    payload["improved_profit_dd"] = (
                        None if d_pd is None else bool(d_pd > 0)
                    )
                return payload

            rows: list[dict[str, Any]] = [
                _row_payload(
                    filter_key="none",
                    label="None",
                    metrics=baseline,
                    is_baseline=True,
                )
            ]
            models = list_comparable_confidence_models(store)
            for model in models:
                filt_parts = list(base_parts) + [f'"{model["pred_col"]}" = ?']
                filt_args = list(base_args) + [1]
                filt_where = " WHERE " + " AND ".join(filt_parts)
                metrics = _sql_comparison_metrics(
                    store, where_sql=filt_where, where_args=filt_args
                )
                rows.append(
                    _row_payload(
                        filter_key=str(model["model_key"]),
                        label=str(model["label"]),
                        metrics=metrics,
                    )
                )

            return {
                "ok": True,
                "available": n0 > 0,
                "evaluation_set": eval_set,
                "evaluation_set_label": eval_meta.get("evaluation_set_label")
                or EVAL_SET_LABELS.get(eval_set, "Seen + Unseen"),
                "filter_trading_day": day_filter or None,
                "baseline_rows": n0,
                "rows": rows,
                "models_compared": len(models),
                "models": models,
            }
    except Exception as exc:
        return {**empty, "ok": False, "error": str(exc)}
