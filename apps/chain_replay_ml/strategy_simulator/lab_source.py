"""Map Research Lab prediction_dataset rows → strategy engine row shape."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.model_lab.confidence_inference import INFERENCE_COLUMNS
from chain_replay_ml.model_lab.confidence_manifest import CONFIDENCE_TARGETS, TARGET_BY_KEY
from chain_replay_ml.training.prediction_packages import PROBABILITY_OUTPUT_COLUMNS

_DISABLED = "disabled"

# UI label → model_key (keep rows where classifier pred == 1)
CLASSIFIER_FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Disabled", _DISABLED),
    *[(str(t["label"]), str(t["key"])) for t in CONFIDENCE_TARGETS],
)

_LAB_BASE_COLUMNS = [
    "id",
    "prediction_id",
    "trading_day",
    "timestamp",
    "token",
    "strike",
    "option_type",
    "current_spot",
    "current_ltp",
    "predicted_future_ltp",
    "actual_future_ltp",
]

_LAB_CONFIDENCE_PRED_COLUMNS = [cols["pred"] for cols in INFERENCE_COLUMNS.values()]

# Triple Barrier side-scorer columns — read-only, never inferred here.
_LAB_TB_COLUMNS = ["tb_model_name", "tb_pred_probability", "tb_pred_class"]


def classifier_filter_labels() -> list[str]:
    return [label for label, _key in CLASSIFIER_FILTER_OPTIONS]


def classifier_key_from_label(label: str) -> str:
    text = str(label or "").strip()
    for lbl, key in CLASSIFIER_FILTER_OPTIONS:
        if lbl == text:
            return key
    # Allow raw model keys too.
    if text in INFERENCE_COLUMNS or text == _DISABLED:
        return text
    return _DISABLED


def classifier_label_from_key(model_key: str | None) -> str:
    key = str(model_key or _DISABLED).strip().lower()
    if key in ("", _DISABLED, "none", "off"):
        return "Disabled"
    return str((TARGET_BY_KEY.get(key) or {}).get("label") or key)


def lab_row_to_engine_row(row: dict[str, Any], *, confidence_pred_col: str | None = None) -> dict[str, Any]:
    """Convert a prediction_dataset row into simulate_prediction_rows input."""
    conf_col = confidence_pred_col or "confidence_target_hit_pred"
    conf = row.get(conf_col)
    if conf is None and conf_col != "confidence_target_hit_pred":
        conf = row.get("confidence_target_hit_pred")
    try:
        confidence = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        confidence = None
    out = {
        "prediction_id": row.get("prediction_id"),
        "timestamp": row.get("timestamp"),
        "trading_day": row.get("trading_day"),
        "token": row.get("token"),
        "strike": row.get("strike"),
        "option_type": row.get("option_type"),
        "spot": row.get("current_spot"),
        "ltp": row.get("current_ltp"),
        "predicted_ltp": row.get("predicted_future_ltp"),
        "actual_ltp": row.get("actual_future_ltp"),
        "fold_id": "",
        "row_index": int(row.get("id") or 0),
        "confidence": confidence,
    }
    # Preserve raw classifier + package probability + Triple Barrier columns
    # for post-load filtering.
    for col in (*_LAB_CONFIDENCE_PRED_COLUMNS, *PROBABILITY_OUTPUT_COLUMNS, *_LAB_TB_COLUMNS):
        if col in row:
            out[col] = row.get(col)
    return out


def _day_filter_sql(
    *,
    trading_days: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> tuple[str, list[Any]]:
    if trading_days:
        days = [str(d).strip() for d in trading_days if str(d).strip()]
        if not days:
            return "", []
        placeholders = ", ".join("?" for _ in days)
        return f"trading_day IN ({placeholders})", days
    clauses: list[str] = []
    args: list[Any] = []
    if date_from:
        clauses.append("trading_day >= ?")
        args.append(str(date_from).strip())
    if date_to:
        clauses.append("trading_day <= ?")
        args.append(str(date_to).strip())
    return (" AND ".join(clauses), args) if clauses else ("", [])


def load_lab_prediction_rows_for_simulation(
    lab_db_path: str,
    *,
    trading_days: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    chunk_size: int = 50_000,
    confidence_classifier: str | None = None,
) -> list[dict[str, Any]]:
    """Bulk-load lab prediction rows mapped for the strategy engine."""
    from chain_replay_ml.model_lab.store import ModelLabStore

    where_sql, where_args = _day_filter_sql(
        trading_days=trading_days,
        date_from=date_from,
        date_to=date_to,
    )
    key = str(confidence_classifier or _DISABLED).strip().lower()
    pred_col = None
    if key not in ("", _DISABLED, "none", "off"):
        cols = INFERENCE_COLUMNS.get(key)
        if not cols:
            raise ValueError(f"Unknown classifier: {confidence_classifier}")
        pred_col = cols["pred"]

    columns = list(_LAB_BASE_COLUMNS)
    for c in (*_LAB_CONFIDENCE_PRED_COLUMNS, *PROBABILITY_OUTPUT_COLUMNS, *_LAB_TB_COLUMNS):
        if c not in columns:
            columns.append(c)

    out: list[dict[str, Any]] = []
    offset = 0
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        while True:
            cols, rows = store.query_predictions(
                columns=columns,
                where_sql=where_sql,
                where_args=where_args,
                order_by="timestamp ASC",
                limit=chunk_size,
                offset=offset,
            )
            if not rows:
                break
            for tup in rows:
                raw = {cols[i]: tup[i] for i in range(len(cols))}
                out.append(lab_row_to_engine_row(raw, confidence_pred_col=pred_col))
            if len(rows) < chunk_size:
                break
            offset += chunk_size
    return out


def apply_classifier_filter(
    rows: list[dict[str, Any]],
    *,
    confidence_classifier: str | None,
    keep_value: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Keep rows where the selected classifier prediction equals *keep_value*.

    Pipeline stage: Prediction Dataset → Classifier Filter → Strategy Rules.
    """
    key = str(confidence_classifier or _DISABLED).strip().lower()
    meta: dict[str, Any] = {
        "active": False,
        "model_key": None,
        "label": "Disabled",
        "keep_value": None,
        "pred_col": None,
        "rows_before": len(rows),
        "rows_after": len(rows),
        "rows_removed": 0,
        "rows_null": 0,
    }
    if key in ("", _DISABLED, "none", "off"):
        return rows, meta

    cols = INFERENCE_COLUMNS.get(key)
    if not cols:
        raise ValueError(f"Unknown classifier: {confidence_classifier}")
    pred_col = cols["pred"]
    label = classifier_label_from_key(key)
    meta.update({
        "active": True,
        "model_key": key,
        "label": label,
        "keep_value": int(keep_value),
        "pred_col": pred_col,
    })

    kept: list[dict[str, Any]] = []
    nulls = 0
    for row in rows:
        raw = row.get(pred_col)
        if raw is None:
            nulls += 1
            continue
        try:
            val = int(float(raw))
        except (TypeError, ValueError):
            nulls += 1
            continue
        if val == int(keep_value):
            kept.append(row)

    meta["rows_null"] = nulls
    meta["rows_after"] = len(kept)
    meta["rows_removed"] = max(0, len(rows) - len(kept))

    if nulls == len(rows):
        raise ValueError(
            f"Classifier '{label}' has no inferred predictions on this dataset. "
            "Run Confidence Model inference first, then retry."
        )
    if not kept:
        raise ValueError(
            f"Classifier '{label}' (keep={keep_value}) removed all "
            f"{len(rows):,} date-filtered predictions."
        )
    return kept, meta


def classifier_row_summary(
    lab_db_path: str,
    *,
    confidence_classifier: str | None,
    trading_days: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keep_value: int = 1,
) -> dict[str, Any]:
    """Fast SQL summary of how a classifier would filter the Prediction Dataset."""
    from chain_replay_ml.model_lab.store import ModelLabStore

    key = str(confidence_classifier or _DISABLED).strip().lower()
    where_sql, where_args = _day_filter_sql(
        trading_days=trading_days,
        date_from=date_from,
        date_to=date_to,
    )
    clause = f"WHERE {where_sql}" if where_sql else ""

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        total = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset {clause}",
                where_args,
            ).fetchone()[0]
            or 0
        )
        if key in ("", _DISABLED, "none", "off"):
            return {
                "ok": True,
                "active": False,
                "label": "Disabled",
                "model_key": None,
                "pred_col": None,
                "prediction_rows": total,
                "rows_kept": total,
                "rows_removed": 0,
                "rows_kept_pct": 100.0 if total else 0.0,
                "rows_removed_pct": 0.0,
                "rows_null": 0,
            }

        cols = INFERENCE_COLUMNS.get(key)
        if not cols:
            return {"ok": False, "error": f"Unknown classifier: {confidence_classifier}"}
        pred_col = cols["pred"]
        table_cols = set(store._prediction_table_columns())
        if pred_col not in table_cols:
            return {
                "ok": False,
                "error": f"Column {pred_col} missing — run inference first.",
                "label": classifier_label_from_key(key),
                "pred_col": pred_col,
                "prediction_rows": total,
            }

        where_keep = f'"{pred_col}" = ?'
        where_null = f'"{pred_col}" IS NULL'
        if where_sql:
            where_keep = f"({where_sql}) AND {where_keep}"
            where_null = f"({where_sql}) AND {where_null}"
        kept = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset WHERE {where_keep}",
                list(where_args) + [int(keep_value)],
            ).fetchone()[0]
            or 0
        )
        nulls = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset WHERE {where_null}",
                where_args,
            ).fetchone()[0]
            or 0
        )
        removed = max(0, total - kept)
        return {
            "ok": True,
            "active": True,
            "label": classifier_label_from_key(key),
            "model_key": key,
            "pred_col": pred_col,
            "prediction_rows": total,
            "rows_kept": kept,
            "rows_removed": removed,
            "rows_null": nulls,
            "rows_kept_pct": round(100.0 * kept / total, 2) if total else 0.0,
            "rows_removed_pct": round(100.0 * removed / total, 2) if total else 0.0,
        }
