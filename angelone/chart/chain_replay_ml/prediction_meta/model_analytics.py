"""Post-build model analytics — leaderboard, agreement/spread buckets, timeline."""

from __future__ import annotations

import math
import os
import re
import sqlite3
from typing import Any

from chain_replay_ml.model_lab.prediction_schema import (
    actual_ltp_column_from_target,
    horizon_label_from_target,
)

from .model_registry import read_model_registry, slot_pred_column
from .store import PredictionMetaStore


def resolve_evaluation_target_column(
    *,
    catalog: list[dict[str, Any]],
    target_column: str | None = None,
    project_config: dict[str, Any] | None = None,
) -> str:
    """Authoritative regression target — never defaults to 5m.

    Prefer explicit ``target_column``, then project config, then unique model
    registry ``target`` values.
    """
    explicit = str(target_column or "").strip()
    if explicit:
        return explicit

    cfg = project_config if isinstance(project_config, dict) else {}
    for key in ("target_column", "target"):
        t = str(cfg.get(key) or "").strip()
        if t:
            return t

    targets: list[str] = []
    for row in catalog or []:
        t = str(row.get("target") or row.get("target_column") or "").strip()
        if t:
            targets.append(t)
    unique = list(dict.fromkeys(targets))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(
            "Ambiguous model targets for analytics — expected one target_column, "
            f"got {unique}"
        )
    raise ValueError(
        "Cannot resolve analytics target_column from models or project config"
    )


def _direction_expr(pred_col: str, actual_col: str) -> str:
    """SQL Direction Accuracy matching evaluator (exclude flat actual moves)."""
    return (
        f"CASE WHEN ({actual_col} - current_ltp) = 0 THEN NULL "
        f"WHEN ({pred_col} - current_ltp) * ({actual_col} - current_ltp) > 0 THEN 1.0 "
        f"ELSE 0.0 END"
    )


def _model_slot_index(slot: str) -> int:
    m = re.match(r"model_(\d+)$", str(slot or ""))
    return int(m.group(1)) if m else 0


def _model_label(slot: str, model_name: str | None) -> str:
    idx = _model_slot_index(slot)
    if model_name:
        short = str(model_name).replace("Future_LTP_5m_WF_", "").replace("_", " ")
        return f"Model {idx} ({short})"
    return f"Model {idx}"


def _round(v: float | None, nd: int = 2) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return round(float(v), nd)


def _read_model_catalog(conn: sqlite3.Connection, data_dir: str) -> list[dict[str, Any]]:
    rows = read_model_registry(conn)
    if rows:
        return sorted(rows, key=lambda r: _model_slot_index(str(r.get("slot") or "")))
    return []


def _eval_where(actual_col: str, scope_sql: str = "") -> str:
    base = (
        f"{actual_col} IS NOT NULL AND current_ltp IS NOT NULL "
        f"AND current_ltp > 0"
    )
    scope = str(scope_sql or "").strip()
    if scope:
        return f"{base} AND ({scope})"
    return base


def _signed_error_metrics_sql(pred_col: str, actual_col: str, prefix: str) -> list[str]:
    err = f"({pred_col} - {actual_col})"
    return [
        f"AVG(CASE WHEN {pred_col} > {actual_col} THEN {err} ELSE NULL END) AS {prefix}_mpe",
        f"AVG(CASE WHEN {pred_col} < {actual_col} THEN {err} ELSE NULL END) AS {prefix}_mne",
        f"SUM(CASE WHEN {pred_col} > {actual_col} THEN 1.0 ELSE 0.0 END) * 1.0 / COUNT({pred_col}) AS {prefix}_over",
        f"SUM(CASE WHEN {pred_col} < {actual_col} THEN 1.0 ELSE 0.0 END) * 1.0 / COUNT({pred_col}) AS {prefix}_under",
        f"AVG({err}) AS {prefix}_bias",
    ]


def compute_model_leaderboard(
    conn: sqlite3.Connection,
    *,
    model_slots: list[dict[str, Any]],
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    if not model_slots:
        return []

    selects: list[str] = []
    for row in model_slots:
        slot = str(row.get("slot") or "")
        idx = _model_slot_index(slot)
        if idx <= 0:
            continue
        pred = slot_pred_column(idx)
        prefix = f"m{idx}"
        selects.extend([
            f"AVG(ABS({pred} - {actual_col})) AS {prefix}_mae",
            f"AVG(({pred} - {actual_col}) * ({pred} - {actual_col})) AS {prefix}_se",
            *_signed_error_metrics_sql(pred, actual_col, prefix),
            f"AVG({_direction_expr(pred, actual_col)}) AS {prefix}_dir",
            f"COUNT({pred}) AS {prefix}_n",
        ])

    sql = f"SELECT {', '.join(selects)} FROM samples WHERE {_eval_where(actual_col, scope_sql)}"
    agg = conn.execute(sql).fetchone()
    if not agg:
        return []

    out: list[dict[str, Any]] = []
    col_i = 0
    for row in model_slots:
        slot = str(row.get("slot") or "")
        idx = _model_slot_index(slot)
        if idx <= 0:
            continue
        mae, se, mpe, mne, over_frac, under_frac, bias, dir_frac, n = agg[col_i:col_i + 9]
        col_i += 9
        if not n:
            continue
        rmse = math.sqrt(float(se)) if se is not None else None
        out.append({
            "rank": 0,
            "slot": slot,
            "model_index": idx,
            "model_name": row.get("model_name"),
            "label": _model_label(slot, row.get("model_name")),
            "mae": _round(mae),
            "rmse": _round(rmse),
            "direction_pct": _round(float(dir_frac) * 100.0 if dir_frac is not None else None, 1),
            "mean_positive_error": _round(mpe),
            "mean_negative_error": _round(mne),
            "over_prediction_pct": _round(float(over_frac) * 100.0 if over_frac is not None else None, 1),
            "under_prediction_pct": _round(float(under_frac) * 100.0 if under_frac is not None else None, 1),
            "bias": _round(bias),
            "rows": int(n),
        })

    out.sort(key=lambda r: (r.get("mae") if r.get("mae") is not None else 1e9, r.get("model_index", 0)))
    for i, row in enumerate(out, start=1):
        row["rank"] = i
        row["medal"] = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
    return out


def compute_ensemble_comparison(
    conn: sqlite3.Connection,
    *,
    leaderboard: list[dict[str, Any]],
    actual_col: str,
    scope_sql: str = "",
) -> dict[str, Any]:
    sql = f"""
        SELECT
            AVG(ABS(ensemble_mean - {actual_col})) AS mean_mae,
            AVG((ensemble_mean - {actual_col}) * (ensemble_mean - {actual_col})) AS mean_se,
            AVG(ABS(ensemble_median - {actual_col})) AS median_mae,
            AVG((ensemble_median - {actual_col}) * (ensemble_median - {actual_col})) AS median_se,
            COUNT(*) AS n
        FROM samples
        WHERE {_eval_where(actual_col, scope_sql)}
          AND ensemble_mean IS NOT NULL
          AND ensemble_median IS NOT NULL
    """
    row = conn.execute(sql).fetchone()
    items: list[dict[str, Any]] = []
    max_mae = 0.0

    for lb in leaderboard:
        mae = lb.get("mae")
        if mae is None:
            continue
        max_mae = max(max_mae, float(mae))
        items.append({
            "key": f"model_{lb['model_index']}",
            "label": f"Model {lb['model_index']}",
            "mae": mae,
            "kind": "model",
        })

    if row and row[4]:
        mean_mae = _round(row[0])
        median_mae = _round(row[2])
        if mean_mae is not None:
            max_mae = max(max_mae, float(mean_mae))
            items.append({"key": "ensemble_mean", "label": "Mean", "mae": mean_mae, "kind": "ensemble", "best": False})
        if median_mae is not None:
            max_mae = max(max_mae, float(median_mae))
            items.append({"key": "ensemble_median", "label": "Median", "mae": median_mae, "kind": "ensemble", "best": False})

    if items and max_mae > 0:
        best_mae = min(float(i["mae"]) for i in items if i.get("mae") is not None)
        for item in items:
            item["bar_pct"] = _round(float(item["mae"]) / max_mae * 100.0, 1)
            item["best"] = float(item["mae"]) == best_mae

    return {
        "items": items,
        "max_mae": _round(max_mae),
        "ensemble_beats_all": bool(
            items
            and any(i.get("kind") == "ensemble" and i.get("best") for i in items)
        ),
        "rows": int(row[4]) if row else 0,
    }


def compute_agreement_buckets(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT bucket, COUNT(*) AS rows,
               AVG(direction_correct) AS direction_frac,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(actual_max_profit_5m) AS avg_profit
        FROM (
            SELECT
                CASE
                    WHEN agreement >= 0.9 THEN '90-100%'
                    WHEN agreement >= 0.8 THEN '80-90%'
                    WHEN agreement >= 0.7 THEN '70-80%'
                    WHEN agreement >= 0.6 THEN '60-70%'
                    ELSE '<60%'
                END AS bucket,
                agreement,
                direction_correct,
                ensemble_mean,
                {actual_col},
                actual_max_profit_5m
            FROM samples
            WHERE agreement IS NOT NULL AND {_eval_where(actual_col, scope_sql)}
              AND ensemble_mean IS NOT NULL
        )
        GROUP BY bucket
    """
    order = ["90-100%", "80-90%", "70-80%", "60-70%", "<60%"]
    rows_by_bucket = {str(r[0]): r for r in conn.execute(sql).fetchall()}
    out: list[dict[str, Any]] = []
    for bucket in order:
        r = rows_by_bucket.get(bucket)
        if not r:
            continue
        out.append({
            "bucket": bucket,
            "rows": int(r[1]),
            "direction_pct": _round(float(r[2]) * 100.0 if r[2] is not None else None, 1),
            "avg_mae": _round(r[3]),
            "mae": _round(r[3]),
            "avg_profit": _round(r[4]),
        })
    return out


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}


def _deep_otm_expr(cols: set[str]) -> str | None:
    if not {"strike", "current_spot", "option_type"}.issubset(cols):
        return None
    return (
        "CASE WHEN option_type = 'CE' AND strike > current_spot * 1.015 THEN 1.0 "
        "WHEN option_type = 'PE' AND strike < current_spot * 0.985 THEN 1.0 "
        "ELSE 0.0 END"
    )


def _expiry_day_expr(cols: set[str]) -> str | None:
    if "expiry" not in cols:
        if "minutes_to_expiry" in cols:
            return "CASE WHEN minutes_to_expiry IS NOT NULL AND minutes_to_expiry <= 390 THEN 1.0 ELSE 0.0 END"
        return None
    return "CASE WHEN date(trading_day) = date(expiry) THEN 1.0 ELSE 0.0 END"


def _lunchtime_expr() -> str:
    return (
        "CASE WHEN ("
        "CAST(strftime('%H', timestamp, 'unixepoch', '+330 minutes') AS INTEGER) = 12 "
        "OR (CAST(strftime('%H', timestamp, 'unixepoch', '+330 minutes') AS INTEGER) = 13 "
        "AND CAST(strftime('%M', timestamp, 'unixepoch', '+330 minutes') AS INTEGER) < 30)"
        ") THEN 1.0 ELSE 0.0 END"
    )


_PREMIUM_BUCKET_ORDER = [
    "0-10", "10-20", "20-50", "50-100",
    "100-200", "200-300", "300-400", "400-500", "500+",
]


def _premium_bucket_case_sql() -> str:
    return """
        CASE
            WHEN current_ltp < 10 THEN '0-10'
            WHEN current_ltp < 20 THEN '10-20'
            WHEN current_ltp < 50 THEN '20-50'
            WHEN current_ltp < 100 THEN '50-100'
            WHEN current_ltp < 200 THEN '100-200'
            WHEN current_ltp < 300 THEN '200-300'
            WHEN current_ltp < 400 THEN '300-400'
            WHEN current_ltp < 500 THEN '400-500'
            ELSE '500+'
        END
    """


def _rows_by_premium_order(rows_by_bucket: dict[str, Any], order: list[str] | None = None) -> list[Any]:
    bucket_order = order or _PREMIUM_BUCKET_ORDER
    out: list[Any] = []
    for bucket in bucket_order:
        r = rows_by_bucket.get(bucket)
        if r:
            out.append(r)
    return out


def compute_premium_buckets(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    """Error breakdown by current option premium (LTP)."""
    sql = f"""
        SELECT bucket, COUNT(*) AS rows,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(ensemble_mean - {actual_col}) AS bias,
               AVG(direction_correct) AS direction_frac,
               AVG(current_ltp) AS avg_premium,
               AVG(ensemble_mean) AS avg_prediction,
               AVG({actual_col}) AS avg_actual,
               SUM(CASE WHEN ensemble_mean > {actual_col} THEN 1.0 ELSE 0.0 END) * 1.0 / COUNT(*) AS over_frac
        FROM (
            SELECT
                {_premium_bucket_case_sql()} AS bucket,
                current_ltp,
                ensemble_mean,
                {actual_col},
                direction_correct
            FROM samples
            WHERE {_eval_where(actual_col, scope_sql)} AND ensemble_mean IS NOT NULL
        )
        GROUP BY bucket
    """
    rows_by_bucket = {str(r[0]): r for r in conn.execute(sql).fetchall()}
    out: list[dict[str, Any]] = []
    for r in _rows_by_premium_order(rows_by_bucket):
        bucket = str(r[0])
        out.append({
            "bucket": bucket,
            "rows": int(r[1]),
            "mae": _round(r[2]),
            "bias": _round(r[3]),
            "direction_pct": _round(float(r[4]) * 100.0 if r[4] is not None else None, 1),
            "avg_premium": _round(r[5]),
            "avg_prediction": _round(r[6]),
            "avg_actual": _round(r[7]),
            "overestimate_pct": _round(float(r[8]) * 100.0 if r[8] is not None else None, 1),
        })
    return out


def compute_premium_error_context(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    """Per premium bucket: how often deep OTM / expiry day / lunch coincide with errors."""
    cols = _table_columns(conn)
    deep_otm = _deep_otm_expr(cols)
    expiry_day = _expiry_day_expr(cols)
    lunch = _lunchtime_expr()
    extra_selects = [
        f"AVG({lunch}) AS lunch_frac",
        "AVG(CASE WHEN current_ltp < 10 THEN 1.0 ELSE 0.0 END) AS premium_low_frac",
    ]
    if deep_otm:
        extra_selects.append(f"AVG({deep_otm}) AS deep_otm_frac")
    if expiry_day:
        extra_selects.append(f"AVG({expiry_day}) AS expiry_day_frac")

    sql = f"""
        SELECT bucket, COUNT(*) AS rows,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(ensemble_mean - {actual_col}) AS bias,
               {', '.join(extra_selects)}
        FROM (
            SELECT
                {_premium_bucket_case_sql()} AS bucket,
                current_ltp, ensemble_mean, {actual_col},
                strike, current_spot, option_type, trading_day, expiry,
                minutes_to_expiry, timestamp
            FROM samples
            WHERE {_eval_where(actual_col, scope_sql)} AND ensemble_mean IS NOT NULL
        )
        GROUP BY bucket
    """
    rows_by_bucket = {str(r[0]): r for r in conn.execute(sql).fetchall()}
    out: list[dict[str, Any]] = []
    for r in _rows_by_premium_order(rows_by_bucket):
        bucket = str(r[0])
        item: dict[str, Any] = {
            "bucket": bucket,
            "rows": int(r[1]),
            "mae": _round(r[2]),
            "bias": _round(r[3]),
            "lunchtime_pct": _round(float(r[4]) * 100.0 if r[4] is not None else None, 1),
            "premium_low_pct": _round(float(r[5]) * 100.0 if r[5] is not None else None, 1),
        }
        idx = 6
        if deep_otm:
            item["deep_otm_pct"] = _round(float(r[idx]) * 100.0 if r[idx] is not None else None, 1)
            idx += 1
        if expiry_day:
            item["expiry_day_pct"] = _round(float(r[idx]) * 100.0 if r[idx] is not None else None, 1)
        out.append(item)
    return out


def compute_calibration_bins(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    bin_width: float = 10.0,
    min_count: int = 20,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    """Binned ensemble_mean vs actual — calibration curve."""
    sql = f"""
        SELECT pred_bin,
               AVG(ensemble_mean) AS avg_pred,
               AVG({actual_col}) AS avg_actual,
               COUNT(*) AS rows,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(ensemble_mean - {actual_col}) AS bias
        FROM (
            SELECT ensemble_mean, {actual_col},
                   CAST(ensemble_mean / {bin_width} AS INTEGER) * {bin_width} AS pred_bin
            FROM samples
            WHERE {_eval_where(actual_col, scope_sql)} AND ensemble_mean IS NOT NULL AND ensemble_mean >= 0
        )
        GROUP BY pred_bin
        HAVING COUNT(*) >= ?
        ORDER BY pred_bin
    """
    out: list[dict[str, Any]] = []
    for r in conn.execute(sql, (min_count,)).fetchall():
        pred_lo = float(r[0] or 0)
        out.append({
            "pred_lo": _round(pred_lo),
            "pred_hi": _round(pred_lo + bin_width),
            "avg_pred": _round(r[1]),
            "avg_actual": _round(r[2]),
            "rows": int(r[3]),
            "mae": _round(r[4]),
            "bias": _round(r[5]),
        })
    return out


def compute_calibration_by_premium(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    """Avg prediction vs avg actual within each premium bucket."""
    return compute_premium_buckets(conn, actual_col=actual_col, scope_sql=scope_sql)


def compute_calibration_scatter(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    limit: int = 500,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT ensemble_mean, {actual_col}, current_ltp
        FROM samples
        WHERE {_eval_where(actual_col, scope_sql)} AND ensemble_mean IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """
    out: list[dict[str, Any]] = []
    for r in conn.execute(sql, (limit,)).fetchall():
        out.append({
            "pred": _round(r[0]),
            "actual": _round(r[1]),
            "premium": _round(r[2]),
        })
    return out


def diagnose_row_error(row: dict[str, Any], *, actual_col: str) -> dict[str, Any]:
    """Explain likely drivers for a single prediction row."""
    current_ltp = float(row.get("current_ltp") or 0)
    pred = row.get("ensemble_mean")
    actual = row.get(actual_col)
    bias = None
    if pred is not None and actual is not None:
        bias = _round(float(pred) - float(actual))

    ts = row.get("timestamp")
    lunch = False
    if ts is not None:
        import datetime as dt
        ist = dt.datetime.fromtimestamp(float(ts), dt.timezone.utc) + dt.timedelta(minutes=330)
        lunch = (ist.hour == 12) or (ist.hour == 13 and ist.minute < 30)

    strike = row.get("strike")
    spot = row.get("current_spot")
    opt = str(row.get("option_type") or "").upper()
    deep_otm = False
    if strike is not None and spot is not None and float(spot) > 0:
        strike_f = float(strike)
        spot_f = float(spot)
        if opt == "CE":
            deep_otm = strike_f > spot_f * 1.015
        elif opt == "PE":
            deep_otm = strike_f < spot_f * 0.985

    expiry_day = False
    expiry = row.get("expiry")
    trading_day = row.get("trading_day")
    if expiry and trading_day:
        expiry_day = str(expiry)[:10] == str(trading_day)[:10]
    elif row.get("minutes_to_expiry") is not None:
        expiry_day = float(row["minutes_to_expiry"]) <= 390

    flags: list[str] = []
    if current_ltp < 10:
        flags.append("premium < 10")
    if deep_otm:
        flags.append("deep OTM")
    if expiry_day:
        flags.append("expiry day")
    if lunch:
        flags.append("lunchtime")

    return {
        "current_ltp": _round(current_ltp),
        "ensemble_mean": _round(pred),
        "actual_ltp": _round(actual),
        actual_col: _round(actual),
        "actual_column": actual_col,
        "error": _round(abs(float(pred) - float(actual)) if pred is not None and actual is not None else None),
        "bias": bias,
        "overestimate": bool(pred is not None and actual is not None and float(pred) > float(actual)),
        "flags": flags,
        "premium_low": current_ltp < 10,
        "deep_otm": deep_otm,
        "expiry_day": expiry_day,
        "lunchtime": lunch,
    }


def fetch_high_error_sample(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> dict[str, Any] | None:
    row = conn.execute(
        f"""
        SELECT * FROM samples
        WHERE {_eval_where(actual_col, scope_sql)}
          AND ensemble_mean IS NOT NULL
          AND ABS(ensemble_mean - {actual_col}) > 0
        ORDER BY ABS(ensemble_mean - {actual_col}) DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(samples)").fetchall()]
    sample = dict(zip(cols, row))
    sample["error_diagnosis"] = diagnose_row_error(sample, actual_col=actual_col)
    return sample


def compute_spread_buckets(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT bucket, COUNT(*) AS rows,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(direction_correct) AS direction_frac,
               AVG(actual_max_profit_5m) AS avg_profit
        FROM (
            SELECT
                CASE
                    WHEN ensemble_spread < 2 THEN '0-2'
                    WHEN ensemble_spread < 5 THEN '2-5'
                    WHEN ensemble_spread < 10 THEN '5-10'
                    WHEN ensemble_spread < 20 THEN '10-20'
                    ELSE '20+'
                END AS bucket,
                ensemble_spread,
                ensemble_mean,
                {actual_col},
                direction_correct,
                actual_max_profit_5m
            FROM samples
            WHERE ensemble_spread IS NOT NULL AND {_eval_where(actual_col, scope_sql)}
        )
        GROUP BY bucket
    """
    order = ["0-2", "2-5", "5-10", "10-20", "20+"]
    rows_by_bucket = {str(r[0]): r for r in conn.execute(sql).fetchall()}
    out: list[dict[str, Any]] = []
    for bucket in order:
        r = rows_by_bucket.get(bucket)
        if not r:
            continue
        out.append({
            "bucket": bucket,
            "rows": int(r[1]),
            "mae": _round(r[2]),
            "direction_pct": _round(float(r[3]) * 100.0 if r[3] is not None else None, 1),
            "avg_profit": _round(r[4]),
        })
    return out


def compute_trading_day_timeline(
    conn: sqlite3.Connection,
    *,
    actual_col: str,
    limit: int = 60,
    scope_sql: str = "",
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT trading_day,
               COUNT(*) AS rows,
               AVG(ABS(ensemble_mean - {actual_col})) AS mae,
               AVG(direction_correct) AS direction_frac,
               AVG(agreement) AS avg_agreement,
               AVG(ensemble_spread) AS avg_spread,
               AVG(actual_max_profit_5m) AS avg_profit
        FROM samples
        WHERE {_eval_where(actual_col, scope_sql)} AND ensemble_mean IS NOT NULL
        GROUP BY trading_day
        ORDER BY trading_day
    """
    out: list[dict[str, Any]] = []
    for r in conn.execute(sql).fetchall():
        out.append({
            "trading_day": str(r[0]),
            "rows": int(r[1]),
            "mae": _round(r[2]),
            "direction_pct": _round(float(r[3]) * 100.0 if r[3] is not None else None, 1),
            "agreement_pct": _round(float(r[4]) * 100.0 if r[4] is not None else None, 1),
            "avg_spread": _round(r[5]),
            "avg_profit": _round(r[6]),
        })
    return out


def fetch_random_sample(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM samples ORDER BY RANDOM() LIMIT 1").fetchone()
    if not row:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(samples)").fetchall()]
    return dict(zip(cols, row))


def read_prediction_model_analytics(
    db_path: str,
    *,
    data_dir: str = "",
    target_column: str | None = None,
    project_config: dict[str, Any] | None = None,
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    if not os.path.isfile(db_path):
        return {"exists": False, "db_path": db_path}

    from .training_context import build_training_scope_sql, resolve_training_context

    training_context = resolve_training_context(data_dir, selected_models)
    with PredictionMetaStore(db_path) as store:
        conn = store.conn
        catalog = _read_model_catalog(conn, data_dir)
        try:
            target = resolve_evaluation_target_column(
                catalog=catalog,
                target_column=target_column,
                project_config=project_config,
            )
            actual_col = actual_ltp_column_from_target(target)
            actual_horizon = horizon_label_from_target(target)
        except ValueError as exc:
            return {
                "exists": True,
                "db_path": db_path,
                "error": str(exc),
                "row_count": store.row_count(),
                "leaderboard": [],
                "ensemble_comparison": {},
                "agreement_buckets": [],
                "premium_buckets": [],
                "spread_buckets": [],
                "timeline": [],
            }

        # Ensure the mapped actual column exists in this prediction DB.
        sample_cols = {d[1] for d in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if actual_col not in sample_cols:
            return {
                "exists": True,
                "db_path": db_path,
                "error": (
                    f"Actual column {actual_col!r} missing for target {target!r}. "
                    "Rebuild prediction meta with that horizon."
                ),
                "target_column": target,
                "actual_horizon": actual_horizon,
                "actual_column": actual_col,
                "row_count": store.row_count(),
                "leaderboard": [],
                "ensemble_comparison": {},
                "agreement_buckets": [],
                "premium_buckets": [],
                "spread_buckets": [],
                "timeline": [],
            }

        scope_sql, scope_info = build_training_scope_sql(
            conn, training_context.get("criteria") if training_context.get("resolved") else None
        )
        total = store.row_count()
        eval_count = conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE {_eval_where(actual_col)}"
        ).fetchone()[0]
        scoped_eval_count = conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE {_eval_where(actual_col, scope_sql)}"
        ).fetchone()[0]
        leaderboard = compute_model_leaderboard(
            conn, model_slots=catalog, actual_col=actual_col, scope_sql=scope_sql,
        )
        ensemble = compute_ensemble_comparison(
            conn, leaderboard=leaderboard, actual_col=actual_col, scope_sql=scope_sql,
        )
        agreement = compute_agreement_buckets(conn, actual_col=actual_col, scope_sql=scope_sql)
        premium = compute_premium_buckets(conn, actual_col=actual_col, scope_sql=scope_sql)
        premium_context = compute_premium_error_context(conn, actual_col=actual_col, scope_sql=scope_sql)
        calibration_bins = compute_calibration_bins(conn, actual_col=actual_col, scope_sql=scope_sql)
        calibration_by_premium = compute_calibration_by_premium(conn, actual_col=actual_col, scope_sql=scope_sql)
        calibration_scatter = compute_calibration_scatter(conn, actual_col=actual_col, scope_sql=scope_sql)
        spread = compute_spread_buckets(conn, actual_col=actual_col, scope_sql=scope_sql)
        timeline = compute_trading_day_timeline(conn, actual_col=actual_col, scope_sql=scope_sql)
        sample = fetch_random_sample(conn)
        high_error = fetch_high_error_sample(conn, actual_col=actual_col, scope_sql=scope_sql)
        if sample:
            sample = dict(sample)
            sample["error_diagnosis"] = diagnose_row_error(sample, actual_col=actual_col)
        if high_error:
            high_error = dict(high_error)

    return {
        "exists": True,
        "db_path": db_path,
        "target_column": target,
        "actual_horizon": actual_horizon,
        "actual_column": actual_col,
        "row_count": total,
        "eval_rows": int(scoped_eval_count if scope_info.get("active") else eval_count),
        "eval_rows_total": int(eval_count),
        "eval_rows_scoped": int(scoped_eval_count),
        "training_context": {
            **training_context,
            "scope": scope_info,
        },
        "leaderboard": leaderboard,
        "ensemble_comparison": ensemble,
        "agreement_buckets": agreement,
        "premium_buckets": premium,
        "premium_error_context": premium_context,
        "calibration_bins": calibration_bins,
        "calibration_by_premium": calibration_by_premium,
        "calibration_scatter": calibration_scatter,
        "spread_buckets": spread,
        "timeline": timeline,
        "sample_row": sample,
        "high_error_sample": high_error,
    }
