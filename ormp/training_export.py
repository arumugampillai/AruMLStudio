"""Slice an ORMP build into a labeled training frame (no rebuild)."""

from __future__ import annotations

import sqlite3
from typing import Any, Literal

import pandas as pd

from .feature_groups import FEATURE_GROUPS, all_grouped_features


LabelType = Literal["points", "percent", "direction"]
HORIZONS_MIN = (5, 10, 15)


def ormp_sample_columns(ormp_db_path: str) -> set[str]:
    """Column names present on ``ormp_samples`` for a built artifact."""
    with sqlite3.connect(ormp_db_path) as conn:
        rows = conn.execute("PRAGMA table_info(ormp_samples)").fetchall()
    return {str(r[1]) for r in rows}


def feature_group_catalog_for_build(ormp_db_path: str) -> list[dict[str, Any]]:
    """Feature groups limited to columns that exist in this ORMP build."""
    available = ormp_sample_columns(ormp_db_path)
    out: list[dict[str, Any]] = []
    for gid, label, feats in FEATURE_GROUPS:
        present = [f for f in feats if f in available]
        if present:
            out.append({"id": gid, "label": label, "features": present})
    return out


def label_column_name(label_type: LabelType, horizon_min: int) -> str:
    if label_type == "direction":
        return f"ormp_direction_{int(horizon_min)}m"
    return f"ormp_return_{int(horizon_min)}m_{label_type}"


def suggest_dataset_name(
    *,
    band_size_pct: float | None,
    price_source: str,
    path_mode: str,
    horizon_min: int,
    label_type: LabelType,
) -> str:
    bs = band_size_pct
    bs_tag = f"bs{str(bs).replace('.', 'p')}" if bs is not None else "bs"
    return (
        f"ormp_{bs_tag}_{price_source}_{path_mode}"
        f"_ret{int(horizon_min)}m_{label_type}"
    )


def _compute_label(
    spot: pd.Series,
    future: pd.Series,
    label_type: LabelType,
) -> pd.Series:
    if label_type == "points":
        return future - spot
    if label_type == "percent":
        return (future - spot) / spot
    # direction: -1 / 0 / +1
    delta = future - spot
    return delta.map(lambda x: 0.0 if x == 0 else (1.0 if x > 0 else -1.0))


def export_training_frame(
    ormp_db_path: str,
    *,
    feature_columns: list[str],
    label_type: LabelType,
    horizon_min: int,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    """Read ORMP samples → labeled DataFrame; drop rows without future LTP."""
    if horizon_min not in HORIZONS_MIN:
        raise ValueError(f"horizon_min must be one of {HORIZONS_MIN}")
    if label_type not in ("points", "percent", "direction"):
        raise ValueError(f"unsupported label_type: {label_type}")

    allowed = set(all_grouped_features())
    feats = [c for c in feature_columns if c in allowed]
    if not feats:
        raise ValueError("Select at least one feature")

    available = ormp_sample_columns(ormp_db_path)
    missing = [c for c in feats if c not in available]
    if missing:
        hint = ""
        if any("ema" in c and "to_spot_ratio" in c for c in missing):
            hint = (
                " Market Context (multi-TF EMA ratios) requires rebuilding the "
                "ORMP artifact on the Builds tab, then exporting from that new build."
            )
        preview = ", ".join(missing[:6])
        more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
        raise ValueError(
            f"{len(missing)} selected feature(s) are not in this ORMP build "
            f"(built before those columns existed).{hint}\n\n"
            f"Missing: {preview}{more}"
        )

    future_col = f"future_ltp_{int(horizon_min)}m"
    label_col = label_column_name(label_type, horizon_min)
    identity = ["trading_day", "timestamp", "spot_open"]
    select_cols = identity + ["spot_ltp", future_col] + feats

    # De-dupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in select_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    quoted = ", ".join(f'"{c}"' for c in ordered)
    sql = f'SELECT {quoted} FROM ormp_samples WHERE 1=1'
    params: list[object] = []
    if from_date:
        sql += " AND trading_day >= ?"
        params.append(from_date)
    if to_date:
        sql += " AND trading_day <= ?"
        params.append(to_date)
    sql += " ORDER BY trading_day, timestamp"

    with sqlite3.connect(ormp_db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    rows_in_range = int(len(df))
    if rows_in_range == 0:
        return {
            "ok": False,
            "df": df,
            "label_column": label_col,
            "feature_columns": feats,
            "rows_in_range": 0,
            "rows_exported": 0,
            "rows_dropped_no_future": 0,
            "error": "No rows in selected date range",
        }

    valid = df[future_col].notna() & df["spot_ltp"].notna()
    dropped = int((~valid).sum())
    out = df.loc[valid].copy()
    out[label_col] = _compute_label(out["spot_ltp"], out[future_col], label_type)
    # Drop forward price (label already computed); keep spot_ltp for reference? exclude from features.
    out = out.drop(columns=[future_col])
    # Keep identity + spot_ltp + features + label
    keep = identity + ["spot_ltp"] + feats + [label_col]
    out = out[keep]
    out = out.reset_index(drop=True)

    enabled_groups = [
        {"id": gid, "label": label, "features": [f for f in gfeats if f in feats]}
        for gid, label, gfeats in FEATURE_GROUPS
        if any(f in feats for f in gfeats)
    ]

    return {
        "ok": True,
        "df": out,
        "label_column": label_col,
        "feature_columns": feats,
        "feature_groups": enabled_groups,
        "rows_in_range": rows_in_range,
        "rows_exported": int(len(out)),
        "rows_dropped_no_future": dropped,
        "horizon_min": int(horizon_min),
        "label_type": label_type,
        "from_date": from_date,
        "to_date": to_date,
        "future_column": future_col,
    }
