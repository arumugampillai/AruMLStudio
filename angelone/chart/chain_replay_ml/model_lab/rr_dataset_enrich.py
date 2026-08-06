"""Enrich an existing training dataset with RR classifier labels from a Prediction Lab.

Prediction Dataset is the source of truth for RR hit labels
(rr_1_1_hit / rr_2_3_hit / rr_1_2_hit / rr_1_3_hit / rr_1_4_hit).
Features are never recomputed; Master Dataset is never touched.

Important: enrichment joins **Seen** prediction rows only. Unseen rows are ignored
(held out for evaluation). Matching a training row to an Unseen prediction is a
validation failure (leak / lab mismatch).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.master_registry_export import _resolve_unique_dataset_name
from chain_replay_ml.dataset_builder.writer import (
    _safe_filename,
    _write_parquet,
    datasets_dir,
    ensure_parquet_engine,
)
from chain_replay_ml.training.dataset_loader import parquet_column_names

from .paths import iter_all_lab_db_paths
from .prediction_io import load_dataset_meta, resolve_dataset_parquet
from .prediction_schema import DATASET_TYPE_SEEN, DATASET_TYPE_UNSEEN, RR_HIT_COLUMNS
from .service import load_lab
from .store import ModelLabStore

RR_LABEL_COLUMNS: tuple[str, ...] = RR_HIT_COLUMNS
_NATURAL_KEY = ("trading_day", "timestamp", "token")
_NATURAL_KEY_WITH_SYMBOL = ("trading_day", "timestamp", "token", "symbol")

_INCONSISTENT_MSG = (
    "❌ RR Enrichment aborted\n\n"
    "Reason:\n"
    "Training dataset and Prediction Lab are inconsistent.\n"
    "Found unmatched or non-Seen prediction rows."
)


def list_labs_with_rr_labels(
    *,
    research_dir: str | None = None,
    parent_dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Labs that have persisted RR columns (matching parent_dataset sorted first)."""
    out: list[dict[str, Any]] = []
    want = str(parent_dataset or "").strip() or None
    for path in iter_all_lab_db_paths(research_dir=research_dir):
        info = load_lab(path)
        if info is None:
            continue
        try:
            with ModelLabStore(info.db_path) as store:
                store.ensure_prediction_schema()
                cols = store._prediction_table_columns()
                if not all(c in cols for c in RR_LABEL_COLUMNS):
                    continue
                summary = store.read_prediction_summary() or {}
                parent = str(summary.get("parent_dataset") or "").strip()
                n = int(store.prediction_row_count() or 0)
                labeled = store.conn.execute(
                    """
                    SELECT COUNT(*) FROM prediction_dataset
                    WHERE rr_1_2_hit IS NOT NULL
                    """
                ).fetchone()
                labeled_n = int(labeled[0] or 0) if labeled else 0
        except Exception:
            continue
        if labeled_n <= 0:
            continue
        parent_match = bool(want and parent and parent == want)
        out.append(
            {
                "lab_name": info.lab_name,
                "parent_model_name": info.parent_model_name,
                "version": info.version,
                "db_path": info.db_path,
                "parent_dataset": parent or None,
                "parent_match": parent_match,
                "prediction_rows": n,
                "rr_labeled_rows": labeled_n,
                "target_column": summary.get("target_column"),
                "prediction_build_timestamp": summary.get("created_at"),
                "label": (
                    f"{info.parent_model_name} · v{info.version}"
                    f" ({labeled_n:,} RR rows"
                    + (f" · parent={parent}" if parent else "")
                    + ")"
                ),
            }
        )
    out.sort(
        key=lambda r: (
            0 if r.get("parent_match") else 1,
            str(r.get("parent_model_name") or ""),
            -int(r.get("version") or 0),
        )
    )
    return out


def _resolve_join_keys(
    train_cols: set[str],
    pred_cols: set[str],
) -> list[str]:
    if "master_row_id" in train_cols and "master_row_id" in pred_cols:
        return ["master_row_id"]
    if all(c in train_cols and c in pred_cols for c in _NATURAL_KEY_WITH_SYMBOL):
        return list(_NATURAL_KEY_WITH_SYMBOL)
    if all(c in train_cols and c in pred_cols for c in _NATURAL_KEY):
        return list(_NATURAL_KEY)
    raise ValueError(
        "Cannot join training dataset to prediction rows: need master_row_id "
        "or (trading_day, timestamp, token[, symbol]) on both sides."
    )


def _day_type_sets(store: ModelLabStore) -> tuple[set[str], set[str], bool]:
    """Return (seen_days, unseen_days, catalog_available)."""
    summary = store.read_prediction_summary() or {}
    lab_uuid = str(summary.get("lab_uuid") or "").strip()
    if not lab_uuid:
        return set(), set(), False
    days = store.list_build_days(lab_uuid)
    if not days:
        return set(), set(), False
    seen = {
        str(d["trading_day"])
        for d in days
        if str(d.get("dataset_type") or "") == DATASET_TYPE_SEEN
    }
    unseen = {
        str(d["trading_day"])
        for d in days
        if str(d.get("dataset_type") or "") == DATASET_TYPE_UNSEEN
    }
    return seen, unseen, True


def _load_prediction_rr_frames(
    lab_db_path: str,
    join_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load RR-labeled prediction rows split into Seen (join) vs Unseen (ignored)."""
    select = list(dict.fromkeys(["trading_day", *join_keys, *RR_LABEL_COLUMNS]))
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        cols = store._prediction_table_columns()
        missing = [c for c in select if c not in cols]
        if missing:
            raise ValueError(f"Prediction dataset missing columns: {', '.join(missing)}")
        col_sql = ", ".join(f'"{c}"' for c in select)
        raw = store.conn.execute(
            f"""
            SELECT {col_sql}
            FROM prediction_dataset
            WHERE rr_1_2_hit IS NOT NULL
            """
        ).fetchall()
        seen_days, unseen_days, catalog = _day_type_sets(store)

    if not raw:
        raise ValueError("No RR-labeled prediction rows (rr_1_2_hit IS NULL for all rows).")

    all_pred = pd.DataFrame.from_records(raw, columns=select)
    day_s = all_pred["trading_day"].astype(str)

    if catalog:
        seen_mask = day_s.isin(seen_days)
        unseen_mask = day_s.isin(unseen_days)
        # Days not catalogued: do not use for training labels
        other_mask = ~(seen_mask | unseen_mask)
        seen_pred = all_pred.loc[seen_mask].copy()
        unseen_pred = all_pred.loc[unseen_mask | other_mask].copy()
    else:
        # Legacy labs without day catalog → treat all RR rows as Seen
        seen_pred = all_pred.copy()
        unseen_pred = all_pred.iloc[0:0].copy()

    stats = {
        "seen_prediction_rows": int(len(seen_pred)),
        "unseen_prediction_rows": int(len(unseen_pred)) if catalog else 0,
        "unseen_ignored": int(len(unseen_pred)),
        "rr_labeled_rows": int(len(all_pred)),
        "day_catalog_available": catalog,
        "seen_days": len(seen_days) if catalog else None,
        "unseen_days": len(unseen_days) if catalog else None,
    }
    if stats["seen_prediction_rows"] <= 0:
        raise ValueError(
            "No Seen RR-labeled prediction rows to join. "
            "Unseen rows are ignored and cannot enrich a training dataset."
        )
    return seen_pred, unseen_pred, stats


def _count_key_matches(
    train: pd.DataFrame,
    other: pd.DataFrame,
    join_keys: list[str],
) -> int:
    if other is None or len(other) == 0 or len(train) == 0:
        return 0
    keys = other[join_keys].drop_duplicates()
    merged = train.merge(keys, on=join_keys, how="inner")
    return int(len(merged))


def _validate_join(
    train: pd.DataFrame,
    pred: pd.DataFrame,
    join_keys: list[str],
) -> dict[str, Any]:
    dataset_rows = int(len(train))
    prediction_rows = int(len(pred))

    pred_keys = pred[join_keys].copy()
    train_keys = train[join_keys].copy()

    pred_dup_mask = pred_keys.duplicated(keep=False)
    pred_dup_n = int(pred_dup_mask.sum())
    pred_dup_groups = (
        int(pred_keys[pred_dup_mask].drop_duplicates().shape[0]) if pred_dup_n else 0
    )

    train_null = train_keys.isna().any(axis=1)
    train_null_n = int(train_null.sum())

    pred_unique = pred.drop_duplicates(subset=join_keys, keep=False)
    if pred_dup_n > 0:
        pred_for_match = pred.drop_duplicates(subset=join_keys, keep="first")
    else:
        pred_for_match = pred_unique if len(pred_unique) == len(pred) else pred

    merged = train.merge(
        pred_for_match[join_keys],
        on=join_keys,
        how="left",
        indicator=True,
    )
    matched = int((merged["_merge"] == "both").sum())
    missing = int((merged["_merge"] == "left_only").sum()) + train_null_n

    pred_counts = pred.groupby(join_keys, dropna=False).size().reset_index(name="_pc")
    multi = train.merge(pred_counts, on=join_keys, how="left")
    multi_hit = multi["_pc"].fillna(0).astype(int)
    duplicate_matches = int((multi_hit > 1).sum())

    ok = missing == 0 and duplicate_matches == 0 and pred_dup_groups == 0
    return {
        "ok": ok,
        "dataset_rows": dataset_rows,
        "prediction_rows": prediction_rows,
        "matched": matched,
        "missing": missing,
        "duplicates": duplicate_matches if duplicate_matches else pred_dup_groups,
        "prediction_duplicate_key_groups": pred_dup_groups,
        "training_null_join_keys": train_null_n,
        "message": (
            "✓ Validation passed"
            if ok
            else _INCONSISTENT_MSG
        ),
    }


def enrich_training_dataset_with_rr_labels(
    data_dir: str,
    dataset_name: str,
    lab_db_path: str,
    *,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Append rr_*_hit columns from **Seen** Prediction Lab rows onto a registry dataset.

    Creates a new dataset (default ``{name}_rr``). Never overwrites the source.
    Aborts when Missing > 0, Duplicates > 0, parent mismatch, or training rows
    match Unseen predictions.
    """
    ensure_parquet_engine()
    src_name = _safe_filename(dataset_name)
    parquet_path, meta_path = resolve_dataset_parquet(data_dir, src_name)
    meta = load_dataset_meta(meta_path)

    train_cols = parquet_column_names(parquet_path)
    if not train_cols:
        raise ValueError(f"Cannot read parquet schema: {parquet_path}")
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        pred_cols = store._prediction_table_columns()
        summary = store.read_prediction_summary() or {}

    join_keys = _resolve_join_keys(train_cols, set(pred_cols))

    existing_rr = [c for c in RR_LABEL_COLUMNS if c in train_cols]
    if existing_rr == list(RR_LABEL_COLUMNS):
        return {
            "ok": False,
            "error": f"Dataset {src_name} already has RR label columns.",
            "dataset_name": src_name,
        }

    # Verify training export ↔ Prediction Lab parent
    parent = str(summary.get("parent_dataset") or "").strip()
    if parent and parent != src_name:
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT_MSG}\n\n"
                f"Training dataset: {src_name}\n"
                f"Prediction Lab parent_dataset: {parent}"
            ),
            "report": {
                "dataset": src_name,
                "parent_dataset": parent,
                "validation_message": _INCONSISTENT_MSG,
            },
            "dataset_name": src_name,
        }

    train_df = pd.read_parquet(parquet_path)
    seen_pred, unseen_pred, type_stats = _load_prediction_rr_frames(lab_db_path, join_keys)

    # Training rows must never resolve via Unseen labels
    unseen_matches = _count_key_matches(train_df, unseen_pred, join_keys)
    if unseen_matches > 0:
        report = {
            "dataset": src_name,
            "dataset_rows": int(len(train_df)),
            "seen_prediction_rows": type_stats["seen_prediction_rows"],
            "unseen_prediction_rows": type_stats["unseen_prediction_rows"],
            "unseen_ignored": type_stats["unseen_ignored"],
            "matched": 0,
            "missing": int(len(train_df)),
            "duplicates": 0,
            "unseen_matches": unseen_matches,
            "join_keys": join_keys,
            "validation_message": _INCONSISTENT_MSG,
        }
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT_MSG}\n\n"
                f"Training rows matching Unseen predictions: {unseen_matches:,}"
            ),
            "report": report,
            "dataset_name": src_name,
        }

    validation = _validate_join(train_df, seen_pred, join_keys)
    report = {
        "dataset": src_name,
        "dataset_rows": validation["dataset_rows"],
        "seen_prediction_rows": type_stats["seen_prediction_rows"],
        "unseen_prediction_rows": type_stats["unseen_prediction_rows"],
        "unseen_ignored": type_stats["unseen_ignored"],
        "prediction_rows": type_stats["seen_prediction_rows"],
        "matched": validation["matched"],
        "missing": validation["missing"],
        "duplicates": validation["duplicates"],
        "unseen_matches": 0,
        "join_keys": join_keys,
        "validation_message": validation["message"],
        "parent_dataset": parent or None,
        "day_catalog_available": type_stats["day_catalog_available"],
    }
    if not validation["ok"]:
        return {
            "ok": False,
            "error": validation["message"],
            "report": report,
            "dataset_name": src_name,
        }

    pred_unique = seen_pred.drop_duplicates(subset=join_keys, keep=False)
    label_frame = pred_unique[join_keys + list(RR_LABEL_COLUMNS)]
    enriched = train_df.merge(label_frame, on=join_keys, how="left", validate="one_to_one")

    null_labels = enriched[list(RR_LABEL_COLUMNS)].isna().any(axis=1).sum()
    if int(null_labels) > 0:
        return {
            "ok": False,
            "error": f"Post-join null RR labels on {int(null_labels):,} rows.",
            "report": report,
            "dataset_name": src_name,
        }

    base_out = str(output_name or "").strip() or f"{src_name}_rr"
    out_name = _resolve_unique_dataset_name(data_dir, base_out)

    out_meta = dict(meta) if isinstance(meta, dict) else {}
    out_meta["dataset_name"] = out_name
    out_meta["source_dataset"] = src_name
    out_meta["export_source"] = "rr_label_enrichment"
    out_meta["created_at"] = datetime.now(timezone.utc).isoformat()
    out_meta.pop("preserve_created_at", None)

    feature_cols = list(out_meta.get("feature_columns") or out_meta.get("selected_features") or [])
    if not feature_cols:
        skip = set(RR_LABEL_COLUMNS) | {
            "trading_day",
            "market",
            "expiry",
            "timestamp",
            "strike",
            "option_type",
            "token",
            "symbol",
            "master_row_id",
            "ltp",
            "spot",
        }
        feature_cols = [
            c
            for c in train_df.columns
            if c not in skip and not str(c).startswith("future_ltp")
        ]
    out_meta["feature_columns"] = feature_cols
    out_meta["selected_features"] = feature_cols
    out_meta["feature_count"] = len(feature_cols)

    pred_targets = list(out_meta.get("prediction_target_columns") or [])
    for col in RR_LABEL_COLUMNS:
        if col not in pred_targets:
            pred_targets.append(col)
    for col in train_df.columns:
        if str(col).startswith("future_ltp") and col not in pred_targets:
            pred_targets.insert(0, col)
    out_meta["prediction_target_columns"] = pred_targets
    out_meta["target_count"] = len(pred_targets)

    lab = load_lab(lab_db_path)
    out_meta["classifier_labels"] = {col: True for col in RR_LABEL_COLUMNS}
    out_meta["rr_enrichment"] = {
        "source_dataset": src_name,
        "prediction_lab": lab.lab_name if lab else os.path.basename(lab_db_path),
        "prediction_lab_db": lab_db_path,
        "parent_model_name": (lab.parent_model_name if lab else None)
        or summary.get("parent_model_name"),
        "prediction_dataset_version": int(lab.version) if lab else None,
        "prediction_build_timestamp": summary.get("created_at"),
        "prediction_parent_dataset": summary.get("parent_dataset"),
        "prediction_target_column": summary.get("target_column"),
        "join_keys": join_keys,
        "seen_only": True,
        "seen_prediction_rows": type_stats["seen_prediction_rows"],
        "unseen_prediction_rows": type_stats["unseen_prediction_rows"],
        "unseen_ignored": type_stats["unseen_ignored"],
        "matched": validation["matched"],
        "missing": 0,
        "duplicates": 0,
        "columns_added": list(RR_LABEL_COLUMNS),
        "enriched_at": out_meta["created_at"],
    }
    out_meta["description"] = (
        f"RR-enriched training set from {src_name} "
        f"(Seen labels from Prediction Lab {out_meta['rr_enrichment']['prediction_lab']})"
    )
    out_meta["row_count"] = int(len(enriched))
    out_meta["column_count"] = int(len(enriched.columns))

    out_dir = datasets_dir(data_dir)
    parquet_out = os.path.join(out_dir, f"{out_name}.parquet")
    json_out = os.path.join(out_dir, f"{out_name}.json")
    out_meta["output_parquet"] = os.path.relpath(parquet_out, data_dir).replace("\\", "/")
    out_meta["output_json"] = os.path.relpath(json_out, data_dir).replace("\\", "/")
    out_meta.setdefault("version", meta.get("version") if isinstance(meta, dict) else None)

    _write_parquet(enriched, parquet_out)
    import json

    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(out_meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    report.update(
        {
            "columns_added": 3,
            "saved_as": out_name,
            "output_parquet": parquet_out,
            "output_json": json_out,
            "completed": True,
            "validation_message": "✓ Enrichment completed",
        }
    )
    return {
        "ok": True,
        "dataset_name": out_name,
        "source_dataset": src_name,
        "report": report,
        "classifier_labels": list(RR_LABEL_COLUMNS),
        "join_keys": join_keys,
    }


def format_rr_enrichment_report(result: dict[str, Any]) -> str:
    """Human-readable export report for UI dialogs."""
    rep = result.get("report") if isinstance(result.get("report"), dict) else {}

    def _n(key: str, default: Any = "—") -> str:
        val = rep.get(key, default)
        if val is None or val == "—":
            return "—"
        try:
            return f"{int(val):,}"
        except (TypeError, ValueError):
            return str(val)

    if not result.get("ok"):
        lines = [
            "RR Enrichment Report",
            "",
            f"Training Dataset Rows : {_n('dataset_rows')}",
            f"Seen Prediction Rows  : {_n('seen_prediction_rows', rep.get('prediction_rows', '—'))}",
            f"Unseen Prediction Rows: {_n('unseen_prediction_rows', 0)} (ignored)",
            f"Matched               : {_n('matched')}",
            f"Missing               : {_n('missing')}",
            f"Duplicates            : {_n('duplicates')}",
            "",
            str(result.get("error") or rep.get("validation_message") or "Failed"),
        ]
        return "\n".join(str(x) for x in lines)

    lines = [
        "RR Enrichment Report",
        "",
        f"Training Dataset Rows : {_n('dataset_rows')}",
        f"Seen Prediction Rows  : {_n('seen_prediction_rows', rep.get('prediction_rows', 0))}",
        f"Unseen Prediction Rows: {_n('unseen_prediction_rows', 0)} (ignored)",
        f"Matched               : {_n('matched')}",
        f"Missing               : {_n('missing')}",
        f"Duplicates            : {_n('duplicates')}",
        "",
        f"Columns Added: {int(rep.get('columns_added') or 3)}",
        f"Saved As: {rep.get('saved_as') or result.get('dataset_name')}",
        "",
        "✓ Enrichment completed",
    ]
    return "\n".join(lines)
