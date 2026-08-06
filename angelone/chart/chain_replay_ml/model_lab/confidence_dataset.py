"""Lab-local Confidence Dataset — Seen training features + RR / hit labels."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.dataset_builder.writer import (
    _write_parquet,
    ensure_parquet_engine,
)
from chain_replay_ml.training.dataset_loader import parquet_column_names

from .prediction_builder import _dataset_name_from_lab, _guess_data_dirs_for_lab
from .prediction_io import load_dataset_meta, resolve_dataset_parquet
from .rr_dataset_enrich import (
    RR_LABEL_COLUMNS,
    _count_key_matches,
    _day_type_sets,
    _resolve_join_keys,
    _validate_join,
)
from .service import load_lab
from .store import ModelLabStore
from .target_spec import market_label_columns, replay_label_columns

CONFIDENCE_DATASET_NAME = "confidence_dataset"
# Market Outcomes always; Replay-Based Outcomes merged when a Label Run exists.
LABEL_COLUMNS: tuple[str, ...] = (*market_label_columns(), *replay_label_columns())

_INCONSISTENT = (
    "❌ Confidence Dataset aborted\n\n"
    "Training dataset and Prediction Lab are inconsistent.\n"
    "Found unmatched or non-Seen prediction rows."
)


def resolve_regression_selected_features(
    lab_db_path: str,
    lab: Any | None = None,
) -> dict[str, Any]:
    """
    Load the parent regression model's persisted selected feature list.

    Confidence Models must train on this subset only — never the full export
    feature matrix and never a separate RFC / feature-selection pass.
    """
    info = lab if lab is not None else load_lab(lab_db_path)
    if info is None:
        return {
            "ok": False,
            "features": [],
            "source": None,
            "error": "Research Lab not found.",
        }

    feats: list[str] = []
    source = None

    snap = info.selected_features_snapshot
    if isinstance(snap, list) and snap:
        feats = [str(x).strip() for x in snap if str(x).strip()]
        source = "lab.selected_features_snapshot"

    if not feats:
        # Walk-forward CSV / package pointers
        pointers = info.artifact_pointers or {}
        for key in ("selected_features_csv", "selected_features.csv"):
            item = pointers.get(key)
            path = None
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("abs") or "").strip()
            elif item:
                path = str(item).strip()
            if path and os.path.isfile(path):
                try:
                    import csv

                    names: list[str] = []
                    with open(path, encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            feat = str(
                                row.get("feature") or row.get("Feature") or ""
                            ).strip()
                            if not feat:
                                continue
                            selected = str(
                                row.get("selected") or row.get("Selected") or ""
                            ).strip().lower()
                            if selected in ("", "yes", "true", "1", "y"):
                                names.append(feat)
                    if names:
                        feats = names
                        source = f"artifact:{os.path.basename(path)}"
                        break
                except OSError:
                    pass

    if not feats:
        count = info.selected_feature_count
        return {
            "ok": False,
            "features": [],
            "source": None,
            "selected_feature_count": count,
            "error": (
                "Regression model has no persisted selected feature list. "
                "Open / recreate the Research Lab from a trained model that "
                "stores selected_features (walk-forward CSV or snapshot)."
            ),
        }

    return {
        "ok": True,
        "features": feats,
        "source": source or "regression_model",
        "feature_count": len(feats),
        "selected_feature_count": len(feats),
        "parent_model_name": info.parent_model_name,
    }


def confidence_package_dir(lab_db_path: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(lab_db_path))
    return f"{base}_confidence"


def confidence_manifest_path(lab_db_path: str) -> str:
    return os.path.join(confidence_package_dir(lab_db_path), "confidence.json")


def confidence_legacy_sidecar_path(lab_db_path: str) -> str:
    base, _ = os.path.splitext(lab_db_path)
    return f"{base}.confidence.json"


def confidence_dataset_paths(lab_db_path: str) -> dict[str, str]:
    root = confidence_package_dir(lab_db_path)
    ds = os.path.join(root, "datasets")
    return {
        "package_dir": root,
        "datasets_dir": ds,
        "parquet": os.path.join(ds, f"{CONFIDENCE_DATASET_NAME}.parquet"),
        "json": os.path.join(ds, f"{CONFIDENCE_DATASET_NAME}.json"),
        "models_dir": os.path.join(root, "models"),
        "manifest": confidence_manifest_path(lab_db_path),
    }


def resolve_training_dataset(
    lab_db_path: str,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Resolve parent training export name + parquet for this Research Lab."""
    lab = load_lab(lab_db_path)
    if lab is None:
        raise FileNotFoundError(f"Research Lab not found: {lab_db_path}")

    name = _dataset_name_from_lab(lab)
    with ModelLabStore(lab_db_path) as store:
        summary = store.read_prediction_summary() or {}
    parent = str(summary.get("parent_dataset") or "").strip()
    if not name:
        name = parent
    if not name:
        raise ValueError("Research Lab has no training dataset (dataset_snapshot / parent_dataset).")

    candidates: list[str] = []
    if data_dir and os.path.isdir(data_dir):
        candidates.append(os.path.abspath(data_dir))
    candidates.extend(_guess_data_dirs_for_lab(lab, lab_db_path))

    last_err: Exception | None = None
    for cand in candidates:
        try:
            parquet_path, meta_path = resolve_dataset_parquet(cand, name)
            meta = load_dataset_meta(meta_path)
            return {
                "dataset_name": name,
                "data_dir": cand,
                "parquet_path": parquet_path,
                "meta_path": meta_path,
                "meta": meta,
                "parent_dataset": parent or name,
                "lab": lab,
                "prediction_summary": summary,
            }
        except Exception as exc:
            last_err = exc
            continue
    raise FileNotFoundError(
        f"Training dataset '{name}' not found for lab. Last error: {last_err}"
    )


def _load_seen_label_frames(
    lab_db_path: str,
    join_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    select = list(
        dict.fromkeys(
            [
                "trading_day",
                "prediction_id",
                *join_keys,
                *market_label_columns(),
                "maximum_profit",
                "maximum_drawdown",
            ]
        )
    )
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        cols = store._prediction_table_columns()
        # Labels required; profit/dd optional (used to backfill missing RR cols)
        required = ["trading_day", *join_keys, "target_reached"]
        # At least one legacy RR column should exist for labeled rows
        for c in ("rr_1_2_hit", "rr_1_3_hit", "rr_1_4_hit"):
            if c in cols:
                required.append(c)
                break
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"Prediction dataset missing columns: {', '.join(missing)}")
        select = [c for c in select if c in cols]
        col_sql = ", ".join(f'"{c}"' for c in select)
        # Prefer rows with RR labels; fall back to target_reached-only if RR absent
        raw = store.conn.execute(
            f"""
            SELECT {col_sql}
            FROM prediction_dataset
            WHERE rr_1_2_hit IS NOT NULL
            """
        ).fetchall()
        if not raw:
            raw = store.conn.execute(
                f"""
                SELECT {col_sql}
                FROM prediction_dataset
                WHERE target_reached IS NOT NULL
                """
            ).fetchall()
        seen_days, unseen_days, catalog = _day_type_sets(store)

    if not raw:
        raise ValueError("No labeled prediction rows (RR / target_reached).")

    all_pred = pd.DataFrame.from_records(raw, columns=select)

    # Ensure every market LABEL column exists, then backfill null RR from profit/dd
    for col in market_label_columns():
        if col not in all_pred.columns:
            all_pred[col] = pd.NA
    backfilled_cols: list[str] = []
    if "maximum_profit" in all_pred.columns and "maximum_drawdown" in all_pred.columns:
        from .prediction_schema import RR_HIT_THRESHOLDS

        hit = pd.to_numeric(all_pred["target_reached"], errors="coerce")
        profit = pd.to_numeric(all_pred["maximum_profit"], errors="coerce")
        risk = pd.to_numeric(all_pred["maximum_drawdown"], errors="coerce")
        can_score = hit.notna() & ((hit != 1) | ((profit.notna()) & (risk.notna()) & (risk > 0)))
        ratio = profit / risk

        for col, threshold in RR_HIT_THRESHOLDS:
            if col not in all_pred.columns:
                continue
            null_mask = all_pred[col].isna() & can_score
            if not bool(null_mask.any()):
                continue
            fill = ((hit == 1) & (ratio >= threshold)).fillna(False).astype(int)
            all_pred.loc[null_mask, col] = fill.loc[null_mask]
            backfilled_cols.append(col)

    day_s = all_pred["trading_day"].astype(str)
    if catalog:
        seen_mask = day_s.isin(seen_days)
        unseen_mask = day_s.isin(unseen_days)
        other_mask = ~(seen_mask | unseen_mask)
        seen_pred = all_pred.loc[seen_mask].copy()
        unseen_pred = all_pred.loc[unseen_mask | other_mask].copy()
    else:
        seen_pred = all_pred.copy()
        unseen_pred = all_pred.iloc[0:0].copy()

    # Drop profit/dd helpers — not classifier labels
    for helper in ("maximum_profit", "maximum_drawdown"):
        if helper in seen_pred.columns:
            seen_pred = seen_pred.drop(columns=[helper])
        if helper in unseen_pred.columns:
            unseen_pred = unseen_pred.drop(columns=[helper])

    stats = {
        "seen_prediction_rows": int(len(seen_pred)),
        "unseen_prediction_rows": int(len(unseen_pred)) if catalog else 0,
        "unseen_ignored": int(len(unseen_pred)),
        "day_catalog_available": catalog,
        "rr_labels_backfilled_columns": backfilled_cols,
    }
    if stats["seen_prediction_rows"] <= 0:
        raise ValueError("No Seen labeled prediction rows to build Confidence Dataset.")
    return seen_pred, unseen_pred, stats


def create_confidence_dataset(
    lab_db_path: str,
    data_dir: str | None = None,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Build lab-local Confidence Dataset = training features + Seen prediction labels.

    Does not write to Dataset Registry. Seen rows only; Unseen ignored.
    ``on_progress`` receives ``{"phase", "message", "percent", ...}``.
    """
    def _prog(
        percent: float,
        message: str,
        *,
        phase: str = "create",
        **extra: Any,
    ) -> None:
        if on_progress is None:
            return
        try:
            on_progress(
                {
                    "phase": phase,
                    "message": message,
                    "percent": float(percent),
                    **extra,
                }
            )
        except Exception:
            pass

    ensure_parquet_engine()
    _prog(2, "Resolving training dataset…", phase="resolve")
    resolved = resolve_training_dataset(lab_db_path, data_dir=data_dir)
    lab = resolved["lab"]
    train_name = resolved["dataset_name"]
    parquet_path = resolved["parquet_path"]
    meta = resolved["meta"] if isinstance(resolved["meta"], dict) else {}
    summary = resolved["prediction_summary"]
    train_rows_meta = int(meta.get("row_count") or 0)
    _prog(
        8,
        f"Training dataset: {train_name} ({train_rows_meta:,} rows)",
        phase="resolve",
        dataset_rows=train_rows_meta,
        training_dataset=train_name,
    )

    parent = str(summary.get("parent_dataset") or "").strip()
    if parent and parent != train_name:
        _prog(100, "Aborted — parent mismatch", phase="error")
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT}\n\n"
                f"Training dataset: {train_name}\n"
                f"Prediction Lab parent_dataset: {parent}"
            ),
            "report": {
                "training_dataset": train_name,
                "dataset_rows": train_rows_meta,
                "matched": 0,
                "missing": train_rows_meta or None,
                "duplicates": 0,
            },
        }

    train_cols = parquet_column_names(parquet_path)
    if not train_cols:
        _prog(100, "Aborted — cannot read parquet schema", phase="error")
        return {
            "ok": False,
            "error": f"Cannot read parquet schema: {parquet_path}",
            "report": {
                "training_dataset": train_name,
                "dataset_rows": train_rows_meta,
            },
        }

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        pred_cols = set(store._prediction_table_columns())

    join_keys = _resolve_join_keys(train_cols, pred_cols)
    _prog(18, f"Loading training parquet ({train_rows_meta:,} rows)…", phase="load_train")
    train_df = pd.read_parquet(parquet_path)
    _prog(
        32,
        f"Loaded training rows: {len(train_df):,}",
        phase="load_train",
        dataset_rows=int(len(train_df)),
    )

    _prog(40, "Loading Seen prediction labels…", phase="load_pred")
    seen_pred, unseen_pred, type_stats = _load_seen_label_frames(lab_db_path, join_keys)

    # Merge Replay-Based Outcome binaries from latest Confidence Label Run (if any).
    replay_meta = None
    try:
        from .confidence_label_builder import load_replay_outcome_frames

        loaded = load_replay_outcome_frames(lab_db_path)
        if loaded.get("ok") and loaded.get("binary_labels") is not None:
            bin_df = loaded["binary_labels"]
            if (
                isinstance(bin_df, pd.DataFrame)
                and not bin_df.empty
                and "prediction_id" in bin_df.columns
                and "prediction_id" in seen_pred.columns
            ):
                replay_cols = [
                    c for c in replay_label_columns() if c in bin_df.columns
                ]
                if replay_cols:
                    merge_cols = ["prediction_id", *replay_cols]
                    seen_pred = seen_pred.merge(
                        bin_df[merge_cols].drop_duplicates(subset=["prediction_id"]),
                        on="prediction_id",
                        how="left",
                    )
                    replay_meta = {
                        "label_run_id": loaded.get("label_run_id"),
                        "strategy_version_id": (loaded.get("meta") or {}).get(
                            "strategy_version_id"
                        ),
                        "strategy_config_hash": (loaded.get("meta") or {}).get(
                            "strategy_config_hash"
                        ),
                        "binary_columns": replay_cols,
                    }
                    type_stats["replay_label_columns"] = replay_cols
                    _prog(
                        51,
                        (
                            "Merged Replay-Based labels from Confidence Label Run "
                            f"{loaded.get('label_run_id')}"
                        ),
                        phase="load_pred",
                    )
    except Exception as exc:
        type_stats["replay_label_merge_error"] = str(exc)

    if type_stats.get("rr_labels_backfilled_columns"):
        _prog(
            53,
            (
                "Backfilled RR labels from profit/drawdown: "
                + ", ".join(type_stats["rr_labels_backfilled_columns"])
            ),
            phase="load_pred",
        )
    _prog(
        52,
        (
            f"Seen labels: {type_stats['seen_prediction_rows']:,}  ·  "
            f"Unseen ignored: {type_stats['unseen_ignored']:,}"
        ),
        phase="load_pred",
        seen_prediction_rows=type_stats["seen_prediction_rows"],
        unseen_prediction_rows=type_stats["unseen_prediction_rows"],
        dataset_rows=int(len(train_df)),
    )

    unseen_matches = _count_key_matches(train_df, unseen_pred, join_keys)
    if unseen_matches > 0:
        _prog(100, "Aborted — Unseen matches", phase="error")
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT}\n\n"
                f"Training rows matching Unseen predictions: {unseen_matches:,}"
            ),
            "report": {
                "training_dataset": train_name,
                "dataset_rows": int(len(train_df)),
                "seen_prediction_rows": type_stats["seen_prediction_rows"],
                "unseen_prediction_rows": type_stats["unseen_prediction_rows"],
                "matched": 0,
                "missing": int(len(train_df)),
                "duplicates": 0,
                "unseen_matches": unseen_matches,
            },
        }

    _prog(58, "Validating join (Missing / Duplicates)…", phase="validate")
    validation = _validate_join(train_df, seen_pred, join_keys)
    report = {
        "training_dataset": train_name,
        "dataset_rows": validation["dataset_rows"],
        "seen_prediction_rows": type_stats["seen_prediction_rows"],
        "unseen_prediction_rows": type_stats["unseen_prediction_rows"],
        "matched": validation["matched"],
        "missing": validation["missing"],
        "duplicates": validation["duplicates"],
        "dropped_unmatched": 0,
        "join_keys": join_keys,
        "rr_labels_backfilled_columns": list(
            type_stats.get("rr_labels_backfilled_columns") or []
        ),
    }
    _prog(
        65,
        (
            f"Matched {validation['matched']:,}  ·  "
            f"Missing {validation['missing']:,}  ·  "
            f"Duplicates {validation['duplicates']:,}"
        ),
        phase="validate",
        **{k: report[k] for k in (
            "dataset_rows", "seen_prediction_rows", "unseen_prediction_rows",
            "matched", "missing", "duplicates",
        )},
    )
    # Duplicates are still fatal. Missing rows are trimmed (e.g. last-horizon
    # samples present in an older training export but dropped from Prediction).
    if int(validation["duplicates"] or 0) > 0:
        _prog(100, "Aborted — duplicate join keys", phase="error")
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT}\n\n"
                f"Duplicates={int(validation['duplicates']):,} "
                "(cannot safely join)."
            ),
            "report": report,
        }
    if int(validation["matched"] or 0) <= 0:
        _prog(100, "Aborted — no matched rows", phase="error")
        return {
            "ok": False,
            "error": (
                f"{_INCONSISTENT}\n\n"
                "No training rows matched Seen prediction rows."
            ),
            "report": report,
        }

    dropped = int(validation["missing"] or 0)
    if dropped > 0:
        _prog(
            68,
            f"Dropping {dropped:,} unmatched training rows (horizon / Seen gap)…",
            phase="trim",
            dropped_unmatched=dropped,
            dataset_rows=validation["matched"],
        )
        pred_keys = seen_pred[join_keys].drop_duplicates()
        train_df = train_df.merge(pred_keys, on=join_keys, how="inner")
        report["dropped_unmatched"] = dropped
        report["dataset_rows_after_trim"] = int(len(train_df))
        # Re-check duplicates after trim
        post = _validate_join(train_df, seen_pred, join_keys)
        if int(post["duplicates"] or 0) > 0 or int(post["missing"] or 0) > 0:
            _prog(100, "Aborted — trim left gaps/duplicates", phase="error")
            report["missing"] = post["missing"]
            report["duplicates"] = post["duplicates"]
            report["matched"] = post["matched"]
            return {
                "ok": False,
                "error": (
                    f"{_INCONSISTENT}\n\n"
                    f"After trimming unmatched rows: "
                    f"Missing={int(post['missing']):,}, "
                    f"Duplicates={int(post['duplicates']):,}"
                ),
                "report": report,
            }
        report["matched"] = post["matched"]
        report["missing"] = 0

    _prog(72, "Appending RR / Target Hit labels…", phase="merge")
    label_frame = seen_pred.drop_duplicates(subset=join_keys, keep=False)
    keep_labels = [c for c in LABEL_COLUMNS if c in label_frame.columns]
    enriched = train_df.merge(
        label_frame[join_keys + keep_labels],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )
    null_any = enriched[keep_labels].isna().any(axis=1) if keep_labels else pd.Series(
        False, index=enriched.index
    )
    null_labels = int(null_any.sum())
    dropped_all_null_cols: list[str] = []
    dropped_null_rows = 0
    if null_labels > 0:
        # Drop label columns that are entirely null (e.g. new RR cols not yet rebuilt)
        usable: list[str] = []
        for c in keep_labels:
            if bool(enriched[c].isna().all()):
                dropped_all_null_cols.append(c)
            else:
                usable.append(c)
        keep_labels = usable
        if keep_labels:
            null_any = enriched[keep_labels].isna().any(axis=1)
            dropped_null_rows = int(null_any.sum())
            if dropped_null_rows > 0:
                enriched = enriched.loc[~null_any].copy()
        warn_bits = []
        if dropped_all_null_cols:
            warn_bits.append(
                f"omitted all-null labels: {', '.join(dropped_all_null_cols)}"
            )
        if dropped_null_rows > 0:
            warn_bits.append(f"dropped {dropped_null_rows:,} rows with null labels")
        elif null_labels > 0 and dropped_all_null_cols:
            warn_bits.append(
                f"null labels on {null_labels:,} rows resolved by omitting empty columns"
            )
        msg = "Warning — " + ("; ".join(warn_bits) if warn_bits else f"null labels on {null_labels:,} rows")
        _prog(73, msg, phase="warn")
        report["null_label_warning"] = msg
        report["dropped_all_null_label_columns"] = dropped_all_null_cols
        report["dropped_null_label_rows"] = dropped_null_rows
        report["null_label_rows"] = null_labels
    if not keep_labels:
        _prog(100, "Aborted — no usable classifier labels after null handling", phase="error")
        return {
            "ok": False,
            "error": (
                "No usable classifier labels after join. "
                "Rebuild the Prediction Dataset so RR labels are populated."
            ),
            "report": report,
        }
    if len(enriched) == 0:
        _prog(100, "Aborted — no rows left after dropping null labels", phase="error")
        return {
            "ok": False,
            "error": "All matched rows had null classifier labels.",
            "report": report,
        }

    feature_resolve = resolve_regression_selected_features(lab_db_path, lab=lab)
    if not feature_resolve.get("ok"):
        _prog(100, "Aborted — no regression selected features", phase="error")
        return {
            "ok": False,
            "error": str(feature_resolve.get("error") or "No regression selected features."),
            "report": report,
        }
    regression_feats = list(feature_resolve["features"])
    feature_cols = [c for c in regression_feats if c in enriched.columns]
    missing_in_export = [c for c in regression_feats if c not in enriched.columns]
    if not feature_cols:
        _prog(100, "Aborted — selected features missing from training export", phase="error")
        return {
            "ok": False,
            "error": (
                "None of the regression model's selected features are present in the "
                f"training dataset '{train_name}'. Missing examples: "
                f"{missing_in_export[:8]}"
            ),
            "report": report,
        }
    if missing_in_export:
        _prog(
            74,
            (
                f"Using {len(feature_cols):,}/{len(regression_feats):,} regression "
                f"selected features ({len(missing_in_export)} absent from export)"
            ),
            phase="features",
        )
    else:
        _prog(
            74,
            f"Using regression selected features: {len(feature_cols):,}",
            phase="features",
        )

    paths = confidence_dataset_paths(lab_db_path)
    os.makedirs(paths["datasets_dir"], exist_ok=True)
    os.makedirs(paths["models_dir"], exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    out_meta = {
        "dataset_name": CONFIDENCE_DATASET_NAME,
        "source": "research_lab_confidence",
        "export_source": "confidence_dataset",
        "build_profile": "research_lab",
        "validation_skipped": True,
        "parent_model_name": lab.parent_model_name if lab else None,
        "lab_db_path": lab_db_path,
        "training_dataset": train_name,
        "prediction_lab_version": int(lab.version) if lab else None,
        "prediction_build_timestamp": summary.get("created_at"),
        "seen_only": True,
        "join_keys": join_keys,
        # Inherited from regression model — never the full export feature matrix
        "feature_source": "regression_model",
        "feature_source_detail": feature_resolve.get("source"),
        "feature_columns": feature_cols,
        "selected_features": feature_cols,
        "feature_count": len(feature_cols),
        "regression_selected_feature_count": len(regression_feats),
        "regression_features_missing_from_export": missing_in_export,
        "prediction_target_columns": list(keep_labels),
        "target_count": len(keep_labels),
        "classifier_labels": {c: True for c in keep_labels},
        "replay_label_run": replay_meta,
        "row_count": int(len(enriched)),
        "column_count": int(len(enriched.columns)),
        "created_at": created_at,
        "description": (
            f"Confidence Dataset for {lab.parent_model_name if lab else 'lab'} "
            f"(Seen rows from {train_name}; "
            f"{len(feature_cols)} regression selected features)"
        ),
    }

    # Persist only identity + regression selected features + classifier labels
    identity_cols = []
    for c in (
        "trading_day",
        "timestamp",
        "token",
        "symbol",
        "master_row_id",
        *join_keys,
    ):
        if c in enriched.columns and c not in identity_cols:
            identity_cols.append(c)
    keep_cols = identity_cols + [
        c for c in feature_cols if c not in identity_cols
    ] + [c for c in keep_labels if c not in identity_cols and c not in feature_cols]
    to_write = enriched[keep_cols].copy()

    _prog(
        82,
        f"Writing parquet ({len(to_write):,} rows · {len(feature_cols)} features)…",
        phase="write",
        dataset_rows=int(len(to_write)),
    )
    _write_parquet(to_write, paths["parquet"])
    out_meta["column_count"] = int(len(to_write.columns))
    out_meta["row_count"] = int(len(to_write))
    _prog(90, "Writing metadata…", phase="write")
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(out_meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    from .confidence_manifest import update_manifest_after_dataset

    _prog(95, "Updating Confidence manifest…", phase="manifest")
    update_manifest_after_dataset(
        lab_db_path,
        dataset_meta=out_meta,
        paths=paths,
        report=report,
        type_stats=type_stats,
    )

    report.update(
        {
            "columns_added": len(keep_labels),
            "saved_as": CONFIDENCE_DATASET_NAME,
            "package_dir": paths["package_dir"],
            "completed": True,
        }
    )
    _prog(
        100,
        f"Complete · {len(enriched):,} rows",
        phase="done",
        dataset_rows=int(len(enriched)),
        matched=report.get("matched"),
    )
    return {
        "ok": True,
        "dataset_name": CONFIDENCE_DATASET_NAME,
        "row_count": int(len(enriched)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "labels": keep_labels,
        "warning": report.get("null_label_warning"),
        "report": report,
        "paths": paths,
    }
