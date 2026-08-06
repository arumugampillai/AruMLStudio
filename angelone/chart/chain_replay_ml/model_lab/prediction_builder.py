"""Model Lab Phase 2 — prediction research dataset builder (multi-worker)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.training.dataset_loader import DatasetLoaderError
from chain_replay_ml.training.model_runtime import (
    load_prediction_model_cached,
    resolve_production_model_path,
)
from chain_replay_ml.training.paths import model_package_dir

from .prediction_io import (
    catalog_trading_days,
    dataset_row_counts_from_meta,
    estimate_sample_total,
    load_dataset_meta,
    load_parent_dataset_row_counts,
    resolve_dataset_parquet,
)
from .prediction_parallel import (
    DEFAULT_PREDICTION_WORKERS,
    DayJobContext,
    run_parallel_day_build,
)
from .prediction_progress import (
    STAGE_CATALOG_DAYS,
    STAGE_FINISHED,
    STAGE_LOAD_MODEL,
    STAGE_READ_METADATA,
    STAGE_TOTAL,
    ProgressHub,
    stage_label,
)
from .prediction_schema import (
    LAB_PHASE_PREDICTION,
    LAB_SCHEMA_VERSION_PREDICTION,
    PRED_STATUS_BUILDING,
    PRED_STATUS_ERROR,
    PRED_STATUS_NOT_GENERATED,
    PRED_STATUS_PAUSED,
    PRED_STATUS_READY,
    align_features_to_model,
    feature_column_map,
    horizon_sec_from_target,
)
from .prediction_control import BuildControl, set_active_build_control
from .store import ModelLabInfo, ModelLabStore

ProgressFn = Callable[[dict[str, Any]], None]

_IDENTITY_LOAD = (
    "trading_day",
    "timestamp",
    "token",
    "strike",
    "option_type",
    "symbol",
    "market",
    "expiry",
    "spot",
    "ltp",
    "minutes_to_expiry",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(on_progress: ProgressFn | None, **payload: Any) -> None:
    if on_progress:
        on_progress(payload)


def _fmt_timing(sec: float | None) -> str:
    if sec is None:
        return "—"
    total = max(0, int(round(float(sec))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _print_timing_summary(timing: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> str:
    lines = [
        "— Prediction Dataset Timing —",
        f"Workers:            {timing.get('workers')}",
        f"Total runtime:      {_fmt_timing(timing.get('total_runtime_sec'))} "
        f"({timing.get('total_runtime_sec')}s)",
        f"Trading days:       {timing.get('trading_days')}",
        f"Avg sec / day:      {timing.get('avg_sec_per_day')}",
        f"Predictions:        {timing.get('predictions_written')}",
        f"Failed days:        {timing.get('failed_days')}",
    ]
    if baseline and baseline.get("total_runtime_sec") and timing.get("total_runtime_sec"):
        try:
            speedup = float(baseline["total_runtime_sec"]) / float(timing["total_runtime_sec"])
            lines.append(
                f"Speedup vs {baseline.get('workers')} worker(s): {speedup:.2f}x"
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def _dataset_name_from_lab(lab: ModelLabInfo) -> str:
    snap = lab.dataset_snapshot or {}
    return str(
        snap.get("dataset_name")
        or snap.get("name")
        or snap.get("dataset")
        or ""
    ).strip()


def _guess_data_dirs_for_lab(lab: ModelLabInfo, lab_db_path: str) -> list[str]:
    """Candidate chart data_dirs for resolving the parent training export."""
    out: list[str] = []

    def _push(path: str) -> None:
        text = str(path or "").strip()
        if not text or not os.path.isdir(text):
            return
        abs_path = os.path.abspath(text)
        if abs_path not in out:
            out.append(abs_path)

    pointers = lab.artifact_pointers or {}
    if isinstance(pointers, dict):
        pkg = pointers.get("package_dir")
        if isinstance(pkg, dict):
            pkg_path = str(pkg.get("path") or pkg.get("abs") or "").strip()
        else:
            pkg_path = str(pkg or "").strip()
        if pkg_path:
            # .../models/<name> → data_dir is parent of models
            pkg_dir = pkg_path if os.path.isdir(pkg_path) else os.path.dirname(pkg_path)
            models_dir = os.path.dirname(pkg_dir)
            if os.path.basename(models_dir).lower() in ("models", "model"):
                _push(os.path.dirname(models_dir))
            _push(os.path.dirname(pkg_dir))
            _push(pkg_dir)

    lab_dir = os.path.dirname(os.path.abspath(lab_db_path))
    _push(lab_dir)
    _push(os.path.dirname(lab_dir))
    # Common layout: <data_dir>/model_labs/<lab>/lab.db
    parent2 = os.path.dirname(os.path.dirname(lab_dir))
    _push(parent2)
    return out


def _parent_days_for_lab(
    lab: ModelLabInfo,
    *,
    lab_db_path: str,
    data_dir: str | None = None,
) -> list[str]:
    """Trading days in this model's parent training export (Seen basis)."""
    dataset = _dataset_name_from_lab(lab)
    if not dataset:
        return []
    candidates: list[str] = []
    if data_dir and os.path.isdir(data_dir):
        candidates.append(os.path.abspath(data_dir))
    candidates.extend(_guess_data_dirs_for_lab(lab, lab_db_path))
    for cand in candidates:
        try:
            parquet_path, meta_path = resolve_dataset_parquet(cand, dataset)
            meta = load_dataset_meta(meta_path)
            days = list(catalog_trading_days(parquet_path, meta=meta))
            if days:
                return days
        except Exception:
            continue
    return []


def validate_prediction_inputs(
    data_dir: str,
    lab: ModelLabInfo,
) -> dict[str, Any]:
    """Pre-generation integrity checks."""
    errors: list[str] = []
    warnings: list[str] = []

    features = [str(f) for f in (lab.selected_features_snapshot or []) if str(f).strip()]
    if not features:
        errors.append("Selected features snapshot is empty")
    elif lab.selected_feature_count is not None and len(features) != int(lab.selected_feature_count):
        warnings.append(
            f"Feature count mismatch: snapshot has {len(features)}, "
            f"lab records {lab.selected_feature_count}"
        )

    target = str(lab.target or "").strip()
    if not target:
        errors.append("Target column is missing on Model Lab")

    dataset = _dataset_name_from_lab(lab)
    if not dataset:
        errors.append("Parent dataset name is missing from lab snapshot")

    model_name = str(lab.parent_model_name or "").strip()
    if not model_name:
        errors.append("Parent model name is missing")

    parquet_path = ""
    model_path = ""
    if dataset:
        safe = _safe_filename(dataset)
        parquet_path = os.path.join(datasets_dir(data_dir), f"{safe}.parquet")
        if not os.path.isfile(parquet_path):
            errors.append(f"Parent dataset parquet not found: {parquet_path}")

    if model_name:
        pkg = model_package_dir(data_dir, model_name)
        model_path = resolve_production_model_path(pkg, algorithm=lab.algorithm)
        if not model_path or not os.path.isfile(model_path):
            errors.append(f"Parent model artifact not found for {model_name}")

    if parquet_path and os.path.isfile(parquet_path) and features and target:
        try:
            from chain_replay_ml.training.dataset_loader import missing_parquet_columns

            missing = missing_parquet_columns(parquet_path, [*features, target])
            if missing:
                show = ", ".join(missing[:8])
                more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
                errors.append(f"Dataset missing columns: {show}{more}")
        except Exception as exc:
            warnings.append(f"Could not verify parquet columns: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "features": features,
        "target": target,
        "dataset": dataset,
        "model_name": model_name,
        "model_path": model_path,
        "parquet_path": parquet_path,
    }


def validate_prediction_output(db_path: str, *, expected_rows: int | None = None) -> dict[str, Any]:
    """Post-generation integrity checks."""
    errors: list[str] = []
    with ModelLabStore(db_path) as store:
        store.ensure_prediction_schema()
        n = store.prediction_row_count()
        dup = store.count_duplicate_prediction_ids()
        missing_ts = store.count_missing_timestamps()
        summary = store.read_prediction_summary()

    if expected_rows is not None and n != int(expected_rows):
        errors.append(f"Prediction count {n} != dataset samples {expected_rows}")
    if dup:
        errors.append(f"Found {dup} duplicate prediction_id values")
    if missing_ts:
        errors.append(f"Found {missing_ts} rows with missing timestamps")
    if summary is None or str(summary.get("status") or "") != PRED_STATUS_READY:
        errors.append("prediction_dataset_summary is not READY")

    return {"ok": not errors, "errors": errors, "row_count": n}


def prediction_dataset_status(db_path: str, *, light: bool = False) -> dict[str, Any]:
    """Prediction dataset readiness.

    ``light=True`` reads summary metadata only (no ``COUNT(*)`` on prediction_dataset).
    """
    if not os.path.isfile(db_path):
        return {"status": PRED_STATUS_NOT_GENERATED, "row_count": 0}
    with ModelLabStore(db_path) as store:
        store.ensure_prediction_schema()
        summary = store.read_prediction_summary()
        n = int(summary.get("row_count") or 0) if summary else 0
        if not light and n <= 0:
            n = store.prediction_row_count()
    if summary:
        return {
            "status": str(summary.get("status") or PRED_STATUS_NOT_GENERATED),
            "row_count": int(summary.get("row_count") or n),
            "trading_days": summary.get("trading_days"),
            "start_day": summary.get("start_day"),
            "end_day": summary.get("end_day"),
            "average_error": summary.get("average_error"),
            "average_absolute_error": summary.get("average_absolute_error"),
            "premium_error": summary.get("premium_error"),
            "direction_accuracy": summary.get("direction_accuracy"),
            "generation_time_sec": summary.get("generation_time_sec"),
            "dataset_hash": summary.get("dataset_hash"),
            "selected_feature_count": summary.get("selected_feature_count"),
            "parent_model_name": summary.get("parent_model_name"),
            "parent_dataset": summary.get("parent_dataset"),
            "target_column": summary.get("target_column"),
            "created_at": summary.get("created_at"),
            "error_message": summary.get("error_message"),
        }
    return {
        "status": PRED_STATUS_READY if n > 0 else PRED_STATUS_NOT_GENERATED,
        "row_count": n,
    }


def _day_ui_meta_ready(day: dict[str, Any]) -> bool:
    """True when expensive Trading Days columns can be shown (not placeholders)."""
    if day.get("ui_meta_ready") is False:
        return False
    if day.get("ui_meta_ready") is True:
        return True
    if day.get("rows_expected") is not None:
        return True
    if int(day.get("row_count") or 0) > 0:
        return True
    st = str(day.get("status") or "")
    return st not in ("", "waiting")


def prediction_days_ui_skeleton(
    data_dir: str,
    lab_db_path: str,
) -> dict[str, Any]:
    """
    Fast Trading Days list for initial UI paint.

    Two lightweight sources only:
    - Master Dataset → available trading days (+ optional row counts)
    - ``prediction_metadata.json`` → status / prediction_rows / notes

    Does **not** scan ``prediction_dataset``, run COUNT/GROUP BY, or upsert
    ``prediction_day_metadata``. Refresh Days rebuilds the sidecar from the DB.
    """
    empty = {
        "total_days": 0,
        "completed": 0,
        "remaining": 0,
        "failed": 0,
        "selected": 0,
        "days": [],
        "parent_dataset": None,
        "master_db_path": None,
        "rows_by_day": {},
    }
    if not os.path.isfile(lab_db_path):
        return empty

    from .prediction_dataset_type import (
        load_master_day_row_counts,
        resolve_master_db_path_for_lab,
    )
    from .prediction_metadata import (
        merge_master_and_metadata,
        read_prediction_metadata,
    )

    # Lab info only — resolve Master path / parent name. No day-catalog queries.
    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        if lab is None:
            return empty
        extra_master: list[str] = []
        pred_sum = store.read_prediction_summary() or {}
        if pred_sum.get("master_db_path"):
            extra_master.append(str(pred_sum.get("master_db_path")))
        try:
            row = store.conn.execute(
                "SELECT master_db_path FROM model_lab_info WHERE id = 1"
            ).fetchone()
            if row and row[0]:
                extra_master.append(str(row[0]))
        except Exception:
            pass
        parent_dataset = str(pred_sum.get("parent_dataset") or "") or _dataset_name_from_lab(lab)

    master_path = resolve_master_db_path_for_lab(
        lab,
        data_dir=data_dir,
        parent_meta=None,
        extra_paths=extra_master,
    )
    master_rows = load_master_day_row_counts(master_path) if master_path else {}
    meta_doc = read_prediction_metadata(lab_db_path)
    days_out = merge_master_and_metadata(master_rows, meta_doc)

    completed = sum(1 for d in days_out if str(d.get("status") or "") == "completed")
    rows_by_day = {
        str(d.get("trading_day")): int(d.get("rows_expected") or 0)
        for d in days_out
        if d.get("trading_day") and int(d.get("rows_expected") or 0) > 0
    }
    return {
        "total_days": len(days_out),
        "completed": completed,
        "remaining": max(0, len(days_out) - completed),
        "failed": sum(1 for d in days_out if str(d.get("status") or "") == "failed"),
        "selected": sum(1 for d in days_out if d.get("selected")),
        "days": days_out,
        "parent_dataset": parent_dataset,
        "master_db_path": master_path,
        "rows_by_day": rows_by_day,
        "dataset_type": None,
    }


def enrich_prediction_day_metadata(
    data_dir: str,
    lab_db_path: str,
    trading_day: str,
) -> dict[str, Any]:
    """
    Load expensive per-day Trading Days metadata for one day (reuse if already ready).

    Sets rows_expected, dataset_type, and prediction row_count for ``trading_day``.
    Does not change build/compute pipelines — catalog bookkeeping only.
    """
    day = str(trading_day or "").strip()
    if not day:
        return {"ok": False, "error": "trading_day is required"}

    from .prediction_dataset_type import (
        build_day_dataset_types,
        load_master_day_row_counts,
        resolve_master_db_path_for_lab,
        resolve_model_master_filter,
        resolve_model_seen_trading_days,
    )
    from .prediction_feature_store import count_trading_day_rows_in_master
    from .prediction_io import (
        catalog_trading_days,
        dataset_row_counts_from_meta,
        load_dataset_meta,
        load_parent_dataset_row_counts,
        resolve_dataset_parquet,
    )

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        lab = store.read_info()
        if lab is None:
            return {"ok": False, "error": "Model Lab info not found"}
        lab_uuid = lab.lab_uuid
        existing = next(
            (d for d in store.list_build_days(lab_uuid) if str(d.get("trading_day")) == day),
            None,
        )
        if existing and _day_ui_meta_ready(existing):
            # Refresh cheap single-day pred count in case builds finished.
            pred_n = store.prediction_row_count_for_day(day)
            if pred_n != int(existing.get("row_count") or 0):
                store.set_build_day_status(
                    lab_uuid,
                    day,
                    status=str(existing.get("status") or "waiting"),
                    row_count=pred_n,
                )
                existing = dict(existing)
                existing["row_count"] = pred_n
            existing["ui_meta_ready"] = True
            return {"ok": True, "day": existing, "reused": True}

        extra_master: list[str] = []
        pred_sum = store.read_prediction_summary() or {}
        if pred_sum.get("master_db_path"):
            extra_master.append(str(pred_sum.get("master_db_path")))
        try:
            row = store.conn.execute(
                "SELECT master_db_path FROM model_lab_info WHERE id = 1"
            ).fetchone()
            if row and row[0]:
                extra_master.append(str(row[0]))
        except Exception:
            pass

    check = validate_prediction_inputs(data_dir, lab)
    dataset = str(check.get("dataset") or "") if check.get("ok") else _dataset_name_from_lab(lab)
    parent_days: list[str] = []
    row_counts: dict[str, int] = {}
    meta: dict[str, Any] = {}
    if dataset:
        try:
            parquet_path, meta_path = resolve_dataset_parquet(data_dir, dataset)
            meta = load_dataset_meta(meta_path)
            parent_days = list(catalog_trading_days(parquet_path, meta=meta))
            row_counts = dataset_row_counts_from_meta(meta)
        except (DatasetLoaderError, Exception):
            parent_days = _parent_days_for_lab(lab, lab_db_path=lab_db_path, data_dir=data_dir)
            if check.get("ok") and dataset:
                try:
                    row_counts = load_parent_dataset_row_counts(data_dir, dataset)
                except Exception:
                    row_counts = {}

    master_path = resolve_master_db_path_for_lab(
        lab,
        data_dir=data_dir,
        parent_meta=meta or None,
        extra_paths=extra_master,
    )
    master_filter = resolve_model_master_filter(lab, parent_meta=meta or None)
    master_rows = load_master_day_row_counts(master_path) if master_path else {}

    expected = int(row_counts.get(day) or 0)
    if expected <= 0 and master_path:
        if master_filter:
            expected = int(
                count_trading_day_rows_in_master(
                    master_path, day, master_filter=master_filter
                )
                or 0
            )
        if expected <= 0:
            expected = int(master_rows.get(day) or 0)

    seen_days = resolve_model_seen_trading_days(lab, parent_trading_days=parent_days or None)
    day_type = build_day_dataset_types([day], seen_days).get(day)

    with ModelLabStore(lab_db_path) as store:
        store.ensure_build_days(
            lab_uuid,
            [day],
            day_dataset_types={day: day_type} if day_type else None,
            sync_pred_counts=False,
        )
        pred_n = store.prediction_row_count_for_day(day)
        if pred_n > 0:
            from .prediction_schema import resolve_day_completion_status

            status = resolve_day_completion_status(pred_n, expected if expected > 0 else None)
        else:
            status = "waiting"
        store.set_build_day_status(
            lab_uuid,
            day,
            status=status,
            row_count=pred_n,
            rows_expected=expected if expected > 0 else None,
            progress_pct=100.0 if pred_n > 0 else None,
            finished=pred_n > 0,
        )
        if day_type:
            store.apply_day_dataset_types(lab_uuid, {day: day_type})
        if expected > 0:
            store.set_day_rows_expected(lab_uuid, day, expected)
        enriched = next(
            (d for d in store.list_build_days(lab_uuid) if str(d.get("trading_day")) == day),
            None,
        )
    if not enriched:
        return {"ok": False, "error": f"Could not enrich day {day}"}
    enriched = dict(enriched)
    enriched["ui_meta_ready"] = True
    return {
        "ok": True,
        "day": enriched,
        "reused": False,
        "dataset": dataset,
        "rows_by_day": {day: expected} if expected > 0 else {},
    }


def parent_registry_day_rows(
    data_dir: str,
    lab_db_path: str,
) -> dict[str, Any]:
    """
    Fast parent dataset day row counts from registry JSON ``sources[].rows``.

    No parquet scan. Used by the Trading Days UI for Dataset rows.
    """
    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        if lab is None:
            return {"ok": False, "error": "Model Lab info not found", "rows_by_day": {}}
        check = validate_prediction_inputs(data_dir, lab)
        if not check["ok"]:
            return {
                "ok": False,
                "error": "; ".join(check["errors"]),
                "rows_by_day": {},
                "dataset": check.get("dataset"),
            }
        dataset = str(check["dataset"])
    try:
        rows_by_day = load_parent_dataset_row_counts(data_dir, dataset)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "rows_by_day": {},
            "dataset": dataset,
        }
    return {
        "ok": True,
        "dataset": dataset,
        "rows_by_day": rows_by_day,
    }


def sync_prediction_build_catalog(
    data_dir: str,
    lab_db_path: str,
    *,
    selected_days: list[str] | None = None,
) -> dict[str, Any]:
    """
    Discover trading days from parent registry meta (prefer days/sources) and
    upsert prediction_day_metadata. Also writes rows_expected from sources[].rows.

    Merges Master Dataset trading days when a master DB is linked so Unseen days
    can appear even when they were never exported into the training parquet.

    Dataset Type is per model: Seen = days used in this model's training /
    walk-forward validation metadata; Unseen = catalog days never used by it.

    Parquet day-column scan only if registry meta has no day list.
    Normal UI open/refresh must read prediction_day_metadata (+ registry rows)
    and must not call this.
    """
    from .prediction_dataset_type import (
        build_day_dataset_types,
        load_master_day_row_counts,
        resolve_master_db_path_for_lab,
        resolve_model_master_filter,
        resolve_model_seen_trading_days,
    )
    from .prediction_feature_store import count_trading_day_rows_in_master

    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        if lab is None:
            return {"ok": False, "error": "Model Lab info not found"}
        store.ensure_prediction_schema()
        check = validate_prediction_inputs(data_dir, lab)
        if not check["ok"]:
            return {
                "ok": False,
                "error": "; ".join(check["errors"]),
                "summary": store.build_summary(lab.lab_uuid),
            }
        dataset = str(check["dataset"])
        lab_uuid = lab.lab_uuid
        extra_master: list[str] = []
        pred_sum = store.read_prediction_summary() or {}
        if pred_sum.get("master_db_path"):
            extra_master.append(str(pred_sum.get("master_db_path")))
        try:
            row = store.conn.execute(
                "SELECT master_db_path FROM model_lab_info WHERE id = 1"
            ).fetchone()
            if row and row[0]:
                extra_master.append(str(row[0]))
        except Exception:
            pass

    try:
        parquet_path, meta_path = resolve_dataset_parquet(data_dir, dataset)
        meta = load_dataset_meta(meta_path)
        parent_days = list(catalog_trading_days(parquet_path, meta=meta))
        row_counts = dataset_row_counts_from_meta(meta)
    except DatasetLoaderError as exc:
        return {"ok": False, "error": str(exc)}

    master_path = resolve_master_db_path_for_lab(
        lab,
        data_dir=data_dir,
        parent_meta=meta,
        extra_paths=extra_master,
    )
    master_filter = resolve_model_master_filter(lab, parent_meta=meta)
    master_rows = load_master_day_row_counts(master_path) if master_path else {}
    day_set = {str(d).strip() for d in parent_days if str(d).strip()}
    for day, n in master_rows.items():
        d = str(day).strip()
        if not d:
            continue
        day_set.add(d)
        # Prefer parent registry counts for Seen parent days; for Master-only /
        # missing days use THIS model's training filter row count.
        if d not in row_counts or int(row_counts.get(d) or 0) <= 0:
            if master_path and master_filter:
                filtered_n = count_trading_day_rows_in_master(
                    master_path, d, master_filter=master_filter
                )
                row_counts[d] = int(filtered_n or n or 0)
            else:
                row_counts[d] = int(n or 0)
    days = sorted(day_set)

    # Parent export days = Seen; Master-only extras (e.g. newer days) = Unseen.
    seen_days = resolve_model_seen_trading_days(lab, parent_trading_days=parent_days)
    day_types = build_day_dataset_types(days, seen_days)

    sel = set(selected_days) if selected_days is not None else None
    with ModelLabStore(lab_db_path) as store:
        store.ensure_build_days(
            lab_uuid,
            days,
            selected=sel,
            day_dataset_types=day_types,
        )
        if selected_days is not None:
            store.set_days_selected(lab_uuid, list(selected_days))
        for day, n in row_counts.items():
            if day and int(n or 0) > 0:
                store.set_day_rows_expected(
                    lab_uuid, str(day), int(n), sync_ui_meta=False
                )
        # Types already written by ensure_build_days; skip per-day JSON until rebuild.
        summary = store.build_summary(lab_uuid)

    from .prediction_metadata import rebuild_prediction_metadata_from_db

    # Refresh Days: rebuild UI sidecar from verified DB catalog (+ pred counts).
    meta_doc = rebuild_prediction_metadata_from_db(lab_db_path)
    # Prefer sidecar-shaped days for UI when available.
    if meta_doc.get("days") and summary.get("days"):
        from .prediction_metadata import merge_master_and_metadata

        # Keep DB summary counts; replace day rows with merged master+meta view.
        master_for_merge = {
            str(d.get("trading_day")): int(d.get("rows_expected") or row_counts.get(str(d.get("trading_day")), 0) or 0)
            for d in (summary.get("days") or [])
            if d.get("trading_day")
        }
        for day, n in row_counts.items():
            master_for_merge[str(day)] = int(n or master_for_merge.get(str(day), 0) or 0)
        summary = dict(summary)
        summary["days"] = merge_master_and_metadata(master_for_merge, meta_doc)

    for d in summary.get("days") or []:
        if isinstance(d, dict):
            d["ui_meta_ready"] = True
    return {
        "ok": True,
        "days": days,
        "summary": summary,
        "dataset": dataset,
        "rows_by_day": row_counts,
        "seen_days": sorted(seen_days),
        "master_db_path": master_path,
    }


def prediction_build_summary(
    lab_db_path: str,
    *,
    data_dir: str | None = None,
    light: bool = False,
) -> dict[str, Any]:
    """
    Read Trading Days UI rows from Lab DB prediction_day_metadata (no parent scan).

    If rows already exist in prediction_dataset but metadata is thin, upsert day
    keys from committed days only (still Lab DB — not parent parquet).

    Re-applies per-model Seen/Unseen from training metadata onto existing catalog
    days when seen-day labels (or parent export days) are available.

    ``light=True``: catalog-only read (no prediction_dataset GROUP BY, no Seen
    re-label). Used for fast post-action UI refresh.
    """
    if not os.path.isfile(lab_db_path):
        return {
            "total_days": 0,
            "completed": 0,
            "remaining": 0,
            "failed": 0,
            "selected": 0,
            "days": [],
        }
    from .prediction_dataset_type import (
        build_day_dataset_types,
        resolve_model_seen_trading_days,
    )

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        lab = store.read_info()
        if lab is None:
            return {
                "total_days": 0,
                "completed": 0,
                "remaining": 0,
                "failed": 0,
                "selected": 0,
                "days": [],
            }
        if light:
            summary = store.build_summary(lab.lab_uuid)
            for d in summary.get("days") or []:
                if isinstance(d, dict):
                    d["ui_meta_ready"] = _day_ui_meta_ready(d)
            return summary
        # Seed from committed prediction rows so Completed days appear immediately
        counts = store.prediction_row_counts_by_day()
        if counts:
            store.ensure_build_days(lab.lab_uuid, sorted(counts.keys()))
        # Refresh Seen/Unseen from this model's training metadata (no Master merge).
        catalog_days = [
            str(d.get("trading_day") or "")
            for d in store.list_build_days(lab.lab_uuid)
            if str(d.get("trading_day") or "").strip()
        ]
        parent_days = _parent_days_for_lab(
            lab, lab_db_path=lab_db_path, data_dir=data_dir
        )
        seen_days = resolve_model_seen_trading_days(
            lab, parent_trading_days=parent_days or None
        )
        if catalog_days and seen_days:
            store.apply_day_dataset_types(
                lab.lab_uuid,
                build_day_dataset_types(catalog_days, seen_days),
            )
        summary = store.build_summary(lab.lab_uuid)
        for d in summary.get("days") or []:
            if isinstance(d, dict):
                d["ui_meta_ready"] = _day_ui_meta_ready(d)
        return summary


def build_prediction_dataset(
    data_dir: str,
    lab_db_path: str,
    *,
    overwrite: bool = False,
    resume: bool = True,
    selected_days: list[str] | None = None,
    enrich_path_outcomes: bool = True,
    workers: int = DEFAULT_PREDICTION_WORKERS,
    on_progress: ProgressFn | None = None,
    print_timing: bool = True,
    control: BuildControl | None = None,
    row_limit: int | None = None,
    mark_day_complete: bool = True,
    tb_model_name: str | None = None,
) -> dict[str, Any]:
    """
    Run parent model on parent dataset (selected features only).

    Unit of work is one trading day. Completed days in prediction_day_metadata
    are skipped on resume. Pause/Cancel finish the current day, then stop.

    row_limit: if set (e.g. 1000 for Test), only the first N valid rows per day.
    mark_day_complete: False for test samples so full Start can still run later.
    """
    t0 = time.perf_counter()
    ctrl = control or BuildControl()
    ctrl.reset()
    set_active_build_control(ctrl)

    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        if lab is None:
            set_active_build_control(None)
            return {"ok": False, "error": "Model Lab info not found"}
        store.ensure_prediction_schema()
        existing = store.prediction_row_count()
        # Legacy: overwrite=False + resume=False blocks when rows exist
        # (Test samples use row_limit and are allowed to append/replace a day.)
        if existing > 0 and not overwrite and not resume and row_limit is None:
            set_active_build_control(None)
            return {
                "ok": False,
                "error": "Prediction Dataset already exists.",
                "code": "exists",
                "row_count": existing,
            }

        check = validate_prediction_inputs(data_dir, lab)
        if not check["ok"]:
            store.write_prediction_summary(
                lab_uuid=lab.lab_uuid,
                status=PRED_STATUS_ERROR,
                row_count=0,
                trading_days=0,
                error_message="; ".join(check["errors"]),
                parent_model_name=lab.parent_model_name,
                parent_dataset=check.get("dataset") or None,
                target_column=lab.target,
                selected_feature_count=len(check.get("features") or []),
            )
            set_active_build_control(None)
            return {"ok": False, "error": "; ".join(check["errors"]), "warnings": check["warnings"]}

        features: list[str] = list(check["features"])
        target = str(check["target"])
        dataset = str(check["dataset"])
        model_name = str(check["model_name"])
        model_path = str(check["model_path"])
        store.write_prediction_summary(
            lab_uuid=lab.lab_uuid,
            status=PRED_STATUS_BUILDING,
            row_count=existing if (resume and not overwrite) else 0,
            trading_days=0,
            parent_model_name=model_name,
            parent_dataset=dataset,
            target_column=target,
            selected_feature_count=len(features),
            created_at=_utc_now(),
        )
        lab_uuid = lab.lab_uuid
        algorithm = lab.algorithm

    hub = ProgressHub(on_progress, started_at=t0)
    hub.start()
    days: list[str] = []
    feat_map: dict[str, str] = {}
    parallel: dict[str, Any] = {}
    storage_mode = "embedded"
    master_id: str | None = None
    master_rel_store: str | None = None
    try:
        hub.update(
            stage=STAGE_LOAD_MODEL,
            stage_label=stage_label(STAGE_LOAD_MODEL),
            stage_detail="Opening parent model artifact…",
            worker_count=workers,
            samples_done=0,
            samples_total=0,
            trading_days_done=0,
            trading_days_total=0,
        )

        try:
            parquet_path, meta_path = resolve_dataset_parquet(data_dir, dataset)
            hub.update(
                stage=STAGE_READ_METADATA,
                stage_label=stage_label(STAGE_READ_METADATA),
                stage_detail="Reading dataset JSON + parquet footer…",
            )
            meta = load_dataset_meta(meta_path)
            total_estimate = estimate_sample_total(parquet_path, meta=meta)
            hub.update(
                stage=STAGE_CATALOG_DAYS,
                stage_label=stage_label(STAGE_CATALOG_DAYS),
                stage_detail="Discovering trading days…",
                samples_total=total_estimate,
            )

            def _catalog_prog(p: dict[str, Any]) -> None:
                hub.update(stage=STAGE_CATALOG_DAYS, **p)

            days = catalog_trading_days(parquet_path, meta=meta, on_progress=_catalog_prog)
            hub.update(
                stage=STAGE_CATALOG_DAYS,
                stage_detail=f"✓ {len(days)} trading days found",
                trading_days_total=len(days),
                samples_total=total_estimate,
            )
        except DatasetLoaderError as exc:
            with ModelLabStore(lab_db_path) as store:
                store.write_prediction_summary(
                    lab_uuid=lab_uuid,
                    status=PRED_STATUS_ERROR,
                    row_count=0,
                    trading_days=0,
                    error_message=str(exc),
                    parent_model_name=model_name,
                    parent_dataset=dataset,
                    target_column=target,
                    selected_feature_count=len(features),
                )
            return {"ok": False, "error": str(exc)}

        if not days:
            return {"ok": False, "error": "No trading days found in parent dataset"}

        from .prediction_dataset_type import resolve_model_master_filter

        master_filter = resolve_model_master_filter(lab, parent_meta=meta)

        hub.update(
            stage=STAGE_LOAD_MODEL,
            stage_label=stage_label(STAGE_LOAD_MODEL),
            stage_detail="Loading model into memory…",
            trading_days_total=len(days),
            samples_total=total_estimate,
            worker_count=workers,
        )

        probe, _ms, _disk = load_prediction_model_cached(model_path, algorithm)
        features = align_features_to_model(features, probe)
        from chain_replay_ml.training.dataset_loader import parquet_column_names
        from .prediction_feature_store import (
            detect_feature_storage_mode,
            ensure_master_row_id_light,
            master_dataset_id_from_path,
            master_has_row_id_column,
            referenced_feature_column_map,
        )
        from .prediction_schema import (
            FEATURE_STORAGE_EMBEDDED,
            FEATURE_STORAGE_REFERENCED,
        )

        pq_cols = parquet_column_names(parquet_path) or set()
        master_rel = (
            str(meta.get("master_db_path") or "").strip()
            or str((lab.dataset_snapshot or {}).get("master_db_path") or "").strip()
            or None
        )
        storage_mode, master_abs = detect_feature_storage_mode(
            parquet_columns=pq_cols,
            master_db_path=master_rel,
            data_dir=data_dir,
        )
        if storage_mode == FEATURE_STORAGE_REFERENCED and master_abs:
            # Lightweight ensure — do NOT open MasterStore (conflicts with UI lock).
            ensured = ensure_master_row_id_light(master_abs)
            if not ensured.get("ok"):
                if master_has_row_id_column(master_abs):
                    ensured = {"ok": True, "has_column": True, "wrote": False}
                else:
                    # Master locked / missing IDs → safe legacy embed for this build
                    storage_mode = FEATURE_STORAGE_EMBEDDED
                    hub.update(
                        stage=STAGE_LOAD_MODEL,
                        stage_detail=(
                            "Master DB busy/locked — using embedded features for this build. "
                            f"({ensured.get('error') or 'no master_row_id'})"
                        ),
                    )
        embed_features = storage_mode == FEATURE_STORAGE_EMBEDDED
        if embed_features:
            feat_map = feature_column_map(features)
        else:
            feat_map = referenced_feature_column_map(features)
        master_rel_store = None
        if master_abs:
            # Prefer absolute path so Research Lab can resolve after moves of data_dir.
            master_rel_store = os.path.abspath(master_abs).replace("\\", "/")
        elif master_rel:
            master_rel_store = master_rel.replace("\\", "/")
        master_id = master_dataset_id_from_path(master_abs or master_rel)
        stamp_ids = (
            storage_mode == FEATURE_STORAGE_REFERENCED
            and "master_row_id" not in pq_cols
            and bool(master_abs)
        )

        del probe
        hub.update(
            stage=STAGE_LOAD_MODEL,
            stage_detail=(
                f"✓ Model ready · {len(features)} features · {workers} worker(s)"
                f" · storage={storage_mode}"
            ),
        )

        # Prediction-package members: probability-ladder classifiers predicted
        # in the same feature pass (union of member features, one day load).
        from chain_replay_ml.training.prediction_packages import (
            discover_prediction_package_members,
            is_package_anchor_target,
            package_members_summary,
        )

        package_members: list[dict[str, Any]] = []
        if is_package_anchor_target(target):
            package_members = discover_prediction_package_members(
                data_dir,
                dataset=dataset,
                anchor_target=target,
                anchor_model_name=model_name,
            )
            hub.update(
                stage=STAGE_LOAD_MODEL,
                stage_detail=(
                    f"Prediction package · {package_members_summary(package_members)}"
                ),
            )

        wanted = list(dict.fromkeys([*features, target, *_IDENTITY_LOAD]))
        member_feature_union = sorted({
            str(f)
            for member in package_members
            if member.get("available")
            for f in (member.get("features") or [])
        })
        wanted.extend(c for c in member_feature_union if c in pq_cols and c not in wanted)
        if tb_model_name:
            try:
                from chain_replay_ml.training.model_runtime import (
                    resolve_prediction_model_package,
                )

                # Merge unconditionally (like the primary model's own
                # ``features``, not gated on raw pq_cols membership) — TB
                # features may only materialize after Master transforms.
                # Gating on pq_cols here would silently drop those columns
                # from wanted_columns, so the day frame never carries them
                # and TB scoring degrades to NULL even when it could run.
                tb_pkg = resolve_prediction_model_package(data_dir, tb_model_name)
                if tb_pkg.get("ok"):
                    for tf in tb_pkg.get("features") or []:
                        tf_s = str(tf).strip()
                        if tf_s and tf_s not in wanted:
                            wanted.append(tf_s)
            except Exception:
                pass
        if "master_row_id" in pq_cols:
            wanted.append("master_row_id")

        from .prediction_transformations import (
            expand_columns_for_master_load,
            sample_interval_sec_from_meta,
            transformation_config_from_dataset_meta,
        )

        transformation_config = transformation_config_from_dataset_meta(meta)
        sample_interval_sec = sample_interval_sec_from_meta(meta)
        wanted = expand_columns_for_master_load(wanted, transformation_config)
        sel_set = set(selected_days) if selected_days is not None else None
        catalog_days = list(days)
        if selected_days:
            for d in selected_days:
                ds = str(d or "").strip()
                if ds and ds not in catalog_days:
                    catalog_days.append(ds)
        with ModelLabStore(lab_db_path) as store:
            if overwrite:
                store.clear_prediction_dataset()
            store.ensure_prediction_schema()
            if embed_features:
                store.ensure_feature_columns(list(feat_map.values()))
            store.ensure_build_days(lab_uuid, catalog_days, selected=sel_set)
            if selected_days is not None:
                store.set_days_selected(lab_uuid, list(selected_days))
            # When row_limit (test sample): force selected days only; do not mark complete
            if row_limit is not None and int(row_limit) > 0:
                mark_day_complete = False
                if selected_days:
                    days_to_run = [str(d) for d in selected_days if str(d or "").strip()]
                else:
                    days_to_run = [
                        d["trading_day"]
                        for d in store.list_build_days(lab_uuid)
                        if d.get("selected")
                    ]
            elif resume and not overwrite:
                if selected_days is not None:
                    # See prediction_job_prepare.prepare_prediction_exec_config for
                    # why this bypasses the shared ``selected`` column: it is
                    # rewritten wholesale (all *other* days deselected) by every
                    # single-day build, so gating on it here can silently drop a
                    # day that was just explicitly requested.
                    pending_all = store.pending_build_days(lab_uuid, selected_only=False)
                    wanted_sel = {
                        str(d).strip() for d in selected_days if str(d or "").strip()
                    }
                    days_to_run = [d for d in pending_all if d in wanted_sel]
                    if tb_model_name:
                        tb_stale = store.days_needing_tb_rescore(
                            lab_uuid, sorted(wanted_sel), tb_model_name
                        )
                        days_to_run.extend(d for d in tb_stale if d not in days_to_run)
                else:
                    days_to_run = store.pending_build_days(lab_uuid, selected_only=True)
                    if tb_model_name:
                        candidate_days = [
                            str(d["trading_day"])
                            for d in store.list_build_days(lab_uuid)
                            if d.get("selected")
                        ]
                        tb_stale = store.days_needing_tb_rescore(
                            lab_uuid, candidate_days, tb_model_name
                        )
                        days_to_run.extend(d for d in tb_stale if d not in days_to_run)
            else:
                if selected_days:
                    days_to_run = [
                        str(d) for d in selected_days if str(d or "").strip()
                    ]
                else:
                    days_to_run = [
                        d["trading_day"]
                        for d in store.list_build_days(lab_uuid)
                        if d.get("selected")
                    ]
            store.write_prediction_summary(
                lab_uuid=lab_uuid,
                status=PRED_STATUS_BUILDING,
                row_count=store.prediction_row_count(),
                trading_days=len(catalog_days),
                parent_model_name=model_name,
                parent_dataset=dataset,
                target_column=target,
                selected_feature_count=len(features),
                feature_columns_json=json.dumps(feat_map, ensure_ascii=False),
                feature_storage_mode=storage_mode,
                master_dataset_id=master_id,
                master_db_path=master_rel_store,
                created_at=_utc_now(),
            )

        if not days_to_run:
            with ModelLabStore(lab_db_path) as store:
                n = store.prediction_row_count()
                summary = store.build_summary(lab_uuid)
                store.write_prediction_summary(
                    lab_uuid=lab_uuid,
                    status=PRED_STATUS_READY if n > 0 else PRED_STATUS_ERROR,
                    row_count=n,
                    trading_days=len(days),
                    start_day=days[0] if days else None,
                    end_day=days[-1] if days else None,
                    parent_model_name=model_name,
                    parent_dataset=dataset,
                    target_column=target,
                    selected_feature_count=len(features),
                    feature_columns_json=json.dumps(feat_map, ensure_ascii=False),
                    feature_storage_mode=storage_mode,
                    master_dataset_id=master_id,
                    master_db_path=master_rel_store,
                    created_at=_utc_now(),
                    error_message=None if n > 0 else "No days selected to build",
                )
            return {
                "ok": n > 0,
                "row_count": n,
                "trading_days": len(days),
                "days_processed": [],
                "stopped": "none",
                "already_complete": True,
                "build_summary": summary,
                "error": None if n > 0 else "No days selected to build",
            }

        ctx = DayJobContext(
            data_dir=data_dir,
            parquet_path=parquet_path,
            features=features,
            target=target,
            wanted_columns=wanted,
            lab_uuid=lab_uuid,
            feat_map=feat_map,
            horizon_sec=horizon_sec_from_target(target),
            enrich_path_outcomes=enrich_path_outcomes,
            model_path=model_path,
            algorithm=algorithm,
            days=days,
            row_limit=int(row_limit) if row_limit else None,
            mark_day_complete=bool(mark_day_complete),
            embed_features=embed_features,
            master_db_path=master_abs,
            stamp_master_row_ids=stamp_ids,
            master_filter=master_filter or None,
            package_members=package_members,
            transformation_config=transformation_config,
            sample_interval_sec=sample_interval_sec,
            parent_dataset=dataset,
            tb_model_name=tb_model_name,
        )

        parallel = run_parallel_day_build(
            ctx=ctx,
            lab_db_path=lab_db_path,
            feature_columns=list(feat_map.values()) if embed_features else [],
            workers=workers,
            total_estimate=total_estimate,
            on_progress=on_progress,
            hub=hub,
            days_to_run=days_to_run,
            control=ctrl,
        )
    finally:
        hub.stop()
        set_active_build_control(None)

    run_rows = int(parallel.get("row_count") or 0)
    failed_days = list(parallel.get("failed_days") or [])
    timing = dict(parallel.get("timing") or {})
    stopped = str(parallel.get("stopped") or "none")
    n_abs = int(parallel.get("n_abs") or 0)
    n_err = int(parallel.get("n_err") or 0)
    n_prem = int(parallel.get("n_prem") or 0)
    dir_n = int(parallel.get("dir_n") or 0)
    avg_abs = (float(parallel["sum_abs"]) / n_abs) if n_abs else None
    avg_err = (float(parallel["sum_err"]) / n_err) if n_err else None
    avg_prem = (float(parallel["sum_prem"]) / n_prem) if n_prem else None
    dir_acc = (float(parallel["dir_hits"]) / dir_n) if dir_n else None
    ds_hash = str(parallel.get("dataset_hash") or "")
    elapsed = float(parallel.get("generation_time_sec") or (time.perf_counter() - t0))

    err_msg = None
    if failed_days:
        err_msg = (
            f"{len(failed_days)} trading day(s) failed: "
            + ", ".join(d.get("trading_day", "?") for d in failed_days[:8])
        )

    with ModelLabStore(lab_db_path) as store:
        done = store.prediction_row_count()
        build_sum = store.build_summary(lab_uuid)
        if stopped == "paused":
            status = PRED_STATUS_PAUSED
        elif stopped == "cancelled":
            status = PRED_STATUS_PAUSED if done > 0 else PRED_STATUS_ERROR
        elif done > 0 and build_sum.get("remaining", 0) == 0 and not failed_days:
            status = PRED_STATUS_READY
        elif done > 0:
            status = PRED_STATUS_READY if not failed_days else PRED_STATUS_ERROR
        else:
            status = PRED_STATUS_ERROR

        store.write_prediction_summary(
            lab_uuid=lab_uuid,
            status=status,
            row_count=done,
            trading_days=len(days),
            start_day=days[0] if days else None,
            end_day=days[-1] if days else None,
            average_error=round(avg_err, 6) if avg_err is not None else None,
            average_absolute_error=round(avg_abs, 6) if avg_abs is not None else None,
            premium_error=round(avg_prem, 6) if avg_prem is not None else None,
            direction_accuracy=round(dir_acc, 6) if dir_acc is not None else None,
            generation_time_sec=round(elapsed, 2),
            dataset_hash=ds_hash,
            selected_feature_count=len(features),
            feature_columns_json=json.dumps(feat_map, ensure_ascii=False),
            parent_model_name=model_name,
            parent_dataset=dataset,
            target_column=target,
            feature_storage_mode=storage_mode,
            master_dataset_id=master_id,
            master_db_path=master_rel_store,
            created_at=_utc_now(),
            error_message=err_msg,
        )
        if done > 0:
            store.update_lab_phase(
                phase=LAB_PHASE_PREDICTION,
                lab_schema_version=LAB_SCHEMA_VERSION_PREDICTION,
            )

    # Confidence inference over Prediction Dataset is now Out of Date
    if done > 0:
        try:
            from chain_replay_ml.model_lab.confidence_inference import (
                mark_stale_on_prediction_rebuild,
            )

            mark_stale_on_prediction_rebuild(lab_db_path)
        except Exception:
            pass

    # Precompute Research Dashboard summary tables (instant UI load)
    if done > 0:
        try:
            from chain_replay_ml.model_lab.research_dashboard import (
                refresh_research_dashboard_cache,
            )

            refresh_research_dashboard_cache(lab_db_path, force=True, data_dir=data_dir)
        except Exception:
            pass

    post = validate_prediction_output(lab_db_path, expected_rows=None)
    if done == 0 and stopped == "none":
        post = {"ok": False, "errors": ["No samples with non-null target"], "row_count": 0}
    elif stopped in ("paused", "cancelled") and done > 0:
        post = {"ok": True, "errors": [], "row_count": done}

    timing_text = ""
    if print_timing and timing:
        timing_text = _print_timing_summary(timing)

    _emit(
        on_progress,
        phase="done",
        status=status,
        stage=STAGE_FINISHED,
        stage_total=STAGE_TOTAL,
        stage_label=stage_label(STAGE_FINISHED),
        stage_detail=(
            f"Paused · {build_sum.get('completed', 0)}/{len(days)} days"
            if stopped == "paused"
            else (
                f"Cancelled · {build_sum.get('completed', 0)}/{len(days)} days"
                if stopped == "cancelled"
                else ("Complete" if post.get("ok") else "; ".join(post.get("errors") or []))
            )
        ),
        current_day=days[-1] if days else "",
        samples_done=done,
        samples_total=done,
        predictions_written=done,
        trading_days_total=len(days),
        trading_days_done=int(build_sum.get("completed") or 0),
        days_completed=int(build_sum.get("completed") or 0),
        days_remaining=int(build_sum.get("remaining") or 0),
        elapsed_sec=round(elapsed, 1),
        eta_sec=0.0,
        percent=100.0 if status == PRED_STATUS_READY and build_sum.get("remaining", 0) == 0 else None,
        worker_count=int(parallel.get("workers_used") or workers),
        workers=[],
        failed_days=failed_days,
        message="Complete" if post.get("ok") else "; ".join(post.get("errors") or []),
        timing=timing,
        stopped=stopped,
    )

    if not post.get("ok"):
        return {
            "ok": False,
            "error": "; ".join(post.get("errors") or [err_msg or "Generation failed"]),
            "row_count": done,
            "dataset_hash": ds_hash,
            "generation_time_sec": round(elapsed, 2),
            "failed_days": failed_days,
            "timing": timing,
            "timing_text": timing_text,
            "workers_used": parallel.get("workers_used"),
            "stopped": stopped,
            "days_processed": parallel.get("days_processed") or [],
            "build_summary": build_sum,
            "rows_this_run": run_rows,
        }

    return {
        "ok": True,
        "row_count": done,
        "trading_days": len(days),
        "dataset_hash": ds_hash,
        "generation_time_sec": round(elapsed, 2),
        "warnings": list(check.get("warnings") or []),
        "selected_feature_count": len(features),
        "parent_model_name": model_name,
        "parent_dataset": dataset,
        "target_column": target,
        "failed_days": failed_days,
        "timing": timing,
        "timing_text": timing_text,
        "workers_used": parallel.get("workers_used"),
        "stopped": stopped,
        "days_processed": parallel.get("days_processed") or [],
        "build_summary": build_sum,
        "rows_this_run": run_rows,
    }


def benchmark_prediction_workers(
    data_dir: str,
    lab_db_path: str,
    *,
    worker_counts: tuple[int, ...] = (1, 3),
    enrich_path_outcomes: bool = False,
) -> dict[str, Any]:
    """
    Run prediction generation for each worker count (overwrite each time).
    Prints total runtime, avg/day, and speedup vs the first (baseline) run.
    """
    runs: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for n in worker_counts:
        print(f"\n=== Benchmark workers={n} ===", flush=True)
        result = build_prediction_dataset(
            data_dir,
            lab_db_path,
            overwrite=True,
            enrich_path_outcomes=enrich_path_outcomes,
            workers=int(n),
            print_timing=False,
        )
        timing = dict(result.get("timing") or {})
        if baseline is None:
            baseline = timing
            _print_timing_summary(timing)
        else:
            _print_timing_summary(timing, baseline=baseline)
        runs.append({
            "workers": n,
            "ok": result.get("ok"),
            "row_count": result.get("row_count"),
            "timing": timing,
            "error": result.get("error"),
        })

    speedup = None
    if len(runs) >= 2:
        t0 = (runs[0].get("timing") or {}).get("total_runtime_sec")
        t1 = (runs[-1].get("timing") or {}).get("total_runtime_sec")
        if t0 and t1:
            try:
                speedup = round(float(t0) / float(t1), 3)
            except (TypeError, ValueError, ZeroDivisionError):
                speedup = None

    summary = {
        "ok": all(r.get("ok") for r in runs),
        "runs": runs,
        "speedup": speedup,
        "baseline_workers": worker_counts[0] if worker_counts else None,
        "compare_workers": worker_counts[-1] if worker_counts else None,
    }
    if speedup is not None:
        print(
            f"\nSpeedup ({worker_counts[-1]} vs {worker_counts[0]} workers): {speedup}x",
            flush=True,
        )
    return summary
