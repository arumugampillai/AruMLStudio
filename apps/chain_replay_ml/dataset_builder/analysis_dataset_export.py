"""Analysis Dataset export — Phase 1A Auto Feature Transformation.

Builds Registry Features ∪ Pipeline Features into one analysis dataset.
Does not prune, select, or invent new transformation rules.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .feature_sources_catalog import (
    FEATURE_SOURCE_PIPELINE,
    FEATURE_SOURCE_REGISTRY,
    feature_sources_catalog,
    pipeline_feature_names,
)
from .master_naming import resolve_master_db_path
from .master_registry_export import MasterRegistryExportError, create_master_registry_dataset

_IST = ZoneInfo("Asia/Kolkata")

ProgressFn = Callable[[dict[str, Any]], None]

STAGE_REGISTRY = "registry"
STAGE_BASELINE_PIPELINE = "baseline_pipeline"
STAGE_EXPERIMENTAL_PIPELINE = "experimental_pipeline"
STAGE_NO_NULL = "no_null"
STAGE_PREMIUM = "premium"
STAGE_FINALIZE = "finalize"

STAGE_DEFS = (
    {"id": STAGE_REGISTRY, "label": "Registry Features"},
    {"id": STAGE_BASELINE_PIPELINE, "label": "Baseline Pipeline Features"},
    {"id": STAGE_EXPERIMENTAL_PIPELINE, "label": "Experimental Pipeline Features"},
    {"id": STAGE_NO_NULL, "label": "No-Null Filter"},
    {"id": STAGE_PREMIUM, "label": "Premium Filter"},
    {"id": STAGE_FINALIZE, "label": "Dataset Finalization"},
)


def _now_label() -> str:
    return datetime.now(_IST).strftime("%H:%M:%S")


def _stage_states(
    active: str,
    *,
    include_registry: bool,
    include_baseline_pipeline: bool,
    include_experimental_pipeline: bool,
    no_null_data: bool = False,
    premium_enabled: bool = False,
) -> list[dict[str, Any]]:
    order: list[str] = []
    if include_registry:
        order.append(STAGE_REGISTRY)
    if include_baseline_pipeline:
        order.append(STAGE_BASELINE_PIPELINE)
    if include_experimental_pipeline:
        order.append(STAGE_EXPERIMENTAL_PIPELINE)
    if no_null_data:
        order.append(STAGE_NO_NULL)
    if premium_enabled:
        order.append(STAGE_PREMIUM)
    order.append(STAGE_FINALIZE)
    if active not in order:
        active = order[-2] if len(order) > 1 else order[0]
    states = []
    for sid in order:
        label = next(s["label"] for s in STAGE_DEFS if s["id"] == sid)
        if sid == active:
            mark = "running"
        elif order.index(sid) < order.index(active):
            mark = "done"
        else:
            mark = "pending"
        states.append({"id": sid, "label": label, "status": mark})
    return states


def create_analysis_dataset(
    data_dir: str,
    *,
    market: str = "NIFTY",
    interval_sec: int = 3,
    include_registry: bool = True,
    include_baseline_pipeline: bool = True,
    include_experimental_pipeline: bool = False,
    include_pipeline: bool | None = None,
    all_days: bool = True,
    selected_days: list[str] | None = None,
    trading_day: str | None = None,
    trading_day_filter: dict[str, Any] | None = None,
    master_db_path: str | None = None,
    dataset_name: str | None = None,
    dataset_kind: str = "analysis",
    transformation_config: dict[str, Any] | None = None,
    no_null_data: bool = False,
    pipeline_no_null_report: bool = False,
    premium_enabled: bool = False,
    premium_min: float | None = None,
    premium_max: float | None = None,
    pipeline_id: str | None = None,
    token: str | None = None,
    atm_band_filter: int | None = None,
    delta_enabled: bool = False,
    delta_min: float | None = None,
    delta_max: float | None = None,
    on_progress: ProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
    feature_project_id: str | None = None,
) -> dict[str, Any]:
    """Create an Analysis Dataset from selected feature sources.

    Phase 1A materialisation supports 3 independent sources:
    - Registry Features from the canonical catalogue.
    - Baseline Pipeline Features regenerated via the approved baseline config.
    - Experimental Pipeline Features from selected experimental pipeline.
    - Optional No-Null filter (same policy as Master Dataset export) runs
      *only after* Feature Transformations finish.
    - Optional Pipeline No-Null Report (diagnostics) streams to Activity Log.
    - Optional LTP premium filter (same band as Master) runs *after* No-Null.
    """
    if include_pipeline is not None:
        include_baseline_pipeline = bool(include_pipeline)

    if not include_registry and not include_baseline_pipeline and not include_experimental_pipeline:
        raise MasterRegistryExportError("Select at least one feature source")

    from .master_feature_project import (
        MasterFeatureProjectError,
        active_project_feature_names,
        normalize_feature_project_id,
        resolve_master_feature_project_id_for_path,
        validate_feature_project_id,
    )
    from .master_naming import resolve_master_db_path

    path = master_db_path or resolve_master_db_path(
        data_dir,
        market=str(market or "NIFTY").upper(),
        sampling_interval_sec=int(interval_sec),
    )
    if not os.path.isfile(path):
        raise MasterRegistryExportError("Master database file does not exist")
    if feature_project_id:
        bound_pid = validate_feature_project_id(
            data_dir,
            normalize_feature_project_id(feature_project_id),
        )
    else:
        try:
            bound_pid = resolve_master_feature_project_id_for_path(path, data_dir)
        except MasterFeatureProjectError as exc:
            raise MasterRegistryExportError(exc.detail) from exc

    from .pipeline_features_prefs import (
        load_pipeline_output_prune_features,
        load_retired_pipeline_features,
        load_transformation_forbidden_features,
    )

    retired = load_retired_pipeline_features(data_dir)
    output_exclude_for_pipeline = load_pipeline_output_prune_features(data_dir)
    forbidden = load_transformation_forbidden_features(data_dir)
    interaction_operand_skip = set(forbidden)
    catalog = feature_sources_catalog(
        data_dir=data_dir,
        retired=retired,
        feature_project_id=bound_pid,
    )
    project_features = active_project_feature_names(data_dir, bound_pid)
    from .registry_features_prefs import resolve_registry_export_features

    reg_export = resolve_registry_export_features(data_dir) if include_registry else frozenset()
    if include_registry and project_features:
        reg_export = frozenset(reg_export & project_features)
    reg_names = sorted(reg_export) if include_registry else []
    if include_registry and not reg_names:
        raise MasterRegistryExportError(
            "No Registry Features selected for export. "
            "Open “Click to Select Features” and select at least one."
        )
    base_pipe_names: list[str] = []
    if include_baseline_pipeline:
        from .feature_sources_catalog import base_pipeline_feature_names

        base_pipe_names = sorted(base_pipeline_feature_names(data_dir))
        if not base_pipe_names:
            raise MasterRegistryExportError(
                "No Baseline Pipeline features available in catalogue."
            )

    exp_pipe_names: list[str] = []
    pipeline_provenance: dict[str, Any] | None = None
    experimental_transform_config: dict[str, Any] | None = None
    if include_experimental_pipeline:
        pid = str(pipeline_id or "").strip().upper()
        if not pid:
            raise MasterRegistryExportError(
                "Select an experimental pipeline when Experimental Pipeline Features is enabled."
            )
        from .pipeline_registry_store import (
            build_pipeline_snapshot,
            ensure_default_existing_pipeline,
            get_pipeline,
            is_base_pipeline_record,
            resolve_pipeline_dataset_feature_names,
        )

        doc = ensure_default_existing_pipeline(data_dir)
        rec = get_pipeline(doc, pid)
        if not rec:
            raise MasterRegistryExportError(f"Pipeline {pid} not found.")
        if is_base_pipeline_record(rec):
            raise MasterRegistryExportError(
                "The Base pipeline cannot be used as an experimental pipeline source. "
                "Select an experimental pipeline (Auto/Manual)."
            )
        pipeline_provenance = build_pipeline_snapshot(rec, pipeline_id=pid)
        exp_pipe_names = sorted(resolve_pipeline_dataset_feature_names(data_dir, doc, pid))
        if isinstance(pipeline_provenance.get("transformation_config"), dict):
            experimental_transform_config = pipeline_provenance.get("transformation_config")
        if not exp_pipe_names:
            raise MasterRegistryExportError(
                f"Pipeline {pipeline_provenance.get('pipeline_name') or pid} has no candidate features."
            )

    combined_names = list(dict.fromkeys(reg_names + base_pipe_names + exp_pipe_names))
    if not combined_names:
        raise MasterRegistryExportError("No features selected from the chosen feature source(s).")

    reg_total = len(reg_names)
    base_pipe_total = len(base_pipe_names)
    exp_pipe_total = len(exp_pipe_names)
    overall_total = len(combined_names)
    started = time.monotonic()
    log_lines: list[str] = []
    no_null_data = bool(no_null_data)
    pipeline_no_null_report = bool(pipeline_no_null_report)
    premium_enabled = bool(premium_enabled)
    prem_lo = float(premium_min) if premium_enabled and premium_min is not None else None
    prem_hi = float(premium_max) if premium_enabled and premium_max is not None else None
    if premium_enabled and (prem_lo is None or prem_hi is None):
        raise MasterRegistryExportError("LTP premium filter requires both min and max")
    premium_active = bool(premium_enabled and prem_lo is not None and prem_hi is not None)

    # Build transformation pipeline config
    transformation_config = None
    if include_baseline_pipeline or include_experimental_pipeline:
        from .transformations.config import merge_transformation_configs

        base_transform_config = None
        if include_baseline_pipeline:
            from .pipeline_features_config import build_pipeline_features_transformation_config

            base_transform_config = build_pipeline_features_transformation_config(
                sample_interval_sec=float(interval_sec),
                exclude_features=output_exclude_for_pipeline,
                interaction_operand_skip=interaction_operand_skip,
                source_forbidden=forbidden,
            )

        if base_transform_config is not None and experimental_transform_config is not None:
            transformation_config = merge_transformation_configs(
                base_transform_config,
                experimental_transform_config,
            )
        elif base_transform_config is not None:
            transformation_config = base_transform_config
        elif experimental_transform_config is not None:
            transformation_config = experimental_transform_config

    if transformation_config is not None:
        from .transformations.config import prune_transformation_config_for_interval

        transformation_config = prune_transformation_config_for_interval(
            transformation_config,
            float(interval_sec),
        )

    if transformation_config and output_exclude_for_pipeline:
        from .pipeline_features_config import prune_pipeline_transformation_config

        transformation_config = prune_pipeline_transformation_config(
            transformation_config,
            output_exclude_for_pipeline,
            interaction_operand_skip=interaction_operand_skip,
            source_exclude=forbidden,
        )

    def _emit(payload: dict[str, Any]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            pass

    def _log(msg: str) -> None:
        text = str(msg)
        # Multiline diagnostics (Pipeline No-Null Report) — keep full lines.
        chunks = text.splitlines() if "\n" in text else [text]
        for chunk in chunks:
            line = f"{_now_label()}  {chunk}"
            log_lines.append(line)
        # Allow larger buffer when streaming long diagnostic reports.
        limit = 2500 if pipeline_no_null_report else 500
        keep = 2000 if pipeline_no_null_report else 400
        if len(log_lines) > limit:
            del log_lines[:-keep]

    def _check_cancel() -> None:
        if cancel_check and cancel_check():
            raise MasterRegistryExportError("Analysis dataset build cancelled")

    def _snapshot(
        *,
        status: str = "running",
        stage: str,
        registry_done: int = 0,
        baseline_pipeline_done: int = 0,
        experimental_pipeline_done: int = 0,
        pipeline_done: int | None = None,
        current_feature: str = "",
        current_source: str = "",
        message: str = "",
        rows_processed: int | None = None,
        columns_per_sec: float | None = None,
        percent: float | None = None,
        extra: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if pipeline_done is not None:
            if include_baseline_pipeline and not include_experimental_pipeline:
                baseline_pipeline_done = pipeline_done
            elif include_experimental_pipeline and not include_baseline_pipeline:
                experimental_pipeline_done = pipeline_done
            elif include_baseline_pipeline and include_experimental_pipeline:
                baseline_pipeline_done = min(pipeline_done, base_pipe_total)
                experimental_pipeline_done = max(0, pipeline_done - base_pipe_total)

        overall_done = registry_done + baseline_pipeline_done + experimental_pipeline_done
        elapsed = time.monotonic() - started
        if percent is None and overall_total > 0:
            percent = round(100.0 * overall_done / overall_total, 1)
        eta = None
        if percent and percent > 0.5 and elapsed > 0.5:
            eta = elapsed * (100.0 - percent) / percent
        body = {
            "status": status,
            "job_kind": "analysis_dataset_build",
            "stage": stage,
            "stages": _stage_states(
                stage,
                include_registry=include_registry,
                include_baseline_pipeline=include_baseline_pipeline,
                include_experimental_pipeline=include_experimental_pipeline,
                no_null_data=no_null_data,
                premium_enabled=premium_active,
            ),
            "registry_done": registry_done,
            "registry_total": reg_total,
            "baseline_pipeline_done": baseline_pipeline_done,
            "baseline_pipeline_total": base_pipe_total,
            "experimental_pipeline_done": experimental_pipeline_done,
            "experimental_pipeline_total": exp_pipe_total,
            "pipeline_done": baseline_pipeline_done + experimental_pipeline_done,
            "pipeline_total": base_pipe_total + exp_pipe_total,
            "overall_done": overall_done,
            "overall_total": overall_total,
            "current_feature": current_feature,
            "current_source": current_source,
            "message": message,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "percent": percent if percent is not None else 0.0,
            "rows_processed": rows_processed,
            "columns_per_sec": columns_per_sec,
            "log_lines": list(
                log_lines if pipeline_no_null_report else log_lines[-80:]
            ),
            "include_registry": include_registry,
            "include_baseline_pipeline": include_baseline_pipeline,
            "include_experimental_pipeline": include_experimental_pipeline,
            "include_pipeline": include_baseline_pipeline or include_experimental_pipeline,
            "no_null_data": no_null_data,
            "premium_enabled": premium_active,
            "premium_min": prem_lo,
            "premium_max": prem_hi,
        }
        if extra:
            body.update(extra)
        return body

    if not os.path.isfile(path):
        raise MasterRegistryExportError("Master database file does not exist")

    created_dt = datetime.now(_IST)
    if not dataset_name:
        stamp = created_dt.strftime("%Y%m%d_%H%M%S")
        tags = []
        if include_registry:
            tags.append(f"{reg_total}r")
        if include_baseline_pipeline:
            tags.append(f"{base_pipe_total}b")
        if include_experimental_pipeline:
            pid_tag = str(pipeline_provenance.get("pipeline_id") or "exp").replace("_", "") if pipeline_provenance else "exp"
            tags.append(f"{pid_tag}_{exp_pipe_total}e")
        tag_str = "_".join(tags) or "empty"
        dataset_name = f"analysis_{tag_str}_{interval_sec}s_{stamp}"

    selected_sources_labels = []
    if include_registry:
        selected_sources_labels.append("Registry Features")
    if include_baseline_pipeline:
        selected_sources_labels.append("Baseline Pipeline Features")
    if include_experimental_pipeline:
        selected_sources_labels.append("Experimental Pipeline Features")
    _log(f"Analysis dataset build started — Selected Sources: {' + '.join(selected_sources_labels)}")
    _log(f"Configuration: Registry={reg_total} | Baseline Pipeline={base_pipe_total} | Experimental Pipeline={exp_pipe_total} | Combined={overall_total}")
    _emit(_snapshot(
        stage=STAGE_REGISTRY if include_registry else (
            STAGE_BASELINE_PIPELINE if include_baseline_pipeline else (
                STAGE_EXPERIMENTAL_PIPELINE if include_experimental_pipeline else STAGE_FINALIZE
            )
        ),
        message="Preparing feature sources…",
    ))

    # --- Stage 1: Registry Features catalogue walk ---
    registry_done = 0
    if include_registry:
        _log(f"Registry Features — starting ({reg_total})")
        for i, name in enumerate(reg_names, start=1):
            _check_cancel()
            registry_done = i
            if i == 1 or i == reg_total or i % 25 == 0:
                _emit(_snapshot(
                    stage=STAGE_REGISTRY,
                    registry_done=registry_done,
                    baseline_pipeline_done=0,
                    experimental_pipeline_done=0,
                    current_feature=name,
                    current_source="Registry Features",
                    message=f"Registry {registry_done} / {reg_total}",
                    columns_per_sec=(registry_done / max(0.001, time.monotonic() - started)),
                ))
            if i % 40 == 0:
                _log(f"Registry  {name}")
        _log(f"Registry Features completed  {reg_total} columns")
        _emit(_snapshot(
            stage=STAGE_REGISTRY,
            registry_done=reg_total,
            baseline_pipeline_done=0,
            experimental_pipeline_done=0,
            current_feature="",
            current_source="Registry Features",
            message="Registry Features completed",
        ))

    # --- Stage 2: export Master (keep pipeline columns) + Pipeline walk ---
    baseline_done = 0
    if include_baseline_pipeline:
        _log(f"Baseline Pipeline Features — regenerating via transformation pipeline ({base_pipe_total})")
        _emit(_snapshot(
            stage=STAGE_BASELINE_PIPELINE,
            registry_done=reg_total,
            baseline_pipeline_done=0,
            experimental_pipeline_done=0,
            current_source="Baseline Pipeline Features",
            message="Exporting Master + regenerating Baseline Pipeline Features…",
            percent=round(100.0 * reg_total / max(1, overall_total), 1) if overall_total else 0.0,
        ))

    if include_experimental_pipeline:
        _log(f"Experimental Pipeline Features — transforming candidate features ({exp_pipe_total})")

    export_started = time.monotonic()

    def _export_progress(msg: str, cur: int = 0, tot: int = 0, **detail: Any) -> None:
        _check_cancel()
        low = str(msg).lower()
        if premium_active and "premium filter" in low:
            stage_now = STAGE_PREMIUM
            source_now = "Premium Filter"
            pct = 97.5
            pipe_approx = pipe_total if include_pipeline else 0
        elif pipeline_no_null_report and (
            "pipeline no-null report" in low or low.startswith("====")
        ):
            stage_now = STAGE_NO_NULL if no_null_data else (
                STAGE_PIPELINE if include_pipeline else STAGE_FINALIZE
            )
            source_now = "Pipeline No-Null Report"
            pct = 96.5 if no_null_data else 94.0
            pipe_approx = pipe_total if include_pipeline else 0
        elif no_null_data and ("no-null" in low or "no null" in low):
            stage_now = STAGE_NO_NULL
            source_now = "No-Null Filter"
            pct = 96.0
            pipe_approx = pipe_total if include_pipeline else 0
        else:
            stage_now = STAGE_PIPELINE if include_pipeline else STAGE_FINALIZE
            source_now = "Pipeline Features" if include_pipeline else "Dataset Finalization"
            # Map day progress into mid-pipeline percent band.
            base = reg_total
            mid = pipe_total * 0.55 if include_pipeline else 0
            frac = 0.0
            day_i = detail.get("day_index")
            day_tot = detail.get("day_total")
            if day_i is not None and day_tot:
                try:
                    frac = min(1.0, float(day_i) / float(day_tot))
                except (TypeError, ValueError):
                    frac = 0.0
            elif tot and tot > 0:
                frac = min(1.0, float(cur) / float(tot))
            pipe_approx = int(mid * frac) if include_pipeline else 0
            cols_done = base + pipe_approx
            pct = round(100.0 * cols_done / max(1, overall_total), 1)
            if (no_null_data or premium_active) and pct > 94:
                pct = 94.0
        elapsed = time.monotonic() - started
        cols_done = reg_total + (pipe_approx if include_pipeline else 0)
        extra = {"export_current": cur, "export_total": tot}
        # Day-at-a-time progress card fields.
        for key in (
            "day",
            "day_index",
            "day_total",
            "mode",
            "rows",
            "features",
            "token_index",
            "token_total",
            "wave_index",
            "wave_total",
            "peak_ram_bytes",
            "resumed",
        ):
            if key in detail and detail.get(key) is not None:
                extra[key] = detail.get(key)
        rows_processed = detail.get("rows")
        if rows_processed is None and cur:
            rows_processed = int(cur)
        _emit(_snapshot(
            stage=stage_now,
            registry_done=reg_total,
            pipeline_done=pipe_approx if include_pipeline else 0,
            current_feature="",
            current_source=source_now,
            message=str(msg),
            rows_processed=int(rows_processed) if rows_processed is not None else None,
            percent=pct,
            columns_per_sec=(cols_done / max(0.001, elapsed)),
            extra=extra,
        ))
        if msg:
            text = str(msg)
            # Full diagnostic report lines — do not truncate to 160 chars.
            if "\n" in text or text.startswith("=") or "Pipeline No-Null Report" in text:
                for chunk in text.splitlines() or [text]:
                    if chunk.strip():
                        _log(chunk)
            elif "row" not in low or cur == 0 or (tot and cur >= tot) or detail.get("day"):
                _log(text[:160])

    kind = str(dataset_kind or "analysis").strip().lower() or "analysis"
    try:
        payload = create_master_registry_dataset(
            data_dir,
            market=market,
            interval_sec=int(interval_sec),
            trading_day=trading_day,
            selected_days=selected_days,
            trading_day_filter=trading_day_filter,
            master_db_path=path,
            all_days=bool(all_days),
            dataset_name=dataset_name,
            token=token,
            atm_band_filter=atm_band_filter,
            transformation_config=transformation_config,
            keep_pipeline_owned=bool(include_baseline_pipeline or include_experimental_pipeline),
            dataset_kind=kind,
            no_null_data=no_null_data,
            pipeline_no_null_report=pipeline_no_null_report,
            premium_enabled=premium_active,
            premium_min=prem_lo,
            premium_max=prem_hi,
            delta_enabled=bool(delta_enabled),
            delta_min=delta_min,
            delta_max=delta_max,
            on_progress=_export_progress,
            registry_export_features=reg_export if include_registry else frozenset(),
            pipeline_provenance=pipeline_provenance if include_experimental_pipeline else None,
            base_pipeline_export_features=frozenset(base_pipe_names) if include_baseline_pipeline else None,
            feature_project_id=bound_pid,
        )
    except MasterRegistryExportError:
        raise
    except Exception as exc:
        raise MasterRegistryExportError(str(exc)) from exc

    # Resolve feature columns from written metadata / parquet (export return is a summary only).
    feature_columns: list[str] = []
    json_path = str(payload.get("json_path") or "")
    parquet_path = str(payload.get("parquet_path") or "")
    if parquet_path and os.path.isfile(parquet_path):
        try:
            import pyarrow.parquet as pq

            feature_columns = list(pq.read_schema(parquet_path).names)
        except Exception:
            feature_columns = []
    if not feature_columns and json_path and os.path.isfile(json_path):
        try:
            import json

            with open(json_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            feature_columns = list(meta.get("feature_columns") or [])
            if not payload.get("column_count"):
                payload["column_count"] = int(meta.get("column_count") or len(feature_columns))
            if not payload.get("feature_count"):
                payload["feature_count"] = int(meta.get("feature_count") or len(feature_columns))
        except Exception:
            feature_columns = []
    elif json_path and os.path.isfile(json_path):
        try:
            import json

            with open(json_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if not payload.get("column_count"):
                payload["column_count"] = int(meta.get("column_count") or len(feature_columns))
            if not payload.get("feature_count"):
                # Prefer catalogue intersection count for analysis summary.
                payload["feature_count"] = int(meta.get("feature_count") or 0)
        except Exception:
            pass
    feature_set = set(feature_columns)

    # Pull No-Null / premium reports from dataset metadata when enabled.
    no_null_dropped: list[str] = []
    no_null_report: dict[str, Any] | None = None
    premium_report: dict[str, Any] | None = None
    transformation_summary: dict[str, Any] | None = None
    day_stats: dict[str, Any] | None = None
    if json_path and os.path.isfile(json_path):
        try:
            import json

            with open(json_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            exec_block = meta.get("execution") if isinstance(meta.get("execution"), dict) else {}
            if isinstance(exec_block.get("transformation_summary"), dict):
                transformation_summary = dict(exec_block["transformation_summary"])
            if isinstance(exec_block.get("day_stats"), dict):
                day_stats = dict(exec_block["day_stats"])
            if no_null_data or premium_active:
                no_null_dropped = list(meta.get("no_null_dropped_columns") or [])
                if isinstance(meta.get("no_null_report"), dict):
                    no_null_report = dict(meta.get("no_null_report") or {})
                if isinstance(meta.get("premium_report"), dict):
                    premium_report = dict(meta.get("premium_report") or {})
                if meta.get("row_count") is not None:
                    payload["row_count"] = int(meta.get("row_count") or 0)
                if meta.get("column_count") is not None:
                    payload["column_count"] = int(meta.get("column_count") or 0)
                # Prefer parquet schema for presence checks. Do not replace with
                # metadata feature_columns when parquet was already read — metadata
                # can lag behind No-Null drops on older builds.
                if not feature_columns and meta.get("feature_columns"):
                    feature_columns = list(meta.get("feature_columns") or [])
                    feature_set = set(feature_columns)
                elif meta.get("feature_columns") and feature_columns:
                    # Keep parquet feature_set; optionally narrow using synced metadata.
                    meta_feats = {
                        str(c) for c in (meta.get("feature_columns") or []) if str(c).strip()
                    }
                    if meta_feats:
                        feature_set = {c for c in feature_set if c in meta_feats} or feature_set
        except Exception:
            pass

    output_parquet = None
    output_json = None
    if parquet_path:
        try:
            from .master_naming import path_relative_to_data_dir

            output_parquet = path_relative_to_data_dir(parquet_path, data_dir)
            if json_path:
                output_json = path_relative_to_data_dir(json_path, data_dir)
        except Exception:
            output_parquet = parquet_path
            output_json = json_path or None
    payload["output_parquet"] = output_parquet
    payload["output_json"] = output_json
    payload["feature_columns"] = feature_columns

    has_pipeline = bool(include_baseline_pipeline or include_experimental_pipeline)
    all_pipe_names = list(dict.fromkeys(base_pipe_names + exp_pipe_names))
    all_pipe_total = len(all_pipe_names)
    if has_pipeline:
        for i, name in enumerate(all_pipe_names, start=1):
            _check_cancel()
            pipeline_done = i
            present = name in feature_set
            if i == 1 or i == all_pipe_total or i % 20 == 0 or present:
                _emit(_snapshot(
                    stage=STAGE_BASELINE_PIPELINE if include_baseline_pipeline else STAGE_EXPERIMENTAL_PIPELINE,
                    registry_done=reg_total,
                    pipeline_done=pipeline_done,
                    current_feature=name,
                    current_source="Pipeline Features",
                    message=(
                        f"Pipeline {pipeline_done} / {all_pipe_total}"
                        + ("" if present else " (not on Master)")
                    ),
                    columns_per_sec=(
                        (reg_total + pipeline_done)
                        / max(0.001, time.monotonic() - started)
                    ),
                ))
            if i % 35 == 0:
                _log(f"Pipeline  {name}")
        _log(f"Pipeline Features scan completed  {all_pipe_total} catalogue columns")

    # --- Stage 3: Finalization summary ---
    _emit(_snapshot(
        stage=STAGE_FINALIZE,
        registry_done=reg_total,
        pipeline_done=all_pipe_total if has_pipeline else 0,
        current_source="Dataset Finalization",
        message="Writing completion summary…",
        percent=99.0,
    ))

    registry_present = sorted(n for n in reg_names if n in feature_set) if include_registry else []
    pipeline_present = sorted(n for n in all_pipe_names if n in feature_set) if has_pipeline else []
    pipeline_missing = sorted(n for n in all_pipe_names if n not in feature_set) if has_pipeline else []
    analysis_feature_count = len(registry_present) + len(pipeline_present)

    parquet_rel = str(payload.get("output_parquet") or "")
    parquet_abs = os.path.join(data_dir, parquet_rel) if parquet_rel and not os.path.isabs(parquet_rel) else (
        str(payload.get("parquet_path") or parquet_rel)
    )
    output_size = None
    if parquet_abs and os.path.isfile(parquet_abs):
        output_size = os.path.getsize(parquet_abs)

    if no_null_data:
        _log(
            f"No-Null Filter applied · dropped columns={len(no_null_dropped)} · "
            f"rows={int(payload.get('row_count') or 0):,}"
        )
    if premium_active:
        kept = (premium_report or {}).get("rows_after")
        before = (premium_report or {}).get("rows_before")
        _log(
            f"Premium Filter applied · LTP {prem_lo:g}–{prem_hi:g}"
            + (
                f" · kept {int(kept):,} / {int(before):,}"
                if kept is not None and before is not None
                else f" · rows={int(payload.get('row_count') or 0):,}"
            )
        )

    elapsed = time.monotonic() - started
    _log("Build completed")
    summary = {
        "status": "completed",
        "job_kind": "analysis_dataset_build",
        "dataset_name": payload.get("dataset_name") or dataset_name,
        "dataset_kind": "analysis",
        "include_registry": include_registry,
        "include_pipeline": include_pipeline,
        "pipeline_id": (pipeline_provenance or {}).get("pipeline_id") if pipeline_provenance else None,
        "pipeline_name": (pipeline_provenance or {}).get("pipeline_name") if pipeline_provenance else None,
        "pipeline_type": (pipeline_provenance or {}).get("pipeline_type") if pipeline_provenance else None,
        "pipeline_snapshot_id": (pipeline_provenance or {}).get("pipeline_snapshot_id") if pipeline_provenance else None,
        "pipeline_feature_count": len((pipeline_provenance or {}).get("candidate_features") or []) if pipeline_provenance else 0,
        "base_pipeline_export_count": len(base_pipe_names) if include_pipeline else 0,
        "experimental_pipeline_export_count": len(experimental_pipe_names) if include_pipeline else 0,
        "pipeline_provenance": dict(pipeline_provenance) if pipeline_provenance else None,
        "no_null_data": no_null_data,
        "pipeline_no_null_report_enabled": pipeline_no_null_report,
        "pipeline_no_null_report": payload.get("pipeline_no_null_report"),
        "no_null_dropped_columns": no_null_dropped,
        "no_null_dropped_count": len(no_null_dropped),
        "no_null_report": no_null_report,
        "premium_enabled": premium_active,
        "premium_min": prem_lo,
        "premium_max": prem_hi,
        "premium_report": premium_report,
        "registry_total": reg_total,
        "registry_present": len(registry_present),
        "baseline_pipeline_total": base_pipe_total,
        "baseline_pipeline_present": len([n for n in base_pipe_names if n in feature_set]),
        "experimental_pipeline_total": exp_pipe_total,
        "experimental_pipeline_present": len([n for n in exp_pipe_names if n in feature_set]),
        "pipeline_total": all_pipe_total,
        "pipeline_present": len(pipeline_present),
        "pipeline_missing": pipeline_missing,
        "pipeline_missing_count": len(pipeline_missing),
        "total_columns": int(payload.get("column_count") or len(feature_columns) or analysis_feature_count),
        "feature_count": analysis_feature_count,
        "row_count": int(payload.get("row_count") or 0),
        "build_time_sec": elapsed,
        "export_elapsed_sec": time.monotonic() - export_started,
        "output_size_bytes": output_size,
        "output_parquet": payload.get("output_parquet"),
        "output_json": payload.get("output_json"),
        "built_at": created_dt.isoformat(timespec="seconds"),
        "built_at_display": created_dt.strftime("%d-%b-%Y %H:%M"),
        "feature_sources": catalog["totals"],
        "transformation_summary": transformation_summary,
        "day_stats": day_stats,
        "log_lines": list(log_lines),
        "payload": payload,
    }
    _emit({
        **_snapshot(
            status="completed",
            stage=STAGE_FINALIZE,
            registry_done=reg_total,
            baseline_pipeline_done=base_pipe_total if include_baseline_pipeline else 0,
            experimental_pipeline_done=exp_pipe_total if include_experimental_pipeline else 0,
            message="Build completed",
            percent=100.0,
            rows_processed=summary["row_count"],
        ),
        "summary": summary,
    })
    return summary


__all__ = [
    "STAGE_FINALIZE",
    "STAGE_NO_NULL",
    "STAGE_PIPELINE",
    "STAGE_REGISTRY",
    "create_analysis_dataset",
]
