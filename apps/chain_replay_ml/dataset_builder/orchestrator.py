"""Dataset build orchestrator — 8 stages with progress callbacks."""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from storage.chain_replay_export import ChainReplayError

from .day_context import DayContext, SourceSpec, load_day_context, validate_day_context
from .feature_plugins import (
    horizon_column_name,
    horizon_label,
    resolve_implemented_features_for_selection,
)
from .progress import TOTAL_STAGES, BuildProgress, feature_group_progress_fields
from .production_day_build import build_production_day_rows
from .stages import build_sample_timestamps
from .timing import PipelineTimer
from .validation import validate_dataset
from .expected_spec import strike_selection_metadata, write_expected_spec
from .master_naming import path_relative_to_data_dir
from .writer import ensure_parquet_engine, patch_dataset_metadata, read_dataset_parquet, write_dataset

from path_config import CHART_DATA_ROOT as _CHART_DIR
@dataclass
class DatasetBuildConfig:
    dataset_name: str
    sources: list[dict[str, Any]]
    sampling: dict[str, Any]
    strike_selection: dict[str, Any]
    prediction_targets: dict[str, Any]
    feature_selection: dict[str, Any]
    feature_registry: dict[str, Any] | None = None
    data_dir: str | None = None
    build_mode: str = "new"
    append_to: str | None = None
    build_profile: str = "production"
    skip_data_validation: bool = False
    storage_backend: str = "parquet"  # parquet | master_sqlite
    master_db_path: str | None = None
    skip_parquet_export: bool = False
    also_write_master_db: bool = False
    preview_snapshot: dict[str, Any] | None = None
    gap_policy: dict[str, Any] | None = None
    build_profiler: bool = True

    def resolved_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        return os.path.join(_CHART_DIR, "data")


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass
class DatasetBuildOrchestrator:
    config: DatasetBuildConfig
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self, on_progress: ProgressCallback | None = None) -> BuildProgress:
        backend = str(getattr(self.config, "storage_backend", "parquet") or "parquet").lower()
        if backend in ("master", "master_sqlite", "sqlite_master"):
            from .master_build import MasterDatasetBuildOrchestrator

            return MasterDatasetBuildOrchestrator(
                self.config,
                cancel_event=self.cancel_event,
            ).run(on_progress=on_progress)

        job_id = str(uuid.uuid4())
        progress = BuildProgress(job_id=job_id)
        registry = self.config.feature_registry or _load_feature_registry()
        enabled_groups, implemented, pending, per_group = resolve_implemented_features_for_selection(
            self.config.feature_selection, registry, data_dir=self.config.resolved_data_dir(),
        )
        group_labels = {
            gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
            for gid in enabled_groups
        }
        horizons_sec = [int(h) for h in (self.config.prediction_targets.get("horizonsSec") or [])]
        target_columns = [horizon_column_name(h) for h in horizons_sec]

        from .schema_registry import validate_build_schema

        schema_errors = validate_build_schema(
            enabled_groups=enabled_groups,
            target_columns=target_columns,
        )
        if schema_errors:
            raise RuntimeError("Schema integrity failed:\n" + "\n".join(f"• {e}" for e in schema_errors))

        from .sliding_stride_policy import (
            resolve_feature_window_sec,
            resolve_sliding_stride_sec,
            validate_sliding_stride,
        )

        window_sec = resolve_feature_window_sec(self.config.sampling)
        stride_sec = resolve_sliding_stride_sec(self.config.sampling)
        stride_err = validate_sliding_stride(window_sec, stride_sec)
        if stride_err:
            raise RuntimeError(f"Invalid sliding stride: {stride_err}")
        atm_band = int(self.config.strike_selection.get("atmBand") or 10)
        strike_selection = dict(self.config.strike_selection or {})
        strike_mode = str(strike_selection.get("mode") or "atm_band").lower()
        from .lookback_policy import build_dataset_configuration
        from .gap_policy import gap_max_sec_from_policy

        gap_max_sec = gap_max_sec_from_policy(self.config.gap_policy)
        dataset_configuration = build_dataset_configuration(
            sampling=self.config.sampling,
            horizons_sec=horizons_sec,
            gap_max_sec=gap_max_sec,
        )
        lb_policy_doc = dataset_configuration["lookback_policy"]
        build_mode = str(self.config.build_mode or "new").lower()
        append_target = str(self.config.append_to or "").strip()
        data_dir = self.config.resolved_data_dir()
        existing_meta: dict[str, Any] | None = None
        existing_parquet_path: str | None = None
        existing_df = None
        preserve_created_at: str | None = None

        if build_mode in ("append", "rebuild"):
            if not append_target:
                raise RuntimeError("Existing dataset name is required for append/rebuild")
            from .append_ops import (
                load_dataset_metadata,
                merge_metadata_for_append,
                plan_append_sessions,
                validate_append_compatible,
                validate_append_merge,
            )

            existing_meta, paths = load_dataset_metadata(data_dir, append_target)
            existing_parquet_path = paths["parquet"]
            preserve_created_at = str(existing_meta.get("created_at") or "")
            if build_mode == "append":
                if not os.path.isfile(existing_parquet_path):
                    raise RuntimeError(f"Existing parquet not found for {append_target}")
                compat_errors = validate_append_compatible(
                    existing_meta,
                    sampling=self.config.sampling,
                    strike_selection=self.config.strike_selection,
                    prediction_targets=self.config.prediction_targets,
                    feature_selection=self.config.feature_selection,
                    registry=registry,
                )
                if compat_errors:
                    raise RuntimeError(
                        "Append configuration incompatible with existing dataset:\n"
                        + "\n".join(f"• {e}" for e in compat_errors)
                    )
                append_plan = plan_append_sessions(existing_meta, list(self.config.sources))
                if not append_plan["new_sources"]:
                    raise RuntimeError("No new trading sessions to append")
                source_dicts = list(append_plan["new_sources"])
            else:
                source_dicts = list(self.config.sources)
                if not source_dicts:
                    raise RuntimeError("Select at least one chain source to rebuild the dataset")
            self.config.dataset_name = paths["safe_name"]
        else:
            source_dicts = list(self.config.sources)

        sources = [_source_from_dict(s) for s in source_dicts]
        n_sources = len(sources)
        group_list = [
            (gid, group_labels.get(gid, gid))
            for gid in enabled_groups
            if gid in per_group
        ]
        timer = PipelineTimer(group_list)
        est_total_rows = 0

        expected_path = write_expected_spec(
            data_dir=data_dir,
            dataset_name=self.config.dataset_name,
            sources=source_dicts if build_mode != "append" else _merged_source_dicts(
                existing_meta,
                source_dicts,
            ),
            sampling=self.config.sampling,
            strike_selection=self.config.strike_selection,
            prediction_targets=self.config.prediction_targets,
            feature_selection=self.config.feature_selection,
            registry=registry,
        )
        progress.output_expected_json = expected_path

        ensure_parquet_engine()

        def emit(**kwargs: Any) -> None:
            pl = timer.snapshot(rows=len(all_rows), estimated_total_rows=est_total_rows)
            payload = progress.emit(**kwargs, pipeline=pl)
            if on_progress:
                on_progress(payload)

        heartbeat = _BuildHeartbeat(emit)

        all_rows: list[dict[str, Any]] = []
        loaded: list[DayContext] = []
        source_results: list[dict[str, Any]] = []
        total_source_ticks = 0
        total_sample_points = 0
        total_target_trimmed = 0
        total_delta_range_stats: dict[str, Any] | None = None
        total_readiness_stats: dict[str, Any] = {}

        try:
            # Stage 1 — Load Database (all sources into memory before row processing)
            timer.start_stage(1)
            emit(
                stage=1,
                current=0,
                total=n_sources,
                message=f"Phase 1/4: Loading all {n_sources} tick database(s) into memory…",
                source_day_index=None,
                source_day_total=n_sources,
                clear_substage=True,
            )
            for i, src in enumerate(sources):
                if self._cancelled():
                    raise _Cancelled()
                msg = f"{src.date_label or src.trading_day} • {src.market} • {src.expiry}"
                emit(
                    current=i,
                    message=f"Loading database {i + 1}/{n_sources}: {msg}",
                    source_day_index=i + 1,
                    source_day_total=n_sources,
                )
                try:
                    ctx = load_day_context(_CHART_DIR, src, feature_grid_step_sec=stride_sec)
                    loaded.append(ctx)
                    total_source_ticks += ctx.source_ticks
                    source_results.append({
                        "source_id": src.source_id,
                        "trading_day": src.trading_day,
                        "market": src.market,
                        "expiry": src.expiry,
                        "status": "loaded",
                        "lines": ctx.validation_lines,
                    })
                except (ChainReplayError, OSError, sqlite3.Error) as exc:
                    progress.warnings.append(f"{msg}: skipped — {exc}")
                    source_results.append({
                        "source_id": src.source_id,
                        "trading_day": src.trading_day,
                        "status": "skipped",
                        "error": str(exc),
                    })
                emit(
                    current=i + 1,
                    rows=len(all_rows),
                    source_ticks=total_source_ticks,
                    message=f"Loaded database {i + 1}/{n_sources}: {msg}",
                    source_day_index=i + 1,
                    source_day_total=n_sources,
                )

            if not loaded:
                raise RuntimeError("No sources could be loaded")
            timer.end_stage(1)
            emit(
                stage=1,
                current=n_sources,
                total=n_sources,
                message=(
                    f"All {len(loaded)}/{n_sources} database(s) loaded — "
                    f"next: process each day and merge rows"
                ),
                source_ticks=total_source_ticks,
                clear_substage=True,
            )

            # Stage 2 — Validate Sources
            timer.start_stage(2)
            emit(stage=2, current=0, total=len(loaded), message="Validating sources…", clear_substage=True)
            valid_ctx: list[DayContext] = []
            for i, ctx in enumerate(loaded):
                if self._cancelled():
                    raise _Cancelled()
                msg = f"{ctx.source.date_label or ctx.source.trading_day} • {ctx.source.market}"
                issues = validate_day_context(ctx)
                if issues:
                    progress.warnings.append(f"{msg}: skipped — {', '.join(issues)}")
                else:
                    valid_ctx.append(ctx)
                emit(current=i + 1, message=msg)

            if not valid_ctx:
                raise RuntimeError("No valid sources after validation")
            timer.end_stage(2)

            max_horizon = max(horizons_sec) if horizons_sec else 0
            for ctx in valid_ctx:
                total_sample_points += len(
                    build_sample_timestamps(ctx, step_sec=stride_sec, max_horizon_sec=max_horizon)
                )
            strikes_per_sample = 18 if strike_mode == "delta_range" else (2 * atm_band + 1) * 2
            est_total_rows = total_sample_points * strikes_per_sample

            timer.start_stage(3)
            emit(
                stage=3,
                current=len(valid_ctx),
                total=len(valid_ctx),
                message="Sampling grid ready",
                rows=0,
                sample_points=total_sample_points,
                source_ticks=total_source_ticks,
                clear_substage=True,
            )
            timer.end_stage(3)

            n_valid = len(valid_ctx)
            for si, ctx in enumerate(valid_ctx):
                if self._cancelled():
                    raise _Cancelled()
                src_msg = f"{ctx.source.trading_day} • {ctx.source.market} • {ctx.expiry_norm}"
                day_label = f"Day {si + 1}/{n_valid}"
                rows_before_day = len(all_rows)
                timer.start_stage(4, rows=rows_before_day)
                emit(
                    stage=4,
                    message=f"Phase 2/4: {day_label} · {src_msg} — strike selection",
                    source_day_index=si + 1,
                    source_day_total=n_valid,
                    rows=rows_before_day,
                )

                def on_strike_progress(partial: int, _unused: int) -> None:
                    total_now = rows_before_day + partial
                    timer.set_stage_progress(4, current=total_now, total=est_total_rows)
                    emit(
                        stage=4,
                        message=f"{day_label} · {src_msg} — strike selection",
                        source_day_index=si + 1,
                        source_day_total=n_valid,
                        rows=total_now,
                    )

                def on_targets_progress(cur: int, tot: int) -> None:
                    if timer._stages[4].status == "running":
                        timer.end_stage(4)
                    timer.start_stage(5)
                    total_now = rows_before_day + cur
                    timer.set_stage_progress(5, current=total_now, total=rows_before_day + tot)
                    emit(
                        stage=5,
                        message=f"{day_label} · {src_msg} — prediction targets",
                        source_day_index=si + 1,
                        source_day_total=n_valid,
                        rows=total_now,
                    )

                groups_total = len([g for g in enabled_groups if g in per_group_features])
                groups_done = 0
                active_gid: str | None = None

                def on_group_start(gid: str, label: str) -> None:
                    nonlocal active_gid
                    if timer._stages[5].status == "running":
                        timer.end_stage(5)
                    timer.start_stage(6)
                    timer.start_substage(6, gid)
                    active_gid = gid

                def on_group_progress(label: str, cur: int, tot: int) -> None:
                    timer.set_stage_progress(6, current=groups_done, total=max(1, groups_total), unit="groups")
                    emit(
                        stage=6,
                        substage=label,
                        message=f"{day_label} · {src_msg} — features: {label}",
                        source_day_index=si + 1,
                        source_day_total=n_valid,
                        rows=rows_before_day + tot,
                        **feature_group_progress_fields(
                            groups_done=groups_done,
                            groups_total=groups_total,
                            group_id=active_gid,
                            group_label=label,
                            row_current=cur,
                            row_total=tot,
                        ),
                    )

                def on_group_done(gid: str) -> None:
                    nonlocal groups_done, active_gid
                    timer.end_substage(6, gid)
                    if active_gid == gid:
                        active_gid = None
                    groups_done += 1
                    timer.set_stage_progress(6, current=groups_done, total=max(1, groups_total), unit="groups")

                day_rows, day_stats = build_production_day_rows(
                    ctx,
                    step_sec=stride_sec,
                    strike_selection=strike_selection,
                    horizons_sec=horizons_sec,
                    enabled_groups=enabled_groups,
                    group_labels=group_labels,
                    implemented_features=implemented,
                    per_group_features=per_group,
                    lookback_policy_doc=lb_policy_doc,
                    on_strike_progress=on_strike_progress,
                    on_targets_progress=on_targets_progress,
                    on_group_start=on_group_start,
                    on_group_progress=on_group_progress,
                    on_group_done=on_group_done,
                    cancel_check=self._cancelled,
                    gap_max_sec=gap_max_sec,
                )
                trimmed = int(day_stats.get("target_trimmed_rows") or 0)
                if trimmed:
                    total_target_trimmed += trimmed
                    progress.warnings.append(
                        f"{src_msg}: dropped {trimmed:,} rows with missing future LTP targets "
                        f"(illiquid / no tick yet)"
                    )
                day_delta = day_stats.get("delta_range_stats")
                if day_delta:
                    from .delta_range_stats import merge_delta_range_stats_dict

                    total_delta_range_stats = merge_delta_range_stats_dict(
                        total_delta_range_stats, day_delta,
                    )
                day_ready = day_stats.get("feature_readiness")
                if day_ready:
                    from chain_replay_ml.feature_policy.build_readiness import _merge_readiness_stats

                    total_readiness_stats = _merge_readiness_stats(total_readiness_stats, day_ready)
                if self._cancelled():
                    raise _Cancelled()
                if timer._stages[4].status == "running":
                    timer.end_stage(4)
                if timer._stages[5].status == "running":
                    timer.end_stage(5)
                all_rows.extend(day_rows)
                emit(
                    stage=6,
                    message=f"{day_label} · {src_msg} — day complete ({len(day_rows):,} rows, {len(all_rows):,} total)",
                    source_day_index=si + 1,
                    source_day_total=n_valid,
                    rows=len(all_rows),
                    clear_substage=True,
                )

            if timer._stages[6].status == "running":
                timer.end_stage(6)
            emit(
                stage=6,
                message=f"Phase 3/4: All {n_valid} day(s) merged — {len(all_rows):,} rows in memory",
                rows=len(all_rows),
                source_day_index=None,
                source_day_total=n_valid,
                clear_substage=True,
            )

            # Stage 7 — Dataset Validation (live per-check progress)
            last_validation_checks: list[dict[str, Any]] = []
            skip_validation = bool(self.config.skip_data_validation)
            if bool(getattr(self.config, "also_write_master_db", False)):
                skip_validation = False
            if skip_validation:
                timer.skip_stage(7)
                last_validation_checks = [{
                    "id": "skipped",
                    "label": "Dataset validation",
                    "status": "skipped",
                    "detail": "Skipped for fast experiment build",
                }]
                emit(
                    stage=7,
                    current=1,
                    total=1,
                    message=f"Validation skipped — {len(all_rows):,} rows ready to write",
                    rows=len(all_rows),
                    clear_substage=True,
                    validation_checks=last_validation_checks,
                )
                progress.validation_checks = []
            else:
                timer.start_stage(7)

                def on_validation_progress(payload: dict[str, Any]) -> None:
                    nonlocal last_validation_checks
                    last_validation_checks = payload.get("checks") or []
                    timer.set_stage_progress(
                        7,
                        current=int(payload.get("sub_current") or payload.get("current") or 0),
                        total=int(payload.get("sub_total") or payload.get("total") or 1),
                        unit="rows" if payload.get("sub_total") else "checks",
                    )
                    heartbeat.update(
                        stage=7,
                        message=payload.get("message") or "Validating dataset…",
                        rows=len(all_rows),
                    )
                    emit(
                        stage=7,
                        current=payload.get("current", 0),
                        total=payload.get("total", 1),
                        message=payload.get("message") or "Validating dataset…",
                        rows=len(all_rows),
                        sub_current=payload.get("sub_current"),
                        sub_total=payload.get("sub_total"),
                        validation_checks=payload.get("checks") or [],
                        current_check=payload.get("current_check"),
                    )

                emit(
                    stage=7,
                    current=0,
                    total=7 if target_columns else 6,
                    message=f"Phase 4/4: Validating merged dataset ({len(all_rows):,} rows)…",
                    rows=len(all_rows),
                    clear_substage=True,
                    validation_checks=[],
                )
                heartbeat.start(
                    stage=7,
                    message=f"Validating {len(all_rows):,} rows…",
                    rows=len(all_rows),
                )
                if build_mode == "append" and existing_parquet_path:
                    heartbeat.update(message="Loading existing parquet for append merge check…")
                    existing_df = read_dataset_parquet(existing_parquet_path)
                    merge_issues = validate_append_merge(existing_df, all_rows)
                    if merge_issues:
                        heartbeat.stop()
                        progress.status = "failed"
                        progress.error = "Validation failed: " + "; ".join(merge_issues)
                        emit(
                            stage=7,
                            current=0,
                            total=1,
                            message="Append merge validation failed",
                            rows=len(all_rows),
                            validation_checks=[],
                        )
                        return progress

                ok, issues = validate_dataset(
                    all_rows,
                    target_columns=target_columns,
                    feature_columns=implemented,
                    expected_feature_count=len(implemented),
                    on_check=on_validation_progress,
                    cancel_check=self._cancelled,
                )
                if self._cancelled():
                    raise _Cancelled()
                if not ok:
                    heartbeat.stop()
                    progress.status = "failed"
                    progress.error = "Validation failed: " + "; ".join(issues)
                    emit(
                        stage=7,
                        current=len(last_validation_checks),
                        total=max(len(last_validation_checks), 1),
                        message="Validation failed",
                        rows=len(all_rows),
                        validation_checks=last_validation_checks,
                    )
                    return progress
                timer.end_stage(7)
                progress.validation_checks = list(last_validation_checks)
                heartbeat.stop()

            # Stage 8 — Master DB only (optional) or Parquet write
            timer.start_stage(8)
            also_master = bool(getattr(self.config, "also_write_master_db", False))
            master_sync_stats: dict[str, Any] | None = None

            if also_master:
                emit(
                    stage=8,
                    current=0,
                    total=2,
                    message=f"Persisting {len(all_rows):,} validated rows to Master Dataset DB…",
                    rows=len(all_rows),
                    clear_substage=True,
                    validation_checks=[],
                )
                heartbeat.start(
                    stage=8,
                    message=f"Persisting validated rows to Master Dataset DB…",
                    rows=len(all_rows),
                )

                def on_master_progress(msg: str, cur: int, tot: int) -> None:
                    timer.set_stage_progress(8, current=cur, total=tot, unit="days")
                    heartbeat.update(message=msg)
                    emit(
                        stage=8,
                        current=cur,
                        total=2,
                        message=msg,
                        rows=len(all_rows),
                        clear_substage=True,
                    )

                from .master_sync import sync_build_rows_to_master

                master_sync_stats = sync_build_rows_to_master(
                    self.config,
                    all_rows=all_rows,
                    valid_ctx=valid_ctx,
                    implemented=implemented,
                    target_columns=target_columns,
                    job_id=job_id,
                    step_sec=stride_sec,
                    atm_band=atm_band,
                    on_progress=on_master_progress,
                    cancel_check=self._cancelled,
                )
                if self._cancelled():
                    raise _Cancelled()
                synced = int(master_sync_stats.get("rows_synced") or 0)
                timer.end_stage(8)
                heartbeat.stop()
                pipeline_snapshot = timer.snapshot(rows=len(all_rows), estimated_total_rows=est_total_rows)
                progress.status = "completed"
                progress.pipeline = pipeline_snapshot
                progress.dataset_stats = {
                    "trading_days": int(master_sync_stats.get("days_synced") or 0),
                    "rows": synced,
                    "rows_added": synced,
                    "columns": len(implemented) + len(target_columns) + 8,
                    "master_db_path": master_sync_stats.get("master_db_path"),
                    "row_counts_by_day": master_sync_stats.get("row_counts_by_day"),
                    "coverage_by_day": master_sync_stats.get("coverage_by_day"),
                    "master_dataset_only": True,
                }
                emit(
                    stage=8,
                    current=2,
                    total=2,
                    message=(
                        f"Master Dataset complete — {synced:,} rows in "
                        f"{master_sync_stats.get('master_db_path') or 'master DB'} (Parquet skipped)"
                    ),
                    rows=len(all_rows),
                    clear_substage=True,
                )
                if on_progress:
                    on_progress(progress.to_dict())
                return progress

            emit(
                stage=8,
                current=0,
                total=3,
                message=f"Writing merged dataset ({len(all_rows):,} rows) to Parquet…",
                rows=len(all_rows),
                clear_substage=True,
                validation_checks=[],
            )
            heartbeat.start(
                stage=8,
                message=f"Writing {len(all_rows):,} rows to Parquet…",
                rows=len(all_rows),
            )

            def on_write_progress(msg: str, cur: int, tot: int) -> None:
                timer.set_stage_progress(8, current=cur, total=tot, unit="steps")
                heartbeat.update(message=msg)
                emit(
                    stage=8,
                    current=cur,
                    total=tot,
                    message=msg,
                    rows=len(all_rows),
                    clear_substage=True,
                )
            days_meta = [
                {
                    "trading_day": c.source.trading_day,
                    "market": c.source.market,
                    "expiry": c.expiry_norm,
                    "source_id": c.source.source_id,
                }
                for c in valid_ctx
            ]
            lb_method = lb_policy_doc["method"]
            from .pipeline_identity import (
                build_pipeline_fingerprint,
                build_version_metadata_fields,
            )

            pipeline_fingerprint = build_pipeline_fingerprint(
                sampling_interval_sec=window_sec,
                atm_band=atm_band,
                feature_count=len(implemented),
                target_horizons_sec=horizons_sec,
                lookback_policy=lb_method,
                registry=registry,
            )
            from .spec_identity import compute_spec_hash_from_fingerprint

            spec_hash = compute_spec_hash_from_fingerprint(pipeline_fingerprint, dataset_configuration)

            from .build_summary import build_summary_metadata
            from .gap_policy import normalize_gap_policy

            build_summary = build_summary_metadata(
                feature_names=implemented,
                sampling_interval_sec=float(window_sec),
                sliding_stride_sec=float(stride_sec),
                strike_selection=strike_selection,
                gap_policy=self.config.gap_policy,
                prediction_targets=self.config.prediction_targets,
                feature_count=len(implemented),
                target_count=len(target_columns),
            )
            dataset_configuration["strike_selection"] = strike_selection_metadata(strike_selection)
            dataset_configuration["gap_policy"] = normalize_gap_policy(self.config.gap_policy)

            metadata = {
                "dataset_name": self.config.dataset_name,
                "market": sources[0].market if len({s.market for s in sources}) == 1 else "MIXED",
                "days": days_meta,
                "sources": source_results,
                "expected_spec": path_relative_to_data_dir(expected_path, data_dir),
                "sampling": {
                    "interval_sec": window_sec,
                    "sliding_stride_sec": stride_sec,
                    "method": self.config.sampling.get("samplingMethod") or "fixed_interval",
                },
                "strike_selection": strike_selection_metadata(strike_selection),
                "gap_policy": normalize_gap_policy(self.config.gap_policy),
                "prediction_targets": [horizon_label(h) for h in horizons_sec],
                "prediction_target_columns": target_columns,
                "build_summary": build_summary,
                "feature_profile": self.config.feature_selection.get("profile") or "default",
                "feature_groups": enabled_groups,
                "enabled_features": list(
                    self.config.feature_selection.get("enabledFeatures") or implemented
                ),
                "feature_groups_implemented": list(per_group.keys()),
                "feature_columns": implemented,
                "feature_columns_pending": pending,
                "feature_count": len(implemented),
                "target_count": len(target_columns),
                "dataset_configuration": dataset_configuration,
                "lookback_policy": lb_method,
                "source_ticks": total_source_ticks,
                "sample_points_estimate": total_sample_points,
                "warnings": list(progress.warnings),
                "target_trimmed_rows": total_target_trimmed,
                "job_id": job_id,
                "build_profile": str(self.config.build_profile or "production"),
                "validation_skipped": bool(self.config.skip_data_validation),
                **build_version_metadata_fields(pipeline_fingerprint),
                "dataset_spec_hash": spec_hash,
            }
            if total_delta_range_stats:
                metadata["delta_range_stats"] = total_delta_range_stats
                strike_meta = dict(metadata.get("strike_selection") or {})
                strike_meta["build_stats"] = total_delta_range_stats
                metadata["strike_selection"] = strike_meta
            if build_mode == "append" and existing_meta is not None:
                metadata = merge_metadata_for_append(
                    existing_meta,
                    new_days_meta=days_meta,
                    new_source_results=source_results,
                    new_warnings=list(progress.warnings),
                    new_target_trimmed=total_target_trimmed,
                    build_performance={},
                    append_job_id=job_id,
                )
                metadata.update({
                    "expected_spec": path_relative_to_data_dir(expected_path, data_dir),
                    "dataset_configuration": dataset_configuration,
                    "lookback_policy": lb_method,
                    "feature_columns": implemented,
                    "feature_columns_pending": pending,
                    "feature_count": len(implemented),
                    "target_count": len(target_columns),
                    "prediction_targets": [horizon_label(h) for h in horizons_sec],
                    "prediction_target_columns": target_columns,
                    "feature_groups": enabled_groups,
                    "feature_groups_implemented": list(per_group.keys()),
                    "dataset_spec_hash": spec_hash,
                    "source_ticks": int(existing_meta.get("source_ticks") or 0) + total_source_ticks,
                    **build_version_metadata_fields(pipeline_fingerprint),
                })
                if metadata.get("append_history"):
                    metadata["append_history"][-1]["rows_added"] = len(all_rows)
            elif build_mode == "rebuild" and existing_meta is not None:
                from datetime import datetime, timezone
                metadata["rebuilt_at"] = datetime.now(timezone.utc).isoformat()
                metadata["previous_row_count"] = int(existing_meta.get("row_count") or 0)

            fp_manifest = None
            try:
                from chain_replay_ml.feature_policy import build_policy_report, finalize_build_policy_manifest

                fp_manifest = finalize_build_policy_manifest(
                    implemented,
                    sampling_interval_sec=float(window_sec),
                    gap_max_sec=gap_max_sec,
                    rows=all_rows,
                    build_stats={
                        "rows": len(all_rows),
                        "trading_days": len(valid_ctx),
                        **total_readiness_stats,
                    },
                )
                dataset_configuration["feature_policy"] = fp_manifest
                metadata["feature_policy"] = fp_manifest
            except Exception:
                fp_manifest = None

            parquet_path, json_path, parquet_bytes, json_bytes = write_dataset(
                data_dir=data_dir,
                dataset_name=self.config.dataset_name,
                rows=all_rows,
                metadata=metadata,
                on_progress=on_write_progress,
                existing_df=existing_df if build_mode == "append" else None,
                preserve_created_at=preserve_created_at if build_mode == "append" else None,
            )
            progress.status = "completed"
            progress.output_parquet = parquet_path
            progress.output_json = json_path
            progress.output_parquet_bytes = parquet_bytes
            progress.output_json_bytes = json_bytes
            progress.dataset_stats = _dataset_stats(
                all_rows,
                trading_days=len(valid_ctx),
                existing_rows=int(existing_meta.get("row_count") or 0) if build_mode == "append" and existing_meta else 0,
                delta_range_stats=total_delta_range_stats,
            )
            if fp_manifest:
                progress.dataset_stats["feature_policy_report"] = build_policy_report(fp_manifest)
            try:
                from .selection_preview_calibration import record_build_calibration

                record_build_calibration(
                    data_dir,
                    build_kind="parquet_build",
                    strike_selection=strike_selection,
                    sources=list(self.config.sources),
                    actual_rows=len(all_rows),
                    market=str(self.config.sources[0].get("market") or "NIFTY") if self.config.sources else "NIFTY",
                    interval_sec=window_sec,
                    build_job_id=job_id,
                    preview_snapshot=self.config.preview_snapshot,
                )
            except Exception:
                pass
            timer.end_stage(8)
            heartbeat.stop()
            pipeline_snapshot = timer.snapshot(rows=len(all_rows), estimated_total_rows=est_total_rows)
            peak_ram_mb = _peak_ram_mb()
            metadata["build_performance"] = {
                "total_elapsed_sec": pipeline_snapshot.get("total_elapsed_sec"),
                "total_elapsed_label": pipeline_snapshot.get("total_elapsed_label"),
                "rows_per_sec": pipeline_snapshot.get("avg_rows_per_sec"),
                "peak_ram_mb": peak_ram_mb,
                "stages": [
                    {
                        "name": s.get("name"),
                        "elapsed_sec": s.get("elapsed_sec"),
                        "elapsed_label": s.get("elapsed_label"),
                    }
                    for s in (pipeline_snapshot.get("stages") or [])
                    if s.get("status") == "done"
                ],
            }
            patch_updates: dict[str, Any] = {"build_performance": metadata["build_performance"]}
            if build_mode == "append" and metadata.get("append_history"):
                append_history = list(metadata["append_history"])
                append_history[-1]["build_performance"] = metadata["build_performance"]
                patch_updates["append_history"] = append_history
            patch_dataset_metadata(json_path, patch_updates)
            emit(
                stage=8,
                current=3,
                total=3,
                message="Completed",
                rows=len(all_rows),
                validation_checks=last_validation_checks,
            )
            if on_progress:
                on_progress(progress.to_dict())
            return progress

        except _Cancelled:
            heartbeat.stop()
            progress.status = "cancelled"
            progress.error = "Build cancelled"
            progress.pipeline = timer.snapshot(rows=len(all_rows), estimated_total_rows=est_total_rows)
            if on_progress:
                on_progress(progress.to_dict())
            return progress
        except Exception as exc:
            heartbeat.stop()
            progress.status = "failed"
            progress.error = str(exc)
            progress.pipeline = timer.snapshot(rows=len(all_rows), estimated_total_rows=est_total_rows)
            if on_progress:
                on_progress(progress.to_dict())
            return progress


def _peak_ram_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _dataset_stats(
    rows: list[dict[str, Any]],
    *,
    trading_days: int,
    existing_rows: int = 0,
    delta_range_stats: dict[str, Any] | None = None,
) -> dict[str, int]:
    sample = rows[0] if rows else {}
    ce_rows = sum(1 for r in rows if str(r.get("option_type") or "").upper() == "CE")
    pe_rows = sum(1 for r in rows if str(r.get("option_type") or "").upper() == "PE")
    total_rows = len(rows) + int(existing_rows)
    out: dict[str, Any] = {
        "trading_days": int(trading_days),
        "rows": total_rows,
        "rows_added": len(rows),
        "columns": len(sample),
        "ce_rows": ce_rows,
        "pe_rows": pe_rows,
    }
    if delta_range_stats:
        out["delta_range_stats"] = delta_range_stats
    return out


def _merged_source_dicts(
    existing_meta: dict[str, Any] | None,
    new_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for day in (existing_meta or {}).get("days") or []:
        key = (
            str(day.get("trading_day") or ""),
            str(day.get("market") or "").upper(),
            str(day.get("expiry") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "source_id": str(day.get("source_id") or f"{key[0]}|{key[1]}|{key[2]}"),
            "trading_day": key[0],
            "market": key[1],
            "expiry": key[2],
        })
    for src in new_sources:
        key = (
            str(src.get("trading_day") or ""),
            str(src.get("market") or "NIFTY").upper(),
            str(src.get("expiry") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(src))
    return merged


class _Cancelled(Exception):
    pass


class _BuildHeartbeat:
    """Emit periodic progress ticks during long stretches without granular callbacks."""

    def __init__(self, emit_fn: Callable[..., None], interval: float = 2.0) -> None:
        self._emit = emit_fn
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = 1
        self._message = ""
        self._rows = 0

    def start(self, *, stage: int, message: str, rows: int) -> None:
        self.stop()
        self._stage = stage
        self._message = message
        self._rows = rows
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="build-heartbeat")
        self._thread.start()

    def update(self, *, stage: int | None = None, message: str | None = None, rows: int | None = None) -> None:
        if stage is not None:
            self._stage = stage
        if message is not None:
            self._message = message
        if rows is not None:
            self._rows = rows

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._emit(stage=self._stage, message=self._message, rows=self._rows)


def _source_from_dict(raw: dict[str, Any]) -> SourceSpec:
    return SourceSpec(
        source_id=str(raw.get("source_id") or ""),
        trading_day=str(raw.get("trading_day") or ""),
        market=str(raw.get("market") or "NIFTY").upper(),
        expiry=str(raw.get("expiry") or ""),
        date_label=str(raw.get("date") or raw.get("date_label") or ""),
    )


@functools.lru_cache(maxsize=1)
def _load_feature_registry() -> dict[str, Any]:
    from .schema_registry import load_feature_registry

    return load_feature_registry()
