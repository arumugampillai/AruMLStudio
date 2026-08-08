"""Master SQLite dataset build — one trading day in RAM at a time."""

from __future__ import annotations

import gc
import os
import uuid
from typing import Any, Callable

from storage.chain_replay_export import ChainReplayError

from .day_context import SourceSpec, load_day_context, validate_day_context
from .expected_spec import strike_selection_metadata, write_expected_spec
from .feature_plugins import (
    horizon_column_name,
    horizon_label,
    resolve_implemented_features_for_selection,
)
from .master_export import export_master_to_parquet
from .master_naming import path_relative_to_data_dir, resolve_master_db_path
from .master_store import MasterStore
from .orchestrator import (
    DatasetBuildConfig,
    ProgressCallback,
    _Cancelled,
    _load_feature_registry,
    _source_from_dict,
)
from .progress import BuildProgress, feature_group_progress_fields
from .production_day_build import build_production_day_rows
from .stages import build_sample_timestamps
from .timing import PipelineTimer
from .writer import datasets_dir, patch_dataset_metadata, write_dataset, _safe_filename

from path_config import CHART_DATA_ROOT as _CHART_DIR
_METADATA_COLS = (
    "trading_day",
    "market",
    "expiry",
    "timestamp",
    "strike",
    "option_type",
    "token",
    "symbol",
)

BACKFILL_STAGE_NAME = "Backfilling existing master"


def make_backfill_progress_handler(
    timer: PipelineTimer,
    emit: Callable[..., dict[str, Any]],
    *,
    stage_name: str = BACKFILL_STAGE_NAME,
) -> Callable[[str, int, int, str], None]:
    """Build the on_progress callback wired into target-horizon backfill.

    Backfill borrows stage 1's timer slot (so elapsed time stays visible) but
    must never be reported to the UI as "Load Database", and its per-day row
    counts must never leak into the day-load / sample counters those widgets
    also read from the same payload shape. This factory keeps that mapping in
    one place: outer "days" callbacks move the day counter forward and reset
    the per-day row counter; inner "rows" callbacks advance the row counter
    for whichever day is currently in flight without touching day counters.
    """
    state: dict[str, int] = {"day_idx": 0, "days_total": 0}

    def _on_progress(msg: str, cur: int, tot: int, unit: str = "days") -> None:
        timer.set_stage_progress(1, current=cur, total=max(1, tot), unit=unit)
        if unit == "days":
            state["day_idx"] = cur + 1
            state["days_total"] = max(1, tot)
            day_frac = cur / state["days_total"]
            emit(
                stage=1,
                stage_name=stage_name,
                message=f"Backfilling existing master data — {msg}",
                backfill_active=True,
                backfill_days_current=state["day_idx"],
                backfill_days_total=state["days_total"],
                backfill_rows_current=0,
                backfill_rows_total=0,
                backfill_percent=round(100.0 * day_frac, 1),
                clear_substage=True,
            )
        else:
            days_total = max(1, state["days_total"])
            day_idx = state["day_idx"] or 1
            row_frac = (cur / tot) if tot else 0.0
            overall_frac = min(1.0, ((day_idx - 1) + row_frac) / days_total)
            emit(
                stage=1,
                stage_name=stage_name,
                message=f"Backfilling existing master data — {msg}",
                backfill_active=True,
                backfill_days_current=day_idx,
                backfill_days_total=days_total,
                backfill_rows_current=cur,
                backfill_rows_total=tot,
                backfill_percent=round(100.0 * overall_frac, 1),
                sub_current=cur,
                sub_total=tot,
                clear_substage=False,
            )

    return _on_progress


class MasterDatasetBuildOrchestrator:
    """Build into SQLite master DB; optional Parquet export for legacy UI/training."""

    def __init__(self, config: DatasetBuildConfig, cancel_event: Any = None) -> None:
        self.config = config
        self.cancel_event = cancel_event

    def _cancelled(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def run(self, on_progress: ProgressCallback | None = None) -> BuildProgress:
        from time import perf_counter

        from chain_replay_ml.performance.runtime import (
            begin_create_dataset_session,
            end_create_dataset_session,
        )

        job_id = str(uuid.uuid4())
        progress = BuildProgress(job_id=job_id)
        create_dataset_t0 = perf_counter()
        phase_timings: dict[str, float] = {
            "loading_ticks_sec": 0.0,
            "feature_computation_sec": 0.0,
            "prediction_targets_sec": 0.0,
            "sqlite_insert_sec": 0.0,
            "polars_duckdb_sec": 0.0,
            "write_output_sec": 0.0,
        }
        begin_create_dataset_session(verbose=True)
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
        data_dir = self.config.resolved_data_dir()
        source_dicts = sorted(
            list(self.config.sources),
            key=lambda s: str(s.get("trading_day") or ""),
        )
        sources = [_source_from_dict(s) for s in source_dicts]
        n_sources = len(sources)
        market = sources[0].market if sources else "NIFTY"

        master_path = self.config.master_db_path or resolve_master_db_path(
            data_dir,
            market=market,
            sampling_interval_sec=window_sec,
        )
        if master_path and not os.path.isabs(master_path):
            master_path = os.path.join(data_dir, master_path.replace("/", os.sep))

        from .lookback_policy import build_dataset_configuration
        from .gap_policy import gap_max_sec_from_policy

        gap_max_sec = gap_max_sec_from_policy(self.config.gap_policy)
        dataset_configuration = build_dataset_configuration(
            sampling=self.config.sampling,
            horizons_sec=horizons_sec,
            gap_max_sec=gap_max_sec,
        )
        lb_policy_doc = dataset_configuration["lookback_policy"]

        expected_path = write_expected_spec(
            data_dir=data_dir,
            dataset_name=self.config.dataset_name,
            sources=source_dicts,
            sampling=self.config.sampling,
            strike_selection=self.config.strike_selection,
            prediction_targets=self.config.prediction_targets,
            feature_selection=self.config.feature_selection,
            registry=registry,
        )
        progress.output_expected_json = expected_path

        export_columns = list(dict.fromkeys([*_METADATA_COLS, *implemented, *target_columns]))
        row_cols = list(export_columns)
        timer = PipelineTimer([
            (gid, group_labels.get(gid, gid))
            for gid in enabled_groups
            if gid in per_group
        ])
        total_rows = 0
        source_results: list[dict[str, Any]] = []
        total_source_ticks = 0
        total_target_trimmed = 0
        total_readiness_stats: dict[str, Any] = {}
        pipeline_bootstrapped = False
        active_feature_substage: str | None = None

        def emit(**kwargs: Any) -> None:
            pl = timer.snapshot(rows=total_rows, estimated_total_rows=0)
            payload = progress.emit(**kwargs, pipeline=pl)
            if on_progress:
                on_progress(payload)

        store = MasterStore(master_path)
        store.open()
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel
        from .build_profiler import BuildProfiler, profile_block, set_profiler
        from .production_day_build import production_performance_debug

        profiler_enabled = bool(getattr(self.config, "build_profiler", False))
        perf_debug = (
            PerformanceDebugConfig(level=PerformanceDebugLevel.BASIC)
            if profiler_enabled
            else production_performance_debug()
        )
        build_profiler: BuildProfiler | None = None
        build_profiler_report: dict[str, Any] | None = None
        if profiler_enabled:
            build_profiler = BuildProfiler()
            build_profiler.start_build()
            set_profiler(build_profiler)
            from .spot_controllers_profiler import reset_spot_controllers_profiler

            reset_spot_controllers_profiler()
        try:
            store.set_meta("master_config", {
                "market": market,
                "sampling_interval_sec": window_sec,
                "sliding_stride_sec": stride_sec,
                "atm_band": atm_band,
                "storage_backend": "master_sqlite",
            })
            store.start_build_job(job_id=job_id, days_total=n_sources)

            from .target_backfill import maybe_backfill_expanded_targets

            # Track backfill under stage 1's timer slot so elapsed time is
            # visible, but give it its own stage name + progress counters
            # (backfill_* fields, sub_current/total) so the UI never mislabels
            # it "Load Database" or conflates its per-day row counts with the
            # real day-load / sample counters.
            timer.start_stage(1)
            timer.set_stage_name(1, BACKFILL_STAGE_NAME)
            _on_backfill_progress = make_backfill_progress_handler(timer, emit)

            try:
                bf_result = maybe_backfill_expanded_targets(
                    store,
                    target_columns=target_columns,
                    horizons_sec=horizons_sec,
                    chart_dir=_CHART_DIR,
                    step_sec=stride_sec,
                    default_market=market,
                    cancel_check=self._cancelled,
                    on_progress=_on_backfill_progress,
                )
                if bf_result.get("backfilled"):
                    emit(
                        stage=1,
                        stage_name=BACKFILL_STAGE_NAME,
                        message=(
                            f"Target backfill: {bf_result.get('rows_updated', 0):,} rows, "
                            f"{', '.join(bf_result.get('columns') or [])}"
                        ),
                        backfill_active=True,
                        backfill_percent=100.0,
                        clear_substage=True,
                    )
                progress.warnings.extend(bf_result.get("warnings") or [])
            except Exception as exc:
                if isinstance(exc, _Cancelled):
                    raise
                progress.warnings.append(f"Target backfill: {exc}")

            # Backfill (if any) is done — restore stage 1's real name/counters
            # before loading the newly selected day(s) below.
            timer.set_stage_name(1, None)
            timer.set_stage_progress(1, current=0, total=n_sources, unit="days")
            emit(
                stage=1,
                current=0,
                total=n_sources,
                message=f"Phase 1: Master build — {os.path.basename(master_path)}",
                backfill_active=False,
                clear_substage=True,
            )

            for si, src in enumerate(sources):
                if self._cancelled():
                    raise _Cancelled()
                day_label = f"Day {si + 1}/{n_sources}"
                src_msg = f"{src.trading_day} • {src.market} • {src.expiry}"

                if store.should_skip_day(src.trading_day):
                    emit(
                        current=si + 1,
                        message=f"{day_label} · {src_msg} — skipped (already in master)",
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        rows=total_rows,
                    )
                    continue

                emit(
                    stage=1,
                    current=si,
                    total=n_sources,
                    message=f"{day_label} · Loading tick DB: {src_msg}",
                    source_day_index=si + 1,
                    source_day_total=n_sources,
                )
                timer.start_stage(1)
                timer.set_stage_progress(1, current=si, total=n_sources, unit="days")

                try:
                    _load_t0 = perf_counter()
                    with profile_block("stage.load_database"):
                        ctx = load_day_context(_CHART_DIR, src, feature_grid_step_sec=stride_sec)
                    phase_timings["loading_ticks_sec"] += perf_counter() - _load_t0
                except (ChainReplayError, OSError) as exc:
                    progress.warnings.append(f"{src_msg}: skipped — {exc}")
                    source_results.append({
                        "trading_day": src.trading_day,
                        "status": "skipped",
                        "error": str(exc),
                    })
                    emit(current=si + 1, message=f"Skipped {src.trading_day}: {exc}")
                    continue

                with profile_block("stage.validate"):
                    issues = validate_day_context(ctx)
                if issues:
                    progress.warnings.append(f"{src_msg}: skipped — {', '.join(issues)}")
                    source_results.append({
                        "trading_day": src.trading_day,
                        "status": "skipped",
                        "error": ", ".join(issues),
                    })
                    continue

                timer.end_stage(1)
                if not pipeline_bootstrapped:
                    for mid_stage in (2, 3, 4, 5):
                        timer.skip_stage(mid_stage)
                    pipeline_bootstrapped = True

                total_source_ticks += ctx.source_ticks
                # Feature stage is a single pass (one-pick or token-parallel), not per-group walks.
                groups_total = 1
                groups_done = 0
                timer.start_stage(6, rows=total_rows)
                timer.set_stage_progress(6, current=0, total=1, unit="groups")
                emit(
                    stage=6,
                    message=f"{day_label} · {src_msg} — feature generation",
                    source_day_index=si + 1,
                    source_day_total=n_sources,
                    **feature_group_progress_fields(groups_done=0, groups_total=groups_total),
                )

                def on_prep_progress(phase: str, msg: str) -> None:
                    emit(
                        stage=6,
                        message=f"{day_label} · {src_msg} — {msg}",
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        **feature_group_progress_fields(
                            groups_done=groups_done,
                            groups_total=groups_total,
                        ),
                    )

                def on_group_start(gid: str, label: str) -> None:
                    nonlocal active_feature_substage
                    if active_feature_substage and active_feature_substage != gid:
                        timer.end_substage(6, active_feature_substage)
                    timer.start_substage(6, gid)
                    active_feature_substage = gid
                    timer.set_stage_progress(6, current=groups_done, total=max(1, groups_total), unit="groups")
                    emit(
                        stage=6,
                        substage=label,
                        message=f"{day_label} · {src_msg} — features: {label}",
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        **feature_group_progress_fields(
                            groups_done=groups_done,
                            groups_total=groups_total,
                            group_id=gid,
                            group_label=label,
                        ),
                    )

                def on_group_progress(label: str, cur: int, tot: int) -> None:
                    timer.set_stage_progress(6, current=groups_done, total=max(1, groups_total), unit="groups")
                    emit(
                        stage=6,
                        substage=label,
                        message=f"{day_label} · {src_msg} — features: {label}",
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        **feature_group_progress_fields(
                            groups_done=groups_done,
                            groups_total=groups_total,
                            group_id=active_feature_substage,
                            group_label=label,
                            row_current=cur,
                            row_total=tot,
                        ),
                    )

                def on_group_done(gid: str) -> None:
                    nonlocal groups_done, active_feature_substage
                    if active_feature_substage == gid:
                        timer.end_substage(6, gid)
                        active_feature_substage = None
                    groups_done += 1
                    timer.set_stage_progress(6, current=groups_done, total=max(1, groups_total), unit="groups")
                    label = group_labels.get(gid, gid)
                    emit(
                        stage=6,
                        substage=label,
                        message=f"{day_label} · {src_msg} — done: {label}",
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        **feature_group_progress_fields(
                            groups_done=groups_done,
                            groups_total=groups_total,
                            group_id=gid,
                            group_label=label,
                        ),
                    )

                _feat_t0 = perf_counter()
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
                    on_group_start=on_group_start,
                    on_group_progress=on_group_progress,
                    on_group_done=on_group_done,
                    on_prep_progress=on_prep_progress,
                    cancel_check=self._cancelled,
                    gap_max_sec=gap_max_sec,
                    performance_debug=perf_debug,
                )
                # build_production_day_rows includes sampling/strike/targets + feature gen.
                phase_timings["feature_computation_sec"] += perf_counter() - _feat_t0
                trimmed = int(day_stats.get("target_trimmed_rows") or 0)
                total_target_trimmed += trimmed

                day_ready = day_stats.get("feature_readiness")
                if day_ready:
                    from chain_replay_ml.feature_policy.build_readiness import _merge_readiness_stats

                    total_readiness_stats = _merge_readiness_stats(total_readiness_stats, day_ready)

                if not day_rows:
                    progress.warnings.append(f"{src_msg}: no rows produced")
                    del ctx
                    gc.collect()
                    continue

                row_cols = list(dict.fromkeys([*row_cols, *(day_rows[0].keys())]))
                try:
                    _sql_t0 = perf_counter()
                    with profile_block("stage.sqlite_insert", rows=len(day_rows)):
                        store.begin_day(src.trading_day, row_cols)
                        inserted = store.insert_rows(day_rows)
                        store.commit_day(src.trading_day)
                    phase_timings["sqlite_insert_sec"] += perf_counter() - _sql_t0
                except Exception:
                    store.rollback_day()
                    raise

                try:
                    from .day_metadata import (
                        build_and_persist_day_metadata,
                        feature_family_map_from_registry,
                    )
                    from .feature_expectation import build_expectation_index
                    from .gap_policy import GAP_POLICY_VERSION

                    category_by_name: dict[str, str] = {}
                    try:
                        from chain_replay_ml.feature_policy import load_feature_policy_registry

                        pol = load_feature_policy_registry(feature_names=implemented)
                        for fname, meta in (pol.features or {}).items():
                            cat = getattr(meta, "feature_category", None)
                            category_by_name[str(fname)] = str(
                                getattr(cat, "value", cat) or ""
                            )
                    except Exception:
                        pass
                    family_by_name = feature_family_map_from_registry(registry)
                    expectation_by_name = build_expectation_index(
                        registry if isinstance(registry, dict) else None
                    )
                    build_ver = str(
                        getattr(self.config, "dataset_name", None)
                        or os.path.basename(str(master_path or ""))
                        or ""
                    )
                    day_stats_local = day_stats if isinstance(day_stats, dict) else {}
                    build_duration = day_stats_local.get("elapsed_sec") or day_stats_local.get(
                        "duration_sec"
                    )
                    build_and_persist_day_metadata(
                        store.conn,
                        day_rows,
                        trading_day=src.trading_day,
                        registry_features=list(implemented),
                        meta_columns=_METADATA_COLS,
                        gap_max_sec=float(gap_max_sec),
                        sampling_interval_sec=float(window_sec),
                        build_version=build_ver,
                        category_by_name=category_by_name,
                        family_by_name=family_by_name,
                        expectation_by_name=expectation_by_name,
                        ingestion={
                            "dataset_version": build_ver,
                            "registry_version": (registry or {}).get("version")
                            if isinstance(registry, dict)
                            else None,
                            "feature_engine_version": str(
                                (registry or {}).get("version")
                                if isinstance(registry, dict)
                                else ""
                            ),
                            "gap_policy_version": str(GAP_POLICY_VERSION),
                            "build_duration_sec": build_duration,
                        },
                    )
                except Exception as meta_exc:
                    progress.warnings.append(
                        f"{src.trading_day}: day metadata failed ({meta_exc})"
                    )

                timer.set_stage_progress(6, current=inserted, total=max(1, inserted), unit="rows")
                timer.end_stage(6)

                total_rows = store.read_master_meta().total_rows
                cov = dict(day_stats.get("coverage") or {})
                cov["samples_written"] = inserted
                store.set_day_coverage(src.trading_day, cov)
                source_results.append({
                    "trading_day": src.trading_day,
                    "market": src.market,
                    "expiry": src.expiry,
                    "status": "loaded",
                    "rows": inserted,
                    "coverage": cov,
                })
                emit(
                    stage=6,
                    message=f"{day_label} · {src_msg} — committed {inserted:,} rows ({total_rows:,} total)",
                    rows=total_rows,
                    source_day_index=si + 1,
                    source_day_total=n_sources,
                    clear_substage=True,
                )

                del ctx, day_rows
                gc.collect()

            from time import perf_counter as _perf_counter
            from .build_profiler import get_profiler as _get_profiler

            _meta_t0 = _perf_counter() if _get_profiler() else None
            store.mark_build_complete()
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
            store.set_meta("build_summary", build_summary)
            store.set_meta("master_config", {
                "market": market,
                "sampling_interval_sec": window_sec,
                "sliding_stride_sec": stride_sec,
                "atm_band": atm_band,
                "storage_backend": "master_sqlite",
                "feature_count": len(implemented),
                "target_count": len(target_columns),
                "strike_selection": strike_selection_metadata(strike_selection),
                "gap_policy": normalize_gap_policy(self.config.gap_policy),
                "prediction_targets": build_summary["prediction_targets"],
            })
            store.set_meta("build_schema", {
                "feature_count": len(implemented),
                "target_count": len(target_columns),
                "feature_columns": implemented,
                "target_columns": target_columns,
            })
            try:
                from chain_replay_ml.feature_policy import (
                    build_policy_report,
                    finalize_build_policy_manifest,
                    sample_rows_from_sqlite,
                )

                emit(
                    stage=6,
                    message="Finalizing feature policy manifest…",
                    rows=total_rows,
                    clear_substage=True,
                )
                sample_rows = sample_rows_from_sqlite(
                    store.conn, implemented, limit=50_000,
                )
                fp_manifest = finalize_build_policy_manifest(
                    implemented,
                    sampling_interval_sec=float(window_sec),
                    gap_max_sec=gap_max_sec,
                    rows=sample_rows,
                    build_stats={
                        "rows": total_rows,
                        "trading_days": len(store.distinct_trading_days()),
                        **total_readiness_stats,
                    },
                )
                dataset_configuration["feature_policy"] = fp_manifest
                store.set_meta("feature_policy", fp_manifest)
            except Exception:
                fp_manifest = None
            dataset_configuration["strike_selection"] = strike_selection_metadata(strike_selection)
            dataset_configuration["gap_policy"] = normalize_gap_policy(self.config.gap_policy)
            store.set_meta("dataset_configuration", dataset_configuration)
            store.sync_schema_meta_fields()
            total_rows = store.read_master_meta().total_rows

            from .master_fingerprint import build_identity_from_build

            lb_method = str(lb_policy_doc.get("method") or "calendar")
            store.update_build_identity(build_identity_from_build(
                market=market,
                sampling_interval_sec=window_sec,
                atm_band=atm_band,
                feature_count=len(implemented),
                target_horizons_sec=horizons_sec,
                lookback_policy=lb_method,
                registry=registry,
                target_columns=target_columns,
                created_from="master_build",
            ))
            if _meta_t0 is not None:
                _prof = _get_profiler()
                if _prof is not None:
                    _prof.record("stage.post_metadata", _perf_counter() - _meta_t0)

            skip_export = bool(getattr(self.config, "skip_parquet_export", False))
            parquet_path: str | None = None
            json_path: str | None = None

            if skip_export:
                timer.skip_stage(8)
                emit(
                    stage=8,
                    current=1,
                    total=1,
                    message=f"Master SQLite complete — {total_rows:,} rows (Parquet export skipped)",
                    rows=total_rows,
                    clear_substage=True,
                )
            else:
                timer.start_stage(8)
                # Export Parquet + JSON metadata for legacy Model Builder / registry
                emit(
                    stage=8,
                    current=0,
                    total=3,
                    message=f"Exporting master → Parquet ({total_rows:,} rows)…",
                    rows=total_rows,
                    clear_substage=True,
                )
                safe_name = _safe_filename(self.config.dataset_name)
                parquet_path = os.path.join(datasets_dir(data_dir), f"{safe_name}.parquet")
                json_path = os.path.join(datasets_dir(data_dir), f"{safe_name}.json")

                def on_export(msg: str, cur: int, tot: int) -> None:
                    emit(stage=8, current=cur, total=tot, message=msg, rows=total_rows)

                _write_t0 = perf_counter()
                export_master_to_parquet(store, parquet_path, row_cols, on_progress=on_export)
                phase_timings["write_output_sec"] += perf_counter() - _write_t0

                from .pipeline_identity import build_pipeline_fingerprint, build_version_metadata_fields
                from .spec_identity import compute_spec_hash_from_fingerprint

                lb_method = lb_policy_doc["method"]
                pipeline_fingerprint = build_pipeline_fingerprint(
                    sampling_interval_sec=window_sec,
                    atm_band=atm_band,
                    feature_count=len(implemented),
                    target_horizons_sec=horizons_sec,
                    lookback_policy=lb_method,
                    registry=registry,
                )
                spec_hash = compute_spec_hash_from_fingerprint(pipeline_fingerprint, dataset_configuration)
                metadata = {
                    "dataset_name": self.config.dataset_name,
                    "storage_backend": "master_sqlite",
                    "master_db_path": path_relative_to_data_dir(master_path, data_dir),
                    "market": market,
                    "days": store.distinct_trading_days(),
                    "sources": source_results,
                    "expected_spec": path_relative_to_data_dir(expected_path, data_dir),
                    "sampling": {
                        "interval_sec": window_sec,
                        "sliding_stride_sec": stride_sec,
                        "method": str(self.config.sampling.get("samplingMethod") or "fixed_interval"),
                    },
                    "strike_selection": strike_selection_metadata(strike_selection),
                    "gap_policy": normalize_gap_policy(self.config.gap_policy),
                    "prediction_targets": [horizon_label(h) for h in horizons_sec],
                    "prediction_target_columns": target_columns,
                    "feature_columns": implemented,
                    "feature_count": len(implemented),
                    "target_count": len(target_columns),
                    "dataset_configuration": dataset_configuration,
                    "build_summary": build_summary,
                    "row_count": total_rows,
                    "job_id": job_id,
                    **build_version_metadata_fields(pipeline_fingerprint),
                    "dataset_spec_hash": spec_hash,
                }
                import json
                from datetime import datetime, timezone

                metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(metadata, fh, indent=2, ensure_ascii=False)

                emit(
                    stage=8,
                    current=3,
                    total=3,
                    message=f"Master build complete — {total_rows:,} rows",
                    rows=total_rows,
                )
                timer.end_stage(8)

            progress.status = "completed"
            progress.output_parquet = parquet_path
            progress.output_json = json_path
            progress.output_parquet_bytes = (
                os.path.getsize(parquet_path) if parquet_path and os.path.isfile(parquet_path) else 0
            )
            progress.output_json_bytes = (
                os.path.getsize(json_path) if json_path and os.path.isfile(json_path) else 0
            )
            progress.dataset_stats = {
                "trading_days": len(store.distinct_trading_days()),
                "rows": total_rows,
                "rows_added": total_rows,
                "columns": len(row_cols),
                "feature_count": len(implemented),
                "target_count": len(target_columns),
                "row_counts_by_day": store.row_counts_by_day(),
                "coverage_by_day": store.get_coverage_by_day(),
                "sources": source_results,
            }
            if build_profiler is not None:
                build_profiler.finish_build(total_rows=total_rows)
                build_profiler_report = build_profiler.to_report()
                from .spot_controllers_profiler import snapshot_spot_controllers_profiler

                spot_update_stats = snapshot_spot_controllers_profiler()
                if spot_update_stats is not None:
                    build_profiler_report["spot_controllers_update"] = spot_update_stats.to_dict()
                progress.dataset_stats["build_profiler_report"] = build_profiler_report
                # Prefer profiler splits when available (prediction targets vs feature gen).
                by_name = {
                    str(e.get("name")): e
                    for e in (build_profiler_report.get("stages") or [])
                    if isinstance(e, dict)
                }
                pred = by_name.get("stage.prediction_targets") or {}
                feat = by_name.get("stage.feature_generation") or {}
                if pred.get("total_sec") is not None:
                    phase_timings["prediction_targets_sec"] = float(pred["total_sec"])
                if feat.get("total_sec") is not None:
                    # Narrow feature_computation to Stage-6 feature gen when profiler ran.
                    phase_timings["feature_computation_sec"] = float(feat["total_sec"])
                load = by_name.get("stage.load_database") or {}
                if load.get("total_sec") is not None:
                    phase_timings["loading_ticks_sec"] = float(load["total_sec"])
                sql = by_name.get("stage.sqlite_insert") or {}
                if sql.get("total_sec") is not None:
                    phase_timings["sqlite_insert_sec"] = float(sql["total_sec"])
            wall_sec = perf_counter() - create_dataset_t0
            phase_timings["create_dataset_wall_sec"] = wall_sec
            numba_stats = end_create_dataset_session(
                verbose=True,
                create_dataset_wall_sec=wall_sec,
                feature_computation_sec=phase_timings.get("feature_computation_sec"),
            )
            progress.dataset_stats["feature_engine_numba"] = numba_stats
            progress.dataset_stats["create_dataset_phase_timings"] = {
                k: round(float(v), 6) for k, v in phase_timings.items()
            }
            try:
                from chain_replay_ml.performance.create_dataset_timing import (
                    build_timing_report,
                    write_timing_report,
                )

                pipe_snap = timer.snapshot(rows=total_rows, estimated_total_rows=0)
                timing_doc = build_timing_report(
                    numba_stats=numba_stats,
                    phase_timings=phase_timings,
                    pipeline_stages=list(pipe_snap.get("stages") or []),
                    build_profiler_report=build_profiler_report,
                    meta={
                        "job_id": job_id,
                        "market": market,
                        "rows": total_rows,
                        "feature_count": len(implemented),
                        "target_count": len(target_columns),
                        "sources": len(source_results),
                    },
                )
                json_rep, md_rep = write_timing_report(timing_doc)
                progress.dataset_stats["create_dataset_timing_report_json"] = json_rep
                progress.dataset_stats["create_dataset_timing_report_md"] = md_rep
            except Exception as exc:
                progress.warnings.append(f"Create Dataset timing report: {exc}")
            if fp_manifest:
                progress.dataset_stats["feature_policy_report"] = build_policy_report(fp_manifest)
            if progress.status == "completed":
                try:
                    from .selection_preview_calibration import record_build_calibration

                    record_build_calibration(
                        data_dir,
                        build_kind="master_build",
                        strike_selection=strike_selection,
                        sources=source_dicts,
                        actual_rows=total_rows,
                        market=market,
                        interval_sec=window_sec,
                        master_db_path=master_path,
                        build_job_id=job_id,
                        preview_snapshot=getattr(self.config, "preview_snapshot", None),
                    )
                except Exception:
                    pass
            if skip_export:
                emit(
                    message=f"Done — {total_rows:,} rows in {os.path.basename(master_path)}",
                    rows=total_rows,
                )
        except _Cancelled:
            store.mark_build_failed("cancelled")
            progress.status = "cancelled"
        except Exception as exc:
            store.mark_build_failed(str(exc))
            progress.status = "failed"
            progress.error = str(exc)
            raise
        finally:
            set_profiler(None)
            store.close()

        return progress
