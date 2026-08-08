"""Run master dataset insert pipeline in-process (no WebSocket / FastAPI)."""

from __future__ import annotations

import multiprocessing
import os
import queue as thread_queue
import threading
import uuid
from typing import Any, Callable

from chain_replay_ml.dataset_builder.master_defaults import (
    default_master_feature_selection,
    default_master_prediction_targets,
    default_master_sampling,
)
from chain_replay_ml.dataset_builder.gap_policy import default_gap_policy, normalize_gap_policy
from chain_replay_ml.dataset_builder.orchestrator import DatasetBuildConfig, _load_feature_registry
from chain_replay_ml.dataset_builder.dataset_pipeline import PipelineOptions

from .progress_adapter import enrich_master_build_payload, master_build_done_payload
from .strike_selection_engine import MASTER_DATASET_ATM_BAND, strike_selection_for_master

ProgressCallback = Callable[[dict[str, Any]], None]

MASTER_ATM_BAND = MASTER_DATASET_ATM_BAND


def chart_data_dir(chart_dir: str) -> str:
    return os.path.join(chart_dir, "data")


def build_master_insert_config(
    chart_dir: str,
    *,
    sources: list[dict[str, Any]],
    interval_sec: int = 10,
    sampling: dict[str, Any] | None = None,
    feature_selection: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    low_memory: bool = False,
    build_profiler: bool = False,
) -> DatasetBuildConfig:
    registry = _load_feature_registry()
    data_dir = chart_data_dir(chart_dir)
    sampling_doc = dict(sampling or default_master_sampling(interval_sec))
    sampling_doc["trainingIntervalSec"] = int(sampling_doc.get("trainingIntervalSec") or interval_sec)
    if sampling_doc.get("slidingStrideSec") is None:
        sampling_doc["slidingStrideSec"] = int(sampling_doc["trainingIntervalSec"])
    from .feature_registry_service import sanitize_feature_selection

    feat_sel = sanitize_feature_selection(
        chart_dir,
        dict(feature_selection or default_master_feature_selection(registry)),
        registry,
    )
    market = str(sources[0].get("market") or "NIFTY").upper() if sources else "NIFTY"

    from chain_replay_ml.dataset_builder.master_naming import resolve_master_db_path

    master_path = resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=int(sampling_doc["trainingIntervalSec"]),
    )

    return DatasetBuildConfig(
        dataset_name=f"master_insert_{uuid.uuid4().hex[:8]}",
        sources=list(sources),
        sampling=sampling_doc,
        strike_selection=strike_selection_for_master(strike_selection),
        prediction_targets=dict(prediction_targets or default_master_prediction_targets()),
        feature_selection=dict(feat_sel),
        feature_registry=registry,
        data_dir=data_dir,
        build_mode="new",
        skip_data_validation=False,
        also_write_master_db=True,
        storage_backend="master_sqlite",
        master_db_path=master_path,
        skip_parquet_export=True,
        gap_policy=normalize_gap_policy(gap_policy or default_gap_policy()),
        build_profiler=bool(build_profiler),
    )


def master_pipeline_options() -> PipelineOptions:
    """Master-only: validate during build; skip audit / post-validation / training."""
    return PipelineOptions(
        build_profile="production",
        run_audit=False,
        run_validation=False,
        train_model=False,
        skip_data_validation=False,
    )


def run_master_build_streaming(
    config: DatasetBuildConfig,
    *,
    cancel_event: Any,
    on_progress: ProgressCallback,
) -> dict[str, Any]:
    """Run MasterDatasetBuildOrchestrator — shared by in-process callers and child process."""
    from chain_replay_ml.dataset_builder.master_build import MasterDatasetBuildOrchestrator

    orch = MasterDatasetBuildOrchestrator(config, cancel_event=cancel_event)
    progress = orch.run(on_progress=on_progress)
    rel_path = config.master_db_path or ""
    data_dir = config.data_dir or ""
    if rel_path and data_dir and not os.path.isabs(rel_path):
        rel_path = os.path.relpath(rel_path, data_dir).replace("\\", "/")
    done = master_build_done_payload(progress.to_dict(), rel_path)
    if progress.status == "failed":
        done["status"] = "failed"
        done["error"] = progress.error
    elif progress.status == "cancelled":
        done["status"] = "cancelled"
    return done


class MasterBuildRunner:
    """Run Create Dataset in a single child process (parent never builds)."""

    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._ctx = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._progress_queue: Any = None
        self._cancel_event: Any = None
        self._on_progress: ProgressCallback | None = None
        self._on_done: Callable[[dict[str, Any]], None] | None = None
        self._terminal_received = False

    @property
    def running(self) -> bool:
        proc = self._process
        return bool(proc and proc.is_alive())

    def cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    def drain_ipc(self) -> bool:
        """Drain child-process progress queue into UI callbacks. Returns True when build finished."""
        if self._progress_queue is None:
            return False
        done = False
        while True:
            try:
                msg = self._progress_queue.get_nowait()
            except thread_queue.Empty:
                break
            if msg.get("_done"):
                done = True
                self._terminal_received = True
                payload = {k: v for k, v in msg.items() if k != "_done"}
                if self._on_done is not None:
                    self._on_done(payload)
            elif self._on_progress is not None:
                self._on_progress(msg)
        if not done and self._process is not None and not self._process.is_alive() and not self._terminal_received:
            self._terminal_received = True
            done = True
            exit_code = self._process.exitcode
            error = (
                f"Build process exited unexpectedly (code {exit_code})"
                if exit_code not in (0, None)
                else "Build process exited without completion message"
            )
            if self._on_done is not None:
                self._on_done({"status": "failed", "error": error})
        if done and self._process is not None:
            self._process.join(timeout=2.0)
            self._reset_ipc()
        return done

    def _reset_ipc(self) -> None:
        self._process = None
        self._progress_queue = None
        self._cancel_event = None
        self._on_progress = None
        self._on_done = None
        self._terminal_received = False

    def start(
        self,
        *,
        sources: list[dict[str, Any]],
        interval_sec: int,
        sampling: dict[str, Any] | None = None,
        feature_selection: dict[str, Any] | None = None,
        prediction_targets: dict[str, Any] | None = None,
        strike_selection: dict[str, Any] | None = None,
        gap_policy: dict[str, Any] | None = None,
        low_memory: bool = False,
        build_profiler: bool = False,
        on_progress: ProgressCallback,
        on_done: Callable[[dict[str, Any]], None],
    ) -> None:
        if self.running:
            raise RuntimeError("A build is already running")

        from .master_build_process import run_master_build_process

        build_kwargs = dict(
            chart_dir=self.chart_dir,
            sources=sources,
            interval_sec=interval_sec,
            sampling=sampling,
            feature_selection=feature_selection,
            prediction_targets=prediction_targets,
            strike_selection=strike_selection,
            gap_policy=gap_policy,
            low_memory=low_memory,
            build_profiler=build_profiler,
        )
        self._on_progress = on_progress
        self._on_done = on_done
        self._terminal_received = False
        self._progress_queue = self._ctx.Queue()
        self._cancel_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=run_master_build_process,
            args=(build_kwargs, self._progress_queue, self._cancel_event),
            name="master-dataset-build",
            daemon=True,
        )
        self._process.start()

class MasterDebugLoadRunner:
    """Load tick DBs into memory only — debug step before full feature build."""

    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def cancel(self) -> None:
        self._cancel_event.set()

    def start(
        self,
        *,
        sources: list[dict[str, Any]],
        interval_sec: int,
        on_progress: ProgressCallback,
        on_done: Callable[[dict[str, Any]], None],
    ) -> None:
        if self.running:
            raise RuntimeError("A debug load is already running")

        self._cancel_event.clear()

        def _worker() -> None:
            import gc
            import time

            from chain_replay_ml.dataset_builder.day_context import load_day_context
            from chain_replay_ml.dataset_builder.orchestrator import _source_from_dict

            from .progress_adapter import debug_load_done_payload, debug_load_running_payload

            t0 = time.perf_counter()
            total_spot = 0
            total_chain = 0
            total_ticks = 0
            results: list[dict[str, Any]] = []
            n = len(sources)
            status = "completed"

            def emit(**kwargs: Any) -> None:
                elapsed = time.perf_counter() - t0
                on_progress(debug_load_running_payload(
                    elapsed_sec=elapsed,
                    ticks_in_memory=total_ticks,
                    spot_ticks=total_spot,
                    chain_ticks=total_chain,
                    source_day_total=n,
                    total=n,
                    **kwargs,
                ))

            try:
                emit(
                    current=0,
                    message=f"Debug load — {n} source(s) into memory",
                    source_day_index=0,
                )
                for si, src_dict in enumerate(sources):
                    if self._cancel_event.is_set():
                        status = "cancelled"
                        break
                    src = _source_from_dict(src_dict)
                    label = f"{src.trading_day} · {src.market} · {src.expiry}"
                    emit(
                        current=si,
                        message=f"Loading tick DB: {label}",
                        source_day_index=si + 1,
                    )
                    ctx = load_day_context(
                        self.chart_dir,
                        src,
                        feature_grid_step_sec=int(interval_sec),
                    )
                    total_spot += int(ctx.spot_ticks)
                    total_chain += int(ctx.chain_ticks)
                    total_ticks += int(ctx.source_ticks)
                    results.append({
                        "trading_day": src.trading_day,
                        "market": src.market,
                        "expiry": src.expiry,
                        "db_path": ctx.db_path,
                        "spot_ticks": ctx.spot_ticks,
                        "chain_ticks": ctx.chain_ticks,
                        "total_ticks": ctx.source_ticks,
                        "strikes": len(ctx.strike_mapping),
                    })
                    emit(
                        current=si + 1,
                        message=f"Loaded {label} — {ctx.source_ticks:,} ticks",
                        source_day_index=si + 1,
                    )
                    del ctx
                    gc.collect()

                elapsed = time.perf_counter() - t0
                if status == "cancelled":
                    on_done({
                        "status": "cancelled",
                        "debug_load": True,
                        "error": "Cancelled",
                        "ticks_in_memory": total_ticks,
                    })
                else:
                    on_done(debug_load_done_payload(
                        ticks_in_memory=total_ticks,
                        spot_ticks=total_spot,
                        chain_ticks=total_chain,
                        sources_loaded=len(results),
                        elapsed_sec=elapsed,
                        load_results=results,
                    ))
            except Exception as exc:
                on_done({
                    "status": "failed",
                    "debug_load": True,
                    "error": str(exc),
                    "ticks_in_memory": total_ticks,
                    "spot_ticks": total_spot,
                    "chain_ticks": total_chain,
                })

        self._thread = threading.Thread(target=_worker, name="master-debug-load", daemon=True)
        self._thread.start()


class MasterDebugFeatureRunner:
    """Load ticks and run feature generation — no master DB write."""

    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def cancel(self) -> None:
        self._cancel_event.set()

    def start(
        self,
        *,
        sources: list[dict[str, Any]],
        interval_sec: int,
        feature_selection: dict[str, Any],
        prediction_targets: dict[str, Any],
        strike_selection: dict[str, Any],
        gap_policy: dict[str, Any] | None = None,
        on_progress: ProgressCallback,
        on_done: Callable[[dict[str, Any]], None],
    ) -> None:
        if self.running:
            raise RuntimeError("A debug feature run is already running")

        self._cancel_event.clear()

        def _worker() -> None:
            import gc
            import time

            from chain_replay_ml.dataset_builder.day_context import load_day_context, validate_day_context
            from chain_replay_ml.dataset_builder.feature_plugins import resolve_implemented_features_for_selection
            from chain_replay_ml.dataset_builder.gap_policy import gap_max_sec_from_policy, normalize_gap_policy
            from chain_replay_ml.dataset_builder.lookback_policy import build_dataset_configuration
            from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry, _source_from_dict
            from chain_replay_ml.dataset_builder.progress import feature_group_progress_fields
            from chain_replay_ml.dataset_builder.production_day_build import build_production_day_rows
            from chain_replay_ml.dataset_builder.timing import PipelineTimer
            from storage.chain_replay_export import ChainReplayError

            from .progress_adapter import debug_feature_done_payload, enrich_master_build_payload

            t0 = time.perf_counter()
            registry = _load_feature_registry()
            enabled_groups, implemented, _pending, per_group = resolve_implemented_features_for_selection(
                feature_selection, registry, data_dir=chart_data_dir(self.chart_dir),
            )
            group_labels = {
                gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
                for gid in enabled_groups
            }
            horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
            step_sec = int(interval_sec)
            strike_sel = strike_selection_for_master(strike_selection)
            gap_max_sec = gap_max_sec_from_policy(normalize_gap_policy(gap_policy or default_gap_policy()))
            lb_policy_doc = build_dataset_configuration(
                sampling={"trainingIntervalSec": step_sec, "samplingMethod": "fixed_interval"},
                horizons_sec=horizons_sec,
                gap_max_sec=gap_max_sec,
            )["lookback_policy"]

            source_dicts = sorted(sources, key=lambda s: str(s.get("trading_day") or ""))
            sources_specs = [_source_from_dict(s) for s in source_dicts]
            n_sources = len(sources_specs)
            timer = PipelineTimer([
                (gid, group_labels.get(gid, gid))
                for gid in enabled_groups
                if gid in per_group
            ])
            total_rows = 0
            total_ticks = 0
            total_spot = 0
            total_chain = 0
            total_readiness: dict[str, Any] = {}
            active_feature_substage: str | None = None
            pipeline_bootstrapped = False
            status = "completed"

            def emit(**kwargs: Any) -> None:
                pl = timer.snapshot(rows=total_rows, estimated_total_rows=0)
                raw = {
                    "status": "running",
                    "debug_features": True,
                    "pipeline": pl,
                    "rows": total_rows,
                    "ticks_in_memory": total_ticks,
                    "spot_ticks": total_spot,
                    "chain_ticks": total_chain,
                    "feature_count": len(implemented),
                    **kwargs,
                }
                on_progress(enrich_master_build_payload(raw))

            try:
                emit(
                    stage=1,
                    current=0,
                    total=n_sources,
                    message=f"Debug features — {len(implemented)} features, {n_sources} day(s)",
                    source_day_index=0,
                    source_day_total=n_sources,
                )
                for si, src in enumerate(sources_specs):
                    if self._cancel_event.is_set():
                        status = "cancelled"
                        break
                    day_label = f"Day {si + 1}/{n_sources}"
                    src_msg = f"{src.trading_day} · {src.market} · {src.expiry}"

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
                        ctx = load_day_context(
                            self.chart_dir, src, feature_grid_step_sec=step_sec,
                        )
                    except (ChainReplayError, OSError) as exc:
                        emit(
                            current=si + 1,
                            message=f"Skipped {src_msg}: {exc}",
                            source_day_index=si + 1,
                            source_day_total=n_sources,
                        )
                        continue

                    issues = validate_day_context(ctx)
                    if issues:
                        emit(
                            current=si + 1,
                            message=f"Skipped {src_msg}: {', '.join(issues)}",
                            source_day_index=si + 1,
                            source_day_total=n_sources,
                        )
                        continue

                    timer.end_stage(1)
                    total_ticks += int(ctx.source_ticks)
                    total_spot += int(ctx.spot_ticks)
                    total_chain += int(ctx.chain_ticks)
                    if not pipeline_bootstrapped:
                        for mid_stage in (2, 3, 4, 5):
                            timer.skip_stage(mid_stage)
                        pipeline_bootstrapped = True

                    groups_total = len(per_group)
                    groups_done = 0
                    timer.start_stage(6, rows=total_rows)
                    timer.set_stage_progress(6, current=0, total=max(1, groups_total), unit="groups")
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

                    day_rows, day_stats = build_production_day_rows(
                        ctx,
                        step_sec=step_sec,
                        strike_selection=strike_sel,
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
                        cancel_check=self._cancel_event.is_set,
                        gap_max_sec=gap_max_sec,
                    )
                    day_ready = day_stats.get("feature_readiness")
                    if day_ready:
                        from chain_replay_ml.feature_policy.build_readiness import _merge_readiness_stats

                        total_readiness = _merge_readiness_stats(total_readiness, day_ready)

                    inserted = len(day_rows)
                    total_rows += inserted
                    timer.set_stage_progress(6, current=inserted, total=max(1, inserted), unit="rows")
                    timer.end_stage(6)
                    emit(
                        stage=6,
                        message=f"{day_label} · {src_msg} — {inserted:,} rows in memory",
                        rows=total_rows,
                        source_day_index=si + 1,
                        source_day_total=n_sources,
                        clear_substage=True,
                    )
                    del ctx, day_rows
                    gc.collect()

                for end_stage in (7, 8):
                    timer.skip_stage(end_stage)

                elapsed = time.perf_counter() - t0
                pl = timer.snapshot(rows=total_rows, estimated_total_rows=0)
                if status == "cancelled":
                    on_done({
                        "status": "cancelled",
                        "debug_features": True,
                        "error": "Cancelled",
                        "rows": total_rows,
                        "ticks_in_memory": total_ticks,
                        "pipeline": pl,
                    })
                else:
                    on_done(debug_feature_done_payload(
                        rows=total_rows,
                        feature_count=len(implemented),
                        groups_run=len(per_group),
                        sources_loaded=n_sources,
                        elapsed_sec=elapsed,
                        ticks_in_memory=total_ticks,
                        spot_ticks=total_spot,
                        chain_ticks=total_chain,
                        pipeline=pl,
                        feature_readiness=total_readiness or None,
                    ))
            except Exception as exc:
                pl = timer.snapshot(rows=total_rows, estimated_total_rows=0)
                on_done({
                    "status": "failed",
                    "debug_features": True,
                    "error": str(exc),
                    "rows": total_rows,
                    "ticks_in_memory": total_ticks,
                    "pipeline": pl,
                })

        self._thread = threading.Thread(target=_worker, name="master-debug-features", daemon=True)
        self._thread.start()


class MasterRegistryExportRunner:
    """Background export of filtered master SQLite rows to dataset registry."""

    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def start(
        self,
        *,
        export_kwargs: dict[str, Any],
        on_done: Callable[[dict[str, Any]], None],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if self.running:
            raise RuntimeError("A registry export is already running")

        def _worker() -> None:
            from chain_replay_ml.dataset_builder.master_registry_export import (
                MasterRegistryExportError,
                create_master_registry_dataset,
            )

            result: dict[str, Any]
            try:
                payload = create_master_registry_dataset(
                    chart_data_dir(self.chart_dir),
                    on_progress=on_progress,
                    **export_kwargs,
                )
                result = {**payload, "status": "completed"}
            except MasterRegistryExportError as exc:
                result = {"status": "failed", "error": exc.detail}
            except Exception as exc:
                result = {"status": "failed", "error": str(exc)}
            on_done(result)

        self._thread = threading.Thread(target=_worker, name="master-registry-export", daemon=True)
        self._thread.start()


class AnalysisDatasetRunner:
    """Background Analysis Dataset build (Registry ∪ Pipeline Features)."""

    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._thread: threading.Thread | None = None
        self._cancel = False

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def cancel(self) -> None:
        self._cancel = True

    def start(
        self,
        *,
        export_kwargs: dict[str, Any],
        on_done: Callable[[dict[str, Any]], None],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if self.running:
            raise RuntimeError("An analysis dataset build is already running")
        self._cancel = False

        def _worker() -> None:
            from chain_replay_ml.dataset_builder.analysis_dataset_export import (
                create_analysis_dataset,
            )
            from chain_replay_ml.dataset_builder.master_registry_export import (
                MasterRegistryExportError,
            )

            result: dict[str, Any]
            try:
                payload = create_analysis_dataset(
                    chart_data_dir(self.chart_dir),
                    on_progress=on_progress,
                    cancel_check=lambda: self._cancel,
                    **export_kwargs,
                )
                result = {**payload, "status": payload.get("status") or "completed"}
            except MasterRegistryExportError as exc:
                result = {"status": "failed", "error": getattr(exc, "detail", None) or str(exc)}
            except Exception as exc:
                result = {"status": "failed", "error": str(exc)}
            on_done(result)

        self._thread = threading.Thread(target=_worker, name="analysis-dataset-build", daemon=True)
        self._thread.start()
