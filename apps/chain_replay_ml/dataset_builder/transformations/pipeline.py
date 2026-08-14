"""Ordered transformation pipeline executor."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .config import (
    TRANSFORMATION_PIPELINE_VERSION,
    default_transformation_config,
    normalize_transformation_config,
)
from .registry import (
    get_transformation,
    list_registered_transformations,
    registered_transformation_count,
)
from .time_shift import (
    extract_sample_interval_from_config,
    normalize_sample_interval_value,
)


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""

    frame: pd.DataFrame
    config: dict[str, Any] = field(default_factory=default_transformation_config)
    registered: int = 0
    enabled: int = 0
    executed: int = 0
    elapsed_sec: float = 0.0
    executed_ids: list[str] = field(default_factory=list)
    step_results: list[TransformationResult] = field(default_factory=list)
    created_columns: list[str] = field(default_factory=list)
    # Experiment identity: wall-clock meaning of row-span packaging features.
    sample_interval_sec: float | int | None = None
    # Day-at-a-time / Fast-Safe instrumentation (optional).
    execution: dict[str, Any] = field(default_factory=dict)
    # Lineage gap warnings (missing parents in dependency graph).
    lineage_warnings: list[str] = field(default_factory=list)

    @property
    def metadata_block(self) -> dict[str, Any]:
        """Versioned configuration persisted into dataset metadata."""
        cfg = normalize_transformation_config(self.config)
        block: dict[str, Any] = {
            "transformation_pipeline_version": int(
                cfg.get("transformation_pipeline_version") or TRANSFORMATION_PIPELINE_VERSION
            ),
            "transformations": list(cfg.get("transformations") or []),
        }
        interval = normalize_sample_interval_value(self.sample_interval_sec)
        if interval is None:
            interval = extract_sample_interval_from_config(cfg)
        if interval is not None:
            # Top-level experiment identity (also mirrored under sampling.interval_sec).
            block["sample_interval_sec"] = interval
        if self.execution:
            block["execution"] = dict(self.execution)
        return block

    @property
    def metadata_transformations(self) -> list[Any]:
        """Backward-compatible list accessor."""
        return list(self.metadata_block.get("transformations") or [])


def _lineage_warnings_for_config(
    config: dict[str, Any],
    *,
    known_columns: list[str] | None = None,
) -> list[str]:
    """Validate pipeline dependency edges; return warning strings."""
    try:
        from ..pipeline_no_null_report import (
            build_pipeline_lineage_map,
            validate_pipeline_lineage_parents,
        )

        lineage = build_pipeline_lineage_map(config)
        return validate_pipeline_lineage_parents(
            lineage,
            known_columns=known_columns,
        )
    except Exception:
        return []


def describe_pipeline(config: dict[str, Any] | None = None) -> PipelineResult:
    """Inspect registry + config without touching a DataFrame."""
    cfg = normalize_transformation_config(config)
    enabled = _resolve_enabled_transforms(cfg)
    warnings = _lineage_warnings_for_config(cfg, known_columns=None)
    return PipelineResult(
        frame=pd.DataFrame(),
        config=cfg,
        registered=registered_transformation_count(),
        enabled=len(enabled),
        executed=0,
        elapsed_sec=0.0,
        executed_ids=[],
        step_results=[],
        created_columns=[],
        lineage_warnings=warnings,
    )


def format_pipeline_log_lines(result: PipelineResult) -> list[str]:
    lines = [
        "Transformation Pipeline",
        f"    Registered : {int(result.registered)}",
        f"    Enabled    : {int(result.enabled)}",
        f"    Executed   : {int(result.executed)}",
        f"    Elapsed    : {float(result.elapsed_sec):.2f} s",
    ]
    for step in result.step_results:
        label = step.transformation_name or step.transformation_id or "transform"
        lines.append(f"    {label}")
        lines.append(f"        Created Columns : {len(step.created_columns)}")
        lines.append(f"        Elapsed         : {float(step.elapsed_sec):.2f} s")
        lines.append(f"        Rows            : {int(step.rows_processed)}")
    for w in list(result.lineage_warnings or [])[:20]:
        lines.append(f"    WARNING: {w}")
    if len(result.lineage_warnings or []) > 20:
        lines.append(
            f"    WARNING: … and {len(result.lineage_warnings) - 20} more lineage gaps"
        )
    return lines


def run_transformation_pipeline(
    df: pd.DataFrame,
    config: dict[str, Any] | list[Any] | None = None,
    *,
    context: TransformContext | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Execute enabled transformations in dependency-aware order.

    When nothing is enabled, returns the same DataFrame object unchanged.
    """
    cfg = normalize_transformation_config(config)
    registered = registered_transformation_count()
    ctx = context or TransformContext(config=cfg)
    ctx.config = cfg
    if getattr(ctx, "data_dir", None):
        from ..pipeline_features_config import sanitize_transformation_config_before_execution

        cfg, skipped_retired = sanitize_transformation_config_before_execution(
            cfg,
            str(ctx.data_dir),
        )
        ctx.config = cfg
        log = log_fn or getattr(ctx, "logger", None)
        if log and skipped_retired:
            for name in skipped_retired:
                log(f"SKIPPED_RETIRED_SOURCE: {name}")
    enabled = _resolve_enabled_transforms(cfg)
    if log_fn is not None and ctx.logger is None:
        ctx.logger = log_fn

    t0 = time.perf_counter()
    executed_ids: list[str] = []
    step_results: list[TransformationResult] = []
    created_all: list[str] = []
    out = df
    if enabled:
        for transform in enabled:
            if ctx.cancelled():
                break
            step = _run_one(transform, out, ctx)
            # Drop prior frame reference so GC can reclaim between stages.
            if out is not step.frame and out is not df:
                del out
            out = step.frame
            executed_ids.append(str(transform.id))
            step_results.append(step)
            created_all.extend(list(step.created_columns))
    elapsed = max(time.perf_counter() - t0, 0.0)

    interval = normalize_sample_interval_value(getattr(ctx, "sample_interval_sec", None))
    if interval is None:
        interval = extract_sample_interval_from_config(cfg)

    result = PipelineResult(
        frame=out,
        config=cfg,
        registered=registered,
        enabled=len(enabled),
        executed=len(executed_ids),
        elapsed_sec=round(elapsed, 4),
        executed_ids=executed_ids,
        step_results=step_results,
        created_columns=list(dict.fromkeys(created_all)),
        sample_interval_sec=interval,
        lineage_warnings=_lineage_warnings_for_config(
            cfg,
            known_columns=[str(c) for c in out.columns],
        ),
    )
    if log_fn is not None:
        for line in format_pipeline_log_lines(result):
            try:
                log_fn(line)
            except Exception:
                pass
    return result


_MAX_PARTITION_ROWS = 20_000
_INTERACTION_PAIR_BATCH = 25
# Soft Fast-mode budget: rows × cols × 8 × copy_factor (pandas).
_FAST_COPY_FACTOR = 2.5
_FAST_RAM_BUDGET_BYTES = 6 * (1024**3)  # ~6 GiB peak estimate


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    return "unable to allocate" in str(exc).lower()


def _estimate_peak_bytes(rows: int, cols: int) -> int:
    return int(max(0, rows) * max(1, cols) * 8 * _FAST_COPY_FACTOR)


def _log(log_fn: Callable[[str], None] | None, msg: str) -> None:
    if log_fn is None:
        return
    try:
        log_fn(msg)
    except Exception:
        pass


def _parquet_day_token_index(parquet_path: str) -> dict[str, list[str]]:
    """Map trading_day → ordered unique tokens (empty string if no token col)."""
    try:
        peek = pd.read_parquet(parquet_path, columns=["trading_day", "token"])
    except Exception:
        peek = pd.read_parquet(parquet_path, columns=["trading_day"])
        peek["token"] = ""
    if peek.empty:
        return {}
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for day, token in zip(
        peek["trading_day"].astype(str),
        peek["token"].astype(str) if "token" in peek.columns else [""] * len(peek),
    ):
        day_s = str(day).strip()
        tok_s = str(token).strip()
        if not day_s:
            continue
        if day_s not in out:
            out[day_s] = []
            seen[day_s] = set()
        if tok_s not in seen[day_s]:
            seen[day_s].add(tok_s)
            out[day_s].append(tok_s)
    return out


def _parquet_partition_keys(parquet_path: str) -> list[tuple[str, str]]:
    """Return unique (trading_day, token) keys without loading numeric columns."""
    index = _parquet_day_token_index(parquet_path)
    keys: list[tuple[str, str]] = []
    for day, tokens in index.items():
        for token in tokens:
            keys.append((day, token))
    return keys


def _read_partition_frame(
    parquet_path: str,
    *,
    trading_day: str,
    token: str | None = None,
) -> pd.DataFrame:
    """Read one trading_day (optionally one token) via pyarrow dataset filters.

    Phase P2: Arrow → Polars → Pandas adapter (same caller contract).
    """
    import pyarrow.dataset as ds

    from chain_replay_ml.frame_backend import arrow_table_to_pandas

    dataset = ds.dataset(parquet_path, format="parquet")
    filt = ds.field("trading_day") == trading_day
    if token:
        filt = filt & (ds.field("token") == token)
    table = dataset.to_table(filter=filt)
    df, _bridge = arrow_table_to_pandas(table, via_polars=True)
    return df


def _split_config_waves(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a pipeline config into low-memory waves (Safe mode; Interaction batched)."""
    entries = [e for e in (cfg.get("transformations") or []) if isinstance(e, dict)]
    waves: list[dict[str, Any]] = []
    for entry in entries:
        if not bool(entry.get("enabled", False)):
            continue
        tid = str(entry.get("id") or "").strip()
        if tid == "interaction":
            params = dict(entry.get("params") or {})
            pairs = list(params.get("pairs") or [])
            if not pairs:
                continue
            for i in range(0, len(pairs), _INTERACTION_PAIR_BATCH):
                chunk = pairs[i : i + _INTERACTION_PAIR_BATCH]
                wave_entry = dict(entry)
                wave_params = dict(params)
                wave_params["pairs"] = chunk
                wave_params["fail_on_duplicate_output"] = False
                wave_params["overwrite"] = True
                wave_entry["params"] = wave_params
                waves.append({
                    "transformation_pipeline_version": cfg.get(
                        "transformation_pipeline_version", TRANSFORMATION_PIPELINE_VERSION
                    ),
                    "transformations": [wave_entry],
                })
        else:
            waves.append({
                "transformation_pipeline_version": cfg.get(
                    "transformation_pipeline_version", TRANSFORMATION_PIPELINE_VERSION
                ),
                "transformations": [entry],
            })
    return waves


def _chunk_frame_with_warmup(
    df: pd.DataFrame,
    *,
    max_rows: int,
    warmup_rows: int,
) -> list[tuple[pd.DataFrame, int]]:
    """Split a large partition into chunks; each chunk overlaps ``warmup_rows`` with the prior.

    Returns list of (chunk_df, drop_prefix_rows) — drop_prefix_rows are discarded after
    transform so lag/rolling warmup is not double-written.
    """
    n = len(df)
    if n <= max_rows:
        return [(df, 0)]
    out: list[tuple[pd.DataFrame, int]] = []
    start = 0
    while start < n:
        if start == 0:
            end = min(n, max_rows)
            out.append((df.iloc[start:end].copy(), 0))
            start = end
        else:
            warm_start = max(0, start - warmup_rows)
            end = min(n, start + max_rows)
            chunk = df.iloc[warm_start:end].copy()
            drop = start - warm_start
            out.append((chunk, drop))
            start = end
    return out


def _child_context(
    ctx: TransformContext,
    cfg: dict[str, Any],
    *,
    quiet: bool = True,
) -> TransformContext:
    return TransformContext(
        config=cfg,
        data_dir=ctx.data_dir,
        dataset_name=ctx.dataset_name,
        sample_interval_sec=ctx.sample_interval_sec,
        dataset_info=dict(ctx.dataset_info or {}),
        metadata=dict(ctx.metadata or {}),
        logger=None if quiet else ctx.logger,
        progress_callback=None if quiet else ctx.progress_callback,
        cancel_token=ctx.cancel_token,
        extras={},
    )


def _align_table_to_writer(table: Any, writer: Any, pa_mod: Any) -> Any:
    """Deprecated silent align — prefer validate_table_against_locked_schema."""
    del pa_mod
    return table.select(list(writer.schema.names))


def _call_day_progress(
    progress_fn: Callable[..., None] | None,
    msg: str,
    cur: int,
    tot: int,
    **detail: Any,
) -> None:
    if progress_fn is None:
        return
    try:
        progress_fn(msg, cur, tot, **detail)
    except TypeError:
        try:
            progress_fn(msg, cur, tot)
        except Exception:
            pass
    except Exception:
        pass


def _transform_frame_waves(
    df: pd.DataFrame,
    waves: list[dict[str, Any]],
    *,
    ctx: TransformContext,
    warmup_rows: int,
    max_partition_rows: int,
    created_all: list[str],
    executed_ids: list[str],
    transform_timings: dict[str, float],
) -> pd.DataFrame:
    """Safe-mode path for one in-memory partition: waves + optional chunking."""
    import gc

    out = df
    for wave_cfg in waves:
        pieces = _chunk_frame_with_warmup(
            out,
            max_rows=max(1000, int(max_partition_rows)),
            warmup_rows=warmup_rows,
        )
        merged: list[pd.DataFrame] = []
        for chunk_df, drop_prefix in pieces:
            t_step = time.perf_counter()
            pipe = run_transformation_pipeline(
                chunk_df,
                wave_cfg,
                context=_child_context(ctx, wave_cfg),
                log_fn=None,
            )
            frame = pipe.frame
            if drop_prefix:
                frame = frame.iloc[drop_prefix:].copy()
            for c in pipe.created_columns:
                if c not in created_all:
                    created_all.append(c)
            for eid in pipe.executed_ids:
                if eid not in executed_ids:
                    executed_ids.append(eid)
                transform_timings[eid] = transform_timings.get(eid, 0.0) + (
                    time.perf_counter() - t_step
                )
            merged.append(frame)
            del chunk_df, pipe
        out = pd.concat(merged, ignore_index=True) if len(merged) > 1 else merged[0]
        del merged
        gc.collect()
    return out


def run_transformation_pipeline_on_parquet(
    parquet_path: str,
    config: dict[str, Any] | list[Any] | None = None,
    *,
    context: TransformContext | None = None,
    log_fn: Callable[[str], None] | None = None,
    partition_col: str = "trading_day",
    on_partition_progress: Callable[..., None] | None = None,
    max_partition_rows: int = _MAX_PARTITION_ROWS,
    resume: bool = True,
) -> PipelineResult:
    """Apply the pipeline one trading day at a time, then move to the next day.

    Fast (per day): load day once → run full pipeline in memory → write day part.
    Safe (per day, on OOM / high estimate): token partitions, optional waves/chunks.
    Schema is locked from the first written day; later days fail hard on mismatch.
    Completed days are checkpointed so a crash can resume without restarting Day 1.
    """
    from .day_at_a_time import run_day_at_a_time_pipeline

    del partition_col
    return run_day_at_a_time_pipeline(
        parquet_path,
        config,
        context=context,
        log_fn=log_fn,
        on_partition_progress=on_partition_progress,
        max_partition_rows=max_partition_rows,
        resume=resume,
        helpers={
            "PipelineResult": PipelineResult,
            "TransformContext": TransformContext,
            "_resolve_enabled_transforms": _resolve_enabled_transforms,
            "_parquet_day_token_index": _parquet_day_token_index,
            "_read_partition_frame": _read_partition_frame,
            "_split_config_waves": _split_config_waves,
            "_estimate_peak_bytes": _estimate_peak_bytes,
            "_FAST_RAM_BUDGET_BYTES": _FAST_RAM_BUDGET_BYTES,
            "_is_oom_error": _is_oom_error,
            "_log": _log,
            "_call_day_progress": _call_day_progress,
            "run_transformation_pipeline": run_transformation_pipeline,
            "_child_context": _child_context,
            "_transform_frame_waves": _transform_frame_waves,
            "format_pipeline_log_lines": format_pipeline_log_lines,
        },
    )


def _run_one(
    transform: FeatureTransformation,
    df: pd.DataFrame,
    context: TransformContext,
) -> TransformationResult:
    t0 = time.perf_counter()
    raw = transform.transform(df, context)
    elapsed = max(time.perf_counter() - t0, 0.0)
    if isinstance(raw, TransformationResult):
        if not raw.transformation_id:
            raw.transformation_id = str(transform.id)
        if not raw.transformation_name:
            raw.transformation_name = str(transform.name or transform.id)
        if raw.elapsed_sec <= 0:
            raw.elapsed_sec = round(elapsed, 4)
        if raw.rows_processed <= 0:
            raw.rows_processed = int(len(raw.frame))
        return raw
    # Defensive: older-style DataFrame return
    return TransformationResult.passthrough(
        raw if isinstance(raw, pd.DataFrame) else df,
        transformation_id=str(transform.id),
        transformation_name=str(transform.name or transform.id),
        elapsed_sec=round(elapsed, 4),
    )


def _resolve_enabled_transforms(config: dict[str, Any]) -> list[FeatureTransformation]:
    """Build enabled transform instances, sorted by order then dependency order."""
    entries = list(config.get("transformations") or [])
    if not entries:
        return []

    registry_meta = {
        t.id: t for t in list_registered_transformations()
    }
    resolved: list[FeatureTransformation] = []
    enabled_ids: set[str] = set()
    entry_by_id: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not bool(entry.get("enabled", False)):
            continue
        tid = str(entry.get("id") or "").strip()
        if not tid:
            continue
        try:
            inst = get_transformation(tid)
        except KeyError:
            continue
        inst.enabled = True
        if entry.get("order") is not None:
            try:
                inst.order = int(entry["order"])
            except (TypeError, ValueError):
                inst.order = int(getattr(registry_meta.get(tid), "order", inst.order) or inst.order)
        cfg_deps = entry.get("depends_on")
        if isinstance(cfg_deps, (list, tuple)) and cfg_deps:
            inst.depends_on = [str(d).strip() for d in cfg_deps if str(d).strip()]
        else:
            inst.depends_on = list(getattr(type(inst), "depends_on", []) or [])
        params = entry.get("params")
        if isinstance(params, dict):
            # Stash params on instance for transforms that read them.
            setattr(inst, "params", dict(params))
        resolved.append(inst)
        enabled_ids.add(tid)
        entry_by_id[tid] = entry

    # Soft dependency check — missing deps raise so misconfig is obvious.
    for inst in resolved:
        missing = [d for d in (inst.depends_on or []) if d not in enabled_ids]
        if missing:
            raise ValueError(
                f"Transformation '{inst.id}' depends on {missing}, "
                f"but those are not enabled in the pipeline config."
            )

    # Order by declared order, then stable topological pass on depends_on.
    resolved.sort(key=lambda t: (int(t.order), str(t.id)))
    return _topo_sort(resolved)


def _topo_sort(items: list[FeatureTransformation]) -> list[FeatureTransformation]:
    by_id = {t.id: t for t in items}
    pending = list(items)
    done: list[FeatureTransformation] = []
    seen: set[str] = set()
    while pending:
        progress = False
        nxt: list[FeatureTransformation] = []
        for t in pending:
            deps = [d for d in (t.depends_on or []) if d in by_id]
            if all(d in seen for d in deps):
                done.append(t)
                seen.add(t.id)
                progress = True
            else:
                nxt.append(t)
        if not progress:
            # Cycle or unresolved — fall back to current order remainder.
            done.extend(nxt)
            break
        pending = nxt
    return done
