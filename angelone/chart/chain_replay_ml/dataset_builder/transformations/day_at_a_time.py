"""Day-at-a-time Fast/Safe transform runner with schema lock + resume."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import normalize_transformation_config
from .day_pipeline_support import (
    SchemaMismatchError,
    build_transformation_summary,
    checkpoint_path_for,
    clear_day_artifacts,
    load_checkpoint,
    parts_dir_for,
    safe_day_filename,
    save_checkpoint,
    serialize_schema,
    source_fingerprint,
    validate_table_against_locked_schema,
)
from .registry import registered_transformation_count
from .time_shift import (
    extract_sample_interval_from_config,
    normalize_sample_interval_value,
)


CODEC = "zstd"
WARMUP_MODE = "within_day"


def run_day_at_a_time_pipeline(
    parquet_path: str,
    config: dict[str, Any] | list[Any] | None = None,
    *,
    context: Any = None,
    log_fn: Callable[[str], None] | None = None,
    on_partition_progress: Callable[..., None] | None = None,
    max_partition_rows: int = 20_000,
    resume: bool = True,
    # Injected from pipeline module to avoid circular imports at definition time.
    helpers: dict[str, Any],
) -> Any:
    """Transform one trading day at a time with schema lock and optional resume."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    PipelineResult = helpers["PipelineResult"]
    TransformContext = helpers["TransformContext"]
    _resolve_enabled_transforms = helpers["_resolve_enabled_transforms"]
    _parquet_day_token_index = helpers["_parquet_day_token_index"]
    _read_partition_frame = helpers["_read_partition_frame"]
    _split_config_waves = helpers["_split_config_waves"]
    _estimate_peak_bytes = helpers["_estimate_peak_bytes"]
    _FAST_RAM_BUDGET_BYTES = helpers["_FAST_RAM_BUDGET_BYTES"]
    _is_oom_error = helpers["_is_oom_error"]
    _log = helpers["_log"]
    _call_day_progress = helpers["_call_day_progress"]
    run_transformation_pipeline = helpers["run_transformation_pipeline"]
    _child_context = helpers["_child_context"]
    _transform_frame_waves = helpers["_transform_frame_waves"]
    format_pipeline_log_lines = helpers["format_pipeline_log_lines"]

    cfg = normalize_transformation_config(config)
    enabled = _resolve_enabled_transforms(cfg)
    if not enabled:
        return PipelineResult(
            frame=pd.DataFrame(),
            config=cfg,
            registered=registered_transformation_count(),
            enabled=0,
            executed=0,
            created_columns=[],
        )

    ctx = context or TransformContext(config=cfg)
    ctx.config = cfg
    if log_fn is not None and ctx.logger is None:
        ctx.logger = log_fn

    day_index = _parquet_day_token_index(parquet_path)
    days = list(day_index.keys())
    if not days:
        raise ValueError("No trading_day partitions found in parquet export.")

    try:
        total_rows_est = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        total_rows_est = 0
    try:
        base_cols = len(pq.read_schema(parquet_path).names)
    except Exception:
        base_cols = 64

    enabled_count = len(enabled)
    created_est = max(32, enabled_count * 4)
    waves = _split_config_waves(cfg)
    interval = float(ctx.sample_interval_sec or 3.0) or 3.0
    warmup_rows = max(64, int(1800.0 / interval) + 2)

    progress_fn = on_partition_progress or (
        (lambda msg, cur, tot, **_kw: ctx.report_progress(msg, cur, tot))
    )

    created_all: list[str] = []
    executed_ids: list[str] = []
    transform_timings: dict[str, float] = {}
    mode_by_day: dict[str, str] = {}
    day_timings: dict[str, dict[str, float]] = {}
    day_stats: dict[str, dict[str, Any]] = {}

    t0 = time.perf_counter()
    read_sec = 0.0
    transform_sec = 0.0
    write_sec = 0.0
    gc_sec = 0.0
    peak_ram_bytes = 0
    total_days = len(days)

    parts_dir = parts_dir_for(parquet_path)
    os.makedirs(parts_dir, exist_ok=True)
    checkpoint = load_checkpoint(parquet_path) if resume else {}
    fp = source_fingerprint(parquet_path)
    if checkpoint and fp and checkpoint.get("source") != fp:
        _log(log_fn, "Checkpoint source parquet changed — starting fresh (not resuming)")
        clear_day_artifacts(parquet_path)
        checkpoint = {}
    completed = {
        str(d)
        for d in (checkpoint.get("completed_days") or [])
        if str(d).strip()
    }
    locked_schema = list(checkpoint.get("schema") or [])
    resumed = bool(completed)
    # Drop stale completed days not in current day list.
    completed = {d for d in completed if d in set(days)}
    if resumed:
        for day in list(completed):
            part = os.path.join(parts_dir, f"{safe_day_filename(day)}.parquet")
            if not os.path.isfile(part):
                completed.discard(day)
        for day in completed:
            mode_by_day[day] = str((checkpoint.get("mode_by_day") or {}).get(day) or "fast")
            prev_stats = dict((checkpoint.get("day_stats") or {}).get(day) or {})
            if prev_stats:
                day_stats[day] = prev_stats
                day_timings[day] = {
                    "read_sec": float(prev_stats.get("read_sec") or 0.0),
                    "transform_sec": float(prev_stats.get("transform_sec") or 0.0),
                    "write_sec": float(prev_stats.get("write_sec") or 0.0),
                    "total_sec": float(prev_stats.get("total_sec") or 0.0),
                }
        for c in checkpoint.get("created_columns") or []:
            if c not in created_all:
                created_all.append(str(c))
        for eid in checkpoint.get("executed_ids") or []:
            if eid not in executed_ids:
                executed_ids.append(str(eid))

    _log(
        log_fn,
        f"Feature Transformations: day-at-a-time · {total_days} day(s) · "
        f"{enabled_count} enabled stages · codec={CODEC} · warmup={WARMUP_MODE}"
        + (f" · resume {len(completed)} done" if resumed else ""),
    )

    def _persist_checkpoint() -> None:
        save_checkpoint(
            parquet_path,
            {
                "version": 1,
                "completed_days": [d for d in days if d in completed],
                "schema": locked_schema,
                "mode_by_day": {d: mode_by_day.get(d) for d in days if d in completed},
                "day_stats": {d: day_stats.get(d) for d in days if d in completed},
                "created_columns": list(created_all),
                "executed_ids": list(executed_ids),
                "warmup_mode": WARMUP_MODE,
                "codec": CODEC,
                "source": source_fingerprint(parquet_path),
            },
        )

    def _write_day_part(day: str, frame: pd.DataFrame) -> int:
        nonlocal locked_schema, write_sec, peak_ram_bytes
        if frame is None or frame.empty:
            return 0
        tw = time.perf_counter()
        try:
            from chain_replay_ml.frame_backend import frame_to_arrow_table_via_polars

            table = frame_to_arrow_table_via_polars(frame, coerce=False)
        except Exception:
            table = pa.Table.from_pandas(frame, preserve_index=False)
        if not locked_schema:
            locked_schema = serialize_schema(table.schema)
        else:
            validate_table_against_locked_schema(table, locked_schema, day=day)
            # Enforce Day-1 column order.
            table = table.select(schema_names_safe(locked_schema))
        part_path = os.path.join(parts_dir, f"{safe_day_filename(day)}.parquet")
        pq.write_table(table, part_path, compression=CODEC)
        nrows = int(table.num_rows)
        ncols = int(table.num_columns)
        est = _estimate_peak_bytes(nrows, ncols)
        if est > peak_ram_bytes:
            peak_ram_bytes = est
        elapsed_w = time.perf_counter() - tw
        write_sec += elapsed_w
        del table
        return nrows

    def schema_names_safe(schema_doc: list[dict[str, str]]) -> list[str]:
        return [str(f.get("name") or "") for f in schema_doc]

    try:
        for day_i, day in enumerate(days, start=1):
            if ctx.cancelled():
                break
            if day in completed:
                _call_day_progress(
                    progress_fn,
                    f"Feature Transformations: day {day_i}/{total_days} ({day}) · resumed skip",
                    day_i,
                    total_days,
                    day=day,
                    day_index=day_i,
                    day_total=total_days,
                    mode=str(mode_by_day.get(day) or "resumed"),
                    rows=int((day_stats.get(day) or {}).get("rows") or 0),
                    features=len(locked_schema) or None,
                    resumed=True,
                )
                continue

            tokens = list(day_index.get(day) or [""])
            day_t0 = time.perf_counter()
            day_read = 0.0
            day_xform = 0.0
            day_write_start = write_sec
            day_mode = "fast"
            day_rows = 0
            day_peak = 0

            _call_day_progress(
                progress_fn,
                f"Feature Transformations: day {day_i}/{total_days} ({day})",
                day_i - 1,
                total_days,
                day=day,
                day_index=day_i,
                day_total=total_days,
                mode="pending",
                features=len(locked_schema) or (base_cols + created_est),
            )

            def _accumulate_pipe(pipe: Any) -> None:
                for step in pipe.step_results:
                    tid = str(step.transformation_id or "transform")
                    transform_timings[tid] = transform_timings.get(tid, 0.0) + float(
                        step.elapsed_sec or 0.0
                    )
                for c in pipe.created_columns:
                    if c not in created_all:
                        created_all.append(c)
                for eid in pipe.executed_ids:
                    if eid not in executed_ids:
                        executed_ids.append(eid)

            fast_ok = False
            df_day: pd.DataFrame | None = None
            try:
                tr = time.perf_counter()
                df_day = _read_partition_frame(parquet_path, trading_day=day, token=None)
                elapsed_r = time.perf_counter() - tr
                day_read += elapsed_r
                read_sec += elapsed_r

                if df_day.empty:
                    mode_by_day[day] = "empty"
                    day_stats[day] = {
                        "rows": 0,
                        "mode": "empty",
                        "total_sec": round(time.perf_counter() - day_t0, 3),
                        "peak_ram_bytes": 0,
                    }
                    completed.add(day)
                    _persist_checkpoint()
                    continue

                if (
                    total_rows_est > 0
                    and len(df_day) >= total_rows_est
                    and total_days > 1
                ):
                    raise MemoryError(
                        f"Day filter failed for {day}: loaded {len(df_day):,} rows "
                        "(full table). Refusing Fast mode."
                    )

                peak = _estimate_peak_bytes(len(df_day), base_cols + created_est)
                day_peak = max(day_peak, peak)
                if peak > _FAST_RAM_BUDGET_BYTES and len(tokens) > 1:
                    raise MemoryError(
                        f"Fast RAM estimate {peak / (1024**3):.1f} GiB exceeds budget "
                        f"for {day} ({len(df_day):,} rows) — using Safe token path"
                    )

                _call_day_progress(
                    progress_fn,
                    f"Feature Transformations: day {day_i}/{total_days} ({day}) · "
                    f"Fast · {len(df_day):,} rows",
                    day_i - 1,
                    total_days,
                    day=day,
                    day_index=day_i,
                    day_total=total_days,
                    mode="fast",
                    rows=len(df_day),
                    features=len(locked_schema) or (base_cols + created_est),
                    peak_ram_bytes=peak,
                )
                tx = time.perf_counter()
                pipe = run_transformation_pipeline(
                    df_day,
                    cfg,
                    context=_child_context(ctx, cfg, quiet=True),
                    log_fn=None,
                )
                elapsed_x = time.perf_counter() - tx
                day_xform += elapsed_x
                transform_sec += elapsed_x
                _accumulate_pipe(pipe)
                day_rows = _write_day_part(day, pipe.frame)
                del df_day, pipe
                df_day = None
                fast_ok = True
                day_mode = "fast"
            except SchemaMismatchError:
                raise
            except Exception as exc:
                if not _is_oom_error(exc) and "exceeds budget" not in str(exc).lower():
                    raise
                _log(log_fn, f"Day {day}: Fast unavailable ({exc}) — Safe token path")
                if df_day is not None:
                    del df_day
                    df_day = None
                tg = time.perf_counter()
                gc.collect()
                gc_sec += time.perf_counter() - tg

            if not fast_ok:
                day_mode = "safe"
                day_frames: list[pd.DataFrame] = []
                for tok_i, token in enumerate(tokens, start=1):
                    if ctx.cancelled():
                        break
                    label = f"{day}" + (f" / {token}" if token else "")
                    _call_day_progress(
                        progress_fn,
                        f"Feature Transformations: day {day_i}/{total_days} · "
                        f"Safe token {tok_i}/{len(tokens)} ({label})",
                        day_i - 1,
                        total_days,
                        day=day,
                        day_index=day_i,
                        day_total=total_days,
                        mode="safe",
                        token_index=tok_i,
                        token_total=len(tokens),
                        features=len(locked_schema) or (base_cols + created_est),
                    )
                    tr = time.perf_counter()
                    df_tok = _read_partition_frame(
                        parquet_path, trading_day=day, token=token or None,
                    )
                    elapsed_r = time.perf_counter() - tr
                    day_read += elapsed_r
                    read_sec += elapsed_r
                    if df_tok.empty:
                        continue
                    if (
                        total_rows_est > 0
                        and len(df_tok) >= total_rows_est
                        and (total_days > 1 or len(tokens) > 1)
                    ):
                        raise MemoryError(
                            f"Partition filter failed for {label}: loaded "
                            f"{len(df_tok):,} rows (full table)."
                        )
                    try:
                        tx = time.perf_counter()
                        pipe = run_transformation_pipeline(
                            df_tok,
                            cfg,
                            context=_child_context(ctx, cfg, quiet=True),
                            log_fn=None,
                        )
                        elapsed_x = time.perf_counter() - tx
                        day_xform += elapsed_x
                        transform_sec += elapsed_x
                        _accumulate_pipe(pipe)
                        day_frames.append(pipe.frame)
                        del df_tok, pipe
                    except SchemaMismatchError:
                        raise
                    except Exception as exc:
                        if not _is_oom_error(exc):
                            raise
                        _log(
                            log_fn,
                            f"Token {label}: full pipeline OOM — wave/chunk Safe",
                        )
                        try:
                            del df_tok
                        except Exception:
                            pass
                        tg = time.perf_counter()
                        gc.collect()
                        gc_sec += time.perf_counter() - tg
                        tr = time.perf_counter()
                        df_tok = _read_partition_frame(
                            parquet_path, trading_day=day, token=token or None,
                        )
                        elapsed_r = time.perf_counter() - tr
                        day_read += elapsed_r
                        read_sec += elapsed_r
                        if not waves:
                            raise
                        n_waves = len(waves)
                        for wi, wave_cfg in enumerate(waves, start=1):
                            _call_day_progress(
                                progress_fn,
                                f"Feature Transformations: day {day_i}/{total_days} · "
                                f"Safe wave {wi}/{n_waves}",
                                day_i - 1,
                                total_days,
                                day=day,
                                day_index=day_i,
                                day_total=total_days,
                                mode="safe",
                                token_index=tok_i,
                                token_total=len(tokens),
                                wave_index=wi,
                                wave_total=n_waves,
                            )
                        tx = time.perf_counter()
                        frame = _transform_frame_waves(
                            df_tok,
                            waves,
                            ctx=ctx,
                            warmup_rows=warmup_rows,
                            max_partition_rows=max_partition_rows,
                            created_all=created_all,
                            executed_ids=executed_ids,
                            transform_timings=transform_timings,
                        )
                        elapsed_x = time.perf_counter() - tx
                        day_xform += elapsed_x
                        transform_sec += elapsed_x
                        day_frames.append(frame)
                        del df_tok, frame
                        day_mode = "safe_waves"

                if day_frames:
                    merged = (
                        pd.concat(day_frames, ignore_index=True)
                        if len(day_frames) > 1
                        else day_frames[0]
                    )
                    day_rows = _write_day_part(day, merged)
                    del day_frames, merged

            day_write = write_sec - day_write_start
            mode_by_day[day] = day_mode
            day_timings[day] = {
                "read_sec": round(day_read, 3),
                "transform_sec": round(day_xform, 3),
                "write_sec": round(day_write, 3),
                "total_sec": round(time.perf_counter() - day_t0, 3),
            }
            day_stats[day] = {
                **day_timings[day],
                "rows": int(day_rows),
                "mode": day_mode,
                "peak_ram_bytes": int(day_peak or 0),
            }
            completed.add(day)
            _persist_checkpoint()
            _log(
                log_fn,
                f"Day {day_i}/{total_days} {day} complete · mode={day_mode} · "
                f"rows={day_rows:,} · "
                f"read {day_read:.1f}s · transform {day_xform:.1f}s · "
                f"write {day_write:.1f}s · total {day_timings[day]['total_sec']:.1f}s",
            )
            tg = time.perf_counter()
            gc.collect()
            gc_sec += time.perf_counter() - tg
            _call_day_progress(
                progress_fn,
                f"Feature Transformations: day {day_i}/{total_days} ({day}) done · "
                f"mode={day_mode}",
                day_i,
                total_days,
                day=day,
                day_index=day_i,
                day_total=total_days,
                mode=day_mode,
                rows=day_rows,
                features=len(locked_schema) or None,
                peak_ram_bytes=day_peak or None,
            )
    except SchemaMismatchError:
        raise

    # Merge day parts → final parquet (schema already locked).
    out_tmp = f"{parquet_path}.by_day.tmp"
    if os.path.isfile(out_tmp):
        try:
            os.remove(out_tmp)
        except OSError:
            pass
    writer = None
    try:
        for day in days:
            if mode_by_day.get(day) == "empty":
                continue
            part = os.path.join(parts_dir, f"{safe_day_filename(day)}.parquet")
            if not os.path.isfile(part):
                continue
            table = pq.read_table(part)
            if locked_schema:
                validate_table_against_locked_schema(table, locked_schema, day=day)
                table = table.select([str(f.get("name") or "") for f in locked_schema])
            if writer is None:
                writer = pq.ParquetWriter(out_tmp, table.schema, compression=CODEC)
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()

    if not os.path.isfile(out_tmp):
        raise ValueError("Day-at-a-time Feature Transformations produced no output.")
    os.replace(out_tmp, parquet_path)
    clear_day_artifacts(parquet_path)

    elapsed = max(time.perf_counter() - t0, 0.0)
    interval_out = normalize_sample_interval_value(getattr(ctx, "sample_interval_sec", None))
    if interval_out is None:
        interval_out = extract_sample_interval_from_config(cfg)

    output_columns = len(locked_schema) if locked_schema else None
    summary = build_transformation_summary(
        days=days,
        mode_by_day=mode_by_day,
        day_stats=day_stats,
        created_columns=created_all,
        feature_count=base_cols,
        output_columns=output_columns,
        elapsed_sec=elapsed,
        codec=CODEC,
        warmup_mode=WARMUP_MODE,
        resumed=resumed,
        peak_ram_bytes=peak_ram_bytes or None,
    )
    modes_used = sorted({m for m in mode_by_day.values() if m and m != "empty"})
    _log(
        log_fn,
        f"Feature Transformations complete: {total_days} day(s) · "
        f"modes={','.join(modes_used) or 'none'} · "
        f"{len(created_all)} created columns · {elapsed:.1f}s "
        f"(read {read_sec:.1f}s / transform {transform_sec:.1f}s / "
        f"write {write_sec:.1f}s / gc {gc_sec:.1f}s)",
    )
    from .day_pipeline_support import format_transformation_summary_text

    for line in format_transformation_summary_text(summary).splitlines():
        _log(log_fn, line)
    if transform_timings:
        top = sorted(transform_timings.items(), key=lambda kv: kv[1], reverse=True)[:12]
        for tid, sec in top:
            _log(log_fn, f"    transform {tid}: {sec:.1f}s")

    execution = {
        "strategy": "day_at_a_time",
        "mode_by_day": mode_by_day,
        "day_timings": day_timings,
        "day_stats": day_stats,
        "transformation_summary": summary,
        "warmup_mode": WARMUP_MODE,
        "codec": CODEC,
        "schema_locked": bool(locked_schema),
        "schema_columns": len(locked_schema),
        "resumed": resumed,
        "read_sec": round(read_sec, 3),
        "transform_sec": round(transform_sec, 3),
        "write_sec": round(write_sec, 3),
        "gc_sec": round(gc_sec, 3),
        "transform_timings_sec": {k: round(v, 3) for k, v in transform_timings.items()},
        "peak_ram_bytes": peak_ram_bytes or None,
    }
    result = PipelineResult(
        frame=pd.DataFrame(),
        config=cfg,
        registered=registered_transformation_count(),
        enabled=len(enabled),
        executed=len(executed_ids),
        elapsed_sec=round(elapsed, 4),
        executed_ids=executed_ids,
        step_results=[],
        created_columns=list(dict.fromkeys(created_all)),
        sample_interval_sec=interval_out,
        execution=execution,
    )
    if log_fn is not None:
        for line in format_pipeline_log_lines(result):
            _log(log_fn, line)
    return result
