"""Build prediction meta SQLite from master dataset rows."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.dataset_builder.master_naming import normalize_market_slug, resolve_master_db_path
from chain_replay_ml.dataset_builder.master_store import MasterStore
from chain_replay_ml.dataset_builder.day_context import (
    SourceSpec,
    load_day_context,
    token_timelines_from_day_context,
)
from chain_replay_ml.replay_feature_scoring import resolve_scoring_expiry, chart_dir_from_data_dir

from live_inference.history import PredictionSnapshotHistory
from live_inference.registry_cache import acquire_inference_registry
from live_inference.snapshot import PredictionSnapshot
from live_inference.versions import feature_version

from .batch_predict import BatchModelRunner
from .meta_features import (
    build_prediction_id,
    extend_ensemble_meta,
    model_deltas_and_ranks,
    resolve_minutes_to_expiry,
)
from .live_metrics import LiveMetricsTracker
from .model_registry import resolve_or_register_prediction_version, read_prediction_versions
from .outcomes import (
    compute_path_outcomes,
    compute_prediction_quality,
    map_point_outcomes,
    prepare_path_outcome_timelines,
)
from chain_replay_ml.model_lab.prediction_schema import (
    actual_ltp_column_from_target,
    horizon_sec_from_target,
)
from .store import PredictionMetaStore

ProgressFn = Callable[[dict[str, Any]], None]

_IDENTITY_COLS = (
    "prediction_id", "trading_day", "timestamp", "token", "strike", "option_type", "symbol", "market", "expiry",
)
_RECORD_COLS = (
    "prediction_timestamp", "feature_version", "prediction_version", "model_registry_version", "prediction_time_ms",
)
_CONTEXT_COLS = ("current_ltp", "current_spot", "minutes_to_expiry")
_CONFIDENCE_COLS = (
    "prediction_min", "prediction_max", "prediction_range_pct",
    "mean_minus_current_ltp", "median_minus_current_ltp",
)
_META_COLS = (
    "ensemble_mean", "ensemble_median", "ensemble_std", "ensemble_spread",
    "agreement", "models_ok", "models_failed",
    "prediction_velocity", "prediction_acceleration", "prediction_trend",
)
_OUTCOME_COLS = (
    "actual_30s_ltp", "actual_1m_ltp", "actual_3m_ltp", "actual_5m_ltp",
    "actual_high_5m", "actual_low_5m", "actual_max_profit_5m", "actual_max_drawdown_5m",
    "ticks_above_entry_5m", "ticks_below_entry_5m",
    "time_to_max_profit", "time_to_max_drawdown",
    "prediction_error", "direction_correct",
)


def resolve_prediction_meta_db_path(
    data_dir: str,
    *,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    filename: str | None = None,
) -> str:
    datasets = os.path.join(data_dir, "datasets")
    os.makedirs(datasets, exist_ok=True)
    slug = normalize_market_slug(market)
    name = filename or f"prediction_meta_{slug}_{int(sampling_interval_sec)}s.db"
    return os.path.join(datasets, name)


def _emit(on_progress: ProgressFn | None, **payload: Any) -> None:
    if on_progress:
        on_progress(payload)


def _registry_version(reg_meta: Any) -> str:
    if hasattr(reg_meta, "models_dir_signature"):
        return str(reg_meta.models_dir_signature or "")
    if isinstance(reg_meta, dict):
        return str(reg_meta.get("models_dir_signature") or "")
    return str(reg_meta or "")


def _slope_over(history: PredictionSnapshotHistory, key: str, window_sec: float) -> float | None:
    from live_inference.meta_engine import _slope_over as _slope

    return _slope(history, key, window_sec)


def _meta_momentum(
    history: PredictionSnapshotHistory,
    meta: dict[str, Any],
    *,
    timestamp: float,
    token: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    vel = _slope_over(history, "ensemble_mean", 1.0)
    if vel is not None:
        out["prediction_velocity"] = vel
    prior_vel = history.meta_value_at_offset("prediction_velocity", 1.0)
    if vel is not None and prior_vel is not None:
        out["prediction_acceleration"] = round(vel - prior_vel, 6)
    trend = _slope_over(history, "ensemble_mean", 5.0)
    if trend is not None:
        out["prediction_trend"] = trend

    combined = {**meta, **out}
    history.append(
        PredictionSnapshot.create(
            timestamp=float(timestamp),
            token=str(token),
            results={},
            feature_version=feature_version(),
            prediction_version="meta-offline",
        ),
        combined,
    )
    return out


def _load_timelines_for_day(
    data_dir: str,
    trading_day: str,
    *,
    market: str,
    step_sec: int,
) -> dict[str, Any]:
    chart_dir = chart_dir_from_data_dir(data_dir)
    expiry_resolution = resolve_scoring_expiry(chart_dir, trading_day, None, underlying=market)
    resolved_expiry = str(expiry_resolution.get("resolved_expiry") or "").strip()
    if not resolved_expiry:
        return {}
    source = SourceSpec(
        source_id=f"{trading_day}|{market}|{resolved_expiry}",
        trading_day=trading_day,
        market=market,
        expiry=resolved_expiry,
    )
    try:
        ctx = load_day_context(chart_dir, source, feature_grid_step_sec=step_sec)
    except Exception:
        return {}
    return token_timelines_from_day_context(ctx)


def _resume_where_clause(prog: Any) -> tuple[str, list[Any]]:
    if not prog.last_trading_day:
        return "", []
    return (
        "(trading_day > ?) OR (trading_day = ? AND (timestamp > ? OR (timestamp = ? AND token > ?)))",
        [
            prog.last_trading_day,
            prog.last_trading_day,
            float(prog.last_timestamp or 0),
            float(prog.last_timestamp or 0),
            str(prog.last_token or ""),
        ],
    )


def _compose_query_filters(
    prog: Any,
    *,
    resume: bool,
    trading_days_filter: list[str] | None,
    selection_spec: Any | None = None,
) -> tuple[str, list[Any]]:
    from chain_replay_ml.dataset_builder.dataset_selection_engine import (
        DatasetSelectionSpec,
        build_selection_sql_where,
    )

    clauses: list[str] = []
    args: list[Any] = []
    if selection_spec is not None:
        where_sql, where_args = build_selection_sql_where(
            selection_spec,
            profile="master_samples",
        )
        if where_sql != "1=1":
            clauses.append(f"({where_sql})")
            args.extend(where_args)
    elif trading_days_filter:
        spec = DatasetSelectionSpec(selected_days=list(trading_days_filter))
        where_sql, where_args = build_selection_sql_where(
            spec,
            profile="master_samples",
        )
        if where_sql != "1=1":
            clauses.append(f"({where_sql})")
            args.extend(where_args)
    if resume:
        resume_sql, resume_args = _resume_where_clause(prog)
        if resume_sql:
            clauses.append(resume_sql)
            args.extend(resume_args)
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), args


def _count_distinct_days(out_store: PredictionMetaStore) -> int:
    row = out_store.conn.execute("SELECT COUNT(DISTINCT trading_day) FROM samples").fetchone()
    return int(row[0]) if row else 0


def _resolve_path_outcome_horizon_sec(
    specs: list[dict[str, Any]],
    project_config: dict[str, Any] | None,
) -> float:
    """Authoritative horizon for path MFE/MAE — from project or model targets."""
    cfg = project_config if isinstance(project_config, dict) else {}
    raw_hz = cfg.get("horizon_sec")
    if raw_hz is not None and str(raw_hz).strip() != "":
        hz = float(raw_hz)
        if hz > 0:
            return hz
    for key in ("target", "target_column"):
        t = str(cfg.get(key) or "").strip()
        if t:
            return horizon_sec_from_target(t)
    horizons: list[float] = []
    for spec in specs:
        t = str(spec.get("target") or "").strip()
        if not t:
            continue
        try:
            horizons.append(horizon_sec_from_target(t))
        except ValueError:
            continue
    if horizons:
        return max(horizons)
    raise ValueError(
        "Cannot resolve path-outcome horizon from project_config or model targets"
    )


def build_prediction_meta_dataset(
    data_dir: str,
    *,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    master_db_path: str | None = None,
    output_db_path: str | None = None,
    batch_size: int = 1000,
    resume: bool = True,
    enrich_path_outcomes: bool = True,
    selected_models: list[str] | None = None,
    trading_days_filter: list[str] | None = None,
    project_config: dict[str, Any] | None = None,
    project_id: str | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Stream master rows → batch predict → project prediction SQLite."""
    master_path = master_db_path or resolve_master_db_path(
        data_dir, market=market, sampling_interval_sec=sampling_interval_sec,
    )
    if not os.path.isfile(master_path):
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    out_path = output_db_path or resolve_prediction_meta_db_path(
        data_dir, market=market, sampling_interval_sec=sampling_interval_sec,
    )

    specs, _merged, union_features, reg_meta = acquire_inference_registry(
        data_dir, status_filter="ready",
    )
    if not specs:
        raise RuntimeError("No ready regression models found in model registry")

    if selected_models:
        from .projects import filter_specs_by_selection

        specs = filter_specs_by_selection(specs, selected_models)
        if not specs:
            raise RuntimeError("None of the selected models are available in the registry")
    if not specs:
        raise RuntimeError("No models to run")

    runner = BatchModelRunner(specs)
    path_horizon_sec = _resolve_path_outcome_horizon_sec(specs, project_config)
    # Point-actual column for quality metrics — same authoritative target as path horizon.
    try:
        if isinstance(project_config, dict) and (
            project_config.get("target") or project_config.get("target_column")
        ):
            tgt = str(
                project_config.get("target_column") or project_config.get("target") or ""
            ).strip()
        else:
            tgt = str(specs[0].get("target") or "").strip()
        actual_ltp_col = actual_ltp_column_from_target(tgt) if tgt else None
    except ValueError:
        actual_ltp_col = None
    if not actual_ltp_col:
        raise ValueError(
            "Cannot resolve actual LTP column from project/model target — "
            "no default prediction horizon"
        )
    model_cols = runner.pred_columns
    delta_cols = runner.delta_columns
    rank_cols = runner.rank_columns
    model_names = [str(s.get("model_name") or "") for s in specs]
    registry_version = _registry_version(reg_meta) + "|models:" + ",".join(model_names)
    inference_sig = _registry_version(reg_meta)
    feat_ver = feature_version()

    with MasterStore(master_path) as master:
        build_schema = master.get_meta("build_schema") or {}
        feature_cols = list(build_schema.get("feature_columns") or union_features)
        if not feature_cols:
            feature_cols = list(union_features)

        total = master.total_row_count()
        if trading_days_filter:
            placeholders = ", ".join("?" for _ in trading_days_filter)
            row = master.conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE trading_day IN ({placeholders})",
                list(trading_days_filter),
            ).fetchone()
            total = int(row[0]) if row else 0
        read_cols = list(dict.fromkeys([
            *_IDENTITY_COLS[1:], "ltp", "spot", "minutes_to_expiry", *feature_cols,
            "future_ltp_30s", "future_ltp_1m", "future_ltp_3m", "future_ltp_180s", "future_ltp_5m",
        ]))
        existing = {row[1] for row in master.conn.execute("PRAGMA table_info(samples)").fetchall()}
        read_cols = [
            c for c in read_cols
            if c in existing or c in _IDENTITY_COLS or c in ("ltp", "spot", "minutes_to_expiry")
        ]

        output_cols = list(dict.fromkeys([
            *_IDENTITY_COLS,
            *_RECORD_COLS,
            *_CONTEXT_COLS,
            *model_cols,
            *delta_cols,
            *rank_cols,
            *_CONFIDENCE_COLS,
            *_META_COLS,
            *_OUTCOME_COLS,
        ]))

        with PredictionMetaStore(out_path, batch_size=batch_size) as out_store:
            prog_before = out_store.read_progress()
            stored_reg = out_store.get_meta("model_registry_version")
            active_ver = out_store.get_meta("active_prediction_version")
            if (
                resume
                and stored_reg == registry_version
                and active_ver is not None
                and prog_before.status in ("running", "paused", "failed", "complete")
            ):
                pred_version = int(active_ver)
            else:
                pred_version = resolve_or_register_prediction_version(
                    out_store.conn,
                    data_dir,
                    specs,
                    model_registry_version=registry_version,
                    default_feature_version=feat_ver,
                )
            prog = out_store.start_job(rows_total=total, resume=resume)
            out_store.prepare_insert(output_cols)
            out_store.set_meta("source_master_db", os.path.basename(master_path))
            out_store.set_meta("model_catalog", runner.model_catalog())
            out_store.set_meta("registry_meta", reg_meta.as_dict() if hasattr(reg_meta, "as_dict") else reg_meta)
            out_store.set_meta("active_prediction_version", pred_version)
            out_store.set_meta("model_registry_version", registry_version)
            if project_config:
                blob = {**project_config, "prediction_version": pred_version, "feature_version": feat_ver}
                out_store.set_meta("project_config", blob)
            if project_id:
                from .projects import update_project_after_build

                slot_count = len(read_prediction_versions(out_store.conn))
                update_project_after_build(
                    data_dir,
                    project_id,
                    prediction_version=pred_version,
                    started=True,
                    build_status="running",
                    rows_planned=total,
                    model_registry_version=registry_version,
                    inference_registry_signature=inference_sig,
                    model_registry_slot_count=slot_count,
                )

            where_sql, where_args = _compose_query_filters(
                prog_before, resume=resume, trading_days_filter=trading_days_filter,
            )
            col_sql = ", ".join(f'"{c}"' for c in read_cols)
            query = (
                f"SELECT {col_sql} FROM samples {where_sql} "
                "ORDER BY token, trading_day, timestamp"
            )
            cur = master.conn.execute(query, where_args)

            histories: dict[str, PredictionSnapshotHistory] = defaultdict(PredictionSnapshotHistory)
            timeline_cache: dict[str, dict[str, Any]] = {}
            rows_done = int(prog.rows_done or 0)
            stats = {
                "rows_written": 0,
                "batches": 0,
                "models_per_row": len(specs),
                "predictions_ok": 0,
                "predictions_failed": 0,
            }
            live = LiveMetricsTracker(
                models_per_row=len(specs),
                rows_at_start=rows_done,
                rows_done=rows_done,
                rows_total=total,
                prediction_version=pred_version,
            )
            live.set_cache_stats(
                registry_cache_hit=bool(getattr(reg_meta, "cache_hit", False)),
                models_loaded_from_disk=runner.models_loaded_from_disk,
                model_count=len(specs),
            )
            out_store.set_meta("live_dashboard", live.snapshot())

            _emit(
                on_progress,
                phase="start",
                rows_done=rows_done,
                rows_total=total,
                models=len(specs),
                prediction_version=pred_version,
            )

            try:
                while True:
                    raw = cur.fetchmany(batch_size)
                    if not raw:
                        break
                    t_batch0 = time.perf_counter()
                    batch_df = pd.DataFrame(raw, columns=read_cols)
                    stats["batches"] += 1
                    t_feat0 = time.perf_counter()
                    feature_ms = (t_feat0 - t_batch0) * 1000.0

                    t_pred0 = time.perf_counter()
                    pred_df, ok_df, batch_ms, batch_pred_ts, feature_valid_pct = runner.predict_batch(
                        batch_df, feature_cols,
                    )
                    prediction_ms = (time.perf_counter() - t_pred0) * 1000.0
                    per_row_ms = round(batch_ms / max(len(batch_df), 1), 3)

                    out_rows: list[dict[str, Any]] = []
                    batch_live = {
                        "predictions_ok": 0,
                        "predictions_failed": 0,
                        "skipped_rows": 0,
                        "failed_model_rows": 0,
                        "outcome_completed": 0,
                        "outcome_pending": 0,
                        "agreement_values": [],
                        "spread_values": [],
                        "direction_values": [],
                    }
                    for i in range(len(batch_df)):
                        src = batch_df.iloc[i].to_dict()
                        token = str(src.get("token") or "")
                        ts = float(src.get("timestamp") or 0)
                        trading_day = str(src.get("trading_day") or "")
                        strike = src.get("strike")
                        option_type = src.get("option_type")
                        entry_ltp = src.get("ltp")
                        try:
                            entry_f = float(entry_ltp) if entry_ltp is not None else None
                        except (TypeError, ValueError):
                            entry_f = None

                        preds_row = pred_df.iloc[i]
                        ok_row = ok_df.iloc[i]
                        model_pred_vals: list[float | None] = []
                        for col in model_cols:
                            val = preds_row.get(col)
                            model_pred_vals.append(float(val) if pd.notna(val) else None)
                        ok_vals = [v for v in model_pred_vals if v is not None]
                        models_ok = int(sum(int(ok_row.get(f"model_{j}_ok", 0)) for j in range(1, len(specs) + 1)))
                        models_failed = len(specs) - models_ok
                        stats["predictions_ok"] += models_ok
                        stats["predictions_failed"] += models_failed
                        batch_live["predictions_ok"] += models_ok
                        batch_live["predictions_failed"] += models_failed
                        if models_ok == 0:
                            batch_live["skipped_rows"] += 1
                        if models_failed > 0:
                            batch_live["failed_model_rows"] += 1

                        meta = extend_ensemble_meta(
                            ok_vals,
                            models_ok=models_ok,
                            models_failed=models_failed,
                            entry_ltp=entry_f,
                        )
                        deltas, ranks = model_deltas_and_ranks(
                            model_pred_vals,
                            ensemble_mean=meta.get("ensemble_mean"),
                        )

                        hist = histories[token]
                        momentum = _meta_momentum(hist, meta, timestamp=ts, token=token)
                        meta.update(momentum)

                        outcomes = map_point_outcomes(src, entry_f)
                        if enrich_path_outcomes and entry_f is not None and token:
                            if trading_day not in timeline_cache:
                                timeline_cache[trading_day] = _load_timelines_for_day(
                                    data_dir, trading_day,
                                    market=str(src.get("market") or market),
                                    step_sec=sampling_interval_sec,
                                )
                                prepare_path_outcome_timelines(timeline_cache[trading_day])
                            tl = timeline_cache[trading_day].get(token)
                            outcomes.update(
                                compute_path_outcomes(
                                    tl,
                                    ts=ts,
                                    entry_ltp=entry_f,
                                    horizon_sec=path_horizon_sec,
                                )
                            )

                        quality = compute_prediction_quality(
                            ensemble_mean=meta.get("ensemble_mean"),
                            entry_ltp=entry_f,
                            actual_ltp=outcomes.get(actual_ltp_col),
                        )
                        actual_ref = outcomes.get(actual_ltp_col)
                        if actual_ref is not None:
                            batch_live["outcome_completed"] += 1
                        else:
                            batch_live["outcome_pending"] += 1
                        if meta.get("agreement") is not None:
                            batch_live["agreement_values"].append(meta.get("agreement"))
                        if meta.get("ensemble_spread") is not None:
                            batch_live["spread_values"].append(meta.get("ensemble_spread"))
                        if quality.get("direction_correct") is not None:
                            batch_live["direction_values"].append(quality.get("direction_correct"))

                        pred_id = build_prediction_id(
                            trading_day=trading_day,
                            timestamp=ts,
                            strike=strike,
                            option_type=option_type,
                            token=token,
                            prediction_version=pred_version,
                        )
                        mte = resolve_minutes_to_expiry(src, timestamp=ts)

                        row_out: dict[str, Any] = {
                            "prediction_id": pred_id,
                            "trading_day": trading_day,
                            "timestamp": ts,
                            "token": token,
                            "strike": strike,
                            "option_type": option_type,
                            "symbol": src.get("symbol"),
                            "market": src.get("market"),
                            "expiry": src.get("expiry"),
                            "prediction_timestamp": batch_pred_ts,
                            "feature_version": feat_ver,
                            "prediction_version": pred_version,
                            "model_registry_version": registry_version,
                            "prediction_time_ms": per_row_ms,
                            "current_ltp": entry_f,
                            "current_spot": src.get("spot"),
                            "minutes_to_expiry": mte,
                        }
                        for j, col in enumerate(model_cols, start=1):
                            row_out[col] = model_pred_vals[j - 1]
                        row_out.update(deltas)
                        row_out.update(ranks)
                        row_out.update(meta)
                        row_out.update(outcomes)
                        row_out.update(quality)
                        out_rows.append(row_out)

                    t_sql0 = time.perf_counter()
                    written = out_store.insert_rows(out_rows)
                    sqlite_ms = (time.perf_counter() - t_sql0) * 1000.0
                    stats["rows_written"] += written
                    rows_done += len(out_rows)
                    live.update_batch(
                        batch_rows=len(out_rows),
                        feature_ms=feature_ms,
                        prediction_ms=prediction_ms,
                        sqlite_ms=sqlite_ms,
                        feature_valid_pct=feature_valid_pct,
                        rows_done=rows_done,
                        batch_stats=batch_live,
                    )
                    out_store.set_meta("live_dashboard", live.snapshot())
                    last = out_rows[-1]
                    out_store.update_checkpoint(
                        rows_done=rows_done,
                        trading_day=str(last["trading_day"]),
                        timestamp=float(last["timestamp"]),
                        token=str(last["token"]),
                    )
                    _emit(
                        on_progress,
                        phase="batch",
                        rows_done=rows_done,
                        rows_total=total,
                        batch=stats["batches"],
                        pct=round(100.0 * rows_done / max(total, 1), 2),
                        dashboard=live.snapshot(),
                    )

                live.status = "complete"
                out_store.set_meta("live_dashboard", live.snapshot())
                out_store.mark_complete(rows_done)
                if project_id:
                    from .projects import get_project, update_project_after_build

                    update_project_after_build(
                        data_dir,
                        project_id,
                        prediction_version=pred_version,
                        finished=True,
                        build_status="complete",
                        rows_planned=total,
                        prediction_row_count=rows_done,
                        model_registry_version=registry_version,
                        inference_registry_signature=inference_sig,
                        model_registry_slot_count=len(read_prediction_versions(out_store.conn)),
                    )
                if project_config:
                    from .projects import get_project

                    proj = get_project(data_dir, project_id) if project_id else None
                    fp = (proj.build_fingerprint if proj else None) or {}
                    out_store.set_meta("project_config", {
                        **project_config,
                        "prediction_version": pred_version,
                        "feature_version": feat_ver,
                        "trading_days": _count_distinct_days(out_store),
                        "rows_written": rows_done,
                        "build_fingerprint": fp,
                    })
                    if fp:
                        out_store.set_meta("build_fingerprint", fp)
            except Exception as exc:
                live.status = "failed"
                out_store.set_meta("live_dashboard", live.snapshot())
                out_store.mark_failed(str(exc))
                raise

            _emit(on_progress, phase="complete", rows_done=rows_done, rows_total=total, stats=stats, dashboard=live.snapshot())

            try:
                from chain_replay_ml.dataset_builder.dataset_selection_engine import DatasetSelectionSpec
                from chain_replay_ml.dataset_builder.selection_preview_calibration import record_selection_calibration

                day_spec = DatasetSelectionSpec(
                    selected_days=list(trading_days_filter or []),
                    market=market,
                    interval_sec=sampling_interval_sec,
                )
                record_selection_calibration(
                    data_dir,
                    build_kind="prediction_meta",
                    spec=day_spec,
                    preview=None,
                    actual_rows=rows_done,
                    actual_days=_count_distinct_days(out_store),
                    build_job_id=pred_version,
                    master_db_path=master_path,
                )
            except Exception:
                pass

            return {
                "status": "complete",
                "master_db": master_path,
                "output_db": out_path,
                "rows_total": total,
                "rows_written": rows_done,
                "prediction_version": pred_version,
                "model_registry_version": registry_version,
                "model_catalog": runner.model_catalog(),
                "stats": stats,
            }
