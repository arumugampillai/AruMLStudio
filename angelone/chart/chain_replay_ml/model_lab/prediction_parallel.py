"""Multi-worker trading-day parallelization for Model Lab prediction builds."""

from __future__ import annotations

import gc
import hashlib
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chain_replay_ml.prediction_meta.meta_features import (
    build_prediction_id,
    resolve_minutes_to_expiry,
)
from chain_replay_ml.prediction_meta.outcomes import (
    PATH_OUTCOME_PROFILE_KEYS,
    compute_path_outcomes,
    move_trend,
    prepare_path_outcome_timelines,
)
from chain_replay_ml.training.dataset_loader import DatasetLoaderError
from chain_replay_ml.training.inference_runtime import (
    InferenceRuntimeInfo,
    batch_predict_day,
    format_day_stage_timings,
    format_path_outcome_microprofile,
    format_per_prediction_outcome_timings,
    load_prediction_model_for_inference,
    resolve_outcome_profile_rows,
)

from .prediction_io import count_trading_day_rows, load_trading_day_frame
from .prediction_progress import (
    STAGE_ENRICH,
    STAGE_LOAD_DAY,
    STAGE_PREDICT,
    STAGE_WRITE,
    ProgressHub,
)
from .prediction_schema import (
    compute_error_metrics,
    compute_rr_hit_labels,
    has_complete_prediction_horizon,
)
from .store import ModelLabStore

ProgressFn = Callable[[dict[str, Any]], None]

DEFAULT_PREDICTION_WORKERS = 1
_SENTINEL = object()


def _f(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _timeline_last_ts(timeline: Any) -> float | None:
    stamps = getattr(timeline, "timestamps", None) if timeline is not None else None
    if not stamps:
        return None
    try:
        return float(stamps[-1])
    except (TypeError, ValueError, IndexError):
        return None


def _exclude_incomplete_horizon_rows(
    day_df: Any,
    *,
    horizon_sec: float,
    timelines: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """Drop rows whose future window exceeds available tick data.

    Requires per-token timelines so the cutoff is the last available tick for
    that option — never a hardcoded market close, and never the last sample
    timestamp (which is already clipped and would double-trim).
    """
    if day_df is None or len(day_df) == 0 or float(horizon_sec) <= 0:
        return day_df, 0
    if not timelines or "timestamp" not in day_df.columns or "token" not in day_df.columns:
        return day_df, 0

    token_ends: dict[str, float] = {}
    for tok, tl in timelines.items():
        end = _timeline_last_ts(tl)
        if end is not None:
            token_ends[str(tok)] = end
    if not token_ends:
        return day_df, 0

    before = int(len(day_df))
    ts_series = day_df["timestamp"].map(_f)
    ends = day_df["token"].map(lambda t: token_ends.get(str(t or ""), None))
    keep = [
        has_complete_prediction_horizon(
            timestamp=ts,
            data_end_ts=end,
            horizon_sec=horizon_sec,
        )
        for ts, end in zip(ts_series.tolist(), ends.tolist())
    ]
    out = day_df.loc[keep].copy()
    return out, before - int(len(out))


@dataclass
class DayJobContext:
    data_dir: str
    parquet_path: str
    features: list[str]
    target: str
    wanted_columns: list[str]
    lab_uuid: str
    feat_map: dict[str, str]
    horizon_sec: float
    enrich_path_outcomes: bool
    model_path: str
    algorithm: str | None
    days: list[str]
    row_limit: int | None = None
    mark_day_complete: bool = True
    embed_features: bool = True
    master_db_path: str | None = None
    stamp_master_row_ids: bool = False
    master_filter: dict[str, Any] | None = None
    inference_device: str = "CPU"
    inference_algorithm: str | None = None
    profile_outcome_rows: int | None = None
    # Prediction-package member contracts (role / model_path / features /
    # prediction_type / output_column). Executed generically per member;
    # unavailable members still materialize their output column as NULL.
    package_members: list[dict[str, Any]] = field(default_factory=list)
    # Parent dataset Feature Transformation pipeline (shared framework).
    # Applied when a day is loaded from Master so Lag / future transforms match
    # the training feature space before model-feature validation.
    transformation_config: dict[str, Any] | None = None
    sample_interval_sec: float | None = None
    parent_dataset: str | None = None
    tb_model_name: str | None = None


@dataclass
class DayBatchResult:
    day_index: int
    day: str
    worker_id: int
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    elapsed_sec: float = 0.0
    sum_abs: float = 0.0
    n_abs: int = 0
    sum_err: float = 0.0
    n_err: int = 0
    sum_prem: float = 0.0
    n_prem: int = 0
    dir_hits: int = 0
    dir_n: int = 0
    skipped: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    inference_device: str = "CPU"
    stage_timing_log: str = ""
    outcome_profile_ms: list[float] = field(default_factory=list)
    per_prediction_timing_log: str = ""
    path_microprofile_log: str = ""


def tb_all_null_warning(
    *, tb_model_name: str | None, rows: list[dict[str, Any]]
) -> str | None:
    """Warning text when Triple Barrier was enabled but scored 0 of N rows.

    Catches silent TB failure (feature mismatch, model resolve failure,
    predict exception, ...) that degrades every row for a day to NULL
    ``tb_*`` without ever failing the day/build itself.
    """
    name = str(tb_model_name or "").strip()
    if not name or not rows:
        return None
    if any(r.get("tb_pred_probability") is not None for r in rows):
        return None
    return (
        f"Triple Barrier enabled ('{name}') but scored 0 of {len(rows):,} rows — "
        "tb_model_name/tb_label_run/tb_pred_probability/tb_pred_class are all "
        "NULL for this day. Check the TB model's features are available on "
        "this day and that the model resolves correctly."
    )


def _load_day_timelines(data_dir: str, trading_day: str, *, market: str, step_sec: int = 3) -> dict[str, Any]:
    try:
        from chain_replay_ml.prediction_meta.builder import _load_timelines_for_day

        return _load_timelines_for_day(
            data_dir,
            trading_day,
            market=market or "NIFTY",
            step_sec=step_sec,
        )
    except Exception:
        return {}


def run_package_member_predictions(
    *,
    members: list[dict[str, Any]],
    day_df: Any,
) -> tuple[dict[str, Any], list[str]]:
    """
    Generic prediction-package member pass over one day frame.

    Every member is executed by contract only: slice its feature list from the
    already-loaded day frame, predict, and collect values under
    ``output_column``. There is no role-specific branching — the wrapper
    (``prediction_type``) decides how raw output is shaped. Members that are
    unavailable or whose features are absent yield ``None`` (NULL column)
    without failing the day.

    Returns (output_column → ndarray | None, skip notes).
    """
    import numpy as np

    from chain_replay_ml.training.inference_runtime import force_cpu_inference
    from chain_replay_ml.training.model_runtime import load_prediction_model_cached

    from .prediction_schema import align_features_to_model

    outputs: dict[str, Any] = {}
    notes: list[str] = []
    for member in members or []:
        column = str(member.get("output_column") or "").strip()
        if not column:
            continue
        outputs[column] = None
        role = str(member.get("role") or column)
        if not member.get("available") or not member.get("model_path"):
            continue
        features = [str(f) for f in (member.get("features") or []) if str(f or "").strip()]
        missing = [f for f in features if f not in day_df.columns]
        if not features or missing:
            notes.append(f"{role}: day frame missing features {missing[:3]}")
            continue
        try:
            member_model, _ms, _disk = load_prediction_model_cached(
                str(member["model_path"]), member.get("algorithm")
            )
            # Ladder members are secondary scorers riding along with the
            # primary GPU-configured model in this worker process — never
            # let them re-acquire GPU/CUDA context (see force_cpu_inference).
            force_cpu_inference(member_model, member.get("algorithm"))
            ordered = align_features_to_model(features, member_model)
            X_member = day_df.loc[:, ordered]
            if hasattr(member_model, "predict_proba"):
                raw = np.asarray(member_model.predict_proba(X_member), dtype=np.float64)
                values = raw[:, -1] if raw.ndim > 1 else raw
            else:
                values = np.asarray(batch_predict_day(member_model, X_member), dtype=np.float64)
            if str(member.get("prediction_type") or "") == "probability":
                values = np.clip(values, 0.0, 1.0)
            outputs[column] = values
        except Exception as exc:
            outputs[column] = None
            notes.append(f"{role}: {exc}")
    return outputs, notes


def run_tb_model_predictions(
    *,
    data_dir: str,
    tb_model_name: str | None,
    day_df: Any,
) -> tuple[str | None, str | None, Any, Any, list[str]]:
    """Run optional Triple Barrier model scorer on day frame.

    Returns (model_name, label_run_id, probabilities_array, classes_array, notes).
    """
    if not tb_model_name or day_df is None or len(day_df) == 0:
        return None, None, None, None, []
    import numpy as np
    from chain_replay_ml.training.inference_runtime import force_cpu_inference
    from chain_replay_ml.training.model_runtime import (
        load_prediction_model_cached,
        resolve_prediction_model_package,
    )
    from .prediction_schema import align_features_to_model

    notes: list[str] = []
    try:
        pkg = resolve_prediction_model_package(data_dir, tb_model_name)
        label_run = str(pkg.get("label_run_id") or "triple_barrier").strip() or None
        if not pkg.get("ok"):
            notes.append(str(pkg.get("error") or f"Triple Barrier model '{tb_model_name}' could not be resolved"))
            return tb_model_name, label_run, None, None, notes

        features = list(pkg.get("features") or [])
        missing = [f for f in features if f not in day_df.columns]
        if not features or missing:
            notes.append(f"Triple Barrier model missing features: {missing[:3]}")
            return tb_model_name, label_run, None, None, notes

        tb_model, _load_ms, _from_disk = load_prediction_model_cached(
            str(pkg["model_path"]), pkg.get("algorithm")
        )
        # The TB side-scorer shares this worker process with the primary
        # GPU-configured model. A saved booster restores whatever device it
        # was trained with (often cuda); force CPU here so TB scoring can
        # never contend for GPU memory with the primary model — that
        # contention is a plausible cause of hard (non-Python-catchable)
        # worker crashes when TB scoring is enabled.
        force_cpu_inference(tb_model, pkg.get("algorithm"))

        ordered = align_features_to_model(features, tb_model)
        X_tb = day_df.loc[:, ordered]

        # Native-format model loaders in this codebase (xgboost booster,
        # lightgbm booster, catboost/RF wrappers) never expose a real
        # sklearn ``predict_proba`` — ``.predict()`` already returns
        # whatever shape the trained objective produces: a 1-D probability
        # for binary-remapped TP-hit models, or a (n, n_classes) probability
        # matrix for true multiclass Triple Barrier models. Prefer
        # ``predict_proba`` when a wrapper does provide it (defensive).
        has_proba = hasattr(tb_model, "predict_proba")
        if has_proba:
            raw = np.asarray(tb_model.predict_proba(X_tb), dtype=np.float64)
        else:
            raw = np.asarray(tb_model.predict(X_tb), dtype=np.float64)

        # Side-scorer probability: P(barrier outcome == TP / positive class).
        # Label encoding is TP=0, SL=1, TIME=2 (outcome_label_engine.triple_barrier),
        # so a true multiclass probability matrix's column 0 is P(TP). Binary
        # remapped TP-hit models put P(TP) in column 1 of a 2-column matrix
        # (label 1 = "TP hit"), or directly as a 1-D probability.
        if raw.ndim > 1 and raw.shape[1] >= 3:
            probs = np.clip(raw[:, 0], 0.0, 1.0)
        elif raw.ndim > 1 and raw.shape[1] == 2:
            probs = np.clip(raw[:, 1], 0.0, 1.0)
        elif raw.ndim > 1:
            probs = np.clip(raw[:, 0], 0.0, 1.0)
        else:
            probs = np.clip(raw, 0.0, 1.0)

        if has_proba and hasattr(tb_model, "predict"):
            classes = np.asarray(tb_model.predict(X_tb), dtype=np.int32)
        elif raw.ndim > 1:
            classes = np.argmax(raw, axis=1).astype(np.int32)
        else:
            classes = (probs >= 0.5).astype(np.int32)

        return tb_model_name, label_run, probs, classes, notes
    except Exception as exc:
        notes.append(f"Triple Barrier inference failed: {exc}")
        return tb_model_name, None, None, None, notes


def process_trading_day(
    *,
    ctx: DayJobContext,
    day: str,
    day_index: int,
    worker_id: int,
    model: Any,
    hub: ProgressHub | None = None,
    on_status: Callable[[str], None] | None = None,
    inference_info: InferenceRuntimeInfo | None = None,
    on_day_progress: Callable[[dict[str, Any]], None] | None = None,
) -> DayBatchResult:
    """
    Load one trading day (batched), predict, build rows. Never writes to SQLite.
    Releases the day frame before return.
    """
    t0 = time.perf_counter()
    target = ctx.target
    features = ctx.features
    device_label = (
        (inference_info.device_label if inference_info else None)
        or getattr(ctx, "inference_device", None)
        or "CPU"
    )
    algo = getattr(ctx, "inference_algorithm", None) or ctx.algorithm or "xgboost"
    timings: dict[str, float] = {
        "load_master": 0.0,
        "load_timeline": 0.0,
        "prepare_matrix": 0.0,
        "predict": 0.0,
        "outcomes": 0.0,
        "sqlite_write": 0.0,
    }

    def status(state: str) -> None:
        if on_status:
            on_status(state)

    def day_progress(**fields: Any) -> None:
        if on_day_progress:
            on_day_progress(fields)

    def report(**fields: Any) -> None:
        if hub is None:
            return
        payload = {k: v for k, v in fields.items() if k != "state"}
        payload.setdefault("current_day", day)
        loaded = payload.get("rows_loaded")
        day_tot = payload.get("rows_day_total")
        day_pct = payload.get("day_progress_pct")
        if day_pct is None and loaded is not None and day_tot:
            try:
                day_pct = 100.0 * min(1.0, float(loaded) / float(day_tot))
            except (TypeError, ValueError, ZeroDivisionError):
                day_pct = None
        if day_pct is not None:
            payload["day_progress_pct"] = day_pct
        hub.update(
            workers=[{
                "worker_id": worker_id,
                "day": day,
                "state": fields.get("state") or "",
                "rows_loaded": fields.get("rows_loaded"),
                "rows_day_total": fields.get("rows_day_total"),
                "predictions": fields.get("predictions"),
                "day_progress_pct": day_pct,
            }],
            **payload,
        )

    def _empty_ok(**extra: Any) -> DayBatchResult:
        timings["total"] = time.perf_counter() - t0
        log_txt = format_day_stage_timings(
            timings, device_label=str(device_label), algorithm=str(algo)
        )
        return DayBatchResult(
            day_index=day_index,
            day=day,
            worker_id=worker_id,
            ok=True,
            rows=[],
            elapsed_sec=timings["total"],
            timings=dict(timings),
            inference_device=str(device_label),
            stage_timing_log=log_txt,
            **extra,
        )

    try:
        status("loading")
        report(
            stage=STAGE_LOAD_DAY,
            stage_label="Loading trading day",
            stage_detail="Counting day rows…",
            state="Loading",
            rows_loaded=0,
            rows_day_total=0,
        )

        def _count_prog(p: dict[str, Any]) -> None:
            report(stage=STAGE_LOAD_DAY, state="Counting", **p)

        t_load = time.perf_counter()
        day_total = count_trading_day_rows(
            ctx.parquet_path,
            day,
            on_progress=_count_prog,
        )
        report(
            stage=STAGE_LOAD_DAY,
            stage_detail=f"Loading {day_total:,} rows…",
            rows_day_total=day_total,
            rows_loaded=0,
            state="Loading",
        )

        def _load_prog(p: dict[str, Any]) -> None:
            report(stage=STAGE_LOAD_DAY, state="Loading", **p)

        day_df = load_trading_day_frame(
            ctx.parquet_path,
            day,
            columns=ctx.wanted_columns,
            on_progress=_load_prog,
            day_row_total=day_total or None,
        )
        loaded_from_master = False
        stamp_from_master = bool(getattr(ctx, "stamp_master_row_ids", False))
        if (day_df.empty or target not in day_df.columns) and ctx.master_db_path:
            from .prediction_feature_store import (
                count_trading_day_rows_in_master,
                load_trading_day_frame_from_master,
            )

            master_n = count_trading_day_rows_in_master(
                str(ctx.master_db_path),
                day,
                master_filter=getattr(ctx, "master_filter", None),
            )
            if master_n > 0:
                filt = getattr(ctx, "master_filter", None) or {}
                from .prediction_dataset_type import master_filter_summary_label

                filt_lbl = master_filter_summary_label(filt)
                report(
                    stage=STAGE_LOAD_DAY,
                    stage_detail=(
                        f"Day missing from parent parquet — "
                        f"loading {master_n:,} rows from Master ({filt_lbl})…"
                    ),
                    rows_day_total=master_n,
                    rows_loaded=0,
                    state="Loading",
                )
                from .prediction_transformations import (
                    expand_columns_for_master_load,
                    pipeline_has_enabled_transforms,
                )

                master_cols = expand_columns_for_master_load(
                    list(ctx.wanted_columns or []),
                    getattr(ctx, "transformation_config", None),
                )
                day_df = load_trading_day_frame_from_master(
                    str(ctx.master_db_path),
                    day,
                    master_cols,
                    master_filter=filt if filt else None,
                )
                day_total = int(len(day_df))
                loaded_from_master = not day_df.empty
                if loaded_from_master:
                    # Prefer Master-frame row ids; fill gaps via map lookup.
                    stamp_from_master = True
                    if pipeline_has_enabled_transforms(
                        getattr(ctx, "transformation_config", None)
                    ):
                        from .prediction_transformations import (
                            apply_parent_dataset_transformations,
                        )

                        report(
                            stage=STAGE_LOAD_DAY,
                            stage_detail=(
                                "Applying parent dataset Feature Transformations "
                                "to Master-loaded day…"
                            ),
                            state="Loading",
                        )
                        day_df = apply_parent_dataset_transformations(
                            day_df,
                            transformation_config=getattr(
                                ctx, "transformation_config", None
                            ),
                            sample_interval_sec=getattr(
                                ctx, "sample_interval_sec", None
                            ),
                            data_dir=ctx.data_dir,
                            dataset_name=getattr(ctx, "parent_dataset", None),
                            log_fn=lambda msg: report(
                                stage=STAGE_LOAD_DAY,
                                stage_detail=str(msg),
                                state="Loading",
                            ),
                        )
                        day_total = int(len(day_df))

        if day_df.empty or target not in day_df.columns:
            if ctx.master_db_path:
                status("failed")
                timings["load_master"] = time.perf_counter() - t_load
                return DayBatchResult(
                    day_index=day_index,
                    day=day,
                    worker_id=worker_id,
                    ok=False,
                    error=(
                        f"No rows for {day} in parent parquet or Master Dataset "
                        f"(target={target})"
                    ),
                    elapsed_sec=time.perf_counter() - t0,
                    timings=dict(timings),
                    inference_device=str(device_label),
                )
            status("finished")
            report(stage=STAGE_LOAD_DAY, stage_detail="Empty day", state="Finished")
            timings["load_master"] = time.perf_counter() - t_load
            return _empty_ok()

        if loaded_from_master:
            report(
                stage=STAGE_LOAD_DAY,
                stage_detail=f"Loaded {len(day_df):,} Master rows for {day}",
                rows_day_total=int(len(day_df)),
                rows_loaded=int(len(day_df)),
                state="Loading",
            )

        missing = [c for c in features if c not in day_df.columns]
        if missing:
            status("failed")
            timings["load_master"] = time.perf_counter() - t_load
            return DayBatchResult(
                day_index=day_index,
                day=day,
                worker_id=worker_id,
                ok=False,
                error=f"Features missing: {missing[:5]}",
                elapsed_sec=time.perf_counter() - t0,
                timings=dict(timings),
                inference_device=str(device_label),
            )

        day_df = day_df.dropna(subset=[target])
        if day_df.empty:
            if loaded_from_master or ctx.master_db_path:
                status("failed")
                timings["load_master"] = time.perf_counter() - t_load
                return DayBatchResult(
                    day_index=day_index,
                    day=day,
                    worker_id=worker_id,
                    ok=False,
                    error=f"No rows with target {target} for {day}",
                    elapsed_sec=time.perf_counter() - t0,
                    timings=dict(timings),
                    inference_device=str(device_label),
                )
            status("finished")
            timings["load_master"] = time.perf_counter() - t_load
            del day_df
            gc.collect()
            return _empty_ok()

        if ctx.row_limit is not None and int(ctx.row_limit) > 0:
            lim = int(ctx.row_limit)
            if len(day_df) > lim:
                day_df = day_df.iloc[:lim].copy()
                report(
                    stage=STAGE_LOAD_DAY,
                    stage_detail=f"Test sample · first {lim:,} of {day_total:,} rows",
                    rows_loaded=lim,
                    rows_day_total=lim,
                    state="Loading",
                )

        master_id_map: dict[tuple[float, str], int] = {}
        if stamp_from_master and ctx.master_db_path:
            from .prediction_feature_store import load_day_master_row_id_map

            master_id_map = load_day_master_row_id_map(ctx.master_db_path, day)

        timings["load_master"] = time.perf_counter() - t_load
        day_progress(
            phase="load",
            current_row=0,
            total_rows=int(len(day_df)),
            percent=8.0,
            message=f"{day}: loaded {len(day_df):,} rows",
        )

        market = "NIFTY"
        if "market" in day_df.columns:
            vals = [
                str(v).strip()
                for v in day_df["market"].dropna().astype(str).tolist()
                if str(v).strip()
            ]
            if vals:
                market = vals[0].upper()

        timelines: dict[str, Any] = {}
        if ctx.enrich_path_outcomes:
            status("enriching")
            report(
                stage=STAGE_ENRICH,
                stage_label="Enriching path outcomes",
                stage_detail="Loading tick timelines…",
                state="Enriching",
                rows_loaded=int(len(day_df)),
                rows_day_total=int(day_total or len(day_df)),
            )
            t_tl = time.perf_counter()
            timelines = _load_day_timelines(ctx.data_dir, day, market=market)
            # Convert LTP/ts arrays once per day — reused by every prediction.
            prepare_path_outcome_timelines(timelines)
            timings["load_timeline"] = time.perf_counter() - t_tl
            day_progress(
                phase="timeline",
                current_row=0,
                total_rows=int(len(day_df)),
                percent=18.0,
                message=f"{day}: tick timelines ready",
            )

        # Exclude samples that cannot be evaluated over the full regression horizon.
        day_df, n_horizon_drop = _exclude_incomplete_horizon_rows(
            day_df,
            horizon_sec=float(ctx.horizon_sec),
            timelines=timelines or None,
        )
        if n_horizon_drop > 0:
            report(
                stage=STAGE_LOAD_DAY,
                stage_detail=(
                    f"Excluded {n_horizon_drop:,} incomplete-horizon rows "
                    f"(last {float(ctx.horizon_sec):.0f}s of available data)"
                ),
                rows_loaded=int(len(day_df)),
                rows_day_total=int(day_total or len(day_df)),
                state="Loading",
            )
        if day_df.empty:
            status("failed")
            return DayBatchResult(
                day_index=day_index,
                day=day,
                worker_id=worker_id,
                ok=False,
                error=(
                    f"No rows with complete {float(ctx.horizon_sec):.0f}s "
                    f"prediction horizon for {day}"
                ),
                elapsed_sec=time.perf_counter() - t0,
                timings=dict(timings),
                inference_device=str(device_label),
            )

        status("predicting")
        report(
            stage=STAGE_PREDICT,
            stage_label="Predicting",
            stage_detail=f"Running model on {len(day_df):,} rows ({device_label})…",
            state="Predicting",
            rows_loaded=int(len(day_df)),
            rows_day_total=int(day_total or len(day_df)),
            predictions=0,
        )
        day_progress(
            phase="predict",
            current_row=0,
            total_rows=int(len(day_df)),
            percent=25.0,
            message=f"{day}: predicting ({device_label})",
        )
        t_prep = time.perf_counter()
        X = day_df.loc[:, features]
        timings["prepare_matrix"] = time.perf_counter() - t_prep

        t_pred = time.perf_counter()
        preds = batch_predict_day(model, X, info=inference_info)
        timings["predict"] = time.perf_counter() - t_pred

        # Prediction-package member pass (generic executor). Stable schema:
        # every ladder output column is always written, NULL when missing.
        from chain_replay_ml.training.prediction_packages import PROBABILITY_OUTPUT_COLUMNS

        member_outputs: dict[str, Any] = {col: None for col in PROBABILITY_OUTPUT_COLUMNS}
        if ctx.package_members:
            t_members = time.perf_counter()
            computed, member_notes = run_package_member_predictions(
                members=ctx.package_members,
                day_df=day_df,
            )
            member_outputs.update(computed)
            timings["predict_members"] = time.perf_counter() - t_members
            n_member_ok = sum(1 for v in member_outputs.values() if v is not None)
            report(
                stage=STAGE_PREDICT,
                stage_detail=(
                    f"Package members: {n_member_ok}/{len(ctx.package_members)} "
                    f"predicted ({len(day_df):,} rows)"
                ),
                state="Predicting",
            )
            for note in member_notes:
                report(
                    stage=STAGE_PREDICT,
                    stage_detail=f"Package member skipped — {note}",
                    state="Predicting",
                )
        member_output_items = list(member_outputs.items())

        tb_name, tb_run, tb_probs, tb_classes, tb_notes = None, None, None, None, []
        tb_model_name = ctx.tb_model_name
        if tb_model_name:
            t_tb = time.perf_counter()
            try:
                tb_name, tb_run, tb_probs, tb_classes, tb_notes = run_tb_model_predictions(
                    data_dir=ctx.data_dir,
                    tb_model_name=tb_model_name,
                    day_df=day_df,
                )
            except Exception as exc:
                # Belt-and-suspenders: run_tb_model_predictions already catches
                # its own errors, but the Triple Barrier side-scorer must never
                # be able to take the whole day down even if that contract
                # changes. Degrade to NULL tb_* columns and keep building.
                tb_name, tb_run, tb_probs, tb_classes = tb_model_name, None, None, None
                tb_notes = [f"Triple Barrier inference failed: {exc}"]
            timings["predict_tb"] = time.perf_counter() - t_tb
            if tb_notes:
                for note in tb_notes:
                    report(stage=STAGE_PREDICT, stage_detail=f"Triple Barrier — {note}", state="Predicting")

        day_progress(
            phase="outcomes",
            current_row=0,
            total_rows=int(len(day_df)),
            percent=40.0,
            message=f"{day}: CPU outcome metrics",
        )
        report(
            stage=STAGE_PREDICT,
            stage_detail=f"Building {len(day_df):,} research rows…",
            state="Predicting",
            predictions=0,
        )

        rows_out: list[dict[str, Any]] = []
        sum_abs = sum_err = sum_prem = 0.0
        n_abs = n_err = n_prem = 0
        dir_hits = dir_n = 0
        n_rows = len(day_df)
        profile_n = resolve_outcome_profile_rows(
            row_limit=ctx.row_limit,
            profile_outcome_rows=getattr(ctx, "profile_outcome_rows", None),
        )
        # Micro-profile first 1,000 path-outcome calls (or fewer if sample is smaller).
        micro_n = min(1000, profile_n) if profile_n > 0 else 0
        raw_micro = os.environ.get("PREDICTION_PATH_MICROPROFILE_ROWS")
        if raw_micro is not None and str(raw_micro).strip() != "":
            try:
                micro_n = max(0, int(raw_micro))
            except (TypeError, ValueError):
                pass
        outcome_profile_ms: list[float] = []
        path_micro_samples: dict[str, list[float]] = {
            k: [] for k in PATH_OUTCOME_PROFILE_KEYS
        }

        t_out = time.perf_counter()
        t_read = t_path = t_build = t_append = t_ckpt = 0.0
        t_gate = time.perf_counter()
        for i, (_, row) in enumerate(day_df.iterrows()):
            # Time spent waiting on / advancing pandas iterrows()
            t_now = time.perf_counter()
            t_read += t_now - t_gate

            t_mark = time.perf_counter()
            src = row.to_dict()
            ts = _f(src.get("timestamp"))
            token = str(src.get("token") or "")
            entry = _f(src.get("ltp"))
            actual = _f(src.get(target))
            predicted = _f(preds[i] if hasattr(preds, "__getitem__") else preds)
            if predicted is None:
                try:
                    predicted = float(preds[i])
                except Exception:
                    predicted = None
            t_read += time.perf_counter() - t_mark

            t_mark = time.perf_counter()
            err = compute_error_metrics(predicted=predicted, actual=actual, entry_ltp=entry)
            path = {
                "actual_max_profit": None,
                "actual_max_drawdown": None,
                "time_to_max_profit": None,
                "time_to_max_drawdown": None,
                "time_to_target": None,
                "target_reached_at": None,
                "max_profit_at": None,
                "max_drawdown_at": None,
                "dd_before_target": None,
                "time_to_dd_before_target": None,
                "exit_at": (float(ts) + ctx.horizon_sec) if ts is not None else None,
            }
            if ctx.enrich_path_outcomes and ts is not None and entry is not None and token:
                path_prof: dict[str, float] | None = None
                if micro_n > 0 and len(path_micro_samples["timeline_lookup"]) < micro_n:
                    path_prof = {}
                path = compute_path_outcomes(
                    timelines.get(token),
                    ts=float(ts),
                    entry_ltp=float(entry),
                    horizon_sec=ctx.horizon_sec,
                    predicted_ltp=predicted,
                    profile=path_prof,
                )
                if path_prof is not None:
                    for key in PATH_OUTCOME_PROFILE_KEYS:
                        path_micro_samples[key].append(
                            float(path_prof.get(key) or 0.0) * 1000.0
                        )
            elif ts is not None and path.get("exit_at") is None:
                path["exit_at"] = float(ts) + ctx.horizon_sec
            path_dt = time.perf_counter() - t_mark
            t_path += path_dt
            if profile_n > 0 and len(outcome_profile_ms) < profile_n:
                outcome_profile_ms.append(path_dt * 1000.0)

            t_mark = time.perf_counter()
            max_profit = _f(path.get("actual_max_profit"))
            if max_profit is None:
                max_profit = _f(path.get("actual_max_profit_5m"))
            pred_id = build_prediction_id(
                trading_day=day,
                timestamp=float(ts or 0),
                strike=src.get("strike"),
                option_type=src.get("option_type"),
                token=token,
                prediction_version=1,
            )

            if err.get("absolute_error") is not None:
                sum_abs += float(err["absolute_error"])
                n_abs += 1
            if err.get("prediction_error") is not None:
                sum_err += float(err["prediction_error"])
                n_err += 1
            if err.get("premium_error_pct") is not None:
                sum_prem += float(err["premium_error_pct"])
                n_prem += 1
            if err.get("direction_correct") is not None:
                dir_n += 1
                dir_hits += int(err["direction_correct"])

            ttt = path.get("time_to_target")
            target_at = path.get("target_reached_at")
            if ttt is None:
                target_reached_flag = None
            elif target_at is not None or (isinstance(ttt, (int, float)) and float(ttt) >= 0):
                target_reached_flag = 1
            else:
                target_reached_flag = 0  # miss (-1)

            expected_move = None
            actual_move = None
            if predicted is not None and entry is not None:
                expected_move = float(predicted) - float(entry)
            if actual is not None and entry is not None:
                actual_move = float(actual) - float(entry)

            master_row_id = src.get("master_row_id")
            if master_row_id is None and master_id_map and ts is not None:
                master_row_id = master_id_map.get((float(ts), token))
            try:
                master_row_id = int(master_row_id) if master_row_id is not None else None
            except (TypeError, ValueError):
                master_row_id = None

            out: dict[str, Any] = {
                "lab_uuid": ctx.lab_uuid,
                "prediction_id": pred_id,
                "trading_day": day,
                "timestamp": ts,  # Entry Timestamp
                "token": token,
                "strike": _f(src.get("strike")),
                "option_type": str(src.get("option_type") or "") or None,
                "expiry": str(src.get("expiry") or "") or None,
                "market": str(src.get("market") or market) or None,
                "current_spot": _f(src.get("spot")),
                "current_ltp": entry,
                "minutes_to_expiry": resolve_minutes_to_expiry(src, timestamp=float(ts or 0)),
                "target_column": target,
                "predicted_future_ltp": predicted,
                "actual_future_ltp": actual,
                "expected_move": expected_move,
                "actual_move": actual_move,
                "predicted_trend": move_trend(expected_move),
                "actual_trend": move_trend(actual_move),
                "absolute_error": err.get("absolute_error"),
                "prediction_error": err.get("prediction_error"),
                "premium_error_pct": err.get("premium_error_pct"),
                "direction_correct": err.get("direction_correct"),
                "maximum_profit": max_profit,
                "maximum_drawdown": (
                    _f(path.get("actual_max_drawdown"))
                    if path.get("actual_max_drawdown") is not None
                    else _f(path.get("actual_max_drawdown_5m"))
                ),
                "dd_before_target": _f(path.get("dd_before_target")),
                "time_to_max_profit": _f(path.get("time_to_max_profit")),
                "time_to_max_drawdown": _f(path.get("time_to_max_drawdown")),
                "time_to_dd_before_target": _f(path.get("time_to_dd_before_target")),
                "time_to_target": ttt,
                "target_reached": target_reached_flag,
                "target_reached_at": target_at,
                "max_profit_at": path.get("max_profit_at"),
                "max_drawdown_at": path.get("max_drawdown_at"),
                "exit_at": path.get("exit_at"),
                "master_row_id": master_row_id,
                "tb_model_name": tb_name,
                "tb_label_run": tb_run,
                "tb_pred_probability": round(_f(tb_probs[i]), 6) if (tb_probs is not None and i < len(tb_probs) and tb_probs[i] is not None) else None,
                "tb_pred_class": int(tb_classes[i]) if (tb_classes is not None and i < len(tb_classes) and tb_classes[i] is not None) else None,
            }
            for member_col, member_vals in member_output_items:
                member_val = None
                if member_vals is not None:
                    try:
                        member_val = _f(member_vals[i])
                    except (IndexError, TypeError, ValueError):
                        member_val = None
                out[member_col] = round(member_val, 6) if member_val is not None else None
            out.update(
                compute_rr_hit_labels(
                    target_reached=target_reached_flag,
                    maximum_profit=max_profit,
                    maximum_drawdown=out.get("maximum_drawdown"),
                )
            )
            if getattr(ctx, "embed_features", True):
                for feat, col in ctx.feat_map.items():
                    out[col] = _f(src.get(feat))
            t_build += time.perf_counter() - t_mark

            t_mark = time.perf_counter()
            rows_out.append(out)
            t_append += time.perf_counter() - t_mark

            # Progress + hub updates every 2k rows (SQLite heartbeat in worker path)
            if i == 0 or (i + 1) % 2000 == 0 or (i + 1) == n_rows:
                t_mark = time.perf_counter()
                frac = float(i + 1) / max(n_rows, 1)
                day_progress(
                    phase="outcomes",
                    current_row=i + 1,
                    total_rows=n_rows,
                    percent=40.0 + 55.0 * frac,
                    message=f"{day}: outcomes {i + 1:,}/{n_rows:,}",
                )
                report(
                    stage=STAGE_PREDICT,
                    state="Predicting",
                    stage_detail=f"Outcome metrics {i + 1:,}/{n_rows:,}…",
                    rows_loaded=i + 1,
                    rows_day_total=n_rows,
                    predictions=i + 1,
                    day_progress_pct=100.0 * (i + 1) / max(n_rows, 1),
                )
                if hub is not None and (i + 1) % 2000 == 0:
                    report(
                        stage=STAGE_PREDICT,
                        state="Predicting",
                        stage_detail=f"Built rows {i + 1:,} / {n_rows:,}",
                        predictions=i + 1,
                        rows_loaded=n_rows,
                        rows_day_total=int(day_total or n_rows),
                    )
                t_ckpt += time.perf_counter() - t_mark

            t_gate = time.perf_counter()

        timings["outcomes"] = time.perf_counter() - t_out
        timings["outcomes_read"] = t_read
        timings["outcomes_path"] = t_path
        timings["outcomes_build"] = t_build
        timings["outcomes_append"] = t_append
        timings["outcomes_checkpoint"] = t_ckpt

        per_pred_log = ""
        if outcome_profile_ms:
            per_pred_log = format_per_prediction_outcome_timings(outcome_profile_ms)
        micro_log = ""
        if any(path_micro_samples[k] for k in PATH_OUTCOME_PROFILE_KEYS):
            micro_n_done = max(len(path_micro_samples[k]) for k in PATH_OUTCOME_PROFILE_KEYS)
            micro_log = format_path_outcome_microprofile(
                path_micro_samples, n_predictions=micro_n_done
            )

        rows_out.sort(
            key=lambda r: (
                float(r.get("timestamp") or 0.0),
                str(r.get("token") or ""),
            )
        )

        status("finished")
        report(
            stage=STAGE_PREDICT,
            state="Finished",
            stage_detail=f"Day complete · {len(rows_out):,} rows",
            predictions=len(rows_out),
        )
        timings["total"] = time.perf_counter() - t0
        log_txt = format_day_stage_timings(
            timings, device_label=str(device_label), algorithm=str(algo)
        )
        result = DayBatchResult(
            day_index=day_index,
            day=day,
            worker_id=worker_id,
            ok=True,
            rows=rows_out,
            elapsed_sec=timings["total"],
            sum_abs=sum_abs,
            n_abs=n_abs,
            sum_err=sum_err,
            n_err=n_err,
            sum_prem=sum_prem,
            n_prem=n_prem,
            dir_hits=dir_hits,
            dir_n=dir_n,
            timings=dict(timings),
            inference_device=str(device_label),
            stage_timing_log=log_txt,
            outcome_profile_ms=list(outcome_profile_ms),
            per_prediction_timing_log=per_pred_log,
            path_microprofile_log=micro_log,
        )
        del day_df, X, timelines, preds
        gc.collect()
        return result
    except DatasetLoaderError as exc:
        status("failed")
        return DayBatchResult(
            day_index=day_index,
            day=day,
            worker_id=worker_id,
            ok=False,
            error=str(exc),
            elapsed_sec=time.perf_counter() - t0,
            timings=dict(timings),
            inference_device=str(device_label),
        )
    except Exception as exc:
        status("failed")
        return DayBatchResult(
            day_index=day_index,
            day=day,
            worker_id=worker_id,
            ok=False,
            error=f"Predict failed on {day}: {exc}",
            elapsed_sec=time.perf_counter() - t0,
            timings=dict(timings),
            inference_device=str(device_label),
        )


def _worker_main(
    worker_id: int,
    job_queue: queue.Queue,
    result_queue: queue.Queue,
    ctx: DayJobContext,
    worker_status: dict[int, dict[str, Any]],
    status_lock: threading.Lock,
    hub: ProgressHub | None = None,
    lab_db_path: str | None = None,
    control: Any | None = None,
    stop_new_work: threading.Event | None = None,
) -> None:
    # Private model copy per worker; verify inference device (train cuda ≠ infer cuda)
    model, infer_info = load_prediction_model_for_inference(ctx.model_path, ctx.algorithm)
    ctx.inference_device = infer_info.device_label
    ctx.inference_algorithm = infer_info.algorithm

    def set_status(day: str, state: str) -> None:
        with status_lock:
            worker_status[worker_id] = {"day": day, "state": state}

    set_status("", "idle")
    while True:
        item = job_queue.get()
        try:
            if item is _SENTINEL:
                set_status("", "idle")
                return
            day_index, day = item

            def on_status(state: str, _day: str = day) -> None:
                set_status(_day, state)

            # Pause/Cancel: finish in-flight day only; do not start the next.
            from .prediction_control import BuildControl

            ctrl = control if isinstance(control, BuildControl) else None
            if (stop_new_work and stop_new_work.is_set()) or (
                ctrl and ctrl.should_stop_after_day
            ):
                set_status(day, "skipped")
                result_queue.put(
                    DayBatchResult(
                        day_index=day_index,
                        day=day,
                        worker_id=worker_id,
                        ok=True,
                        skipped=True,
                    )
                )
                continue

            set_status(day, "loading")
            if lab_db_path:
                from .prediction_schema import DAY_RUNNING
                from .store import ModelLabStore

                with ModelLabStore(lab_db_path) as store:
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        day,
                        status=DAY_RUNNING,
                        started=True,
                        progress_pct=0.0,
                        error_message="",
                    )

            batch = process_trading_day(
                ctx=ctx,
                day=day,
                day_index=day_index,
                worker_id=worker_id,
                model=model,
                hub=hub,
                on_status=on_status,
                inference_info=infer_info,
            )
            result_queue.put(batch)
            set_status(day, "finished" if batch.ok else "failed")
        finally:
            job_queue.task_done()


def run_parallel_day_build(
    *,
    ctx: DayJobContext,
    lab_db_path: str,
    feature_columns: list[str],
    workers: int = DEFAULT_PREDICTION_WORKERS,
    total_estimate: int = 0,
    on_progress: ProgressFn | None = None,
    hub: ProgressHub | None = None,
    days_to_run: list[str] | None = None,
    control: Any | None = None,
) -> dict[str, Any]:
    """
    Process trading days with a fixed worker pool; one ordered SQLite writer.
    Skips days not in ``days_to_run`` (resume). Honors pause/cancel after the
    current day commits.
    """
    from .prediction_control import BuildControl
    from .prediction_schema import (
        DAY_CANCELLED,
        DAY_FAILED,
        DAY_WAITING,
        resolve_day_completion_status,
    )
    import logging

    log = logging.getLogger(__name__)

    all_days = list(ctx.days)
    run_set = set(days_to_run) if days_to_run is not None else set(all_days)
    ordered_run = [d for d in all_days if d in run_set]
    n_workers = max(1, min(int(workers), len(ordered_run) or 1))
    t0 = time.perf_counter()
    own_hub = hub is None
    if hub is None:
        hub = ProgressHub(on_progress, started_at=t0)
        hub.start()

    base_completed = max(0, len(all_days) - len(ordered_run))
    hub.update(
        stage=STAGE_LOAD_DAY,
        samples_total=total_estimate,
        trading_days_total=len(all_days),
        trading_days_done=base_completed,
        samples_done=0,
        predictions_written=0,
        worker_count=n_workers,
        stage_detail=f"Queue {len(ordered_run)} day(s)…",
        days_completed=base_completed,
        days_remaining=len(ordered_run),
    )

    empty_timing = {
        "workers": n_workers,
        "total_runtime_sec": 0.0,
        "trading_days": len(all_days),
        "avg_sec_per_day": None,
        "predictions_written": 0,
        "failed_days": 0,
    }
    if not ordered_run:
        if own_hub:
            hub.stop()
        return {
            "ok": True,
            "row_count": 0,
            "trading_days": len(all_days),
            "failed_days": [],
            "dataset_hash": "",
            "generation_time_sec": 0.0,
            "sum_abs": 0.0,
            "n_abs": 0,
            "sum_err": 0.0,
            "n_err": 0,
            "sum_prem": 0.0,
            "n_prem": 0,
            "dir_hits": 0,
            "dir_n": 0,
            "timing": empty_timing,
            "workers_used": n_workers,
            "stopped": "none",
            "days_processed": [],
        }

    job_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    status_lock = threading.Lock()
    stop_new_work = threading.Event()
    worker_status: dict[int, dict[str, Any]] = {
        i: {"day": "", "state": "idle"} for i in range(1, n_workers + 1)
    }

    day_index_map = {d: i for i, d in enumerate(ordered_run)}
    for day in ordered_run:
        job_queue.put((day_index_map[day], day))
    for _ in range(n_workers):
        job_queue.put(_SENTINEL)

    threads = [
        threading.Thread(
            target=_worker_main,
            name=f"pred-day-w{i}",
            args=(
                i,
                job_queue,
                result_queue,
                ctx,
                worker_status,
                status_lock,
                hub,
                lab_db_path,
                control,
                stop_new_work,
            ),
            daemon=True,
        )
        for i in range(1, n_workers + 1)
    ]
    for th in threads:
        th.start()

    digest = hashlib.sha256()
    digest.update(f"features={len(ctx.features)}|rows_order=day_ts_token|".encode("utf-8"))
    done_rows = 0
    days_done_in_run = 0
    failed_days: list[dict[str, str]] = []
    sum_abs = sum_err = sum_prem = 0.0
    n_abs = n_err = n_prem = 0
    dir_hits = dir_n = 0
    day_elapsed: list[float] = []
    days_processed: list[str] = []
    stopped = "none"

    pending: dict[int, DayBatchResult] = {}
    next_flush = 0
    received = 0
    ctrl = control if isinstance(control, BuildControl) else None

    def snapshot_workers() -> list[dict[str, Any]]:
        with status_lock:
            return [
                {
                    "worker_id": wid,
                    "day": info.get("day") or "",
                    "state": info.get("state") or "idle",
                }
                for wid, info in sorted(worker_status.items())
            ]

    def flush_batch(batch: DayBatchResult) -> None:
        nonlocal done_rows, days_done_in_run, sum_abs, n_abs, sum_err, n_err
        nonlocal sum_prem, n_prem, dir_hits, dir_n, stopped

        if batch.skipped:
            st = DAY_CANCELLED if (ctrl and ctrl.is_cancel) else DAY_WAITING
            with ModelLabStore(lab_db_path) as store:
                store.set_build_day_status(
                    ctx.lab_uuid,
                    batch.day,
                    status=st,
                    progress_pct=0.0,
                )
            return

        days_done_in_run += 1
        day_elapsed.append(float(batch.elapsed_sec or 0.0))
        days_processed.append(batch.day)
        overall_done = base_completed + days_done_in_run
        remaining = max(0, len(all_days) - overall_done)

        if not batch.ok:
            failed_days.append({"trading_day": batch.day, "error": str(batch.error or "failed")})
            with ModelLabStore(lab_db_path) as store:
                store.set_build_day_status(
                    ctx.lab_uuid,
                    batch.day,
                    status=DAY_FAILED,
                    error_message=str(batch.error or "failed"),
                    finished=True,
                    progress_pct=0.0,
                )
            hub.update(
                stage=STAGE_WRITE,
                current_day=batch.day,
                trading_days_done=overall_done,
                trading_days_total=len(all_days),
                samples_done=done_rows,
                samples_total=max(total_estimate, done_rows),
                predictions_written=done_rows,
                workers=snapshot_workers(),
                stage_detail=f"Failed {batch.day}: {batch.error}",
                failed_days=list(failed_days),
                days_completed=overall_done,
                days_remaining=remaining,
                day_progress_pct=0.0,
            )
            if ctrl and ctrl.should_stop_after_day:
                stop_new_work.set()
                stopped = "cancelled" if ctrl.is_cancel else "paused"
            return

        rows = batch.rows
        if rows:
            hub.update(
                stage=STAGE_WRITE,
                stage_label="Writing SQLite",
                stage_detail=f"Committing {len(rows):,} rows for {batch.day}…",
                current_day=batch.day,
                workers=snapshot_workers(),
                day_progress_pct=100.0,
            )
            tb_warn = tb_all_null_warning(tb_model_name=ctx.tb_model_name, rows=rows)
            if tb_warn:
                log.warning("[%s] %s", batch.day, tb_warn)
                report(
                    stage=STAGE_WRITE,
                    stage_detail=f"⚠ {tb_warn}",
                    state="Finished",
                )
            t_write = time.perf_counter()
            with ModelLabStore(lab_db_path) as store:
                store.delete_predictions_for_day(ctx.lab_uuid, batch.day)
                store.insert_prediction_rows(rows, feature_columns=feature_columns)
                if ctx.mark_day_complete:
                    # Preserve the true parent-dataset/Master expected count —
                    # never clobber it with the just-built row count, or
                    # Complete vs Partial can never be told apart later.
                    existing_expected = store.day_rows_expected(ctx.lab_uuid, batch.day)
                    expected = (
                        existing_expected
                        if existing_expected and existing_expected > 0
                        else len(rows)
                    )
                    day_status = resolve_day_completion_status(len(rows), expected)
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        batch.day,
                        status=day_status,
                        row_count=len(rows),
                        rows_expected=expected,
                        progress_pct=100.0,
                        error_message="",
                        finished=True,
                    )
                else:
                    # Test sample — leave day Waiting so full Start still runs
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        batch.day,
                        status=DAY_WAITING,
                        row_count=len(rows),
                        rows_expected=None,
                        progress_pct=None,
                        error_message=f"test sample {len(rows)} rows",
                        finished=False,
                    )
            write_sec = time.perf_counter() - t_write
            timings = dict(batch.timings or {})
            timings["sqlite_write"] = write_sec
            timings["total"] = (
                float(timings.get("load_master") or 0.0)
                + float(timings.get("load_timeline") or 0.0)
                + float(timings.get("prepare_matrix") or 0.0)
                + float(timings.get("predict") or 0.0)
                + float(timings.get("outcomes") or 0.0)
                + write_sec
            )
            timing_block = format_day_stage_timings(
                timings,
                device_label=str(
                    batch.inference_device
                    or getattr(ctx, "inference_device", None)
                    or "CPU"
                ),
                algorithm=str(
                    getattr(ctx, "inference_algorithm", None) or ctx.algorithm or "xgboost"
                ),
            )
            batch.timings = timings
            batch.stage_timing_log = timing_block
            for row in rows:
                pid = str(row.get("prediction_id") or "")
                digest.update(pid.encode("utf-8"))
                digest.update(b"\n")
            done_rows += len(rows)
        else:
            with ModelLabStore(lab_db_path) as store:
                if ctx.mark_day_complete:
                    existing_expected = store.day_rows_expected(ctx.lab_uuid, batch.day)
                    day_status = resolve_day_completion_status(0, existing_expected)
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        batch.day,
                        status=day_status,
                        row_count=0,
                        progress_pct=100.0,
                        finished=True,
                    )
                else:
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        batch.day,
                        status=DAY_WAITING,
                        row_count=0,
                    )

        sum_abs += batch.sum_abs
        n_abs += batch.n_abs
        sum_err += batch.sum_err
        n_err += batch.n_err
        sum_prem += batch.sum_prem
        n_prem += batch.n_prem
        dir_hits += batch.dir_hits
        dir_n += batch.dir_n

        hub.update(
            stage=STAGE_WRITE,
            current_day=batch.day,
            trading_days_done=overall_done,
            trading_days_total=len(all_days),
            samples_done=done_rows,
            samples_total=max(total_estimate, done_rows),
            predictions_written=done_rows,
            workers=snapshot_workers(),
            stage_detail=(
                f"Completed Days {overall_done}/{len(all_days)} · committed {batch.day}\n"
                f"{batch.stage_timing_log}"
                if batch.stage_timing_log
                else (
                    f"Completed Days {overall_done}/{len(all_days)} · "
                    f"committed {batch.day}"
                )
            ),
            failed_days=list(failed_days),
            days_completed=overall_done,
            days_remaining=remaining,
            day_progress_pct=100.0,
        )

        if ctrl and ctrl.should_stop_after_day:
            stop_new_work.set()
            stopped = "cancelled" if ctrl.is_cancel else "paused"

    try:
        while received < len(ordered_run):
            batch = result_queue.get()
            received += 1
            pending[batch.day_index] = batch
            while next_flush in pending:
                flush_batch(pending.pop(next_flush))
                next_flush += 1

        for th in threads:
            th.join(timeout=120.0)
        while next_flush in pending:
            flush_batch(pending.pop(next_flush))
            next_flush += 1
    finally:
        if own_hub:
            hub.stop()

    elapsed = time.perf_counter() - t0
    avg_day = (sum(day_elapsed) / len(day_elapsed)) if day_elapsed else None
    timing = {
        "workers": n_workers,
        "total_runtime_sec": round(elapsed, 3),
        "trading_days": len(all_days),
        "avg_sec_per_day": round(avg_day, 3) if avg_day is not None else None,
        "predictions_written": done_rows,
        "failed_days": len(failed_days),
    }

    return {
        "ok": True,
        "row_count": done_rows,
        "trading_days": len(all_days),
        "failed_days": failed_days,
        "dataset_hash": digest.hexdigest(),
        "generation_time_sec": round(elapsed, 2),
        "sum_abs": sum_abs,
        "n_abs": n_abs,
        "sum_err": sum_err,
        "n_err": n_err,
        "sum_prem": sum_prem,
        "n_prem": n_prem,
        "dir_hits": dir_hits,
        "dir_n": dir_n,
        "timing": timing,
        "workers_used": n_workers,
        "stopped": stopped,
        "days_processed": days_processed,
    }

