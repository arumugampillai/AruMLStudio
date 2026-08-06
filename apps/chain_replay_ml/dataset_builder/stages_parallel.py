"""Per-token parallel Stage-6 feature enrichment (master build + replay)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable

from chain_replay_ml.ticks import TickTimeline

from .chain_maps import ChainMaps
from .day_context import DayContext
from .extended_features import OptionFeatureState
from .feature_enrichment import build_feature_raw_for_row, attach_scoring_infra_columns
from .feature_plugins import pick_features_from_row
from .feature_grid_policy import resolve_feature_grid_step_sec
from .rolling_controllers import build_spot_rv_cache

_PAR_CTX: DayContext | None = None
_PAR_CHAIN_MAPS: ChainMaps | None = None
_PAR_STRIKE_STEP: int = 50
_PAR_GROUP_IDS: list[str] = []
_PAR_PER_GROUP: dict[str, list[str]] = {}
_PAR_LOOKBACK: dict[str, Any] | None = None
_PAR_ACTIVE_FEATURES: frozenset[str] | None = None
_PAR_GAP_MAX_SEC: float | None = None
_PAR_SPOT_RV_CACHE: dict[float, dict[str, float | None]] | None = None


def _parallel_worker_init(
    ctx: DayContext,
    chain_maps: ChainMaps,
    strike_step: int,
    group_ids: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None,
    active_features: frozenset[str] | None = None,
    gap_max_sec: float | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> None:
    global _PAR_CTX, _PAR_CHAIN_MAPS, _PAR_STRIKE_STEP, _PAR_GROUP_IDS, _PAR_PER_GROUP, _PAR_LOOKBACK, _PAR_ACTIVE_FEATURES, _PAR_GAP_MAX_SEC, _PAR_SPOT_RV_CACHE
    _PAR_CTX = ctx
    _PAR_CHAIN_MAPS = chain_maps
    _PAR_STRIKE_STEP = strike_step
    _PAR_GROUP_IDS = group_ids
    _PAR_PER_GROUP = per_group_features
    _PAR_LOOKBACK = lookback_policy_doc
    _PAR_ACTIVE_FEATURES = active_features
    _PAR_GAP_MAX_SEC = gap_max_sec
    _PAR_SPOT_RV_CACHE = spot_rv_cache
    # Process workers start cold; thread workers share warm state with parent.
    try:
        from chain_replay_ml.performance.runtime import warm_kernels

        warm_kernels(verbose=False)
    except Exception:
        pass


def _parallel_process_token(
    batch: tuple[str, TickTimeline, list[tuple[int, dict[str, Any]]]],
) -> list[tuple[int, dict[str, Any]]]:
    _token, opt_tl, indexed_rows = batch
    if _PAR_CTX is None or _PAR_CHAIN_MAPS is None:
        raise RuntimeError("parallel worker not initialized")
    opt_state = OptionFeatureState()
    out: list[tuple[int, dict[str, Any]]] = []
    for row_index, snap in indexed_rows:
        row = {
            **snap,
            "token": str(snap.get("token") or _token),
            "_opt_tl": opt_tl,
        }
        raw = build_feature_raw_for_row(
            row,
            ctx=_PAR_CTX,
            chain_maps=_PAR_CHAIN_MAPS,
            strike_step=_PAR_STRIKE_STEP,
            lookback_policy_doc=_PAR_LOOKBACK,
            opt_state=opt_state,
            active_features=_PAR_ACTIVE_FEATURES,
            gap_max_sec=_PAR_GAP_MAX_SEC,
            spot_rv_cache=_PAR_SPOT_RV_CACHE,
        )
        picked: dict[str, Any] = {}
        for gid in _PAR_GROUP_IDS:
            picked.update(pick_features_from_row(raw, _PAR_PER_GROUP[gid], [gid]))
        attach_scoring_infra_columns(picked, raw)
        out.append((row_index, picked))
    return out


def _group_rows_by_token(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, row in enumerate(rows):
        token = str(row["token"])
        snap = {
            "token": token,
            "timestamp": row["timestamp"],
            "strike": row["strike"],
            "option_type": row["option_type"],
            "symbol": row.get("symbol"),
            "_atm": row["_atm"],
        }
        grouped.setdefault(token, []).append((row_index, snap))
    for token_rows in grouped.values():
        token_rows.sort(key=lambda item: item[1]["timestamp"])
    return grouped


def _max_parallel_workers(token_count: int) -> int:
    raw = os.getenv("REPLAY_PARALLEL_WORKERS", "").strip()
    if raw.isdigit():
        cap = max(1, int(raw))
    else:
        cpu = os.cpu_count() or 4
        cap = max(1, cpu - 1)
    return max(1, min(cap, token_count))


def _parallel_executor(token_count: int):
    """Windows default: threads (avoid pickling multi-GB DayContext).

    Set MASTER_BUILD_PARALLEL_BACKEND=process (or REPLAY_PARALLEL_BACKEND=process)
    to force process pool when measuring CPU-bound speedups.
    """
    backend = (
        os.getenv("MASTER_BUILD_PARALLEL_BACKEND")
        or os.getenv("REPLAY_PARALLEL_BACKEND")
        or "auto"
    ).strip().lower()
    workers = _max_parallel_workers(token_count)
    if backend == "thread" or (backend == "auto" and sys.platform == "win32"):
        return ThreadPoolExecutor(max_workers=workers), True
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=_parallel_worker_init,
        initargs=(),
    ), False


def enrich_rows_features_parallel_token(
    rows: list[dict[str, Any]],
    *,
    ctx: DayContext,
    chain_maps: ChainMaps,
    strike_step: int,
    enabled_groups: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None,
    active_features: frozenset[str] | None = None,
    gap_max_sec: float | None = None,
    cancel_check: Any | None = None,
    on_progress: Any | None = None,
) -> None:
    """Mutates rows in place with feature columns (Stage 6).

    Partitions work by token so each worker owns its OptionFeatureState and
    timeline locality. Progress is overall rows completed (not per feature group).
    """
    group_ids = [g for g in enabled_groups if g in per_group_features]
    if not group_ids:
        return

    grouped = _group_rows_by_token(rows)
    grid_step = resolve_feature_grid_step_sec(ctx=ctx)
    spot_rv_cache = build_spot_rv_cache(
        ctx.index_tl,
        [r["timestamp"] for r in rows],
        grid_step_sec=grid_step,
    )
    if len(grouped) < 2:
        _enrich_rows_features_serial_token(
            rows,
            ctx=ctx,
            chain_maps=chain_maps,
            strike_step=strike_step,
            group_ids=group_ids,
            per_group_features=per_group_features,
            lookback_policy_doc=lookback_policy_doc,
            active_features=active_features,
            gap_max_sec=gap_max_sec,
            spot_rv_cache=spot_rv_cache,
        )
        return

    batches: list[tuple[str, TickTimeline, list[tuple[int, dict[str, Any]]]]] = []
    for token, indexed_rows in grouped.items():
        first_row = rows[indexed_rows[0][0]]
        batches.append((token, first_row["_opt_tl"], indexed_rows))

    workers = _max_parallel_workers(len(batches))
    init_args = (
        ctx,
        chain_maps,
        strike_step,
        group_ids,
        per_group_features,
        lookback_policy_doc,
        active_features,
        gap_max_sec,
        spot_rv_cache,
    )
    total_rows = len(rows)
    done_rows = 0
    progress_fn = on_progress

    executor, use_threads = _parallel_executor(len(batches))
    if use_threads:
        _parallel_worker_init(*init_args)
        with executor as pool:
            futures = [pool.submit(_parallel_process_token, batch) for batch in batches]
            for fut in as_completed(futures):
                if cancel_check and cancel_check():
                    for f in futures:
                        f.cancel()
                    break
                for row_index, picked in fut.result():
                    rows[row_index].update(picked)
                    done_rows += 1
                if progress_fn is not None:
                    try:
                        progress_fn(min(done_rows, total_rows), total_rows)
                    except Exception:
                        pass
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_parallel_worker_init,
            initargs=init_args,
        ) as pool:
            for token_updates in pool.map(_parallel_process_token, batches, chunksize=1):
                if cancel_check and cancel_check():
                    break
                for row_index, picked in token_updates:
                    rows[row_index].update(picked)
                    done_rows += 1
                if progress_fn is not None:
                    try:
                        progress_fn(min(done_rows, total_rows), total_rows)
                    except Exception:
                        pass


def _enrich_rows_features_serial_token(
    rows: list[dict[str, Any]],
    *,
    ctx: DayContext,
    chain_maps: ChainMaps,
    strike_step: int,
    group_ids: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None,
    active_features: frozenset[str] | None = None,
    gap_max_sec: float | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> None:
    """Serial per-token path used when only one token is present."""
    grouped = _group_rows_by_token(rows)
    if spot_rv_cache is None:
        grid_step = resolve_feature_grid_step_sec(ctx=ctx)
        spot_rv_cache = build_spot_rv_cache(
            ctx.index_tl,
            [r["timestamp"] for r in rows],
            grid_step_sec=grid_step,
        )
    for token, indexed_rows in grouped.items():
        opt_state = OptionFeatureState()
        opt_tl = rows[indexed_rows[0][0]]["_opt_tl"]
        for row_index, snap in indexed_rows:
            row = {**snap, "_opt_tl": opt_tl, "token": token}
            raw = build_feature_raw_for_row(
                row,
                ctx=ctx,
                chain_maps=chain_maps,
                strike_step=strike_step,
                lookback_policy_doc=lookback_policy_doc,
                opt_state=opt_state,
                active_features=active_features,
                gap_max_sec=gap_max_sec,
                spot_rv_cache=spot_rv_cache,
            )
            picked: dict[str, Any] = {}
            for gid in group_ids:
                picked.update(pick_features_from_row(raw, per_group_features[gid], [gid]))
            attach_scoring_infra_columns(picked, raw)
            rows[row_index].update(picked)
