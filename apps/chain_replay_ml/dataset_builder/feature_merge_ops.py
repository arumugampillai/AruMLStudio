"""Merge missing registry feature columns into an existing dataset parquet."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name
from chain_replay_ml.features_atm_band import find_atm_strike

from .append_ops import load_dataset_metadata
from .chain_maps import precompute_chain_maps
from .day_context import DayContext, SourceSpec, load_day_context
from .extended_features import OptionFeatureState
from .feature_plugins import GROUP_FEATURE_SOURCES
from .feature_grid_policy import resolve_feature_grid_step_sec
from .lookback_policy import read_dataset_configuration, lookback_policy
from .registry_features import build_registry_features_at_ts
from .rolling_controllers import SpotControllers
from .schema_registry import load_schema_registry
from .writer import ensure_parquet_engine, read_dataset_parquet

from path_config import CHART_DATA_ROOT as _CHART_DIR
_merge_jobs: dict[str, dict[str, Any]] = {}
_merge_jobs_lock = threading.Lock()


def _implemented_registry_features() -> dict[str, str]:
    out: dict[str, str] = {}
    for _gid, mapping in GROUP_FEATURE_SOURCES.items():
        for feat, src in mapping.items():
            if src is not None:
                out[feat] = _gid
    return out


def _groups_for_features(features: list[str], registry: dict[str, Any]) -> list[str]:
    impl = _implemented_registry_features()
    groups_meta = registry.get("groups") or {}
    group_order = list(registry.get("groupOrder") or groups_meta.keys())
    feat_to_gid: dict[str, str] = {}
    for gid in group_order:
        for feat in (groups_meta.get(gid) or {}).get("features") or []:
            feat_to_gid[feat] = gid
    gids: list[str] = []
    for feat in features:
        gid = feat_to_gid.get(feat) or impl.get(feat)
        if gid and gid not in gids:
            gids.append(gid)
    return gids


def plan_feature_merge(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Compare dataset parquet columns vs registry — return merge candidates."""
    meta, paths = load_dataset_metadata(data_dir, dataset_name)
    registry = load_schema_registry()
    impl = _implemented_registry_features()

    if not os.path.isfile(paths["parquet"]):
        raise FileNotFoundError(f"Parquet not found for {dataset_name}")

    ensure_parquet_engine()
    df = read_dataset_parquet(paths["parquet"])
    parquet_cols = set(df.columns)

    stored_features = list(meta.get("feature_columns") or [])
    stored_set = set(stored_features)

    present = [f for f in stored_features if f in parquet_cols]
    missing_from_build = [f for f in stored_features if f not in parquet_cols]

    merge_candidates: list[dict[str, Any]] = []
    for feat in sorted(impl.keys()):
        if feat in parquet_cols:
            continue
        gid = impl[feat]
        groups_meta = registry.get("groups") or {}
        group_label = (groups_meta.get(gid) or {}).get("label") or gid
        merge_candidates.append({
            "name": feat,
            "group_id": gid,
            "group": group_label,
            "is_new_since_build": feat not in stored_set,
            "was_expected": feat in stored_set,
        })

    new_since_build = [c["name"] for c in merge_candidates if c["is_new_since_build"]]
    by_group: dict[str, list[str]] = {}
    for c in merge_candidates:
        by_group.setdefault(c["group_id"], []).append(c["name"])

    return {
        "dataset_name": meta.get("dataset_name") or dataset_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_count_present": len(present),
        "feature_count_stored": len(stored_features),
        "missing_from_build": missing_from_build,
        "merge_candidates": merge_candidates,
        "merge_candidate_count": len(merge_candidates),
        "new_since_build": new_since_build,
        "new_since_build_count": len(new_since_build),
        "by_group": {k: sorted(v) for k, v in by_group.items()},
        "can_merge": len(merge_candidates) > 0,
        "note": (
            "Merge recomputes selected features from replay tick DB for every row "
            "(same pipeline as build stage 6). Skips full audit/validation."
        ),
    }


def merge_features_into_dataset(
    data_dir: str,
    dataset_name: str,
    features: list[str],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Add feature columns to an existing parquet by replaying the feature pipeline."""
    if not features:
        raise ValueError("No features selected for merge")

    meta, paths = load_dataset_metadata(data_dir, dataset_name)
    registry = load_schema_registry()
    impl = _implemented_registry_features()

    requested = []
    for f in features:
        name = str(f).strip()
        if not name:
            continue
        if name not in impl:
            raise ValueError(f"Feature not implemented in registry: {name}")
        requested.append(name)
    if not requested:
        raise ValueError("No valid features to merge")

    ensure_parquet_engine()
    df = read_dataset_parquet(paths["parquet"])
    parquet_cols = set(df.columns)
    to_add = [f for f in requested if f not in parquet_cols]
    if not to_add:
        return {
            "dataset_name": dataset_name,
            "features_added": [],
            "rows": int(len(df)),
            "message": "All selected features already present in parquet",
        }

    required_cols = {"trading_day", "timestamp", "strike", "option_type", "token"}
    missing_cols = required_cols - parquet_cols
    if missing_cols:
        raise ValueError(f"Dataset missing required columns for merge: {sorted(missing_cols)}")

    lb_policy_doc = lookback_policy(read_dataset_configuration(meta))
    dataset_cfg = read_dataset_configuration(meta)
    grid_step = resolve_feature_grid_step_sec(dataset_configuration=dataset_cfg)
    existing_groups = list(meta.get("feature_groups") or meta.get("feature_groups_implemented") or [])
    merge_groups = _groups_for_features(to_add, registry)
    enabled_groups = list(dict.fromkeys([*existing_groups, *merge_groups]))

    ctx_cache: dict[str, DayContext] = {}
    chain_cache: dict[str, Any] = {}
    opt_states: dict[str, OptionFeatureState] = {}
    spot_controllers_by_day: dict[str, SpotControllers] = {}
    rows_total = len(df)
    rows_done = 0

    def _emit(**kwargs: Any) -> None:
        if on_progress:
            on_progress({"rows_done": rows_done, "rows_total": rows_total, **kwargs})

    _emit(phase="loading", message="Starting feature merge…")

    for (trading_day, token), _ in df.groupby(["trading_day", "token"], dropna=False).groups.items():
        day = str(trading_day)
        tok = str(token)
        if day not in ctx_cache:
            day_info = next(
                (d for d in (meta.get("days") or []) if str(d.get("trading_day")) == day),
                None,
            )
            if not day_info:
                continue
            try:
                ctx_cache[day] = load_day_context(
                    _CHART_DIR,
                    SourceSpec(
                        source_id=str(day_info.get("source_id") or day),
                        trading_day=day,
                        market=str(day_info.get("market") or "NIFTY"),
                        expiry=str(day_info.get("expiry") or ""),
                    ),
                    feature_grid_step_sec=grid_step,
                )
            except Exception:
                continue

        ctx = ctx_cache.get(day)
        if not ctx:
            continue

        index_key = normalize_index_name(ctx.source.market)
        strike_step = STRIKE_STEP.get(index_key, 50)

        token_df = df[(df["trading_day"] == trading_day) & (df["token"] == tok)].sort_values("timestamp")
        if day not in chain_cache:
            all_day_ts = sorted(df.loc[df["trading_day"] == trading_day, "timestamp"].unique())
            chain_cache[day] = precompute_chain_maps(
                index_tl=ctx.index_tl,
                strike_mapping=ctx.strike_mapping,
                timestamps=all_day_ts,
                strike_step=strike_step,
            )
        chain_maps = chain_cache[day]

        state_key = f"{day}:{tok}"
        if state_key not in opt_states:
            opt_states[state_key] = OptionFeatureState()

        if day not in spot_controllers_by_day:
            spot_controllers_by_day[day] = SpotControllers()
        spot_ctrl = spot_controllers_by_day[day]

        for idx, row in token_df.iterrows():
            strike_r = float(row["strike"])
            opt_type = str(row["option_type"])
            entry = ctx.strike_mapping.get((strike_r, opt_type))
            if not entry:
                rows_done += 1
                continue
            _tok, _sym, opt_tl = entry
            ts = float(row["timestamp"])
            spot = ctx.index_tl.ltp_rupees_at(ts)
            if spot is None or spot <= 0:
                rows_done += 1
                continue
            atm = find_atm_strike(spot, strike_step)
            picked = build_registry_features_at_ts(
                ts=ts,
                strike=strike_r,
                option_type=opt_type,
                opt_tl=opt_tl,
                index_tl=ctx.index_tl,
                strike_mapping=ctx.strike_mapping,
                chain_maps=chain_maps,
                opt_state=opt_states[state_key],
                strike_step=strike_step,
                expiry_ts=float(ctx.expiry_ts),
                open_ts=float(ctx.open_ts),
                close_ts=float(ctx.close_ts),
                enabled_groups=enabled_groups,
                trading_day=day,
                expiry_norm=str(ctx.expiry_norm),
                lookback_policy_doc=lb_policy_doc,
                atm_strike=atm,
                feature_grid_step_sec=grid_step,
                spot_controllers=spot_ctrl,
            )
            for feat in to_add:
                df.at[idx, feat] = picked.get(feat)
            rows_done += 1
            if rows_done % 2000 == 0:
                _emit(phase="merge", trading_day=day, message=f"Merging… {rows_done:,}/{rows_total:,}")

    merged_at = datetime.now(timezone.utc).isoformat()
    stored_features = list(meta.get("feature_columns") or [])
    updated_features = list(dict.fromkeys([*stored_features, *to_add]))
    merge_entry = {
        "merged_at": merged_at,
        "features_added": to_add,
        "row_count": int(len(df)),
        "skipped_audit": True,
        "skipped_validation": True,
    }
    history = list(meta.get("merge_history") or [])
    history.append(merge_entry)

    meta_updates = {
        "feature_columns": updated_features,
        "feature_count": len(updated_features),
        "enabled_features": list(dict.fromkeys([*(meta.get("enabled_features") or []), *to_add])),
        "feature_groups": list(dict.fromkeys([*existing_groups, *merge_groups])),
        "feature_groups_implemented": list(dict.fromkeys([
            *(meta.get("feature_groups_implemented") or []),
            *merge_groups,
        ])),
        "column_count": int(len(df.columns)),
        "row_count": int(len(df)),
        "last_merged_at": merged_at,
        "merge_history": history,
        "audit_stale": True,
        "validation_stale": True,
    }

    try:
        from chain_replay_ml.frame_backend import write_parquet_via_polars

        write_parquet_via_polars(df, paths["parquet"])
    except Exception:
        df.to_parquet(paths["parquet"], index=False)
    with open(paths["json"], encoding="utf-8") as fh:
        full_meta = json.load(fh)
    full_meta.update(meta_updates)
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    _emit(phase="done", message="Merge complete")

    return {
        "dataset_name": dataset_name,
        "features_added": to_add,
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_count": len(updated_features),
        "merged_at": merged_at,
        "skipped_audit": True,
        "skipped_validation": True,
    }


def start_feature_merge_job(
    data_dir: str,
    dataset_name: str,
    features: list[str],
) -> dict[str, Any]:
    """Run merge in a background thread; poll with get_feature_merge_job."""
    job_id = uuid.uuid4().hex
    with _merge_jobs_lock:
        _merge_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "dataset_name": dataset_name,
            "features": list(features),
            "rows_done": 0,
            "rows_total": 0,
            "phase": "starting",
            "message": "Starting merge…",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _run() -> None:
        def on_progress(payload: dict[str, Any]) -> None:
            with _merge_jobs_lock:
                if job_id in _merge_jobs:
                    _merge_jobs[job_id].update(payload)

        try:
            result = merge_features_into_dataset(
                data_dir,
                dataset_name,
                features,
                on_progress=on_progress,
            )
            with _merge_jobs_lock:
                _merge_jobs[job_id].update({
                    "status": "done",
                    "phase": "done",
                    "message": "Merge complete",
                    "result": result,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            with _merge_jobs_lock:
                _merge_jobs[job_id].update({
                    "status": "error",
                    "phase": "error",
                    "message": str(exc),
                    "error": str(exc),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def get_feature_merge_job(job_id: str) -> dict[str, Any] | None:
    with _merge_jobs_lock:
        job = _merge_jobs.get(job_id)
        return dict(job) if job else None
