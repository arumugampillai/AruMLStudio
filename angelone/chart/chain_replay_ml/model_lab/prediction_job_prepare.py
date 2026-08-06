"""Prepare durable prediction job config (no prediction generation)."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from chain_replay_ml.training.dataset_loader import DatasetLoaderError, parquet_column_names
from chain_replay_ml.training.model_runtime import (
    load_prediction_model_cached,
    resolve_prediction_model_package,
)

from .prediction_io import (
    catalog_trading_days,
    count_trading_day_rows,
    estimate_sample_total,
    load_dataset_meta,
    resolve_dataset_parquet,
)
from .prediction_schema import (
    FEATURE_STORAGE_EMBEDDED,
    FEATURE_STORAGE_REFERENCED,
    DAY_COMPLETED,
    DAY_SKIPPED,
    PRED_STATUS_BUILDING,
    PRED_STATUS_ERROR,
    align_features_to_model,
    feature_column_map,
    horizon_sec_from_target,
)
from .store import ModelLabStore


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


_IDENTITY_LOAD = (
    "trading_day",
    "timestamp",
    "token",
    "strike",
    "option_type",
    "symbol",
    "market",
    "expiry",
    "spot",
    "ltp",
    "minutes_to_expiry",
)


def _filter_days_with_target_coverage(
    days: list[str],
    *,
    parquet_path: str,
    pq_cols: set[str],
    target: str,
    master_abs: str | None,
    master_filter: dict[str, Any] | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split ``days`` into (buildable, [(day, reason), ...] unbuildable).

    Mirrors the worker's runtime check (``prediction_parallel.process_trading_day``)
    but runs *before* a job/worker is spawned, so a day with zero usable rows
    for ``target`` never silently burns worker time only to fail after the
    fact with an opaque "Worker process exited before day completed".

    A day is buildable if either:
      - the parent parquet carries the ``target`` column and has >0 rows for
        that day, or
      - Master Dataset has the ``target`` column *and* >0 rows for that day
        under ``master_filter``.

    Master never backfills long-horizon labels (e.g. ``future_ltp_5m``) that
    only exist because the analysis-dataset build computed them — it only
    stores near-term derived columns such as ``future_ltp_1m``/``future_ltp_10s``.
    That mismatch is called out explicitly in the skip reason so the user
    knows to rebuild the analysis dataset (to include the day) rather than
    rely on the Master fallback.
    """
    from .prediction_feature_store import (
        count_trading_day_rows_in_master,
        master_sample_columns,
    )

    parquet_has_target = target in pq_cols
    master_cols = master_sample_columns(master_abs) if master_abs else set()
    master_has_target = bool(master_abs) and target in master_cols

    buildable: list[str] = []
    unbuildable: list[tuple[str, str]] = []
    for day in days:
        day_s = str(day or "").strip()
        if not day_s:
            continue

        pq_rows = 0
        if parquet_has_target:
            try:
                pq_rows = count_trading_day_rows(parquet_path, day_s)
            except Exception:
                pq_rows = 0
        if pq_rows > 0:
            buildable.append(day_s)
            continue

        if master_has_target:
            try:
                master_rows = count_trading_day_rows_in_master(
                    master_abs, day_s, master_filter=master_filter
                )
            except Exception:
                master_rows = 0
            if master_rows > 0:
                buildable.append(day_s)
                continue

        if master_abs and not master_has_target:
            reason = (
                f"No rows for {day_s} in parent parquet, and Master Dataset does "
                f"not have target column '{target}' (Master only stores near-term "
                "forward labels like future_ltp_1m/future_ltp_10s — longer "
                "horizons are computed when the analysis dataset is built). "
                "Rebuild the analysis dataset including this day, or exclude "
                "the day from this build."
            )
        else:
            reason = (
                f"No rows for {day_s} in parent parquet"
                + (" or Master Dataset" if master_abs else "")
                + f" (target={target}). Rebuild the analysis dataset including "
                "this day, or exclude the day from this build."
            )
        unbuildable.append((day_s, reason))
    return buildable, unbuildable


def prepare_prediction_exec_config(
    data_dir: str,
    lab_db_path: str,
    *,
    overwrite: bool = False,
    resume: bool = True,
    selected_days: list[str] | None = None,
    enrich_path_outcomes: bool = True,
    row_limit: int | None = None,
    mark_day_complete: bool = True,
    tb_model_name: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Validate inputs, catalog days, resolve feature storage, prepare lab schema.

    Does **not** run predictions. Returns a serializable config for workers.
    ``on_stage`` receives short progress lines for the UI log.
    """

    def stage(msg: str) -> None:
        if on_stage is not None:
            try:
                on_stage(msg)
            except Exception:
                pass

    from .prediction_builder import validate_prediction_inputs
    from .prediction_feature_store import (
        detect_feature_storage_mode,
        ensure_master_row_id_light,
        master_dataset_id_from_path,
        master_has_row_id_column,
        referenced_feature_column_map,
    )

    stage("Prepare: opening lab + validating inputs…")
    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        if lab is None:
            return {"ok": False, "error": "Model Lab info not found"}
        store.ensure_prediction_schema()
        # Prefer summary row_count — avoid full-table COUNT(*) on large prediction DBs.
        summary_doc = store.read_prediction_summary() or {}
        try:
            existing = int(summary_doc.get("row_count") or 0)
        except (TypeError, ValueError):
            existing = 0
        if existing <= 0 and not overwrite and not resume:
            existing = store.prediction_row_count()
        if existing > 0 and not overwrite and not resume and row_limit is None:
            return {
                "ok": False,
                "error": "Prediction Dataset already exists.",
                "code": "exists",
                "row_count": existing,
            }

        check = validate_prediction_inputs(data_dir, lab)
        if not check["ok"]:
            store.write_prediction_summary(
                lab_uuid=lab.lab_uuid,
                status=PRED_STATUS_ERROR,
                row_count=0,
                trading_days=0,
                error_message="; ".join(check["errors"]),
                parent_model_name=lab.parent_model_name,
                parent_dataset=check.get("dataset") or None,
                target_column=lab.target,
                selected_feature_count=len(check.get("features") or []),
            )
            return {
                "ok": False,
                "error": "; ".join(check["errors"]),
                "warnings": check.get("warnings"),
            }

        features = list(check["features"])
        target = str(check["target"])
        dataset = str(check["dataset"])
        model_name = str(check["model_name"])
        model_path = str(check["model_path"])
        lab_uuid = lab.lab_uuid
        algorithm = lab.algorithm
        dataset_snapshot = lab.dataset_snapshot or {}

        store.write_prediction_summary(
            lab_uuid=lab_uuid,
            status=PRED_STATUS_BUILDING,
            row_count=existing if (resume and not overwrite) else 0,
            trading_days=0,
            parent_model_name=model_name,
            parent_dataset=dataset,
            target_column=target,
            selected_feature_count=len(features),
            created_at=_utc_now(),
        )

    try:
        stage(f"Prepare: resolving parent dataset “{dataset}”…")
        parquet_path, meta_path = resolve_dataset_parquet(data_dir, dataset)
        meta = load_dataset_meta(meta_path)
        total_estimate = estimate_sample_total(parquet_path, meta=meta) or 0
        days = catalog_trading_days(parquet_path, meta=meta)
    except DatasetLoaderError as exc:
        return {"ok": False, "error": str(exc)}

    if not days:
        return {"ok": False, "error": "No trading days found in parent dataset"}

    from .prediction_dataset_type import (
        master_filter_summary_label,
        resolve_model_master_filter,
    )

    master_filter = resolve_model_master_filter(lab, parent_meta=meta)

    stage("Prepare: loading model…")
    probe, _ms, _disk = load_prediction_model_cached(model_path, algorithm)
    features = align_features_to_model(features, probe)
    del probe

    pq_cols = parquet_column_names(parquet_path) or set()
    master_rel = (
        str(meta.get("master_db_path") or "").strip()
        or str(dataset_snapshot.get("master_db_path") or "").strip()
        or None
    )
    storage_mode, master_abs = detect_feature_storage_mode(
        parquet_columns=pq_cols,
        master_db_path=master_rel,
        data_dir=data_dir,
    )
    if storage_mode == FEATURE_STORAGE_REFERENCED and master_abs:
        stage("Prepare: checking master_row_id (fast probe)…")
        ensured = ensure_master_row_id_light(master_abs)
        if not ensured.get("ok") and not master_has_row_id_column(master_abs):
            storage_mode = FEATURE_STORAGE_EMBEDDED

    embed_features = storage_mode == FEATURE_STORAGE_EMBEDDED
    if embed_features:
        feat_map = feature_column_map(features)
    else:
        feat_map = referenced_feature_column_map(features)

    master_rel_store = None
    if master_abs:
        master_rel_store = os.path.abspath(master_abs).replace("\\", "/")
    elif master_rel:
        master_rel_store = master_rel.replace("\\", "/")
    master_id = master_dataset_id_from_path(master_abs or master_rel)
    stamp_ids = (
        storage_mode == FEATURE_STORAGE_REFERENCED
        and "master_row_id" not in pq_cols
        and bool(master_abs)
    )

    # Prediction-package members: discover the probability ladder for anchor
    # targets and extend the loaded columns with the member feature union so
    # one feature pass serves every member.
    from chain_replay_ml.training.prediction_packages import (
        discover_prediction_package_members,
        is_package_anchor_target,
        package_members_summary,
    )

    package_members: list[dict[str, Any]] = []
    if is_package_anchor_target(target):
        stage("Prepare: discovering prediction-package members…")
        package_members = discover_prediction_package_members(
            data_dir,
            dataset=dataset,
            anchor_target=target,
            anchor_model_name=model_name,
        )
        stage(f"Prepare: prediction package · {package_members_summary(package_members)}")

    wanted = list(dict.fromkeys([*features, target, *_IDENTITY_LOAD]))
    tb_label_run: str | None = None
    if tb_model_name:
        try:
            tb_pkg = resolve_prediction_model_package(data_dir, tb_model_name)
            if tb_pkg.get("ok"):
                tb_label_run = str(tb_pkg.get("label_run_id") or "triple_barrier").strip()
                # Merge unconditionally (like the primary model's own
                # ``features``, not gated on raw pq_cols membership) — TB
                # features may only materialize after Master transforms
                # (e.g. via expand_columns_for_master_load below). Gating on
                # pq_cols here silently dropped those columns from
                # wanted_columns, so the day frame never carried them and TB
                # scoring degraded to NULL even when it could have run.
                for tf in tb_pkg.get("features") or []:
                    tf_s = str(tf).strip()
                    if tf_s and tf_s not in wanted:
                        wanted.append(tf_s)
                stage(f"Prepare: Triple Barrier model '{tb_model_name}' linked (label_run={tb_label_run})")
            else:
                stage(f"Prepare: Triple Barrier model warning: {tb_pkg.get('error')}")
        except Exception as exc:
            stage(f"Prepare: Triple Barrier model warning: {exc}")

    member_feature_union = sorted({
        str(f)
        for member in package_members
        if member.get("available")
        for f in (member.get("features") or [])
    })
    wanted.extend(c for c in member_feature_union if c in pq_cols and c not in wanted)
    if "master_row_id" in pq_cols:
        wanted.append("master_row_id")

    from .prediction_transformations import (
        expand_columns_for_master_load,
        sample_interval_sec_from_meta,
        transformation_config_from_dataset_meta,
    )

    transformation_config = transformation_config_from_dataset_meta(meta)
    sample_interval_sec = sample_interval_sec_from_meta(meta)
    # Master Unseen days need transform source columns (e.g. atm_pcr for Lag),
    # not only the final model feature names.
    wanted = expand_columns_for_master_load(wanted, transformation_config)

    sel_set = set(selected_days) if selected_days is not None else None
    catalog_days = list(days)
    if selected_days:
        for d in selected_days:
            ds = str(d or "").strip()
            if ds and ds not in catalog_days:
                catalog_days.append(ds)
    stage("Prepare: updating day catalog (no prediction scan)…")
    with ModelLabStore(lab_db_path) as store:
        if overwrite:
            stage("Prepare: clearing prediction dataset (overwrite)…")
            store.clear_prediction_dataset()
            existing = 0
        store.ensure_prediction_schema()
        if embed_features:
            stage(f"Prepare: ensuring {len(feat_map)} feature columns…")
            store.ensure_feature_columns(list(feat_map.values()))
        store.ensure_build_days(
            lab_uuid,
            catalog_days,
            selected=sel_set,
            sync_pred_counts=False,
        )
        if selected_days is not None:
            store.set_days_selected(lab_uuid, list(selected_days))
            # EXISTS probe only (no COUNT(*)) — marks resume-complete days.
            for day in selected_days:
                day_s = str(day or "").strip()
                if not day_s:
                    continue
                try:
                    hit = store.conn.execute(
                        """
                        SELECT 1 FROM prediction_dataset
                        WHERE lab_uuid = ? AND trading_day = ?
                        LIMIT 1
                        """,
                        (lab_uuid, day_s),
                    ).fetchone()
                except Exception:
                    hit = None
                if hit:
                    store.set_build_day_status(
                        lab_uuid,
                        day_s,
                        status=DAY_COMPLETED,
                        progress_pct=100.0,
                        finished=True,
                    )

        skipped_days: list[tuple[str, str]] = []
        if row_limit is not None and int(row_limit) > 0:
            mark_day_complete = False
            if selected_days:
                days_to_run = [str(d) for d in selected_days if str(d or "").strip()]
            else:
                days_to_run = [
                    d["trading_day"]
                    for d in store.list_build_days(lab_uuid)
                    if d.get("selected")
                ]
        elif resume and not overwrite:
            if selected_days is not None:
                # Caller told us exactly which days to build — evaluate pending-ness
                # for those days directly rather than gating on the shared
                # ``selected`` column. That column is rewritten wholesale by every
                # single-day build (ensure_build_days/set_days_selected deselect
                # every *other* day as a side effect), so a build for day A that
                # races with — or simply follows — a build for day B can leave A's
                # ``selected`` flag at 0 even though A was just explicitly
                # requested here. Filtering on the caller-provided list instead
                # makes day selection immune to that shared, mutable state.
                pending_all = store.pending_build_days(lab_uuid, selected_only=False)
                selected_only_days = {
                    str(d).strip() for d in selected_days if str(d or "").strip()
                }
                days_to_run = [d for d in pending_all if d in selected_only_days]
                if tb_model_name:
                    # A day already Complete from a build that ran without
                    # Triple Barrier (or with a different TB model) is not
                    # "pending" by row-count alone — resume must not skip it,
                    # or its tb_* columns stay NULL forever. Only rescan days
                    # the caller actually asked for here.
                    tb_stale = store.days_needing_tb_rescore(
                        lab_uuid, sorted(selected_only_days), tb_model_name
                    )
                    days_to_run.extend(d for d in tb_stale if d not in days_to_run)
            else:
                days_to_run = store.pending_build_days(lab_uuid, selected_only=True)
                if tb_model_name:
                    candidate_days = [
                        str(d["trading_day"])
                        for d in store.list_build_days(lab_uuid)
                        if d.get("selected")
                    ]
                    tb_stale = store.days_needing_tb_rescore(
                        lab_uuid, candidate_days, tb_model_name
                    )
                    days_to_run.extend(d for d in tb_stale if d not in days_to_run)
        else:
            if selected_days:
                days_to_run = [str(d) for d in selected_days if str(d or "").strip()]
            else:
                days_to_run = [
                    d["trading_day"]
                    for d in store.list_build_days(lab_uuid)
                    if d.get("selected")
                ]

        if days_to_run:
            stage("Prepare: checking day eligibility (target column coverage)…")
            days_to_run, skipped_days = _filter_days_with_target_coverage(
                days_to_run,
                parquet_path=parquet_path,
                pq_cols=pq_cols,
                target=target,
                master_abs=master_abs,
                master_filter=master_filter,
            )
            for day_s, why in skipped_days:
                stage(f"Prepare: skipping {day_s} — unbuildable ({why})")
                store.set_build_day_status(
                    lab_uuid,
                    day_s,
                    status=DAY_SKIPPED,
                    finished=True,
                    error_message=why,
                )

        store.write_prediction_summary(
            lab_uuid=lab_uuid,
            status=PRED_STATUS_BUILDING,
            row_count=int(existing or 0),
            trading_days=len(catalog_days),
            parent_model_name=model_name,
            parent_dataset=dataset,
            target_column=target,
            selected_feature_count=len(features),
            feature_columns_json=json.dumps(feat_map, ensure_ascii=False),
            feature_storage_mode=storage_mode,
            master_dataset_id=master_id,
            master_db_path=master_rel_store,
            created_at=_utc_now(),
        )

    stage(f"Prepare done · days_to_run={len(days_to_run)} · storage={storage_mode}")
    if not days_to_run:
        error = "No days selected to build"
        if skipped_days:
            skipped_names = ", ".join(d for d, _why in skipped_days)
            error = f"No buildable days: {skipped_names} — {skipped_days[0][1]}"
        return {
            "ok": False,
            "error": error,
            "lab_uuid": lab_uuid,
            "days": [],
            "all_days": catalog_days,
            "skipped_days": skipped_days,
        }

    config = {
        "data_dir": data_dir,
        "parquet_path": parquet_path,
        "features": features,
        "target": target,
        "wanted_columns": wanted,
        "lab_uuid": lab_uuid,
        "feat_map": feat_map,
        "horizon_sec": horizon_sec_from_target(target),
        "enrich_path_outcomes": bool(enrich_path_outcomes),
        "model_path": model_path,
        "algorithm": algorithm,
        "days": catalog_days,
        "row_limit": int(row_limit) if row_limit else None,
        "mark_day_complete": bool(mark_day_complete),
        "embed_features": embed_features,
        "master_db_path": master_abs,
        "stamp_master_row_ids": stamp_ids,
        "master_filter": master_filter,
        "master_filter_label": master_filter_summary_label(master_filter),
        "feature_columns": list(feat_map.values()) if embed_features else [],
        "storage_mode": storage_mode,
        "master_dataset_id": master_id,
        "master_db_path_store": master_rel_store,
        "parent_model_name": model_name,
        "parent_dataset": dataset,
        "total_estimate": int(total_estimate or 0),
        "package_members": package_members,
        "transformation_config": transformation_config,
        "sample_interval_sec": sample_interval_sec,
        "tb_model_name": tb_model_name,
        "tb_label_run": tb_label_run,
    }
    return {
        "ok": True,
        "lab_uuid": lab_uuid,
        "days_to_run": days_to_run,
        "all_days": days,
        "config": config,
        "skipped_days": skipped_days,
    }
