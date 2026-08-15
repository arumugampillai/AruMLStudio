"""In-process pipeline registry operations."""

from __future__ import annotations

from typing import Any

from .build_service import chart_data_dir


def data_dir_for(chart_dir: str) -> str:
    return chart_data_dir(chart_dir)


def load_pipelines(chart_dir: str) -> list[dict[str, Any]]:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        list_pipelines,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    return list_pipelines(doc)


def get_experimental_pipelines(chart_dir: str) -> list[dict[str, Any]]:
    """Experimental pipelines only (excludes Base pipeline)."""
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        list_experimental_pipelines,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    return list_experimental_pipelines(doc)


def get_pipeline(chart_dir: str, pipeline_id: str) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        get_pipeline_summary,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    return get_pipeline_summary(doc, pipeline_id)


def is_base_pipeline(chart_dir: str, pipeline_id: str) -> bool:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        get_pipeline,
        is_base_pipeline_record,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    rec = get_pipeline(doc, pipeline_id)
    return is_base_pipeline_record(rec)


def build_pipeline_snapshot(chart_dir: str, pipeline_id: str) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        build_pipeline_snapshot as _build_snapshot,
        ensure_default_existing_pipeline,
        get_pipeline,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    rec = get_pipeline(doc, pipeline_id)
    if not rec:
        return None
    return _build_snapshot(rec, pipeline_id=pipeline_id)


def resolve_pipeline_dataset_feature_names(chart_dir: str, pipeline_id: str) -> list[str]:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        load_store,
        resolve_pipeline_dataset_feature_names as _resolve,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    return _resolve(data_dir, doc, pipeline_id)


def peek_next_pipeline_identity(chart_dir: str) -> dict[str, str]:
    """Preview the permanent ID and display name for the next pipeline create."""
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        peek_next_pipeline_identity as _peek,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    pipeline_id, name = _peek(doc)
    return {"pipeline_id": pipeline_id, "name": name}


def create_pipeline(
    chart_dir: str,
    *,
    name: str | None = None,
    pipeline_type: str = "manual",
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        create_pipeline as _create,
        get_pipeline_summary,
        load_store,
        save_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = load_store(data_dir)
    rec = _create(doc, name=name, pipeline_type=pipeline_type, status="draft")
    save_store(data_dir, doc)
    return get_pipeline_summary(doc, rec["pipeline_id"]) or rec


def update_pipeline_name(chart_dir: str, pipeline_id: str, name: str) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        get_pipeline_summary,
        load_store,
        save_store,
        update_pipeline,
    )

    data_dir = data_dir_for(chart_dir)
    doc = load_store(data_dir)
    rec = update_pipeline(doc, pipeline_id, name=name)
    if rec is None:
        return None
    save_store(data_dir, doc)
    return get_pipeline_summary(doc, pipeline_id)


def delete_pipeline(chart_dir: str, pipeline_id: str) -> bool:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        delete_pipeline as _delete,
        load_store,
        save_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = load_store(data_dir)
    deleted = _delete(doc, pipeline_id)
    if deleted:
        save_store(data_dir, doc)
    return deleted


def update_pipeline_transformation_config(
    chart_dir: str,
    pipeline_id: str,
    transformation_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        get_pipeline_summary,
        load_store,
        save_store,
        update_pipeline,
    )

    data_dir = data_dir_for(chart_dir)
    if isinstance(transformation_config, dict):
        from master_dataset_tk.auto_candidate_generation import (
            sanitize_transformation_config_for_data_dir,
        )

        transformation_config = sanitize_transformation_config_for_data_dir(
            transformation_config,
            data_dir,
        )
    doc = load_store(data_dir)
    rec = update_pipeline(doc, pipeline_id, transformation_config=transformation_config)
    if rec is None:
        return None
    save_store(data_dir, doc)
    return get_pipeline_summary(doc, pipeline_id)


def set_pipeline_registry_members(
    chart_dir: str,
    pipeline_id: str,
    feature_ids: list[str],
) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        get_pipeline_summary,
        load_store,
        save_store,
        set_registry_members,
    )

    data_dir = data_dir_for(chart_dir)
    doc = load_store(data_dir)
    rec = set_registry_members(doc, pipeline_id, feature_ids)
    if rec is None:
        return None
    save_store(data_dir, doc)
    return get_pipeline_summary(doc, pipeline_id)


def add_pipeline_candidates(
    chart_dir: str,
    pipeline_id: str,
    names: list[str],
    *,
    replace: bool = False,
) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        add_candidate_features,
        get_pipeline_summary,
        load_store,
        save_store,
    )

    data_dir = data_dir_for(chart_dir)
    from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
        is_excluded_pipeline_feature,
    )

    clean_names = [
        str(n).strip()
        for n in names
        if str(n).strip() and not is_excluded_pipeline_feature(str(n).strip(), data_dir)
    ]
    doc = load_store(data_dir)
    rec = add_candidate_features(doc, pipeline_id, clean_names, replace=replace)
    if rec is None:
        return None
    save_store(data_dir, doc)
    return get_pipeline_summary(doc, pipeline_id)


def registry_catalog_features(chart_dir: str) -> list[dict[str, Any]]:
    from chain_replay_ml.dataset_builder.feature_registry_catalog import build_feature_registry_catalog

    catalog = build_feature_registry_catalog(data_dir_for(chart_dir))
    feats = catalog.get("features") or []
    out: list[dict[str, Any]] = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("feature_id") or "").strip().upper()
        name = str(f.get("name") or "").strip()
        if not fid or not name:
            continue
        if str(f.get("disabled") or "").lower() == "true" or f.get("retired"):
            continue
        out.append(
            {
                "feature_id": fid,
                "name": name,
                "group": str(f.get("group") or f.get("group_id") or ""),
                "label": str(f.get("display_name") or name),
            }
        )
    out.sort(key=lambda r: (r.get("group") or "", r.get("name") or ""))
    return out
