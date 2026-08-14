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


def get_pipeline(chart_dir: str, pipeline_id: str) -> dict[str, Any] | None:
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        ensure_default_existing_pipeline,
        get_pipeline_summary,
        load_store,
    )

    data_dir = data_dir_for(chart_dir)
    doc = ensure_default_existing_pipeline(data_dir)
    return get_pipeline_summary(doc, pipeline_id)


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
    doc = load_store(data_dir)
    rec = add_candidate_features(doc, pipeline_id, names, replace=replace)
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
