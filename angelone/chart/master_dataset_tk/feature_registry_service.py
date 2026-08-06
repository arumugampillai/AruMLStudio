"""In-process feature registry operations (no HTTP / chart server)."""

from __future__ import annotations

import json
from typing import Any

from .build_service import chart_data_dir


def data_dir_for(chart_dir: str) -> str:
    return chart_data_dir(chart_dir)


def load_catalog(chart_dir: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_catalog import build_feature_registry_catalog

    return build_feature_registry_catalog(data_dir_for(chart_dir))


def save_planned_feature(chart_dir: str, entry: dict[str, Any]) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_catalog import save_planned_feature

    saved = save_planned_feature(data_dir_for(chart_dir), entry)
    return {"ok": True, "feature": saved}


def preview_import(
    chart_dir: str,
    *,
    payload: dict[str, Any],
    import_type: str,
    target_group: str | None = None,
    new_group: dict[str, Any] | None = None,
    conflict_policy: str = "skip",
    resolutions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_catalog import build_feature_registry_catalog
    from chain_replay_ml.dataset_builder.feature_registry_store import (
        load_store,
        parse_import_payload,
        preview_import as _preview,
    )

    data_dir = data_dir_for(chart_dir)
    meta, incoming, group = parse_import_payload(payload)
    catalog = build_feature_registry_catalog(data_dir)
    store = load_store(data_dir)
    result = _preview(
        store=store,
        catalog_features=catalog.get("features") or [],
        catalog_groups=catalog.get("groups") or [],
        incoming=incoming,
        import_type=import_type,
        target_group=target_group,
        new_group=new_group or group,
        conflict_policy=conflict_policy,
        resolutions=resolutions or {},
    )
    result["import_meta"] = meta
    return result


def apply_import(
    chart_dir: str,
    *,
    payload: dict[str, Any],
    import_type: str,
    target_group: str | None = None,
    new_group: dict[str, Any] | None = None,
    conflict_policy: str = "skip",
    resolutions: dict[str, Any] | None = None,
    bulk_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_catalog import build_feature_registry_catalog
    from chain_replay_ml.dataset_builder.feature_registry_store import (
        apply_import as _apply,
        load_store,
        parse_import_payload,
    )

    data_dir = data_dir_for(chart_dir)
    meta, incoming, group = parse_import_payload(payload)
    catalog = build_feature_registry_catalog(data_dir)
    store = load_store(data_dir)
    return _apply(
        data_dir=data_dir,
        store=store,
        catalog_features=catalog.get("features") or [],
        catalog_groups=catalog.get("groups") or [],
        incoming=incoming,
        import_type=import_type,
        target_group=target_group,
        new_group=new_group or group,
        conflict_policy=conflict_policy,
        resolutions=resolutions or {},
        import_meta=meta,
        bulk_defaults=bulk_defaults,
    )


def bulk_update(chart_dir: str, names: list[str], updates: dict[str, Any]) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import bulk_update_features, load_store

    data_dir = data_dir_for(chart_dir)
    store = load_store(data_dir)
    return bulk_update_features(data_dir, store, names, updates)


def set_feature_registry_active(
    chart_dir: str,
    name: str,
    *,
    active: bool,
    home_group_id: str | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import (
        load_store,
        set_feature_registry_active as _set_active,
    )

    data_dir = data_dir_for(chart_dir)
    store = load_store(data_dir)
    return _set_active(
        data_dir,
        store,
        name,
        active=active,
        home_group_id=home_group_id,
    )


def disabled_registry_features(chart_dir: str) -> set[str]:
    from chain_replay_ml.dataset_builder.feature_registry_store import (
        disabled_registry_feature_names,
        load_store,
    )

    return disabled_registry_feature_names(load_store(data_dir_for(chart_dir)))


def filter_active_registry_names(chart_dir: str | None, names: list[str]) -> list[str]:
    """Drop features marked inactive in the feature registry overlay."""
    if not chart_dir:
        return list(names)
    disabled = disabled_registry_features(chart_dir)
    if not disabled:
        return list(names)
    return [n for n in names if n not in disabled]


def create_project(
    chart_dir: str,
    *,
    label: str,
    project_id: str | None = None,
    description: str = "",
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    warmup_minutes: int | None = None,
    default_sampling: str = "",
    notes: str = "",
    version: str = "1",
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import create_project

    doc = create_project(
        data_dir_for(chart_dir),
        label=label,
        project_id=project_id,
        description=description,
        group_ids=group_ids or [],
        feature_names=feature_names or [],
        warmup_minutes=warmup_minutes,
        default_sampling=default_sampling,
        notes=notes,
        version=version,
    )
    return {"ok": True, "project": doc}


def update_project(
    chart_dir: str,
    project_id: str,
    *,
    label: str | None = None,
    description: str | None = None,
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    warmup_minutes: int | None | object = ...,
    default_sampling: str | None = None,
    notes: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import update_project

    kwargs: dict[str, Any] = {
        "label": label,
        "description": description,
        "group_ids": group_ids,
        "feature_names": feature_names,
        "default_sampling": default_sampling,
        "notes": notes,
        "version": version,
    }
    if warmup_minutes is not ...:
        kwargs["warmup_minutes"] = warmup_minutes
    doc = update_project(data_dir_for(chart_dir), project_id, **kwargs)
    return {"ok": True, "project": doc}


def clone_project(
    chart_dir: str,
    source_project_id: str,
    *,
    label: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import clone_project

    doc = clone_project(
        data_dir_for(chart_dir),
        source_project_id,
        label=label,
        project_id=project_id,
    )
    return {"ok": True, "project": doc}


def delete_project(chart_dir: str, project_id: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_store import delete_project

    return {"ok": True, **delete_project(data_dir_for(chart_dir), project_id)}


def delete_preview(chart_dir: str, feature_ref: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_delete import preview_feature_delete

    return preview_feature_delete(data_dir_for(chart_dir), feature_ref)


def delete_feature(chart_dir: str, *, feature_id: str | None = None, name: str | None = None) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_registry_delete import apply_feature_delete

    ref = (feature_id or name or "").strip()
    if not ref:
        raise ValueError("feature_id or name is required")
    return apply_feature_delete(data_dir_for(chart_dir), ref)


def parity_rules() -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_grid_policy import (
        FEATURE_PARITY_VERSION,
        FeatureComputationKind,
        FeatureSharedScope,
        feature_parity_audit_rows,
    )

    return {
        "version": FEATURE_PARITY_VERSION,
        "kinds": [
            FeatureComputationKind.GRID_BAR.value,
            FeatureComputationKind.CALENDAR_SEC.value,
            FeatureComputationKind.STATIC.value,
        ],
        "scopes": [s.value for s in FeatureSharedScope],
        "golden_rule": (
            "trainingIntervalSec drives feature_grid_step_sec for all grid_bar rollings; "
            "dataset build, replay inference, and live trading share build_day_rows."
        ),
        "rows": feature_parity_audit_rows(),
    }


def run_pipeline_parity_audit(
    chart_dir: str,
    *,
    trading_day: str | None = None,
    timestamp: float | None = None,
    token: str | None = None,
    dataset_name: str | None = None,
    tolerance: float = 1e-6,
    include_parquet: bool = True,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.feature_pipeline_parity import run_feature_pipeline_parity_audit

    return run_feature_pipeline_parity_audit(
        data_dir_for(chart_dir),
        trading_day=trading_day,
        timestamp=timestamp,
        token=token,
        dataset_name=dataset_name,
        tolerance=tolerance,
        include_parquet=include_parquet,
    )


def export_catalog_json(catalog: dict[str, Any], *, filtered_features: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    import datetime as dt

    selected_name = context.get("selected_feature")
    selected = next((f for f in (catalog.get("features") or []) if f.get("name") == selected_name), None)
    return {
        "registry_version": catalog.get("registry_version"),
        "created_by": catalog.get("registry_created_by"),
        "created_on": catalog.get("registry_created_on"),
        "description": catalog.get("registry_description"),
        "exported_at": dt.datetime.utcnow().isoformat() + "Z",
        "export_context": context,
        "schema_version": catalog.get("schema_version"),
        "schema_registry_hash": catalog.get("schema_registry_hash"),
        "stats": catalog.get("stats"),
        "groups": catalog.get("groups"),
        "projects": catalog.get("projects"),
        "status_options": catalog.get("status_options"),
        "pipeline_surfaces": catalog.get("pipeline_surfaces"),
        "features": catalog.get("features"),
        "filtered_features": filtered_features,
        "selected_feature_detail": selected,
    }


def parse_expected_range(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    if "," in raw:
        parts = [s.strip() for s in raw.split(",")]
        if len(parts) == 2 and all(p != "" and not _is_nan(p) for p in parts):
            return [float(parts[0]), float(parts[1])]
    return raw


def _is_nan(s: str) -> bool:
    try:
        float(s)
        return False
    except ValueError:
        return True


def split_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text or "").split(",") if s.strip()]


def load_json_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data
