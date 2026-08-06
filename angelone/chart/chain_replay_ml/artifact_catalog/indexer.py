"""Discover existing on-disk artifacts and register them (do not rewrite)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .store import ArtifactCatalogStore
from .types import ArtifactRecord
from .uri import (
    diagnostics_uri,
    feature_studio_uri,
    model_uri,
    prediction_uri,
    training_uri,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
    except OSError:
        return _utc_now()


def index_models(store: ArtifactCatalogStore, data_dir: str) -> list[str]:
    """Scan ``data/models/<name>/`` and Feature Studio / planner sidecars."""
    from chain_replay_ml.training.paths import models_dir

    registered: list[str] = []
    root = models_dir(data_dir)
    if not os.path.isdir(root):
        return registered

    studio_map = {
        "feature_importance_studio": "importance",
        "feature_distribution_studio": "distribution",
        "feature_drift_studio": "drift",
        "diagnostics_studio": "diagnostics",
        "experiment_planner": "experiment_planner",
    }

    for name in sorted(os.listdir(root)):
        pkg = os.path.join(root, name)
        if not os.path.isdir(pkg):
            continue
        m_uri = model_uri(name)
        parents: list[str] = []
        meta: dict[str, Any] = {"model_name": name}
        cfg_path = os.path.join(pkg, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as fh:
                    cfg = json.loads(fh.read())
                if isinstance(cfg, dict):
                    meta["config_keys"] = sorted(cfg.keys())[:40]
                    ds = cfg.get("dataset") or (cfg.get("training") or {}).get("dataset")
                    if ds:
                        meta["dataset"] = ds
            except (OSError, json.JSONDecodeError):
                pass
        store.register(
            ArtifactRecord(
                artifact_uri=m_uri,
                artifact_type="model",
                created_at=_mtime_iso(pkg),
                local_path=pkg,
                parent_artifact_uris=parents,
                metadata=meta,
                capabilities=["comparable", "deployable", "visualizable"],
                status="completed",
            )
        )
        registered.append(m_uri)

        for dirname, studio_key in studio_map.items():
            studio_path = os.path.join(pkg, dirname)
            if not os.path.isdir(studio_path):
                continue
            if dirname == "diagnostics_studio":
                uri = diagnostics_uri(name)
                atype = "diagnostics"
            elif dirname == "experiment_planner":
                uri = feature_studio_uri("planner", name)
                atype = "feature_studio"
            else:
                uri = feature_studio_uri(studio_key, name)
                atype = "feature_studio"
            store.register(
                ArtifactRecord(
                    artifact_uri=uri,
                    artifact_type=atype,
                    created_at=_mtime_iso(studio_path),
                    local_path=studio_path,
                    parent_artifact_uris=[m_uri],
                    metadata={"studio": studio_key, "model_name": name},
                    capabilities=["visualizable", "comparable"],
                    status="completed",
                )
            )
            registered.append(uri)
    return registered


def index_datasets(store: ArtifactCatalogStore, data_dir: str) -> list[str]:
    """Scan ``data/datasets/`` parquet / names as prediction or training pointers."""
    registered: list[str] = []
    ds_root = os.path.join(data_dir, "datasets")
    if not os.path.isdir(ds_root):
        return registered
    for name in sorted(os.listdir(ds_root)):
        path = os.path.join(ds_root, name)
        if not (os.path.isfile(path) or os.path.isdir(path)):
            continue
        base = os.path.splitext(name)[0]
        # Heuristic: training_* → training, else prediction/other dataset.
        if base.startswith("training_dataset") or base.startswith("training_"):
            uri = training_uri(base)
            atype = "training"
            caps = ["trainable", "comparable"]
        else:
            uri = prediction_uri(base)
            atype = "prediction"
            caps = ["comparable", "visualizable"]
        store.register(
            ArtifactRecord(
                artifact_uri=uri,
                artifact_type=atype,
                created_at=_mtime_iso(path),
                local_path=path,
                parent_artifact_uris=[],
                metadata={"name": base},
                capabilities=caps,
                status="completed",
            )
        )
        registered.append(uri)
    return registered


def index_ole_training_root(
    store: ArtifactCatalogStore,
    training_root: str,
    *,
    parent_uris: list[str] | None = None,
) -> list[str]:
    """Index OLE immutable training_dataset_* dirs under a root."""
    registered: list[str] = []
    if not os.path.isdir(training_root):
        return registered
    parents = list(parent_uris or [])
    for name in sorted(os.listdir(training_root)):
        if not name.startswith("training_dataset_"):
            continue
        path = os.path.join(training_root, name)
        if not os.path.isdir(path):
            continue
        meta: dict[str, Any] = {"artifact_id": name}
        run_meta_path = os.path.join(path, "run_meta.json")
        if os.path.isfile(run_meta_path):
            try:
                with open(run_meta_path, encoding="utf-8") as fh:
                    meta.update(json.loads(fh.read()))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        uri = training_uri(name)
        store.register(
            ArtifactRecord(
                artifact_uri=uri,
                artifact_type="training",
                created_at=str(meta.get("created_at_utc") or _mtime_iso(path)),
                local_path=path,
                parent_artifact_uris=parents,
                metadata={
                    "strategy": meta.get("strategy"),
                    "engine_version": meta.get("engine_version"),
                    "label_version": meta.get("version"),
                    "rows": meta.get("rows"),
                },
                capabilities=["trainable", "comparable"],
                status="completed",
            )
        )
        registered.append(uri)
    return registered


def rebuild_catalog_index(
    store: ArtifactCatalogStore,
    data_dir: str,
    *,
    ole_training_root: str | None = None,
) -> dict[str, Any]:
    """Indexer entrypoint — discover existing artifacts into the catalog."""
    models = index_models(store, data_dir)
    datasets = index_datasets(store, data_dir)
    ole: list[str] = []
    if ole_training_root:
        ole = index_ole_training_root(store, ole_training_root)
    return {
        "models": len(models),
        "datasets": len(datasets),
        "ole_training": len(ole),
        "total_registered": len(models) + len(datasets) + len(ole),
    }
