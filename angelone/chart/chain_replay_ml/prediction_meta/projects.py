"""Prediction project registry — many prediction DBs from one master dataset."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.dataset_builder.master_naming import resolve_master_db_path
from chain_replay_ml.dataset_builder.master_store import MasterStore

from live_inference.registry_cache import acquire_inference_registry
from live_inference.versions import feature_version

from .store import PredictionMetaStore

_PROJECTS_DIR = "prediction_projects"
_INDEX_FILE = "index.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_created(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except (TypeError, ValueError):
        return iso[:16] if iso else "—"


def slugify_project_name(display_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(display_name or "").strip()).strip("_").lower()
    if not s:
        raise ValueError("Prediction dataset name is required")
    return s


def db_filename_from_display_name(display_name: str) -> str:
    return f"{slugify_project_name(display_name)}.db"


def projects_index_path(data_dir: str) -> str:
    root = os.path.join(data_dir, "datasets", _PROJECTS_DIR)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, _INDEX_FILE)


def prediction_db_path(data_dir: str, db_filename: str) -> str:
    name = str(db_filename or "").strip()
    if not name.endswith(".db"):
        name = f"{name}.db"
    return os.path.join(data_dir, "datasets", name)


def _load_index(data_dir: str) -> dict[str, Any]:
    path = projects_index_path(data_dir)
    if not os.path.isfile(path):
        return {"version": 1, "projects": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "projects": {}}
    if not isinstance(data.get("projects"), dict):
        data["projects"] = {}
    return data


def _save_index(data_dir: str, data: dict[str, Any]) -> None:
    path = projects_index_path(data_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


@dataclass
class PredictionProject:
    project_id: str
    display_name: str
    db_filename: str
    source_master_db: str
    market: str
    sampling_interval_sec: int
    selected_models: list[str] = field(default_factory=list)
    batch_size: int = 1000
    enrich_path_outcomes: bool = True
    trading_days_filter: list[str] | None = None
    created_at: str = ""
    updated_at: str = ""
    last_build_started_at: str | None = None
    last_build_finished_at: str | None = None
    cloned_from: str | None = None
    feature_version: str | None = None
    prediction_version: int | None = None
    build_fingerprint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PredictionProject:
        return cls(
            project_id=str(raw.get("project_id") or ""),
            display_name=str(raw.get("display_name") or ""),
            db_filename=str(raw.get("db_filename") or ""),
            source_master_db=str(raw.get("source_master_db") or ""),
            market=str(raw.get("market") or "NIFTY").upper(),
            sampling_interval_sec=int(raw.get("sampling_interval_sec") or 3),
            selected_models=list(raw.get("selected_models") or []),
            batch_size=int(raw.get("batch_size") or 1000),
            enrich_path_outcomes=bool(raw.get("enrich_path_outcomes", True)),
            trading_days_filter=list(raw["trading_days_filter"]) if raw.get("trading_days_filter") else None,
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            last_build_started_at=raw.get("last_build_started_at"),
            last_build_finished_at=raw.get("last_build_finished_at"),
            cloned_from=raw.get("cloned_from"),
            feature_version=raw.get("feature_version"),
            prediction_version=int(raw["prediction_version"]) if raw.get("prediction_version") is not None else None,
            build_fingerprint=raw.get("build_fingerprint") if isinstance(raw.get("build_fingerprint"), dict) else None,
        )

    def db_path(self, data_dir: str) -> str:
        return prediction_db_path(data_dir, self.db_filename)

    def master_path(self, data_dir: str) -> str:
        src = str(self.source_master_db or "").strip()
        if os.path.isabs(src) and os.path.isfile(src):
            return src
        if os.path.sep in src or "/" in src:
            return os.path.join(data_dir, src.replace("/", os.sep))
        return os.path.join(data_dir, "datasets", os.path.basename(src))

    def project_config_blob(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "display_name": self.display_name,
            "db_filename": self.db_filename,
            "source_master_db": self.source_master_db,
            "market": self.market,
            "sample_interval_sec": self.sampling_interval_sec,
            "selected_models": self.selected_models,
            "models_selected_count": len(self.selected_models),
            "batch_size": self.batch_size,
            "enrich_path_outcomes": self.enrich_path_outcomes,
            "trading_days_filter": self.trading_days_filter,
            "feature_version": self.feature_version or feature_version(),
            "prediction_version": self.prediction_version,
            "created_at": self.created_at,
            "cloned_from": self.cloned_from,
        }


def _master_row_count_for_project(data_dir: str, project: PredictionProject) -> int | None:
    path = resolve_master_path_from_project(data_dir, project)
    if not os.path.isfile(path):
        return None
    try:
        with MasterStore(path) as store:
            return store.total_row_count()
    except Exception:
        return None


def refresh_build_fingerprint(
    data_dir: str,
    project: PredictionProject,
    *,
    build_status: str | None = None,
    master_row_count: int | None = None,
    rows_planned: int | None = None,
    prediction_row_count: int | None = None,
    prediction_version: int | None = None,
    model_registry_version: str | None = None,
    inference_registry_signature: str | None = None,
    model_registry_slot_count: int | None = None,
    completed_at: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    from .build_fingerprint import compute_build_fingerprint, merge_fingerprint

    if master_row_count is None:
        master_row_count = _master_row_count_for_project(data_dir, project)

    db_path = project.db_path(data_dir)
    if prediction_row_count is None and os.path.isfile(db_path):
        try:
            with PredictionMetaStore(db_path) as store:
                prediction_row_count = store.row_count()
                if prediction_version is None:
                    cfg = store.get_meta("project_config") or {}
                    if cfg.get("prediction_version") is not None:
                        prediction_version = int(cfg["prediction_version"])
                if model_registry_version is None:
                    model_registry_version = store.get_meta("model_registry_version")
                if inference_registry_signature is None:
                    reg_meta = store.get_meta("registry_meta") or {}
                    if isinstance(reg_meta, dict):
                        inference_registry_signature = reg_meta.get("models_dir_signature")
                if model_registry_slot_count is None:
                    from .model_registry import read_prediction_versions

                    model_registry_slot_count = len(read_prediction_versions(store.conn))
                if build_status is None:
                    build_status = store.read_progress().status
        except Exception:
            pass

    fp = compute_build_fingerprint(
        project,
        master_row_count=master_row_count,
        rows_planned=rows_planned,
        prediction_row_count=prediction_row_count,
        prediction_version=prediction_version or project.prediction_version,
        model_registry_version=model_registry_version,
        inference_registry_signature=inference_registry_signature,
        model_registry_slot_count=model_registry_slot_count,
        build_status=build_status or "pending",
        completed_at=completed_at,
    )
    project.build_fingerprint = fp

    if persist:
        index = _load_index(data_dir)
        raw = (index.get("projects") or {}).get(project.project_id)
        if raw:
            raw["build_fingerprint"] = fp
            raw["updated_at"] = _utc_now()
            index["projects"][project.project_id] = raw
            _save_index(data_dir, index)
        if os.path.isfile(db_path):
            try:
                with PredictionMetaStore(db_path) as store:
                    cfg = store.get_meta("project_config") or project.project_config_blob()
                    store.set_meta("project_config", {**cfg, "build_fingerprint": fp})
                    store.set_meta("build_fingerprint", fp)
            except Exception:
                pass

    return fp


def list_master_datasets(data_dir: str) -> list[dict[str, Any]]:
    datasets_dir = os.path.join(data_dir, "datasets")
    out: list[dict[str, Any]] = []
    if not os.path.isdir(datasets_dir):
        return out
    for name in sorted(os.listdir(datasets_dir)):
        if not name.startswith("master_dataset_") or not name.endswith(".db"):
            continue
        path = os.path.join(datasets_dir, name)
        if not os.path.isfile(path):
            continue
        m = re.match(r"master_dataset_([a-z]+)_(\d+)s\.db$", name, re.I)
        market = m.group(1).upper() if m else "NIFTY"
        interval = int(m.group(2)) if m else 3
        info: dict[str, Any] = {
            "filename": name,
            "path": path,
            "market": market,
            "sampling_interval_sec": interval,
            "row_count": None,
            "trading_days": None,
        }
        try:
            with MasterStore(path) as store:
                info["row_count"] = store.total_row_count()
                days = store.distinct_trading_days()
                info["trading_days"] = len(days)
                info["trading_day_list"] = days
        except Exception:
            pass
        out.append(info)
    return out


def list_available_models(data_dir: str) -> list[dict[str, Any]]:
    specs, _merged, _union, _meta = acquire_inference_registry(data_dir, status_filter="ready")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        reg = spec.get("registry") or {}
        rows.append({
            "model_name": str(spec.get("model_name") or ""),
            "target": str(spec.get("target") or reg.get("target") or ""),
            "algorithm": str(spec.get("algorithm") or reg.get("algorithm") or ""),
            "mae": spec.get("mae") or reg.get("mae"),
            "rmse": spec.get("rmse") or reg.get("rmse"),
            "feature_count": len(spec.get("features") or []),
        })
    return rows


def list_projects(data_dir: str) -> list[dict[str, Any]]:
    index = _load_index(data_dir)
    projects = index.get("projects") or {}
    rows: list[dict[str, Any]] = []
    for pid, raw in projects.items():
        proj = PredictionProject.from_dict({**raw, "project_id": pid})
        rows.append(enrich_project_stats(data_dir, proj))
    rows.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return rows


def get_project(data_dir: str, project_id: str) -> PredictionProject | None:
    index = _load_index(data_dir)
    raw = (index.get("projects") or {}).get(str(project_id))
    if not raw:
        return None
    return PredictionProject.from_dict({**raw, "project_id": project_id})


def _count_trading_days(db_path: str) -> int:
    if not os.path.isfile(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT COUNT(DISTINCT trading_day) FROM samples").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def enrich_project_stats(data_dir: str, project: PredictionProject) -> dict[str, Any]:
    db_path = project.db_path(data_dir)
    exists = os.path.isfile(db_path)
    row_count = 0
    status = "missing"
    build_status = None
    if exists:
        try:
            with PredictionMetaStore(db_path) as store:
                row_count = store.row_count()
                prog = store.read_progress()
                status = prog.status
                build_status = prog.status
                cfg = store.get_meta("project_config") or {}
                if cfg.get("prediction_version") is not None:
                    project.prediction_version = int(cfg["prediction_version"])
        except Exception:
            status = "unknown"

    master_name = os.path.basename(project.source_master_db)
    master_short = master_name.replace("master_dataset_", "Master_").replace(".db", "")
    if project.sampling_interval_sec:
        master_short = f"{master_short}_{project.sampling_interval_sec}s"

    fp = project.build_fingerprint
    if not fp and exists:
        fp = _load_fingerprint_from_db(data_dir, project)

    return {
        **project.to_dict(),
        "db_path": db_path,
        "exists": exists,
        "row_count": row_count,
        "trading_days": _count_trading_days(db_path) if exists else None,
        "status": build_status or status,
        "created_label": _fmt_created(project.created_at),
        "source_master_label": master_name,
        "source_master_short": master_short.replace("_nifty", "").replace("_", " "),
        "models_count": len(project.selected_models),
        "build_fingerprint": fp,
    }


def _load_fingerprint_from_db(data_dir: str, project: PredictionProject) -> dict[str, Any] | None:
    db_path = project.db_path(data_dir)
    if not os.path.isfile(db_path):
        return refresh_build_fingerprint(data_dir, project, persist=False)
    try:
        with PredictionMetaStore(db_path) as store:
            fp = store.get_meta("build_fingerprint")
            if isinstance(fp, dict):
                return fp
            cfg = store.get_meta("project_config") or {}
            if isinstance(cfg.get("build_fingerprint"), dict):
                return cfg["build_fingerprint"]
    except Exception:
        pass
    return refresh_build_fingerprint(data_dir, project, persist=False)


def create_project(
    data_dir: str,
    *,
    display_name: str,
    source_master_db: str,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    selected_models: list[str],
    batch_size: int = 1000,
    enrich_path_outcomes: bool = True,
    trading_days_filter: list[str] | None = None,
    cloned_from: str | None = None,
) -> dict[str, Any]:
    if not selected_models:
        raise ValueError("Select at least one model")

    project_id = slugify_project_name(display_name)
    db_filename = db_filename_from_display_name(display_name)
    db_path = prediction_db_path(data_dir, db_filename)

    index = _load_index(data_dir)
    projects = index.setdefault("projects", {})
    if project_id in projects:
        raise ValueError(f"Project already exists: {display_name}")
    if os.path.isfile(db_path):
        raise ValueError(f"Database file already exists: {db_filename}")

    now = _utc_now()
    proj = PredictionProject(
        project_id=project_id,
        display_name=display_name.strip(),
        db_filename=db_filename,
        source_master_db=os.path.basename(source_master_db),
        market=str(market or "NIFTY").upper(),
        sampling_interval_sec=int(sampling_interval_sec),
        selected_models=list(selected_models),
        batch_size=int(batch_size),
        enrich_path_outcomes=bool(enrich_path_outcomes),
        trading_days_filter=trading_days_filter,
        created_at=now,
        updated_at=now,
        cloned_from=cloned_from,
        feature_version=feature_version(),
    )
    projects[project_id] = proj.to_dict()
    _save_index(data_dir, index)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with PredictionMetaStore(db_path) as store:
        store.set_meta("project_config", proj.project_config_blob())

    refresh_build_fingerprint(data_dir, proj, build_status="pending")
    return enrich_project_stats(data_dir, proj)


def clone_project_config(data_dir: str, source_project_id: str, *, new_display_name: str) -> dict[str, Any]:
    src = get_project(data_dir, source_project_id)
    if not src:
        raise ValueError(f"Source project not found: {source_project_id}")
    return create_project(
        data_dir,
        display_name=new_display_name,
        source_master_db=src.source_master_db,
        market=src.market,
        sampling_interval_sec=src.sampling_interval_sec,
        selected_models=list(src.selected_models),
        batch_size=src.batch_size,
        enrich_path_outcomes=src.enrich_path_outcomes,
        trading_days_filter=src.trading_days_filter,
        cloned_from=source_project_id,
    )


def update_project_after_build(
    data_dir: str,
    project_id: str,
    *,
    prediction_version: int | None,
    started: bool = False,
    finished: bool = False,
    build_status: str | None = None,
    rows_planned: int | None = None,
    prediction_row_count: int | None = None,
    model_registry_version: str | None = None,
    inference_registry_signature: str | None = None,
    model_registry_slot_count: int | None = None,
) -> None:
    index = _load_index(data_dir)
    raw = (index.get("projects") or {}).get(project_id)
    if not raw:
        return
    now = _utc_now()
    raw["updated_at"] = now
    if started:
        raw["last_build_started_at"] = now
    if finished:
        raw["last_build_finished_at"] = now
    if prediction_version is not None:
        raw["prediction_version"] = prediction_version
    raw["feature_version"] = feature_version()
    index["projects"][project_id] = raw
    _save_index(data_dir, index)

    proj = PredictionProject.from_dict({**raw, "project_id": project_id})
    status = build_status or ("complete" if finished else "running" if started else None)
    refresh_build_fingerprint(
        data_dir,
        proj,
        build_status=status or "pending",
        rows_planned=rows_planned,
        prediction_row_count=prediction_row_count,
        prediction_version=prediction_version,
        model_registry_version=model_registry_version,
        inference_registry_signature=inference_registry_signature,
        model_registry_slot_count=model_registry_slot_count,
        completed_at=now if finished else None,
    )


def delete_project(data_dir: str, project_id: str, *, delete_db: bool = True) -> dict[str, Any]:
    proj = get_project(data_dir, project_id)
    if not proj:
        raise ValueError(f"Project not found: {project_id}")
    db_path = proj.db_path(data_dir)
    removed = False
    if delete_db and os.path.isfile(db_path):
        os.remove(db_path)
        removed = True
    index = _load_index(data_dir)
    index.get("projects", {}).pop(project_id, None)
    _save_index(data_dir, index)
    return {"deleted": project_id, "db_removed": removed}


def resolve_master_path_from_project(data_dir: str, project: PredictionProject) -> str:
    path = project.master_path(data_dir)
    if os.path.isfile(path):
        return path
    return resolve_master_db_path(
        data_dir,
        market=project.market,
        sampling_interval_sec=project.sampling_interval_sec,
    )


def filter_specs_by_selection(
    specs: list[dict[str, Any]],
    selected_models: list[str],
) -> list[dict[str, Any]]:
    by_name = {str(s.get("model_name") or ""): s for s in specs}
    out: list[dict[str, Any]] = []
    for name in selected_models:
        spec = by_name.get(name)
        if spec:
            out.append(spec)
    return out
