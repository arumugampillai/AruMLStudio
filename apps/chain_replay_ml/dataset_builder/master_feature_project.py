"""Master Dataset ↔ Feature Project identity binding."""

from __future__ import annotations

from typing import Any

from .feature_project_organization import RESERVED_ALL_PROJECT_ID
from .feature_registry_store import load_store as load_feature_registry_store

META_CONFIG_KEY = "feature_project_id"


class MasterFeatureProjectError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def normalize_feature_project_id(project_id: str | None) -> str:
    pid = str(project_id or "").strip().lower()
    return pid or RESERVED_ALL_PROJECT_ID


def project_exists(data_dir: str, project_id: str) -> bool:
    from .feature_project_organization import is_reserved_all_project_id

    pid = normalize_feature_project_id(project_id)
    if is_reserved_all_project_id(pid):
        return True
    doc = load_feature_registry_store(data_dir)
    projects = doc.get("projects") if isinstance(doc.get("projects"), dict) else {}
    return pid in projects


def validate_feature_project_id(data_dir: str, project_id: str) -> str:
    pid = normalize_feature_project_id(project_id)
    if not project_exists(data_dir, pid):
        raise MasterFeatureProjectError(
            f"Feature project '{pid}' does not exist. "
            "Select a valid project on the Create Dataset page."
        )
    return pid


def read_master_config_feature_project_id(store: Any) -> str | None:
    cfg = store.get_meta("master_config")
    if isinstance(cfg, dict):
        raw = cfg.get(META_CONFIG_KEY)
        if raw is not None and str(raw).strip():
            return normalize_feature_project_id(str(raw))
    return None


def read_master_feature_project_id(store: Any) -> str | None:
    """Read project id from master_dataset_meta column or master_config fallback."""
    meta = store.read_master_meta_dict()
    col = meta.get(META_CONFIG_KEY)
    if col is not None and str(col).strip():
        return normalize_feature_project_id(str(col))
    return read_master_config_feature_project_id(store)


def _write_master_feature_project_id(store: Any, project_id: str) -> None:
    pid = normalize_feature_project_id(project_id)
    store.conn.execute(
        "UPDATE master_dataset_meta SET feature_project_id = ? WHERE id = 1",
        (pid,),
    )
    cfg = store.get_meta("master_config")
    if not isinstance(cfg, dict):
        cfg = {}
    cfg = dict(cfg)
    cfg[META_CONFIG_KEY] = pid
    store.set_meta("master_config", cfg)
    store.conn.commit()


def ensure_master_feature_project_id(
    store: Any,
    data_dir: str,
    *,
    default: str = RESERVED_ALL_PROJECT_ID,
) -> str:
    """Migrate/backfill and return canonical project id for this master dataset."""
    pid = read_master_feature_project_id(store)
    if pid is None:
        pid = normalize_feature_project_id(default)
    pid = validate_feature_project_id(data_dir, pid)
    current_col = store.read_master_meta_dict().get(META_CONFIG_KEY)
    current_cfg = read_master_config_feature_project_id(store)
    if (
        normalize_feature_project_id(str(current_col or "")) != pid
        or normalize_feature_project_id(str(current_cfg or "")) != pid
    ):
        _write_master_feature_project_id(store, pid)
    return pid


def require_master_feature_project_id(store: Any, data_dir: str) -> str:
    pid = read_master_feature_project_id(store)
    if not pid:
        raise MasterFeatureProjectError(
            "Master Dataset has no feature_project_id. "
            "Open Create Dataset, select a Feature Project, and rebuild or migrate the master DB."
        )
    return validate_feature_project_id(data_dir, pid)


def set_master_feature_project_id(store: Any, data_dir: str, project_id: str) -> str:
    pid = validate_feature_project_id(data_dir, project_id)
    _write_master_feature_project_id(store, pid)
    return pid


def resolve_master_feature_project_id_for_path(
    master_db_path: str,
    data_dir: str,
    *,
    migrate: bool = True,
) -> str:
    from .master_store import MasterStore

    store = MasterStore(master_db_path)
    store.open()
    try:
        if migrate:
            return ensure_master_feature_project_id(store, data_dir)
        return require_master_feature_project_id(store, data_dir)
    finally:
        store.close()


def active_project_feature_names(data_dir: str, project_id: str) -> frozenset[str]:
    from .feature_project_organization import project_registry_feature_source

    pid = validate_feature_project_id(data_dir, project_id)
    src = project_registry_feature_source(data_dir=data_dir, project_id=pid)
    names = {str(n).strip() for n in (src.get("features") or []) if str(n).strip()}
    return frozenset(names)
