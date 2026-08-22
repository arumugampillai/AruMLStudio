"""Core Data Root & Canonical Path Resolution Service (Phase 1, Doc 17)."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Literal

DEFAULT_CANONICAL_DATA_ROOT = r"D:\data"
_GLOBAL_DATA_ROOT_SERVICE: DataRootService | None = None


def normalize_storage_path(path: str) -> str:
    """Normalize and return absolute path with uniform Windows separators."""
    if not path:
        return ""
    return os.path.abspath(os.path.normpath(str(path).strip()))


def _config_file_path() -> str:
    """Return the authoritative path to ml_research_studio.json in AppData."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "AruMLStudio") if os.environ.get("APPDATA") else os.path.join(base, ".arumlstudio")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "ml_research_studio.json")
    if not os.path.isfile(target):
        legacy = os.path.join(base, "AruNeo", "ml_research_studio.json") if os.environ.get("APPDATA") else os.path.join(base, ".aruneo", "ml_research_studio.json")
        if os.path.isfile(legacy):
            try:
                shutil.copy2(legacy, target)
            except Exception:
                pass
    return target


def load_application_config() -> dict[str, Any]:
    """Load global application configuration dictionary from AppData."""
    path = _config_file_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_application_config(updates: dict[str, Any]) -> None:
    """Update global application configuration dictionary in AppData."""
    doc = load_application_config()
    doc.update(updates)
    path = _config_file_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def resolve_data_root(custom_root: str | None = None) -> str:
    """Determine the authoritative Data Root path.
    
    Resolution Priority:
    1. Direct argument (`custom_root`)
    2. Environment Variable `ARUMLSTUDIO_DATA_ROOT` (fallback: `ARUNEO_DATA_ROOT`)
    3. `data_root` field in `%APPDATA%/AruMLStudio/ml_research_studio.json`
    4. Canonical default: `D:\\data`
    """
    if custom_root and str(custom_root).strip():
        return normalize_storage_path(custom_root)

    env = (
        os.environ.get("ARUMLSTUDIO_DATA_ROOT")
        or os.environ.get("ARUNEO_DATA_ROOT")
        or ""
    ).strip()
    if env:
        return normalize_storage_path(env)

    saved = str(load_application_config().get("data_root") or "").strip()
    if saved:
        return normalize_storage_path(saved)

    return normalize_storage_path(DEFAULT_CANONICAL_DATA_ROOT)


def save_data_root(data_root: str) -> None:
    """Persist the configured Data Root path into ml_research_studio.json."""
    norm = normalize_storage_path(data_root)
    save_application_config({"data_root": norm})
    global _GLOBAL_DATA_ROOT_SERVICE
    _GLOBAL_DATA_ROOT_SERVICE = DataRootService(data_root=norm)


class DataRootService:
    """Single Source of Truth for all persistent application storage paths (Doc 17).
    
    Provides deterministic, canonical path resolution without side-effects.
    Resolving a path NEVER creates a directory or file on disk.
    """

    def __init__(self, data_root: str | None = None) -> None:
        self._data_root = resolve_data_root(data_root)

    @property
    def data_root(self) -> str:
        """The canonical root directory for all application-owned persistent data."""
        return self._data_root

    def get_data_root(self) -> str:
        """Return the canonical root directory."""
        return self._data_root

    def get_database_path(
        self,
        db_type: Literal["analysis", "feature_evidence", "angel_historic", "predictions", "strategies"] | str,
    ) -> str:
        """Return the canonical path to an authoritative SQLite database.
        
        Standard Mappings:
        - "analysis" -> <data_root>/databases/analysis.db
        - "feature_evidence" -> <data_root>/databases/feature_recommendation_evidence.db
        - "angel_historic" -> <data_root>/databases/angel_historic_bars.db
        - "predictions" -> <data_root>/databases/prediction_runs.db
        - "strategies" -> <data_root>/databases/strategy_runs.db
        """
        db_map = {
            "analysis": "analysis.db",
            "feature_evidence": "feature_recommendation_evidence.db",
            "feature_recommendation_evidence": "feature_recommendation_evidence.db",
            "angel_historic": "angel_historic_bars.db",
            "angel_historic_bars": "angel_historic_bars.db",
            "predictions": "prediction_runs.db",
            "prediction_runs": "prediction_runs.db",
            "strategies": "strategy_runs.db",
            "strategy_runs": "strategy_runs.db",
        }
        filename = db_map.get(str(db_type).lower())
        if not filename:
            filename = str(db_type) if str(db_type).endswith(".db") else f"{db_type}.db"
        return os.path.join(self._data_root, "databases", filename)

    def get_registry_path(
        self,
        reg_type: Literal["feature", "pipeline", "model"] | str,
    ) -> str:
        """Return the canonical path to an authoritative registry file.
        
        Standard Mappings:
        - "feature" -> <data_root>/registries/feature_registry_store.json
        - "pipeline" -> <data_root>/registries/pipeline_registry_store.json
        - "model" -> <data_root>/registries/model_registry.db
        """
        reg_map = {
            "feature": "feature_registry_store.json",
            "feature_registry": "feature_registry_store.json",
            "pipeline": "pipeline_registry_store.json",
            "pipeline_registry": "pipeline_registry_store.json",
            "model": "model_registry.db",
            "model_registry": "model_registry.db",
        }
        filename = reg_map.get(str(reg_type).lower())
        if not filename:
            filename = str(reg_type)
        return os.path.join(self._data_root, "registries", filename)

    def get_datasets_dir(
        self,
        category: Literal["master", "analysis", "labels", "exports"] | str = "analysis",
    ) -> str:
        """Return canonical datasets directory by category (master, analysis, labels, exports)."""
        cat = str(category).strip().lower() or "analysis"
        return os.path.join(self._data_root, "datasets", cat)

    def get_models_dir(
        self,
        category: Literal["production", "candidates", "research"] | str = "research",
    ) -> str:
        """Return canonical models directory by category (production, candidates, research)."""
        cat = str(category).strip().lower() or "research"
        return os.path.join(self._data_root, "models", cat)

    def get_research_dir(
        self,
        category: Literal["campaigns", "discovery", "snapshots", "dossiers"] | str = "discovery",
    ) -> str:
        """Return canonical autonomous research directory by category."""
        cat = str(category).strip().lower() or "discovery"
        return os.path.join(self._data_root, "research", cat)

    def get_predictions_dir(
        self,
        category: Literal["datasets", "artifacts"] | str = "datasets",
    ) -> str:
        """Return canonical prediction evaluations directory by category."""
        cat = str(category).strip().lower() or "datasets"
        return os.path.join(self._data_root, "predictions", cat)

    def get_ticks_dir(self) -> str:
        """Return canonical tick market feed directory (<data_root>/ticks)."""
        return os.path.join(self._data_root, "ticks")

    def get_logs_dir(self) -> str:
        """Return canonical application and execution logs directory (<data_root>/logs)."""
        return os.path.join(self._data_root, "logs")

    def get_cache_dir(self) -> str:
        """Return canonical ephemeral cache directory (<data_root>/cache)."""
        return os.path.join(self._data_root, "cache")

    def get_all_canonical_dirs(self) -> list[str]:
        """Return a list of all canonical subdirectories defined in the Doc 17 hierarchy."""
        return [
            os.path.join(self._data_root, "databases"),
            os.path.join(self._data_root, "registries"),
            os.path.join(self._data_root, "datasets", "master"),
            os.path.join(self._data_root, "datasets", "analysis"),
            os.path.join(self._data_root, "datasets", "labels"),
            os.path.join(self._data_root, "datasets", "exports"),
            os.path.join(self._data_root, "models", "production"),
            os.path.join(self._data_root, "models", "candidates"),
            os.path.join(self._data_root, "models", "research"),
            os.path.join(self._data_root, "research", "campaigns"),
            os.path.join(self._data_root, "research", "discovery"),
            os.path.join(self._data_root, "research", "snapshots"),
            os.path.join(self._data_root, "research", "dossiers"),
            os.path.join(self._data_root, "predictions", "datasets"),
            os.path.join(self._data_root, "predictions", "artifacts"),
            os.path.join(self._data_root, "ticks"),
            os.path.join(self._data_root, "logs"),
            os.path.join(self._data_root, "cache"),
        ]

    def ensure_layout(self) -> list[str]:
        """Explicitly and idempotently create the canonical directory tree on disk."""
        created: list[str] = []
        for p in self.get_all_canonical_dirs():
            if not os.path.isdir(p):
                os.makedirs(p, exist_ok=True)
                created.append(p)
        return created

    def validate_layout(self) -> dict[str, Any]:
        """Perform a read-only health check of the Data Root and return diagnostic metadata."""
        root_exists = os.path.isdir(self._data_root)
        disk_free_gb = 0.0
        disk_total_gb = 0.0

        if root_exists:
            try:
                usage = shutil.disk_usage(self._data_root)
                disk_free_gb = round(usage.free / (1024 ** 3), 2)
                disk_total_gb = round(usage.total / (1024 ** 3), 2)
            except Exception:
                pass

        subdirs_status = {
            os.path.relpath(p, self._data_root): os.path.isdir(p)
            for p in self.get_all_canonical_dirs()
        }

        return {
            "data_root": self._data_root,
            "root_exists": root_exists,
            "disk_free_gb": disk_free_gb,
            "disk_total_gb": disk_total_gb,
            "subdirs": subdirs_status,
            "is_valid": root_exists or bool(os.path.splitdrive(self._data_root)[0]),
        }


def get_data_root_service(data_root: str | None = None) -> DataRootService:
    """Return the global DataRootService singleton or an instance for a custom root."""
    global _GLOBAL_DATA_ROOT_SERVICE
    if data_root is not None:
        return DataRootService(data_root)
    if _GLOBAL_DATA_ROOT_SERVICE is None:
        _GLOBAL_DATA_ROOT_SERVICE = DataRootService()
    return _GLOBAL_DATA_ROOT_SERVICE
