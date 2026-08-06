"""Centralized configuration loader for Feature Intelligence Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _yaml_lite
from .paths import CONFIG_DIR, default_db_path


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path
    timeout_seconds: float = 30.0
    journal_mode: str = "WAL"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    to_file: bool = False
    file_path: Path | None = None


@dataclass(frozen=True)
class FeatureIntelligenceConfig:
    schema_version: str = "0.0.1"
    environment: str = "development"
    data_dir: Path | None = None


@dataclass(frozen=True)
class FicConfig:
    feature_intelligence: FeatureIntelligenceConfig = field(
        default_factory=FeatureIntelligenceConfig
    )
    database: DatabaseConfig = field(
        default_factory=lambda: DatabaseConfig(path=default_db_path())
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def db_path(self) -> Path:
        return self.database.path


def _as_path(value: Any, *, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("path value is required")
        return default
    return Path(str(value)).expanduser()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = _yaml_lite.load(path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(config_dir: Path | None = None) -> FicConfig:
    """Load FIC YAML configs from ``config_dir`` (defaults to package config/)."""
    root = config_dir or CONFIG_DIR
    fi_raw = _load_yaml(root / "feature_intelligence.yaml")
    db_raw = _load_yaml(root / "database.yaml")
    log_raw = _load_yaml(root / "logging.yaml")

    data_dir = _as_path(
        fi_raw.get("data_dir"),
        default=default_db_path().parent,
    )
    db_path = _as_path(
        db_raw.get("path"),
        default=data_dir / "feature_intelligence.db",
    )

    log_file = log_raw.get("file_path")
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "INFO")).upper(),
        format=str(
            log_raw.get(
                "format",
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            )
        ),
        to_file=bool(log_raw.get("to_file", False)),
        file_path=_as_path(log_file, default=None) if log_file else None,
    )

    return FicConfig(
        feature_intelligence=FeatureIntelligenceConfig(
            schema_version=str(fi_raw.get("schema_version", "0.0.1")),
            environment=str(fi_raw.get("environment", "development")),
            data_dir=data_dir,
        ),
        database=DatabaseConfig(
            path=db_path,
            timeout_seconds=float(db_raw.get("timeout_seconds", 30.0)),
            journal_mode=str(db_raw.get("journal_mode", "WAL")),
        ),
        logging=logging_cfg,
    )
