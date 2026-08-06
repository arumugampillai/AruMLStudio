"""SQLite database initialization for Feature Intelligence Core."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import FicConfig, load_config
from .logging import get_logger

logger = get_logger("core.database")


class Database:
    """Thin SQLite connection wrapper used by migrations and infrastructure."""

    def __init__(self, path: Path, *, timeout_seconds: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.timeout_seconds)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn
        return conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def executescript(self, sql: str) -> None:
        self.connect().executescript(sql)

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.commit()
        self.close()


def init_database(
    config: FicConfig | None = None,
    *,
    apply_migrations: bool = True,
) -> Path:
    """Create ``feature_intelligence.db`` and optionally apply migrations.

    Returns the database path.
    """
    cfg = config or load_config()
    db = Database(cfg.database.path, timeout_seconds=cfg.database.timeout_seconds)
    try:
        conn = db.connect()
        # Avoid WAL lock issues on Windows temp dirs during tests/CI.
        mode = cfg.database.journal_mode
        conn.execute(f"PRAGMA journal_mode={mode}")
        logger.info("Initialized database at %s", cfg.database.path)
        db.commit()
    finally:
        db.close()

    if apply_migrations:
        from feature_intelligence.migrations.runner import MigrationRunner

        runner = MigrationRunner(cfg.database.path)
        runner.upgrade()

    return cfg.database.path
