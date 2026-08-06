"""Migration runner with version tracking, history, and rollback."""

from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from feature_intelligence.core.logging import get_logger
from feature_intelligence.core.paths import MIGRATION_VERSIONS_DIR

logger = get_logger("migrations")


class MigrationModule(Protocol):
    version: str
    description: str

    def upgrade(self, conn: sqlite3.Connection) -> None: ...

    def downgrade(self, conn: sqlite3.Connection) -> None: ...


@dataclass(frozen=True)
class MigrationInfo:
    version: str
    description: str
    path: Path


_META_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS schema_migration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('upgrade', 'downgrade')),
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


class MigrationRunner:
    """Apply and roll back ordered SQL/Python migrations."""

    def __init__(
        self,
        db_path: Path,
        *,
        versions_dir: Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.versions_dir = Path(versions_dir or MIGRATION_VERSIONS_DIR)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_meta(self, conn: sqlite3.Connection | None = None) -> None:
        owns = conn is None
        conn = conn or self._connect()
        try:
            conn.executescript(_META_SQL)
            conn.commit()
        finally:
            if owns:
                conn.close()

    def discover(self) -> list[MigrationInfo]:
        if not self.versions_dir.is_dir():
            return []
        found: list[MigrationInfo] = []
        for path in sorted(self.versions_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            mod = self._load_module(path)
            found.append(
                MigrationInfo(
                    version=str(mod.version),
                    description=str(mod.description),
                    path=path,
                )
            )
        found.sort(key=lambda m: m.version)
        return found

    def _load_module(self, path: Path) -> MigrationModule:
        spec = importlib.util.spec_from_file_location(
            f"feature_intelligence.migrations.versions.{path.stem}",
            path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load migration: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in ("version", "description", "upgrade", "downgrade"):
            if not hasattr(module, attr):
                raise AttributeError(f"Migration {path.name} missing {attr}")
        return module  # type: ignore[return-value]

    def current_version(self) -> str | None:
        conn = self._connect()
        try:
            self.ensure_meta(conn)
            row = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return None if row is None else str(row["version"])
        finally:
            conn.close()

    def applied_versions(self) -> list[str]:
        conn = self._connect()
        try:
            self.ensure_meta(conn)
            rows = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version ASC"
            ).fetchall()
            return [str(r["version"]) for r in rows]
        finally:
            conn.close()

    def history(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            self.ensure_meta(conn)
            rows = conn.execute(
                """
                SELECT id, version, action, description, applied_at
                FROM schema_migration_history
                ORDER BY id ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def upgrade(self, target: str | None = None) -> list[str]:
        """Apply pending migrations up to ``target`` (inclusive)."""
        applied: list[str] = []
        conn = self._connect()
        try:
            self.ensure_meta(conn)
            done = {
                str(r["version"])
                for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for info in self.discover():
                if info.version in done:
                    continue
                if target is not None and info.version > target:
                    break
                mod = self._load_module(info.path)
                logger.info("Applying migration %s — %s", info.version, info.description)
                mod.upgrade(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version, description) VALUES (?, ?)",
                    (info.version, info.description),
                )
                conn.execute(
                    """
                    INSERT INTO schema_migration_history(version, action, description)
                    VALUES (?, 'upgrade', ?)
                    """,
                    (info.version, info.description),
                )
                conn.commit()
                applied.append(info.version)
        finally:
            conn.close()
        return applied

    def downgrade(self, steps: int = 1) -> list[str]:
        """Roll back the last ``steps`` applied migrations."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        rolled: list[str] = []
        conn = self._connect()
        try:
            self.ensure_meta(conn)
            rows = conn.execute(
                "SELECT version, description FROM schema_migrations ORDER BY version DESC"
            ).fetchall()
            to_roll = rows[:steps]
            by_version = {m.version: m for m in self.discover()}
            for row in to_roll:
                version = str(row["version"])
                info = by_version.get(version)
                if info is None:
                    raise RuntimeError(f"Migration file missing for version {version}")
                mod = self._load_module(info.path)
                logger.info("Rolling back migration %s", version)
                mod.downgrade(conn)
                conn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?",
                    (version,),
                )
                conn.execute(
                    """
                    INSERT INTO schema_migration_history(version, action, description)
                    VALUES (?, 'downgrade', ?)
                    """,
                    (version, str(row["description"])),
                )
                conn.commit()
                rolled.append(version)
        finally:
            conn.close()
        return rolled
