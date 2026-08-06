"""SQLite store for strategy profiles and immutable versions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .paths import strategy_registry_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyRegistryStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = strategy_registry_db_path(data_dir)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StrategyRegistryStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StrategyRegistryStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS strategy_profiles (
                strategy_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT,
                slug TEXT,
                current_version_id TEXT,
                current_version_label TEXT,
                current_version_number INTEGER NOT NULL DEFAULT 0,
                champion_config_hash TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_on TEXT NOT NULL,
                updated_on TEXT NOT NULL,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_strategy_profiles_slug
                ON strategy_profiles(slug);
            CREATE INDEX IF NOT EXISTS idx_strategy_profiles_updated
                ON strategy_profiles(updated_on DESC);

            CREATE TABLE IF NOT EXISTS strategy_versions (
                version_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                version_label TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                parent_version_id TEXT,
                lifecycle TEXT NOT NULL DEFAULT 'new_strategy',
                status TEXT NOT NULL DEFAULT 'active',
                config_hash TEXT NOT NULL,
                display_name TEXT,
                description TEXT,
                created_on TEXT NOT NULL,
                updated_on TEXT NOT NULL,
                config_json TEXT NOT NULL,
                meta_json TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategy_profiles(strategy_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_versions_family_num
                ON strategy_versions(strategy_id, version_number);
            CREATE INDEX IF NOT EXISTS idx_strategy_versions_hash
                ON strategy_versions(config_hash);
            """
        )

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key in ("meta_json", "config_json"):
            if d.get(key):
                try:
                    parsed = json.loads(d[key])
                    if key == "meta_json":
                        d["meta"] = parsed
                    else:
                        d["config"] = parsed
                except json.JSONDecodeError:
                    if key == "meta_json":
                        d["meta"] = {}
                    else:
                        d["config"] = {}
                del d[key]
            elif key == "meta_json":
                d["meta"] = {}
        return d

    def insert_profile(self, doc: dict[str, Any]) -> dict[str, Any]:
        strategy_id = str(doc.get("strategy_id") or uuid.uuid4().hex)
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO strategy_profiles (
                strategy_id, display_name, description, slug,
                current_version_id, current_version_label, current_version_number,
                champion_config_hash, status, created_on, updated_on, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                doc["display_name"],
                doc.get("description"),
                doc.get("slug"),
                doc.get("current_version_id"),
                doc.get("current_version_label"),
                int(doc.get("current_version_number") or 0),
                doc.get("champion_config_hash"),
                doc.get("status") or "active",
                now,
                now,
                json.dumps(doc.get("meta") or {}, default=str),
            ),
        )
        self.conn.commit()
        return self.get_profile(strategy_id) or {"strategy_id": strategy_id}

    def update_profile_champion(
        self,
        strategy_id: str,
        *,
        version_id: str,
        version_label: str,
        version_number: int,
        config_hash: str,
        display_name: str | None = None,
        description: str | None = None,
    ) -> None:
        now = _utc_now()
        if display_name is not None and description is not None:
            self.conn.execute(
                """
                UPDATE strategy_profiles
                SET current_version_id = ?, current_version_label = ?,
                    current_version_number = ?, champion_config_hash = ?,
                    display_name = ?, description = ?, updated_on = ?
                WHERE strategy_id = ?
                """,
                (version_id, version_label, version_number, config_hash, display_name, description, now, strategy_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE strategy_profiles
                SET current_version_id = ?, current_version_label = ?,
                    current_version_number = ?, champion_config_hash = ?, updated_on = ?
                WHERE strategy_id = ?
                """,
                (version_id, version_label, version_number, config_hash, now, strategy_id),
            )
        self.conn.commit()

    def set_profile_status(self, strategy_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE strategy_profiles SET status = ?, updated_on = ? WHERE strategy_id = ?",
            (status, _utc_now(), strategy_id),
        )
        self.conn.commit()

    def insert_version(self, doc: dict[str, Any]) -> dict[str, Any]:
        version_id = str(doc.get("version_id") or uuid.uuid4().hex)
        now = _utc_now()
        config = doc.get("config") or {}
        self.conn.execute(
            """
            INSERT INTO strategy_versions (
                version_id, strategy_id, version_label, version_number,
                parent_version_id, lifecycle, status, config_hash,
                display_name, description, created_on, updated_on,
                config_json, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                doc["strategy_id"],
                doc["version_label"],
                int(doc["version_number"]),
                doc.get("parent_version_id"),
                doc.get("lifecycle") or "new_strategy",
                doc.get("status") or "active",
                doc["config_hash"],
                doc.get("display_name"),
                doc.get("description"),
                now,
                now,
                json.dumps(config, default=str),
                json.dumps(doc.get("meta") or {}, default=str),
            ),
        )
        self.conn.commit()
        return self.get_version(version_id) or {"version_id": version_id}

    def get_profile(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategy_profiles WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def get_profile_by_slug(self, slug: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategy_profiles WHERE slug = ?", (slug,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_profiles(self, *, limit: int = 100, include_archived: bool = False) -> list[dict[str, Any]]:
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM strategy_profiles ORDER BY updated_on DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM strategy_profiles
                WHERE status != 'archived'
                ORDER BY updated_on DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategy_versions WHERE version_id = ?", (version_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_versions(self, strategy_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM strategy_versions
            WHERE strategy_id = ?
            ORDER BY version_number DESC
            """,
            (strategy_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def next_version_number(self, strategy_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(version_number) FROM strategy_versions WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        return int(row[0] or 0) + 1

    def find_version_by_hash(self, strategy_id: str, config_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM strategy_versions
            WHERE strategy_id = ? AND config_hash = ?
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (strategy_id, config_hash),
        ).fetchone()
        return self._row_to_dict(row)
