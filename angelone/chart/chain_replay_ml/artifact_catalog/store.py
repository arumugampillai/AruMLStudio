"""SQLite Artifact Catalog store — register / get / list / DAG parents."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from .paths import catalog_db_path
from .types import ArtifactRecord
from .uri import ArtifactUriError, is_artifact_uri, parse_uri


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactCatalogError(ValueError):
    pass


class ArtifactCatalogStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = catalog_db_path(data_dir)
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

    def __enter__(self) -> ArtifactCatalogStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ArtifactCatalogStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_uri TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                local_path TEXT,
                parent_uris_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                capabilities_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'completed'
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_type
                ON artifacts(artifact_type);
            CREATE INDEX IF NOT EXISTS idx_artifacts_created
                ON artifacts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_status
                ON artifacts(status);
            """
        )
        self.conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_uri=str(row["artifact_uri"]),
            artifact_type=str(row["artifact_type"]),
            created_at=str(row["created_at"]),
            local_path=row["local_path"],
            parent_artifact_uris=list(json.loads(row["parent_uris_json"] or "[]")),
            metadata=dict(json.loads(row["metadata_json"] or "{}")),
            capabilities=list(json.loads(row["capabilities_json"] or "[]")),
            status=str(row["status"] or "completed"),
        )

    def would_create_cycle(
        self,
        artifact_uri: str,
        parent_uris: Iterable[str],
    ) -> bool:
        """True if adding parents would introduce a cycle (ancestor of self)."""
        parents = [str(p) for p in parent_uris if p]
        if artifact_uri in parents:
            return True
        # Walk ancestors of each parent; if we reach artifact_uri, cycle.
        stack = list(parents)
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur == artifact_uri:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            rec = self.get(cur)
            if rec:
                stack.extend(rec.parent_artifact_uris)
        return False

    def register(
        self,
        record: ArtifactRecord,
        *,
        replace: bool = True,
    ) -> ArtifactRecord:
        if not is_artifact_uri(record.artifact_uri):
            raise ArtifactUriError(f"invalid artifact_uri: {record.artifact_uri!r}")
        family, _ = parse_uri(record.artifact_uri)
        # Soft-check type family alignment (allow feature_studio URIs with that type).
        if record.artifact_type and record.artifact_type not in (
            family,
            "feature_studio",
            "other",
        ):
            # Still allow; metadata may diverge slightly (e.g. type=model, family=model).
            pass
        for p in record.parent_artifact_uris:
            if not is_artifact_uri(p):
                raise ArtifactUriError(f"invalid parent URI: {p!r}")
        if self.would_create_cycle(record.artifact_uri, record.parent_artifact_uris):
            raise ArtifactCatalogError(
                f"parent_artifact_uris would create a cycle for {record.artifact_uri}"
            )
        existing = self.get(record.artifact_uri)
        if existing is not None and not replace:
            raise ArtifactCatalogError(f"artifact already registered: {record.artifact_uri}")
        created = record.created_at or _utc_now()
        self.conn.execute(
            """
            INSERT INTO artifacts (
                artifact_uri, artifact_type, created_at, local_path,
                parent_uris_json, metadata_json, capabilities_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_uri) DO UPDATE SET
                artifact_type=excluded.artifact_type,
                created_at=excluded.created_at,
                local_path=excluded.local_path,
                parent_uris_json=excluded.parent_uris_json,
                metadata_json=excluded.metadata_json,
                capabilities_json=excluded.capabilities_json,
                status=excluded.status
            """,
            (
                record.artifact_uri,
                record.artifact_type,
                created,
                record.local_path,
                json.dumps(list(record.parent_artifact_uris)),
                json.dumps(dict(record.metadata)),
                json.dumps(list(record.capabilities)),
                record.status or "completed",
            ),
        )
        self.conn.commit()
        out = self.get(record.artifact_uri)
        assert out is not None
        return out

    def get(self, artifact_uri: str) -> ArtifactRecord | None:
        row = self.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_uri = ?",
            (str(artifact_uri),),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self) -> list[ArtifactRecord]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts ORDER BY created_at ASC, artifact_uri ASC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_type(self, artifact_type: str) -> list[ArtifactRecord]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_type = ? ORDER BY created_at ASC",
            (str(artifact_type),),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_capability(self, capability: str) -> list[ArtifactRecord]:
        """Future-facing filter; works once capabilities are populated."""
        want = str(capability)
        return [r for r in self.list_all() if want in r.capabilities]

    def children_of(self, parent_uri: str) -> list[ArtifactRecord]:
        parent = str(parent_uri)
        return [r for r in self.list_all() if parent in r.parent_artifact_uris]

    def update_status(self, artifact_uri: str, status: str) -> ArtifactRecord:
        rec = self.get(artifact_uri)
        if rec is None:
            raise ArtifactCatalogError(f"unknown artifact: {artifact_uri}")
        updated = ArtifactRecord(
            artifact_uri=rec.artifact_uri,
            artifact_type=rec.artifact_type,
            created_at=rec.created_at,
            local_path=rec.local_path,
            parent_artifact_uris=list(rec.parent_artifact_uris),
            metadata=dict(rec.metadata),
            capabilities=list(rec.capabilities),
            status=str(status),
        )
        return self.register(updated, replace=True)
