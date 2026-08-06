"""Public types for Production Validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnseenDatasetResolveResult:
    """Outcome of resolve-or-create for an ``unseen_*`` registry dataset."""

    ok: bool
    model_name: str
    dataset_name: str | None = None
    parquet_path: str | None = None
    json_path: str | None = None
    reused: bool = False
    created: bool = False
    seen_days: list[str] = field(default_factory=list)
    unseen_days: list[str] = field(default_factory=list)
    master_db_path: str | None = None
    identity_hash: str | None = None
    message: str = ""
    error: str | None = None
    status: str = "pending"  # pending | ready | empty | error
    compute_note: str = "compute coming"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "parquet_path": self.parquet_path,
            "json_path": self.json_path,
            "reused": self.reused,
            "created": self.created,
            "seen_day_count": len(self.seen_days),
            "unseen_day_count": len(self.unseen_days),
            "seen_days": list(self.seen_days),
            "unseen_days": list(self.unseen_days),
            "master_db_path": self.master_db_path,
            "identity_hash": self.identity_hash,
            "message": self.message,
            "error": self.error,
            "status": self.status,
            "compute_note": self.compute_note,
        }


@dataclass
class ProductionValidationResult:
    """Holdout → Unseen rank-based importance compare result."""

    ok: bool
    model_name: str
    package_dir: str
    artifacts_dir: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    unseen_status: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_name": self.model_name,
            "package_dir": self.package_dir,
            "artifacts_dir": self.artifacts_dir,
            "row_count": len(self.rows),
            "rows": list(self.rows),
            "summary": dict(self.summary),
            "meta": dict(self.meta),
            "unseen_status": dict(self.unseen_status),
            "error": self.error,
        }
