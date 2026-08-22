"""Safe Migration Assistant Service for Canonical Data Root (Doc 17, Phase 4)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .data_root import DataRootService, normalize_storage_path, save_data_root


def _sha256_file(filepath: str) -> str | None:
    if not os.path.isfile(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _verify_sqlite_integrity(filepath: str) -> str:
    if not os.path.isfile(filepath):
        return "missing"
    try:
        conn = sqlite3.connect(filepath, timeout=10.0)
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        res = cur.fetchone()[0]
        conn.close()
        return str(res)
    except Exception as e:
        return f"error: {e}"


@dataclass
class MigrationPlanItem:
    name: str
    category: str
    src_path: str
    dst_path: str
    size_bytes: int
    is_sqlite: bool = False
    is_dir_tree: bool = False
    status: str = "ready"  # ready | identical | conflict_diff | missing_src
    message: str = ""


@dataclass
class MigrationPlan:
    target_data_root: str
    source_dir: str
    items: list[MigrationPlanItem] = field(default_factory=list)
    total_bytes: int = 0
    ready_count: int = 0
    identical_count: int = 0
    conflict_count: int = 0
    disk_free_bytes: int = 0
    is_safe_to_execute: bool = True
    issues: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "target_data_root": self.target_data_root,
            "source_dir": self.source_dir,
            "total_items": len(self.items),
            "ready_count": self.ready_count,
            "identical_count": self.identical_count,
            "conflict_count": self.conflict_count,
            "total_size_mb": round(self.total_bytes / (1024 * 1024), 2),
            "disk_free_gb": round(self.disk_free_bytes / (1024 ** 3), 2),
            "is_safe": self.is_safe_to_execute,
            "issues": self.issues,
        }


class DataMigrationService:
    """Safe 5-stage migration workflow engine for consolidating data under DataRoot."""

    def __init__(self, target_data_root: str = r"D:\data") -> None:
        self.svc = DataRootService(target_data_root)
        self.target_data_root = self.svc.data_root

    def build_plan(self, source_dir: str | None = None) -> MigrationPlan:
        """Stage 1 & 2: Discover source files, detect conflicts, compute sizes and dry-run plan."""
        src_root = normalize_storage_path(source_dir or os.path.join(os.getcwd(), "data"))
        plan = MigrationPlan(
            target_data_root=self.target_data_root,
            source_dir=src_root,
        )

        try:
            usage = shutil.disk_usage(self.target_data_root if os.path.exists(self.target_data_root) else os.path.splitdrive(self.target_data_root)[0] or "C:")
            plan.disk_free_bytes = usage.free
        except Exception:
            plan.disk_free_bytes = 100 * (1024 ** 3)

        discovered: list[MigrationPlanItem] = []

        # 1. Databases
        db_candidates = [
            ("analysis.db", "databases", self.svc.get_database_path("analysis"), True),
            ("feature_recommendation_evidence.db", "databases", self.svc.get_database_path("feature_evidence"), True),
            ("angel_historic_bars.db", "databases", self.svc.get_database_path("angel_historic"), True),
            (os.path.join("prediction_runs", "registry.db"), "databases", self.svc.get_database_path("predictions"), True),
            (os.path.join("strategy_runs", "registry.db"), "databases", self.svc.get_database_path("strategies"), True),
        ]
        for rel_src, cat, dst_p, is_sql in db_candidates:
            src_p = os.path.join(src_root, rel_src)
            if os.path.isfile(src_p):
                sz = os.path.getsize(src_p)
                discovered.append(MigrationPlanItem(
                    name=os.path.basename(rel_src),
                    category=cat,
                    src_path=src_p,
                    dst_path=dst_p,
                    size_bytes=sz,
                    is_sqlite=is_sql,
                ))

        # 2. Registries
        reg_candidates = [
            ("feature_registry_store.json", "registries", self.svc.get_registry_path("feature"), False),
            ("pipeline_registry_store.json", "registries", self.svc.get_registry_path("pipeline"), False),
            (os.path.join("models", ".lifecycle_registry.db"), "registries", self.svc.get_registry_path("model"), True),
        ]
        for rel_src, cat, dst_p, is_sql in reg_candidates:
            src_p = os.path.join(src_root, rel_src)
            if os.path.isfile(src_p):
                sz = os.path.getsize(src_p)
                discovered.append(MigrationPlanItem(
                    name=os.path.basename(rel_src),
                    category=cat,
                    src_path=src_p,
                    dst_path=dst_p,
                    size_bytes=sz,
                    is_sqlite=is_sql,
                ))

        # 3. Datasets Analysis Parquets
        src_datasets = os.path.join(src_root, "datasets")
        if os.path.isdir(src_datasets):
            for fname in os.listdir(src_datasets):
                sf = os.path.join(src_datasets, fname)
                if os.path.isfile(sf):
                    df = os.path.join(self.svc.get_datasets_dir("analysis"), fname)
                    discovered.append(MigrationPlanItem(
                        name=fname,
                        category="datasets/analysis",
                        src_path=sf,
                        dst_path=df,
                        size_bytes=os.path.getsize(sf),
                    ))

        # 4. Labels
        src_labels = os.path.join(src_root, "label_runs")
        if os.path.isdir(src_labels):
            for fname in os.listdir(src_labels):
                sf = os.path.join(src_labels, fname)
                if os.path.isfile(sf):
                    df = os.path.join(self.svc.get_datasets_dir("labels"), fname)
                    discovered.append(MigrationPlanItem(
                        name=fname,
                        category="datasets/labels",
                        src_path=sf,
                        dst_path=df,
                        size_bytes=os.path.getsize(sf),
                    ))

        # 5. Exports
        src_exports = os.path.join(src_root, "chain_exports")
        if os.path.isdir(src_exports):
            for fname in os.listdir(src_exports):
                sf = os.path.join(src_exports, fname)
                if os.path.isfile(sf):
                    df = os.path.join(self.svc.get_datasets_dir("exports"), fname)
                    discovered.append(MigrationPlanItem(
                        name=fname,
                        category="datasets/exports",
                        src_path=sf,
                        dst_path=df,
                        size_bytes=os.path.getsize(sf),
                    ))

        # Evaluate status and conflicts for each item
        total_sz = 0
        ready_cnt = 0
        ident_cnt = 0
        conf_cnt = 0

        for item in discovered:
            total_sz += item.size_bytes
            if not os.path.exists(item.dst_path):
                item.status = "ready"
                item.message = "Destination is clear; ready to migrate."
                ready_cnt += 1
            else:
                dst_sz = os.path.getsize(item.dst_path)
                if dst_sz == item.size_bytes:
                    src_hash = _sha256_file(item.src_path)
                    dst_hash = _sha256_file(item.dst_path)
                    if src_hash == dst_hash:
                        item.status = "identical"
                        item.message = "Already present and verified in canonical location."
                        ident_cnt += 1
                    else:
                        item.status = "conflict_diff"
                        item.message = f"Conflict: Same size ({dst_sz} bytes) but hash mismatch."
                        conf_cnt += 1
                else:
                    item.status = "conflict_diff"
                    item.message = f"Conflict: Size differs (src: {item.size_bytes} vs dst: {dst_sz})."
                    conf_cnt += 1

        plan.items = discovered
        plan.total_bytes = total_sz
        plan.ready_count = ready_cnt
        plan.identical_count = ident_cnt
        plan.conflict_count = conf_cnt

        if plan.total_bytes > plan.disk_free_bytes:
            plan.is_safe_to_execute = False
            plan.issues.append(f"Insufficient disk space on target: Need {plan.total_bytes / (1024*1024):.1f} MB, Free: {plan.disk_free_bytes / (1024*1024):.1f} MB")

        if conf_cnt > 0:
            plan.issues.append(f"{conf_cnt} conflicting file(s) detected at destination. Conflicts will not be silently overwritten.")

        return plan

    def execute_migration(
        self,
        plan: MigrationPlan,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Stage 3, 4, 5: Provision layout, copy files, verify hashes, record manifest, and update pointer."""
        if not plan.is_safe_to_execute:
            return {
                "success": False,
                "message": "Migration blocked: " + "; ".join(plan.issues),
                "migrated_count": 0,
                "errors": plan.issues,
            }

        # Stage 2: Provision layout
        self.svc.ensure_layout()

        manifest_items: list[dict[str, Any]] = []
        errors: list[str] = []
        total_steps = len(plan.items)
        copied_cnt = 0

        for idx, item in enumerate(plan.items, start=1):
            if on_progress:
                on_progress(f"Migrating {item.name} ({idx}/{total_steps})…", idx, total_steps)

            if item.status == "identical":
                manifest_items.append({
                    "name": item.name,
                    "category": item.category,
                    "src": item.src_path,
                    "dst": item.dst_path,
                    "size_bytes": item.size_bytes,
                    "status": "already_verified",
                })
                continue

            if item.status == "conflict_diff":
                errors.append(f"Skipping conflicting item: {item.name}")
                continue

            try:
                os.makedirs(os.path.dirname(item.dst_path), exist_ok=True)
                shutil.copy2(item.src_path, item.dst_path)

                # Post-copy integrity check
                dst_sz = os.path.getsize(item.dst_path)
                assert dst_sz == item.size_bytes, f"Size mismatch on {item.name}: {item.size_bytes} != {dst_sz}"

                src_sha = _sha256_file(item.src_path)
                dst_sha = _sha256_file(item.dst_path)
                assert src_sha == dst_sha, f"SHA-256 mismatch on {item.name}"

                if item.is_sqlite:
                    chk = _verify_sqlite_integrity(item.dst_path)
                    assert chk == "ok", f"SQLite integrity failed for {item.dst_path}: {chk}"
                else:
                    chk = "valid"

                copied_cnt += 1
                manifest_items.append({
                    "name": item.name,
                    "category": item.category,
                    "src": item.src_path,
                    "dst": item.dst_path,
                    "size_bytes": dst_sz,
                    "sha256": dst_sha,
                    "integrity": chk,
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                errors.append(f"Failed to migrate {item.name}: {e}")

        # Stage 4: Record Manifest
        manifest_path = os.path.join(self.target_data_root, "migration_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_dir": plan.source_dir,
                    "target_data_root": self.target_data_root,
                    "total_items_processed": len(manifest_items),
                    "items": manifest_items,
                    "errors": errors,
                }, fh, indent=2)
        except Exception:
            pass

        # Stage 5: Switch Data Root setting atomically
        save_data_root(self.target_data_root)

        return {
            "success": len(errors) == 0,
            "message": f"Migration complete: {copied_cnt} files copied, {plan.identical_count} verified identical." if len(errors) == 0 else f"Completed with {len(errors)} issues.",
            "copied_count": copied_cnt,
            "verified_count": len(manifest_items),
            "manifest_path": manifest_path,
            "errors": errors,
        }
