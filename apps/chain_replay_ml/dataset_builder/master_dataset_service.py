"""Master dataset read API — metadata tables only for dashboard stats."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from .master_fingerprint import normalize_metadata_status
from .master_naming import path_relative_to_data_dir, resolve_master_db_path
from .master_store import MasterStore


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path) if os.path.isfile(path) else 0
    except OSError:
        return 0


def _read_table_info(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [
        {
            "name": str(r[1]),
            "type": str(r[2] or ""),
            "notnull": bool(r[3]),
            "pk": bool(r[5]),
        }
        for r in rows
    ]


class MasterDatasetService:
    """Single read layer for master dataset dashboard APIs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)

    @classmethod
    def for_market(
        cls,
        data_dir: str,
        *,
        market: str,
        interval_sec: int,
        master_db_path: str | None = None,
    ) -> MasterDatasetService:
        path = master_db_path or resolve_master_db_path(
            data_dir,
            market=market,
            sampling_interval_sec=interval_sec,
        )
        return cls(path)

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.db_path)

    def refresh_metadata(self) -> dict[str, Any]:
        store = MasterStore(self.db_path)
        store.open()
        try:
            meta = store.refresh_metadata_from_samples(reason="BACKFILL")
            return {
                "metadata_version": meta.metadata_version,
                "total_rows": meta.total_rows,
                "total_days": meta.total_days,
                "metadata_status": normalize_metadata_status(meta.metadata_status),
            }
        finally:
            store.close()

    def read_status(
        self,
        *,
        market: str,
        interval_sec: int,
        data_dir: str,
    ) -> dict[str, Any]:
        rel = path_relative_to_data_dir(self.db_path, data_dir)
        out: dict[str, Any] = {
            "market": str(market or "NIFTY").upper(),
            "interval_sec": int(interval_sec),
            "master_db_path": rel,
            "master_db_abs": self.db_path,
            "exists": self.exists,
            "row_count": 0,
            "trading_days": [],
            "days_in_master": [],
            "row_counts_by_day": {},
            "coverage_by_day": {},
            "feature_count": 0,
            "target_count": 0,
            "builder_progress": None,
            "metadata_version": 0,
            "metadata_status": "VALID",
            "distributions": [],
            "dataset_fingerprint": None,
        }
        try:
            from .master_defaults import default_master_column_counts

            fc, tc = default_master_column_counts()
            out["feature_count"] = fc
            out["target_count"] = tc
        except Exception:
            pass

        if not self.exists:
            return out

        store = MasterStore(self.db_path)
        store.open()
        try:
            meta_dict = store.read_master_meta_dict()
            meta = store.read_master_meta()
            days = store.read_master_days()
            prog = store.read_builder_progress()
            day_keys = sorted({d["trading_day"] for d in days})
            out["row_count"] = int(meta.total_rows)
            out["trading_days"] = day_keys
            out["days_in_master"] = day_keys
            out["row_counts_by_day"] = {
                d["trading_day"]: int(d["row_count"]) for d in days
            }
            out["metadata_version"] = int(meta.metadata_version or 0)
            out["metadata_status"] = normalize_metadata_status(meta.metadata_status)
            out["dataset_fingerprint"] = store.read_dataset_fingerprint()
            out["distributions"] = store.read_master_distributions()
            if meta.feature_count is not None:
                out["feature_count"] = int(meta.feature_count)
            if meta.target_count is not None:
                out["target_count"] = int(meta.target_count)
            out["builder_progress"] = {
                "last_completed_day": prog.last_completed_day,
                "current_day": prog.current_day,
                "status": prog.status,
                "days_total": prog.days_total,
                "days_done": prog.days_done,
                "error_message": prog.error_message,
            }
            cfg = store.get_meta("master_config")
            if isinstance(cfg, dict):
                out["master_config"] = cfg
            fpid = store.read_master_meta_dict().get("feature_project_id")
            if fpid is not None and str(fpid).strip():
                out["feature_project_id"] = str(fpid).strip().lower()
            elif isinstance(cfg, dict) and cfg.get("feature_project_id"):
                out["feature_project_id"] = str(cfg.get("feature_project_id")).strip().lower()
            out["coverage_by_day"] = store.get_coverage_by_day()
            schema_meta = store.get_meta("build_schema")
            if isinstance(schema_meta, dict):
                out["build_schema"] = schema_meta
                if schema_meta.get("feature_count") is not None:
                    out["feature_count"] = int(schema_meta["feature_count"])
                if schema_meta.get("target_count") is not None:
                    out["target_count"] = int(schema_meta["target_count"])
            fp = store.get_meta("feature_policy")
            if isinstance(fp, dict):
                out["feature_policy"] = fp
                try:
                    from chain_replay_ml.feature_policy import build_policy_report

                    out["feature_policy_report"] = build_policy_report(fp)
                except Exception:
                    pass
            out["master_meta"] = meta_dict
        finally:
            store.close()
        return out

    def read_fingerprint(self) -> dict[str, Any]:
        if not self.exists:
            return {}
        store = MasterStore(self.db_path)
        store.open()
        try:
            return store.read_dataset_fingerprint()
        finally:
            store.close()

    def read_distributions(self) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        store = MasterStore(self.db_path)
        store.open()
        try:
            return store.read_master_distributions()
        finally:
            store.close()

    def read_day_details(self, coverage_by_day: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        store = MasterStore(self.db_path)
        store.open()
        try:
            days = store.read_master_days()
        finally:
            store.close()
        coverage = coverage_by_day or {}
        out: list[dict[str, Any]] = []
        for day in days:
            td = day["trading_day"]
            cov = coverage.get(td) if isinstance(coverage, dict) else None
            if cov is None and day.get("coverage_percent") is not None:
                cov = {"coverage_pct": day["coverage_percent"]}
            if cov is None and day.get("rejected_rows") is not None:
                cov = {"rejected_samples": day["rejected_rows"]}
            out.append({
                "trading_day": td,
                "row_count": int(day.get("row_count") or 0),
                "token_count": day.get("token_count"),
                "expiry_count": day.get("expiry_count"),
                "dominant_expiry": day.get("dominant_expiry"),
                "is_expiry_day": day.get("is_expiry_day"),
                "option_type_count": None,
                "timestamp_min": day.get("first_timestamp"),
                "timestamp_max": day.get("last_timestamp"),
                "coverage": cov,
                "status": day.get("status"),
            })
        return out

    def read_sqlite_tables(self, *, include_row_counts: bool = True) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            table_names = [
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                if not str(r[0]).startswith("sqlite_")
            ]
            meta_counts: dict[str, int | None] = {}
            if not include_row_counts:
                meta_counts["samples"] = None
            else:
                store = MasterStore(self.db_path)
                store.open()
                try:
                    meta = store.read_master_meta()
                    meta_counts["samples"] = int(meta.total_rows)
                    for row in store.read_master_days():
                        pass
                    day_count = store.count_metadata_days()
                    meta_counts["master_dataset_days"] = day_count
                    meta_counts["master_dataset_meta"] = 1
                    meta_counts["master_dataset_meta_history"] = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM master_dataset_meta_history"
                        ).fetchone()[0]
                    )
                finally:
                    store.close()
            tables: list[dict[str, Any]] = []
            for table in table_names:
                row_count: int | None
                if table in meta_counts:
                    row_count = meta_counts[table]
                elif include_row_counts:
                    count_row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
                    row_count = int(count_row[0]) if count_row else 0
                else:
                    row_count = None
                tables.append({
                    "name": table,
                    "row_count": row_count,
                    "columns": _read_table_info(conn, table),
                })
            return tables
        finally:
            conn.close()
