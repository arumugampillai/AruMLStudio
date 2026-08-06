"""Model Lab SQLite store — Phase 1+2 schema (lab meta + prediction dataset)."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .prediction_schema import (
    CORE_COLUMN_NAMES,
    DATASET_TYPE_SEEN,
    DAY_CANCELLED,
    DAY_COMPLETED,
    DAY_FAILED,
    DAY_RUNNING,
    DAY_SKIPPED,
    DAY_WAITING,
    PRED_STATUS_NOT_GENERATED,
    create_prediction_dataset_sql,
    create_prediction_day_metadata_sql,
    create_prediction_summary_sql,
    create_research_dashboard_sql,
    normalize_dataset_type,
    resolve_day_completion_status,
)

LAB_SCHEMA_VERSION = 1
LAB_PHASE = 1

STATUS_CREATED = "CREATED"
STATUS_READY = "READY"
STATUS_ARCHIVED = "ARCHIVED"
STATUS_ERROR = "ERROR"

_STATUS_ALIASES = {
    "ready": STATUS_READY,
    "created": STATUS_CREATED,
    "archived": STATUS_ARCHIVED,
    "error": STATUS_ERROR,
}


def normalize_lab_status(status: str | None) -> str:
    raw = str(status or STATUS_READY).strip()
    if raw.upper() in {STATUS_CREATED, STATUS_READY, STATUS_ARCHIVED, STATUS_ERROR}:
        return raw.upper()
    return _STATUS_ALIASES.get(raw.lower(), STATUS_READY)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


@dataclass
class ModelLabInfo:
    lab_uuid: str
    lab_id: str
    lab_name: str
    parent_model_id: str
    parent_model_name: str
    model_checksum: str | None
    description: str | None
    purpose: str | None
    created_at: str
    created_by: str | None
    status: str
    version: int
    lab_schema_version: int
    phase: int
    original_feature_count: int | None
    selected_feature_count: int | None
    training_rows: int | None
    target: str | None
    algorithm: str | None
    dataset_snapshot: dict[str, Any] | None
    model_snapshot: dict[str, Any] | None
    training_config_snapshot: dict[str, Any] | None
    wf_snapshot: dict[str, Any] | None
    metrics_snapshot: dict[str, Any] | None
    selected_features_snapshot: list[str] | None
    feature_ranking_snapshot: dict[str, Any] | None
    artifact_pointers: dict[str, Any] | None
    db_path: str = ""

    def to_overview_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "lab_uuid": self.lab_uuid,
            "lab_id": self.lab_id,
            "lab_name": self.lab_name,
            "parent_model_id": self.parent_model_id,
            "parent_model_name": self.parent_model_name,
            "model_checksum": self.model_checksum,
            "description": self.description,
            "purpose": self.purpose,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "status": self.status,
            "version": self.version,
            "lab_schema_version": self.lab_schema_version,
            "phase": self.phase,
            "original_feature_count": self.original_feature_count,
            "selected_feature_count": self.selected_feature_count,
            "training_rows": self.training_rows,
            "target": self.target,
            "algorithm": self.algorithm,
            "dataset": (self.dataset_snapshot or {}).get("dataset_name")
            or (self.dataset_snapshot or {}).get("name"),
            "training_metrics": (self.metrics_snapshot or {}).get("training"),
            "holdout_metrics": (self.metrics_snapshot or {}).get("holdout")
            or (self.metrics_snapshot or {}).get("test"),
            "walk_forward_metrics": (self.metrics_snapshot or {}).get("walk_forward")
            or (self.metrics_snapshot or {}).get("production_walk_forward"),
            "selected_features": self.selected_features_snapshot or [],
            "feature_ranking": self.feature_ranking_snapshot,
            "artifact_pointers": self.artifact_pointers or {},
            "db_path": self.db_path,
            "prediction_dataset_status": PRED_STATUS_NOT_GENERATED,
        }
        extra = getattr(self, "_prediction_overview", None)
        if isinstance(extra, dict):
            out.update(extra)
        return out


class ModelLabStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        # Keep short so GUI polls never stick for tens of seconds on write locks.
        self._conn = sqlite3.connect(self.db_path, timeout=5.0)
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ModelLabStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ModelLabStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_lab_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                lab_uuid TEXT NOT NULL,
                lab_id TEXT NOT NULL,
                lab_name TEXT NOT NULL,
                parent_model_id TEXT NOT NULL,
                parent_model_name TEXT NOT NULL,
                model_checksum TEXT,
                description TEXT,
                purpose TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT,
                status TEXT NOT NULL DEFAULT 'READY',
                version INTEGER NOT NULL DEFAULT 1,
                lab_schema_version INTEGER NOT NULL DEFAULT 1,
                phase INTEGER NOT NULL DEFAULT 1,
                original_feature_count INTEGER,
                selected_feature_count INTEGER,
                training_rows INTEGER,
                target TEXT,
                algorithm TEXT,
                dataset_snapshot_json TEXT,
                model_snapshot_json TEXT,
                training_config_snapshot_json TEXT,
                wf_snapshot_json TEXT,
                metrics_snapshot_json TEXT,
                selected_features_snapshot_json TEXT,
                feature_ranking_snapshot_json TEXT,
                artifact_pointers_json TEXT
            );

            CREATE TABLE IF NOT EXISTS prediction_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lab_uuid TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_explanation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lab_uuid TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lab_uuid TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lab_uuid TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lab_uuid TEXT NOT NULL,
                key TEXT,
                value TEXT
            );
            """
        )
        self._migrate_columns()
        self.ensure_prediction_schema()
        self.conn.commit()

    def _migrate_columns(self) -> None:
        existing = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(model_lab_info)").fetchall()
        }
        extras = {
            "lab_uuid": "TEXT",
            "model_checksum": "TEXT",
            "description": "TEXT",
            "purpose": "TEXT",
            "lab_schema_version": "INTEGER NOT NULL DEFAULT 1",
            "phase": "INTEGER NOT NULL DEFAULT 1",
            "original_feature_count": "INTEGER",
            "master_dataset_id": "TEXT",
            "master_db_path": "TEXT",
        }
        for col, sql_type in extras.items():
            if col not in existing:
                self.conn.execute(f'ALTER TABLE model_lab_info ADD COLUMN "{col}" {sql_type}')
        self.conn.execute(
            """
            UPDATE model_lab_info SET status = 'READY'
            WHERE lower(COALESCE(status, '')) IN ('ready', 'ok', '')
            """
        )

    def write_info(
        self,
        *,
        lab_uuid: str,
        lab_id: str,
        lab_name: str,
        parent_model_id: str,
        parent_model_name: str,
        model_checksum: str | None,
        description: str | None,
        purpose: str | None,
        version: int,
        original_feature_count: int | None,
        selected_feature_count: int | None,
        training_rows: int | None,
        target: str | None,
        algorithm: str | None,
        dataset_snapshot: dict[str, Any] | None,
        model_snapshot: dict[str, Any] | None,
        training_config_snapshot: dict[str, Any] | None,
        wf_snapshot: dict[str, Any] | None,
        metrics_snapshot: dict[str, Any] | None,
        selected_features_snapshot: list[str] | None,
        feature_ranking_snapshot: dict[str, Any] | None,
        artifact_pointers: dict[str, Any] | None,
        created_by: str | None = "ml_research_studio",
        status: str = STATUS_READY,
        lab_schema_version: int = LAB_SCHEMA_VERSION,
        phase: int = LAB_PHASE,
    ) -> ModelLabInfo:
        created_at = _utc_now()
        status_n = normalize_lab_status(status)
        self.conn.execute(
            """
            INSERT INTO model_lab_info (
                id, lab_uuid, lab_id, lab_name, parent_model_id, parent_model_name,
                model_checksum, description, purpose,
                created_at, created_by, status, version, lab_schema_version, phase,
                original_feature_count, selected_feature_count, training_rows, target, algorithm,
                dataset_snapshot_json, model_snapshot_json, training_config_snapshot_json,
                wf_snapshot_json, metrics_snapshot_json,
                selected_features_snapshot_json, feature_ranking_snapshot_json,
                artifact_pointers_json
            ) VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                lab_uuid = excluded.lab_uuid,
                lab_id = excluded.lab_id,
                lab_name = excluded.lab_name,
                parent_model_id = excluded.parent_model_id,
                parent_model_name = excluded.parent_model_name,
                model_checksum = excluded.model_checksum,
                description = excluded.description,
                purpose = excluded.purpose,
                created_at = excluded.created_at,
                created_by = excluded.created_by,
                status = excluded.status,
                version = excluded.version,
                lab_schema_version = excluded.lab_schema_version,
                phase = excluded.phase,
                original_feature_count = excluded.original_feature_count,
                selected_feature_count = excluded.selected_feature_count,
                training_rows = excluded.training_rows,
                target = excluded.target,
                algorithm = excluded.algorithm,
                dataset_snapshot_json = excluded.dataset_snapshot_json,
                model_snapshot_json = excluded.model_snapshot_json,
                training_config_snapshot_json = excluded.training_config_snapshot_json,
                wf_snapshot_json = excluded.wf_snapshot_json,
                metrics_snapshot_json = excluded.metrics_snapshot_json,
                selected_features_snapshot_json = excluded.selected_features_snapshot_json,
                feature_ranking_snapshot_json = excluded.feature_ranking_snapshot_json,
                artifact_pointers_json = excluded.artifact_pointers_json
            """,
            (
                lab_uuid,
                lab_id,
                lab_name,
                parent_model_id,
                parent_model_name,
                model_checksum,
                description,
                purpose,
                created_at,
                created_by,
                status_n,
                int(version),
                int(lab_schema_version),
                int(phase),
                original_feature_count,
                selected_feature_count,
                training_rows,
                target,
                algorithm,
                _dumps(dataset_snapshot) if dataset_snapshot is not None else None,
                _dumps(model_snapshot) if model_snapshot is not None else None,
                _dumps(training_config_snapshot) if training_config_snapshot is not None else None,
                _dumps(wf_snapshot) if wf_snapshot is not None else None,
                _dumps(metrics_snapshot) if metrics_snapshot is not None else None,
                _dumps(selected_features_snapshot) if selected_features_snapshot is not None else None,
                _dumps(feature_ranking_snapshot) if feature_ranking_snapshot is not None else None,
                _dumps(artifact_pointers) if artifact_pointers is not None else None,
            ),
        )
        self.conn.commit()
        info = self.read_info()
        if info is None:
            raise RuntimeError("Failed to write model_lab_info")
        return info

    def read_info(self) -> ModelLabInfo | None:
        row = self.conn.execute(
            """
            SELECT lab_uuid, lab_id, lab_name, parent_model_id, parent_model_name,
                   model_checksum, description, purpose,
                   created_at, created_by, status, version, lab_schema_version, phase,
                   original_feature_count, selected_feature_count, training_rows, target, algorithm,
                   dataset_snapshot_json, model_snapshot_json, training_config_snapshot_json,
                   wf_snapshot_json, metrics_snapshot_json,
                   selected_features_snapshot_json, feature_ranking_snapshot_json,
                   artifact_pointers_json
            FROM model_lab_info WHERE id = 1
            """
        ).fetchone()
        if not row:
            return None

        def as_dict(raw: str | None) -> dict[str, Any] | None:
            val = _loads(raw)
            if val is None:
                return None
            return val if isinstance(val, dict) else {"raw": val}

        selected = _loads(row[24])
        ranking = _loads(row[25])
        return ModelLabInfo(
            lab_uuid=str(row[0] or ""),
            lab_id=str(row[1]),
            lab_name=str(row[2]),
            parent_model_id=str(row[3]),
            parent_model_name=str(row[4]),
            model_checksum=str(row[5]) if row[5] is not None else None,
            description=str(row[6]) if row[6] is not None else None,
            purpose=str(row[7]) if row[7] is not None else None,
            created_at=str(row[8]),
            created_by=str(row[9]) if row[9] is not None else None,
            status=normalize_lab_status(row[10]),
            version=int(row[11] or 1),
            lab_schema_version=int(row[12] or LAB_SCHEMA_VERSION),
            phase=int(row[13] or LAB_PHASE),
            original_feature_count=int(row[14]) if row[14] is not None else None,
            selected_feature_count=int(row[15]) if row[15] is not None else None,
            training_rows=int(row[16]) if row[16] is not None else None,
            target=str(row[17]) if row[17] is not None else None,
            algorithm=str(row[18]) if row[18] is not None else None,
            dataset_snapshot=as_dict(row[19]),
            model_snapshot=as_dict(row[20]),
            training_config_snapshot=as_dict(row[21]),
            wf_snapshot=as_dict(row[22]),
            metrics_snapshot=as_dict(row[23]),
            selected_features_snapshot=selected if isinstance(selected, list) else None,
            feature_ranking_snapshot=ranking if isinstance(ranking, dict) else None,
            artifact_pointers=as_dict(row[26]),
            db_path=self.db_path,
        )

    # --- Phase 2: prediction dataset -----------------------------------------

    def _prediction_table_columns(self) -> set[str]:
        return {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(prediction_dataset)").fetchall()
        }

    def _prediction_is_stub(self) -> bool:
        cols = self._prediction_table_columns()
        if not cols:
            return True
        return "prediction_id" not in cols

    def ensure_prediction_schema(self) -> None:
        """Upgrade stub prediction_dataset → full Phase-2 schema + summary table.

        Also creates prediction_exec_* orchestration tables used by Build+Compute
        so legacy labs open without a manual DB recreate.
        """
        if self._prediction_is_stub():
            self.conn.execute("DROP TABLE IF EXISTS prediction_dataset")
            self.conn.executescript(create_prediction_dataset_sql())
        else:
            existing = self._prediction_table_columns()
            # Rename ambiguous legacy columns (SQLite 3.25+)
            renames = (
                ("time_to_profit", "time_to_max_profit"),
                ("time_to_drawdown", "time_to_max_drawdown"),
            )
            for old, new in renames:
                if old in existing and new not in existing:
                    try:
                        self.conn.execute(
                            f'ALTER TABLE prediction_dataset RENAME COLUMN "{old}" TO "{new}"'
                        )
                    except sqlite3.Error:
                        pass
            existing = self._prediction_table_columns()
            from .prediction_schema import CORE_COLUMNS

            for name, typedef in CORE_COLUMNS:
                if name == "id" or name in existing:
                    continue
                # Strip PRIMARY KEY / NOT NULL / UNIQUE from ALTER adds
                simple = typedef.split()[0]
                self.conn.execute(
                    f'ALTER TABLE prediction_dataset ADD COLUMN "{name}" {simple}'
                )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pred_lab_day
                    ON prediction_dataset(lab_uuid, trading_day)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pred_timestamp
                    ON prediction_dataset(timestamp)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pred_master_row_id
                    ON prediction_dataset(master_row_id)
                """
            )

        self.conn.executescript(create_prediction_summary_sql())
        self._migrate_prediction_summary_columns()
        self.conn.executescript(create_prediction_day_metadata_sql())
        self._migrate_prediction_day_metadata_table()
        self._migrate_prediction_day_metadata_columns()
        self.conn.executescript(create_research_dashboard_sql())
        from .prediction_job_schema import create_prediction_exec_tables_sql

        self.conn.executescript(create_prediction_exec_tables_sql())
        self.conn.commit()

    def _migrate_prediction_day_metadata_table(self) -> None:
        """Rename/copy legacy prediction_build_status → prediction_day_metadata."""
        tables = {
            str(r[0])
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "prediction_build_status" not in tables:
            return
        if "prediction_day_metadata" not in tables:
            self.conn.execute(
                "ALTER TABLE prediction_build_status RENAME TO prediction_day_metadata"
            )
        else:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO prediction_day_metadata (
                    lab_uuid, trading_day, status, row_count, rows_expected,
                    progress_pct, error_message, started_at, finished_at,
                    selected, updated_at
                )
                SELECT lab_uuid, trading_day, status, row_count, rows_expected,
                       progress_pct, error_message, started_at, finished_at,
                       selected, updated_at
                FROM prediction_build_status
                """
            )
            self.conn.execute("DROP TABLE prediction_build_status")
        try:
            self.conn.execute("DROP INDEX IF EXISTS idx_pred_build_status")
        except sqlite3.Error:
            pass
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_day_metadata
                ON prediction_day_metadata(lab_uuid, status)
            """
        )

    def _migrate_prediction_day_metadata_columns(self) -> None:
        """Add newer day-metadata columns (e.g. dataset_type) without wiping catalog."""
        tables = {
            str(r[0])
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "prediction_day_metadata" not in tables:
            return
        existing = {
            str(row[1])
            for row in self.conn.execute(
                "PRAGMA table_info(prediction_day_metadata)"
            ).fetchall()
        }
        if "dataset_type" not in existing:
            self.conn.execute(
                "ALTER TABLE prediction_day_metadata "
                "ADD COLUMN dataset_type TEXT NOT NULL DEFAULT 'Seen'"
            )
            # Explicit backfill for engines that leave new cols NULL on old rows.
            self.conn.execute(
                """
                UPDATE prediction_day_metadata
                SET dataset_type = 'Seen'
                WHERE dataset_type IS NULL OR TRIM(COALESCE(dataset_type, '')) = ''
                """
            )

    def _migrate_prediction_summary_columns(self) -> None:
        existing = {
            str(row[1])
            for row in self.conn.execute(
                "PRAGMA table_info(prediction_dataset_summary)"
            ).fetchall()
        }
        from .prediction_schema import SUMMARY_COLUMNS

        for name, typedef in SUMMARY_COLUMNS:
            if name == "id" or name in existing:
                continue
            simple = typedef.split()[0]
            self.conn.execute(
                f'ALTER TABLE prediction_dataset_summary ADD COLUMN "{name}" {simple}'
            )

    def ensure_feature_columns(self, columns: list[str]) -> None:
        existing = self._prediction_table_columns()
        for col in columns:
            if col in existing:
                continue
            self.conn.execute(f'ALTER TABLE prediction_dataset ADD COLUMN "{col}" REAL')
        self.conn.commit()

    def clear_prediction_dataset(self) -> None:
        self.conn.execute("DELETE FROM prediction_dataset")
        self.conn.execute("DELETE FROM prediction_dataset_summary")
        self.conn.execute("DELETE FROM prediction_day_metadata")
        self.clear_research_dashboard_cache()
        self.conn.commit()
        try:
            from .prediction_metadata import clear_prediction_metadata_days

            clear_prediction_metadata_days(self.db_path)
        except Exception:
            pass

    def clear_research_dashboard_cache(self) -> None:
        self.ensure_prediction_schema()
        for table in (
            "research_dashboard_meta",
            "research_dashboard_premium_band",
            "research_dashboard_trading_day",
            "research_dashboard_feature",
        ):
            try:
                self.conn.execute(f"DELETE FROM {table}")
            except sqlite3.Error:
                pass
        self.conn.commit()

    def research_dashboard_source_fingerprint(self) -> tuple[int, str]:
        """(row_count, fingerprint) used to detect whether cache must rebuild."""
        self.ensure_prediction_schema()
        n = self.prediction_row_count()
        max_id = 0
        try:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM prediction_dataset"
            ).fetchone()
            max_id = int(row[0] or 0) if row else 0
        except sqlite3.Error:
            max_id = 0
        ds_hash = ""
        summary = self.read_prediction_summary() or {}
        ds_hash = str(summary.get("dataset_hash") or "")
        return n, f"{n}:{max_id}:{ds_hash}"

    def read_research_dashboard_meta(self) -> dict[str, Any] | None:
        self.ensure_prediction_schema()
        try:
            row = self.conn.execute(
                """
                SELECT schema_version, source_row_count, source_fingerprint, computed_at,
                       kpi_json, quality_json, risk_json, error_json, distribution_json
                FROM research_dashboard_meta WHERE id = 1
                """
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None

        def _loads(raw: Any) -> dict[str, Any]:
            if not raw:
                return {}
            try:
                val = json.loads(str(raw))
                return val if isinstance(val, dict) else {}
            except (TypeError, json.JSONDecodeError):
                return {}

        return {
            "schema_version": int(row[0] or 0),
            "source_row_count": int(row[1] or 0),
            "source_fingerprint": str(row[2] or ""),
            "computed_at": str(row[3] or ""),
            "kpi": _loads(row[4]),
            "quality": _loads(row[5]),
            "risk": _loads(row[6]),
            "error_metrics": _loads(row[7]),
            "distribution": _loads(row[8]),
        }

    def write_research_dashboard_cache(
        self,
        *,
        schema_version: int,
        source_row_count: int,
        source_fingerprint: str,
        computed_at: str,
        kpi: dict[str, Any],
        quality: dict[str, Any],
        risk: dict[str, Any],
        error_metrics: dict[str, Any],
        distribution: dict[str, Any],
        premium_bands: list[dict[str, Any]],
        trading_days: list[dict[str, Any]],
    ) -> None:
        """Persist dashboard outcome stats only (not Feature Research)."""
        self.ensure_prediction_schema()
        self.conn.execute("DELETE FROM research_dashboard_meta")
        self.conn.execute("DELETE FROM research_dashboard_premium_band")
        self.conn.execute("DELETE FROM research_dashboard_trading_day")
        self.conn.execute(
            """
            INSERT INTO research_dashboard_meta (
                id, schema_version, source_row_count, source_fingerprint, computed_at,
                kpi_json, quality_json, risk_json, error_json, distribution_json
            ) VALUES (1,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(schema_version),
                int(source_row_count),
                str(source_fingerprint),
                str(computed_at),
                json.dumps(kpi, ensure_ascii=False),
                json.dumps(quality, ensure_ascii=False),
                json.dumps(risk, ensure_ascii=False),
                json.dumps(error_metrics, ensure_ascii=False),
                json.dumps(distribution, ensure_ascii=False),
            ),
        )
        for i, band in enumerate(premium_bands):
            self.conn.execute(
                """
                INSERT INTO research_dashboard_premium_band (
                    band, sort_order, rows, hit_rate, direction_accuracy,
                    mae, avg_dd_before_target, avg_time_to_target, premium_mae
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(band.get("band") or ""),
                    i,
                    int(band.get("rows") or 0),
                    band.get("hit_rate"),
                    band.get("direction_accuracy"),
                    band.get("mae"),
                    band.get("avg_dd_before_target"),
                    band.get("avg_time_to_target"),
                    band.get("premium_mae"),
                ),
            )
        for day in trading_days:
            self.conn.execute(
                """
                INSERT INTO research_dashboard_trading_day (
                    trading_day, rows, hit_rate, direction_accuracy,
                    mae, avg_dd_before_target, avg_time_to_target, premium_mae
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    str(day.get("trading_day") or ""),
                    int(day.get("rows") or 0),
                    day.get("hit_rate"),
                    day.get("direction_accuracy"),
                    day.get("mae"),
                    day.get("avg_dd_before_target"),
                    day.get("avg_time_to_target"),
                    day.get("premium_mae"),
                ),
            )
        self.conn.commit()

    def read_research_dashboard_cache(self) -> dict[str, Any] | None:
        meta = self.read_research_dashboard_meta()
        if meta is None:
            return None

        bands = []
        for row in self.conn.execute(
            """
            SELECT band, rows, hit_rate, direction_accuracy, mae,
                   avg_dd_before_target, avg_time_to_target, premium_mae
            FROM research_dashboard_premium_band
            ORDER BY sort_order, band
            """
        ).fetchall():
            bands.append(
                {
                    "band": row[0],
                    "rows": int(row[1] or 0),
                    "hit_rate": row[2],
                    "direction_accuracy": row[3],
                    "mae": row[4],
                    "avg_dd_before_target": row[5],
                    "avg_time_to_target": row[6],
                    "premium_mae": row[7],
                }
            )
        days = []
        for row in self.conn.execute(
            """
            SELECT trading_day, rows, hit_rate, direction_accuracy, mae,
                   avg_dd_before_target, avg_time_to_target, premium_mae
            FROM research_dashboard_trading_day
            ORDER BY trading_day
            """
        ).fetchall():
            days.append(
                {
                    "trading_day": row[0],
                    "rows": int(row[1] or 0),
                    "hit_rate": row[2],
                    "direction_accuracy": row[3],
                    "mae": row[4],
                    "avg_dd_before_target": row[5],
                    "avg_time_to_target": row[6],
                    "premium_mae": row[7],
                }
            )
        n = int(meta.get("source_row_count") or 0)
        return {
            "available": n > 0,
            "error": None,
            "cached": True,
            "computed_at": meta.get("computed_at"),
            "source_fingerprint": meta.get("source_fingerprint"),
            "total_predictions": n,
            "kpi": meta.get("kpi") or {},
            "quality": meta.get("quality") or {},
            "risk": meta.get("risk") or {},
            "error_metrics": meta.get("error_metrics") or {},
            "distribution": meta.get("distribution") or {},
            "premium_bands": bands,
            "trading_days": days,
            # Feature Research is a separate workload / cache.
            "features": [],
        }

    def delete_predictions_for_day(self, lab_uuid: str, trading_day: str) -> int:
        cur = self.conn.execute(
            "DELETE FROM prediction_dataset WHERE lab_uuid = ? AND trading_day = ?",
            (lab_uuid, trading_day),
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def prediction_row_counts_by_day(self) -> dict[str, int]:
        try:
            rows = self.conn.execute(
                """
                SELECT trading_day, COUNT(*) FROM prediction_dataset
                WHERE trading_day IS NOT NULL AND trading_day != ''
                GROUP BY trading_day
                """
            ).fetchall()
            return {str(r[0]): int(r[1] or 0) for r in rows}
        except sqlite3.Error:
            return {}

    def prediction_row_count_for_day(
        self,
        trading_day: str,
        *,
        lab_uuid: str | None = None,
    ) -> int:
        """Single-day COUNT — uses ``(lab_uuid, trading_day)`` index when uuid given."""
        day = str(trading_day or "").strip()
        if not day:
            return 0
        try:
            if lab_uuid:
                row = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM prediction_dataset
                    WHERE lab_uuid = ? AND trading_day = ?
                    """,
                    (str(lab_uuid), day),
                ).fetchone()
            else:
                row = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM prediction_dataset
                    WHERE trading_day = ?
                    """,
                    (day,),
                ).fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def day_rows_expected(
        self, lab_uuid: str, trading_day: str
    ) -> int | None:
        """Previously-recorded expected dataset row count for one day.

        Read before overwriting ``rows_expected`` at build-completion time —
        the true parent-dataset/Master expected count (set by catalog sync)
        must not be clobbered with the just-built row count, or Complete vs
        Partial can never be told apart on a later read.
        """
        try:
            row = self.conn.execute(
                """
                SELECT rows_expected FROM prediction_day_metadata
                WHERE lab_uuid = ? AND trading_day = ?
                """,
                (lab_uuid, trading_day),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except sqlite3.Error:
            return None

    def days_needing_tb_rescore(
        self,
        lab_uuid: str,
        days: list[str],
        tb_model_name: str,
    ) -> list[str]:
        """Days with existing prediction rows missing/stale Triple Barrier scoring.

        Triple Barrier is an inference side-scorer (like Regression /
        Probability Ladder) — it must run for every built row, Seen and
        Unseen. A day already marked Complete (or Partial) from a build that
        ran without Triple Barrier enabled (or with a different TB model)
        must be treated as pending again once this ``tb_model_name`` is
        requested; otherwise normal resume semantics skip already-built days
        and its ``tb_*`` columns stay NULL forever.

        Also flags a day whose rows are already stamped with *this*
        ``tb_model_name`` but carry a NULL ``tb_pred_probability`` — the
        "Triple Barrier enabled but scored 0 of N rows" failure mode (model
        resolve failure, missing features, predict exception; see
        ``tb_all_null_warning``). Without this, a day that silently failed
        TB scoring on a prior build looks identical to a fully-scored day
        (same ``tb_model_name``) and would never be retried on resume, even
        though the requested TB model is enabled again.
        """
        name = str(tb_model_name or "").strip()
        day_list = [str(d).strip() for d in (days or []) if str(d or "").strip()]
        if not name or not day_list:
            return []
        placeholders = ",".join("?" for _ in day_list)
        try:
            rows = self.conn.execute(
                f"""
                SELECT trading_day, COUNT(*) FROM prediction_dataset
                WHERE lab_uuid = ? AND trading_day IN ({placeholders})
                  AND (
                    tb_model_name IS NULL
                    OR tb_model_name != ?
                    OR tb_pred_probability IS NULL
                  )
                GROUP BY trading_day
                """,
                (lab_uuid, *day_list, name),
            ).fetchall()
            return [str(r[0]) for r in rows if int(r[1] or 0) > 0]
        except sqlite3.Error:
            return []

    def ensure_build_days(
        self,
        lab_uuid: str,
        days: list[str],
        *,
        selected: set[str] | None = None,
        day_dataset_types: dict[str, str] | None = None,
        sync_pred_counts: bool = True,
    ) -> None:
        """Upsert catalog days as waiting; preserve completed/running status.

        ``day_dataset_types`` maps trading_day → Seen|Unseen for this model.
        When provided, types are written on insert and refreshed on existing rows.

        ``sync_pred_counts``: when True (default), scans ``prediction_dataset`` with
        GROUP BY to sync row_count/status. Set False for fast UI seeding from Master
        day lists — avoids multi-second COUNT on large prediction tables.
        """
        self.ensure_prediction_schema()
        now = _utc_now()
        existing_rows = {
            str(r[0]): (str(r[1] or ""), r[2])
            for r in self.conn.execute(
                "SELECT trading_day, status, rows_expected FROM prediction_day_metadata WHERE lab_uuid = ?",
                (lab_uuid,),
            ).fetchall()
        }
        existing = {day: st for day, (st, _exp) in existing_rows.items()}
        counts = self.prediction_row_counts_by_day() if sync_pred_counts else {}
        type_map = {
            str(k): normalize_dataset_type(v)
            for k, v in (day_dataset_types or {}).items()
            if str(k or "").strip()
        }
        for day in days:
            day_s = str(day).strip()
            if not day_s:
                continue
            is_sel = 1 if (selected is None or day_s in selected) else 0
            dtype = type_map.get(day_s, DATASET_TYPE_SEEN)
            if day_s in existing:
                if not sync_pred_counts:
                    if selected is not None:
                        self.conn.execute(
                            """
                            UPDATE prediction_day_metadata SET selected = ?, updated_at = ?
                            WHERE lab_uuid = ? AND trading_day = ?
                            """,
                            (is_sel, now, lab_uuid, day_s),
                        )
                    if day_s in type_map:
                        self.conn.execute(
                            """
                            UPDATE prediction_day_metadata
                            SET dataset_type = ?, updated_at = ?
                            WHERE lab_uuid = ? AND trading_day = ?
                            """,
                            (dtype, now, lab_uuid, day_s),
                        )
                    continue
                # Sync row_count from committed predictions; mark complete only
                # when every dataset row for the day has a prediction row.
                n = int(counts.get(day_s, 0))
                prev, prev_expected = existing_rows[day_s]
                # Crash recovery: stuck "running" with no rows → waiting
                if prev == DAY_RUNNING and n == 0:
                    self.conn.execute(
                        """
                        UPDATE prediction_day_metadata
                        SET status = ?, progress_pct = 0, updated_at = ?
                        WHERE lab_uuid = ? AND trading_day = ?
                        """,
                        (DAY_WAITING, now, lab_uuid, day_s),
                    )
                elif n > 0 and prev not in (DAY_RUNNING,):
                    day_status = resolve_day_completion_status(n, prev_expected)
                    self.conn.execute(
                        """
                        UPDATE prediction_day_metadata
                        SET row_count = ?, status = ?, progress_pct = 100,
                            selected = COALESCE(selected, ?), updated_at = ?
                        WHERE lab_uuid = ? AND trading_day = ?
                        """,
                        (n, day_status, is_sel, now, lab_uuid, day_s),
                    )
                elif selected is not None:
                    self.conn.execute(
                        """
                        UPDATE prediction_day_metadata SET selected = ?, updated_at = ?
                        WHERE lab_uuid = ? AND trading_day = ?
                        """,
                        (is_sel, now, lab_uuid, day_s),
                    )
                if day_s in type_map:
                    self.conn.execute(
                        """
                        UPDATE prediction_day_metadata
                        SET dataset_type = ?, updated_at = ?
                        WHERE lab_uuid = ? AND trading_day = ?
                        """,
                        (dtype, now, lab_uuid, day_s),
                    )
                continue
            n = int(counts.get(day_s, 0)) if sync_pred_counts else 0
            status = DAY_COMPLETED if n > 0 else DAY_WAITING
            self.conn.execute(
                """
                INSERT INTO prediction_day_metadata (
                    lab_uuid, trading_day, status, row_count, progress_pct,
                    selected, updated_at, finished_at, dataset_type
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    lab_uuid,
                    day_s,
                    status,
                    n,
                    100.0 if status == DAY_COMPLETED else None,
                    is_sel,
                    now,
                    now if status == DAY_COMPLETED else None,
                    dtype if day_s in type_map else DATASET_TYPE_SEEN,
                ),
            )
        self.conn.commit()

    def apply_day_dataset_types(
        self,
        lab_uuid: str,
        day_dataset_types: dict[str, str],
        *,
        sync_ui_meta: bool = True,
    ) -> int:
        """Update Dataset Type on existing catalog rows for this lab. Returns rows touched."""
        self.ensure_prediction_schema()
        now = _utc_now()
        n = 0
        for day, raw_type in (day_dataset_types or {}).items():
            day_s = str(day or "").strip()
            if not day_s:
                continue
            dtype = normalize_dataset_type(raw_type)
            cur = self.conn.execute(
                """
                UPDATE prediction_day_metadata
                SET dataset_type = ?, updated_at = ?
                WHERE lab_uuid = ? AND trading_day = ?
                """,
                (dtype, now, lab_uuid, day_s),
            )
            n += int(cur.rowcount or 0)
        self.conn.commit()
        if n and sync_ui_meta:
            try:
                from .prediction_metadata import upsert_prediction_day_metadata

                for day, raw_type in (day_dataset_types or {}).items():
                    day_s = str(day or "").strip()
                    if not day_s:
                        continue
                    upsert_prediction_day_metadata(
                        self.db_path,
                        day_s,
                        dataset_type=normalize_dataset_type(raw_type),
                    )
            except Exception:
                pass
        return n

    def list_build_days(self, lab_uuid: str) -> list[dict[str, Any]]:
        self.ensure_prediction_schema()
        cols = {
            str(row[1])
            for row in self.conn.execute(
                "PRAGMA table_info(prediction_day_metadata)"
            ).fetchall()
        }
        has_type = "dataset_type" in cols
        select_type = ", dataset_type" if has_type else ""
        rows = self.conn.execute(
            f"""
            SELECT trading_day, status, row_count, rows_expected, progress_pct,
                   error_message, started_at, finished_at, selected, updated_at
                   {select_type}
            FROM prediction_day_metadata
            WHERE lab_uuid = ?
            ORDER BY trading_day ASC
            """,
            (lab_uuid,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            dtype = normalize_dataset_type(r[10] if has_type and len(r) > 10 else None)
            out.append({
                "trading_day": str(r[0]),
                "status": str(r[1] or DAY_WAITING),
                "row_count": int(r[2] or 0),
                "rows_expected": int(r[3]) if r[3] is not None else None,
                "progress_pct": float(r[4]) if r[4] is not None else None,
                "error_message": r[5],
                "started_at": r[6],
                "finished_at": r[7],
                "selected": bool(int(r[8] or 0)),
                "updated_at": r[9],
                "dataset_type": dtype,
            })
        return out

    def build_summary(self, lab_uuid: str) -> dict[str, Any]:
        days = self.list_build_days(lab_uuid)
        counts = {
            DAY_WAITING: 0,
            DAY_RUNNING: 0,
            DAY_COMPLETED: 0,
            DAY_FAILED: 0,
            DAY_CANCELLED: 0,
            DAY_SKIPPED: 0,
        }
        selected = 0
        for d in days:
            st = d.get("status") or DAY_WAITING
            counts[st] = counts.get(st, 0) + 1
            if d.get("selected"):
                selected += 1
        remaining = sum(
            1
            for d in days
            if d.get("selected")
            and d.get("status") not in (DAY_COMPLETED, DAY_SKIPPED)
        )
        summary = self.read_prediction_summary() or {}
        return {
            "total_days": len(days),
            "completed": counts.get(DAY_COMPLETED, 0),
            "remaining": remaining,
            "failed": counts.get(DAY_FAILED, 0),
            "cancelled": counts.get(DAY_CANCELLED, 0),
            "running": counts.get(DAY_RUNNING, 0),
            "waiting": counts.get(DAY_WAITING, 0),
            "selected": selected,
            "days": days,
            "dataset_type": normalize_dataset_type(summary.get("dataset_type")),
            "parent_dataset": summary.get("parent_dataset"),
        }

    def set_build_day_status(
        self,
        lab_uuid: str,
        trading_day: str,
        *,
        status: str,
        row_count: int | None = None,
        rows_expected: int | None = None,
        progress_pct: float | None = None,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
        sync_ui_meta: bool = True,
    ) -> None:
        self.ensure_prediction_schema()
        now = _utc_now()
        exists = self.conn.execute(
            """
            SELECT 1 FROM prediction_day_metadata
            WHERE lab_uuid = ? AND trading_day = ?
            """,
            (lab_uuid, trading_day),
        ).fetchone()
        if exists:
            sets = ["status = ?", "updated_at = ?"]
            args: list[Any] = [status, now]
            if row_count is not None:
                sets.append("row_count = ?")
                args.append(int(row_count))
            if rows_expected is not None:
                sets.append("rows_expected = ?")
                args.append(int(rows_expected))
            if progress_pct is not None:
                sets.append("progress_pct = ?")
                args.append(float(progress_pct))
            if error_message is not None:
                sets.append("error_message = ?")
                args.append(error_message)
            if started:
                sets.append("started_at = COALESCE(started_at, ?)")
                args.append(now)
            if finished:
                sets.append("finished_at = ?")
                args.append(now)
            args.extend([lab_uuid, trading_day])
            self.conn.execute(
                f"""
                UPDATE prediction_day_metadata
                SET {", ".join(sets)}
                WHERE lab_uuid = ? AND trading_day = ?
                """,
                tuple(args),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO prediction_day_metadata (
                    lab_uuid, trading_day, status, row_count, rows_expected,
                    progress_pct, error_message, started_at, finished_at,
                    selected, updated_at, dataset_type
                ) VALUES (?,?,?,?,?,?,?,?,?,1,?,?)
                """,
                (
                    lab_uuid,
                    trading_day,
                    status,
                    int(row_count or 0),
                    rows_expected,
                    progress_pct,
                    error_message,
                    now if started else None,
                    now if finished else None,
                    now,
                    DATASET_TYPE_SEEN,
                ),
            )
        self.conn.commit()
        if sync_ui_meta:
            try:
                from .prediction_metadata import sync_day_metadata_from_status

                sync_day_metadata_from_status(
                    self.db_path,
                    trading_day,
                    status=status,
                    row_count=row_count,
                    rows_expected=rows_expected,
                    error_message=error_message,
                    started=started,
                    finished=finished,
                )
            except Exception:
                pass

    def set_day_rows_expected(
        self,
        lab_uuid: str,
        trading_day: str,
        rows_expected: int,
        *,
        sync_ui_meta: bool = True,
    ) -> None:
        """Store parent dataset row count on day metadata (no status change)."""
        self.ensure_prediction_schema()
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE prediction_day_metadata
            SET rows_expected = ?, updated_at = ?
            WHERE lab_uuid = ? AND trading_day = ?
            """,
            (int(rows_expected), now, lab_uuid, trading_day),
        )
        self.conn.commit()
        if sync_ui_meta:
            try:
                from .prediction_metadata import upsert_prediction_day_metadata

                upsert_prediction_day_metadata(
                    self.db_path,
                    trading_day,
                    dataset_rows=int(rows_expected),
                )
            except Exception:
                pass

    def set_days_selected(self, lab_uuid: str, selected_days: list[str]) -> None:
        self.ensure_prediction_schema()
        now = _utc_now()
        sel = {str(d) for d in selected_days}
        rows = self.conn.execute(
            "SELECT trading_day FROM prediction_day_metadata WHERE lab_uuid = ?",
            (lab_uuid,),
        ).fetchall()
        for (day,) in rows:
            self.conn.execute(
                """
                UPDATE prediction_day_metadata SET selected = ?, updated_at = ?
                WHERE lab_uuid = ? AND trading_day = ?
                """,
                (1 if str(day) in sel else 0, now, lab_uuid, day),
            )
        self.conn.commit()

    def pending_build_days(
        self,
        lab_uuid: str,
        *,
        selected_only: bool = True,
        include_failed: bool = True,
    ) -> list[str]:
        """Days still needing work.

        Completed days with 0 prediction rows are treated as pending so a
        Master-only / Unseen day that was marked Complete after an empty
        parquet load can be rebuilt.
        """
        days = self.list_build_days(lab_uuid)
        out: list[str] = []
        for d in days:
            if selected_only and not d.get("selected"):
                continue
            st = d.get("status")
            n = int(d.get("row_count") or 0)
            expected = int(d.get("rows_expected") or 0)
            if st == DAY_SKIPPED:
                continue
            if st == DAY_COMPLETED and n > 0:
                continue
            if st == DAY_COMPLETED and n <= 0 and expected <= 0:
                # Truly empty parent day with no Master expectation — leave alone
                # unless user re-selects / overwrite.
                continue
            if st == DAY_COMPLETED and n <= 0:
                # Bogus complete (0 rows) while rows were expected — rebuild.
                out.append(str(d["trading_day"]))
                continue
            if st == DAY_FAILED and not include_failed:
                continue
            out.append(str(d["trading_day"]))
        return out

    def prediction_row_count(self) -> int:
        try:
            row = self.conn.execute("SELECT COUNT(*) FROM prediction_dataset").fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def count_duplicate_prediction_ids(self) -> int:
        try:
            row = self.conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT prediction_id FROM prediction_dataset
                    GROUP BY prediction_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def count_missing_timestamps(self) -> int:
        try:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM prediction_dataset WHERE timestamp IS NULL"
            ).fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def insert_prediction_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        feature_columns: list[str] | None = None,
    ) -> int:
        if not rows:
            return 0
        cols = list(CORE_COLUMN_NAMES) + list(feature_columns or [])
        # Prefer keys present on first row
        present = [c for c in cols if c in rows[0]]
        placeholders = ",".join("?" for _ in present)
        col_sql = ",".join(f'"{c}"' for c in present)
        sql = f"INSERT OR REPLACE INTO prediction_dataset ({col_sql}) VALUES ({placeholders})"
        batch = [tuple(row.get(c) for c in present) for row in rows]
        self.conn.executemany(sql, batch)
        self.conn.commit()
        return len(batch)

    def write_prediction_summary(
        self,
        *,
        lab_uuid: str,
        status: str,
        row_count: int,
        trading_days: int,
        start_day: str | None = None,
        end_day: str | None = None,
        average_error: float | None = None,
        average_absolute_error: float | None = None,
        premium_error: float | None = None,
        direction_accuracy: float | None = None,
        generation_time_sec: float | None = None,
        dataset_hash: str | None = None,
        selected_feature_count: int | None = None,
        feature_columns_json: str | None = None,
        parent_model_name: str | None = None,
        parent_dataset: str | None = None,
        target_column: str | None = None,
        created_at: str | None = None,
        error_message: str | None = None,
        feature_storage_mode: str | None = None,
        master_dataset_id: str | None = None,
        master_db_path: str | None = None,
        dataset_type: str | None = None,
    ) -> None:
        self.ensure_prediction_schema()
        existing = self.read_prediction_summary()
        resolved_type = normalize_dataset_type(
            dataset_type
            if dataset_type is not None
            else (existing.get("dataset_type") if existing else None)
        )
        self.conn.execute(
            """
            INSERT INTO prediction_dataset_summary (
                id, lab_uuid, status, row_count, trading_days,
                start_day, end_day, average_error, average_absolute_error,
                premium_error, direction_accuracy, generation_time_sec,
                dataset_hash, selected_feature_count, feature_columns_json,
                parent_model_name, parent_dataset, target_column,
                created_at, error_message,
                feature_storage_mode, master_dataset_id, master_db_path,
                dataset_type
            ) VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                lab_uuid = excluded.lab_uuid,
                status = excluded.status,
                row_count = excluded.row_count,
                trading_days = excluded.trading_days,
                start_day = excluded.start_day,
                end_day = excluded.end_day,
                average_error = excluded.average_error,
                average_absolute_error = excluded.average_absolute_error,
                premium_error = excluded.premium_error,
                direction_accuracy = excluded.direction_accuracy,
                generation_time_sec = excluded.generation_time_sec,
                dataset_hash = excluded.dataset_hash,
                selected_feature_count = excluded.selected_feature_count,
                feature_columns_json = excluded.feature_columns_json,
                parent_model_name = excluded.parent_model_name,
                parent_dataset = excluded.parent_dataset,
                target_column = excluded.target_column,
                created_at = COALESCE(excluded.created_at, prediction_dataset_summary.created_at),
                error_message = excluded.error_message,
                feature_storage_mode = COALESCE(
                    excluded.feature_storage_mode,
                    prediction_dataset_summary.feature_storage_mode
                ),
                master_dataset_id = COALESCE(
                    excluded.master_dataset_id,
                    prediction_dataset_summary.master_dataset_id
                ),
                master_db_path = COALESCE(
                    excluded.master_db_path,
                    prediction_dataset_summary.master_db_path
                ),
                dataset_type = COALESCE(
                    excluded.dataset_type,
                    prediction_dataset_summary.dataset_type,
                    'Seen'
                )
            """,
            (
                lab_uuid,
                status,
                int(row_count),
                int(trading_days),
                start_day,
                end_day,
                average_error,
                average_absolute_error,
                premium_error,
                direction_accuracy,
                generation_time_sec,
                dataset_hash,
                selected_feature_count,
                feature_columns_json,
                parent_model_name,
                parent_dataset,
                target_column,
                created_at or _utc_now(),
                error_message,
                feature_storage_mode,
                master_dataset_id,
                master_db_path,
                resolved_type,
            ),
        )
        self.conn.commit()
        if master_db_path or master_dataset_id:
            self.set_lab_master_pointer(
                master_dataset_id=master_dataset_id,
                master_db_path=master_db_path,
            )

    def set_lab_master_pointer(
        self,
        *,
        master_dataset_id: str | None = None,
        master_db_path: str | None = None,
    ) -> None:
        """Persist Master Dataset pointer on model_lab_info (immutable lab metadata)."""
        existing = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(model_lab_info)").fetchall()
        }
        if "master_db_path" not in existing and "master_dataset_id" not in existing:
            return
        sets: list[str] = []
        args: list[Any] = []
        if master_dataset_id is not None and "master_dataset_id" in existing:
            sets.append("master_dataset_id = ?")
            args.append(master_dataset_id)
        if master_db_path is not None and "master_db_path" in existing:
            sets.append("master_db_path = ?")
            args.append(master_db_path)
        if not sets:
            return
        self.conn.execute(
            f"UPDATE model_lab_info SET {', '.join(sets)} WHERE id = 1",
            args,
        )
        self.conn.commit()

    def read_prediction_summary(self) -> dict[str, Any] | None:
        self.ensure_prediction_schema()
        cols = {
            str(row[1])
            for row in self.conn.execute(
                "PRAGMA table_info(prediction_dataset_summary)"
            ).fetchall()
        }
        has_type = "dataset_type" in cols
        type_select = ", dataset_type" if has_type else ""
        row = self.conn.execute(
            f"""
            SELECT lab_uuid, status, row_count, trading_days, start_day, end_day,
                   average_error, average_absolute_error, premium_error, direction_accuracy,
                   generation_time_sec, dataset_hash, selected_feature_count,
                   feature_columns_json, parent_model_name, parent_dataset,
                   target_column, created_at, error_message,
                   feature_storage_mode, master_dataset_id, master_db_path
                   {type_select}
            FROM prediction_dataset_summary WHERE id = 1
            """
        ).fetchone()
        if not row:
            return None
        return {
            "lab_uuid": row[0],
            "status": row[1],
            "row_count": int(row[2] or 0),
            "trading_days": int(row[3] or 0),
            "start_day": row[4],
            "end_day": row[5],
            "average_error": row[6],
            "average_absolute_error": row[7],
            "premium_error": row[8],
            "direction_accuracy": row[9],
            "generation_time_sec": row[10],
            "dataset_hash": row[11],
            "selected_feature_count": row[12],
            "feature_columns_json": row[13],
            "parent_model_name": row[14],
            "parent_dataset": row[15],
            "target_column": row[16],
            "created_at": row[17],
            "error_message": row[18],
            "feature_storage_mode": row[19] if len(row) > 19 else None,
            "master_dataset_id": row[20] if len(row) > 20 else None,
            "master_db_path": row[21] if len(row) > 21 else None,
            "dataset_type": normalize_dataset_type(
                row[22] if has_type and len(row) > 22 else None
            ),
        }

    def set_dataset_type(
        self,
        lab_uuid: str,
        dataset_type: str,
        *,
        trading_days: list[str] | None = None,
    ) -> str:
        """Manually persist Dataset Type on summary and day catalog rows.

        Prefer automatic per-model classification via training metadata
        (``sync_prediction_build_catalog`` / ``prediction_build_summary``).
        Use this for explicit overrides. If ``trading_days`` is None, all
        existing catalog days for the lab are updated.
        """
        self.ensure_prediction_schema()
        resolved = normalize_dataset_type(dataset_type)
        now = _utc_now()
        existing = self.read_prediction_summary()
        if existing:
            self.conn.execute(
                """
                UPDATE prediction_dataset_summary
                SET dataset_type = ?, lab_uuid = COALESCE(lab_uuid, ?)
                WHERE id = 1
                """,
                (resolved, lab_uuid),
            )
        else:
            self.write_prediction_summary(
                lab_uuid=lab_uuid,
                status=PRED_STATUS_NOT_GENERATED,
                row_count=0,
                trading_days=0,
                dataset_type=resolved,
            )
        day_cols = {
            str(row[1])
            for row in self.conn.execute(
                "PRAGMA table_info(prediction_day_metadata)"
            ).fetchall()
        }
        if "dataset_type" in day_cols:
            if trading_days is None:
                self.conn.execute(
                    """
                    UPDATE prediction_day_metadata
                    SET dataset_type = ?, updated_at = ?
                    WHERE lab_uuid = ?
                    """,
                    (resolved, now, lab_uuid),
                )
            else:
                for day in trading_days:
                    day_s = str(day or "").strip()
                    if not day_s:
                        continue
                    self.conn.execute(
                        """
                        UPDATE prediction_day_metadata
                        SET dataset_type = ?, updated_at = ?
                        WHERE lab_uuid = ? AND trading_day = ?
                        """,
                        (resolved, now, lab_uuid, day_s),
                    )
        self.conn.commit()
        return resolved

    def update_lab_phase(self, *, phase: int, lab_schema_version: int | None = None) -> None:
        if lab_schema_version is None:
            self.conn.execute(
                "UPDATE model_lab_info SET phase = ? WHERE id = 1",
                (int(phase),),
            )
        else:
            self.conn.execute(
                "UPDATE model_lab_info SET phase = ?, lab_schema_version = ? WHERE id = 1",
                (int(phase), int(lab_schema_version)),
            )
        self.conn.commit()

    def query_predictions(
        self,
        *,
        columns: list[str] | None = None,
        where_sql: str = "",
        where_args: list[Any] | None = None,
        order_by: str = "timestamp ASC",
        limit: int = 200,
        offset: int = 0,
        search: str | None = None,
        data_dir: str | None = None,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Browse helper for Prediction Dataset explorer."""
        self.ensure_prediction_schema()
        available = self._prediction_table_columns()
        default_cols = [
            "prediction_id",
            "trading_day",
            "timestamp",
            "token",
            "strike",
            "option_type",
            "current_ltp",
            "predicted_future_ltp",
            "actual_future_ltp",
            "absolute_error",
            "direction_correct",
        ]
        cols = [c for c in (columns or default_cols) if c in available]
        if not cols:
            cols = [c for c in default_cols if c in available]
        if not cols:
            return [], []

        from .prediction_feature_store import PredictionFeatureStore

        access = PredictionFeatureStore.from_store(self, data_dir=data_dir)
        use_join = access.is_referenced() and "m." in str(where_sql)

        clauses: list[str] = []
        args: list[Any] = list(where_args or [])
        if where_sql.strip():
            clauses.append(f"({where_sql})")
        if search and search.strip():
            q = f"%{search.strip()}%"
            prefix = "p." if use_join else ""
            clauses.append(
                f"({prefix}prediction_id LIKE ? OR {prefix}token LIKE ? OR "
                f"{prefix}trading_day LIKE ? OR CAST({prefix}strike AS TEXT) LIKE ?)"
            )
            args.extend([q, q, q, q])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # Whitelist order_by column
        order = "timestamp ASC"
        if order_by:
            parts = order_by.strip().split()
            col = parts[0]
            direction = parts[1].upper() if len(parts) > 1 else "ASC"
            if col in available and direction in ("ASC", "DESC"):
                order = f'{"p." if use_join else ""}"{col}" {direction}'

        if use_join:
            access.attach()
            col_sql = ", ".join(f'p."{c}"' for c in cols)
            from_sql = access.from_clause(pred_alias="p")
            sql = (
                f"SELECT {col_sql} FROM {from_sql}{where} "
                f"ORDER BY {order} LIMIT ? OFFSET ?"
            )
        else:
            col_sql = ", ".join(f'"{c}"' for c in cols)
            sql = (
                f"SELECT {col_sql} FROM prediction_dataset{where} "
                f"ORDER BY {order} LIMIT ? OFFSET ?"
            )
        args.extend([int(limit), int(offset)])
        rows = self.conn.execute(sql, args).fetchall()
        return cols, list(rows)

    def count_predictions(
        self,
        *,
        where_sql: str = "",
        where_args: list[Any] | None = None,
        search: str | None = None,
        data_dir: str | None = None,
    ) -> int:
        self.ensure_prediction_schema()
        available = self._prediction_table_columns()
        from .prediction_feature_store import PredictionFeatureStore

        access = PredictionFeatureStore.from_store(self, data_dir=data_dir)
        use_join = access.is_referenced() and "m." in str(where_sql)
        clauses: list[str] = []
        args: list[Any] = list(where_args or [])
        if where_sql.strip():
            clauses.append(f"({where_sql})")
        if search and search.strip():
            q = f"%{search.strip()}%"
            prefix = "p." if use_join else ""
            clauses.append(
                f"({prefix}prediction_id LIKE ? OR {prefix}token LIKE ? OR "
                f"{prefix}trading_day LIKE ? OR CAST({prefix}strike AS TEXT) LIKE ?)"
            )
            args.extend([q, q, q, q])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            if use_join:
                access.attach()
                from_sql = access.from_clause(pred_alias="p")
                row = self.conn.execute(
                    f"SELECT COUNT(*) FROM {from_sql}{where}",
                    args,
                ).fetchone()
            else:
                row = self.conn.execute(
                    f"SELECT COUNT(*) FROM prediction_dataset{where}",
                    args,
                ).fetchone()
            return int(row[0] or 0) if row else 0
        except sqlite3.Error:
            return 0

    def list_prediction_columns(self) -> list[str]:
        self.ensure_prediction_schema()
        rows = self.conn.execute("PRAGMA table_info(prediction_dataset)").fetchall()
        return [str(r[1]) for r in rows if str(r[1]) != "id"]
