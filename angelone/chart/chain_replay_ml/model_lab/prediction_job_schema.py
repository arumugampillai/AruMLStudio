"""Orchestration schema for external prediction worker processes."""

from __future__ import annotations

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_PAUSED = "paused"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

WORKER_STATUS_IDLE = "idle"
WORKER_STATUS_RUNNING = "running"
WORKER_STATUS_WAITING = "waiting"  # paused / no work
WORKER_STATUS_DONE = "done"
WORKER_STATUS_FAILED = "failed"

DAY_CP_PENDING = "pending"
DAY_CP_IN_PROGRESS = "in_progress"
DAY_CP_COMPLETED = "completed"
DAY_CP_FAILED = "failed"
DAY_CP_CANCELLED = "cancelled"

DEFAULT_EXEC_WORKERS = 3
MAX_EXEC_WORKERS = 4
CHECKPOINT_BATCH_ROWS = 2000
WORKER_POLL_SEC = 1.0


def create_prediction_exec_tables_sql() -> str:
    return """
            CREATE TABLE IF NOT EXISTS prediction_exec_job (
                job_id TEXT PRIMARY KEY,
                lab_uuid TEXT NOT NULL,
                lab_db_path TEXT NOT NULL,
                data_dir TEXT NOT NULL,
                worker_count INTEGER NOT NULL DEFAULT 3,
                status TEXT NOT NULL DEFAULT 'pending',
                overwrite INTEGER NOT NULL DEFAULT 0,
                resume INTEGER NOT NULL DEFAULT 1,
                row_limit INTEGER,
                mark_day_complete INTEGER NOT NULL DEFAULT 1,
                config_json TEXT NOT NULL,
                days_total INTEGER NOT NULL DEFAULT 0,
                days_completed INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pred_exec_job_status
                ON prediction_exec_job(status);

            CREATE TABLE IF NOT EXISTS prediction_exec_worker (
                job_id TEXT NOT NULL,
                worker_id INTEGER NOT NULL,
                pid INTEGER,
                assigned_day TEXT,
                current_row INTEGER NOT NULL DEFAULT 0,
                total_rows INTEGER NOT NULL DEFAULT 0,
                percent REAL NOT NULL DEFAULT 0,
                eta_sec REAL,
                status TEXT NOT NULL DEFAULT 'idle',
                log_path TEXT,
                last_message TEXT,
                heartbeat_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, worker_id)
            );

            CREATE TABLE IF NOT EXISTS prediction_exec_checkpoint (
                job_id TEXT NOT NULL,
                trading_day TEXT NOT NULL,
                worker_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                rows_committed INTEGER NOT NULL DEFAULT 0,
                rows_expected INTEGER,
                error_message TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, trading_day)
            );
            CREATE INDEX IF NOT EXISTS idx_pred_exec_cp_job_status
                ON prediction_exec_checkpoint(job_id, status);
            CREATE INDEX IF NOT EXISTS idx_pred_exec_cp_worker
                ON prediction_exec_checkpoint(job_id, worker_id);
            """
