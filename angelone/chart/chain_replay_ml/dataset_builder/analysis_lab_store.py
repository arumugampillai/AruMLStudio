"""Phase 2 Analysis Lab — SQLite result store (read-only over analysis datasets).

Separate from Master / build pipelines. Auto builds parquet; Analysis loads it
and writes module results into ``analysis.db``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ANALYSIS_DB_NAME = "analysis.db"

# Independent module lifecycle (order = dependency / Run All sequence).
ANALYSIS_MODULES: tuple[str, ...] = (
    "correlation",
    "hca",
    "vif",
    "mutual_information",
    "permutation",
    "feature_scorecard",
    "shap",
)

MODULE_LABELS: dict[str, str] = {
    "correlation": "Correlation",
    "hca": "HCA (Feature Families)",
    "vif": "VIF",
    "mutual_information": "Mutual Information",
    "permutation": "Permutation",
    "feature_scorecard": "Feature Rating (Discovery)",
    "shap": "Model Explanation (SHAP)",
}

# Soft dependencies: child may reuse parent results; parent must be Completed.
MODULE_DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "hca": ("correlation",),
    "vif": ("hca",),
    # Stage 1 Feature Discovery — SHAP is Stage 2 (Model Validation).
    "feature_scorecard": (
        "correlation",
        "mutual_information",
        "permutation",
    ),
}

STATUS_NOT_RUN = "not_run"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"


@dataclass(frozen=True)
class DatasetFingerprint:
    name: str
    path: str
    rows: int
    features: int
    columns_hash: str
    dataset_hash: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "rows": self.rows,
            "features": self.features,
            "columns_hash": self.columns_hash,
            "dataset_hash": self.dataset_hash,
            "created_at": self.created_at,
        }


def analysis_db_path(data_dir: str) -> str:
    root = str(data_dir or "").strip() or "."
    return os.path.join(root, ANALYSIS_DB_NAME)


def connect_analysis_db(data_dir: str) -> sqlite3.Connection:
    path = analysis_db_path(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_analysis_schema(conn)
    return conn


class _AnalysisDb:
    """Context manager that always closes the SQLite connection."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = connect_analysis_db(self._data_dir)
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self.conn is not None:
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            finally:
                self.conn.close()
                self.conn = None


def ensure_analysis_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            features INTEGER NOT NULL DEFAULT 0,
            columns_json TEXT NOT NULL DEFAULT '[]',
            columns_hash TEXT NOT NULL DEFAULT '',
            dataset_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            registered_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );

        -- Lab UI prefs (dataset selection, etc.) — survives panel restarts.
        CREATE TABLE IF NOT EXISTS lab_prefs (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS module_runs (
            run_id TEXT NOT NULL,
            module_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'not_run',
            started_at TEXT,
            finished_at TEXT,
            elapsed_sec REAL,
            version INTEGER NOT NULL DEFAULT 1,
            message TEXT,
            PRIMARY KEY (run_id, module_id),
            FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id)
        );

        -- Result tables (module-owned). Scaffold only; runners fill later.
        CREATE TABLE IF NOT EXISTS correlation (
            run_id TEXT NOT NULL,
            feature_a TEXT NOT NULL,
            feature_b TEXT NOT NULL,
            correlation REAL,
            PRIMARY KEY (run_id, feature_a, feature_b)
        );

        CREATE TABLE IF NOT EXISTS clusters (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            cluster TEXT NOT NULL,
            representative INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (run_id, feature)
        );

        CREATE TABLE IF NOT EXISTS vif (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            cluster TEXT,
            vif REAL,
            PRIMARY KEY (run_id, feature)
        );

        CREATE TABLE IF NOT EXISTS mutual_information (
            run_id TEXT NOT NULL,
            target TEXT NOT NULL DEFAULT '',
            feature TEXT NOT NULL,
            score REAL,
            rank INTEGER,
            percentile REAL,
            interpretation TEXT,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, target, feature)
        );

        CREATE TABLE IF NOT EXISTS mi_runs (
            run_id TEXT NOT NULL,
            target TEXT NOT NULL,
            n_features INTEGER,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, target)
        );

        CREATE TABLE IF NOT EXISTS shap (
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT '',
            feature TEXT NOT NULL,
            importance REAL,
            rank INTEGER,
            percentile REAL,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, model_name, feature)
        );

        CREATE TABLE IF NOT EXISTS shap_runs (
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            n_features INTEGER,
            n_samples INTEGER,
            algorithm TEXT,
            elapsed_sec REAL,
            created_at TEXT,
            PRIMARY KEY (run_id, model_name)
        );

        CREATE TABLE IF NOT EXISTS permutation_importance (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL DEFAULT '',
            model_id TEXT NOT NULL DEFAULT '',
            target TEXT NOT NULL DEFAULT '',
            feature_name TEXT NOT NULL,
            baseline_rmse REAL,
            permuted_rmse REAL,
            delta_rmse REAL,
            baseline_mae REAL,
            permuted_mae REAL,
            delta_mae REAL,
            baseline_accuracy REAL,
            permuted_accuracy REAL,
            delta_accuracy REAL,
            baseline_f1 REAL,
            permuted_f1 REAL,
            delta_f1 REAL,
            baseline_auc REAL,
            permuted_auc REAL,
            delta_auc REAL,
            importance REAL,
            importance_rank INTEGER,
            importance_percentile REAL,
            interpretation TEXT,
            task_type TEXT,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, model_id, target, feature_name)
        );

        CREATE TABLE IF NOT EXISTS permutation_runs (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT,
            task_type TEXT,
            n_features INTEGER,
            n_done INTEGER,
            n_samples INTEGER,
            baseline_json TEXT,
            features_json TEXT,
            elapsed_sec REAL,
            message TEXT,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (run_id, model_id, target)
        );

        CREATE TABLE IF NOT EXISTS permutation (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            importance REAL,
            rank INTEGER,
            PRIMARY KEY (run_id, feature)
        );

        CREATE TABLE IF NOT EXISTS feature_scorecard (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            recommendation TEXT,
            notes TEXT,
            PRIMARY KEY (run_id, feature)
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            run_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT,
            PRIMARY KEY (run_id, feature)
        );
        """
    )
    conn.commit()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha16(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def fingerprint_parquet(path: str) -> DatasetFingerprint:
    """Fingerprint an analysis parquet without loading the full frame."""
    import pyarrow.parquet as pq

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    schema = pq.read_schema(abs_path)
    columns = [str(n) for n in schema.names]
    meta = pq.read_metadata(abs_path)
    rows = int(meta.num_rows or 0)
    features = len(columns)
    col_hash = _sha16("\n".join(columns))
    # Include size + mtime so silent file replaces invalidate analysis.
    st = os.stat(abs_path)
    ds_hash = _sha16(
        col_hash,
        str(rows),
        str(features),
        str(int(st.st_size)),
        str(int(st.st_mtime_ns)),
    )
    name = os.path.splitext(os.path.basename(abs_path))[0]
    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
    return DatasetFingerprint(
        name=name,
        path=abs_path,
        rows=rows,
        features=features,
        columns_hash=col_hash,
        dataset_hash=ds_hash,
        created_at=created,
    )


def register_dataset(
    data_dir: str,
    parquet_path: str,
    *,
    name: str | None = None,
    relative_path: str | None = None,
) -> dict[str, Any]:
    """Upsert dataset fingerprint; mark prior runs stale if hash changed."""
    fp = fingerprint_parquet(parquet_path)
    store_path = relative_path or fp.path
    dataset_id = name or fp.name
    now = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        prev = conn.execute(
            "SELECT dataset_hash FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, name, path, rows, features, columns_json,
                columns_hash, dataset_hash, created_at, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                name=excluded.name,
                path=excluded.path,
                rows=excluded.rows,
                features=excluded.features,
                columns_json=excluded.columns_json,
                columns_hash=excluded.columns_hash,
                dataset_hash=excluded.dataset_hash,
                created_at=excluded.created_at,
                registered_at=excluded.registered_at
            """,
            (
                dataset_id,
                name or fp.name,
                store_path,
                fp.rows,
                fp.features,
                json.dumps([str(n) for n in pq_column_names(parquet_path)]),
                fp.columns_hash,
                fp.dataset_hash,
                fp.created_at,
                now,
            ),
        )
        if prev is not None and str(prev["dataset_hash"] or "") != fp.dataset_hash:
            _mark_runs_stale(conn, dataset_id)
        conn.commit()
        return get_dataset(conn, dataset_id) or fp.as_dict()


def pq_column_names(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return [str(n) for n in pq.read_schema(path).names]


def _mark_runs_stale(conn: sqlite3.Connection, dataset_id: str) -> None:
    rows = conn.execute(
        "SELECT run_id FROM analysis_runs WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()
    for row in rows:
        run_id = str(row["run_id"])
        conn.execute(
            "UPDATE analysis_runs SET status = ? WHERE run_id = ?",
            (STATUS_STALE, run_id),
        )
        conn.execute(
            """
            UPDATE module_runs
            SET status = ?
            WHERE run_id = ? AND status = ?
            """,
            (STATUS_STALE, run_id, STATUS_COMPLETED),
        )


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM datasets WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    return dict(row) if row else None


def list_datasets(data_dir: str) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY registered_at DESC, name ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def scan_and_register_datasets(
    data_dir: str,
    *,
    datasets_subdir: str = "datasets",
) -> list[dict[str, Any]]:
    """Register all ``*.parquet`` under data_dir/datasets into analysis.db."""
    root = os.path.join(str(data_dir), datasets_subdir)
    if not os.path.isdir(root):
        return list_datasets(data_dir)
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith(".parquet"):
            continue
        path = os.path.join(root, name)
        try:
            rel = os.path.join(datasets_subdir, name).replace("\\", "/")
            out.append(
                register_dataset(
                    data_dir,
                    path,
                    name=os.path.splitext(name)[0],
                    relative_path=rel,
                )
            )
        except Exception:
            continue
    return out or list_datasets(data_dir)


def load_analysis_dataset_catalog(
    data_dir: str,
    *,
    datasets_subdir: str = "datasets",
    force_rescan: bool = False,
) -> list[dict[str, Any]]:
    """Return Analysis Dataset meta from ``analysis.db`` (no research compute).

    Default trusts registered rows and only fingerprints parquet files that are
    missing from the catalog. ``force_rescan=True`` re-fingerprints every file
    (Refresh datasets).
    """
    if force_rescan:
        return scan_and_register_datasets(
            data_dir, datasets_subdir=datasets_subdir
        )

    existing = list_datasets(data_dir)
    known = {
        str(d.get("dataset_id") or d.get("name") or "").strip()
        for d in existing
        if str(d.get("dataset_id") or d.get("name") or "").strip()
    }
    root = os.path.join(str(data_dir), datasets_subdir)
    if not os.path.isdir(root):
        return existing

    added = False
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith(".parquet"):
            continue
        dataset_id = os.path.splitext(name)[0]
        if dataset_id in known:
            continue
        path = os.path.join(root, name)
        try:
            rel = os.path.join(datasets_subdir, name).replace("\\", "/")
            register_dataset(
                data_dir,
                path,
                name=dataset_id,
                relative_path=rel,
            )
            added = True
        except Exception:
            continue
    return list_datasets(data_dir) if added else existing


def ensure_analysis_run(data_dir: str, dataset_id: str) -> dict[str, Any]:
    """Return the latest non-stale run for dataset, or create a new one."""
    with _AnalysisDb(data_dir) as conn:
        ds = get_dataset(conn, dataset_id)
        if not ds:
            raise KeyError(f"Unknown dataset_id={dataset_id!r}")
        row = conn.execute(
            """
            SELECT * FROM analysis_runs
            WHERE dataset_id = ? AND dataset_hash = ? AND status != ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset_id, ds["dataset_hash"], STATUS_STALE),
        ).fetchone()
        if row:
            run = dict(row)
        else:
            run_id = str(uuid.uuid4())
            now = _now_iso()
            conn.execute(
                """
                INSERT INTO analysis_runs (run_id, dataset_id, dataset_hash, created_at, status)
                VALUES (?, ?, ?, ?, 'ready')
                """,
                (run_id, dataset_id, ds["dataset_hash"], now),
            )
            for mid in ANALYSIS_MODULES:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO module_runs (run_id, module_id, status, version)
                    VALUES (?, ?, ?, 1)
                    """,
                    (run_id, mid, STATUS_NOT_RUN),
                )
            conn.commit()
            run = dict(
                conn.execute(
                    "SELECT * FROM analysis_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            )
        _ensure_module_rows(conn, str(run["run_id"]))
        conn.commit()
        return run


def _ensure_module_rows(conn: sqlite3.Connection, run_id: str) -> None:
    for mid in ANALYSIS_MODULES:
        conn.execute(
            """
            INSERT OR IGNORE INTO module_runs (run_id, module_id, status, version)
            VALUES (?, ?, ?, 1)
            """,
            (run_id, mid, STATUS_NOT_RUN),
        )


def module_statuses(data_dir: str, run_id: str) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        _ensure_module_rows(conn, run_id)
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM module_runs WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        by_id = {str(r["module_id"]): dict(r) for r in rows}
        out: list[dict[str, Any]] = []
        for mid in ANALYSIS_MODULES:
            row = by_id.get(mid) or {
                "run_id": run_id,
                "module_id": mid,
                "status": STATUS_NOT_RUN,
                "version": 1,
            }
            row["label"] = MODULE_LABELS.get(mid, mid)
            out.append(row)
        return out


def set_module_status(
    data_dir: str,
    run_id: str,
    module_id: str,
    status: str,
    *,
    message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    with _AnalysisDb(data_dir) as conn:
        _ensure_module_rows(conn, run_id)
        cur = conn.execute(
            "SELECT version, started_at FROM module_runs WHERE run_id = ? AND module_id = ?",
            (run_id, module_id),
        ).fetchone()
        version = int(cur["version"] or 1) if cur else 1
        if status == STATUS_COMPLETED:
            version = version + 1
        keep_started = started_at
        if keep_started is None and cur is not None:
            keep_started = cur["started_at"]
        conn.execute(
            """
            INSERT INTO module_runs (
                run_id, module_id, status, started_at, finished_at,
                elapsed_sec, version, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, module_id) DO UPDATE SET
                status=excluded.status,
                started_at=COALESCE(excluded.started_at, module_runs.started_at),
                finished_at=excluded.finished_at,
                elapsed_sec=excluded.elapsed_sec,
                version=excluded.version,
                message=excluded.message
            """,
            (
                run_id,
                module_id,
                status,
                keep_started,
                finished_at,
                elapsed_sec,
                version,
                message,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM module_runs WHERE run_id = ? AND module_id = ?",
            (run_id, module_id),
        ).fetchone()
        return dict(row) if row else {}


def dependency_blockers(
    data_dir: str,
    run_id: str,
    module_id: str,
) -> list[str]:
    """Return parent module ids that are not Completed."""
    need = MODULE_DEPENDS_ON.get(module_id) or ()
    if not need:
        return []
    statuses = {m["module_id"]: m["status"] for m in module_statuses(data_dir, run_id)}
    return [p for p in need if statuses.get(p) != STATUS_COMPLETED]


def format_module_status_label(status: str) -> str:
    return {
        STATUS_NOT_RUN: "Not Run",
        STATUS_RUNNING: "Running",
        STATUS_COMPLETED: "Completed",
        STATUS_FAILED: "Failed",
        STATUS_STALE: "Stale",
    }.get(str(status or ""), str(status or "Not Run"))


def resolve_parquet_path(data_dir: str, dataset: dict[str, Any]) -> str:
    path = str(dataset.get("path") or "").strip()
    if not path:
        return ""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    joined = os.path.join(str(data_dir), path)
    if os.path.isfile(joined):
        return joined
    return path


PREF_SELECTED_DATASET = "selected_dataset_id"


def get_lab_pref(data_dir: str, key: str, default: str = "") -> str:
    """Read a lab UI preference from analysis.db."""
    k = str(key or "").strip()
    if not k:
        return default
    with _AnalysisDb(data_dir) as conn:
        row = conn.execute(
            "SELECT value FROM lab_prefs WHERE key = ?",
            (k,),
        ).fetchone()
    if not row:
        return default
    return str(row["value"] if row["value"] is not None else default)


def set_lab_pref(data_dir: str, key: str, value: str) -> None:
    """Persist a lab UI preference into analysis.db."""
    k = str(key or "").strip()
    if not k:
        return
    stamp = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        conn.execute(
            """
            INSERT INTO lab_prefs (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (k, str(value if value is not None else ""), stamp),
        )


def get_selected_analysis_dataset(data_dir: str) -> str:
    """Last Analysis Dataset selection (dataset_id or name)."""
    return get_lab_pref(data_dir, PREF_SELECTED_DATASET, "")


def set_selected_analysis_dataset(data_dir: str, dataset_id: str) -> None:
    """Remember Analysis Dataset selection in analysis.db."""
    set_lab_pref(data_dir, PREF_SELECTED_DATASET, str(dataset_id or "").strip())


__all__ = [
    "ANALYSIS_DB_NAME",
    "ANALYSIS_MODULES",
    "MODULE_DEPENDS_ON",
    "MODULE_LABELS",
    "PREF_SELECTED_DATASET",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_NOT_RUN",
    "STATUS_RUNNING",
    "STATUS_STALE",
    "DatasetFingerprint",
    "analysis_db_path",
    "connect_analysis_db",
    "dependency_blockers",
    "ensure_analysis_run",
    "ensure_analysis_schema",
    "fingerprint_parquet",
    "format_module_status_label",
    "get_dataset",
    "get_lab_pref",
    "get_selected_analysis_dataset",
    "list_datasets",
    "load_analysis_dataset_catalog",
    "module_statuses",
    "register_dataset",
    "resolve_parquet_path",
    "scan_and_register_datasets",
    "set_lab_pref",
    "set_module_status",
    "set_selected_analysis_dataset",
]
