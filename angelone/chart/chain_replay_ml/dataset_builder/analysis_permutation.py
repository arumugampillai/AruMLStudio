"""Permutation Importance for Phase 2 Research Lab.

Measures how much model performance degrades when each predictor is shuffled.
No retraining, no feature regeneration. Resume-safe across cancel/interrupt.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Sequence

from .analysis_feature_roles import (
    ROLE_LABEL,
    classify_feature_role,
)
from .analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _AnalysisDb,
    _now_iso,
    resolve_parquet_path,
    set_module_status,
)
from .analysis_mutual_information import mi_stars

ProgressCb = Callable[[dict[str, Any]], None]

DEFAULT_PERM_SAMPLE = 5_000


def _short_error(exc: BaseException, *, limit: int = 240) -> str:
    msg = str(exc)
    if "feature_names mismatch" in msg:
        # XGBoost dumps both full feature lists — keep a readable summary.
        if "expected " in msg:
            tail = msg.split("expected ", 1)[-1].strip()
            return f"feature_names mismatch — expected {tail[:180]}"
        return "feature_names mismatch between model and analysis feature matrix"
    if len(msg) > limit:
        return msg[: limit - 1] + "…"
    return msg


class CancelToken:
    """Thread-safe cancel flag for long-running permutation jobs."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


def ensure_permutation_schema(conn: Any) -> None:
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(permutation_importance)").fetchall()
    }
    if not cols:
        legacy = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(permutation)").fetchall()
        }
        if legacy and "delta_rmse" not in legacy and "model_id" not in legacy:
            try:
                conn.execute(
                    "ALTER TABLE permutation RENAME TO permutation_stub_legacy"
                )
            except Exception:
                pass

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS permutation_importance (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            target TEXT NOT NULL,
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
        )
        """
    )
    conn.execute(
        """
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
        )
        """
    )
    # Migrate slim CREATE from analysis_lab_store → full column set
    pi_cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(permutation_importance)").fetchall()
    }
    for col, typ in (
        ("baseline_accuracy", "REAL"),
        ("permuted_accuracy", "REAL"),
        ("delta_accuracy", "REAL"),
        ("baseline_f1", "REAL"),
        ("permuted_f1", "REAL"),
        ("delta_f1", "REAL"),
        ("baseline_auc", "REAL"),
        ("permuted_auc", "REAL"),
        ("delta_auc", "REAL"),
        ("importance", "REAL"),
        ("importance_rank", "INTEGER"),
        ("importance_percentile", "REAL"),
        ("interpretation", "TEXT"),
        ("task_type", "TEXT"),
        ("n_samples", "INTEGER"),
        ("created_at", "TEXT"),
        ("baseline_rmse", "REAL"),
        ("permuted_rmse", "REAL"),
        ("delta_rmse", "REAL"),
        ("baseline_mae", "REAL"),
        ("permuted_mae", "REAL"),
        ("delta_mae", "REAL"),
        ("dataset_id", "TEXT"),
        ("model_id", "TEXT"),
        ("target", "TEXT"),
        ("feature_name", "TEXT"),
        ("run_id", "TEXT"),
    ):
        if col not in pi_cols:
            try:
                conn.execute(
                    f"ALTER TABLE permutation_importance ADD COLUMN {col} {typ}"
                )
            except Exception:
                pass

    prof_cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(feature_profiles)").fetchall()
    }
    if prof_cols:
        for col, typ in (
            ("permutation_percentile", "REAL"),
            ("permutation_interpretation", "TEXT"),
            ("permutation_model", "TEXT"),
            ("permutation_target", "TEXT"),
            ("permutation_delta_rmse", "REAL"),
            ("permutation_delta_mae", "REAL"),
            ("permutation_baseline_rmse", "REAL"),
            ("permutation_permuted_rmse", "REAL"),
        ):
            if col not in prof_cols:
                conn.execute(
                    f"ALTER TABLE feature_profiles ADD COLUMN {col} {typ}"
                )


def interpret_permutation(percentile: float) -> str:
    p = float(percentile)
    if p >= 95:
        return "Critical"
    if p >= 80:
        return "Important"
    if p >= 60:
        return "Useful"
    if p >= 40:
        return "Minor"
    return "Negligible"


def permutation_detail(delta: float | None, *, task_type: str) -> str:
    if delta is None:
        return "Pending"
    d = float(delta)
    if task_type == "classification":
        if d >= 0.05:
            return "Model performance drops significantly."
        if d >= 0.02:
            return "Noticeable drop when this feature is shuffled."
        if d >= 0.005:
            return "Small but measurable impact."
        return "Negligible impact on classification metrics."
    if d >= 1.0:
        return "Model performance drops significantly."
    if d >= 0.25:
        return "Noticeable increase in error when shuffled."
    if d >= 0.05:
        return "Small but measurable impact."
    return "Negligible impact on regression error."


def perm_already_complete(
    data_dir: str, run_id: str, model_id: str, target: str
) -> bool:
    with _AnalysisDb(data_dir) as conn:
        ensure_permutation_schema(conn)
        row = conn.execute(
            """
            SELECT status, n_features, n_done FROM permutation_runs
            WHERE run_id = ? AND model_id = ? AND target = ?
            """,
            (run_id, model_id, target),
        ).fetchone()
        if not row:
            return False
        return (
            str(row["status"]) == STATUS_COMPLETED
            and int(row["n_done"] or 0) >= int(row["n_features"] or 0)
            and int(row["n_features"] or 0) > 0
        )


def load_permutation_results(
    data_dir: str,
    run_id: str,
    model_id: str,
    target: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_permutation_schema(conn)
        sql = """
            SELECT * FROM permutation_importance
            WHERE run_id = ? AND model_id = ? AND target = ?
            ORDER BY importance_rank ASC, feature_name ASC
        """
        params: list[Any] = [run_id, model_id, target]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _is_classification_target(
    target: str, y: Any, sidecar: dict[str, Any] | None
) -> bool:
    role = classify_feature_role(target, sidecar=sidecar)
    if role == ROLE_LABEL or str(target).lower().startswith("label_"):
        return True
    import pandas as pd

    s = pd.Series(y).dropna()
    if s.empty:
        return False
    nunique = int(s.nunique())
    return nunique <= 12 and (
        pd.api.types.is_integer_dtype(s) or nunique <= 6
    )


def _regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    err = yp - yt
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }


def _classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    if yp.ndim > 1:
        yp = yp[:, -1] if yp.shape[1] > 1 else yp.ravel()
    y_hat = (
        (yp >= 0.5).astype(int)
        if float(np.nanmax(yp)) <= 1.5
        else np.rint(yp).astype(int)
    )
    yt_i = np.rint(np.asarray(yt, dtype=float)).astype(int)
    mask = np.isfinite(yp)
    yt_i, y_hat, yp = yt_i[mask], y_hat[mask], yp[mask]
    out = {
        "accuracy": float(accuracy_score(yt_i, y_hat)) if len(yt_i) else float("nan"),
        "f1": float(
            f1_score(yt_i, y_hat, average="binary", zero_division=0)
        )
        if len(set(yt_i.tolist())) >= 2
        else float("nan"),
        "auc": float("nan"),
    }
    try:
        if len(set(yt_i.tolist())) >= 2:
            out["auc"] = float(roc_auc_score(yt_i, yp))
    except Exception:
        pass
    return out


def _predict(model: Any, X: Any) -> Any:
    import numpy as np

    return np.asarray(model.predict(X))


def _load_sidecar(parquet_path: str) -> dict[str, Any]:
    base, _ = os.path.splitext(parquet_path)
    js = base + ".json"
    if not os.path.isfile(js):
        return {}
    try:
        with open(js, encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _patch_profiles(
    conn: Any,
    run_id: str,
    model_id: str,
    target: str,
    rows: Sequence[dict[str, Any]],
) -> int:
    ensure_permutation_schema(conn)
    exists = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'feature_profiles'
        """
    ).fetchone()
    if not exists:
        return 0
    now = _now_iso()
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            UPDATE feature_profiles
            SET permutation_importance = ?, permutation_rank = ?,
                permutation_percentile = ?, permutation_interpretation = ?,
                permutation_model = ?, permutation_target = ?,
                permutation_delta_rmse = ?, permutation_delta_mae = ?,
                permutation_baseline_rmse = ?, permutation_permuted_rmse = ?,
                updated_at = ?
            WHERE run_id = ? AND feature_name = ?
            """,
            (
                float(r["importance"]),
                int(r["importance_rank"]),
                float(r["importance_percentile"]),
                str(r.get("interpretation") or ""),
                model_id,
                target,
                r.get("delta_rmse"),
                r.get("delta_mae"),
                r.get("baseline_rmse"),
                r.get("permuted_rmse"),
                now,
                run_id,
                str(r["feature_name"]),
            ),
        )
        n += int(cur.rowcount or 0)
    return n


def rehydrate_permutation_into_profiles(
    data_dir: str,
    run_id: str,
    *,
    model_id: str | None = None,
    target: str | None = None,
) -> int:
    with _AnalysisDb(data_dir) as conn:
        ensure_permutation_schema(conn)
        mid = str(model_id or "").strip()
        tgt = str(target or "").strip()
        if not mid or not tgt:
            row = conn.execute(
                """
                SELECT model_id, target FROM permutation_runs
                WHERE run_id = ? AND status = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (run_id, STATUS_COMPLETED),
            ).fetchone()
            if not row:
                return 0
            mid = str(row["model_id"])
            tgt = str(row["target"])
        rows = conn.execute(
            """
            SELECT * FROM permutation_importance
            WHERE run_id = ? AND model_id = ? AND target = ?
            """,
            (run_id, mid, tgt),
        ).fetchall()
        if not rows:
            return 0
        return _patch_profiles(conn, run_id, mid, tgt, [dict(r) for r in rows])


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda r: (
            -float(r.get("importance") or 0.0),
            str(r.get("feature_name") or ""),
        )
    )
    n = len(rows)
    for i, r in enumerate(rows, start=1):
        pct = 100.0 * (n - i + 1) / n if n else 0.0
        r["importance_rank"] = i
        r["importance_percentile"] = pct
        r["interpretation"] = interpret_permutation(pct)
    return rows


def _upsert_feature_row(
    conn: Any,
    *,
    run_id: str,
    dataset_id: str,
    model_id: str,
    target: str,
    row: dict[str, Any],
    task_type: str,
    n_samples: int,
) -> None:
    conn.execute(
        """
        INSERT INTO permutation_importance (
            run_id, dataset_id, model_id, target, feature_name,
            baseline_rmse, permuted_rmse, delta_rmse,
            baseline_mae, permuted_mae, delta_mae,
            baseline_accuracy, permuted_accuracy, delta_accuracy,
            baseline_f1, permuted_f1, delta_f1,
            baseline_auc, permuted_auc, delta_auc,
            importance, importance_rank, importance_percentile,
            interpretation, task_type, n_samples, created_at
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        ON CONFLICT(run_id, model_id, target, feature_name) DO UPDATE SET
            baseline_rmse=excluded.baseline_rmse,
            permuted_rmse=excluded.permuted_rmse,
            delta_rmse=excluded.delta_rmse,
            baseline_mae=excluded.baseline_mae,
            permuted_mae=excluded.permuted_mae,
            delta_mae=excluded.delta_mae,
            baseline_accuracy=excluded.baseline_accuracy,
            permuted_accuracy=excluded.permuted_accuracy,
            delta_accuracy=excluded.delta_accuracy,
            baseline_f1=excluded.baseline_f1,
            permuted_f1=excluded.permuted_f1,
            delta_f1=excluded.delta_f1,
            baseline_auc=excluded.baseline_auc,
            permuted_auc=excluded.permuted_auc,
            delta_auc=excluded.delta_auc,
            importance=excluded.importance,
            task_type=excluded.task_type,
            n_samples=excluded.n_samples,
            created_at=excluded.created_at
        """,
        (
            run_id,
            dataset_id,
            model_id,
            target,
            str(row["feature_name"]),
            row.get("baseline_rmse"),
            row.get("permuted_rmse"),
            row.get("delta_rmse"),
            row.get("baseline_mae"),
            row.get("permuted_mae"),
            row.get("delta_mae"),
            row.get("baseline_accuracy"),
            row.get("permuted_accuracy"),
            row.get("delta_accuracy"),
            row.get("baseline_f1"),
            row.get("permuted_f1"),
            row.get("delta_f1"),
            row.get("baseline_auc"),
            row.get("permuted_auc"),
            row.get("delta_auc"),
            row.get("importance"),
            None,
            None,
            None,
            task_type,
            n_samples,
            _now_iso(),
        ),
    )


def run_permutation_importance(
    data_dir: str,
    run_id: str,
    dataset: dict[str, Any],
    model_id: str,
    target: str,
    *,
    force: bool = False,
    sample_size: int = DEFAULT_PERM_SAMPLE,
    progress: ProgressCb | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Any]:
    """Full permutation lifecycle with resume + cancel support."""
    from .analysis_shap import _resolve_model_bundle

    model_id = str(model_id or "").strip()
    target = str(target or "").strip()
    if not model_id:
        raise ValueError("Select a trained model for Permutation Importance")
    if not target:
        raise ValueError("Select a target for Permutation Importance")

    dataset_id = str(dataset.get("dataset_id") or dataset.get("name") or "")
    cancel = cancel or CancelToken()

    if not force and perm_already_complete(data_dir, run_id, model_id, target):
        rows = load_permutation_results(data_dir, run_id, model_id, target)
        rehydrate_permutation_into_profiles(
            data_dir, run_id, model_id=model_id, target=target
        )
        set_module_status(
            data_dir,
            run_id,
            "permutation",
            STATUS_COMPLETED,
            message=f"Reused Permutation · {model_id} · {len(rows)} features",
            finished_at=_now_iso(),
        )
        return {
            "reused": True,
            "model_id": model_id,
            "target": target,
            "features": len(rows),
            "elapsed_sec": 0.0,
            "message": (
                f"Permutation done (reused) · {model_id} · {target} · "
                f"{len(rows)} features"
            ),
            "cancelled": False,
        }

    path = resolve_parquet_path(data_dir, dataset)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path!r}")

    started = _now_iso()
    t0 = time.perf_counter()
    set_module_status(
        data_dir,
        run_id,
        "permutation",
        STATUS_RUNNING,
        started_at=started,
        message=f"Permutation starting · {model_id} · {target}",
    )

    def _emit(**kwargs: Any) -> None:
        if progress:
            progress(
                {
                    "done": 0,
                    "total": 0,
                    "elapsed": max(time.perf_counter() - t0, 0.0),
                    "eta": None,
                    "message": "",
                    **kwargs,
                }
            )

    try:
        import numpy as np
        import pandas as pd

        _emit(message=f"Loading model {model_id}…")
        bundle = _resolve_model_bundle(data_dir, model_id)
        features = list(bundle["features"])
        model = bundle["model"]

        _emit(message="Loading analysis parquet…")
        sidecar = _load_sidecar(path)
        df = pd.read_parquet(path)
        if target not in df.columns:
            raise KeyError(f"Target {target!r} not in analysis dataset")

        missing = [f for f in features if f not in df.columns]
        if missing:
            preview = ", ".join(missing[:8])
            raise KeyError(
                f"{len(missing)} model feature(s) missing from analysis "
                f"dataset: {preview}"
            )

        if not features:
            raise ValueError("No predictor features available for permutation")

        X_all = df[features].apply(pd.to_numeric, errors="coerce")
        y_all = pd.to_numeric(df[target], errors="coerce")
        mask = y_all.notna() & X_all.notna().all(axis=1)
        X_all = X_all.loc[mask].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_all = y_all.loc[mask]
        if len(y_all) < 50:
            raise ValueError(
                f"Too few complete rows for permutation ({len(y_all)})"
            )

        if len(y_all) > int(sample_size):
            idx = y_all.sample(n=int(sample_size), random_state=42).index
            X_all = X_all.loc[idx]
            y_all = y_all.loc[idx]
        n_samples = int(len(y_all))
        task_type = (
            "classification"
            if _is_classification_target(target, y_all, sidecar)
            else "regression"
        )

        _emit(
            message=f"Baseline predict · sample={n_samples} · {task_type}…",
            total=len(features),
        )
        base_pred = _predict(model, X_all)
        if task_type == "classification":
            baseline = _classification_metrics(y_all, base_pred)
        else:
            baseline = _regression_metrics(y_all, base_pred)

        done_map: dict[str, dict[str, Any]] = {}
        with _AnalysisDb(data_dir) as conn:
            ensure_permutation_schema(conn)
            if force:
                conn.execute(
                    """
                    DELETE FROM permutation_importance
                    WHERE run_id = ? AND model_id = ? AND target = ?
                    """,
                    (run_id, model_id, target),
                )
            else:
                for r in conn.execute(
                    """
                    SELECT * FROM permutation_importance
                    WHERE run_id = ? AND model_id = ? AND target = ?
                    """,
                    (run_id, model_id, target),
                ).fetchall():
                    done_map[str(r["feature_name"])] = dict(r)
            conn.execute(
                """
                INSERT INTO permutation_runs (
                    run_id, dataset_id, model_id, target, status, task_type,
                    n_features, n_done, n_samples, baseline_json, features_json,
                    elapsed_sec, message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, model_id, target) DO UPDATE SET
                    status=excluded.status,
                    task_type=excluded.task_type,
                    n_features=excluded.n_features,
                    n_done=excluded.n_done,
                    n_samples=excluded.n_samples,
                    baseline_json=excluded.baseline_json,
                    features_json=excluded.features_json,
                    message=excluded.message,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    dataset_id,
                    model_id,
                    target,
                    STATUS_RUNNING,
                    task_type,
                    len(features),
                    len(done_map),
                    n_samples,
                    json.dumps(baseline),
                    json.dumps(features),
                    None,
                    "Running",
                    started,
                    _now_iso(),
                ),
            )

        pending = [f for f in features if f not in done_map]
        total = len(features)
        done_n = len(done_map)
        rng = np.random.default_rng(42)
        cancelled = False

        for feat in pending:
            if cancel.cancelled():
                cancelled = True
                break
            X_perm = X_all.copy()
            shuffled = X_perm[feat].to_numpy(copy=True)
            rng.shuffle(shuffled)
            X_perm[feat] = shuffled
            pred = _predict(model, X_perm)
            if task_type == "classification":
                m = _classification_metrics(y_all, pred)
                importance = float(
                    (baseline.get("accuracy") or 0.0) - (m.get("accuracy") or 0.0)
                )
                row = {
                    "feature_name": feat,
                    "baseline_rmse": None,
                    "permuted_rmse": None,
                    "delta_rmse": None,
                    "baseline_mae": None,
                    "permuted_mae": None,
                    "delta_mae": None,
                    "baseline_accuracy": baseline.get("accuracy"),
                    "permuted_accuracy": m.get("accuracy"),
                    "delta_accuracy": importance,
                    "baseline_f1": baseline.get("f1"),
                    "permuted_f1": m.get("f1"),
                    "delta_f1": float(
                        (baseline.get("f1") or 0.0) - (m.get("f1") or 0.0)
                    ),
                    "baseline_auc": baseline.get("auc"),
                    "permuted_auc": m.get("auc"),
                    "delta_auc": float(
                        (baseline.get("auc") or 0.0) - (m.get("auc") or 0.0)
                    )
                    if baseline.get("auc") == baseline.get("auc")
                    else None,
                    "importance": importance,
                    "task_type": task_type,
                    "n_samples": n_samples,
                }
            else:
                m = _regression_metrics(y_all, pred)
                delta_rmse = float(m["rmse"] - baseline["rmse"])
                delta_mae = float(m["mae"] - baseline["mae"])
                row = {
                    "feature_name": feat,
                    "baseline_rmse": baseline.get("rmse"),
                    "permuted_rmse": m.get("rmse"),
                    "delta_rmse": delta_rmse,
                    "baseline_mae": baseline.get("mae"),
                    "permuted_mae": m.get("mae"),
                    "delta_mae": delta_mae,
                    "baseline_accuracy": None,
                    "permuted_accuracy": None,
                    "delta_accuracy": None,
                    "baseline_f1": None,
                    "permuted_f1": None,
                    "delta_f1": None,
                    "baseline_auc": None,
                    "permuted_auc": None,
                    "delta_auc": None,
                    "importance": delta_rmse,
                    "task_type": task_type,
                    "n_samples": n_samples,
                }

            done_map[feat] = row
            done_n += 1
            elapsed = max(time.perf_counter() - t0, 0.0)
            rate = done_n / elapsed if elapsed > 0 else 0.0
            remaining = total - done_n
            eta = (remaining / rate) if rate > 0 else None

            with _AnalysisDb(data_dir) as conn:
                ensure_permutation_schema(conn)
                _upsert_feature_row(
                    conn,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    target=target,
                    row=row,
                    task_type=task_type,
                    n_samples=n_samples,
                )
                conn.execute(
                    """
                    UPDATE permutation_runs
                    SET n_done = ?, elapsed_sec = ?, message = ?, updated_at = ?
                    WHERE run_id = ? AND model_id = ? AND target = ?
                    """,
                    (
                        done_n,
                        round(elapsed, 3),
                        f"{done_n}/{total} · {feat}",
                        _now_iso(),
                        run_id,
                        model_id,
                        target,
                    ),
                )

            _emit(
                done=done_n,
                total=total,
                elapsed=elapsed,
                eta=eta,
                message=f"Permuting {done_n}/{total} · {feat}",
                feature=feat,
            )

        rows = _rank_rows(list(done_map.values()))
        now = _now_iso()
        with _AnalysisDb(data_dir) as conn:
            ensure_permutation_schema(conn)
            for r in rows:
                conn.execute(
                    """
                    UPDATE permutation_importance
                    SET importance_rank = ?, importance_percentile = ?,
                        interpretation = ?, importance = ?
                    WHERE run_id = ? AND model_id = ? AND target = ?
                      AND feature_name = ?
                    """,
                    (
                        int(r["importance_rank"]),
                        float(r["importance_percentile"]),
                        str(r["interpretation"]),
                        float(r["importance"]),
                        run_id,
                        model_id,
                        target,
                        str(r["feature_name"]),
                    ),
                )
            if not cancelled:
                _patch_profiles(conn, run_id, model_id, target, rows)
            elapsed = max(time.perf_counter() - t0, 0.0)
            if cancelled:
                status = "cancelled"
                msg = (
                    f"Permutation cancelled · {done_n}/{total} saved · "
                    f"resume later · {elapsed:.1f}s"
                )
            else:
                status = STATUS_COMPLETED
                msg = (
                    f"Permutation done · {model_id} · {target} · "
                    f"{len(rows)} features · sample={n_samples} · {elapsed:.1f}s"
                )
            conn.execute(
                """
                UPDATE permutation_runs
                SET status = ?, n_done = ?, elapsed_sec = ?, message = ?,
                    updated_at = ?
                WHERE run_id = ? AND model_id = ? AND target = ?
                """,
                (
                    status,
                    done_n,
                    round(elapsed, 3),
                    msg,
                    now,
                    run_id,
                    model_id,
                    target,
                ),
            )

        elapsed = max(time.perf_counter() - t0, 0.0)
        if cancelled:
            set_module_status(
                data_dir,
                run_id,
                "permutation",
                STATUS_FAILED,
                started_at=started,
                finished_at=_now_iso(),
                elapsed_sec=round(elapsed, 3),
                message=(
                    f"Cancelled · {done_n}/{total} features saved "
                    "(resume supported)"
                ),
            )
            _emit(
                done=done_n,
                total=total,
                elapsed=elapsed,
                eta=0.0,
                message=f"Cancelled · {done_n}/{total} saved",
            )
            return {
                "reused": False,
                "cancelled": True,
                "model_id": model_id,
                "target": target,
                "features": done_n,
                "total": total,
                "elapsed_sec": elapsed,
                "message": (
                    f"Permutation cancelled · {done_n}/{total} saved · "
                    f"re-run to resume · {elapsed:.1f}s"
                ),
            }

        done_msg = (
            f"Permutation done · {model_id} · {target} · {len(rows)} features · "
            f"sample={n_samples} · {elapsed:.1f}s"
        )
        set_module_status(
            data_dir,
            run_id,
            "permutation",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=done_msg,
        )
        _emit(
            done=total,
            total=total,
            elapsed=elapsed,
            eta=0.0,
            message=done_msg,
        )
        return {
            "reused": False,
            "cancelled": False,
            "model_id": model_id,
            "target": target,
            "features": len(rows),
            "elapsed_sec": elapsed,
            "task_type": task_type,
            "baseline": baseline,
            "rows": rows[:50],
            "message": done_msg,
        }
    except Exception as exc:
        elapsed = max(time.perf_counter() - t0, 0.0)
        set_module_status(
            data_dir,
            run_id,
            "permutation",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=_short_error(exc),
        )
        raise


__all__ = [
    "CancelToken",
    "DEFAULT_PERM_SAMPLE",
    "ensure_permutation_schema",
    "interpret_permutation",
    "load_permutation_results",
    "mi_stars",
    "perm_already_complete",
    "permutation_detail",
    "rehydrate_permutation_into_profiles",
    "run_permutation_importance",
]
