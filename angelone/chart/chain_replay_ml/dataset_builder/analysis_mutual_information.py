"""Mutual Information module for Phase 2 Research Lab.

Computes MI(feature, target) once per (run, target). No model training,
no dataset rebuild, no automatic deletion.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Sequence

from .analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _AnalysisDb,
    _now_iso,
    resolve_parquet_path,
    set_module_status,
)
from .analysis_feature_roles import (
    ROLE_LABEL,
    classify_feature_role,
    predictor_columns,
)

ProgressCb = Callable[[dict[str, Any]], None]

# Prefer these names when discovering targets from parquet / sidecar.
PREFERRED_TARGETS = (
    "future_ltp_1m",
    "future_ltp_3m",
    "future_ltp_5m",
    "Future_LTP_1m",
    "Future_LTP_3m",
    "Future_LTP_5m",
)


def ensure_mi_schema(conn: Any) -> None:
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(mutual_information)").fetchall()
    }
    if cols and "target" not in cols:
        conn.execute(
            "ALTER TABLE mutual_information RENAME TO mutual_information_legacy"
        )
        cols = set()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mutual_information (
            run_id TEXT NOT NULL,
            target TEXT NOT NULL,
            feature TEXT NOT NULL,
            score REAL,
            rank INTEGER,
            percentile REAL,
            interpretation TEXT,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, target, feature)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mi_runs (
            run_id TEXT NOT NULL,
            target TEXT NOT NULL,
            n_features INTEGER,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, target)
        )
        """
    )
    # Enrich feature_profiles with MI display fields
    prof_cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(feature_profiles)").fetchall()
    }
    if prof_cols:
        for col, typ in (
            ("mi_rank", "INTEGER"),
            ("mi_percentile", "REAL"),
            ("mi_interpretation", "TEXT"),
            ("mi_target", "TEXT"),
        ):
            if col not in prof_cols:
                conn.execute(
                    f"ALTER TABLE feature_profiles ADD COLUMN {col} {typ}"
                )


def interpret_mi(percentile: float) -> str:
    p = float(percentile)
    if p >= 95:
        return "Excellent predictor"
    if p >= 85:
        return "Strong"
    if p >= 60:
        return "Moderate"
    if p >= 30:
        return "Weak"
    return "Very weak"


def mi_stars(percentile: float | None) -> str:
    if percentile is None:
        return "Pending"
    # 5 stars at 95+, 4 at 80+, 3 at 60+, 2 at 40+, 1 at 20+, else empty
    p = float(percentile)
    n = 5 if p >= 95 else 4 if p >= 80 else 3 if p >= 60 else 2 if p >= 40 else 1 if p >= 20 else 0
    return ("★" * n) + ("☆" * (5 - n))


def discover_mi_targets(
    data_dir: str,
    dataset: dict[str, Any],
) -> list[str]:
    """List usable prediction targets from sidecar + parquet schema."""
    path = resolve_parquet_path(data_dir, dataset)
    found: list[str] = []
    if path:
        base, _ = os.path.splitext(path)
        js = base + ".json"
        if os.path.isfile(js):
            try:
                with open(js, encoding="utf-8") as f:
                    doc = json.load(f)
                for key in ("prediction_target_columns", "prediction_targets"):
                    raw = doc.get(key) or []
                    if isinstance(raw, list):
                        for x in raw:
                            s = str(x).strip()
                            if s and s not in found:
                                # prediction_targets may be horizons like "1m" — skip those
                                if s in {"10s", "30s", "1m", "5m", "3m"}:
                                    continue
                                found.append(s)
                labels = doc.get("classification_labels_5m") or {}
                for x in labels.get("columns") or []:
                    s = str(x).strip()
                    if s and s not in found:
                        found.append(s)
            except Exception:
                pass
        try:
            import pyarrow.parquet as pq

            names = [str(n) for n in pq.read_schema(path).names]
            name_set = set(names)
            ordered: list[str] = []
            for pref in PREFERRED_TARGETS:
                if pref in name_set and pref not in ordered:
                    ordered.append(pref)
            for n in names:
                low = n.lower()
                if n in ordered:
                    continue
                if low.startswith("future_ltp_") or low.startswith("label_"):
                    ordered.append(n)
            # Prefer ordered discovery, then sidecar extras present in parquet
            for s in found:
                if s in name_set and s not in ordered:
                    ordered.append(s)
            return ordered
        except Exception:
            pass
    return found


def mi_already_computed(data_dir: str, run_id: str, target: str) -> bool:
    with _AnalysisDb(data_dir) as conn:
        ensure_mi_schema(conn)
        row = conn.execute(
            "SELECT 1 FROM mi_runs WHERE run_id = ? AND target = ?",
            (run_id, target),
        ).fetchone()
        return row is not None


def load_mi_results(
    data_dir: str,
    run_id: str,
    target: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_mi_schema(conn)
        sql = """
            SELECT feature, score, rank, percentile, interpretation, n_samples
            FROM mutual_information
            WHERE run_id = ? AND target = ?
            ORDER BY rank ASC, feature ASC
        """
        params: list[Any] = [run_id, target]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def load_mi_for_feature(
    data_dir: str,
    run_id: str,
    target: str,
    feature: str,
) -> dict[str, Any] | None:
    with _AnalysisDb(data_dir) as conn:
        ensure_mi_schema(conn)
        row = conn.execute(
            """
            SELECT feature, score, rank, percentile, interpretation, n_samples
            FROM mutual_information
            WHERE run_id = ? AND target = ? AND feature = ?
            """,
            (run_id, target, feature),
        ).fetchone()
        return dict(row) if row else None


def _is_classification_target(y: Any) -> bool:
    import pandas as pd

    s = pd.Series(y).dropna()
    if s.empty:
        return False
    nunique = int(s.nunique())
    if nunique <= 12 and (
        pd.api.types.is_integer_dtype(s) or nunique <= 6
    ):
        return True
    # label_* columns are usually 0/1
    return False


def compute_mutual_information(
    parquet_path: str,
    target: str,
    *,
    max_rows: int | None = 80_000,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    """Compute MI for all numeric features vs target. Returns ranked rows."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    df = pd.read_parquet(parquet_path)
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not in dataset columns")

    y = pd.to_numeric(df[target], errors="coerce")
    # Predictors only — never score other targets/labels/metadata against the target.
    sidecar: dict[str, Any] = {}
    base, _ = os.path.splitext(parquet_path)
    js = base + ".json"
    if os.path.isfile(js):
        try:
            with open(js, encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                sidecar = doc
        except Exception:
            sidecar = {}
    feats = [
        c
        for c in predictor_columns([str(c) for c in df.columns], sidecar=sidecar)
        if str(c) != target
    ]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    mask = y.notna() & X.notna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    if len(y) < 50:
        raise ValueError(f"Too few valid rows for MI ({len(y)})")

    if max_rows is not None and len(y) > int(max_rows):
        idx = y.sample(n=int(max_rows), random_state=random_state).index
        X = X.loc[idx]
        y = y.loc[idx]

    # Fill remaining feature NaNs with column median (sklearn requires finite)
    X = X.fillna(X.median(numeric_only=True))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    clf = (
        _is_classification_target(y)
        or classify_feature_role(str(target), sidecar=sidecar) == ROLE_LABEL
        or str(target).lower().startswith("label_")
    )
    if clf:
        y_use = y.astype(int) if y.nunique() <= 20 else y.round().astype(int)
        scores = mutual_info_classif(
            X, y_use, discrete_features=False, random_state=random_state, n_neighbors=3
        )
    else:
        scores = mutual_info_regression(
            X, y, discrete_features=False, random_state=random_state, n_neighbors=3
        )

    pairs = list(zip([str(c) for c in X.columns], [float(s) for s in scores]))
    pairs.sort(key=lambda t: (-t[1], t[0]))
    n = len(pairs)
    out: list[dict[str, Any]] = []
    for i, (feat, score) in enumerate(pairs, start=1):
        # Percentile: higher MI → higher percentile
        percentile = 100.0 * (n - i + 1) / n if n else 0.0
        out.append(
            {
                "feature": feat,
                "score": score,
                "rank": i,
                "percentile": percentile,
                "interpretation": interpret_mi(percentile),
                "n_samples": int(len(y)),
            }
        )
    return out


def _patch_profiles_from_mi_rows(
    conn: Any,
    run_id: str,
    target: str,
    rows: Sequence[dict[str, Any]],
    *,
    now: str | None = None,
) -> int:
    """Copy MI scores onto feature_profiles (no-op if profiles missing)."""
    ensure_mi_schema(conn)
    exists = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'feature_profiles'
        """
    ).fetchone()
    if not exists:
        return 0
    stamp = now or _now_iso()
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            UPDATE feature_profiles
            SET mi_score = ?, mi_rank = ?, mi_percentile = ?,
                mi_interpretation = ?, mi_target = ?, updated_at = ?
            WHERE run_id = ? AND feature_name = ?
            """,
            (
                float(r["score"]),
                int(r["rank"]),
                float(r["percentile"]),
                str(r["interpretation"]),
                target,
                stamp,
                run_id,
                str(r["feature"]),
            ),
        )
        n += int(cur.rowcount or 0)
    return n


def rehydrate_mi_into_profiles(
    data_dir: str,
    run_id: str,
    *,
    target: str | None = None,
) -> int:
    """Re-apply stored MI onto feature_profiles after a profile rebuild.

    When *target* is omitted, uses the most recent mi_runs row for this run.
    """
    with _AnalysisDb(data_dir) as conn:
        ensure_mi_schema(conn)
        tgt = str(target or "").strip()
        if not tgt:
            row = conn.execute(
                """
                SELECT target FROM mi_runs
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if not row:
                return 0
            tgt = str(row["target"])
        rows = conn.execute(
            """
            SELECT feature, score, rank, percentile, interpretation, n_samples
            FROM mutual_information
            WHERE run_id = ? AND target = ?
            """,
            (run_id, tgt),
        ).fetchall()
        if not rows:
            return 0
        return _patch_profiles_from_mi_rows(
            conn, run_id, tgt, [dict(r) for r in rows]
        )


def persist_mi_results(
    data_dir: str,
    run_id: str,
    target: str,
    rows: Sequence[dict[str, Any]],
) -> int:
    now = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_mi_schema(conn)
        conn.execute(
            "DELETE FROM mutual_information WHERE run_id = ? AND target = ?",
            (run_id, target),
        )
        conn.execute(
            "DELETE FROM mi_runs WHERE run_id = ? AND target = ?",
            (run_id, target),
        )
        conn.executemany(
            """
            INSERT INTO mutual_information (
                run_id, target, feature, score, rank, percentile,
                interpretation, n_samples, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    target,
                    str(r["feature"]),
                    float(r["score"]),
                    int(r["rank"]),
                    float(r["percentile"]),
                    str(r["interpretation"]),
                    int(r.get("n_samples") or 0),
                    now,
                )
                for r in rows
            ],
        )
        conn.execute(
            """
            INSERT INTO mi_runs (run_id, target, n_features, n_samples, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                target,
                len(rows),
                int(rows[0]["n_samples"]) if rows else 0,
                now,
            ),
        )
        _patch_profiles_from_mi_rows(conn, run_id, target, rows, now=now)
    return len(rows)


def run_mutual_information(
    data_dir: str,
    run_id: str,
    dataset: dict[str, Any],
    target: str,
    *,
    force: bool = False,
    max_rows: int | None = 80_000,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Full MI module lifecycle. Reuses results unless force=True."""
    target = str(target or "").strip()
    if not target:
        raise ValueError("Select a prediction target for Mutual Information")

    def _tick(frac: float, message: str, *, elapsed: float = 0.0, **extra: Any) -> None:
        if not progress:
            return
        progress(
            {
                "frac": max(0.0, min(1.0, float(frac))),
                "elapsed": float(elapsed),
                "message": str(message),
                **extra,
            }
        )

    if not force and mi_already_computed(data_dir, run_id, target):
        _tick(0.2, f"Reusing MI vs {target}…")
        rows = load_mi_results(data_dir, run_id, target)
        # Profiles may have been rebuilt after MI — re-apply scores.
        rehydrate_mi_into_profiles(data_dir, run_id, target=target)
        set_module_status(
            data_dir,
            run_id,
            "mutual_information",
            STATUS_COMPLETED,
            message=f"Reused MI vs {target} · {len(rows)} features",
            finished_at=_now_iso(),
        )
        _tick(1.0, f"Reused MI vs {target} · {len(rows)} features")
        return {
            "reused": True,
            "target": target,
            "features": len(rows),
            "rows": rows[:50],
        }

    path = resolve_parquet_path(data_dir, dataset)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path!r}")

    started = _now_iso()
    t0 = time.perf_counter()
    set_module_status(
        data_dir,
        run_id,
        "mutual_information",
        STATUS_RUNNING,
        started_at=started,
        message=f"Computing MI vs {target}…",
    )
    _tick(0.02, f"Starting MI vs {target}…", elapsed=0.0)
    try:
        _tick(
            0.15,
            f"Loading parquet + scoring vs {target}…",
            elapsed=max(time.perf_counter() - t0, 0.0),
        )
        rows = compute_mutual_information(path, target, max_rows=max_rows)
        _tick(
            0.85,
            f"Persisting {len(rows)} MI scores…",
            elapsed=max(time.perf_counter() - t0, 0.0),
        )
        n = persist_mi_results(data_dir, run_id, target, rows)
        elapsed = max(time.perf_counter() - t0, 0.0)
        set_module_status(
            data_dir,
            run_id,
            "mutual_information",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=f"MI vs {target} · {n} features · {elapsed:.1f}s",
        )
        _tick(1.0, f"MI vs {target} · {n} features · {elapsed:.1f}s", elapsed=elapsed)
        return {
            "reused": False,
            "target": target,
            "features": n,
            "elapsed_sec": elapsed,
            "rows": rows[:50],
        }
    except Exception as exc:
        set_module_status(
            data_dir,
            run_id,
            "mutual_information",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(max(time.perf_counter() - t0, 0.0), 3),
            message=str(exc),
        )
        raise


def analysis_timeline(
    data_dir: str,
    run_id: str,
    feature: str,
    *,
    mi_target: str | None = None,
) -> list[dict[str, str]]:
    """Progress checklist for Feature Explorer."""
    from .analysis_feature_profiles import load_feature_profile
    from .analysis_lab_store import module_statuses

    statuses = {m["module_id"]: m["status"] for m in module_statuses(data_dir, run_id)}
    prof = load_feature_profile(data_dir, run_id, feature) or {}

    def _mark(done: bool, pending_label: str = "Pending") -> str:
        return "done" if done else "pending"

    corr_done = statuses.get("correlation") == "completed" or bool(prof.get("cluster_id"))
    cluster_done = bool(prof.get("cluster_id"))
    rec_done = bool(prof.get("recommendation"))
    mi_done = prof.get("mi_score") is not None
    if mi_target and not mi_done:
        mi_done = mi_already_computed(data_dir, run_id, mi_target)
    shap_done = (
        prof.get("shap_importance") is not None
        or statuses.get("shap") == "completed"
    )
    perm_done = (
        prof.get("permutation_importance") is not None
        or statuses.get("permutation") == "completed"
    )

    return [
        {"id": "correlation", "label": "Correlation", "state": _mark(corr_done)},
        {"id": "cluster", "label": "Cluster", "state": _mark(cluster_done)},
        {
            "id": "recommendation",
            "label": "Recommendation",
            "state": _mark(rec_done),
        },
        {
            "id": "mutual_information",
            "label": "Mutual Information",
            "state": _mark(mi_done),
        },
        {
            "id": "permutation",
            "label": "Permutation",
            "state": _mark(perm_done),
        },
        {
            "id": "discovery_rating",
            "label": "Discovery Rating",
            "state": _mark(
                prof.get("feature_score") is not None
                or prof.get("rating_score") is not None
            ),
        },
        {
            "id": "shap",
            "label": "Model Explanation (SHAP)",
            "state": _mark(shap_done),
        },
        {"id": "vif", "label": "VIF", "state": _mark(prof.get("vif") is not None)},
    ]


__all__ = [
    "analysis_timeline",
    "compute_mutual_information",
    "discover_mi_targets",
    "ensure_mi_schema",
    "interpret_mi",
    "load_mi_for_feature",
    "load_mi_results",
    "mi_already_computed",
    "mi_stars",
    "persist_mi_results",
    "rehydrate_mi_into_profiles",
    "run_mutual_information",
]
