"""SHAP Analysis module for Phase 2 Research Lab.

Uses a selected trained model + analysis parquet. No feature regeneration,
no retraining, no automatic deletion. Computes mean |SHAP| once per
(run_id, model_name) and reuses stored results.
"""

from __future__ import annotations

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

ProgressCb = Callable[[str, float], None]

DEFAULT_SHAP_SAMPLE = 800


def ensure_shap_schema(conn: Any) -> None:
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(shap)").fetchall()
    }
    if cols and "model_name" not in cols:
        conn.execute("ALTER TABLE shap RENAME TO shap_legacy")
        cols = set()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shap (
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            feature TEXT NOT NULL,
            importance REAL,
            rank INTEGER,
            percentile REAL,
            n_samples INTEGER,
            created_at TEXT,
            PRIMARY KEY (run_id, model_name, feature)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shap_runs (
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            n_features INTEGER,
            n_samples INTEGER,
            algorithm TEXT,
            elapsed_sec REAL,
            created_at TEXT,
            PRIMARY KEY (run_id, model_name)
        )
        """
    )
    prof_cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(feature_profiles)").fetchall()
    }
    if prof_cols:
        for col, typ in (
            ("shap_percentile", "REAL"),
            ("shap_model", "TEXT"),
        ):
            if col not in prof_cols:
                conn.execute(
                    f"ALTER TABLE feature_profiles ADD COLUMN {col} {typ}"
                )


def list_shap_models(data_dir: str) -> list[str]:
    """Trained model names available for SHAP (newest first).

    Includes Analysis Lab experiment packages — research SHAP needs them.
    Model Registry listing excludes those packages separately.
    """
    from chain_replay_ml.training.registry import list_trained_models

    rows = list_trained_models(
        data_dir, lightweight=True, include_experiments=True
    )
    return [str(r.get("model_name") or "") for r in rows if r.get("model_name")]


def default_shap_model(data_dir: str) -> str:
    from chain_replay_ml.training.registry import get_active_model

    active = get_active_model(data_dir)
    names = list_shap_models(data_dir)
    if active and active in names:
        return active
    return names[0] if names else ""


FULL_CATALOGUE_WARN_RATIO = 0.85

SHAP_PRE_SELECTION_WARNING = (
    "This model was trained before feature selection.\n"
    "SHAP may be influenced by highly correlated features.\n"
    "Use Correlation + MI + Permutation for feature selection.\n"
    "Use SHAP after retraining on the reduced feature set."
)


def assess_model_feature_selection(
    data_dir: str,
    model_name: str,
    *,
    predictor_count: int | None = None,
    dataset: dict[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Compare model selected_features vs analysis predictors.

    Returns stage metadata + optional warning when the model looks like it
    was trained on the full (pre-selection) catalogue.
    """
    from chain_replay_ml.training.registry import get_model_summary

    from .analysis_feature_roles import (
        ROLE_LABEL,
        ROLE_PREDICTOR,
        ROLE_TARGET,
        classify_feature_role,
        predictor_columns,
    )
    from .analysis_lab_store import resolve_parquet_path

    name = str(model_name or "").strip()
    out: dict[str, Any] = {
        "model_name": name,
        "stage": "Model Validation",
        "model_feature_count": 0,
        "predictor_count": int(predictor_count or 0),
        "ratio": None,
        "pre_selection": False,
        "warning": "",
    }
    if not name:
        return out

    summary = get_model_summary(data_dir, name)
    feats = [
        str(f)
        for f in (summary.get("selected_features") or [])
        if str(f).strip()
        and classify_feature_role(str(f)) not in (ROLE_TARGET, ROLE_LABEL)
    ]
    out["model_feature_count"] = len(feats)

    pred_n = int(predictor_count or 0)
    if pred_n <= 0 and dataset is not None:
        try:
            path = resolve_parquet_path(data_dir, dataset)
            import pyarrow.parquet as pq

            cols = [str(c) for c in pq.read_schema(path).names]
            pred_n = len(predictor_columns(cols))
        except Exception:
            pred_n = 0
    if pred_n <= 0 and run_id:
        try:
            from .analysis_lab_store import _AnalysisDb

            with _AnalysisDb(data_dir) as conn:
                rows = conn.execute(
                    """
                    SELECT feature_name, feature_role FROM feature_profiles
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()
            pred_n = 0
            for r in rows:
                role = str(r["feature_role"] or "") or classify_feature_role(
                    str(r["feature_name"] or "")
                )
                if role == ROLE_PREDICTOR or (
                    not role
                    and classify_feature_role(str(r["feature_name"] or ""))
                    == ROLE_PREDICTOR
                ):
                    pred_n += 1
        except Exception:
            pred_n = 0

    out["predictor_count"] = pred_n
    if pred_n > 0 and feats:
        ratio = len(feats) / float(pred_n)
        out["ratio"] = ratio
        # Trained on (nearly) the full catalogue → SHAP less trustworthy for selection
        if ratio >= FULL_CATALOGUE_WARN_RATIO or len(feats) >= max(pred_n - 5, 1):
            out["pre_selection"] = True
            out["warning"] = SHAP_PRE_SELECTION_WARNING
            out["stage"] = "Model Validation (pre-selection model)"
        else:
            out["stage"] = "Model Validation"
    return out


def shap_already_computed(data_dir: str, run_id: str, model_name: str) -> bool:
    with _AnalysisDb(data_dir) as conn:
        ensure_shap_schema(conn)
        row = conn.execute(
            "SELECT 1 FROM shap_runs WHERE run_id = ? AND model_name = ?",
            (run_id, model_name),
        ).fetchone()
        return row is not None


def load_shap_results(
    data_dir: str,
    run_id: str,
    model_name: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_shap_schema(conn)
        sql = """
            SELECT feature, importance, rank, percentile, n_samples
            FROM shap
            WHERE run_id = ? AND model_name = ?
            ORDER BY rank ASC, feature ASC
        """
        params: list[Any] = [run_id, model_name]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _explainer_model(model: Any) -> Any:
    """Prefer raw booster / native tree model for TreeExplainer."""
    if hasattr(model, "_booster") and getattr(model, "_booster") is not None:
        return model._booster
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None:
        return booster
    return model


def _resolve_model_bundle(data_dir: str, model_name: str) -> dict[str, Any]:
    from chain_replay_ml.training.model_runtime import (
        load_prediction_model,
        resolve_production_model_path,
    )
    from chain_replay_ml.training.paths import model_package_dir
    from chain_replay_ml.training.registry import get_model_summary

    summary = get_model_summary(data_dir, model_name)
    features = [str(f) for f in (summary.get("selected_features") or []) if str(f).strip()]
    if not features:
        raise ValueError(f"Model {model_name!r} has no selected_features")
    # Keep the exact trained feature list / order for predict().
    # Do NOT drop identity-looking columns the model was trained on (e.g. strike).
    # Only exclude leaked targets / labels.
    from .analysis_feature_roles import ROLE_LABEL, ROLE_TARGET, classify_feature_role

    features = [
        f
        for f in features
        if classify_feature_role(f) not in (ROLE_TARGET, ROLE_LABEL)
    ]
    if not features:
        raise ValueError(f"Model {model_name!r} has no usable features for inference")
    pkg = model_package_dir(data_dir, model_name)
    algo = summary.get("algorithm")
    path = resolve_production_model_path(pkg, algorithm=algo)
    if not path:
        raise FileNotFoundError(f"No production model file for {model_name!r}")
    model = load_prediction_model(path, algo)
    return {
        "model_name": str(summary.get("model_name") or model_name),
        "algorithm": algo,
        "features": features,
        "target": summary.get("target"),
        "model_path": path,
        "model": model,
        "summary": summary,
    }


def _mean_abs_shap_native(
    model: Any,
    X: Any,
    features: Sequence[str],
) -> list[float] | None:
    """Fallback mean |contribution| via booster pred_contribs (no shap pkg)."""
    import numpy as np

    explainer_target = _explainer_model(model)
    # LightGBM Booster
    if hasattr(explainer_target, "predict") and type(explainer_target).__name__ == "Booster":
        try:
            # lightgbm.Booster
            import lightgbm as lgb  # noqa: F401

            if explainer_target.__class__.__module__.startswith("lightgbm"):
                contribs = np.asarray(
                    explainer_target.predict(X, pred_contrib=True), dtype=float
                )
                if contribs.ndim != 2 or contribs.shape[1] < len(features) + 1:
                    return None
                return [float(v) for v in np.abs(contribs[:, : len(features)]).mean(axis=0)]
        except Exception:
            pass
    # XGBoost
    try:
        import xgboost as xgb

        booster = explainer_target
        if hasattr(model, "get_booster"):
            booster = model.get_booster() or explainer_target
        dmat = xgb.DMatrix(X, feature_names=list(features))
        contribs = np.asarray(booster.predict(dmat, pred_contribs=True), dtype=float)
        if contribs.ndim != 2 or contribs.shape[1] < len(features) + 1:
            return None
        return [float(v) for v in np.abs(contribs[:, : len(features)]).mean(axis=0)]
    except Exception:
        return None


def _mean_abs_shap_values(
    model: Any,
    X: Any,
    features: Sequence[str],
    *,
    progress: ProgressCb | None = None,
    started: float,
) -> list[float]:
    import numpy as np

    def _tick(msg: str) -> None:
        if progress:
            progress(msg, max(time.perf_counter() - started, 0.0))

    n_samples = int(len(X))
    prefer_gpu = False
    try:
        from chain_replay_ml.training.model_device import (
            verify_xgboost_booster_device,
        )

        booster = _explainer_model(model)
        executed = verify_xgboost_booster_device(booster)
        prefer_gpu = str(executed).startswith("cuda")
    except Exception:
        prefer_gpu = False

    if prefer_gpu:
        _tick(f"SHAP GPU path · sample={n_samples}…")
        native = _mean_abs_shap_native(model, X, features)
        if native is not None:
            return native
        try:
            import shap

            gpu_cls = getattr(shap, "GPUTreeExplainer", None)
            if gpu_cls is not None:
                explainer = gpu_cls(_explainer_model(model))
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                arr = np.abs(np.asarray(shap_vals, dtype=float))
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)
                return [float(v) for v in arr.mean(axis=0)]
        except Exception as exc:
            _tick(f"GPU SHAP failed ({exc}); falling back to CPU…")

    _tick(f"Building TreeExplainer · sample={n_samples}…")
    try:
        import shap

        explainer_target = _explainer_model(model)
        explainer = shap.TreeExplainer(explainer_target)
        _tick(
            f"Computing SHAP values · {n_samples} rows × {len(features)} features…"
        )
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        arr = np.abs(np.asarray(shap_vals, dtype=float))
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return [float(v) for v in arr.mean(axis=0)]
    except ImportError:
        _tick(
            f"Computing native pred_contribs · {n_samples} rows × "
            f"{len(features)} features (shap pkg not installed)…"
        )
        native = _mean_abs_shap_native(model, X, features)
        if native is None:
            raise ImportError(
                "shap package is required for SHAP Analysis "
                "(native pred_contribs fallback failed)"
            )
        return native
    except Exception as exc:
        _tick(f"TreeExplainer failed ({exc}); trying native pred_contribs…")
        native = _mean_abs_shap_native(model, X, features)
        if native is None:
            raise RuntimeError(f"SHAP compute failed: {exc}") from exc
        return native


def compute_shap_for_dataset(
    parquet_path: str,
    model: Any,
    features: Sequence[str],
    *,
    sample_size: int = DEFAULT_SHAP_SAMPLE,
    random_state: int = 42,
    progress: ProgressCb | None = None,
    t0: float | None = None,
) -> list[dict[str, Any]]:
    """Mean |SHAP| for each model feature on analysis parquet rows."""
    import numpy as np
    import pandas as pd

    started = t0 if t0 is not None else time.perf_counter()

    def _tick(msg: str) -> None:
        if progress:
            progress(msg, max(time.perf_counter() - started, 0.0))

    _tick("Loading analysis parquet…")
    df = pd.read_parquet(parquet_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise KeyError(
            f"{len(missing)} model feature(s) missing from analysis dataset: "
            f"{preview}{more}"
        )

    _tick(f"Preparing matrix · {len(features)} features…")
    X = df[list(features)].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    if len(X) < 30:
        raise ValueError(f"Too few complete rows for SHAP ({len(X)})")

    if len(X) > int(sample_size):
        X = X.sample(n=int(sample_size), random_state=random_state)
    n_samples = int(len(X))

    mean_abs = _mean_abs_shap_values(
        model, X, features, progress=progress, started=started
    )
    if len(mean_abs) != len(features):
        raise RuntimeError(
            f"SHAP length mismatch: got {len(mean_abs)} for {len(features)} features"
        )

    _tick("Ranking features…")
    pairs = list(zip([str(c) for c in features], [float(v) for v in mean_abs]))
    pairs.sort(key=lambda t: (-t[1], t[0]))
    n = len(pairs)
    out: list[dict[str, Any]] = []
    for i, (feat, score) in enumerate(pairs, start=1):
        percentile = 100.0 * (n - i + 1) / n if n else 0.0
        out.append(
            {
                "feature": feat,
                "importance": score,
                "rank": i,
                "percentile": percentile,
                "n_samples": n_samples,
            }
        )
    return out


def _patch_profiles_from_shap_rows(
    conn: Any,
    run_id: str,
    model_name: str,
    rows: Sequence[dict[str, Any]],
    *,
    now: str | None = None,
) -> int:
    ensure_shap_schema(conn)
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
            SET shap_importance = ?, shap_rank = ?, shap_percentile = ?,
                shap_model = ?, updated_at = ?
            WHERE run_id = ? AND feature_name = ?
            """,
            (
                float(r["importance"]),
                int(r["rank"]),
                float(r["percentile"]),
                model_name,
                stamp,
                run_id,
                str(r["feature"]),
            ),
        )
        n += int(cur.rowcount or 0)
    return n


def rehydrate_shap_into_profiles(
    data_dir: str,
    run_id: str,
    *,
    model_name: str | None = None,
) -> int:
    """Re-apply stored SHAP onto feature_profiles after a profile rebuild."""
    with _AnalysisDb(data_dir) as conn:
        ensure_shap_schema(conn)
        name = str(model_name or "").strip()
        if not name:
            row = conn.execute(
                """
                SELECT model_name FROM shap_runs
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if not row:
                return 0
            name = str(row["model_name"])
        rows = conn.execute(
            """
            SELECT feature, importance, rank, percentile, n_samples
            FROM shap
            WHERE run_id = ? AND model_name = ?
            """,
            (run_id, name),
        ).fetchall()
        if not rows:
            return 0
        return _patch_profiles_from_shap_rows(
            conn, run_id, name, [dict(r) for r in rows]
        )


def persist_shap_results(
    data_dir: str,
    run_id: str,
    model_name: str,
    rows: Sequence[dict[str, Any]],
    *,
    algorithm: str | None = None,
    elapsed_sec: float | None = None,
) -> int:
    now = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_shap_schema(conn)
        conn.execute(
            "DELETE FROM shap WHERE run_id = ? AND model_name = ?",
            (run_id, model_name),
        )
        conn.execute(
            "DELETE FROM shap_runs WHERE run_id = ? AND model_name = ?",
            (run_id, model_name),
        )
        conn.executemany(
            """
            INSERT INTO shap (
                run_id, model_name, feature, importance, rank, percentile,
                n_samples, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    model_name,
                    str(r["feature"]),
                    float(r["importance"]),
                    int(r["rank"]),
                    float(r["percentile"]),
                    int(r.get("n_samples") or 0),
                    now,
                )
                for r in rows
            ],
        )
        conn.execute(
            """
            INSERT INTO shap_runs (
                run_id, model_name, n_features, n_samples, algorithm,
                elapsed_sec, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_name,
                len(rows),
                int(rows[0]["n_samples"]) if rows else 0,
                str(algorithm or ""),
                float(elapsed_sec) if elapsed_sec is not None else None,
                now,
            ),
        )
        _patch_profiles_from_shap_rows(conn, run_id, model_name, rows, now=now)
    return len(rows)


def run_shap_analysis(
    data_dir: str,
    run_id: str,
    dataset: dict[str, Any],
    model_name: str,
    *,
    force: bool = False,
    sample_size: int = DEFAULT_SHAP_SAMPLE,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Full SHAP module lifecycle. Reuses results unless force=True."""
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("Select a trained model for SHAP Analysis")

    def _tick(msg: str, elapsed: float) -> None:
        if progress:
            progress(msg, elapsed)

    if not force and shap_already_computed(data_dir, run_id, model_name):
        rows = load_shap_results(data_dir, run_id, model_name)
        rehydrate_shap_into_profiles(data_dir, run_id, model_name=model_name)
        set_module_status(
            data_dir,
            run_id,
            "shap",
            STATUS_COMPLETED,
            message=f"Reused SHAP · {model_name} · {len(rows)} features",
            finished_at=_now_iso(),
        )
        return {
            "reused": True,
            "model_name": model_name,
            "features": len(rows),
            "rows": rows[:50],
            "elapsed_sec": 0.0,
            "message": f"SHAP done (reused) · {model_name} · {len(rows)} features",
        }

    path = resolve_parquet_path(data_dir, dataset)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path!r}")

    started = _now_iso()
    t0 = time.perf_counter()
    set_module_status(
        data_dir,
        run_id,
        "shap",
        STATUS_RUNNING,
        started_at=started,
        message=f"SHAP starting · {model_name}",
    )
    _tick(f"Loading model {model_name}…", 0.0)

    try:
        bundle = _resolve_model_bundle(data_dir, model_name)
        _tick(
            f"Model loaded · {bundle.get('algorithm')} · "
            f"{len(bundle['features'])} features",
            max(time.perf_counter() - t0, 0.0),
        )
        rows = compute_shap_for_dataset(
            path,
            bundle["model"],
            bundle["features"],
            sample_size=sample_size,
            progress=progress,
            t0=t0,
        )
        elapsed = max(time.perf_counter() - t0, 0.0)
        _tick("Persisting SHAP to analysis.db…", elapsed)
        n = persist_shap_results(
            data_dir,
            run_id,
            model_name,
            rows,
            algorithm=str(bundle.get("algorithm") or ""),
            elapsed_sec=elapsed,
        )
        elapsed = max(time.perf_counter() - t0, 0.0)
        done_msg = (
            f"SHAP done · {model_name} · {n} features · "
            f"sample={rows[0]['n_samples'] if rows else 0} · {elapsed:.1f}s"
        )
        set_module_status(
            data_dir,
            run_id,
            "shap",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=done_msg,
        )
        _tick(done_msg, elapsed)
        return {
            "reused": False,
            "model_name": model_name,
            "features": n,
            "elapsed_sec": elapsed,
            "algorithm": bundle.get("algorithm"),
            "rows": rows[:50],
            "message": done_msg,
        }
    except Exception as exc:
        elapsed = max(time.perf_counter() - t0, 0.0)
        set_module_status(
            data_dir,
            run_id,
            "shap",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=str(exc),
        )
        raise


__all__ = [
    "DEFAULT_SHAP_SAMPLE",
    "SHAP_PRE_SELECTION_WARNING",
    "assess_model_feature_selection",
    "compute_shap_for_dataset",
    "default_shap_model",
    "ensure_shap_schema",
    "list_shap_models",
    "load_shap_results",
    "persist_shap_results",
    "rehydrate_shap_into_profiles",
    "run_shap_analysis",
    "shap_already_computed",
]
