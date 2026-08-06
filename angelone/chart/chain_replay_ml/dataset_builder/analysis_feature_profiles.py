"""Feature Profiles — Phase 2.2 Research Lab convergence record.

Read-only. Built from analysis parquet + analysis.db correlation/insights +
registry/pipeline metadata. Future MI/SHAP/VIF/Permutation enrich the same rows.
Never rebuilds datasets, never deletes features.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

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
    ROLE_METADATA,
    ROLE_PREDICTOR,
    ROLE_TARGET,
    classify_feature_role,
    explorer_columns,
)

# Re-export for older imports
from .analysis_feature_roles import META_COLS  # noqa: F401


def ensure_feature_profiles_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_profiles (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            category TEXT,
            source TEXT,
            transformation TEXT,
            parents_json TEXT,
            children_json TEXT,
            cluster_id TEXT,
            cluster_members INTEGER,
            representative TEXT,
            coverage REAL,
            null_pct REAL,
            warmup_pct REAL,
            unique_values INTEGER,
            mean REAL,
            std_dev REAL,
            min_val REAL,
            max_val REAL,
            top_corr_json TEXT,
            recommendation TEXT,
            reason TEXT,
            mi_score REAL,
            shap_importance REAL,
            shap_rank INTEGER,
            permutation_importance REAL,
            permutation_rank INTEGER,
            vif REAL,
            feature_score REAL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, feature_name)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_profiles_dataset
        ON feature_profiles(dataset_id, feature_name)
        """
    )
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(feature_profiles)").fetchall()
    }
    for col, typ in (
        ("mi_rank", "INTEGER"),
        ("mi_percentile", "REAL"),
        ("mi_interpretation", "TEXT"),
        ("mi_target", "TEXT"),
        ("shap_percentile", "REAL"),
        ("shap_model", "TEXT"),
        ("feature_role", "TEXT"),
        ("permutation_percentile", "REAL"),
        ("permutation_interpretation", "TEXT"),
        ("permutation_model", "TEXT"),
        ("permutation_target", "TEXT"),
        ("permutation_delta_rmse", "REAL"),
        ("permutation_delta_mae", "REAL"),
        ("permutation_baseline_rmse", "REAL"),
        ("permutation_permuted_rmse", "REAL"),
        ("rating_action", "TEXT"),
        ("rating_confidence", "TEXT"),
        ("rating_reason", "TEXT"),
        ("rating_score", "REAL"),
        ("rating_stars", "TEXT"),
        ("rating_peer", "TEXT"),
        ("rating_abs_corr", "REAL"),
        ("rating_mi_pct", "REAL"),
        ("rating_shap_pct", "REAL"),
        ("rating_perm_pct", "REAL"),
        ("rating_model", "TEXT"),
        ("rating_target", "TEXT"),
        ("rating_stage", "TEXT"),
        ("rating_family_id", "TEXT"),
        ("rating_family_label", "TEXT"),
        ("validation_score", "REAL"),
        ("validation_stars", "TEXT"),
        ("validation_action", "TEXT"),
        ("validation_confidence", "TEXT"),
        ("validation_reason", "TEXT"),
        ("validation_shap_pct", "REAL"),
        ("validation_model", "TEXT"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE feature_profiles ADD COLUMN {col} {typ}")


def _is_feature_col(name: str) -> bool:
    """Explorer-eligible columns (predictors + targets + labels)."""
    return classify_feature_role(name) != ROLE_METADATA


def _warmup_like(name: str) -> bool:
    n = str(name or "").lower()
    return any(
        tok in n
        for tok in (
            "_lag_",
            "_diff_",
            "_return_",
            "_change_",
            "_zscore_",
            "zscore_",
            "_slope_",
            "_roll_",
        )
    )


def _category_for(name: str) -> str:
    try:
        from .feature_domains import primary_domain_label

        return primary_domain_label(name)
    except Exception:
        pass
    from .analysis_correlation import _family_label

    return _family_label(name)


def _source_and_transform(
    name: str,
    *,
    in_lineage: bool = False,
) -> tuple[str, str]:
    from .feature_migration import PIPELINE_OWNED_GENERATORS, is_pipeline_owned

    if is_pipeline_owned(name) or in_lineage or _warmup_like(name):
        gen = str(PIPELINE_OWNED_GENERATORS.get(name) or "pipeline")
        return "Pipeline", gen
    return "Registry", "registry"


def _load_dataset_sidecar(parquet_path: str) -> dict[str, Any]:
    base, _ = os.path.splitext(parquet_path)
    json_path = base + ".json"
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _lineage_maps(transformations: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return parents_by_feature, children_by_feature from transformation config."""
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    if isinstance(transformations, list):
        cfg: dict[str, Any] = {"transformations": transformations}
    elif isinstance(transformations, dict):
        cfg = (
            transformations
            if "transformations" in transformations
            else {"transformations": list(transformations.get("transformations") or [])}
        )
    else:
        return parents, children
    try:
        from .pipeline_no_null_report import build_pipeline_lineage_map

        lineage = build_pipeline_lineage_map(cfg)
        for child, info in lineage.items():
            pars = [str(p) for p in (info.get("parents") or []) if str(p).strip()]
            parents[str(child)] = pars
            for p in pars:
                children.setdefault(p, []).append(str(child))
    except Exception:
        pass
    return parents, children


def _top_correlated(
    data_dir: str,
    run_id: str,
    feature: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_a, feature_b, correlation
            FROM correlation
            WHERE run_id = ?
              AND (feature_a = ? OR feature_b = ?)
            ORDER BY ABS(correlation) DESC
            LIMIT ?
            """,
            (run_id, feature, feature, int(limit)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        a, b = str(r["feature_a"]), str(r["feature_b"])
        other = b if a == feature else a
        out.append({"feature": other, "correlation": float(r["correlation"])})
    return out


def build_feature_profiles(
    data_dir: str,
    run_id: str,
    dataset: dict[str, Any],
    *,
    max_rows_for_stats: int | None = None,
) -> dict[str, Any]:
    """Build/replace feature_profiles for a run from parquet + correlation/insights."""
    import pandas as pd

    from .analysis_correlation_insights import load_correlation_insights
    from .analysis_correlation import load_clusters

    path = resolve_parquet_path(data_dir, dataset)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path!r}")

    dataset_id = str(dataset.get("dataset_id") or dataset.get("name") or "")
    sidecar = _load_dataset_sidecar(path)
    parents_map, children_map = _lineage_maps(sidecar.get("transformations"))

    df = pd.read_parquet(path)
    if max_rows_for_stats is not None and len(df) > int(max_rows_for_stats):
        # Stats still meaningful on a sample for huge frames; coverage uses full if cheap
        sample = df.sample(n=int(max_rows_for_stats), random_state=42)
    else:
        sample = df

    features = explorer_columns([str(c) for c in df.columns], sidecar=sidecar)
    n_full = len(df)
    now = _now_iso()

    # Cluster + insight lookup
    cluster_by_feat: dict[str, dict[str, Any]] = {}
    for c in load_clusters(data_dir, run_id):
        for m in c.get("members") or []:
            cluster_by_feat[str(m)] = c
    insight_by_cluster = {
        str(i.get("cluster")): i for i in load_correlation_insights(data_dir, run_id)
    }

    rows_out: list[tuple[Any, ...]] = []
    for feat in features:
        role = classify_feature_role(feat, sidecar=sidecar)
        series_full = pd.to_numeric(df[feat], errors="coerce")
        series = pd.to_numeric(sample[feat], errors="coerce") if sample is not df else series_full
        null_n = int(series_full.isna().sum())
        null_pct = (100.0 * null_n / n_full) if n_full else 0.0
        coverage = max(0.0, 100.0 - null_pct)
        warmup_pct = float(null_pct) if _warmup_like(feat) else 0.0
        valid = series_full.dropna()
        unique_n = int(valid.nunique()) if len(valid) else 0
        mean = float(valid.mean()) if len(valid) else None
        std = float(valid.std(ddof=0)) if len(valid) else None
        vmin = float(valid.min()) if len(valid) else None
        vmax = float(valid.max()) if len(valid) else None

        source, transform = _source_and_transform(
            feat,
            in_lineage=(feat in parents_map) or (feat in children_map),
        )
        category = _category_for(feat)
        if role == ROLE_TARGET:
            category = "Target"
            source = "Target"
            transform = "prediction_target"
            cluster_id = None
            members_n = None
            rep = None
            recommendation = "Excluded"
            reason = (
                "Prediction target — available for MI / model target selection only. "
                "Not included in feature evaluation."
            )
            top_corr = []
        elif role == ROLE_LABEL:
            category = "Label"
            source = "Label"
            transform = "classification_label"
            cluster_id = None
            members_n = None
            rep = None
            recommendation = "Excluded"
            reason = (
                "Supervised learning label — excluded from feature scoring. "
                "Available as a classification target only."
            )
            top_corr = []
        else:
            cl = cluster_by_feat.get(feat) or {}
            cluster_id = str(cl.get("cluster") or "") or None
            rep = str(cl.get("representative") or "") or None
            members_n = int(cl.get("size") or len(cl.get("members") or []) or 0) or None
            insight = insight_by_cluster.get(cluster_id or "") or {}
            recommendation = str(insight.get("recommendation") or "Keep")
            reason = str(insight.get("reason") or "")
            if not reason and not cluster_id:
                reason = (
                    "No multi-member correlation cluster. Keep for now — "
                    "wait for MI + Permutation / Discovery Rating before "
                    "any removal decisions."
                )
            elif recommendation == "Review" and "Discovery Rating" not in reason:
                reason = (
                    reason + " Continue with MI → Permutation → Discovery Rating."
                ).strip()
            top_corr = _top_correlated(data_dir, run_id, feat, limit=8)

        parents = parents_map.get(feat) or []
        children = children_map.get(feat) or []

        rows_out.append(
            (
                run_id,
                dataset_id,
                feat,
                category,
                source,
                transform,
                json.dumps(parents, separators=(",", ":")),
                json.dumps(children, separators=(",", ":")),
                cluster_id,
                members_n,
                rep,
                coverage,
                null_pct,
                warmup_pct,
                unique_n,
                mean,
                std,
                vmin,
                vmax,
                json.dumps(top_corr, separators=(",", ":")),
                recommendation,
                reason,
                None,  # mi_score
                None,  # shap_importance
                None,  # shap_rank
                None,  # permutation_importance
                None,  # permutation_rank
                None,  # vif
                None,  # feature_score
                now,
                role,
            )
        )

    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        conn.execute("DELETE FROM feature_profiles WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO feature_profiles (
                run_id, dataset_id, feature_name, category, source, transformation,
                parents_json, children_json, cluster_id, cluster_members, representative,
                coverage, null_pct, warmup_pct, unique_values, mean, std_dev,
                min_val, max_val, top_corr_json, recommendation, reason,
                mi_score, shap_importance, shap_rank, permutation_importance,
                permutation_rank, vif, feature_score, updated_at, feature_role
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            rows_out,
        )

    # Preserve MI / SHAP computed earlier (profiles wipe research metrics).
    try:
        from .analysis_mutual_information import rehydrate_mi_into_profiles

        rehydrate_mi_into_profiles(data_dir, run_id)
    except Exception:
        pass
    try:
        from .analysis_shap import rehydrate_shap_into_profiles

        rehydrate_shap_into_profiles(data_dir, run_id)
    except Exception:
        pass
    try:
        from .analysis_permutation import rehydrate_permutation_into_profiles

        rehydrate_permutation_into_profiles(data_dir, run_id)
    except Exception:
        pass
    # Re-apply Discovery Rating (never SHAP). Validation rating is separate.
    try:
        from .analysis_lab_store import STATUS_COMPLETED, module_statuses
        from .analysis_feature_rating import (
            STAGE_DISCOVERY,
            STAGE_VALIDATION,
            run_feature_rating,
            shap_module_completed,
        )

        st = {
            str(r.get("module_id") or ""): str(r.get("status") or "")
            for r in module_statuses(data_dir, run_id)
        }
        if st.get("feature_scorecard") == STATUS_COMPLETED:
            run_feature_rating(data_dir, run_id, stage=STAGE_DISCOVERY)
        if shap_module_completed(data_dir, run_id):
            try:
                run_feature_rating(data_dir, run_id, stage=STAGE_VALIDATION)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "features": len(rows_out),
        "dataset_id": dataset_id,
        "run_id": run_id,
    }


def list_profile_features(data_dir: str, run_id: str) -> list[str]:
    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        rows = conn.execute(
            """
            SELECT feature_name FROM feature_profiles
            WHERE run_id = ?
            ORDER BY feature_name
            """,
            (run_id,),
        ).fetchall()
        return [str(r["feature_name"]) for r in rows]


def load_feature_profile(
    data_dir: str,
    run_id: str,
    feature_name: str,
) -> dict[str, Any] | None:
    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        row = conn.execute(
            """
            SELECT * FROM feature_profiles
            WHERE run_id = ? AND feature_name = ?
            """,
            (run_id, feature_name),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        for key, out_key in (
            ("parents_json", "parents"),
            ("children_json", "children"),
            ("top_corr_json", "top_correlated"),
        ):
            try:
                item[out_key] = json.loads(str(item.get(key) or "[]"))
            except Exception:
                item[out_key] = []
        return item


def load_feature_scorecard(
    data_dir: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Scorecard rows — predictors only (targets/labels/metadata excluded)."""
    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        rows = conn.execute(
            """
            SELECT feature_name, category, source, cluster_id, coverage, null_pct,
                   recommendation, reason, mi_score, mi_rank, mi_percentile,
                   mi_interpretation, mi_target, shap_importance, shap_rank,
                   shap_percentile, shap_model,
                   permutation_importance, permutation_rank, vif, feature_score,
                   representative, feature_role,
                   permutation_percentile, permutation_interpretation,
                   permutation_delta_rmse, permutation_delta_mae,
                   permutation_baseline_rmse, permutation_permuted_rmse,
                   rating_action, rating_confidence, rating_reason,
                   rating_score, rating_stars, rating_peer, rating_abs_corr,
                   rating_mi_pct, rating_perm_pct,
                   rating_model, rating_target, rating_stage,
                   rating_family_id, rating_family_label,
                   validation_score, validation_stars, validation_action,
                   validation_confidence, validation_reason,
                   validation_shap_pct, validation_model
            FROM feature_profiles
            WHERE run_id = ?
              AND (
                    feature_role = ?
                    OR feature_role IS NULL
                    OR feature_role = ''
              )
            ORDER BY
                CASE WHEN feature_score IS NULL THEN 1 ELSE 0 END,
                feature_score DESC,
                CASE rating_action
                    WHEN 'RETIRE CANDIDATE' THEN 0
                    WHEN 'MERGE CANDIDATE' THEN 1
                    WHEN 'REVIEW FAMILY' THEN 2
                    WHEN 'REVIEW' THEN 2
                    WHEN 'KEEP' THEN 3
                    ELSE 4
                END,
                CASE recommendation
                    WHEN 'Duplicate Candidate' THEN 0
                    WHEN 'Review' THEN 1
                    ELSE 2
                END,
                CASE WHEN mi_rank IS NULL THEN 1 ELSE 0 END,
                mi_rank ASC,
                CASE WHEN permutation_rank IS NULL THEN 1 ELSE 0 END,
                permutation_rank ASC,
                feature_name
            """,
            (run_id, ROLE_PREDICTOR),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            name = str(item.get("feature_name") or "")
            # Always re-classify by name so stale profile roles (e.g. market /
            # master_row_id stored as predictor) cannot leak into Scorecard.
            role = classify_feature_role(name)
            item["feature_role"] = role
            if role != ROLE_PREDICTOR:
                continue
            out.append(item)
        return out


def enrich_feature_profile(
    data_dir: str,
    run_id: str,
    feature_name: str,
    **fields: Any,
) -> None:
    """Future modules patch MI/SHAP/VIF/Permutation onto the same profile row."""
    allowed = {
        "mi_score",
        "shap_importance",
        "shap_rank",
        "permutation_importance",
        "permutation_rank",
        "vif",
        "feature_score",
        "recommendation",
        "reason",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    sets = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [_now_iso(), run_id, feature_name]
    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        conn.execute(
            f"""
            UPDATE feature_profiles
            SET {sets}, updated_at = ?
            WHERE run_id = ? AND feature_name = ?
            """,
            vals,
        )


def profiles_exist(data_dir: str, run_id: str) -> bool:
    with _AnalysisDb(data_dir) as conn:
        ensure_feature_profiles_schema(conn)
        row = conn.execute(
            "SELECT 1 FROM feature_profiles WHERE run_id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
        return row is not None


__all__ = [
    "build_feature_profiles",
    "enrich_feature_profile",
    "ensure_feature_profiles_schema",
    "list_profile_features",
    "load_feature_profile",
    "load_feature_scorecard",
    "profiles_exist",
]
