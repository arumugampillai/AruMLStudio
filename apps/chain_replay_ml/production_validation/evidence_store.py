"""SQLite Feature Recommendation Evidence Store & Dual Projections Engine."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from .dataset_context import DatasetContext, LEGACY_UNKNOWN_CONTEXT_ID
from .recommendation_policy import (
    RecommendationPolicy,
    compute_evidence_score,
    load_recommendation_policy,
)

EVIDENCE_DB_NAME = "feature_recommendation_evidence.db"


def evidence_db_path(data_dir: str) -> str:
    return os.path.join(data_dir, EVIDENCE_DB_NAME)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(data_dir: str) -> sqlite3.Connection:
    path = evidence_db_path(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dataset_contexts (
                context_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                sampling_interval_sec INTEGER NOT NULL,
                sampling_label TEXT NOT NULL,
                sliding_window TEXT NOT NULL DEFAULT 'standard',
                feature_project_id TEXT NOT NULL DEFAULT 'all',
                context_key TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ctx_market_interval 
                ON dataset_contexts(market, sampling_interval_sec);

            CREATE TABLE IF NOT EXISTS recommendation_evidence (
                evidence_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_source TEXT NOT NULL CHECK (feature_source IN ('registry', 'base_pipeline', 'experimental')),
                feature_identity_key TEXT NOT NULL,
                pipeline_id TEXT,
                pipeline_snapshot_id TEXT,
                recommendation TEXT NOT NULL CHECK (recommendation IN ('KEEP', 'WATCH', 'REMOVE')),
                validation_run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                target_column TEXT,
                holdout_rank INTEGER,
                unseen_rank INTEGER,
                rank_change INTEGER,
                relative_imp_drop REAL,
                drift_severity INTEGER,
                evidence_detail_json TEXT,
                run_timestamp TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id)
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_lookup 
                ON recommendation_evidence(context_id, feature_name, run_timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_lineage 
                ON recommendation_evidence(context_id, feature_identity_key);
            CREATE INDEX IF NOT EXISTS idx_evidence_run 
                ON recommendation_evidence(validation_run_id, model_name);

            CREATE TABLE IF NOT EXISTS feature_context_summary (
                summary_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                feature_source TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                total_runs INTEGER NOT NULL DEFAULT 0,
                keep_runs INTEGER NOT NULL DEFAULT 0,
                watch_runs INTEGER NOT NULL DEFAULT 0,
                remove_runs INTEGER NOT NULL DEFAULT 0,
                unique_models_count INTEGER NOT NULL DEFAULT 0,
                consecutive_remove_count INTEGER NOT NULL DEFAULT 0,
                consecutive_keep_count INTEGER NOT NULL DEFAULT 0,
                evidence_score REAL NOT NULL DEFAULT 0.0,
                priority_rank INTEGER,
                lifecycle_status TEXT NOT NULL DEFAULT 'active' 
                    CHECK (lifecycle_status IN ('active', 'held', 'blocked', 'alert')),
                last_recommendation TEXT,
                last_validated_at TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id),
                UNIQUE(context_id, feature_source, feature_name)
            );
            CREATE INDEX IF NOT EXISTS idx_ctx_summary_gate 
                ON feature_context_summary(context_id, feature_name, lifecycle_status);
            CREATE INDEX IF NOT EXISTS idx_ctx_summary_ranking 
                ON feature_context_summary(context_id, feature_source, evidence_score DESC);

            CREATE TABLE IF NOT EXISTS experimental_lineage_summary (
                lineage_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                pipeline_id TEXT NOT NULL,
                pipeline_snapshot_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_identity_key TEXT NOT NULL,
                total_runs INTEGER NOT NULL DEFAULT 0,
                keep_runs INTEGER NOT NULL DEFAULT 0,
                watch_runs INTEGER NOT NULL DEFAULT 0,
                remove_runs INTEGER NOT NULL DEFAULT 0,
                unique_models_count INTEGER NOT NULL DEFAULT 0,
                consecutive_keep_count INTEGER NOT NULL DEFAULT 0,
                consecutive_remove_count INTEGER NOT NULL DEFAULT 0,
                lineage_evidence_score REAL NOT NULL DEFAULT 0.0,
                lifecycle_status TEXT NOT NULL DEFAULT 'active' 
                    CHECK (lifecycle_status IN ('active', 'held', 'blocked', 'promotion_candidate')),
                last_recommendation TEXT,
                last_validated_at TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id),
                UNIQUE(context_id, pipeline_id, pipeline_snapshot_id, feature_name)
            );
            CREATE INDEX IF NOT EXISTS idx_lineage_summary_prom 
                ON experimental_lineage_summary(context_id, lifecycle_status, lineage_evidence_score DESC);

            CREATE TABLE IF NOT EXISTS migration_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def ensure_dataset_context(conn: sqlite3.Connection, context: DatasetContext) -> None:
    conn.execute(
        """
        INSERT INTO dataset_contexts (
            context_id, market, sampling_interval_sec, sampling_label,
            sliding_window, feature_project_id, context_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(context_id) DO UPDATE SET
            market=excluded.market,
            sampling_interval_sec=excluded.sampling_interval_sec,
            sampling_label=excluded.sampling_label,
            sliding_window=excluded.sliding_window,
            feature_project_id=excluded.feature_project_id,
            context_key=excluded.context_key;
        """,
        (
            context.context_id,
            context.market,
            context.sampling_interval_sec,
            context.sampling_label,
            context.sliding_window,
            context.feature_project_id,
            context.context_key,
            context.created_at,
        ),
    )


def compute_feature_identity_key(
    feature_source: str,
    feature_name: str,
    pipeline_id: str | None = None,
    pipeline_snapshot_id: str | None = None,
) -> str:
    src = str(feature_source or "registry").strip().lower()
    fn = str(feature_name or "").strip()
    if src == "registry":
        return f"registry:{fn}"
    if src == "base_pipeline":
        return f"base_pipeline:{fn}"
    pid = str(pipeline_id or "").strip().upper() or "PL_NONE"
    snap = str(pipeline_snapshot_id or "").strip() or "snap_none"
    return f"exp:{fn}:{pid}:{snap}"


def append_validation_evidence(
    conn: sqlite3.Connection,
    *,
    context: DatasetContext,
    evidence_rows: list[dict[str, Any]],
    policy: RecommendationPolicy | None = None,
) -> dict[str, int]:
    """Single authoritative write path: appends raw evidence and updates dual projections atomically."""
    pol = policy or RecommendationPolicy()
    inserted_count = 0
    ensure_dataset_context(conn, context)

    with conn:
        for row in evidence_rows:
            ev_id = str(row.get("evidence_id") or f"ev_{uuid.uuid4().hex[:12]}")
            feat_name = str(row.get("feature_name") or row.get("feature") or "").strip()
            if not feat_name:
                continue
            feat_source = str(row.get("feature_source") or "registry").strip().lower()
            if feat_source not in ("registry", "base_pipeline", "experimental"):
                feat_source = "registry"

            pid = str(row.get("pipeline_id") or "").strip().upper() or None
            snap_id = str(row.get("pipeline_snapshot_id") or "").strip() or None
            identity_key = str(
                row.get("feature_identity_key")
                or compute_feature_identity_key(feat_source, feat_name, pid, snap_id)
            )

            rec = str(row.get("recommendation") or "").strip().upper()
            if rec not in ("KEEP", "WATCH", "REMOVE"):
                rec = "KEEP"

            val_run_id = str(row.get("validation_run_id") or row.get("production_validation_run_id") or "")
            model_name = str(row.get("model_name") or "")
            target_col = str(row.get("target_column") or row.get("target") or "")
            ho_rank = int(row.get("holdout_rank")) if row.get("holdout_rank") is not None else None
            un_rank = int(row.get("unseen_rank")) if row.get("unseen_rank") is not None else None
            rank_chg = int(row.get("rank_change")) if row.get("rank_change") is not None else None
            rel_drop = float(row.get("relative_imp_drop")) if row.get("relative_imp_drop") is not None else None
            drift_sev = int(row.get("drift_severity")) if row.get("drift_severity") is not None else None
            detail = row.get("evidence_detail_json")
            if isinstance(detail, dict):
                detail = json.dumps(detail)
            elif detail is None and row.get("recommendation_detail"):
                detail = json.dumps(row.get("recommendation_detail"))
            timestamp = str(row.get("run_timestamp") or row.get("generated_date") or _utc_now())

            conn.execute(
                """
                INSERT INTO recommendation_evidence (
                    evidence_id, context_id, feature_name, feature_source,
                    feature_identity_key, pipeline_id, pipeline_snapshot_id,
                    recommendation, validation_run_id, model_name, target_column,
                    holdout_rank, unseen_rank, rank_change, relative_imp_drop,
                    drift_severity, evidence_detail_json, run_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO NOTHING;
                """,
                (
                    ev_id,
                    context.context_id,
                    feat_name,
                    feat_source,
                    identity_key,
                    pid,
                    snap_id,
                    rec,
                    val_run_id,
                    model_name,
                    target_col,
                    ho_rank,
                    un_rank,
                    rank_chg,
                    rel_drop,
                    drift_sev,
                    detail,
                    timestamp,
                ),
            )
            inserted_count += 1

            # Update Projection 1: feature_context_summary
            _update_context_summary_for_feature(
                conn,
                context_id=context.context_id,
                feature_source=feat_source,
                feature_name=feat_name,
                policy=pol,
            )

            # Update Projection 2: experimental_lineage_summary (only for experimental features)
            if feat_source == "experimental" and pid and snap_id:
                _update_lineage_summary_for_candidate(
                    conn,
                    context_id=context.context_id,
                    pipeline_id=pid,
                    pipeline_snapshot_id=snap_id,
                    feature_name=feat_name,
                    feature_identity_key=identity_key,
                    policy=pol,
                )

    return {"inserted": inserted_count}


def _update_context_summary_for_feature(
    conn: sqlite3.Connection,
    *,
    context_id: str,
    feature_source: str,
    feature_name: str,
    policy: RecommendationPolicy,
) -> None:
    # Query all historical evidence for this feature in this context ordered chronologically
    cur = conn.execute(
        """
        SELECT recommendation, model_name, run_timestamp
        FROM recommendation_evidence
        WHERE context_id = ? AND feature_source = ? AND feature_name = ?
        ORDER BY run_timestamp ASC
        """,
        (context_id, feature_source, feature_name),
    )
    rows = cur.fetchall()
    if not rows:
        return

    total_runs = len(rows)
    keep_runs = sum(1 for r in rows if r["recommendation"] == "KEEP")
    watch_runs = sum(1 for r in rows if r["recommendation"] == "WATCH")
    remove_runs = sum(1 for r in rows if r["recommendation"] == "REMOVE")

    unique_models = set(r["model_name"] for r in rows if r["model_name"])
    unique_models_count = len(unique_models)

    # Calculate model counts by recommendation
    remove_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "REMOVE" and r["model_name"]))
    keep_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "KEEP" and r["model_name"]))
    watch_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "WATCH" and r["model_name"]))

    # Calculate current streaks from end of chronological sequence
    consecutive_removes = 0
    for r in reversed(rows):
        if r["recommendation"] == "REMOVE":
            consecutive_removes += 1
        else:
            break

    consecutive_keeps = 0
    for r in reversed(rows):
        if r["recommendation"] == "KEEP":
            consecutive_keeps += 1
        else:
            break

    score = compute_evidence_score(
        keep_models=keep_models_count,
        remove_models=remove_models_count,
        watch_models=watch_models_count,
        consecutive_keeps=consecutive_keeps,
        consecutive_removes=consecutive_removes,
        policy=policy.scoring,
    )

    last_rec = rows[-1]["recommendation"]
    last_val_at = rows[-1]["run_timestamp"]

    # Determine lifecycle_status based on feature_source and policy
    if feature_source == "experimental":
        exp_pol = policy.experimental_lifecycle
        if (
            consecutive_removes >= exp_pol.remove_block_consecutive_threshold
            or remove_runs >= exp_pol.remove_block_total_threshold
        ):
            status = "blocked"
        elif last_rec == "WATCH":
            status = "held"
        else:
            status = "active"
    elif feature_source == "base_pipeline":
        base_pol = policy.base_pipeline
        if score <= base_pol.negative_alert_score_threshold:
            status = "alert"
        elif last_rec == "WATCH":
            status = "held"
        else:
            status = "active"
    else:  # registry
        reg_pol = policy.feature_registry
        if (
            remove_runs >= reg_pol.remove_audit_alert_threshold
            and remove_models_count >= reg_pol.min_unique_models
        ):
            status = "alert"
        elif last_rec == "WATCH":
            status = "held"
        else:
            status = "active"

    summary_id = f"sum_{context_id}_{feature_source}_{feature_name}"
    conn.execute(
        """
        INSERT INTO feature_context_summary (
            summary_id, context_id, feature_source, feature_name,
            total_runs, keep_runs, watch_runs, remove_runs,
            unique_models_count, consecutive_remove_count, consecutive_keep_count,
            evidence_score, priority_rank, lifecycle_status,
            last_recommendation, last_validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        ON CONFLICT(context_id, feature_source, feature_name) DO UPDATE SET
            total_runs=excluded.total_runs,
            keep_runs=excluded.keep_runs,
            watch_runs=excluded.watch_runs,
            remove_runs=excluded.remove_runs,
            unique_models_count=excluded.unique_models_count,
            consecutive_remove_count=excluded.consecutive_remove_count,
            consecutive_keep_count=excluded.consecutive_keep_count,
            evidence_score=excluded.evidence_score,
            lifecycle_status=excluded.lifecycle_status,
            last_recommendation=excluded.last_recommendation,
            last_validated_at=excluded.last_validated_at;
        """,
        (
            summary_id,
            context_id,
            feature_source,
            feature_name,
            total_runs,
            keep_runs,
            watch_runs,
            remove_runs,
            unique_models_count,
            consecutive_removes,
            consecutive_keeps,
            score,
            status,
            last_rec,
            last_val_at,
        ),
    )


def _update_lineage_summary_for_candidate(
    conn: sqlite3.Connection,
    *,
    context_id: str,
    pipeline_id: str,
    pipeline_snapshot_id: str,
    feature_name: str,
    feature_identity_key: str,
    policy: RecommendationPolicy,
) -> None:
    cur = conn.execute(
        """
        SELECT recommendation, model_name, run_timestamp
        FROM recommendation_evidence
        WHERE context_id = ? AND pipeline_id = ? AND pipeline_snapshot_id = ? AND feature_name = ?
        ORDER BY run_timestamp ASC
        """,
        (context_id, pipeline_id, pipeline_snapshot_id, feature_name),
    )
    rows = cur.fetchall()
    if not rows:
        return

    total_runs = len(rows)
    keep_runs = sum(1 for r in rows if r["recommendation"] == "KEEP")
    watch_runs = sum(1 for r in rows if r["recommendation"] == "WATCH")
    remove_runs = sum(1 for r in rows if r["recommendation"] == "REMOVE")

    unique_models = set(r["model_name"] for r in rows if r["model_name"])
    unique_models_count = len(unique_models)

    remove_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "REMOVE" and r["model_name"]))
    keep_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "KEEP" and r["model_name"]))
    watch_models_count = len(set(r["model_name"] for r in rows if r["recommendation"] == "WATCH" and r["model_name"]))

    consecutive_removes = 0
    for r in reversed(rows):
        if r["recommendation"] == "REMOVE":
            consecutive_removes += 1
        else:
            break

    consecutive_keeps = 0
    for r in reversed(rows):
        if r["recommendation"] == "KEEP":
            consecutive_keeps += 1
        else:
            break

    score = compute_evidence_score(
        keep_models=keep_models_count,
        remove_models=remove_models_count,
        watch_models=watch_models_count,
        consecutive_keeps=consecutive_keeps,
        consecutive_removes=consecutive_removes,
        policy=policy.scoring,
    )

    last_rec = rows[-1]["recommendation"]
    last_val_at = rows[-1]["run_timestamp"]

    exp_pol = policy.experimental_lifecycle
    if (
        consecutive_removes >= exp_pol.remove_block_consecutive_threshold
        or remove_runs >= exp_pol.remove_block_total_threshold
    ):
        status = "blocked"
    elif (
        consecutive_keeps >= exp_pol.promotion_candidate_consecutive_keep
        and score >= exp_pol.promotion_candidate_min_score
        and unique_models_count >= exp_pol.min_unique_models
    ):
        status = "promotion_candidate"
    elif last_rec == "WATCH":
        status = "held"
    else:
        status = "active"

    lineage_id = f"lin_{context_id}_{pipeline_id}_{pipeline_snapshot_id}_{feature_name}"
    conn.execute(
        """
        INSERT INTO experimental_lineage_summary (
            lineage_id, context_id, pipeline_id, pipeline_snapshot_id,
            feature_name, feature_identity_key,
            total_runs, keep_runs, watch_runs, remove_runs,
            unique_models_count, consecutive_keep_count, consecutive_remove_count,
            lineage_evidence_score, lifecycle_status,
            last_recommendation, last_validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(context_id, pipeline_id, pipeline_snapshot_id, feature_name) DO UPDATE SET
            total_runs=excluded.total_runs,
            keep_runs=excluded.keep_runs,
            watch_runs=excluded.watch_runs,
            remove_runs=excluded.remove_runs,
            unique_models_count=excluded.unique_models_count,
            consecutive_keep_count=excluded.consecutive_keep_count,
            consecutive_remove_count=excluded.consecutive_remove_count,
            lineage_evidence_score=excluded.lineage_evidence_score,
            lifecycle_status=excluded.lifecycle_status,
            last_recommendation=excluded.last_recommendation,
            last_validated_at=excluded.last_validated_at;
        """,
        (
            lineage_id,
            context_id,
            pipeline_id,
            pipeline_snapshot_id,
            feature_name,
            feature_identity_key,
            total_runs,
            keep_runs,
            watch_runs,
            remove_runs,
            unique_models_count,
            consecutive_keeps,
            consecutive_removes,
            score,
            status,
            last_rec,
            last_val_at,
        ),
    )


def rebuild_all_projections(
    conn: sqlite3.Connection,
    policy: RecommendationPolicy | None = None,
) -> dict[str, int]:
    """Deterministic disaster-recovery rebuild of all materialized summary projections."""
    pol = policy or RecommendationPolicy()
    with conn:
        conn.execute("DELETE FROM feature_context_summary;")
        conn.execute("DELETE FROM experimental_lineage_summary;")

        # Fetch all distinct (context_id, feature_source, feature_name)
        cur_ctx = conn.execute(
            """
            SELECT DISTINCT context_id, feature_source, feature_name
            FROM recommendation_evidence
            """
        )
        ctx_targets = cur_ctx.fetchall()
        for t in ctx_targets:
            _update_context_summary_for_feature(
                conn,
                context_id=t["context_id"],
                feature_source=t["feature_source"],
                feature_name=t["feature_name"],
                policy=pol,
            )

        # Fetch all distinct experimental lineages
        cur_lin = conn.execute(
            """
            SELECT DISTINCT context_id, pipeline_id, pipeline_snapshot_id, feature_name, feature_identity_key
            FROM recommendation_evidence
            WHERE feature_source = 'experimental' AND pipeline_id IS NOT NULL AND pipeline_snapshot_id IS NOT NULL
            """
        )
        lin_targets = cur_lin.fetchall()
        for l in lin_targets:
            _update_lineage_summary_for_candidate(
                conn,
                context_id=l["context_id"],
                pipeline_id=l["pipeline_id"],
                pipeline_snapshot_id=l["pipeline_snapshot_id"],
                feature_name=l["feature_name"],
                feature_identity_key=l["feature_identity_key"],
                policy=pol,
            )

    return {
        "context_summaries_rebuilt": len(ctx_targets),
        "lineage_summaries_rebuilt": len(lin_targets),
    }


def query_blocked_candidates(
    conn: sqlite3.Connection,
    *,
    context_id: str,
    candidate_names: Iterable[str],
) -> set[str]:
    """Query Pre-Training Elimination Gate for blocked experimental features."""
    names = [str(n).strip() for n in candidate_names if str(n).strip()]
    if not names or not context_id or context_id == LEGACY_UNKNOWN_CONTEXT_ID:
        return set()

    placeholders = ",".join("?" for _ in names)
    cur = conn.execute(
        f"""
        SELECT feature_name
        FROM feature_context_summary
        WHERE context_id = ?
          AND feature_source = 'experimental'
          AND lifecycle_status = 'blocked'
          AND feature_name IN ({placeholders})
        """,
        [context_id, *names],
    )
    return set(row["feature_name"] for row in cur.fetchall())


def get_feature_context_summaries(
    conn: sqlite3.Connection,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    if context_id:
        cur = conn.execute(
            """
            SELECT * FROM feature_context_summary
            WHERE context_id = ?
            ORDER BY feature_source, evidence_score DESC
            """,
            (context_id,),
        )
    else:
        cur = conn.execute(
            """
            SELECT * FROM feature_context_summary
            ORDER BY context_id, feature_source, evidence_score DESC
            """
        )
    return [dict(row) for row in cur.fetchall()]


def get_experimental_lineage_summaries(
    conn: sqlite3.Connection,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    if context_id:
        cur = conn.execute(
            """
            SELECT * FROM experimental_lineage_summary
            WHERE context_id = ?
            ORDER BY lifecycle_status, lineage_evidence_score DESC
            """,
            (context_id,),
        )
    else:
        cur = conn.execute(
            """
            SELECT * FROM experimental_lineage_summary
            ORDER BY context_id, lifecycle_status, lineage_evidence_score DESC
            """
        )
    return [dict(row) for row in cur.fetchall()]
