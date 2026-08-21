"""Isolated Persistence Layer for Autonomous Research Discovery Pipeline in analysis.db (Phase 2).

Manages storage, retrieval, and snapshot history for:
1. discovery_pipelines: Campaign-scoped Discovery Pipeline headers.
2. discovery_pipeline_features: Discovered feature provenance, formulas, and governance status.
3. discovery_pipeline_snapshots: Cryptographic, reproducible generation snapshots.

Invariants:
1. Isolated Sandbox: All operations strictly target `<data_dir>/analysis.db`.
2. Zero Registry Contamination: NEVER touches feature_registry_store.json or pipeline_registry_store.json.
3. Duplicate Prevention: Enforces UNIQUE(pipeline_id, formula_hash) for mathematical uniqueness.
4. Non-Destructive: REMOVE updates lifecycle_status without physical deletion of records.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Sequence

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    _utc_now_iso,
    compute_formula_hash,
)


DISCOVERY_PIPELINES_DDL = """
CREATE TABLE IF NOT EXISTS discovery_pipelines (
    pipeline_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    context_key TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_snapshot_hash TEXT NOT NULL,
    base_feature_count INTEGER NOT NULL,
    base_feature_names_json TEXT NOT NULL DEFAULT '[]',
    base_pipeline_id TEXT NOT NULL DEFAULT 'PL_0001',
    base_pipeline_snapshot_hash TEXT NOT NULL DEFAULT '',
    active_features_count INTEGER NOT NULL DEFAULT 0,
    total_generated_count INTEGER NOT NULL DEFAULT 0,
    parent_snapshot_hash TEXT NOT NULL DEFAULT '',
    current_snapshot_hash TEXT NOT NULL DEFAULT '',
    current_generation INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    budget_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dp_campaign 
    ON discovery_pipelines(campaign_id);

CREATE INDEX IF NOT EXISTS idx_dp_context 
    ON discovery_pipelines(context_key, status);

CREATE TABLE IF NOT EXISTS discovery_pipeline_features (
    feature_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    formula_expression TEXT NOT NULL,
    formula_hash TEXT NOT NULL,
    generator_strategy TEXT NOT NULL,
    parent_features_json TEXT NOT NULL,
    generation_discovered INTEGER NOT NULL,
    lifecycle_status TEXT NOT NULL DEFAULT 'candidate' 
        CHECK (lifecycle_status IN ('candidate', 'KEEP', 'WATCH', 'REMOVE', 'promoted', 'keep', 'watch', 'remove')),
    evidence_score REAL NOT NULL DEFAULT 0.0,
    total_evaluations INTEGER NOT NULL DEFAULT 0,
    holdout_rank INTEGER,
    relative_imp_drop REAL,
    drift_severity INTEGER NOT NULL DEFAULT 0,
    ks_statistic REAL NOT NULL DEFAULT 0.0,
    ks_pvalue REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (pipeline_id) REFERENCES discovery_pipelines(pipeline_id),
    UNIQUE(pipeline_id, formula_hash)
);

CREATE INDEX IF NOT EXISTS idx_dp_features_lookup 
    ON discovery_pipeline_features(pipeline_id, lifecycle_status);

CREATE INDEX IF NOT EXISTS idx_dp_features_score 
    ON discovery_pipeline_features(pipeline_id, evidence_score DESC);

CREATE TABLE IF NOT EXISTS discovery_pipeline_snapshots (
    snapshot_hash TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    generation_number INTEGER NOT NULL,
    active_feature_names_json TEXT NOT NULL,
    feature_count INTEGER NOT NULL,
    keep_count INTEGER NOT NULL DEFAULT 0,
    watch_count INTEGER NOT NULL DEFAULT 0,
    remove_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (pipeline_id) REFERENCES discovery_pipelines(pipeline_id)
);

CREATE INDEX IF NOT EXISTS idx_dp_snapshots_pipe 
    ON discovery_pipeline_snapshots(pipeline_id, generation_number DESC);
"""


def _ensure_discovery_pipeline_columns(conn: sqlite3.Connection) -> None:
    """Safe, non-destructive schema migration ensuring discovery_pipelines columns exist."""
    try:
        cur = conn.execute("PRAGMA table_info(discovery_pipelines);")
        cols = {row[1] for row in cur.fetchall()}
        if cols:
            if "parent_snapshot_hash" not in cols:
                conn.execute("ALTER TABLE discovery_pipelines ADD COLUMN parent_snapshot_hash TEXT NOT NULL DEFAULT '';")
            if "base_pipeline_id" not in cols:
                conn.execute("ALTER TABLE discovery_pipelines ADD COLUMN base_pipeline_id TEXT NOT NULL DEFAULT 'PL_0001';")
            if "base_pipeline_snapshot_hash" not in cols:
                conn.execute("ALTER TABLE discovery_pipelines ADD COLUMN base_pipeline_snapshot_hash TEXT NOT NULL DEFAULT '';")
    except Exception:
        pass


def init_discovery_pipeline_tables(data_dir: str) -> None:
    """Idempotently initialize Discovery Pipeline schema in analysis.db."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            conn.executescript(DISCOVERY_PIPELINES_DDL)
            _ensure_discovery_pipeline_columns(conn)
    finally:
        conn.close()


def persist_discovery_pipeline(data_dir: str, spec: DiscoveryPipelineSpec) -> None:
    """Upsert a DiscoveryPipelineSpec header record into analysis.db."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    now = _utc_now_iso()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO discovery_pipelines (
                    pipeline_id, campaign_id, context_key, dataset_name, dataset_snapshot_hash,
                    base_feature_count, base_feature_names_json, base_pipeline_id, base_pipeline_snapshot_hash,
                    active_features_count, total_generated_count, parent_snapshot_hash, current_snapshot_hash,
                    current_generation, status, budget_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pipeline_id) DO UPDATE SET
                    context_key = excluded.context_key,
                    dataset_name = excluded.dataset_name,
                    dataset_snapshot_hash = excluded.dataset_snapshot_hash,
                    base_feature_count = excluded.base_feature_count,
                    base_feature_names_json = excluded.base_feature_names_json,
                    base_pipeline_id = excluded.base_pipeline_id,
                    base_pipeline_snapshot_hash = excluded.base_pipeline_snapshot_hash,
                    active_features_count = excluded.active_features_count,
                    total_generated_count = excluded.total_generated_count,
                    parent_snapshot_hash = excluded.parent_snapshot_hash,
                    current_snapshot_hash = excluded.current_snapshot_hash,
                    current_generation = excluded.current_generation,
                    status = excluded.status,
                    budget_json = excluded.budget_json,
                    updated_at = excluded.updated_at
                """,
                (
                    spec.pipeline_id,
                    spec.campaign_id,
                    spec.context_key,
                    spec.dataset_name,
                    spec.dataset_snapshot_hash,
                    spec.base_feature_count,
                    json.dumps(spec.base_feature_names),
                    spec.base_pipeline_id,
                    spec.base_pipeline_snapshot_hash,
                    spec.active_features_count,
                    spec.total_generated_count,
                    spec.parent_snapshot_hash,
                    spec.current_snapshot_hash,
                    spec.current_generation,
                    spec.status,
                    json.dumps(spec.budget.to_dict()),
                    spec.created_at or now,
                    now,
                ),
            )
    finally:
        conn.close()


def load_discovery_pipeline(data_dir: str, pipeline_id: str) -> DiscoveryPipelineSpec | None:
    """Load a DiscoveryPipelineSpec by pipeline_id from analysis.db."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM discovery_pipelines WHERE pipeline_id = ?",
            (pipeline_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_pipeline_spec(row)
    finally:
        conn.close()


def load_discovery_pipeline_by_campaign(data_dir: str, campaign_id: str) -> DiscoveryPipelineSpec | None:
    """Load a DiscoveryPipelineSpec by owning campaign_id from analysis.db."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM discovery_pipelines WHERE campaign_id = ? ORDER BY created_at DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_pipeline_spec(row)
    finally:
        conn.close()


def persist_discovered_features(
    data_dir: str,
    features: Sequence[DiscoveredFeatureSpec],
) -> int:
    """Batch insert or update discovered features enforcing formula uniqueness per pipeline.
    
    Returns count of persisted features.
    """
    if not features:
        return 0
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    now = _utc_now_iso()
    count = 0
    try:
        with conn:
            for f in features:
                # Ensure parent pipeline exists defensively
                conn.execute(
                    """
                    INSERT INTO discovery_pipelines (
                        pipeline_id, campaign_id, context_key, dataset_name, dataset_snapshot_hash,
                        base_feature_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pipeline_id) DO NOTHING
                    """,
                    (f.pipeline_id, f.pipeline_id.replace("DP_", ""), "CONTEXT_DEFAULT", "unknown", "none", 0, now, now),
                )

                f_hash = f.formula_hash or compute_formula_hash(f.formula_expression)
                strat_str = f.generator_strategy.value if isinstance(f.generator_strategy, GeneratorStrategy) else str(f.generator_strategy)
                status_str = f.lifecycle_status.value if isinstance(f.lifecycle_status, DiscoveryLifecycleStatus) else str(f.lifecycle_status)
                
                conn.execute(
                    """
                    INSERT INTO discovery_pipeline_features (
                        feature_id, pipeline_id, feature_name, formula_expression,
                        formula_hash, generator_strategy, parent_features_json,
                        generation_discovered, lifecycle_status, evidence_score,
                        total_evaluations, holdout_rank, relative_imp_drop,
                        drift_severity, ks_statistic, ks_pvalue, metadata_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pipeline_id, formula_hash) DO UPDATE SET
                        feature_name = excluded.feature_name,
                        formula_expression = excluded.formula_expression,
                        generator_strategy = excluded.generator_strategy,
                        parent_features_json = excluded.parent_features_json,
                        generation_discovered = excluded.generation_discovered,
                        lifecycle_status = excluded.lifecycle_status,
                        evidence_score = excluded.evidence_score,
                        total_evaluations = excluded.total_evaluations,
                        holdout_rank = excluded.holdout_rank,
                        relative_imp_drop = excluded.relative_imp_drop,
                        drift_severity = excluded.drift_severity,
                        ks_statistic = excluded.ks_statistic,
                        ks_pvalue = excluded.ks_pvalue,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        f.feature_id,
                        f.pipeline_id,
                        f.feature_name,
                        f.formula_expression,
                        f_hash,
                        strat_str,
                        json.dumps(f.parent_features),
                        f.generation_discovered,
                        status_str,
                        float(f.evidence_score),
                        int(f.total_evaluations),
                        f.holdout_rank,
                        f.relative_imp_drop,
                        int(f.drift_severity),
                        float(f.ks_statistic),
                        float(f.ks_pvalue),
                        json.dumps(f.metadata),
                        f.created_at or now,
                        now,
                    ),
                )
                count += 1
    finally:
        conn.close()
    return count


def load_discovered_features(
    data_dir: str,
    pipeline_id: str,
    status: DiscoveryLifecycleStatus | str | None = None,
) -> list[DiscoveredFeatureSpec]:
    """Load all discovered features for a pipeline, optionally filtered by lifecycle status."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        if status:
            st = status.value if isinstance(status, DiscoveryLifecycleStatus) else str(status)
            rows = conn.execute(
                """
                SELECT * FROM discovery_pipeline_features 
                WHERE pipeline_id = ? AND lifecycle_status = ?
                ORDER BY evidence_score DESC, created_at ASC
                """,
                (pipeline_id, st),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM discovery_pipeline_features 
                WHERE pipeline_id = ?
                ORDER BY evidence_score DESC, created_at ASC
                """,
                (pipeline_id,),
            ).fetchall()
        return [_row_to_feature_spec(r) for r in rows]
    finally:
        conn.close()


def load_discovered_feature_by_hash(
    data_dir: str,
    pipeline_id: str,
    formula_hash: str,
) -> DiscoveredFeatureSpec | None:
    """Load a specific discovered feature by formula hash for deduplication check."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            """
            SELECT * FROM discovery_pipeline_features 
            WHERE pipeline_id = ? AND formula_hash = ?
            """,
            (pipeline_id, formula_hash),
        ).fetchone()
        if not row:
            return None
        return _row_to_feature_spec(row)
    finally:
        conn.close()


def update_discovered_feature_status(
    data_dir: str,
    pipeline_id: str,
    feature_name: str,
    new_status: DiscoveryLifecycleStatus | str,
    *,
    evidence_score: float | None = None,
    ks_statistic: float | None = None,
    holdout_rank: int | None = None,
    relative_imp_drop: float | None = None,
) -> bool:
    """Update lifecycle status and telemetry for a specific feature."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    status_str = new_status.value if isinstance(new_status, DiscoveryLifecycleStatus) else str(new_status)
    now = _utc_now_iso()
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE discovery_pipeline_features 
                SET lifecycle_status = ?,
                    evidence_score = COALESCE(?, evidence_score),
                    ks_statistic = COALESCE(?, ks_statistic),
                    holdout_rank = COALESCE(?, holdout_rank),
                    relative_imp_drop = COALESCE(?, relative_imp_drop),
                    total_evaluations = total_evaluations + 1,
                    updated_at = ?
                WHERE pipeline_id = ? AND feature_name = ?
                """,
                (
                    status_str,
                    evidence_score,
                    ks_statistic,
                    holdout_rank,
                    relative_imp_drop,
                    now,
                    pipeline_id,
                    feature_name,
                ),
            )
            return cur.rowcount > 0
    finally:
        conn.close()


def persist_discovery_snapshot(
    data_dir: str,
    snapshot: DiscoveryPipelineSnapshot,
) -> None:
    """Persist a reproducible generation snapshot."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    now = _utc_now_iso()
    try:
        with conn:
            # Ensure parent pipeline exists defensively
            conn.execute(
                """
                INSERT INTO discovery_pipelines (
                    pipeline_id, campaign_id, context_key, dataset_name, dataset_snapshot_hash,
                    base_feature_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pipeline_id) DO NOTHING
                """,
                (snapshot.pipeline_id, snapshot.pipeline_id.replace("DP_", ""), "CONTEXT_DEFAULT", "unknown", "none", 0, now, now),
            )

            conn.execute(
                """
                INSERT INTO discovery_pipeline_snapshots (
                    snapshot_hash, pipeline_id, generation_number,
                    active_feature_names_json, feature_count,
                    keep_count, watch_count, remove_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_hash) DO UPDATE SET
                    active_feature_names_json = excluded.active_feature_names_json,
                    feature_count = excluded.feature_count,
                    keep_count = excluded.keep_count,
                    watch_count = excluded.watch_count,
                    remove_count = excluded.remove_count
                """,
                (
                    snapshot.snapshot_hash,
                    snapshot.pipeline_id,
                    snapshot.generation_number,
                    json.dumps(snapshot.active_feature_names),
                    snapshot.feature_count,
                    snapshot.keep_count,
                    snapshot.watch_count,
                    snapshot.remove_count,
                    snapshot.created_at or now,
                ),
            )
    finally:
        conn.close()


def load_discovery_snapshot(
    data_dir: str,
    snapshot_hash: str,
) -> DiscoveryPipelineSnapshot | None:
    """Load a discovery pipeline snapshot by hash."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM discovery_pipeline_snapshots WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        if not row:
            return None
        return _row_to_snapshot(row)
    finally:
        conn.close()


def load_discovery_snapshots_for_pipeline(
    data_dir: str,
    pipeline_id: str,
) -> list[DiscoveryPipelineSnapshot]:
    """Load all snapshots for a pipeline ordered by generation."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM discovery_pipeline_snapshots 
            WHERE pipeline_id = ? 
            ORDER BY generation_number ASC
            """,
            (pipeline_id,),
        ).fetchall()
        return [_row_to_snapshot(r) for r in rows]
    finally:
        conn.close()


def get_discovery_pipeline_summary(data_dir: str, pipeline_id: str) -> dict[str, Any]:
    """Compute aggregate governance and count metrics for a Discovery Pipeline."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        pipe = load_discovery_pipeline(data_dir, pipeline_id)
        if not pipe:
            return {}

        counts = {
            "total_generated": 0,
            "candidate": 0,
            "KEEP": 0,
            "WATCH": 0,
            "REMOVE": 0,
            "promoted": 0,
        }

        rows = conn.execute(
            """
            SELECT lifecycle_status, COUNT(*) as cnt 
            FROM discovery_pipeline_features 
            WHERE pipeline_id = ? 
            GROUP BY lifecycle_status
            """,
            (pipeline_id,),
        ).fetchall()

        for r in rows:
            st = r["lifecycle_status"]
            if st in counts:
                counts[st] = int(r["cnt"])
            counts["total_generated"] += int(r["cnt"])

        top_feats = conn.execute(
            """
            SELECT feature_name, generator_strategy, lifecycle_status, evidence_score, ks_statistic
            FROM discovery_pipeline_features
            WHERE pipeline_id = ? AND lifecycle_status = 'KEEP'
            ORDER BY evidence_score DESC
            LIMIT 10
            """,
            (pipeline_id,),
        ).fetchall()

        return {
            "pipeline_id": pipe.pipeline_id,
            "campaign_id": pipe.campaign_id,
            "context_key": pipe.context_key,
            "dataset_name": pipe.dataset_name,
            "current_generation": pipe.current_generation,
            "current_snapshot_hash": pipe.current_snapshot_hash,
            "counts": counts,
            "top_keep_features": [dict(r) for r in top_feats],
        }
    finally:
        conn.close()


def _row_to_pipeline_spec(row: sqlite3.Row) -> DiscoveryPipelineSpec:
    budget_raw = json.loads(row["budget_json"]) if "budget_json" in row.keys() and row["budget_json"] else {}
    base_names = json.loads(row["base_feature_names_json"]) if "base_feature_names_json" in row.keys() and row["base_feature_names_json"] else []
    return DiscoveryPipelineSpec(
        pipeline_id=str(row["pipeline_id"]),
        campaign_id=str(row["campaign_id"]),
        context_key=str(row["context_key"]),
        dataset_name=str(row["dataset_name"]),
        dataset_snapshot_hash=str(row["dataset_snapshot_hash"]),
        base_feature_count=int(row["base_feature_count"]),
        base_feature_names=base_names,
        base_pipeline_id=str(row["base_pipeline_id"]) if "base_pipeline_id" in row.keys() else "PL_0001",
        base_pipeline_snapshot_hash=str(row["base_pipeline_snapshot_hash"]) if "base_pipeline_snapshot_hash" in row.keys() else "",
        active_features_count=int(row["active_features_count"]),
        total_generated_count=int(row["total_generated_count"]),
        parent_snapshot_hash=str(row["parent_snapshot_hash"]) if "parent_snapshot_hash" in row.keys() else "",
        current_snapshot_hash=str(row["current_snapshot_hash"]),
        current_generation=int(row["current_generation"]),
        status=str(row["status"]),
        budget=DiscoveryPipelineBudget.from_dict(budget_raw),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_feature_spec(row: sqlite3.Row) -> DiscoveredFeatureSpec:
    parent_feats = json.loads(row["parent_features_json"]) if "parent_features_json" in row.keys() and row["parent_features_json"] else []
    meta = json.loads(row["metadata_json"]) if "metadata_json" in row.keys() and row["metadata_json"] else {}
    return DiscoveredFeatureSpec(
        feature_id=str(row["feature_id"]),
        pipeline_id=str(row["pipeline_id"]),
        feature_name=str(row["feature_name"]),
        formula_expression=str(row["formula_expression"]),
        formula_hash=str(row["formula_hash"]),
        generator_strategy=GeneratorStrategy(row["generator_strategy"]) if row["generator_strategy"] in GeneratorStrategy._value2member_map_ else GeneratorStrategy.RATIO,
        parent_features=parent_feats,
        generation_discovered=int(row["generation_discovered"]),
        lifecycle_status=DiscoveryLifecycleStatus(row["lifecycle_status"]) if row["lifecycle_status"] in DiscoveryLifecycleStatus._value2member_map_ else DiscoveryLifecycleStatus.CANDIDATE,
        evidence_score=float(row["evidence_score"]),
        total_evaluations=int(row["total_evaluations"]),
        holdout_rank=row["holdout_rank"] if row["holdout_rank"] is not None else None,
        relative_imp_drop=row["relative_imp_drop"] if row["relative_imp_drop"] is not None else None,
        drift_severity=int(row["drift_severity"]),
        ks_statistic=float(row["ks_statistic"]),
        ks_pvalue=float(row["ks_pvalue"]),
        metadata=meta,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_snapshot(row: sqlite3.Row) -> DiscoveryPipelineSnapshot:
    names = json.loads(row["active_feature_names_json"]) if "active_feature_names_json" in row.keys() and row["active_feature_names_json"] else []
    return DiscoveryPipelineSnapshot(
        snapshot_hash=str(row["snapshot_hash"]),
        pipeline_id=str(row["pipeline_id"]),
        generation_number=int(row["generation_number"]),
        active_feature_names=names,
        feature_count=int(row["feature_count"]),
        keep_count=int(row["keep_count"]),
        watch_count=int(row["watch_count"]),
        remove_count=int(row["remove_count"]),
        created_at=str(row["created_at"]),
    )
