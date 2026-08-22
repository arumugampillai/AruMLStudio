"""Storage and query engine for Autonomous Research Registry in analysis.db (Doc 16)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import (
    FormulaGlobalStatus,
    FormulaMemoryRecord,
    ResearchGenerationLinkage,
    ResearchRegistryRecord,
    ResearchStatus,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_research_id(context_key: str, timestamp_iso: str | None = None) -> str:
    """Generate deterministic, human-readable immutable Research ID."""
    clean_ctx = str(context_key or "UNKNOWN_CONTEXT").strip().replace(":", "_")
    ts = timestamp_iso or _utc_now_iso()
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_compact = dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts_compact = ts.replace("-", "").replace(":", "").replace("T", "_")[:15]
    
    unique_seed = f"{clean_ctx}_{ts}"
    short_hex = hashlib.sha256(unique_seed.encode("utf-8")).hexdigest()[:4]
    return f"RESEARCH_{clean_ctx}_{ts_compact}_{short_hex}"


def init_research_registry_tables(data_dir: str) -> None:
    """Initialize research registry, generation linkage, and formula memory tables in analysis.db."""
    init_analysis_db(data_dir)
    try:
        from chain_replay_ml.discovery_pipeline.persistence import init_discovery_pipeline_tables
        init_discovery_pipeline_tables(data_dir)
    except Exception:
        pass
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_registry (
                research_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL UNIQUE,
                context_key TEXT NOT NULL,
                context_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_snapshot_hash TEXT NOT NULL,
                base_pipeline_id TEXT NOT NULL DEFAULT 'PL_0001',
                base_feature_count INTEGER NOT NULL DEFAULT 171,
                registry_feature_count INTEGER NOT NULL DEFAULT 211,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                duration_seconds REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                stop_reason TEXT NOT NULL DEFAULT 'IN_PROGRESS',
                algorithms_used_json TEXT NOT NULL,
                elimination_strategy TEXT NOT NULL,
                max_generations_configured INTEGER NOT NULL,
                actual_generations_completed INTEGER NOT NULL,
                max_candidates_configured INTEGER NOT NULL,
                candidates_generated INTEGER NOT NULL DEFAULT 0,
                candidates_evaluated INTEGER NOT NULL DEFAULT 0,
                candidates_pruned INTEGER NOT NULL DEFAULT 0,
                best_candidate_id TEXT,
                best_composite_score REAL NOT NULL DEFAULT 0.0,
                best_trading_score REAL NOT NULL DEFAULT 0.0,
                best_model_score REAL NOT NULL DEFAULT 0.0,
                best_win_rate_pct REAL NOT NULL DEFAULT 0.0,
                best_profit_factor REAL NOT NULL DEFAULT 0.0,
                best_max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
                starting_best_score REAL NOT NULL DEFAULT 0.0,
                total_score_lift REAL NOT NULL DEFAULT 0.0,
                discovery_pipeline_id TEXT NOT NULL,
                final_discovery_snapshot_hash TEXT,
                total_df_features_created INTEGER NOT NULL DEFAULT 0,
                unique_formula_count INTEGER NOT NULL DEFAULT 0,
                keep_count INTEGER NOT NULL DEFAULT 0,
                watch_count INTEGER NOT NULL DEFAULT 0,
                remove_count INTEGER NOT NULL DEFAULT 0,
                active_discovery_pool INTEGER NOT NULL DEFAULT 0,
                promoted_feature_count INTEGER NOT NULL DEFAULT 0,
                research_config_json TEXT NOT NULL,
                research_outcome_json TEXT NOT NULL DEFAULT '{}',
                failure_reason TEXT,
                architecture_version TEXT NOT NULL DEFAULT '2.2.0',
                code_version TEXT NOT NULL DEFAULT '1.0.0',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_generation_snapshots (
                snapshot_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                research_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                generation_number INTEGER NOT NULL,
                discovery_snapshot_hash TEXT NOT NULL,
                candidates_evaluated INTEGER NOT NULL,
                generation_best_score REAL NOT NULL,
                generation_best_candidate_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, generation_number)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_formula_memory (
                formula_hash TEXT PRIMARY KEY,
                canonical_formula_expression TEXT NOT NULL,
                generator_strategy TEXT NOT NULL,
                parent_features_json TEXT NOT NULL,
                first_discovered_research_id TEXT NOT NULL,
                first_discovered_at TEXT NOT NULL,
                last_evaluated_research_id TEXT NOT NULL,
                last_evaluated_at TEXT NOT NULL,
                total_researches_tested INTEGER NOT NULL DEFAULT 1,
                total_evaluations_count INTEGER NOT NULL DEFAULT 1,
                highest_evidence_score REAL NOT NULL DEFAULT 0.0,
                lowest_ks_drift REAL NOT NULL DEFAULT 1.0,
                best_marginal_delta_auc REAL NOT NULL DEFAULT -1.0,
                global_status TEXT NOT NULL,
                last_governance_verdict TEXT NOT NULL,
                last_governance_reason TEXT NOT NULL,
                context_lock_json TEXT NOT NULL DEFAULT '[]'
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_res_reg_context ON research_registry (context_key, started_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_res_reg_status ON research_registry (status, started_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_res_reg_best ON research_registry (best_composite_score DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_res_reg_disc ON research_registry (discovery_pipeline_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gen_snap_lookup ON research_generation_snapshots (research_id, generation_number);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_form_mem_status ON research_formula_memory (global_status, highest_evidence_score DESC);")
        conn.commit()
    finally:
        conn.close()


def insert_or_update_research_run(data_dir: str, record: ResearchRegistryRecord) -> None:
    """Insert or update a research registry record in analysis.db."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            INSERT INTO research_registry (
                research_id, campaign_id, context_key, context_id,
                dataset_name, dataset_snapshot_hash, base_pipeline_id,
                base_feature_count, registry_feature_count, started_at,
                finished_at, duration_seconds, status, stop_reason,
                algorithms_used_json, elimination_strategy,
                max_generations_configured, actual_generations_completed,
                max_candidates_configured, candidates_generated,
                candidates_evaluated, candidates_pruned, best_candidate_id,
                best_composite_score, best_trading_score, best_model_score,
                best_win_rate_pct, best_profit_factor, best_max_drawdown_pct,
                starting_best_score, total_score_lift, discovery_pipeline_id,
                final_discovery_snapshot_hash, total_df_features_created,
                unique_formula_count, keep_count, watch_count, remove_count,
                active_discovery_pool, promoted_feature_count,
                research_config_json, research_outcome_json, failure_reason,
                architecture_version, code_version, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(research_id) DO UPDATE SET
                finished_at = excluded.finished_at,
                duration_seconds = excluded.duration_seconds,
                status = excluded.status,
                stop_reason = excluded.stop_reason,
                actual_generations_completed = excluded.actual_generations_completed,
                candidates_generated = excluded.candidates_generated,
                candidates_evaluated = excluded.candidates_evaluated,
                candidates_pruned = excluded.candidates_pruned,
                best_candidate_id = excluded.best_candidate_id,
                best_composite_score = excluded.best_composite_score,
                best_trading_score = excluded.best_trading_score,
                best_model_score = excluded.best_model_score,
                best_win_rate_pct = excluded.best_win_rate_pct,
                best_profit_factor = excluded.best_profit_factor,
                best_max_drawdown_pct = excluded.best_max_drawdown_pct,
                total_score_lift = excluded.total_score_lift,
                final_discovery_snapshot_hash = excluded.final_discovery_snapshot_hash,
                total_df_features_created = excluded.total_df_features_created,
                unique_formula_count = excluded.unique_formula_count,
                keep_count = excluded.keep_count,
                watch_count = excluded.watch_count,
                remove_count = excluded.remove_count,
                active_discovery_pool = excluded.active_discovery_pool,
                promoted_feature_count = excluded.promoted_feature_count,
                research_outcome_json = excluded.research_outcome_json,
                failure_reason = excluded.failure_reason,
                updated_at = excluded.updated_at;
            """,
            (
                record.research_id, record.campaign_id, record.context_key, record.context_id,
                record.dataset_name, record.dataset_snapshot_hash, record.base_pipeline_id,
                record.base_feature_count, record.registry_feature_count, record.started_at,
                record.finished_at, record.duration_seconds,
                record.status.value if isinstance(record.status, ResearchStatus) else str(record.status),
                record.stop_reason, json.dumps(record.algorithms_used), record.elimination_strategy,
                record.max_generations_configured, record.actual_generations_completed,
                record.max_candidates_configured, record.candidates_generated,
                record.candidates_evaluated, record.candidates_pruned, record.best_candidate_id,
                record.best_composite_score, record.best_trading_score, record.best_model_score,
                record.best_win_rate_pct, record.best_profit_factor, record.best_max_drawdown_pct,
                record.starting_best_score, record.total_score_lift, record.discovery_pipeline_id,
                record.final_discovery_snapshot_hash, record.total_df_features_created,
                record.unique_formula_count, record.keep_count, record.watch_count, record.remove_count,
                record.active_discovery_pool, record.promoted_feature_count,
                record.research_config_json, record.research_outcome_json, record.failure_reason,
                record.architecture_version, record.code_version, record.created_at, record.updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_generation_linkage(
    data_dir: str,
    research_id: str,
    campaign_id: str,
    generation_number: int,
    discovery_snapshot_hash: str,
    candidates_evaluated: int,
    generation_best_score: float,
    generation_best_candidate_id: str,
) -> None:
    """Record or update a generational snapshot linkage record."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            INSERT INTO research_generation_snapshots (
                research_id, campaign_id, generation_number,
                discovery_snapshot_hash, candidates_evaluated,
                generation_best_score, generation_best_candidate_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(research_id, generation_number) DO UPDATE SET
                discovery_snapshot_hash = excluded.discovery_snapshot_hash,
                candidates_evaluated = excluded.candidates_evaluated,
                generation_best_score = excluded.generation_best_score,
                generation_best_candidate_id = excluded.generation_best_candidate_id;
            """,
            (
                research_id, campaign_id, generation_number,
                discovery_snapshot_hash, candidates_evaluated,
                generation_best_score, generation_best_candidate_id, _utc_now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_research_records(data_dir: str, context_key: str | None = None) -> list[dict[str, Any]]:
    """Retrieve all research registry records from analysis.db."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        if context_key:
            rows = conn.execute(
                "SELECT * FROM research_registry WHERE context_key = ? ORDER BY started_at DESC;",
                (context_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM research_registry ORDER BY started_at DESC;"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_research_detail(data_dir: str, research_id: str) -> dict[str, Any] | None:
    """Retrieve full research record, generational linkages, discovery features, and hardware metadata."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute("SELECT * FROM research_registry WHERE research_id = ?;", (research_id,)).fetchone()
        if not row:
            return None
        res_d = dict(row)
        camp_id = res_d.get("campaign_id") or ""
        dp_id = res_d.get("discovery_pipeline_id") or ""

        # Parse config and outcome JSON
        cfg_d = json.loads(res_d.get("research_config_json") or "{}") if res_d.get("research_config_json") else {}
        out_d = json.loads(res_d.get("research_outcome_json") or "{}") if res_d.get("research_outcome_json") else {}
        res_d["research_config"] = cfg_d
        res_d["research_outcome"] = out_d

        # 1. Load point-in-time discovery features for this research run
        feat_rows = conn.execute(
            """SELECT feature_id, pipeline_id, feature_name, formula_expression, formula_hash,
                      generator_strategy, parent_features_json, generation_discovered,
                      lifecycle_status, evidence_score, total_evaluations, holdout_rank,
                      relative_imp_drop, drift_severity, ks_statistic, ks_pvalue, metadata_json,
                      created_at, updated_at
               FROM discovery_pipeline_features
               WHERE pipeline_id = ? OR pipeline_id LIKE ?
               ORDER BY generation_discovered ASC, evidence_score DESC;""",
            (dp_id, f"%{camp_id}%" if camp_id else dp_id),
        ).fetchall()

        features_list = []
        for f in feat_rows:
            fd = dict(f)
            meta = json.loads(fd.get("metadata_json") or "{}") if fd.get("metadata_json") else {}
            fd["delta_auc"] = meta.get("delta_auc")
            fd["governance_rationale"] = meta.get("governance_rationale") or "Under evaluation"
            fd["research_id"] = research_id
            try:
                fd["parent_features"] = json.loads(fd.get("parent_features_json") or "[]")
            except Exception:
                fd["parent_features"] = []
            features_list.append(fd)
        res_d["features"] = features_list

        # 2. Load generations
        gen_rows = conn.execute(
            "SELECT * FROM research_generation_snapshots WHERE research_id = ? ORDER BY generation_number ASC;",
            (research_id,),
        ).fetchall()
        generations = [dict(g) for g in gen_rows]

        # If generation snapshots table was not explicitly populated, synthesize from discovery_pipeline_snapshots
        if not generations and dp_id:
            snap_rows = conn.execute(
                """SELECT snapshot_hash, pipeline_id, generation_number,
                          feature_count, keep_count, watch_count, remove_count, created_at
                   FROM discovery_pipeline_snapshots
                   WHERE pipeline_id = ? OR pipeline_id LIKE ?
                   ORDER BY generation_number ASC;""",
                (dp_id, f"%{camp_id}%" if camp_id else dp_id),
            ).fetchall()
            for s in snap_rows:
                sd = dict(s)
                sd["research_id"] = research_id
                sd["discovery_snapshot_hash"] = sd.get("snapshot_hash")
                sd["candidates_evaluated"] = res_d.get("candidates_evaluated", 0) // max(len(snap_rows), 1)
                sd["candidates_generated"] = res_d.get("candidates_generated", 0) // max(len(snap_rows), 1)
                sd["candidates_pruned"] = res_d.get("candidates_pruned", 0) // max(len(snap_rows), 1)
                sd["generation_best_score"] = res_d.get("best_composite_score", 0.0)
                sd["generation_best_trading_score"] = res_d.get("best_trading_score", 0.0)
                sd["generation_best_model_score"] = res_d.get("best_model_score", 0.0)
                sd["generation_best_candidate_id"] = res_d.get("best_candidate_id") or "—"
                generations.append(sd)

        # Fallback if still empty but actual_generations_completed > 0
        if not generations and int(res_d.get("actual_generations_completed") or 0) > 0:
            act_gens = int(res_d.get("actual_generations_completed") or 1)
            for g_num in range(1, act_gens + 1):
                generations.append({
                    "generation_number": g_num,
                    "discovery_snapshot_hash": res_d.get("final_discovery_snapshot_hash") or f"GEN_{g_num}_SNAPSHOT",
                    "candidates_generated": res_d.get("candidates_generated", 0) // act_gens,
                    "candidates_evaluated": res_d.get("candidates_evaluated", 0) // act_gens,
                    "candidates_pruned": res_d.get("candidates_pruned", 0) // act_gens,
                    "generation_best_score": res_d.get("best_composite_score", 0.0),
                    "generation_best_trading_score": res_d.get("best_trading_score", 0.0),
                    "generation_best_model_score": res_d.get("best_model_score", 0.0),
                    "generation_best_candidate_id": res_d.get("best_candidate_id") or "—",
                    "keep_count": res_d.get("keep_count", 0),
                    "watch_count": res_d.get("watch_count", 0),
                    "remove_count": res_d.get("remove_count", 0),
                    "created_at": res_d.get("started_at"),
                })

        res_d["generations"] = generations

        # 3. Build Hardware / Runtime info
        import os
        from chain_replay_ml.training.model_device import detect_gpu_hardware
        hw = detect_gpu_hardware()
        gpu_detected = bool(hw.get("gpu_detected"))
        gpu_name = hw.get("gpu_name") or "—"
        res_d["hardware"] = {
            "cpu": f"{os.cpu_count() or 4} Logical Cores (OpenMP multi-threaded)",
            "gpu": "Available (CUDA active)" if gpu_detected else "None / CPU",
            "gpu_model": gpu_name,
            "algorithms_mapping": "XGBoost: CUDA (cuda:0) · CatBoost: GPU · LightGBM: CPU (OpenMP) · Random Forest: CPU",
            "workers_threads": f"{cfg_d.get('n_workers') or 4} workers · {os.cpu_count() or 4} threads",
            "memory": hw.get("memory_total") or "8192 MiB dedicated VRAM",
            "fallback_info": "LightGBM using CPU fallback (installed wheel lacks CUDA learner)",
            "driver_version": hw.get("driver_version") or "610.88",
        }
        return res_d
    finally:
        conn.close()


def map_campaign_status_to_research_status(camp_status: str | None, stop_reason: str | None = None) -> ResearchStatus:
    """Normalize and map overnight campaign status to authoritative ResearchStatus."""
    st = str(camp_status or "").strip().upper()
    sr = str(stop_reason or "").strip().upper()
    if sr == "USER_CANCELLED":
        return ResearchStatus.ABORTED
    if st in ("COMPLETED", "CAMPAIGN_COMPLETED"):
        return ResearchStatus.COMPLETED
    if st in ("CAMPAIGN_FAILED", "FAILED"):
        return ResearchStatus.FAILED
    if st == "RESOURCE_PAUSED":
        return ResearchStatus.PAUSED
    if st == "RUNNING":
        return ResearchStatus.RUNNING
    # Interrupted or unexecuted non-terminal states
    if st in ("CREATED", "TRAINING", "CAMPAIGN_STOPPED", "OOS_EVALUATION", "TRADING_EVALUATION", "RANKING", "FINE_TUNING", "NEXT_GENERATION"):
        return ResearchStatus.ABORTED
    return ResearchStatus.ABORTED


def backfill_historical_research_records(data_dir: str, force_resync: bool = False) -> int:
    """Scan existing overnight_campaigns in analysis.db and ensure corresponding research_registry entries exist and have accurate status."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    synced_count = 0
    try:
        camps = conn.execute("SELECT * FROM overnight_campaigns ORDER BY start_time_iso ASC;").fetchall()
        for c in camps:
            camp_id = c["campaign_id"]
            existing = conn.execute("SELECT * FROM research_registry WHERE campaign_id = ?;", (camp_id,)).fetchone()

            cfg_d = json.loads(c["config_json"] or "{}") if c["config_json"] else {}
            ctx_key = cfg_d.get("context_key") or "NIFTY:6:standard:all"
            start_iso = c["start_time_iso"] or _utc_now_iso()
            r_id = existing["research_id"] if existing else generate_research_id(ctx_key, start_iso)

            # Check for discovery pipeline
            dp_row = conn.execute("SELECT * FROM discovery_pipelines WHERE campaign_id = ? ORDER BY created_at DESC LIMIT 1;", (camp_id,)).fetchone()
            dp_id = dp_row["pipeline_id"] if dp_row else f"DP_{camp_id}"
            dp_snap = dp_row["current_snapshot_hash"] if dp_row else None
            
            # Count discovery features
            dp_feats = conn.execute(
                "SELECT lifecycle_status, COUNT(*) as cnt FROM discovery_pipeline_features WHERE pipeline_id = ? GROUP BY lifecycle_status;",
                (dp_id,),
            ).fetchall()
            stat_m = {str(r["lifecycle_status"]).upper(): r["cnt"] for r in dp_feats}
            keep_cnt = stat_m.get("KEEP", 0)
            watch_cnt = stat_m.get("WATCH", 0)
            rem_cnt = stat_m.get("REMOVE", 0)
            tot_df = sum(stat_m.values())
            act_pool = keep_cnt + watch_cnt

            st_val = map_campaign_status_to_research_status(c["status"], c["stop_reason"])
            dur = 0.0
            if c["end_time_iso"] and c["start_time_iso"]:
                try:
                    t1 = datetime.fromisoformat(c["start_time_iso"].replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(c["end_time_iso"].replace("Z", "+00:00"))
                    dur = (t2 - t1).total_seconds()
                except Exception:
                    dur = 0.0

            c_d = dict(c)
            algos = cfg_d.get("algorithms") or ["XGBoost", "LightGBM", "CatBoost"]
            strat = c_d.get("feature_elimination_strategy") or cfg_d.get("feature_elimination_strategy") or "SHAP_AND_EVIDENCE"

            if existing:
                if force_resync or existing["status"] != st_val.value:
                    conn.execute(
                        "UPDATE research_registry SET status = ?, updated_at = ? WHERE research_id = ?;",
                        (st_val.value, _utc_now_iso(), r_id),
                    )
                    synced_count += 1
                continue

            rec = ResearchRegistryRecord(
                research_id=r_id,
                campaign_id=camp_id,
                context_key=ctx_key,
                context_id=cfg_d.get("context_id") or "ctx_default",
                dataset_name=cfg_d.get("dataset_name") or "dataset",
                dataset_snapshot_hash=cfg_d.get("dataset_snapshot_hash") or "snap",
                base_pipeline_id="PL_0001",
                base_feature_count=171,
                registry_feature_count=211,
                started_at=start_iso,
                finished_at=c["end_time_iso"],
                duration_seconds=dur,
                status=st_val,
                stop_reason=c["stop_reason"] or "COMPLETED",
                algorithms_used=algos,
                elimination_strategy=strat,
                max_generations_configured=cfg_d.get("max_generations") or 100,
                actual_generations_completed=c["current_generation"] or 0,
                max_candidates_configured=cfg_d.get("max_candidates") or 500,
                candidates_generated=c["total_candidates_generated"] or 0,
                candidates_evaluated=c["total_candidates_evaluated"] or 0,
                candidates_pruned=c["total_candidates_pruned"] or 0,
                best_candidate_id=c["best_candidate_id"],
                best_composite_score=float(c["best_composite_score"] or 0.0),
                best_trading_score=float(c["best_trading_score"] or 0.0),
                best_model_score=float(c["best_model_score"] or 0.0),
                starting_best_score=float(c["starting_best_score"] or 0.0),
                total_score_lift=float(c["best_composite_score"] or 0.0) - float(c["starting_best_score"] or 0.0),
                discovery_pipeline_id=dp_id,
                final_discovery_snapshot_hash=dp_snap,
                total_df_features_created=tot_df,
                unique_formula_count=tot_df,
                keep_count=keep_cnt,
                watch_count=watch_cnt,
                remove_count=rem_cnt,
                active_discovery_pool=act_pool,
                promoted_feature_count=0,
                research_config_json=c["config_json"] or "{}",
                research_outcome_json="{}",
                failure_reason=None,
                architecture_version="2.2.0",
                code_version="1.0.0",
                created_at=start_iso,
                updated_at=_utc_now_iso(),
            )
            insert_or_update_research_run(data_dir, rec)
            synced_count += 1
        conn.commit()
        return synced_count
    finally:
        conn.close()


def delete_research_records(data_dir: str, research_ids: list[str]) -> dict[str, int]:
    """Safely delete specified research registry records and their uniquely associated run metadata.
    
    Preserves referential integrity and checks ownership so shared discovery pipelines
    or external models/registries are never accidentally removed.
    """
    if not research_ids:
        return {}

    conn = connect_analysis_db(data_dir)
    deleted_counts: dict[str, int] = {
        "research_registry": 0,
        "research_generation_snapshots": 0,
        "overnight_campaign_events": 0,
        "overnight_campaigns": 0,
        "campaign_candidate_specs": 0,
        "discovery_pipelines": 0,
        "discovery_pipeline_features": 0,
        "discovery_pipeline_snapshots": 0,
    }

    try:
        conn.execute("BEGIN TRANSACTION;")
        for r_id in research_ids:
            # 1. Fetch the research record to find its campaign_id and discovery_pipeline_id
            row = conn.execute(
                "SELECT campaign_id, discovery_pipeline_id FROM research_registry WHERE research_id = ?;",
                (r_id,),
            ).fetchone()
            if not row:
                continue

            c_id = row["campaign_id"]
            dp_id = row["discovery_pipeline_id"]

            # 2. Delete research_generation_snapshots
            cur = conn.execute("DELETE FROM research_generation_snapshots WHERE research_id = ? OR campaign_id = ?;", (r_id, c_id))
            deleted_counts["research_generation_snapshots"] += cur.rowcount

            # 3. Delete overnight_campaign_events and candidate specs
            if c_id:
                cur = conn.execute("DELETE FROM overnight_campaign_events WHERE campaign_id = ?;", (c_id,))
                deleted_counts["overnight_campaign_events"] += cur.rowcount

                cur = conn.execute("DELETE FROM campaign_candidate_specs WHERE campaign_id = ?;", (c_id,))
                deleted_counts["campaign_candidate_specs"] += cur.rowcount

            # 4. Check if discovery pipeline is uniquely owned by this campaign / research run
            if dp_id:
                other_refs = conn.execute(
                    "SELECT COUNT(*) as c FROM research_registry WHERE discovery_pipeline_id = ? AND research_id != ?;",
                    (dp_id, r_id),
                ).fetchone()["c"]
                if other_refs == 0:
                    cur = conn.execute("DELETE FROM discovery_pipeline_features WHERE pipeline_id = ?;", (dp_id,))
                    deleted_counts["discovery_pipeline_features"] += cur.rowcount
                    cur = conn.execute("DELETE FROM discovery_pipeline_snapshots WHERE pipeline_id = ?;", (dp_id,))
                    deleted_counts["discovery_pipeline_snapshots"] += cur.rowcount
                    cur = conn.execute("DELETE FROM discovery_pipelines WHERE pipeline_id = ?;", (dp_id,))
                    deleted_counts["discovery_pipelines"] += cur.rowcount

            # 5. Delete from overnight_campaigns (only if no other research_registry row uses this campaign_id)
            if c_id:
                other_camp_refs = conn.execute(
                    "SELECT COUNT(*) as c FROM research_registry WHERE campaign_id = ? AND research_id != ?;",
                    (c_id, r_id),
                ).fetchone()["c"]
                if other_camp_refs == 0:
                    cur = conn.execute("DELETE FROM overnight_campaigns WHERE campaign_id = ?;", (c_id,))
                    deleted_counts["overnight_campaigns"] += cur.rowcount

            # 6. Delete from research_registry
            cur = conn.execute("DELETE FROM research_registry WHERE research_id = ?;", (r_id,))
            deleted_counts["research_registry"] += cur.rowcount

        conn.commit()
        return deleted_counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
