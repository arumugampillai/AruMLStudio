"""Cross-Research Formula Memory and Longitudinal Prior Registry (Doc 16)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from chain_replay_ml.research_memory.db import connect_analysis_db
from .store import init_research_registry_tables
from .types import FormulaGlobalStatus, FormulaMemoryRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_blacklisted_formula_hashes(data_dir: str, context_key: str | None = None) -> set[str]:
    """Retrieve formula hashes that have proven to suffer severe distribution drift (D_KS > 0.35) under this context."""
    if not data_dir:
        return set()
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            "SELECT formula_hash, context_lock_json FROM research_formula_memory WHERE global_status = ?;",
            (FormulaGlobalStatus.REJECTED_DRIFT.value,),
        ).fetchall()
        blacklisted = set()
        for r in rows:
            f_hash = str(r["formula_hash"])
            ctx_lock = json.loads(r["context_lock_json"] or "[]")
            if not context_key or not ctx_lock or context_key in ctx_lock:
                blacklisted.add(f_hash)
        return blacklisted
    finally:
        conn.close()


def update_formula_memory_from_discovery(
    data_dir: str,
    research_id: str,
    campaign_id: str,
    context_key: str,
) -> int:
    """Sync newly evaluated discovery pipeline features into research_formula_memory."""
    init_research_registry_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    updated_count = 0
    now = _utc_now_iso()
    try:
        # Find features for this campaign's discovery pipeline
        dp_rows = conn.execute(
            """
            SELECT f.feature_id, f.formula_expression, f.formula_hash, f.generator_strategy,
                   f.parent_features_json, f.lifecycle_status, f.evidence_score, f.ks_statistic,
                   f.drift_severity, f.metadata_json
            FROM discovery_pipeline_features f
            JOIN discovery_pipelines p ON f.pipeline_id = p.pipeline_id
            WHERE p.campaign_id = ?;
            """,
            (campaign_id,),
        ).fetchall()

        for r in dp_rows:
            f_hash = str(r["formula_hash"])
            formula_expr = str(r["formula_expression"])
            strategy = str(r["generator_strategy"])
            parents = r["parent_features_json"] or "[]"
            status_str = str(r["lifecycle_status"]).upper()
            ev_score = float(r["evidence_score"] or 0.0)
            ks_stat = float(r["ks_statistic"] or 0.0)
            drift_sev = int(r["drift_severity"] or 0)
            
            meta_d = json.loads(r["metadata_json"] or "{}") if r["metadata_json"] else {}
            delta_auc = float(meta_d.get("delta_auc", 0.0))
            gov_reason = str(meta_d.get("governance_rationale", "Automated Evaluation"))

            if status_str == "KEEP":
                g_status = FormulaGlobalStatus.PROMISING
            elif status_str == "WATCH":
                g_status = FormulaGlobalStatus.WATCH
            elif drift_sev >= 2 or ks_stat > 0.35:
                g_status = FormulaGlobalStatus.REJECTED_DRIFT
            else:
                g_status = FormulaGlobalStatus.REJECTED_NOISE

            existing = conn.execute(
                "SELECT * FROM research_formula_memory WHERE formula_hash = ?;",
                (f_hash,),
            ).fetchone()

            if not existing:
                conn.execute(
                    """
                    INSERT INTO research_formula_memory (
                        formula_hash, canonical_formula_expression, generator_strategy,
                        parent_features_json, first_discovered_research_id, first_discovered_at,
                        last_evaluated_research_id, last_evaluated_at, total_researches_tested,
                        total_evaluations_count, highest_evidence_score, lowest_ks_drift,
                        best_marginal_delta_auc, global_status, last_governance_verdict,
                        last_governance_reason, context_lock_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        f_hash, formula_expr, strategy, parents,
                        research_id, now, research_id, now,
                        1, 1, ev_score, ks_stat, delta_auc,
                        g_status.value, status_str, gov_reason,
                        json.dumps([context_key]),
                    ),
                )
            else:
                ex_d = dict(existing)
                tot_res = int(ex_d.get("total_researches_tested") or 1)
                if ex_d.get("last_evaluated_research_id") != research_id:
                    tot_res += 1
                tot_ev = int(ex_d.get("total_evaluations_count") or 1) + 1
                high_score = max(float(ex_d.get("highest_evidence_score") or 0.0), ev_score)
                low_ks = min(float(ex_d.get("lowest_ks_drift") or 1.0), ks_stat)
                best_d_auc = max(float(ex_d.get("best_marginal_delta_auc") or -1.0), delta_auc)
                
                ctx_locks = json.loads(ex_d.get("context_lock_json") or "[]")
                if context_key not in ctx_locks:
                    ctx_locks.append(context_key)

                conn.execute(
                    """
                    UPDATE research_formula_memory SET
                        last_evaluated_research_id = ?,
                        last_evaluated_at = ?,
                        total_researches_tested = ?,
                        total_evaluations_count = ?,
                        highest_evidence_score = ?,
                        lowest_ks_drift = ?,
                        best_marginal_delta_auc = ?,
                        global_status = ?,
                        last_governance_verdict = ?,
                        last_governance_reason = ?,
                        context_lock_json = ?
                    WHERE formula_hash = ?;
                    """,
                    (
                        research_id, now, tot_res, tot_ev,
                        high_score, low_ks, best_d_auc,
                        g_status.value, status_str, gov_reason,
                        json.dumps(ctx_locks), f_hash,
                    ),
                )
            updated_count += 1

        conn.commit()
        return updated_count
    finally:
        conn.close()
