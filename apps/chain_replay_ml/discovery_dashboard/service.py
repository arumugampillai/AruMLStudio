"""Backend service engine for the Discovery Feature Dashboard & Pipeline Builder (Doc 18).

Invariants:
1. Pure Storage Consumption: Operates directly against analysis.db (discovery_pipelines, discovery_pipeline_features).
2. Authoritative PL_0001 Resolution: Always resolves PL_0001 base features directly from pipeline_registry_store.json.
3. Zero Permanent Contamination: Discovered features remain DF_* in candidate pipelines and never contaminate feature_registry_store.json.
4. Deterministic Deduplication: Deduplicates identical formula hashes across multiple Discovery Pipelines.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
from typing import Any

from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    add_candidate_features,
    build_pipeline_snapshot,
    create_pipeline,
    load_store as load_pl_store,
    save_store as save_pl_store,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovery_pipeline,
)
from chain_replay_ml.research_memory.db import connect_analysis_db
from .types import (
    CrossPipelineSelectionBasket,
    PipelineCreationRequest,
    PipelineCreationResult,
    SelectedDiscoveryFeatureRef,
    _utc_now_iso,
)


def list_discovery_pipelines(
    data_dir: str,
    context_key: str | None = None,
) -> list[dict[str, Any]]:
    """Query and return all Discovery Pipelines matching context_key with aggregated governance metrics."""
    init_discovery_pipeline_tables(data_dir)
    try:
        from chain_replay_ml.research_registry.store import init_research_registry_tables
        init_research_registry_tables(data_dir)
    except Exception:
        pass

    conn = connect_analysis_db(data_dir)
    try:
        query = "SELECT * FROM discovery_pipelines"
        params: list[Any] = []
        if context_key and context_key != "ALL":
            query += " WHERE context_key = ? OR context_key LIKE ?"
            params.extend([context_key, f"%{context_key}%"])
        query += " ORDER BY created_at DESC;"

        rows = conn.execute(query, params).fetchall()
        results: list[dict[str, Any]] = []

        for r in rows:
            dp_id = r["pipeline_id"]
            camp_id = r["campaign_id"]

            # Query associated research_registry record if present
            res_row = None
            try:
                res_row = conn.execute(
                    "SELECT research_id, duration_seconds, started_at, finished_at, best_candidate_id, best_composite_score FROM research_registry WHERE campaign_id = ? LIMIT 1;",
                    (camp_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                pass

            r_id = res_row["research_id"] if res_row else f"RESEARCH_{camp_id}"
            dur = float(res_row["duration_seconds"]) if res_row and res_row["duration_seconds"] else 0.0

            # Count governance verdicts
            f_rows = conn.execute(
                "SELECT lifecycle_status, COUNT(*) as cnt, COUNT(DISTINCT formula_hash) as uniq FROM discovery_pipeline_features WHERE pipeline_id = ? GROUP BY lifecycle_status;",
                (dp_id,),
            ).fetchall()

            stat_m = {str(fr["lifecycle_status"]).upper(): fr["cnt"] for fr in f_rows}
            keep_cnt = stat_m.get("KEEP", 0)
            watch_cnt = stat_m.get("WATCH", 0)
            remove_cnt = stat_m.get("REMOVE", 0)
            cand_cnt = stat_m.get("CANDIDATE", 0)
            total_created = sum(stat_m.values())
            active_pool = keep_cnt + watch_cnt

            total_uniq_row = conn.execute(
                "SELECT COUNT(DISTINCT formula_hash) as uniq_total FROM discovery_pipeline_features WHERE pipeline_id = ?;",
                (dp_id,),
            ).fetchone()
            uniq_total = total_uniq_row["uniq_total"] if total_uniq_row else total_created

            results.append({
                "pipeline_id": dp_id,
                "research_id": r_id,
                "campaign_id": camp_id,
                "context_key": r["context_key"],
                "dataset_name": r["dataset_name"],
                "dataset_snapshot_hash": r["dataset_snapshot_hash"],
                "base_pipeline_id": r["base_pipeline_id"],
                "base_feature_count": r["base_feature_count"],
                "current_generation": r["current_generation"],
                "status": r["status"],
                "current_snapshot_hash": r["current_snapshot_hash"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "duration_seconds": dur,
                "total_df_features_created": total_created,
                "unique_formula_count": uniq_total,
                "keep_count": keep_cnt,
                "watch_count": watch_cnt,
                "remove_count": remove_cnt,
                "candidate_count": cand_cnt,
                "active_discovery_pool": active_pool,
                "best_candidate_id": res_row["best_candidate_id"] if res_row else None,
                "best_composite_score": res_row["best_composite_score"] if res_row else None,
            })

        return results
    finally:
        conn.close()


def derive_human_readable_feature_name(
    formula_expression: str,
    generator_strategy: str,
    parent_features: list[str],
    fallback_id: str = "",
) -> str:
    """Deterministically derive a clean, readable feature display name from strategy, parents, and formula AST."""
    strat = str(generator_strategy or "").upper().strip()
    parents = [str(p).strip() for p in parent_features if str(p).strip()]
    expr = str(formula_expression or "").strip()

    if strat == "RATIO" and len(parents) >= 2:
        return f"{parents[0]}_to_{parents[1]}_ratio"
    elif strat == "INTERACTION" and len(parents) >= 2:
        return f"{parents[0]}_x_{parents[1]}"
    elif strat == "SPREAD" and len(parents) >= 2:
        return f"{parents[0]}_minus_{parents[1]}"
    elif strat == "NONLINEAR" and len(parents) >= 1:
        p0 = parents[0]
        if "log1p" in expr or "log" in expr:
            return f"log1p_{p0}"
        elif "tanh" in expr:
            return f"tanh_{p0}"
        elif "** 2" in expr or "sq" in expr:
            return f"sq_{p0}"
        elif "sqrt" in expr:
            return f"sqrt_{p0}"
        elif "sign" in expr:
            return f"sign_{p0}"
        return f"nonl_{p0}"
    elif strat == "COMPOSITE" and len(parents) >= 3:
        return f"{parents[0]}_minus_{parents[1]}_div_{parents[2]}"

    if len(parents) == 2:
        return f"{parents[0]}_{strat.lower()}_{parents[1]}"
    elif len(parents) == 1:
        return f"{strat.lower()}_{parents[0]}"

    return fallback_id or "discovered_feature"


def list_discovery_features(
    data_dir: str,
    pipeline_id: str | Sequence[str],
    generation: int | None = None,
    verdicts: list[str] | None = None,
    strategy: str | None = None,
    search_text: str | None = None,
    deduplicate_by_hash: bool = True,
) -> list[dict[str, Any]]:
    """Query features across one or multiple Discovery Pipelines with filtering and formula deduplication."""
    init_discovery_pipeline_tables(data_dir)
    try:
        from chain_replay_ml.research_registry.store import init_research_registry_tables
        init_research_registry_tables(data_dir)
    except Exception:
        pass

    if isinstance(pipeline_id, str):
        p_ids = [pipeline_id.strip()] if pipeline_id.strip() else []
    else:
        p_ids = [str(p).strip() for p in pipeline_id if str(p).strip()]

    if not p_ids:
        return []

    conn = connect_analysis_db(data_dir)
    try:
        # Load pipeline metadata map
        pipeline_info: dict[str, dict[str, str]] = {}
        placeholders = ",".join(["?"] * len(p_ids))
        dp_rows = conn.execute(
            f"SELECT pipeline_id, campaign_id, context_key, current_snapshot_hash FROM discovery_pipelines WHERE pipeline_id IN ({placeholders});",
            p_ids,
        ).fetchall()
        for dp in dp_rows:
            camp_id = dp["campaign_id"]
            res_row = None
            try:
                res_row = conn.execute(
                    "SELECT research_id FROM research_registry WHERE campaign_id = ? LIMIT 1;",
                    (camp_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                pass
            r_id = res_row["research_id"] if res_row else f"RESEARCH_{camp_id}"
            pipeline_info[dp["pipeline_id"]] = {
                "campaign_id": camp_id,
                "context_key": dp["context_key"],
                "snapshot_hash": dp["current_snapshot_hash"] or "",
                "research_id": r_id,
            }

        query = f"SELECT * FROM discovery_pipeline_features WHERE pipeline_id IN ({placeholders})"
        params: list[Any] = list(p_ids)

        if generation is not None and generation > 0:
            query += " AND generation_discovered = ?"
            params.append(generation)

        if strategy and strategy != "ALL":
            query += " AND generator_strategy = ?"
            params.append(strategy)

        if verdicts:
            norm_v = [v.upper() for v in verdicts if v]
            v_placeholders = ",".join(["?"] * len(norm_v))
            query += f" AND UPPER(lifecycle_status) IN ({v_placeholders})"
            params.extend(norm_v)

        query += " ORDER BY evidence_score DESC, feature_id ASC;"
        rows = conn.execute(query, params).fetchall()

        raw_features: list[dict[str, Any]] = []
        search_lower = str(search_text or "").strip().lower()

        for r in rows:
            meta = json.loads(r["metadata_json"] or "{}") if r["metadata_json"] else {}
            parents = json.loads(r["parent_features_json"] or "[]") if r["parent_features_json"] else []

            delta_auc = float(meta.get("delta_auc") or meta.get("delta_auc_mean") or 0.0)
            fold_cons = float(meta.get("fold_consistency") or meta.get("positive_fold_ratio") or 0.0)
            gov_rat = str(meta.get("governance_rationale") or meta.get("rationale") or "")

            f_id = r["feature_id"]
            p_id = r["pipeline_id"]
            p_meta = pipeline_info.get(p_id, {})
            camp_id = p_meta.get("campaign_id") or p_id.replace("DP_", "")
            ctx_k = p_meta.get("context_key") or "UNKNOWN"
            snap_hash = p_meta.get("snapshot_hash") or ""
            r_id = p_meta.get("research_id") or f"RESEARCH_{camp_id}"

            f_name = r["feature_name"]
            f_expr = r["formula_expression"]
            f_hash = r["formula_hash"]
            strat = r["generator_strategy"]
            status_norm = str(r["lifecycle_status"]).upper()
            d_ks = float(r["ks_statistic"] or 0.0)
            drift_sev = int(r["drift_severity"] or 0)
            ev_score = float(r["evidence_score"] or 0.0)
            gen = int(r["generation_discovered"] or 1)
            display_name = derive_human_readable_feature_name(f_expr, strat, parents, f_id)

            # Search text filter matching formula, feature_name, strategy, or parents
            if search_lower:
                matchable = f"{f_id} {f_name} {display_name} {f_expr} {strat} {' '.join(parents)} {gov_rat} {p_id} {r_id}".lower()
                if search_lower not in matchable:
                    continue

            raw_features.append({
                "feature_id": f_id,
                "feature_name": f_name,
                "display_name": display_name,
                "pipeline_id": p_id,
                "research_id": r_id,
                "campaign_id": camp_id,
                "context_key": ctx_k,
                "formula_hash": f_hash,
                "formula_expression": f_expr,
                "generator_strategy": strat,
                "parent_features": parents,
                "generation_discovered": gen,
                "discovery_snapshot_hash": snap_hash,
                "discovery_verdict": status_norm,
                "marginal_delta_auc": delta_auc,
                "ks_statistic": d_ks,
                "drift_severity": drift_sev,
                "evidence_score": ev_score,
                "fold_consistency": fold_cons,
                "governance_rationale": gov_rat,
                "co_discovered_pipelines": [],
                "co_discovery_count": 0,
                "created_at": r["created_at"],
            })

        if not deduplicate_by_hash:
            return raw_features

        # Deduplicate across multiple pipelines by mathematical formula hash
        grouped: dict[str, list[dict[str, Any]]] = {}
        for f in raw_features:
            grouped.setdefault(f["formula_hash"], []).append(f)

        results: list[dict[str, Any]] = []
        for f_hash, flist in grouped.items():
            if len(flist) == 1:
                results.append(flist[0])
            else:
                sorted_flist = sorted(
                    flist,
                    key=lambda x: (x["evidence_score"], x["marginal_delta_auc"]),
                    reverse=True,
                )
                primary = dict(sorted_flist[0])
                other_pids = sorted(list({o["pipeline_id"] for o in sorted_flist[1:] if o["pipeline_id"] != primary["pipeline_id"]}))
                primary["co_discovered_pipelines"] = other_pids
                primary["co_discovery_count"] = len(other_pids)
                results.append(primary)

        # Sort final results by evidence score descending
        results.sort(key=lambda x: (x["evidence_score"], x["marginal_delta_auc"]), reverse=True)
        return results
    finally:
        conn.close()


def validate_cross_pipeline_selection(
    basket: CrossPipelineSelectionBasket,
    target_context: str | None = None,
) -> tuple[bool, str, list[SelectedDiscoveryFeatureRef], list[dict[str, Any]]]:
    """Validate cross-pipeline feature selection, enforce context compatibility, and deduplicate identical formulas.

    Returns:
        tuple[bool, str, list[SelectedDiscoveryFeatureRef], list[dict]]:
            (is_valid, error_message, deduplicated_items, co_discovery_records)
    """
    items = basket.get_all()
    if not items:
        return False, "No features selected in basket.", [], []

    # Rule 1: No REMOVE features
    for it in items:
        if str(it.discovery_verdict).upper() == "REMOVE":
            return False, f"Feature '{it.feature_id}' is marked REMOVE and cannot be used in pipeline creation.", [], []

    # Rule 2: Context Key Compatibility
    if target_context:
        base_ctx_norm = target_context.strip().lower()
        for it in items:
            it_ctx = str(it.context_key or "").strip().lower()
            if it_ctx and it_ctx != base_ctx_norm:
                # Basic market & task compatibility check
                m_it = it_ctx.split(":")[0] if ":" in it_ctx else it_ctx
                m_target = base_ctx_norm.split(":")[0] if ":" in base_ctx_norm else base_ctx_norm
                if m_it != m_target:
                    return False, f"Context mismatch: Feature '{it.feature_id}' belongs to context '{it.context_key}' which is incompatible with target context '{target_context}'.", [], []

    # Rule 3: Deduplicate Identical Mathematical Formulas across Pipelines
    by_hash: dict[str, list[SelectedDiscoveryFeatureRef]] = {}
    for it in items:
        by_hash.setdefault(it.formula_hash, []).append(it)

    deduped_items: list[SelectedDiscoveryFeatureRef] = []
    co_discovery_records: list[dict[str, Any]] = []

    for f_hash, hash_items in by_hash.items():
        if len(hash_items) == 1:
            deduped_items.append(hash_items[0])
        else:
            # Sort by evidence_score desc, delta_auc desc
            sorted_items = sorted(
                hash_items,
                key=lambda x: (x.evidence_score, x.marginal_delta_auc),
                reverse=True,
            )
            primary = sorted_items[0]
            deduped_items.append(primary)

            co_discovery_records.append({
                "formula_hash": f_hash,
                "formula_expression": primary.formula_expression,
                "primary_feature_id": primary.feature_id,
                "primary_pipeline_id": primary.pipeline_id,
                "primary_evidence_score": primary.evidence_score,
                "co_discovered_instances": [
                    {
                        "feature_id": other.feature_id,
                        "pipeline_id": other.pipeline_id,
                        "research_id": other.research_id,
                        "evidence_score": other.evidence_score,
                        "marginal_delta_auc": other.marginal_delta_auc,
                    }
                    for other in sorted_items[1:]
                ],
            })

    return True, "Selection validated successfully.", deduped_items, co_discovery_records


def create_candidate_discovery_pipeline(
    data_dir: str,
    req: PipelineCreationRequest,
    basket: CrossPipelineSelectionBasket,
) -> PipelineCreationResult:
    """Create a new discovery_experimental pipeline in pipeline_registry_store.json.

    Invariants:
    1. Pipeline contains ONLY the selected, deduplicated Discovery Features (DF_*).
    2. ZERO Base Pipeline features (PL_0001) are injected.
    3. ZERO Permanent Feature Registry features are injected.
    4. PL_0001 and permanent Feature Registry remain strictly immutable.
    5. New pipeline type is strictly 'discovery_experimental'.
    6. Full multi-pipeline and human-readable AST provenance is persisted in pipeline_registry_store.json.
    """
    is_valid, msg, dedup_items, co_disc = validate_cross_pipeline_selection(basket, req.context_key)
    if not is_valid:
        return PipelineCreationResult(
            success=False,
            pipeline_id="",
            pipeline_name=req.name,
            base_feature_count=0,
            discovered_feature_count=0,
            total_feature_count=0,
            pipeline_snapshot_id="",
            message=msg,
            errors=[msg],
        )

    # 1. Load pipeline registry store
    pr_doc = load_pl_store(data_dir)

    # 2. Extract strictly the selected Discovery Features (Zero Base / Registry features)
    df_features = [it.feature_id for it in dedup_items]
    total_count = len(df_features)

    # 3. Create new pipeline entry
    try:
        new_rec = create_pipeline(
            pr_doc,
            name=req.name,
            pipeline_type="discovery_experimental",
            status="ready",
        )
        new_pid = new_rec["pipeline_id"]

        # Populate features and metadata strictly with selected discovery features
        target_rec = pr_doc["pipelines"][new_pid]
        target_rec["type"] = "discovery_experimental"
        target_rec["candidate_features"] = df_features
        target_rec["registry_feature_ids"] = []
        target_rec["context_key"] = req.context_key
        target_rec["base_feature_count"] = 0
        target_rec["discovered_feature_count"] = total_count
        target_rec["total_feature_count"] = total_count

        # Build provenance metadata
        source_dps = sorted(list({it.pipeline_id for it in dedup_items}))
        source_rids = sorted(list({it.research_id for it in dedup_items if it.research_id}))
        source_camps = sorted(list({it.campaign_id for it in dedup_items if it.campaign_id}))

        target_rec["provenance_metadata"] = {
            "creation_source": "DISCOVERY_FEATURE_DASHBOARD",
            "creation_mode": "CROSS_DISCOVERY_PIPELINE_SELECTION",
            "created_by": "QUANTITATIVE_RESEARCHER",
            "created_at": _utc_now_iso(),
            "description": req.description,
            "pipeline_type": "DISCOVERY_EXPERIMENTAL",
            "source_discovery_pipelines": source_dps,
            "source_research_ids": source_rids,
            "source_campaign_ids": source_camps,
            "selected_features_provenance": [it.to_dict() for it in dedup_items],
            "co_discovery_features": co_disc,
        }

        # Build cryptographic pipeline snapshot
        snap_id = f"PL_SNAP_{hashlib.sha256((new_pid + str(df_features)).encode()).hexdigest()[:16]}"
        try:
            built_snap = build_pipeline_snapshot(pr_doc, new_pid)
            if built_snap:
                snap_id = built_snap
        except Exception:
            pass
        target_rec["pipeline_snapshot_id"] = snap_id

        # Save store
        save_pl_store(data_dir, pr_doc)

        return PipelineCreationResult(
            success=True,
            pipeline_id=new_pid,
            pipeline_name=req.name,
            base_feature_count=0,
            discovered_feature_count=total_count,
            total_feature_count=total_count,
            pipeline_snapshot_id=snap_id,
            message=f"Discovery Feature Pipeline '{new_pid}' ({req.name}) successfully created with {total_count} Discovery features.",
            co_discovery_count=len(co_disc),
        )

    except Exception as ex:
        return PipelineCreationResult(
            success=False,
            pipeline_id="",
            pipeline_name=req.name,
            base_feature_count=0,
            discovered_feature_count=len(dedup_items),
            total_feature_count=len(dedup_items),
            pipeline_snapshot_id="",
            message=f"Failed to create discovery pipeline: {ex}",
            errors=[str(ex)],
        )
