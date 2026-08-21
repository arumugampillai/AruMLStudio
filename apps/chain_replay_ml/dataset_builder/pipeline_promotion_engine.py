"""Automated Pipeline Snapshot Promotion Engine (Phase A).

Promotes validated Research Champions and High-Evidence Feature Sets from Autonomous
Research into authoritative, immutable Pipeline Snapshots (PL_XXXX) and Feature Registry linkages.

Preserves the exact champion feature set without modification or secondary filtering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from chain_replay_ml.candidate_generation.types import CandidateSpec
from chain_replay_ml.dataset_builder.feature_registry_store import (
    ensure_feature_ids,
    load_store as load_fr_store,
    save_store as save_fr_store,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    add_candidate_features,
    build_pipeline_snapshot,
    create_pipeline,
    load_store as load_pl_store,
    save_store as save_pl_store,
    set_registry_members,
)
from chain_replay_ml.production_validation.evidence_store import (
    get_connection as get_evidence_conn,
)
from chain_replay_ml.research_memory.db import connect_analysis_db

logger = logging.getLogger(__name__)

# Standard forbidden targets and timestamp/price metadata
FORBIDDEN_METADATA_COLUMNS = frozenset({
    "timestamp", "datetime", "date", "time", "token", "symbol", "expiry",
    "option_type", "instrument_type", "day", "trading_day", "open", "high", "low", "close", "ltp",
})


@dataclass(frozen=True)
class PromotionValidationReport:
    """Audit report from candidate feature set validation."""
    eligible: bool
    status: str  # "READY" | "WARNING" | "BLOCKED"
    feature_count: int
    exact_features: list[str]
    keep_count: int = 0
    watch_count: int = 0
    remove_count: int = 0
    mean_evidence_score: float = 0.0
    blocked_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelinePromotionResult:
    """Payload returned upon successful promotion of a research candidate to a pipeline snapshot."""
    pipeline_id: str
    pipeline_snapshot_id: str
    pipeline_name: str
    feature_count: int
    registry_feature_ids_count: int
    dataset_name: str
    dataset_snapshot_hash: str
    candidate_id: str
    campaign_id: str | None
    promoted_at: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_candidate_for_promotion(
    data_dir: str,
    spec: CandidateSpec,
    *,
    dataset_target_columns: Sequence[str] | None = None,
) -> PromotionValidationReport:
    """Validate a candidate model's feature set against promotion governance invariants."""
    features = list(dict.fromkeys(str(f).strip() for f in spec.features if f and str(f).strip()))
    blocked_reasons: list[str] = []
    warnings: list[str] = []

    if len(features) < 5:
        blocked_reasons.append(f"Insufficient feature cardinality ({len(features)} features; minimum 5 required).")

    # 1. Target leakage check
    targets = set(dataset_target_columns or [
        "future_ltp_10s", "future_ltp_1m", "label_up_2pct_5m", "label_up_3pct_5m",
        "label_up_4pct_5m", "label_up_5pct_5m", "label_up_6pct_5m", "label_up_gt6pct_5m"
    ])
    leaked = [f for f in features if f in targets or f.startswith("label_") or f.startswith("target_") or f.startswith("future_")]
    if leaked:
        blocked_reasons.append(f"Target leakage detected in candidate feature set: {leaked}")

    # 2. Metadata columns check
    meta_cols = [f for f in features if f.lower() in FORBIDDEN_METADATA_COLUMNS]
    if meta_cols:
        blocked_reasons.append(f"Non-feature metadata columns detected: {meta_cols}")

    # 3. Feature Studio Evidence Query
    keep_cnt = 0
    watch_cnt = 0
    rem_cnt = 0
    scores_list: list[float] = []

    try:
        conn = get_evidence_conn(data_dir)
        try:
            placeholders = ",".join("?" for _ in features)
            cur = conn.execute(
                f"SELECT feature_name, recommendation, relative_imp_drop, evidence_detail_json FROM recommendation_evidence WHERE model_name = ? AND feature_name IN ({placeholders})",
                (spec.candidate_id, *features),
            )
            rows = cur.fetchall()
            ev_map: dict[str, dict[str, Any]] = {}
            for r in rows:
                fn = r["feature_name"]
                rec = str(r["recommendation"] or "").upper()
                ks = float(r["relative_imp_drop"] or 0.0)
                d_json = {}
                try:
                    d_json = json.loads(r["evidence_detail_json"] or "{}")
                except Exception:
                    pass
                ev_map[fn] = {"rec": rec, "ks": ks, "score": float(d_json.get("evidence_score", 50.0))}

            # Query feature_context_summary as fallback for any unmapped
            unmapped = [f for f in features if f not in ev_map]
            if unmapped:
                u_placeholders = ",".join("?" for _ in unmapped)
                cur2 = conn.execute(
                    f"SELECT feature_name, last_recommendation, evidence_score FROM feature_context_summary WHERE feature_name IN ({u_placeholders})",
                    tuple(unmapped),
                )
                for r2 in cur2.fetchall():
                    fn = r2["feature_name"]
                    ev_map[fn] = {
                        "rec": str(r2["last_recommendation"] or "WATCH").upper(),
                        "ks": 0.10,
                        "score": float(r2["evidence_score"] or 50.0),
                    }

            for f in features:
                info = ev_map.get(f, {"rec": "WATCH", "ks": 0.0, "score": 50.0})
                rec = info["rec"]
                score = info["score"]
                ks = info["ks"]
                scores_list.append(score)

                if rec == "KEEP":
                    keep_cnt += 1
                elif rec == "REMOVE":
                    rem_cnt += 1
                    if ks > 0.35:
                        warnings.append(f"Feature '{f}' marked REMOVE with severe drift (KS={ks:.3f}).")
                else:
                    watch_cnt += 1

        finally:
            conn.close()
    except Exception as exc:
        warnings.append(f"Could not query Feature Studio evidence store: {exc}")

    mean_score = sum(scores_list) / max(1, len(scores_list)) if scores_list else 50.0
    status = "BLOCKED" if blocked_reasons else ("WARNING" if rem_cnt > 0 else "READY")
    eligible = len(blocked_reasons) == 0

    return PromotionValidationReport(
        eligible=eligible,
        status=status,
        feature_count=len(features),
        exact_features=features,
        keep_count=keep_cnt,
        watch_count=watch_cnt,
        remove_count=rem_cnt,
        mean_evidence_score=round(mean_score, 2),
        blocked_reasons=blocked_reasons,
        warnings=warnings,
    )


def promote_candidate_to_pipeline_snapshot(
    data_dir: str,
    spec: CandidateSpec,
    *,
    campaign_id: str | None = None,
    dataset_name: str | None = None,
    override_warnings: bool = True,
) -> PipelinePromotionResult:
    """Promote a candidate model's exact feature set to an authoritative Pipeline Snapshot (PL_XXXX).

    Preserves 100% of the candidate's exact feature universe without secondary pruning.
    """
    val_report = validate_candidate_for_promotion(data_dir, spec)
    if not val_report.eligible:
        raise ValueError(f"Candidate {spec.candidate_id} is BLOCKED from promotion: {val_report.blocked_reasons}")

    camp_id = campaign_id or (spec.lineage.campaign_id if spec.lineage else "CAMP_PROMOTED")
    ds_name = dataset_name or "analysis_dataset"
    now_iso = datetime.now(timezone.utc).isoformat()
    features = list(val_report.exact_features)

    # 1. Resolve or allocate permanent Feature Registry IDs (FR_XXXX)
    fr_store = load_fr_store(data_dir)
    name_to_fid = ensure_feature_ids(fr_store, features)
    reg_feature_ids = [name_to_fid[f] for f in features if f in name_to_fid]
    save_fr_store(data_dir, fr_store)

    # 2. Allocate and Create Pipeline Record in pipeline_registry_store.json
    pl_store = load_pl_store(data_dir)
    display_title = f"Pipeline — Promoted from {spec.candidate_id}"
    pl_rec = create_pipeline(
        pl_store,
        name=display_title,
        pipeline_type="auto",
        status="ready",
    )
    pipeline_id = pl_rec["pipeline_id"]

    # Associate Feature Registry IDs and exact Candidate Features
    set_registry_members(pl_store, pipeline_id, reg_feature_ids)
    add_candidate_features(pl_store, pipeline_id, features, replace=True)

    # Embed complete research provenance into pipeline metadata
    provenance_doc = {
        "source_candidate_id": spec.candidate_id,
        "source_campaign_id": camp_id,
        "source_context_key": spec.context_key,
        "source_dataset_name": ds_name,
        "source_dataset_snapshot_hash": spec.dataset_snapshot_hash,
        "feature_elimination_strategy": spec.feature_elimination_strategy or "NONE",
        "algorithm": spec.algorithm,
        "exact_feature_count": len(features),
        "governance_breakdown": {
            "keep_count": val_report.keep_count,
            "watch_count": val_report.watch_count,
            "remove_count": val_report.remove_count,
            "mean_evidence_score": val_report.mean_evidence_score,
        },
        "promoted_at": now_iso,
    }

    # Update pipeline record with provenance and build snapshot
    updated_rec = pl_store["pipelines"][pipeline_id]
    updated_rec["provenance"] = provenance_doc
    save_pl_store(data_dir, pl_store)

    snapshot = build_pipeline_snapshot(updated_rec, pipeline_id=pipeline_id)
    snapshot_id = snapshot["pipeline_snapshot_id"]

    # 3. Update Experimental Lineage in feature_recommendation_evidence.db
    try:
        from chain_replay_ml.production_validation.dataset_context import build_dataset_context
        ctx_obj = build_dataset_context(
            market=spec.market or "NIFTY",
            sampling_interval_sec=spec.sampling_interval_sec or 6,
        )
        ev_conn = get_evidence_conn(data_dir)
        try:
            # Ensure context exists in dataset_contexts
            ev_conn.execute(
                """
                INSERT OR IGNORE INTO dataset_contexts (
                    context_id, market, sampling_interval_sec, sampling_label, sliding_window, feature_project_id, context_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ctx_obj.context_id, ctx_obj.market, ctx_obj.sampling_interval_sec,
                    ctx_obj.sampling_label, ctx_obj.sliding_window, ctx_obj.feature_project_id,
                    spec.context_key, now_iso,
                ),
            )
            for feat in features:
                fid = name_to_fid.get(feat, "FR_UNKNOWN")
                lineage_id = f"lin_{ctx_obj.context_id}_{pipeline_id}_{snapshot_id}_{feat}"
                ev_conn.execute(
                    """
                    INSERT INTO experimental_lineage_summary (
                        lineage_id, context_id, pipeline_id, pipeline_snapshot_id,
                        feature_name, feature_identity_key, total_runs, keep_runs, watch_runs, remove_runs,
                        unique_models_count, consecutive_keep_count, consecutive_remove_count,
                        lineage_evidence_score, lifecycle_status, last_recommendation, last_validated_at,
                        projection_policy_id, projection_policy_version, projection_rebuilt_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, 0, 0, 1, 1, 0, 100.0, 'promotion_candidate', 'KEEP', ?, 'promotion_v1', 1, ?)
                    ON CONFLICT(context_id, pipeline_id, pipeline_snapshot_id, feature_name) DO UPDATE SET
                        lifecycle_status = 'promotion_candidate',
                        last_validated_at = excluded.last_validated_at,
                        projection_rebuilt_at = excluded.projection_rebuilt_at
                    """,
                    (
                        lineage_id, ctx_obj.context_id, pipeline_id, snapshot_id,
                        feat, fid, now_iso, now_iso,
                    ),
                )
            ev_conn.commit()
        finally:
            ev_conn.close()
    except Exception as exc:
        logger.warning(f"Could not record experimental lineage for {pipeline_id}: {exc}")

    # 4. Log PIPELINE_SNAPSHOT_PROMOTED Event in analysis.db
    try:
        an_conn = connect_analysis_db(data_dir)
        try:
            # Ensure campaign_id exists in overnight_campaigns to satisfy foreign key
            an_conn.execute(
                """
                INSERT OR IGNORE INTO overnight_campaigns (
                    campaign_id, config_hash, config_json, status, stop_reason, current_generation,
                    total_candidates_generated, total_candidates_trained, total_candidates_evaluated,
                    total_candidates_excluded, total_candidates_pruned, total_failures,
                    best_composite_score, best_trading_score, best_model_score, starting_best_score,
                    start_time_iso, last_update_iso, warnings_json
                ) VALUES (?, 'promoted_hash', '{}', 'COMPLETED', 'PROMOTED', 0, 1, 1, 1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, ?, ?, '[]')
                """,
                (camp_id, now_iso, now_iso),
            )
            an_conn.execute(
                """
                INSERT INTO overnight_campaign_events (
                    campaign_id, generation_number, event_type, candidate_id, message, event_details_json, created_at
                ) VALUES (?, ?, 'PIPELINE_SNAPSHOT_PROMOTED', ?, ?, ?, ?)
                """,
                (
                    camp_id,
                    spec.lineage.generation_number if spec.lineage else 0,
                    spec.candidate_id,
                    f"Promoted candidate {spec.candidate_id} to Pipeline Snapshot {pipeline_id} ({snapshot_id}, {len(features)} features).",
                    json.dumps(provenance_doc),
                    now_iso,
                ),
            )
            an_conn.commit()
        finally:
            an_conn.close()
    except Exception as exc:
        logger.warning(f"Could not log promotion event: {exc}")

    return PipelinePromotionResult(
        pipeline_id=pipeline_id,
        pipeline_snapshot_id=snapshot_id,
        pipeline_name=display_title,
        feature_count=len(features),
        registry_feature_ids_count=len(reg_feature_ids),
        dataset_name=ds_name,
        dataset_snapshot_hash=spec.dataset_snapshot_hash,
        candidate_id=spec.candidate_id,
        campaign_id=camp_id,
        promoted_at=now_iso,
        status="PROMOTED",
    )
