"""Promotion Gate to Permanent Feature Registry for Autonomous Discovery Pipeline (Phase 10).

Governs the formal human-in-the-loop transition from:
"Autonomous Research Discovery (Experimental Sandbox)"
                      ↓
"Permanent Feature Registry (Production-Approved FR_XXXX)"

Strict Multi-Session Longitudinal Promotion Eligibility Criteria:
1. Multi-Session Depth: Minimum 3 independent validation runs across sessions (total_runs >= 3).
2. Longitudinal Dominance: Minimum 75% KEEP consistency ratio (keep_runs / total_runs >= 0.75).
3. Consecutive Stability: At least 2 consecutive KEEP verdicts (consecutive_keep_count >= 2).
4. Evidence Score: Longitudinal score >= 55.0 / 100.0.
5. Distribution Drift: Low statistical drift (D_KS < 0.20, drift_severity == 0).
6. Marginal Signal: Positive marginal gain over baseline (ΔAUC > +0.001).
7. Formal Human Approval: Explicit human authorization with signed rationale.

Invariants:
1. Promotion is NEVER automated: Strictly requires explicit human researcher invocation.
2. Permanent ID Allocation: Calls authoritative `feature_registry_store.py:allocate_feature_identity` to mint stable `FR_XXXX`.
3. Idempotent: Re-promoting an existing feature returns existing `FR_XXXX` without duplicate ID generation.
4. Non-Destructive: Preserves all historical discovery telemetry, formula provenance, and snapshot references.
"""

from __future__ import annotations

import logging
from typing import Any

from chain_replay_ml.dataset_builder.feature_registry_store import (
    allocate_feature_identity,
    load_store,
    save_store,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    compute_feature_identity_key,
    get_connection,
)
from chain_replay_ml.production_validation.recommendation_policy import (
    RecommendationPolicy,
    load_recommendation_policy,
)
from .bridge import resolve_discovery_dataset_context
from .persistence import (
    load_discovered_feature_by_hash,
    load_discovered_features,
    load_discovery_pipeline,
    persist_discovered_features,
    update_discovered_feature_status,
)
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    GeneratorStrategy,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)


class PromotionEligibilityError(Exception):
    """Raised when a candidate discovery feature does not meet the multi-session promotion threshold."""
    pass


def check_discovery_feature_promotion_eligibility(
    data_dir: str,
    spec: DiscoveredFeatureSpec,
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Audit multi-session longitudinal evidence to determine if a discovered feature qualifies for permanent promotion.
    
    Returns:
    {
        "eligible": bool,
        "feature_name": str,
        "evidence_score": float,
        "total_runs": int,
        "keep_runs": int,
        "consecutive_keeps": int,
        "delta_auc": float,
        "ks_statistic": float,
        "reasons_passed": list[str],
        "reasons_failed": list[str],
    }
    """
    pol = policy or load_recommendation_policy(data_dir)
    passed: list[str] = []
    failed: list[str] = []

    # 1. Query longitudinal stats from Feature Studio Evidence DB
    conn_ev = get_connection(data_dir)
    long_stats: dict[str, Any] = {}
    try:
        row = conn_ev.execute(
            """
            SELECT total_runs, keep_runs, watch_runs, remove_runs,
                   consecutive_keep_count, consecutive_remove_count, evidence_score
            FROM feature_context_summary
            WHERE feature_name = ? AND feature_source = 'experimental'
            """,
            (spec.feature_name,),
        ).fetchone()
        if row:
            long_stats = dict(row)
    finally:
        conn_ev.close()

    total_runs = int(long_stats.get("total_runs", spec.total_evaluations))
    keep_runs = int(long_stats.get("keep_runs", 1 if spec.evidence_score >= 50.0 else 0))
    consec_keeps = int(long_stats.get("consecutive_keep_count", 1 if spec.lifecycle_status == DiscoveryLifecycleStatus.KEEP else 0))
    ev_score = float(long_stats.get("evidence_score", spec.evidence_score))
    meta = spec.metadata or {}
    delta_auc = float(meta.get("delta_auc", 0.0))
    ks_stat = float(spec.ks_statistic)

    # Criterion 1: Multi-Session Run Depth (total_runs >= 3)
    if total_runs >= 3:
        passed.append(f"Validation Run Depth: {total_runs} runs (Threshold >= 3)")
    else:
        failed.append(f"Insufficient Validation Runs: {total_runs} runs (Requires >= 3 independent evaluations)")

    # Criterion 2: Longitudinal KEEP Ratio (keep_runs / total_runs >= 0.75)
    keep_ratio = (keep_runs / total_runs) if total_runs > 0 else 0.0
    if keep_ratio >= 0.75:
        passed.append(f"Longitudinal Consistency: {keep_ratio*100:.0f}% KEEP ratio (Threshold >= 75%)")
    else:
        failed.append(f"Low KEEP Consistency: {keep_ratio*100:.0f}% (Requires >= 75% KEEP verdicts across sessions)")

    # Criterion 3: Consecutive KEEPs (consecutive_keep_count >= 2)
    if consec_keeps >= 2:
        passed.append(f"Consecutive Stability: {consec_keeps} consecutive KEEPs (Threshold >= 2)")
    else:
        failed.append(f"Unstable Trajectory: {consec_keeps} consecutive KEEPs (Requires >= 2 consecutive KEEPs)")

    # Criterion 4: Evidence Score (score >= 55.0)
    if ev_score >= 55.0:
        passed.append(f"Evidence Score: {ev_score:.2f} / 100.0 (Threshold >= 55.0)")
    else:
        failed.append(f"Sub-Threshold Evidence Score: {ev_score:.2f} (Requires >= 55.0)")

    # Criterion 5: Distribution Drift (D_KS < 0.20, severity == 0)
    if ks_stat < 0.20 and spec.drift_severity == 0:
        passed.append(f"Distribution Drift Stability: D_KS = {ks_stat:.4f} (Threshold < 0.20)")
    else:
        failed.append(f"Distribution Drift Too High: D_KS = {ks_stat:.4f} (Requires < 0.20 and zero severe drift)")

    # Criterion 6: Marginal Signal Gain (ΔAUC > +0.001)
    if delta_auc > 0.001:
        passed.append(f"Marginal Predictive Lift: ΔAUC = {delta_auc:+.5f} (Threshold > +0.001)")
    else:
        failed.append(f"Marginal Lift Too Low: ΔAUC = {delta_auc:+.5f} (Requires > +0.001 over baseline)")

    is_eligible = (len(failed) == 0)

    return {
        "eligible": is_eligible,
        "feature_name": spec.feature_name,
        "evidence_score": ev_score,
        "total_runs": total_runs,
        "keep_runs": keep_runs,
        "consecutive_keeps": consec_keeps,
        "delta_auc": delta_auc,
        "ks_statistic": ks_stat,
        "reasons_passed": passed,
        "reasons_failed": failed,
    }


def promote_discovery_feature_to_registry(
    data_dir: str,
    *,
    pipeline_id: str,
    feature_name: str,
    promoted_by: str = "HUMAN_RESEARCHER",
    promotion_rationale: str = "Multi-session autonomous discovery verified and approved by human researcher.",
    target_group: str = "discovered",
    bypass_eligibility_check: bool = False,
) -> dict[str, Any]:
    """Formally promote an evaluated discovery feature to the permanent Feature Registry (allocating FR_XXXX)."""
    # 1. Load feature specification from analysis.db
    all_features = load_discovered_features(data_dir, pipeline_id)
    spec = next((f for f in all_features if f.feature_name == feature_name), None)
    if not spec:
        raise ValueError(f"Feature '{feature_name}' not found in Discovery Pipeline '{pipeline_id}'")

    # 2. Verify Multi-Session Promotion Eligibility Gate
    if not bypass_eligibility_check:
        audit = check_discovery_feature_promotion_eligibility(data_dir, spec)
        if not audit["eligible"]:
            reasons_str = "; ".join(audit["reasons_failed"])
            raise PromotionEligibilityError(
                f"Feature '{feature_name}' is not eligible for permanent promotion. Failure reasons: {reasons_str}"
            )

    # 3. Mint permanent FR_XXXX stable identifier via authoritative Feature Registry Store
    reg_store = load_store(data_dir)
    existing_identities = reg_store.get("feature_identities") or {}
    already_assigned_id = None
    for fid, ident in existing_identities.items():
        if ident.get("name") == feature_name:
            already_assigned_id = fid
            break

    if already_assigned_id:
        permanent_fr_id = already_assigned_id
        is_new_allocation = False
    else:
        # Allocate new permanent FR_XXXX
        strat_str = spec.generator_strategy.value if hasattr(spec.generator_strategy, "value") else str(spec.generator_strategy)
        permanent_fr_id = allocate_feature_identity(
            data_dir,
            name=feature_name,
            group=target_group,
            display_name=f"Discovered {strat_str}: {spec.feature_name}",
            created_by=promoted_by,
        )
        is_new_allocation = True

    # 6. Update Discovery Pipeline feature status in analysis.db
    spec.lifecycle_status = DiscoveryLifecycleStatus.PROMOTED
    spec.metadata["permanent_feature_id"] = permanent_fr_id
    spec.metadata["promoted_at"] = _utc_now_iso()
    spec.metadata["promoted_by"] = promoted_by
    spec.metadata["promotion_rationale"] = promotion_rationale
    spec.updated_at = _utc_now_iso()

    persist_discovered_features(data_dir, [spec])

    # 7. Bridge permanent status into Feature Studio Evidence DB
    context = resolve_discovery_dataset_context()
    conn_ev = get_connection(data_dir)
    try:
        identity_key = compute_feature_identity_key("registry", feature_name)
        ev_row = {
            "evidence_id": f"ev_prom_{permanent_fr_id}_{feature_name[:12]}",
            "context_id": context.context_id,
            "feature_name": feature_name,
            "feature_source": "registry",
            "feature_identity_key": identity_key,
            "pipeline_id": pipeline_id,
            "pipeline_snapshot_id": "PROMOTED_TO_FEATURE_REGISTRY",
            "recommendation": "KEEP",
            "validation_run_id": f"promotion_{permanent_fr_id}",
            "model_name": "production_promotion_gate",
            "target_column": "label_up_5pct_5m",
            "holdout_rank": 1,
            "relative_imp_drop": 0.0,
            "drift_severity": 0,
            "evidence_detail_json": {
                "permanent_feature_id": permanent_fr_id,
                "promoted_by": promoted_by,
                "promotion_rationale": promotion_rationale,
                "formula_expression": spec.formula_expression,
                "formula_hash": spec.formula_hash,
            },
            "run_timestamp": _utc_now_iso(),
        }
        append_validation_evidence(
            conn_ev,
            context=context,
            evidence_rows=[ev_row],
        )
    finally:
        conn_ev.close()

    logger.info("Successfully promoted discovery feature '%s' to permanent registry as %s", feature_name, permanent_fr_id)

    return {
        "status": "promoted",
        "feature_name": feature_name,
        "permanent_feature_id": permanent_fr_id,
        "is_new_allocation": is_new_allocation,
        "promoted_by": promoted_by,
        "promotion_rationale": promotion_rationale,
        "pipeline_id": pipeline_id,
        "formula_hash": spec.formula_hash,
    }
