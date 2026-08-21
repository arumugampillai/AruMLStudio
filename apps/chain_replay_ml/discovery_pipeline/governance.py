"""Autonomous Research Discovery Pipeline Governance Decision Engine (Phase 6).

Applies multi-factor empirical governance decisions (KEEP / WATCH / REMOVE)
to experimental Discovery Pipeline features based on Phase 4/5 evaluation evidence
and longitudinal history from `feature_recommendation_evidence.db`.

Flow:
Phase 4 Walk-Forward Telemetry (ΔAUC, D_KS, Consistency)
      ↓
Phase 5 Longitudinal History (total_runs, keep_runs, remove_runs)
      ↓
Phase 6 Governance Decision Engine
      ├── 🟢 KEEP: Strong positive marginal gain, low drift, high fold consistency
      ├── 🟡 WATCH: Marginal gain or moderate drift; retained for observation
      └── 🔴 REMOVE: Consistently negative contribution or severe distribution drift
      ↓
Discovery Pipeline Update in analysis.db: discovery_pipeline_features

Invariants:
1. Zero Permanent Registry Mutation: NEVER touches feature_registry_store.json or pipeline_registry_store.json.
2. Non-Destructive REMOVE: Pruning excludes features from active candidate pools without deleting historical evidence.
3. Longitudinal Stability: Decisions account for accumulated evidence over multiple generations, not single-run noise.
4. Campaign Isolation: Governance decisions strictly apply to target DP_<campaign_id>.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from chain_replay_ml.production_validation.evidence_store import get_connection
from chain_replay_ml.production_validation.recommendation_policy import (
    RecommendationPolicy,
    load_recommendation_policy,
)
from .persistence import (
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


def evaluate_discovery_governance_decision(
    spec: DiscoveredFeatureSpec,
    longitudinal_stats: dict[str, Any] | None = None,
    policy: RecommendationPolicy | None = None,
) -> tuple[DiscoveryLifecycleStatus, str]:
    """Evaluate deterministic KEEP / WATCH / REMOVE verdict based on empirical telemetry and longitudinal history.
    
    Returns:
    (verdict, rationale_string)
    """
    # Invariant: Base features and promoted features are strictly protected from REMOVE governance
    if spec.lifecycle_status == DiscoveryLifecycleStatus.PROMOTED:
        return DiscoveryLifecycleStatus.PROMOTED, "Permanent Feature Registry member — immutable under experimental governance."

    pol = policy or RecommendationPolicy()
    stats_dict = longitudinal_stats or {}

    # Extract Phase 4 metrics
    meta = spec.metadata or {}
    delta_auc = float(meta.get("delta_auc", 0.0))
    consistency = float(meta.get("fold_consistency", 0.5))
    ks_stat = float(spec.ks_statistic)
    drift_sev = int(spec.drift_severity)
    evidence_score = float(spec.evidence_score)
    total_evals = int(spec.total_evaluations)

    # Extract longitudinal counters from Evidence DB
    total_runs = int(stats_dict.get("total_runs", total_evals))
    keep_runs = int(stats_dict.get("keep_runs", 1 if evidence_score >= 50.0 else 0))
    remove_runs = int(stats_dict.get("remove_runs", 0))
    consec_remove = int(stats_dict.get("consecutive_remove_count", 0))
    consec_keep = int(stats_dict.get("consecutive_keep_count", 0))

    # 1. REMOVE Conditions (Consistently poor or severe drift)
    # Severe drift (D_KS > 0.35) or heavy negative contribution (ΔAUC < -0.005)
    if ks_stat > 0.35 or drift_sev >= 2:
        return DiscoveryLifecycleStatus.REMOVE, f"Severe KS distribution drift (D_KS={ks_stat:.4f})"

    if consec_remove >= 2 or (total_runs >= 2 and remove_runs > keep_runs and delta_auc < -0.002):
        return DiscoveryLifecycleStatus.REMOVE, f"Consecutive negative validation runs (ΔAUC={delta_auc:.5f}, remove_runs={remove_runs})"

    if total_runs >= 1 and delta_auc < -0.008:
        return DiscoveryLifecycleStatus.REMOVE, f"Excessive negative marginal predictive contribution (ΔAUC={delta_auc:.5f})"

    if consistency < 0.25 and total_runs >= 1:
        return DiscoveryLifecycleStatus.REMOVE, f"Low fold consistency across walk-forward splits ({consistency*100:.0f}%)"

    # 2. KEEP Conditions (Strong positive marginal gain, low drift, high fold consistency)
    if delta_auc > 0.001 and consistency >= 0.60 and ks_stat < 0.20 and evidence_score >= 52.0:
        return DiscoveryLifecycleStatus.KEEP, f"Statistically positive marginal gain (ΔAUC=+{delta_auc:.5f}, consistency={consistency*100:.0f}%, D_KS={ks_stat:.4f})"

    if consec_keep >= 2 or (total_runs >= 2 and keep_runs >= 2 and evidence_score >= 50.0 and ks_stat < 0.25):
        return DiscoveryLifecycleStatus.KEEP, f"Longitudinally stable positive validation across {total_runs} runs (score={evidence_score:.1f})"

    if delta_auc > 0.0 and consistency >= 0.50 and ks_stat < 0.15 and evidence_score >= 50.0:
        return DiscoveryLifecycleStatus.KEEP, f"Positive marginal gain over baseline (ΔAUC=+{delta_auc:.5f}, score={evidence_score:.1f})"

    # 3. WATCH Conditions (Marginal / mixed evidence, candidate observation)
    return DiscoveryLifecycleStatus.WATCH, f"Marginal / mixed evidence retained for observation (ΔAUC={delta_auc:.5f}, consistency={consistency*100:.0f}%, score={evidence_score:.1f})"


def run_discovery_pipeline_governance(
    data_dir: str,
    *,
    pipeline_id: str,
    campaign_id: str,
    context_key: str = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Execute complete governance review on all discovered features for an active Discovery Pipeline."""
    pol = policy or load_recommendation_policy(data_dir)
    features = load_discovered_features(data_dir, pipeline_id)
    if not features:
        return {
            "pipeline_id": pipeline_id,
            "campaign_id": campaign_id,
            "total_reviewed": 0,
            "keep_count": 0,
            "watch_count": 0,
            "remove_count": 0,
            "decisions": [],
        }

    # Query longitudinal evidence summary from feature_recommendation_evidence.db
    conn_ev = get_connection(data_dir)
    longitudinal_map: dict[str, dict[str, Any]] = {}
    try:
        rows = conn_ev.execute(
            """
            SELECT feature_name, total_runs, keep_runs, watch_runs, remove_runs,
                   consecutive_remove_count, consecutive_keep_count, evidence_score
            FROM feature_context_summary
            WHERE feature_source = 'experimental'
            """
        ).fetchall()
        for r in rows:
            longitudinal_map[r["feature_name"]] = dict(r)
    finally:
        conn_ev.close()

    decisions: list[dict[str, Any]] = []
    keep_cnt = 0
    watch_cnt = 0
    remove_cnt = 0

    updated_features: list[DiscoveredFeatureSpec] = []

    for spec in features:
        long_stats = longitudinal_map.get(spec.feature_name)
        new_status, rationale = evaluate_discovery_governance_decision(spec, long_stats, pol)

        old_status = spec.lifecycle_status
        spec.lifecycle_status = new_status
        spec.metadata["governance_rationale"] = rationale
        spec.metadata["governed_at"] = _utc_now_iso()
        spec.updated_at = _utc_now_iso()

        if new_status == DiscoveryLifecycleStatus.KEEP:
            keep_cnt += 1
        elif new_status == DiscoveryLifecycleStatus.WATCH:
            watch_cnt += 1
        elif new_status == DiscoveryLifecycleStatus.REMOVE:
            remove_cnt += 1

        decisions.append({
            "feature_id": spec.feature_id,
            "feature_name": spec.feature_name,
            "strategy": spec.generator_strategy.value if isinstance(spec.generator_strategy, GeneratorStrategy) else str(spec.generator_strategy),
            "previous_status": old_status.value if isinstance(old_status, DiscoveryLifecycleStatus) else str(old_status),
            "new_status": new_status.value,
            "evidence_score": spec.evidence_score,
            "delta_auc": spec.metadata.get("delta_auc", 0.0),
            "ks_statistic": spec.ks_statistic,
            "rationale": rationale,
        })
        updated_features.append(spec)

    # Persist updated status back to analysis.db
    persist_discovered_features(data_dir, updated_features)

    return {
        "pipeline_id": pipeline_id,
        "campaign_id": campaign_id,
        "context_key": context_key,
        "total_reviewed": len(features),
        "keep_count": keep_cnt,
        "watch_count": watch_cnt,
        "remove_count": remove_cnt,
        "decisions": decisions,
    }
