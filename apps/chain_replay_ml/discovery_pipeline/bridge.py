"""Feature Studio Evidence DB Bridge for Autonomous Research Discovery Pipeline (Phase 5).

Ingests Phase 4 Discovery Feature Evaluation telemetry and records authoritative
longitudinal empirical evidence into `feature_recommendation_evidence.db`.

Flow:
Discovery Pipeline (DP_<campaign_id>)
      ↓
Phase 4 Walk-Forward Evaluation Results
      ↓
Phase 5 Evidence Bridge
      ↓
feature_recommendation_evidence.db:
      ├── recommendation_evidence (raw append-only runs)
      ├── feature_context_summary (longitudinal context aggregates)
      └── experimental_lineage_summary (campaign/snapshot lineage projection)

Invariants:
1. Zero Permanent Registry Mutation: NEVER modifies feature_registry_store.json or pipeline_registry_store.json.
2. Single Authoritative Evidence Store: Strictly writes to feature_recommendation_evidence.db without parallel DBs.
3. Feature Source Tagging: All discovery features strictly marked `feature_source='experimental'`.
4. Append-Only Accumulation: Subsequent evaluations increment run counters without overwriting past evidence.
5. Campaign & Snapshot Traceability: Every record carries pipeline_id (DP_...) and snapshot_id (DP_SNAP_...).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Sequence

from chain_replay_ml.production_validation.dataset_context import (
    DatasetContext,
    build_dataset_context,
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
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    GeneratorStrategy,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)


def resolve_discovery_dataset_context(
    context_key: str | None = None,
    market: str = "NIFTY",
    sampling_interval_sec: int = 6,
) -> DatasetContext:
    """Resolve or construct DatasetContext for Discovery Pipeline telemetry."""
    ck = str(context_key or "").strip()
    if ck:
        # Extract market and interval if formatted like NIFTY_6s_...
        parts = ck.split("_")
        m = parts[0] if parts else market
        iv = sampling_interval_sec
        for p in parts:
            if p.endswith("s") and p[:-1].isdigit():
                iv = int(p[:-1])
                break
        return build_dataset_context(
            market=m,
            sampling_interval_sec=iv,
            sliding_window="standard",
            feature_project_id="all",
        )
    return build_dataset_context(
        market=market,
        sampling_interval_sec=sampling_interval_sec,
        sliding_window="standard",
        feature_project_id="all",
    )


def bridge_discovery_evaluation_to_evidence_db(
    data_dir: str,
    *,
    pipeline_id: str,
    campaign_id: str,
    snapshot_hash: str,
    evaluated_features: Sequence[DiscoveredFeatureSpec],
    target_column: str = "label_up_5pct_5m",
    context_key: str = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
    generation_number: int = 1,
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Ingest Phase 4 evaluated discovery features into feature_recommendation_evidence.db."""
    if not evaluated_features:
        return {"inserted": 0, "contexts_updated": 0}

    pol = policy or load_recommendation_policy(data_dir)
    context = resolve_discovery_dataset_context(context_key)
    conn = get_connection(data_dir)
    now_iso = _utc_now_iso()
    val_run_id = f"eval_{pipeline_id}_g{generation_number}_{int(time.time())}"
    model_name = f"discovery_model_{pipeline_id}_g{generation_number}"

    evidence_rows: list[dict[str, Any]] = []

    for spec in evaluated_features:
        strat_str = spec.generator_strategy.value if isinstance(spec.generator_strategy, GeneratorStrategy) else str(spec.generator_strategy)
        status_str = spec.lifecycle_status.value if isinstance(spec.lifecycle_status, DiscoveryLifecycleStatus) else str(spec.lifecycle_status)

        # Preliminary recommendation tag based on evidence score and drift
        if spec.evidence_score >= 52.0 and spec.drift_severity == 0:
            rec = "KEEP"
        elif spec.evidence_score >= 48.0 or spec.drift_severity == 1:
            rec = "WATCH"
        else:
            rec = "REMOVE"

        identity_key = compute_feature_identity_key(
            "experimental",
            spec.feature_name,
            pipeline_id,
            snapshot_hash,
        )

        detail_payload = {
            "formula_hash": spec.formula_hash,
            "formula_expression": spec.formula_expression,
            "parent_features": spec.parent_features,
            "generator_strategy": strat_str,
            "generation_discovered": spec.generation_discovered,
            "evidence_score": spec.evidence_score,
            "ks_statistic": spec.ks_statistic,
            "ks_pvalue": spec.ks_pvalue,
            "drift_severity": spec.drift_severity,
            "metadata": spec.metadata,
        }

        row = {
            "evidence_id": f"ev_{uuid.uuid4().hex[:12]}",
            "context_id": context.context_id,
            "feature_name": spec.feature_name,
            "feature_source": "experimental",
            "feature_identity_key": identity_key,
            "pipeline_id": pipeline_id,
            "pipeline_snapshot_id": snapshot_hash,
            "recommendation": rec,
            "validation_run_id": val_run_id,
            "model_name": model_name,
            "target_column": target_column,
            "holdout_rank": spec.holdout_rank,
            "unseen_rank": None,
            "rank_change": None,
            "relative_imp_drop": spec.relative_imp_drop,
            "drift_severity": spec.drift_severity,
            "evidence_detail_json": detail_payload,
            "run_timestamp": now_iso,
        }
        evidence_rows.append(row)

    try:
        res = append_validation_evidence(
            conn,
            context=context,
            evidence_rows=evidence_rows,
            policy=pol,
        )
        return {
            "pipeline_id": pipeline_id,
            "campaign_id": campaign_id,
            "snapshot_hash": snapshot_hash,
            "validation_run_id": val_run_id,
            "context_id": context.context_id,
            "context_key": context.context_key,
            "features_bridged": len(evidence_rows),
            "inserted_evidence_rows": res.get("inserted", 0),
        }
    finally:
        conn.close()
