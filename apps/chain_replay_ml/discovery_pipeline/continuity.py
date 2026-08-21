"""Next-Day Multi-Session Continuity & Warm-Start Engine for Discovery Pipeline (Phase 9).

Enables autonomous research to resume across trading days / multi-session campaigns:
Day 1 Research Session (DP_CAMP_DAY1)
      ↓
Generation Snapshots (DP_SNAP_<hash1>)
      ↓
Active Discovered Feature Pool & Provenance Saved to analysis.db
      ↓
==================== NEXT TRADING DAY ====================
      ↓
Day 2 Research Session (DP_CAMP_DAY2)
      ↓
Select Historical Snapshot (DP_SNAP_<hash1>)
      ↓
Warm-Start Pipeline Initialization:
      ├── 1. Load active discovered feature formulas & provenance
      ├── 2. Re-evaluate against Day 2 real market dataset (Walk-Forward CV)
      ├── 3. Longitudinal Evidence DB Ingestion (feature_recommendation_evidence.db)
      ├── 4. Apply Governance: KEEP persisting features, prune degraded features
      └── 5. Seed Parent Pool with Base Features + Surviving Day 1 Discoveries
      ↓
Continue Autonomous Evolutionary Discovery on Day 2 Dataset

Invariants:
1. Zero Permanent Registry Mutation: NEVER touches feature_registry_store.json or pipeline_registry_store.json.
2. Cross-Session Immutability: Day 1 campaign records remain 100% immutable; Day 2 runs in its own pipeline namespace.
3. Feature Source Tagging: All warm-started features remain strictly `feature_source='experimental'`.
4. Longitudinal Accumulation: Re-evaluation on Day 2 data increments longitudinal run counters in Evidence DB.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Sequence

import pandas as pd

from .bridge import bridge_discovery_evaluation_to_evidence_db
from .evaluator import DiscoveryFeatureEvaluator
from .governance import run_discovery_pipeline_governance
from .persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_snapshot,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
)
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    _utc_now_iso,
    compute_discovery_snapshot_hash,
    format_discovered_feature_id,
    format_discovery_pipeline_id,
)
from chain_replay_ml.research_memory.db import connect_analysis_db

logger = logging.getLogger(__name__)


def list_available_discovery_snapshots(
    data_dir: str,
    context_key: str | None = None,
) -> list[dict[str, Any]]:
    """Query analysis.db for all available historical discovery snapshots."""
    init_discovery_pipeline_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        if context_key:
            rows = conn.execute(
                """
                SELECT s.*, p.campaign_id, p.context_key, p.dataset_name, p.dataset_snapshot_hash
                FROM discovery_pipeline_snapshots s
                JOIN discovery_pipelines p ON s.pipeline_id = p.pipeline_id
                WHERE p.context_key = ?
                ORDER BY s.created_at DESC
                """,
                (context_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT s.*, p.campaign_id, p.context_key, p.dataset_name, p.dataset_snapshot_hash
                FROM discovery_pipeline_snapshots s
                JOIN discovery_pipelines p ON s.pipeline_id = p.pipeline_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_discovery_snapshot_bundle(
    data_dir: str,
    snapshot_hash: str,
) -> dict[str, Any] | None:
    """Load snapshot metadata, parent pipeline spec, and all active discovered features in the snapshot."""
    snap = load_discovery_snapshot(data_dir, snapshot_hash)
    if not snap:
        return None

    pipe = load_discovery_pipeline(data_dir, snap.pipeline_id)
    if not pipe:
        return None

    all_features = load_discovered_features(data_dir, snap.pipeline_id)
    active_set = set(snap.active_feature_names)
    active_features = [f for f in all_features if f.feature_name in active_set]

    return {
        "snapshot": snap,
        "pipeline": pipe,
        "active_features": active_features,
        "all_features_count": len(all_features),
    }


def warm_start_discovery_pipeline(
    df: pd.DataFrame,
    *,
    data_dir: str,
    source_snapshot_hash: str,
    new_campaign_id: str,
    base_features: Sequence[str],
    context_key: str = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
    dataset_name: str = "next_day_dataset",
    dataset_snapshot_hash: str = "next_day_snap",
    target_column: str = "label_up_5pct_5m",
    revalidate_features: bool = True,
    budget: DiscoveryPipelineBudget | None = None,
) -> dict[str, Any]:
    """Initialize a new Discovery Pipeline warm-started from a previous day's snapshot."""
    b = budget or DiscoveryPipelineBudget()
    init_discovery_pipeline_tables(data_dir)

    # 1. Load source snapshot bundle
    bundle = load_discovery_snapshot_bundle(data_dir, source_snapshot_hash)
    if not bundle:
        raise ValueError(f"Discovery snapshot '{source_snapshot_hash}' not found in {data_dir}")

    source_snap: DiscoveryPipelineSnapshot = bundle["snapshot"]
    source_pipe: DiscoveryPipelineSpec = bundle["pipeline"]
    imported_features: list[DiscoveredFeatureSpec] = bundle["active_features"]

    # 2. Create new Day 2 pipeline spec
    new_pipe_id = format_discovery_pipeline_id(new_campaign_id)
    new_pipe = DiscoveryPipelineSpec(
        pipeline_id=new_pipe_id,
        campaign_id=new_campaign_id,
        context_key=context_key,
        dataset_name=dataset_name,
        dataset_snapshot_hash=dataset_snapshot_hash,
        base_feature_count=len(base_features),
        base_feature_names=list(base_features),
        parent_snapshot_hash=source_snapshot_hash,
        current_generation=1,
        budget=b,
    )
    persist_discovery_pipeline(data_dir, new_pipe)

    # 3. Clone imported features into new pipeline namespace
    cloned_features: list[DiscoveredFeatureSpec] = []
    for idx, f in enumerate(imported_features, start=1):
        strat_code = f.generator_strategy.value if hasattr(f.generator_strategy, "value") else str(f.generator_strategy)
        new_fid = format_discovered_feature_id(new_pipe_id, strat_code, idx)

        new_f = DiscoveredFeatureSpec(
            feature_id=new_fid,
            pipeline_id=new_pipe_id,
            feature_name=f.feature_name,
            formula_expression=f.formula_expression,
            formula_hash=f.formula_hash,
            generator_strategy=f.generator_strategy,
            parent_features=list(f.parent_features),
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
            evidence_score=f.evidence_score,
            ks_statistic=f.ks_statistic,
            ks_pvalue=f.ks_pvalue,
            drift_severity=f.drift_severity,
            total_evaluations=f.total_evaluations,
            metadata=copy.deepcopy(f.metadata),
        )
        new_f.metadata["warm_started_from_snapshot"] = source_snapshot_hash
        new_f.metadata["warm_started_from_pipeline"] = source_pipe.pipeline_id
        cloned_features.append(new_f)

    persist_discovered_features(data_dir, cloned_features)

    # 4. Revalidate imported discoveries on Day 2 dataset if requested
    if revalidate_features and cloned_features:
        logger.info("Revalidating %d warm-started features on new dataset...", len(cloned_features))
        DiscoveryFeatureEvaluator.evaluate_features_on_dataset(
            df,
            data_dir=data_dir,
            pipeline_id=new_pipe_id,
            campaign_id=new_campaign_id,
            base_feature_names=base_features,
            discovery_features=cloned_features,
            target_column=target_column,
            generation_number=1,
            dataset_name=dataset_name,
            dataset_snapshot_hash=dataset_snapshot_hash,
            n_splits=5,
            budget=b,
        )

        # Bridge Day 2 evaluation into Evidence DB
        bridge_discovery_evaluation_to_evidence_db(
            data_dir,
            pipeline_id=new_pipe_id,
            campaign_id=new_campaign_id,
            snapshot_hash=source_snapshot_hash,
            evaluated_features=cloned_features,
            target_column=target_column,
            context_key=context_key,
            generation_number=1,
        )

        # Apply Governance
        run_discovery_pipeline_governance(
            data_dir,
            pipeline_id=new_pipe_id,
            campaign_id=new_campaign_id,
            context_key=context_key,
        )

    # 5. Compute surviving pool and initial snapshot for new pipeline
    governed_features = load_discovered_features(data_dir, new_pipe_id)
    surviving_keeps = [f for f in governed_features if f.lifecycle_status == DiscoveryLifecycleStatus.KEEP]
    surviving_watches = [f for f in governed_features if f.lifecycle_status == DiscoveryLifecycleStatus.WATCH]
    removed_features = [f for f in governed_features if f.lifecycle_status == DiscoveryLifecycleStatus.REMOVE]

    active_pool_names = [f.feature_name for f in surviving_keeps] + [f.feature_name for f in surviving_watches]

    snap_hash = compute_discovery_snapshot_hash(new_pipe_id, 1, active_pool_names)
    initial_snap = DiscoveryPipelineSnapshot(
        snapshot_hash=snap_hash,
        pipeline_id=new_pipe_id,
        generation_number=1,
        active_feature_names=active_pool_names,
        feature_count=len(active_pool_names),
        keep_count=len(surviving_keeps),
        watch_count=len(surviving_watches),
        remove_count=len(removed_features),
    )
    persist_discovery_snapshot(data_dir, initial_snap)

    # Update pipeline header
    new_pipe.current_snapshot_hash = snap_hash
    new_pipe.active_features_count = len(active_pool_names)
    new_pipe.total_generated_count = len(governed_features)
    persist_discovery_pipeline(data_dir, new_pipe)

    return {
        "pipeline_id": new_pipe_id,
        "campaign_id": new_campaign_id,
        "source_snapshot_hash": source_snapshot_hash,
        "source_pipeline_id": source_pipe.pipeline_id,
        "imported_features_count": len(cloned_features),
        "revalidated": revalidate_features,
        "surviving_keeps": len(surviving_keeps),
        "surviving_watches": len(surviving_watches),
        "removed_features": len(removed_features),
        "initial_snapshot_hash": snap_hash,
        "active_pool_count": len(active_pool_names),
    }
