"""Autonomous Research Discovery Pipeline Evolutionary Loop Engine (Phase 7).

Orchestrates multi-generational, closed-loop feature discovery:
Generation N (Surviving Active Pool)
      ↓
Feature Synthesis on Real Data (Ratios, Interactions, Non-Linear, Spreads, Composites)
      ↓
Deduplication against Canonical Formula Hashes
      ↓
Phase 4 Chronological 5-Fold Walk-Forward Evaluation (Marginal ΔAUC, D_KS)
      ↓
Phase 5 Feature Studio Evidence DB Ingestion (feature_recommendation_evidence.db)
      ↓
Phase 6 Empirical & Longitudinal Governance (KEEP / WATCH / REMOVE)
      ↓
Active Discovery Pool Update & Pruning (REMOVE excludes, KEEP/WATCH survive)
      ↓
Immutable Cryptographic Generation Snapshot (DP_SNAP_<hash>)
      ↓
Generation N+1 (Parent Universe = Base Features + Discovered KEEPs)

Invariants:
1. Zero Permanent Registry Mutation: NEVER touches feature_registry_store.json or pipeline_registry_store.json.
2. Campaign Isolation: All mutations, features, and snapshots strictly scoped to DP_<campaign_id>.
3. Append-Only Longitudinal Evidence: Each generation appends evidence and increments run counters.
4. Non-Destructive REMOVE: Pruning excludes features from future candidate pools while preserving historical evidence.
5. Workstation Safety: Strictly bounded by DiscoveryPipelineBudget (max_new_features_per_gen, max_total_candidate_features).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .bridge import bridge_discovery_evaluation_to_evidence_db
from .evaluator import DiscoveryFeatureEvaluator
from .governance import run_discovery_pipeline_governance
from .persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
)
from .synthesizer import (
    generate_discovery_features_from_dataset,
    is_eligible_base_feature,
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
    format_discovery_pipeline_id,
)

logger = logging.getLogger(__name__)


def run_discovery_generation(
    df: pd.DataFrame,
    *,
    data_dir: str,
    pipeline_id: str,
    campaign_id: str,
    generation_number: int,
    base_features: Sequence[str],
    target_column: str = "label_up_5pct_5m",
    context_key: str = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
    dataset_name: str = "real_dataset",
    dataset_snapshot_hash: str = "snap_hash",
    budget: DiscoveryPipelineBudget | None = None,
    strategies: Sequence[GeneratorStrategy] | None = None,
) -> dict[str, Any]:
    """Execute a single complete discovery generation cycle (Synthesis → Evaluation → Ingest → Governance → Snapshot)."""
    t0 = time.perf_counter()
    b = budget or DiscoveryPipelineBudget()
    init_discovery_pipeline_tables(data_dir)

    # 1. Load active pipeline header
    from chain_replay_ml.dataset_builder.pipeline_registry_store import get_base_pipeline_for_context

    base_pipe = get_base_pipeline_for_context(data_dir, context_key)
    base_pipe_id = base_pipe.get("pipeline_id", "PL_0001") if base_pipe else "PL_0001"
    base_pipe_snap = base_pipe.get("pipeline_snapshot_id", "1714b8dddb455a95") if base_pipe else ""

    pipe = load_discovery_pipeline(data_dir, pipeline_id)
    if not pipe:
        pipe = DiscoveryPipelineSpec(
            pipeline_id=pipeline_id,
            campaign_id=campaign_id,
            context_key=context_key,
            dataset_name=dataset_name,
            dataset_snapshot_hash=dataset_snapshot_hash,
            base_feature_count=len(base_features),
            base_feature_names=list(base_features),
            base_pipeline_id=base_pipe_id,
            base_pipeline_snapshot_hash=base_pipe_snap,
            budget=b,
        )
        persist_discovery_pipeline(data_dir, pipe)
    else:
        if not pipe.base_pipeline_id:
            pipe.base_pipeline_id = base_pipe_id
        if not pipe.base_pipeline_snapshot_hash and base_pipe_snap:
            pipe.base_pipeline_snapshot_hash = base_pipe_snap

    # 2. Identify active discovery pool from prior generations
    all_existing_features = load_discovered_features(data_dir, pipeline_id)
    existing_hashes = {f.formula_hash for f in all_existing_features}
    try:
        from chain_replay_ml.research_registry.memory import get_blacklisted_formula_hashes
        blacklisted = get_blacklisted_formula_hashes(data_dir, context_key)
        existing_hashes.update(blacklisted)
    except Exception:
        pass

    # Surviving features = KEEPs + WATCHes (REMOVEs are excluded from parent pool)
    active_discovered = [
        f for f in all_existing_features
        if f.lifecycle_status in (DiscoveryLifecycleStatus.KEEP, DiscoveryLifecycleStatus.WATCH, DiscoveryLifecycleStatus.CANDIDATE)
    ]
    active_discovered_names = [f.feature_name for f in active_discovered]

    # Ensure surviving features from prior generations are computed on df in-memory so synthesizer can use them
    for f in active_discovered:
        if f.feature_name not in df.columns and f.formula_expression:
            try:
                df[f.feature_name] = evaluate_discovery_formula(df, f.formula_expression)
            except Exception:
                pass

    # Parent candidate pool for generation N+1 combines base features + surviving discovered features
    parent_candidate_pool = list(base_features) + [n for n in active_discovered_names if n in df.columns]

    # 3. Synthesize novel experimental features
    start_seq = len(all_existing_features) + 1
    new_specs, generated_df = generate_discovery_features_from_dataset(
        df,
        pipeline_id=pipeline_id,
        generation_number=generation_number,
        base_feature_candidates=parent_candidate_pool,
        existing_formula_hashes=existing_hashes,
        start_sequence=start_seq,
        budget=b,
        strategies=strategies,
    )

    if new_specs:
        persist_discovered_features(data_dir, new_specs)

    # 4. Features to evaluate in this generation: newly synthesized + active un-evaluated/watch features
    to_evaluate = list(new_specs)
    # Also re-evaluate WATCH features to gather longitudinal evidence
    watch_features = [f for f in active_discovered if f.lifecycle_status == DiscoveryLifecycleStatus.WATCH]
    to_evaluate.extend(watch_features[:5])  # Sample up to 5 WATCH features for re-evaluation

    eval_result: dict[str, Any] = {}
    if to_evaluate:
        # Augment df in-memory with newly generated feature columns if needed
        eval_df = df.copy(deep=False)
        for col_name in generated_df.columns:
            eval_df[col_name] = generated_df[col_name]

        # 5. Run Phase 4 Walk-Forward Feature Evaluation
        eval_result = DiscoveryFeatureEvaluator.evaluate_features_on_dataset(
            eval_df,
            data_dir=data_dir,
            pipeline_id=pipeline_id,
            campaign_id=campaign_id,
            base_feature_names=base_features,
            discovery_features=to_evaluate,
            target_column=target_column,
            generation_number=generation_number,
            dataset_name=dataset_name,
            dataset_snapshot_hash=dataset_snapshot_hash,
            n_splits=5,
            budget=b,
        )

        # 6. Bridge telemetry into Feature Studio Evidence DB
        bridge_discovery_evaluation_to_evidence_db(
            data_dir,
            pipeline_id=pipeline_id,
            campaign_id=campaign_id,
            snapshot_hash=pipe.current_snapshot_hash or "DP_SNAP_INITIAL",
            evaluated_features=to_evaluate,
            target_column=target_column,
            context_key=context_key,
            generation_number=generation_number,
        )

    # 7. Apply Phase 6 Governance to update lifecycle states
    gov_result = run_discovery_pipeline_governance(
        data_dir,
        pipeline_id=pipeline_id,
        campaign_id=campaign_id,
        context_key=context_key,
    )

    # 8. Compute updated active discovery pool after governance
    all_governed = load_discovered_features(data_dir, pipeline_id)
    surviving_keeps = [f for f in all_governed if f.lifecycle_status == DiscoveryLifecycleStatus.KEEP]
    surviving_watches = [f for f in all_governed if f.lifecycle_status == DiscoveryLifecycleStatus.WATCH]
    removed_features = [f for f in all_governed if f.lifecycle_status == DiscoveryLifecycleStatus.REMOVE]

    active_pool_names = [f.feature_name for f in surviving_keeps] + [f.feature_name for f in surviving_watches]

    # 9. Create immutable cryptographic generation snapshot
    snapshot_hash = compute_discovery_snapshot_hash(pipeline_id, generation_number, active_pool_names)
    snapshot = DiscoveryPipelineSnapshot(
        snapshot_hash=snapshot_hash,
        pipeline_id=pipeline_id,
        generation_number=generation_number,
        active_feature_names=active_pool_names,
        feature_count=len(active_pool_names),
        keep_count=len(surviving_keeps),
        watch_count=len(surviving_watches),
        remove_count=len(removed_features),
    )
    persist_discovery_snapshot(data_dir, snapshot)

    # 10. Update Discovery Pipeline Header
    pipe.current_generation = generation_number
    pipe.current_snapshot_hash = snapshot_hash
    pipe.active_features_count = len(active_pool_names)
    pipe.total_generated_count = len(all_governed)
    persist_discovery_pipeline(data_dir, pipe)

    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "pipeline_id": pipeline_id,
        "campaign_id": campaign_id,
        "generation_number": generation_number,
        "snapshot_hash": snapshot_hash,
        "new_features_generated": len(new_specs),
        "total_evaluated": len(to_evaluate),
        "keep_count": len(surviving_keeps),
        "watch_count": len(surviving_watches),
        "remove_count": len(removed_features),
        "active_pool_count": len(active_pool_names),
        "duration_sec": elapsed,
        "governance_summary": gov_result,
    }


def run_autonomous_discovery_loop(
    df: pd.DataFrame,
    *,
    data_dir: str,
    campaign_id: str,
    total_generations: int = 3,
    base_features: Sequence[str],
    target_column: str = "label_up_5pct_5m",
    context_key: str = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
    dataset_name: str = "real_dataset",
    dataset_snapshot_hash: str = "snap_hash",
    budget: DiscoveryPipelineBudget | None = None,
    strategies: Sequence[GeneratorStrategy] | None = None,
    stop_requested_fn: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute complete multi-generation Autonomous Discovery Evolutionary Loop."""
    t_start = time.perf_counter()
    pipeline_id = format_discovery_pipeline_id(campaign_id)
    b = budget or DiscoveryPipelineBudget()

    generation_history: list[dict[str, Any]] = []

    for gen in range(1, total_generations + 1):
        if stop_requested_fn and stop_requested_fn():
            logger.info("Stop requested; safely halting discovery loop at generation %d", gen)
            break

        logger.info("Starting Discovery Generation %d/%d for %s", gen, total_generations, pipeline_id)
        try:
            gen_res = run_discovery_generation(
                df,
                data_dir=data_dir,
                pipeline_id=pipeline_id,
                campaign_id=campaign_id,
                generation_number=gen,
                base_features=base_features,
                target_column=target_column,
                context_key=context_key,
                dataset_name=dataset_name,
                dataset_snapshot_hash=dataset_snapshot_hash,
                budget=b,
                strategies=strategies,
            )
            generation_history.append(gen_res)
        except Exception as e:
            logger.error("Error during Discovery Generation %d: %s", gen, e, exc_info=True)
            generation_history.append({
                "generation_number": gen,
                "error": str(e),
                "status": "failed",
            })
            break

    total_time = round(time.perf_counter() - t_start, 3)

    # Load final pipeline state
    final_pipe = load_discovery_pipeline(data_dir, pipeline_id)
    final_snaps = load_discovery_snapshots_for_pipeline(data_dir, pipeline_id)

    return {
        "pipeline_id": pipeline_id,
        "campaign_id": campaign_id,
        "total_generations_completed": len(generation_history),
        "total_duration_sec": total_time,
        "current_generation": final_pipe.current_generation if final_pipe else 0,
        "current_snapshot_hash": final_pipe.current_snapshot_hash if final_pipe else "",
        "active_features_count": final_pipe.active_features_count if final_pipe else 0,
        "total_generated_count": final_pipe.total_generated_count if final_pipe else 0,
        "generation_history": generation_history,
        "snapshots_recorded": [s.to_dict() for s in final_snaps],
    }
