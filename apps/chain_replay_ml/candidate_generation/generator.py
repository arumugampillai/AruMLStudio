"""Pure candidate specification generator (Phase 4F.2).

Generates valid candidate specifications strictly across the 5 verified algorithms
and verified hyperparameter schemas in AruMLStudio.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Sequence

from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from chain_replay_ml.research_memory.signature import (
    build_canonical_experiment_payload,
    compute_experiment_signature,
)
from chain_replay_ml.training.trainers.base import (
    ALGORITHM_CATBOOST,
    ALGORITHM_EXTRA_TREES,
    ALGORITHM_LIGHTGBM,
    ALGORITHM_RANDOM_FOREST,
    ALGORITHM_XGBOOST,
    normalize_algorithm_id,
)
from .types import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateLineageRecord,
    CandidateSpec,
    MutationType,
)

# Canonical baseline hyperparameter defaults per verified algorithm
VERIFIED_ALGORITHM_DEFAULTS: dict[str, dict[str, Any]] = {
    ALGORITHM_XGBOOST: {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    ALGORITHM_LIGHTGBM: {
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    ALGORITHM_CATBOOST: {
        "learning_rate": 0.05,
        "depth": 6,
        "iterations": 500,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    ALGORITHM_RANDOM_FOREST: {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_seed": 42,
    },
    ALGORITHM_EXTRA_TREES: {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_seed": 42,
    },
}

DEFAULT_WALK_FORWARD_CONFIG: dict[str, Any] = {
    "n_folds": 5,
    "window_mode": "expanding",
    "fold_placement": "anchored",
    "train_pct": 70,
    "val_pct": 15,
    "test_pct": 15,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_hyperparameters_for_algorithm(
    algorithm: str,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Validate and filter hyperparameters strictly to supported keys for the given algorithm."""
    algo = normalize_algorithm_id(algorithm)
    defaults = VERIFIED_ALGORITHM_DEFAULTS.get(algo, VERIFIED_ALGORITHM_DEFAULTS[ALGORITHM_XGBOOST])
    valid_keys = set(defaults.keys())

    cleaned: dict[str, Any] = {}
    for k, v in hyperparameters.items():
        if k in valid_keys:
            cleaned[k] = v

    # Fill in missing defaults
    for k, v in defaults.items():
        if k not in cleaned:
            cleaned[k] = v

    return cleaned


def create_candidate_spec(
    *,
    context_key: str,
    algorithm: str,
    features: Sequence[str],
    hyperparameters: dict[str, Any] | None = None,
    walk_forward_config: dict[str, Any] | None = None,
    regime_definition_hash: str = "regime_hash_universal",
    dataset_snapshot_hash: str = "dataset_snapshot_v1",
    random_seed: int = 42,
    parent_spec: CandidateSpec | None = None,
    mutation_type: MutationType = MutationType.COLD_START,
    mutation_description: str = "Initial baseline candidate",
    campaign_id: str | None = None,
    candidate_id_suffix: str = "",
    opportunity_id: str | None = None,
    opportunity_type: str | None = None,
    priority_score: float | None = None,
    feature_elimination_strategy: str | None = None,
) -> CandidateSpec:
    """Construct a complete, deterministically signed CandidateSpec."""
    ctx_obj = ModelContextKey.from_key_str(context_key)
    algo_norm = normalize_algorithm_id(algorithm)
    cleaned_params = validate_hyperparameters_for_algorithm(algo_norm, hyperparameters or {})
    wf_config = dict(walk_forward_config or DEFAULT_WALK_FORWARD_CONFIG)
    deduped_features = sorted(list(set(str(f).strip() for f in features if f and str(f).strip())))

    # Compute Canonical Experiment Signature (Phase 4D.2)
    payload = build_canonical_experiment_payload(
        market=ctx_obj.market,
        sampling_interval_sec=ctx_obj.sampling_interval_sec,
        task_type=ctx_obj.task_type.value,
        prediction_horizon=ctx_obj.prediction_horizon,
        regime_id=ctx_obj.regime_id,
        regime_definition_hash=regime_definition_hash,
        dataset_snapshot_hash=dataset_snapshot_hash,
        features=deduped_features,
        algorithm=algo_norm,
        hyperparameters=cleaned_params,
        walk_forward_config=wf_config,
        random_seed=random_seed,
    )
    sig_hash, _, _ = compute_experiment_signature(payload)

    # Determine generation and candidate ID
    gen_num = (parent_spec.lineage.generation_number + 1) if (parent_spec and parent_spec.lineage) else 0
    p_id = parent_spec.candidate_id if parent_spec else None
    p_sig = parent_spec.signature_hash if parent_spec else None

    if p_id:
        c_id = f"{p_id}_G{gen_num}_{algo_norm[:3].upper()}{candidate_id_suffix}"
    else:
        c_id = f"CAND_{ctx_obj.market}_{algo_norm[:3].upper()}_{sig_hash[:8]}"

    lineage = CandidateLineageRecord(
        candidate_id=c_id,
        signature_hash=sig_hash,
        parent_candidate_id=p_id,
        parent_signature_hash=p_sig,
        generation_number=gen_num,
        campaign_id=campaign_id,
        context_key=ctx_obj.canonical_key_str(),
        mutation_type=mutation_type,
        mutation_description=mutation_description,
        opportunity_id=opportunity_id,
        opportunity_type=opportunity_type,
        priority_score=priority_score,
        feature_elimination_strategy=feature_elimination_strategy,
        created_at=_utc_now_iso(),
    )

    return CandidateSpec(
        candidate_id=c_id,
        context_key=ctx_obj.canonical_key_str(),
        market=ctx_obj.market,
        sampling_interval_sec=ctx_obj.sampling_interval_sec,
        task_type=ctx_obj.task_type.value,
        prediction_horizon=ctx_obj.prediction_horizon,
        regime_id=ctx_obj.regime_id,
        regime_definition_hash=regime_definition_hash,
        dataset_snapshot_hash=dataset_snapshot_hash,
        features=deduped_features,
        algorithm=algo_norm,
        hyperparameters=cleaned_params,
        walk_forward_config=wf_config,
        random_seed=random_seed,
        signature_hash=sig_hash,
        lineage=lineage,
        eligibility=CandidateEligibility.ELIGIBLE,
        feature_elimination_strategy=feature_elimination_strategy,
    )


def generate_cold_start_candidates(
    context_key: str,
    base_features: Sequence[str],
    *,
    algorithms: Sequence[str] | None = None,
    regime_definition_hash: str = "regime_hash_universal",
    dataset_snapshot_hash: str = "dataset_snapshot_v1",
    campaign_id: str | None = None,
    budget: CandidateGenerationBudget | None = None,
    mutation_type: MutationType = MutationType.FULL_FEATURE_BASELINE,
    feature_elimination_strategy: str | None = None,
) -> list[CandidateSpec]:
    """Generate initial full-universe baseline candidate specifications across verified algorithms."""
    b = budget or CandidateGenerationBudget()
    algos = algorithms or [
        ALGORITHM_XGBOOST,
        ALGORITHM_LIGHTGBM,
        ALGORITHM_CATBOOST,
        ALGORITHM_RANDOM_FOREST,
        ALGORITHM_EXTRA_TREES,
    ]
    features_list = list(dict.fromkeys(str(f).strip() for f in base_features if str(f).strip()))
    if b.max_features_per_candidate and len(features_list) > b.max_features_per_candidate:
        features_list = features_list[:b.max_features_per_candidate]

    candidates: list[CandidateSpec] = []
    for algo in algos:
        if len(candidates) >= b.max_candidates_per_campaign:
            break
        cand = create_candidate_spec(
            context_key=context_key,
            algorithm=algo,
            features=features_list,
            regime_definition_hash=regime_definition_hash,
            dataset_snapshot_hash=dataset_snapshot_hash,
            campaign_id=campaign_id,
            mutation_type=mutation_type,
            mutation_description=f"Full feature baseline ({len(features_list)} features) — {algo.upper()}",
            feature_elimination_strategy=feature_elimination_strategy,
        )
        candidates.append(cand)

    return candidates
