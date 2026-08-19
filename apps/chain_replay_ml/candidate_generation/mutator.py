"""Deterministic descendant candidate mutation engine (Phase 4F.2).

Generates reproducible fine-tuning descendant specifications along 5 supported mutation dimensions:
1. ALGORITHM_MUTATION: Explores alternative verified algorithm engines.
2. HYPERPARAMETER_MUTATION: Localized search in tree depth, learning rate, and regularization.
3. FEATURE_SUBSET_MUTATION: Incremental addition/removal of top Phase 4E affinity features & interaction pairs.
4. TARGET_HORIZON_MUTATION: Research outcome candidate formulations (2%, 3%, 4% hypotheses).
5. REGIME_SPECIALIZATION: Specialized regime routing within hierarchy.

Invariants:
- StrategyEvaluationPolicy is NEVER modified during candidate mutation.
- Context isolation: candidates remain strictly inside the target context key.
- 16 GB memory safety: respects max_features_per_candidate budget cap.
"""

from __future__ import annotations

from typing import Any, Sequence

from chain_replay_ml.training.trainers.base import (
    ALGORITHM_CATBOOST,
    ALGORITHM_EXTRA_TREES,
    ALGORITHM_LIGHTGBM,
    ALGORITHM_RANDOM_FOREST,
    ALGORITHM_XGBOOST,
    normalize_algorithm_id,
)
from .generator import create_candidate_spec
from .types import (
    CandidateGenerationBudget,
    CandidateSpec,
    MutationType,
)


def mutate_algorithm(
    parent: CandidateSpec,
    target_algorithm: str,
    *,
    campaign_id: str | None = None,
) -> CandidateSpec:
    """Mutate candidate algorithm engine while preserving context and feature set."""
    target_algo_norm = normalize_algorithm_id(target_algorithm)
    return create_candidate_spec(
        context_key=parent.context_key,
        algorithm=target_algo_norm,
        features=parent.features,
        regime_definition_hash=parent.regime_definition_hash,
        dataset_snapshot_hash=parent.dataset_snapshot_hash,
        random_seed=parent.random_seed,
        parent_spec=parent,
        mutation_type=MutationType.ALGORITHM_MUTATION,
        mutation_description=f"Algorithm mutation: {parent.algorithm} -> {target_algo_norm}",
        campaign_id=campaign_id or (parent.lineage.campaign_id if parent.lineage else None),
        candidate_id_suffix=f"_{target_algo_norm[:3].upper()}",
    )


def mutate_hyperparameters(
    parent: CandidateSpec,
    param_updates: dict[str, Any],
    *,
    description: str | None = None,
    campaign_id: str | None = None,
    suffix: str = "_HP",
) -> CandidateSpec:
    """Mutate specific hyperparameters of parent candidate."""
    new_params = dict(parent.hyperparameters)
    new_params.update(param_updates)
    desc = description or f"Hyperparameter mutation: {list(param_updates.keys())}"

    return create_candidate_spec(
        context_key=parent.context_key,
        algorithm=parent.algorithm,
        features=parent.features,
        hyperparameters=new_params,
        walk_forward_config=parent.walk_forward_config,
        regime_definition_hash=parent.regime_definition_hash,
        dataset_snapshot_hash=parent.dataset_snapshot_hash,
        random_seed=parent.random_seed,
        parent_spec=parent,
        mutation_type=MutationType.HYPERPARAMETER_MUTATION,
        mutation_description=desc,
        campaign_id=campaign_id or (parent.lineage.campaign_id if parent.lineage else None),
        candidate_id_suffix=suffix,
    )


def mutate_feature_subset(
    parent: CandidateSpec,
    *,
    features_to_add: Sequence[str] | None = None,
    features_to_remove: Sequence[str] | None = None,
    budget: CandidateGenerationBudget | None = None,
    campaign_id: str | None = None,
    suffix: str = "_FEAT",
    description: str | None = None,
) -> CandidateSpec:
    """Mutate feature composition by adding recommended affinity features or pruning weak features."""
    b = budget or CandidateGenerationBudget()
    curr_features = set(parent.features)

    if features_to_remove:
        curr_features.difference_update(set(features_to_remove))

    if features_to_add:
        curr_features.update(set(features_to_add))

    # Cap to memory safety ceiling
    new_feat_list = sorted(list(curr_features))[:b.max_features_per_candidate]
    added = [f for f in (features_to_add or []) if f in new_feat_list]
    removed = list(features_to_remove or [])
    desc = description or f"Feature mutation: +{len(added)} added, -{len(removed)} removed (total {len(new_feat_list)})"

    return create_candidate_spec(
        context_key=parent.context_key,
        algorithm=parent.algorithm,
        features=new_feat_list,
        hyperparameters=parent.hyperparameters,
        walk_forward_config=parent.walk_forward_config,
        regime_definition_hash=parent.regime_definition_hash,
        dataset_snapshot_hash=parent.dataset_snapshot_hash,
        random_seed=parent.random_seed,
        parent_spec=parent,
        mutation_type=MutationType.FEATURE_SUBSET_MUTATION,
        mutation_description=desc,
        campaign_id=campaign_id or (parent.lineage.campaign_id if parent.lineage else None),
        candidate_id_suffix=suffix,
    )


def generate_descendant_mutations(
    parent: CandidateSpec,
    *,
    top_affinity_features: Sequence[str] | None = None,
    interaction_pairs: Sequence[tuple[str, str]] | None = None,
    budget: CandidateGenerationBudget | None = None,
    campaign_id: str | None = None,
) -> list[CandidateSpec]:
    """Generate a diverse batch of descendant mutations up to max_descendants_per_parent budget."""
    b = budget or CandidateGenerationBudget()
    if parent.lineage and parent.lineage.generation_number >= b.max_generations:
        return []

    descendants: list[CandidateSpec] = []
    max_desc = b.max_descendants_per_parent

    # 1. Hyperparameter Depth / Learning Rate Variation
    curr_algo = parent.algorithm
    if curr_algo in (ALGORITHM_XGBOOST, ALGORITHM_LIGHTGBM):
        curr_depth = int(parent.hyperparameters.get("max_depth", 6))
        new_depth = 8 if curr_depth <= 6 else 4
        h1 = mutate_hyperparameters(
            parent,
            {"max_depth": new_depth, "learning_rate": 0.03},
            description=f"Refine depth to {new_depth} and lr to 0.03",
            campaign_id=campaign_id,
            suffix=f"_D{new_depth}",
        )
        descendants.append(h1)
    elif curr_algo == ALGORITHM_CATBOOST:
        curr_depth = int(parent.hyperparameters.get("depth", 6))
        new_depth = 7 if curr_depth <= 6 else 5
        h1 = mutate_hyperparameters(
            parent,
            {"depth": new_depth, "l2_leaf_reg": 5.0},
            description=f"Refine CatBoost depth to {new_depth} and l2_leaf_reg to 5.0",
            campaign_id=campaign_id,
            suffix=f"_D{new_depth}",
        )
        descendants.append(h1)

    if len(descendants) >= max_desc:
        return descendants

    # 2. Non-Linear Interaction Synergy Mutation (Phase 4E.3)
    if interaction_pairs:
        for f_a, f_b in interaction_pairs:
            if f_a not in parent.features or f_b not in parent.features:
                new_pair = [f for f in (f_a, f_b) if f not in parent.features]
                if new_pair:
                    syn_child = mutate_feature_subset(
                        parent,
                        features_to_add=new_pair,
                        budget=b,
                        campaign_id=campaign_id,
                        suffix="_SYN",
                        description=f"Interaction synergy mutation: +({f_a}, {f_b})",
                    )
                    descendants.append(syn_child)
                    break

    if len(descendants) >= max_desc:
        return descendants

    # 3. Univariate Feature Affinity Expansion (Phase 4E.3)
    if top_affinity_features:
        new_feats = [f for f in top_affinity_features if f not in parent.features][:5]
        if new_feats:
            f1 = mutate_feature_subset(
                parent,
                features_to_add=new_feats,
                budget=b,
                campaign_id=campaign_id,
                suffix="_AFF5",
            )
            descendants.append(f1)

    if len(descendants) >= max_desc:
        return descendants

    # 4. Algorithm Cross-Exploration
    alt_algo = ALGORITHM_LIGHTGBM if curr_algo == ALGORITHM_XGBOOST else (ALGORITHM_CATBOOST if curr_algo == ALGORITHM_LIGHTGBM else ALGORITHM_XGBOOST)
    a1 = mutate_algorithm(parent, alt_algo, campaign_id=campaign_id)
    descendants.append(a1)

    return descendants[:max_desc]
