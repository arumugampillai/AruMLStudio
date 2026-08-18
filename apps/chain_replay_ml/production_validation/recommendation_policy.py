"""Configurable recommendation policy, thresholds, and scoring model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

POLICY_FILE_NAME = "recommendation_policy.json"


@dataclass
class ScoringPolicy:
    weight_keep: float = 25.0
    weight_remove: float = -35.0
    weight_watch: float = -10.0
    bonus_consecutive_keep: float = 15.0
    penalty_consecutive_remove: float = -25.0
    min_score: float = -100.0
    max_score: float = 100.0


@dataclass
class ExperimentalLifecyclePolicy:
    remove_block_consecutive_threshold: int = 2
    remove_block_total_threshold: int = 4
    promotion_candidate_consecutive_keep: int = 3
    promotion_candidate_min_score: float = 75.0
    min_unique_models: int = 2


@dataclass
class BasePipelinePolicy:
    negative_alert_score_threshold: float = -40.0
    strong_keep_min_score: float = 50.0
    min_validation_runs_for_ranking: int = 2


@dataclass
class FeatureRegistryPolicy:
    remove_audit_alert_threshold: int = 3
    min_unique_models: int = 2


@dataclass
class RecommendationPolicy:
    version: int = 1
    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    experimental_lifecycle: ExperimentalLifecyclePolicy = field(
        default_factory=ExperimentalLifecyclePolicy
    )
    base_pipeline: BasePipelinePolicy = field(default_factory=BasePipelinePolicy)
    feature_registry: FeatureRegistryPolicy = field(
        default_factory=FeatureRegistryPolicy
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RecommendationPolicy:
        if not data or not isinstance(data, dict):
            return cls()
        scoring_data = data.get("scoring") or {}
        exp_data = data.get("experimental_lifecycle") or {}
        base_data = data.get("base_pipeline") or {}
        reg_data = data.get("feature_registry") or {}

        return cls(
            version=int(data.get("version") or 1),
            scoring=ScoringPolicy(**{k: v for k, v in scoring_data.items() if k in ScoringPolicy.__dataclass_fields__}),
            experimental_lifecycle=ExperimentalLifecyclePolicy(
                **{k: v for k, v in exp_data.items() if k in ExperimentalLifecyclePolicy.__dataclass_fields__}
            ),
            base_pipeline=BasePipelinePolicy(
                **{k: v for k, v in base_data.items() if k in BasePipelinePolicy.__dataclass_fields__}
            ),
            feature_registry=FeatureRegistryPolicy(
                **{k: v for k, v in reg_data.items() if k in FeatureRegistryPolicy.__dataclass_fields__}
            ),
        )


def policy_file_path(data_dir: str) -> str:
    return os.path.join(data_dir, POLICY_FILE_NAME)


def load_recommendation_policy(data_dir: str) -> RecommendationPolicy:
    path = policy_file_path(data_dir)
    if not os.path.isfile(path):
        return RecommendationPolicy()
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return RecommendationPolicy.from_dict(doc)
    except Exception:
        return RecommendationPolicy()


def save_recommendation_policy(data_dir: str, policy: RecommendationPolicy) -> str:
    path = policy_file_path(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(policy.to_dict(), fh, indent=2)
    return path


def compute_evidence_score(
    *,
    keep_models: int,
    remove_models: int,
    watch_models: int,
    consecutive_keeps: int,
    consecutive_removes: int,
    policy: ScoringPolicy | None = None,
) -> float:
    """Compute evidence score bounded strictly within [min_score, max_score]."""
    p = policy or ScoringPolicy()
    raw = (
        p.weight_keep * float(keep_models)
        + p.weight_remove * float(remove_models)
        + p.weight_watch * float(watch_models)
        + p.bonus_consecutive_keep * float(consecutive_keeps)
        + p.penalty_consecutive_remove * float(consecutive_removes)
    )
    return round(max(p.min_score, min(p.max_score, raw)), 2)
