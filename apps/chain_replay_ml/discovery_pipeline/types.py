"""Data types, schemas, and identity primitives for the Autonomous Research Discovery Pipeline.

Defines:
1. GeneratorStrategy: Categorical strategies for synthesizing experimental features.
2. DiscoveryLifecycleStatus: Governance lifecycle states (CANDIDATE, KEEP, WATCH, REMOVE, PROMOTED).
3. DiscoveryPipelineBudget: Workstation-safe memory and computation bounds.
4. DiscoveredFeatureSpec: Full mathematical specification and telemetry record for a synthetic feature.
5. DiscoveryPipelineSnapshot: Cryptographic snapshot representation of an active feature set.
6. DiscoveryPipelineSpec: Campaign-scoped Discovery Pipeline header and configuration.
7. Utility hashing and ID formatting functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GeneratorStrategy(str, Enum):
    """Categorical strategy used to synthesize a discovery feature."""
    RATIO = "RATIO"
    INTERACTION = "INTERACTION"
    NONLINEAR = "NONLINEAR"
    SPREAD = "SPREAD"
    COMPOSITE = "COMPOSITE"


class DiscoveryLifecycleStatus(str, Enum):
    """Governance lifecycle status of an experimental discovery feature."""
    CANDIDATE = "candidate"
    KEEP = "KEEP"
    WATCH = "WATCH"
    REMOVE = "REMOVE"
    PROMOTED = "promoted"


@dataclass(frozen=True)
class DiscoveryPipelineBudget:
    """Resource budget and safety constraints for discovery feature generation (16 GB RAM safety)."""
    max_new_features_per_gen: int = 30           # Max synthetic features generated in a single step
    max_active_discovery_features: int = 200      # Max synthetic features actively kept in candidate sets
    max_total_candidate_features: int = 500       # Total features ceiling per candidate model
    max_total_pool_features: int = 1000           # Global maximum synthetic pool size per campaign
    max_generation_depth: int = 10                # Maximum descendant generations for discovery
    min_gain_threshold: float = 0.001             # Minimum gain to avoid automatic REMOVE verdict
    max_ks_drift_threshold: float = 0.45          # Maximum KS statistic before triggering REMOVE verdict
    max_nan_fraction: float = 0.01                # Maximum tolerated NaN/Inf percentage (1%)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DiscoveryPipelineBudget:
        if not isinstance(d, dict):
            return cls()
        valid_fields = {
            "max_new_features_per_gen",
            "max_active_discovery_features",
            "max_total_candidate_features",
            "max_total_pool_features",
            "max_generation_depth",
            "min_gain_threshold",
            "max_ks_drift_threshold",
            "max_nan_fraction",
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


def format_discovery_pipeline_id(campaign_id: str) -> str:
    """Format canonical Discovery Pipeline ID scoped to owning campaign."""
    clean_id = str(campaign_id or "").strip()
    if clean_id.startswith("DP_"):
        return clean_id
    if clean_id.startswith("CAMP_"):
        return f"DP_{clean_id}"
    return f"DP_CAMP_{clean_id}" if clean_id else f"DP_CAMP_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def format_discovered_feature_id(pipeline_id: str, strategy: str, seq: int) -> str:
    """Format unique deterministic feature ID."""
    clean_pipe = str(pipeline_id).replace("DP_", "")
    strat_code = str(strategy).lower()[:4]
    return f"DF_{clean_pipe}_{strat_code}_{seq:04d}"


def normalize_formula_expression(formula_expression: str) -> str:
    """Normalize formula string whitespace and casing for canonical hashing."""
    s = str(formula_expression or "").strip()
    # Normalize multiple whitespace characters into single space
    s = re.sub(r"\s+", " ", s)
    return s


def compute_formula_hash(formula_expression: str) -> str:
    """Compute 16-character MD5 hash of canonical formula string."""
    norm = normalize_formula_expression(formula_expression)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


def compute_discovery_snapshot_hash(
    pipeline_id: str,
    generation_number: int,
    feature_names: Sequence[str],
) -> str:
    """Compute unique cryptographic snapshot hash for active discovery feature set."""
    payload = {
        "pipeline_id": str(pipeline_id),
        "generation": int(generation_number),
        "features": sorted(list(set(feature_names))),
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"DP_SNAP_{digest}"


@dataclass
class DiscoveredFeatureSpec:
    """Specification and empirical telemetry for a synthetic discovery feature."""
    feature_id: str
    pipeline_id: str
    feature_name: str
    formula_expression: str
    formula_hash: str
    generator_strategy: GeneratorStrategy
    parent_features: list[str]
    generation_discovered: int
    lifecycle_status: DiscoveryLifecycleStatus = DiscoveryLifecycleStatus.CANDIDATE
    evidence_score: float = 0.0
    total_evaluations: int = 0
    holdout_rank: int | None = None
    relative_imp_drop: float | None = None
    drift_severity: int = 0
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "pipeline_id": self.pipeline_id,
            "feature_name": self.feature_name,
            "formula_expression": self.formula_expression,
            "formula_hash": self.formula_hash,
            "generator_strategy": self.generator_strategy.value if isinstance(self.generator_strategy, GeneratorStrategy) else str(self.generator_strategy),
            "parent_features": list(self.parent_features),
            "generation_discovered": self.generation_discovered,
            "lifecycle_status": self.lifecycle_status.value if isinstance(self.lifecycle_status, DiscoveryLifecycleStatus) else str(self.lifecycle_status),
            "evidence_score": float(self.evidence_score),
            "total_evaluations": int(self.total_evaluations),
            "holdout_rank": self.holdout_rank,
            "relative_imp_drop": self.relative_imp_drop,
            "drift_severity": int(self.drift_severity),
            "ks_statistic": float(self.ks_statistic),
            "ks_pvalue": float(self.ks_pvalue),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiscoveredFeatureSpec:
        strat_raw = d.get("generator_strategy", "RATIO")
        try:
            strat = GeneratorStrategy(strat_raw)
        except Exception:
            strat = GeneratorStrategy.RATIO

        status_raw = d.get("lifecycle_status", "candidate")
        try:
            status = DiscoveryLifecycleStatus(status_raw)
        except Exception:
            status = DiscoveryLifecycleStatus.CANDIDATE

        return cls(
            feature_id=str(d.get("feature_id", "")),
            pipeline_id=str(d.get("pipeline_id", "")),
            feature_name=str(d.get("feature_name", "")),
            formula_expression=str(d.get("formula_expression", "")),
            formula_hash=str(d.get("formula_hash") or compute_formula_hash(str(d.get("formula_expression", "")))),
            generator_strategy=strat,
            parent_features=list(d.get("parent_features") or []),
            generation_discovered=int(d.get("generation_discovered", 0)),
            lifecycle_status=status,
            evidence_score=float(d.get("evidence_score", 0.0)),
            total_evaluations=int(d.get("total_evaluations", 0)),
            holdout_rank=d.get("holdout_rank"),
            relative_imp_drop=d.get("relative_imp_drop"),
            drift_severity=int(d.get("drift_severity", 0)),
            ks_statistic=float(d.get("ks_statistic", 0.0)),
            ks_pvalue=float(d.get("ks_pvalue", 1.0)),
            metadata=dict(d.get("metadata") or {}),
            created_at=str(d.get("created_at") or _utc_now_iso()),
            updated_at=str(d.get("updated_at") or _utc_now_iso()),
        )


@dataclass
class DiscoveryPipelineSnapshot:
    """Reproducible snapshot of an active discovery pipeline state at a specific generation."""
    snapshot_hash: str
    pipeline_id: str
    generation_number: int
    active_feature_names: list[str]
    feature_count: int
    keep_count: int = 0
    watch_count: int = 0
    remove_count: int = 0
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_hash": self.snapshot_hash,
            "pipeline_id": self.pipeline_id,
            "generation_number": self.generation_number,
            "active_feature_names": list(self.active_feature_names),
            "feature_count": self.feature_count,
            "keep_count": self.keep_count,
            "watch_count": self.watch_count,
            "remove_count": self.remove_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiscoveryPipelineSnapshot:
        return cls(
            snapshot_hash=str(d.get("snapshot_hash", "")),
            pipeline_id=str(d.get("pipeline_id", "")),
            generation_number=int(d.get("generation_number", 0)),
            active_feature_names=list(d.get("active_feature_names") or []),
            feature_count=int(d.get("feature_count", len(d.get("active_feature_names") or []))),
            keep_count=int(d.get("keep_count", 0)),
            watch_count=int(d.get("watch_count", 0)),
            remove_count=int(d.get("remove_count", 0)),
            created_at=str(d.get("created_at") or _utc_now_iso()),
        )


@dataclass
class DiscoveryPipelineSpec:
    """Complete specification of a campaign-isolated Discovery Pipeline."""
    pipeline_id: str
    campaign_id: str
    context_key: str
    dataset_name: str
    dataset_snapshot_hash: str
    base_feature_count: int
    base_feature_names: list[str] = field(default_factory=list)
    base_pipeline_id: str = "PL_0001"
    base_pipeline_snapshot_hash: str = ""
    active_features_count: int = 0
    total_generated_count: int = 0
    parent_snapshot_hash: str = ""
    current_snapshot_hash: str = ""
    current_generation: int = 0
    status: str = "active"  # "active" | "completed" | "archived"
    budget: DiscoveryPipelineBudget = field(default_factory=DiscoveryPipelineBudget)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "campaign_id": self.campaign_id,
            "context_key": self.context_key,
            "dataset_name": self.dataset_name,
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "base_feature_count": self.base_feature_count,
            "base_feature_names": list(self.base_feature_names),
            "base_pipeline_id": self.base_pipeline_id,
            "base_pipeline_snapshot_hash": self.base_pipeline_snapshot_hash,
            "active_features_count": self.active_features_count,
            "total_generated_count": self.total_generated_count,
            "parent_snapshot_hash": self.parent_snapshot_hash,
            "current_snapshot_hash": self.current_snapshot_hash,
            "current_generation": self.current_generation,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiscoveryPipelineSpec:
        return cls(
            pipeline_id=str(d.get("pipeline_id", "")),
            campaign_id=str(d.get("campaign_id", "")),
            context_key=str(d.get("context_key", "")),
            dataset_name=str(d.get("dataset_name", "")),
            dataset_snapshot_hash=str(d.get("dataset_snapshot_hash", "")),
            base_feature_count=int(d.get("base_feature_count", 0)),
            base_feature_names=list(d.get("base_feature_names") or []),
            base_pipeline_id=str(d.get("base_pipeline_id", "PL_0001")),
            base_pipeline_snapshot_hash=str(d.get("base_pipeline_snapshot_hash", "")),
            active_features_count=int(d.get("active_features_count", 0)),
            total_generated_count=int(d.get("total_generated_count", 0)),
            parent_snapshot_hash=str(d.get("parent_snapshot_hash", "")),
            current_snapshot_hash=str(d.get("current_snapshot_hash", "")),
            current_generation=int(d.get("current_generation", 0)),
            status=str(d.get("status", "active")),
            budget=DiscoveryPipelineBudget.from_dict(d.get("budget")),
            created_at=str(d.get("created_at") or _utc_now_iso()),
            updated_at=str(d.get("updated_at") or _utc_now_iso()),
        )
