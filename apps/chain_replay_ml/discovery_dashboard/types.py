"""Data types and schemas for the Discovery Feature Dashboard & Pipeline Builder (Doc 18)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SelectedDiscoveryFeatureRef:
    """In-memory reference and rich provenance for a selected discovery feature."""
    feature_id: str                          # e.g. "DF_CAMP_..._RATIO_00003"
    pipeline_id: str                         # e.g. "DP_CAMP_..._20260822_002913"
    research_id: str                         # e.g. "RESEARCH_..._20260822_002913_a1b2"
    campaign_id: str                         # e.g. "CAMP_..._20260822_002913"
    formula_hash: str                        # 16-char MD5 hash
    formula_expression: str                  # Canonical AST formula
    generator_strategy: str                  # RATIO, INTERACTION, etc.
    parent_features: list[str] = field(default_factory=list) # Input base features
    generation_discovered: int = 1           # e.g. 2
    discovery_snapshot_hash: str = ""        # e.g. "DP_SNAP_f839ab10e927c34d"
    discovery_verdict: str = "KEEP"          # "KEEP" or "WATCH"
    marginal_delta_auc: float = 0.0          # e.g. +0.00182
    ks_statistic: float = 0.0                # e.g. 0.0842
    drift_severity: int = 0                  # 0 (<=0.20), 1 (0.20-0.35), 2 (>0.35)
    evidence_score: float = 0.0              # e.g. 58.4
    fold_consistency: float = 0.0            # e.g. 0.80
    governance_rationale: str = ""           # Algorithmic verdict justification
    context_key: str = ""                    # e.g. "NIFTY:6:standard:all"
    display_name: str = ""                   # Deterministic human-readable name derived from AST
    co_discovered_pipelines: list[str] = field(default_factory=list) # Co-discovering DP IDs
    selection_timestamp_iso: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossPipelineSelectionBasket:
    """Persistent in-session selection container across multiple Discovery Pipelines."""
    _items: dict[str, SelectedDiscoveryFeatureRef] = field(default_factory=dict) # Key: feature_id

    def add(self, item: SelectedDiscoveryFeatureRef) -> bool:
        """Add item to basket. REMOVE features are strictly rejected."""
        if str(item.discovery_verdict).upper() == "REMOVE":
            return False
        self._items[item.feature_id] = item
        return True

    def remove(self, feature_id: str) -> bool:
        if feature_id in self._items:
            del self._items[feature_id]
            return True
        return False

    def toggle(self, item: SelectedDiscoveryFeatureRef) -> bool:
        """Toggle inclusion. Returns True if now in basket, False if removed."""
        if item.feature_id in self._items:
            del self._items[item.feature_id]
            return False
        return self.add(item)

    def contains(self, feature_id: str) -> bool:
        return feature_id in self._items

    def clear(self) -> None:
        self._items.clear()

    def get_all(self) -> list[SelectedDiscoveryFeatureRef]:
        return list(self._items.values())

    def get_by_pipeline(self) -> dict[str, list[SelectedDiscoveryFeatureRef]]:
        grouped: dict[str, list[SelectedDiscoveryFeatureRef]] = {}
        for item in self._items.values():
            grouped.setdefault(item.pipeline_id, []).append(item)
        return grouped

    @property
    def total_count(self) -> int:
        return len(self._items)

    @property
    def pipeline_count(self) -> int:
        return len({item.pipeline_id for item in self._items.values()})

    @property
    def keep_count(self) -> int:
        return sum(1 for item in self._items.values() if item.discovery_verdict.upper() == "KEEP")

    @property
    def watch_count(self) -> int:
        return sum(1 for item in self._items.values() if item.discovery_verdict.upper() == "WATCH")


@dataclass
class PipelineCreationRequest:
    """Request payload for constructing a candidate discovery pipeline."""
    name: str                                # e.g. "Pipeline_002 — Discovery Synthesis V1"
    description: str                         # Human notes and context
    context_key: str                         # e.g. "NIFTY:6:standard:all"
    pipeline_id: str | None = None           # Optional specific ID (otherwise auto-allocated)


@dataclass
class PipelineCreationResult:
    """Result payload from pipeline construction."""
    success: bool
    pipeline_id: str
    pipeline_name: str
    base_feature_count: int
    discovered_feature_count: int
    total_feature_count: int
    pipeline_snapshot_id: str
    message: str
    co_discovery_count: int = 0
    errors: list[str] = field(default_factory=list)
