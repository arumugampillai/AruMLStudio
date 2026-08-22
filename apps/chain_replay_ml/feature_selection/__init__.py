"""Phase 4B: Feature Selection & Attribution Integration Package."""

from .composite_pre import PreTrainingCompositeSelector
from .types import (
    AttributionStage,
    CanonicalFeatureAction,
    CompositeAttributionResult,
    CompositeProvenanceRecord,
    CompositeSelectionConfig,
    CompositeStrategy,
    DEFAULT_COMPOSITE_SELECTION_CONFIG,
    DiscoveryDiagnosticAction,
    FeatureAttributionRecord,
    ValidationDiagnosticAction,
    map_discovery_action_to_canonical,
    map_validation_action_to_canonical,
)

__all__ = [
    "AttributionStage",
    "CanonicalFeatureAction",
    "CompositeAttributionResult",
    "CompositeProvenanceRecord",
    "CompositeSelectionConfig",
    "CompositeStrategy",
    "DEFAULT_COMPOSITE_SELECTION_CONFIG",
    "DiscoveryDiagnosticAction",
    "FeatureAttributionRecord",
    "PreTrainingCompositeSelector",
    "ValidationDiagnosticAction",
    "map_discovery_action_to_canonical",
    "map_validation_action_to_canonical",
]
