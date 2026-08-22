"""Public exports for the Discovery Feature Dashboard & Pipeline Builder (Doc 18)."""

from .types import (
    CrossPipelineSelectionBasket,
    PipelineCreationRequest,
    PipelineCreationResult,
    SelectedDiscoveryFeatureRef,
)
from .service import (
    create_candidate_discovery_pipeline,
    list_discovery_features,
    list_discovery_pipelines,
    validate_cross_pipeline_selection,
)

__all__ = [
    "CrossPipelineSelectionBasket",
    "PipelineCreationRequest",
    "PipelineCreationResult",
    "SelectedDiscoveryFeatureRef",
    "create_candidate_discovery_pipeline",
    "list_discovery_features",
    "list_discovery_pipelines",
    "validate_cross_pipeline_selection",
]
