"""Model Taxonomy Subsystem (Phase 4C.1).

Defines canonical task types, market regime metadata contracts, population tiers,
lifecycle statuses, context keys, and legacy backward-compatibility adapters.
"""

from __future__ import annotations

from .enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelLifecycleStatus,
    ModelPopulationTier,
    RegimeScope,
    TaskType,
)
from .specs import (
    ModelContextKey,
    ModelMetadata,
    RegimeSpec,
    TaskSpec,
)
from .adapter import (
    infer_task_type_from_target,
    resolve_model_metadata_or_legacy,
)
from .regime_registry_store import (
    compute_regime_definition_hash,
    get_regime_record,
    list_regimes,
    load_regime_registry,
    reactivate_regime,
    regime_registry_path,
    register_regime,
    retire_regime,
    save_regime_registry,
    update_regime_definition,
    validate_regime_id_format,
)
from .filtering import (
    filter_model_records,
    format_model_taxonomy_display,
    get_context_champions_map,
)

__all__ = [
    "BASELINE_REGIME_CATALOG",
    "DEFAULT_REGIME_ID",
    "DEFAULT_REGIME_NAME",
    "ModelContextKey",
    "ModelLifecycleStatus",
    "ModelMetadata",
    "ModelPopulationTier",
    "RegimeScope",
    "RegimeSpec",
    "TaskSpec",
    "TaskType",
    "infer_task_type_from_target",
    "resolve_model_metadata_or_legacy",
    "compute_regime_definition_hash",
    "get_regime_record",
    "list_regimes",
    "load_regime_registry",
    "reactivate_regime",
    "regime_registry_path",
    "register_regime",
    "retire_regime",
    "save_regime_registry",
    "update_regime_definition",
    "validate_regime_id_format",
    "filter_model_records",
    "format_model_taxonomy_display",
    "get_context_champions_map",
]
