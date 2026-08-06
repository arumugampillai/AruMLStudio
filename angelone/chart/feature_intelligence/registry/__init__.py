"""FIC registry: Primitive Catalog (Sprint 1) + Feature Registry (Sprint 2)."""

from __future__ import annotations

from feature_intelligence.registry.catalog import (
    CATALOG_VERSION,
    EXPECTED_SEED_CATALOG_HASH,
    SEED_PRIMITIVES,
    compute_seed_catalog_hash,
)
from feature_intelligence.registry.feature_ids import (
    generate_compilation_uuid,
    generate_feature_uuid,
    is_valid_feature_uuid,
    normalize_feature_uuid,
    derive_transformation_uuid,
    expression_hash,
)
from feature_intelligence.registry.feature_models import (
    FeatureRecord,
    SyncFailure,
    SyncReport,
    SyncSummary,
)
from feature_intelligence.registry.feature_registry_synchronizer import (
    synchronize_feature_registry,
)
from feature_intelligence.registry.feature_service import (
    FeatureNotFoundError,
    FeatureRegistryService,
)
from feature_intelligence.registry.models import PrimitiveRecord, ValidationReport
from feature_intelligence.registry.service import (
    PrimitiveCatalogService,
    PrimitiveNotFoundError,
    get_default_service,
)
from feature_intelligence.registry.traceability import (
    PRIMITIVE_ID_FIELD,
    is_valid_primitive_id,
)
from feature_intelligence.registry.validation import validate_primitives

__all__ = [
    "CATALOG_VERSION",
    "EXPECTED_SEED_CATALOG_HASH",
    "PRIMITIVE_ID_FIELD",
    "FeatureNotFoundError",
    "FeatureRecord",
    "FeatureRegistryService",
    "PrimitiveCatalogService",
    "PrimitiveNotFoundError",
    "PrimitiveRecord",
    "SEED_PRIMITIVES",
    "SyncFailure",
    "SyncReport",
    "SyncSummary",
    "ValidationReport",
    "compute_seed_catalog_hash",
    "derive_transformation_uuid",
    "expression_hash",
    "generate_compilation_uuid",
    "generate_feature_uuid",
    "get_default_service",
    "is_valid_feature_uuid",
    "is_valid_primitive_id",
    "normalize_feature_uuid",
    "synchronize_feature_registry",
    "validate_primitives",
]
