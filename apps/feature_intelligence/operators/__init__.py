"""Operator Registry / Catalog (Sprint 3)."""

from __future__ import annotations

from feature_intelligence.operators.catalog import (
    EXPECTED_OPERATOR_CATALOG_HASH,
    OPERATOR_CATALOG_VERSION,
    OPERATOR_PACK_VERSION,
    SEED_OPERATORS,
    compute_operator_catalog_hash,
    write_catalog_artifacts,
)
from feature_intelligence.operators.operator_models import OperatorRecord
from feature_intelligence.operators.operator_service import (
    OperatorNotFoundError,
    OperatorRegistryService,
)
from feature_intelligence.operators.operator_validation import validate_operators

__all__ = [
    "EXPECTED_OPERATOR_CATALOG_HASH",
    "OPERATOR_CATALOG_VERSION",
    "OPERATOR_PACK_VERSION",
    "OperatorNotFoundError",
    "OperatorRecord",
    "OperatorRegistryService",
    "SEED_OPERATORS",
    "compute_operator_catalog_hash",
    "validate_operators",
    "write_catalog_artifacts",
]
