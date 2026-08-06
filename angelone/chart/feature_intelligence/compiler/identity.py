"""Transformation / compilation identity helpers (Sprint 5)."""

from __future__ import annotations

from feature_intelligence.registry.feature_ids import (
    COMPILATION_UUID_PATTERN,
    TRANSFORM_UUID_PATTERN,
    derive_transformation_uuid,
    expression_hash,
    generate_compilation_uuid,
    is_valid_compilation_uuid,
    is_valid_transformation_uuid,
    normalize_compilation_uuid,
    normalize_transformation_uuid,
)

__all__ = [
    "COMPILATION_UUID_PATTERN",
    "TRANSFORM_UUID_PATTERN",
    "derive_transformation_uuid",
    "expression_hash",
    "generate_compilation_uuid",
    "is_valid_compilation_uuid",
    "is_valid_transformation_uuid",
    "normalize_compilation_uuid",
    "normalize_transformation_uuid",
]
