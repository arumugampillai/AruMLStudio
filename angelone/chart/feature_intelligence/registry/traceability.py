"""Primitive-to-feature traceability contract helpers (Sprint 1 stub)."""

from __future__ import annotations

import re

from feature_intelligence.registry.catalog import PRIMITIVE_ID_PATTERN

# Downstream AST / lineage JSON must use this field name.
PRIMITIVE_ID_FIELD = "primitive_id"

_ID_RE = re.compile(PRIMITIVE_ID_PATTERN)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def is_valid_primitive_id(primitive_id: str) -> bool:
    """Return True if ``primitive_id`` matches the frozen ``PR_*`` pattern."""
    return bool(_ID_RE.fullmatch(primitive_id))


def looks_like_uuid(value: str) -> bool:
    """Return True if ``value`` looks like an RFC-4122 UUID (forbidden for primitives)."""
    return bool(_UUID_RE.fullmatch(value))
