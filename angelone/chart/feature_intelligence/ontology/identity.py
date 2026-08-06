"""Deterministic ONT_* ontology object identity (Sprint 6)."""

from __future__ import annotations

import hashlib
import re

ONT_UUID_PATTERN = re.compile(r"^ONT_[0-9A-F]{32}$")

OBJECT_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "PRIMITIVE": re.compile(r"^PR_[A-Z][A-Z0-9_]*$"),
    "OPERATOR": re.compile(r"^OP_[A-Z][A-Z0-9_]*$"),
    "TRANSFORMATION": re.compile(r"^TR_[0-9A-F]{32}$"),
    "FEATURE": re.compile(r"^FEAT_[0-9A-F]{32}$"),
}


def normalize_feature_object_id(object_id: str) -> str:
    """Normalize FEAT_* ids by stripping hyphens (freeze §2)."""
    if object_id.startswith("FEAT_"):
        return "FEAT_" + object_id[5:].replace("-", "").upper()
    return object_id


def derive_ontology_uuid(object_type: str, object_id: str) -> str:
    """
    ontology_uuid = ONT_ + SHA256(f"{object_type}:{object_id}")[:32].upper()

    Input material is object_type + object_id only — not classification fields.
    """
    oid = object_id
    if object_type == "FEATURE":
        oid = normalize_feature_object_id(object_id)
    material = f"{object_type}:{oid}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:32].upper()
    return f"ONT_{digest}"
