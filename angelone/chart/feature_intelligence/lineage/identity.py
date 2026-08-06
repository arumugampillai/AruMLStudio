"""Deterministic LINEAGE_* edge identity (Sprint 7)."""

from __future__ import annotations

import hashlib
import re

LINEAGE_UUID_PATTERN = re.compile(r"^LINEAGE_[0-9A-F]{32}$")
RELATIONSHIP_ID_PATTERN = re.compile(r"^REL_[A-Z][A-Z0-9_]*$")

PR_PATTERN = re.compile(r"^PR_[A-Z][A-Z0-9_]*$")
OP_PATTERN = re.compile(r"^OP_[A-Z][A-Z0-9_]*$")
TR_PATTERN = re.compile(r"^TR_[0-9A-F]{32}$")
FEAT_PATTERN = re.compile(r"^FEAT_[0-9A-F]{32}$")

FORBIDDEN_NODE_PREFIXES = ("ONT_", "COMP_", "DOM_", "REL_", "SIG_", "MATH_", "HOR_", "OUT_", "FREQ_", "STAB_")

OBJECT_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "PRIMITIVE": PR_PATTERN,
    "OPERATOR": OP_PATTERN,
    "TRANSFORMATION": TR_PATTERN,
    "FEATURE": FEAT_PATTERN,
}


def normalize_feature_object_id(object_id: str) -> str:
    """Normalize FEAT_* ids by stripping hyphens (freeze §3)."""
    if object_id.startswith("FEAT_"):
        return "FEAT_" + object_id[5:].replace("-", "").upper()
    return object_id


def normalize_object_id(object_id: str) -> str:
    if object_id.startswith("FEAT_"):
        return normalize_feature_object_id(object_id)
    if object_id.startswith("TR_"):
        return "TR_" + object_id[3:].replace("-", "").upper()
    return object_id


def infer_object_type(object_id: str) -> str | None:
    oid = normalize_object_id(object_id)
    if PR_PATTERN.match(oid):
        return "PRIMITIVE"
    if OP_PATTERN.match(oid):
        return "OPERATOR"
    if TR_PATTERN.match(oid):
        return "TRANSFORMATION"
    if FEAT_PATTERN.match(oid):
        return "FEATURE"
    return None


def is_valid_lineage_node(object_id: str) -> bool:
    """True iff id matches an allowed FIC node pattern (PR/OP/TR/FEAT)."""
    return infer_object_type(object_id) is not None


def derive_lineage_uuid(
    parent_object: str, child_object: str, relationship_id: str
) -> str:
    """
    lineage_uuid = LINEAGE_ + SHA256(f"{parent}|{child}|{relationship_id}")[:32].upper()

    Input material is the three fields only — pipe separator, no extra whitespace.
    """
    parent = normalize_object_id(parent_object)
    child = normalize_object_id(child_object)
    material = f"{parent}|{child}|{relationship_id}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:32].upper()
    return f"LINEAGE_{digest}"
