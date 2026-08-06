"""Deterministic FRR_* research identity (Sprint 8)."""

from __future__ import annotations

import hashlib
import re

RESEARCH_UUID_PATTERN = re.compile(r"^FRR_[0-9A-F]{32}$")
FEAT_PATTERN = re.compile(r"^FEAT_[0-9A-F]{32}$")
ONT_PATTERN = re.compile(r"^ONT_[0-9A-F]{32}$")
TR_PATTERN = re.compile(r"^TR_[0-9A-F]{32}$")


def normalize_feature_uuid(feature_uuid: str) -> str:
    """Normalize FEAT_* by stripping hyphens and uppercasing hex."""
    if feature_uuid.startswith("FEAT_"):
        return "FEAT_" + feature_uuid[5:].replace("-", "").upper()
    return feature_uuid


def derive_research_uuid(feature_uuid: str) -> str:
    """
    research_uuid = FRR_ + SHA256(feature_uuid UTF-8)[:32].upper()

    Input is the full FEAT_* string (after normalize). Never a random UUID.
    """
    feat = normalize_feature_uuid(feature_uuid)
    digest = hashlib.sha256(feat.encode("utf-8")).hexdigest()[:32].upper()
    return f"FRR_{digest}"


def is_feat_uuid(object_id: str) -> bool:
    return bool(FEAT_PATTERN.match(normalize_feature_uuid(object_id)))
