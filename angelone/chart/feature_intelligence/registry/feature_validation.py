"""Feature Registry validation (Sprint 2)."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.registry.feature_definition_hash import compute_definition_hash
from feature_intelligence.registry.feature_ids import is_valid_feature_uuid, normalize_transformation_uuid
from feature_intelligence.registry.feature_models import (
    CANONICAL_NAME_PATTERN,
    GAP_POLICIES,
    MEMORY_MODELS,
    RESEARCH_STATES,
)
from feature_intelligence.registry.feature_store import FeatureStore
from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.registry.store import PrimitiveStore
from feature_intelligence.registry.traceability import looks_like_uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _registry_content_hash(rows: list) -> str:
    lines = [
        f"{r.feature_uuid}|{r.canonical_name}|{r.definition_hash}"
        for r in sorted(rows, key=lambda x: x.feature_uuid)
    ]
    doc = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def validate_features(db_path: Path) -> ValidationReport:
    store = FeatureStore(db_path)
    prim_store = PrimitiveStore(db_path)
    failed: list[str] = []
    warnings: list[str] = []

    if not store.table_exists():
        return ValidationReport(
            passed=False,
            failed_rules=["TABLE_EXISTS: feature_registry missing"],
            warnings=warnings,
            validated_objects="0 features",
            timestamp=_utc_now(),
        )

    if not store.index_exists("idx_feature_uuid"):
        failed.append("INDEX_FEATURE_UUID: idx_feature_uuid missing")

    rows = store.list_all()
    names = [r.canonical_name for r in rows]
    uuids = [r.feature_uuid for r in rows]
    if len(names) != len(set(names)):
        failed.append("NAME_UNIQUE: duplicate canonical_name")
    if len(uuids) != len(set(uuids)):
        failed.append("UUID_UNIQUE: duplicate feature_uuid")

    for r in rows:
        if not is_valid_feature_uuid(r.feature_uuid):
            failed.append(f"UUID_FORMAT: {r.feature_uuid}")
        if looks_like_uuid(r.feature_uuid):
            failed.append(f"NO_PRIMITIVE_UUID: unexpected bare UUID {r.feature_uuid}")
        if not re.fullmatch(CANONICAL_NAME_PATTERN, r.canonical_name):
            failed.append(f"NAME_FORMAT: {r.canonical_name}")
        if not r.primitive_ids:
            failed.append(f"PRIMITIVES_MIN_ONE: {r.feature_uuid}")
        for pid in r.primitive_ids:
            if prim_store.get_by_id(pid) is None:
                failed.append(f"PRIMITIVE_FK: {r.feature_uuid}->{pid}")
        if r.research_state not in RESEARCH_STATES:
            failed.append(f"RESEARCH_STATE: {r.feature_uuid}={r.research_state}")
        if not r.definition_version:
            failed.append(f"DEFINITION_VERSION: empty on {r.feature_uuid}")
        if not r.implementation_version:
            failed.append(f"IMPLEMENTATION_VERSION: empty on {r.feature_uuid}")
        if r.gap_policy not in GAP_POLICIES:
            failed.append(f"GAP_POLICY: {r.feature_uuid}={r.gap_policy}")
        if r.memory_model not in MEMORY_MODELS:
            failed.append(f"MEMORY_MODEL: {r.feature_uuid}={r.memory_model}")
        expected = compute_definition_hash(
            canonical_name=r.canonical_name,
            warmup_periods=r.warmup_periods,
            gap_policy=r.gap_policy,
            memory_model=r.memory_model,
            primitive_ids=r.primitive_ids,
        )
        if r.definition_hash != expected:
            failed.append(f"DEFINITION_HASH_MATCH: {r.feature_uuid}")
        if r.transformation_uuid is not None:
            try:
                normalize_transformation_uuid(r.transformation_uuid)
            except ValueError:
                failed.append(f"TRANSFORMATION_UUID_FORMAT: {r.feature_uuid}")

    seen: set[str] = set()
    uniq: list[str] = []
    for item in failed:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    return ValidationReport(
        passed=len(uniq) == 0,
        failed_rules=uniq,
        warnings=warnings,
        seed_hash=_registry_content_hash(rows),
        expected_seed_hash="",
        validated_objects=f"{len(rows)} features",
        timestamp=_utc_now(),
    )
