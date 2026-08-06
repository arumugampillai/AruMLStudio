"""Primitive Catalog validation (Sprint 1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.registry.catalog import (
    CATALOG_VERSION,
    DATA_SOURCES,
    EXPECTED_SEED_CATALOG_HASH,
    PRIMITIVE_TYPES,
    SEED_BY_ID,
    SEED_PRIMITIVES,
    UNITS,
    compute_seed_catalog_hash,
)
from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.registry.store import PrimitiveStore
from feature_intelligence.registry.traceability import is_valid_primitive_id, looks_like_uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def validate_primitives(db_path: Path) -> ValidationReport:
    """Run the frozen Sprint 1 validation checklist."""
    store = PrimitiveStore(db_path)
    failed: list[str] = []
    warnings: list[str] = []
    seed_hash = compute_seed_catalog_hash()

    if seed_hash != EXPECTED_SEED_CATALOG_HASH:
        failed.append(
            f"SEED_HASH_MISMATCH: computed={seed_hash} expected={EXPECTED_SEED_CATALOG_HASH}"
        )

    if not store.table_exists():
        failed.append("TABLE_EXISTS: primitive_registry missing")
        return ValidationReport(
            passed=False,
            failed_rules=failed,
            warnings=warnings,
            seed_hash=seed_hash,
            expected_seed_hash=EXPECTED_SEED_CATALOG_HASH,
            validated_objects="0 primitives",
            timestamp=_utc_now(),
        )

    if not store.index_exists("idx_primitive_id"):
        failed.append("INDEX_PRIMITIVE_ID: idx_primitive_id missing")

    rows = store.list_all()
    by_id = {r.primitive_id: r for r in rows}
    seed_ids = {p.primitive_id for p in SEED_PRIMITIVES}

    missing = sorted(seed_ids - set(by_id))
    if missing:
        failed.append(f"SEED_COMPLETE: missing {missing}")

    extra = sorted(set(by_id) - seed_ids)
    if extra:
        failed.append(f"NO_EXTRA_ROWS: unexpected {extra}")

    names = [r.name for r in rows]
    if len(names) != len(set(names)):
        failed.append("UNIQUENESS: duplicate name values")
    if len(by_id) != len(rows):
        failed.append("UNIQUENESS: duplicate primitive_id values")

    for row in rows:
        if not is_valid_primitive_id(row.primitive_id):
            failed.append(f"ID_PATTERN: invalid {row.primitive_id}")
        if looks_like_uuid(row.primitive_id):
            failed.append(f"NO_UUID_IDS: forbidden UUID-shaped id {row.primitive_id}")

        if not row.name or not row.primitive_type or not row.data_source or not row.units:
            failed.append(f"REQUIRED_FIELDS: incomplete row {row.primitive_id}")
        if not row.catalog_version:
            failed.append(f"REQUIRED_FIELDS: missing catalog_version on {row.primitive_id}")

        if row.primitive_type not in PRIMITIVE_TYPES:
            failed.append(f"VOCAB_TYPE: {row.primitive_id}={row.primitive_type}")
        if row.data_source not in DATA_SOURCES:
            failed.append(f"VOCAB_SOURCE: {row.primitive_id}={row.data_source}")
        if row.units not in UNITS:
            failed.append(f"VOCAB_UNITS: {row.primitive_id}={row.units}")

        seed = SEED_BY_ID.get(row.primitive_id)
        if seed is not None:
            if (
                row.name != seed.name
                or row.primitive_type != seed.primitive_type
                or row.data_source != seed.data_source
                or row.units != seed.units
                or row.catalog_version != seed.catalog_version
            ):
                failed.append(f"SEED_FIELD_MATCH: drift on {row.primitive_id}")
            if row.catalog_version != CATALOG_VERSION:
                failed.append(
                    f"CATALOG_VERSION_PRESENT: {row.primitive_id} "
                    f"expected {CATALOG_VERSION} got {row.catalog_version}"
                )

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq_failed: list[str] = []
    for item in failed:
        if item not in seen:
            seen.add(item)
            uniq_failed.append(item)

    return ValidationReport(
        passed=len(uniq_failed) == 0,
        failed_rules=uniq_failed,
        warnings=warnings,
        seed_hash=seed_hash,
        expected_seed_hash=EXPECTED_SEED_CATALOG_HASH,
        validated_objects=f"{len(SEED_PRIMITIVES)} primitives",
        timestamp=_utc_now(),
    )
