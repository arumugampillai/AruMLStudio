"""Operator Registry validation (Sprint 3)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from feature_intelligence.operators.catalog import (
    EXPECTED_OPERATOR_CATALOG_HASH,
    OPERATOR_CATALOG_VERSION,
    OPERATOR_ID_PATTERN,
    OPERATOR_PACK_VERSION,
    SEED_BY_ID,
    SEED_OPERATORS,
    catalog_artifact_path,
    compute_operator_catalog_hash,
)
from feature_intelligence.operators.operator_models import (
    CATEGORIES,
    COMPLEXITY_CLASSES,
    MISSING_DATA_POLICIES,
    PARAMETER_TYPES,
    WARMUP_POLICIES,
)
from feature_intelligence.operators.operator_store import OperatorStore
from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.registry.traceability import looks_like_uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_param_schema(schema_json: str, operator_id: str, failed: list[str]) -> None:
    try:
        obj = json.loads(schema_json)
    except json.JSONDecodeError:
        failed.append(f"PARAM_SCHEMA_VALID: invalid JSON on {operator_id}")
        return
    if not isinstance(obj, dict) or obj.get("type") != "object":
        failed.append(f"PARAM_SCHEMA_VALID: root object required on {operator_id}")
        return
    if obj.get("additionalProperties") is not False:
        failed.append(f"PARAM_SCHEMA_VALID: additionalProperties false required on {operator_id}")
    props = obj.get("properties") or {}
    if not isinstance(props, dict):
        failed.append(f"PARAM_SCHEMA_VALID: properties map required on {operator_id}")
        return
    required = obj.get("required") or []
    if not isinstance(required, list):
        failed.append(f"PARAM_SCHEMA_VALID: required must be list on {operator_id}")
        return
    for pname, pschema in props.items():
        if not isinstance(pschema, dict):
            failed.append(f"PARAM_TYPE_VALID: {operator_id}.{pname}")
            continue
        ptype = pschema.get("type")
        if ptype not in PARAMETER_TYPES:
            failed.append(f"PARAM_TYPE_VALID: {operator_id}.{pname}={ptype}")
            continue
        if "default" in pschema:
            # Structural: default present is OK; type-check light
            default = pschema["default"]
            if ptype == "integer" and not isinstance(default, int):
                failed.append(f"PARAM_DEFAULT_INVALID: {operator_id}.{pname}")
            if ptype == "float" and not isinstance(default, (int, float)):
                failed.append(f"PARAM_DEFAULT_INVALID: {operator_id}.{pname}")
            if ptype == "boolean" and not isinstance(default, bool):
                failed.append(f"PARAM_DEFAULT_INVALID: {operator_id}.{pname}")
            if ptype == "string" and not isinstance(default, str):
                failed.append(f"PARAM_DEFAULT_INVALID: {operator_id}.{pname}")
        if ptype == "list" and "items" not in pschema:
            failed.append(f"PARAM_SCHEMA_VALID: list needs items on {operator_id}.{pname}")
    for r in required:
        if r not in props:
            failed.append(f"PARAM_SCHEMA_VALID: required missing prop {operator_id}.{r}")


def validate_operators(db_path: Path) -> ValidationReport:
    store = OperatorStore(db_path)
    failed: list[str] = []
    warnings: list[str] = []
    seed_hash = compute_operator_catalog_hash()

    if seed_hash != EXPECTED_OPERATOR_CATALOG_HASH:
        failed.append(
            f"SEED_HASH_MISMATCH: computed={seed_hash} expected={EXPECTED_OPERATOR_CATALOG_HASH}"
        )

    art = catalog_artifact_path()
    if not art.is_file():
        failed.append("CATALOG_ARTIFACT: operator_catalog.json missing")
    else:
        try:
            payload = json.loads(art.read_text(encoding="utf-8"))
            if payload.get("expected_catalog_hash") != EXPECTED_OPERATOR_CATALOG_HASH:
                failed.append("CATALOG_ARTIFACT: hash drift vs EXPECTED_OPERATOR_CATALOG_HASH")
            if len(payload.get("operators") or []) != len(SEED_OPERATORS):
                failed.append("CATALOG_ARTIFACT: operator count mismatch")
        except (OSError, json.JSONDecodeError) as exc:
            failed.append(f"CATALOG_ARTIFACT: {exc}")

    if not store.table_exists():
        return ValidationReport(
            passed=False,
            failed_rules=failed + ["TABLE_EXISTS: operator_registry missing"],
            warnings=warnings,
            seed_hash=seed_hash,
            expected_seed_hash=EXPECTED_OPERATOR_CATALOG_HASH,
            validated_objects="0 operators",
            timestamp=_utc_now(),
        )

    rows = store.list_all()
    by_id = {r.operator_id: r for r in rows}
    seed_ids = {o.operator_id for o in SEED_OPERATORS}

    missing = sorted(seed_ids - set(by_id))
    if missing:
        failed.append(f"SEED_COMPLETE: missing {missing}")
    extra = sorted(set(by_id) - seed_ids)
    if extra:
        failed.append(f"NO_EXTRA_ROWS: unexpected {extra}")

    names = [r.canonical_name for r in rows]
    if len(names) != len(set(names)):
        failed.append("NAME_UNIQUE: duplicate canonical_name")
    if len(by_id) != len(rows):
        failed.append("ID_UNIQUE: duplicate operator_id")

    id_re = re.compile(OPERATOR_ID_PATTERN)
    for r in rows:
        if not id_re.fullmatch(r.operator_id):
            failed.append(f"ID_PATTERN: {r.operator_id}")
        if looks_like_uuid(r.operator_id):
            failed.append(f"NO_UUID_IDS: {r.operator_id}")
        if r.category not in CATEGORIES:
            failed.append(f"CATEGORY_VALID: {r.operator_id}={r.category}")
        if r.warmup_policy not in WARMUP_POLICIES:
            failed.append(f"WARMUP_POLICY_VALID: {r.operator_id}")
        if r.missing_data_policy not in MISSING_DATA_POLICIES:
            failed.append(f"MISSING_DATA_POLICY_VALID: {r.operator_id}")
        if r.complexity_class not in COMPLEXITY_CLASSES:
            failed.append(f"COMPLEXITY_CLASS_VALID: {r.operator_id}")
        if r.operator_pack_version != OPERATOR_PACK_VERSION:
            failed.append(f"PACK_VERSION: {r.operator_id}")
        if r.catalog_version != OPERATOR_CATALOG_VERSION:
            failed.append(f"CATALOG_VERSION: {r.operator_id}")
        if r.depends_on_operator_ids is not None:
            failed.append(f"DEPENDS_ON_NULL_SPRINT3: {r.operator_id}")
        _validate_param_schema(r.parameter_schema_json, r.operator_id, failed)

        seed = SEED_BY_ID.get(r.operator_id)
        if seed is not None:
            if (
                r.canonical_name != seed.canonical_name
                or r.category != seed.category
                or r.warmup_policy != seed.warmup_policy
                or r.missing_data_policy != seed.missing_data_policy
                or int(r.deterministic) != seed.deterministic
                or r.complexity_class != seed.complexity_class
            ):
                failed.append(f"SEED_FIELD_MATCH: {r.operator_id}")
            else:
                # semantics id present in freeze table
                pass

    # SEMANTICS_ID_STABLE: all freeze IDs present
    if not missing:
        # already covered by SEED_COMPLETE
        pass

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
        seed_hash=seed_hash,
        expected_seed_hash=EXPECTED_OPERATOR_CATALOG_HASH,
        validated_objects=f"{len(rows)} operators",
        timestamp=_utc_now(),
    )
