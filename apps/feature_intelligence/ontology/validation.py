"""Ontology validation (strict / present modes) — Sprint 6."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from feature_intelligence.ontology import error_codes as ec
from feature_intelligence.ontology.catalog import (
    EXPECTED_ONTOLOGY_SEED_HASH,
    EXPECTED_VOCAB_SEED_HASH,
    SEED_ONTOLOGY_ROWS,
    SEED_VOCABULARIES,
    compute_ontology_seed_hash,
    compute_vocab_seed_hash,
)
from feature_intelligence.ontology.identity import (
    OBJECT_ID_PATTERNS,
    derive_ontology_uuid,
)
from feature_intelligence.ontology.models import (
    CLASSIFICATION_SOURCES,
    ONTOLOGY_VERSION,
    OBJECT_TYPE_FEATURE,
    OBJECT_TYPE_OPERATOR,
    OBJECT_TYPE_PRIMITIVE,
    OBJECT_TYPE_TRANSFORMATION,
    OntologyRecord,
    normalize_id_list,
)
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.ontology.vocab import (
    FIELD_VOCAB_TYPE,
    OBJECT_REF_PATTERN,
)
from feature_intelligence.registry.models import ValidationReport

_REQUIRED_TYPES = (OBJECT_TYPE_PRIMITIVE, OBJECT_TYPE_OPERATOR)

_SINGLE_FIELDS = ("domain", "horizon", "output_type", "frequency", "stability")
_MULTI_MIN1 = ("signal_type", "mathematical_family")

# Display/canonical lookalikes that must not appear as classification values
_FREE_TEXT_HINT = re.compile(r"^[a-z][a-z0-9_ ]*$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def validate_record_fields(
    record: OntologyRecord,
    *,
    vocab_types: dict[str, str],
    vocab_active: dict[str, bool],
    ontology_versions: set[str],
    strict_refs: bool = False,
    store: OntologyStore | None = None,
    for_new_assignment: bool = True,
) -> tuple[list[str], list[str]]:
    """Validate one ontology row. Returns (failed_rules, warnings)."""
    failed: list[str] = []
    warnings: list[str] = []

    expected_uuid = derive_ontology_uuid(record.object_type, record.object_id)
    if record.ontology_uuid != expected_uuid:
        _fail(failed, ec.ONT_ID_MISMATCH)

    pat = OBJECT_ID_PATTERNS.get(record.object_type)
    if pat is None or not pat.match(record.object_id):
        _fail(failed, ec.INVALID_OBJECT_REF)

    if record.ontology_version not in ontology_versions:
        _fail(failed, ec.ONTOLOGY_VERSION_MISMATCH)

    if record.confidence is not None:
        _fail(failed, ec.CONFIDENCE_NOT_NULL)

    if record.classification_source is not None:
        if record.classification_source not in CLASSIFICATION_SOURCES:
            _fail(failed, ec.INVALID_CLASSIFICATION_SOURCE)

    # Single fields
    for fname in _SINGLE_FIELDS:
        val = getattr(record, fname)
        if val is None or val == "":
            _fail(failed, ec.MISSING_FIELD)
            continue
        _check_vocab_ref(
            val,
            FIELD_VOCAB_TYPE[fname],
            vocab_types,
            vocab_active,
            failed,
            for_new_assignment=for_new_assignment,
        )

    # Multi min-1
    for fname in _MULTI_MIN1:
        vals = getattr(record, fname) or []
        if not vals:
            _fail(failed, ec.INVALID_CARDINALITY)
            continue
        normalized = normalize_id_list(vals)
        if list(vals) != normalized:
            # Either unsorted or has dupes
            if sorted(vals) != list(vals) or len(vals) != len(set(vals)):
                _fail(failed, ec.UNSORTED_MULTI)
            else:
                _fail(failed, ec.UNSORTED_MULTI)
        expected_type = FIELD_VOCAB_TYPE[fname]
        for vid in vals:
            _check_vocab_ref(
                vid,
                expected_type,
                vocab_types,
                vocab_active,
                failed,
                for_new_assignment=for_new_assignment,
            )

    # input_dependencies 0+
    deps = record.input_dependencies or []
    if list(deps) != normalize_id_list(deps):
        _fail(failed, ec.UNSORTED_MULTI)
    for dep in deps:
        if not OBJECT_REF_PATTERN.match(dep):
            _fail(failed, ec.INVALID_OBJECT_REF)
            continue
        if strict_refs and store is not None:
            dep_type = _infer_object_type(dep)
            if dep_type and not store.object_exists_in_registry(dep_type, dep):
                warnings.append(f"missing_ref:{dep}")

    if store is not None:
        if not store.object_exists_in_registry(record.object_type, record.object_id):
            _fail(failed, ec.OBJECT_NOT_FOUND)

    return failed, warnings


def _infer_object_type(object_id: str) -> str | None:
    if object_id.startswith("PR_"):
        return OBJECT_TYPE_PRIMITIVE
    if object_id.startswith("OP_"):
        return OBJECT_TYPE_OPERATOR
    if object_id.startswith("TR_"):
        return OBJECT_TYPE_TRANSFORMATION
    if object_id.startswith("FEAT_"):
        return OBJECT_TYPE_FEATURE
    return None


def _check_vocab_ref(
    value: str,
    expected_type: str,
    vocab_types: dict[str, str],
    vocab_active: dict[str, bool],
    failed: list[str],
    *,
    for_new_assignment: bool,
) -> None:
    if value not in vocab_types:
        # Free-text / display name heuristic
        if _FREE_TEXT_HINT.match(value) and not value.startswith(
            ("DOM_", "SIG_", "MATH_", "HOR_", "OUT_", "FREQ_", "STAB_")
        ):
            _fail(failed, ec.FREE_TEXT_CLASSIFICATION)
        else:
            _fail(failed, ec.INVALID_VOCAB_REF)
        return
    if vocab_types[value] != expected_type:
        _fail(failed, ec.VOCAB_TYPE_MISMATCH)
        return
    if for_new_assignment and not vocab_active.get(value, False):
        _fail(failed, ec.VOCAB_INACTIVE)


def validate_ontology(
    store: OntologyStore,
    *,
    mode: str = "strict",
    strict_refs: bool = False,
) -> ValidationReport:
    """Validate vocabulary + ontology rows. Does NOT write statistics (service does)."""
    if mode not in ("strict", "present"):
        raise ValueError(f"Unknown mode: {mode!r}")

    failed: list[str] = []
    warnings: list[str] = []

    seed_hash = compute_vocab_seed_hash()
    if seed_hash != EXPECTED_VOCAB_SEED_HASH:
        _fail(failed, ec.VOCAB_SEED_HASH_MISMATCH)

    ont_seed_hash = compute_ontology_seed_hash()
    if ont_seed_hash != EXPECTED_ONTOLOGY_SEED_HASH:
        _fail(failed, ec.ONTOLOGY_SEED_HASH_MISMATCH)

    expected_vids = {v.vocabulary_id for v in SEED_VOCABULARIES}
    db_vids = store.vocab_id_set()
    if expected_vids - db_vids:
        _fail(failed, ec.VOCAB_INCOMPLETE)
    if db_vids - expected_vids:
        _fail(failed, ec.VOCAB_EXTRA)

    vocab_types = store.vocab_type_map()
    vocab_active = store.vocab_active_map()
    ont_versions = store.ontology_versions() or {ONTOLOGY_VERSION}

    # Required coverage
    if mode == "strict":
        for ot in _REQUIRED_TYPES:
            registry_ids = set(store.registry_object_ids(ot))
            classified = {r.object_id for r in store.list_ontology(ot)}
            missing = registry_ids - classified
            if missing:
                _fail(failed, ec.MISSING_REQUIRED_ONTOLOGY)

    # Validate present rows
    seen_uuid: set[str] = set()
    seen_obj: set[tuple[str, str]] = set()
    counts: dict[str, int] = {
        "vocab": len(db_vids),
        "primitive_ontology": 0,
        "operator_ontology": 0,
        "transformation_ontology": 0,
        "feature_ontology": 0,
    }
    type_to_key = {
        OBJECT_TYPE_PRIMITIVE: "primitive_ontology",
        OBJECT_TYPE_OPERATOR: "operator_ontology",
        OBJECT_TYPE_TRANSFORMATION: "transformation_ontology",
        OBJECT_TYPE_FEATURE: "feature_ontology",
    }

    for rec in store.list_ontology():
        counts[type_to_key[rec.object_type]] = (
            counts.get(type_to_key[rec.object_type], 0) + 1
        )
        key = (rec.object_type, rec.object_id)
        if key in seen_obj or rec.ontology_uuid in seen_uuid:
            _fail(failed, ec.DUPLICATE_ONTOLOGY)
        seen_obj.add(key)
        seen_uuid.add(rec.ontology_uuid)

        # Historical rows may reference inactive vocab — treat as existing assignment
        row_failed, row_warn = validate_record_fields(
            rec,
            vocab_types=vocab_types,
            vocab_active=vocab_active,
            ontology_versions=ont_versions,
            strict_refs=strict_refs,
            store=store,
            for_new_assignment=False,
        )
        for c in row_failed:
            _fail(failed, c)
        warnings.extend(row_warn)

        # Also check multi JSON stored order
        for fname in _MULTI_MIN1 + ("input_dependencies",):
            vals = getattr(rec, fname) or []
            if fname == "input_dependencies" and not vals:
                continue
            if fname in _MULTI_MIN1 and not vals:
                continue
            if list(vals) != normalize_id_list(vals):
                _fail(failed, ec.UNSORTED_MULTI)

    # Ontology seed row presence check (hash already covers catalog; soft note)
    expected_seed_pairs = {(r.object_type, r.object_id) for r in SEED_ONTOLOGY_ROWS}
    present_pairs = {
        (r.object_type, r.object_id) for r in store.list_ontology()
    }
    if mode == "strict" and expected_seed_pairs - present_pairs:
        _fail(failed, ec.MISSING_REQUIRED_ONTOLOGY)

    validated = ";".join(f"{k}={v}" for k, v in counts.items())
    warnings.append(f"ontology_seed_hash={ont_seed_hash}")

    return ValidationReport(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warnings,
        seed_hash=seed_hash,
        expected_seed_hash=EXPECTED_VOCAB_SEED_HASH,
        validated_objects=validated,
        timestamp=_now(),
    )


def is_sorted_json_array(raw: str) -> bool:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, list):
        return False
    strs = [str(x) for x in data]
    return strs == normalize_id_list(strs)
