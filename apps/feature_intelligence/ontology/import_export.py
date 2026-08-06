"""Ontology JSON / YAML / CSV import-export (Sprint 6)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.core import _yaml_lite
from feature_intelligence.ontology.catalog import (
    ONTOLOGY_VERSION,
    VOCAB_PACK_VERSION,
)
from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.models import (
    OBJECT_TYPES,
    OntologyRecord,
    normalize_id_list,
)
from feature_intelligence.ontology.service import OntologyService
from feature_intelligence.ontology.validation import validate_record_fields
from feature_intelligence.registry.models import ValidationReport

SCHEMA_VERSION = "1.0"

_CSV_FIELDS = [
    "object_type",
    "object_id",
    "ontology_uuid",
    "ontology_version",
    "domain",
    "signal_type",
    "mathematical_family",
    "horizon",
    "output_type",
    "frequency",
    "stability",
    "input_dependencies",
    "meaning",
    "classification_source",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _join_multi(vals: list[str]) -> str:
    return "|".join(normalize_id_list(vals))


def _split_multi(raw: str | None) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    return normalize_id_list(parts)


def record_to_envelope(r: OntologyRecord) -> dict[str, Any]:
    """Public envelope — never includes vocabulary_pk or confidence noise."""
    out: dict[str, Any] = {
        "ontology_uuid": r.ontology_uuid,
        "object_type": r.object_type,
        "object_id": r.object_id,
        "ontology_version": r.ontology_version,
        "domain": r.domain,
        "signal_type": normalize_id_list(r.signal_type),
        "mathematical_family": normalize_id_list(r.mathematical_family),
        "horizon": r.horizon,
        "output_type": r.output_type,
        "frequency": r.frequency,
        "stability": r.stability,
        "input_dependencies": normalize_id_list(r.input_dependencies),
        "meaning": r.meaning,
        "confidence": None,
        "classification_source": r.classification_source,
    }
    return out


def _canonical_records_payload(records: list[OntologyRecord]) -> list[dict[str, Any]]:
    rows = sorted(records, key=lambda r: (r.object_type, r.object_id))
    return [record_to_envelope(r) for r in rows]


def export_ontology(
    service: OntologyService,
    path: Path,
    *,
    fmt: str = "json",
    object_type: str | None = None,
) -> Path:
    records = service.list_ontology(object_type)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "vocab_pack_version": VOCAB_PACK_VERSION,
        "exported_at": _now(),
        "records": _canonical_records_payload(records),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    elif fmt == "yaml":
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(envelope, sort_keys=False), encoding="utf-8")
        except ImportError:
            out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x.object_type, x.object_id)):
            writer.writerow(
                {
                    "object_type": r.object_type,
                    "object_id": r.object_id,
                    "ontology_uuid": r.ontology_uuid,
                    "ontology_version": r.ontology_version,
                    "domain": r.domain,
                    "signal_type": _join_multi(r.signal_type),
                    "mathematical_family": _join_multi(r.mathematical_family),
                    "horizon": r.horizon,
                    "output_type": r.output_type,
                    "frequency": r.frequency,
                    "stability": r.stability,
                    "input_dependencies": _join_multi(r.input_dependencies),
                    "meaning": r.meaning or "",
                    "classification_source": r.classification_source or "",
                }
            )
        out.write_text(buf.getvalue(), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out


def _load_json_yaml(path: Path, fmt: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if fmt == "json":
        data = json.loads(text)
    elif fmt == "yaml":
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            data = _yaml_lite.loads(text)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if not isinstance(data, dict):
        raise ValueError("Envelope must be a mapping")
    return data


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        rows.append(
            {
                "object_type": raw.get("object_type", ""),
                "object_id": raw.get("object_id", ""),
                "ontology_uuid": raw.get("ontology_uuid") or None,
                "ontology_version": raw.get("ontology_version") or ONTOLOGY_VERSION,
                "domain": raw.get("domain", ""),
                "signal_type": _split_multi(raw.get("signal_type")),
                "mathematical_family": _split_multi(raw.get("mathematical_family")),
                "horizon": raw.get("horizon", ""),
                "output_type": raw.get("output_type", ""),
                "frequency": raw.get("frequency", ""),
                "stability": raw.get("stability", ""),
                "input_dependencies": _split_multi(raw.get("input_dependencies")),
                "meaning": (raw.get("meaning") or None) or None,
                "classification_source": (raw.get("classification_source") or None)
                or None,
            }
        )
    return rows


def _parse_record(raw: dict[str, Any]) -> OntologyRecord:
    object_type = str(raw["object_type"])
    object_id = str(raw["object_id"])
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"Unknown object_type: {object_type!r}")
    sig = raw.get("signal_type") or []
    math = raw.get("mathematical_family") or []
    deps = raw.get("input_dependencies") or []
    if isinstance(sig, str):
        sig = _split_multi(sig)
    if isinstance(math, str):
        math = _split_multi(math)
    if isinstance(deps, str):
        deps = _split_multi(deps)
    uuid = str(raw.get("ontology_uuid") or "") or derive_ontology_uuid(
        object_type, object_id
    )
    src = raw.get("classification_source")
    if src is None or src == "":
        src = "IMPORT"
    else:
        src = str(src)
    return OntologyRecord(
        ontology_uuid=uuid,
        object_type=object_type,
        object_id=object_id,
        ontology_version=str(raw.get("ontology_version") or ONTOLOGY_VERSION),
        domain=str(raw["domain"]),
        signal_type=normalize_id_list(list(sig)),
        mathematical_family=normalize_id_list(list(math)),
        horizon=str(raw["horizon"]),
        output_type=str(raw["output_type"]),
        frequency=str(raw["frequency"]),
        stability=str(raw["stability"]),
        input_dependencies=normalize_id_list(list(deps)),
        meaning=(None if raw.get("meaning") in (None, "") else str(raw["meaning"])),
        confidence=None,
        classification_source=src,
    )


def import_ontology(
    service: OntologyService,
    path: Path,
    *,
    fmt: str = "json",
) -> ValidationReport:
    """
    Upsert ontology rows. Sets classification_source='IMPORT' when absent.
    Does NOT write ontology_statistics (freeze §9.4.1).
    """
    failed: list[str] = []
    warnings: list[str] = []
    touched = 0

    if fmt == "csv":
        raw_records = _records_from_csv(Path(path))
    else:
        envelope = _load_json_yaml(Path(path), fmt)
        if str(envelope.get("schema_version") or "") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version: {envelope.get('schema_version')!r}"
            )
        raw_records = list(envelope.get("records") or [])

    store = service.store
    vocab_types = store.vocab_type_map()
    vocab_active = store.vocab_active_map()
    ont_versions = store.ontology_versions() or {ONTOLOGY_VERSION}

    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Ontology record must be a mapping")
        rec = _parse_record(raw)
        # Ensure IMPORT source when caller left blank (already defaulted)
        if not rec.classification_source:
            rec.classification_source = "IMPORT"
        row_failed, row_warn = validate_record_fields(
            rec,
            vocab_types=vocab_types,
            vocab_active=vocab_active,
            ontology_versions=ont_versions,
            store=store,
            for_new_assignment=True,
        )
        if row_failed:
            failed.extend(row_failed)
            warnings.append(f"skipped:{rec.object_type}:{rec.object_id}")
            continue
        warnings.extend(row_warn)
        service.upsert_ontology(rec)
        touched += 1

    # Deduplicate failed codes while preserving order
    seen: set[str] = set()
    uniq_failed: list[str] = []
    for c in failed:
        if c not in seen:
            seen.add(c)
            uniq_failed.append(c)

    return ValidationReport(
        passed=len(uniq_failed) == 0,
        failed_rules=uniq_failed,
        warnings=warnings + [f"imported={touched}"],
        seed_hash="",
        expected_seed_hash="",
        validated_objects=f"imported={touched}",
        timestamp=_now(),
    )
