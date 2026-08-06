"""Research JSON / YAML / CSV import-export (Sprint 8)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.research import error_codes as ec
from feature_intelligence.research.identity import (
    derive_research_uuid,
    is_feat_uuid,
    normalize_feature_uuid,
)
from feature_intelligence.research.models import (
    RESEARCH_EXPORT_VERSION,
    RESEARCH_VERSION,
    SCHEMA_VERSION,
    SOURCE_IMPORT,
    FeatureResearchRecord,
    normalize_experiment_ids,
)
from feature_intelligence.research.service import ResearchService
from feature_intelligence.research.validation import validate_record

_CSV_FIELDS = [
    "research_uuid",
    "feature_uuid",
    "ontology_uuid",
    "transformation_uuid",
    "lineage_version",
    "compiler_version",
    "grammar_version",
    "research_status",
    "validation_status",
    "evidence_json",
    "strengths_json",
    "weaknesses_json",
    "regimes_json",
    "failure_modes_json",
    "experiment_ids",
    "notes",
    "record_source",
    "research_version",
    "schema_version",
    "research_export_version",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def record_to_envelope(r: FeatureResearchRecord) -> dict[str, Any]:
    return {
        "research_uuid": r.research_uuid,
        "feature_uuid": r.feature_uuid,
        "ontology_uuid": r.ontology_uuid,
        "transformation_uuid": r.transformation_uuid,
        "lineage_version": r.lineage_version,
        "compiler_version": r.compiler_version,
        "grammar_version": r.grammar_version,
        "research_status": r.research_status,
        "validation_status": r.validation_status,
        "evidence_json": r.evidence_json,
        "strengths_json": r.strengths_json,
        "weaknesses_json": r.weaknesses_json,
        "regimes_json": r.regimes_json,
        "failure_modes_json": r.failure_modes_json,
        "experiment_ids": normalize_experiment_ids(r.experiment_ids),
        "notes": r.notes,
        "record_source": r.record_source,
        "research_version": RESEARCH_VERSION,
        "schema_version": SCHEMA_VERSION,
        "research_export_version": RESEARCH_EXPORT_VERSION,
    }


def _canonical_records_payload(
    records: list[FeatureResearchRecord],
) -> list[dict[str, Any]]:
    rows = sorted(records, key=lambda r: r.research_uuid)
    return [record_to_envelope(r) for r in rows]


def export_research(
    service: ResearchService,
    path: Path,
    *,
    fmt: str = "json",
) -> Path:
    records = service.list_research()
    envelope = {
        "research_version": RESEARCH_VERSION,
        "schema_version": SCHEMA_VERSION,
        "research_export_version": RESEARCH_EXPORT_VERSION,
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
        for r in sorted(records, key=lambda x: x.research_uuid):
            exp = normalize_experiment_ids(r.experiment_ids)
            writer.writerow(
                {
                    "research_uuid": r.research_uuid,
                    "feature_uuid": r.feature_uuid,
                    "ontology_uuid": r.ontology_uuid or "",
                    "transformation_uuid": r.transformation_uuid or "",
                    "lineage_version": r.lineage_version or "",
                    "compiler_version": r.compiler_version or "",
                    "grammar_version": r.grammar_version or "",
                    "research_status": r.research_status,
                    "validation_status": r.validation_status,
                    "evidence_json": r.evidence_json or "",
                    "strengths_json": r.strengths_json or "",
                    "weaknesses_json": r.weaknesses_json or "",
                    "regimes_json": r.regimes_json or "",
                    "failure_modes_json": r.failure_modes_json or "",
                    "experiment_ids": (
                        json.dumps(exp, separators=(",", ":")) if exp is not None else ""
                    ),
                    "notes": r.notes or "",
                    "record_source": r.record_source or "",
                    "research_version": RESEARCH_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "research_export_version": RESEARCH_EXPORT_VERSION,
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
            from feature_intelligence.core import _yaml_lite

            data = _yaml_lite.safe_load(text)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if not isinstance(data, dict):
        raise ValueError("Envelope must be a mapping")
    return data


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _rows_from_payload(
    data: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if "records" in data:
        return list(data["records"])
    raise ValueError("No records in envelope")


def _empty_to_none(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, str) and val.strip() == "":
        return None
    return val


def _parse_experiment_ids(raw: Any) -> list[str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return normalize_experiment_ids([str(x) for x in raw])
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return normalize_experiment_ids([str(x) for x in parsed])
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _row_to_record(raw: dict[str, Any]) -> FeatureResearchRecord:
    feat = normalize_feature_uuid(str(raw["feature_uuid"]))
    uuid = str(raw.get("research_uuid") or derive_research_uuid(feat))
    return FeatureResearchRecord(
        research_uuid=uuid,
        feature_uuid=feat,
        ontology_uuid=_empty_to_none(raw.get("ontology_uuid")),
        transformation_uuid=_empty_to_none(raw.get("transformation_uuid")),
        lineage_version=_empty_to_none(raw.get("lineage_version")),
        compiler_version=_empty_to_none(raw.get("compiler_version")),
        grammar_version=_empty_to_none(raw.get("grammar_version")),
        research_status=str(raw.get("research_status") or "EMPTY"),
        validation_status=str(raw.get("validation_status") or "pending"),
        evidence_json=_empty_to_none(raw.get("evidence_json")),
        strengths_json=_empty_to_none(raw.get("strengths_json")),
        weaknesses_json=_empty_to_none(raw.get("weaknesses_json")),
        regimes_json=_empty_to_none(raw.get("regimes_json")),
        failure_modes_json=_empty_to_none(raw.get("failure_modes_json")),
        experiment_ids=_parse_experiment_ids(raw.get("experiment_ids")),
        notes=_empty_to_none(raw.get("notes")),
        record_source=_empty_to_none(raw.get("record_source")) or SOURCE_IMPORT,
    )


def import_research(
    service: ResearchService,
    path: Path,
    *,
    fmt: str = "json",
) -> ValidationReport:
    """
    Upsert FRR rows; identity must match formula; refresh checksum;
    do NOT write statistics.
    """
    failed: list[str] = []
    warnings: list[str] = []

    if fmt == "csv":
        rows = _parse_csv(path)
    else:
        data = _load_json_yaml(path, fmt)
        rows = _rows_from_payload(data)

    store = service.store
    imported = 0

    for raw in rows:
        rec = _row_to_record(raw)
        if not is_feat_uuid(rec.feature_uuid):
            if rec.feature_uuid.startswith(("PR_", "OP_", "TR_")):
                code = ec.PR_OP_TR_FRR_FORBIDDEN
            else:
                code = ec.INVALID_FEATURE_REF
            if code not in failed:
                failed.append(code)
            continue

        expected = derive_research_uuid(rec.feature_uuid)
        if rec.research_uuid != expected:
            if ec.FRR_ID_MISMATCH not in failed:
                failed.append(ec.FRR_ID_MISMATCH)
            continue

        if not store.feature_exists(rec.feature_uuid):
            if ec.FEATURE_NOT_FOUND not in failed:
                failed.append(ec.FEATURE_NOT_FOUND)
            continue

        e_fail, e_warn = validate_record(rec, store=store, strict_refs=True)
        # Skip ORPHAN since we already checked feature exists
        e_fail = [c for c in e_fail if c != ec.ORPHAN_FRR]
        for code in e_fail:
            if code not in failed:
                failed.append(code)
        warnings.extend(e_warn)
        if e_fail:
            continue

        if rec.record_source is None:
            rec.record_source = SOURCE_IMPORT
        store.upsert_record(rec)
        imported += 1

    checksum = store.recompute_and_store_checksum()
    return ValidationReport(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warnings,
        seed_hash=checksum,
        expected_seed_hash=checksum,
        validated_objects=f"imported={imported}",
        timestamp=_now(),
    )
