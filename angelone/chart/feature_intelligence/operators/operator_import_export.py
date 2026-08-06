"""Operator Registry JSON/YAML import-export (Sprint 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_intelligence.core import _yaml_lite
from feature_intelligence.operators.catalog import (
    OPERATOR_CATALOG_VERSION,
    OPERATOR_PACK_VERSION,
)
from feature_intelligence.operators.operator_models import OperatorRecord
from feature_intelligence.operators.operator_service import OperatorRegistryService

SCHEMA_VERSION = "1.0"


def _to_export(r: OperatorRecord) -> dict[str, Any]:
    return {
        "operator_id": r.operator_id,
        "canonical_name": r.canonical_name,
        "display_name": r.display_name,
        "category": r.category,
        "description": r.description,
        "formula": r.formula,
        "definition_text": r.definition_text,
        "parameter_schema": json.loads(r.parameter_schema_json),
        "depends_on_operator_ids": (
            None
            if r.depends_on_operator_ids is None
            else json.loads(r.depends_on_operator_ids)
        ),
        "input_arity_min": r.input_arity_min,
        "input_arity_max": r.input_arity_max,
        "output_count": r.output_count,
        "warmup_policy": r.warmup_policy,
        "missing_data_policy": r.missing_data_policy,
        "deterministic": r.deterministic,
        "stateful": r.stateful,
        "streaming_supported": r.streaming_supported,
        "incremental_supported": r.incremental_supported,
        "complexity_class": r.complexity_class,
        "extras_json": r.extras_json,
        "operator_version": r.operator_version,
        "catalog_version": r.catalog_version,
        "operator_pack_version": r.operator_pack_version,
    }


def export_operators(
    service: OperatorRegistryService,
    path: Path,
    *,
    fmt: str = "json",
) -> Path:
    from datetime import datetime, timezone

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": OPERATOR_CATALOG_VERSION,
        "operator_pack_version": OPERATOR_PACK_VERSION,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "operators": [_to_export(r) for r in service.list_operators()],
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
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out


def _load(path: Path, fmt: str) -> dict[str, Any]:
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


def import_operators(
    service: OperatorRegistryService,
    path: Path,
    *,
    fmt: str = "json",
    force: bool = False,
) -> list[str]:
    envelope = _load(Path(path), fmt)
    if str(envelope.get("schema_version") or "") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {envelope.get('schema_version')!r}")
    pack = str(envelope.get("operator_pack_version") or "")
    if pack != OPERATOR_PACK_VERSION and not force:
        raise ValueError(f"Pack mismatch: {pack} != {OPERATOR_PACK_VERSION}")

    touched: list[str] = []
    for raw in envelope.get("operators") or []:
        if not isinstance(raw, dict):
            raise ValueError("Operator record must be a mapping")
        oid = str(raw["operator_id"])
        name = str(raw["canonical_name"])
        existing = service._store.get_by_name(name)  # noqa: SLF001
        if existing is not None and existing.operator_id != oid:
            raise ValueError(f"NAME_ID_CONFLICT:{name}")
        if service.operator_exists(oid):
            service.update_metadata(
                oid,
                display_name=str(raw.get("display_name") or name),
                description=raw.get("description"),
            )
            touched.append(oid)
            continue
        schema = raw.get("parameter_schema") or {}
        record = OperatorRecord(
            operator_id=oid,
            canonical_name=name,
            display_name=str(raw.get("display_name") or name),
            category=str(raw["category"]),
            description=raw.get("description"),
            formula=str(raw["formula"]),
            definition_text=str(raw["definition_text"]),
            parameter_schema_json=json.dumps(schema, sort_keys=True, separators=(",", ":")),
            depends_on_operator_ids=None,
            input_arity_min=int(raw["input_arity_min"]),
            input_arity_max=(
                None if raw.get("input_arity_max") is None else int(raw["input_arity_max"])
            ),
            output_count=int(raw.get("output_count") or 1),
            warmup_policy=str(raw["warmup_policy"]),
            missing_data_policy=str(raw["missing_data_policy"]),
            deterministic=bool(raw.get("deterministic", True)),
            stateful=bool(raw.get("stateful", False)),
            streaming_supported=bool(raw.get("streaming_supported", False)),
            incremental_supported=bool(raw.get("incremental_supported", False)),
            complexity_class=str(raw["complexity_class"]),
            extras_json=raw.get("extras_json"),
            operator_version=str(raw.get("operator_version") or "1.0"),
            catalog_version=str(raw.get("catalog_version") or OPERATOR_CATALOG_VERSION),
            operator_pack_version=str(raw.get("operator_pack_version") or OPERATOR_PACK_VERSION),
        )
        service.register_operator(record)
        touched.append(oid)
    return touched
