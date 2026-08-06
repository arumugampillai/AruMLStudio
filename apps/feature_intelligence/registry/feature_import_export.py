"""Feature Registry JSON/YAML import-export (Sprint 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_intelligence.core import _yaml_lite
from feature_intelligence.registry.feature_ids import normalize_feature_uuid
from feature_intelligence.registry.feature_models import FeatureRecord
from feature_intelligence.registry.feature_service import FeatureRegistryService

SCHEMA_VERSION = "1.0"


def _record_to_export(r: FeatureRecord) -> dict[str, Any]:
    return {
        "feature_uuid": r.feature_uuid,
        "canonical_name": r.canonical_name,
        "display_name": r.display_name,
        "definition_version": r.definition_version,
        "implementation_version": r.implementation_version,
        "feature_version": r.feature_version,
        "definition_hash": r.definition_hash,
        "transformation_uuid": r.transformation_uuid,
        "legacy_feature_id": r.legacy_feature_id,
        "description": r.description,
        "created_by": r.created_by,
        "controller_owner": r.controller_owner,
        "warmup_periods": r.warmup_periods,
        "gap_policy": r.gap_policy,
        "memory_model": r.memory_model,
        "research_state": r.research_state,
        "primitive_ids": list(r.primitive_ids),
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


def export_features(
    service: FeatureRegistryService,
    path: Path,
    *,
    fmt: str = "json",
) -> Path:
    from datetime import datetime, timezone

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "features": [_record_to_export(r) for r in service.list_features()],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    elif fmt == "yaml":
        # Minimal YAML dump (enough for round-trip via our loader / PyYAML)
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(envelope, sort_keys=False), encoding="utf-8")
        except ImportError:
            # JSON-compatible YAML subset
            out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out


def _load_envelope(path: Path, fmt: str) -> dict[str, Any]:
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
        raise ValueError("Envelope root must be a mapping")
    return data


def import_features(
    service: FeatureRegistryService,
    path: Path,
    *,
    fmt: str = "json",
) -> list[str]:
    """Import features. Preserves UUIDs when present. Returns registered/updated names."""
    envelope = _load_envelope(Path(path), fmt)
    if str(envelope.get("schema_version") or "") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version: {envelope.get('schema_version')!r}"
        )
    features = envelope.get("features") or []
    if not isinstance(features, list):
        raise ValueError("features must be a list")

    touched: list[str] = []
    for raw in features:
        if not isinstance(raw, dict):
            raise ValueError("Feature record must be a mapping")
        name = str(raw["canonical_name"])
        uuid_raw = raw.get("feature_uuid")
        existing_by_name = service._store.get_by_name(name)  # noqa: SLF001
        if existing_by_name is not None:
            if uuid_raw is not None:
                try:
                    norm = normalize_feature_uuid(str(uuid_raw))
                except ValueError as exc:
                    raise ValueError(f"Invalid feature_uuid on import: {uuid_raw}") from exc
                if existing_by_name.feature_uuid != norm:
                    raise ValueError(f"NAME_UUID_CONFLICT:{name}")
            service.update_metadata(
                existing_by_name.feature_uuid,
                display_name=str(raw.get("display_name") or name),
                description=raw.get("description"),
                controller_owner=str(raw.get("controller_owner") or "import"),
                research_state=str(raw.get("research_state") or "EXPERIMENTAL"),
                implementation_version=str(raw.get("implementation_version") or "1"),
            )
            touched.append(name)
            continue

        service.register_feature(
            canonical_name=name,
            display_name=str(raw.get("display_name") or name),
            primitive_ids=list(raw.get("primitive_ids") or []),
            created_by=str(raw.get("created_by") or "import"),
            controller_owner=str(raw.get("controller_owner") or "import"),
            warmup_periods=int(raw.get("warmup_periods") or 0),
            gap_policy=str(raw.get("gap_policy") or "CONTINUOUS"),
            memory_model=str(raw.get("memory_model") or "TICK"),
            definition_version=str(raw.get("definition_version") or "1.0"),
            implementation_version=str(raw.get("implementation_version") or "1"),
            research_state=str(raw.get("research_state") or "EXPERIMENTAL"),
            description=raw.get("description"),
            feature_version=raw.get("feature_version"),
            transformation_uuid=raw.get("transformation_uuid"),
            legacy_feature_id=raw.get("legacy_feature_id"),
            feature_uuid=None if uuid_raw is None else str(uuid_raw),
        )
        touched.append(name)
    return touched
