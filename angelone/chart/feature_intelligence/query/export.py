"""Query export helpers — JSON / YAML / CSV (Sprint 9 dump-only)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from feature_intelligence.query.models import (
    QUERY_ENGINE_VERSION,
    QUERY_EXPORT_VERSION,
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
)

_SEARCH_CSV_FIELDS = [
    "research_uuid",
    "feature_uuid",
    "canonical_name",
    "research_status",
    "validation_status",
    "ontology_uuid",
    "transformation_uuid",
]


def _yaml_dump(payload: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(payload, sort_keys=False)
    except ImportError:
        return json.dumps(payload, indent=2)


def wrap_export(
    data: Any,
    *,
    kind: str,
) -> dict[str, Any]:
    return {
        "query_export_version": QUERY_EXPORT_VERSION,
        "query_engine_version": QUERY_ENGINE_VERSION,
        "query_language_version": QUERY_LANGUAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "data": data,
    }


def export_payload(
    payload: dict[str, Any],
    out: Path,
    *,
    fmt: str = "json",
    kind: str = "search",
) -> Path:
    """
    Write search/inspect dump. CSV is tabular for search hits;
    inspect flattens key columns + JSON blob for nested sections.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    envelope = wrap_export(payload, kind=kind)

    if fmt == "json":
        out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        return out

    if fmt == "yaml":
        out.write_text(_yaml_dump(envelope), encoding="utf-8")
        return out

    if fmt == "csv":
        buf = io.StringIO()
        if kind == "search":
            items = []
            if isinstance(payload, dict):
                items = list(payload.get("items") or [])
            writer = csv.DictWriter(
                buf, fieldnames=_SEARCH_CSV_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for item in items:
                writer.writerow({k: item.get(k, "") for k in _SEARCH_CSV_FIELDS})
        else:
            # inspect: flat keys + sections_json
            fields = [
                "research_uuid",
                "feature_uuid",
                "canonical_name",
                "sections_present",
                "payload_json",
            ]
            writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            row = {
                "research_uuid": payload.get("research_uuid", "")
                if isinstance(payload, dict)
                else "",
                "feature_uuid": payload.get("feature_uuid", "")
                if isinstance(payload, dict)
                else "",
                "canonical_name": payload.get("canonical_name", "")
                if isinstance(payload, dict)
                else "",
                "sections_present": json.dumps(
                    payload.get("sections_present")
                    if isinstance(payload, dict)
                    else {},
                    separators=(",", ":"),
                ),
                "payload_json": json.dumps(payload, separators=(",", ":")),
            }
            writer.writerow(row)
        # prepend version comment line as first metadata via separate sidecar? keep simple:
        # write CSV body; versions live in a companion .meta.json if needed.
        # Spec: envelope carries versions — for CSV write a header comment block:
        text = (
            f"# query_export_version={QUERY_EXPORT_VERSION}\n"
            f"# query_engine_version={QUERY_ENGINE_VERSION}\n"
            f"# query_language_version={QUERY_LANGUAGE_VERSION}\n"
            f"# schema_version={SCHEMA_VERSION}\n"
            f"# kind={kind}\n"
        ) + buf.getvalue()
        out.write_text(text, encoding="utf-8")
        return out

    raise ValueError(f"Unsupported export format: {fmt}")
