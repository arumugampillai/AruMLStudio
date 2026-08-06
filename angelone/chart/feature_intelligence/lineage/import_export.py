"""Lineage JSON / YAML / CSV import-export (Sprint 7)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_intelligence.lineage.graph import has_cycle, would_introduce_cycle
from feature_intelligence.lineage.identity import (
    derive_lineage_uuid,
    normalize_object_id,
)
from feature_intelligence.lineage.models import (
    GRAPH_EXPORT_VERSION,
    GRAPH_SCHEMA_VERSION,
    LINEAGE_VERSION,
    LineageEdge,
)
from feature_intelligence.lineage.relationships import RELATIONSHIP_PACK_VERSION
from feature_intelligence.lineage.service import LineageService
from feature_intelligence.lineage.validation import validate_edge
from feature_intelligence.registry.models import ValidationReport

_CSV_FIELDS = [
    "lineage_uuid",
    "parent_object",
    "child_object",
    "relationship_id",
    "edge_source",
    "lineage_version",
    "graph_schema_version",
    "graph_export_version",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def edge_to_envelope(e: LineageEdge) -> dict[str, Any]:
    return {
        "lineage_uuid": e.lineage_uuid,
        "parent_object": e.parent_object,
        "child_object": e.child_object,
        "relationship_id": e.relationship_id,
        "edge_source": e.edge_source,
        "lineage_version": LINEAGE_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_export_version": GRAPH_EXPORT_VERSION,
    }


def _canonical_edges_payload(edges: list[LineageEdge]) -> list[dict[str, Any]]:
    rows = sorted(
        edges,
        key=lambda e: (e.parent_object, e.child_object, e.relationship_id),
    )
    return [edge_to_envelope(e) for e in rows]


def export_lineage(
    service: LineageService,
    path: Path,
    *,
    fmt: str = "json",
    relationship_id: str | None = None,
) -> Path:
    edges = service.list_edges(relationship_id)
    envelope = {
        "lineage_version": LINEAGE_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_export_version": GRAPH_EXPORT_VERSION,
        "relationship_pack_version": RELATIONSHIP_PACK_VERSION,
        "exported_at": _now(),
        "edges": _canonical_edges_payload(edges),
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
        for e in sorted(
            edges, key=lambda x: (x.parent_object, x.child_object, x.relationship_id)
        ):
            writer.writerow(
                {
                    "lineage_uuid": e.lineage_uuid,
                    "parent_object": e.parent_object,
                    "child_object": e.child_object,
                    "relationship_id": e.relationship_id,
                    "edge_source": e.edge_source or "",
                    "lineage_version": LINEAGE_VERSION,
                    "graph_schema_version": GRAPH_SCHEMA_VERSION,
                    "graph_export_version": GRAPH_EXPORT_VERSION,
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


def _rows_from_payload(data: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if "edges" in data:
        return list(data["edges"])
    if "records" in data:
        return list(data["records"])
    raise ValueError("No edges/records in envelope")


def import_lineage(
    service: LineageService,
    path: Path,
    *,
    fmt: str = "json",
) -> ValidationReport:
    """
    Upsert edges by lineage_uuid; refresh checksum; do NOT write statistics.
    """
    failed: list[str] = []
    warnings: list[str] = []

    if fmt == "csv":
        rows = _parse_csv(path)
    else:
        data = _load_json_yaml(path, fmt)
        rows = _rows_from_payload(data)

    store = service.store
    rel_ids = store.relationship_id_set()
    rel_active = store.relationship_active_map()
    pack_versions = store.list_pack_versions()
    applied_pairs = store.edge_pairs()

    for raw in rows:
        parent = normalize_object_id(str(raw["parent_object"]))
        child = normalize_object_id(str(raw["child_object"]))
        rid = str(raw["relationship_id"])
        uuid = str(raw.get("lineage_uuid") or derive_lineage_uuid(parent, child, rid))
        expected = derive_lineage_uuid(parent, child, rid)
        if uuid != expected:
            uuid = expected
            warnings.append(f"uuid_normalized:{parent}|{child}|{rid}")
        source = raw.get("edge_source") or "IMPORT"
        if source == "":
            source = "IMPORT"
        edge = LineageEdge(
            lineage_uuid=uuid,
            parent_object=parent,
            child_object=child,
            relationship_id=rid,
            edge_source=str(source),
        )
        e_fail, e_warn = validate_edge(
            edge,
            rel_ids=rel_ids,
            rel_active=rel_active,
            pack_versions=pack_versions,
            store=store,
            for_new_assignment=True,
        )
        warnings.extend(e_warn)
        if e_fail:
            for c in e_fail:
                if c not in failed:
                    failed.append(c)
            continue
        if would_introduce_cycle(applied_pairs, parent, child):
            existing = store.get_edge_by_triple(parent, child, rid)
            if existing is None:
                from feature_intelligence.lineage import error_codes as ec

                if ec.CYCLE_DETECTED not in failed:
                    failed.append(ec.CYCLE_DETECTED)
                warnings.append(f"cycle_reject:{parent}->{child}")
                continue
        store.upsert_edge(edge)
        if (parent, child) not in applied_pairs:
            applied_pairs.append((parent, child))

    # Shared checksum writer; never stats
    checksum = store.recompute_and_store_graph_checksum()
    if has_cycle(store.edge_pairs()):
        from feature_intelligence.lineage import error_codes as ec

        if ec.CYCLE_DETECTED not in failed:
            failed.append(ec.CYCLE_DETECTED)

    return ValidationReport(
        passed=len(failed) == 0,
        failed_rules=failed,
        warnings=warnings,
        seed_hash="",
        expected_seed_hash="",
        validated_objects=f"imported_rows={len(rows)};checksum={checksum}",
        timestamp=_now(),
    )
