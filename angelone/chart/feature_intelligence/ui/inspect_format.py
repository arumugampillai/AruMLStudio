"""Pure formatting helpers for Feature Inspector / Explorer (Sprint 9 UI).

No DB access. Operates on inspect / search envelopes only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from feature_intelligence.query.models import QUERY_FIELDS

ABSENT = "(absent)"
# Include language aliases so UI structured detection accepts feat:…
from feature_intelligence.query.language import FIELD_ALIASES

_FIELD_SET = frozenset(QUERY_FIELDS) | frozenset(FIELD_ALIASES)
_TOKEN_RE = re.compile(r"^([a-z_]+):(\S+)$")

SCOPE_NAME = "name"
SCOPE_FEAT = "feat"
SCOPE_PRIMITIVE = "primitive"
SCOPE_OPERATOR = "operator"
SCOPE_ONTOLOGY = "ontology"

ALL_SCOPES: tuple[str, ...] = (
    SCOPE_NAME,
    SCOPE_FEAT,
    SCOPE_PRIMITIVE,
    SCOPE_OPERATOR,
    SCOPE_ONTOLOGY,
)

SCOPE_LABELS: dict[str, str] = {
    SCOPE_NAME: "Name",
    SCOPE_FEAT: "FEAT",
    SCOPE_PRIMITIVE: "Primitive",
    SCOPE_OPERATOR: "Operator",
    SCOPE_ONTOLOGY: "Ontology",
}


def display_value(value: Any, *, absent: str = ABSENT) -> str:
    if value is None:
        return absent
    if isinstance(value, str) and not value.strip():
        return absent
    if isinstance(value, (list, tuple)):
        if not value:
            return absent
        return ", ".join(display_value(v, absent=absent) for v in value)
    if isinstance(value, dict):
        if not value:
            return absent
        return json.dumps(value, indent=2, sort_keys=True)
    return str(value)


def short_id(value: Any, *, max_len: int = 28) -> str:
    text = display_value(value)
    if text == ABSENT:
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _looks_structured(text: str) -> bool:
    tokens = [t for t in text.split() if t]
    if not tokens:
        return False
    for raw in tokens:
        m = _TOKEN_RE.match(raw)
        if m is None or m.group(1) not in _FIELD_SET:
            return False
    return True


@dataclass(frozen=True)
class SearchPlan:
    """How Studio should call search_features + optional client filters."""

    mode: str  # structured | match_all | scoped
    structured_query: str | None = None
    match_all: bool = False
    engine_queries: tuple[str, ...] = ()
    name_substring: str | None = None
    feat_substring: str | None = None
    research_substring: str | None = None
    note: str = ""


def build_search_plan(
    text: str | None,
    scopes: set[str] | frozenset[str] | None = None,
) -> SearchPlan:
    """
    Map Explorer search box + scope checkboxes → query engine plan.

    - Advanced: all tokens are ``field:value`` → pass through as structured query.
    - Empty text → ``match_all`` (list FRRs).
    - Simple text → union of scope strategies (default scope: Name).
      Name uses client-side case-insensitive substring on ``canonical_name``
      after ``match_all`` (engine ``feature:`` is exact-only).
    """
    raw = "" if text is None else str(text).strip()
    active = set(scopes or ()) & set(ALL_SCOPES)
    if not active:
        active = {SCOPE_NAME}

    if raw and _looks_structured(raw):
        return SearchPlan(
            mode="structured",
            structured_query=raw,
            note="Advanced structured query (field:value); scopes ignored",
        )

    if not raw:
        return SearchPlan(
            mode="match_all",
            match_all=True,
            note="Empty search lists all FRR subjects (match_all)",
        )

    upper = raw.upper()
    # Strong id prefixes override / complement scopes
    if upper.startswith("FEAT_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"feature:{raw}",),
            note="Prefix FEAT_ → feature:<id>",
        )
    if upper.startswith("FRR_"):
        return SearchPlan(
            mode="match_all",
            match_all=True,
            research_substring=raw,
            note="Prefix FRR_ → client filter on research_uuid",
        )
    if upper.startswith("PR_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"primitive:{raw}",),
            note="Prefix PR_ → primitive:<id>",
        )
    if upper.startswith("OP_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"operator:{raw}",),
            note="Prefix OP_ → operator:<id>",
        )
    if upper.startswith("TR_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"transformation:{raw}",),
            note="Prefix TR_ → transformation:<id>",
        )
    if upper.startswith("DOM_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"domain:{raw}",),
            note="Prefix DOM_ → domain:<id>",
        )
    if upper.startswith("SIG_"):
        return SearchPlan(
            mode="scoped",
            engine_queries=(f"signal:{raw}",),
            note="Prefix SIG_ → signal:<id>",
        )

    engine_queries: list[str] = []
    name_substring: str | None = None
    feat_substring: str | None = None
    notes: list[str] = []

    if SCOPE_NAME in active:
        name_substring = raw
        notes.append("Name → substring on canonical_name (client)")
    if SCOPE_FEAT in active:
        feat_substring = raw
        notes.append("FEAT → substring on feature_uuid (client)")
    if SCOPE_PRIMITIVE in active:
        engine_queries.append(f"primitive:{raw}")
        notes.append("Primitive → primitive:<token>")
    if SCOPE_OPERATOR in active:
        engine_queries.append(f"operator:{raw}")
        notes.append("Operator → operator:<token>")
    if SCOPE_ONTOLOGY in active:
        engine_queries.append(f"domain:{raw}")
        engine_queries.append(f"signal:{raw}")
        notes.append("Ontology → domain:<token> ∪ signal:<token>")

    needs_match_all = name_substring is not None or feat_substring is not None
    return SearchPlan(
        mode="scoped",
        match_all=needs_match_all,
        engine_queries=tuple(engine_queries),
        name_substring=name_substring,
        feat_substring=feat_substring,
        note="; ".join(notes),
    )


def filter_hits_by_plan(
    items: list[dict[str, Any]],
    plan: SearchPlan,
) -> list[dict[str, Any]]:
    """Apply client-side substring filters from a SearchPlan."""
    out = list(items)
    if plan.name_substring:
        needle = plan.name_substring.lower()
        out = [
            h
            for h in out
            if needle in str(h.get("canonical_name") or "").lower()
        ]
    if plan.feat_substring:
        needle = plan.feat_substring.lower()
        out = [
            h
            for h in out
            if needle in str(h.get("feature_uuid") or "").lower()
        ]
    if plan.research_substring:
        needle = plan.research_substring.lower()
        out = [
            h
            for h in out
            if needle in str(h.get("research_uuid") or "").lower()
        ]
    return out


def merge_hit_lists(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe search hits by research_uuid, stable order."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for items in lists:
        for hit in items:
            key = str(hit.get("research_uuid") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    return merged


def hit_list_label(item: dict[str, Any]) -> str:
    name = item.get("canonical_name") or "?"
    status = item.get("research_status") or ABSENT
    return f"{name}  [{status}]"


# Results grid column keys (Treeview)
HIT_GRID_COLUMNS: tuple[str, ...] = (
    "canonical_name",
    "feature_uuid",
    "research_status",
    "domain",
    "primary_operator",
    "primary_primitive",
    "compiler_version",
    "ontology_version",
)

HIT_GRID_HEADINGS: dict[str, str] = {
    "canonical_name": "Feature Name",
    "feature_uuid": "FEAT ID",
    "research_status": "Research Status",
    "domain": "Domain",
    "primary_operator": "Primary Operator",
    "primary_primitive": "Primary Primitive",
    "compiler_version": "Compiler Version",
    "ontology_version": "Ontology Version",
}


def hit_grid_values(item: dict[str, Any]) -> tuple[str, ...]:
    """Tuple of display strings aligned with HIT_GRID_COLUMNS."""
    return tuple(display_value(item.get(k)) for k in HIT_GRID_COLUMNS)


def _section_present(payload: dict[str, Any], key: str) -> bool:
    present = payload.get("sections_present") or {}
    if key in present:
        return bool(present[key])
    return payload.get(key) is not None


def feature_display_name(payload: dict[str, Any]) -> str:
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    for key in ("display_name", "canonical_name"):
        if identity and identity.get(key):
            return str(identity[key])
    if payload.get("canonical_name"):
        return str(payload["canonical_name"])
    if payload.get("feature_uuid"):
        return str(payload["feature_uuid"])
    return "?"


def architecture_strip(payload: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """
    FEAT → TR → ONT → FRR as (label, id_or_absent, present).
    """
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
    compiler = payload.get("compiler") if isinstance(payload.get("compiler"), dict) else {}
    ontology = payload.get("ontology") if isinstance(payload.get("ontology"), dict) else {}
    refs = payload.get("references") if isinstance(payload.get("references"), dict) else {}

    feat = (
        payload.get("feature_uuid")
        or (identity or {}).get("feature_uuid")
        or (research or {}).get("feature_uuid")
    )
    tr = (
        (research or {}).get("transformation_uuid")
        or (compiler or {}).get("transformation_uuid")
        or (refs or {}).get("transformation_uuid")
        or (identity or {}).get("transformation_uuid")
    )
    ont = (
        (research or {}).get("ontology_uuid")
        or (ontology or {}).get("ontology_uuid")
        or (refs or {}).get("ontology_uuid")
    )
    frr = payload.get("research_uuid") or (research or {}).get("research_uuid")

    return [
        ("FEAT", display_value(feat), feat is not None and bool(str(feat).strip())),
        ("TR", display_value(tr), tr is not None and bool(str(tr).strip())),
        ("ONT", display_value(ont), ont is not None and bool(str(ont).strip())),
        ("FRR", display_value(frr), frr is not None and bool(str(frr).strip())),
    ]


def ontology_chip_text(payload: dict[str, Any]) -> str:
    if not _section_present(payload, "ontology"):
        return ABSENT
    ont = payload.get("ontology") or {}
    if not isinstance(ont, dict):
        return ABSENT
    domain = ont.get("domain_display") or ont.get("domain")
    signals = ont.get("signal_type_display") or []
    sig_bits: list[str] = []
    if isinstance(signals, list):
        for s in signals:
            if isinstance(s, dict):
                sig_bits.append(str(s.get("display_name") or s.get("vocabulary_id") or ""))
            elif s:
                sig_bits.append(str(s))
    parts = [p for p in (display_value(domain), ", ".join(x for x in sig_bits if x)) if p and p != ABSENT]
    return " / ".join(parts) if parts else ABSENT


def lineage_summary_text(payload: dict[str, Any]) -> str:
    if not _section_present(payload, "lineage"):
        return ABSENT
    lin = payload.get("lineage") or {}
    if not isinstance(lin, dict):
        return ABSENT
    n = lin.get("parent_count")
    if n is None:
        return ABSENT
    return f"{n} parent{'s' if n != 1 else ''}"


def header_summary_lines(payload: dict[str, Any]) -> list[str]:
    """Top-of-panel feel: Feature : Name + identity chips."""
    name = feature_display_name(payload)
    lines = [f"Feature : {name}"]
    arch = architecture_strip(payload)
    id_bits = [f"{lab}={short_id(val)}" for lab, val, ok in arch if ok]
    research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    status = (research or {}).get("research_status")
    prims = (identity or {}).get("primitive_ids") if identity else None
    prim_txt = ABSENT
    if isinstance(prims, list) and prims:
        prim_txt = ", ".join(str(p) for p in prims[:3])
        if len(prims) > 3:
            prim_txt += f" (+{len(prims) - 3})"
    ops = []
    lin = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    if lin:
        ops = list(lin.get("operator_ancestors") or [])[:2]
    chips = [
        f"Status={display_value(status)}",
        f"Primitive={prim_txt}",
        f"Operator={display_value(ops) if ops else ABSENT}",
        f"Ontology={ontology_chip_text(payload)}",
        f"Research={display_value((research or {}).get('validation_status'))}",
    ]
    if id_bits:
        lines.append(" · ".join(id_bits))
    lines.append(" · ".join(chips))
    return lines


@dataclass
class FieldRow:
    label: str
    value: str
    present: bool = True


def platform_summary_count_rows(summary: dict[str, Any] | None) -> list[FieldRow]:
    counts = (summary or {}).get("counts") if isinstance(summary, dict) else None
    if not isinstance(counts, dict):
        return [FieldRow("Platform counts", ABSENT, False)]
    labels = (
        ("Primitives", "primitives"),
        ("Features", "features"),
        ("Operators", "operators"),
        ("Transformations", "transformations"),
        ("Ontology Records", "ontology_records"),
        ("Research Records", "research_records"),
    )
    rows: list[FieldRow] = []
    for label, key in labels:
        val = counts.get(key)
        present = val is not None
        rows.append(FieldRow(label, display_value(val), present))
    return rows


def platform_summary_version_rows(summary: dict[str, Any] | None) -> list[FieldRow]:
    vers = (summary or {}).get("versions") if isinstance(summary, dict) else None
    if not isinstance(vers, dict):
        return [FieldRow("Platform versions", ABSENT, False)]
    labels = (
        ("Compiler", "compiler_version"),
        ("Grammar Pack", "grammar_pack_version"),
        ("Grammar Version", "grammar_version"),
        ("Ontology Version", "ontology_version"),
    )
    rows: list[FieldRow] = []
    for label, key in labels:
        val = vers.get(key)
        rows.append(FieldRow(label, display_value(val), bool(val)))
    return rows


def overview_fields(payload: dict[str, Any]) -> list[FieldRow]:
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
    compiler = payload.get("compiler") if isinstance(payload.get("compiler"), dict) else {}
    ontology = payload.get("ontology") if isinstance(payload.get("ontology"), dict) else {}
    created = (research or {}).get("created_at") or (identity or {}).get("created_at")
    updated = (research or {}).get("updated_at") or (identity or {}).get("updated_at")
    summary = payload.get("overview_summary")
    return [
        FieldRow("Feature Name", feature_display_name(payload), True),
        FieldRow(
            "FEAT",
            display_value(payload.get("feature_uuid") or (identity or {}).get("feature_uuid")),
            bool(payload.get("feature_uuid") or (identity or {}).get("feature_uuid")),
        ),
        FieldRow(
            "FRR",
            display_value(payload.get("research_uuid") or (research or {}).get("research_uuid")),
            bool(payload.get("research_uuid")),
        ),
        FieldRow(
            "TR",
            display_value(
                (research or {}).get("transformation_uuid")
                or (compiler or {}).get("transformation_uuid")
            ),
            bool(
                (research or {}).get("transformation_uuid")
                or (compiler or {}).get("transformation_uuid")
            ),
        ),
        FieldRow(
            "Status",
            display_value((research or {}).get("research_status")),
            (research or {}).get("research_status") is not None,
        ),
        FieldRow(
            "Compiler version",
            display_value(
                (research or {}).get("compiler_version")
                or (compiler or {}).get("compiler_version")
            ),
            bool(
                (research or {}).get("compiler_version")
                or (compiler or {}).get("compiler_version")
            ),
        ),
        FieldRow(
            "Grammar version",
            display_value(
                (research or {}).get("grammar_version")
                or (compiler or {}).get("grammar_version")
            ),
            bool(
                (research or {}).get("grammar_version")
                or (compiler or {}).get("grammar_version")
            ),
        ),
        FieldRow(
            "Ontology version",
            display_value((ontology or {}).get("ontology_version")),
            bool((ontology or {}).get("ontology_version")),
        ),
        FieldRow(
            "Lineage version",
            display_value((research or {}).get("lineage_version")),
            bool((research or {}).get("lineage_version")),
        ),
        FieldRow(
            "Ontology summary",
            ontology_chip_text(payload),
            _section_present(payload, "ontology"),
        ),
        FieldRow(
            "Lineage summary",
            lineage_summary_text(payload),
            _section_present(payload, "lineage"),
        ),
        FieldRow("Created", display_value(created), bool(created)),
        FieldRow("Updated", display_value(updated), bool(updated)),
        FieldRow(
            "Summary",
            display_value(summary),
            bool(summary),
        ),
    ]


def identity_fields(payload: dict[str, Any]) -> list[FieldRow]:
    if not _section_present(payload, "identity"):
        return [FieldRow("Identity", ABSENT, False)]
    ident = payload.get("identity") or {}
    if not isinstance(ident, dict):
        return [FieldRow("Identity", ABSENT, False)]
    keys = (
        ("UUID (FEAT)", "feature_uuid"),
        ("Canonical name", "canonical_name"),
        ("Display name", "display_name"),
        ("Definition version", "definition_version"),
        ("Implementation version", "implementation_version"),
        ("Checksum (definition_hash)", "definition_hash"),
        ("Research state", "research_state"),
        ("Controller owner", "controller_owner"),
        ("Created by", "created_by"),
        ("Transformation", "transformation_uuid"),
        ("Warmup periods", "warmup_periods"),
        ("Gap policy", "gap_policy"),
        ("Memory model", "memory_model"),
        ("Primitive dependencies", "primitive_ids"),
        ("Description", "description"),
        ("Created", "created_at"),
        ("Updated", "updated_at"),
    )
    rows: list[FieldRow] = []
    for label, key in keys:
        val = ident.get(key)
        present = val is not None and not (isinstance(val, str) and not val.strip())
        if isinstance(val, list):
            present = bool(val)
        rows.append(FieldRow(label, display_value(val), present))
    return rows


def ontology_fields(payload: dict[str, Any]) -> list[FieldRow]:
    if not _section_present(payload, "ontology"):
        return [FieldRow("Ontology", ABSENT, False)]
    ont = payload.get("ontology") or {}
    if not isinstance(ont, dict):
        return [FieldRow("Ontology", ABSENT, False)]

    def _math_family() -> str:
        fam = ont.get("mathematical_family") or []
        if not fam:
            return ABSENT
        return ", ".join(str(x) for x in fam)

    def _signals() -> str:
        disp = ont.get("signal_type_display") or []
        if isinstance(disp, list) and disp:
            bits = []
            for s in disp:
                if isinstance(s, dict):
                    bits.append(str(s.get("display_name") or s.get("vocabulary_id") or ""))
                else:
                    bits.append(str(s))
            return ", ".join(b for b in bits if b) or ABSENT
        return display_value(ont.get("signal_type"))

    pairs = (
        ("Ontology UUID", ont.get("ontology_uuid")),
        ("Domain", ont.get("domain_display") or ont.get("domain")),
        ("Signal", _signals()),
        ("Math family", _math_family()),
        ("Output", ont.get("output_type")),
        ("Frequency", ont.get("frequency")),
        ("Stability", ont.get("stability")),
        ("Horizon", ont.get("horizon")),
        ("Meaning", ont.get("meaning")),
    )
    rows: list[FieldRow] = []
    for label, val in pairs:
        present = val is not None and val != ABSENT and not (isinstance(val, str) and not str(val).strip())
        rows.append(FieldRow(label, display_value(val) if not isinstance(val, str) or val != ABSENT else val, present))
    return rows


def research_fields(payload: dict[str, Any]) -> list[FieldRow]:
    if not _section_present(payload, "research"):
        return [FieldRow("Research", ABSENT, False)]
    res = payload.get("research") or {}
    if not isinstance(res, dict):
        return [FieldRow("Research", ABSENT, False)]
    exp = res.get("experiment_ids")
    exp_count = len(exp) if isinstance(exp, list) else ABSENT
    return [
        FieldRow("FRR", display_value(res.get("research_uuid")), bool(res.get("research_uuid"))),
        FieldRow("Status", display_value(res.get("research_status")), bool(res.get("research_status"))),
        FieldRow(
            "Validated",
            display_value(res.get("validation_status")),
            bool(res.get("validation_status")),
        ),
        FieldRow(
            "Experiments count",
            str(exp_count) if exp_count != ABSENT else ABSENT,
            isinstance(exp, list),
        ),
        FieldRow("Evidence", display_value(res.get("evidence_json")), bool(res.get("evidence_json"))),
        FieldRow("Notes", display_value(res.get("notes")), bool(res.get("notes"))),
        FieldRow("Record source", display_value(res.get("record_source")), bool(res.get("record_source"))),
        FieldRow("Created", display_value(res.get("created_at")), bool(res.get("created_at"))),
        FieldRow("Updated", display_value(res.get("updated_at")), bool(res.get("updated_at"))),
    ]


def references_fields(payload: dict[str, Any]) -> list[FieldRow]:
    """Phase 1: experiment ids from research; models/datasets honestly empty."""
    research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
    refs = payload.get("references") if isinstance(payload.get("references"), dict) else {}

    models = (refs or {}).get("models") if isinstance(refs, dict) else None
    datasets = (refs or {}).get("datasets") if isinstance(refs, dict) else None
    programs = (refs or {}).get("research_programs") if isinstance(refs, dict) else None
    exp = None
    if isinstance(refs, dict) and isinstance(refs.get("experiments"), list):
        exp = refs["experiments"]
    elif isinstance(research, dict) and isinstance(research.get("experiment_ids"), list):
        exp = research["experiment_ids"]

    def _list_row(label: str, value: Any) -> FieldRow:
        if value is None:
            return FieldRow(label, "No references found.", False)
        if isinstance(value, list):
            if not value:
                return FieldRow(label, "No references found.", False)
            return FieldRow(label, str(len(value)), True)
        return FieldRow(label, display_value(value), True)

    rows = [
        _list_row("Models", models),
        _list_row("Datasets", datasets),
        _list_row("Experiments", exp),
        _list_row("Research Programs", programs),
    ]
    if isinstance(refs, dict) and _section_present(payload, "references"):
        for label, key in (
            ("FRR", "research_uuid"),
            ("FEAT", "feature_uuid"),
            ("TR", "transformation_uuid"),
            ("ONT", "ontology_uuid"),
        ):
            rows.append(
                FieldRow(label, display_value(refs.get(key)), bool(refs.get(key)))
            )
    return rows


def references_are_empty(payload: dict[str, Any]) -> bool:
    """True when Phase 1 linkage lists are all empty / absent."""
    refs = payload.get("references") if isinstance(payload.get("references"), dict) else {}
    research = payload.get("research") if isinstance(payload.get("research"), dict) else {}
    for key in ("models", "datasets", "research_programs"):
        val = (refs or {}).get(key)
        if isinstance(val, list) and val:
            return False
    exp = (refs or {}).get("experiments")
    if isinstance(exp, list) and exp:
        return False
    exp2 = (research or {}).get("experiment_ids")
    if isinstance(exp2, list) and exp2:
        return False
    return True


def compiler_stack_summary(payload: dict[str, Any]) -> list[FieldRow]:
    """Transformation → Grammar → AST → Manifest summaries."""
    present = payload.get("sections_present") or {}
    compiler = payload.get("compiler") if isinstance(payload.get("compiler"), dict) else None
    ast = payload.get("ast") if isinstance(payload.get("ast"), dict) else None
    rows: list[FieldRow] = []

    if not present.get("compiler") and not present.get("ast"):
        return [FieldRow("Compiler / AST", ABSENT, False)]

    if compiler:
        rows.extend(
            [
                FieldRow(
                    "Transformation",
                    display_value(compiler.get("transformation_uuid")),
                    bool(compiler.get("transformation_uuid")),
                ),
                FieldRow(
                    "Canonical text",
                    display_value(compiler.get("canonical_text")),
                    bool(compiler.get("canonical_text")),
                ),
                FieldRow(
                    "Expression hash",
                    display_value(compiler.get("expression_hash")),
                    bool(compiler.get("expression_hash")),
                ),
                FieldRow(
                    "Grammar version",
                    display_value(compiler.get("grammar_version")),
                    bool(compiler.get("grammar_version")),
                ),
                FieldRow(
                    "Compiler version",
                    display_value(compiler.get("compiler_version")),
                    bool(compiler.get("compiler_version")),
                ),
                FieldRow(
                    "Operator pack",
                    display_value(compiler.get("operator_pack_version")),
                    bool(compiler.get("operator_pack_version")),
                ),
            ]
        )
        manifest = compiler.get("manifest_summary")
        if isinstance(manifest, dict):
            rows.append(
                FieldRow(
                    "Manifest compilation",
                    display_value(manifest.get("compilation_uuid")),
                    bool(manifest.get("compilation_uuid")),
                )
            )
            rows.append(
                FieldRow(
                    "Manifest ast_hash",
                    display_value(manifest.get("ast_hash")),
                    bool(manifest.get("ast_hash")),
                )
            )
            rows.append(
                FieldRow(
                    "Manifest cache_hit",
                    display_value(manifest.get("cache_hit")),
                    manifest.get("cache_hit") is not None,
                )
            )
        else:
            rows.append(FieldRow("Compiler Manifest", ABSENT, False))
    else:
        rows.append(FieldRow("Transformation / Grammar", ABSENT, False))

    if ast:
        rows.extend(
            [
                FieldRow(
                    "AST root operator",
                    display_value(ast.get("root_operator")),
                    bool(ast.get("root_operator")),
                ),
                FieldRow(
                    "AST node count",
                    display_value(ast.get("node_count")),
                    ast.get("node_count") is not None,
                ),
                FieldRow(
                    "AST hash",
                    display_value(ast.get("ast_hash")),
                    bool(ast.get("ast_hash")),
                ),
                FieldRow(
                    "AST schema",
                    display_value(ast.get("ast_schema_version")),
                    bool(ast.get("ast_schema_version")),
                ),
            ]
        )
    else:
        rows.append(FieldRow("AST summary", ABSENT, False))
    return rows


def compiler_raw_json(payload: dict[str, Any]) -> str:
    blob = {
        "compiler": payload.get("compiler"),
        "ast": payload.get("ast"),
    }
    return json.dumps(blob, indent=2, sort_keys=True)


def lineage_tree_text(payload: dict[str, Any]) -> str:
    """Indented parent → subject → children text graph."""
    if not _section_present(payload, "lineage"):
        return "No lineage edges linked yet"
    lin = payload.get("lineage")
    if not isinstance(lin, dict):
        return "No lineage edges linked yet"

    parents = list(lin.get("sample_parents") or [])
    children = list(lin.get("sample_children") or [])
    ancestors = list(lin.get("primitive_ancestors") or []) + list(
        lin.get("operator_ancestors") or []
    )
    prim_inputs = list(lin.get("primitive_inputs") or [])
    ops_used = list(lin.get("operators_used") or [])
    chain = list(lin.get("transformation_chain") or [])
    subject = payload.get("feature_uuid") or payload.get("canonical_name") or "feature"

    if (
        not parents
        and not children
        and not ancestors
        and not prim_inputs
        and not ops_used
    ):
        if (lin.get("parent_count") or 0) == 0 and (lin.get("child_count") or 0) == 0:
            return "No lineage edges linked yet"

    lines: list[str] = []
    if prim_inputs:
        lines.append("Primitive inputs:")
        for p in prim_inputs[:12]:
            lines.append(f"  • {p}")
    if ops_used:
        lines.append("Operators used:")
        for o in ops_used[:12]:
            lines.append(f"  • {o}")
    if ancestors and not (prim_inputs or ops_used):
        lines.append("Ancestors (sample):")
        for a in ancestors[:12]:
            lines.append(f"  ↑ {a}")
    if parents:
        lines.append("Parents:")
        for p in parents:
            lines.append(f"  {p}")
            lines.append("    ↓")
    lines.append(f"  ● {subject}")
    if children:
        for c in children:
            lines.append("    ↓")
            lines.append(f"      {c}")
    if chain:
        lines.append("")
        lines.append("Transformation chain:")
        lines.append("  " + " → ".join(str(x) for x in chain))
    lines.append("")
    lines.append(
        f"Counts: parents={lin.get('parent_count', 0)}  "
        f"children={lin.get('child_count', 0)}  "
        f"ancestors={lin.get('ancestor_count', 0)}"
    )
    return "\n".join(lines)


def ontology_chip_labels(payload: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(category, chip_text, present) for Ontology tab chips."""
    rows = ontology_fields(payload)
    wanted = {
        "Domain",
        "Signal",
        "Math family",
        "Output",
        "Horizon",
        "Frequency",
        "Stability",
    }
    return [(r.label, r.value, r.present) for r in rows if r.label in wanted]
