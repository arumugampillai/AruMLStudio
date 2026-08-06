"""Validation and dependency-graph generation for FEATURE_CLASSIFICATION.md."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
    FEATURE_REGISTRY_VERSION,
    FUTURE_CONTROLLER_IDS,
    all_registry_controller_ids,
    detect_dependency_cycles,
)

# Bump when classification_validate.py validation rules or markdown layout change.
GENERATOR_VERSION = 3


def _parse_source_controllers(label: str) -> list[str]:
    if not label or label.strip() in ("—", "-"):
        return []
    return [p.strip() for p in label.split(",") if p.strip() and p.strip() not in ("—", "-")]


def collect_used_controller_ids(rows: list[Any]) -> set[str]:
    used: set[str] = set()
    for row in rows:
        if row.controller_owner not in ("—", ""):
            used.add(row.controller_owner)
        for cid in _parse_source_controllers(row.source_controllers):
            used.add(cid)
    return used


def validate_registry_coverage(rows: list[Any]) -> None:
    """Every registry controller must be referenced; every used ID must exist in registry.

    Controllers that only produce Pipeline Owned / Interaction products (empty
    ``CONTROLLER_FEATURES`` or solely pipeline-owned names) are exempt from the
    orphan check — classification covers canonical registry features only.
    """
    from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned

    registry = all_registry_controller_ids()
    used = collect_used_controller_ids(rows)

    orphans = registry - used
    exempt = set()
    for cid in orphans:
        feats = CONTROLLER_FEATURES.get(cid) or []
        if not feats or all(is_pipeline_owned(f) for f in feats):
            exempt.add(cid)
    orphans -= exempt
    if orphans:
        raise RuntimeError(
            "Controllers in registry but not referenced by any feature: "
            + ", ".join(sorted(orphans)),
        )

    unknown = used - registry - FUTURE_CONTROLLER_IDS
    if unknown:
        raise RuntimeError(
            "Features reference unknown controller IDs (check spelling): "
            + ", ".join(sorted(unknown)),
        )

    for cid, deps in ((c, CONTROLLER_REGISTRY[c].source_controllers) for c in registry):
        for dep in deps:
            if dep not in registry:
                raise RuntimeError(f"{cid}: source controller {dep!r} not in registry")


def validate_warmup_consistency(rows: list[Any]) -> None:
    """Classification warmup must match controller_registry warmup specs."""
    for row in rows:
        cid = row.controller_owner
        if cid in ("—", ""):
            continue
        spec = CONTROLLER_REGISTRY.get(cid)
        if spec is None:
            continue
        if row.warmup_type != spec.warmup_type:
            raise RuntimeError(
                f"{row.feature} ({cid}): warmup_type {row.warmup_type!r} != registry {spec.warmup_type!r}",
            )
        if row.warmup_value != spec.warmup_value:
            raise RuntimeError(
                f"{row.feature} ({cid}): warmup_value {row.warmup_value!r} != registry {spec.warmup_value!r}",
            )


def validate_build_phase_order(rows: list[Any]) -> None:
    """Every source controller must exist and be same-or-earlier phase than consumer."""
    for cid, spec in CONTROLLER_REGISTRY.items():
        for dep in spec.source_controllers:
            dep_phase = CONTROLLER_REGISTRY[dep].phase
            if dep_phase > spec.phase:
                raise RuntimeError(
                    f"Registry {cid} (phase {spec.phase}) depends on {dep} (phase {dep_phase}) "
                    f"— source must be same or earlier build phase",
                )

    for row in rows:
        if row.phase in ("n/a", "—", ""):
            continue
        try:
            consumer_phase = int(row.phase)
        except (TypeError, ValueError):
            continue
        owner = row.controller_owner
        if owner not in ("—", "") and owner in CONTROLLER_REGISTRY:
            consumer_phase = max(consumer_phase, CONTROLLER_REGISTRY[owner].phase)

        for src in _parse_source_controllers(row.source_controllers):
            if src not in CONTROLLER_REGISTRY:
                raise RuntimeError(
                    f"{row.feature}: source controller {src!r} not in registry",
                )
            src_phase = CONTROLLER_REGISTRY[src].phase
            if src_phase > consumer_phase:
                raise RuntimeError(
                    f"{row.feature}: source {src} (phase {src_phase}) consumed in phase "
                    f"{consumer_phase} — invalid build order (e.g. Phase 1 composite "
                    f"depending on Phase 2 controller)",
                )


def generated_document_header() -> str:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"FEATURE_REGISTRY_VERSION = {FEATURE_REGISTRY_VERSION}\n"
        f"GENERATOR_VERSION = {GENERATOR_VERSION}\n"
        f"Generated at = {ts}"
    )


def validate_governance(
    rows: list[Any],
    names: list[str],
    feature_to_controller: dict[str, str],
) -> None:
    from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned

    if len(names) != 206:
        raise RuntimeError(f"Expected 206 features, got {len(names)}")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"Duplicate features in registry: {dupes}")

    for row in rows:
        if row.memory_model == "Controller" and row.controller_owner in ("—", ""):
            raise RuntimeError(f"{row.feature}: Controller memory model requires exactly one owner")

    mapped = set(feature_to_controller)
    registry = set(names)
    # CONTROLLER_FEATURES may still list Pipeline Owned products for readiness;
    # classification covers canonical registry names only.
    unmapped = sorted(f for f in (mapped - registry) if not is_pipeline_owned(f))
    if unmapped:
        raise RuntimeError(f"Controller map references unknown features: {unmapped}")

    for feat, ctrl in feature_to_controller.items():
        if feat not in registry:
            continue
        row = next(r for r in rows if r.feature == feat)
        if row.controller_owner != ctrl:
            raise RuntimeError(
                f"{feat}: owner mismatch — map={ctrl}, classified={row.controller_owner}",
            )

    validate_registry_coverage(rows)
    validate_warmup_consistency(rows)
    validate_build_phase_order(rows)

    cycles = detect_dependency_cycles()
    if cycles:
        raise RuntimeError(f"Controller dependency cycles detected: {cycles}")


def render_dependency_graph_markdown() -> str:
    """Mermaid graphs grouped by phase — parent → child emit order."""
    lines = [
        "# Controller Dependency Graph",
        "",
        "```",
        generated_document_header(),
        "```",
        "",
        "Auto-generated from `controller_registry.py`. Regenerate via "
        "`scripts/generate_feature_classification.py`.",
        "",
        "Edges: source controller → dependent controller/emitter (build update order).",
        "Phase rule: every dependency must be same or **earlier** build phase than consumer.",
        "",
    ]

    def _section(title: str, controller_ids: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph TD")
        seen_edges: set[tuple[str, str]] = set()
        for cid in controller_ids:
            spec = CONTROLLER_REGISTRY[cid]
            safe = cid.replace(".", "_")
            lines.append(f'  {safe}["{cid}"]')
            for dep in spec.source_controllers:
                dep_safe = dep.replace(".", "_")
                edge = (dep, cid)
                if edge not in seen_edges:
                    lines.append(f"  {dep_safe} --> {safe}")
                    seen_edges.add(edge)
        lines.append("```")
        lines.append("")

    phase1 = [c for c, s in CONTROLLER_REGISTRY.items() if s.phase == 1]
    phase2 = [c for c, s in CONTROLLER_REGISTRY.items() if s.phase == 2]
    phase3 = [c for c, s in CONTROLLER_REGISTRY.items() if s.phase == 3]

    _section("Phase 1 — token LTP EMA", sorted(phase1))
    _section("Phase 2 — token rolling", sorted(phase2))
    _section("Phase 3 — spot / composites", sorted(phase3))

    lines.append("## Feature coverage")
    lines.append("")
    lines.append("| Controller | Features |")
    lines.append("|------------|----------|")
    for cid in sorted(CONTROLLER_REGISTRY):
        feats = CONTROLLER_FEATURES.get(cid, [])
        lines.append(f"| `{cid}` | {len(feats)} |")
    lines.append("")
    return "\n".join(lines)
