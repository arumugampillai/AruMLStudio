"""Plain-text formatters for feature registry views (Tk)."""

from __future__ import annotations

import json
from typing import Any


def format_registry_meta(catalog: dict[str, Any]) -> str:
    ver = catalog.get("registry_version") or "1.0"
    count = catalog.get("feature_count") or len(catalog.get("features") or [])
    disabled = int((catalog.get("stats") or {}).get("disabled") or 0)
    if disabled:
        return f"v{ver} · {count} features · {disabled} retired"
    return f"v{ver} · {count} features"


def format_expected_range(f: dict[str, Any]) -> str:
    val = f.get("expected_range")
    if val is None or val == "":
        return "—"
    if isinstance(val, list) and len(val) == 2:
        return f"{val[0]} to {val[1]}"
    return str(f.get("expected_range_label") or val)


def format_date(value: Any) -> str:
    if not value:
        return "—"
    return str(value).replace("T", " ")[:10]


def format_dep_tree(nodes: list[dict[str, Any]], indent: int = 0) -> str:
    if not nodes:
        return ""
    lines: list[str] = []
    for node in nodes:
        pad = "  " * indent
        prefix = "↓ " if indent else ""
        lines.append(f"{pad}{prefix}{node.get('label') or node.get('name') or '—'}")
        child = format_dep_tree(node.get("children") or [], indent + 1)
        if child:
            lines.append(child)
    return "\n".join(lines)


def format_feature_detail(f: dict[str, Any], catalog: dict[str, Any] | None = None) -> str:
    if not f:
        return "Select a feature to view details, dependencies, and usage."

    lines: list[str] = []
    title = f.get("display_name") or f.get("name") or "—"
    lines.append(title)
    lines.append("=" * min(72, max(len(title), 40)))
    lines.append(
        f"{f.get('name', '—')} · {f.get('primary_domain_label') or f.get('domain') or f.get('group', '—')} · "
        f"{f.get('ownership') or ''} · {f.get('category') or ''}".rstrip(" ·")
    )
    lines.append("")

    def section(title: str) -> None:
        lines.append(title)
        lines.append("-" * 40)

    section("Identity")
    lines.append(f"  Feature ID: {f.get('feature_id') or '—'}")
    prev = f.get("previous_names") or []
    if prev:
        lines.append(f"  Previous names: {', '.join(prev)}")
    lines.append(f"  Primary domain: {f.get('primary_domain_label') or '—'}")
    lines.append(f"  Ownership: {f.get('ownership') or '—'}")
    lines.append(f"  Domain data type: {f.get('domain_data_type') or f.get('expected_data_type') or '—'}")
    lines.append(f"  Created: {format_date(f.get('created_at'))}")
    lines.append(f"  Created by: {f.get('created_by') or 'System'}")
    lines.append(f"  Last modified: {format_date(f.get('updated_at') or f.get('last_updated'))}")
    lines.append(f"  Version: {f.get('feature_version') or '1.0'}")
    lines.append(f"  State: {f.get('implementation_label') or f.get('implementation_status') or '—'}")
    if f.get("registry_active") is False:
        lines.append(f"  Registry active: No (home group: {f.get('home_group_filter') or f.get('home_group') or '—'})")
    else:
        lines.append("  Registry active: Yes")
    lines.append(f"  Owner: {f.get('owner') or '—'}")
    lines.append("")

    section("Auto Feature Generation")
    lines.append(f"  Lag: {'Yes' if f.get('can_apply_lag') else 'No'}")
    lines.append(f"  Difference: {'Yes' if f.get('can_apply_difference') else 'No'}")
    lines.append(f"  Return: {'Yes' if f.get('can_apply_return') else 'No'}")
    lines.append(f"  Rolling: {'Yes' if f.get('can_apply_rolling') else 'No'}")
    lines.append(f"  Z-score: {'Yes' if f.get('can_apply_zscore') else 'No'}")
    lines.append(f"  Interaction: {'Yes' if f.get('can_participate_in_interaction') else 'No'}")
    lines.append("")

    if f.get("description"):
        section("Description")
        lines.append(f"  {f['description']}")
        lines.append("")

    if f.get("why_needed"):
        section("Why this feature?")
        lines.append(f"  {f['why_needed']}")
        lines.append("")

    section("Formula")
    lines.append(f"  {f.get('formula') or '—'}")
    lines.append("")

    section("Data type")
    lines.append(f"  Expected type: {f.get('expected_data_type') or 'float'}")
    lines.append(f"  Expected range: {format_expected_range(f)}")
    lines.append("")

    section("Dependencies")
    deps_resolved = f.get("dependencies_resolved") or []
    if deps_resolved:
        for d in deps_resolved:
            if not (d.get("name") or d.get("feature_id")):
                continue
            fid = d.get("feature_id") or d.get("ref") or ""
            name = d.get("display_name") or d.get("name") or fid
            lines.append(f"  • {fid} — {name}" if fid else f"  • {name}")
    else:
        deps = [d for d in (f.get("dependencies") or []) if d not in ("timestamp", "token", "symbol")]
        lines.append(f"  {', '.join(deps) if deps else '—'}")
    tree = format_dep_tree(f.get("dependency_tree") or [])
    if tree:
        lines.append("")
        lines.append(tree)
    lines.append("")

    from . import feature_policy_format as pol_fmt

    pol_section = pol_fmt.format_feature_policy_section(f)
    if pol_section.strip():
        lines.append(pol_section.rstrip())
        lines.append("")

    surfaces = catalog.get("pipeline_surfaces") if catalog else []
    used_in = {u.get("id") for u in (f.get("used_in") or [])}
    if surfaces:
        section("Feature pipeline")
        for s in surfaces:
            mark = "✓" if s.get("id") in used_in else "·"
            lines.append(f"  {mark} {s.get('label') or s.get('id')}")
        lines.append("")

    section("Implementation")
    if f.get("implementation_module"):
        lines.append(f"  {f.get('implementation_module')} → {f.get('implementation_function') or ''}")
    elif f.get("source") == "planned":
        lines.append("  Backlog only — not implemented in code")
    else:
        lines.append("  —")
    if f.get("priority"):
        lines.append(f"  Priority: {f.get('priority')}")
    if f.get("developer_notes"):
        lines.append(f"  Notes: {f.get('developer_notes')}")
    lines.append("")

    imp = f.get("importance") or {}
    section("Feature importance")
    if imp.get("average_pct") is not None:
        lines.append(f"  Average: {imp.get('average_pct')}%")
        if imp.get("best_pct") is not None:
            lines.append(f"  Best model: {imp.get('best_pct')}% ({imp.get('best_model') or '—'})")
    else:
        lines.append("  No training importance data yet.")
    lines.append("")

    models = f.get("models_using") or []
    section("Models using it")
    if models:
        for m in models[:30]:
            lines.append(f"  • {m}")
        if len(models) > 30:
            lines.append(f"  … +{len(models) - 30} more")
    else:
        lines.append("  Not used in any trained model yet.")

    return "\n".join(lines)


def format_delete_preview(report: dict[str, Any]) -> str:
    lines = [
        f"Delete preview — {report.get('name') or report.get('feature_id') or '—'}",
        "=" * 60,
        f"Status: {report.get('status_label') or ('Safe' if report.get('can_delete') else 'Blocked')}",
        f"Can delete: {report.get('can_delete')}",
        "",
        "Currently used by:",
    ]
    used = report.get("currently_used_by") or {}
    for key, label in (
        ("datasets", "Datasets"),
        ("models", "Models"),
        ("registry_dependencies", "Dependencies"),
        ("feature_groups", "Feature groups"),
        ("dataset_schemas", "Dataset schemas"),
    ):
        block = used.get(key) or {}
        lines.append(f"  {label}: {block.get('count', 0)}")
    blockers = report.get("blockers") or []
    if blockers:
        lines.append("")
        lines.append("Blockers:")
        for b in blockers:
            lines.append(f"  • {b}")
    return "\n".join(lines)


def format_import_preview(result: dict[str, Any]) -> str:
    lines = [
        "Import preview",
        "=" * 60,
        f"Import type: {result.get('import_type') or '—'}",
        f"Valid: {result.get('valid')}",
        f"Can apply: {result.get('can_apply')}",
    ]
    summary = result.get("summary") or {}
    if summary:
        lines.append("")
        for k, v in summary.items():
            lines.append(f"  {k}: {v}")
    conflicts = result.get("conflicts") or []
    if conflicts:
        lines.append("")
        lines.append(f"Conflicts ({len(conflicts)}):")
        for c in conflicts[:40]:
            if isinstance(c, dict):
                lines.append(f"  • {c.get('name') or c.get('feature')}: {c.get('reason') or c.get('status')}")
            else:
                lines.append(f"  • {c}")
    errors = result.get("errors") or []
    if errors:
        lines.append("")
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  • {e}")
    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings[:20]:
            lines.append(f"  • {w}")
    return "\n".join(lines)


def format_parity_audit(data: dict[str, Any]) -> str:
    cmp = data.get("comparison") or {}
    probe = data.get("probe") or {}
    lines = [
        f"Pipeline parity — {cmp.get('label') or cmp.get('status') or '—'}",
        "=" * 60,
        f"Parity: {cmp.get('parity_pct')}%",
        f"Matched: {cmp.get('match_count')}/{cmp.get('feature_count')}",
        f"Mismatches: {cmp.get('mismatch_count', 0)}",
        "",
        f"Probe: {probe.get('trading_day') or '—'} · ts={probe.get('timestamp')} · token={probe.get('token')}",
        f"Step: {probe.get('step_sec')}s",
        "",
        data.get("golden_rule") or "",
        "",
    ]
    mismatches = cmp.get("mismatches") or []
    if mismatches:
        lines.append("Mismatches:")
        for m in mismatches[:50]:
            lines.append(
                f"  {m.get('feature')}: dataset={m.get('value_dataset')} "
                f"replay={m.get('value_replay')} live={m.get('value_live')}"
            )
    else:
        lines.append("All features matched across dataset, replay, and live.")
    return "\n".join(lines)


def format_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)
