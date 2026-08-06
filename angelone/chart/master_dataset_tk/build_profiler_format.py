"""Format build profiler report for the master dataset progress panel."""

from __future__ import annotations

from typing import Any


def _fmt_sec(sec: float | None) -> str:
    if sec is None:
        return "—"
    s = float(sec)
    if s < 60:
        return f"{s:.2f}s"
    return f"{s / 60:.2f}m"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{pct:.1f}%"


def _fmt_rps(rps: float | None) -> str:
    if rps is None:
        return "—"
    return f"{rps:,.0f}"


def _table_header() -> str:
    return (
        f"{'Name':<42} {'Total':>8} {'Calls':>10} {'Avg ms':>9} {'Max ms':>9} "
        f"{'Rows':>10} {'Rows/s':>10} {'%Build':>7}\n"
        + "-" * 108 + "\n"
    )


def _table_rows(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for e in entries:
        name = str(e.get("name") or "")[:42]
        lines.append(
            f"{name:<42} "
            f"{_fmt_sec(e.get('total_sec')):>8} "
            f"{int(e.get('call_count') or 0):>10,} "
            f"{float(e.get('avg_ms') or 0):>9.3f} "
            f"{float(e.get('max_ms') or 0):>9.3f} "
            f"{int(e.get('rows') or 0):>10,} "
            f"{_fmt_rps(e.get('rows_per_sec')):>10} "
            f"{_fmt_pct(e.get('pct')):>7}"
        )
    return "\n".join(lines)


def format_build_profiler_report(report: dict[str, Any] | None) -> str:
    if not report:
        return "No profiler data — enable Build Profiler in config and run a build."

    total = float(report.get("total_build_sec") or 0)
    rows = int(report.get("total_rows") or 0)
    rps = report.get("build_rows_per_sec")
    header = (
        f"Build total: {_fmt_sec(total)}"
        f"  ·  {rows:,} rows"
        f"  ·  {_fmt_rps(rps)} rows/sec overall\n\n"
    )

    sections: list[str] = []

    # Compact Measure-phase summary at the top of the Profiler tab.
    stages = list(report.get("stages") or [])
    if stages and total > 0:
        sections.append("=== Build Profiling (stages) ===\n")
        sections.append(
            f"{'Stage':<34} {'Time':>10} {'% Total':>9}\n" + "-" * 55 + "\n"
        )
        for e in stages:
            name = str(e.get("name") or "")
            if name.startswith("stage."):
                name = name[len("stage.") :]
            name = name.replace("_", " ").replace(".", " / ")[:34]
            sections.append(
                f"{name:<34} {_fmt_sec(e.get('total_sec')):>10} {_fmt_pct(e.get('pct')):>9}"
            )
        sections.append("")

    functions = list(report.get("functions") or [])
    if functions and total > 0:
        kernel = next(
            (
                e
                for e in stages
                if str(e.get("name") or "") == "stage.feature_generation"
            ),
            None,
        )
        kernel_sec = float((kernel or {}).get("total_sec") or 0) or total
        sections.append("=== Feature Kernel (top functions) ===\n")
        sections.append(
            f"{'Bucket':<34} {'Time':>10} {'% Kernel':>9}\n" + "-" * 55 + "\n"
        )
        for e in functions[:12]:
            name = str(e.get("name") or "")
            if name.startswith("function."):
                name = name[len("function.") :]
            name = name.replace("_", " ")[:34]
            sec = float(e.get("total_sec") or 0)
            pct = (100.0 * sec / kernel_sec) if kernel_sec > 0 else 0.0
            sections.append(
                f"{name:<34} {_fmt_sec(sec):>10} {_fmt_pct(pct):>9}"
            )
        sections.append("")

    def add_section(title: str, key: str) -> None:
        entries = list(report.get(key) or [])
        if not entries:
            return
        sections.append(f"=== {title} (slowest → fastest) ===\n")
        sections.append(_table_header())
        sections.append(_table_rows(entries))
        sections.append("")

    add_section("Stage timeline", "stages")
    add_section("Function timeline", "functions")
    add_section("Controller timeline", "controllers")
    add_section("Feature group timeline", "feature_groups")
    add_section("Feature family rollup", "feature_families")

    ranked = list(report.get("ranked") or [])[:25]
    if ranked:
        sections.append("=== Top 25 overall (slowest → fastest) ===\n")
        sections.append(_table_header())
        sections.append(_table_rows(ranked))
        sections.append("")

    tree = _feature_generation_tree(report)
    if tree:
        sections.append("=== Feature generation breakdown ===\n")
        sections.append(tree)
        sections.append("")

    spot_update = report.get("spot_controllers_update")
    if isinstance(spot_update, dict):
        sections.append(_format_spot_controllers_update(spot_update))

    return header + "\n".join(sections).rstrip()


def _format_spot_controllers_update(stats: dict[str, Any]) -> str:
    lines = [
        "=== SpotControllers.update() path breakdown ===",
        (
            f"Total calls: {int(stats.get('total_calls') or 0):,}"
            f"  ·  full updates: {int(stats.get('full_updates_executed') or 0):,}"
            f"  ·  duplicate-ts early returns: {int(stats.get('early_returns_duplicate_timestamp') or 0):,}"
            f"  ·  invalid-spot early returns: {int(stats.get('early_returns_invalid_spot') or 0):,}"
        ),
        (
            f"Time in early-return paths: {_fmt_sec(stats.get('total_time_early_return_sec'))}"
            f"  ({float(stats.get('pct_time_early_returns') or 0):.1f}% of tracked update time)"
            f"  ·  avg {_float_ms(stats.get('avg_ms_early_return'))} ms/call"
        ),
        (
            f"Time in full-update path: {_fmt_sec(stats.get('total_time_full_update_sec'))}"
            f"  ({float(stats.get('pct_time_full_updates') or 0):.1f}% of tracked update time)"
            f"  ·  avg {_float_ms(stats.get('avg_ms_full_update'))} ms/call"
        ),
        "",
        f"{'Path':<24} {'Calls':>12} {'Total':>10} {'Avg ms':>10}",
        "-" * 60,
    ]
    for row in stats.get("summary_table") or []:
        lines.append(
            f"{str(row.get('path') or ''):<24} "
            f"{int(row.get('calls') or 0):>12,} "
            f"{_fmt_sec(row.get('total_sec')):>10} "
            f"{_float_ms(row.get('avg_ms')):>10}"
        )
    interpretation = str(stats.get("interpretation") or "").strip()
    if interpretation:
        lines.append("")
        lines.append(f"→ {interpretation}")

    breakdown = stats.get("full_update_breakdown") or []
    if breakdown and int(stats.get("full_updates_executed") or 0) > 0:
        lines.extend([
            "",
            "=== Full update breakdown (% of full-update time) ===",
            f"{'Section':<22} {'Time':>10} {'% Full':>8} {'Calls':>10} {'Avg ms':>10}",
            "-" * 64,
        ])
        for row in breakdown:
            lines.append(
                f"{str(row.get('section') or ''):<22} "
                f"{_fmt_sec(row.get('total_sec')):>10} "
                f"{float(row.get('pct_of_full_update') or 0):>7.1f}% "
                f"{int(row.get('calls') or 0):>10,} "
                f"{_float_ms(row.get('avg_ms')):>10}"
            )

    lines.append("")
    return "\n".join(lines)


def _float_ms(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _feature_generation_tree(report: dict[str, Any]) -> str:
    functions = {e["name"]: e for e in (report.get("functions") or []) if e.get("name")}
    root = functions.get("function.build_feature_raw_for_row")
    if not root:
        return ""

    children_order = [
        "function.extract_timeline_features",
        "function.update_token_ltp_controllers",
        "function.update_token_rv_controllers",
        "function.spot_controllers.update",
        "function.enrich_spot_momentum_registry",
        "function.enrich_dataset_features",
        "function.enrich_with_chain_maps",
        "function.enrich_sharp_momentum",
        "function.enrich_iv_zscore",
        "function.enrich_advanced_composites",
        "function.enrich_spot_hl_ratio",
        "function.enrich_spot_hl_composite",
        "function.enrich_current_to_atm6_flow",
    ]
    labels = {
        "function.extract_timeline_features": "extract_timeline_features",
        "function.update_token_ltp_controllers": "update_token_controllers",
        "function.update_token_rv_controllers": "update_token_rv_controllers",
        "function.spot_controllers.update": "spot_controllers.update",
        "function.enrich_spot_momentum_registry": "enrich_spot_momentum",
        "function.enrich_dataset_features": "enrich_dataset_features",
        "function.enrich_with_chain_maps": "enrich_chain_maps",
        "function.enrich_sharp_momentum": "enrich_sharp_momentum",
        "function.enrich_iv_zscore": "enrich_iv",
        "function.enrich_advanced_composites": "advanced_features",
        "function.enrich_spot_hl_ratio": "enrich_spot_hl_ratio",
        "function.enrich_spot_hl_composite": "enrich_spot_hl_composite",
        "function.enrich_current_to_atm6_flow": "enrich_atm6_flow",
    }

    lines = [
        f"build_feature_raw_for_row .......... {_fmt_sec(root.get('total_sec'))}",
        "│",
    ]
    for i, key in enumerate(children_order):
        entry = functions.get(key)
        if not entry:
            continue
        branch = "└──" if i == len(children_order) - 1 else "├──"
        label = labels.get(key, key.replace("function.", ""))
        lines.append(
            f"{branch} {label:<32} {_fmt_sec(entry.get('total_sec'))}"
            f"  ({int(entry.get('call_count') or 0):,} calls)"
        )
    return "\n".join(lines)
