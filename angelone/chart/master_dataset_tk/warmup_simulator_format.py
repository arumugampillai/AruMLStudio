"""Text formatters for warm-up simulator output."""

from __future__ import annotations

import math
from typing import Any

from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel
from chain_replay_ml.feature_policy.warmup_simulator import WarmupSimulationResult


def _mark(ready: bool) -> str:
    return "✓" if ready else "✗"


def format_transition_section(result: WarmupSimulationResult) -> str:
    trans = result.transition or {}
    last_nr = trans.get("last_not_ready")
    first_r = trans.get("first_ready")
    if not last_nr and not first_r:
        if result.ready_at_sample:
            return (
                "Ready Transition\n"
                "-" * 40 + "\n"
                f"  Ready at sample {result.ready_at_sample}\n"
            )
        return ""
    lines = ["Ready Transition", "-" * 40, ""]
    if last_nr:
        lines.extend([
            "  Last NOT READY",
            f"  Sample {last_nr.get('samples', '—')}",
            f"  {last_nr.get('time', '—')}",
            "",
        ])
    if first_r:
        lines.extend([
            "  First READY",
            f"  Sample {first_r.get('samples', '—')}",
            f"  {first_r.get('time', '—')}",
            "",
        ])
    return "\n".join(lines).rstrip()


def format_output_values(result: WarmupSimulationResult) -> str:
    samples = result.output_samples or []
    if not samples:
        return ""
    ctrl = result.controller_label or "Controller"
    lines = [
        "Output Values",
        "-" * 56,
        f"  {'Time':<10}  {'Sample':>6}  {ctrl[:10]:<10}  Feature  Output",
    ]
    for row in samples:
        lines.append(
            f"  {row.get('time', '—'):<10}  {row.get('samples', 0):>6}  "
            f"{_mark(bool(row.get('ctrl_ready'))):>10}  "
            f"{_mark(bool(row.get('feature_ready'))):>7}  "
            f"{row.get('output_display', 'NULL')}",
        )
    lines.extend([
        "",
        "  Rule: until the controller is ready the dataset contains NULL.",
    ])
    return "\n".join(lines)


def format_dependency_timeline(result: WarmupSimulationResult) -> str:
    rows = result.dependency_timeline or []
    labels = result.dependency_labels or []
    if not rows or not labels:
        return ""
    col_w = 8
    header = f"  {'Time':<10}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in labels) + f"{'Feature':>10}"
    lines = ["Dependency Timeline", "-" * max(56, len(header)), header]
    for row in rows:
        deps = row.get("deps") or {}
        dep_marks = "".join(
            f"{_mark(bool(deps.get(lbl, True))):>{col_w}}" for lbl in labels
        )
        lines.append(
            f"  {row.get('time', '—'):<10}{dep_marks}{_mark(bool(row.get('feature_ready'))):>10}",
        )
    return "\n".join(lines)


def format_timeline_table(result: WarmupSimulationResult) -> str:
    ctrl = result.controller_label or "Controller"
    lines = [
        "Timeline",
        "-" * 56,
        f"  {'Time':<10}  {'Samples':>7}  {ctrl[:14]:<14}  Feature Ready",
    ]
    for row in result.timeline:
        lines.append(
            f"  {row['time']:<10}  {row['samples']:>7}  "
            f"{_mark(bool(row['ctrl_ready'])):>14}  {_mark(bool(row['feature_ready'])):>13}",
        )
    return "\n".join(lines)


def format_readiness_chart(result: WarmupSimulationResult) -> str:
    points = result.chart_points
    if not points:
        return ""
    width = 20
    height = 6
    lines = ["Readiness Chart", "-" * 40, "  Ready %", ""]
    for level in range(height, -1, -1):
        threshold = level * 100 / height
        row_chars: list[str] = []
        for _label, pct in points:
            row_chars.append("█" if pct >= threshold else " ")
        lines.append(f"  {threshold:3.0f} |{''.join(row_chars)}")
    labels = "     " + "".join(f"{p[0][-5:]:>2}" for p in points[:width])
    lines.extend(["", labels])
    return "\n".join(lines)


def format_dependency_status(result: WarmupSimulationResult) -> str:
    lines = ["Dependency Status (final)", "-" * 40]
    for dep in result.dependency_status:
        lines.append(f"  {dep['label']:<16} {_mark(bool(dep['ready']))}")
    lines.extend(["", "Overall Feature", ""])
    feature_ready = bool(result.ready_at_sample)
    if not feature_ready and result.timeline:
        feature_ready = bool(result.timeline[-1].get("feature_ready"))
    if feature_ready:
        lines.append("  READY")
    else:
        lines.append("  NOT READY")
        lines.append("  Reason:")
        blocking = [d["label"] for d in result.dependency_status if not d["ready"]]
        if blocking:
            lines.append(f"  {blocking[0]} still warming up")
        elif result.controller_label:
            lines.append(f"  {result.controller_label} still warming up")
        else:
            lines.append("  Warm-up incomplete")
    return "\n".join(lines)


def format_gap_section(result: WarmupSimulationResult) -> str:
    if not result.gap_events:
        return ""
    lines = ["Gap Replay", "=" * 40, ""]
    trace = result.full_trace or []
    for evt in result.gap_events:
        ctrl = result.controller_label or "Rolling"
        lines.extend([
            f"  {evt.get('time', '—')}",
            "",
            "  Gap detected",
            f"  {evt.get('gap_sec', 0):.0f} sec",
            "",
            "  ↓",
            "",
            "  Policy",
            f"  {evt.get('policy', f'Reset {ctrl}')}",
            "",
            "  ↓",
            "",
            "  Samples reset",
            f"  {evt.get('samples_after_reset', 0)}",
            "",
            "  ↓",
            "",
            "  Feature",
            "  NOT READY" if not evt.get("feature_ready", False) else "  READY",
            "",
        ])
        after = int(evt.get("after_sample", 0))
        re_ready = None
        for row in trace:
            if int(row.get("samples", 0)) <= after:
                continue
            if row.get("feature_ready"):
                re_ready = row
                break
        if re_ready:
            lines.extend([
                "  ↓",
                "",
                "  … simulation continues …",
                "",
                "  ↓",
                "",
                "  First READY after gap",
                f"  Sample {re_ready.get('samples')}",
                f"  {re_ready.get('time')}",
                "",
            ])
        else:
            lines.extend([
                "  ↓",
                "",
                "  … simulation continues (not ready within window) …",
                "",
            ])
    return "\n".join(lines).rstrip()


def format_event_log(result: WarmupSimulationResult) -> str:
    lines = ["Event Log", "-" * 40]
    for evt in result.events:
        lines.append(f"  {evt}")
    return "\n".join(lines)


def format_dataset_impact(result: WarmupSimulationResult) -> str:
    impact = result.dataset_impact or {}
    if not impact:
        total = result.samples_processed
        ready_n = sum(1 for r in (result.full_trace or []) if r.get("feature_ready"))
        impact = {
            "total_samples": total,
            "ready_samples": ready_n,
            "null_samples": total - ready_n,
            "ready_pct": round(ready_n / max(total, 1) * 100.0, 1),
            "gap_resets": result.gap_resets,
            "effective_training_rows": ready_n,
        }
    lines = [
        "Simulation Summary",
        "=" * 40,
        "",
        f"  Total samples           {impact.get('total_samples', 0)}",
        f"  Ready samples           {impact.get('ready_samples', 0)}",
        f"  NULL samples            {impact.get('null_samples', 0)}",
        f"  Ready %                 {impact.get('ready_pct', 0)}%",
        f"  Gap resets              {impact.get('gap_resets', 0)}",
        f"  Effective training rows {impact.get('effective_training_rows', 0)}",
    ]
    return "\n".join(lines)


def format_simulation_summary(result: WarmupSimulationResult) -> str:
    lines = [
        "Policy Check",
        "-" * 40,
        "",
        f"  Feature:",
        f"  {result.feature_name}",
        "",
    ]
    trans = format_transition_section(result)
    if trans:
        lines.extend([trans, ""])
    if result.ready_at_ts:
        from chain_replay_ml.feature_policy.warmup_simulator import fmt_ist_time

        lines.extend([
            f"  Ready At",
            f"  {fmt_ist_time(result.ready_at_ts)}",
            "",
            f"  Sample",
            f"  {result.ready_at_sample}",
            "",
        ])
    else:
        lines.extend([
            f"  Ready At",
            f"  Never",
            "",
            f"  Reason",
            f"  {result.policy_reason}",
            "",
        ])
    lines.extend([
        f"  Gap Reset",
        f"  {'Yes' if result.gap_resets else 'No'}",
        "",
        f"  Policy {'PASS' if result.policy_pass else 'FAIL'}",
        f"  {'✓' if result.policy_pass else '✗'}",
    ])
    return "\n".join(lines)


def format_compare_features_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparison results."
    lines = [
        "Compare Features — Ready Time",
        "=" * 40,
        "",
        f"  {'Feature':<10}  {'Ready':>12}  {'Sample':>8}  {'Clock':>10}",
        "  " + "-" * 44,
    ]
    for row in rows:
        lines.append(
            f"  {row.get('ema', '—'):<10}  {row.get('ready_time', '—'):>12}  "
            f"{row.get('ready_sample') or '—':>8}  {row.get('ready_clock', '—'):>10}",
        )
    lines.extend([
        "",
        "  (Each row uses a representative derived feature for that EMA controller.)",
    ])
    return "\n".join(lines)


def format_calc_row_breakdown(breakdown: dict[str, Any]) -> str:
    lines = [
        f"Sample {breakdown.get('sample', '—')}",
        f"Time {breakdown.get('time', '—')}",
        "",
        "Dependencies",
        "-" * 40,
    ]
    deps = breakdown.get("dependencies") or []
    if deps:
        for dep in deps:
            lines.append(f"  {dep}")
    else:
        lines.append("  —")
    lines.extend(["", "Formula", "-" * 40, f"  {breakdown.get('formula_doc', '—')}", ""])
    steps = breakdown.get("steps") or []
    if steps:
        lines.append("Calculation")
        lines.append("-" * 40)
        for step in steps:
            label = step.get("label", "")
            text = step.get("text", "")
            if label == "=":
                lines.extend(["", f"  = {text}"])
            else:
                lines.append(f"  {label}: {text}")
    tree = breakdown.get("tree_lines") or []
    if tree:
        lines.extend(["", "Formula Tree", "-" * 40])
        for ln in tree:
            lines.append(f"  {ln}")
    return "\n".join(lines)


def format_calc_debug_summary(calc_debug: dict[str, Any], feature_name: str) -> str:
    if not calc_debug or not calc_debug.get("ok"):
        err = (calc_debug or {}).get("error") or "No calculation data"
        return f"Feature Calculation\n{'-' * 40}\n  {err}\n"
    spec = calc_debug.get("formula_spec") or {}
    lines = [
        "Feature Calculation",
        "=" * 40,
        "",
        f"  Feature: {feature_name}",
        f"  Formula: {spec.get('formula_doc', '—')}",
        "",
        "  Select a row below to verify operands and formula substitution.",
    ]
    return "\n".join(lines)


def format_maturity_gauge(row: dict[str, Any], *, total: int) -> str:
    ready = int(row.get("ready") or 0)
    pct = float(row.get("ready_pct") or 0)
    width = 20
    filled = int(round(pct / 100.0 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"Dataset Readiness\n\n  {bar}\n\n  {ready} / {total}   ({pct}%)"


def format_maturity_chart(summary: dict[str, Any], *, total: int) -> str:
    points = summary.get("chart_points") or []
    if not points:
        return ""
    max_ready = max(int(p[1]) for p in points) or total
    height = 6
    lines = ["Timeline Chart — Ready Features", "-" * 44, ""]
    for level in range(height, -1, -1):
        threshold = level * max_ready / height if height else 0
        chars = "".join("█" if r >= threshold else " " for _s, r in points)
        label = int(threshold)
        lines.append(f"  {label:>3} |{chars}")
    labels = "      " + "".join(f"{str(s)[-3:]:>3}" for s, _r in points[:16])
    lines.extend(["", labels])
    return "\n".join(lines)


def _fmt_timing_sec(val: Any) -> str:
    if val is None:
        return "—"
    try:
        sec = float(val)
    except (TypeError, ValueError):
        return "—"
    if sec < 1:
        return f"{sec:.3f} s"
    if sec < 60:
        return f"{sec:.2f} s"
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins}m {rem:.1f}s"


def _fmt_tick_count(val: Any) -> str:
    if val is None:
        return "—"
    try:
        n = int(val)
    except (TypeError, ValueError):
        return "—"
    return f"{n:,}"


def _fmt_bytes(val: Any) -> str:
    if val is None:
        return "—"
    try:
        b = int(val)
    except (TypeError, ValueError):
        return "—"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


def _fmt_run_timestamp(timing: dict[str, Any] | None) -> str:
    ts = (timing or {}).get("run_completed_at")
    return f" ({ts})" if ts else ""


def timing_tab_title(result: WarmupSimulationResult | None) -> str:
    """Notebook tab label for the Time Taken tab."""
    if result and result.timing:
        return f"Time Taken{_fmt_run_timestamp(result.timing)}"
    return "Time Taken"


def format_gap_pass_comparison(timing: dict[str, Any] | None) -> str:
    """Single-pass gap OFF vs ON function diff table sorted by slowdown."""
    cmp_doc = (timing or {}).get("gap_pass_comparison")
    if not isinstance(cmp_doc, dict) or not cmp_doc.get("by_function"):
        return ""
    lines = [
        "",
        "Function diff — gap OFF vs ON (isolated single-pass, no dual lookback)",
        "-" * 72,
        f"  Wall time OFF / ON / Δ:  {cmp_doc.get('gap_off_wall_sec', '—')} / "
        f"{cmp_doc.get('gap_on_wall_sec', '—')} / {cmp_doc.get('delta_wall_sec', '—')} s",
        f"  Changed functions: {cmp_doc.get('changed_function_count', '—')}",
        "",
        f"  {'Function':<36} {'Gap OFF':>9} {'Gap ON':>9} {'Difference':>11}",
        f"  {'-' * 36} {'-' * 9} {'-' * 9} {'-' * 11}",
    ]
    for row in cmp_doc.get("by_function") or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or row.get("function") or "?")[:36]
        off_s = float(row.get("gap_off_sec", 0))
        on_s = float(row.get("gap_on_sec", 0))
        delta = float(row.get("delta_sec", 0))
        calls_off = int(row.get("calls_off", 0))
        calls_on = int(row.get("calls_on", 0))
        call_note = ""
        if calls_off != calls_on:
            call_note = f"  ({calls_off:,}→{calls_on:,} calls)"
        lines.append(
            f"  {label:<36} {off_s:>8.3f}s {on_s:>8.3f}s {delta:>+10.3f}s{call_note}",
        )
    lines.extend([
        "",
        "  Sorted by Difference (Gap ON − OFF) descending; only changed functions shown.",
        "  Call counts appended when OFF vs ON differ.",
    ])
    return "\n".join(lines)


def format_readiness_profiler(timing: dict[str, Any] | None) -> str:
    """Readiness enforcement profiler from main replay pass."""
    prof = (timing or {}).get("readiness_profiler")
    if not isinstance(prof, dict):
        return ""
    lines = [
        "",
        "Readiness Enforcement Profiler",
        "-" * 72,
        f"  enforce_readiness_on_rows wall     {float(prof.get('enforce_wall_sec', 0)):>7.3f} s",
        f"  validate_readiness_compliance wall {float(prof.get('validate_wall_sec', 0)):>7.3f} s",
        f"  Total                              {float(prof.get('total_wall_sec', 0)):>7.3f} s",
        f"  gap_max_sec (readiness)            {prof.get('gap_max_sec', '—')}",
        "",
        f"  {'Function':<28} {'Time':>9} {'Calls':>10} {'Avg (µs)':>10}",
        f"  {'-' * 28} {'-' * 9} {'-' * 10} {'-' * 10}",
    ]
    for row in prof.get("by_function") or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("function") or "?")[:28]
        lines.append(
            f"  {label:<28} {float(row.get('time_sec', 0)):>8.3f}s "
            f"{int(row.get('calls', 0)):>10,} {float(row.get('avg_time_us', 0)):>10.1f}",
        )
    lines.append("  Readiness is per-timestamp; rows at same ts share engine state.")
    return "\n".join(lines)


def format_readiness_comparison(timing: dict[str, Any] | None) -> str:
    """Gap OFF vs ON readiness profiler diff (from gap pass compare)."""
    cmp_doc = (timing or {}).get("gap_pass_comparison")
    if not isinstance(cmp_doc, dict):
        return ""
    rc = cmp_doc.get("readiness_comparison")
    if not isinstance(rc, dict) or not rc.get("by_function"):
        return ""
    lines = [
        "",
        "Readiness diff — gap OFF vs ON (inside build_day_rows)",
        "-" * 72,
        f"  Total OFF / ON / Δ:  {rc.get('off_total_sec')} / {rc.get('on_total_sec')} / "
        f"{rc.get('delta_total_sec')} s",
        "",
        f"  {'Function':<24} {'OFF':>9} {'ON':>9} {'Δ':>9} {'Calls OFF':>10} {'Calls ON':>10}",
        f"  {'-' * 24} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 10} {'-' * 10}",
    ]
    for row in rc.get("by_function") or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("function") or "?")[:24]
        lines.append(
            f"  {label:<24} {float(row.get('off_sec', 0)):>8.3f}s "
            f"{float(row.get('on_sec', 0)):>8.3f}s {float(row.get('delta_sec', 0)):>+8.3f}s "
            f"{int(row.get('calls_off', 0)):>10,} {int(row.get('calls_on', 0)):>10,}",
        )
    return "\n".join(lines)


def format_replay_pipeline(timing: dict[str, Any] | None) -> str:
    """Stage-by-stage breakdown of the main replay attach path."""
    pipe = (timing or {}).get("replay_pipeline")
    if not isinstance(pipe, dict):
        return ""
    rows = [
        ("load_day_context", "load_day_context_sec"),
        ("build_day_rows", "build_day_rows_sec"),
        ("to_dataframe", "to_dataframe_sec"),
        ("serialize_replay_rows", "serialize_replay_rows_sec"),
        ("replay_statistics", "replay_statistics_sec"),
        ("build_replay_lookup", "build_replay_lookup_sec"),
    ]
    lines = [
        "",
        "Replay Pipeline (main pass)",
        "-" * 44,
    ]
    for label, key in rows:
        sec = pipe.get(key)
        if sec is None:
            continue
        lines.append(f"  {label:<32} {float(sec):>7.3f} s")
    total = pipe.get("total_sec")
    frame = pipe.get("build_replay_day_frame_sec")
    if total is not None:
        lines.extend(["-" * 44, f"  {'Pipeline total':<32} {float(total):>7.3f} s"])
    if frame is not None and total is not None and abs(float(frame) - float(total)) > 0.05:
        lines.append(f"  {'build_replay_day_frame (wall)':<32} {float(frame):>7.3f} s")
    return "\n".join(lines)


def format_replay_context_benchmark(timing: dict[str, Any] | None) -> str:
    """Cold vs warm build_day_rows on reused DayContext."""
    bench = (timing or {}).get("replay_context_benchmark")
    if not isinstance(bench, dict):
        err = (timing or {}).get("replay_context_benchmark_error")
        if err:
            return f"\nReplay context benchmark error: {err}"
        return ""
    cold = float(bench.get("cold_build_day_rows_sec") or 0.0)
    warm = float(bench.get("warm_build_day_rows_sec") or 0.0)
    save = float(bench.get("cache_savings_sec") or 0.0)
    lines = [
        "",
        "build_day_rows — context cache effect",
        "-" * 44,
        f"  {'Cold build (caches cleared)':<32} {cold:>7.3f} s",
        f"  {'Warm build (immediate rerun)':<32} {warm:>7.3f} s",
        f"  {'Cache savings':<32} {save:>7.3f} s",
        "  Cold = reset spot EMA / momentum / HL caches then build.",
        "  Warm = second build on same ctx (production rerun pattern).",
    ]
    return "\n".join(lines)


def format_gap_policy_profiler(timing: dict[str, Any] | None) -> str:
    """Gap Policy Profiler — counters, gap-function timings, and root-cause note."""
    prof = (timing or {}).get("gap_policy_profiler")
    if not isinstance(prof, dict):
        return ""
    lines = [
        "",
        "Gap Policy Profiler",
        "-" * 44,
        f"  Gap max (sec)                         {prof.get('gap_max_sec', '—')}",
        f"  Row gap checks (O(1))                 {prof.get('gap_checks', 0):,}",
        f"  Actual gaps detected                  {prof.get('gaps_detected', 0):,}",
        f"  reset_option_rolling_state calls        {prof.get('reset_count', 0):,}",
        f"  ensure_ltp_ema_state cache hits       {prof.get('ltp_ema_cache_hits', 0):,}",
        f"  ensure_ltp_ema_state full rebuilds    {prof.get('ltp_ema_rebuilds', 0):,}",
        "",
        "Gap-related function time",
        "-" * 44,
    ]
    fn_times = prof.get("gap_function_times_sec") or prof.get("function_times_sec") or {}
    if isinstance(fn_times, dict):
        for name, sec in list(fn_times.items())[:12]:
            calls = (prof.get("function_calls") or {}).get(name, 0)
            call_note = f" ({calls:,} calls)" if calls else ""
            lines.append(f"  {name:<36} {float(sec):.3f}s{call_note}")
    top = prof.get("top_functions") or []
    if top:
        lines.extend(["", "Top functions (cumulative, gap build pass)", "-" * 44])
        for row in top[:20]:
            if not isinstance(row, dict):
                continue
            fn = row.get("function", "?")
            cum = row.get("cumtime_sec", row.get("total_sec"))
            ncalls = row.get("ncalls", row.get("calls", ""))
            lines.append(f"  {str(fn):<36} {float(cum or 0):.3f}s  {ncalls}")
    lines.extend([
        "",
        "Why gap ON can be slower than gap OFF (single pass)",
        "-" * 44,
        "  Gap max (sec) is the configured reset threshold, not runtime.",
        "  Row gap checks are O(1); resets only when sample gap > threshold.",
        "  True single-pass delta = gap-aware spot/token EMA precompute",
        "  (ensure_ltp_ema_state, SpotControllers streaming) on first use per token.",
        "  Use the Function diff table below for gap OFF vs ON (runs automatically).",
        "  Dual lookback benchmark is separate (2× feature calc) — not gap policy.",
    ])
    dominant = _dominant_from_gap_comparison(timing)
    if dominant:
        label, delta = dominant
        lines.append(f"  Dominant slowdown (gap OFF→ON diff): {label} (+{delta:.3f}s)")
    return "\n".join(lines)


def _dominant_from_gap_comparison(timing: dict[str, Any] | None) -> tuple[str, float] | None:
    cmp_doc = (timing or {}).get("gap_pass_comparison")
    if not isinstance(cmp_doc, dict):
        return None
    label = cmp_doc.get("dominant_function")
    delta = float(cmp_doc.get("dominant_delta_sec") or 0.0)
    if label and delta > 0:
        return str(label), delta
    rows = cmp_doc.get("by_function") or []
    if not rows:
        return None
    top = max(rows, key=lambda r: float(r.get("delta_sec", 0)))
    delta = float(top.get("delta_sec", 0))
    if delta <= 0:
        return None
    return str(top.get("label") or top.get("function") or "?"), delta


def _timing_debug_config(result: WarmupSimulationResult | None) -> PerformanceDebugConfig:
    timing = (result.timing if result else None) or {}
    return PerformanceDebugConfig.resolve(timing.get("performance_debug_level"))


def format_gap_policy_summary(timing: dict[str, Any] | None) -> str:
    """Gap policy counters without cProfile / top-function tables (Basic mode)."""
    prof = (timing or {}).get("gap_policy_profiler")
    if not isinstance(prof, dict):
        return ""
    lines = [
        "",
        "Gap Policy Summary",
        "-" * 44,
        f"  Gap max (sec)                         {prof.get('gap_max_sec', '—')}",
        f"  Row gap checks (O(1))                 {prof.get('gap_checks', 0):,}",
        f"  Actual gaps detected                  {prof.get('gaps_detected', 0):,}",
        f"  reset_option_rolling_state calls        {prof.get('reset_count', 0):,}",
        f"  ensure_ltp_ema_state cache hits       {prof.get('ltp_ema_cache_hits', 0):,}",
        f"  ensure_ltp_ema_state full rebuilds    {prof.get('ltp_ema_rebuilds', 0):,}",
    ]
    return "\n".join(lines)


def format_timing_summary(
    result: WarmupSimulationResult | None,
    *,
    performance_debug: PerformanceDebugConfig | None = None,
) -> str:
    """Human-readable timing breakdown for the Time Taken tab."""
    if not result or not result.timing:
        return "Run a simulation to see timing breakdown."
    perf = performance_debug or _timing_debug_config(result)
    if perf.level == PerformanceDebugLevel.OFF:
        return _format_timing_summary_production(result)
    if perf.level == PerformanceDebugLevel.BASIC:
        return _format_timing_summary_basic(result, perf=perf)
    return _format_timing_summary_verbose(result, perf=perf)


def _format_timing_summary_production(result: WarmupSimulationResult) -> str:
    t = result.timing
    lines = [
        f"Time Taken{_fmt_run_timestamp(t)}",
        "-" * 40,
        f"  Fetch ticks                           {_fmt_timing_sec(t.get('load_ticks_sec'))}",
        f"  Build sample grid                     {_fmt_timing_sec(t.get('build_grid_sec'))}",
        f"  Policy simulation                     {_fmt_timing_sec(t.get('policy_engine_sec'))}",
        f"  Build features                        {_fmt_timing_sec(t.get('feature_calc_sec'))}",
    ]
    if t.get("temp_sqlite_insert_sec") is not None:
        lines.append(f"  SQLite write                          {_fmt_timing_sec(t.get('temp_sqlite_insert_sec'))}")
    if t.get("temp_parquet_export_sec") is not None:
        lines.append(f"  Parquet export                        {_fmt_timing_sec(t.get('temp_parquet_export_sec'))}")
    lines.extend([
        "-" * 40,
        f"  Total                                 {_fmt_timing_sec(t.get('total_sec'))}",
    ])
    return "\n".join(lines)


def _format_timing_summary_basic(
    result: WarmupSimulationResult,
    *,
    perf: PerformanceDebugConfig,
) -> str:
    t = result.timing
    lines = [
        f"Time Taken{_fmt_run_timestamp(t)}",
        "-" * 40,
        f"  Load ticks                            {_fmt_timing_sec(t.get('load_ticks_sec'))}",
        f"  Build sample grid                     {_fmt_timing_sec(t.get('build_grid_sec'))}",
        f"  Policy simulation                     {_fmt_timing_sec(t.get('policy_engine_sec'))}",
        f"  Build features                        {_fmt_timing_sec(t.get('feature_calc_sec'))}",
    ]
    pipe_block = format_replay_pipeline(t) if perf.show_replay_pipeline() else ""
    if pipe_block:
        lines.append(pipe_block)
    if t.get("temp_sqlite_insert_sec") is not None:
        lines.append(f"  SQLite write                          {_fmt_timing_sec(t.get('temp_sqlite_insert_sec'))}")
    if t.get("temp_parquet_export_sec") is not None:
        lines.append(f"  Parquet export                        {_fmt_timing_sec(t.get('temp_parquet_export_sec'))}")
    lines.extend([
        "-" * 40,
        f"  Total                                 {_fmt_timing_sec(t.get('total_sec'))}",
    ])
    return "\n".join(lines)


def _format_timing_summary_verbose(
    result: WarmupSimulationResult,
    *,
    perf: PerformanceDebugConfig,
) -> str:
    t = result.timing
    spot = t.get("spot_ticks")
    chain = t.get("chain_ticks")
    total_ticks = t.get("source_ticks")
    if spot is not None and chain is not None:
        tick_detail = (
            f"{_fmt_tick_count(spot)} spot + {_fmt_tick_count(chain)} chain"
            f" ({_fmt_tick_count(total_ticks)} total)"
        )
    else:
        tick_detail = ""
    dur_min = t.get("tick_load_duration_min")
    if dur_min is not None:
        tick_detail = (tick_detail + f" · {int(dur_min)} min window") if tick_detail else f"{int(dur_min)} min window"
    fetch_line = f"  Fetching ticks (load_day_context)     {_fmt_timing_sec(t.get('load_ticks_sec'))}"
    if tick_detail:
        fetch_line += f"  ·  {tick_detail}"
    maturity_replay_line = f"  Dataset maturity replay               {_fmt_timing_sec(t.get('maturity_replay_sec'))}"
    if t.get("maturity_replay_skipped"):
        maturity_replay_line += "  (skipped)"
    elif t.get("maturity_replay_shared"):
        maturity_replay_line += "  (shared with all-features calc)"
    all_features = bool(t.get("all_features_calc"))
    calc_label = (
        "All features calculation"
        if all_features
        else "Feature calculation (calc debug)"
    )
    lines = [
        f"Time Taken{_fmt_run_timestamp(t)}",
        "=" * 44,
        "",
        fetch_line,
        f"  Building sample grid                  {_fmt_timing_sec(t.get('build_grid_sec'))}",
        f"  Policy engine simulation              {_fmt_timing_sec(t.get('policy_engine_sec'))}",
        f"  {calc_label:<37} {_fmt_timing_sec(t.get('feature_calc_sec'))}",
        maturity_replay_line,
    ]
    pipe_block = format_replay_pipeline(t) if perf.show_replay_pipeline() else ""
    if pipe_block:
        lines.append(pipe_block)
    if perf.show_full_diagnostics() and t.get("feature_calc_without_lookback_sec") is not None:
        lines.append(
            f"  Feature calc (exact_timestamp baseline) {_fmt_timing_sec(t.get('feature_calc_without_lookback_sec'))}",
        )
    if perf.show_full_diagnostics() and t.get("lookback_nearest_snapshot_sec") is not None:
        lines.append(
            f"  Lookback nearest_snapshot overhead      {_fmt_timing_sec(t.get('lookback_nearest_snapshot_sec'))}",
        )
    if perf.show_full_diagnostics() and t.get("all_features_calc_wall_sec") is not None:
        lines.append(
            f"  All features wall time (both passes)    {_fmt_timing_sec(t.get('all_features_calc_wall_sec'))}",
        )
    if perf.show_full_diagnostics() and t.get("lookback_benchmark_skipped"):
        lines.append("  Lookback benchmark                    skipped")
        lines.append(f"    {t.get('lookback_benchmark_skipped')}")
    if perf.show_full_diagnostics():
        cmp_doc = t.get("gap_pass_comparison")
        if isinstance(cmp_doc, dict) and cmp_doc.get("gap_off_wall_sec") is not None:
            lines.append(
                f"  Gap pass compare (OFF / ON / Δ)         "
                f"{cmp_doc.get('gap_off_wall_sec')} / {cmp_doc.get('gap_on_wall_sec')} / "
                f"{cmp_doc.get('delta_wall_sec')} s",
            )
        if t.get("gap_pass_comparison_error"):
            lines.append(f"  Gap pass compare error                {t.get('gap_pass_comparison_error')}")
    sqlite_sec = t.get("temp_sqlite_insert_sec")
    parquet_sec = t.get("temp_parquet_export_sec")
    if sqlite_sec is not None or parquet_sec is not None:
        lines.append(f"  Temp SQLite insert (build I/O)          {_fmt_timing_sec(sqlite_sec)}")
        sqlite_rows = t.get("temp_sqlite_rows")
        sqlite_bytes = t.get("temp_sqlite_bytes")
        if sqlite_rows is not None:
            lines.append(
                f"    Rows written                        {_fmt_tick_count(sqlite_rows)}"
                + (f" · {_fmt_bytes(sqlite_bytes)}" if sqlite_bytes is not None else ""),
            )
        lines.append(f"  Temp Parquet export (build I/O)         {_fmt_timing_sec(parquet_sec)}")
        parquet_rows = t.get("temp_parquet_rows")
        parquet_bytes = t.get("temp_parquet_bytes")
        if parquet_rows is not None:
            lines.append(
                f"    Rows exported                       {_fmt_tick_count(parquet_rows)}"
                + (f" · {_fmt_bytes(parquet_bytes)}" if parquet_bytes is not None else ""),
            )
    if t.get("temp_build_io_error"):
        lines.append(f"  Temp build I/O error                  {t.get('temp_build_io_error')}")
    lines.extend([
        "",
        "-" * 44,
        f"  Total                                 {_fmt_timing_sec(t.get('total_sec'))}",
    ])
    if result.feature_name:
        policy_n = t.get("policy_grid_samples", result.samples_processed)
        chain_n = t.get("chain_rows_total")
        sample_line = f"Day: {result.trading_day or '—'} · Policy samples: {policy_n}"
        if chain_n is not None:
            sample_line += f" · Chain rows: {_fmt_tick_count(chain_n)}"
        lines.extend([
            "",
            f"Feature: {result.feature_name}",
            sample_line,
        ])
    calc = result.calc_debug or {}
    if calc.get("ok"):
        rows_n = len(calc.get("rows") or [])
        lines.append(f"Calc debug rows: {rows_n}")
    elif calc.get("error"):
        lines.append(f"Calc debug: {calc.get('error')}")
    if result.maturity_replay_error:
        if t.get("maturity_replay_skipped"):
            lines.append("Maturity replay: skipped")
        else:
            lines.append(f"Maturity replay: {result.maturity_replay_error}")
    elif result.maturity_replay_lookup:
        lines.append(f"Maturity replay buckets: {len(result.maturity_replay_lookup)}")
    if result.all_features_lookup:
        feat_n = len(result.maturity_feature_names or [])
        lines.append(
            f"All features lookup: {len(result.all_features_lookup)} buckets"
            + (f" · {feat_n} features" if feat_n else ""),
        )
    chain_total = t.get("chain_rows_total")
    if chain_total is not None:
        grid_n = t.get("chain_grid_timestamps") or t.get("policy_grid_samples", result.samples_processed)
        lines.extend([
            "",
            "Sample counts",
            "-" * 44,
            f"  Grid timestamps (duration window)     {_fmt_tick_count(grid_n)}",
        ])
        strikes_band = t.get("strikes_in_band")
        rows_per_ts = t.get("rows_per_timestamp")
        if strikes_band is not None and rows_per_ts is not None:
            strike_lbl = t.get("strike_selection_label") or "ATM band"
            lines.append(
                f"  Strikes in band ({strike_lbl})            {_fmt_tick_count(strikes_band)}",
            )
            lines.append(
                f"  Rows per timestamp ({strikes_band} × CE/PE)   {_fmt_tick_count(rows_per_ts)}",
            )
            expected = t.get("expected_chain_rows")
            if expected is not None:
                lines.append(
                    f"  Expected chain rows ({grid_n} × {rows_per_ts})  {_fmt_tick_count(expected)}",
                )
        lines.append(
            f"  Actual chain rows (all tokens)        {_fmt_tick_count(chain_total)}",
        )
        if t.get("avg_rows_per_timestamp") is not None:
            lines.append(
                f"  Avg rows per timestamp (actual)       {t.get('avg_rows_per_timestamp')}",
            )
        if t.get("unique_tokens") is not None:
            lines.append(
                f"  Unique tokens in window               {_fmt_tick_count(t.get('unique_tokens'))}",
            )
        target_cols = t.get("target_columns") or result.target_columns or []
        if target_cols:
            lines.append(f"  Prediction targets                    {', '.join(target_cols)}")
        trimmed = t.get("target_trimmed_rows")
        if trimmed:
            lines.append(
                f"  Rows trimmed (missing targets)        {_fmt_tick_count(trimmed)}",
            )
        if t.get("match_build_dataset_selection"):
            lines.append("  Build config (trim, enabled features)  ON")
            feat_n = t.get("build_feature_count")
            if feat_n is not None:
                lines.append(f"  Enabled build features                {_fmt_tick_count(feat_n)}")
        elif chain_total is not None or t.get("match_build_dataset_selection") is False:
            lines.append("  Build config (trim, enabled features)  OFF")
        if t.get("match_build_gap_parity"):
            lines.append("  Build gap parity                      ON")
            gap_b = t.get("build_gap_max_sec")
            if gap_b is not None:
                lines.append(f"  Gap policy (build)                    {int(float(gap_b))}s")
        elif chain_total is not None or t.get("match_build_gap_parity") is False:
            lines.append("  Build gap parity                      OFF")
    if t.get("lookback_nearest_snapshot") is not None:
        lb_state = "ON (nearest_snapshot)" if t.get("lookback_nearest_snapshot") else "OFF (exact_timestamp)"
        lines.append(f"  Lookback nearest_snapshot             {lb_state}")
    lb_method = t.get("lookback_policy_method")
    if lb_method:
        lines.append(f"  Lookback policy applied               {lb_method}")
    if chain_total is not None:
        strike_lbl = t.get("strike_selection_label")
        if strike_lbl and strikes_band is None:
            lines.append(f"  Strike selection                      {strike_lbl}")
    if perf.show_gap_policy_summary():
        gap_prof = format_gap_policy_profiler(t)
        if gap_prof:
            lines.append(gap_prof)
    if perf.show_full_diagnostics():
        gap_cmp = format_gap_pass_comparison(t)
        if gap_cmp:
            lines.append(gap_cmp)
        readiness_cmp = format_readiness_comparison(t)
        if readiness_cmp:
            lines.append(readiness_cmp)
        readiness_prof = format_readiness_profiler(t)
        if readiness_prof:
            lines.append(readiness_prof)
        ctx_bench = format_replay_context_benchmark(t)
        if ctx_bench:
            lines.append(ctx_bench)
    return "\n".join(lines)


def format_maturity_buckets(summary: dict[str, Any]) -> str:
    buckets = summary.get("buckets") or []
    if not buckets:
        return ""
    lines = ["Dataset Maturity", "=" * 40, ""]
    for b in buckets:
        lines.append(f"  {b.get('label', '—'):<10}  {b.get('avg_ready_pct', 0)}%")
    return "\n".join(lines)


def format_maturity_row_detail(row: dict[str, Any], *, total: int) -> str:
    sample = row.get("sample", "—")
    ready = int(row.get("ready") or 0)
    pct = float(row.get("ready_pct") or 0)
    lines = [
        f"Sample {sample}",
        f"Time {row.get('time', '—')}",
        "",
        f"Ready {ready} / {total}  ({pct}%)",
        f"Skip Row: {'YES' if row.get('skip_row') else 'NO'}",
    ]
    explain = row.get("simulated_feature_explain") or {}
    if explain.get("ok"):
        feat = explain.get("feature", "—")
        mark = "YES" if explain.get("ready") else "NO"
        lines.extend([
            "",
            "Simulated Feature",
            "-" * 40,
            f"  {feat}",
            f"  Ready at this sample: {mark}",
            f"  {explain.get('reason', '')}",
        ])
        feat_blocking = explain.get("blocking_controllers") or []
        if feat_blocking:
            lines.append(f"  Blocked by rolling: {', '.join(feat_blocking)}")
        elif not explain.get("ready"):
            lines.append("  Blocked by rolling: (none — not an EMA-derived feature)")
    lines.extend([
        "",
        "By Category",
        "-" * 40,
        f"  Raw       {row.get('raw', 0)}",
        f"  Rolling   {row.get('rolling', 0)}",
        f"  Derived   {row.get('derived', 0)}",
        f"  Lookback  {row.get('lookback', 0)}",
    ])
    detail = row.get("detail") or {}
    controllers = detail.get("controllers") or []
    if controllers:
        lines.extend(["", "Controllers", "-" * 40])
        for c in controllers:
            mark = "✓" if c.get("ready") else "✗"
            lines.append(f"  {mark} {c.get('label', '—')}")
        blocked = int(row.get("derived_blocked_count") or detail.get("derived_blocked_count") or 0)
        if blocked:
            lines.append(f"\n  Derived blocked by rolling: {blocked}")
    blocking_ctrl = detail.get("blocking_controllers") or []
    if blocking_ctrl:
        lines.extend([
            "",
            "Blocking Rolling (other features)",
            "-" * 40,
            "  EMA controllers still warming derived features —",
            "  not necessarily why the simulated feature is NULL.",
        ])
        for label in blocking_ctrl:
            lines.append(f"  ✗ {label}")
    not_ready_ctrl = detail.get("not_ready_controllers") or []
    warming_only = [
        lbl for lbl in not_ready_ctrl
        if lbl not in set(blocking_ctrl)
    ]
    if warming_only:
        lines.extend(["", "Still Warming (no feature blocked yet)", "-" * 40])
        for label in warming_only:
            lines.append(f"  · {label}")
    elif controllers and row.get("skip_row") and not blocking_ctrl:
        lines.extend([
            "",
            "Rolling Controllers",
            "-" * 40,
            "  All EMA rolling controllers are ready at this sample.",
            "  Skip Row is YES because lookback / calendar features",
            "  still need more session time.",
        ])
    not_ready_lookback = detail.get("not_ready_lookback") or []
    lookback_count = int(detail.get("not_ready_lookback_count") or 0)
    if not_ready_lookback:
        lines.extend(["", "Lookback Features Not Ready", "-" * 40])
        for name in not_ready_lookback[:20]:
            lines.append(f"  ✗ {name}")
        if len(not_ready_lookback) > 20:
            lines.append(f"  … +{len(not_ready_lookback) - 20} more")
    elif lookback_count and not detail.get("not_ready_lookback"):
        lines.extend([
            "",
            "Lookback Features Not Ready",
            "-" * 40,
            f"  {lookback_count} lookback feature(s) — double-click row for full list.",
        ])
    not_ready_derived = detail.get("not_ready_derived") or []
    if not_ready_derived:
        lines.extend(["", "Affected Derived", "-" * 40])
        for name in not_ready_derived[:25]:
            lines.append(f"  ✗ {name}")
        if len(not_ready_derived) > 25:
            lines.append(f"  … +{len(not_ready_derived) - 25} more")
    elif not detail:
        lines.extend([
            "",
            "Detail",
            "-" * 40,
            "  Double-click a row to open all feature values.",
        ])
    lines.extend([
        "",
        "Tip: Double-click a maturity row for the full feature value panel.",
    ])
    return "\n".join(lines)


def format_maturity_features_summary(panel: dict[str, Any]) -> str:
    summary = panel.get("summary") or {}
    return (
        f"Sample {panel.get('sample', '—')} @ {panel.get('time', '—')}  ·  "
        f"{summary.get('with_value', 0)} with value  ·  "
        f"{summary.get('null_policy', 0)} NULL (policy)  ·  "
        f"{summary.get('missing', 0)} missing replay"
    )


def maturity_features_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["feature", "category", "ready", "status", "value"])
    for row in rows:
        writer.writerow([
            row.get("name", ""),
            row.get("category", ""),
            "1" if row.get("ready") else "0",
            row.get("status", ""),
            row.get("display", ""),
        ])
    return buf.getvalue()


def format_progress_log(result: WarmupSimulationResult) -> str:
    lines = ["Background Steps", "=" * 40, ""]
    for step in result.progress_steps:
        lines.append(f"  {step}")
    cov = result.coverage_info or {}
    if cov and not result.ok:
        lines.extend(["", "Diagnostics", "-" * 40])
        if cov.get("grid_mode"):
            lines.append(f"  Grid mode: {cov.get('grid_mode')}")
        if cov.get("grid_points") is not None:
            lines.append(f"  Grid points: {cov.get('grid_points')}")
        if cov.get("grid_start"):
            lines.append(f"  Grid span: {cov.get('grid_start')} → {cov.get('grid_end', '—')}")
        if cov.get("fresh_grid_points") is not None:
            lines.append(f"  Fresh points: {cov.get('fresh_grid_points')}")
    return "\n".join(lines)


def format_simulation_results_body(result: WarmupSimulationResult) -> str:
    if not result.ok:
        parts = [
            "Simulation failed",
            "-" * 40,
            f"  {result.error or 'Unknown error'}",
        ]
        if result.timeline:
            parts.extend(["", format_timeline_table(result)])
        return "\n".join(parts)
    parts: list[str] = []
    trans = format_transition_section(result)
    if trans:
        parts.extend([trans, ""])
    parts.extend([
        format_timeline_table(result),
        "",
        format_output_values(result),
        "",
        format_dependency_timeline(result),
        "",
        format_readiness_chart(result),
        "",
        format_dependency_status(result),
    ])
    gap = format_gap_section(result)
    if gap:
        parts.extend(["", gap])
    parts.extend([
        "",
        format_event_log(result),
        "",
        format_dataset_impact(result),
        "",
        format_simulation_summary(result),
    ])
    return "\n".join(p for p in parts if p)


def format_simulation_result(result: WarmupSimulationResult) -> str:
    parts = [format_progress_log(result), "", format_simulation_results_body(result)]
    return "\n".join(parts)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def simulation_trace_csv(result: WarmupSimulationResult) -> str:
    """Full sample trace as CSV (one row per grid sample)."""
    import csv
    import io

    trace = result.full_trace or []
    if not trace:
        return ""
    dep_labels = list(result.dependency_labels or [])
    ctrl_col = f"{result.controller_label or 'controller'}_ready"
    fieldnames = [
        "feature",
        "trading_day",
        "sample",
        "time",
        "timestamp",
        ctrl_col,
        "controller_samples",
        "feature_ready",
        "output",
        *dep_labels,
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in trace:
        deps = row.get("deps") or {}
        out: dict[str, Any] = {
            "feature": result.feature_name,
            "trading_day": result.trading_day,
            "sample": row.get("samples"),
            "time": row.get("time"),
            "timestamp": row.get("ts"),
            ctrl_col: row.get("ctrl_ready"),
            "controller_samples": row.get("ctrl_samples"),
            "feature_ready": row.get("feature_ready"),
            "output": row.get("output_display"),
        }
        for label in dep_labels:
            out[label] = deps.get(label)
        writer.writerow({k: _csv_cell(out.get(k)) for k in fieldnames})
    return buf.getvalue()


def compare_features_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    if not rows:
        return ""
    fieldnames = ["ema", "feature", "ready_time", "ready_sample", "ready_clock", "policy_pass"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})
    return buf.getvalue()


def _policy_sample_for_ts(
    trace: list[dict[str, Any]],
    ts: float,
    *,
    step_sec: int,
) -> tuple[int | None, str | None]:
    """Map a chain row timestamp to policy trace sample number and time label."""
    best_sample: int | None = None
    best_time: str | None = None
    best_diff = float("inf")
    tol = max(float(step_sec) * 0.51, 0.5)
    for tr in trace:
        try:
            diff = abs(float(tr.get("ts", 0)) - float(ts))
        except (TypeError, ValueError):
            continue
        if diff <= tol and diff < best_diff:
            best_diff = diff
            samp = tr.get("samples")
            best_sample = int(samp) if samp is not None else None
            best_time = str(tr.get("time") or "") or None
    return best_sample, best_time


def build_data_filter_table(
    result: WarmupSimulationResult | None,
    feature_names: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Build columns + display rows for Data Filter tab (one row per policy sample)."""
    from chain_replay_ml.feature_policy.warmup_calc_debug import lookup_replay_values

    cols = ["sample", "time", *feature_names]
    if not result or not feature_names:
        return cols, []
    trace = list(result.full_trace or [])
    if not trace:
        return cols, []

    lookup = result.all_features_lookup or result.maturity_replay_lookup or {}
    step_sec = int(max(result.sampling_interval_sec, 1))
    rows: list[list[str]] = []
    for tr in trace:
        ts = tr.get("ts")
        row = [
            str(tr.get("samples") or "—"),
            str(tr.get("time") or "—"),
        ]
        if lookup and ts is not None:
            replay_vals = lookup_replay_values(lookup, float(ts), step_sec=step_sec)
        else:
            replay_vals = {}
        for feat in feature_names:
            raw = replay_vals.get(feat) if replay_vals else None
            if raw is None and feat == result.feature_name:
                disp = tr.get("output_display")
                if disp not in (None, "VALUE"):
                    raw = disp
            row.append(_csv_cell(raw) if raw is not None else "NULL")
        rows.append(row)
    return cols, rows


_LTP_EMA_TO_LTP_RATIO_RE = __import__("re").compile(r"^ltp_ema(\d+)_to_ltp_ratio$")
_LTP_EMA_TO_SPOT_RATIO_RE = __import__("re").compile(r"^ltp_ema(\d+)_to_spot_ratio$")


def ratio_split_spec(feature_name: str) -> tuple[int, str] | None:
    """Return (period, denominator) for ltp_ema ratio features — ltp or spot."""
    name = str(feature_name or "").strip()
    m = _LTP_EMA_TO_LTP_RATIO_RE.match(name)
    if m:
        return int(m.group(1)), "ltp"
    m = _LTP_EMA_TO_SPOT_RATIO_RE.match(name)
    if m:
        return int(m.group(1)), "spot"
    return None


def ltp_ema_period_from_ratio_feature(feature_name: str) -> int | None:
    spec = ratio_split_spec(feature_name)
    return spec[0] if spec else None


def ratio_split_column_plan(
    feature_names: list[str],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Columns for Ratio Split: ltp/spot bases + derived ltp_ema{N} from ratio features."""
    derived: list[tuple[str, str, str]] = []
    seen: set[tuple[int, str]] = set()
    need_ltp = False
    need_spot = False
    for feat in feature_names:
        spec = ratio_split_spec(feat)
        if spec is None:
            continue
        period, denom = spec
        key = (period, denom)
        if key in seen:
            continue
        seen.add(key)
        if denom == "ltp":
            need_ltp = True
        else:
            need_spot = True
        derived.append((f"ltp_ema{period}", feat, denom))
    derived.sort(key=lambda t: int(t[0].replace("ltp_ema", "") or 0))
    cols = ["sample", "time"]
    if need_ltp:
        cols.append("ltp")
    if need_spot:
        cols.append("spot")
    cols.extend(name for name, _, _ in derived)
    return cols, derived


def _numeric_or_none(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, str) and val.strip().upper() == "NULL":
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def derive_ltp_ema_from_ratio(*, base: Any, ratio: Any) -> float | None:
    """EMA{N}(ltp) = ratio × base (base is ltp or spot depending on ratio feature)."""
    base_f = _numeric_or_none(base)
    ratio_f = _numeric_or_none(ratio)
    if base_f is None or ratio_f is None or base_f <= 0:
        return None
    return ratio_f * base_f


def build_data_filter_ratio_split_table(
    result: WarmupSimulationResult | None,
    feature_names: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Ratio Split: base ltp + derived ltp_ema{N} from filtered ratio features."""
    from chain_replay_ml.feature_policy.warmup_calc_debug import lookup_replay_values

    cols, derived = ratio_split_column_plan(feature_names)
    if not result or not derived:
        return cols, []
    trace = list(result.full_trace or [])
    if not trace:
        return cols, []

    lookup = result.all_features_lookup or result.maturity_replay_lookup or {}
    step_sec = int(max(result.sampling_interval_sec, 1))
    rows: list[list[str]] = []
    for tr in trace:
        ts = tr.get("ts")
        replay_vals = (
            lookup_replay_values(lookup, float(ts), step_sec=step_sec)
            if lookup and ts is not None
            else {}
        )
        ltp_val = replay_vals.get("ltp")
        spot_val = replay_vals.get("spot")
        row_map: dict[str, str] = {
            "sample": str(tr.get("samples") or "—"),
            "time": str(tr.get("time") or "—"),
            "ltp": _csv_cell(ltp_val) if ltp_val is not None else "NULL",
            "spot": _csv_cell(spot_val) if spot_val is not None else "NULL",
        }
        for ema_col, ratio_feat, denom in derived:
            ratio_raw = replay_vals.get(ratio_feat)
            if ratio_raw is None and ratio_feat == result.feature_name:
                disp = tr.get("output_display")
                if disp not in (None, "NULL", "VALUE"):
                    ratio_raw = disp
            base = ltp_val if denom == "ltp" else spot_val
            ema_val = derive_ltp_ema_from_ratio(base=base, ratio=ratio_raw)
            row_map[ema_col] = _csv_cell(ema_val) if ema_val is not None else "NULL"
        rows.append([row_map.get(c, "NULL") for c in cols])
    return cols, rows


def table_cols_rows_to_csv(cols: list[str], rows: list[list[str]]) -> str:
    """Serialize a column/row table to CSV text."""
    import csv
    import io

    if not cols:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def data_filter_values_csv(
    result: WarmupSimulationResult | None,
    feature_names: list[str],
) -> str:
    cols, rows = build_data_filter_table(result, feature_names)
    if not rows:
        return ""
    return table_cols_rows_to_csv(cols, rows)


def data_filter_ratio_split_csv(
    result: WarmupSimulationResult | None,
    feature_names: list[str],
) -> str:
    cols, rows = build_data_filter_ratio_split_table(result, feature_names)
    if not rows:
        return ""
    return table_cols_rows_to_csv(cols, rows)


def build_controller_warmup_regression_table(
    result: WarmupSimulationResult | None,
) -> tuple[list[str], list[list[str]]]:
    """Warmup Regression tab: expected vs actual first valid sample per feature."""
    from chain_replay_ml.dataset_builder.controller_warmup_regression import (
        validate_all_warmup_regressions_from_result,
    )

    cols = [
        "feature",
        "category",
        "source",
        "expected_first_sample",
        "actual_first_sample",
        "status",
        "note",
    ]
    rows: list[list[str]] = []
    for item in validate_all_warmup_regressions_from_result(result):
        actual = item.get("actual")
        expected = item.get("expected")
        rows.append([
            str(item.get("feature") or ""),
            str(item.get("category") or ""),
            str(item.get("source") or ""),
            str(expected) if expected is not None else "—",
            str(actual) if actual is not None else "—",
            str(item.get("status") or "PENDING"),
            str(item.get("note") or ""),
        ])
    return cols, rows


def controller_warmup_regression_csv(
    result: WarmupSimulationResult | None,
) -> str:
    cols, rows = build_controller_warmup_regression_table(result)
    if not rows:
        return ""
    return table_cols_rows_to_csv(cols, rows)


def controller_warmup_regression_hint(result: WarmupSimulationResult | None) -> str:
    from chain_replay_ml.dataset_builder.controller_warmup_regression import (
        WARMUP_REGRESSION_SPEC,
        warmup_regression_summary,
        validate_all_warmup_regressions_from_result,
    )

    rows = validate_all_warmup_regressions_from_result(result)
    summary = warmup_regression_summary(rows)
    n = len(WARMUP_REGRESSION_SPEC)
    parts = [f"{n} warmup feature(s)"]
    if summary.get("PASS"):
        parts.append(f"{summary['PASS']} PASS")
    if summary.get("FAIL"):
        parts.append(f"{summary['FAIL']} FAIL")
    if summary.get("SKIP"):
        parts.append(f"{summary['SKIP']} SKIP")
    if summary.get("PENDING"):
        parts.append("run simulation to validate")
    return " · ".join(parts)


def _null_audit_category(feature_name: str) -> str:
    from chain_replay_ml.dataset_builder.controller_warmup_regression import (
        _WARMUP_REGRESSION_BY_FEATURE,
    )

    spec = _WARMUP_REGRESSION_BY_FEATURE.get(feature_name)
    return str(spec.category) if spec else "—"


def _classify_null_audit_status(
    *,
    null_count: int,
    value_count: int,
    missing_count: int,
    total: int,
) -> str:
    if total <= 0:
        return "ALL_MISSING"
    if missing_count >= total:
        return "ALL_MISSING"
    if value_count >= total:
        return "HAS_VALUE"
    if null_count >= total:
        return "ALL_NULL"
    if value_count > 0:
        return "PARTIAL"
    return "ALL_NULL"


def build_null_audit_table(
    result: WarmupSimulationResult | None,
) -> tuple[list[str], list[list[str]]]:
    """Null Audit tab: session-wide NULL / value / missing status per feature column."""
    from chain_replay_ml.dataset_builder.controller_warmup_regression import is_valid_feature_value
    from chain_replay_ml.feature_policy.warmup_calc_debug import (
        lookup_replay_values,
        resolve_replay_lookup_from_result,
    )

    cols = [
        "feature",
        "category",
        "status",
        "null_count",
        "value_count",
        "missing_count",
        "total_samples",
    ]
    if result is None or not getattr(result, "ok", False):
        return cols, []

    trace = list(result.full_trace or [])
    lookup = resolve_replay_lookup_from_result(result)
    if not trace or not lookup:
        return cols, []

    step_sec = int(max(result.sampling_interval_sec, 1))
    feature_names = list(result.maturity_feature_names or [])
    if not feature_names:
        seen: set[str] = set()
        for bucket in lookup.values():
            if isinstance(bucket, dict):
                seen.update(str(k) for k in bucket)
        feature_names = sorted(seen)

    rows: list[list[str]] = []
    for feat in feature_names:
        null_n = 0
        value_n = 0
        missing_n = 0
        for tr in trace:
            ts = tr.get("ts")
            if ts is None:
                missing_n += 1
                continue
            replay_vals = lookup_replay_values(lookup, float(ts), step_sec=step_sec)
            if not replay_vals or feat not in replay_vals:
                missing_n += 1
                continue
            val = replay_vals.get(feat)
            if feat == result.feature_name and not is_valid_feature_value(val):
                disp = tr.get("output_display")
                if disp not in (None, "NULL", "VALUE"):
                    val = disp
            if is_valid_feature_value(val):
                value_n += 1
            else:
                null_n += 1
        total = len(trace)
        status = _classify_null_audit_status(
            null_count=null_n,
            value_count=value_n,
            missing_count=missing_n,
            total=total,
        )
        rows.append([
            feat,
            _null_audit_category(feat),
            status,
            str(null_n),
            str(value_n),
            str(missing_n),
            str(total),
        ])
    return cols, rows


def null_audit_csv(result: WarmupSimulationResult | None) -> str:
    cols, rows = build_null_audit_table(result)
    if not rows:
        return ""
    return table_cols_rows_to_csv(cols, rows)


def null_audit_hint(result: WarmupSimulationResult | None) -> str:
    cols, rows = build_null_audit_table(result)
    if not rows:
        return "run simulation with Dataset maturity replay"
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row[2] or "")
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{len(rows)} feature(s)"]
    for key in ("HAS_VALUE", "PARTIAL", "ALL_NULL", "ALL_MISSING"):
        if counts.get(key):
            parts.append(f"{counts[key]} {key}")
    return " · ".join(parts)


def all_features_export_status(result: WarmupSimulationResult | None) -> dict[str, Any]:
    """Summarize whether all-features CSV export has data."""
    if result is None:
        return {"ok": False, "reason": "no_result"}
    chain_rows = list(result.all_features_rows or [])
    trace = list(result.full_trace or [])
    lookup = result.all_features_lookup or result.maturity_replay_lookup or {}
    timing = result.timing or {}
    if chain_rows:
        return {
            "ok": True,
            "mode": "chain_rows",
            "chain_rows": len(chain_rows),
            "trace_samples": len(trace),
        }
    if lookup and trace:
        return {
            "ok": True,
            "mode": "lookup",
            "lookup_buckets": len(lookup),
            "trace_samples": len(trace),
        }
    return {
        "ok": False,
        "reason": "no_export_data",
        "chain_rows": len(chain_rows),
        "lookup_buckets": len(lookup),
        "trace_samples": len(trace),
        "all_features_calc": bool(timing.get("all_features_calc")),
        "maturity_replay_skipped": bool(timing.get("maturity_replay_skipped")),
    }


def all_features_csv(result: WarmupSimulationResult) -> str:
    """Export all-features replay as CSV (one row per strike/token per grid timestamp)."""
    import csv
    import io

    from chain_replay_ml.feature_policy.warmup_calc_debug import lookup_replay_values

    chain_rows = list(result.all_features_rows or [])
    trace = result.full_trace or []
    step_sec = int(max(result.sampling_interval_sec, 1))
    lookup = (
        result.all_features_lookup
        or result.maturity_replay_lookup
        or {}
    )

    feature_names = list(result.maturity_feature_names or [])
    target_cols = list(result.target_columns or [])
    if chain_rows:
        seen_feats: set[str] = set()
        id_and_target = {
            "trading_day", "market", "expiry", "timestamp",
            "strike", "option_type", "token", "symbol",
            *target_cols,
        }
        for row in chain_rows:
            for key in row:
                if key not in id_and_target:
                    seen_feats.add(str(key))
        if not feature_names:
            feature_names = sorted(seen_feats)

    if not chain_rows and not (lookup and trace):
        return ""

    id_cols = ["sample", "time", "timestamp", "token", "symbol", "strike", "option_type"]
    feat_cols = [
        f for f in feature_names
        if f not in id_cols and f not in target_cols
    ]
    fieldnames = [*id_cols, *target_cols, *feat_cols]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    if chain_rows:
        for row in chain_rows:
            ts = row.get("timestamp")
            sample, time_lbl = (None, None)
            if ts is not None:
                sample, time_lbl = _policy_sample_for_ts(trace, float(ts), step_sec=step_sec)
            if time_lbl is None and ts is not None:
                from chain_replay_ml.feature_policy.warmup_simulator import fmt_ist_time

                time_lbl = fmt_ist_time(float(ts))
            out: dict[str, Any] = {
                "sample": sample,
                "time": time_lbl,
                "timestamp": ts,
                "token": row.get("token"),
                "symbol": row.get("symbol"),
                "strike": row.get("strike"),
                "option_type": row.get("option_type"),
            }
            for name in feat_cols:
                out[name] = row.get(name)
            for col in target_cols:
                out[col] = row.get(col)
            writer.writerow({k: _csv_cell(out.get(k)) for k in fieldnames})
        return buf.getvalue()

    for row in trace:
        replay_vals = lookup_replay_values(lookup, row["ts"], step_sec=step_sec)
        out = {
            "sample": row.get("samples"),
            "time": row.get("time"),
            "timestamp": row.get("ts"),
            "token": replay_vals.get("token"),
            "symbol": None,
            "strike": None,
            "option_type": None,
        }
        for name in feat_cols:
            out[name] = replay_vals.get(name)
        writer.writerow({k: _csv_cell(out.get(k)) for k in fieldnames})
    return buf.getvalue()


def default_csv_filename(
    *,
    feature_name: str = "",
    trading_day: str = "",
    prefix: str = "warmup_sim",
) -> str:
    safe_feat = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature_name)[:40]
    day = trading_day.replace("-", "") if trading_day else "run"
    parts = [prefix, day]
    if safe_feat:
        parts.append(safe_feat)
    return "_".join(p for p in parts if p) + ".csv"

