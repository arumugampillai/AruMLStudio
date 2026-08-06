"""Format GIL diagnostic report for the progress panel."""

from __future__ import annotations

from typing import Any


def format_gil_report(report: dict[str, Any] | None) -> str:
    if not report:
        return "No GIL report captured."
    lines = [
        "GIL / Event-loop diagnostic",
        "=" * 40,
        str(report.get("interpretation") or ""),
        "",
        f"GIL contention likely: {report.get('gil_contention_likely')}",
        f"Progress callback yields: {report.get('progress_callback_yields')}",
    ]
    burst = report.get("worker_burst_tracker") or {}
    probe = report.get("gil_hold_probe") or {}
    lag = report.get("main_thread_lag") or {}
    lines.extend(
        [
            "",
            "Worker Python bursts (sys.setprofile):",
            f"  Longest continuous Python: {burst.get('longest_continuous_python_ms')} ms",
            f"  Location: {burst.get('longest_burst_location')}",
            f"  Yield gaps detected: {burst.get('voluntary_yield_gaps_detected')}",
            f"  Bursts > 50 ms: {len(burst.get('bursts_over_50ms') or [])}",
        ]
    )
    for item in (burst.get("bursts_over_50ms") or [])[:8]:
        lines.append(f"    · {item.get('duration_ms')} ms @ {item.get('location')}")
    lines.extend(
        [
            "",
            "GIL hold probe (1 ms sleeper):",
            f"  Longest estimated hold: {probe.get('longest_estimated_gil_hold_ms')} ms",
            f"  Samples > 5 ms: {probe.get('hold_samples_over_5ms')}",
            f"  p95 hold: {probe.get('estimated_hold_p95_ms')} ms",
            f"  Holds > 50 ms: {len(probe.get('holds_over_50ms') or [])}",
        ]
    )
    for item in (probe.get("holds_over_50ms") or [])[:3]:
        lines.append(f"    · {item.get('hold_ms')} ms")
        stack = str(item.get("worker_stack") or "").strip()
        if stack:
            for stack_line in stack.splitlines()[-4:]:
                lines.append(f"      {stack_line}")
    lines.extend(
        [
            "",
            "Main thread (Tk poll) lag:",
            f"  Longest poll gap: {lag.get('longest_poll_gap_ms')} ms (expected {lag.get('expected_poll_interval_ms')} ms)",
            f"  Gaps > 50 ms over expected: {lag.get('poll_gaps_over_50ms')}",
        ]
    )
    path = report.get("report_path")
    if path:
        lines.extend(["", f"Full report: {path}"])
    return "\n".join(lines)
