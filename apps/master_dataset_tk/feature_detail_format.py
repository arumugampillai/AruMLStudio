"""Text formatters for the unified feature detail panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from typing import Any

from .source_navigation import open_source_location


def _insert_line(widget: scrolledtext.ScrolledText, text: str, *, end: str = "\n") -> None:
    widget.insert(tk.END, text + end)


def _insert_source_link(
    widget: scrolledtext.ScrolledText,
    label: str,
    location: dict[str, Any],
    *,
    tag_prefix: str,
    counter: list[int],
) -> None:
    if not label or not location.get("ok"):
        _insert_line(widget, label or "—", end="")
        return
    tag = f"{tag_prefix}_{counter[0]}"
    counter[0] += 1
    widget.insert(tk.END, label, (tag, "source_link"))
    widget.tag_configure(tag, foreground="#1565c0", underline=True)

    def _open(_event: tk.Event, loc: dict[str, Any] = location) -> str:
        open_source_location(loc)
        return "break"

    widget.tag_bind(tag, "<Button-1>", _open)
    widget.tag_bind(tag, "<Enter>", lambda _e: widget.configure(cursor="hand2"))
    widget.tag_bind(tag, "<Leave>", lambda _e: widget.configure(cursor=""))


def render_feature_detail_widget(
    widget: scrolledtext.ScrolledText,
    detail: dict[str, Any],
) -> None:
    """Fill *widget* with feature detail text and clickable source paths."""
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)

    if not detail.get("ok"):
        _insert_line(widget, f"Feature detail unavailable.\n\n{detail.get('error') or 'Unknown error'}")
        return

    link_counter = [0]
    title = str(detail.get("display_name") or detail.get("name") or "—")
    _insert_line(widget, title)
    _insert_line(widget, "=" * min(72, max(len(title), 24)))
    _insert_line(
        widget,
        f"{detail.get('name')}  ·  group={detail.get('group')}  ·  {detail.get('category')}",
    )
    _insert_line(widget, "")

    ctx = detail.get("context") or {}
    if ctx:
        _insert_line(widget, "Sample Context")
        _insert_line(widget, "-" * 40)
        if ctx.get("sample") is not None:
            _insert_line(widget, f"  Sample: {ctx.get('sample')}")
        if ctx.get("time"):
            _insert_line(widget, f"  Time: {ctx.get('time')}")
        if ctx.get("ready") is not None:
            _insert_line(widget, f"  Policy ready: {'yes' if ctx.get('ready') else 'no'}")
        if ctx.get("status"):
            _insert_line(widget, f"  Status: {ctx.get('status')}")
        if ctx.get("display") is not None:
            _insert_line(widget, f"  Value: {ctx.get('display')}")
        _insert_line(widget, "")

    if detail.get("description"):
        _insert_line(widget, "What it is")
        _insert_line(widget, "-" * 40)
        _insert_line(widget, f"  {detail['description']}")
        _insert_line(widget, "")

    if detail.get("interpretation"):
        _insert_line(widget, "How to read it")
        _insert_line(widget, "-" * 40)
        _insert_line(widget, f"  {detail['interpretation']}")
        _insert_line(widget, "")

    _insert_line(widget, "Formula")
    _insert_line(widget, "-" * 40)
    _insert_line(widget, f"  {detail.get('formula_doc') or '—'}")
    _insert_line(widget, "")

    inputs = detail.get("inputs") or []
    _insert_line(widget, "Inputs")
    _insert_line(widget, "-" * 40)
    if inputs:
        for inp in inputs:
            _insert_line(widget, f"  • {inp.get('name')} — {inp.get('label')}")
    else:
        _insert_line(widget, "  (raw tick / chain fields at sample time)")
    _insert_line(widget, "")

    parity = detail.get("parity") or {}
    if parity:
        _insert_line(widget, "Computation")
        _insert_line(widget, "-" * 40)
        _insert_line(widget, f"  Kind: {parity.get('kind') or '—'}")
        _insert_line(widget, f"  Rule: {parity.get('rule') or '—'}")
        _insert_line(widget, f"  Warm-up: {parity.get('warmup') or '—'}")
        _insert_line(widget, "")

    policy = detail.get("policy") or {}
    if policy:
        _insert_line(widget, "Policy (NULL rules)")
        _insert_line(widget, "-" * 40)
        _insert_line(widget, f"  Category: {policy.get('feature_category') or '—'}")
        _insert_line(widget, f"  Effective warm-up: {policy.get('effective_warmup_samples') or 0} samples")
        if policy.get("intrinsic_warmup_sec"):
            _insert_line(widget, f"  Time warm-up: {policy.get('intrinsic_warmup_sec')} sec")
        if policy.get("policy_anchor"):
            _insert_line(widget, f"  Rolling anchor: {policy.get('policy_anchor')}")
        if detail.get("expected_null_reason"):
            _insert_line(widget, f"  NULL when: {detail.get('expected_null_reason')}")
        _insert_line(widget, "")

    impl = detail.get("implementation") or {}
    source = detail.get("source_location") or {}
    _insert_line(widget, "Implementation")
    _insert_line(widget, "-" * 40)
    widget.insert(tk.END, "  ")
    module_label = str(impl.get("module") or "—")
    if source.get("ok"):
        line = int(source.get("line") or 1)
        _insert_source_link(
            widget,
            f"{module_label}:{line}",
            source,
            tag_prefix="src_mod",
            counter=link_counter,
        )
    else:
        _insert_line(widget, module_label, end="")
    _insert_line(widget, f"  {impl.get('function') or '—'}")
    if source.get("ok"):
        _insert_line(widget, "  (click path to open in editor)")
    _insert_line(widget, "")

    _insert_line(widget, "Python (pseudocode)")
    _insert_line(widget, "-" * 40)
    code = str(detail.get("python_code") or "—")
    widget.insert(tk.END, code + "\n")
    _insert_line(widget, "")

    meta_bits: list[str] = []
    if detail.get("unit") and detail.get("unit") != "—":
        meta_bits.append(f"unit={detail['unit']}")
    if detail.get("example"):
        meta_bits.append(f"example={detail['example']}")
    if detail.get("expected_range"):
        meta_bits.append(f"range={detail['expected_range']}")
    if meta_bits:
        _insert_line(widget, "Reference")
        _insert_line(widget, "-" * 40)
        _insert_line(widget, "  " + "  ·  ".join(meta_bits))


def format_feature_detail_text(detail: dict[str, Any]) -> str:
    if not detail.get("ok"):
        return f"Feature detail unavailable.\n\n{detail.get('error') or 'Unknown error'}"

    lines: list[str] = []
    title = detail.get("display_name") or detail.get("name") or "—"
    lines.append(title)
    lines.append("=" * min(72, max(len(str(title)), 24)))
    lines.append(
        f"{detail.get('name')}  ·  group={detail.get('group')}  ·  {detail.get('category')}"
    )
    lines.append("")

    ctx = detail.get("context") or {}
    if ctx:
        lines.extend(["Sample Context", "-" * 40])
        if ctx.get("sample") is not None:
            lines.append(f"  Sample: {ctx.get('sample')}")
        if ctx.get("time"):
            lines.append(f"  Time: {ctx.get('time')}")
        if ctx.get("ready") is not None:
            lines.append(f"  Policy ready: {'yes' if ctx.get('ready') else 'no'}")
        if ctx.get("status"):
            lines.append(f"  Status: {ctx.get('status')}")
        if ctx.get("display") is not None:
            lines.append(f"  Value: {ctx.get('display')}")
        lines.append("")

    if detail.get("description"):
        lines.extend(["What it is", "-" * 40, f"  {detail['description']}", ""])

    if detail.get("interpretation"):
        lines.extend(["How to read it", "-" * 40, f"  {detail['interpretation']}", ""])

    lines.extend(["Formula", "-" * 40, f"  {detail.get('formula_doc') or '—'}", ""])

    inputs = detail.get("inputs") or []
    lines.extend(["Inputs", "-" * 40])
    if inputs:
        for inp in inputs:
            lines.append(f"  • {inp.get('name')} — {inp.get('label')}")
    else:
        lines.append("  (raw tick / chain fields at sample time)")
    lines.append("")

    parity = detail.get("parity") or {}
    if parity:
        lines.extend([
            "Computation",
            "-" * 40,
            f"  Kind: {parity.get('kind') or '—'}",
            f"  Rule: {parity.get('rule') or '—'}",
            f"  Warm-up: {parity.get('warmup') or '—'}",
            "",
        ])

    policy = detail.get("policy") or {}
    if policy:
        lines.extend([
            "Policy (NULL rules)",
            "-" * 40,
            f"  Category: {policy.get('feature_category') or '—'}",
            f"  Effective warm-up: {policy.get('effective_warmup_samples') or 0} samples",
        ])
        if policy.get("intrinsic_warmup_sec"):
            lines.append(f"  Time warm-up: {policy.get('intrinsic_warmup_sec')} sec")
        if policy.get("policy_anchor"):
            lines.append(f"  Rolling anchor: {policy.get('policy_anchor')}")
        if detail.get("expected_null_reason"):
            lines.append(f"  NULL when: {detail.get('expected_null_reason')}")
        lines.append("")

    impl = detail.get("implementation") or {}
    source = detail.get("source_location") or {}
    module_line = impl.get("module") or "—"
    if source.get("ok"):
        module_line = f"{module_line}:{source.get('line')}"
    lines.extend([
        "Implementation",
        "-" * 40,
        f"  {module_line}",
        f"  {impl.get('function') or '—'}",
        "",
    ])

    lines.extend(["Python (pseudocode)", "-" * 40, detail.get("python_code") or "—", ""])

    meta_bits: list[str] = []
    if detail.get("unit") and detail.get("unit") != "—":
        meta_bits.append(f"unit={detail['unit']}")
    if detail.get("example"):
        meta_bits.append(f"example={detail['example']}")
    if detail.get("expected_range"):
        meta_bits.append(f"range={detail['expected_range']}")
    if meta_bits:
        lines.extend(["Reference", "-" * 40, "  " + "  ·  ".join(meta_bits)])

    return "\n".join(lines)
