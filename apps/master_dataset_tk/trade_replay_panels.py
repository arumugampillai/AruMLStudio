"""Plugin-style panels for Trade Replay window."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Callable

from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, fmt_rupee, metric_table

PanelBuilder = Callable[[tk.Misc, dict[str, Any]], None]

_REGISTRY: list[tuple[str, str, PanelBuilder]] = []


def register_trade_replay_panel(panel_id: str, title: str) -> Callable[[PanelBuilder], PanelBuilder]:
    def decorator(fn: PanelBuilder) -> PanelBuilder:
        _REGISTRY.append((panel_id, title, fn))
        return fn

    return decorator


def iter_trade_replay_panels() -> list[tuple[str, str, PanelBuilder]]:
    return list(_REGISTRY)


def _signed_rupee(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
        sign = "+" if n >= 0 else "−"
        return f"{sign}₹{abs(n):,.2f}"
    except (TypeError, ValueError):
        return "—"


@register_trade_replay_panel("counterfactual", "What If?")
def render_counterfactual_panel(parent: tk.Misc, ctx: dict[str, Any]) -> None:
    doc = ctx["doc"]
    counter = doc.get("counterfactuals") or {}
    fr = ttk.LabelFrame(parent, text="Counterfactual Replay — What If?", padding=6)
    fr.pack(fill="x", pady=(0, 6))
    if not counter.get("available"):
        ttk.Label(fr, text="Counterfactual analysis unavailable.", foreground=COL_MUTED).pack(anchor="w")
        return
    scenarios = counter.get("scenarios") or []
    for sc in scenarios:
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=2)
        profit = sc.get("profit")
        is_actual = sc.get("is_actual")
        is_best = sc.get("label") == counter.get("best_label") and not is_actual
        label_font = ("Segoe UI", 9, "bold") if is_actual or is_best else ("Segoe UI", 9)
        ttk.Label(row, text=str(sc.get("label") or ""), font=label_font, width=18).pack(side="left")
        color = COL_OK if profit is not None and float(profit) > 0 else COL_WARN
        ttk.Label(row, text=f"Profit = {_signed_rupee(profit)}", foreground=color, font=label_font).pack(side="left")
        if is_best:
            ttk.Label(row, text="best alt", foreground=COL_OK, font=("Segoe UI", 8)).pack(side="left", padx=4)
    ttk.Separator(fr, orient="horizontal").pack(fill="x", pady=4)
    ttk.Label(
        fr,
        text=f"{counter.get('scenario_count', 0)} alternate strategies tested on this premium path (no retraining).",
        foreground=COL_MUTED,
        font=("Segoe UI", 8),
        wraplength=340,
    ).pack(anchor="w")


@register_trade_replay_panel("explain", "AI Observation")
def render_explain_panel(parent: tk.Misc, ctx: dict[str, Any]) -> None:
    doc = ctx["doc"]
    win = ctx["win"]
    explanation = doc.get("trade_explanation") or {}
    fr = ttk.LabelFrame(parent, text="Explain This Trade", padding=6)
    fr.pack(fill="x", pady=(0, 6))

    narrative_box = scrolledtext.ScrolledText(fr, height=5, font=("Segoe UI", 9), wrap="word")
    narrative_box.pack(fill="x", pady=(0, 4))
    narrative_box.insert("1.0", str(explanation.get("narrative") or ""))
    narrative_box.configure(state="disabled")

    def _refresh() -> None:
        from chain_replay_ml.fold_research import explain_trade_narrative

        fresh = explain_trade_narrative(doc)
        narrative_box.configure(state="normal")
        narrative_box.delete("1.0", "end")
        narrative_box.insert("1.0", str(fresh.get("narrative") or ""))
        narrative_box.configure(state="disabled")

    ttk.Button(fr, text="Explain This Trade", command=_refresh).pack(anchor="w")


@register_trade_replay_panel("feature_time_machine", "Feature Time Machine")
def render_feature_time_machine_panel(parent: tk.Misc, ctx: dict[str, Any]) -> None:
    doc = ctx["doc"]
    ftm = doc.get("feature_time_machine") or {}
    trails = ftm.get("trails") or {}
    fr = ttk.LabelFrame(parent, text="Feature Time Machine", padding=6)
    fr.pack(fill="x", pady=(0, 6))
    if not trails:
        ttk.Label(fr, text="Feature trails unavailable.", foreground=COL_MUTED).pack(anchor="w")
        return
    nb = ttk.Notebook(fr)
    nb.pack(fill="both", expand=True)
    for fname, trail in list(trails.items())[:5]:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text=fname[:16])
        cols = ("time", "value", "contrib")
        tree = ttk.Treeview(tab, columns=cols, show="headings", height=2)
        for c, w, label in (("time", 64, "Time"), ("value", 72, "Value"), ("contrib", 72, "Contribution")):
            tree.heading(c, text=label)
            tree.column(c, width=w)
        tree.pack(fill="both", expand=True)
        tree.tag_configure("up", foreground=COL_OK)
        tree.tag_configure("down", foreground=COL_WARN)
        for pt in trail:
            c = pt.get("contribution_pct")
            tag = ""
            if c is not None:
                tag = "up" if float(c) > 0 else ("down" if float(c) < 0 else "")
            c_txt = f"{pt.get('arrow', '')} {c:+.1f}%" if c is not None else "—"
            tree.insert(
                "",
                "end",
                tags=(tag,) if tag else (),
                values=(pt.get("rel_label"), pt.get("value"), c_txt),
            )


@register_trade_replay_panel("multi_model", "Multi-Model")
def render_multi_model_stub(parent: tk.Misc, ctx: dict[str, Any]) -> None:
    doc = ctx["doc"]
    pred = (doc.get("decision") or {}).get("prediction") or {}
    fr = ttk.LabelFrame(parent, text="Multi-Model Replay", padding=6)
    fr.pack(fill="x", pady=(0, 6))
    metric_table(
        fr,
        [
            ("Current Model", fmt_rupee(pred.get("predicted_ltp"))),
            ("Previous Model", "—"),
            ("Champion", "—"),
            ("Actual", fmt_rupee(pred.get("actual_ltp"))),
        ],
        label_width=14,
    )
    ttk.Label(
        fr,
        text="Load additional prediction runs for the same fold to compare models side-by-side.",
        foreground=COL_MUTED,
        font=("Segoe UI", 8),
        wraplength=320,
    ).pack(anchor="w", pady=(4, 0))


@register_trade_replay_panel("research_conclusion", "Research Conclusion")
def render_research_conclusion_panel(parent: tk.Misc, ctx: dict[str, Any]) -> None:
    doc = ctx["doc"]
    rc = doc.get("research_conclusion") or {}
    fr = ttk.LabelFrame(parent, text="Research Conclusion", padding=6)
    fr.pack(fill="x", pady=(0, 6))
    if not rc:
        ttk.Label(fr, text="Conclusion unavailable.", foreground=COL_MUTED).pack(anchor="w")
        return
    metric_table(
        fr,
        [
            ("Root Cause", str(rc.get("root_cause") or "—")),
            ("Classification", str(rc.get("classification") or "—")),
            ("Expected Improvement", str(rc.get("expected_improvement") or "—")),
        ],
        label_width=18,
    )
    recs = rc.get("recommendations") or []
    if recs:
        ttk.Label(fr, text="Recommendation — avoid entries when:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 2))
        for r in recs:
            ttk.Label(fr, text=f"• {r}", font=("Segoe UI", 9), wraplength=320).pack(anchor="w", padx=(8, 0))


def mount_plugin_panels(parent: tk.Misc, *, doc: dict[str, Any], win: tk.Misc) -> None:
    """Render all registered trade replay panels."""
    ctx = {"doc": doc, "win": win}
    for _pid, _title, builder in iter_trade_replay_panels():
        builder(parent, ctx)
