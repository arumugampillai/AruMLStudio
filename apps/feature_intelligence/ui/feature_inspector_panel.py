"""Feature Intelligence Inspector panels (Sprint 9) — Tk metadata view.

Card / chip layout over inspect payloads. Deep-link navigation to
OP_/TR_/ONT_/FRR_ registries is **reserved** (post Phase 1).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from feature_intelligence.ui.inspect_format import (
    ALL_SCOPES,
    SCOPE_LABELS,
    architecture_strip,
    compiler_raw_json,
    compiler_stack_summary,
    feature_display_name,
    header_summary_lines,
    identity_fields,
    lineage_tree_text,
    ontology_chip_labels,
    ontology_fields,
    overview_fields,
    platform_summary_count_rows,
    platform_summary_version_rows,
    references_are_empty,
    references_fields,
    research_fields,
    FieldRow,
)


def _style_absent(label: ttk.Label, present: bool) -> None:
    if present:
        label.configure(foreground="")
    else:
        try:
            label.configure(foreground="#888888")
        except tk.TclError:
            pass


class _ScrollBody(ttk.Frame):
    """Simple scrollable frame for card tabs."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=scroll.set)
        self.body = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body)
        self._canvas.bind("<Configure>", self._on_canvas)

    def _on_body(self, _event: object = None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._canvas.itemconfigure(self._win, width=event.width)

    def clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()


class _CardTab(ttk.Frame):
    """One inspector subtab with a scrollable card body."""

    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text=title, font=("", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 2)
        )
        self._scroll = _ScrollBody(self)
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

    @property
    def body(self) -> ttk.Frame:
        return self._scroll.body

    def clear(self) -> None:
        self._scroll.clear_body()


def _add_field_rows(parent: ttk.Frame, rows: list[FieldRow], *, start_row: int = 0) -> int:
    parent.columnconfigure(1, weight=1)
    r = start_row
    for row in rows:
        lab = ttk.Label(parent, text=row.label, font=("", 9, "bold"))
        lab.grid(row=r, column=0, sticky="nw", padx=(8, 12), pady=3)
        val = ttk.Label(parent, text=row.value, wraplength=520, justify="left")
        val.grid(row=r, column=1, sticky="ew", padx=(0, 8), pady=3)
        _style_absent(val, row.present)
        r += 1
    return r


def _add_chip_row(parent: ttk.Frame, chips: list[tuple[str, str, bool]], *, row: int) -> None:
    frame = ttk.Frame(parent)
    frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
    for i, (label, value, present) in enumerate(chips):
        chip = ttk.Label(
            frame,
            text=f" {label}: {value} ",
            relief="solid",
            padding=(6, 2),
        )
        chip.pack(side="left", padx=(0, 6), pady=2)
        _style_absent(chip, present)


def _add_arch_strip(parent: ttk.Frame, payload: dict[str, Any], *, row: int) -> None:
    box = ttk.LabelFrame(parent, text="Architecture", padding=6)
    box.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
    parts = architecture_strip(payload)
    for i, (lab, val, present) in enumerate(parts):
        cell = ttk.Frame(box)
        cell.pack(side="left", padx=(0, 4))
        ttk.Label(cell, text=lab, font=("", 8, "bold")).pack(anchor="w")
        id_lab = ttk.Label(cell, text=val, wraplength=140)
        id_lab.pack(anchor="w")
        _style_absent(id_lab, present)
        if i < len(parts) - 1:
            ttk.Label(box, text="→", font=("", 12)).pack(side="left", padx=6)


class FeatureInspectorPanel(ttk.Frame):
    """
    Overview / Identity / Compiler / Ontology / Lineage / Research / References.

    Data source: inspect API payload only (read-only). No model metrics.
    When no feature is loaded, shows platform summary dashboard (API-fed).
    """

    TAB_KEYS = (
        ("overview", "Overview"),
        ("identity", "Identity"),
        ("compiler", "Compiler"),
        ("ontology", "Ontology"),
        ("lineage", "Lineage"),
        ("research", "Research"),
        ("references", "References"),
    )

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_deep_link: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._on_deep_link = on_deep_link
        self._payload: dict[str, Any] | None = None
        self._platform_summary: dict[str, Any] | None = None

        self._status = tk.StringVar(value="No feature loaded — search and select a result")
        self._sub = tk.StringVar(value="")
        hdr = ttk.Frame(self)
        hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        hdr.columnconfigure(0, weight=1)
        self._title_lbl = ttk.Label(hdr, textvariable=self._status, font=("", 11, "bold"))
        self._title_lbl.grid(row=0, column=0, sticky="w")
        self._sub_lbl = ttk.Label(hdr, textvariable=self._sub, wraplength=720, justify="left")
        self._sub_lbl.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self._nb = ttk.Notebook(self)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._tabs: dict[str, _CardTab] = {}
        for key, label in self.TAB_KEYS:
            tab = _CardTab(self._nb, label)
            self._tabs[key] = tab
            self._nb.add(tab, text=label)
        self.clear()

    def set_platform_summary(self, summary: dict[str, Any] | None) -> None:
        """Feed empty-state dashboard from get_platform_summary()."""
        self._platform_summary = summary
        if self._payload is None:
            self._render_dashboard()

    def clear(self) -> None:
        self._payload = None
        self._status.set("No feature loaded — search and select a result")
        self._sub.set("Platform summary (read-only)")
        self._render_dashboard()

    def show_loading(self, message: str = "Loading feature…") -> None:
        self._status.set(message)
        self._sub.set("")

    def _render_dashboard(self) -> None:
        """Empty-state: counts + versions from Public API summary."""
        for key in self._tabs:
            tab = self._tabs[key]
            tab.clear()
            if key != "overview":
                ttk.Label(
                    tab.body,
                    text="(no feature selected)",
                    foreground="#888888",
                ).grid(row=0, column=0, sticky="w", padx=8, pady=8)

        ov = self._tabs["overview"]
        counts_card = ttk.LabelFrame(ov.body, text="Registry counts", padding=6)
        counts_card.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        counts_card.columnconfigure(1, weight=1)
        _add_field_rows(counts_card, platform_summary_count_rows(self._platform_summary))

        vers_card = ttk.LabelFrame(ov.body, text="Pack / registry versions", padding=6)
        vers_card.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        vers_card.columnconfigure(1, weight=1)
        _add_field_rows(vers_card, platform_summary_version_rows(self._platform_summary))

        note_row = 2
        if self._platform_summary is None:
            ttk.Label(
                ov.body,
                text="Summary not loaded yet — host will refresh from get_platform_summary().",
                foreground="#666666",
            ).grid(row=note_row, column=0, sticky="w", padx=8, pady=4)
            note_row += 1
        else:
            counts = self._platform_summary.get("counts") or {}
            feat_c = counts.get("features")
            frr_c = counts.get("research_records")
            hint = self._platform_summary.get("empty_hint")
            if not hint and isinstance(feat_c, int) and isinstance(frr_c, int):
                if frr_c == 0 and feat_c > 0:
                    hint = (
                        "No research records — List All / search start from FRR. "
                        "Run: python -m feature_intelligence research sync"
                    )
                elif frr_c == 0 and feat_c == 0:
                    hint = (
                        "Empty registry. Initialize DB, register features, then "
                        "run: python -m feature_intelligence research sync"
                    )
            if hint:
                ttk.Label(
                    ov.body,
                    text=str(hint),
                    foreground="#a15c00",
                    wraplength=640,
                    justify="left",
                ).grid(row=note_row, column=0, sticky="w", padx=8, pady=4)

    def load_inspect(self, payload: dict[str, Any] | None) -> None:
        """Render inspect_feature data dict (with sections_present)."""
        if not payload:
            self.clear()
            return
        self._payload = payload
        lines = header_summary_lines(payload)
        self._status.set(lines[0] if lines else f"Feature : {feature_display_name(payload)}")
        self._sub.set("\n".join(lines[1:]) if len(lines) > 1 else "")

        self._render_overview(payload)
        self._render_identity(payload)
        self._render_compiler(payload)
        self._render_ontology(payload)
        self._render_lineage(payload)
        self._render_research(payload)
        self._render_references(payload)

    def _render_overview(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["overview"]
        tab.clear()
        body = tab.body
        card = ttk.LabelFrame(body, text="Identity summary", padding=6)
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        card.columnconfigure(1, weight=1)
        _add_field_rows(card, overview_fields(payload))
        _add_arch_strip(body, payload, row=1)
        present = payload.get("sections_present") or {}
        chip_frame = ttk.LabelFrame(body, text="Sections present", padding=6)
        chip_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        chips = [(k, "yes" if present.get(k) else "absent", bool(present.get(k))) for k in (
            "identity", "compiler", "ast", "ontology", "lineage", "research", "references"
        )]
        _add_chip_row(chip_frame, chips, row=0)

    def _render_identity(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["identity"]
        tab.clear()
        card = ttk.LabelFrame(tab.body, text="Feature identity", padding=6)
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        card.columnconfigure(1, weight=1)
        _add_field_rows(card, identity_fields(payload))

    def _render_compiler(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["compiler"]
        tab.clear()
        stack = ttk.LabelFrame(tab.body, text="Compiler stack", padding=6)
        stack.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        stack.columnconfigure(1, weight=1)
        _add_field_rows(stack, compiler_stack_summary(payload))

        raw_box = ttk.LabelFrame(tab.body, text="Raw (optional)", padding=4)
        raw_box.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        raw_box.columnconfigure(0, weight=1)
        show = tk.BooleanVar(value=False)
        text = tk.Text(raw_box, wrap="word", height=10, state="disabled")

        def _toggle() -> None:
            text.configure(state="normal")
            text.delete("1.0", "end")
            if show.get():
                text.insert("1.0", compiler_raw_json(payload))
                text.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
            else:
                text.grid_remove()
            text.configure(state="disabled")

        ttk.Checkbutton(
            raw_box, text="Show raw JSON", variable=show, command=_toggle
        ).grid(row=0, column=0, sticky="w", padx=4, pady=2)

    def _render_ontology(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["ontology"]
        tab.clear()
        chips_box = ttk.LabelFrame(tab.body, text="Vocabulary chips", padding=6)
        chips_box.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        _add_chip_row(chips_box, ontology_chip_labels(payload), row=0)
        detail = ttk.LabelFrame(tab.body, text="Ontology detail", padding=6)
        detail.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        detail.columnconfigure(1, weight=1)
        _add_field_rows(detail, ontology_fields(payload))

    def _render_lineage(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["lineage"]
        tab.clear()
        box = ttk.LabelFrame(tab.body, text="Lineage graph", padding=6)
        box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        box.columnconfigure(0, weight=1)
        txt = tk.Text(box, wrap="word", height=18, state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, command=txt.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        txt.configure(yscrollcommand=scroll.set)
        txt.configure(state="normal")
        txt.insert("1.0", lineage_tree_text(payload))
        txt.configure(state="disabled")

    def _render_research(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["research"]
        tab.clear()
        card = ttk.LabelFrame(tab.body, text="Research record (FRR)", padding=6)
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        card.columnconfigure(1, weight=1)
        _add_field_rows(card, research_fields(payload))

    def _render_references(self, payload: dict[str, Any]) -> None:
        tab = self._tabs["references"]
        tab.clear()
        if references_are_empty(payload):
            ttk.Label(
                tab.body,
                text="No references found.",
                font=("", 10),
            ).grid(row=0, column=0, sticky="w", padx=8, pady=12)
            ttk.Label(
                tab.body,
                text="Phase 1 has no model / dataset / research-program linkage tables.",
                foreground="#666666",
            ).grid(row=1, column=0, sticky="w", padx=8, pady=2)
            return
        card = ttk.LabelFrame(tab.body, text="Linked references", padding=6)
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        card.columnconfigure(1, weight=1)
        _add_field_rows(card, references_fields(payload))

    def open_registry_stub(self, object_id: str) -> None:
        if self._on_deep_link is not None:
            return
        _ = object_id


class FeatureIntelligenceSearchBar(ttk.Frame):
    """Feature Explorer search: text + scope checkboxes + advanced hint."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_search: Callable[[], None] | None = None,
        on_sync_registry: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._on_search = on_search
        self._on_sync_registry = on_sync_registry
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="Search").grid(row=0, column=0, sticky="w")
        self.query_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.query_var)
        entry.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        entry.bind("<Return>", lambda _e: self._search())
        ttk.Button(self, text="Search", command=self._search).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(self, text="List all", command=self._list_all).grid(
            row=0, column=3, padx=2
        )
        if on_sync_registry is not None:
            ttk.Button(
                self,
                text="Sync from Feature Registry",
                command=self._sync_registry,
            ).grid(row=0, column=4, padx=(8, 2))

        scope_row = ttk.Frame(self)
        scope_row.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 2))
        ttk.Label(scope_row, text="Scope:").pack(side="left", padx=(0, 6))
        self._scope_vars: dict[str, tk.BooleanVar] = {}
        for key in ALL_SCOPES:
            var = tk.BooleanVar(value=(key == "name"))
            self._scope_vars[key] = var
            ttk.Checkbutton(
                scope_row, text=SCOPE_LABELS[key], variable=var
            ).pack(side="left", padx=(0, 8))

        ttk.Label(
            self,
            text=(
                "Simple text uses scopes (Name = substring). "
                "Advanced: field:value tokens, e.g. status:ACTIVE domain:price feat:FEAT_…"
            ),
            foreground="#555555",
            wraplength=720,
            justify="left",
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(2, 0))

        # Back-compat aliases used by older hosts (unused by Studio redesign)
        self.subject_var = self.query_var

    def selected_scopes(self) -> set[str]:
        return {k for k, v in self._scope_vars.items() if v.get()}

    def _search(self) -> None:
        if self._on_search:
            self._on_search()

    def _list_all(self) -> None:
        self.query_var.set("")
        if self._on_search:
            self._on_search()

    def _sync_registry(self) -> None:
        if self._on_sync_registry:
            self._on_sync_registry()

