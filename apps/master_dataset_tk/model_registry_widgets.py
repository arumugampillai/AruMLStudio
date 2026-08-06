"""Reusable Tk widgets for Model Registry detail (web UI parity)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Iterable

ACCENT = "#58a6ff"
COL_TRAINING = "#1976D2"
COL_PRODUCTION = "#2E7D32"
COL_HOLDOUT = "#E65100"
COL_MUTED = "#666666"
COL_OK = "#2E7D32"
COL_WARN = "#C62828"
SECTION_FONT = ("Segoe UI", 10, "bold")
BODY_FONT = ("Segoe UI", 9)
MONO_FONT = ("Consolas", 9)


def pnl_foreground(v: Any) -> str:
    """Green for profit, red for loss."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return COL_MUTED
    if n > 0:
        return COL_OK
    if n < 0:
        return COL_WARN
    return COL_MUTED


def clear_children(widget: tk.Misc) -> None:
    for child in widget.winfo_children():
        child.destroy()


def fmt_num(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rupee(v: Any, digits: int = 2) -> str:
    try:
        return f"₹{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_signed_rupee(v: Any, digits: int = 2) -> str:
    try:
        n = float(v)
        sign = "+" if n > 0 else ("−" if n < 0 else "")
        return f"{sign}₹{abs(n):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_signed_pct(v: Any, digits: int = 2) -> str:
    try:
        n = float(v)
        sign = "+" if n > 0 else ("−" if n < 0 else "")
        return f"{sign}{abs(n):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_rows(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_val(v: Any) -> str:
    if v is None or v == "":
        return "—"
    return str(v)


class ScrollableFrame(ttk.Frame):
    """Vertical scroll container for rich detail tabs."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._scroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas, padding=(4, 6))
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scroll.pack(side="right", fill="y")
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self._canvas)
        self._bind_mousewheel(self.inner)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._win, width=event.width)

    def _bind_mousewheel(self, widget: tk.Misc) -> None:
        def _on_wheel(event: tk.Event) -> str | None:
            if getattr(event.widget, "winfo_class", lambda: "")() in ("Treeview", "TCombobox"):
                return None
            if event.delta:
                self._canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                self._canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self._canvas.yview_scroll(3, "units")
            return "break"

        widget.bind("<Enter>", lambda _e: widget.bind_all("<MouseWheel>", _on_wheel), add="+")
        widget.bind("<Leave>", lambda _e: widget.unbind_all("<MouseWheel>"), add="+")
        widget.bind("<Button-4>", _on_wheel, add="+")
        widget.bind("<Button-5>", _on_wheel, add="+")


def section_title(parent: tk.Misc, text: str, *, color: str = ACCENT) -> ttk.Label:
    lbl = ttk.Label(parent, text=text, font=SECTION_FONT, foreground=color)
    lbl.pack(anchor="w", pady=(10, 4))
    return lbl


def section_desc(parent: tk.Misc, text: str) -> ttk.Label:
    lbl = ttk.Label(parent, text=text, font=BODY_FONT, foreground=COL_MUTED, wraplength=900, justify="left")
    lbl.pack(anchor="w", pady=(0, 8))
    return lbl


def inline_spec_rows(parent: tk.Misc, rows: Iterable[tuple[str, Any]], *, label_width: int = 18) -> ttk.Frame:
    """Key-value rows with label and value on the same line."""
    frame = ttk.Frame(parent, relief="groove", borderwidth=1, padding=8)
    frame.pack(fill="x", pady=(0, 6))
    for key, val in rows:
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=str(key),
            width=label_width,
            anchor="w",
            font=(BODY_FONT[0], BODY_FONT[1], "bold"),
            foreground=COL_MUTED,
        ).pack(side="left")
        ttk.Label(row, text=fmt_val(val), anchor="w", font=BODY_FONT, wraplength=280).pack(
            side="left", fill="x", expand=True,
        )
    return frame


def metric_cards_grid(
    parent: tk.Misc,
    cards: Iterable[tuple[str, str]],
    *,
    columns: int = 4,
) -> ttk.Frame:
    """Small summary cards in a responsive grid."""
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=(0, 6))
    for i, (label, value) in enumerate(cards):
        card = ttk.Frame(frame, relief="solid", borderwidth=1, padding=(8, 6))
        card.grid(row=i // columns, column=i % columns, padx=4, pady=4, sticky="nsew")
        ttk.Label(
            card,
            text=str(label),
            font=(BODY_FONT[0], BODY_FONT[1], "bold"),
            foreground=COL_MUTED,
        ).pack(anchor="w")
        ttk.Label(card, text=str(value), font=("Segoe UI", 11, "bold")).pack(anchor="w")
    for c in range(columns):
        frame.columnconfigure(c, weight=1)
    return frame


def metric_table(parent: tk.Misc, rows: Iterable[tuple[str, Any]], *, label_width: int = 18) -> ttk.Frame:
    """Two-column metric table (label / value)."""
    frame = ttk.Frame(parent, relief="groove", borderwidth=1, padding=8)
    frame.pack(fill="x", pady=(0, 6))
    for key, val in rows:
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=str(key),
            width=label_width,
            anchor="w",
            font=(BODY_FONT[0], BODY_FONT[1], "bold"),
            foreground=COL_MUTED,
        ).pack(side="left")
        ttk.Label(row, text=fmt_val(val), anchor="w", font=BODY_FONT).pack(side="left", fill="x", expand=True)
    return frame


def dual_spec_sections(
    parent: tk.Misc,
    left_rows: Iterable[tuple[str, Any]],
    right_rows: Iterable[tuple[str, Any]],
    *,
    left_title: str = "Performance",
    right_title: str = "Risk & Returns",
    label_width: int = 26,
    extra_rows: Iterable[tuple[str, Any]] | None = None,
    extra_title: str = "Metrics Debug",
    extra_label_width: int | None = None,
) -> ttk.Frame:
    """Side-by-side labeled sections (2 columns, or 3 when ``extra_rows`` is set)."""
    container = ttk.Frame(parent)
    container.pack(fill="both", expand=True)

    def _fill_section(
        frame: ttk.LabelFrame,
        rows: Iterable[tuple[str, Any, ...]],
        *,
        width: int,
    ) -> None:
        for item in rows:
            key = item[0]
            val = item[1]
            color = pnl_foreground(item[2]) if len(item) >= 3 else None
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(
                row,
                text=str(key),
                width=width,
                anchor="w",
                font=(BODY_FONT[0], BODY_FONT[1], "bold"),
                foreground=COL_MUTED,
            ).pack(side="left", padx=(0, 12))
            val_lbl = ttk.Label(row, text=fmt_val(val), anchor="w", font=BODY_FONT)
            if color is not None:
                val_lbl.configure(foreground=color)
            val_lbl.pack(side="left", fill="x", expand=True)

    sections: list[tuple[str, Iterable[tuple[str, Any, ...]], tuple[int, int], int]] = [
        (left_title, left_rows, (0, 4), label_width),
        (right_title, right_rows, (4, 4) if extra_rows is not None else (4, 0), label_width),
    ]
    if extra_rows is not None:
        sections.append(
            (
                extra_title,
                extra_rows,
                (4, 0),
                extra_label_width if extra_label_width is not None else label_width,
            )
        )

    for title, rows, pad, width in sections:
        section = ttk.LabelFrame(container, text=title, padding=8)
        section.pack(side="left", fill="both", expand=True, padx=pad)
        _fill_section(section, rows, width=width)
    return container


def spec_grid(parent: tk.Misc, rows: Iterable[tuple[str, Any]], *, columns: int = 2) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=(0, 6))
    grid = ttk.Frame(frame, relief="groove", borderwidth=1, padding=8)
    grid.pack(fill="x")
    for idx, (key, val) in enumerate(rows):
        row, col = divmod(idx, columns)
        kf = ttk.Frame(grid)
        kf.grid(row=row, column=col * 2, sticky="nw", padx=(0, 8), pady=2)
        ttk.Label(kf, text=str(key), font=(BODY_FONT[0], BODY_FONT[1], "bold"), foreground=COL_MUTED).pack(anchor="w")
        ttk.Label(kf, text=fmt_val(val), font=BODY_FONT, wraplength=360, justify="left").pack(anchor="w")
    return frame


def kv_block(parent: tk.Misc, title: str, items: Iterable[tuple[str, str]]) -> ttk.LabelFrame:
    box = ttk.LabelFrame(parent, text=title, padding=8)
    box.pack(fill="x", pady=(0, 8))
    for key, val in items:
        row = ttk.Frame(box)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=key, width=22, anchor="w", foreground=COL_MUTED, font=BODY_FONT).pack(side="left")
        ttk.Label(row, text=val, anchor="w", font=BODY_FONT).pack(side="left", fill="x", expand=True)
    return box


def drift_bar_text(pct: float, *, blocks: int = 9) -> str:
    try:
        val = float(pct)
    except (TypeError, ValueError):
        return "░" * blocks
    val = max(0.0, min(100.0, val))
    filled = int(round(blocks * val / 100.0))
    return "█" * filled + "░" * (blocks - filled)


def drift_score_bars(parent: tk.Misc, scores: list[tuple[str, float]], *, bar_width: int = 200) -> ttk.Frame:
    frame = ttk.Frame(parent)
    if not scores:
        ttk.Label(frame, text="No drift scores available.", foreground=COL_MUTED).pack(anchor="w")
        return frame
    for label, pct in scores:
        row_f = ttk.Frame(frame)
        row_f.pack(fill="x", pady=3)
        ttk.Label(row_f, text=label, width=14, anchor="w", font=BODY_FONT).pack(side="left")
        ttk.Label(row_f, text=drift_bar_text(pct), font=MONO_FONT, foreground=ACCENT).pack(side="left", padx=(4, 8))
        bar_wrap = tk.Frame(row_f, height=8, bg="#e8eaed")
        bar_wrap.pack(side="left", fill="x", expand=True, padx=(0, 8))
        filled_px = max(2, int(bar_width * float(pct) / 100.0))
        tk.Frame(bar_wrap, width=filled_px, height=8, bg=ACCENT).pack(side="left", fill="y")
        ttk.Label(row_f, text=f"{float(pct):.0f}%", width=5, anchor="e", font=BODY_FONT).pack(side="right")
    return frame


def outlier_impact_card(parent: tk.Misc, impact: dict[str, Any]) -> ttk.LabelFrame:
    """Compact card for top-tier outlier contribution to Premium RMSE."""
    box = ttk.LabelFrame(parent, text="Outlier Impact", padding=10)
    if not impact:
        ttk.Label(box, text="Not available", foreground=COL_MUTED).pack(anchor="w")
        return box

    tier = str(impact.get("label") or "Top 1% rows")
    ttk.Label(box, text=tier, font=("Segoe UI", 10, "bold")).pack(anchor="w")
    ttk.Separator(box, orient="horizontal").pack(fill="x", pady=6)

    row_count = impact.get("row_count")
    if row_count is not None:
        ttk.Label(box, text=f"Rows: {int(row_count):,}", font=BODY_FONT).pack(anchor="w", pady=(0, 8))

    contrib = float(impact.get("contribution_pct") or 0)
    ttk.Label(
        box,
        text="Contribution to Premium RMSE",
        font=BODY_FONT,
        foreground=COL_MUTED,
    ).pack(anchor="w")
    bar_row = ttk.Frame(box)
    bar_row.pack(fill="x", pady=(4, 2))
    bar_color = COL_WARN if contrib >= 60 else ACCENT
    ttk.Label(bar_row, text=drift_bar_text(contrib, blocks=26), font=MONO_FONT, foreground=bar_color).pack(side="left")
    ttk.Label(bar_row, text=f"{contrib:.1f}%", font=("Segoe UI", 10, "bold"), foreground=bar_color).pack(side="left", padx=8)

    status = str(impact.get("status") or "—")
    status_color = COL_WARN if status in ("Extreme", "High") else COL_MUTED
    ttk.Label(box, text="Status", font=BODY_FONT, foreground=COL_MUTED).pack(anchor="w", pady=(10, 0))
    ttk.Label(box, text=status, font=("Segoe UI", 11, "bold"), foreground=status_color).pack(anchor="w")
    return box


def metrics_stage(
    parent: tk.Misc,
    title: str,
    description: str,
    metrics: Iterable[tuple[str, str]],
    *,
    color: str,
    footnote: str | None = None,
) -> ttk.Frame:
    outer = tk.Frame(parent, highlightbackground=color, highlightthickness=1, padx=8, pady=8)
    outer.pack(side="left", fill="both", expand=True, padx=4, pady=4)
    tk.Label(outer, text=title, font=SECTION_FONT, fg=color, anchor="w").pack(fill="x")
    tk.Label(outer, text=description, font=BODY_FONT, fg=COL_MUTED, wraplength=280, justify="left", anchor="w").pack(
        fill="x", pady=(2, 8),
    )
    grid = ttk.Frame(outer)
    grid.pack(fill="both", expand=True)
    for idx, (label, value) in enumerate(metrics):
        cell = ttk.Frame(grid, padding=(0, 4))
        cell.grid(row=idx // 2, column=idx % 2, sticky="nw", padx=4, pady=2)
        tk.Label(cell, text=value, font=("Segoe UI", 11, "bold"), fg=color).pack(anchor="w")
        ttk.Label(cell, text=label, font=BODY_FONT, foreground=COL_MUTED).pack(anchor="w")
    if footnote:
        tk.Label(outer, text=footnote, font=BODY_FONT, fg=COL_WARN if "unreliable" in footnote.lower() else COL_MUTED,
                 wraplength=280, justify="left", anchor="w").pack(fill="x", pady=(6, 0))
    return outer


def data_table(
    parent: tk.Misc,
    columns: list[tuple[str, str, int]],
    rows: list[tuple[Any, ...]],
    *,
    height: int = 8,
    expand: bool = True,
    style: str | None = None,
) -> ttk.Treeview:
    col_ids = [c[0] for c in columns]
    kwargs: dict[str, Any] = {
        "columns": col_ids,
        "show": "headings",
        "height": min(height, max(3, len(rows))),
    }
    if style:
        kwargs["style"] = style
    tree = ttk.Treeview(parent, **kwargs)
    for col_id, _label, width in columns:
        tree.heading(col_id, text=_label)
        tree.column(col_id, width=width, anchor="center" if col_id != col_ids[0] else "w")
    for row in rows:
        tree.insert("", "end", values=row)
    tree.pack(fill="both" if expand else "x", expand=expand, pady=(4, 8))
    return tree


def importance_list(parent: tk.Misc, features: list[dict[str, Any]], *, limit: int = 20) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=(0, 8))
    rows = sorted(features, key=lambda x: float(x.get("importance_pct") or 0), reverse=True)[:limit]
    if not rows:
        ttk.Label(frame, text="No feature importance data.", foreground=COL_MUTED).pack(anchor="w")
        return frame
    max_pct = max(float(r.get("importance_pct") or 0) for r in rows) or 0.01
    hdr = ttk.Frame(frame)
    hdr.pack(fill="x")
    ttk.Label(hdr, text="Feature", font=(BODY_FONT[0], BODY_FONT[1], "bold")).pack(side="left")
    ttk.Label(hdr, text="Final Model Gain", font=(BODY_FONT[0], BODY_FONT[1], "bold")).pack(side="right")
    for idx, row in enumerate(rows):
        feat = str(row.get("feature") or "")
        pct = float(row.get("importance_pct") or 0)
        row_f = ttk.Frame(frame)
        row_f.pack(fill="x", pady=2)
        ttk.Label(row_f, text=f"{idx + 1}. {feat}", font=BODY_FONT).pack(side="left")
        ttk.Label(row_f, text=f"{pct:.2f}%", font=BODY_FONT, foreground=ACCENT).pack(side="right", padx=(8, 0))
        bar_wrap = tk.Frame(row_f, height=6, bg="#e8eaed")
        bar_wrap.pack(fill="x", pady=(2, 0))
        width_px = max(4, int(400 * (pct / max_pct)))
        tk.Frame(bar_wrap, width=width_px, height=6, bg=ACCENT).pack(side="left", fill="y")

    return frame


def json_block(parent: tk.Misc, data: Any, *, height: int = 10) -> tk.Text:
    from tkinter import scrolledtext
    import json

    txt = scrolledtext.ScrolledText(parent, height=height, font=MONO_FONT, wrap="none")
    txt.pack(fill="both", expand=True, pady=(4, 8))
    try:
        txt.insert("end", json.dumps(data, indent=2, default=str))
    except Exception as exc:
        txt.insert("end", f"Error: {exc}")
    txt.configure(state="disabled")
    return txt
