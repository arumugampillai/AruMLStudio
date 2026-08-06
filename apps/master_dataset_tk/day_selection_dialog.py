"""Reusable trading-day multi-select dialog.

Mirrors the Master Dataset panel's checkbox day list (click a row to toggle,
Select all / Clear, explicit dates only) so any panel that needs a "Selected
days" picker (e.g. Feature Transformation → Auto) gets the same behaviour.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Iterable


def select_trading_days(
    parent: tk.Misc,
    *,
    days: list[str],
    initial: Iterable[str] | None = None,
    title: str = "Select Trading Days",
) -> list[str] | None:
    """Modal checkbox picker for trading days.

    Returns the sorted list of chosen days, or ``None`` if the dialog was
    cancelled (caller should leave the previous selection untouched).
    """
    ordered_days = sorted({str(d).strip() for d in days if str(d).strip()})
    chosen: set[str] = {str(d).strip() for d in (initial or []) if str(d).strip() in set(ordered_days)}
    result: dict[str, list[str] | None] = {"value": None}

    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("340x440")
    win.transient(parent.winfo_toplevel())
    win.grab_set()

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=f"{len(ordered_days)} trading day(s) available — click a row to toggle.",
        foreground="#666",
        wraplength=310,
    ).pack(anchor="w")

    tree_wrap = ttk.Frame(frame)
    tree_wrap.pack(fill="both", expand=True, pady=(6, 4))
    tree = ttk.Treeview(tree_wrap, columns=("sel", "day"), show="headings", height=14)
    tree.heading("sel", text="✓")
    tree.heading("day", text="Trading Day")
    tree.column("sel", width=32, anchor="center", stretch=False)
    tree.column("day", width=220, anchor="w")
    sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    hint_var = tk.StringVar()

    def _update_hint() -> None:
        hint_var.set(f"{len(chosen)} of {len(ordered_days)} selected")

    def _render() -> None:
        tree.delete(*tree.get_children())
        for day in ordered_days:
            mark = "☑" if day in chosen else "☐"
            tree.insert("", "end", iid=day, values=(mark, day))

    def _toggle(event: tk.Event) -> None:
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        iid = tree.identify_row(event.y)
        if not iid:
            return
        if iid in chosen:
            chosen.discard(iid)
        else:
            chosen.add(iid)
        _render()
        _update_hint()

    tree.bind("<Button-1>", _toggle)

    ttk.Label(frame, textvariable=hint_var, foreground="#888").pack(anchor="w")

    btns = ttk.Frame(frame)
    btns.pack(fill="x", pady=(6, 0))

    def _select_all() -> None:
        chosen.clear()
        chosen.update(ordered_days)
        _render()
        _update_hint()

    def _clear_all() -> None:
        chosen.clear()
        _render()
        _update_hint()

    ttk.Button(btns, text="Select all", command=_select_all).pack(side="left")
    ttk.Button(btns, text="Clear", command=_clear_all).pack(side="left", padx=(6, 0))

    action_row = ttk.Frame(frame)
    action_row.pack(fill="x", pady=(10, 0))

    def _on_ok() -> None:
        result["value"] = sorted(chosen)
        win.destroy()

    def _on_cancel() -> None:
        result["value"] = None
        win.destroy()

    ttk.Button(action_row, text="Cancel", command=_on_cancel).pack(side="right")
    ttk.Button(action_row, text="OK", command=_on_ok).pack(side="right", padx=(0, 6))

    win.protocol("WM_DELETE_WINDOW", _on_cancel)

    _render()
    _update_hint()

    win.update_idletasks()
    try:
        win.focus_set()
    except tk.TclError:
        pass
    win.wait_window()

    return result["value"]


__all__ = ["select_trading_days"]
