"""Feature selection side panel — opens beside build configuration."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .feature_selection_picker import FeatureSelectionPicker
from .fold_replay_widgets import place_toplevel_beside_main


def open_feature_selection_panel(
    master: tk.Misc,
    *,
    registry: dict[str, Any],
    profile_var: tk.StringVar,
    initial_config: dict[str, Any],
    on_apply: Callable[[dict[str, Any]], None],
    excluded_features: set[str] | frozenset[str] | None = None,
    chart_dir: str | None = None,
    feature_project_id: str | None = None,
) -> tk.Toplevel:
    """Open picker beside main window; Apply commits config and closes."""
    win = tk.Toplevel(master)
    win.title("Feature selection")
    win.transient(master.winfo_toplevel())

    body = ttk.Frame(win, padding=8)
    body.pack(fill="both", expand=True)

    picker = FeatureSelectionPicker(
        registry,
        profile_var=profile_var,
        on_change=None,
        chart_dir=chart_dir,
        feature_project_id=feature_project_id,
    )
    if excluded_features:
        picker.set_excluded_features(excluded_features)
    picker.apply_config(initial_config)
    picker.mount(body, show_search=True, always_expand_features=True, canvas_height=480)

    btn_row = ttk.Frame(win, padding=8)
    btn_row.pack(fill="x")

    def apply_and_close() -> None:
        on_apply(picker.get_config())
        picker.unmount()
        win.destroy()

    def cancel() -> None:
        picker.unmount()
        win.destroy()

    ttk.Button(btn_row, text="Cancel", command=cancel).pack(side="right", padx=4)
    ttk.Button(btn_row, text="Apply", command=apply_and_close).pack(side="right")

    win.protocol("WM_DELETE_WINDOW", cancel)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win
