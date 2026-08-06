"""Comparison shell — Model / Fold / Dataset comparison tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .dataset_comparison_panel import DatasetComparisonPanel
from .fold_comparison_panel import FoldComparisonPanel
from .model_comparison_panel import ModelComparisonPanel
from .ui_state import get_ui_state_manager


class ComparisonPanel(ttk.Frame):
    """Hosts Model / Fold / Dataset comparison UIs under one nav page."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._ui_state = get_ui_state_manager()
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew")

        self.model = ModelComparisonPanel(self._notebook, chart_dir=chart_dir)
        self.fold = FoldComparisonPanel(self._notebook, chart_dir=chart_dir)
        self.dataset = DatasetComparisonPanel(self._notebook, chart_dir=chart_dir)

        self._tabs: dict[str, ttk.Frame] = {
            "model": self.model,
            "fold": self.fold,
            "dataset": self.dataset,
        }
        self._notebook.add(self.model, text="Model")
        self._notebook.add(self.fold, text="Fold")
        self._notebook.add(self.dataset, text="Dataset")

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._ui_state.bind_notebook(self._notebook, "comparison.tab")

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        for panel in self._tabs.values():
            if hasattr(panel, "chart_dir"):
                panel.chart_dir = chart_dir

    def select_tab(self, key: str) -> None:
        panel = self._tabs.get(key)
        if panel is None:
            return
        try:
            self._notebook.select(panel)
        except tk.TclError:
            return
        self._show_active_tab()

    def on_show(self) -> None:
        self._show_active_tab()

    def _active_panel(self) -> ttk.Frame | None:
        try:
            selected = self._notebook.select()
        except tk.TclError:
            return None
        if not selected:
            return None
        widget = self.nametowidget(selected)
        for panel in self._tabs.values():
            if panel is widget:
                return panel
        return None

    def _show_active_tab(self) -> None:
        panel = self._active_panel()
        on_show = getattr(panel, "on_show", None) if panel is not None else None
        if callable(on_show):
            on_show()

    def _on_tab_changed(self, _event: object | None = None) -> None:
        self._show_active_tab()
