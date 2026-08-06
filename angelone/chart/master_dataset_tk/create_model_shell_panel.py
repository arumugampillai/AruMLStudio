"""Create Model shell — Create Model / Outcome Labels tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .model_builder_panel import CreateModelPanel
from .outcome_label_engine_panel import OutcomeLabelEnginePanel
from .ui_state import get_ui_state_manager


class CreateModelShell(ttk.Frame):
    """Hosts Create Model and Outcome Labels under one Model Builder nav page."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_registry: Callable[[str], None] | None = None,
        on_title_changed: Callable[[], None] | None = None,
        on_label_run_created: Callable[[str], None] | None = None,
        on_open_create_model: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_title_changed = on_title_changed
        self._on_open_create_model = on_open_create_model
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._ui_state = get_ui_state_manager()
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew")

        self.create = CreateModelPanel(
            self._notebook,
            chart_dir=chart_dir,
            on_open_registry=on_open_registry,
            on_title_changed=on_title_changed,
            on_open_outcome_label_engine=self._open_outcome_labels_tab,
        )
        self.outcome_labels = OutcomeLabelEnginePanel(
            self._notebook,
            chart_dir=chart_dir,
            on_label_run_created=on_label_run_created,
            on_open_create_model=self._return_to_create_model,
        )

        self._tabs: dict[str, ttk.Frame] = {
            "create": self.create,
            "ole": self.outcome_labels,
        }
        self._notebook.add(self.create, text="Create Model")
        self._notebook.add(self.outcome_labels, text="Outcome Labels")

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._ui_state.bind_notebook(self._notebook, "builder.create.tab")

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        for panel in self._tabs.values():
            if hasattr(panel, "set_chart_dir"):
                panel.set_chart_dir(chart_dir)
            else:
                if hasattr(panel, "chart_dir"):
                    panel.chart_dir = chart_dir
                data_dir = getattr(panel, "_data_dir", None)
                if data_dir is not None:
                    from .build_service import chart_data_dir

                    panel._data_dir = chart_data_dir(chart_dir)

    def active_tab_key(self) -> str:
        panel = self._active_panel()
        for key, candidate in self._tabs.items():
            if candidate is panel:
                return key
        return "create"

    def select_tab(self, key: str, *, from_nav: bool = True) -> None:
        panel = self._tabs.get(key)
        if panel is None:
            return
        try:
            self._notebook.select(panel)
        except tk.TclError:
            return
        self._show_active_tab(from_nav=from_nav)

    def on_show(self) -> None:
        self._show_active_tab(from_nav=True)

    def page_title(self) -> str:
        if self.active_tab_key() == "ole":
            return "Outcome Labels"
        return self.create.page_title()

    def _open_outcome_labels_tab(self, prefill: dict[str, Any] | None = None) -> None:
        self.select_tab("ole", from_nav=True)
        try:
            self.outcome_labels.apply_prefill(prefill)
        except Exception:
            pass
        if self._on_title_changed:
            self._on_title_changed()

    def _return_to_create_model(self, payload: dict[str, Any] | None = None) -> None:
        self.select_tab("create", from_nav=False)
        if self._on_open_create_model:
            self._on_open_create_model(payload)
        if self._on_title_changed:
            self._on_title_changed()

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

    def _show_active_tab(self, *, from_nav: bool = False) -> None:
        panel = self._active_panel()
        if panel is self.create and not from_nav:
            # Switching back from Outcome Labels must not wipe the form.
            load = getattr(panel, "_load_catalog", None)
            if callable(load):
                try:
                    load(lazy=True)
                except Exception:
                    pass
            return
        on_show = getattr(panel, "on_show", None) if panel is not None else None
        if callable(on_show):
            on_show()

    def _on_tab_changed(self, _event: object | None = None) -> None:
        self._show_active_tab(from_nav=False)
        if self._on_title_changed:
            self._on_title_changed()
