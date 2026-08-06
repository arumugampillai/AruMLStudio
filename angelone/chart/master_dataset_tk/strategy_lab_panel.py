"""Strategies shell — Registry / Prediction Runs / Simulation / Leaderboard tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .prediction_runs_panel import PredictionRunsPanel
from .research_lab_panel import ResearchLabPanel
from .strategy_registry_panel import StrategyRegistryPanel
from .strategy_simulation_panel import StrategySimulationPanel
from .ui_state import get_ui_state_manager


class StrategiesPanel(ttk.Frame):
    """Hosts strategy registry, prediction runs, simulation, and leaderboard under one nav page."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_model: Callable[[str], None] | None = None,
        on_open_fold_replay: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._ui_state = get_ui_state_manager()
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew")

        self.strategies = StrategyRegistryPanel(self._notebook, chart_dir=chart_dir)
        self.prediction_runs = PredictionRunsPanel(
            self._notebook,
            chart_dir=chart_dir,
            on_open_model=on_open_model,
            on_open_fold_replay=on_open_fold_replay,
        )
        self.simulation = StrategySimulationPanel(self._notebook, chart_dir=chart_dir)
        self.leaderboard = ResearchLabPanel(self._notebook, chart_dir=chart_dir)

        self._tabs: dict[str, ttk.Frame] = {
            "strategies": self.strategies,
            "prediction_runs": self.prediction_runs,
            "simulation": self.simulation,
            "leaderboard": self.leaderboard,
        }
        self._notebook.add(self.strategies, text="Strategies")
        self._notebook.add(self.prediction_runs, text="Prediction Runs")
        self._notebook.add(self.simulation, text="Simulation")
        self._notebook.add(self.leaderboard, text="Leaderboard")

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._ui_state.bind_notebook(self._notebook, "strategies.tab")

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
