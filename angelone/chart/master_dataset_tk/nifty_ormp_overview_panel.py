"""NIFTY ORMP — tabbed hub (Overview / Builds / Dataset Builder / Feature Explorer)."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Any

from .nifty_ormp_builds_panel import NiftyOrmpBuildsPanel
from .nifty_ormp_dataset_builder_panel import NiftyOrmpDatasetBuilderPanel
from .ormp_service import format_size, overview_snapshot
from .placeholder_panel import PlaceholderPanel
from .ui_util import open_path


class NiftyOrmpOverviewPanel(ttk.Frame):
    """Historical Data → ORMP Overview (tabs for Builds, Dataset Builder, Feature Explorer)."""

    def __init__(self, master: tk.Misc, *, chart_dir: str = "") -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._status_var = tk.StringVar(value="Open this page to load ORMP status.")
        self._build_choice_var = tk.StringVar(value="")
        self._builds_by_label: dict[str, str] = {}
        self._detail_vars: dict[str, tk.StringVar] = {
            "version": tk.StringVar(value="—"),
            "build": tk.StringVar(value="—"),
            "params": tk.StringVar(value="—"),
            "coverage": tk.StringVar(value="—"),
            "rows": tk.StringVar(value="—"),
            "size": tk.StringVar(value="—"),
            "built": tk.StringVar(value="—"),
            "path": tk.StringVar(value="—"),
            "candle": tk.StringVar(value="—"),
            "outputs": tk.StringVar(value="—"),
            "count": tk.StringVar(value="—"),
        }
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self.builds_panel.set_chart_dir(chart_dir)
        self.dataset_builder_panel.set_chart_dir(chart_dir)

    def on_show(self) -> None:
        self.refresh_overview()
        self.builds_panel.refresh_list()
        self.dataset_builder_panel.refresh()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 0))
        header.pack(fill="x")
        ttk.Label(header, text="NIFTY ORMP", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Standalone Open Relative Market Profile research sandbox. "
                "Does not modify Master Dataset."
            ),
            foreground="#888",
            wraplength=900,
        ).pack(anchor="w", pady=(4, 8))

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        overview_tab = ttk.Frame(self._notebook)
        self._notebook.add(overview_tab, text="Overview")
        self._build_overview_tab(overview_tab)

        self.builds_panel = NiftyOrmpBuildsPanel(self._notebook, chart_dir=self.chart_dir)
        self._notebook.add(self.builds_panel, text="Builds")

        self.dataset_builder_panel = NiftyOrmpDatasetBuilderPanel(
            self._notebook, chart_dir=self.chart_dir
        )
        self._notebook.add(self.dataset_builder_panel, text="Dataset Builder")

        self.feature_explorer_panel = PlaceholderPanel(
            self._notebook,
            title="ORMP Feature Explorer",
            phase="Later",
            description=(
                "Browse ORMP feature formulas, distributions, and sample values for a selected build."
            ),
        )
        self._notebook.add(self.feature_explorer_panel, text="Feature Explorer")

    def _build_overview_tab(self, parent: ttk.Frame) -> None:
        wrap = ttk.Frame(parent, padding=16)
        wrap.pack(fill="both", expand=True)

        toolbar = ttk.Frame(wrap)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="Refresh", command=self.refresh_overview).pack(side="left")
        ttk.Label(toolbar, text="Build:").pack(side="left", padx=(16, 4))
        self._build_combo = ttk.Combobox(
            toolbar,
            textvariable=self._build_choice_var,
            state="readonly",
            width=56,
        )
        self._build_combo.pack(side="left", fill="x", expand=True)
        self._build_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_selected())

        status = ttk.LabelFrame(wrap, text="Status", padding=10)
        status.pack(fill="x", pady=(0, 10))
        rows = [
            ("ORMP engine", "version"),
            ("Builds on disk", "count"),
            ("Candle source", "candle"),
            ("Outputs folder", "outputs"),
        ]
        for label, key in rows:
            fr = ttk.Frame(status)
            fr.pack(fill="x", pady=2)
            ttk.Label(fr, text=f"{label}:", width=16).pack(side="left")
            ttk.Label(fr, textvariable=self._detail_vars[key], wraplength=720).pack(
                side="left", fill="x", expand=True
            )

        detail = ttk.LabelFrame(wrap, text="Selected build", padding=10)
        detail.pack(fill="x", pady=(0, 10))
        drows = [
            ("Name", "build"),
            ("Parameters", "params"),
            ("Coverage", "coverage"),
            ("Rows / days", "rows"),
            ("File size", "size"),
            ("Built at", "built"),
            ("Path", "path"),
        ]
        for label, key in drows:
            fr = ttk.Frame(detail)
            fr.pack(fill="x", pady=2)
            ttk.Label(fr, text=f"{label}:", width=16).pack(side="left")
            ttk.Label(fr, textvariable=self._detail_vars[key], wraplength=720).pack(
                side="left", fill="x", expand=True
            )

        path_fr = ttk.Frame(wrap)
        path_fr.pack(fill="x", pady=(0, 8))
        ttk.Button(path_fr, text="Open outputs folder", command=self._open_outputs).pack(
            side="left"
        )
        ttk.Button(path_fr, text="Open selected DB folder", command=self._open_selected).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(wrap, textvariable=self._status_var, foreground="#888").pack(anchor="w")

    def _on_tab_changed(self, _event: object = None) -> None:
        try:
            tab = self._notebook.select()
            widget = self._notebook.nametowidget(tab)
            tabs = self._notebook.tabs()
        except tk.TclError:
            return
        if widget is self.builds_panel:
            self.builds_panel.refresh_list()
        elif widget is self.dataset_builder_panel:
            self.dataset_builder_panel.refresh()
        elif tabs and tab == tabs[0]:
            self.refresh_overview()

    def refresh_overview(self) -> None:
        try:
            snap = overview_snapshot(self.chart_dir)
        except Exception as exc:  # noqa: BLE001
            self._status_var.set(f"Failed to load ORMP status: {exc}")
            return
        self._detail_vars["version"].set(str(snap.get("ormp_version") or "—"))
        self._detail_vars["count"].set(str(snap.get("build_count") or 0))
        candle = snap.get("candle_db_path") or ""
        exists = "found" if snap.get("candle_db_exists") else "missing"
        self._detail_vars["candle"].set(f"{candle}  ({exists})")
        self._detail_vars["outputs"].set(str(snap.get("outputs_dir") or "—"))

        builds = snap.get("builds") or []
        labels: list[str] = []
        self._builds_by_label = {}
        for b in builds:
            label = f"{b.display_name}  ·  {b.built_at_label}"
            labels.append(label)
            self._builds_by_label[label] = b.build_id
        self._build_combo["values"] = labels
        if labels:
            cur = self._build_choice_var.get()
            if cur not in self._builds_by_label:
                self._build_choice_var.set(labels[0])
            self._apply_selected(builds=builds)
            self._status_var.set(f"{len(builds)} ORMP build(s) available.")
        else:
            self._build_choice_var.set("")
            self._clear_selected()
            self._status_var.set("No ORMP builds yet — create one on the Builds tab.")

    def _apply_selected(self, *, builds: list[Any] | None = None) -> None:
        label = self._build_choice_var.get()
        build_id = self._builds_by_label.get(label)
        if not build_id:
            self._clear_selected()
            return
        if builds is None:
            snap = overview_snapshot(self.chart_dir, build_id=build_id)
            b = snap.get("selected")
        else:
            b = next((x for x in builds if x.build_id == build_id), None)
        if b is None:
            self._clear_selected()
            return
        bs = f"{b.band_size_pct:g}%" if b.band_size_pct is not None else "?"
        self._detail_vars["build"].set(b.display_name)
        self._detail_vars["params"].set(
            f"band_size={bs}  ·  price_source={b.price_source}  ·  path_mode={b.path_mode}"
        )
        cov = "—"
        if b.from_date or b.to_date:
            cov = f"{b.from_date or '?'} → {b.to_date or '?'}"
        self._detail_vars["coverage"].set(cov)
        rows = f"{b.rows:,}" if b.rows is not None else "—"
        days = f"{b.days:,}" if b.days is not None else "—"
        self._detail_vars["rows"].set(f"{rows} rows  ·  {days} days")
        self._detail_vars["size"].set(format_size(b.file_size_bytes))
        self._detail_vars["built"].set(b.built_at_label)
        self._detail_vars["path"].set(b.path)

    def _clear_selected(self) -> None:
        for key in ("build", "params", "coverage", "rows", "size", "built", "path"):
            self._detail_vars[key].set("—")

    def _open_outputs(self) -> None:
        snap = overview_snapshot(self.chart_dir)
        path = snap.get("outputs_dir") or ""
        if path:
            open_path(path)

    def _open_selected(self) -> None:
        path = self._detail_vars["path"].get()
        if path and path != "—":
            open_path(os.path.dirname(path))
