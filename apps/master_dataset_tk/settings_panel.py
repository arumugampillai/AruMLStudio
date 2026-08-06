"""Settings page — project folder, data paths, and quick links."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .build_service import chart_data_dir
from .project_config import (
    bundled_chart_dir,
    config_path,
    ensure_project_data_dir,
    normalize_chart_dir,
    resolve_chart_dir_from_selection,
    resolve_master_data_dir,
    resolve_tick_data_dir,
    save_master_data_dir,
    save_project_config,
    save_tick_data_dir,
    validate_chart_dir,
    DEFAULT_MASTER_DATA_DIR,
)
from .ui_util import open_path


class SettingsPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_project_changed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_project_changed = on_project_changed
        self._project_var = tk.StringVar(value=chart_dir)
        self._tick_data_var = tk.StringVar(value=self._configured_tick_data_dir())
        self._master_data_var = tk.StringVar(value=self._configured_master_data_dir())
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._project_var.set(chart_dir)
        self._tick_data_var.set(self._configured_tick_data_dir())
        self._master_data_var.set(self._configured_master_data_dir())
        self._refresh_path_rows()
        if hasattr(self, "_disable_gil_var"):
            self._disable_gil_var.set(self._load_disable_gil_monitor())

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def _configured_tick_data_dir(self) -> str:
        from .project_config import load_project_config

        saved = str(load_project_config().get("tick_data_dir") or "").strip()
        if saved:
            return normalize_chart_dir(saved)
        from tick_data_paths import DEFAULT_TICK_DATA_DIR

        return DEFAULT_TICK_DATA_DIR

    def _configured_master_data_dir(self) -> str:
        from .project_config import load_project_config

        saved = str(load_project_config().get("master_data_dir") or "").strip()
        if saved:
            return normalize_chart_dir(saved)
        return DEFAULT_MASTER_DATA_DIR

    def _resolved_tick_data_dir(self) -> str:
        return resolve_tick_data_dir(self.chart_dir)

    def _resolved_master_data_dir(self) -> str:
        return resolve_master_data_dir(self.chart_dir)

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        ttk.Label(wrap, text="Settings", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))
        ttk.Label(
            wrap,
            text="ML Research Studio runs standalone — no chart server required.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 16))

        proj_fr = ttk.LabelFrame(wrap, text="Project Folder", padding=10)
        proj_fr.pack(fill="x", pady=(0, 16))
        ttk.Label(
            proj_fr,
            text="Point to the chart folder that contains data/ (or select the AruNeo repo root).",
            foreground="#888",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(proj_fr)
        row.pack(fill="x", pady=(0, 6))
        ttk.Entry(row, textvariable=self._project_var, width=78).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_project).pack(side="left", padx=(6, 0))

        btn_row = ttk.Frame(proj_fr)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="Open Project Folder", command=lambda: open_path(self.chart_dir)).pack(side="left")
        ttk.Button(btn_row, text="Use Bundled Default", command=self._use_bundled).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Apply Project", command=self._apply_project).pack(side="left", padx=(8, 0))

        ttk.Label(
            proj_fr,
            text=f"Config file: {config_path()}",
            foreground="#666",
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(8, 0))

        tick_fr = ttk.LabelFrame(wrap, text="Tick Data Folder", padding=10)
        tick_fr.pack(fill="x", pady=(0, 16))
        ttk.Label(
            tick_fr,
            text="Primary location for angel_market_YYYY-MM-DD.db tick databases (live writer and replay).",
            foreground="#888",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 8))

        tick_row = ttk.Frame(tick_fr)
        tick_row.pack(fill="x", pady=(0, 6))
        ttk.Entry(tick_row, textvariable=self._tick_data_var, width=78).pack(side="left", fill="x", expand=True)
        ttk.Button(tick_row, text="Browse…", command=self._browse_tick_data).pack(side="left", padx=(6, 0))

        tick_btn_row = ttk.Frame(tick_fr)
        tick_btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(
            tick_btn_row,
            text="Open Tick Data Folder",
            command=lambda: open_path(self._resolved_tick_data_dir()),
        ).pack(side="left")
        ttk.Button(tick_btn_row, text="Apply Tick Folder", command=self._apply_tick_data).pack(side="left", padx=(8, 0))

        self._tick_resolved_var = tk.StringVar()
        ttk.Label(
            tick_fr,
            textvariable=self._tick_resolved_var,
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=760,
        ).pack(anchor="w", pady=(8, 0))

        master_fr = ttk.LabelFrame(wrap, text="Master Dataset Folder", padding=10)
        master_fr.pack(fill="x", pady=(0, 16))
        ttk.Label(
            master_fr,
            text="Location for master_dataset_*.db, master_insert_*.expected.json, and build prefs.",
            foreground="#888",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 8))

        master_row = ttk.Frame(master_fr)
        master_row.pack(fill="x", pady=(0, 6))
        ttk.Entry(master_row, textvariable=self._master_data_var, width=78).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(master_row, text="Browse…", command=self._browse_master_data).pack(
            side="left", padx=(6, 0)
        )

        master_btn_row = ttk.Frame(master_fr)
        master_btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(
            master_btn_row,
            text="Open Master Dataset Folder",
            command=lambda: open_path(self._resolved_master_data_dir()),
        ).pack(side="left")
        ttk.Button(
            master_btn_row, text="Apply Master Folder", command=self._apply_master_data
        ).pack(side="left", padx=(8, 0))

        self._master_resolved_var = tk.StringVar()
        ttk.Label(
            master_fr,
            textvariable=self._master_resolved_var,
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=760,
        ).pack(anchor="w", pady=(8, 0))

        self._paths_host = ttk.Frame(wrap)
        self._paths_host.pack(fill="x")
        self._refresh_path_rows()

        diag_fr = ttk.LabelFrame(wrap, text="Diagnostics", padding=10)
        diag_fr.pack(fill="x", pady=(16, 0))
        self._disable_gil_var = tk.BooleanVar(value=self._load_disable_gil_monitor())
        ttk.Checkbutton(
            diag_fr,
            text="Disable GIL / event-loop diagnostics during builds (recommended)",
            variable=self._disable_gil_var,
            command=self._save_gil_monitor_pref,
        ).pack(anchor="w")
        ttk.Label(
            diag_fr,
            text="When enabled, diagnostics use heavy profiling and slow builds. Leave disabled unless investigating UI lag.",
            foreground="#888",
            wraplength=760,
        ).pack(anchor="w", pady=(6, 0))

    def _refresh_path_rows(self) -> None:
        for child in self._paths_host.winfo_children():
            child.destroy()

        ttk.Label(self._paths_host, text="Data paths", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        paths = [
            ("Chart directory", self.chart_dir),
            ("Tick data directory", self._resolved_tick_data_dir()),
            ("Master dataset folder", self._resolved_master_data_dir()),
            ("Data directory", self._data_dir()),
            ("Models", os.path.join(self._data_dir(), "models")),
            ("Datasets (exports)", os.path.join(self._data_dir(), "datasets")),
            ("Prediction runs DB", os.path.join(self._data_dir(), "prediction_runs", "registry.db")),
        ]
        for label, path in paths:
            row = ttk.Frame(self._paths_host)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=20).pack(side="left")
            var = tk.StringVar(value=path)
            ttk.Entry(row, textvariable=var, state="readonly", width=70).pack(side="left", padx=4)
            ttk.Button(row, text="Open", command=lambda p=path: open_path(p)).pack(side="left")

        self._tick_resolved_var.set(f"Resolved tick path: {self._resolved_tick_data_dir()}")
        self._master_resolved_var.set(
            f"Resolved master path: {self._resolved_master_data_dir()}"
        )

    def _browse_tick_data(self) -> None:
        initial = self._tick_data_var.get().strip() or self._resolved_tick_data_dir()
        if not os.path.isdir(initial):
            initial = self._resolved_tick_data_dir()
        picked = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Select tick data folder",
            initialdir=initial,
        )
        if picked:
            self._tick_data_var.set(normalize_chart_dir(picked))

    def _apply_tick_data(self) -> None:
        raw = self._tick_data_var.get().strip()
        if not raw:
            messagebox.showinfo("Tick Data Folder", "Choose a folder first.", parent=self.winfo_toplevel())
            return
        tick_dir = normalize_chart_dir(raw)
        if not os.path.isdir(tick_dir):
            messagebox.showerror(
                "Tick Data Folder",
                f"Folder does not exist:\n{tick_dir}",
                parent=self.winfo_toplevel(),
            )
            return
        save_tick_data_dir(tick_dir)
        self._tick_data_var.set(tick_dir)
        self._refresh_path_rows()
        messagebox.showinfo(
            "Tick Data Folder",
            f"Tick data folder set to:\n{tick_dir}\n\nResolved path:\n{self._resolved_tick_data_dir()}",
            parent=self.winfo_toplevel(),
        )

    def _browse_master_data(self) -> None:
        initial = self._master_data_var.get().strip() or self._resolved_master_data_dir()
        if not os.path.isdir(initial):
            initial = self._resolved_master_data_dir()
        picked = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Select master dataset folder",
            initialdir=initial,
        )
        if picked:
            self._master_data_var.set(normalize_chart_dir(picked))

    def _apply_master_data(self) -> None:
        raw = self._master_data_var.get().strip()
        if not raw:
            messagebox.showinfo(
                "Master Dataset Folder",
                "Choose a folder first.",
                parent=self.winfo_toplevel(),
            )
            return
        master_dir = normalize_chart_dir(raw)
        os.makedirs(master_dir, exist_ok=True)
        if not os.path.isdir(master_dir):
            messagebox.showerror(
                "Master Dataset Folder",
                f"Folder does not exist:\n{master_dir}",
                parent=self.winfo_toplevel(),
            )
            return
        save_master_data_dir(master_dir)
        self._master_data_var.set(master_dir)
        self._refresh_path_rows()
        messagebox.showinfo(
            "Master Dataset Folder",
            f"Master dataset folder set to:\n{master_dir}\n\nResolved path:\n{self._resolved_master_data_dir()}",
            parent=self.winfo_toplevel(),
        )

    def _load_disable_gil_monitor(self) -> bool:
        from .build_config_prefs import load_build_config_prefs

        studio = (load_build_config_prefs(self.chart_dir) or {}).get("studio") or {}
        return bool(studio.get("disable_gil_monitor", True))

    def _save_gil_monitor_pref(self) -> None:
        from .build_config_prefs import load_build_config_prefs, save_build_config_prefs

        existing = load_build_config_prefs(self.chart_dir) or {}
        studio = dict(existing.get("studio") or {})
        studio["disable_gil_monitor"] = bool(self._disable_gil_var.get())
        save_build_config_prefs(self.chart_dir, {"studio": studio})

    def _browse_project(self) -> None:
        initial = self._project_var.get().strip() or self.chart_dir or bundled_chart_dir()
        if not os.path.isdir(initial):
            initial = bundled_chart_dir()
        picked = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Select project folder",
            initialdir=initial,
        )
        if picked:
            self._project_var.set(resolve_chart_dir_from_selection(picked))

    def _use_bundled(self) -> None:
        self._project_var.set(bundled_chart_dir())

    def _apply_project(self) -> None:
        raw = self._project_var.get().strip()
        if not raw:
            messagebox.showinfo("Project Folder", "Choose a folder first.", parent=self.winfo_toplevel())
            return
        chart_dir = resolve_chart_dir_from_selection(raw)
        ok, err = validate_chart_dir(chart_dir)
        if not ok:
            messagebox.showerror("Project Folder", err, parent=self.winfo_toplevel())
            return
        ensure_project_data_dir(chart_dir)
        save_project_config(chart_dir)
        self.chart_dir = chart_dir
        self._project_var.set(chart_dir)
        self._refresh_path_rows()
        if self._on_project_changed:
            self._on_project_changed(chart_dir)
        messagebox.showinfo(
            "Project Folder",
            f"Project set to:\n{chart_dir}\n\nAll panels now use this data folder.",
            parent=self.winfo_toplevel(),
        )
