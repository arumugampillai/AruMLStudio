"""IDE-style persistent global build status bar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .build_progress_manager import BuildProgressManager, BuildStatusSnapshot

_STATUS_BG = "#ececec"
_STATUS_FG = "#1f1f1f"
_STATUS_ACTIVE_BG = "#d8ecff"


class GlobalBuildStatusBar(ttk.Frame):
    """Bottom-docked status bar — one compact horizontal row (~30px)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        manager: BuildProgressManager,
        on_details: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, padding=0)
        self._manager = manager
        self._on_details = on_details
        self._on_cancel = on_cancel
        self._build_ui()
        manager.subscribe(self.on_snapshot)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("GlobalStatusBar.TFrame", background=_STATUS_BG)
        style.configure(
            "GlobalStatusBar.TLabel",
            background=_STATUS_BG,
            foreground=_STATUS_FG,
            font=("Segoe UI", 9),
        )
        style.configure(
            "GlobalStatusBarActive.TFrame",
            background=_STATUS_ACTIVE_BG,
        )
        style.configure(
            "GlobalStatusBarActive.TLabel",
            background=_STATUS_ACTIVE_BG,
            foreground=_STATUS_FG,
            font=("Segoe UI", 9),
        )

        ttk.Separator(self, orient="horizontal").pack(fill="x", side="top")

        self._outer = ttk.Frame(self, style="GlobalStatusBar.TFrame", padding=(8, 4))
        self._outer.pack(fill="x")

        row = ttk.Frame(self._outer, style="GlobalStatusBar.TFrame")
        row.pack(fill="x")
        self._row = row

        self._icon_var = tk.StringVar(value="⚪")
        self._icon_lbl = ttk.Label(
            row,
            textvariable=self._icon_var,
            width=2,
            style="GlobalStatusBar.TLabel",
        )
        self._icon_lbl.pack(side="left", padx=(0, 6))

        self._title_var = tk.StringVar(value="Ready")
        self._title_lbl = ttk.Label(
            row,
            textvariable=self._title_var,
            width=18,
            style="GlobalStatusBar.TLabel",
            font=("Segoe UI", 9, "bold"),
        )
        self._title_lbl.pack(side="left", padx=(0, 8))

        self._progress = ttk.Progressbar(row, length=120, mode="determinate", maximum=100)
        self._progress.pack(side="left", padx=(0, 6))

        self._percent_var = tk.StringVar(value="")
        self._percent_lbl = ttk.Label(
            row,
            textvariable=self._percent_var,
            width=5,
            style="GlobalStatusBar.TLabel",
            font=("Consolas", 9),
        )
        self._percent_lbl.pack(side="left", padx=(0, 8))

        self._detail_var = tk.StringVar(value="Global build status — idle")
        self._detail_lbl = ttk.Label(
            row,
            textvariable=self._detail_var,
            style="GlobalStatusBar.TLabel",
            font=("Consolas", 9),
            anchor="w",
        )
        self._detail_lbl.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._details_btn = ttk.Button(row, text="Details", width=8, command=self._handle_details)
        self._details_btn.pack(side="right", padx=(4, 0))

        self._cancel_btn = ttk.Button(
            row,
            text="Cancel",
            width=8,
            command=self._handle_cancel,
            state="disabled",
        )
        self._cancel_btn.pack(side="right", padx=(4, 0))

    def _set_active_style(self, active: bool) -> None:
        frame_style = "GlobalStatusBarActive.TFrame" if active else "GlobalStatusBar.TFrame"
        label_style = "GlobalStatusBarActive.TLabel" if active else "GlobalStatusBar.TLabel"
        self._outer.configure(style=frame_style)
        self._row.configure(style=frame_style)
        for widget in (self._icon_lbl, self._title_lbl, self._percent_lbl, self._detail_lbl):
            widget.configure(style=label_style)

    def on_snapshot(self, snap: BuildStatusSnapshot) -> None:
        self._icon_var.set(snap.status_icon)
        self._title_var.set(snap.job_title)
        self._set_active_style(snap.active)
        if snap.active:
            self._progress.configure(value=max(0.0, min(100.0, snap.percent)))
            self._percent_var.set(f"{snap.percent:.0f}%")
            parts = [
                snap.task_label,
                snap.day_label,
                f"Samples {snap.samples_label}",
                snap.speed_label,
                f"RAM {snap.ram_label}",
                f"Workers {snap.worker_count}",
                f"Elapsed {snap.elapsed_label}",
                f"ETA {snap.eta_label}",
            ]
            self._detail_var.set(" | ".join(p for p in parts if p and p != "—"))
            self._cancel_btn.configure(state="normal" if snap.cancellable else "disabled")
            self._details_btn.configure(state="normal")
        else:
            self._progress.configure(value=0.0)
            self._percent_var.set("")
            if snap.status == "completed":
                self._detail_var.set(snap.message or "Build finished successfully.")
            elif snap.status == "failed":
                self._detail_var.set(snap.message or "Build failed.")
            elif snap.status == "cancelled":
                self._detail_var.set("Build cancelled.")
            else:
                self._detail_var.set(f"Ready | RAM {snap.ram_label}")
            self._cancel_btn.configure(state="disabled")
            self._details_btn.configure(state="normal" if snap.status != "idle" else "disabled")

    def _handle_details(self) -> None:
        if callable(self._on_details):
            self._on_details()

    def _handle_cancel(self) -> None:
        if callable(self._on_cancel):
            self._on_cancel()
