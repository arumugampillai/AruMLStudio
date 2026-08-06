"""Placeholder panel for sections not yet implemented in Tk."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class PlaceholderPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        title: str,
        description: str,
        phase: str | None = None,
        bullets: list[str] | None = None,
    ) -> None:
        super().__init__(master)
        self._title = title
        wrap = ttk.Frame(self, padding=24)
        wrap.pack(fill="both", expand=True)

        ttk.Label(wrap, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        if phase:
            ttk.Label(wrap, text=phase, foreground="#58a6ff").pack(anchor="w", pady=(4, 12))
        ttk.Label(wrap, text=description, wraplength=720, justify="left").pack(anchor="w", pady=(0, 12))
        if bullets:
            for item in bullets:
                ttk.Label(wrap, text=f"• {item}", wraplength=700, justify="left").pack(anchor="w", pady=2)

    def on_show(self) -> None:
        return
