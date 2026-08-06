"""Gap policy tab — enable gap reset + threshold for Create Dataset builds."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.gap_policy import (
    GAP_PRESET_SEC,
    default_gap_policy,
    normalize_gap_policy,
)


class GapPolicyTab:
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._enabled_var = tk.BooleanVar(value=True)
        self._preset_var = tk.StringVar(value="20")
        self._hint_var = tk.StringVar(value="")
        self._threshold_widgets: list[ttk.Widget] = []
        self._build(parent)
        self.apply_config(default_gap_policy())

    def apply_config(self, cfg: dict[str, Any]) -> None:
        doc = normalize_gap_policy(cfg)
        self._enabled_var.set(bool(doc.get("enabled", True)))
        preset = str(doc.get("preset") or "20")
        if preset == "custom":
            # Snap custom values onto the nearest UI preset for the radios.
            sec = float(doc.get("gapMaxSec") or 20)
            preset = str(min(GAP_PRESET_SEC, key=lambda p: abs(p - sec)))
        elif preset.isdigit() and int(preset) not in GAP_PRESET_SEC:
            sec = int(preset)
            preset = str(min(GAP_PRESET_SEC, key=lambda p: abs(p - sec)))
        self._preset_var.set(preset)
        self._sync_enabled_state()
        self._update_hint()

    def get_config(self) -> dict[str, Any]:
        preset = str(self._preset_var.get() or "20").strip()
        if preset.isdigit() and int(preset) in GAP_PRESET_SEC:
            gap_max = float(preset)
        else:
            gap_max = 20.0
            preset = "20"
        return normalize_gap_policy({
            "enabled": bool(self._enabled_var.get()),
            "preset": preset,
            "gapMaxSec": gap_max,
            "applied": True,
        })

    def _notify(self) -> None:
        self._update_hint()
        if self._on_change:
            self._on_change()

    def _on_enabled_toggle(self) -> None:
        self._sync_enabled_state()
        self._notify()

    def _sync_enabled_state(self) -> None:
        state = "normal" if self._enabled_var.get() else "disabled"
        for w in self._threshold_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _update_hint(self) -> None:
        if not self._enabled_var.get():
            self._hint_var.set(
                "Gap reset is off: rolling EMAs keep state across sample gaps "
                "(including ATM-band dropouts)."
            )
            return
        sec = self._preset_var.get() or "20"
        self._hint_var.set(
            f"When a token's sample stream pauses longer than {sec}s, rolling "
            "controllers (e.g. ltp_ema300) reset and re-warm. Choose a higher "
            "threshold if short ATM-band absences are wiping warm-up."
        )

    def _build(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Applied when building the Master dataset (before insert). "
                "Controls whether rolling features reset after a sample-stream gap."
            ),
            foreground="#666",
            font=("TkDefaultFont", 8),
            wraplength=360,
        ).pack(anchor="w", pady=(0, 6))

        ttk.Checkbutton(
            parent,
            text="Enable Gap Reset",
            variable=self._enabled_var,
            command=self._on_enabled_toggle,
        ).pack(anchor="w", pady=(0, 8))

        box = ttk.LabelFrame(parent, text="Gap Threshold", padding=6)
        box.pack(fill="x", pady=(0, 4))
        ttk.Label(
            box,
            text="Reset rolling features when the token sample gap exceeds:",
            foreground="#444",
            wraplength=340,
        ).pack(anchor="w", pady=(0, 4))

        row = ttk.Frame(box)
        row.pack(anchor="w")
        for sec in GAP_PRESET_SEC:
            rb = ttk.Radiobutton(
                row,
                text=f"{sec}s",
                variable=self._preset_var,
                value=str(sec),
                command=self._notify,
            )
            rb.pack(side="left", padx=(0, 10))
            self._threshold_widgets.append(rb)

        ttk.Label(
            parent,
            textvariable=self._hint_var,
            foreground="#666",
            font=("TkDefaultFont", 8),
            wraplength=360,
        ).pack(anchor="w", pady=(8, 0))
