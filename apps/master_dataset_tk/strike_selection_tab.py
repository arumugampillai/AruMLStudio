"""Strike selection tab — web-parity strike modes for build config."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .strike_selection_engine import (
    ATM_BAND_OPTIONS,
    CUSTOM_OFFSETS,
    MASTER_DATASET_ATM_BAND,
    atm_band_hint_text,
    default_strike_config,
    delta_preview_rule,
    normalize_strike_config,
)


class StrikeSelectionTab:
    def __init__(
        self,
        parent: ttk.Frame,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._mode_var = tk.StringVar(value="atm_band")
        self._atm_band_var = tk.StringVar(value="15")
        self._premium_min_var = tk.StringVar(value="15")
        self._premium_max_var = tk.StringVar(value="30")
        self._delta_type_var = tk.StringVar(value="absolute")
        self._delta_min_var = tk.StringVar(value="0.15")
        self._delta_max_var = tk.StringVar(value="0.50")
        self._custom_vars: dict[int, tk.BooleanVar] = {}
        self._atm_hint_var = tk.StringVar(value="")
        self._delta_rule_var = tk.StringVar(value="")
        self._panels: dict[str, ttk.Frame] = {}
        self._build(parent)
        self.apply_config(default_strike_config())

    def apply_config(self, cfg: dict[str, Any]) -> None:
        doc = normalize_strike_config(cfg)
        self._mode_var.set(str(doc.get("mode") or "atm_band"))
        band = doc.get("atmBand")
        band_key = "all" if band == "all" else str(int(band) if band is not None else MASTER_DATASET_ATM_BAND)
        if band_key not in {k for k, _ in ATM_BAND_OPTIONS}:
            band_key = "15"
        self._atm_band_var.set(band_key)
        self._premium_min_var.set(str(doc.get("premiumMin") or 15))
        self._premium_max_var.set(str(doc.get("premiumMax") or 30))
        self._delta_type_var.set(str(doc.get("deltaType") or "absolute"))
        self._delta_min_var.set(str(doc.get("deltaMin") or 0.15))
        self._delta_max_var.set(str(doc.get("deltaMax") or 0.50))
        selected = {int(o) for o in (doc.get("customOffsets") or [])}
        for offset, var in self._custom_vars.items():
            var.set(offset in selected)
        self._sync_panels()
        self._update_hints()

    def get_config(self) -> dict[str, Any]:
        band_val = self._atm_band_var.get()
        atm_band: Any = "all" if band_val == "all" else int(band_val)
        mode = self._mode_var.get()
        return normalize_strike_config({
            "mode": mode,
            "atmBand": atm_band,
            "premiumMin": float(self._premium_min_var.get() or 15),
            "premiumMax": float(self._premium_max_var.get() or 30),
            "premiumIgnoreOutside": mode == "premium_band",
            "deltaType": self._delta_type_var.get(),
            "deltaMin": float(self._delta_min_var.get() or 0.15),
            "deltaMax": float(self._delta_max_var.get() or 0.50),
            "customOffsets": sorted(o for o, var in self._custom_vars.items() if var.get()),
            "applied": True,
        })

    def _notify(self) -> None:
        self._update_hints()
        if self._on_change:
            self._on_change()

    def _build(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Selected strike settings are used for master dataset builds.",
            foreground="#666",
            font=("TkDefaultFont", 8),
            wraplength=340,
        ).pack(anchor="w", pady=(0, 4))

        mode_box = ttk.LabelFrame(parent, text="Selection mode", padding=4)
        mode_box.pack(fill="x", pady=(0, 4))
        for mode, label in (
            ("atm_band", "ATM Band"),
            ("premium_band", "Premium Band"),
            ("delta_range", "Delta Range"),
            ("custom", "Custom Strikes"),
        ):
            ttk.Radiobutton(
                mode_box,
                text=label,
                variable=self._mode_var,
                value=mode,
                command=self._on_mode_changed,
            ).pack(anchor="w")

        self._content = ttk.Frame(parent)
        self._content.pack(fill="both", expand=True)

        self._panels["atm_band"] = self._build_atm_panel(self._content)
        self._panels["premium_band"] = self._build_premium_panel(self._content)
        self._panels["delta_range"] = self._build_delta_panel(self._content)
        self._panels["custom"] = self._build_custom_panel(self._content)

    def _build_atm_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="ATM Band", padding=4)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="ATM Band").pack(side="left")
        self._atm_labels = {k: v for k, v in ATM_BAND_OPTIONS}
        self._atm_keys = [k for k, _ in ATM_BAND_OPTIONS]
        self._atm_combo = ttk.Combobox(
            row,
            state="readonly",
            width=14,
            values=[self._atm_labels[k] for k in self._atm_keys],
        )
        self._atm_combo.pack(side="left", padx=(6, 0))
        self._atm_combo.bind("<<ComboboxSelected>>", self._on_atm_band_pick)

        ttk.Label(frame, textvariable=self._atm_hint_var, foreground="#666", wraplength=320).pack(
            anchor="w", pady=(4, 0),
        )
        return frame

    def _on_atm_band_pick(self, _event: tk.Event | None = None) -> None:
        pick = self._atm_combo.get()
        for key, lbl in self._atm_labels.items():
            if lbl == pick:
                self._atm_band_var.set(key)
                break
        self._notify()

    def _build_premium_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="Premium filter", padding=4)
        ttk.Label(
            frame,
            text="Select strikes by option premium (LTP).",
            foreground="#666",
            font=("TkDefaultFont", 8),
            wraplength=320,
        ).pack(anchor="w", pady=(0, 4))
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Min ₹", width=8).pack(side="left")
        e1 = ttk.Entry(row1, textvariable=self._premium_min_var, width=8)
        e1.pack(side="left")
        e1.bind("<KeyRelease>", lambda _e: self._notify())
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Max ₹", width=8).pack(side="left")
        e2 = ttk.Entry(row2, textvariable=self._premium_max_var, width=8)
        e2.pack(side="left")
        e2.bind("<KeyRelease>", lambda _e: self._notify())
        return frame

    def _build_delta_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="Delta range", padding=4)
        type_row = ttk.Frame(frame)
        type_row.pack(fill="x", pady=(0, 4))
        for dt, lbl in (("absolute", "Absolute"), ("ce", "CE"), ("pe", "PE")):
            ttk.Radiobutton(
                type_row,
                text=lbl,
                variable=self._delta_type_var,
                value=dt,
                command=self._notify,
            ).pack(side="left", padx=(0, 6))
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Min", width=8).pack(side="left")
        e1 = ttk.Entry(row1, textvariable=self._delta_min_var, width=8)
        e1.pack(side="left")
        e1.bind("<KeyRelease>", lambda _e: self._notify())
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Max", width=8).pack(side="left")
        e2 = ttk.Entry(row2, textvariable=self._delta_max_var, width=8)
        e2.pack(side="left")
        e2.bind("<KeyRelease>", lambda _e: self._notify())
        ttk.Label(frame, textvariable=self._delta_rule_var, foreground="#666", wraplength=320).pack(
            anchor="w", pady=(4, 0),
        )
        return frame

    def _build_custom_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="Custom strikes", padding=4)
        ttk.Label(
            frame,
            text="Strike offset from ATM (− below spot, + above).",
            foreground="#666",
            font=("TkDefaultFont", 8),
            wraplength=320,
        ).pack(anchor="w", pady=(0, 4))
        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        for i, offset in enumerate(CUSTOM_OFFSETS):
            var = tk.BooleanVar(value=False)
            self._custom_vars[offset] = var
            lbl = f"+{offset}" if offset > 0 else str(offset)
            ttk.Checkbutton(
                grid,
                text=lbl,
                variable=var,
                command=self._notify,
            ).grid(row=i // 6, column=i % 6, sticky="w", padx=2, pady=1)
        return frame

    def _on_mode_changed(self) -> None:
        self._sync_panels()
        self._notify()

    def _sync_panels(self) -> None:
        mode = self._mode_var.get()
        for key, panel in self._panels.items():
            if key == mode:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()
        if hasattr(self, "_atm_combo"):
            labels = getattr(self, "_atm_labels", {})
            key = self._atm_band_var.get()
            self._atm_combo.set(labels.get(key, labels.get("15", "±15")))

    def _update_hints(self) -> None:
        self._atm_hint_var.set(atm_band_hint_text(self._atm_band_var.get()))
        try:
            d_min = float(self._delta_min_var.get() or 0)
            d_max = float(self._delta_max_var.get() or 0)
            rule = delta_preview_rule(self._delta_type_var.get(), d_min, d_max)
            self._delta_rule_var.set(f"Keep strikes where {rule}")
        except (TypeError, ValueError):
            self._delta_rule_var.set("")
