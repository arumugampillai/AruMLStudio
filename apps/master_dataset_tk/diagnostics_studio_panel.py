"""Diagnostics Studio — Phase 4.5 UI (headline + narrative + feature table)."""

from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import (
    COL_MUTED,
    COL_WARN,
    SECTION_FONT,
    fmt_num,
    fmt_val,
)


_TABLE_COLS = (
    ("feature", "Feature", 180),
    ("diagnostic_flag", "Flag", 120),
    ("rank_gain", "Rank Gain", 80),
    ("risk", "Risk", 70),
    ("risk_score", "Risk Score", 90),
    ("drift", "Drift", 70),
    ("drift_pct", "Drift %", 70),
    ("null_pct", "null%", 70),
    ("skew", "Skew", 70),
)


class DiagnosticsStudioPanel(ttk.Frame, LazyLoadMixin):
    """Synthesize studio artifacts into a short holdout diagnosis."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        model_var: tk.StringVar | None = None,
        filter_var: tk.StringVar | None = None,
        top_n_var: tk.StringVar | None = None,
        top_n_only: tk.BooleanVar | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._model_names: list[str] = []
        self._rows: list[dict[str, Any]] = []
        self._display_rows: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self._narrative: list[str] = []
        self._meta: dict[str, Any] = {}
        self._sort_col = "risk_score"
        self._sort_desc = True

        self._status_var = tk.StringVar(value="Select a model and Load or Compute.")
        self._model_var = model_var if model_var is not None else tk.StringVar()
        self._filter_var = filter_var if filter_var is not None else tk.StringVar()
        self._top_n_var = top_n_var if top_n_var is not None else tk.StringVar(value="30")
        self._top_n_only = (
            top_n_only if top_n_only is not None else tk.BooleanVar(value=False)
        )

        self._summary_vars = {
            "model": tk.StringVar(value="—"),
            "cause": tk.StringVar(value="—"),
            "confidence": tk.StringVar(value="—"),
            "similarity": tk.StringVar(value="—"),
            "feat_drift": tk.StringVar(value="—"),
            "mae_chg": tk.StringVar(value="—"),
            "joins": tk.StringVar(value="—"),
            "compute": tk.StringVar(value="—"),
        }

        self._build_ui()
        self._filter_var.trace_add("write", lambda *_: self._apply_filter_sort())
        self._top_n_only.trace_add("write", lambda *_: self._apply_filter_sort())
        self._top_n_var.trace_add("write", lambda *_: self._apply_filter_sort())
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        pass

    def open_with_model(self, model_name: str) -> None:
        self._model_var.set(str(model_name or "").strip())

    def apply_model_names(self, names: list[str]) -> None:
        self._model_names = list(names)
        self._status_var.set(
            f"{len(names)} model(s) available."
            if names
            else "No trained models on disk."
        )

    def refresh(self, *, lazy: bool = False) -> None:
        del lazy
        self.apply_model_names(self._model_names)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        hdr = ttk.Frame(self, padding=(8, 8, 8, 4))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Diagnostics Studio", font=SECTION_FONT).pack(side="left")
        ttk.Label(
            hdr,
            text="Why is holdout degrading? · join studios + metrics · no full Holdout recompute",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(12, 0))

        summary = ttk.LabelFrame(self, text="Diagnosis Headline", padding=8)
        summary.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        for i in range(4):
            summary.columnconfigure(i, weight=1)
        fields = (
            ("Model", "model"),
            ("Primary cause", "cause"),
            ("Confidence", "confidence"),
            ("Similarity", "similarity"),
            ("Feature drift", "feat_drift"),
            ("MAE Δ%", "mae_chg"),
            ("Joins", "joins"),
            ("Compute time", "compute"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(summary)
            cell.grid(row=i // 4, column=i % 4, sticky="ew", padx=4, pady=(0 if i < 4 else 6, 0))
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key]).pack(anchor="w")

        narr = ttk.LabelFrame(self, text="Narrative", padding=8)
        narr.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))
        narr.columnconfigure(0, weight=1)
        self._narr_text = tk.Text(narr, height=5, wrap="word", relief="flat")
        self._narr_text.grid(row=0, column=0, sticky="ew")
        self._narr_text.configure(state="disabled")

        table_wrap = ttk.LabelFrame(self, text="Feature Diagnostic Table", padding=4)
        table_wrap.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)

        cols = tuple(c[0] for c in _TABLE_COLS)
        self._tree = ttk.Treeview(
            table_wrap, columns=cols, show="headings", selectmode="browse"
        )
        for key, title, width in _TABLE_COLS:
            self._tree.heading(
                key, text=title, command=lambda k=key: self._on_sort_header(k)
            )
            anchor = "w" if key in ("feature", "diagnostic_flag", "risk") else "e"
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "feature"))
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.tag_configure("high_risk", background="#ffe8e8")
        self._tree.tag_configure("high_null", background="#fff6e0")
        self._tree.tag_configure("rank_drift_conflict", foreground=COL_WARN)

        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground=COL_MUTED,
            padding=(8, 2, 8, 8),
        ).grid(row=4, column=0, sticky="ew")

    def _selected_model(self) -> str:
        return str(self._model_var.get() or "").strip()

    def apply_artifacts(
        self, loaded: dict[str, Any] | None, model_name: str
    ) -> None:
        """Populate from controller-owned load (viewer only)."""
        if not loaded:
            self.mark_unavailable("Unavailable — no Diagnostics artifacts.")
            return
        self._apply_result(
            loaded.get("summary") or {},
            loaded.get("narrative") or [],
            loaded.get("comparison") or [],
            loaded.get("meta") or {},
            model_name,
        )

    def mark_unavailable(self, message: str) -> None:
        self._rows = []
        self._display_rows = []
        self._summary = {}
        self._narrative = []
        self._meta = {}
        for key in self._summary_vars:
            self._summary_vars[key].set("—")
        if hasattr(self, "_narr_text"):
            self._narr_text.configure(state="normal")
            self._narr_text.delete("1.0", "end")
            self._narr_text.configure(state="disabled")
        self._apply_filter_sort()
        self._status_var.set(message)

    def _on_load(self, *, quiet: bool = True) -> None:
        """Legacy self-load; Feature Studio controller owns Load Artifacts."""
        name = self._selected_model()
        if not name:
            if not quiet:
                messagebox.showwarning(
                    "Diagnostics", "Select a model first.", parent=self
                )
            return
        from chain_replay_ml.diagnostics_studio.writer import load_studio_artifacts
        from chain_replay_ml.training.paths import model_package_dir

        pkg = model_package_dir(self._data_dir(), name)
        loaded = load_studio_artifacts(pkg)
        if not loaded:
            self.mark_unavailable("No artifacts yet — click Compute.")
            if not quiet:
                messagebox.showinfo(
                    "Diagnostics",
                    "No Diagnostics Studio artifacts found.\n"
                    "Click Compute after running Importance / Distribution / Drift.",
                    parent=self,
                )
            return
        self.apply_artifacts(loaded, name)

    def _apply_result(
        self,
        summary: dict[str, Any],
        narrative: list[str],
        rows: list[dict[str, Any]],
        meta: dict[str, Any],
        model_name: str,
    ) -> None:
        self._summary = dict(summary or {})
        self._narrative = [str(b) for b in (narrative or [])]
        self._rows = [r for r in rows if isinstance(r, dict)]
        self._meta = dict(meta or {})

        self._summary_vars["model"].set(model_name)
        self._summary_vars["cause"].set(
            fmt_val(self._summary.get("label") or self._summary.get("primary_cause"))
        )
        conf = self._summary.get("confidence_pct")
        self._summary_vars["confidence"].set(
            f"{float(conf):.0f}%" if conf is not None else "—"
        )
        sim = self._summary.get("similarity_pct")
        self._summary_vars["similarity"].set(
            f"{float(sim):.1f}%" if sim is not None else "—"
        )
        fd = self._summary.get("feature_drift_pct")
        self._summary_vars["feat_drift"].set(
            f"{float(fd):.1f}%" if fd is not None else "—"
        )
        mae = self._summary.get("mae_pct_change")
        self._summary_vars["mae_chg"].set(
            f"{float(mae):+.1f}%" if mae is not None else "—"
        )
        joins = self._summary.get("joins") or {}
        on = [k for k, v in joins.items() if v]
        self._summary_vars["joins"].set(", ".join(on) if on else "none")
        wall = self._meta.get("wall_time_sec")
        self._summary_vars["compute"].set(
            f"{float(wall):.3f}s" if wall is not None else "—"
        )

        self._narr_text.configure(state="normal")
        self._narr_text.delete("1.0", "end")
        self._narr_text.insert("1.0", "\n".join(f"• {b}" for b in self._narrative))
        self._narr_text.configure(state="disabled")

        self._apply_filter_sort()
        self._status_var.set(
            f"Loaded {len(self._rows)} features · cause={self._summary.get('primary_cause')}"
        )

    def _on_sort_header(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col not in (
                "feature",
                "diagnostic_flag",
                "risk",
                "rank_gain",
            )
        self._apply_filter_sort()

    def _top_n(self) -> int:
        try:
            return max(1, int(self._top_n_var.get() or 30))
        except ValueError:
            return 30

    def _apply_filter_sort(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        rows = list(self._rows)
        if needle:
            rows = [
                r
                for r in rows
                if needle in str(r.get("feature") or "").lower()
                or needle in str(r.get("diagnostic_flag") or "").lower()
            ]

        key = self._sort_col

        def sort_key(r: dict[str, Any]) -> tuple:
            val = r.get(key)
            if key in ("feature", "diagnostic_flag", "risk"):
                return (0, str(val or "").lower())
            try:
                return (0, float(val))
            except (TypeError, ValueError):
                return (1, 0.0)

        rows.sort(key=sort_key, reverse=self._sort_desc)

        if self._top_n_only.get():
            by_risk = sorted(
                self._rows,
                key=lambda r: float(r.get("risk_score") or 0),
                reverse=True,
            )[: self._top_n()]
            keep = {str(r.get("feature")) for r in by_risk}
            rows = [r for r in rows if str(r.get("feature")) in keep]

        self._display_rows = rows
        self._render_table()

    def _render_table(self) -> None:
        tree = self._tree
        tree.delete(*tree.get_children())
        for row in self._display_rows:
            flag = str(row.get("diagnostic_flag") or "ok")
            tags = [flag] if flag != "ok" else []
            values = (
                str(row.get("feature") or ""),
                flag,
                fmt_val(row.get("rank_gain")),
                fmt_val(row.get("risk")),
                fmt_num(row.get("risk_score"), 4),
                fmt_num(row.get("drift"), 4),
                fmt_num(row.get("drift_pct"), 1),
                fmt_num(row.get("null_pct"), 2),
                fmt_num(row.get("skew"), 3),
            )
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _on_export(self) -> None:
        if not self._display_rows:
            messagebox.showinfo("Export", "Nothing to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export diagnostics CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._selected_model() or 'diagnostics'}_features.csv",
        )
        if not path:
            return
        fields = [c[0] for c in _TABLE_COLS]
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in self._display_rows:
                    writer.writerow({k: row.get(k) for k in fields})
        except OSError as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return
        self._status_var.set(
            f"Exported {len(self._display_rows)} rows → {os.path.basename(path)}"
        )
