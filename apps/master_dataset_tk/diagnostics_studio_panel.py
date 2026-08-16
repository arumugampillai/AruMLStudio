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

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 4))
        self._notebook.columnconfigure(0, weight=1)
        self._notebook.rowconfigure(0, weight=1)

        cols = tuple(c[0] for c in _TABLE_COLS)

        def _create_tree_tab(parent_nb: ttk.Notebook, tab_name: str) -> tuple[ttk.Frame, ttk.Treeview]:
            tab_frame = ttk.Frame(parent_nb, padding=4)
            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(0, weight=1)
            parent_nb.add(tab_frame, text=tab_name)

            tree = ttk.Treeview(tab_frame, columns=cols, show="headings", selectmode="browse")
            for key, title, width in _TABLE_COLS:
                tree.heading(key, text=title, command=lambda k=key: self._on_sort_header(k))
                anchor = "w" if key in ("feature", "diagnostic_flag", "risk") else "e"
                tree.column(key, width=width, anchor=anchor, stretch=(key == "feature"))

            vsb = ttk.Scrollbar(tab_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tab_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            tree.tag_configure("high_risk", background="#ffe8e8")
            tree.tag_configure("high_null", background="#fff6e0")
            tree.tag_configure("rank_drift_conflict", foreground=COL_WARN)
            return tab_frame, tree

        self._tab_reg, self._tree_reg = _create_tree_tab(self._notebook, "Feature Registry")
        self._tab_base, self._tree_base = _create_tree_tab(self._notebook, "Base Pipeline")
        self._tab_exp, self._tree_exp = _create_tree_tab(self._notebook, "Selected Experimental")

        # Keep legacy _tree pointing to active tab tree for backward compatibility
        self._tree = self._tree_reg

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
        self._rows_reg = []
        self._rows_base = []
        self._rows_exp = []
        self._display_rows_reg = []
        self._display_rows_base = []
        self._display_rows_exp = []
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
        if hasattr(self, "_notebook"):
            self._notebook.tab(0, text="Feature Registry")
            self._notebook.tab(1, text="Base Pipeline")
            self._notebook.tab(2, text="Selected Experimental")
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

        # Partition rows into 3 tabs with ownership invariant enforcement
        from chain_replay_ml.diagnostics_studio.feature_partition import partition_diagnostic_rows
        from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
        from chain_replay_ml.training.paths import model_package_dir

        ds_name = self._summary.get("dataset") or self._meta.get("dataset")
        if not ds_name:
            cfg_path = os.path.join(model_package_dir(self._data_dir(), model_name), "config.json")
            if os.path.isfile(cfg_path):
                try:
                    import json
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        cfg_raw = json.load(fh)
                    ds_name = cfg_raw.get("dataset")
                except Exception:
                    pass

        ds_meta: dict[str, Any] = {}
        if ds_name:
            meta_path = os.path.join(datasets_dir(self._data_dir()), f"{_safe_filename(ds_name)}.json")
            if os.path.isfile(meta_path):
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        loaded_doc = json.load(fh)
                    if isinstance(loaded_doc, dict):
                        ds_meta = loaded_doc
                except Exception:
                    ds_meta = {}

        self._partition = partition_diagnostic_rows(
            self._rows,
            data_dir=self._data_dir(),
            dataset_metadata=ds_meta,
        )

        self._rows_reg = self._partition.registry_rows
        self._rows_base = self._partition.base_pipeline_rows
        self._rows_exp = self._partition.experimental_rows

        self._apply_filter_sort()

        if not self._partition.is_valid:
            self._status_var.set(
                f"Invariant Warning: {self._partition.error_message}"
            )
        else:
            self._status_var.set(
                f"Loaded {len(self._rows)} features (Reg: {self._partition.registry_count}, "
                f"Base: {self._partition.base_pipeline_count}, Exp: {self._partition.experimental_count}) · "
                f"cause={self._summary.get('primary_cause')}"
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

    def _filter_and_sort_subset(self, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        needle = str(self._filter_var.get() or "").strip().lower()
        rows = list(source_rows)
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
                source_rows,
                key=lambda r: float(r.get("risk_score") or 0),
                reverse=True,
            )[: self._top_n()]
            keep = {str(r.get("feature")) for r in by_risk}
            rows = [r for r in rows if str(r.get("feature")) in keep]

        return rows

    def _apply_filter_sort(self) -> None:
        self._display_rows_reg = self._filter_and_sort_subset(getattr(self, "_rows_reg", []))
        self._display_rows_base = self._filter_and_sort_subset(getattr(self, "_rows_base", []))
        self._display_rows_exp = self._filter_and_sort_subset(getattr(self, "_rows_exp", []))
        self._display_rows = self._display_rows_reg + self._display_rows_base + self._display_rows_exp

        # Update tab counts
        reg_tot = len(getattr(self, "_rows_reg", []))
        base_tot = len(getattr(self, "_rows_base", []))
        exp_tot = len(getattr(self, "_rows_exp", []))

        if hasattr(self, "_notebook"):
            self._notebook.tab(0, text=f"Feature Registry ({len(self._display_rows_reg)}/{reg_tot})")
            self._notebook.tab(1, text=f"Base Pipeline ({len(self._display_rows_base)}/{base_tot})")
            self._notebook.tab(2, text=f"Selected Experimental ({len(self._display_rows_exp)}/{exp_tot})")

        self._render_tree(self._tree_reg, self._display_rows_reg)
        self._render_tree(self._tree_base, self._display_rows_base)
        self._render_tree(self._tree_exp, self._display_rows_exp)

    def _render_tree(self, tree: ttk.Treeview, rows: list[dict[str, Any]]) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
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
        active_idx = self._notebook.index(self._notebook.select()) if hasattr(self, "_notebook") else 0
        tab_names = ("registry", "base_pipeline", "experimental")
        active_name = tab_names[active_idx] if active_idx < len(tab_names) else "diagnostics"

        if active_idx == 0:
            export_rows = self._display_rows_reg
        elif active_idx == 1:
            export_rows = self._display_rows_base
        else:
            export_rows = self._display_rows_exp

        if not export_rows:
            messagebox.showinfo("Export", "Nothing to export in the current tab.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export diagnostics CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._selected_model() or 'diagnostics'}_{active_name}_features.csv",
        )
        if not path:
            return
        fields = [c[0] for c in _TABLE_COLS] + ["feature_source"]
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in export_rows:
                    writer.writerow({k: row.get(k) for k in fields})
        except OSError as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return
        self._status_var.set(
            f"Exported {len(export_rows)} {active_name} rows → {os.path.basename(path)}"
        )
