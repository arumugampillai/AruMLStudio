"""Feature Distribution Studio — Phase 4.2 UI (sortable comparison table)."""

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
    ("null_pct", "null%", 70),
    ("mean", "Mean", 90),
    ("std", "Std", 90),
    ("min", "Min", 80),
    ("p25", "p25", 80),
    ("p50", "p50", 80),
    ("p75", "p75", 80),
    ("p95", "p95", 80),
    ("max", "Max", 80),
    ("skew", "Skew", 70),
    ("rank_gain", "Rank Gain", 80),
)


class FeatureDistributionStudioPanel(ttk.Frame, LazyLoadMixin):
    """Simple Feature Distribution Studio: summary + sortable comparison table."""

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
        self._meta: dict[str, Any] = {}
        self._sort_col = "rank_gain"
        self._sort_desc = False

        self._status_var = tk.StringVar(value="Select a model and Load or Compute.")
        self._model_var = model_var if model_var is not None else tk.StringVar()
        self._filter_var = filter_var if filter_var is not None else tk.StringVar()
        self._top_n_var = top_n_var if top_n_var is not None else tk.StringVar(value="20")
        self._top_n_only = (
            top_n_only if top_n_only is not None else tk.BooleanVar(value=False)
        )

        self._summary_vars = {
            "model": tk.StringVar(value="—"),
            "target": tk.StringVar(value="—"),
            "dataset": tk.StringVar(value="—"),
            "rows": tk.StringVar(value="—"),
            "features": tk.StringVar(value="—"),
            "compute": tk.StringVar(value="—"),
            "backend": tk.StringVar(value="—"),
            "importance": tk.StringVar(value="—"),
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
        self.rowconfigure(2, weight=1)

        hdr = ttk.Frame(self, padding=(8, 8, 8, 4))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Feature Distribution Studio", font=SECTION_FONT).pack(
            side="left"
        )
        ttk.Label(
            hdr,
            text="What do important features look like? · holdout only · no charts v1",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(12, 0))

        summary = ttk.LabelFrame(self, text="Model Summary", padding=8)
        summary.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        for i in range(6):
            summary.columnconfigure(i, weight=1)
        fields = (
            ("Model", "model"),
            ("Target", "target"),
            ("Dataset", "dataset"),
            ("Holdout rows", "rows"),
            ("Features", "features"),
            ("Compute time", "compute"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(summary)
            cell.grid(row=0, column=i, sticky="ew", padx=4)
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key]).pack(anchor="w")
        ttk.Label(summary, text="Engine", foreground=COL_MUTED).grid(
            row=1, column=0, sticky="w", padx=4, pady=(6, 0)
        )
        ttk.Label(summary, textvariable=self._summary_vars["backend"]).grid(
            row=1, column=1, sticky="w", padx=4, pady=(6, 0)
        )
        ttk.Label(summary, text="Importance join", foreground=COL_MUTED).grid(
            row=1, column=2, sticky="w", padx=4, pady=(6, 0)
        )
        ttk.Label(summary, textvariable=self._summary_vars["importance"]).grid(
            row=1, column=3, sticky="w", padx=4, pady=(6, 0)
        )

        table_wrap = ttk.LabelFrame(self, text="Comparison Table", padding=4)
        table_wrap.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))
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
            anchor = "w" if key == "feature" else "e"
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "feature"))
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.tag_configure("top", background="#e8f1ff")
        self._tree.tag_configure("high_null", foreground=COL_WARN)
        self._tree.tag_configure("high_skew", foreground=COL_WARN)

        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground=COL_MUTED,
            padding=(8, 2, 8, 8),
        ).grid(row=3, column=0, sticky="ew")

    def _selected_model(self) -> str:
        return str(self._model_var.get() or "").strip()

    def apply_artifacts(
        self, loaded: dict[str, Any] | None, model_name: str
    ) -> None:
        """Populate from controller-owned load (viewer only)."""
        if not loaded:
            self.mark_unavailable("Unavailable — no Distribution artifacts.")
            return
        self._apply_result(
            loaded.get("comparison") or [], loaded.get("meta") or {}, model_name
        )

    def mark_unavailable(self, message: str) -> None:
        self._rows = []
        self._display_rows = []
        self._meta = {}
        for key in self._summary_vars:
            self._summary_vars[key].set("—")
        self._apply_filter_sort()
        self._status_var.set(message)

    def _on_load(self, *, quiet: bool = True) -> None:
        """Legacy self-load; Feature Studio controller owns Load Artifacts."""
        name = self._selected_model()
        if not name:
            if not quiet:
                messagebox.showwarning(
                    "Feature Distribution", "Select a model first.", parent=self
                )
            return
        from chain_replay_ml.feature_distribution_studio.writer import load_studio_artifacts
        from chain_replay_ml.training.paths import model_package_dir

        pkg = model_package_dir(self._data_dir(), name)
        loaded = load_studio_artifacts(pkg)
        if not loaded:
            self.mark_unavailable("No artifacts yet — click Compute.")
            if not quiet:
                messagebox.showinfo(
                    "Feature Distribution",
                    "No Feature Distribution Studio artifacts found for this model.\n"
                    "Click Compute to generate them.",
                    parent=self,
                )
            return
        self.apply_artifacts(loaded, name)

    def _apply_result(
        self, rows: list[dict[str, Any]], meta: dict[str, Any], model_name: str
    ) -> None:
        self._rows = [r for r in rows if isinstance(r, dict)]
        self._meta = dict(meta or {})
        self._summary_vars["model"].set(model_name)
        self._summary_vars["target"].set(fmt_val(self._meta.get("target")))
        self._summary_vars["dataset"].set(fmt_val(self._meta.get("dataset")))
        self._summary_vars["rows"].set(fmt_val(self._meta.get("holdout_row_count")))
        self._summary_vars["features"].set(fmt_val(self._meta.get("feature_count")))
        wall = self._meta.get("wall_time_sec")
        self._summary_vars["compute"].set(
            f"{float(wall):.2f}s" if wall is not None else "—"
        )
        self._summary_vars["backend"].set(
            fmt_val(self._meta.get("dataset_engine_backend"))
        )
        joined = self._meta.get("importance_joined")
        if joined is None and self._rows:
            joined = any(r.get("importance_joined") for r in self._rows)
        self._summary_vars["importance"].set("yes" if joined else "no")
        # Prefer rank_gain sort when importance joined
        if joined and self._sort_col == "rank_gain":
            self._sort_desc = False
        elif not joined and self._sort_col == "rank_gain":
            self._sort_col = "skew"
            self._sort_desc = True
        self._apply_filter_sort()
        self._status_var.set(
            f"Loaded {len(self._rows)} features · sort={self._sort_col}"
            + (" desc" if self._sort_desc else " asc")
        )

    def _on_sort_header(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col not in ("feature", "rank_gain")
        self._apply_filter_sort()

    def _top_n(self) -> int:
        try:
            return max(1, int(self._top_n_var.get() or 20))
        except ValueError:
            return 20

    def _apply_filter_sort(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        rows = list(self._rows)
        if needle:
            rows = [r for r in rows if needle in str(r.get("feature") or "").lower()]

        key = self._sort_col

        def sort_key(r: dict[str, Any]) -> tuple:
            val = r.get(key)
            if key == "feature":
                return (0, str(val or "").lower())
            try:
                return (0, float(val))
            except (TypeError, ValueError):
                return (1, 0.0)

        rows.sort(key=sort_key, reverse=self._sort_desc)

        if self._top_n_only.get():
            has_rank = any(r.get("rank_gain") is not None for r in self._rows)
            if has_rank:
                by_imp = sorted(
                    self._rows,
                    key=lambda r: (
                        int(r["rank_gain"])
                        if r.get("rank_gain") is not None
                        else 10**9
                    ),
                )[: self._top_n()]
            else:
                by_imp = sorted(
                    self._rows,
                    key=lambda r: abs(float(r["skew"])) if r.get("skew") is not None else -1.0,
                    reverse=True,
                )[: self._top_n()]
            keep = {str(r.get("feature")) for r in by_imp}
            rows = [r for r in rows if str(r.get("feature")) in keep]

        self._display_rows = rows
        self._render_table()

    def _render_table(self) -> None:
        tree = self._tree
        tree.delete(*tree.get_children())
        top_n = self._top_n()
        for row in self._display_rows:
            feat = str(row.get("feature") or "")
            try:
                rank_gain = (
                    int(row["rank_gain"]) if row.get("rank_gain") is not None else None
                )
            except (TypeError, ValueError):
                rank_gain = None
            try:
                null_pct = (
                    float(row["null_pct"]) if row.get("null_pct") is not None else 0.0
                )
            except (TypeError, ValueError):
                null_pct = 0.0
            try:
                skew = float(row["skew"]) if row.get("skew") is not None else None
            except (TypeError, ValueError):
                skew = None
            tags: list[str] = []
            if rank_gain is not None and rank_gain <= top_n:
                tags.append("top")
            if null_pct >= 5.0:
                tags.append("high_null")
            if skew is not None and abs(skew) >= 2.0:
                tags.append("high_skew")
            values = (
                feat,
                fmt_num(row.get("null_pct"), 2),
                fmt_num(row.get("mean"), 4),
                fmt_num(row.get("std"), 4),
                fmt_num(row.get("min"), 4),
                fmt_num(row.get("p25"), 4),
                fmt_num(row.get("p50"), 4),
                fmt_num(row.get("p75"), 4),
                fmt_num(row.get("p95"), 4),
                fmt_num(row.get("max"), 4),
                fmt_num(row.get("skew"), 3),
                fmt_val(row.get("rank_gain")),
            )
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _on_export(self) -> None:
        if not self._display_rows:
            messagebox.showinfo("Export", "Nothing to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export distribution CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._selected_model() or 'distribution'}_comparison.csv",
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
