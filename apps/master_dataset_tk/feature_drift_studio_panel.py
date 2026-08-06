"""Feature Drift Studio — Phase 5.2 UI (distribution-aware comparison table)."""

from __future__ import annotations

import csv
import math
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


# Feature | means | Drift % | KS | Wasserstein | W-Norm | Importance | Risk | Risk Score | …
_TABLE_COLS = (
    ("feature", "Feature", 160),
    ("wf_mean", "WF Mean", 90),
    ("holdout_mean", "HO Mean", 90),
    ("drift_pct", "Drift %", 80),
    ("ks_statistic", "KS", 70),
    ("wasserstein_distance", "Wasserstein", 95),
    ("wasserstein_normalized", "W-Norm", 80),
    ("importance", "Importance", 90),
    ("risk", "Risk", 80),
    ("risk_score", "Risk Score", 90),
    ("rank_gain", "Rank Gain", 80),
    ("null_pct_ho", "null% HO", 80),
)

_SIMILARITY_TOOLTIP = (
    "Similarity % = 100 − weighted overall drift.\n"
    "Weights: feature 35%, target 30%, premium 20%, volatility 15%.\n"
    "Each component is a 0–100 WF-vs-holdout drift score "
    "(see compute_similarity_score)."
)

# Display order for risk contributor breakdown (matches user-facing labels).
_RISK_SHARE_LABELS = (
    ("ks", "KS"),
    ("wasserstein_normalized", "Normalized W"),
    ("mean_drift", "Mean Drift"),
    ("null_drift", "Null Drift"),
    ("importance", "Importance"),
)


def format_si_number(value: Any) -> str:
    """Compact magnitude for UI (SI suffixes). Raw floats stay in JSON/CSV."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if x == 0.0:
        return "0"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    for thresh, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if ax >= thresh:
            scaled = ax / thresh
            if scaled >= 100:
                text = f"{scaled:.0f}"
            elif scaled >= 10:
                text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            else:
                text = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    if ax >= 0.01:
        return f"{sign}{ax:.4f}".rstrip("0").rstrip(".")
    return f"{sign}{ax:.2e}"


def format_importance_cell(row: dict[str, Any]) -> str:
    """Show — when importance was not joined / missing; 0.0000 only if joined and zero."""
    joined = row.get("importance_joined")
    val = row.get("importance")
    if joined is False:
        return "—"
    if val is None:
        return "—"
    # Old artifacts without the join flag: don't imply fake zeros.
    if joined is None:
        try:
            if float(val) == 0.0:
                return "—"
        except (TypeError, ValueError):
            return "—"
    return fmt_num(val, 4)


def _bind_tooltip(widget: tk.Misc, text: str) -> None:
    """Minimal hover tooltip (no external dependency)."""
    tip: dict[str, Any] = {"win": None}

    def show(_event: object | None = None) -> None:
        if tip["win"] is not None:
            return
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.attributes("-topmost", True)
        try:
            x = widget.winfo_rootx() + 8
            y = widget.winfo_rooty() + widget.winfo_height() + 4
        except tk.TclError:
            x, y = 0, 0
        win.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            win,
            text=text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 8),
            padx=6,
            pady=4,
        )
        lbl.pack()
        tip["win"] = win

    def hide(_event: object | None = None) -> None:
        win = tip.get("win")
        tip["win"] = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


class FeatureDriftStudioPanel(ttk.Frame, LazyLoadMixin):
    """Feature Drift Studio: summary + sortable comparison table (KS / Wasserstein)."""

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
        self._sort_col = "risk_score"
        self._sort_desc = True

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
            "wf_rows": tk.StringVar(value="—"),
            "ho_rows": tk.StringVar(value="—"),
            "features": tk.StringVar(value="—"),
            "avg_drift": tk.StringVar(value="—"),
            "avg_ks": tk.StringVar(value="—"),
            "avg_w": tk.StringVar(value="—"),
            "similarity": tk.StringVar(value="—"),
            "compute": tk.StringVar(value="—"),
            "joins": tk.StringVar(value="—"),
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
        ttk.Label(hdr, text="Feature Drift Studio", font=SECTION_FONT).pack(side="left")
        ttk.Label(
            hdr,
            text="WF vs holdout · mean/std + KS + Wasserstein · risk 0–100",
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
            ("WF rows", "wf_rows"),
            ("Holdout rows", "ho_rows"),
            ("Features", "features"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(summary)
            cell.grid(row=0, column=i, sticky="ew", padx=4)
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key]).pack(anchor="w")

        row2 = (
            ("Average Drift", "avg_drift"),
            ("Average KS", "avg_ks"),
            ("Average Normalized Wasserstein", "avg_w"),
            ("Similarity %", "similarity"),
            ("Compute time", "compute"),
            ("Joins", "joins"),
        )
        for i, (label, key) in enumerate(row2):
            cell = ttk.Frame(summary)
            cell.grid(row=1, column=i, sticky="ew", padx=4, pady=(6, 0))
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            val_lbl = ttk.Label(cell, textvariable=self._summary_vars[key])
            val_lbl.pack(anchor="w")
            if key == "similarity":
                _bind_tooltip(val_lbl, _SIMILARITY_TOOLTIP)
                # Also bind the muted caption so hover is easy to find.
                caption = cell.winfo_children()[0]
                _bind_tooltip(caption, _SIMILARITY_TOOLTIP)

        table_wrap = ttk.LabelFrame(
            self, text="Comparison Table (double-click row for risk breakdown)", padding=4
        )
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
            anchor = "w" if key in ("feature", "risk") else "e"
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "feature"))
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.tag_configure("high", background="#ffe8e8")
        self._tree.tag_configure("medium", background="#fff6e0")
        self._tree.tag_configure("top", foreground=COL_WARN)
        self._tree.bind("<Double-1>", self._on_row_double_click)

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
            self.mark_unavailable("Unavailable — no Drift artifacts.")
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
                    "Feature Drift", "Select a model first.", parent=self
                )
            return
        from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts
        from chain_replay_ml.training.paths import model_package_dir

        pkg = model_package_dir(self._data_dir(), name)
        loaded = load_studio_artifacts(pkg)
        if not loaded:
            self.mark_unavailable("No artifacts yet — click Compute.")
            if not quiet:
                messagebox.showinfo(
                    "Feature Drift",
                    "No Feature Drift Studio artifacts found for this model.\n"
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
        self._summary_vars["wf_rows"].set(fmt_val(self._meta.get("wf_row_count")))
        self._summary_vars["ho_rows"].set(fmt_val(self._meta.get("holdout_row_count")))
        self._summary_vars["features"].set(fmt_val(self._meta.get("feature_count")))

        avg_drift = self._meta.get("average_drift_pct")
        if avg_drift is None and self._rows:
            vals = [
                abs(float(r["drift_pct"]))
                for r in self._rows
                if r.get("drift_pct") is not None
            ]
            avg_drift = sum(vals) / len(vals) if vals else None
        self._summary_vars["avg_drift"].set(
            f"{float(avg_drift):.1f}%" if avg_drift is not None else "—"
        )

        avg_ks = self._meta.get("average_ks")
        if avg_ks is None and self._rows:
            vals = [
                float(r["ks_statistic"])
                for r in self._rows
                if r.get("ks_statistic") is not None
            ]
            avg_ks = sum(vals) / len(vals) if vals else None
        self._summary_vars["avg_ks"].set(
            f"{float(avg_ks):.4f}" if avg_ks is not None else "—"
        )

        avg_wn = self._meta.get("average_wasserstein_normalized")
        if avg_wn is None and self._rows:
            vals = [
                float(r["wasserstein_normalized"])
                for r in self._rows
                if r.get("wasserstein_normalized") is not None
            ]
            avg_wn = sum(vals) / len(vals) if vals else None
        self._summary_vars["avg_w"].set(
            f"{float(avg_wn):.4f}" if avg_wn is not None else "—"
        )

        sim = self._meta.get("similarity_pct")
        self._summary_vars["similarity"].set(
            f"{float(sim):.1f}%" if sim is not None else "—"
        )
        wall = self._meta.get("wall_time_sec")
        self._summary_vars["compute"].set(
            f"{float(wall):.2f}s" if wall is not None else "—"
        )
        joins = []
        if self._meta.get("importance_joined"):
            joins.append("Importance")
        if self._meta.get("distribution_joined"):
            joins.append("Distribution")
        self._summary_vars["joins"].set(", ".join(joins) if joins else "none")
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
            self._sort_desc = col not in ("feature", "risk", "rank_gain")
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
            if key in ("feature", "risk"):
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
            risk = str(row.get("risk") or "").lower()
            tags: list[str] = []
            if risk == "high":
                tags.append("high")
            elif risk == "medium":
                tags.append("medium")
            w_norm = row.get("wasserstein_normalized")
            values = (
                str(row.get("feature") or ""),
                fmt_num(row.get("wf_mean"), 4),
                fmt_num(row.get("holdout_mean"), 4),
                fmt_num(row.get("drift_pct"), 1),
                fmt_num(row.get("ks_statistic"), 4),
                format_si_number(row.get("wasserstein_distance")),
                fmt_num(w_norm, 4) if w_norm is not None else "—",
                format_importance_cell(row),
                fmt_val(row.get("risk")),
                fmt_num(row.get("risk_score"), 2),
                fmt_val(row.get("rank_gain")),
                fmt_num(row.get("null_pct_ho"), 2),
            )
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _row_for_iid(self, iid: str) -> dict[str, Any] | None:
        vals = self._tree.item(iid, "values")
        if not vals:
            return None
        feat = str(vals[0] or "")
        for row in self._display_rows:
            if str(row.get("feature") or "") == feat:
                return row
        return None

    def _on_row_double_click(self, _event: object | None = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        row = self._row_for_iid(sel[0])
        if row is None:
            return
        self._show_risk_breakdown(row)

    def _show_risk_breakdown(self, row: dict[str, Any]) -> None:
        from chain_replay_ml.training.holdout_performance import (
            composite_drift_risk_components,
        )

        feat = str(row.get("feature") or "")
        # Prefer stored importance only when joined; else treat as 0 for formula parity.
        if row.get("importance_joined") is False or row.get("importance") is None:
            importance = 0.0
        else:
            try:
                importance = float(row.get("importance") or 0.0)
            except (TypeError, ValueError):
                importance = 0.0
        try:
            mean_drift = float(row.get("drift") or 0.0)
        except (TypeError, ValueError):
            mean_drift = 0.0
        ks = row.get("ks_statistic")
        w_norm = row.get("wasserstein_normalized")
        null_pp = row.get("null_drift_pp")
        breakdown = composite_drift_risk_components(
            mean_drift=mean_drift,
            ks_statistic=float(ks) if ks is not None else None,
            wasserstein_normalized=float(w_norm) if w_norm is not None else None,
            null_drift_pp=float(null_pp) if null_pp is not None else None,
            importance=importance,
        )
        risk = breakdown["risk_score"]
        stored = row.get("risk_score")
        if stored is not None:
            try:
                risk = float(stored)
            except (TypeError, ValueError):
                pass
        shares = breakdown["shares_pct"]
        lines = [
            feat,
            f"Risk Score: {risk:.1f}",
            "",
            "Contributors",
        ]
        for key, label in _RISK_SHARE_LABELS:
            pct = float(shares.get(key) or 0.0)
            lines.append(f"  {label:<16} {pct:5.1f}%")
        messagebox.showinfo("Risk breakdown", "\n".join(lines), parent=self)

    def _on_export(self) -> None:
        if not self._display_rows:
            messagebox.showinfo("Export", "Nothing to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export drift CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._selected_model() or 'drift'}_comparison.csv",
        )
        if not path:
            return
        # Table cols plus Phase 5.2 JSON fields useful for offline analysis.
        # CSV keeps raw floats (not SI-formatted Wasserstein).
        fields = [c[0] for c in _TABLE_COLS] + [
            "ks_pvalue",
            "null_pct_wf",
            "null_drift_pp",
            "drift",
            "wf_std",
            "holdout_std",
            "importance_joined",
        ]
        # de-dupe while preserving order
        seen: set[str] = set()
        export_fields: list[str] = []
        for f in fields:
            if f not in seen:
                seen.add(f)
                export_fields.append(f)
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=export_fields, extrasaction="ignore"
                )
                writer.writeheader()
                for row in self._display_rows:
                    writer.writerow({k: row.get(k) for k in export_fields})
        except OSError as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return
        self._status_var.set(
            f"Exported {len(self._display_rows)} rows → {os.path.basename(path)}"
        )
