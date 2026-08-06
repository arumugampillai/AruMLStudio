"""Multi-model Feature Studio — Phase 4.4 UI (join Importance/Dist/Drift)."""

from __future__ import annotations

import csv
import os
import threading
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
    ("feature", "Feature", 160),
    ("in_a", "in A", 50),
    ("in_b", "in B", 50),
    ("rank_gain_a", "Rank A", 70),
    ("rank_gain_b", "Rank B", 70),
    ("rank_gain_delta", "Δ Rank", 70),
    ("risk_a", "Risk A", 70),
    ("risk_b", "Risk B", 70),
    ("risk_score_delta", "Δ Risk Sc", 80),
    ("drift_a", "Drift A", 70),
    ("drift_b", "Drift B", 70),
    ("null_pct_a", "null% A", 70),
    ("null_pct_b", "null% B", 70),
)


class MultiModelStudioPanel(ttk.Frame, LazyLoadMixin):
    """Compare two models' studio artifacts (join-only, no recompute)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        model_a_var: tk.StringVar | None = None,
        model_b_var: tk.StringVar | None = None,
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
        self._sort_col = "rank_gain_delta"
        self._sort_desc = True
        self._sort_abs = True
        self._busy = False
        self._pending: tuple[str, str] | None = None

        self._status_var = tk.StringVar(
            value="Select two models with studio artifacts, then Compute."
        )
        self._model_a_var = (
            model_a_var if model_a_var is not None else tk.StringVar()
        )
        self._model_b_var = (
            model_b_var if model_b_var is not None else tk.StringVar()
        )
        self._filter_var = filter_var if filter_var is not None else tk.StringVar()
        self._top_n_var = top_n_var if top_n_var is not None else tk.StringVar(value="30")
        self._top_n_only = (
            top_n_only if top_n_only is not None else tk.BooleanVar(value=False)
        )
        self._require_imp = tk.BooleanVar(value=False)

        self._summary_vars = {
            "a": tk.StringVar(value="—"),
            "b": tk.StringVar(value="—"),
            "common": tk.StringVar(value="—"),
            "only_a": tk.StringVar(value="—"),
            "only_b": tk.StringVar(value="—"),
            "loaded": tk.StringVar(value="—"),
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
        if self._pending:
            self.apply_model_names(self._model_names)

    def open_with_models(self, model_a: str, model_b: str) -> None:
        self._pending = (str(model_a or "").strip(), str(model_b or "").strip())
        self._model_a_var.set(self._pending[0])
        self._model_b_var.set(self._pending[1])

    def apply_model_names(self, names: list[str]) -> None:
        self._model_names = list(names)
        self._status_var.set(
            f"{len(names)} model(s) available."
            if names
            else "No trained models on disk."
        )
        if self._pending:
            a, b = self._pending
            self._pending = None
            if a in names and b in names and a != b:
                self._model_a_var.set(a)
                self._model_b_var.set(b)
                self._on_compare()

    def refresh(self, *, lazy: bool = False) -> None:
        del lazy
        self.apply_model_names(self._model_names)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        hdr = ttk.Frame(self, padding=(8, 8, 8, 4))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Feature Studio Compare", font=SECTION_FONT).pack(side="left")
        ttk.Label(
            hdr,
            text="Join Importance · Distribution · Drift · does not replace metrics Compare",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            hdr, text="Require Importance", variable=self._require_imp
        ).pack(side="right")

        summary = ttk.LabelFrame(self, text="Pair Summary", padding=8)
        summary.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        for i in range(7):
            summary.columnconfigure(i, weight=1)
        fields = (
            ("Model A", "a"),
            ("Model B", "b"),
            ("Common", "common"),
            ("Only A", "only_a"),
            ("Only B", "only_b"),
            ("Studios loaded", "loaded"),
            ("Join time", "compute"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(summary)
            cell.grid(row=0, column=i, sticky="ew", padx=4)
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key]).pack(anchor="w")

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
            if key in ("in_a", "in_b", "risk_a", "risk_b"):
                anchor = "center"
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "feature"))
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.tag_configure("rank_shift", background="#e8f1ff")
        self._tree.tag_configure("risk_shift", background="#fff6e0")
        self._tree.tag_configure("exclusive", foreground=COL_WARN)

        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground=COL_MUTED,
            padding=(8, 2, 8, 8),
        ).grid(row=3, column=0, sticky="ew")

    def _selection(self) -> tuple[str, str] | None:
        a = str(self._model_a_var.get() or "").strip()
        b = str(self._model_b_var.get() or "").strip()
        if not a or not b:
            messagebox.showwarning(
                "Feature Studio Compare", "Select Model A and Model B.", parent=self
            )
            return None
        if a == b:
            messagebox.showwarning(
                "Feature Studio Compare", "Choose two different models.", parent=self
            )
            return None
        return a, b

    def _on_load(self) -> None:
        sel = self._selection()
        if not sel:
            return
        a, b = sel
        from chain_replay_ml.multi_model_studio.writer import load_studio_artifacts

        loaded = load_studio_artifacts(self._data_dir(), a, b)
        if not loaded:
            self._status_var.set("No pair artifacts yet — click Compute.")
            messagebox.showinfo(
                "Feature Studio Compare",
                "No saved pair artifacts found.\nClick Compute to join studio outputs.",
                parent=self,
            )
            return
        self._apply_result(loaded["comparison"], loaded.get("meta") or {}, a, b)

    def _on_compare(self) -> None:
        sel = self._selection()
        if not sel:
            return
        if self._busy:
            return
        a, b = sel
        self._busy = True
        self._status_var.set(f"Joining studio artifacts: {a} vs {b}…")
        data_dir = self._data_dir()
        require = ("importance",) if self._require_imp.get() else ()

        def work() -> None:
            err: str | None = None
            result: Any = None
            try:
                from chain_replay_ml.multi_model_studio import run_multi_model_studio

                result = run_multi_model_studio(
                    data_dir=data_dir,
                    model_a=a,
                    model_b=b,
                    require=require,
                )
                if not result.ok:
                    err = result.error or "Compare failed"
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._busy = False
                if err:
                    self._status_var.set(f"Compare failed: {err}")
                    messagebox.showerror("Feature Studio Compare", err, parent=self)
                    return
                assert result is not None
                self._apply_result(result.comparison, result.meta, a, b)

            self.after(0, done)

        threading.Thread(target=work, name="mms-compare", daemon=True).start()

    def _apply_result(
        self,
        rows: list[dict[str, Any]],
        meta: dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> None:
        self._rows = [r for r in rows if isinstance(r, dict)]
        self._meta = dict(meta or {})
        self._summary_vars["a"].set(model_a)
        self._summary_vars["b"].set(model_b)
        self._summary_vars["common"].set(fmt_val(self._meta.get("common_count")))
        self._summary_vars["only_a"].set(fmt_val(self._meta.get("only_a_count")))
        self._summary_vars["only_b"].set(fmt_val(self._meta.get("only_b_count")))
        loaded = self._meta.get("artifacts_loaded") or {}
        parts: list[str] = []
        for side, label in (("a", "A"), ("b", "B")):
            flags = loaded.get(side) or {}
            on = [k for k, v in flags.items() if v]
            parts.append(f"{label}:{'+'.join(on) if on else 'none'}")
        self._summary_vars["loaded"].set(" · ".join(parts))
        wall = self._meta.get("wall_time_sec")
        self._summary_vars["compute"].set(
            f"{float(wall):.3f}s" if wall is not None else "—"
        )
        self._apply_filter_sort()
        self._status_var.set(f"Loaded {len(self._rows)} features · sort={self._sort_col}")

    def _on_sort_header(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col not in ("feature", "risk_a", "risk_b", "in_a", "in_b")
            self._sort_abs = col in ("rank_gain_delta", "risk_score_delta", "null_pct_delta")
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
            rows = [r for r in rows if needle in str(r.get("feature") or "").lower()]

        key = self._sort_col
        use_abs = self._sort_abs and key in (
            "rank_gain_delta",
            "risk_score_delta",
            "null_pct_delta",
        )

        def sort_key(r: dict[str, Any]) -> tuple:
            val = r.get(key)
            if key in ("feature", "risk_a", "risk_b"):
                return (0, str(val or "").lower())
            if key in ("in_a", "in_b"):
                return (0, 1 if val else 0)
            try:
                num = float(val)
                return (0, abs(num) if use_abs else num)
            except (TypeError, ValueError):
                return (1, 0.0)

        rows.sort(key=sort_key, reverse=self._sort_desc)

        if self._top_n_only.get():
            by_delta = sorted(
                self._rows,
                key=lambda r: abs(float(r["rank_gain_delta"]))
                if r.get("rank_gain_delta") is not None
                else -1.0,
                reverse=True,
            )[: self._top_n()]
            keep = {str(r.get("feature")) for r in by_delta}
            rows = [r for r in rows if str(r.get("feature")) in keep]

        self._display_rows = rows
        self._render_table()

    def _render_table(self) -> None:
        tree = self._tree
        tree.delete(*tree.get_children())
        for row in self._display_rows:
            tags: list[str] = []
            rd = row.get("rank_gain_delta")
            rs = row.get("risk_score_delta")
            try:
                if rd is not None and abs(float(rd)) >= 5:
                    tags.append("rank_shift")
            except (TypeError, ValueError):
                pass
            try:
                if rs is not None and abs(float(rs)) >= 0.05:
                    tags.append("risk_shift")
            except (TypeError, ValueError):
                pass
            if bool(row.get("in_a")) != bool(row.get("in_b")):
                tags.append("exclusive")
            values = (
                str(row.get("feature") or ""),
                "✓" if row.get("in_a") else "",
                "✓" if row.get("in_b") else "",
                fmt_val(row.get("rank_gain_a")),
                fmt_val(row.get("rank_gain_b")),
                fmt_num(row.get("rank_gain_delta"), 0),
                fmt_val(row.get("risk_a")),
                fmt_val(row.get("risk_b")),
                fmt_num(row.get("risk_score_delta"), 4),
                fmt_num(row.get("drift_a"), 4),
                fmt_num(row.get("drift_b"), 4),
                fmt_num(row.get("null_pct_a"), 2),
                fmt_num(row.get("null_pct_b"), 2),
            )
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _on_export(self) -> None:
        if not self._display_rows:
            messagebox.showinfo("Export", "Nothing to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export multi-model feature CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="multi_model_feature_compare.csv",
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
