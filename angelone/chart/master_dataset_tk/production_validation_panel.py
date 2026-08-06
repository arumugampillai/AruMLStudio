"""Production Validation — Feature Studio tab (Phase A resolve + Phase B compute)."""

from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import COL_MUTED, SECTION_FONT, fmt_num, fmt_val

_KEEP_WATCH = frozenset({"KEEP", "WATCH"})
_REC_TABS = ("KEEP", "WATCH", "REMOVE")


_TABLE_COLS = (
    ("feature", "Feature", 160),
    ("holdout_rank", "Holdout Rank", 90),
    ("unseen_rank", "Unseen Rank", 90),
    ("rank_change", "Rank Change", 90),
    ("holdout_importance", "Holdout Importance", 110),
    ("unseen_importance", "Unseen Importance", 110),
    ("importance_difference", "Importance Difference", 120),
    ("recommendation", "Recommendation", 100),
)

_STAGE_LABELS = {
    "resolve_unseen": "Resolve unseen dataset",
    "load_model": "Load model",
    "configure_inference": "Configure inference device",
    "load_holdout": "Load holdout matrix",
    "load_unseen": "Load unseen day(s)",
    "permutation_holdout": "Permutation importance — Holdout",
    "permutation_unseen": "Permutation importance — Unseen",
    "permutation": "Permutation importance",
    "write_artifacts": "Write artifacts",
    "done": "Done",
    "error": "Error",
}

# Overall bar weights (sum ≈ 100). Permutation dominates wall time.
_STAGE_WEIGHTS = {
    "resolve_unseen": (0, 5),
    "load_model": (5, 7),
    "configure_inference": (7, 10),
    "load_holdout": (10, 20),
    "load_unseen": (20, 32),
    "permutation_holdout": (32, 65),
    "permutation_unseen": (65, 95),
    "permutation": (32, 95),
    "write_artifacts": (95, 100),
    "done": (100, 100),
}


class ProductionValidationPanel(ttk.Frame, LazyLoadMixin):
    """Holdout → True Unseen Days — resolve status + importance table."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        model_var: tk.StringVar | None = None,
        filter_var: tk.StringVar | None = None,
        top_n_var: tk.StringVar | None = None,
        top_n_only: tk.BooleanVar | None = None,
        on_create_model: Callable[[str, list[str], str | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_create_model = on_create_model
        self._model_names: list[str] = []
        self._busy = False
        self._rows: list[dict[str, Any]] = []
        self._filter_var = filter_var if filter_var is not None else tk.StringVar()
        self._top_n_var = top_n_var if top_n_var is not None else tk.StringVar(value="20")
        self._top_n_only = (
            top_n_only if top_n_only is not None else tk.BooleanVar(value=False)
        )

        self._status_var = tk.StringVar(
            value="Select a model — Resolve Unseen Dataset, then Compute."
        )
        self._phase_var = tk.StringVar(value="Idle")
        self._model_var = model_var if model_var is not None else tk.StringVar()
        self._summary_vars = {
            "model": tk.StringVar(value="—"),
            "seen": tk.StringVar(value="—"),
            "unseen": tk.StringVar(value="—"),
            "dataset": tk.StringVar(value="—"),
            "path": tk.StringVar(value="—"),
            "reuse": tk.StringVar(value="—"),
            "compute": tk.StringVar(value="compute coming"),
            "hash": tk.StringVar(value="—"),
        }
        self._diag_var = tk.StringVar(value="—")
        self._prod_var = tk.StringVar(value="—")
        self._coverage_var = tk.StringVar(value="—")
        self._overview_var = tk.StringVar(value="—")
        self._device_var = tk.StringVar(value="—")
        self._feat_summary_var = tk.StringVar(value="—")
        self._last_timeline_stage = ""

        self._build_ui()
        self._filter_var.trace_add("write", lambda *_: self._render_table())
        self._top_n_only.trace_add("write", lambda *_: self._render_table())
        self._top_n_var.trace_add("write", lambda *_: self._render_table())
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        name = self._selected_model()
        if name and self._summary_vars["dataset"].get() in ("—", ""):
            self._load_cached_status(quiet=True)

    def open_with_model(self, model_name: str) -> None:
        self._model_var.set(str(model_name or "").strip())

    def apply_model_names(self, names: list[str]) -> None:
        self._model_names = list(names)
        if not self._busy:
            self._status_var.set(
                f"{len(names)} model(s) available."
                if names
                else "No trained models on disk."
            )

    def refresh(self, *, lazy: bool = False) -> None:
        del lazy
        self.apply_model_names(self._model_names)

    def mark_unavailable(self, message: str) -> None:
        for key in self._summary_vars:
            if key == "compute":
                self._summary_vars[key].set("compute coming")
            else:
                self._summary_vars[key].set("—")
        self._diag_var.set("—")
        self._prod_var.set("—")
        self._coverage_var.set("—")
        self._overview_var.set("—")
        self._device_var.set("—")
        self._feat_summary_var.set("—")
        self._rows = []
        self._clear_table()
        self._status_var.set(message)
        if not self._busy:
            self._phase_var.set("Idle")
            self._set_progress_pct(0)

    def apply_artifacts(
        self, loaded: dict[str, Any] | None, model_name: str
    ) -> None:
        """Apply Phase B artifacts and/or Phase A resolve status."""
        if not loaded:
            self.mark_unavailable("No Production Validation status yet.")
            return
        # Phase B payload from load_validation_artifacts
        if isinstance(loaded.get("rows"), list):
            status = loaded.get("unseen_status") or {}
            if status:
                self._apply_status(status, model_name)
            else:
                self._summary_vars["model"].set(model_name or "—")
            self._apply_compute_payload(loaded, model_name)
            return
        # Phase A status dict only
        self._apply_status(loaded, model_name)

    def _select_tab(self, tab: ttk.Frame) -> None:
        try:
            self._notebook.select(tab)
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        hdr = ttk.Frame(self, padding=(8, 8, 8, 4))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Production Validation", font=SECTION_FONT).pack(
            side="left"
        )
        ttk.Label(
            hdr,
            text="Holdout → True Unseen Days · dual confidence · Phase B",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(12, 0))
        self._btn_compute = ttk.Button(
            hdr,
            text="Compute",
            command=self._on_compute,
        )
        self._btn_compute.pack(side="right", padx=(8, 0))
        self._btn_resolve = ttk.Button(
            hdr,
            text="Resolve Unseen Dataset",
            command=self._on_resolve,
        )
        self._btn_resolve.pack(side="right", padx=(8, 0))
        self._btn_refresh = ttk.Button(
            hdr,
            text="Refresh Status",
            command=lambda: self._load_cached_status(quiet=False),
        )
        self._btn_refresh.pack(side="right", padx=(8, 0))
        # Packed after Refresh so side=right places it immediately before Refresh.
        self._btn_create_model = ttk.Button(
            hdr,
            text="Create Model",
            command=self._on_create_model_clicked,
        )
        self._btn_create_model.pack(side="right")
        self._bind_create_model_tooltip()

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        # Tab 1 — Summary (overview + dual confidence)
        summary_tab = ttk.Frame(self._notebook, padding=8)
        summary_tab.columnconfigure(0, weight=1)
        summary_tab.columnconfigure(1, weight=1)
        ttk.Label(summary_tab, text="Overview", foreground=COL_MUTED).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(summary_tab, textvariable=self._overview_var, wraplength=880).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        ttk.Label(summary_tab, text="Inference device", foreground=COL_MUTED).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Label(summary_tab, textvariable=self._device_var, wraplength=880).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        ttk.Label(summary_tab, text="Diagnosis", foreground=COL_MUTED).grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(summary_tab, textvariable=self._diag_var, wraplength=420).grid(
            row=5, column=0, sticky="w", pady=(2, 0)
        )
        ttk.Label(
            summary_tab, text="Production Confirmation", foreground=COL_MUTED
        ).grid(row=4, column=1, sticky="w", pady=(10, 0))
        ttk.Label(summary_tab, textvariable=self._prod_var, wraplength=420).grid(
            row=5, column=1, sticky="w", pady=(2, 0)
        )
        ttk.Label(
            summary_tab,
            textvariable=self._coverage_var,
            foreground=COL_MUTED,
            wraplength=880,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._tab_summary = summary_tab

        # Tab 2 — Unseen Dataset Status (resolve / reuse / path / days)
        unseen_tab = ttk.Frame(self._notebook, padding=8)
        for i in range(4):
            unseen_tab.columnconfigure(i, weight=1)
        fields = (
            ("Model", "model"),
            ("Seen days", "seen"),
            ("Unseen days", "unseen"),
            ("Dataset", "dataset"),
            ("Path / name", "path"),
            ("Reuse", "reuse"),
            ("Identity hash", "hash"),
            ("Compute", "compute"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(unseen_tab)
            cell.grid(
                row=i // 4,
                column=i % 4,
                sticky="ew",
                padx=4,
                pady=(0 if i < 4 else 8, 0),
            )
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key], wraplength=220).pack(
                anchor="w"
            )
        self._tab_unseen = unseen_tab

        # Tab 3 — Feature Validation (KEEP / WATCH / REMOVE subtabs)
        table_tab = ttk.Frame(self._notebook, padding=4)
        table_tab.columnconfigure(0, weight=1)
        table_tab.rowconfigure(1, weight=1)
        ttk.Label(
            table_tab,
            textvariable=self._feat_summary_var,
            wraplength=960,
            padding=(4, 2, 4, 4),
        ).grid(row=0, column=0, sticky="ew")

        self._feat_rec_nb = ttk.Notebook(table_tab)
        self._feat_rec_nb.grid(row=1, column=0, sticky="nsew")
        self._feat_trees: dict[str, ttk.Treeview] = {}
        cols = tuple(c[0] for c in _TABLE_COLS)

        for rec in _REC_TABS:
            frame = ttk.Frame(self._feat_rec_nb, padding=2)
            frame.columnconfigure(0, weight=1)
            tree_row = 0
            if rec == "REMOVE":
                bar = ttk.Frame(frame)
                bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
                self._btn_update_recs = ttk.Button(
                    bar,
                    text="Update Registry Recommendations",
                    command=self._on_update_registry_recommendations,
                )
                self._btn_update_recs.pack(side="left")
                ttk.Label(
                    bar,
                    text=(
                        "Persists recommendations only — does not remove "
                        "pipeline features or retire registry features."
                    ),
                    foreground=COL_MUTED,
                ).pack(side="left", padx=(8, 0))
                tree_row = 1
            frame.rowconfigure(tree_row, weight=1)

            tree = ttk.Treeview(
                frame, columns=cols, show="headings", selectmode="browse"
            )
            for key, title, width in _TABLE_COLS:
                tree.heading(key, text=title)
                anchor = "w" if key in ("feature", "recommendation") else "e"
                tree.column(
                    key, width=width, anchor=anchor, stretch=(key == "feature")
                )
            vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=tree_row, column=0, sticky="nsew")
            vsb.grid(row=tree_row, column=1, sticky="ns")
            hsb.grid(row=tree_row + 1, column=0, sticky="ew")
            self._feat_trees[rec] = tree
            self._feat_rec_nb.add(frame, text=rec)

        # Legacy alias used by older helpers (points at KEEP tree).
        self._tree = self._feat_trees["KEEP"]
        self._tab_features = table_tab

        # Tab 4 — Compute Progress (bar + timeline)
        progress_tab = ttk.Frame(self._notebook, padding=8)
        progress_tab.columnconfigure(0, weight=1)
        progress_tab.rowconfigure(2, weight=1)
        ttk.Label(progress_tab, textvariable=self._phase_var).grid(
            row=0, column=0, sticky="w"
        )
        self._progress = ttk.Progressbar(
            progress_tab, mode="determinate", maximum=100, value=0
        )
        self._progress.grid(row=1, column=0, sticky="ew", pady=(4, 4))
        timeline_frame = ttk.Frame(progress_tab)
        timeline_frame.grid(row=2, column=0, sticky="nsew")
        timeline_frame.columnconfigure(0, weight=1)
        timeline_frame.rowconfigure(0, weight=1)
        self._timeline = tk.Listbox(
            timeline_frame,
            height=12,
            font=("Consolas", 8),
            activestyle="none",
            exportselection=False,
        )
        tl_sb = ttk.Scrollbar(
            timeline_frame, orient="vertical", command=self._timeline.yview
        )
        self._timeline.configure(yscrollcommand=tl_sb.set)
        self._timeline.grid(row=0, column=0, sticky="nsew")
        tl_sb.grid(row=0, column=1, sticky="ns")
        self._tab_progress = progress_tab

        self._notebook.add(self._tab_summary, text="Summary")
        self._notebook.add(self._tab_unseen, text="Unseen Dataset Status")
        self._notebook.add(self._tab_features, text="Feature Validation")
        self._notebook.add(self._tab_progress, text="Compute Progress")

        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground=COL_MUTED,
            padding=(8, 2, 8, 8),
        ).grid(row=2, column=0, sticky="ew")

    def _selected_model(self) -> str:
        return str(self._model_var.get() or "").strip()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        buttons = [
            self._btn_compute,
            self._btn_resolve,
            self._btn_refresh,
            self._btn_create_model,
        ]
        if hasattr(self, "_btn_update_recs"):
            buttons.append(self._btn_update_recs)
        for btn in buttons:
            try:
                btn.configure(state=state)
            except tk.TclError:
                pass

    def _keep_watch_features(self) -> tuple[list[str], int, int]:
        """KEEP + WATCH feature names from loaded validation rows (excludes REMOVE)."""
        keep: list[str] = []
        watch: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            if not isinstance(row, dict):
                continue
            rec = str(row.get("recommendation") or "").strip().upper()
            if rec not in _KEEP_WATCH:
                continue
            name = str(row.get("feature") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            if rec == "KEEP":
                keep.append(name)
            else:
                watch.append(name)
        return keep + watch, len(keep), len(watch)

    def _create_model_tooltip_text(self) -> str:
        if not self._rows:
            return "Load validation results first (Compute / Refresh Status)."
        features, keep_n, watch_n = self._keep_watch_features()
        if not features:
            return "No KEEP or WATCH features in the current validation results."
        return (
            f"Open Create Model with {len(features)} features "
            f"(KEEP {keep_n} · WATCH {watch_n}; REMOVE excluded)."
        )

    def _bind_create_model_tooltip(self) -> None:
        tip: dict[str, Any] = {"win": None}

        def _hide(_: Any = None) -> None:
            win = tip.get("win")
            if win is not None:
                try:
                    win.destroy()
                except tk.TclError:
                    pass
                tip["win"] = None

        def _show(_: Any = None) -> None:
            _hide()
            text = self._create_model_tooltip_text()
            if not text:
                return
            win = tk.Toplevel(self)
            win.wm_overrideredirect(True)
            try:
                win.attributes("-topmost", True)
            except tk.TclError:
                pass
            ttk.Label(
                win,
                text=text,
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padding=(6, 3),
            ).pack()
            x = self._btn_create_model.winfo_rootx()
            y = self._btn_create_model.winfo_rooty() + self._btn_create_model.winfo_height() + 4
            win.wm_geometry(f"+{x}+{y}")
            tip["win"] = win

        self._btn_create_model.bind("<Enter>", _show)
        self._btn_create_model.bind("<Leave>", _hide)

    def _source_training_dataset(self) -> str | None:
        """Training dataset for the selected model (not the unseen_* resolve dataset)."""
        name = self._selected_model()
        if not name:
            return None
        try:
            from chain_replay_ml.training.paths import model_package_dir

            pkg = model_package_dir(self._data_dir(), name)
        except Exception:
            return None
        for fname in ("config.json", "training_config.json", "metadata.json"):
            path = os.path.join(pkg, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if not isinstance(doc, dict):
                continue
            ds = doc.get("dataset") or doc.get("dataset_name")
            meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
            if not ds and meta:
                ds = meta.get("dataset") or meta.get("dataset_name")
            raw = str(ds or "").strip()
            if raw:
                return raw
        return None

    def _on_create_model_clicked(self) -> None:
        if self._busy:
            return
        if not self._rows:
            messagebox.showinfo(
                "Create Model",
                "No validation results loaded.\n"
                "Resolve Unseen Dataset, then Compute (or Refresh Status) "
                "before creating a model from KEEP/WATCH features.",
                parent=self,
            )
            return
        features, keep_n, watch_n = self._keep_watch_features()
        if not features:
            messagebox.showinfo(
                "Create Model",
                "No KEEP or WATCH features in the current validation results.\n"
                "REMOVE features are excluded from Create Model handoff.",
                parent=self,
            )
            return
        model_name = self._selected_model()
        if not model_name:
            messagebox.showwarning(
                "Create Model", "Select a model first.", parent=self
            )
            return
        if not self._on_create_model:
            messagebox.showerror(
                "Create Model",
                "Create Model navigation is not wired in this window.",
                parent=self,
            )
            return
        dataset = self._source_training_dataset()
        self._status_var.set(
            f"Opening Create Model · {len(features)} features "
            f"(KEEP {keep_n} · WATCH {watch_n})…"
        )
        self._on_create_model(model_name, features, dataset)

    def _set_progress_pct(self, pct: float) -> None:
        try:
            self._progress.configure(value=max(0.0, min(100.0, float(pct))))
        except tk.TclError:
            pass

    def _clear_timeline(self) -> None:
        self._last_timeline_stage = ""
        try:
            self._timeline.delete(0, tk.END)
        except tk.TclError:
            pass

    def _append_timeline(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            self._timeline.insert(tk.END, f"{stamp}  {line}")
            self._timeline.see(tk.END)
        except tk.TclError:
            pass

    def _stage_label(self, stage: str) -> str:
        return _STAGE_LABELS.get(stage, stage.replace("_", " ") or "working")

    def _progress_pct_for(self, info: dict[str, Any]) -> float:
        stage = str(info.get("stage") or "")
        lo, hi = _STAGE_WEIGHTS.get(stage, (0, 100))
        done = info.get("done")
        total = info.get("total")
        if done is not None and total is not None:
            try:
                t = max(1, int(total))
                d = max(0, min(t, int(done)))
                return lo + (hi - lo) * (d / t)
            except (TypeError, ValueError):
                return float(lo)
        return float(lo)

    def _apply_progress(self, info: dict[str, Any], *, model_name: str) -> None:
        stage = str(info.get("stage") or "")
        feat = str(info.get("feature") or "").strip()
        done = info.get("done")
        total = info.get("total")
        label = self._stage_label(stage)
        msg = str(info.get("message") or "").strip()
        device = str(info.get("device") or "").strip()

        phase = f"{label}"
        if device and stage == "configure_inference":
            phase += f" · {device}"
            if info.get("gpu_active") is False and info.get("fallback_reason"):
                phase += " (CPU fallback)"
        if feat:
            phase += f" · {feat}"
        if done is not None and total is not None:
            phase += f" ({done}/{total})"
        self._phase_var.set(phase)
        self._status_var.set(f"{model_name} — {phase}")
        self._set_progress_pct(self._progress_pct_for(info))

        # Timeline: one line per stage enter; feature ticks only every 10 or last.
        if stage and stage != self._last_timeline_stage:
            self._last_timeline_stage = stage
            extra = ""
            if stage == "configure_inference":
                if msg:
                    extra = f" · {msg}"
                elif device:
                    extra = f" · {device}"
                    if info.get("fallback_reason") and not info.get("gpu_active"):
                        extra += f" ({info.get('fallback_reason')})"
            self._append_timeline(f"▶ {label}{extra}")
        elif feat and done is not None and total is not None:
            try:
                d_i, t_i = int(done), int(total)
            except (TypeError, ValueError):
                return
            if d_i == 0 or d_i % 10 == 0 or d_i + 1 >= t_i:
                self._append_timeline(f"  {label}: feature {d_i + 1}/{t_i} · {feat}")

    def _clear_table(self) -> None:
        trees = getattr(self, "_feat_trees", None)
        if isinstance(trees, dict) and trees:
            for tree in trees.values():
                for item in tree.get_children():
                    tree.delete(item)
            return
        if hasattr(self, "_tree"):
            for item in self._tree.get_children():
                self._tree.delete(item)

    def _apply_status(self, doc: dict[str, Any], model_name: str) -> None:
        self._summary_vars["model"].set(model_name or str(doc.get("model_name") or "—"))
        seen_n = doc.get("seen_day_count")
        if seen_n is None and isinstance(doc.get("seen_days"), list):
            seen_n = len(doc["seen_days"])
        unseen_n = doc.get("unseen_day_count")
        if unseen_n is None and isinstance(doc.get("unseen_days"), list):
            unseen_n = len(doc["unseen_days"])
        self._summary_vars["seen"].set(str(seen_n if seen_n is not None else "—"))
        self._summary_vars["unseen"].set(str(unseen_n if unseen_n is not None else "—"))
        ds = str(doc.get("dataset_name") or "").strip()
        self._summary_vars["dataset"].set(ds or "—")
        path = str(doc.get("parquet_path") or doc.get("json_path") or ds or "").strip()
        self._summary_vars["path"].set(path or "—")
        if doc.get("reused"):
            reuse = "Reused"
        elif doc.get("created"):
            reuse = "Created"
        elif str(doc.get("status") or "") == "empty":
            reuse = "N/A (no unseen days)"
        else:
            reuse = str(doc.get("status") or "—")
        self._summary_vars["reuse"].set(reuse)
        self._summary_vars["hash"].set(str(doc.get("identity_hash") or "—"))
        self._summary_vars["compute"].set(
            str(doc.get("compute_note") or "compute coming")
        )
        msg = str(doc.get("message") or doc.get("error") or "").strip()
        if not doc.get("ok") and doc.get("error"):
            self._status_var.set(msg or str(doc.get("error")))
        else:
            note = str(doc.get("compute_note") or "")
            if note and note != "compute coming":
                self._status_var.set(msg or note)
            else:
                self._status_var.set(
                    msg or "Unseen dataset ready — click Compute for Holdout vs Unseen."
                )

    def _apply_compute_payload(self, loaded: dict[str, Any], model_name: str) -> None:
        from chain_replay_ml.production_validation.rules import (
            build_feature_validation_summary,
            enrich_comparison_rows_from_importances,
        )

        rows = [r for r in (loaded.get("rows") or []) if isinstance(r, dict)]
        rows, enriched = enrich_comparison_rows_from_importances(rows)
        summary = loaded.get("summary") if isinstance(loaded.get("summary"), dict) else {}
        meta = loaded.get("meta") if isinstance(loaded.get("meta"), dict) else {}
        if enriched:
            summary = dict(summary)
            summary["feature_validation"] = build_feature_validation_summary(rows)
        self._rows = rows
        self._summary_vars["model"].set(model_name or str(summary.get("model_name") or "—"))
        if summary.get("dataset_name"):
            self._summary_vars["dataset"].set(str(summary["dataset_name"]))
        if summary.get("unseen_day_count") is not None:
            self._summary_vars["unseen"].set(str(summary["unseen_day_count"]))

        scored = summary.get("feature_count_scored") or len(rows)
        selected = summary.get("feature_count_selected")
        ho_rows = summary.get("holdout_rows") or meta.get("holdout_row_count")
        un_rows = summary.get("unseen_rows") or meta.get("unseen_row_count")
        day_n = summary.get("unseen_day_count")
        wall = meta.get("wall_time_sec")
        overview_parts = [
            f"{fmt_val(scored)} features scored"
            + (f" / {fmt_val(selected)} selected" if selected is not None else ""),
            f"holdout {fmt_val(ho_rows)} rows",
            f"unseen {fmt_val(un_rows)} rows"
            + (f" · {fmt_val(day_n)} day(s)" if day_n is not None else ""),
        ]
        if wall is not None:
            try:
                overview_parts.append(f"{float(wall):.1f}s wall")
            except (TypeError, ValueError):
                pass
        self._overview_var.set(" · ".join(overview_parts))

        device = str(
            summary.get("inference_device") or meta.get("inference_device") or ""
        ).strip()
        fallback = str(
            summary.get("inference_fallback_reason")
            or meta.get("inference_fallback_reason")
            or ""
        ).strip()
        gpu_active = summary.get("gpu_active")
        if gpu_active is None:
            gpu_active = meta.get("gpu_active")
        if device:
            if gpu_active:
                self._device_var.set(f"{device} (XGBoost GPU predict)")
            elif fallback:
                self._device_var.set(f"{device} — {fallback}")
            else:
                self._device_var.set(device)
        else:
            self._device_var.set("—")

        diag = summary.get("diagnosis") if isinstance(summary.get("diagnosis"), dict) else {}
        prod = (
            summary.get("production_confirmation")
            if isinstance(summary.get("production_confirmation"), dict)
            else {}
        )
        if diag:
            self._diag_var.set(
                f"{diag.get('label') or '—'} · Confidence {diag.get('confidence_pct', '—')}%"
            )
        else:
            self._diag_var.set("—")
        if prod:
            self._prod_var.set(
                f"{prod.get('status') or '—'} · {prod.get('confidence_pct', '—')}% · "
                f"{prod.get('explanation') or ''}"
            )
        else:
            self._prod_var.set("—")

        coverage = str(meta.get("unseen_coverage") or summary.get("coverage") or "")
        rows_n = summary.get("unseen_rows") or meta.get("unseen_row_count")
        full_n = summary.get("unseen_rows_full") or meta.get("unseen_rows_full")
        capped = summary.get("unseen_rows_capped") or meta.get("unseen_rows_capped")
        if capped:
            self._coverage_var.set(
                f"Coverage: CAPPED unseen rows {fmt_val(rows_n)} / {fmt_val(full_n)} "
                f"(not full day — clear UI cap)."
            )
        elif coverage or rows_n is not None:
            self._coverage_var.set(
                f"Coverage: whole unseen day(s) · {fmt_val(rows_n)} rows · "
                f"{fmt_val(summary.get('feature_count_scored') or len(rows))} features "
                f"(model selected set)."
            )
        else:
            self._coverage_var.set("—")

        feat_val = (
            summary.get("feature_validation")
            if isinstance(summary.get("feature_validation"), dict)
            else {}
        )
        # Rebuild when missing or legacy strip lacks rank stats.
        if rows and (
            not feat_val
            or feat_val.get("average_rank_change") is None
            or feat_val.get("median_rank_change") is None
            or enriched
        ):
            feat_val = build_feature_validation_summary(rows)
        if feat_val:
            self._feat_summary_var.set(
                f"KEEP {fmt_val(feat_val.get('keep_count'))} · "
                f"WATCH {fmt_val(feat_val.get('watch_count'))} · "
                f"REMOVE {fmt_val(feat_val.get('remove_count'))} · "
                f"Avg Rank Change {fmt_num(feat_val.get('average_rank_change'))} · "
                f"Median Rank Change {fmt_num(feat_val.get('median_rank_change'))} · "
                f"Stable Features {fmt_num(feat_val.get('stable_features_pct'))}%"
            )
        else:
            self._feat_summary_var.set("—")

        compute_note = (
            f"computed · {len(rows)} features"
            + (f" · {float(wall):.1f}s" if wall is not None else "")
            + (f" · {device}" if device else "")
        )
        if enriched or loaded.get("rank_fields_enriched") or meta.get(
            "rank_fields_enriched_from_importances"
        ):
            compute_note += " · ranks derived from cached importances"
        self._summary_vars["compute"].set(compute_note)
        self._render_table()
        self._status_var.set(
            f"Loaded {len(rows)} feature validation rows · model selected features"
            + (f" · {device}" if device else "")
            + (
                " · ranks derived from importances (no recompute)"
                if enriched or loaded.get("rank_fields_enriched")
                else ""
            )
            + "."
        )
        self._phase_var.set("Idle — results loaded")
        self._set_progress_pct(100 if rows else 0)

    def _top_n(self) -> int:
        try:
            return max(1, int(self._top_n_var.get() or 20))
        except ValueError:
            return 20

    def _render_table(self) -> None:
        trees = getattr(self, "_feat_trees", None)
        if not isinstance(trees, dict) or not trees:
            return
        needle = str(self._filter_var.get() or "").strip().lower()
        rows = list(self._rows)
        if needle:
            rows = [r for r in rows if needle in str(r.get("feature") or "").lower()]
        if self._top_n_only.get():
            # Top N by worst rank drop (most negative Rank Change first).
            ranked = sorted(
                self._rows,
                key=lambda r: (
                    int(r["rank_change"]) if r.get("rank_change") is not None else 10**9,
                    float(r["importance_difference"])
                    if r.get("importance_difference") is not None
                    else 0.0,
                ),
            )[: self._top_n()]
            keep = {str(r.get("feature")) for r in ranked}
            rows = [r for r in rows if str(r.get("feature")) in keep]

        by_rec: dict[str, list[dict[str, Any]]] = {rec: [] for rec in _REC_TABS}
        for row in rows:
            rec = str(row.get("recommendation") or "").strip().upper()
            if rec in by_rec:
                by_rec[rec].append(row)

        self._clear_table()
        for rec in _REC_TABS:
            tree = trees[rec]
            for row in by_rec[rec]:
                tree.insert(
                    "",
                    "end",
                    values=(
                        str(row.get("feature") or ""),
                        fmt_val(row.get("holdout_rank")),
                        fmt_val(row.get("unseen_rank")),
                        fmt_val(row.get("rank_change")),
                        fmt_num(row.get("holdout_importance")),
                        fmt_num(row.get("unseen_importance")),
                        fmt_num(row.get("importance_difference")),
                        str(row.get("recommendation") or ""),
                    ),
                )
            if hasattr(self, "_feat_rec_nb"):
                try:
                    idx = _REC_TABS.index(rec)
                    self._feat_rec_nb.tab(idx, text=f"{rec} ({len(by_rec[rec])})")
                except tk.TclError:
                    pass

    def _on_update_registry_recommendations(self) -> None:
        """Persist PV recommendations for the selected model (no feature mutation)."""
        if self._busy:
            return
        name = self._selected_model()
        if not name:
            messagebox.showwarning(
                "Production Validation", "Select a model first.", parent=self
            )
            return
        if not self._rows:
            messagebox.showinfo(
                "Update Registry Recommendations",
                "Load validation results first (Compute / Refresh Status).",
                parent=self,
            )
            return

        remove_n = sum(
            1
            for r in self._rows
            if str(r.get("recommendation") or "").strip().upper() == "REMOVE"
        )
        keep_n = sum(
            1
            for r in self._rows
            if str(r.get("recommendation") or "").strip().upper() == "KEEP"
        )
        watch_n = sum(
            1
            for r in self._rows
            if str(r.get("recommendation") or "").strip().upper() == "WATCH"
        )
        if not messagebox.askyesno(
            "Update Registry Recommendations",
            (
                f"Persist recommendations for model «{name}»?\n\n"
                f"KEEP {keep_n} · WATCH {watch_n} · REMOVE {remove_n}\n\n"
                "This only updates the recommendation history.\n"
                "It does not remove pipeline features or retire registry features."
            ),
            parent=self,
        ):
            return

        self._set_busy(True)
        self._status_var.set(f"{name} — Updating registry recommendations…")
        data_dir = self._data_dir()

        def work() -> None:
            err: str | None = None
            result_doc: dict[str, Any] | None = None
            try:
                from chain_replay_ml.production_validation import (
                    persist_registry_recommendations,
                )

                result_doc = persist_registry_recommendations(
                    data_dir=data_dir,
                    model_name=name,
                )
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._set_busy(False)
                if err or not result_doc:
                    self._status_var.set(f"Update recommendations failed: {err or 'unknown'}")
                    messagebox.showerror(
                        "Update Registry Recommendations",
                        err or "Update failed",
                        parent=self,
                    )
                    return
                inserted = int(result_doc.get("inserted") or 0)
                updated = int(result_doc.get("updated") or 0)
                run_id = str(result_doc.get("production_validation_run_id") or "")
                self._status_var.set(
                    f"Recommendations updated · {inserted} new · {updated} refreshed"
                    + (f" · run {run_id[:8]}" if run_id else "")
                )
                messagebox.showinfo(
                    "Update Registry Recommendations",
                    (
                        f"Saved recommendations for «{name}».\n\n"
                        f"Inserted: {inserted}\n"
                        f"Updated: {updated}\n"
                        f"Features in store: {result_doc.get('feature_count') or 0}\n\n"
                        "No pipeline features were removed.\n"
                        "No registry features were retired."
                    ),
                    parent=self,
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _load_cached_status(self, *, quiet: bool = True) -> None:
        name = self._selected_model()
        if not name:
            if not quiet:
                messagebox.showwarning(
                    "Production Validation", "Select a model first.", parent=self
                )
            return
        from chain_replay_ml.production_validation.api import (
            load_unseen_dataset_status,
            load_validation_artifacts,
        )
        from chain_replay_ml.training.paths import model_package_dir

        pkg = model_package_dir(self._data_dir(), name)
        loaded = load_validation_artifacts(pkg)
        if loaded:
            self.apply_artifacts(loaded, name)
            return

        doc = load_unseen_dataset_status(self._data_dir(), name)
        if not doc:
            self.mark_unavailable(
                "No cached status — click Resolve Unseen Dataset."
            )
            if not quiet:
                messagebox.showinfo(
                    "Production Validation",
                    "No cached unseen-dataset status for this model.\n"
                    "Click Resolve Unseen Dataset, then Compute.",
                    parent=self,
                )
            return
        self._apply_status(doc, name)
        self._rows = []
        self._clear_table()
        self._diag_var.set("—")
        self._prod_var.set("—")
        self._coverage_var.set("Resolve ready — click Compute.")
        self._overview_var.set("—")
        self._device_var.set("—")
        self._feat_summary_var.set("—")
        self._select_tab(self._tab_unseen)

    def _on_resolve(self) -> None:
        if self._busy:
            return
        name = self._selected_model()
        if not name:
            messagebox.showwarning(
                "Production Validation", "Select a model first.", parent=self
            )
            return

        self._set_busy(True)
        self._select_tab(self._tab_progress)
        self._clear_timeline()
        self._append_timeline(f"Resolve started · {name}")
        self._phase_var.set("Resolve unseen dataset…")
        self._set_progress_pct(2)
        self._status_var.set(f"{name} — Resolving unseen dataset…")
        data_dir = self._data_dir()

        def work() -> None:
            err: str | None = None
            result_doc: dict[str, Any] | None = None
            try:
                from chain_replay_ml.production_validation import (
                    resolve_unseen_dataset_for_model,
                )

                result = resolve_unseen_dataset_for_model(
                    data_dir=data_dir,
                    model_name=name,
                    create_if_missing=True,
                )
                result_doc = result.as_dict()
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._set_busy(False)
                if err or result_doc is None:
                    self._phase_var.set("Resolve failed")
                    self._set_progress_pct(0)
                    self._append_timeline(f"✗ Resolve failed: {err or 'unknown'}")
                    self._status_var.set(f"Resolve failed: {err or 'unknown'}")
                    messagebox.showerror(
                        "Production Validation",
                        err or "Resolve failed",
                        parent=self,
                    )
                    return
                self._apply_status(result_doc, name)
                self._append_timeline(
                    f"✓ Resolve {result_doc.get('status') or 'done'} · "
                    f"{result_doc.get('dataset_name') or ''}"
                )
                self._phase_var.set("Idle — resolve complete")
                self._set_progress_pct(100 if result_doc.get("ok") else 0)
                if result_doc.get("ok"):
                    self._select_tab(self._tab_unseen)
                if not result_doc.get("ok"):
                    messagebox.showerror(
                        "Production Validation",
                        result_doc.get("error")
                        or result_doc.get("message")
                        or "Resolve failed",
                        parent=self,
                    )
            self.after(0, done)

        threading.Thread(target=work, name="pv-resolve", daemon=True).start()

    def _on_compute(self) -> None:
        if self._busy:
            return
        name = self._selected_model()
        if not name:
            messagebox.showwarning(
                "Production Validation", "Select a model first.", parent=self
            )
            return

        self._set_busy(True)
        self._select_tab(self._tab_progress)
        self._clear_timeline()
        self._append_timeline(f"Compute started · {name}")
        self._append_timeline(
            "Note: prefers XGBoost GPU predict when available; "
            "falls back to CPU with status on Summary"
        )
        self._append_timeline(
            "Note: permutation on all selected features × Holdout+Unseen "
            "can take several minutes — watch this timeline"
        )
        self._phase_var.set("Starting compute…")
        self._set_progress_pct(1)
        self._status_var.set(
            f"{name} — Computing Holdout vs Unseen importance "
            f"(whole unseen day(s); may take several minutes)…"
        )
        data_dir = self._data_dir()
        t0 = time.perf_counter()

        def work() -> None:
            err: str | None = None
            result_doc: dict[str, Any] | None = None
            try:
                from chain_replay_ml.production_validation import run_production_validation

                def _progress(info: dict[str, Any]) -> None:
                    payload = dict(info) if isinstance(info, dict) else {"stage": str(info)}

                    def _ui() -> None:
                        self._apply_progress(payload, model_name=name)

                    self.after(0, _ui)

                result = run_production_validation(
                    data_dir=data_dir,
                    model_name=name,
                    resolve_unseen_if_needed=True,
                    progress=_progress,
                )
                result_doc = result.as_dict()
            except Exception as exc:
                err = str(exc)

            elapsed = time.perf_counter() - t0

            def done() -> None:
                self._set_busy(False)
                if err or result_doc is None:
                    self._phase_var.set("Compute failed")
                    self._set_progress_pct(0)
                    self._append_timeline(f"✗ Compute failed ({elapsed:.1f}s): {err or 'unknown'}")
                    self._status_var.set(f"Compute failed: {err or 'unknown'}")
                    messagebox.showerror(
                        "Production Validation",
                        err or "Compute failed",
                        parent=self,
                    )
                    return
                if not result_doc.get("ok"):
                    fail_msg = str(result_doc.get("error") or "unknown")
                    self._phase_var.set("Compute failed")
                    self._set_progress_pct(0)
                    self._append_timeline(f"✗ Compute failed ({elapsed:.1f}s): {fail_msg}")
                    self._status_var.set(f"Compute failed: {fail_msg}")
                    messagebox.showerror(
                        "Production Validation",
                        fail_msg,
                        parent=self,
                    )
                    if result_doc.get("unseen_status"):
                        self._apply_status(result_doc["unseen_status"], name)
                    return
                payload = {
                    "rows": result_doc.get("rows") or [],
                    "summary": result_doc.get("summary") or {},
                    "meta": result_doc.get("meta") or {},
                    "unseen_status": result_doc.get("unseen_status") or {},
                }
                wall = (result_doc.get("meta") or {}).get("wall_time_sec")
                device = (result_doc.get("meta") or {}).get("inference_device")
                n_rows = len(payload["rows"] or [])
                device_bit = f" · {device}" if device else ""
                self._append_timeline(
                    f"✓ Compute complete · {n_rows} features · "
                    f"{float(wall):.1f}s{device_bit}"
                    if wall is not None
                    else f"✓ Compute complete · {n_rows} features · {elapsed:.1f}s{device_bit}"
                )
                self.apply_artifacts(payload, name)
                self._phase_var.set(
                    f"Done · {n_rows} features"
                    + (f" · {float(wall):.1f}s" if wall is not None else "")
                )
                self._set_progress_pct(100)
                self._select_tab(self._tab_summary)

            self.after(0, done)

        threading.Thread(target=work, name="pv-compute", daemon=True).start()
