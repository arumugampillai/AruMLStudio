"""Fold Research & Replay panel — research dashboard."""

from __future__ import annotations

import json
import tkinter as tk
import tkinter.simpledialog
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .fold_replay_widgets import draw_histogram, draw_line_chart, draw_sparkline, draw_confidence_bar, fmt_ts, place_toplevel_beside_main
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, ScrollableFrame, dual_spec_sections, fmt_num, fmt_rupee, fmt_rows, metric_cards_grid, metric_table
from .trade_replay_panels import mount_plugin_panels


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return "—"


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _event_filter_match(event: dict[str, Any], mode: str) -> bool:
    et = str(event.get("event_type") or "")
    if mode == "trades":
        return et.startswith("trade_")
    if mode == "predictions":
        return et == "prediction"
    return True


class FoldReplayPanel(ttk.Frame, LazyLoadMixin):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_experiment_planner: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_experiment_planner = on_open_experiment_planner
        self._detail: dict[str, Any] | None = None
        self._pred_id_map: dict[str, str] = {}
        self._fold_id_map: dict[str, str] = {}
        self._strat_run_map: dict[str, str] = {}
        self._force_strategy_run_id: str | None = None
        self._timeline_events: list[dict[str, Any]] = []
        self._trade_by_id: dict[str, dict[str, Any]] = {}
        self._trade_replay_win: tk.Toplevel | None = None
        self._last_trade_replay_doc: dict[str, Any] | None = None
        self._compare_trade_id: str | None = None
        self._run_summary: dict[str, Any] | None = None
        self._research_filter: dict[str, Any] | None = None
        self._planner_vars: dict[str, tk.BooleanVar] = {}
        self._timeline_filter = tk.StringVar(value="all")
        self._bookmark_search_var = tk.StringVar()
        self._status_var = tk.StringVar(value="")
        self._chart_canvases: list[tk.Canvas] = []
        self._build_ui()
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self._load_prediction_runs(lazy=True)

    def _maybe_load_research_after_show(self) -> None:
        pred_id, fold_id, _ = self._selected_ids()
        if not pred_id or not fold_id:
            return
        loaded = self._detail or {}
        if loaded.get("ok"):
            same_run = str((loaded.get("prediction_run") or {}).get("run_id") or "") == pred_id
            same_fold = str((loaded.get("fold") or {}).get("fold_id") or "") == fold_id
            if same_run and same_fold:
                return
        self._load_research()

    def prefill(self, prediction_run_id: str, fold_id: str | None = None, strategy_run_id: str | None = None) -> None:
        self._force_strategy_run_id = strategy_run_id
        self._load_prediction_runs(lazy=False)
        for label, rid in self._pred_id_map.items():
            if rid == prediction_run_id:
                self._pred_var.set(label)
                self._on_prediction_selected()
                break
        if fold_id:
            for label, fid in self._fold_id_map.items():
                if fid == fold_id:
                    self._fold_var.set(label)
                    break
        self._load_research()

    def _build_ui(self) -> None:
        form = ttk.LabelFrame(self, text="Fold Research & Replay", padding=8)
        form.pack(fill="x", padx=8, pady=8)

        r1 = ttk.Frame(form)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Prediction Run", width=14).pack(side="left")
        self._pred_var = tk.StringVar()
        self._pred_combo = ttk.Combobox(r1, textvariable=self._pred_var, width=50, state="readonly")
        self._pred_combo.pack(side="left", padx=4)
        self._pred_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_prediction_selected(load=True))

        r2 = ttk.Frame(form)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Fold", width=14).pack(side="left")
        self._fold_var = tk.StringVar()
        self._fold_combo = ttk.Combobox(r2, textvariable=self._fold_var, width=50, state="readonly")
        self._fold_combo.pack(side="left", padx=4)
        self._fold_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_research())

        r3 = ttk.Frame(form)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Strategy Run", width=14).pack(side="left")
        self._strat_var = tk.StringVar(value="(auto)")
        self._strat_combo = ttk.Combobox(r3, textvariable=self._strat_var, width=50, state="readonly")
        self._strat_combo.pack(side="left", padx=4)

        btn_row = ttk.Frame(form)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Research Report", command=self._load_research_report).pack(side="right", padx=(0, 6))
        ttk.Button(btn_row, text="Load Fold Research", command=self._load_research).pack(side="right")
        ttk.Button(btn_row, text="Download CSV", command=self._download_csv).pack(side="right", padx=(0, 6))
        ttk.Button(btn_row, text="Compare All Folds", command=self._load_fold_compare).pack(side="right", padx=(0, 6))

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self._tab_overview = ttk.Frame(self._notebook, padding=4)
        self._tab_prediction = ttk.Frame(self._notebook, padding=4)
        self._tab_trading = ttk.Frame(self._notebook, padding=4)
        self._tab_charts = ttk.Frame(self._notebook, padding=4)
        self._tab_compare = ttk.Frame(self._notebook, padding=4)
        self._tab_errors = ttk.Frame(self._notebook, padding=4)
        self._tab_drift = ttk.Frame(self._notebook, padding=4)
        self._tab_regime = ttk.Frame(self._notebook, padding=4)
        self._tab_notebook = ttk.Frame(self._notebook, padding=4)
        self._tab_bookmarks = ttk.Frame(self._notebook, padding=4)
        self._tab_timeline = ttk.Frame(self._notebook, padding=4)
        self._tab_run_summary = ttk.Frame(self._notebook, padding=4)
        self._notebook.add(self._tab_overview, text="Overview")
        self._notebook.add(self._tab_prediction, text="Prediction")
        self._notebook.add(self._tab_trading, text="Trading")
        self._notebook.add(self._tab_charts, text="Charts")
        self._notebook.add(self._tab_errors, text="Error Explorer")
        self._notebook.add(self._tab_drift, text="Feature Drift")
        self._notebook.add(self._tab_regime, text="Regime")
        self._notebook.add(self._tab_notebook, text="Notebook")
        self._notebook.add(self._tab_bookmarks, text="Bookmarks")
        self._notebook.add(self._tab_compare, text="Compare Folds")
        self._notebook.add(self._tab_timeline, text="Replay Timeline")
        self._notebook.add(self._tab_run_summary, text="Research Report")

        self._build_research_report_ui()

        self._overview_host = ttk.Frame(self._tab_overview)
        self._overview_host.pack(fill="both", expand=True)

        self._prediction_host = ttk.Frame(self._tab_prediction)
        self._prediction_host.pack(fill="both", expand=True)

        trade_cols = ("num", "entry_ts", "exit_ts", "entry", "exit", "target", "stop", "pnl", "reason", "hold")
        self._trade_filter_var = tk.StringVar(value="")
        trade_top = ttk.Frame(self._tab_trading)
        trade_top.pack(fill="x", pady=(0, 4))
        self._trade_filter_label = ttk.Label(trade_top, text="", foreground=COL_WARN, font=("Segoe UI", 9))
        self._trade_filter_label.pack(side="left", padx=(0, 8))
        ttk.Button(trade_top, text="Clear filter", command=self._clear_research_filter).pack(side="left")
        self._trades_tree = ttk.Treeview(self._tab_trading, columns=trade_cols, show="headings", height=14)
        for c, w, label in (
            ("num", 36, "#"),
            ("entry_ts", 70, "Entry Time"),
            ("exit_ts", 70, "Exit Time"),
            ("entry", 58, "Entry"),
            ("exit", 58, "Exit"),
            ("target", 58, "Target"),
            ("stop", 58, "Stop"),
            ("pnl", 68, "PnL"),
            ("reason", 72, "Exit Reason"),
            ("hold", 58, "Hold s"),
        ):
            self._trades_tree.heading(c, text=label)
            self._trades_tree.column(c, width=w)
        self._trades_tree.pack(fill="both", expand=True)
        self._trades_tree.bind("<Double-1>", self._on_trade_double_click)
        ttk.Label(
            self._tab_trading,
            text="Double-click a trade for Trade Replay.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        chart_grid = ttk.Frame(self._tab_charts)
        chart_grid.pack(fill="both", expand=True)
        chart_grid.columnconfigure(0, weight=1)
        chart_grid.columnconfigure(1, weight=1)
        chart_grid.rowconfigure(0, weight=1)
        chart_grid.rowconfigure(1, weight=1)
        self._equity_canvas = tk.Canvas(chart_grid, height=160, bg="#1a2a44", highlightthickness=0)
        self._error_canvas = tk.Canvas(chart_grid, height=160, bg="#1a2a44", highlightthickness=0)
        self._profit_canvas = tk.Canvas(chart_grid, height=160, bg="#1a2a44", highlightthickness=0)
        self._dd_canvas = tk.Canvas(chart_grid, height=160, bg="#1a2a44", highlightthickness=0)
        self._equity_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._error_canvas.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self._profit_canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._dd_canvas.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self._chart_canvases = [self._equity_canvas, self._error_canvas, self._profit_canvas, self._dd_canvas]
        for cv in self._chart_canvases:
            cv.bind("<Configure>", lambda _e: self._redraw_charts())

        cmp_cols = ("fold", "rows", "mae", "rmse", "direction", "trades", "profit", "pf", "dd", "bias")
        self._compare_tree = ttk.Treeview(self._tab_compare, columns=cmp_cols, show="headings", height=10)
        for c, w, label in (
            ("fold", 52, "Fold"),
            ("rows", 52, "Rows"),
            ("mae", 58, "MAE"),
            ("rmse", 58, "RMSE"),
            ("direction", 62, "Direction"),
            ("trades", 52, "Trades"),
            ("profit", 72, "Profit"),
            ("pf", 48, "PF"),
            ("dd", 68, "Max DD"),
            ("bias", 58, "Bias"),
        ):
            self._compare_tree.heading(c, text=label)
            self._compare_tree.column(c, width=w)
        self._compare_tree.pack(fill="both", expand=True)
        self._compare_tree.tag_configure("cmp_pos", foreground=COL_OK)
        self._compare_tree.tag_configure("cmp_neg", foreground=COL_WARN)
        ttk.Label(
            self._tab_compare,
            text="Double-click a fold row to load that fold.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))
        self._compare_tree.bind("<Double-1>", self._on_compare_fold_select)

        err_bar = ttk.Frame(self._tab_errors)
        err_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(err_bar, text="Rank by").pack(side="left")
        self._error_mode_var = tk.StringVar(value="absolute")
        ttk.Combobox(
            err_bar,
            textvariable=self._error_mode_var,
            values=["absolute", "positive", "negative"],
            width=14,
            state="readonly",
        ).pack(side="left", padx=6)
        ttk.Label(err_bar, text="Limit").pack(side="left", padx=(8, 0))
        self._error_limit_var = tk.StringVar(value="100")
        ttk.Combobox(
            err_bar,
            textvariable=self._error_limit_var,
            values=["100", "1000"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Button(err_bar, text="Refresh", command=self._render_errors).pack(side="right")
        err_cols = ("rank", "time", "token", "pred", "actual", "error", "abs")
        self._errors_tree = ttk.Treeview(self._tab_errors, columns=err_cols, show="headings", height=14)
        for c, w, label in (
            ("rank", 36, "#"),
            ("time", 70, "Time"),
            ("token", 70, "Token"),
            ("pred", 65, "Prediction"),
            ("actual", 65, "Actual"),
            ("error", 65, "Error"),
            ("abs", 65, "Abs Error"),
        ):
            self._errors_tree.heading(c, text=label)
            self._errors_tree.column(c, width=w)
        self._errors_tree.pack(fill="both", expand=True)
        self._errors_tree.bind("<Double-1>", self._on_error_double_click)
        ttk.Label(
            self._tab_errors,
            text="Double-click a row to open Prediction Inspector.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        drift_cols = ("feature", "train", "validation", "shift", "severity")
        self._drift_tree = ttk.Treeview(self._tab_drift, columns=drift_cols, show="headings", height=16)
        for c, w, label in (
            ("feature", 180, "Feature"),
            ("train", 90, "Train Mean"),
            ("validation", 90, "Validation Mean"),
            ("shift", 72, "Shift %"),
            ("severity", 72, "Severity"),
        ):
            self._drift_tree.heading(c, text=label)
            self._drift_tree.column(c, width=w)
        self._drift_tree.pack(fill="both", expand=True)
        self._drift_note = ttk.Label(self._tab_drift, text="", foreground=COL_MUTED, wraplength=900, justify="left")
        self._drift_note.pack(anchor="w", pady=4)

        reg_cols = ("regime", "rows", "mae")
        self._regime_tree = ttk.Treeview(self._tab_regime, columns=reg_cols, show="headings", height=10)
        for c, w, label in (("regime", 220, "Regime"), ("rows", 70, "Rows"), ("mae", 80, "MAE")):
            self._regime_tree.heading(c, text=label)
            self._regime_tree.column(c, width=w)
        self._regime_tree.pack(fill="both", expand=True)
        self._regime_summary = ttk.Label(self._tab_regime, text="", foreground=COL_MUTED, wraplength=900)
        self._regime_summary.pack(anchor="w", pady=4)

        nb_top = ttk.Frame(self._tab_notebook)
        nb_top.pack(fill="x", pady=(0, 4))
        ttk.Label(nb_top, text="Search").pack(side="left")
        self._note_search_var = tk.StringVar()
        ttk.Entry(nb_top, textvariable=self._note_search_var, width=30).pack(side="left", padx=4)
        ttk.Button(nb_top, text="Search Notes", command=self._search_notes).pack(side="left")
        ttk.Button(nb_top, text="Generate Observation", command=self._generate_fold_observation).pack(side="right", padx=(0, 6))
        ttk.Button(nb_top, text="Save Fold Note", command=self._save_fold_note).pack(side="right")
        self._note_title_var = tk.StringVar()
        ttk.Entry(self._tab_notebook, textvariable=self._note_title_var).pack(fill="x", pady=(0, 4))
        self._note_text = scrolledtext.ScrolledText(self._tab_notebook, height=8, font=("Segoe UI", 9))
        self._note_text.pack(fill="both", expand=True, pady=(0, 4))
        self._notes_list = ttk.Treeview(
            self._tab_notebook,
            columns=("updated", "title", "preview"),
            show="headings",
            height=6,
        )
        for c, w, label in (("updated", 130, "Updated"), ("title", 140, "Title"), ("preview", 360, "Preview")):
            self._notes_list.heading(c, text=label)
            self._notes_list.column(c, width=w)
        self._notes_list.pack(fill="x")
        self._notes_list.bind("<<TreeviewSelect>>", lambda _e: self._on_note_select())

        bm_bar = ttk.Frame(self._tab_bookmarks)
        bm_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bm_bar, text="Search").pack(side="left")
        bm_search = ttk.Entry(bm_bar, textvariable=self._bookmark_search_var, width=36)
        bm_search.pack(side="left", padx=4)
        bm_search.bind("<Return>", lambda _e: self._render_bookmarks())
        ttk.Button(bm_bar, text="Go", command=self._render_bookmarks).pack(side="left")

        bm_cols = ("time", "title", "reason", "tags", "trade")
        self._bookmarks_tree = ttk.Treeview(self._tab_bookmarks, columns=bm_cols, show="headings", height=12)
        for c, w, label in (
            ("time", 90, "Time"),
            ("title", 120, "Title"),
            ("reason", 220, "Reason"),
            ("tags", 140, "Tags"),
            ("trade", 100, "Trade"),
        ):
            self._bookmarks_tree.heading(c, text=label)
            self._bookmarks_tree.column(c, width=w)
        self._bookmarks_tree.pack(fill="both", expand=True)
        self._bookmarks_tree.bind("<Double-1>", self._on_bookmark_double_click)
        ttk.Label(
            self._tab_bookmarks,
            text="Double-click a bookmark to jump to Trade Replay at that moment.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        tl_bar = ttk.Frame(self._tab_timeline)
        tl_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(tl_bar, text="Show").pack(side="left")
        ttk.Combobox(
            tl_bar,
            textvariable=self._timeline_filter,
            values=["all", "trades", "predictions"],
            width=14,
            state="readonly",
        ).pack(side="left", padx=6)
        self._timeline_filter.trace_add("write", lambda *_a: self._render_timeline())

        cols = ("seq", "time", "type", "token", "detail")
        self.timeline_tree = ttk.Treeview(self._tab_timeline, columns=cols, show="headings", height=14)
        for c, w, label in (
            ("seq", 40, "#"),
            ("time", 70, "Time"),
            ("type", 90, "Event"),
            ("token", 70, "Token"),
            ("detail", 400, "Detail"),
        ):
            self.timeline_tree.heading(c, text=label)
            self.timeline_tree.column(c, width=w)
        self.timeline_tree.pack(fill="both", expand=True)
        self.timeline_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_timeline_select())
        self.timeline_tree.bind("<Double-1>", self._on_timeline_double_click)
        ttk.Label(
            self._tab_timeline,
            text="Double-click a prediction row to open Prediction Inspector (feature contributions coming soon).",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        self._event_detail = scrolledtext.ScrolledText(self._tab_timeline, height=5, font=("Consolas", 9))
        self._event_detail.pack(fill="x", pady=4)

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))

    def _load_prediction_runs(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_prediction_runs,
                apply=self._apply_prediction_runs,
                message="Loading prediction runs…",
                status_var=self._status_var,
            )
            return
        self._apply_prediction_runs(self._fetch_prediction_runs())

    def _fetch_prediction_runs(self) -> list[dict[str, Any]]:
        from chain_replay_ml.prediction_runs import list_all_runs

        return list_all_runs(self._data_dir(), limit=100)

    def _apply_prediction_runs(self, runs: list[dict[str, Any]]) -> None:
        labels = []
        self._pred_id_map = {}
        for r in runs:
            rid = str(r.get("run_id") or "")
            label = f"{rid[:8]}… — {r.get('model_id')}"
            labels.append(label)
            self._pred_id_map[label] = rid
        self._pred_combo.configure(values=labels)
        if labels:
            if not self._pred_var.get() or self._pred_var.get() not in self._pred_id_map:
                self._pred_combo.set(labels[0])
            self._on_prediction_selected()
        self._maybe_load_research_after_show()

    def _on_prediction_selected(self, *, load: bool = False) -> None:
        from chain_replay_ml.fold_research import list_folds_for_replay

        label = self._pred_var.get()
        rid = self._pred_id_map.get(label)
        if not rid:
            return
        try:
            doc = list_folds_for_replay(self._data_dir(), rid)
            folds = doc.get("folds") or []
            fold_labels = []
            self._fold_id_map = {}
            for f in folds:
                fid = str(f.get("fold_id") or "")
                lbl = f"Fold {f.get('fold_number')} — MAE {f.get('mae')} ({fid[:8]}…)"
                fold_labels.append(lbl)
                self._fold_id_map[lbl] = fid
            self._fold_combo.configure(values=fold_labels)
            if fold_labels:
                current = self._fold_var.get()
                if current not in self._fold_id_map:
                    self._fold_combo.set(fold_labels[0])
        except Exception as exc:
            self._status_var.set(str(exc))
            return
        if load:
            pred_id, fold_id, _ = self._selected_ids()
            if pred_id and fold_id:
                self._load_research()

    def _selected_ids(self) -> tuple[str | None, str | None, str | None]:
        pred = self._pred_id_map.get(self._pred_var.get())
        fold = self._fold_id_map.get(self._fold_var.get())
        if self._force_strategy_run_id:
            return pred, fold, self._force_strategy_run_id
        strat_label = self._strat_var.get()
        strat = None if strat_label in ("", "(auto)") else self._strat_run_map.get(strat_label)
        return pred, fold, strat

    def _load_research(self) -> None:
        from chain_replay_ml.fold_research import get_fold_replay_timeline, get_fold_research

        pred_id, fold_id, strat_id = self._selected_ids()
        if not pred_id or not fold_id:
            messagebox.showinfo("Fold Research", "Select prediction run and fold.")
            return
        try:
            self._detail = get_fold_research(
                self._data_dir(),
                prediction_run_id=pred_id,
                fold_id=fold_id,
                strategy_run_id=strat_id,
            )
        except Exception as exc:
            messagebox.showerror("Fold Research", str(exc))
            return
        if not self._detail.get("ok"):
            messagebox.showerror("Fold Research", self._detail.get("error") or "Failed")
            return

        self._sync_strategy_combo(strat_id)
        self._render_overview()
        self._render_prediction()
        self._render_trading()
        self._redraw_charts()
        self._render_errors()
        self._render_drift()
        self._render_regime()
        self._render_notebook()
        self._render_bookmarks()

        trading = self._detail.get("trading") or {}
        selected_strat = trading.get("strategy_run_id") or strat_id
        timeline_doc = get_fold_replay_timeline(
            self._data_dir(),
            prediction_run_id=pred_id,
            fold_id=fold_id,
            strategy_run_id=selected_strat,
            limit=500,
        )
        self._timeline_events = timeline_doc.get("events") or []
        self._render_timeline()
        self._status_var.set(f"Loaded fold research — {len(self._timeline_events)} replay events")

    def _sync_strategy_combo(self, strat_id: str | None) -> None:
        if not self._detail:
            return
        runs = self._detail.get("strategy_runs_available") or []
        strat_labels = ["(auto)"]
        self._strat_run_map = {}
        for r in runs:
            sid = str(r.get("strategy_run_id") or "")
            lbl = f"{sid[:8]}… ({r.get('fold_trade_count')} trades)"
            strat_labels.append(lbl)
            self._strat_run_map[lbl] = sid
        self._strat_combo.configure(values=strat_labels)
        selected_strat = (self._detail.get("trading") or {}).get("strategy_run_id") or strat_id
        if selected_strat:
            for lbl, sid in self._strat_run_map.items():
                if sid == selected_strat:
                    self._strat_var.set(lbl)
                    break
            else:
                sid = str(selected_strat)
                lbl = f"{sid[:8]}… (selected)"
                if lbl not in strat_labels:
                    strat_labels.append(lbl)
                    self._strat_run_map[lbl] = sid
                    self._strat_combo.configure(values=strat_labels)
                self._strat_var.set(lbl)
            self._force_strategy_run_id = None

    def _render_overview(self) -> None:
        for w in self._overview_host.winfo_children():
            w.destroy()
        if not self._detail:
            return

        fold = self._detail.get("fold") or {}
        pq = self._detail.get("prediction_quality") or {}
        trading = self._detail.get("trading") or {}
        tm = trading.get("metrics") or {}
        market = self._detail.get("market_summary") or {}

        mae = fold.get("mae") if fold.get("mae") is not None else pq.get("mae")
        direction = (
            fold.get("directional_accuracy_pct")
            if fold.get("directional_accuracy_pct") is not None
            else pq.get("directional_accuracy_pct")
        )
        pf = tm.get("profit_factor")
        profit = tm.get("profit")

        cards = [
            ("Prediction Rows", fmt_rows(fold.get("validation_rows") or pq.get("row_count"))),
            ("Trades", fmt_rows(trading.get("trade_count") or tm.get("trade_count") or 0)),
            ("Profit", fmt_rupee(profit)),
            ("Win Rate", _fmt_pct(tm.get("win_rate_pct"))),
            ("MAE", fmt_num(mae, digits=2)),
            ("Direction", _fmt_pct(direction)),
            ("Profit Factor", fmt_num(pf, digits=2) if pf is not None else "—"),
            ("Max DD", fmt_rupee(tm.get("max_drawdown"))),
        ]
        ttk.Label(
            self._overview_host,
            text=f"Fold {fold.get('fold_number')} Summary",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        metric_cards_grid(self._overview_host, cards, columns=4)

        ctx = ttk.LabelFrame(self._overview_host, text="Market Context", padding=8)
        ctx.pack(fill="x", pady=(8, 4))
        metric_table(
            ctx,
            [
                ("Trading Days", ", ".join(market.get("trading_days") or []) or "—"),
                ("Spot", f"{market.get('spot_start')} → {market.get('spot_end')} ({market.get('spot_trend_pct')}%)"),
                ("Volatility", f"{market.get('volatility_proxy_pct')}%"),
                ("Tokens", market.get("token_count")),
                ("Time Span", f"{market.get('timestamp_span_sec')}s"),
            ],
            label_width=14,
        )
        note = market.get("regime_note") or ""
        drift = (self._detail.get("feature_drift") or {}).get("note") or ""
        if note or drift:
            notes = ttk.LabelFrame(self._overview_host, text="Notes", padding=8)
            notes.pack(fill="x", pady=4)
            if note:
                ttk.Label(notes, text=note, wraplength=900, justify="left", foreground=COL_MUTED).pack(anchor="w", pady=2)
            if drift:
                ttk.Label(notes, text=drift, wraplength=900, justify="left", foreground=COL_MUTED).pack(anchor="w", pady=2)

        fq = self._detail.get("fold_quality") or {}
        if fq.get("total") is not None:
            fq_fr = ttk.LabelFrame(self._overview_host, text="Fold Quality", padding=8)
            fq_fr.pack(fill="x", pady=4)
            ttk.Label(
                fq_fr,
                text=f"Fold Quality  {fq.get('total')} / {fq.get('max', 100)}",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w")
            for dim in fq.get("dimensions") or []:
                ttk.Label(
                    fq_fr,
                    text=f"{dim.get('label')}  {dim.get('score')}",
                    font=("Segoe UI", 9),
                ).pack(anchor="w", padx=(8, 0))
            if fq.get("note"):
                ttk.Label(fq_fr, text=fq.get("note"), foreground=COL_MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        clusters = self._detail.get("trade_clusters") or {}
        if clusters.get("available"):
            cl_fr = ttk.LabelFrame(self._overview_host, text="Trade Clusters", padding=8)
            cl_fr.pack(fill="both", expand=True, pady=4)
            ccols = ("cluster", "trades", "avg_pnl", "win_rate")
            ctree = ttk.Treeview(cl_fr, columns=ccols, show="headings", height=5)
            for c, w, label in (("cluster", 220, "Cluster"), ("trades", 60, "Trades"), ("avg_pnl", 80, "Avg PnL"), ("win_rate", 70, "Win %")):
                ctree.heading(c, text=label)
                ctree.column(c, width=w)
            ctree.pack(fill="both", expand=True)
            for cl in clusters.get("clusters") or []:
                ctree.insert(
                    "",
                    "end",
                    values=(
                        cl.get("label"),
                        cl.get("trade_count"),
                        fmt_rupee(cl.get("average_pnl")),
                        _fmt_pct(cl.get("win_rate_pct")),
                    ),
                )

    def _render_prediction(self) -> None:
        for w in self._prediction_host.winfo_children():
            w.destroy()
        if not self._detail:
            return
        pq = self._detail.get("prediction_quality") or {}
        metric_table(
            self._prediction_host,
            [
                ("Rows", fmt_rows(pq.get("row_count"))),
                ("MAE", fmt_num(pq.get("mae"), digits=2)),
                ("RMSE", fmt_num(pq.get("rmse"), digits=2)),
                ("Median Error", fmt_num(pq.get("median_error"), digits=2)),
                ("P95 Error", fmt_num(pq.get("p95_error"), digits=2)),
                ("Bias", fmt_num(pq.get("bias"), digits=2)),
                ("Direction", _fmt_pct(pq.get("directional_accuracy_pct"))),
            ],
            label_width=16,
        )

        cal_fr = ttk.LabelFrame(self._prediction_host, text="Calibration", padding=8)
        cal_fr.pack(fill="both", expand=True, pady=(6, 0))
        cal_cols = ("bucket", "count", "prediction", "actual", "difference")
        cal_tree = ttk.Treeview(cal_fr, columns=cal_cols, show="headings", height=6)
        for c, w, label in (
            ("bucket", 56, "Bucket"),
            ("count", 52, "Rows"),
            ("prediction", 90, "Prediction %"),
            ("actual", 90, "Actual %"),
            ("difference", 90, "Difference %"),
        ):
            cal_tree.heading(c, text=label)
            cal_tree.column(c, width=w)
        cal_tree.pack(fill="both", expand=True)
        for b in pq.get("calibration_buckets") or []:
            cal_tree.insert(
                "",
                "end",
                values=(
                    b.get("bin"),
                    b.get("count"),
                    fmt_num(b.get("pred_return_avg_pct"), digits=2),
                    fmt_num(b.get("actual_return_avg_pct"), digits=2),
                    fmt_num(b.get("calibration_error_pct"), digits=2),
                ),
            )

        dbg = ttk.LabelFrame(self._prediction_host, text="Debug JSON", padding=4)
        dbg.pack(fill="x", pady=(6, 0))
        dbg_text = scrolledtext.ScrolledText(dbg, height=4, font=("Consolas", 8))
        dbg_text.pack(fill="x")
        dbg_text.insert("end", json.dumps(pq, indent=2, default=str))
        dbg_text.configure(state="disabled")

    def _render_trading(self) -> None:
        self._trades_tree.delete(*self._trades_tree.get_children())
        self._trade_by_id.clear()
        filt = self._research_filter
        if filt and filt.get("trade_ids"):
            self._trade_filter_label.configure(
                text=f"Filter: {filt.get('label') or 'Research'} — {len(filt.get('trade_ids') or [])} trades across {len(filt.get('fold_ids') or [])} folds",
            )
            trades = self._load_filtered_trades(filt)
        else:
            self._trade_filter_label.configure(text="")
            trades = self._trades_for_current_fold()
        for i, t in enumerate(trades, start=1):
            tid = str(t.get("trade_id") or f"trade_{i}")
            self._trade_by_id[tid] = t
            pnl = t.get("net_pnl")
            tag = "cmp_pos" if pnl is not None and float(pnl) > 0 else ("cmp_neg" if pnl is not None and float(pnl) < 0 else "")
            self._trades_tree.insert(
                "",
                "end",
                iid=tid,
                tags=(tag,) if tag else (),
                values=(
                    i,
                    fmt_ts(t.get("entry_ts")),
                    fmt_ts(t.get("exit_ts")),
                    t.get("entry_price"),
                    t.get("exit_price"),
                    "—",
                    "—",
                    fmt_rupee(pnl) if pnl is not None else "—",
                    t.get("exit_reason"),
                    t.get("holding_seconds"),
                ),
            )
        self._trades_tree.tag_configure("cmp_pos", foreground=COL_OK)
        self._trades_tree.tag_configure("cmp_neg", foreground=COL_WARN)

    def _trades_for_current_fold(self) -> list[dict[str, Any]]:
        if not self._detail:
            return []
        trading = self._detail.get("trading")
        if not trading:
            return []
        trades = list(trading.get("trades") or [])
        strat_id = trading.get("strategy_run_id")
        fold_id = str((self._detail.get("fold") or {}).get("fold_id") or "")
        if strat_id and fold_id:
            from chain_replay_ml.fold_research.service import _load_fold_trades

            trades = _load_fold_trades(self._data_dir(), str(strat_id), fold_id)
        return trades

    def _load_filtered_trades(self, filt: dict[str, Any]) -> list[dict[str, Any]]:
        from chain_replay_ml.fold_research.service import _load_fold_trades

        trade_ids = {str(t) for t in (filt.get("trade_ids") or [])}
        fold_ids = list(filt.get("fold_ids") or [])
        strat_id = filt.get("strategy_run_id")
        if not strat_id:
            _, _, strat_id = self._selected_ids()
        if not strat_id or not trade_ids:
            return []
        out: list[dict[str, Any]] = []
        for fold_id in fold_ids:
            for t in _load_fold_trades(self._data_dir(), str(strat_id), str(fold_id)):
                tid = str(t.get("trade_id") or "")
                if tid in trade_ids:
                    row = dict(t)
                    row["_filter_fold_id"] = fold_id
                    out.append(row)
        out.sort(key=lambda t: (_as_float(t.get("entry_ts")) or 0))
        return out

    def _clear_research_filter(self) -> None:
        self._research_filter = None
        self._render_trading()

    def _apply_root_cause_filter(self, item: dict[str, Any]) -> None:
        doc = self._run_summary or {}
        self._research_filter = {
            "label": item.get("label"),
            "trade_ids": list(item.get("trade_ids") or []),
            "fold_ids": list(item.get("fold_ids") or []),
            "strategy_run_id": doc.get("strategy_run_id"),
        }
        self._render_trading()
        self._notebook.select(self._tab_trading)
        self._status_var.set(f"Filtered trades: {item.get('label')} ({item.get('count')} losing trades)")

    def _build_research_report_ui(self) -> None:
        outer = ttk.Frame(self._tab_run_summary)
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="Generate Report", command=self._load_research_report).pack(side="left")
        ttk.Button(toolbar, text="Save Report", command=self._save_research_report).pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, text="Saved reports →", foreground=COL_MUTED).pack(side="left", padx=(16, 4))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        saved_fr = ttk.LabelFrame(body, text="Research Reports", padding=4, width=180)
        body.add(saved_fr, weight=0)
        self._saved_reports_list = tk.Listbox(saved_fr, height=18, exportselection=False, font=("Segoe UI", 9))
        saved_scroll = ttk.Scrollbar(saved_fr, orient="vertical", command=self._saved_reports_list.yview)
        self._saved_reports_list.configure(yscrollcommand=saved_scroll.set)
        self._saved_reports_list.pack(side="left", fill="both", expand=True)
        saved_scroll.pack(side="right", fill="y")
        self._saved_reports_list.bind("<<ListboxSelect>>", self._on_saved_report_select)
        self._saved_report_index: list[dict[str, Any]] = []

        report_fr = ttk.Frame(body)
        body.add(report_fr, weight=1)

        self._report_notebook = ttk.Notebook(report_fr)
        self._report_notebook.pack(fill="both", expand=True)

        self._sec_exec = ttk.Frame(self._report_notebook, padding=8)
        self._sec_root = ScrollableFrame(self._report_notebook)
        self._sec_opp = ScrollableFrame(self._report_notebook)
        self._sec_folds = ScrollableFrame(self._report_notebook)
        self._sec_recs = ScrollableFrame(self._report_notebook)
        self._sec_action = ScrollableFrame(self._report_notebook)
        self._sec_findings = ScrollableFrame(self._report_notebook)

        self._report_notebook.add(self._sec_exec, text="1. Executive Summary")
        self._report_notebook.add(self._sec_root, text="2. Root Cause")
        self._report_notebook.add(self._sec_opp, text="3. Opportunity")
        self._report_notebook.add(self._sec_folds, text="4. Fold Ranking")
        self._report_notebook.add(self._sec_recs, text="5. Recommendations")
        self._report_notebook.add(self._sec_action, text="6. Experiment Planner")
        self._report_notebook.add(self._sec_findings, text="7. Known Findings")

        self._exec_host = self._sec_exec
        self._root_host = self._sec_root.inner
        self._opp_host = self._sec_opp.inner
        self._folds_host = self._sec_folds.inner
        self._recs_host = self._sec_recs.inner
        self._action_host = self._sec_action.inner
        self._findings_host = self._sec_findings.inner

    def _load_research_report(self) -> None:
        from chain_replay_ml.fold_research import get_research_report

        pred_id, _fold_id, strat_id = self._selected_ids()
        if not pred_id:
            messagebox.showinfo("Research Report", "Select a prediction run first.")
            return
        try:
            doc = get_research_report(
                self._data_dir(),
                pred_id,
                strategy_run_id=strat_id,
            )
        except Exception as exc:
            messagebox.showerror("Research Report", str(exc))
            return
        if not doc.get("ok"):
            messagebox.showerror("Research Report", doc.get("error") or "Failed")
            return
        self._run_summary = doc
        self._render_research_report()
        self._refresh_saved_reports_list(pred_id)
        self._notebook.select(self._tab_run_summary)
        exec_sum = doc.get("executive_summary") or {}
        self._status_var.set(
            f"Research report — grade {exec_sum.get('overall_grade')} · {exec_sum.get('trade_count')} trades",
        )

    def _save_research_report(self) -> None:
        from chain_replay_ml.fold_research import save_research_report_to_store

        doc = self._run_summary
        if not doc or not doc.get("ok"):
            messagebox.showinfo("Research Report", "Generate a report first.")
            return
        try:
            saved = save_research_report_to_store(self._data_dir(), doc)
        except Exception as exc:
            messagebox.showerror("Research Report", str(exc))
            return
        self._run_summary = saved
        pred_id = str(doc.get("prediction_run_id") or "")
        self._refresh_saved_reports_list(pred_id or None)
        messagebox.showinfo("Research Report", "Report saved.")

    def _refresh_saved_reports_list(self, prediction_run_id: str | None = None) -> None:
        from chain_replay_ml.fold_research import list_saved_research_reports

        self._saved_reports_list.delete(0, "end")
        self._saved_report_index.clear()
        try:
            rows = list_saved_research_reports(
                self._data_dir(),
                prediction_run_id=prediction_run_id,
                limit=30,
            )
        except Exception:
            rows = []
        for i, row in enumerate(rows, start=1):
            run_short = str(row.get("prediction_run_id") or "")[:8]
            grade = row.get("grade") or "—"
            created = str(row.get("created_at") or "")[:10]
            self._saved_reports_list.insert("end", f"Run {run_short}…  {grade}  ({created})")
            self._saved_report_index.append(row)

    def _on_saved_report_select(self, _event: tk.Event) -> None:
        from chain_replay_ml.fold_research import load_saved_research_report

        sel = self._saved_reports_list.curselection()
        if not sel:
            return
        row = self._saved_report_index[sel[0]]
        report_id = row.get("report_id")
        if not report_id:
            return
        doc = load_saved_research_report(self._data_dir(), str(report_id))
        if not doc:
            return
        self._run_summary = doc
        self._render_research_report()
        self._report_notebook.select(self._sec_exec)

    def _render_research_report(self) -> None:
        for host in (self._exec_host, self._root_host, self._opp_host, self._folds_host, self._recs_host, self._action_host, self._findings_host):
            for w in host.winfo_children():
                w.destroy()

        doc = self._run_summary or {}
        if not doc.get("ok"):
            ttk.Label(self._exec_host, text="Generate a research report.", foreground=COL_MUTED).pack(anchor="w")
            return

        self._render_exec_summary(doc)
        self._render_root_cause_section(doc)
        self._render_opportunity_section(doc)
        self._render_fold_ranking_section(doc)
        self._render_recommendations_section(doc)
        self._render_action_plan_section(doc)
        self._render_known_findings_section(doc)

    def _render_known_findings_section(self, doc: dict[str, Any]) -> None:
        from chain_replay_ml.fold_research import get_known_findings_for_report

        kb = get_known_findings_for_report(self._data_dir(), doc)
        findings = kb.get("findings") or []
        ttk.Label(
            self._findings_host,
            text="Evidence-backed findings relevant to this run",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        if not findings:
            ttk.Label(
                self._findings_host,
                text="No findings yet — complete experiments to build the knowledge base.",
                foreground=COL_MUTED,
                wraplength=720,
            ).pack(anchor="w")
            return
        if kb.get("source") == "global":
            ttk.Label(
                self._findings_host,
                text="Showing top platform findings (no direct match for this run yet).",
                foreground=COL_MUTED,
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(0, 6))
        for f in findings[:12]:
            stars = "★" * int(f.get("stars") or 0) + "☆" * (5 - int(f.get("stars") or 0))
            card = ttk.LabelFrame(self._findings_host, text=stars, padding=8)
            card.pack(fill="x", pady=4)
            ttk.Label(card, text=str(f.get("finding") or ""), font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(
                card,
                text=(
                    f"Status {f.get('status')} · Confidence {f.get('confidence_pct')}% · "
                    f"{f.get('experiment_count')} experiments · {fmt_rows(f.get('trade_count'))} trades"
                ),
                foreground=COL_MUTED,
                font=("Segoe UI", 9),
            ).pack(anchor="w")

    def _render_exec_summary(self, doc: dict[str, Any]) -> None:
        exec_sum = doc.get("executive_summary") or {}
        pred_run = doc.get("prediction_run") or {}
        flags = exec_sum.get("recommendation_flags") or {}

        grid = ttk.Frame(self._exec_host)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        fields = [
            ("Prediction Run", str(pred_run.get("run_id") or doc.get("prediction_run_id") or "")[:16]),
            ("Model", str(exec_sum.get("model_id") or "—")),
            ("Target", str(exec_sum.get("target") or pred_run.get("target") or "—")),
            ("Strategy", str(exec_sum.get("strategy") or "—")),
            ("Trading Days", fmt_rows(exec_sum.get("trading_days"))),
            ("Folds", fmt_rows(exec_sum.get("fold_count"))),
            ("Trades", fmt_rows(exec_sum.get("trade_count"))),
        ]
        for i, (label, value) in enumerate(fields):
            r, c = divmod(i, 2)
            ttk.Label(grid, text=label, foreground=COL_MUTED, width=14).grid(row=r, column=c * 2, sticky="w", pady=3, padx=(0, 4))
            ttk.Label(grid, text=value, font=("Segoe UI", 10, "bold")).grid(row=r, column=c * 2 + 1, sticky="w", pady=3)

        grade_fr = ttk.Frame(self._exec_host)
        grade_fr.pack(fill="x", pady=(12, 4))
        ttk.Label(grade_fr, text="Overall Grade", foreground=COL_MUTED).pack(side="left")
        ttk.Label(grade_fr, text=str(exec_sum.get("overall_grade") or "—"), font=("Segoe UI", 28, "bold")).pack(side="left", padx=(12, 0))

        rec_fr = ttk.LabelFrame(self._exec_host, text="Recommendation", padding=8)
        rec_fr.pack(fill="x", pady=(8, 0))
        for key, label, ok in (
            ("worth_improving", "Worth improving", flags.get("worth_improving")),
            ("not_production_ready", "Not production ready", flags.get("not_production_ready")),
            ("ready_for_live", "Ready for live", flags.get("ready_for_live")),
        ):
            mark = "✓" if ok else "○"
            color = COL_OK if ok else COL_MUTED
            ttk.Label(rec_fr, text=f"{mark} {label}", foreground=color, font=("Segoe UI", 10)).pack(anchor="w", pady=1)

        if doc.get("note"):
            ttk.Label(self._exec_host, text=str(doc["note"]), foreground=COL_MUTED, wraplength=720).pack(anchor="w", pady=(8, 0))

    def _render_root_cause_section(self, doc: dict[str, Any]) -> None:
        rc = doc.get("root_cause_analysis") or {}
        items = rc.get("items") or []
        ttk.Label(
            self._root_host,
            text="Top Failure Reasons — click a row to filter affected folds and trades",
            foreground=COL_MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 8))
        if not items:
            ttk.Label(self._root_host, text="No losing trades to analyze.", foreground=COL_MUTED).pack(anchor="w")
            return
        for item in items:
            row = ttk.Frame(self._root_host)
            row.pack(fill="x", pady=3)
            pct = float(item.get("pct") or 0)
            label = str(item.get("label") or "")
            btn = ttk.Button(
                row,
                text=f"{pct:.1f}%  {label}  ({item.get('count')})",
                command=lambda it=item: self._apply_root_cause_filter(it),
            )
            btn.pack(side="left", fill="x", expand=True)
            bar = ttk.Progressbar(row, length=180, maximum=100, value=pct)
            bar.pack(side="right", padx=(8, 0))

    def _render_opportunity_section(self, doc: dict[str, Any]) -> None:
        opp = doc.get("opportunity_analysis") or {}
        cards = [
            ("Total", fmt_rows(opp.get("total_trades"))),
            ("Winning", fmt_rows(opp.get("winning"))),
            ("Losing", fmt_rows(opp.get("losing"))),
            ("Recoverable", fmt_rows(opp.get("recoverable"))),
            ("Unrecoverable", fmt_rows(opp.get("unrecoverable"))),
        ]
        metric_cards_grid(self._opp_host, cards, columns=3)

        ttk.Label(self._opp_host, text="Scenario uplift", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 6))
        for sc in opp.get("scenarios") or []:
            delta = sc.get("profit_delta")
            sign = "+" if delta is not None and float(delta) >= 0 else ""
            ttk.Label(
                self._opp_host,
                text=f"{sc.get('label')}  →  {sign}{fmt_rupee(delta)}",
                font=("Segoe UI", 10),
                foreground=COL_OK if delta and float(delta) > 0 else COL_MUTED,
            ).pack(anchor="w", pady=2)

        combined = opp.get("combined") or {}
        if combined:
            pf_b = combined.get("profit_factor_before")
            pf_a = combined.get("profit_factor_after")
            ttk.Label(
                self._opp_host,
                text=f"If all combined — PF {fmt_num(pf_b, digits=2) if pf_b is not None else '—'} → {fmt_num(pf_a, digits=2) if pf_a is not None else '—'}",
                font=("Segoe UI", 10, "bold"),
                foreground=COL_OK,
            ).pack(anchor="w", pady=(8, 2))

    def _render_fold_ranking_section(self, doc: dict[str, Any]) -> None:
        ranking = doc.get("fold_ranking") or []
        if not ranking:
            ttk.Label(self._folds_host, text="No fold data.", foreground=COL_MUTED).pack(anchor="w")
            return
        for fold in ranking:
            card = ttk.LabelFrame(
                self._folds_host,
                text=f"Fold {fold.get('fold_number')}  ·  Score {fold.get('score')}  ·  Grade {fold.get('grade')}",
                padding=8,
            )
            card.pack(fill="x", pady=4)
            ttk.Label(card, text=f"Reason: {fold.get('reason')}", font=("Segoe UI", 10)).pack(anchor="w")
            ttk.Label(card, text=f"{fold.get('trade_count')} trades", foreground=COL_MUTED, font=("Segoe UI", 9)).pack(anchor="w")
            fid = fold.get("fold_id")
            if fid:
                ttk.Button(
                    card,
                    text="Investigate fold",
                    command=lambda f=fid: self._jump_to_fold(f),
                ).pack(anchor="e", pady=(4, 0))

    def _jump_to_fold(self, fold_id: str) -> None:
        for label, fid in self._fold_id_map.items():
            if fid == fold_id:
                self._fold_var.set(label)
                self._load_research()
                self._notebook.select(self._tab_overview)
                break

    def _render_recommendations_section(self, doc: dict[str, Any]) -> None:
        recs = doc.get("recommendations") or []
        if not recs:
            ttk.Label(self._recs_host, text="No recommendations generated.", foreground=COL_MUTED).pack(anchor="w")
            return
        for rec in recs:
            stars = "★" * int(rec.get("stars") or 0) + "☆" * (5 - int(rec.get("stars") or 0))
            card = ttk.LabelFrame(self._recs_host, text=stars, padding=8)
            card.pack(fill="x", pady=4)
            ttk.Label(card, text=str(rec.get("text") or ""), font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(
                card,
                text=f"Observed {rec.get('observed_trades')} trades · Expected PF +{rec.get('expected_pf_delta')}",
                foreground=COL_MUTED,
                font=("Segoe UI", 9),
            ).pack(anchor="w")

    def _render_action_plan_section(self, doc: dict[str, Any]) -> None:
        from chain_replay_ml.fold_research import get_experiment_planner_view

        plan = doc.get("action_plan") or {}
        self._planner_vars.clear()
        view = get_experiment_planner_view(self._data_dir(), doc)
        items = view.get("items") or []

        ttk.Label(
            self._action_host,
            text="Experiment Planner — select recommendations, create a Proposal, then freeze into a Template",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        goal_fr = ttk.Frame(self._action_host)
        goal_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(goal_fr, text="Goal", width=8).pack(side="left")
        self._experiment_goal_var = tk.StringVar(value=str(view.get("suggested_goal") or ""))
        ttk.Entry(goal_fr, textvariable=self._experiment_goal_var, width=58).pack(side="left", fill="x", expand=True)

        rec_fr = ttk.LabelFrame(self._action_host, text="Recommendations", padding=8)
        rec_fr.pack(fill="x", pady=(0, 8))
        if not items:
            for item in plan.get("next_experiment") or []:
                ttk.Label(rec_fr, text=f"✓ {item}", foreground=COL_OK, font=("Segoe UI", 10)).pack(anchor="w", pady=1)
        else:
            for item in items:
                row = ttk.Frame(rec_fr)
                row.pack(fill="x", pady=2)
                key = str(item.get("key") or item.get("text"))
                var = tk.BooleanVar(value=bool(item.get("accepted_default", True)))
                self._planner_vars[key] = var
                ttk.Checkbutton(row, variable=var).pack(side="left")
                ttk.Label(row, text=str(item.get("text") or ""), width=38, anchor="w").pack(side="left", padx=(4, 6))
                ttk.Label(row, text=str(item.get("target_label") or ""), foreground=COL_MUTED, width=26, anchor="w").pack(side="left")

        btn_row = ttk.Frame(self._action_host)
        btn_row.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_row, text="Score Proposal", command=self._score_experiment_proposal).pack(side="left")
        ttk.Button(btn_row, text="Create Proposal", command=self._create_proposal_from_report).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Open Experiment Planner", command=self._open_experiment_planner).pack(side="left", padx=(8, 0))

        self._proposal_scores_fr = ttk.LabelFrame(self._action_host, text="Experiment Score", padding=8)
        self._proposal_scores_fr.pack(fill="x", pady=(0, 8))
        self._proposal_scores_var = tk.StringVar(value="Run Score Proposal to see improvement probability and novelty.")
        ttk.Label(self._proposal_scores_fr, textvariable=self._proposal_scores_var, wraplength=720, justify="left").pack(anchor="w")

        est = plan.get("estimated_improvement") or {}
        if est:
            imp_fr = ttk.LabelFrame(self._action_host, text="Estimated improvement (if accepted filters applied)", padding=8)
            imp_fr.pack(fill="x", pady=(6, 0))
            pf_b = est.get("profit_factor_before")
            pf_a = est.get("profit_factor_after")
            wr_b = est.get("win_rate_before_pct")
            wr_a = est.get("win_rate_after_pct")
            ttk.Label(
                imp_fr,
                text=f"Profit Factor  {fmt_num(pf_b, digits=2) if pf_b is not None else '—'} → {fmt_num(pf_a, digits=2) if pf_a is not None else '—'}",
                font=("Segoe UI", 10),
            ).pack(anchor="w", pady=2)
            ttk.Label(
                imp_fr,
                text=f"Win Rate  {_fmt_pct(wr_b)} → {_fmt_pct(wr_a)}",
                font=("Segoe UI", 10),
            ).pack(anchor="w", pady=2)

    def _planner_accepted_items(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        from chain_replay_ml.fold_research import get_experiment_planner_view

        view = get_experiment_planner_view(self._data_dir(), doc)
        accepted: list[dict[str, Any]] = []
        for item in view.get("items") or []:
            key = str(item.get("key") or item.get("text"))
            if self._planner_vars.get(key) and self._planner_vars[key].get():
                accepted.append(item)
        return accepted

    def _score_experiment_proposal(self) -> None:
        from chain_replay_ml.fold_research import compute_experiment_score

        doc = self._run_summary
        if not doc or not doc.get("ok"):
            return
        accepted = self._planner_accepted_items(doc)
        if not accepted:
            self._proposal_scores_var.set("Select at least one recommendation.")
            return
        goal = getattr(self, "_experiment_goal_var", tk.StringVar()).get().strip() or None
        score = compute_experiment_score(self._data_dir(), doc, accepted_items=accepted, goal=goal)
        self._proposal_scores_var.set(self._format_experiment_score(score))

    def _format_experiment_score(self, score: dict[str, Any]) -> str:
        stars = "★" * int(score.get("stars") or 0) + "☆" * (5 - int(score.get("stars") or 0))
        return (
            f"Overall {score.get('overall')} / 100  {stars}  ·  {score.get('recommendation') or 'Review'}\n"
            f"Novelty {score.get('novelty')} · Evidence {score.get('evidence_strength')} · "
            f"Expected Gain {score.get('expected_gain')}\n"
            f"Estimated Time {score.get('estimated_minutes')} min · GPU Cost {score.get('gpu_cost')}\n"
            f"Tags: {', '.join(score.get('tags') or []) or '—'}"
        )

    def _create_proposal_from_report(self) -> None:
        from chain_replay_ml.fold_research import (
            compute_experiment_score,
            create_experiment_proposal_from_report,
            update_experiment_proposal_selection,
        )
        from .experiment_similarity_ui import confirm_experiment_after_similarity_check

        doc = self._run_summary
        if not doc or not doc.get("ok"):
            messagebox.showinfo("Experiment Planner", "Generate a Research Report first.")
            return
        accepted = self._planner_accepted_items(doc)
        if not accepted:
            messagebox.showinfo("Experiment Planner", "Select at least one recommendation.")
            return
        goal = getattr(self, "_experiment_goal_var", tk.StringVar()).get().strip() or None
        score = compute_experiment_score(self._data_dir(), doc, accepted_items=accepted, goal=goal)
        self._proposal_scores_var.set(self._format_experiment_score(score))
        dup = score.get("duplicate_check") or {}
        if dup.get("should_warn") and not confirm_experiment_after_similarity_check(
            self, dup, action_label="Create Proposal"
        ):
            return
        keys = [str(i.get("key") or i.get("text")) for i in accepted]
        out = create_experiment_proposal_from_report(self._data_dir(), doc, goal=goal)
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        proposal = out.get("proposal") or {}
        out2 = update_experiment_proposal_selection(
            self._data_dir(),
            str(proposal.get("proposal_id") or ""),
            selected_keys=keys,
            goal=goal,
        )
        if not out2.get("ok"):
            messagebox.showerror("Experiment Planner", out2.get("error") or "Failed")
            return
        proposal = out2.get("proposal") or {}
        messagebox.showinfo("Experiment Planner", f"Created Proposal #{proposal.get('proposal_number')}")
        self._open_experiment_planner(proposal_id=str(proposal.get("proposal_id") or ""))

    def _open_experiment_planner(self, *, proposal_id: str = "") -> None:
        doc = self._run_summary if (self._run_summary or {}).get("ok") else None
        if self._on_open_experiment_planner:
            self._on_open_experiment_planner(doc, proposal_id=proposal_id or None)
        else:
            messagebox.showinfo("Experiment Planner", "Open Strategy Lab → Experiment Planner from the nav menu.")

    def _load_run_summary(self) -> None:
        self._load_research_report()

    def _render_run_summary(self) -> None:
        self._render_research_report()

    def _render_errors(self) -> None:
        self._errors_tree.delete(*self._errors_tree.get_children())
        if not self._detail:
            return
        mode = self._error_mode_var.get() or "absolute"
        try:
            limit = int(self._error_limit_var.get() or "100")
        except ValueError:
            limit = 100
        bucket = (self._detail.get("error_explorer") or {}).get(mode) or []
        for i, row in enumerate(bucket[:limit], start=1):
            self._errors_tree.insert(
                "",
                "end",
                iid=str(row.get("prediction_id") or i),
                values=(
                    i,
                    fmt_ts(row.get("timestamp")),
                    row.get("token"),
                    row.get("predicted_ltp"),
                    row.get("actual_ltp"),
                    row.get("prediction_error"),
                    row.get("abs_error"),
                ),
            )

    def _render_drift(self) -> None:
        self._drift_tree.delete(*self._drift_tree.get_children())
        drift = (self._detail or {}).get("feature_drift") or {}
        self._drift_note.configure(text=drift.get("note") or "")
        if not drift.get("available"):
            if drift.get("note"):
                self._drift_note.configure(text=drift.get("note"))
            return
        for row in drift.get("top_drifted") or []:
            sev = row.get("severity") or ""
            tag = ""
            if sev == "High":
                tag = "cmp_neg"
            elif sev == "Medium":
                tag = "cmp_warn"
            self._drift_tree.insert(
                "",
                "end",
                tags=(tag,) if tag else (),
                values=(
                    row.get("feature"),
                    fmt_num(row.get("train_mean"), digits=4),
                    fmt_num(row.get("validation_mean"), digits=4),
                    _fmt_pct(row.get("shift_pct")) if row.get("shift_pct") is not None else "—",
                    sev,
                ),
            )
        self._drift_tree.tag_configure("cmp_neg", foreground=COL_WARN)
        self._drift_tree.tag_configure("cmp_warn", foreground="#E65100")

    def _render_regime(self) -> None:
        self._regime_tree.delete(*self._regime_tree.get_children())
        reg = (self._detail or {}).get("regime_analysis") or {}
        iv_tag = " · IV-aware" if reg.get("iv_enriched") else ""
        summary = f"{reg.get('volatility_regime', '—')}{iv_tag} · vol proxy {reg.get('volatility_proxy_pct', '—')}%"
        self._regime_summary.configure(text=summary)
        if not reg.get("available"):
            self._regime_summary.configure(text=reg.get("note") or summary)
            return
        for row in reg.get("regimes") or []:
            self._regime_tree.insert(
                "",
                "end",
                values=(row.get("regime"), row.get("row_count"), fmt_num(row.get("mae"), digits=2)),
            )
        if reg.get("note"):
            self._regime_summary.configure(text=f"{summary}\n{reg.get('note')}")

    def _render_bookmarks(self) -> None:
        self._bookmarks_tree.delete(*self._bookmarks_tree.get_children())
        pred_id, fold_id, _ = self._selected_ids()
        query = self._bookmark_search_var.get().strip()
        bookmarks: list[dict[str, Any]]
        if query:
            from chain_replay_ml.fold_research import search_research_bookmarks

            bookmarks = search_research_bookmarks(
                self._data_dir(),
                query,
                prediction_run_id=str(pred_id) if pred_id else None,
            )
        elif self._detail:
            bookmarks = list(self._detail.get("bookmarks") or [])
        else:
            return
        for bm in bookmarks:
            ts = bm.get("timestamp")
            time_lbl = fmt_ts(ts) if ts is not None else "—"
            tags = bm.get("tags") or []
            tag_txt = ", ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)
            self._bookmarks_tree.insert(
                "",
                "end",
                iid=str(bm.get("bookmark_id")),
                values=(
                    time_lbl,
                    bm.get("title") or "(bookmark)",
                    (bm.get("reason") or "")[:80],
                    tag_txt[:60],
                    (bm.get("trade_id") or "")[:16],
                ),
            )

    def _render_notebook(self) -> None:
        self._notes_list.delete(*self._notes_list.get_children())
        if not self._detail:
            return
        for note in self._detail.get("notes") or []:
            body = str(note.get("body") or "")
            preview = body.replace("\n", " ")[:80]
            self._notes_list.insert(
                "",
                "end",
                iid=str(note.get("note_id")),
                values=(note.get("updated_at", "")[:19], note.get("title") or "(untitled)", preview),
            )

    def _generate_fold_observation(self) -> None:
        from chain_replay_ml.fold_research import generate_trade_observation

        if not self._last_trade_replay_doc:
            messagebox.showinfo(
                "Research Notebook",
                "Open a Trade Replay first (double-click a trade), then generate an observation.",
            )
            return
        obs = generate_trade_observation(self._last_trade_replay_doc)
        self._note_title_var.set(str(obs.get("title") or ""))
        self._note_text.delete("1.0", "end")
        self._note_text.insert("1.0", str(obs.get("body") or ""))
        self._notebook.select(self._tab_notebook)

    def _save_fold_note(self) -> None:
        from chain_replay_ml.fold_research import save_fold_note

        pred_id, fold_id, _ = self._selected_ids()
        if not pred_id or not fold_id:
            messagebox.showinfo("Notebook", "Select prediction run and fold.")
            return
        body = self._note_text.get("1.0", "end").strip()
        if not body:
            messagebox.showinfo("Notebook", "Write a note first.")
            return
        try:
            save_fold_note(
                self._data_dir(),
                prediction_run_id=pred_id,
                fold_id=fold_id,
                title=self._note_title_var.get().strip() or None,
                body=body,
                model_id=(self._detail or {}).get("prediction_run", {}).get("model_id"),
            )
            self._load_research()
            self._notebook.select(self._tab_notebook)
        except Exception as exc:
            messagebox.showerror("Notebook", str(exc))

    def _search_notes(self) -> None:
        from chain_replay_ml.fold_research import search_research_notes

        q = self._note_search_var.get().strip()
        if not q:
            self._render_notebook()
            return
        try:
            notes = search_research_notes(self._data_dir(), q)
        except Exception as exc:
            messagebox.showerror("Notebook", str(exc))
            return
        self._notes_list.delete(*self._notes_list.get_children())
        for note in notes:
            body = str(note.get("body") or "")
            preview = body.replace("\n", " ")[:80]
            self._notes_list.insert(
                "",
                "end",
                iid=str(note.get("note_id")),
                values=(note.get("updated_at", "")[:19], note.get("title") or "(untitled)", preview),
            )

    def _on_note_select(self) -> None:
        sel = self._notes_list.selection()
        if not sel or not self._detail:
            return
        nid = sel[0]
        for note in self._detail.get("notes") or []:
            if str(note.get("note_id")) == nid:
                self._note_title_var.set(str(note.get("title") or ""))
                self._note_text.delete("1.0", "end")
                self._note_text.insert("end", note.get("body") or "")
                break

    def _on_error_double_click(self, _event: tk.Event) -> None:
        sel = self._errors_tree.selection()
        if not sel:
            item = self._errors_tree.identify_row(_event.y)
            if item:
                sel = (item,)
        if not sel:
            return
        prediction_id = sel[0]
        self._open_prediction_inspector_by_id(prediction_id)

    def _open_prediction_inspector_by_id(self, prediction_id: str) -> None:
        from chain_replay_ml.fold_research import get_prediction_inspector

        pred_id, fold_id, _ = self._selected_ids()
        if not pred_id or not fold_id:
            return
        try:
            doc = get_prediction_inspector(
                self._data_dir(),
                prediction_run_id=pred_id,
                fold_id=fold_id,
                prediction_id=prediction_id,
            )
        except Exception as exc:
            messagebox.showerror("Prediction Inspector", str(exc))
            return
        if not doc.get("ok"):
            messagebox.showerror("Prediction Inspector", doc.get("error") or "Failed")
            return
        self._show_prediction_inspector(doc)

    def _redraw_charts(self) -> None:
        if not self._detail:
            return
        series = self._detail.get("chart_series") or {}
        equity = [p.get("value", 0) for p in series.get("equity_curve") or []]
        drawdown = [p.get("value", 0) for p in series.get("drawdown_curve") or []]
        errors = series.get("prediction_errors") or []
        pnls = series.get("trade_pnls") or []
        draw_line_chart(self._equity_canvas, equity, title="Equity Curve", color="#58a6ff")
        draw_histogram(self._error_canvas, errors, title="Prediction Error", color="#e57373")
        draw_histogram(self._profit_canvas, pnls, title="Profit Distribution", color="#4caf50")
        draw_line_chart(self._dd_canvas, drawdown, title="Drawdown", color="#ffb74d")

    def _download_csv(self) -> None:
        from chain_replay_ml.fold_research import build_fold_research_csv

        if not self._detail or not self._detail.get("ok"):
            messagebox.showinfo("Download CSV", "Load fold research first.")
            return
        fold = self._detail.get("fold") or {}
        fold_num = fold.get("fold_number", "fold")
        default_name = f"fold_research_fold_{fold_num}.csv"
        path = filedialog.asksaveasfilename(
            title="Save Fold Research CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            csv_text = build_fold_research_csv(self._data_dir(), self._detail)
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write(csv_text)
            self._status_var.set(f"Exported CSV → {path}")
            messagebox.showinfo("Download CSV", f"Saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Download CSV", str(exc))

    def _load_fold_compare(self) -> None:
        from chain_replay_ml.fold_research import compare_folds_for_run

        pred_id, _fold_id, strat_id = self._selected_ids()
        if not pred_id:
            messagebox.showinfo("Compare Folds", "Select a prediction run first.")
            return
        try:
            doc = compare_folds_for_run(self._data_dir(), pred_id, strategy_run_id=strat_id)
        except Exception as exc:
            messagebox.showerror("Compare Folds", str(exc))
            return
        if not doc.get("ok"):
            messagebox.showerror("Compare Folds", doc.get("error") or "Failed")
            return
        self._compare_tree.delete(*self._compare_tree.get_children())
        for row in doc.get("folds") or []:
            profit = row.get("profit")
            try:
                tag = "cmp_pos" if profit is not None and float(profit) > 0 else ("cmp_neg" if profit is not None and float(profit) < 0 else "")
            except (TypeError, ValueError):
                tag = ""
            self._compare_tree.insert(
                "",
                "end",
                iid=str(row.get("fold_id") or ""),
                tags=(tag,) if tag else (),
                values=(
                    f"Fold {row.get('fold_number')}",
                    fmt_rows(row.get("validation_rows")),
                    fmt_num(row.get("mae"), digits=2),
                    fmt_num(row.get("rmse"), digits=2),
                    _fmt_pct(row.get("directional_accuracy_pct")),
                    fmt_rows(row.get("trade_count")),
                    fmt_rupee(profit) if profit is not None else "—",
                    fmt_num(row.get("profit_factor"), digits=2) if row.get("profit_factor") is not None else "—",
                    fmt_rupee(row.get("max_drawdown")) if row.get("max_drawdown") is not None else "—",
                    fmt_num(row.get("bias"), digits=2) if row.get("bias") is not None else "—",
                ),
            )
        self._notebook.select(self._tab_compare)
        self._status_var.set(f"Compared {len(doc.get('folds') or [])} folds")

    def _on_compare_fold_select(self, _event: tk.Event) -> None:
        sel = self._compare_tree.selection()
        if not sel:
            return
        fold_id = sel[0]
        for label, fid in self._fold_id_map.items():
            if fid == fold_id:
                self._fold_var.set(label)
                self._load_research()
                self._notebook.select(self._tab_overview)
                break

    def _render_timeline(self) -> None:
        mode = self._timeline_filter.get()
        self.timeline_tree.delete(*self.timeline_tree.get_children())
        for ev in self._timeline_events:
            if not _event_filter_match(ev, mode):
                continue
            seq = ev.get("sequence")
            self.timeline_tree.insert(
                "",
                "end",
                iid=str(seq),
                values=(
                    seq,
                    fmt_ts(ev.get("timestamp")),
                    ev.get("display_type") or ev.get("event_type"),
                    ev.get("token"),
                    ev.get("label"),
                ),
            )

    def _on_timeline_select(self) -> None:
        sel = self.timeline_tree.selection()
        if not sel:
            return
        seq = int(sel[0])
        ev = next((e for e in self._timeline_events if e.get("sequence") == seq), None)
        self._event_detail.delete("1.0", "end")
        if ev:
            self._event_detail.insert("end", json.dumps(ev, indent=2, default=str))

    def _on_timeline_double_click(self, _event: tk.Event) -> None:
        sel = self.timeline_tree.selection()
        if not sel:
            item = self.timeline_tree.identify_row(_event.y)
            if item:
                sel = (item,)
        if not sel:
            return
        seq = int(sel[0])
        ev = next((e for e in self._timeline_events if e.get("sequence") == seq), None)
        if not ev or ev.get("event_type") != "prediction":
            return
        pid = ev.get("prediction_id")
        if pid:
            self._open_prediction_inspector_by_id(str(pid))
        else:
            self._show_prediction_inspector({
                "ok": True,
                "timestamp": ev.get("timestamp"),
                "trading_day": ev.get("trading_day"),
                "token": ev.get("token"),
                "spot": ev.get("spot"),
                "ltp": ev.get("ltp"),
                "predicted_ltp": ev.get("predicted_ltp"),
                "actual_ltp": ev.get("actual_ltp"),
                "prediction_error": ev.get("prediction_error"),
                "direction_correct": ev.get("direction_correct"),
                "confidence": ev.get("confidence"),
                "contributions": [],
            })

    def _show_prediction_inspector(self, doc: dict[str, Any]) -> None:
        win = tk.Toplevel(self)
        win.title("Prediction Inspector")
        win.geometry("640x520")
        win.transient(self.winfo_toplevel())

        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        dir_ok = doc.get("direction_correct")
        dir_label = "Yes" if dir_ok == 1 else ("No" if dir_ok == 0 else "—")
        metric_table(
            body,
            [
                ("Timestamp", fmt_ts(doc.get("timestamp"))),
                ("Trading Day", doc.get("trading_day")),
                ("Token", doc.get("token")),
                ("Spot", doc.get("spot")),
                ("Option LTP", doc.get("ltp")),
                ("Prediction", doc.get("predicted_ltp")),
                ("Actual", doc.get("actual_ltp")),
                ("Error", doc.get("prediction_error")),
                ("Direction", dir_label),
                ("Confidence", doc.get("confidence") if doc.get("confidence") is not None else "—"),
            ],
            label_width=14,
        )

        feat_fr = ttk.LabelFrame(body, text="Top 20 Feature Contributions", padding=8)
        feat_fr.pack(fill="both", expand=True, pady=(8, 0))
        feat_cols = ("feature", "value", "impact")
        feat_tree = ttk.Treeview(feat_fr, columns=feat_cols, show="headings", height=10)
        for c, w, label in zip(feat_cols, (180, 90, 90), ("Feature", "Value", "Impact")):
            feat_tree.heading(c, text=label)
            feat_tree.column(c, width=w)
        feat_tree.pack(fill="both", expand=True)
        contributions = doc.get("contributions") or []
        if contributions:
            for row in contributions:
                impact = row.get("impact")
                tag = "cmp_pos" if impact is not None and float(impact) > 0 else ("cmp_neg" if impact is not None and float(impact) < 0 else "")
                feat_tree.insert(
                    "",
                    "end",
                    tags=(tag,) if tag else (),
                    values=(row.get("feature"), row.get("value"), impact),
                )
            feat_tree.tag_configure("cmp_pos", foreground=COL_OK)
            feat_tree.tag_configure("cmp_neg", foreground=COL_WARN)
        else:
            rehyd = doc.get("rehydration") or {}
            msg = rehyd.get("error") or "SHAP contributions unavailable (install shap or check model package)."
            ttk.Label(feat_fr, text=msg, foreground=COL_MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=4)

        final = doc.get("final_prediction") or doc.get("predicted_ltp")
        ttk.Label(
            body,
            text=f"Final Prediction: {fmt_rupee(final) if final is not None else '—'}",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(8, 0))

    def _open_prediction_inspector(self, ev: dict[str, Any]) -> None:
        pid = ev.get("prediction_id")
        if pid:
            self._open_prediction_inspector_by_id(str(pid))
            return
        self._show_prediction_inspector({
            "ok": True,
            "timestamp": ev.get("timestamp"),
            "trading_day": ev.get("trading_day"),
            "token": ev.get("token"),
            "spot": ev.get("spot"),
            "ltp": ev.get("ltp"),
            "predicted_ltp": ev.get("predicted_ltp"),
            "actual_ltp": ev.get("actual_ltp"),
            "prediction_error": ev.get("prediction_error"),
            "direction_correct": ev.get("direction_correct"),
            "confidence": ev.get("confidence"),
            "contributions": [],
        })

    def _on_trade_double_click(self, _event: tk.Event) -> None:
        sel = self._trades_tree.selection()
        if not sel:
            item = self._trades_tree.identify_row(_event.y)
            if item:
                sel = (item,)
        if not sel:
            return
        self._open_trade_replay(str(sel[0]))

    def _on_bookmark_double_click(self, _event: tk.Event) -> None:
        sel = self._bookmarks_tree.selection()
        if not sel:
            return
        bm = next((b for b in (self._detail or {}).get("bookmarks") or [] if str(b.get("bookmark_id")) == sel[0]), None)
        if not bm:
            return
        trade_id = bm.get("trade_id")
        if trade_id:
            self._open_trade_replay(str(trade_id), focus_sequence=bm.get("sequence"))

    def _open_trade_replay(self, trade_id: str, *, focus_sequence: int | None = None) -> None:
        from chain_replay_ml.fold_research import get_trade_replay

        pred_id, fold_id, _ = self._selected_ids()
        trade = self._trade_by_id.get(trade_id) or {}
        filter_fold = trade.get("_filter_fold_id")
        if filter_fold:
            fold_id = str(filter_fold)
            for label, fid in self._fold_id_map.items():
                if fid == fold_id:
                    self._fold_var.set(label)
                    break
        if not pred_id or not fold_id:
            messagebox.showinfo("Trade Replay", "Select a prediction run and fold, then load fold research.")
            return
        try:
            doc = get_trade_replay(
                self._data_dir(),
                prediction_run_id=pred_id,
                fold_id=fold_id,
                trade_id=trade_id,
            )
        except Exception as exc:
            messagebox.showerror("Trade Replay", str(exc))
            return
        if not doc.get("ok"):
            messagebox.showerror("Trade Replay", doc.get("error") or "Failed")
            return
        self._last_trade_replay_doc = doc
        self._show_trade_replay_window(doc, focus_sequence=focus_sequence)

    def _show_trade_replay_window(self, doc: dict[str, Any], *, focus_sequence: int | None = None) -> None:
        if self._trade_replay_win is not None:
            try:
                if self._trade_replay_win.winfo_exists():
                    self._trade_replay_win.destroy()
            except tk.TclError:
                pass
            self._trade_replay_win = None

        win = tk.Toplevel(self)
        self._trade_replay_win = win
        win.title("Trade Replay — ML Research Studio")

        def _on_close() -> None:
            try:
                _stop_replay()
            except Exception:
                pass
            self._trade_replay_win = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        body = ttk.Frame(win, padding=8)
        body.pack(fill="both", expand=True)

        trade = doc.get("trade") or {}
        decision = doc.get("decision") or {}
        pred = decision.get("prediction") or {}
        exit_a = doc.get("exit_analysis") or {}
        verdict = doc.get("trade_verdict") or {}
        max_opp = doc.get("maximum_opportunity") or {}
        since = doc.get("since_entry") or {}
        rule_tl = doc.get("rule_timeline") or []
        feature_alerts = doc.get("feature_alerts") or []
        similar = doc.get("similar_trades") or []
        pnl_path = doc.get("pnl_path") or []
        mini_shap = doc.get("mini_shap") or []
        regime_badges = doc.get("regime_badges") or []
        decision_quality = decision.get("decision_quality") or {}
        shap_by_feature = {s.get("feature"): s for s in mini_shap}
        trade_class = doc.get("trade_classification") or {}
        pred_failure = doc.get("prediction_failure") or {}
        research_conclusion = doc.get("research_conclusion") or {}

        def _signed_pct(v: Any) -> str:
            if v is None:
                return "—"
            try:
                n = float(v)
                return f"{n:+.2f}%"
            except (TypeError, ValueError):
                return "—"

        def _signed_rupee(v: Any) -> str:
            if v is None:
                return "—"
            try:
                n = float(v)
                sign = "+" if n >= 0 else "−"
                return f"{sign}₹{abs(n):,.2f}"
            except (TypeError, ValueError):
                return "—"

        def _yes_no(v: Any) -> str:
            if v is None:
                return "—"
            return "YES" if v in (1, True, "1") else ("NO" if v in (0, False, "0") else str(v))

        ttk.Label(
            body,
            text=f"Trade Replay — {trade.get('token')} · PnL {fmt_rupee(trade.get('net_pnl'))}",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        header_row = ttk.Frame(body)
        header_row.pack(fill="x", pady=(0, 4))
        badge_host = ttk.Frame(header_row)
        badge_host.pack(side="right")
        class_lbl = str(trade_class.get("primary") or "")
        if class_lbl:
            tk.Label(
                badge_host,
                text=class_lbl,
                bg="#fff3e0",
                fg="#e65100",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=2,
            ).pack(side="right", padx=2)
        for badge in regime_badges[:5]:
            tk.Label(
                badge_host,
                text=str(badge),
                bg="#e3f2fd",
                fg="#1565c0",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=2,
            ).pack(side="left", padx=2)

        footer = ttk.Frame(body)
        footer.pack(side="bottom", fill="x", pady=(6, 0))
        sim_actions_host = ttk.Frame(footer)
        sim_actions_host.pack(side="left", fill="x", expand=True)
        action_row = ttk.Frame(footer)
        action_row.pack(side="right")

        main = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right_scroll = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right_scroll, weight=2)

        left_split = ttk.Panedwindow(left, orient=tk.VERTICAL)
        left_split.pack(fill="both", expand=True)
        tl_host = ttk.Frame(left_split)
        mid_row_host = ttk.Frame(left_split)
        feat_host = ttk.Frame(left_split)
        similar_host = ttk.Frame(left_split)
        left_split.add(tl_host, weight=3)
        left_split.add(mid_row_host, weight=2)
        left_split.add(feat_host, weight=1)
        left_split.add(similar_host, weight=1)

        mid_row = ttk.Panedwindow(mid_row_host, orient=tk.HORIZONTAL)
        mid_row.pack(fill="both", expand=True)
        rule_host = ttk.LabelFrame(mid_row, text="Strategy Rule Timeline", padding=4)
        charts_fr = ttk.LabelFrame(mid_row, text="Price Path", padding=4)
        mid_row.add(rule_host, weight=3)
        mid_row.add(charts_fr, weight=7)

        paths = doc.get("price_paths") or {}
        chart_row = ttk.Frame(charts_fr)
        chart_row.pack(fill="both", expand=True)
        spot_canvas = tk.Canvas(chart_row, height=72, bg="#f8f9fa", highlightthickness=0)
        prem_canvas = tk.Canvas(chart_row, height=72, bg="#f8f9fa", highlightthickness=0)
        pnl_canvas = tk.Canvas(chart_row, height=72, bg="#1a2a44", highlightthickness=0)
        spot_canvas.pack(fill="x", pady=2)
        prem_canvas.pack(fill="x", pady=2)
        pnl_canvas.pack(fill="x", pady=2)
        spot_pts = [_as_float(p.get("value")) for p in paths.get("spot") or []]
        spot_pts = [p for p in spot_pts if p is not None]
        prem_pts = [_as_float(p.get("value")) for p in paths.get("premium") or []]
        prem_pts = [p for p in prem_pts if p is not None]
        pnl_pts = [_as_float(p.get("pnl")) for p in pnl_path]
        pnl_pts = [p for p in pnl_pts if p is not None]

        def _redraw_charts(_event: tk.Event | None = None) -> None:
            draw_sparkline(spot_canvas, spot_pts, title="Spot", color="#5c6bc0")
            draw_sparkline(prem_canvas, prem_pts, title="Premium", color="#2e7d32")
            draw_line_chart(pnl_canvas, pnl_pts, title="PnL During Trade", color="#66bb6a", fill="#1a2a44")

        spot_canvas.bind("<Configure>", _redraw_charts)
        prem_canvas.bind("<Configure>", _redraw_charts)
        pnl_canvas.bind("<Configure>", _redraw_charts)
        win.after_idle(_redraw_charts)

        play_bar = ttk.Frame(tl_host)
        play_bar.pack(fill="x", pady=(0, 4))
        play_state: dict[str, Any] = {"playing": False, "idx": 0, "after_id": None, "speed": 1.0}
        speed_var = tk.StringVar(value="1×")

        def _stop_replay() -> None:
            play_state["playing"] = False
            aid = play_state.get("after_id")
            if aid is not None:
                try:
                    win.after_cancel(aid)
                except tk.TclError:
                    pass
                play_state["after_id"] = None

        def _redraw_live_pnl(upto: int) -> None:
            pts = [_as_float(p.get("pnl")) for p in pnl_path]
            pts = [p for p in pts if p is not None]
            if not pts:
                return
            try:
                live = pts[: max(1, upto)]
                draw_line_chart(pnl_canvas, live, title="PnL During Trade", color="#66bb6a", fill="#1a2a44")
            except tk.TclError:
                return

        def _replay_step() -> None:
            events = self._trade_replay_events
            if not play_state["playing"] or play_state["idx"] >= len(events):
                _stop_replay()
                return
            ev = events[play_state["idx"]]
            seq = ev.get("sequence")
            if seq is not None:
                iid = str(seq)
                if tl.exists(iid):
                    tl.selection_set(iid)
                    tl.see(iid)
            _redraw_live_pnl(play_state["idx"] + 1)
            play_state["idx"] += 1
            speed = float(play_state.get("speed") or 1.0)
            delay = max(80, int(450 / speed))
            play_state["after_id"] = win.after(delay, _replay_step)

        def _toggle_play() -> None:
            if play_state["playing"]:
                _stop_replay()
                play_btn.configure(text="▶ Play")
                return
            play_state["playing"] = True
            play_state["idx"] = 0
            play_btn.configure(text="⏸ Pause")
            _replay_step()

        def _on_speed(_event: tk.Event | None = None) -> None:
            raw = speed_var.get().replace("×", "").strip()
            try:
                play_state["speed"] = float(raw)
            except ValueError:
                play_state["speed"] = 1.0

        play_btn = ttk.Button(play_bar, text="▶ Play", command=_toggle_play, width=10)
        play_btn.pack(side="left")
        ttk.Label(play_bar, text="Speed").pack(side="left", padx=(8, 2))
        speed_combo = ttk.Combobox(
            play_bar, textvariable=speed_var, values=["1×", "2×", "5×", "10×"], width=5, state="readonly",
        )
        speed_combo.pack(side="left")
        speed_combo.bind("<<ComboboxSelected>>", _on_speed)

        tl_cols = ("time", "event", "detail", "spot", "ltp", "pred", "actual")
        tl = ttk.Treeview(tl_host, columns=tl_cols, show="headings", height=12)
        for c, w, label in (
            ("time", 72, "Time"),
            ("event", 96, "Event"),
            ("detail", 200, "Detail"),
            ("spot", 58, "Spot"),
            ("ltp", 52, "LTP"),
            ("pred", 52, "Pred"),
            ("actual", 52, "Actual"),
        ):
            tl.heading(c, text=label)
            tl.column(c, width=w)
        tl.pack(fill="both", expand=True)
        self._trade_replay_events = doc.get("events") or []
        for ev in self._trade_replay_events:
            seq = ev.get("sequence")
            tl.insert(
                "",
                "end",
                iid=str(seq),
                values=(
                    ev.get("time_label") or fmt_ts(ev.get("timestamp")),
                    ev.get("display_type") or ev.get("event_type"),
                    ev.get("label"),
                    ev.get("spot"),
                    ev.get("ltp") or ev.get("price"),
                    ev.get("predicted_ltp"),
                    ev.get("actual_ltp"),
                ),
            )
        if focus_sequence is not None:
            iid = str(focus_sequence)
            if tl.exists(iid):
                tl.selection_set(iid)
                tl.see(iid)

        rule_scroll = ttk.Frame(rule_host)
        rule_scroll.pack(fill="both", expand=True)
        rule_canvas = tk.Canvas(rule_scroll, highlightthickness=0, height=120)
        rule_sb = ttk.Scrollbar(rule_scroll, orient="vertical", command=rule_canvas.yview)
        rule_inner = ttk.Frame(rule_canvas)
        rule_inner.bind("<Configure>", lambda _e: rule_canvas.configure(scrollregion=rule_canvas.bbox("all")))
        rule_canvas.create_window((0, 0), window=rule_inner, anchor="nw")
        rule_canvas.configure(yscrollcommand=rule_sb.set)
        rule_canvas.pack(side="left", fill="both", expand=True)
        rule_sb.pack(side="right", fill="y")
        if not rule_tl:
            ttk.Label(rule_inner, text="No rule timeline available.", foreground=COL_MUTED).pack(anchor="w")
        for i, step in enumerate(rule_tl):
            ttk.Label(rule_inner, text=str(step.get("label") or ""), font=("Segoe UI", 9, "bold")).pack(anchor="w")
            detail = step.get("detail")
            if detail:
                ttk.Label(rule_inner, text=str(detail), foreground=COL_MUTED, font=("Segoe UI", 8), wraplength=220).pack(anchor="w", padx=(10, 0))
            if i < len(rule_tl) - 1:
                ttk.Label(rule_inner, text="↓", foreground=COL_MUTED).pack(anchor="w", pady=(0, 2))

        right_canvas = tk.Canvas(right_scroll, highlightthickness=0)
        right_sb = ttk.Scrollbar(right_scroll, orient="vertical", command=right_canvas.yview)
        right_inner = ttk.Frame(right_canvas)
        right_inner.bind("<Configure>", lambda _e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.create_window((0, 0), window=right_inner, anchor="nw")
        right_canvas.configure(yscrollcommand=right_sb.set)
        right_canvas.pack(side="left", fill="both", expand=True)
        right_sb.pack(side="right", fill="y")

        outcome = str(verdict.get("outcome") or "TRADE")
        verdict_bg = "#e8f5e9" if "SUCCEEDED" in outcome else ("#fff8e1" if "BREAKEVEN" in outcome else "#ffebee")
        verdict_fg = "#1b5e20" if "SUCCEEDED" in outcome else ("#e65100" if "BREAKEVEN" in outcome else "#b71c1c")
        verdict_fr = tk.Frame(right_inner, bg=verdict_bg, padx=8, pady=8)
        verdict_fr.pack(fill="x", pady=(0, 6))
        tk.Label(verdict_fr, text=outcome, bg=verdict_bg, fg=verdict_fg, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        if class_lbl:
            tk.Label(
                verdict_fr,
                text=f"Classification: {class_lbl}",
                bg=verdict_bg,
                fg="#5d4037",
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", pady=(2, 0))

        model_v = verdict.get("model") or {}
        strategy_v = verdict.get("strategy") or {}
        for section_title, section in (("MODEL", model_v), ("STRATEGY", strategy_v)):
            sec_verdict = str(section.get("verdict") or "—")
            sym = "✓" if sec_verdict.lower() in ("correct", "succeeded", "neutral") else "✗"
            if section_title == "MODEL" and sec_verdict.lower() == "wrong":
                sym = "✗"
            elif section_title == "STRATEGY" and sec_verdict.lower() == "failed":
                sym = "✗"
            elif section_title == "STRATEGY" and sec_verdict.lower() == "succeeded":
                sym = "✓"
            tk.Label(
                verdict_fr,
                text=f"{section_title}  {sym} {sec_verdict}",
                bg=verdict_bg,
                fg="#333",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(6, 0))
            tk.Frame(verdict_fr, bg="#ccc", height=1).pack(fill="x", pady=4)
            for reason in section.get("reasons") or []:
                passed = bool(reason.get("passed"))
                rsym = "✓" if passed else "✗"
                color = "#2e7d32" if passed else "#c62828"
                line = f"{rsym} {reason.get('label')}"
                detail = reason.get("detail")
                if detail:
                    line = f"{line} — {detail}"
                tk.Label(
                    verdict_fr, text=line, bg=verdict_bg, fg=color,
                    font=("Segoe UI", 9), wraplength=340, justify="left",
                ).pack(anchor="w")

        if pred_failure.get("failed") or pred_failure.get("difference_pct") is not None:
            pf_fr = tk.Frame(verdict_fr, bg=verdict_bg)
            pf_fr.pack(fill="x", pady=(8, 0))
            tk.Label(pf_fr, text="Why Prediction Failed", bg=verdict_bg, fg="#333", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            exp = pred_failure.get("expected_pct")
            act = pred_failure.get("actual_pct")
            diff = pred_failure.get("difference_pct")
            tk.Label(
                pf_fr,
                text=f"Expected {_signed_pct(exp)}  ·  Actual {_signed_pct(act)}  ·  Difference {_signed_pct(diff)}",
                bg=verdict_bg,
                fg="#333",
                font=("Segoe UI", 9),
                wraplength=340,
            ).pack(anchor="w", pady=(2, 0))
            if pred_failure.get("contributors"):
                tk.Label(pf_fr, text="Largest contributors", bg=verdict_bg, fg="#666", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 0))
                for c in pred_failure.get("contributors") or []:
                    tk.Label(pf_fr, text=f"• {c}", bg=verdict_bg, fg="#c62828", font=("Segoe UI", 9)).pack(anchor="w")

        pred_fr = ttk.LabelFrame(right_inner, text="5-min Prediction", padding=6)
        pred_fr.pack(fill="x", pady=(0, 6))
        prob_success = pred.get("probability_success_pct")
        prob_failure = pred.get("probability_failure_pct")
        dual_spec_sections(
            pred_fr,
            [
                ("Prediction", _signed_pct(pred.get("prediction_pct"))),
                ("Probability Success", f"{prob_success:.0f}%" if prob_success is not None else "—"),
                ("Probability Failure", f"{prob_failure:.0f}%" if prob_failure is not None else "—"),
                ("Current LTP", fmt_rupee(pred.get("current_ltp"))),
                ("Predicted 5m LTP", fmt_rupee(pred.get("predicted_ltp"))),
            ],
            [
                ("Actual 5m LTP", fmt_rupee(pred.get("actual_ltp"))),
                ("Expected Profit", _signed_rupee(pred.get("expected_profit"))),
                ("Actual Profit", _signed_rupee(pred.get("actual_profit"))),
                ("Model Error", _signed_rupee(pred.get("model_error"))),
                ("Prediction Error", _signed_pct(pred.get("prediction_error_pct"))),
            ],
            left_title="Forecast",
            right_title="Outcome",
            label_width=12,
        )
        if pred.get("probability_note"):
            ttk.Label(pred_fr, text=pred.get("probability_note"), foreground=COL_MUTED, font=("Segoe UI", 8), wraplength=320).pack(anchor="w", pady=(4, 0))

        conf_bar_canvas = tk.Canvas(pred_fr, height=40, bg="#f8f9fa", highlightthickness=0)
        conf_bar_canvas.pack(fill="x", pady=(6, 0))
        conf_val = pred.get("confidence_pct") or pred.get("probability_success_pct")

        def _draw_conf_bar(_e: tk.Event | None = None) -> None:
            draw_confidence_bar(conf_bar_canvas, _as_float(conf_val))

        conf_bar_canvas.bind("<Configure>", _draw_conf_bar)
        win.after_idle(_draw_conf_bar)

        conf_fr = ttk.LabelFrame(right_inner, text="Confidence", padding=6)
        conf_fr.pack(fill="x", pady=(0, 6))
        conf_pct = pred.get("confidence_pct")
        metric_table(
            conf_fr,
            [
                ("Confidence", f"{conf_pct:.0f}%" if conf_pct is not None else "—"),
                ("Prediction Rank", "—"),
                ("Model Agreement", "—"),
                ("Prediction Std", "—"),
            ],
            label_width=16,
        )
        if pred.get("confidence_note"):
            ttk.Label(conf_fr, text=pred.get("confidence_note"), foreground=COL_MUTED, font=("Segoe UI", 8), wraplength=320).pack(anchor="w", pady=(4, 0))

        audit_fr = ttk.LabelFrame(right_inner, text="Decision Audit", padding=6)
        audit_fr.pack(fill="x", pady=(0, 6))
        dq_total = decision_quality.get("total")
        if dq_total is not None:
            ttk.Label(
                audit_fr,
                text=f"Decision Quality  {dq_total} / {decision_quality.get('max', 100)}",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(0, 4))
            for dim in decision_quality.get("dimensions") or []:
                ttk.Label(
                    audit_fr,
                    text=f"{dim.get('label')}  {dim.get('score')}/{dim.get('max')}",
                    font=("Segoe UI", 9),
                ).pack(anchor="w", padx=(8, 0))
            ttk.Separator(audit_fr, orient="horizontal").pack(fill="x", pady=4)
        for check in decision.get("audit_checks") or []:
            passed = bool(check.get("passed"))
            sym = "✓" if passed else "✗"
            color = COL_OK if passed else COL_WARN
            row = ttk.Frame(audit_fr)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{sym} {check.get('label')}", foreground=color, font=("Segoe UI", 9, "bold"), width=22).pack(side="left")
            ttk.Label(row, text=str(check.get("detail") or ""), font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True)
        ttk.Label(
            audit_fr,
            text=f"Decision: {decision.get('decision', '—')}",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(6, 0))

        exit_fr = ttk.LabelFrame(right_inner, text="Why This Trade Ended", padding=6)
        exit_fr.pack(fill="x", pady=(0, 6))
        target_missed = (
            "Hit" if exit_a.get("target_hit")
            else fmt_rupee(exit_a.get("target_missed_by")) if exit_a.get("target_missed_by") is not None
            else "—"
        )
        dual_spec_sections(
            exit_fr,
            [
                ("Exit Reason", exit_a.get("exit_reason_label")),
                ("Prediction Correct?", _yes_no(exit_a.get("prediction_correct"))),
                ("Strategy Correct?", _yes_no(exit_a.get("strategy_correct"))),
                ("Target Missed by", target_missed),
                ("Stop", "Hit" if exit_a.get("stop_hit") else "Never Hit"),
            ],
            [
                ("Maximum Profit", fmt_rupee(exit_a.get("maximum_profit"))),
                ("Maximum Drawdown", fmt_rupee(exit_a.get("maximum_drawdown"))),
                ("Held", f"{exit_a.get('held_seconds')} sec" if exit_a.get("held_seconds") is not None else "—"),
                ("Net PnL", fmt_rupee(exit_a.get("net_pnl"))),
                ("Exit Premium", fmt_rupee(max_opp.get("exit_premium"))),
            ],
            left_title="Exit Verdict",
            right_title="Trade Result",
            label_width=12,
        )

        opp_fr = ttk.LabelFrame(right_inner, text="Opportunity Curve", padding=6)
        opp_fr.pack(fill="x", pady=(0, 6))
        dual_spec_sections(
            opp_fr,
            [
                ("Possible", _signed_rupee(max_opp.get("maximum_possible"))),
                ("Captured", _signed_rupee(max_opp.get("captured_profit"))),
                ("Efficiency", (
                    f"{max_opp.get('capture_efficiency_pct'):.0f}%"
                    if max_opp.get("capture_efficiency_pct") is not None else "—"
                )),
            ],
            [
                ("Entry", fmt_rupee(max_opp.get("entry_premium"))),
                ("Highest", fmt_rupee(max_opp.get("highest_premium"))),
                ("Exit", fmt_rupee(max_opp.get("exit_premium"))),
            ],
            left_title="Capture",
            right_title="Premium Path",
            label_width=12,
        )
        opp_note = ""
        mp = _as_float(max_opp.get("maximum_possible"))
        if mp is not None and mp <= 0:
            opp_note = "Market never gave a favorable opportunity."
        elif max_opp.get("capture_efficiency_pct") is not None and float(max_opp["capture_efficiency_pct"]) < 30:
            opp_note = "Strategy failed to capture available opportunity."
        if opp_note:
            ttk.Label(opp_fr, text=opp_note, foreground=COL_WARN, font=("Segoe UI", 8), wraplength=320).pack(anchor="w", pady=(4, 0))

        since_fr = ttk.LabelFrame(right_inner, text="What Changed After Entry", padding=6)
        since_fr.pack(fill="x", pady=(0, 6))
        since_metrics = since.get("metrics") or {}

        def _since_val(key: str) -> str:
            val = since_metrics.get(key)
            if val is None:
                return "—"
            if key in ("delta", "theta", "gamma"):
                return f"{val:+.4f}"
            return f"{val:+.2f}%"

        dual_spec_sections(
            since_fr,
            [
                ("Spot", _since_val("spot")),
                ("Premium", _since_val("premium")),
                ("IV", _since_val("iv")),
                ("PCR", _since_val("pcr")),
            ],
            [
                ("Delta", _since_val("delta")),
                ("Theta", _since_val("theta")),
                ("Gamma", _since_val("gamma")),
                ("Direction", str(since.get("direction") or "—").title()),
            ],
            left_title="Market",
            right_title="Greeks",
            label_width=12,
        )

        mount_plugin_panels(right_inner, doc=doc, win=win)

        feat_fr = ttk.LabelFrame(feat_host, text="Feature Replay", padding=4)
        feat_fr.pack(fill="both", expand=True, pady=(6, 0))

        if feature_alerts:
            alert_fr = ttk.LabelFrame(feat_fr, text="Feature Alerts", padding=4)
            alert_fr.pack(fill="x", pady=(0, 4))
            for alert in feature_alerts[:6]:
                sev = str(alert.get("severity") or "LOW")
                sev_color = COL_WARN if sev == "HIGH" else ("#e65100" if sev == "MEDIUM" else COL_MUTED)
                row = ttk.Frame(alert_fr)
                row.pack(fill="x", pady=1)
                ttk.Label(
                    row,
                    text=f"{alert.get('label')} {alert.get('value')} {alert.get('direction', '')}",
                    font=("Segoe UI", 9, "bold"),
                ).pack(anchor="w")
                ttk.Label(
                    row,
                    text=f"{alert.get('drift_text')} {alert.get('delta_label', '')} · Severity {sev}",
                    foreground=sev_color,
                    font=("Segoe UI", 8),
                ).pack(anchor="w", padx=(8, 0))

        feat_nb = ttk.Notebook(feat_fr)
        feat_nb.pack(fill="both", expand=True)
        feature_series = doc.get("feature_series") or {}
        if not feature_series:
            empty = ttk.Frame(feat_nb, padding=4)
            feat_nb.add(empty, text="Features")
            ttk.Label(empty, text="Feature evolution unavailable for this trade.", foreground=COL_MUTED).pack(anchor="w")
        for fname, series in feature_series.items():
            tab = ttk.Frame(feat_nb, padding=2)
            feat_nb.add(tab, text=fname[:18])
            shap = shap_by_feature.get(fname) or {}
            if shap:
                shap_line = (
                    f"{shap.get('value')}  {shap.get('arrow', '')} {shap.get('pct_change', 0):+.1f}%"
                )
                ttk.Label(tab, text=shap_line, font=("Segoe UI", 9, "bold")).pack(anchor="w")
                ttk.Label(
                    tab,
                    text=str(shap.get("direction_label") or ""),
                    foreground=COL_OK if "bullish" in str(shap.get("direction_label", "")).lower() else COL_WARN,
                    font=("Segoe UI", 8),
                ).pack(anchor="w", pady=(0, 2))
            fcols = ("time", "value", "delta")
            ft = ttk.Treeview(tab, columns=fcols, show="headings", height=3)
            for c, w, label in (("time", 64, "Time"), ("value", 80, "Value"), ("delta", 64, "Δ")):
                ft.heading(c, text=label)
                ft.column(c, width=w)
            ft.pack(fill="both", expand=True)
            ft.tag_configure("up", foreground=COL_OK)
            ft.tag_configure("down", foreground=COL_WARN)
            for pt in series:
                delta = pt.get("delta")
                tag = ""
                if delta is not None:
                    try:
                        d = float(delta)
                        tag = "up" if d > 0 else ("down" if d < 0 else "")
                    except (TypeError, ValueError):
                        pass
                d_txt = f"{delta:+.4f}" if delta is not None else "—"
                ft.insert("", "end", tags=(tag,) if tag else (), values=(pt.get("rel_label") or pt.get("time_label"), pt.get("value"), d_txt))

        def _ask_bookmark() -> tuple[str, str, str] | None:
            from chain_replay_ml.fold_research import generate_trade_observation

            dlg = tk.Toplevel(win)
            dlg.title("Research Notebook")
            dlg.transient(win)
            dlg.grab_set()
            f = ttk.Frame(dlg, padding=10)
            f.pack(fill="both", expand=True)
            ttk.Label(
                f,
                text="Save a bookmark with an auto-generated research observation.",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(0, 8))
            ttk.Label(f, text="Reason").pack(anchor="w")
            reason_var = tk.StringVar()
            ttk.Entry(f, textvariable=reason_var, width=48).pack(fill="x", pady=(0, 8))
            ttk.Label(f, text="Tags (comma-separated)").pack(anchor="w")
            tags_var = tk.StringVar()
            ttk.Entry(f, textvariable=tags_var, width=48).pack(fill="x", pady=(0, 8))
            ttk.Label(f, text="Observation").pack(anchor="w")
            note_box = scrolledtext.ScrolledText(f, height=8, width=48, font=("Segoe UI", 9))
            note_box.pack(fill="x", pady=(0, 8))
            out: dict[str, str | None] = {"ok": None}

            def _fill_observation() -> None:
                obs = generate_trade_observation(doc)
                reason_var.set(str(obs.get("reason") or ""))
                tags_var.set(", ".join(obs.get("tags") or []))
                note_box.delete("1.0", "end")
                note_box.insert("1.0", str(obs.get("body") or ""))

            def _ok() -> None:
                r = reason_var.get().strip()
                body = note_box.get("1.0", "end").strip()
                if not r and not body:
                    messagebox.showinfo("Research Notebook", "Generate an observation or enter a reason.", parent=dlg)
                    return
                if not r:
                    r = body.split("\n", 1)[0][:120]
                out["ok"] = r
                out["tags"] = tags_var.get().strip()
                out["note"] = body
                dlg.destroy()

            btn_row = ttk.Frame(f)
            btn_row.pack(fill="x", pady=(0, 4))
            ttk.Button(btn_row, text="Generate Observation", command=_fill_observation).pack(side="left")
            ttk.Button(btn_row, text="Save", command=_ok).pack(side="right")
            dlg.wait_window()
            if not out.get("ok"):
                return None
            return str(out["ok"]), str(out.get("tags") or ""), str(out.get("note") or "")

        def _save_observation() -> None:
            from chain_replay_ml.fold_research import generate_trade_observation, save_fold_note

            pred_id, fold_id, _ = self._selected_ids()
            if not pred_id or not fold_id:
                messagebox.showinfo("Research Notebook", "Select a prediction run and fold first.", parent=win)
                return
            obs = generate_trade_observation(doc)
            try:
                save_fold_note(
                    self._data_dir(),
                    prediction_run_id=str(pred_id),
                    fold_id=str(fold_id),
                    title=str(obs.get("title") or "Trade observation"),
                    body=str(obs.get("body") or ""),
                    tags=list(obs.get("tags") or []),
                    model_id=(self._detail or {}).get("prediction_run", {}).get("model_id"),
                )
                self._load_research()
                messagebox.showinfo("Research Notebook", "Observation saved to fold notebook.", parent=win)
            except Exception as exc:
                messagebox.showerror("Research Notebook", str(exc), parent=win)

        def _bookmark() -> None:
            from chain_replay_ml.fold_research import generate_trade_observation, save_fold_note, save_research_bookmark

            asked = _ask_bookmark()
            if not asked:
                return
            reason, tags_raw, notebook_note = asked
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            verdict_tags = [str(r.get("label")) for r in (verdict.get("reasons") or []) if r.get("label")]
            for vt in verdict_tags:
                if vt not in tags:
                    tags.append(vt)
            sel = tl.selection()
            ev = None
            if sel:
                ev = next((e for e in self._trade_replay_events if str(e.get("sequence")) == sel[0]), None)
            pred_id, fold_id, strat_id = self._selected_ids()
            observation = generate_trade_observation(doc)
            save_research_bookmark(
                self._data_dir(),
                prediction_run_id=str(pred_id),
                fold_id=str(fold_id),
                trade_id=str(trade.get("trade_id") or ""),
                timestamp=ev.get("timestamp") if ev else trade.get("entry_ts"),
                sequence=ev.get("sequence") if ev else None,
                title=f"Fold {(self._detail or {}).get('fold', {}).get('fold_number')} {ev.get('time_label') if ev else ''}".strip(),
                reason=reason,
                tags=tags,
                context={
                    "token": trade.get("token"),
                    "event_type": ev.get("event_type") if ev else "trade",
                    "tags": tags,
                    "notebook_note": notebook_note,
                    "observation": observation,
                    "prediction": pred,
                    "trade_verdict": verdict,
                    "since_entry": since,
                    "feature_alerts": feature_alerts,
                    "maximum_opportunity": max_opp,
                    "strategy_run_id": strat_id,
                    "price_paths": paths,
                },
            )
            if notebook_note:
                try:
                    save_fold_note(
                        self._data_dir(),
                        prediction_run_id=str(pred_id),
                        fold_id=str(fold_id),
                        title=reason[:120],
                        body=notebook_note,
                        tags=tags,
                        model_id=(self._detail or {}).get("prediction_run", {}).get("model_id"),
                    )
                except Exception:
                    pass
            self._load_research()
            messagebox.showinfo("Research Notebook", "Bookmark saved. Observation synced to fold notebook.", parent=win)

        similar_fr = ttk.LabelFrame(similar_host, text="Similar Trades", padding=6)
        similar_fr.pack(fill="both", expand=True, pady=(6, 0))
        similar_by_id = {str(st.get("trade_id")): st for st in similar}
        if not similar:
            ttk.Label(similar_fr, text="No similar trades in this fold yet.", foreground=COL_MUTED).pack(anchor="w")
        else:
            sim_cols = ("trade", "pnl", "similarity", "reason")
            sim_tree = ttk.Treeview(similar_fr, columns=sim_cols, show="headings", height=4)
            for c, w, label in (("trade", 100, "Trade"), ("pnl", 90, "PnL"), ("similarity", 80, "Similarity"), ("reason", 120, "Exit")):
                sim_tree.heading(c, text=label)
                sim_tree.column(c, width=w)
            sim_tree.pack(fill="x")
            sim_match_var = tk.StringVar(value="")
            ttk.Label(similar_fr, textvariable=sim_match_var, foreground=COL_MUTED, font=("Segoe UI", 8), wraplength=480).pack(anchor="w", pady=(4, 0))

            def _show_similar_match(_event: tk.Event | None = None) -> None:
                sel = sim_tree.selection()
                if not sel:
                    sim_match_var.set("")
                    return
                st = similar_by_id.get(sel[0]) or {}
                parts = []
                for m in st.get("matched_on") or []:
                    sym = "✓" if m.get("matched") else "✗"
                    parts.append(f"{sym} {m.get('label')}")
                sim_match_var.set("Matched on:  " + "  ".join(parts) if parts else "")

            def _selected_similar_id() -> str | None:
                sel = sim_tree.selection()
                return str(sel[0]) if sel else None

            def _replay_similar() -> None:
                tid = _selected_similar_id()
                if tid:
                    self._open_trade_replay(tid)

            def _compare_similar() -> None:
                tid = _selected_similar_id()
                if not tid:
                    return
                self._compare_trade_id = tid
                messagebox.showinfo(
                    "Compare Trades",
                    f"Compare target set: {tid[:12]}…\nOpen Replay on another trade to view side-by-side (coming soon).",
                    parent=win,
                )

            def _notebook_similar() -> None:
                from chain_replay_ml.fold_research import generate_trade_observation, get_trade_replay, save_fold_note

                tid = _selected_similar_id()
                pred_id, fold_id, _ = self._selected_ids()
                if not tid or not pred_id or not fold_id:
                    return
                try:
                    sdoc = get_trade_replay(
                        self._data_dir(),
                        prediction_run_id=str(pred_id),
                        fold_id=str(fold_id),
                        trade_id=tid,
                    )
                    if not sdoc.get("ok"):
                        return
                    obs = generate_trade_observation(sdoc)
                    save_fold_note(
                        self._data_dir(),
                        prediction_run_id=str(pred_id),
                        fold_id=str(fold_id),
                        title=str(obs.get("title") or f"Similar trade {tid[:8]}"),
                        body=str(obs.get("body") or ""),
                        tags=list(obs.get("tags") or []),
                        model_id=(self._detail or {}).get("prediction_run", {}).get("model_id"),
                    )
                    self._load_research()
                    messagebox.showinfo("Research Notebook", "Observation saved for similar trade.", parent=win)
                except Exception as exc:
                    messagebox.showerror("Research Notebook", str(exc), parent=win)

            sim_actions = sim_actions_host
            ttk.Button(sim_actions, text="Replay", command=_replay_similar, width=10).pack(side="left", padx=(0, 4))
            ttk.Button(sim_actions, text="Compare", command=_compare_similar, width=10).pack(side="left", padx=(0, 4))
            ttk.Button(sim_actions, text="Notebook", command=_notebook_similar, width=10).pack(side="left", padx=(0, 8))
            ttk.Label(sim_actions, text="Double-click row = Replay", foreground=COL_MUTED, font=("Segoe UI", 8)).pack(side="left")

            def _on_similar_dbl(_event: tk.Event) -> None:
                _replay_similar()

            sim_tree.bind("<<TreeviewSelect>>", _show_similar_match)
            sim_tree.bind("<Double-1>", _on_similar_dbl)
            for st in similar:
                tid = str(st.get("trade_id") or "")
                pnl = st.get("net_pnl")
                sim_tree.insert(
                    "",
                    "end",
                    iid=tid,
                    values=(
                        tid[:12],
                        fmt_rupee(pnl) if pnl is not None else "—",
                        f"{st.get('similarity_pct')}%",
                        st.get("exit_reason") or "—",
                    ),
                )
            kids = sim_tree.get_children()
            if kids:
                sim_tree.selection_set(kids[0])
                _show_similar_match()

        ttk.Button(action_row, text="📝 Save Observation", command=_save_observation).pack(side="right", padx=(6, 0))
        ttk.Button(action_row, text="⭐ Bookmark this moment", command=_bookmark).pack(side="right")

        win.update_idletasks()
        place_toplevel_beside_main(win, self)
        win.lift()
        win.focus_force()
