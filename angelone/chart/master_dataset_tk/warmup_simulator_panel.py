"""Warm-up Simulator tab — policy engine on real tick grid."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from chain_replay_ml.feature_policy import DEFAULT_GAP_MAX_SEC
from chain_replay_ml.feature_policy.warmup_simulator import (
    WarmupSimulationResult,
    compare_ema_readiness,
    list_trading_days,
    simulate_warmup,
)

from . import feature_policy_format as pol_fmt
from . import warmup_simulator_format as sim_fmt
from .build_config_prefs import (
    infer_simulator_duration_preset,
    load_build_config_prefs,
    resolve_simulator_duration_minutes,
    save_build_config_prefs,
    simulator_duration_prefs_for_save,
)


def _resolve_chart_dir(widget: tk.Misc) -> str | None:
    w: tk.Misc | None = widget
    while w is not None:
        if hasattr(w, "chart_dir"):
            return str(getattr(w, "chart_dir"))
        w = w.master  # type: ignore[assignment]
    return None


class WarmupSimulatorPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str | None = None,
        feature_names: list[str] | None = None,
        features_by_name: dict[str, dict[str, Any]] | None = None,
        sampling_interval_sec: float = 10.0,
        gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
        get_selected_feature: Callable[[], str | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, padding=6, **kwargs)
        self._chart_dir = chart_dir
        self._feature_names = list(feature_names or [])
        self._features_by_name: dict[str, dict[str, Any]] = dict(features_by_name or {})
        self._interval = float(sampling_interval_sec)
        self._gap_max = float(gap_max_sec)
        self._get_selected_feature = get_selected_feature
        self._gap_injections: list[tuple[int, float]] = []
        self._last_trace_len = 0
        self._running = False
        self._last_result: WarmupSimulationResult | None = None
        self._last_compare_rows: list[dict[str, Any]] | None = None
        self._data_filter_features: list[str] = []
        self._pending_data_filter_features: list[str] = []

        self._day_var = tk.StringVar()
        self._duration_var = tk.IntVar(value=15)
        self._custom_duration_var = tk.StringVar(value="30")
        self._feature_var = tk.StringVar()
        self._feature_search_var = tk.StringVar()
        self._feature_match_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="Configure inputs and run simulation.")
        self._maturity_replay_var = tk.BooleanVar(value=True)
        self._all_features_calc_var = tk.BooleanVar(value=False)
        self._build_dataset_sel_var = tk.BooleanVar(value=True)
        self._build_gap_parity_var = tk.BooleanVar(value=True)
        self._lookback_nearest_var = tk.BooleanVar(value=True)
        self._lookback_dual_pass_var = tk.BooleanVar(value=False)
        self._gap_pass_compare_var = tk.BooleanVar(value=False)
        self._temp_build_io_var = tk.BooleanVar(value=True)
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugLevel

        self._perf_level_var = tk.StringVar(value=PerformanceDebugLevel.OFF.value)

        self._build_ui()
        self.refresh_trading_days()
        self.load_prefs()
        self._sync_profiler_option_states()
        self._refresh_policy_summary()
        self._day_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._duration_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._custom_duration_var.trace_add("write", self._on_custom_duration_change)
        self._feature_search_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._maturity_replay_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._all_features_calc_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._build_dataset_sel_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._build_gap_parity_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._lookback_nearest_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._lookback_dual_pass_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._gap_pass_compare_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._temp_build_io_var.trace_add("write", lambda *_a: self._save_sim_prefs())
        self._perf_level_var.trace_add("write", lambda *_a: self._on_perf_level_change())

    def set_chart_dir(self, chart_dir: str | None) -> None:
        self._chart_dir = chart_dir
        self.refresh_trading_days()
        self.load_prefs()

    def _sim_prefs(self) -> dict[str, Any]:
        if not self._chart_dir:
            return {}
        doc = load_build_config_prefs(self._chart_dir) or {}
        sim = doc.get("simulator")
        return sim if isinstance(sim, dict) else {}

    def _save_sim_prefs(self) -> None:
        if not self._chart_dir:
            return
        duration_prefs = simulator_duration_prefs_for_save(
            preset_minutes=int(self._duration_var.get()),
            custom_minutes=self._custom_duration_var.get(),
        )
        save_build_config_prefs(self._chart_dir, {
            "simulator": {
                "trading_day": self._day_var.get().strip(),
                **duration_prefs,
                "feature": self._feature_var.get().strip(),
                "feature_search": self._feature_search_var.get().strip(),
                "dataset_maturity_replay": bool(self._maturity_replay_var.get()),
                "calculate_all_features": bool(self._all_features_calc_var.get()),
                "match_build_dataset_selection": bool(self._build_dataset_sel_var.get()),
                "match_build_gap_parity": bool(self._build_gap_parity_var.get()),
                "lookback_nearest_snapshot": bool(self._lookback_nearest_var.get()),
                "lookback_dual_pass_benchmark": bool(self._lookback_dual_pass_var.get()),
                "run_gap_pass_comparison": bool(self._gap_pass_compare_var.get()),
                "temp_build_io": bool(self._temp_build_io_var.get()),
                "performance_debug_level": self._perf_level_var.get().strip(),
                "data_filter_features": list(self._data_filter_features),
            },
        })

    def load_prefs(self) -> None:
        prefs = self._sim_prefs()
        if not prefs:
            return
        search = str(prefs.get("feature_search") or "").strip()
        if search:
            self._feature_search_var.set(search)
        custom = str(prefs.get("custom_duration") or "").strip()
        if custom:
            self._custom_duration_var.set(custom)
        self._duration_var.set(
            infer_simulator_duration_preset(
                prefs.get("duration_minutes"),
                custom or self._custom_duration_var.get(),
            ),
        )
        day = str(prefs.get("trading_day") or "").strip()
        if day:
            values = list(self._day_combo["values"]) if hasattr(self, "_day_combo") else []
            if not values or day in values:
                self._day_var.set(day)
        feat = str(prefs.get("feature") or "").strip()
        if feat and feat in self._feature_names:
            self._select_feature_in_list(feat)
        elif feat and self._feature_names:
            self._feature_search_var.set(feat)
            self._refresh_feature_list()
        if "dataset_maturity_replay" in prefs:
            self._maturity_replay_var.set(bool(prefs.get("dataset_maturity_replay")))
        if "calculate_all_features" in prefs:
            self._all_features_calc_var.set(bool(prefs.get("calculate_all_features")))
        if "match_build_dataset_selection" in prefs:
            self._build_dataset_sel_var.set(bool(prefs.get("match_build_dataset_selection")))
        elif "match_build_parity" in prefs:
            self._build_dataset_sel_var.set(bool(prefs.get("match_build_parity")))
        if "match_build_gap_parity" in prefs:
            self._build_gap_parity_var.set(bool(prefs.get("match_build_gap_parity")))
        elif "match_build_parity" in prefs:
            self._build_gap_parity_var.set(bool(prefs.get("match_build_parity")))
        if "lookback_nearest_snapshot" in prefs:
            self._lookback_nearest_var.set(bool(prefs.get("lookback_nearest_snapshot")))
        if "lookback_dual_pass_benchmark" in prefs:
            self._lookback_dual_pass_var.set(bool(prefs.get("lookback_dual_pass_benchmark")))
        if "run_gap_pass_comparison" in prefs:
            self._gap_pass_compare_var.set(bool(prefs.get("run_gap_pass_comparison")))
        if "temp_build_io" in prefs:
            self._temp_build_io_var.set(bool(prefs.get("temp_build_io")))
        if prefs.get("performance_debug_level"):
            from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugLevel

            self._perf_level_var.set(
                PerformanceDebugLevel.from_value(prefs.get("performance_debug_level")).value,
            )
        saved_filters = prefs.get("data_filter_features")
        if isinstance(saved_filters, list):
            pending = [str(f).strip() for f in saved_filters if str(f).strip()]
            if self._feature_names:
                self._data_filter_features = [f for f in pending if f in self._feature_names]
            else:
                self._pending_data_filter_features = pending
                self._data_filter_features = []
        self._refresh_data_filter_list()
        self._sync_profiler_option_states()

    def set_features(
        self,
        names: list[str],
        *,
        features_by_name: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._feature_names = list(dict.fromkeys(names))
        if features_by_name is not None:
            self._features_by_name = dict(features_by_name)
        pending = list(self._pending_data_filter_features)
        saved = self._sim_prefs().get("data_filter_features")
        if isinstance(saved, list):
            pending.extend(str(f).strip() for f in saved if str(f).strip())
        if pending:
            merged = list(dict.fromkeys([*self._data_filter_features, *pending]))
            self._data_filter_features = [f for f in merged if f in self._feature_names]
            self._pending_data_filter_features = []
        else:
            self._data_filter_features = [
                f for f in self._data_filter_features if f in self._feature_names
            ]
        if self._feature_names and self._feature_var.get() not in self._feature_names:
            saved = str(self._sim_prefs().get("feature") or "").strip()
            if saved in self._feature_names:
                self._feature_var.set(saved)
            else:
                self._feature_var.set(self._feature_names[0])
        self._refresh_feature_list()
        self._refresh_data_filter_list()
        self._refresh_policy_summary()
        self._save_sim_prefs()
        if self._last_result is not None:
            self._render_data_filter_table(self._last_result)

    def set_features_by_name(self, features_by_name: dict[str, dict[str, Any]]) -> None:
        self._features_by_name = dict(features_by_name)
        self._refresh_policy_summary()

    def set_sampling_interval(self, sec: float) -> None:
        self._interval = max(0.001, float(sec))
        self._sampling_lbl.configure(text=f"{self._interval:g} sec")
        self.refresh_trading_days()
        self._refresh_policy_summary()

    def _resolve_market(self) -> str:
        chart_dir = self._chart_dir or _resolve_chart_dir(self)
        if not chart_dir:
            return "NIFTY"
        try:
            from .build_config_prefs import load_build_config_prefs

            doc = load_build_config_prefs(chart_dir) or {}
            master = doc.get("master_data") if isinstance(doc.get("master_data"), dict) else {}
            build = doc.get("build") if isinstance(doc.get("build"), dict) else {}
            market = str(
                master.get("market") or build.get("market") or "NIFTY"
            ).strip().upper()
            if market in {"NIFTY", "BANKNIFTY", "SENSEX"}:
                return market
        except Exception:
            pass
        return "NIFTY"

    def refresh_trading_days(self) -> None:
        chart_dir = self._chart_dir or _resolve_chart_dir(self)
        days = (
            list_trading_days(
                chart_dir,
                sampling_interval_sec=self._interval,
                market=self._resolve_market(),
            )
            if chart_dir
            else []
        )
        current = self._day_var.get().strip()
        self._day_combo["values"] = days
        if days and (not current or current not in days):
            self._day_var.set(days[0])
        elif not days:
            self._day_var.set("")

    def _build_ui(self) -> None:
        outer = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill="both", expand=True)
        self._outer_paned = outer

        inputs_col = ttk.Frame(outer)
        outer.add(inputs_col, weight=35)

        inputs = ttk.LabelFrame(inputs_col, text="Inputs", padding=6)
        inputs.pack(fill="x")

        day_row = ttk.Frame(inputs)
        day_row.pack(fill="x", pady=(0, 6))
        ttk.Label(day_row, text="Trading Day").pack(side="left")
        self._day_combo = ttk.Combobox(
            day_row, textvariable=self._day_var, width=14, state="readonly",
        )
        self._day_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))

        dur_row = ttk.Frame(inputs)
        dur_row.pack(fill="x", pady=(0, 6))
        ttk.Label(dur_row, text="Duration").pack(side="left")
        dur_opts = ttk.Frame(dur_row)
        dur_opts.pack(side="left", fill="x", expand=True, padx=(6, 0))
        for val, label in ((5, "5"), (10, "10"), (15, "15")):
            ttk.Radiobutton(
                dur_opts,
                text=label,
                variable=self._duration_var,
                value=val,
                command=self._on_duration_preset_change,
            ).pack(side="left", padx=(0, 2))
        ttk.Radiobutton(
            dur_opts,
            text="Custom",
            variable=self._duration_var,
            value=0,
            command=self._save_sim_prefs,
        ).pack(side="left", padx=(4, 2))
        ttk.Entry(dur_opts, textvariable=self._custom_duration_var, width=4).pack(side="left")
        ttk.Label(dur_opts, text="min").pack(side="left", padx=(2, 0))

        sampling_row = ttk.Frame(inputs)
        sampling_row.pack(fill="x", pady=(0, 8))
        ttk.Label(sampling_row, text="Sampling").pack(side="left")
        self._sampling_lbl = ttk.Label(sampling_row, text=f"{self._interval:g} sec")
        self._sampling_lbl.pack(side="left", padx=(6, 0))

        ttk.Label(inputs, text="Feature").pack(anchor="w")
        feat_box = ttk.Frame(inputs)
        feat_box.pack(fill="x", pady=(2, 8))

        search_row = ttk.Frame(feat_box)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="Search").pack(side="left")
        search_entry = ttk.Entry(search_row, textvariable=self._feature_search_var, width=22)
        search_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._feature_search_var.trace_add("write", lambda *_a: self._refresh_feature_list())

        list_frame = ttk.Frame(feat_box)
        list_frame.pack(fill="x", pady=(4, 2))
        self._feature_list = tk.Listbox(
            list_frame, height=7, exportselection=False, activestyle="dotbox",
        )
        feat_vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self._feature_list.yview)
        self._feature_list.configure(yscrollcommand=feat_vsb.set)
        self._feature_list.pack(side="left", fill="both", expand=True)
        feat_vsb.pack(side="right", fill="y")
        self._feature_list.bind("<<ListboxSelect>>", self._on_feature_list_select)
        self._feature_list.bind("<Double-Button-1>", self._on_feature_list_add_data_filter)

        ttk.Label(
            feat_box, textvariable=self._feature_match_var, foreground="#666", font=("Segoe UI", 8),
        ).pack(anchor="w")
        ttk.Label(feat_box, text="Selected:").pack(anchor="w", pady=(2, 0))
        ttk.Label(
            feat_box, textvariable=self._feature_var, foreground="#1a237e", wraplength=260,
        ).pack(anchor="w")

        if self._feature_names:
            self._feature_var.set(self._feature_names[0])
        self._refresh_feature_list()

        feat_btn_row = ttk.Frame(inputs)
        feat_btn_row.pack(fill="x", pady=(0, 8))
        ttk.Button(feat_btn_row, text="Use Selected Feature", command=self._use_selected).pack(side="left")
        ttk.Button(feat_btn_row, text="Feature Detail…", command=self._open_selected_feature_detail).pack(
            side="left", padx=(6, 0),
        )

        gap_row = ttk.Frame(inputs)
        gap_row.pack(fill="x", pady=(0, 4))
        ttk.Label(gap_row, text="Gap Testing", font=("Segoe UI", 9, "bold")).pack(side="left")
        gap_btns = ttk.Frame(gap_row)
        gap_btns.pack(side="left", padx=(6, 0))
        for sec, label in ((5, "5s"), (10, "10s"), (30, "30s"), (60, "1m"), (300, "5m")):
            ttk.Button(
                gap_btns, text=label,
                command=lambda s=sec: self._queue_gap(s),
            ).pack(side="left", padx=(0, 3))

        action_row = ttk.Frame(inputs)
        action_row.pack(fill="x", pady=(0, 6))
        self._run_btn = ttk.Button(action_row, text="Run Simulation", command=self._run_simulation)
        self._run_btn.pack(side="left", padx=(0, 6))
        ttk.Button(action_row, text="Compare Features", command=self._open_compare_dialog).pack(side="left")

        ttk.Checkbutton(
            inputs,
            text="Calculate all features (slower)",
            variable=self._all_features_calc_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            inputs,
            text="Dataset maturity replay (all features — slower)",
            variable=self._maturity_replay_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            inputs,
            text="Match build config (trim targets, enabled features)",
            variable=self._build_dataset_sel_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            inputs,
            text="Match build gap policy",
            variable=self._build_gap_parity_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            inputs,
            text="Lookback nearest_snapshot (main replay pass)",
            variable=self._lookback_nearest_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        perf_box = ttk.LabelFrame(inputs, text="Performance Level", padding=4)
        perf_box.pack(fill="x", pady=(0, 6))
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugLevel

        for level in (PerformanceDebugLevel.OFF, PerformanceDebugLevel.BASIC, PerformanceDebugLevel.FULL):
            ttk.Radiobutton(
                perf_box,
                text=level.ui_label(),
                variable=self._perf_level_var,
                value=level.value,
                command=self._on_perf_level_change,
            ).pack(anchor="w")

        self._lookback_dual_pass_cb = ttk.Checkbutton(
            inputs,
            text="Run exact_timestamp vs nearest_snapshot benchmark (Full Debug only)",
            variable=self._lookback_dual_pass_var,
            command=self._save_sim_prefs,
        )
        self._lookback_dual_pass_cb.pack(anchor="w", pady=(0, 4))

        self._gap_pass_compare_cb = ttk.Checkbutton(
            inputs,
            text="Compare gap OFF vs ON when gap parity OFF (Full Debug only)",
            variable=self._gap_pass_compare_var,
            command=self._save_sim_prefs,
        )
        self._gap_pass_compare_cb.pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(
            inputs,
            text="Temp SQLite + Parquet export (benchmark build I/O)",
            variable=self._temp_build_io_var,
            command=self._save_sim_prefs,
        ).pack(anchor="w", pady=(0, 4))

        self._gap_var = tk.StringVar(value="No gaps queued")
        ttk.Label(inputs, textvariable=self._gap_var, foreground="#666", wraplength=220).pack(anchor="w", pady=(0, 4))

        ttk.Label(inputs, textvariable=self._status_var, foreground="#444", wraplength=240).pack(anchor="w")

        policy_box = ttk.LabelFrame(inputs_col, text="Feature Policy Summary", padding=4)
        policy_box.pack(fill="both", expand=True, pady=(6, 0))
        self._policy_summary = scrolledtext.ScrolledText(
            policy_box, wrap="word", font=("Consolas", 8), height=14,
        )
        self._policy_summary.pack(fill="both", expand=True)
        self._set_readonly_text(self._policy_summary, "")

        output = ttk.LabelFrame(outer, text="Simulation Output", padding=4)
        outer.add(output, weight=65)
        self._outer_paned_split_done = False
        outer.bind("<Configure>", self._apply_outer_paned_split)
        self.after_idle(self._apply_outer_paned_split)
        out_paned = ttk.Panedwindow(output, orient=tk.VERTICAL)
        out_paned.pack(fill="both", expand=True)

        progress_box = ttk.LabelFrame(out_paned, text="Progress", padding=4)
        out_paned.add(progress_box, weight=1)
        self._progress = scrolledtext.ScrolledText(
            progress_box, wrap="word", font=("Consolas", 8), height=8,
        )
        self._progress.pack(fill="both", expand=True)

        results_box = ttk.LabelFrame(out_paned, text="Results", padding=4)
        out_paned.add(results_box, weight=3)
        self._results_notebook = ttk.Notebook(results_box)
        self._results_notebook.pack(fill="both", expand=True)

        timeline_tab = ttk.Frame(self._results_notebook, padding=2)
        self._results_notebook.add(timeline_tab, text="Timeline & Results")
        results_toolbar = ttk.Frame(timeline_tab)
        results_toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(results_toolbar, text="Copy", command=self._copy_results).pack(side="left", padx=(0, 4))
        ttk.Button(results_toolbar, text="Download CSV", command=self._download_results_csv).pack(side="left")
        self._output = scrolledtext.ScrolledText(
            timeline_tab, wrap="word", font=("Consolas", 9), height=20,
        )
        self._output.pack(fill="both", expand=True)

        calc_tab = ttk.Frame(self._results_notebook, padding=2)
        self._results_notebook.add(calc_tab, text="Feature Calculation")
        calc_toolbar = ttk.Frame(calc_tab)
        calc_toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(
            calc_toolbar,
            text="Download All Features CSV",
            command=self._download_all_features_csv,
        ).pack(side="left")
        self._calc_summary_var = tk.StringVar(value="Run a simulation to inspect formula and operands.")
        ttk.Label(
            calc_tab, textvariable=self._calc_summary_var, wraplength=700, foreground="#444",
        ).pack(anchor="w", pady=(0, 4))
        calc_paned = ttk.Panedwindow(calc_tab, orient=tk.VERTICAL)
        calc_paned.pack(fill="both", expand=True)
        calc_table_box = ttk.Frame(calc_paned)
        calc_paned.add(calc_table_box, weight=3)
        self._calc_tree = ttk.Treeview(calc_table_box, show="headings", height=16)
        calc_vsb = ttk.Scrollbar(calc_table_box, orient="vertical", command=self._calc_tree.yview)
        calc_hsb = ttk.Scrollbar(calc_table_box, orient="horizontal", command=self._calc_tree.xview)
        self._calc_tree.configure(yscrollcommand=calc_vsb.set, xscrollcommand=calc_hsb.set)
        self._calc_tree.grid(row=0, column=0, sticky="nsew")
        calc_vsb.grid(row=0, column=1, sticky="ns")
        calc_hsb.grid(row=1, column=0, sticky="ew")
        calc_table_box.rowconfigure(0, weight=1)
        calc_table_box.columnconfigure(0, weight=1)
        self._calc_tree.bind("<<TreeviewSelect>>", self._on_calc_row_select)
        calc_detail_box = ttk.LabelFrame(calc_paned, text="Selected Row", padding=4)
        calc_paned.add(calc_detail_box, weight=2)
        self._calc_detail = scrolledtext.ScrolledText(
            calc_detail_box, wrap="word", font=("Consolas", 9), height=12,
        )
        self._calc_detail.pack(fill="both", expand=True)
        self._calc_rows_index: list[dict[str, Any]] = []

        maturity_tab = ttk.Frame(self._results_notebook, padding=2)
        self._results_notebook.add(maturity_tab, text="Dataset Maturity")
        self._maturity_gauge_var = tk.StringVar(value="Run simulation for dataset maturity.")
        ttk.Label(
            maturity_tab, textvariable=self._maturity_gauge_var, font=("Consolas", 9),
        ).pack(anchor="w", pady=(0, 4))
        mat_paned = ttk.Panedwindow(maturity_tab, orient=tk.VERTICAL)
        mat_paned.pack(fill="both", expand=True)
        mat_top = ttk.Frame(mat_paned)
        mat_paned.add(mat_top, weight=3)
        mat_cols = (
            "sample", "ready", "not_ready", "ready_pct",
            "raw", "rolling", "derived", "skip",
        )
        self._maturity_tree = ttk.Treeview(mat_top, columns=mat_cols, show="headings", height=12)
        headers = {
            "sample": "Sample", "ready": "Ready", "not_ready": "Not Ready",
            "ready_pct": "Ready %", "raw": "Raw", "rolling": "Rolling",
            "derived": "Derived", "skip": "Skip Row",
        }
        for col in mat_cols:
            self._maturity_tree.heading(col, text=headers[col])
            w = 56 if col in ("sample", "raw", "rolling", "derived", "skip") else 64
            if col == "ready_pct":
                w = 52
            self._maturity_tree.column(col, width=w, anchor="center")
        mat_vsb = ttk.Scrollbar(mat_top, orient="vertical", command=self._maturity_tree.yview)
        mat_hsb = ttk.Scrollbar(mat_top, orient="horizontal", command=self._maturity_tree.xview)
        self._maturity_tree.configure(yscrollcommand=mat_vsb.set, xscrollcommand=mat_hsb.set)
        self._maturity_tree.grid(row=0, column=0, sticky="nsew")
        mat_vsb.grid(row=0, column=1, sticky="ns")
        mat_hsb.grid(row=1, column=0, sticky="ew")
        mat_top.rowconfigure(0, weight=1)
        mat_top.columnconfigure(0, weight=1)
        self._maturity_tree.bind("<<TreeviewSelect>>", self._on_maturity_row_select)
        self._maturity_tree.bind("<Double-1>", self._on_maturity_row_double_click)
        mat_bottom = ttk.Frame(mat_paned)
        mat_paned.add(mat_bottom, weight=2)
        self._maturity_chart = scrolledtext.ScrolledText(
            mat_bottom, wrap="none", font=("Consolas", 8), height=7,
        )
        self._maturity_chart.pack(fill="both", expand=True, pady=(0, 4))
        self._maturity_detail = scrolledtext.ScrolledText(
            mat_bottom, wrap="word", font=("Consolas", 9), height=10,
        )
        self._maturity_detail.pack(fill="both", expand=True)
        self._maturity_rows_index: list[dict[str, Any]] = []
        self._maturity_total = 0

        timing_tab = ttk.Frame(self._results_notebook, padding=2)
        self._timing_tab = timing_tab
        self._results_notebook.add(timing_tab, text="Time Taken")
        self._timing_summary_var = tk.StringVar(value="Run a simulation to see timing breakdown.")
        ttk.Label(
            timing_tab, textvariable=self._timing_summary_var, font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._timing_text = scrolledtext.ScrolledText(
            timing_tab, wrap="word", font=("Consolas", 9), height=18,
        )
        self._timing_text.pack(fill="both", expand=True)

        gap_prof_tab = ttk.Frame(self._results_notebook, padding=2)
        self._gap_profiler_tab = gap_prof_tab
        self._results_notebook.add(gap_prof_tab, text="Gap Profiler")
        self._gap_profiler_summary_var = tk.StringVar(
            value="Enable build gap parity and run feature calc to see gap profiler.",
        )
        ttk.Label(
            gap_prof_tab, textvariable=self._gap_profiler_summary_var, font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._gap_profiler_text = scrolledtext.ScrolledText(
            gap_prof_tab, wrap="word", font=("Consolas", 9), height=18,
        )
        self._gap_profiler_text.pack(fill="both", expand=True)

        data_filter_tab = ttk.Frame(self._results_notebook, padding=2)
        self._data_filter_tab = data_filter_tab
        self._results_notebook.add(data_filter_tab, text="Data Filter")
        ttk.Label(
            data_filter_tab,
            text="Double-click a feature in the Inputs list to add it to the filter below.",
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=680,
        ).pack(anchor="w", pady=(0, 4))
        df_list_row = ttk.Frame(data_filter_tab)
        df_list_row.pack(fill="x", pady=(0, 4))
        self._data_filter_list = tk.Listbox(
            df_list_row, height=4, exportselection=False, activestyle="dotbox",
        )
        df_list_vsb = ttk.Scrollbar(df_list_row, orient="vertical", command=self._data_filter_list.yview)
        self._data_filter_list.configure(yscrollcommand=df_list_vsb.set)
        self._data_filter_list.pack(side="left", fill="x", expand=True)
        df_list_vsb.pack(side="right", fill="y")
        df_btn_row = ttk.Frame(data_filter_tab)
        df_btn_row.pack(fill="x", pady=(0, 4))
        ttk.Button(df_btn_row, text="Remove", command=self._remove_data_filter_feature).pack(side="left")
        ttk.Button(df_btn_row, text="Clear", command=self._clear_data_filter_features).pack(side="left", padx=(4, 0))
        ttk.Button(df_btn_row, text="Download CSV", command=self._download_data_filter_csv).pack(side="right")
        self._data_filter_hint_var = tk.StringVar(value="No features in filter — run simulation to populate table.")
        ttk.Label(
            data_filter_tab, textvariable=self._data_filter_hint_var, foreground="#888", font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 4))
        df_table_box = ttk.Frame(data_filter_tab)
        df_table_box.pack(fill="both", expand=True)
        self._data_filter_notebook = ttk.Notebook(df_table_box)
        self._data_filter_notebook.pack(fill="both", expand=True)

        values_tab = ttk.Frame(self._data_filter_notebook, padding=2)
        self._data_filter_notebook.add(values_tab, text="Values")
        values_table = ttk.Frame(values_tab)
        values_table.pack(fill="both", expand=True)
        self._data_filter_tree = ttk.Treeview(values_table, show="headings", height=14)
        df_vsb = ttk.Scrollbar(values_table, orient="vertical", command=self._data_filter_tree.yview)
        df_hsb = ttk.Scrollbar(values_table, orient="horizontal", command=self._data_filter_tree.xview)
        self._data_filter_tree.configure(yscrollcommand=df_vsb.set, xscrollcommand=df_hsb.set)
        self._data_filter_tree.grid(row=0, column=0, sticky="nsew")
        df_vsb.grid(row=0, column=1, sticky="ns")
        df_hsb.grid(row=1, column=0, sticky="ew")
        values_table.rowconfigure(0, weight=1)
        values_table.columnconfigure(0, weight=1)

        ratio_tab = ttk.Frame(self._data_filter_notebook, padding=2)
        self._data_filter_ratio_tab = ratio_tab
        self._data_filter_notebook.add(ratio_tab, text="Ratio Split")
        ttk.Label(
            ratio_tab,
            text="ltp/spot + derived ltp_ema{N}: ltp_ema{N}_to_ltp_ratio × ltp or ltp_ema{N}_to_spot_ratio × spot.",
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=680,
        ).pack(anchor="w", pady=(0, 4))
        ratio_table = ttk.Frame(ratio_tab)
        ratio_table.pack(fill="both", expand=True)
        self._data_filter_ratio_tree = ttk.Treeview(ratio_table, show="headings", height=14)
        ratio_vsb = ttk.Scrollbar(ratio_table, orient="vertical", command=self._data_filter_ratio_tree.yview)
        ratio_hsb = ttk.Scrollbar(ratio_table, orient="horizontal", command=self._data_filter_ratio_tree.xview)
        self._data_filter_ratio_tree.configure(yscrollcommand=ratio_vsb.set, xscrollcommand=ratio_hsb.set)
        self._data_filter_ratio_tree.grid(row=0, column=0, sticky="nsew")
        ratio_vsb.grid(row=0, column=1, sticky="ns")
        ratio_hsb.grid(row=1, column=0, sticky="ew")
        ratio_table.rowconfigure(0, weight=1)
        ratio_table.columnconfigure(0, weight=1)

        warmup_tab = ttk.Frame(self._data_filter_notebook, padding=2)
        self._data_filter_warmup_tab = warmup_tab
        self._data_filter_notebook.add(warmup_tab, text="Warmup Regression")
        ttk.Label(
            warmup_tab,
            text="First valid sample per NullUntilReady feature — validated from replay data after simulation.",
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=680,
        ).pack(anchor="w", pady=(0, 4))
        warmup_table = ttk.Frame(warmup_tab)
        warmup_table.pack(fill="both", expand=True)
        self._data_filter_warmup_tree = ttk.Treeview(warmup_table, show="headings", height=14)
        warmup_vsb = ttk.Scrollbar(warmup_table, orient="vertical", command=self._data_filter_warmup_tree.yview)
        warmup_hsb = ttk.Scrollbar(warmup_table, orient="horizontal", command=self._data_filter_warmup_tree.xview)
        self._data_filter_warmup_tree.configure(yscrollcommand=warmup_vsb.set, xscrollcommand=warmup_hsb.set)
        self._data_filter_warmup_tree.grid(row=0, column=0, sticky="nsew")
        warmup_vsb.grid(row=0, column=1, sticky="ns")
        warmup_hsb.grid(row=1, column=0, sticky="ew")
        warmup_table.rowconfigure(0, weight=1)
        warmup_table.columnconfigure(0, weight=1)

        null_audit_tab = ttk.Frame(self._data_filter_notebook, padding=2)
        self._data_filter_null_audit_tab = null_audit_tab
        self._data_filter_notebook.add(null_audit_tab, text="Null Audit")
        ttk.Label(
            null_audit_tab,
            text="Session-wide NULL / value / missing status for every replay feature column.",
            foreground="#666",
            font=("Segoe UI", 8),
            wraplength=680,
        ).pack(anchor="w", pady=(0, 4))
        null_audit_table = ttk.Frame(null_audit_tab)
        null_audit_table.pack(fill="both", expand=True)
        self._data_filter_null_audit_tree = ttk.Treeview(null_audit_table, show="headings", height=14)
        null_vsb = ttk.Scrollbar(null_audit_table, orient="vertical", command=self._data_filter_null_audit_tree.yview)
        null_hsb = ttk.Scrollbar(null_audit_table, orient="horizontal", command=self._data_filter_null_audit_tree.xview)
        self._data_filter_null_audit_tree.configure(yscrollcommand=null_vsb.set, xscrollcommand=null_hsb.set)
        self._data_filter_null_audit_tree.grid(row=0, column=0, sticky="nsew")
        null_vsb.grid(row=0, column=1, sticky="ns")
        null_hsb.grid(row=1, column=0, sticky="ew")
        null_audit_table.rowconfigure(0, weight=1)
        null_audit_table.columnconfigure(0, weight=1)

        self._clear_progress()
        self._set_output("Run a simulation to see timeline, readiness chart, and event log.")

    def _apply_outer_paned_split(self, _event: tk.Event | None = None) -> None:
        if getattr(self, "_outer_paned_split_done", False):
            return
        paned = getattr(self, "_outer_paned", None)
        if paned is None:
            return
        width = paned.winfo_width()
        if width <= 1:
            return
        paned.sashpos(0, max(160, int(width * 0.35)))
        self._outer_paned_split_done = True

    def _filtered_feature_names(self) -> list[str]:
        query = self._feature_search_var.get().strip().lower()
        if not query:
            return list(self._feature_names)
        return [n for n in self._feature_names if query in n.lower()]

    def _refresh_feature_list(self) -> None:
        if not hasattr(self, "_feature_list"):
            return
        filtered = self._filtered_feature_names()
        current = self._feature_var.get().strip()
        self._feature_list.delete(0, tk.END)
        for name in filtered:
            self._feature_list.insert(tk.END, name)
        total = len(self._feature_names)
        shown = len(filtered)
        if query := self._feature_search_var.get().strip():
            self._feature_match_var.set(
                f"{shown} match{'es' if shown != 1 else ''} for \"{query}\""
                + (f" (of {total})" if shown < total else ""),
            )
        else:
            self._feature_match_var.set(f"{total} features")
        if current in filtered:
            idx = filtered.index(current)
            self._feature_list.selection_set(idx)
            self._feature_list.see(idx)
        elif filtered:
            self._feature_list.selection_set(0)
            self._feature_list.see(0)
            self._feature_var.set(filtered[0])
            self._refresh_policy_summary()

    def _select_feature_in_list(self, name: str) -> None:
        if name not in self._feature_names:
            return
        self._feature_var.set(name)
        if name not in self._filtered_feature_names():
            self._feature_search_var.set("")
        self._refresh_feature_list()

    def _on_feature_list_select(self, _event: tk.Event | None = None) -> None:
        sel = self._feature_list.curselection()
        if not sel:
            return
        name = str(self._feature_list.get(sel[0]))
        self._feature_var.set(name)
        self._refresh_policy_summary()
        self._save_sim_prefs()

    def _on_feature_list_add_data_filter(self, _event: tk.Event | None = None) -> None:
        sel = self._feature_list.curselection()
        if not sel:
            return
        name = str(self._feature_list.get(sel[0]))
        self.add_data_filter_feature(name)

    def _open_selected_feature_detail(self) -> None:
        name = self._feature_var.get().strip()
        if name:
            self._open_feature_detail(name)

    def add_data_filter_feature(self, name: str) -> None:
        feat = str(name or "").strip()
        if not feat or feat in self._data_filter_features:
            return
        if feat not in self._feature_names:
            return
        self._data_filter_features.append(feat)
        self._refresh_data_filter_list()
        self._save_sim_prefs()
        self.focus_data_filter_tab()

    def focus_data_filter_tab(self) -> None:
        if hasattr(self, "_results_notebook") and hasattr(self, "_data_filter_tab"):
            self._results_notebook.select(self._data_filter_tab)

    def _remove_data_filter_feature(self) -> None:
        sel = self._data_filter_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self._data_filter_features):
            del self._data_filter_features[idx]
            self._refresh_data_filter_list()
            self._save_sim_prefs()
            self._render_data_filter_table(self._last_result)

    def _clear_data_filter_features(self) -> None:
        self._data_filter_features.clear()
        self._refresh_data_filter_list()
        self._save_sim_prefs()
        self._render_data_filter_table(self._last_result)

    def _refresh_data_filter_list(self) -> None:
        if not hasattr(self, "_data_filter_list"):
            return
        self._data_filter_list.delete(0, tk.END)
        for feat in self._data_filter_features:
            self._data_filter_list.insert(tk.END, feat)
        n = len(self._data_filter_features)
        if n:
            self._data_filter_hint_var.set(
                f"{n} feature(s) in filter — run simulation to refresh values below.",
            )
        else:
            self._data_filter_hint_var.set("No features in filter — double-click features to add.")

    def _render_data_filter_table(self, result: WarmupSimulationResult | None) -> None:
        if not hasattr(self, "_data_filter_tree"):
            return
        features = list(self._data_filter_features)
        self._populate_data_filter_tree(self._data_filter_tree, *sim_fmt.build_data_filter_table(result, features))
        ratio_cols, ratio_rows = sim_fmt.build_data_filter_ratio_split_table(result, features)
        if hasattr(self, "_data_filter_ratio_tree"):
            self._populate_data_filter_tree(self._data_filter_ratio_tree, ratio_cols, ratio_rows)
        warmup_cols, warmup_rows = sim_fmt.build_controller_warmup_regression_table(result)
        if hasattr(self, "_data_filter_warmup_tree"):
            self._populate_data_filter_tree(self._data_filter_warmup_tree, warmup_cols, warmup_rows)
        null_cols, null_rows = sim_fmt.build_null_audit_table(result)
        if hasattr(self, "_data_filter_null_audit_tree"):
            self._populate_data_filter_tree(self._data_filter_null_audit_tree, null_cols, null_rows)

        if not features:
            hint = sim_fmt.controller_warmup_regression_hint(result)
            null_hint = sim_fmt.null_audit_hint(result)
            self._data_filter_hint_var.set(
                f"No features in filter — double-click features to add. "
                f"Warmup Regression: {hint} · Null Audit: {null_hint}",
            )
            return
        if not result or not result.ok:
            return
        lookup = result.all_features_lookup or result.maturity_replay_lookup
        n_ratio = len(sim_fmt.ratio_split_column_plan(features)[1])
        warmup_hint = sim_fmt.controller_warmup_regression_hint(result)
        null_hint = sim_fmt.null_audit_hint(result)
        if not lookup:
            self._data_filter_hint_var.set(
                "Enable Dataset maturity replay (or Calculate all features) and re-run. "
                f"Warmup Regression: {warmup_hint} · Null Audit: {null_hint}",
            )
            return
        trace_n = len(result.full_trace or [])
        self._data_filter_hint_var.set(
            f"{len(features)} feature(s) · {trace_n} sample row(s)"
            + (f" · Ratio Split: {n_ratio} EMA column(s)" if n_ratio else "")
            + f" · Warmup Regression: {warmup_hint}"
            + f" · Null Audit: {null_hint}",
        )

    def _populate_data_filter_tree(
        self,
        tree: ttk.Treeview,
        cols: list[str],
        rows: list[list[str]],
    ) -> None:
        for item in tree.get_children():
            tree.delete(item)
        if not cols:
            tree["columns"] = ()
            return
        tree["columns"] = cols
        for col in cols:
            hdr = col
            if col == "sample":
                hdr = "Sample"
            elif col == "time":
                hdr = "Time"
            width = 56 if col == "sample" else (72 if col == "time" else 88)
            tree.heading(col, text=hdr)
            tree.column(col, width=width, minwidth=48, anchor="center")
        for i, row in enumerate(rows):
            tree.insert("", "end", iid=f"{tree}-{i}", values=row)

    def _download_data_filter_csv(self) -> None:
        result = self._last_result
        tab_id = self._data_filter_notebook.select() if hasattr(self, "_data_filter_notebook") else ""
        is_null_audit = bool(
            hasattr(self, "_data_filter_null_audit_tab")
            and tab_id == str(self._data_filter_null_audit_tab),
        )
        if is_null_audit:
            if not result or not result.ok:
                messagebox.showinfo("Download CSV", "Run a simulation first, then download.")
                return
            csv_text = sim_fmt.null_audit_csv(result)
            if not csv_text:
                messagebox.showinfo("Download CSV", "No Null Audit rows — run simulation with Dataset maturity replay.")
                return
            default_name = sim_fmt.default_csv_filename(
                feature_name="null_audit",
                trading_day=result.trading_day,
                prefix="warmup_null_audit",
            )
            path = filedialog.asksaveasfilename(
                parent=self.winfo_toplevel(),
                title="Save Null Audit CSV",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(csv_text)
                self._status_var.set(f"Saved Null Audit CSV: {path}")
            except OSError as exc:
                messagebox.showerror("Download CSV", f"Could not save file:\n{exc}")
            return
        is_warmup = bool(
            hasattr(self, "_data_filter_warmup_tab")
            and tab_id == str(self._data_filter_warmup_tab),
        )
        if is_warmup:
            if not result or not result.ok:
                messagebox.showinfo("Download CSV", "Run a simulation first, then download.")
                return
            csv_text = sim_fmt.controller_warmup_regression_csv(result)
            if not csv_text:
                messagebox.showinfo("Download CSV", "No Warmup Regression rows — run simulation first.")
                return
            default_name = sim_fmt.default_csv_filename(
                feature_name="warmup_regression",
                trading_day=result.trading_day,
                prefix="warmup_controller_regression",
            )
            path = filedialog.asksaveasfilename(
                parent=self.winfo_toplevel(),
                title="Save Warmup Regression CSV",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(csv_text)
                self._status_var.set(f"Saved Warmup Regression CSV: {path}")
            except OSError as exc:
                messagebox.showerror("Download CSV", f"Could not save file:\n{exc}")
            return

        features = list(self._data_filter_features)
        if not features:
            messagebox.showinfo("Download CSV", "Add features to the Data Filter list first.")
            return
        result = self._last_result
        if not result or not result.ok:
            messagebox.showinfo("Download CSV", "Run a simulation first, then download.")
            return
        tab_id = self._data_filter_notebook.select() if hasattr(self, "_data_filter_notebook") else ""
        is_ratio = bool(
            hasattr(self, "_data_filter_ratio_tab")
            and tab_id == str(self._data_filter_ratio_tab),
        )
        is_warmup = bool(
            hasattr(self, "_data_filter_warmup_tab")
            and tab_id == str(self._data_filter_warmup_tab),
        )
        if is_warmup:
            csv_text = sim_fmt.controller_warmup_regression_csv(result)
            suffix = "warmup_regression"
            empty_hint = "No Warmup Regression rows — run simulation with Dataset maturity replay."
        elif is_ratio:
            csv_text = sim_fmt.data_filter_ratio_split_csv(result, features)
            suffix = "ratio_split"
            empty_hint = "No Ratio Split rows — add ltp_ema ratio features and re-run simulation."
        else:
            csv_text = sim_fmt.data_filter_values_csv(result, features)
            suffix = "values"
            empty_hint = "No Values rows — run simulation with Dataset maturity replay enabled."
        if not csv_text:
            messagebox.showinfo("Download CSV", empty_hint)
            return
        default_name = sim_fmt.default_csv_filename(
            feature_name="data_filter",
            trading_day=result.trading_day,
            prefix=f"warmup_data_filter_{suffix}",
        )
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save Data Filter CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            self._status_var.set(f"Saved Data Filter CSV: {path}")
        except OSError as exc:
            messagebox.showerror("Download CSV", f"Could not save file:\n{exc}")

    def _dataset_feature_names_for_sim(self) -> list[str] | None:
        base = list(self._feature_names or [])
        merged = list(dict.fromkeys([*base, *self._data_filter_features]))
        need_ltp = False
        need_spot = False
        for feat in self._data_filter_features:
            spec = sim_fmt.ratio_split_spec(feat)
            if spec is None:
                continue
            if spec[1] == "ltp":
                need_ltp = True
            else:
                need_spot = True
        extras: list[str] = []
        if need_ltp:
            extras.append("ltp")
        if need_spot:
            extras.append("spot")
        if extras:
            merged = list(dict.fromkeys([*merged, *extras]))
        return merged if merged else None

    def _on_feature_list_detail(self, _event: tk.Event | None = None) -> None:
        sel = self._feature_list.curselection()
        if not sel:
            return
        name = str(self._feature_list.get(sel[0]))
        self._open_feature_detail(name)

    def _open_feature_detail(
        self,
        feature_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        from .feature_detail_panel import open_feature_detail_window

        open_feature_detail_window(
            self,
            feature_name,
            chart_dir=self._chart_dir or _resolve_chart_dir(self),
            features_by_name=self._features_by_name,
            context=context,
        )

    def _refresh_policy_summary(self) -> None:
        text = pol_fmt.format_simulator_policy_sidebar(
            self._feature_var.get().strip() or None,
            self._feature_names,
            self._features_by_name,
            sampling_interval_sec=self._interval,
            gap_max_sec=self._gap_max,
        )
        self._set_readonly_text(self._policy_summary, text)

    def _use_selected(self) -> None:
        if not self._get_selected_feature:
            return
        name = self._get_selected_feature()
        if name:
            self._select_feature_in_list(name)
            self._refresh_policy_summary()

    def _build_prefs(self) -> dict[str, Any]:
        if not self._chart_dir:
            return {}
        doc = load_build_config_prefs(self._chart_dir) or {}
        build = doc.get("build")
        return build if isinstance(build, dict) else {}

    def _replay_build_settings(self) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.lookback_policy import (
            DEFAULT_LOOKBACK_POLICY,
            POLICY_EXACT_TIMESTAMP,
            build_dataset_configuration,
            normalize_policy_doc,
        )

        settings: dict[str, Any] = {
            "match_build_dataset_selection": bool(self._build_dataset_sel_var.get()),
            "match_build_gap_parity": bool(self._build_gap_parity_var.get()),
            "apply_lookback_nearest": bool(self._lookback_nearest_var.get()),
            "run_lookback_dual_pass_benchmark": bool(self._lookback_dual_pass_var.get()),
            "run_gap_pass_comparison": bool(self._gap_pass_compare_var.get()),
            "performance_debug_level": self._perf_level_var.get().strip(),
        }
        if self._lookback_nearest_var.get():
            settings["lookback_policy"] = normalize_policy_doc(dict(DEFAULT_LOOKBACK_POLICY))
        else:
            settings["lookback_policy"] = normalize_policy_doc({
                "method": POLICY_EXACT_TIMESTAMP,
                "label": "Exact Timestamp",
            })

        if self._build_dataset_sel_var.get():
            build = self._build_prefs()
            settings["feature_selection"] = {
                "profile": str(build.get("feature_profile") or "default"),
                "enabledGroups": list(build.get("enabled_groups") or []),
                "enabledFeatures": list(build.get("enabled_features") or []),
                "applied": True,
            }
            settings["trim_target_rows"] = True

        if self._build_gap_parity_var.get():
            from chain_replay_ml.dataset_builder.gap_policy import (
                default_gap_policy,
                gap_max_sec_from_policy,
                normalize_gap_policy,
            )

            build = self._build_prefs()
            gap_policy = normalize_gap_policy(
                build.get("gap_policy") if isinstance(build.get("gap_policy"), dict) else None,
            )
            if not build.get("gap_policy"):
                gap_policy = default_gap_policy()
            gap_max = gap_max_sec_from_policy(gap_policy)
            hz = self._horizons_sec()
            step = int(max(self._interval, 1))
            ds_cfg = build_dataset_configuration(
                sampling={"trainingIntervalSec": step, "interval_sec": step},
                horizons_sec=hz,
                gap_max_sec=gap_max,
            )
            settings.update({
                "gap_policy": gap_policy,
                "gap_max_sec": gap_max,
                "dataset_configuration": ds_cfg,
            })
        return settings

    def _strike_selection(self) -> dict[str, Any]:
        if not self._chart_dir:
            from chain_replay_ml.dataset_builder.master_defaults import default_master_strike_selection
            return default_master_strike_selection()
        doc = load_build_config_prefs(self._chart_dir) or {}
        build = doc.get("build")
        saved = build.get("strike_selection") if isinstance(build, dict) else None
        from .strike_selection_engine import strike_selection_for_master
        return strike_selection_for_master(saved if isinstance(saved, dict) else None)

    def _horizons_sec(self) -> list[int]:
        if not self._chart_dir:
            from .target_horizons import DEFAULT_HORIZON_SEC
            return list(DEFAULT_HORIZON_SEC)
        doc = load_build_config_prefs(self._chart_dir) or {}
        build = doc.get("build")
        if isinstance(build, dict):
            hz = build.get("horizons_sec")
            if isinstance(hz, list) and hz:
                return sorted(int(h) for h in hz if int(h) > 0)
        from .target_horizons import DEFAULT_HORIZON_SEC
        return list(DEFAULT_HORIZON_SEC)

    def _on_custom_duration_change(self, *_a) -> None:
        raw = self._custom_duration_var.get().strip()
        if raw:
            try:
                if int(raw) > 0:
                    self._duration_var.set(0)
            except ValueError:
                pass
        self._save_sim_prefs()

    def _on_perf_level_change(self) -> None:
        self._sync_profiler_option_states()
        self._save_sim_prefs()

    def _sync_profiler_option_states(self) -> None:
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel

        perf = PerformanceDebugConfig.resolve(self._perf_level_var.get())
        full_debug = perf.level == PerformanceDebugLevel.FULL
        state = "normal" if full_debug else "disabled"
        for widget in (
            getattr(self, "_lookback_dual_pass_cb", None),
            getattr(self, "_gap_pass_compare_cb", None),
        ):
            if widget is not None:
                widget.configure(state=state)

    def _on_duration_preset_change(self) -> None:
        preset = int(self._duration_var.get())
        if preset != 0:
            self._custom_duration_var.set("")
        self._save_sim_prefs()

    def _duration_minutes(self) -> int:
        return resolve_simulator_duration_minutes(
            preset_minutes=int(self._duration_var.get()),
            custom_minutes=self._custom_duration_var.get(),
        )

    def _queue_gap(self, gap_sec: float) -> None:
        idx = max(0, self._last_trace_len // 2) if self._last_trace_len else 10
        injection = (idx, float(gap_sec))
        self._gap_var.set(
            f"Injecting {int(gap_sec)}s gap after sample {idx} — auto-running simulation…",
        )
        self._run_simulation(gap_injections=[injection])

    def _open_compare_dialog(self) -> None:
        chart_dir = self._chart_dir or _resolve_chart_dir(self)
        if not chart_dir:
            messagebox.showwarning("Compare Features", "No chart folder — open a project first.")
            return
        day = self._day_var.get().strip()
        if not day:
            messagebox.showwarning("Compare Features", "Select a trading day first.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Compare EMA Features")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        ttk.Label(dlg, text="Select EMA controllers to compare:", padding=8).pack(anchor="w")
        periods = (9, 20, 50, 100, 200, 300)
        vars_map: dict[int, tk.BooleanVar] = {}
        box = ttk.Frame(dlg, padding=(8, 0))
        box.pack(fill="x")
        for p in periods:
            var = tk.BooleanVar(value=True)
            vars_map[p] = var
            ttk.Checkbutton(box, text=f"EMA{p}", variable=var).pack(anchor="w")

        status = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=status, foreground="#444", wraplength=280, padding=8).pack(anchor="w")

        def run_compare() -> None:
            selected = tuple(p for p, v in vars_map.items() if v.get())
            if not selected:
                messagebox.showwarning("Compare Features", "Select at least one EMA period.")
                return
            if self._running:
                return
            self._running = True
            status.set("Running comparison…")
            self._clear_progress()
            self._append_progress("Compare Features clicked")
            self._append_progress(f"EMA periods: {', '.join(f'EMA{p}' for p in selected)}")
            duration = self._duration_minutes()

            def work() -> None:
                def on_progress(msg: str) -> None:
                    self.after(0, lambda m=msg: self._append_progress(m))

                try:
                    rows = compare_ema_readiness(
                        chart_dir=chart_dir,
                        trading_day=day,
                        duration_minutes=duration,
                        sampling_interval_sec=self._interval,
                        gap_max_sec=self._gap_max,
                        ema_periods=selected,
                        feature_names=self._feature_names or None,
                        on_progress=on_progress,
                    )
                    text = sim_fmt.format_compare_features_table(rows)
                except Exception as exc:
                    rows = []
                    text = f"Compare failed:\n  {exc}"
                self.after(0, lambda r=rows, t=text: self._on_compare_done(t, dlg, status, r))

            threading.Thread(target=work, daemon=True).start()

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Run Compare", command=run_compare).pack(side="left")
        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="left", padx=8)

    def _on_compare_done(
        self,
        text: str,
        dlg: tk.Toplevel,
        status: tk.StringVar,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._running = False
        status.set("Done.")
        dlg.destroy()
        self._last_result = None
        self._last_compare_rows = list(rows) if rows else None
        self._set_output(text)
        self._render_calc_debug(None)
        self._render_maturity_timeline(None)
        self._render_timing_tab(None)
        self._render_gap_profiler_tab(None)
        self._render_data_filter_table(None)
        self._status_var.set("Feature comparison complete.")

    def _results_text(self) -> str:
        return self._output.get("1.0", tk.END).strip()

    def _copy_results(self) -> None:
        text = self._results_text()
        if not text:
            messagebox.showinfo("Copy", "Nothing to copy — run a simulation first.")
            return
        root = self.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update_idletasks()
        self._status_var.set("Timeline & results copied to clipboard.")

    def _download_results_csv(self) -> None:
        csv_text = ""
        default_name = "warmup_sim.csv"
        if self._last_result and (self._last_result.full_trace or []):
            csv_text = sim_fmt.simulation_trace_csv(self._last_result)
            default_name = sim_fmt.default_csv_filename(
                feature_name=self._last_result.feature_name,
                trading_day=self._last_result.trading_day,
                prefix="warmup_sim",
            )
        elif self._last_compare_rows:
            csv_text = sim_fmt.compare_features_csv(self._last_compare_rows)
            day = self._day_var.get().strip().replace("-", "")
            default_name = sim_fmt.default_csv_filename(trading_day=day, prefix="warmup_compare")
        if not csv_text:
            messagebox.showinfo("Download CSV", "No tabular data — run a simulation or feature compare first.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save simulation CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            self._status_var.set(f"Saved CSV: {path}")
        except OSError as exc:
            messagebox.showerror("Download CSV", f"Could not save file:\n{exc}")

    def _download_all_features_csv(self) -> None:
        result = self._last_result
        if not result or not result.ok:
            messagebox.showinfo(
                "Download All Features CSV",
                "Run a simulation first, then download.",
            )
            return
        status = sim_fmt.all_features_export_status(result)
        csv_text = sim_fmt.all_features_csv(result)
        if not csv_text:
            timing = result.timing or {}
            hints = [
                "Enable \"Calculate all features\" or \"Dataset maturity replay\" and click Run.",
                "The replay must finish successfully (check Simulation Output for errors).",
            ]
            if timing.get("maturity_replay_error"):
                hints.append(f"Replay error: {timing['maturity_replay_error']}")
            elif getattr(result, "maturity_replay_error", None):
                hints.append(f"Replay error: {result.maturity_replay_error}")
            detail = (
                f"Chain rows: {status.get('chain_rows', 0):,}\n"
                f"Lookup buckets: {status.get('lookup_buckets', 0):,}\n"
                f"Policy samples: {status.get('trace_samples', 0):,}\n"
                f"All-features calc: {'yes' if status.get('all_features_calc') else 'no'}"
            )
            messagebox.showinfo(
                "Download All Features CSV",
                "No all-features export data in the last run.\n\n"
                + "\n".join(hints)
                + "\n\n"
                + detail,
            )
            return
        default_name = sim_fmt.default_csv_filename(
            feature_name="all_features",
            trading_day=result.trading_day,
            prefix="warmup_all_features",
        )
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save all features CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            self._status_var.set(f"Saved all-features CSV: {path}")
        except OSError as exc:
            messagebox.showerror("Download All Features CSV", f"Could not save file:\n{exc}")

    def _run_simulation(self, *, gap_injections: list[tuple[int, float]] | None = None) -> None:
        if self._running:
            return
        chart_dir = self._chart_dir or _resolve_chart_dir(self)
        if not chart_dir:
            messagebox.showwarning("Warm-up Simulator", "No chart folder — open a project first.")
            return
        feature = self._feature_var.get().strip()
        day = self._day_var.get().strip()
        if not feature or not day:
            messagebox.showwarning("Warm-up Simulator", "Select trading day and feature.")
            return
        duration = self._duration_minutes()
        self._running = True
        self._run_btn.configure(state="disabled")
        self._status_var.set("Running simulation…")
        self._clear_progress()
        self._append_progress("Run Simulation clicked")
        self._append_progress(f"Feature: {feature}")
        self._append_progress(f"Day: {day} · Duration: {duration} min · Grid: {self._interval:g}s")
        self._set_output("Processing… timeline and results will appear below when complete.")
        gaps = list(gap_injections) if gap_injections is not None else list(self._gap_injections)
        self._gap_injections = []
        if gaps:
            self._gap_var.set(
                "Gap replay: " + ", ".join(f"{int(s)}s after sample {i}" for i, s in gaps),
            )
        else:
            self._gap_var.set("No gaps")
        if not self._maturity_replay_var.get():
            self._append_progress("Dataset maturity replay: OFF (skipped)")
        if self._all_features_calc_var.get():
            self._append_progress("All features calculation: ON")
        if self._all_features_calc_var.get() and self._build_dataset_sel_var.get():
            self._append_progress("Build config: ON (trim targets, enabled features)")
        elif self._all_features_calc_var.get():
            self._append_progress("Build config: OFF (all features, keep NULL targets)")
        if self._all_features_calc_var.get() and self._build_gap_parity_var.get():
            self._append_progress("Build gap policy: ON")
        elif self._all_features_calc_var.get():
            self._append_progress("Build gap policy: OFF")
        if self._all_features_calc_var.get() and self._lookback_nearest_var.get():
            if self._lookback_dual_pass_var.get():
                from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugLevel

                if PerformanceDebugLevel.from_value(self._perf_level_var.get()) == PerformanceDebugLevel.FULL:
                    self._append_progress(
                        "Lookback dual-pass benchmark: ON (exact_timestamp + nearest_snapshot)",
                    )
                else:
                    self._append_progress(
                        "Lookback dual-pass benchmark: skipped (Full Debug only)",
                    )
            else:
                self._append_progress("Lookback nearest_snapshot: ON (single pass)")
        elif self._all_features_calc_var.get():
            self._append_progress("Lookback nearest_snapshot: OFF (exact_timestamp)")
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel

        perf = PerformanceDebugConfig.resolve(self._perf_level_var.get())
        self._append_progress(f"Performance level: {perf.level.ui_label()}")
        if self._all_features_calc_var.get() and perf.run_gap_pass_comparison(
            explicit=bool(self._gap_pass_compare_var.get()),
            gap_parity=bool(self._build_gap_parity_var.get()),
        ):
            self._append_progress(
                "Gap pass comparison: ON (isolated single-pass OFF vs ON, independent of dual lookback)",
            )
        if self._all_features_calc_var.get() and self._temp_build_io_var.get():
            self._append_progress("Temp build I/O: ON (SQLite insert + Parquet export)")
        strike_lbl = str(self._strike_selection().get("mode") or "atm_band")
        self._append_progress(f"Strike selection: {strike_lbl} (from build config)")
        hz = self._horizons_sec()
        if hz and self._all_features_calc_var.get():
            from .target_horizons import horizon_label
            self._append_progress(
                "Prediction targets: " + ", ".join(horizon_label(h) for h in hz),
            )

        def work() -> None:
            def on_progress(msg: str) -> None:
                self.after(0, lambda m=msg: self._append_progress(m))

            try:
                result = simulate_warmup(
                    chart_dir=chart_dir,
                    trading_day=day,
                    feature_name=feature,
                    duration_minutes=duration,
                    sampling_interval_sec=self._interval,
                    gap_max_sec=self._gap_max,
                    gap_injections=gaps,
                    dataset_feature_names=self._dataset_feature_names_for_sim(),
                    run_dataset_maturity_replay=bool(self._maturity_replay_var.get()),
                    run_all_features_calc=bool(self._all_features_calc_var.get()),
                    strike_selection=self._strike_selection(),
                    horizons_sec=self._horizons_sec(),
                    build_replay_settings=self._replay_build_settings(),
                    run_temp_build_io=bool(
                        self._all_features_calc_var.get() and self._temp_build_io_var.get(),
                    ),
                    on_progress=on_progress,
                )
            except Exception as exc:
                result = WarmupSimulationResult(ok=False, error=str(exc))
            self.after(0, lambda: self._on_done(result))

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, result: WarmupSimulationResult) -> None:
        self._running = False
        self._run_btn.configure(state="normal")
        self._last_result = result if (result.ok or result.timing) else None
        self._last_compare_rows = None
        if result.ok:
            self._last_trace_len = result.samples_processed
            self._status_var.set(
                f"Done — {result.samples_processed} samples"
                + (f", ready at sample {result.ready_at_sample}" if result.ready_at_sample else ", not ready"),
            )
        else:
            self._status_var.set("Simulation failed.")
        self._set_output(sim_fmt.format_simulation_results_body(result))
        self._render_calc_debug(result)
        self._render_maturity_timeline(result)
        self._render_timing_tab(result)
        self._render_data_filter_table(result)

    def _render_timing_tab(self, result: WarmupSimulationResult | None) -> None:
        if not hasattr(self, "_timing_text"):
            return
        if hasattr(self, "_timing_tab"):
            self._results_notebook.tab(
                self._timing_tab,
                text=sim_fmt.timing_tab_title(result),
            )
        timing = (result.timing if result else None) or {}
        if not timing:
            self._timing_summary_var.set("Run a simulation to see timing breakdown.")
            self._set_readonly_text(self._timing_text, "")
            return
        load_sec = timing.get("load_ticks_sec")
        calc_sec = timing.get("feature_calc_sec")
        total_sec = timing.get("total_sec")
        spot = timing.get("spot_ticks")
        chain = timing.get("chain_ticks")
        tick_part = ""
        if spot is not None and chain is not None:
            tick_part = f" · {int(spot):,} spot + {int(chain):,} chain"
        self._timing_summary_var.set(
            f"Fetch {load_sec:.2f}s{tick_part} · Feature calc {calc_sec:.2f}s"
            + (
                f" · SQLite {timing.get('temp_sqlite_insert_sec', 0):.2f}s"
                f" · Parquet {timing.get('temp_parquet_export_sec', 0):.2f}s"
                if timing.get("temp_sqlite_insert_sec") is not None
                else ""
            )
            + f" · Total {total_sec:.2f}s"
            if load_sec is not None and calc_sec is not None and total_sec is not None
            else "Timing breakdown available below.",
        )
        self._set_readonly_text(self._timing_text, sim_fmt.format_timing_summary(result))
        self._render_gap_profiler_tab(result)

    def _render_gap_profiler_tab(self, result: WarmupSimulationResult | None) -> None:
        if not hasattr(self, "_gap_profiler_text"):
            return
        from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel

        timing = (result.timing if result else None) or {}
        perf = PerformanceDebugConfig.resolve(timing.get("performance_debug_level"))
        if perf.level == PerformanceDebugLevel.OFF:
            self._gap_profiler_summary_var.set("Gap profiler disabled in Production mode.")
            self._set_readonly_text(self._gap_profiler_text, "")
            return
        prof = timing.get("gap_policy_profiler")
        if not isinstance(prof, dict):
            if perf.show_full_diagnostics():
                cmp_doc = timing.get("gap_pass_comparison")
                if isinstance(cmp_doc, dict):
                    self._gap_profiler_summary_var.set(
                        f"Gap pass compare OFF {cmp_doc.get('gap_off_wall_sec', '—')}s · "
                        f"ON {cmp_doc.get('gap_on_wall_sec', '—')}s",
                    )
                    self._set_readonly_text(
                        self._gap_profiler_text,
                        sim_fmt.format_gap_pass_comparison(timing),
                    )
                    return
            self._gap_profiler_summary_var.set(
                "Run All Features Calc with Full Debug to profile gap overhead.",
            )
            self._set_readonly_text(self._gap_profiler_text, "")
            return
        checks = int(prof.get("gap_checks") or 0)
        gaps = int(prof.get("gaps_detected") or 0)
        resets = int(prof.get("reset_count") or 0)
        self._gap_profiler_summary_var.set(
            f"Gap checks {checks:,} · actual gaps {gaps:,} · resets {resets:,}",
        )
        body = sim_fmt.format_gap_policy_profiler(timing)
        self._set_readonly_text(self._gap_profiler_text, body)
        if perf.show_full_diagnostics():
            cmp_doc = timing.get("gap_pass_comparison")
            if isinstance(cmp_doc, dict):
                existing = self._gap_profiler_text.get("1.0", "end").strip()
                extra = sim_fmt.format_gap_pass_comparison(timing)
                if extra:
                    self._set_readonly_text(
                        self._gap_profiler_text,
                        f"{existing}\n\n{extra}" if existing else extra,
                    )

    def _render_maturity_timeline(self, result: WarmupSimulationResult | None) -> None:
        timeline = (result.maturity_timeline if result else None) or []
        summary = (result.maturity_summary if result else None) or {}
        total = int((result.dataset_feature_total if result else 0) or summary.get("feature_total") or 0)
        self._maturity_total = total
        self._maturity_rows_index = list(timeline)

        for item in self._maturity_tree.get_children():
            self._maturity_tree.delete(item)

        if not timeline:
            self._maturity_gauge_var.set("No maturity data — run a simulation.")
            self._set_readonly_text(self._maturity_chart, "")
            self._set_readonly_text(self._maturity_detail, "")
            return

        last = timeline[-1]
        self._maturity_gauge_var.set(
            sim_fmt.format_maturity_gauge(last, total=total).replace("\n", "   "),
        )

        display_rows = timeline

        for i, row in enumerate(display_rows):
            self._maturity_tree.insert(
                "", "end", iid=f"mat-{i}",
                values=(
                    row.get("sample"),
                    row.get("ready"),
                    row.get("not_ready"),
                    f"{row.get('ready_pct', 0)}%",
                    row.get("raw"),
                    row.get("rolling"),
                    row.get("derived"),
                    "YES" if row.get("skip_row") else "NO",
                ),
            )

        chart_text = sim_fmt.format_maturity_chart(summary, total=total)
        buckets = sim_fmt.format_maturity_buckets(summary)
        self._set_readonly_text(
            self._maturity_chart,
            "\n\n".join(p for p in (chart_text, buckets) if p),
        )

        children = self._maturity_tree.get_children()
        if children:
            last = children[-1]
            self._maturity_tree.selection_set(last)
            self._maturity_tree.focus(last)
            self._maturity_tree.see(last)
            self._on_maturity_row_select()

    def _on_maturity_row_select(self, _event: tk.Event | None = None) -> None:
        sel = self._maturity_tree.selection()
        if not sel:
            return
        try:
            idx = int(str(sel[0]).replace("mat-", ""))
        except ValueError:
            return
        if idx < 0 or idx >= len(self._maturity_rows_index):
            return
        row = self._maturity_rows_index[idx]
        self._set_readonly_text(
            self._maturity_detail,
            sim_fmt.format_maturity_row_detail(row, total=self._maturity_total),
        )
        self._maturity_gauge_var.set(
            sim_fmt.format_maturity_gauge(row, total=self._maturity_total).replace("\n", "   "),
        )

    def _maturity_sample_index(self) -> int | None:
        sel = self._maturity_tree.selection()
        if not sel:
            return None
        try:
            return int(str(sel[0]).replace("mat-", ""))
        except ValueError:
            return None

    def _on_maturity_row_double_click(self, _event: tk.Event) -> None:
        idx = self._maturity_sample_index()
        if idx is None:
            return
        result = self._last_result
        if not result or not result.ok:
            messagebox.showinfo(
                "Feature Values",
                "Run a simulation first to inspect feature values.",
            )
            return
        self._open_maturity_features_panel(idx)

    def _open_maturity_features_panel(self, sample_index: int) -> None:
        result = self._last_result
        if not result:
            return
        trace = result.full_trace or []
        if sample_index < 0 or sample_index >= len(trace):
            messagebox.showwarning("Feature Values", "Invalid sample row.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Maturity Feature Values")
        dlg.geometry("920x620")
        dlg.transient(self.winfo_toplevel())

        header = ttk.Label(dlg, text="Loading feature values…", padding=8)
        header.pack(anchor="w")

        filter_row = ttk.Frame(dlg, padding=(8, 0))
        filter_row.pack(fill="x")
        ttk.Label(filter_row, text="Search:").pack(side="left")
        search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_row, textvariable=search_var, width=24)
        search_entry.pack(side="left", padx=(4, 12))
        ttk.Label(filter_row, text="Show:").pack(side="left")
        show_var = tk.StringVar(value="all")
        show_combo = ttk.Combobox(
            filter_row,
            textvariable=show_var,
            values=("all", "null", "value", "missing"),
            state="readonly",
            width=12,
        )
        show_combo.pack(side="left", padx=4)

        tree_frame = ttk.Frame(dlg, padding=8)
        tree_frame.pack(fill="both", expand=True)
        cols = ("feature", "category", "ready", "status", "value")
        feat_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=22)
        headers = {
            "feature": "Feature",
            "category": "Category",
            "ready": "Ready",
            "status": "Status",
            "value": "Value",
        }
        widths = {"feature": 280, "category": 88, "ready": 56, "status": 72, "value": 120}
        for col in cols:
            feat_tree.heading(col, text=headers[col])
            feat_tree.column(col, width=widths[col], anchor="w" if col == "feature" else "center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=feat_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=feat_tree.xview)
        feat_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        feat_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        panel_data: dict[str, Any] = {"rows": []}
        panel_meta: dict[str, Any] = {
            "sample": int(trace[sample_index].get("samples") or sample_index + 1),
            "time": str(trace[sample_index].get("time") or "—"),
        }

        def _render_rows() -> None:
            for item in feat_tree.get_children():
                feat_tree.delete(item)
            query = search_var.get().strip().lower()
            show = show_var.get()
            for i, row in enumerate(panel_data.get("rows") or []):
                name = str(row.get("name") or "")
                if query and query not in name.lower():
                    continue
                status = str(row.get("status") or "")
                if show == "null" and status != "NULL":
                    continue
                if show == "value" and status != "VALUE":
                    continue
                if show == "missing" and status != "MISSING":
                    continue
                ready_mark = "✓" if row.get("ready") else "✗"
                feat_tree.insert(
                    "", "end", iid=f"feat-{i}",
                    values=(
                        name,
                        row.get("category", ""),
                        ready_mark,
                        status,
                        row.get("display", ""),
                    ),
                )

        def _apply_filters(*_args: Any) -> None:
            _render_rows()

        search_var.trace_add("write", _apply_filters)
        show_combo.bind("<<ComboboxSelected>>", _apply_filters)

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack(fill="x")

        def _copy_csv() -> None:
            rows = panel_data.get("rows") or []
            if not rows:
                return
            text = sim_fmt.maturity_features_csv(rows)
            dlg.clipboard_clear()
            dlg.clipboard_append(text)
            messagebox.showinfo("Copy CSV", f"Copied {len(rows)} rows to clipboard.", parent=dlg)

        def _open_feature_detail_from_row() -> None:
            sel = feat_tree.selection()
            if not sel:
                return
            try:
                idx = int(str(sel[0]).replace("feat-", ""))
            except ValueError:
                return
            rows = panel_data.get("rows") or []
            if idx < 0 or idx >= len(rows):
                return
            row = rows[idx]
            self._open_feature_detail(
                str(row.get("name") or ""),
                context={
                    "sample": panel_meta.get("sample"),
                    "time": panel_meta.get("time"),
                    "ready": row.get("ready"),
                    "status": row.get("status"),
                    "display": row.get("display"),
                    "value": row.get("value"),
                },
            )

        feat_tree.bind("<Double-1>", lambda _e: _open_feature_detail_from_row())

        ttk.Button(btn_row, text="Feature Detail…", command=_open_feature_detail_from_row).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Copy CSV", command=_copy_csv).pack(side="left")
        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="left", padx=8)

        def _load_panel() -> None:
            from chain_replay_ml.feature_policy.registry import load_feature_policy_registry
            from chain_replay_ml.feature_policy.warmup_maturity_features import (
                build_sample_feature_panel,
            )

            names = list(result.maturity_feature_names or self._feature_names or [])
            lookup = dict(result.maturity_replay_lookup or {})
            replay_err = result.maturity_replay_error
            step_sec = int(max(result.sampling_interval_sec, 1))
            try:
                reg = load_feature_policy_registry(feature_names=names)
                panel = build_sample_feature_panel(
                    trace=trace,
                    sample_index=sample_index,
                    feature_names=names,
                    registry=reg,
                    replay_lookup=lookup,
                    sampling_interval_sec=float(result.sampling_interval_sec),
                    gap_max_sec=float(result.gap_max_sec),
                    step_sec=step_sec,
                )
            except Exception as exc:
                panel = {"ok": False, "error": str(exc)}

            def _show() -> None:
                if not panel.get("ok"):
                    header.configure(
                        text=f"Sample {sample_index + 1} — failed: {panel.get('error') or replay_err}",
                    )
                    return
                panel_data["rows"] = list(panel.get("rows") or [])
                panel_meta["sample"] = int(panel.get("sample") or panel_meta.get("sample") or 0)
                panel_meta["time"] = str(panel.get("time") or panel_meta.get("time") or "—")
                summary = panel.get("summary") or {}
                hdr = sim_fmt.format_maturity_features_summary(panel)
                if replay_err:
                    hdr += f"  (replay warning: {replay_err})"
                header.configure(text=hdr)
                _render_rows()
                null_n = int(summary.get("null_policy") or 0)
                if null_n:
                    ttk.Label(
                        dlg,
                        text=f"{null_n} features are NULL because policy warm-up is not complete.",
                        foreground="#666",
                        padding=(8, 0),
                    ).pack(anchor="w")

            self.after(0, _show)

        threading.Thread(target=_load_panel, daemon=True).start()

    def _render_calc_debug(self, result: WarmupSimulationResult | None) -> None:
        calc_debug = (result.calc_debug if result else None) or {}
        feature = (result.feature_name if result else None) or self._feature_var.get().strip()

        for item in self._calc_tree.get_children():
            self._calc_tree.delete(item)
        self._calc_rows_index = []
        self._set_readonly_text(self._calc_detail, "Select a row to see formula substitution.")

        if not calc_debug.get("ok"):
            err = calc_debug.get("error") or "Run a simulation to inspect calculations."
            self._calc_summary_var.set(f"{feature} — calculation data unavailable")
            self._set_readonly_text(self._calc_detail, err)
            return

        spec = calc_debug.get("formula_spec") or {}
        formula_line = str(spec.get("formula_doc") or "")
        self._calc_summary_var.set(f"{feature}  ·  {formula_line}")
        columns = spec.get("table_columns") or []
        headers = spec.get("table_headers") or columns
        self._calc_tree["columns"] = columns
        for col, hdr in zip(columns, headers):
            self._calc_tree.heading(col, text=hdr)
            width = 88 if col in ("time", "feature_value") else 72
            if col == "sample":
                width = 56
            self._calc_tree.column(col, width=width, minwidth=48, anchor="center")

        for i, row in enumerate(calc_debug.get("rows") or []):
            display = row.get("display") or {}
            values = [display.get(col, "—") for col in columns]
            iid = f"calc-{i}"
            self._calc_tree.insert("", "end", iid=iid, values=values)
            self._calc_rows_index.append(row)

        children = self._calc_tree.get_children()
        if children:
            self._calc_tree.selection_set(children[0])
            self._calc_tree.focus(children[0])
            self._on_calc_row_select()

    def _on_calc_row_select(self, _event: tk.Event | None = None) -> None:
        sel = self._calc_tree.selection()
        if not sel:
            return
        try:
            idx = int(str(sel[0]).replace("calc-", ""))
        except ValueError:
            return
        if idx < 0 or idx >= len(self._calc_rows_index):
            return
        breakdown = self._calc_rows_index[idx].get("breakdown") or {}
        self._set_readonly_text(self._calc_detail, sim_fmt.format_calc_row_breakdown(breakdown))

    def _clear_progress(self) -> None:
        self._progress.configure(state="normal")
        self._progress.delete("1.0", tk.END)
        self._progress.configure(state="disabled")

    def _append_progress(self, line: str) -> None:
        self._progress.configure(state="normal")
        self._progress.insert(tk.END, line + "\n")
        self._progress.see(tk.END)
        self._progress.configure(state="disabled")
        self.update_idletasks()

    @staticmethod
    def _set_readonly_text(widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_output(self, text: str) -> None:
        self._set_readonly_text(self._output, text)
