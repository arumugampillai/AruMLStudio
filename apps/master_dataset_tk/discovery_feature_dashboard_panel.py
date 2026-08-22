"""Discovery Feature Dashboard & Pipeline Builder Panel (Doc 18).

Interactive workspace for multi-pipeline exploration, filtering mathematical AST features,
curating KEEP and WATCH discoveries across multiple campaigns, and assembling candidate pipelines
anchored by authoritative PL_0001.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.discovery_dashboard.types import (
    CrossPipelineSelectionBasket,
    PipelineCreationRequest,
    PipelineCreationResult,
    SelectedDiscoveryFeatureRef,
)
from chain_replay_ml.discovery_dashboard.service import (
    create_candidate_discovery_pipeline,
    list_discovery_features,
    list_discovery_pipelines,
    validate_cross_pipeline_selection,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    load_store as load_pl_store,
    peek_next_pipeline_identity,
)

COL_MUTED = "#666666"
COL_KEEP = "#2e7d32"
COL_WATCH = "#e65100"
COL_REMOVE = "#c62828"


class DiscoveryFeatureDashboardPanel(ttk.Frame):
    """Discovery Feature Dashboard workspace and Multi-Pipeline Candidate Pipeline Builder."""

    def __init__(
        self,
        parent: tk.Widget,
        data_dir: str,
        *,
        on_pipeline_created: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=8)
        self.data_dir = data_dir
        self.on_pipeline_created = on_pipeline_created

        self.basket = CrossPipelineSelectionBasket()
        self._all_pipelines: list[dict[str, Any]] = []
        self._current_features: list[dict[str, Any]] = []
        self.selected_pipeline_ids: set[str] = set()

        # Context Filter Variables
        self.market_var = tk.StringVar(value="ALL")
        self.interval_var = tk.StringVar(value="ALL")
        self.task_var = tk.StringVar(value="ALL")
        self.horizon_var = tk.StringVar(value="ALL")
        self.regime_var = tk.StringVar(value="ALL")

        # Discovery Scope Variables
        self.selected_gen_var = tk.StringVar(value="ALL")

        # Governance & Common Filters
        self.filter_keep_var = tk.BooleanVar(value=True)
        self.filter_watch_var = tk.BooleanVar(value=True)
        self.filter_remove_var = tk.BooleanVar(value=False)
        self.filter_strategy_var = tk.StringVar(value="ALL")
        self.search_text_var = tk.StringVar(value="")

        self._build_ui()
        self.refresh_pipelines()

    def set_data_dir(self, data_dir: str) -> None:
        """Update the active data directory and refresh pipelines."""
        if data_dir and self.data_dir != data_dir:
            self.data_dir = data_dir
            self.refresh_pipelines()

    def set_chart_dir(self, chart_dir: str) -> None:
        """Update data directory from chart_dir and refresh pipelines."""
        from .build_service import chart_data_dir
        resolved = chart_data_dir(chart_dir) if chart_dir else os.path.join(os.getcwd(), "data")
        self.set_data_dir(resolved)

    def _build_ui(self) -> None:
        """Construct the multi-select Discovery Feature Dashboard UI hierarchy."""
        # Top Header & Context Filter Frame
        top_frame = ttk.LabelFrame(self, text="🧬 Discovery Feature Dashboard — Context & Scope Filters", padding=8)
        top_frame.pack(fill=tk.X, pady=(0, 6))

        c_row1 = ttk.Frame(top_frame)
        c_row1.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(c_row1, text="Market:").pack(side=tk.LEFT, padx=(0, 2))
        self.market_combo = ttk.Combobox(c_row1, textvariable=self.market_var, state="readonly", width=10)
        self.market_combo["values"] = ["ALL", "NIFTY", "BANKNIFTY", "FINNIFTY"]
        self.market_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.market_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pipelines())

        ttk.Label(c_row1, text="Interval:").pack(side=tk.LEFT, padx=(0, 2))
        self.interval_combo = ttk.Combobox(c_row1, textvariable=self.interval_var, state="readonly", width=8)
        self.interval_combo["values"] = ["ALL", "6s", "1s", "1m", "5m"]
        self.interval_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.interval_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pipelines())

        ttk.Label(c_row1, text="Task:").pack(side=tk.LEFT, padx=(0, 2))
        self.task_combo = ttk.Combobox(c_row1, textvariable=self.task_var, state="readonly", width=12)
        self.task_combo["values"] = ["ALL", "Direction", "Volatility", "Magnitude"]
        self.task_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.task_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pipelines())

        ttk.Label(c_row1, text="Horizon:").pack(side=tk.LEFT, padx=(0, 2))
        self.horizon_combo = ttk.Combobox(c_row1, textvariable=self.horizon_var, state="readonly", width=8)
        self.horizon_combo["values"] = ["ALL", "5m", "15m", "30m", "1h"]
        self.horizon_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.horizon_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pipelines())

        ttk.Label(c_row1, text="Regime:").pack(side=tk.LEFT, padx=(0, 2))
        self.regime_combo = ttk.Combobox(c_row1, textvariable=self.regime_var, state="readonly", width=8)
        self.regime_combo["values"] = ["ALL", "R001", "R002", "R003"]
        self.regime_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.regime_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pipelines())

        ttk.Button(c_row1, text="🔄 Refresh All Pipelines", command=self.refresh_pipelines).pack(side=tk.RIGHT, padx=4)

        # Row 2: Pipeline Selection Toolbar
        c_row2 = ttk.Frame(top_frame)
        c_row2.pack(fill=tk.X, pady=(2, 0))

        ttk.Label(c_row2, text="Discovery Pipelines Selection:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(c_row2, text="☑ Select All Pipelines", command=self._select_all_pipelines).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(c_row2, text="☐ Deselect All", command=self._deselect_all_pipelines).pack(side=tk.LEFT, padx=(0, 12))

        self.pipeline_sel_summary_label = ttk.Label(
            c_row2,
            text="Selected Discovery Pipelines: 0 / 0",
            font=("Segoe UI", 9, "bold"),
            foreground="#0d47a1",
        )
        self.pipeline_sel_summary_label.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(c_row2, text="Generation Scope:").pack(side=tk.LEFT, padx=(0, 2))
        self.gen_combo = ttk.Combobox(c_row2, textvariable=self.selected_gen_var, state="readonly", width=10)
        self.gen_combo["values"] = ["ALL"]
        self.gen_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.gen_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_features())

        # Middle Paned Section
        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # Top Pane: Pipelines Overview Grid (Multi-Selectable)
        p_frame = ttk.LabelFrame(paned, text="📋 Available Discovery Pipelines (Matching Context) — Click Row to Toggle Selection", padding=6)
        paned.add(p_frame, weight=1)

        p_cols = (
            "sel", "pipeline_id", "context", "gens", "created",
            "keep", "watch", "remove", "pool", "best_cand", "best_score", "created_at"
        )
        self.pipelines_tree = ttk.Treeview(p_frame, columns=p_cols, show="headings", height=4)
        self.pipelines_tree.heading("sel", text="☑")
        self.pipelines_tree.heading("pipeline_id", text="Pipeline ID")
        self.pipelines_tree.heading("context", text="Context")
        self.pipelines_tree.heading("gens", text="Gens")
        self.pipelines_tree.heading("created", text="Created")
        self.pipelines_tree.heading("keep", text="KEEP")
        self.pipelines_tree.heading("watch", text="WATCH")
        self.pipelines_tree.heading("remove", text="REMOVE")
        self.pipelines_tree.heading("pool", text="Active Pool")
        self.pipelines_tree.heading("best_cand", text="Best Candidate")
        self.pipelines_tree.heading("best_score", text="Best Score")
        self.pipelines_tree.heading("created_at", text="Created (UTC)")

        self.pipelines_tree.column("sel", width=35, anchor=tk.CENTER)
        self.pipelines_tree.column("pipeline_id", width=240)
        self.pipelines_tree.column("context", width=140)
        self.pipelines_tree.column("gens", width=50, anchor=tk.CENTER)
        self.pipelines_tree.column("created", width=60, anchor=tk.CENTER)
        self.pipelines_tree.column("keep", width=55, anchor=tk.CENTER)
        self.pipelines_tree.column("watch", width=60, anchor=tk.CENTER)
        self.pipelines_tree.column("remove", width=65, anchor=tk.CENTER)
        self.pipelines_tree.column("pool", width=75, anchor=tk.CENTER)
        self.pipelines_tree.column("best_cand", width=140)
        self.pipelines_tree.column("best_score", width=75, anchor=tk.E)
        self.pipelines_tree.column("created_at", width=130)

        p_vsb = ttk.Scrollbar(p_frame, orient="vertical", command=self.pipelines_tree.yview)
        self.pipelines_tree.configure(yscrollcommand=p_vsb.set)
        self.pipelines_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        p_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.pipelines_tree.bind("<Button-1>", self._on_pipeline_tree_clicked)

        # Bottom Pane: Combined Multi-Pipeline Feature Browser
        f_frame = ttk.LabelFrame(paned, text="🧪 Discovered Features (DF_*) — Combined Pool across Selected Pipelines", padding=6)
        paned.add(f_frame, weight=3)

        # Selection Aggregate KPI Header
        self.kpi_banner_label = ttk.Label(
            f_frame,
            text="Selected Discovery Pipelines: 0  |  KEEP Features: 0  |  WATCH Features: 0  |  Active Discovery Features: 0",
            font=("Segoe UI", 10, "bold"),
            foreground="#1b5e20",
        )
        self.kpi_banner_label.pack(anchor=tk.W, pady=(0, 6))

        # Filter Toolbar
        f_tb = ttk.Frame(f_frame)
        f_tb.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(f_tb, text="Governance Filter:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        cb_keep = ttk.Checkbutton(f_tb, text="🟢 KEEP", variable=self.filter_keep_var, command=self.refresh_features)
        cb_keep.pack(side=tk.LEFT, padx=(0, 6))
        cb_watch = ttk.Checkbutton(f_tb, text="🟡 WATCH", variable=self.filter_watch_var, command=self.refresh_features)
        cb_watch.pack(side=tk.LEFT, padx=(0, 6))
        cb_rem = ttk.Checkbutton(f_tb, text="🔴 REMOVE (Locked)", variable=self.filter_remove_var, command=self.refresh_features)
        cb_rem.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(f_tb, text="Strategy:").pack(side=tk.LEFT, padx=(0, 2))
        self.strategy_combo = ttk.Combobox(f_tb, textvariable=self.filter_strategy_var, state="readonly", width=14)
        self.strategy_combo["values"] = ["ALL", "RATIO", "INTERACTION", "NONLINEAR", "SPREAD", "COMPOSITE"]
        self.strategy_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.strategy_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_features())

        ttk.Label(f_tb, text="Search:").pack(side=tk.LEFT, padx=(0, 2))
        self.search_entry = ttk.Entry(f_tb, textvariable=self.search_text_var, width=28)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.search_entry.bind("<KeyRelease>", lambda _e: self.refresh_features())

        # Selection Helpers Toolbar
        sel_tb = ttk.Frame(f_frame)
        sel_tb.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(sel_tb, text="☑ Select All KEEP in View", command=self._select_all_keep_in_view).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_tb, text="☑ Select All WATCH in View", command=self._select_all_watch_in_view).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_tb, text="☐ Clear View Selection", command=self._clear_view_selection).pack(side=tk.LEFT, padx=(0, 10))
        self.view_sel_label = ttk.Label(sel_tb, text="Selected in view: 0 features", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED)
        self.view_sel_label.pack(side=tk.LEFT, padx=6)

        # Feature Treeview
        f_tree_frame = ttk.Frame(f_frame)
        f_tree_frame.pack(fill=tk.BOTH, expand=True)

        f_cols = (
            "sel", "verdict", "display_name", "feature_id", "pipeline_id",
            "gen", "strategy", "parents", "formula", "delta_auc",
            "d_ks", "drift_sev", "evidence", "folds", "rationale"
        )
        self.features_tree = ttk.Treeview(f_tree_frame, columns=f_cols, show="headings", height=10)
        self.features_tree.heading("sel", text="☑")
        self.features_tree.heading("verdict", text="Verdict")
        self.features_tree.heading("display_name", text="Feature Name")
        self.features_tree.heading("feature_id", text="DF Feature ID")
        self.features_tree.heading("pipeline_id", text="Source Discovery Pipeline")
        self.features_tree.heading("gen", text="Gen")
        self.features_tree.heading("strategy", text="Strategy")
        self.features_tree.heading("parents", text="Parent Features")
        self.features_tree.heading("formula", text="Mathematical AST Formula")
        self.features_tree.heading("delta_auc", text="ΔAUC")
        self.features_tree.heading("d_ks", text="D_KS")
        self.features_tree.heading("drift_sev", text="Drift")
        self.features_tree.heading("evidence", text="Evidence")
        self.features_tree.heading("folds", text="Folds")
        self.features_tree.heading("rationale", text="Governance Rationale")

        self.features_tree.column("sel", width=35, anchor=tk.CENTER)
        self.features_tree.column("verdict", width=85, anchor=tk.CENTER)
        self.features_tree.column("display_name", width=240)
        self.features_tree.column("feature_id", width=210)
        self.features_tree.column("pipeline_id", width=220)
        self.features_tree.column("gen", width=45, anchor=tk.CENTER)
        self.features_tree.column("strategy", width=95)
        self.features_tree.column("parents", width=160)
        self.features_tree.column("formula", width=260)
        self.features_tree.column("delta_auc", width=75, anchor=tk.E)
        self.features_tree.column("d_ks", width=65, anchor=tk.E)
        self.features_tree.column("drift_sev", width=50, anchor=tk.CENTER)
        self.features_tree.column("evidence", width=75, anchor=tk.E)
        self.features_tree.column("folds", width=60, anchor=tk.CENTER)
        self.features_tree.column("rationale", width=180)

        self.features_tree.tag_configure("tag_keep", foreground=COL_KEEP)
        self.features_tree.tag_configure("tag_watch", foreground=COL_WATCH)
        self.features_tree.tag_configure("tag_remove", foreground=COL_REMOVE)

        f_vsb = ttk.Scrollbar(f_tree_frame, orient="vertical", command=self.features_tree.yview)
        f_hsb = ttk.Scrollbar(f_tree_frame, orient="horizontal", command=self.features_tree.xview)
        self.features_tree.configure(yscrollcommand=f_vsb.set, xscrollcommand=f_hsb.set)

        self.features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        f_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        f_hsb.pack(side=tk.BOTTOM, fill=tk.X)

        self.features_tree.bind("<Button-1>", self._on_feature_tree_clicked)
        self.features_tree.bind("<Double-1>", self._on_feature_tree_double_clicked)

        # Bottom Global Selection Basket Tray
        b_tray = ttk.LabelFrame(self, text="🧺 Global Cross-Pipeline Selection Basket", padding=8)
        b_tray.pack(fill=tk.X, pady=(4, 0))

        self.basket_summary_label = ttk.Label(
            b_tray,
            text="🧺 Global Basket: 0 features selected (0 KEEP · 0 WATCH across 0 Discovery Pipelines)",
            font=("Segoe UI", 10, "bold"),
            foreground="#0d47a1",
        )
        self.basket_summary_label.pack(side=tk.LEFT, padx=(4, 15))

        ttk.Button(b_tray, text="📋 View / Manage Basket", command=self._open_basket_manager_modal).pack(side=tk.LEFT, padx=4)
        ttk.Button(b_tray, text="🗑️ Clear Global Basket", command=self._clear_global_basket).pack(side=tk.LEFT, padx=4)

        self.create_pipeline_btn = ttk.Button(
            b_tray,
            text="[+ CREATE NEW PIPELINE (0 Features) ▶]",
            command=self._open_pipeline_construction_modal,
        )
        self.create_pipeline_btn.pack(side=tk.RIGHT, padx=4)

    def refresh_pipelines(self) -> None:
        """Query and reload Discovery Pipelines matching active context filters."""
        m_val = self.market_var.get()
        ctx_filter = m_val if m_val != "ALL" else None

        self._all_pipelines = list_discovery_pipelines(self.data_dir, ctx_filter)
        available_pids = {p["pipeline_id"] for p in self._all_pipelines}

        # Keep existing selections if still available, or default to all matching pipelines
        if not self.selected_pipeline_ids:
            self.selected_pipeline_ids = set(available_pids)
        else:
            self.selected_pipeline_ids = self.selected_pipeline_ids.intersection(available_pids)

        self._render_pipelines_table()
        self._update_generation_dropdown()
        self.refresh_features()

    def _render_pipelines_table(self) -> None:
        """Render rows in the Discovery Pipelines treeview with checkboxes."""
        for item in self.pipelines_tree.get_children():
            self.pipelines_tree.delete(item)

        for p in self._all_pipelines:
            pid = p["pipeline_id"]
            is_checked = pid in self.selected_pipeline_ids
            sel_icon = "☑" if is_checked else "☐"
            b_score_str = f"{p['best_composite_score']:.2f} pts" if p.get("best_composite_score") is not None else "—"

            self.pipelines_tree.insert(
                "",
                tk.END,
                iid=pid,
                values=(
                    sel_icon,
                    pid,
                    p["context_key"],
                    str(p["current_generation"]),
                    str(p["total_df_features_created"]),
                    str(p["keep_count"]),
                    str(p["watch_count"]),
                    str(p["remove_count"]),
                    str(p["active_discovery_pool"]),
                    p["best_candidate_id"] or "—",
                    b_score_str,
                    str(p["created_at"])[:19].replace("T", " "),
                ),
            )

        total_cnt = len(self._all_pipelines)
        sel_cnt = len(self.selected_pipeline_ids)
        self.pipeline_sel_summary_label.config(
            text=f"Selected Discovery Pipelines: {sel_cnt} / {total_cnt}"
        )

    def _update_generation_dropdown(self) -> None:
        """Update available generation scope based on selected pipelines."""
        max_g = 1
        for p in self._all_pipelines:
            if p["pipeline_id"] in self.selected_pipeline_ids:
                max_g = max(max_g, int(p.get("current_generation") or 1))
        gen_vals = ["ALL"] + [str(g) for g in range(1, max_g + 1)]
        self.gen_combo["values"] = gen_vals
        if self.selected_gen_var.get() not in gen_vals:
            self.selected_gen_var.set("ALL")

    def _select_all_pipelines(self) -> None:
        self.selected_pipeline_ids = {p["pipeline_id"] for p in self._all_pipelines}
        self._render_pipelines_table()
        self._update_generation_dropdown()
        self.refresh_features()

    def _deselect_all_pipelines(self) -> None:
        self.selected_pipeline_ids.clear()
        self._render_pipelines_table()
        self._update_generation_dropdown()
        self.refresh_features()

    def _on_pipeline_tree_clicked(self, event: Any) -> None:
        """Handle clicking on a pipeline row to toggle selection."""
        region = self.pipelines_tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            return
        p_id = self.pipelines_tree.identify_row(event.y)
        if not p_id:
            return

        if p_id in self.selected_pipeline_ids:
            self.selected_pipeline_ids.remove(p_id)
        else:
            self.selected_pipeline_ids.add(p_id)

        self._render_pipelines_table()
        self._update_generation_dropdown()
        self.refresh_features()

    def refresh_features(self) -> None:
        """Query and display features combined across all selected Discovery Pipelines."""
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)

        # Compute pipeline selection summary banner from authoritative records
        sel_pipes = [p for p in self._all_pipelines if p["pipeline_id"] in self.selected_pipeline_ids]
        sum_keep = sum(p["keep_count"] for p in sel_pipes)
        sum_watch = sum(p["watch_count"] for p in sel_pipes)
        sum_active = sum_keep + sum_watch
        self.kpi_banner_label.config(
            text=f"Selected Discovery Pipelines: {len(sel_pipes)}  |  KEEP Features: {sum_keep}  |  WATCH Features: {sum_watch}  |  Active Discovery Features: {sum_active}"
        )

        if not self.selected_pipeline_ids:
            self._current_features = []
            self.view_sel_label.config(text="Selected in view: 0 features (No pipelines selected)")
            self._update_basket_ui()
            return

        # Prepare filters
        gen_val = None
        if self.selected_gen_var.get() != "ALL":
            try:
                gen_val = int(self.selected_gen_var.get())
            except ValueError:
                gen_val = None

        verdicts: list[str] = []
        if self.filter_keep_var.get():
            verdicts.append("KEEP")
        if self.filter_watch_var.get():
            verdicts.append("WATCH")
        if self.filter_remove_var.get():
            verdicts.append("REMOVE")

        strat_val = self.filter_strategy_var.get()
        search_txt = self.search_text_var.get()

        # Query features combined across all selected pipelines with formula deduplication
        self._current_features = list_discovery_features(
            self.data_dir,
            list(self.selected_pipeline_ids),
            generation=gen_val,
            verdicts=verdicts,
            strategy=strat_val,
            search_text=search_txt,
            deduplicate_by_hash=True,
        )

        in_view_sel_count = 0
        for f in self._current_features:
            f_id = f["feature_id"]
            v_norm = f["discovery_verdict"]
            is_in_basket = self.basket.contains(f_id)

            if v_norm == "REMOVE":
                sel_icon = "⛔"
                tag = "tag_remove"
            elif is_in_basket:
                sel_icon = "☑"
                in_view_sel_count += 1
                tag = "tag_keep" if v_norm == "KEEP" else "tag_watch"
            else:
                sel_icon = "☐"
                tag = "tag_keep" if v_norm == "KEEP" else "tag_watch"

            d_auc_str = f"{f['marginal_delta_auc']:+.5f}"
            d_ks_str = f"{f['ks_statistic']:.4f}"
            ev_str = f"{f['evidence_score']:.1f} pts"
            folds_str = f"{f['fold_consistency']*100:.0f}%"

            # Format source pipeline display with co-discovery indicator if present
            pipe_display = f["pipeline_id"]
            if f.get("co_discovery_count", 0) > 0:
                pipe_display += f" (+{f['co_discovery_count']} co-disc)"

            disp_name = f.get("display_name") or f_id

            self.features_tree.insert(
                "",
                tk.END,
                iid=f_id,
                values=(
                    sel_icon,
                    f"🟢 {v_norm}" if v_norm == "KEEP" else (f"🟡 {v_norm}" if v_norm == "WATCH" else f"🔴 {v_norm}"),
                    disp_name,
                    f_id,
                    pipe_display,
                    f"G{f['generation_discovered']}",
                    f["generator_strategy"],
                    ", ".join(f["parent_features"][:2]) + (f" +{len(f['parent_features'])-2}" if len(f['parent_features']) > 2 else ""),
                    f["formula_expression"],
                    d_auc_str,
                    d_ks_str,
                    str(f["drift_severity"]),
                    ev_str,
                    folds_str,
                    f["governance_rationale"] or "—",
                ),
                tags=(tag,),
            )

        self.view_sel_label.config(text=f"Selected in view: {in_view_sel_count} / {len(self._current_features)} unique discovery features")
        self._update_basket_ui()

    def _on_feature_tree_clicked(self, event: Any) -> None:
        """Handle clicking on a feature row or checkbox."""
        region = self.features_tree.identify_region(event.x, event.y)
        if region not in ("cell", "tree"):
            return
        item_id = self.features_tree.identify_row(event.y)
        if not item_id:
            return

        matched = next((f for f in self._current_features if f["feature_id"] == item_id), None)
        if not matched:
            return

        # Rule: REMOVE features can never be selected
        if matched["discovery_verdict"] == "REMOVE":
            messagebox.showwarning(
                "Feature Locked",
                f"Feature '{matched['feature_id']}' holds a REMOVE governance verdict (e.g. severe drift or negative lift) and is permanently locked against pipeline selection."
            )
            return

        # Toggle in basket
        ref = SelectedDiscoveryFeatureRef(
            feature_id=matched["feature_id"],
            display_name=matched.get("display_name") or matched["feature_id"],
            pipeline_id=matched["pipeline_id"],
            research_id=matched["research_id"],
            campaign_id=matched["campaign_id"],
            formula_hash=matched["formula_hash"],
            formula_expression=matched["formula_expression"],
            generator_strategy=matched["generator_strategy"],
            parent_features=matched["parent_features"],
            generation_discovered=matched["generation_discovered"],
            discovery_snapshot_hash=matched["discovery_snapshot_hash"],
            discovery_verdict=matched["discovery_verdict"],
            marginal_delta_auc=matched["marginal_delta_auc"],
            ks_statistic=matched["ks_statistic"],
            drift_severity=matched["drift_severity"],
            evidence_score=matched["evidence_score"],
            fold_consistency=matched["fold_consistency"],
            governance_rationale=matched["governance_rationale"],
            context_key=matched["context_key"],
            co_discovered_pipelines=matched.get("co_discovered_pipelines", []),
        )
        self.basket.toggle(ref)
        self.refresh_features()

    def _on_feature_tree_double_clicked(self, _event: Any) -> None:
        sel = self.features_tree.selection()
        if not sel:
            return
        f_id = sel[0]
        matched = next((f for f in self._current_features if f["feature_id"] == f_id), None)
        if not matched:
            return

        top = tk.Toplevel(self)
        top.title(f"Discovered Feature AST Detail — {f_id}")
        top.geometry("680x480")
        top.transient(self)

        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text=f"Feature: {f_id}", font=("Segoe UI", 11, "bold"), foreground="#0d47a1").pack(anchor=tk.W, pady=(0, 8))

        co_dps = matched.get("co_discovered_pipelines", [])
        co_str = ", ".join(co_dps) if co_dps else "None (Single discovery)"

        detail_pairs = [
            ("Mathematical Formula", matched["formula_expression"]),
            ("Formula Hash (16-char MD5)", matched["formula_hash"]),
            ("Primary Discovery Pipeline", matched["pipeline_id"]),
            ("Co-Discovered Pipelines", co_str),
            ("Research ID", matched["research_id"]),
            ("Campaign ID", matched["campaign_id"]),
            ("Context Key", matched["context_key"]),
            ("Strategy", matched["generator_strategy"]),
            ("Generation Discovered", f"Generation {matched['generation_discovered']}"),
            ("Parent Features", ", ".join(matched["parent_features"])),
            ("Governance Verdict", matched["discovery_verdict"]),
            ("Marginal ΔAUC", f"{matched['marginal_delta_auc']:+.6f}"),
            ("KS Statistic (Drift)", f"{matched['ks_statistic']:.4f} (Severity {matched['drift_severity']})"),
            ("Evidence Score", f"{matched['evidence_score']:.2f} pts"),
            ("Fold Consistency", f"{matched['fold_consistency']*100:.1f}%"),
            ("Governance Rationale", matched["governance_rationale"] or "—"),
        ]

        grid = ttk.Frame(frm)
        grid.pack(fill=tk.BOTH, expand=True, pady=4)
        for idx, (k, v) in enumerate(detail_pairs):
            ttk.Label(grid, text=f"{k}:", font=("Segoe UI", 9, "bold")).grid(row=idx, column=0, sticky=tk.W, pady=2, padx=(0, 8))
            ttk.Label(grid, text=str(v), font=("Segoe UI", 9)).grid(row=idx, column=1, sticky=tk.W, pady=2)

        ttk.Button(frm, text="Close", command=top.destroy).pack(side=tk.RIGHT, pady=(8, 0))

    def _select_all_keep_in_view(self) -> None:
        """Add all KEEP features currently visible in view to the basket."""
        count_added = 0
        for f in self._current_features:
            if f["discovery_verdict"] == "KEEP":
                ref = SelectedDiscoveryFeatureRef(
                    feature_id=f["feature_id"],
                    display_name=f.get("display_name") or f["feature_id"],
                    pipeline_id=f["pipeline_id"],
                    research_id=f["research_id"],
                    campaign_id=f["campaign_id"],
                    formula_hash=f["formula_hash"],
                    formula_expression=f["formula_expression"],
                    generator_strategy=f["generator_strategy"],
                    parent_features=f["parent_features"],
                    generation_discovered=f["generation_discovered"],
                    discovery_snapshot_hash=f["discovery_snapshot_hash"],
                    discovery_verdict=f["discovery_verdict"],
                    marginal_delta_auc=f["marginal_delta_auc"],
                    ks_statistic=f["ks_statistic"],
                    drift_severity=f["drift_severity"],
                    evidence_score=f["evidence_score"],
                    fold_consistency=f["fold_consistency"],
                    governance_rationale=f["governance_rationale"],
                    context_key=f["context_key"],
                    co_discovered_pipelines=f.get("co_discovered_pipelines", []),
                )
                if self.basket.add(ref):
                    count_added += 1
        self.refresh_features()

    def _select_all_watch_in_view(self) -> None:
        """Add all WATCH features currently visible in view to the basket."""
        count_added = 0
        for f in self._current_features:
            if f["discovery_verdict"] == "WATCH":
                ref = SelectedDiscoveryFeatureRef(
                    feature_id=f["feature_id"],
                    display_name=f.get("display_name") or f["feature_id"],
                    pipeline_id=f["pipeline_id"],
                    research_id=f["research_id"],
                    campaign_id=f["campaign_id"],
                    formula_hash=f["formula_hash"],
                    formula_expression=f["formula_expression"],
                    generator_strategy=f["generator_strategy"],
                    parent_features=f["parent_features"],
                    generation_discovered=f["generation_discovered"],
                    discovery_snapshot_hash=f["discovery_snapshot_hash"],
                    discovery_verdict=f["discovery_verdict"],
                    marginal_delta_auc=f["marginal_delta_auc"],
                    ks_statistic=f["ks_statistic"],
                    drift_severity=f["drift_severity"],
                    evidence_score=f["evidence_score"],
                    fold_consistency=f["fold_consistency"],
                    governance_rationale=f["governance_rationale"],
                    context_key=f["context_key"],
                    co_discovered_pipelines=f.get("co_discovered_pipelines", []),
                )
                if self.basket.add(ref):
                    count_added += 1
        self.refresh_features()

    def _clear_view_selection(self) -> None:
        """Remove features in the current view from the global basket."""
        for f in self._current_features:
            self.basket.remove(f["feature_id"])
        self.refresh_features()

    def _clear_global_basket(self) -> None:
        """Clear all items in the global cross-pipeline basket."""
        if self.basket.total_count == 0:
            return
        if messagebox.askyesno("Clear Basket", "Are you sure you want to clear all selected features from the Global Selection Basket?"):
            self.basket.clear()
            self.refresh_features()

    def _update_basket_ui(self) -> None:
        """Update the bottom basket summary label and create button state."""
        tot = self.basket.total_count
        p_cnt = self.basket.pipeline_count
        k_cnt = self.basket.keep_count
        w_cnt = self.basket.watch_count

        self.basket_summary_label.config(
            text=f"🧺 Global Basket: {tot} features selected ({k_cnt} KEEP · {w_cnt} WATCH across {p_cnt} Discovery Pipelines)"
        )
        self.create_pipeline_btn.config(
            text=f"[+ CREATE NEW PIPELINE ({tot} Selected Features) ▶]",
            state=tk.NORMAL if tot > 0 else tk.DISABLED,
        )

    def _open_basket_manager_modal(self) -> None:
        """Open modal dialog to view and manage cross-pipeline basket items."""
        items = self.basket.get_all()
        if not items:
            messagebox.showinfo("Selection Basket", "The Global Selection Basket is currently empty.")
            return

        top = tk.Toplevel(self)
        top.title("Global Cross-Pipeline Selection Basket")
        top.geometry("920x520")
        top.transient(self)

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text=f"🧺 Selected Features Basket ({len(items)} features across {self.basket.pipeline_count} Discovery Pipelines)",
            font=("Segoe UI", 11, "bold"),
            foreground="#0d47a1",
        ).pack(anchor=tk.W, pady=(0, 6))

        t_frame = ttk.Frame(frm)
        t_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        cols = ("verdict", "feature_id", "pipeline_id", "strategy", "formula", "delta_auc", "evidence")
        b_tree = ttk.Treeview(t_frame, columns=cols, show="headings", height=14)
        b_tree.heading("verdict", text="Verdict")
        b_tree.heading("feature_id", text="Feature ID")
        b_tree.heading("pipeline_id", text="Source Discovery Pipeline")
        b_tree.heading("strategy", text="Strategy")
        b_tree.heading("formula", text="Mathematical Formula")
        b_tree.heading("delta_auc", text="ΔAUC")
        b_tree.heading("evidence", text="Evidence")

        b_tree.column("verdict", width=80, anchor=tk.CENTER)
        b_tree.column("feature_id", width=200)
        b_tree.column("pipeline_id", width=230)
        b_tree.column("strategy", width=90)
        b_tree.column("formula", width=230)
        b_tree.column("delta_auc", width=75, anchor=tk.E)
        b_tree.column("evidence", width=75, anchor=tk.E)

        for it in items:
            b_tree.insert(
                "",
                tk.END,
                iid=it.feature_id,
                values=(
                    f"🟢 KEEP" if it.discovery_verdict == "KEEP" else f"🟡 WATCH",
                    it.feature_id,
                    it.pipeline_id,
                    it.generator_strategy,
                    it.formula_expression,
                    f"{it.marginal_delta_auc:+.5f}",
                    f"{it.evidence_score:.1f} pts",
                ),
            )

        b_vsb = ttk.Scrollbar(t_frame, orient="vertical", command=b_tree.yview)
        b_tree.configure(yscrollcommand=b_vsb.set)
        b_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        b_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _remove_selected_from_basket():
            sel = b_tree.selection()
            if not sel:
                return
            for fid in sel:
                self.basket.remove(fid)
                b_tree.delete(fid)
            self.refresh_features()
            if self.basket.total_count == 0:
                top.destroy()

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Remove Selected Item", command=_remove_selected_from_basket).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Close", command=top.destroy).pack(side=tk.RIGHT)

    def _open_pipeline_construction_modal(self) -> None:
        """Open the candidate pipeline construction dialog using the Global Selection Basket."""
        if self.basket.total_count == 0:
            messagebox.showinfo("Empty Selection", "Please select at least one KEEP or WATCH feature to create a candidate pipeline.")
            return

        # Load authoritative PL_0001 features directly from pipeline_registry_store.json
        pr_doc = load_pl_store(self.data_dir)
        pl_0001 = pr_doc.get("pipelines", {}).get("PL_0001", {})
        base_feats = list(pl_0001.get("candidate_features") or pl_0001.get("feature_names") or [])
        base_count = len(base_feats)

        # Validate selection
        is_valid, msg, dedup_items, co_disc = validate_cross_pipeline_selection(self.basket)
        if not is_valid:
            messagebox.showerror("Validation Error", msg)
            return

        # Next suggested pipeline identity
        suggested_pid, suggested_name = peek_next_pipeline_identity(pr_doc)
        ctx_k = dedup_items[0].context_key if dedup_items and dedup_items[0].context_key else "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001"

        top = tk.Toplevel(self)
        top.title("🛠️ Create New Candidate Discovery Pipeline")
        top.geometry("740x620")
        top.transient(self)

        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text="🛠️ Discovery Feature Pipeline Construction Engine",
            font=("Segoe UI", 12, "bold"),
            foreground="#0d47a1",
        ).pack(anchor=tk.W, pady=(0, 6))

        # Pipeline Accounting Card
        acc_frame = ttk.LabelFrame(frm, text="Discovery Feature Population Accounting", padding=8)
        acc_frame.pack(fill=tk.X, pady=(0, 10))

        tot_cand_features = len(dedup_items)
        ttk.Label(acc_frame, text=f"• Pipeline Category / Type:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=1)
        ttk.Label(acc_frame, text=f"DISCOVERY_EXPERIMENTAL (Pure Discovery Features)").grid(row=0, column=1, sticky=tk.W, pady=1)

        ttk.Label(acc_frame, text=f"• Discovered Features in Global Basket:", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=1)
        ttk.Label(acc_frame, text=f"{len(dedup_items)} Unique Features ({self.basket.keep_count} KEEP · {self.basket.watch_count} WATCH)").grid(row=1, column=1, sticky=tk.W, pady=1)

        if co_disc:
            ttk.Label(acc_frame, text=f"• Formula Deduplications:", font=("Segoe UI", 9, "bold"), foreground="#e65100").grid(row=2, column=0, sticky=tk.W, pady=1)
            ttk.Label(acc_frame, text=f"{len(co_disc)} duplicate formulas unified across pipelines", foreground="#e65100").grid(row=2, column=1, sticky=tk.W, pady=1)

        ttk.Label(acc_frame, text=f"• Base / Registry Features Injected:", font=("Segoe UI", 9, "bold")).grid(row=3, column=0, sticky=tk.W, pady=1)
        ttk.Label(acc_frame, text=f"0 (Zero Base or Registry Features)").grid(row=3, column=1, sticky=tk.W, pady=1)

        ttk.Label(acc_frame, text=f"• Resulting Pipeline Universe:", font=("Segoe UI", 9, "bold"), foreground=COL_KEEP).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(acc_frame, text=f"{tot_cand_features} Total Discovered Features", font=("Segoe UI", 9, "bold"), foreground=COL_KEEP).grid(row=4, column=1, sticky=tk.W, pady=2)

        # Source Discovery Pipelines Breakdown
        src_frame = ttk.LabelFrame(frm, text="Contributing Discovery Pipelines", padding=8)
        src_frame.pack(fill=tk.X, pady=(0, 10))
        grouped = self.basket.get_by_pipeline()
        for idx, (p_id, p_items) in enumerate(grouped.items()):
            k_c = sum(1 for it in p_items if it.discovery_verdict == "KEEP")
            w_c = sum(1 for it in p_items if it.discovery_verdict == "WATCH")
            ttk.Label(src_frame, text=f"{idx+1}. {p_id} ── {len(p_items)} Features ({k_c} KEEP, {w_c} WATCH)").pack(anchor=tk.W, pady=1)

        # Configuration Form
        form_frame = ttk.LabelFrame(frm, text="Target Pipeline Configuration", padding=8)
        form_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(form_frame, text="Pipeline ID:").grid(row=0, column=0, sticky=tk.W, pady=3)
        pid_var = tk.StringVar(value=suggested_pid)
        ttk.Entry(form_frame, textvariable=pid_var, width=20, state="readonly").grid(row=0, column=1, sticky=tk.W, pady=3)

        ttk.Label(form_frame, text="Pipeline Display Name:").grid(row=1, column=0, sticky=tk.W, pady=3)
        pname_var = tk.StringVar(value=f"{suggested_name} — Discovery Synthesis V1")
        ttk.Entry(form_frame, textvariable=pname_var, width=45).grid(row=1, column=1, sticky=tk.W, pady=3)

        ttk.Label(form_frame, text="Context Key:").grid(row=2, column=0, sticky=tk.W, pady=3)
        pctx_var = tk.StringVar(value=ctx_k)
        ttk.Entry(form_frame, textvariable=pctx_var, width=45).grid(row=2, column=1, sticky=tk.W, pady=3)

        ttk.Label(form_frame, text="Description:").grid(row=3, column=0, sticky=tk.NW, pady=3)
        desc_text = tk.Text(form_frame, height=3, width=45, font=("Segoe UI", 9))
        desc_text.insert("1.0", f"Cross-discovery candidate pipeline assembling {len(dedup_items)} high-conviction features across {len(grouped)} discovery runs.")
        desc_text.grid(row=3, column=1, sticky=tk.W, pady=3)

        def _do_create_pipeline():
            req = PipelineCreationRequest(
                name=pname_var.get().strip() or suggested_name,
                description=desc_text.get("1.0", tk.END).strip(),
                context_key=pctx_var.get().strip() or ctx_k,
                pipeline_id=pid_var.get().strip() or None,
            )
            res = create_candidate_discovery_pipeline(self.data_dir, req, self.basket)
            if res.success:
                messagebox.showinfo("Pipeline Created", res.message)
                if self.on_pipeline_created:
                    self.on_pipeline_created(res.pipeline_id)
                top.destroy()
            else:
                messagebox.showerror("Error Creating Pipeline", res.message)

        # Action Buttons
        b_row = ttk.Frame(frm)
        b_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(b_row, text="Create Candidate Pipeline ▶", command=_do_create_pipeline).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(b_row, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=4)
