"""Feature Studio Recommendation & Evidence Viewer Dialog.

Dataset-context-aware recommendation inspector for the three feature populations:
1. Feature Registry (Health, score, accumulated runs, alert state)
2. Base Pipeline (Evidence score, priority ranking, health state)
3. Selected Experimental (Lineage streaks, PROMOTION_CANDIDATE, BLOCKED gate)
4. Raw Evidence Log (Immutable validation history)
5. Policy Settings (Configurable policy thresholds, preview impact, history/rollback)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.production_validation.api import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    build_dataset_context,
    get_population_recommendations,
    ignore_recommendation,
    list_policy_history,
    load_recommendation_policy,
    preview_policy_impact,
    rebuild_all_projections,
    restore_policy_version,
    save_recommendation_policy,
    validate_recommendation_policy,
)
from chain_replay_ml.production_validation.evidence_store import (
    get_connection,
)
from .build_service import chart_data_dir


class FeatureRecommendationViewerDialog(tk.Toplevel):
    """Dataset-context-aware recommendation and evidence viewer."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        initial_market: str = "NIFTY",
        initial_interval_sec: int = 3,
        initial_sliding_window: str = "standard",
        initial_feature_project_id: str = "all",
        on_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Feature Recommendation Evidence Studio")
        self.transient(master.winfo_toplevel())
        self.geometry("1060x680")
        self.minsize(850, 520)

        self._chart_dir = chart_dir
        self._data_dir = chart_data_dir(chart_dir)
        self._on_changed = on_changed

        # Context filters
        self._market_var = tk.StringVar(value=initial_market.upper())
        self._interval_var = tk.StringVar(value=str(initial_interval_sec))
        self._window_var = tk.StringVar(value=initial_sliding_window.lower())
        self._fpid_var = tk.StringVar(value=initial_feature_project_id.lower())
        self._include_legacy_var = tk.BooleanVar(value=False)
        self._context_id_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="")

        # Policy form variables
        self._policy_scope_mode = tk.StringVar(value="context")  # 'context' or 'global'
        self._active_policy_info_var = tk.StringVar(value="")
        self._policy_validation_err_var = tk.StringVar(value="")

        # 1. Scoring
        self._weight_keep_var = tk.StringVar(value="25.0")
        self._weight_watch_var = tk.StringVar(value="-10.0")
        self._weight_remove_var = tk.StringVar(value="-35.0")
        self._bonus_keep_var = tk.StringVar(value="15.0")
        self._penalty_remove_var = tk.StringVar(value="-25.0")
        self._min_score_var = tk.StringVar(value="-100.0")
        self._max_score_var = tk.StringVar(value="100.0")

        # 2. Experimental Lifecycle
        self._exp_promo_keep_streak_var = tk.StringVar(value="3")
        self._exp_promo_min_models_var = tk.StringVar(value="2")
        self._exp_promo_min_score_var = tk.StringVar(value="75.0")
        self._exp_block_consec_remove_var = tk.StringVar(value="2")
        self._exp_block_total_remove_var = tk.StringVar(value="4")

        # 3. Base & Registry
        self._base_neg_alert_score_var = tk.StringVar(value="-40.0")
        self._base_strong_keep_score_var = tk.StringVar(value="50.0")
        self._reg_audit_remove_runs_var = tk.StringVar(value="3")
        self._reg_audit_unique_models_var = tk.StringVar(value="2")

        # Idempotently ensure legacy migration has run before querying projections
        try:
            from chain_replay_ml.production_validation.recommendation_migration import (
                migrate_legacy_recommendation_json,
            )
            migrate_legacy_recommendation_json(self._data_dir)
        except Exception:
            pass

        self._build_ui()
        self._update_context_id()
        self._reload_all_tabs()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 1. Top Context Filter Toolbar
        ctx_box = ttk.LabelFrame(main_frame, text="Dataset Context Filter", padding=(8, 6))
        ctx_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(ctx_box, text="Market:").pack(side="left", padx=(0, 4))
        m_cb = ttk.Combobox(
            ctx_box,
            textvariable=self._market_var,
            values=["NIFTY", "SENSEX", "ALL"],
            width=8,
            state="readonly",
        )
        m_cb.pack(side="left", padx=(0, 10))
        m_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed())

        ttk.Label(ctx_box, text="Interval:").pack(side="left", padx=(0, 4))
        int_cb = ttk.Combobox(
            ctx_box,
            textvariable=self._interval_var,
            values=["1", "3", "6", "15", "ALL"],
            width=6,
            state="readonly",
        )
        int_cb.pack(side="left", padx=(0, 10))
        int_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed())

        ttk.Label(ctx_box, text="Sliding Window:").pack(side="left", padx=(0, 4))
        win_cb = ttk.Combobox(
            ctx_box,
            textvariable=self._window_var,
            values=["standard", "atm_15", "atm_25", "ALL"],
            width=10,
            state="readonly",
        )
        win_cb.pack(side="left", padx=(0, 10))
        win_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed())

        ttk.Label(ctx_box, text="Project ID:").pack(side="left", padx=(0, 4))
        fpid_cb = ttk.Combobox(
            ctx_box,
            textvariable=self._fpid_var,
            values=["all", "ALL"],
            width=6,
            state="readonly",
        )
        fpid_cb.pack(side="left", padx=(0, 10))
        fpid_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed())

        ttk.Checkbutton(
            ctx_box,
            text="Include Legacy Unknown",
            variable=self._include_legacy_var,
            command=self._on_filter_changed,
        ).pack(side="left", padx=(10, 10))

        ttk.Label(ctx_box, text="Context ID:").pack(side="left", padx=(10, 4))
        ttk.Label(ctx_box, textvariable=self._context_id_var, font=("TkDefaultFont", 9, "bold")).pack(
            side="left"
        )

        # 2. Population Notebook Tabs
        self._notebook = ttk.Notebook(main_frame)
        self._notebook.grid(row=1, column=0, sticky="nsew")

        # Tab 1: Feature Registry
        self._tab_registry = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_registry, text="1. Feature Registry")
        self._build_registry_tab(self._tab_registry)

        # Tab 2: Base Pipeline
        self._tab_base = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_base, text="2. Base Pipeline")
        self._build_base_tab(self._tab_base)

        # Tab 3: Selected Experimental
        self._tab_exp = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_exp, text="3. Selected Experimental")
        self._build_exp_tab(self._tab_exp)

        # Tab 4: Raw Evidence Log
        self._tab_raw = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_raw, text="4. Raw Evidence Log")
        self._build_raw_tab(self._tab_raw)

        # Tab 5: Policy Settings
        self._tab_policy = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_policy, text="5. Policy Settings")
        self._build_policy_tab(self._tab_policy)

        # 3. Status Bar & Bottom Actions
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        bottom_frame.columnconfigure(0, weight=1)

        ttk.Label(bottom_frame, textvariable=self._status_var, foreground="#555").grid(
            row=0, column=0, sticky="w"
        )

        btn_box = ttk.Frame(bottom_frame)
        btn_box.grid(row=0, column=1, sticky="e")

        ttk.Button(
            btn_box,
            text="🚀 Export Training Candidates to Model Builder",
            command=self._on_export_training_candidates,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_box,
            text="Rebuild Projections",
            command=self._on_rebuild_projections,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_box,
            text="Refresh",
            command=self._reload_all_tabs,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(btn_box, text="Close", command=self.destroy).pack(side="left")

    def _build_registry_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Feature Registry evidence from Production Validation. Status 'alert' highlights features with repeated degradation. Registry features are never auto-blocked.",
            foreground="#555",
            wraplength=960,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tf = ttk.Frame(parent)
        tf.grid(row=1, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = ("feature", "status", "score", "confidence", "stability", "generalization", "consensus", "freshness", "keep", "watch", "remove", "models", "last_validated")
        self._reg_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._reg_tree.heading("feature", text="Feature Name")
        self._reg_tree.heading("status", text="Health Status")
        self._reg_tree.heading("score", text="Evidence Score")
        self._reg_tree.heading("confidence", text="Confidence")
        self._reg_tree.heading("stability", text="Stability")
        self._reg_tree.heading("generalization", text="Generalization")
        self._reg_tree.heading("consensus", text="Model Consensus")
        self._reg_tree.heading("freshness", text="Freshness")
        self._reg_tree.heading("keep", text="KEEP")
        self._reg_tree.heading("watch", text="WATCH")
        self._reg_tree.heading("remove", text="REMOVE")
        self._reg_tree.heading("models", text="Unique Models")
        self._reg_tree.heading("last_validated", text="Last Validated")

        self._reg_tree.column("feature", width=180, anchor="w")
        self._reg_tree.column("status", width=85, anchor="center")
        self._reg_tree.column("score", width=85, anchor="e")
        self._reg_tree.column("confidence", width=80, anchor="center")
        self._reg_tree.column("stability", width=115, anchor="center")
        self._reg_tree.column("generalization", width=125, anchor="center")
        self._reg_tree.column("consensus", width=145, anchor="w")
        self._reg_tree.column("freshness", width=110, anchor="center")
        self._reg_tree.column("keep", width=50, anchor="e")
        self._reg_tree.column("watch", width=50, anchor="e")
        self._reg_tree.column("remove", width=50, anchor="e")
        self._reg_tree.column("models", width=70, anchor="e")
        self._reg_tree.column("last_validated", width=120, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._reg_tree.yview)
        self._reg_tree.configure(yscrollcommand=sb.set)
        self._reg_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_base_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Base Pipeline accepted features. Priority rank ordered by evidence score. Strong KEEP features rank higher. Base features are never auto-blocked or deleted.",
            foreground="#555",
            wraplength=960,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tf = ttk.Frame(parent)
        tf.grid(row=1, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = ("rank", "feature", "status", "score", "confidence", "adj_score", "advisory_rank", "stability", "generalization", "consensus", "freshness", "models", "last_validated")
        self._base_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._base_tree.heading("rank", text="Priority Rank")
        self._base_tree.heading("feature", text="Feature Name")
        self._base_tree.heading("status", text="Health Status")
        self._base_tree.heading("score", text="Evidence Score")
        self._base_tree.heading("confidence", text="Confidence")
        self._base_tree.heading("adj_score", text="Adj Score")
        self._base_tree.heading("advisory_rank", text="Advisory Rank")
        self._base_tree.heading("stability", text="Stability")
        self._base_tree.heading("generalization", text="Generalization")
        self._base_tree.heading("consensus", text="Model Consensus")
        self._base_tree.heading("freshness", text="Freshness")
        self._base_tree.heading("models", text="Unique Models")
        self._base_tree.heading("last_validated", text="Last Validated")

        self._base_tree.column("rank", width=65, anchor="center")
        self._base_tree.column("feature", width=170, anchor="w")
        self._base_tree.column("status", width=85, anchor="center")
        self._base_tree.column("score", width=85, anchor="e")
        self._base_tree.column("confidence", width=80, anchor="center")
        self._base_tree.column("adj_score", width=75, anchor="e")
        self._base_tree.column("advisory_rank", width=85, anchor="center")
        self._base_tree.column("stability", width=115, anchor="center")
        self._base_tree.column("generalization", width=125, anchor="center")
        self._base_tree.column("consensus", width=145, anchor="w")
        self._base_tree.column("freshness", width=110, anchor="center")
        self._base_tree.column("models", width=70, anchor="e")
        self._base_tree.column("last_validated", width=120, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._base_tree.yview)
        self._base_tree.configure(yscrollcommand=sb.set)
        self._base_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_exp_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Selected Experimental features tracked by exact pipeline & snapshot lineage. Satisfying consecutive KEEPs on unique models produces PROMOTION_CANDIDATE eligibility. Repeated REMOVEs trigger context-level BLOCKED gate.",
            foreground="#555",
            wraplength=960,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tf = ttk.Frame(parent)
        tf.grid(row=1, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = (
            "pipeline_id",
            "snapshot_id",
            "feature",
            "lineage_status",
            "context_gate",
            "score",
            "confidence",
            "stability",
            "generalization",
            "consensus",
            "freshness",
            "streak_keep",
            "streak_remove",
            "models",
            "last_validated",
        )
        self._exp_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._exp_tree.heading("pipeline_id", text="Pipeline ID")
        self._exp_tree.heading("snapshot_id", text="Snapshot ID")
        self._exp_tree.heading("feature", text="Feature Name")
        self._exp_tree.heading("lineage_status", text="Lineage Status")
        self._exp_tree.heading("context_gate", text="Context Gate")
        self._exp_tree.heading("score", text="Lineage Score")
        self._exp_tree.heading("confidence", text="Confidence")
        self._exp_tree.heading("stability", text="Stability")
        self._exp_tree.heading("generalization", text="Generalization")
        self._exp_tree.heading("consensus", text="Model Consensus")
        self._exp_tree.heading("freshness", text="Freshness")
        self._exp_tree.heading("streak_keep", text="Streak KEEP")
        self._exp_tree.heading("streak_remove", text="Streak REMOVE")
        self._exp_tree.heading("models", text="Unique Models")
        self._exp_tree.heading("last_validated", text="Last Validated")

        self._exp_tree.column("pipeline_id", width=80, anchor="center")
        self._exp_tree.column("snapshot_id", width=85, anchor="w")
        self._exp_tree.column("feature", width=150, anchor="w")
        self._exp_tree.column("lineage_status", width=120, anchor="center")
        self._exp_tree.column("context_gate", width=85, anchor="center")
        self._exp_tree.column("score", width=75, anchor="e")
        self._exp_tree.column("confidence", width=75, anchor="center")
        self._exp_tree.column("stability", width=115, anchor="center")
        self._exp_tree.column("generalization", width=125, anchor="center")
        self._exp_tree.column("consensus", width=140, anchor="w")
        self._exp_tree.column("freshness", width=105, anchor="center")
        self._exp_tree.column("streak_keep", width=70, anchor="e")
        self._exp_tree.column("streak_remove", width=70, anchor="e")
        self._exp_tree.column("models", width=65, anchor="e")
        self._exp_tree.column("last_validated", width=115, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._exp_tree.yview)
        self._exp_tree.configure(yscrollcommand=sb.set)
        self._exp_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_raw_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Raw append-only recommendation evidence log. Sole authoritative source of truth. Never modified by policy changes.",
            foreground="#555",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tf = ttk.Frame(parent)
        tf.grid(row=1, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = ("timestamp", "context", "source", "feature", "recommendation", "model", "run_id", "details")
        self._raw_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._raw_tree.heading("timestamp", text="Timestamp")
        self._raw_tree.heading("context", text="Context ID")
        self._raw_tree.heading("source", text="Source")
        self._raw_tree.heading("feature", text="Feature Name")
        self._raw_tree.heading("recommendation", text="Recommendation")
        self._raw_tree.heading("model", text="Model Name")
        self._raw_tree.heading("run_id", text="Validation Run ID")
        self._raw_tree.heading("details", text="Evidence Details")

        self._raw_tree.column("timestamp", width=140, anchor="w")
        self._raw_tree.column("context", width=110, anchor="w")
        self._raw_tree.column("source", width=90, anchor="center")
        self._raw_tree.column("feature", width=180, anchor="w")
        self._raw_tree.column("recommendation", width=110, anchor="center")
        self._raw_tree.column("model", width=140, anchor="w")
        self._raw_tree.column("run_id", width=110, anchor="w")
        self._raw_tree.column("details", width=180, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._raw_tree.yview)
        self._raw_tree.configure(yscrollcommand=sb.set)
        self._raw_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_policy_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Scrollable container for policy settings
        canvas = tk.Canvas(parent, highlightthickness=0)
        v_sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_sb.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_sb.grid(row=0, column=1, sticky="ns")

        form = ttk.Frame(canvas, padding=8)
        canvas_window = canvas.create_window((0, 0), window=form, anchor="nw")

        def _on_frame_config(event: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_config(event: Any) -> None:
            canvas.itemconfig(canvas_window, width=event.width)

        form.bind("<Configure>", _on_frame_config)
        canvas.bind("<Configure>", _on_canvas_config)

        # 1. Header Frame: Scope & Active Policy Info
        hdr_frame = ttk.LabelFrame(form, text="Active Policy Status & Scope", padding=(10, 6))
        hdr_frame.pack(fill="x", pady=(0, 8))
        hdr_frame.columnconfigure(1, weight=1)

        ttk.Label(hdr_frame, textvariable=self._active_policy_info_var, font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        scope_box = ttk.Frame(hdr_frame)
        scope_box.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(scope_box, text="Policy Scope:").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(
            scope_box,
            text="Context-Specific Override",
            variable=self._policy_scope_mode,
            value="context",
            command=self._on_policy_scope_toggle,
        ).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(
            scope_box,
            text="Global Default Policy",
            variable=self._policy_scope_mode,
            value="global",
            command=self._on_policy_scope_toggle,
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            hdr_frame,
            text="Policy History / Rollback",
            command=self._on_open_policy_history,
        ).grid(row=1, column=2, sticky="e")

        # 2. Section 1: Experimental Promotion Thresholds
        exp_promo_box = ttk.LabelFrame(form, text="1. Experimental Promotion Thresholds (PROMOTION_CANDIDATE)", padding=(10, 6))
        exp_promo_box.pack(fill="x", pady=(0, 8))

        row1 = ttk.Frame(exp_promo_box)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Min Consecutive KEEP Streak:", width=30, anchor="w").pack(side="left")
        ttk.Entry(row1, textvariable=self._exp_promo_keep_streak_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(row1, text="Min Unique Models:", width=22, anchor="w").pack(side="left")
        ttk.Entry(row1, textvariable=self._exp_promo_min_models_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(row1, text="Min Evidence Score:", width=20, anchor="w").pack(side="left")
        ttk.Entry(row1, textvariable=self._exp_promo_min_score_var, width=8).pack(side="left")

        # 3. Section 2: Experimental Blocking Thresholds
        exp_block_box = ttk.LabelFrame(form, text="2. Experimental Blocking Gates (Pre-Training Gate)", padding=(10, 6))
        exp_block_box.pack(fill="x", pady=(0, 8))

        row2 = ttk.Frame(exp_block_box)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Consecutive REMOVE Threshold:", width=30, anchor="w").pack(side="left")
        ttk.Entry(row2, textvariable=self._exp_block_consec_remove_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(row2, text="Total REMOVE Threshold:", width=22, anchor="w").pack(side="left")
        ttk.Entry(row2, textvariable=self._exp_block_total_remove_var, width=8).pack(side="left")

        # 4. Section 3: Evidence Score Weights & Bounds
        score_box = ttk.LabelFrame(form, text="3. Evidence Score Weights & Mathematical Bounds", padding=(10, 6))
        score_box.pack(fill="x", pady=(0, 8))

        r3a = ttk.Frame(score_box)
        r3a.pack(fill="x", pady=2)
        ttk.Label(r3a, text="KEEP Weight (per model):", width=30, anchor="w").pack(side="left")
        ttk.Entry(r3a, textvariable=self._weight_keep_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r3a, text="WATCH Weight (per model):", width=24, anchor="w").pack(side="left")
        ttk.Entry(r3a, textvariable=self._weight_watch_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r3a, text="REMOVE Weight (per model):", width=24, anchor="w").pack(side="left")
        ttk.Entry(r3a, textvariable=self._weight_remove_var, width=8).pack(side="left")

        r3b = ttk.Frame(score_box)
        r3b.pack(fill="x", pady=2)
        ttk.Label(r3b, text="KEEP Streak Bonus (multiplier):", width=30, anchor="w").pack(side="left")
        ttk.Entry(r3b, textvariable=self._bonus_keep_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r3b, text="REMOVE Streak Penalty (mult):", width=24, anchor="w").pack(side="left")
        ttk.Entry(r3b, textvariable=self._penalty_remove_var, width=8).pack(side="left", padx=(0, 20))

        r3c = ttk.Frame(score_box)
        r3c.pack(fill="x", pady=2)
        ttk.Label(r3c, text="Score Minimum Bound:", width=30, anchor="w").pack(side="left")
        ttk.Entry(r3c, textvariable=self._min_score_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r3c, text="Score Maximum Bound:", width=24, anchor="w").pack(side="left")
        ttk.Entry(r3c, textvariable=self._max_score_var, width=8).pack(side="left")

        # 5. Section 4: Base Pipeline & Feature Registry Governance
        gov_box = ttk.LabelFrame(form, text="4. Base Pipeline & Feature Registry Governance", padding=(10, 6))
        gov_box.pack(fill="x", pady=(0, 8))

        r4 = ttk.Frame(gov_box)
        r4.pack(fill="x", pady=2)
        ttk.Label(r4, text="Base Negative Alert Score:", width=30, anchor="w").pack(side="left")
        ttk.Entry(r4, textvariable=self._base_neg_alert_score_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r4, text="Registry Audit REMOVE Runs:", width=26, anchor="w").pack(side="left")
        ttk.Entry(r4, textvariable=self._reg_audit_remove_runs_var, width=8).pack(side="left", padx=(0, 20))
        ttk.Label(r4, text="Registry Alert Models:", width=20, anchor="w").pack(side="left")
        ttk.Entry(r4, textvariable=self._reg_audit_unique_models_var, width=8).pack(side="left")

        # Error banner
        ttk.Label(
            form,
            textvariable=self._policy_validation_err_var,
            foreground="#c62828",
            wraplength=920,
            font=("TkDefaultFont", 9, "bold"),
        ).pack(fill="x", pady=(0, 6))

        # 6. Action buttons
        act_box = ttk.Frame(form)
        act_box.pack(fill="x", pady=(4, 0))

        ttk.Button(
            act_box,
            text="Preview Policy Impact (Read-Only)",
            command=self._on_preview_policy_impact,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            act_box,
            text="Save Policy",
            command=self._on_save_policy,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            act_box,
            text="Rebuild Projections with Policy",
            command=self._on_rebuild_projections,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            act_box,
            text="Reset Defaults",
            command=self._on_reset_policy_defaults,
        ).pack(side="right")

    def _resolved_dataset_context(self) -> Any:
        m = self._market_var.get().strip()
        sec = self._interval_var.get().strip()
        win = self._window_var.get().strip()
        fpid = self._fpid_var.get().strip()

        if m == "ALL" or sec == "ALL" or win == "ALL" or fpid == "ALL":
            return None
        try:
            sec_int = int(sec)
        except ValueError:
            sec_int = 3

        return build_dataset_context(
            market=m,
            sampling_interval_sec=sec_int,
            sliding_window=win,
            feature_project_id=fpid,
        )

    def _resolved_context_id(self) -> str | None:
        ctx = self._resolved_dataset_context()
        return ctx.context_id if ctx else None

    def _on_export_training_candidates(self) -> None:
        from .training_candidate_handoff_dialog import TrainingCandidateHandoffDialog

        ctx = self._resolved_dataset_context()
        cid = ctx.context_id if ctx else None
        policy = load_recommendation_policy(self._data_dir, context_id=cid)

        TrainingCandidateHandoffDialog(
            self,
            chart_dir=self._chart_dir,
            data_dir=self._data_dir,
            context_id=cid,
            context=ctx,
            policy=policy,
        )

    def _update_context_id(self) -> None:
        cid = self._resolved_context_id()
        self._context_id_var.set(cid if cid else "ALL_CONTEXTS")
        self._reload_policy_form()

    def _on_filter_changed(self) -> None:
        self._update_context_id()
        self._reload_all_tabs()

    def _on_policy_scope_toggle(self) -> None:
        self._reload_policy_form()

    def _reload_policy_form(self) -> None:
        cid = self._resolved_context_id()
        scope_mode = self._policy_scope_mode.get()
        target_cid = cid if scope_mode == "context" and cid else None

        policy = load_recommendation_policy(self._data_dir, context_id=target_cid)

        # Populate active policy info
        scope_label = f"Context Override ({cid})" if policy.context_id else "Global Default"
        self._active_policy_info_var.set(
            f"Active Policy: {policy.policy_id} (Version {policy.policy_version}) · Scope: {scope_label} · Updated: {policy.updated_at[:19]}"
        )
        self._policy_validation_err_var.set("")

        # Scoring
        self._weight_keep_var.set(str(policy.scoring.weight_keep))
        self._weight_watch_var.set(str(policy.scoring.weight_watch))
        self._weight_remove_var.set(str(policy.scoring.weight_remove))
        self._bonus_keep_var.set(str(policy.scoring.bonus_consecutive_keep))
        self._penalty_remove_var.set(str(policy.scoring.penalty_consecutive_remove))
        self._min_score_var.set(str(policy.scoring.min_score))
        self._max_score_var.set(str(policy.scoring.max_score))

        # Exp Lifecycle
        self._exp_promo_keep_streak_var.set(str(policy.experimental_lifecycle.promotion_candidate_consecutive_keep))
        self._exp_promo_min_models_var.set(str(policy.experimental_lifecycle.experimental_promotion_min_unique_models))
        self._exp_promo_min_score_var.set(str(policy.experimental_lifecycle.promotion_candidate_min_score))
        self._exp_block_consec_remove_var.set(str(policy.experimental_lifecycle.remove_block_consecutive_threshold))
        self._exp_block_total_remove_var.set(str(policy.experimental_lifecycle.remove_block_total_threshold))

        # Base & Registry
        self._base_neg_alert_score_var.set(str(policy.base_pipeline.negative_alert_score_threshold))
        self._base_strong_keep_score_var.set(str(policy.base_pipeline.strong_keep_min_score))
        self._reg_audit_remove_runs_var.set(str(policy.feature_registry.remove_audit_alert_threshold))
        self._reg_audit_unique_models_var.set(str(policy.feature_registry.registry_alert_min_unique_models))

    def _build_policy_from_form(self) -> tuple[RecommendationPolicy | None, list[str]]:
        try:
            cid = self._resolved_context_id()
            scope_mode = self._policy_scope_mode.get()
            target_cid = cid if scope_mode == "context" and cid else None

            current = load_recommendation_policy(self._data_dir, context_id=target_cid)

            scoring = ScoringPolicy(
                weight_keep=float(self._weight_keep_var.get()),
                weight_watch=float(self._weight_watch_var.get()),
                weight_remove=float(self._weight_remove_var.get()),
                bonus_consecutive_keep=float(self._bonus_keep_var.get()),
                penalty_consecutive_remove=float(self._penalty_remove_var.get()),
                min_score=float(self._min_score_var.get()),
                max_score=float(self._max_score_var.get()),
            )
            exp = ExperimentalLifecyclePolicy(
                promotion_candidate_consecutive_keep=int(self._exp_promo_keep_streak_var.get()),
                experimental_promotion_min_unique_models=int(self._exp_promo_min_models_var.get()),
                promotion_candidate_min_score=float(self._exp_promo_min_score_var.get()),
                remove_block_consecutive_threshold=int(self._exp_block_consec_remove_var.get()),
                remove_block_total_threshold=int(self._exp_block_total_remove_var.get()),
            )
            base = BasePipelinePolicy(
                negative_alert_score_threshold=float(self._base_neg_alert_score_var.get()),
                strong_keep_min_score=float(self._base_strong_keep_score_var.get()),
                min_validation_runs_for_ranking=2,
            )
            reg = FeatureRegistryPolicy(
                remove_audit_alert_threshold=int(self._reg_audit_remove_runs_var.get()),
                registry_alert_min_unique_models=int(self._reg_audit_unique_models_var.get()),
            )

            pol = RecommendationPolicy(
                policy_id=current.policy_id,
                policy_version=current.policy_version,
                context_id=target_cid,
                scoring=scoring,
                experimental_lifecycle=exp,
                base_pipeline=base,
                feature_registry=reg,
            )
            errs = validate_recommendation_policy(pol)
            return (pol, errs) if not errs else (None, errs)
        except Exception as exc:
            return None, [f"Form parsing error: {exc}"]

    def _on_save_policy(self) -> None:
        pol, errs = self._build_policy_from_form()
        if errs or pol is None:
            err_msg = "Validation failed:\n• " + "\n• ".join(errs)
            self._policy_validation_err_var.set(err_msg)
            messagebox.showerror("Policy Validation Error", err_msg, parent=self)
            return

        self._policy_validation_err_var.set("")
        cid = pol.context_id
        scope_str = f"Context '{cid}'" if cid else "Global Default"

        current = load_recommendation_policy(self._data_dir, context_id=cid)
        if pol.parameters_equal(current) and (cid == current.context_id or cid is None):
            messagebox.showinfo("Save Policy", "No changes detected from active policy. Version remains unchanged.", parent=self)
            return

        next_ver = current.policy_version + 1
        confirm_msg = (
            f"Save new Feature Recommendation Policy Version {next_ver} for {scope_str}?\n\n"
            f"• Prior version ({current.policy_id}) will be archived into Policy History.\n"
            f"• Future Production Validation runs will use Version {next_ver}.\n"
            f"• Raw evidence in recommendation_evidence is immutable and will not be altered.\n\n"
            f"Proceed?"
        )
        if not messagebox.askyesno("Confirm Policy Save", confirm_msg, parent=self):
            return

        try:
            saved = save_recommendation_policy(self._data_dir, pol, context_id=cid)
            messagebox.showinfo(
                "Policy Saved",
                f"Successfully saved {saved.policy_id} (Version {saved.policy_version}) for {scope_str}.",
                parent=self,
            )
            self._reload_policy_form()
            self._reload_all_tabs()
            if self._on_changed:
                self._on_changed()
        except Exception as exc:
            messagebox.showerror("Save Policy Failed", str(exc), parent=self)

    def _on_reset_policy_defaults(self) -> None:
        if not messagebox.askyesno("Reset Defaults", "Reset all form fields to default policy settings?", parent=self):
            return
        default_pol = RecommendationPolicy()
        self._weight_keep_var.set(str(default_pol.scoring.weight_keep))
        self._weight_watch_var.set(str(default_pol.scoring.weight_watch))
        self._weight_remove_var.set(str(default_pol.scoring.weight_remove))
        self._bonus_keep_var.set(str(default_pol.scoring.bonus_consecutive_keep))
        self._penalty_remove_var.set(str(default_pol.scoring.penalty_consecutive_remove))
        self._min_score_var.set(str(default_pol.scoring.min_score))
        self._max_score_var.set(str(default_pol.scoring.max_score))

        self._exp_promo_keep_streak_var.set(str(default_pol.experimental_lifecycle.promotion_candidate_consecutive_keep))
        self._exp_promo_min_models_var.set(str(default_pol.experimental_lifecycle.experimental_promotion_min_unique_models))
        self._exp_promo_min_score_var.set(str(default_pol.experimental_lifecycle.promotion_candidate_min_score))
        self._exp_block_consec_remove_var.set(str(default_pol.experimental_lifecycle.remove_block_consecutive_threshold))
        self._exp_block_total_remove_var.set(str(default_pol.experimental_lifecycle.remove_block_total_threshold))

        self._base_neg_alert_score_var.set(str(default_pol.base_pipeline.negative_alert_score_threshold))
        self._base_strong_keep_score_var.set(str(default_pol.base_pipeline.strong_keep_min_score))
        self._reg_audit_remove_runs_var.set(str(default_pol.feature_registry.remove_audit_alert_threshold))
        self._reg_audit_unique_models_var.set(str(default_pol.feature_registry.registry_alert_min_unique_models))
        self._policy_validation_err_var.set("")

    def _on_preview_policy_impact(self) -> None:
        cid = self._resolved_context_id()
        if not cid:
            messagebox.showwarning("Preview Policy Impact", "Please select a specific Dataset Context (Market/Interval/Window) to preview policy impact.", parent=self)
            return

        pol, errs = self._build_policy_from_form()
        if errs or pol is None:
            messagebox.showerror("Policy Validation Error", "Cannot preview invalid policy:\n• " + "\n• ".join(errs), parent=self)
            return

        try:
            conn = get_connection(self._data_dir)
            try:
                preview = preview_policy_impact(conn, context_id=cid, proposed_policy=pol)
            finally:
                conn.close()

            # Open Preview Modal
            win = tk.Toplevel(self)
            win.title(f"Policy Impact Preview — {cid}")
            win.transient(self)
            win.geometry("880x560")
            win.minsize(700, 400)

            pad = ttk.Frame(win, padding=10)
            pad.pack(fill="both", expand=True)
            pad.columnconfigure(0, weight=1)
            pad.rowconfigure(2, weight=1)

            ttk.Label(
                pad,
                text=f"Read-Only In-Memory Simulation for Context: {cid}",
                font=("TkDefaultFont", 11, "bold"),
            ).grid(row=0, column=0, sticky="w", pady=(0, 4))

            ttk.Label(
                pad,
                text="ℹ️ In-Memory Simulation Only: No changes have been written to the database.",
                foreground="#1565c0",
            ).grid(row=1, column=0, sticky="w", pady=(0, 8))

            # Summary comparison table
            sum_frame = ttk.LabelFrame(pad, text="Summary Metrics Comparison", padding=8)
            sum_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
            sum_frame.columnconfigure(0, weight=1)
            sum_frame.rowconfigure(0, weight=1)

            cols = ("metric", "current", "proposed", "delta")
            tree_sum = ttk.Treeview(sum_frame, columns=cols, show="headings", height=6)
            tree_sum.heading("metric", text="Metric / Lifecycle State")
            tree_sum.heading("current", text="Current Policy")
            tree_sum.heading("proposed", text="Proposed Policy")
            tree_sum.heading("delta", text="Net Delta")

            tree_sum.column("metric", width=220, anchor="w")
            tree_sum.column("current", width=120, anchor="e")
            tree_sum.column("proposed", width=120, anchor="e")
            tree_sum.column("delta", width=120, anchor="e")
            tree_sum.pack(fill="x", pady=(0, 8))

            cur_c = preview["current_counts"]
            prop_c = preview["proposed_counts"]
            deltas = preview["deltas"]

            metrics_data = [
                ("PROMOTION_CANDIDATE (Lineage)", cur_c["promotion_candidates"], prop_c["promotion_candidates"], deltas["promotion_candidates"]),
                ("BLOCKED Experimental (Gated)", cur_c["blocked"], prop_c["blocked"], deltas["blocked"]),
                ("ALERT (Registry & Base)", cur_c["alert"], prop_c["alert"], deltas["alert"]),
                ("HELD (WATCH state)", cur_c["held"], prop_c["held"], deltas["held"]),
                ("ACTIVE Features", cur_c["active"], prop_c["active"], deltas["active"]),
            ]
            for m_name, c_val, p_val, d_val in metrics_data:
                sign = "+" if d_val > 0 else ""
                tree_sum.insert("", "end", values=(m_name, c_val, p_val, f"{sign}{d_val}"))

            # Detailed feature transitions
            trans_frame = ttk.LabelFrame(pad, text=f"Feature State & Score Transitions ({len(preview['feature_transitions'])} features)", padding=8)
            trans_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
            pad.rowconfigure(3, weight=2)
            trans_frame.columnconfigure(0, weight=1)
            trans_frame.rowconfigure(0, weight=1)

            t_cols = ("feature", "source", "cur_status", "prop_status", "cur_score", "prop_score", "delta")
            t_tree = ttk.Treeview(trans_frame, columns=t_cols, show="headings")
            t_tree.heading("feature", text="Feature Name")
            t_tree.heading("source", text="Source")
            t_tree.heading("cur_status", text="Current Status")
            t_tree.heading("prop_status", text="Proposed Status")
            t_tree.heading("cur_score", text="Current Score")
            t_tree.heading("prop_score", text="Proposed Score")
            t_tree.heading("delta", text="Score Delta")

            t_tree.column("feature", width=200, anchor="w")
            t_tree.column("source", width=100, anchor="center")
            t_tree.column("cur_status", width=130, anchor="center")
            t_tree.column("prop_status", width=130, anchor="center")
            t_tree.column("cur_score", width=90, anchor="e")
            t_tree.column("prop_score", width=90, anchor="e")
            t_tree.column("delta", width=80, anchor="e")

            t_sb = ttk.Scrollbar(trans_frame, orient="vertical", command=t_tree.yview)
            t_tree.configure(yscrollcommand=t_sb.set)
            t_tree.grid(row=0, column=0, sticky="nsew")
            t_sb.grid(row=0, column=1, sticky="ns")

            for t in preview["feature_transitions"]:
                sign = "+" if t["score_delta"] > 0 else ""
                t_tree.insert(
                    "",
                    "end",
                    values=(
                        t["feature_name"],
                        t["feature_source"],
                        t["current_status"].upper(),
                        t["proposed_status"].upper(),
                        f"{t['current_score']:.1f}",
                        f"{t['proposed_score']:.1f}",
                        f"{sign}{t['score_delta']:.1f}",
                    ),
                )

            btn_box = ttk.Frame(pad)
            btn_box.grid(row=4, column=0, sticky="e")
            ttk.Button(btn_box, text="Close Preview", command=win.destroy).pack(side="right")

        except Exception as exc:
            messagebox.showerror("Preview Failed", str(exc), parent=self)

    def _on_open_policy_history(self) -> None:
        cid = self._resolved_context_id()
        scope_mode = self._policy_scope_mode.get()
        target_cid = cid if scope_mode == "context" and cid else None

        hist = list_policy_history(self._data_dir, context_id=target_cid)
        if not hist:
            messagebox.showinfo("Policy History", "No policy history found.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Policy History & Rollback")
        win.transient(self)
        win.geometry("780x420")
        win.minsize(600, 300)

        pad = ttk.Frame(win, padding=10)
        pad.pack(fill="both", expand=True)
        pad.columnconfigure(0, weight=1)
        pad.rowconfigure(1, weight=1)

        ttk.Label(
            pad,
            text=f"Policy Revision History for: {target_cid or 'GLOBAL DEFAULT'}",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tf = ttk.Frame(pad)
        tf.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = ("version", "policy_id", "updated_at", "restored_from", "status")
        tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="browse")
        tree.heading("version", text="Version")
        tree.heading("policy_id", text="Policy ID")
        tree.heading("updated_at", text="Timestamp")
        tree.heading("restored_from", text="Restored From")
        tree.heading("status", text="Status")

        tree.column("version", width=70, anchor="center")
        tree.column("policy_id", width=180, anchor="w")
        tree.column("updated_at", width=160, anchor="w")
        tree.column("restored_from", width=110, anchor="center")
        tree.column("status", width=90, anchor="center")

        sb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        for h in hist:
            ver = h.get("policy_version") or h.get("version") or 1
            is_active = h.get("is_active", False)
            stat = "ACTIVE" if is_active else "Archived"
            rf = f"v{h.get('restored_from_version')}" if h.get("restored_from_version") else "—"
            tree.insert(
                "",
                "end",
                iid=str(ver),
                values=(
                    f"v{ver}",
                    h.get("policy_id", ""),
                    str(h.get("updated_at") or h.get("created_at") or "")[:19],
                    rf,
                    stat,
                ),
            )

        btn_box = ttk.Frame(pad)
        btn_box.grid(row=2, column=0, sticky="ew")

        def _do_restore() -> None:
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Rollback", "Please select a historical version to restore.", parent=win)
                return
            target_v = int(sel[0])
            cur = load_recommendation_policy(self._data_dir, context_id=target_cid)
            if target_v == cur.policy_version:
                messagebox.showinfo("Rollback", f"Version {target_v} is already active.", parent=win)
                return

            next_v = cur.policy_version + 1
            if not messagebox.askyesno(
                "Confirm Policy Restore",
                f"Restore settings from Version {target_v}?\n\n"
                f"• A NEW Version {next_v} will be created containing Version {target_v}'s thresholds.\n"
                f"• No historical versions will be deleted or overwritten.\n"
                f"• Audit record: v{next_v} restored from v{target_v}.",
                parent=win,
            ):
                return

            try:
                new_pol = restore_policy_version(self._data_dir, target_v, context_id=target_cid)
                messagebox.showinfo(
                    "Policy Restored",
                    f"Successfully created {new_pol.policy_id} (Version {new_pol.policy_version}) restored from Version {target_v}.",
                    parent=win,
                )
                win.destroy()
                self._reload_policy_form()
                self._reload_all_tabs()
                if self._on_changed:
                    self._on_changed()
            except Exception as exc:
                messagebox.showerror("Restore Failed", str(exc), parent=win)

        ttk.Button(btn_box, text="Restore Selected Version as New Active Version", command=_do_restore).pack(side="left")
        ttk.Button(btn_box, text="Close", command=win.destroy).pack(side="right")

    def _reload_all_tabs(self) -> None:
        cid = self._resolved_context_id()
        inc_legacy = self._include_legacy_var.get()

        # 1. Reload Feature Registry
        self._reg_tree.delete(*self._reg_tree.get_children())
        reg_rows = get_population_recommendations(
            self._data_dir,
            population="registry",
            context_id=cid,
            include_legacy_unknown=inc_legacy,
        )
        for r in reg_rows:
            self._reg_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature_name"),
                    r.get("lifecycle_status", "active").upper(),
                    f"{float(r.get('evidence_score') or 0.0):.1f}",
                    r.get("confidence_display", "—"),
                    r.get("stability_display", "—"),
                    r.get("generalization_display", "—"),
                    r.get("consensus_display", "—"),
                    r.get("freshness_display", "—"),
                    int(r.get("keep_runs") or 0),
                    int(r.get("watch_runs") or 0),
                    int(r.get("remove_runs") or 0),
                    int(r.get("unique_models_count") or 0),
                    str(r.get("last_validated_at") or "")[:19],
                ),
            )

        # 2. Reload Base Pipeline
        self._base_tree.delete(*self._base_tree.get_children())
        base_rows = get_population_recommendations(
            self._data_dir,
            population="base_pipeline",
            context_id=cid,
            include_legacy_unknown=inc_legacy,
        )
        for r in base_rows:
            adv_rank_str = f"#{r.get('advisory_rank')}" if r.get("advisory_rank") else "—"
            op_score_str = f"{float(r.get('operational_priority_score') or 0.0):.1f}"
            self._base_tree.insert(
                "",
                "end",
                values=(
                    r.get("priority_rank", "—"),
                    r.get("feature_name"),
                    r.get("lifecycle_status", "active").upper(),
                    f"{float(r.get('evidence_score') or 0.0):.1f}",
                    r.get("confidence_display", "—"),
                    op_score_str,
                    adv_rank_str,
                    r.get("stability_display", "—"),
                    r.get("generalization_display", "—"),
                    r.get("consensus_display", "—"),
                    r.get("freshness_display", "—"),
                    int(r.get("unique_models_count") or 0),
                    str(r.get("last_validated_at") or "")[:19],
                ),
            )

        # 3. Reload Selected Experimental
        self._exp_tree.delete(*self._exp_tree.get_children())
        exp_rows = get_population_recommendations(
            self._data_dir,
            population="experimental",
            context_id=cid,
            include_legacy_unknown=inc_legacy,
        )
        for r in exp_rows:
            ctx_stat = str(r.get("context_status") or "active").upper()
            gate_label = "BLOCKED" if ctx_stat == "BLOCKED" else "CLEAR"
            self._exp_tree.insert(
                "",
                "end",
                values=(
                    r.get("pipeline_id") or "—",
                    r.get("pipeline_snapshot_id") or "—",
                    r.get("feature_name"),
                    r.get("lifecycle_status", "active").upper(),
                    gate_label,
                    f"{float(r.get('lineage_evidence_score') or 0.0):.1f}",
                    r.get("confidence_display", "—"),
                    r.get("stability_display", "—"),
                    r.get("generalization_display", "—"),
                    r.get("consensus_display", "—"),
                    r.get("freshness_display", "—"),
                    int(r.get("consecutive_keep_count") or 0),
                    int(r.get("consecutive_remove_count") or 0),
                    int(r.get("unique_models_count") or 0),
                    str(r.get("last_validated_at") or "")[:19],
                ),
            )

        # 4. Reload Raw Evidence Log
        self._raw_tree.delete(*self._raw_tree.get_children())
        try:
            conn = get_connection(self._data_dir)
            try:
                if cid:
                    cur = conn.execute(
                        "SELECT * FROM recommendation_evidence WHERE context_id = ? ORDER BY run_timestamp DESC LIMIT 200",
                        (cid,),
                    )
                elif not inc_legacy:
                    cur = conn.execute(
                        "SELECT * FROM recommendation_evidence WHERE context_id != 'legacy_unknown' ORDER BY run_timestamp DESC LIMIT 200"
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM recommendation_evidence ORDER BY run_timestamp DESC LIMIT 200"
                    )
                raw_rows = [dict(row) for row in cur.fetchall()]
                for r in raw_rows:
                    self._raw_tree.insert(
                        "",
                        "end",
                        values=(
                            str(r.get("run_timestamp") or "")[:19],
                            r.get("context_id"),
                            r.get("feature_source"),
                            r.get("feature_name"),
                            r.get("recommendation"),
                            r.get("model_name"),
                            str(r.get("validation_run_id") or "")[:8],
                            str(r.get("evidence_detail_json") or ""),
                        ),
                    )
            finally:
                conn.close()
        except Exception:
            pass

        self._status_var.set(
            f"Loaded: Registry ({len(reg_rows)}) · Base Pipeline ({len(base_rows)}) · Experimental ({len(exp_rows)})"
        )

    def _on_rebuild_projections(self) -> None:
        cid = self._resolved_context_id()
        scope_mode = self._policy_scope_mode.get()
        target_cid = cid if scope_mode == "context" and cid else None
        active_pol = load_recommendation_policy(self._data_dir, context_id=target_cid)

        try:
            conn = get_connection(self._data_dir)
            try:
                res = rebuild_all_projections(conn, policy=active_pol, context_id=cid)
            finally:
                conn.close()
            self._reload_all_tabs()
            messagebox.showinfo(
                "Rebuild Projections",
                f"Successfully rebuilt summary projections using {active_pol.policy_id} (v{active_pol.policy_version}):\n\n"
                f"• Context Summaries: {res.get('context_summaries_rebuilt', 0)}\n"
                f"• Lineage Summaries: {res.get('lineage_summaries_rebuilt', 0)}\n"
                f"• Raw evidence in recommendation_evidence is unchanged.",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Rebuild Projections Failed", str(exc), parent=self)


def open_feature_recommendation_viewer(
    master: tk.Misc,
    *,
    chart_dir: str,
    initial_market: str = "NIFTY",
    initial_interval_sec: int = 3,
    initial_sliding_window: str = "standard",
    initial_feature_project_id: str = "all",
    on_changed: Callable[[], None] | None = None,
) -> FeatureRecommendationViewerDialog:
    return FeatureRecommendationViewerDialog(
        master,
        chart_dir=chart_dir,
        initial_market=initial_market,
        initial_interval_sec=initial_interval_sec,
        initial_sliding_window=initial_sliding_window,
        initial_feature_project_id=initial_feature_project_id,
        on_changed=on_changed,
    )
