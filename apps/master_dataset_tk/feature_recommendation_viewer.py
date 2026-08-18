"""Feature Studio Recommendation & Evidence Viewer Dialog.

Dataset-context-aware recommendation inspector for the three feature populations:
1. Feature Registry (Health, score, accumulated runs, alert state)
2. Base Pipeline (Evidence score, priority ranking, health state)
3. Selected Experimental (Lineage streaks, PROMOTION_CANDIDATE, BLOCKED gate)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.production_validation.api import (
    build_dataset_context,
    get_population_recommendations,
    ignore_recommendation,
    rebuild_all_projections,
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
        self.geometry("1020x640")
        self.minsize(800, 500)

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

        cols = ("feature", "status", "score", "keep", "watch", "remove", "models", "last_validated")
        self._reg_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._reg_tree.heading("feature", text="Feature Name")
        self._reg_tree.heading("status", text="Health Status")
        self._reg_tree.heading("score", text="Evidence Score")
        self._reg_tree.heading("keep", text="KEEP")
        self._reg_tree.heading("watch", text="WATCH")
        self._reg_tree.heading("remove", text="REMOVE")
        self._reg_tree.heading("models", text="Unique Models")
        self._reg_tree.heading("last_validated", text="Last Validated")

        self._reg_tree.column("feature", width=220, anchor="w")
        self._reg_tree.column("status", width=100, anchor="center")
        self._reg_tree.column("score", width=100, anchor="e")
        self._reg_tree.column("keep", width=70, anchor="e")
        self._reg_tree.column("watch", width=70, anchor="e")
        self._reg_tree.column("remove", width=70, anchor="e")
        self._reg_tree.column("models", width=100, anchor="e")
        self._reg_tree.column("last_validated", width=150, anchor="w")

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

        cols = ("rank", "feature", "status", "score", "keep", "watch", "remove", "models", "last_validated")
        self._base_tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="extended")
        self._base_tree.heading("rank", text="Priority Rank")
        self._base_tree.heading("feature", text="Feature Name")
        self._base_tree.heading("status", text="Health Status")
        self._base_tree.heading("score", text="Evidence Score")
        self._base_tree.heading("keep", text="KEEP")
        self._base_tree.heading("watch", text="WATCH")
        self._base_tree.heading("remove", text="REMOVE")
        self._base_tree.heading("models", text="Unique Models")
        self._base_tree.heading("last_validated", text="Last Validated")

        self._base_tree.column("rank", width=80, anchor="center")
        self._base_tree.column("feature", width=220, anchor="w")
        self._base_tree.column("status", width=100, anchor="center")
        self._base_tree.column("score", width=100, anchor="e")
        self._base_tree.column("keep", width=70, anchor="e")
        self._base_tree.column("watch", width=70, anchor="e")
        self._base_tree.column("remove", width=70, anchor="e")
        self._base_tree.column("models", width=100, anchor="e")
        self._base_tree.column("last_validated", width=150, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._base_tree.yview)
        self._base_tree.configure(yscrollcommand=sb.set)
        self._base_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_exp_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Selected Experimental features tracked by exact pipeline & snapshot lineage. Satisfying 3 consecutive KEEPs on unique models produces PROMOTION_CANDIDATE eligibility. Repeated REMOVEs trigger context-level BLOCKED gate.",
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
        self._exp_tree.heading("streak_keep", text="Streak KEEP")
        self._exp_tree.heading("streak_remove", text="Streak REMOVE")
        self._exp_tree.heading("models", text="Unique Models")
        self._exp_tree.heading("last_validated", text="Last Validated")

        self._exp_tree.column("pipeline_id", width=90, anchor="center")
        self._exp_tree.column("snapshot_id", width=100, anchor="w")
        self._exp_tree.column("feature", width=180, anchor="w")
        self._exp_tree.column("lineage_status", width=140, anchor="center")
        self._exp_tree.column("context_gate", width=100, anchor="center")
        self._exp_tree.column("score", width=90, anchor="e")
        self._exp_tree.column("streak_keep", width=80, anchor="e")
        self._exp_tree.column("streak_remove", width=90, anchor="e")
        self._exp_tree.column("models", width=90, anchor="e")
        self._exp_tree.column("last_validated", width=130, anchor="w")

        sb = ttk.Scrollbar(tf, orient="vertical", command=self._exp_tree.yview)
        self._exp_tree.configure(yscrollcommand=sb.set)
        self._exp_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    def _build_raw_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Raw append-only recommendation evidence log. Sole authoritative source of truth.",
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

    def _resolved_context_id(self) -> str | None:
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

        ctx = build_dataset_context(
            market=m,
            sampling_interval_sec=sec_int,
            sliding_window=win,
            feature_project_id=fpid,
        )
        return ctx.context_id

    def _update_context_id(self) -> None:
        cid = self._resolved_context_id()
        self._context_id_var.set(cid if cid else "ALL_CONTEXTS")

    def _on_filter_changed(self) -> None:
        self._update_context_id()
        self._reload_all_tabs()

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
            self._base_tree.insert(
                "",
                "end",
                values=(
                    r.get("priority_rank", "—"),
                    r.get("feature_name"),
                    r.get("lifecycle_status", "active").upper(),
                    f"{float(r.get('evidence_score') or 0.0):.1f}",
                    int(r.get("keep_runs") or 0),
                    int(r.get("watch_runs") or 0),
                    int(r.get("remove_runs") or 0),
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
        try:
            conn = get_connection(self._data_dir)
            try:
                res = rebuild_all_projections(conn)
            finally:
                conn.close()
            self._reload_all_tabs()
            messagebox.showinfo(
                "Rebuild Projections",
                f"Successfully rebuilt summary projections from immutable evidence:\n\n"
                f"• Context Summaries: {res.get('context_summaries_rebuilt', 0)}\n"
                f"• Lineage Summaries: {res.get('lineage_summaries_rebuilt', 0)}",
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
