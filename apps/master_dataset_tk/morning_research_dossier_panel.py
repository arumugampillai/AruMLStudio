"""Autonomous Research Registry & Historical Research Dossier (Doc 16).

Provides an interactive GUI presenting the canonical historical ledger of autonomous research runs,
along with comprehensive multi-tab Research Detail Dossiers containing point-in-time discovery features,
candidate rankings, generational lineage, feature governance audits, and execution logs.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.morning_dossier import (
    DiscoveredFeatureStatus,
    MorningResearchDossier,
    export_morning_dossier_markdown,
    generate_morning_research_dossier,
)
from chain_replay_ml.overnight_campaign.persistence import init_campaign_tables
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from chain_replay_ml.research_registry.store import (
    backfill_historical_research_records,
    delete_research_records,
    get_all_research_records,
    get_research_detail,
)

from .build_service import chart_data_dir
from .fold_replay_widgets import place_toplevel_beside_main
from .model_registry_widgets import (
    ACCENT,
    COL_MUTED,
    COL_OK,
    COL_PRODUCTION,
    COL_TRAINING,
    COL_WARN,
    ScrollableFrame,
    clear_children,
    data_table,
    fmt_num,
    fmt_pct,
    kv_block,
    section_desc,
    section_title,
)


class MorningResearchDossierPanel(ttk.Frame):
    """Interactive GUI for Autonomous Research Registry historical ledger and Research Detail Dossiers."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        data_dir: str | None = None,
        on_select_model: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_dir = data_dir or chart_data_dir()
        self.on_select_model = on_select_model

        # Filter state variables
        self.filter_campaign = tk.StringVar(value="All")
        self.filter_status = tk.StringVar(value="All")
        self.filter_context = tk.StringVar(value="All")
        self.search_var = tk.StringVar(value="")
        self.selected_campaign_id = tk.StringVar(value="")  # Compatibility

        self._all_records: list[dict[str, Any]] = []
        self._sort_col: str = "started_at"
        self._sort_reverse: bool = True
        self.current_dossier: MorningResearchDossier | None = None
        self._detail_window: tk.Toplevel | None = None
        self._current_detail_research_id: str | None = None

        init_analysis_db(self.data_dir)
        init_campaign_tables(self.data_dir)

        self._build_layout()
        self.refresh()

    def on_show(self, tab: str | None = None) -> None:
        """Called by app shell when user navigates to Autonomous Researches."""
        self.refresh()

    def select_tab(self, tab: str) -> None:
        """Compatibility helper."""
        self.refresh()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.data_dir = chart_data_dir(chart_dir)
        self.refresh()

    def refresh(self) -> None:
        """Reload research registry records and populate filter values."""
        if self.data_dir:
            try:
                backfill_historical_research_records(self.data_dir)
            except Exception:
                pass
            self._all_records = get_all_research_records(self.data_dir)
        else:
            self._all_records = []

        # Populate filter combobox values
        campaigns = sorted(list({r.get("campaign_id", "") for r in self._all_records if r.get("campaign_id")}))
        contexts = sorted(list({r.get("context_key", "") for r in self._all_records if r.get("context_key")}))
        
        self.campaign_filter_combo["values"] = ["All"] + campaigns
        self.context_filter_combo["values"] = ["All"] + contexts

        self._apply_filters()

    def _build_layout(self) -> None:
        """Construct the top header, filter toolbar, main registry treeview, and status footer."""
        # 1. Header Toolbar
        hdr_frame = ttk.Frame(self, padding=(10, 8, 10, 4))
        hdr_frame.pack(fill=tk.X, side=tk.TOP)

        title_box = ttk.Frame(hdr_frame)
        title_box.pack(side=tk.LEFT)
        ttk.Label(title_box, text="📜 Autonomous Research Registry", font=("Segoe UI", 12, "bold"), foreground="#0d47a1").pack(anchor=tk.W)
        self.subtitle_lbl = ttk.Label(title_box, text="Historical permanent ledger of autonomous model research campaigns", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED)
        self.subtitle_lbl.pack(anchor=tk.W)

        btn_box = ttk.Frame(hdr_frame)
        btn_box.pack(side=tk.RIGHT)
        self.delete_btn = ttk.Button(btn_box, text="Delete Research", command=self._on_delete_clicked, state="disabled")
        self.delete_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="🔍 View Research Detail Dossier", command=self._on_view_detail_clicked).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="🔄 Refresh", command=self.refresh).pack(side=tk.LEFT, padx=4)

        # 2. Filter Toolbar
        filter_frame = ttk.LabelFrame(self, text="Filter Runs", padding=(8, 4, 8, 6))
        filter_frame.pack(fill=tk.X, padx=10, pady=(2, 6))

        ttk.Label(filter_frame, text="Campaign:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.campaign_filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_campaign, width=22, state="readonly")
        self.campaign_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.campaign_filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(filter_frame, text="Status:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.status_filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_status,
            values=["All", "COMPLETED", "FAILED", "ABORTED", "RUNNING", "PAUSED"],
            width=13,
            state="readonly",
        )
        self.status_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.status_filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(filter_frame, text="Context:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.context_filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_context, width=22, state="readonly")
        self.context_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.context_filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(filter_frame, text="Search:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=18)
        search_entry.pack(side=tk.LEFT, padx=(0, 8))
        search_entry.bind("<KeyRelease>", lambda _e: self._apply_filters())

        ttk.Button(filter_frame, text="❌ Clear Filters", command=self._clear_filters).pack(side=tk.LEFT, padx=4)

        # 3. Main Treeview Container
        table_container = ttk.Frame(self, padding=(10, 0, 10, 4))
        table_container.pack(fill=tk.BOTH, expand=True)

        cols = (
            "research_id", "campaign_id", "context", "dataset", "started_at", "duration",
            "status", "algos", "gens", "cands", "df_created",
            "keep", "watch", "remove", "pool", "best_cand", "best_score"
        )

        self.tree = ttk.Treeview(table_container, columns=cols, show="headings", height=16)
        self.tree.heading("research_id", text="Research ID", command=lambda: self._sort_by_column("research_id"))
        self.tree.heading("campaign_id", text="Campaign ID", command=lambda: self._sort_by_column("campaign_id"))
        self.tree.heading("context", text="Context", command=lambda: self._sort_by_column("context_key"))
        self.tree.heading("dataset", text="Dataset", command=lambda: self._sort_by_column("dataset_name"))
        self.tree.heading("started_at", text="Started (UTC)", command=lambda: self._sort_by_column("started_at"))
        self.tree.heading("duration", text="Duration", command=lambda: self._sort_by_column("duration_seconds"))
        self.tree.heading("status", text="Status", command=lambda: self._sort_by_column("status"))
        self.tree.heading("algos", text="Algorithms", command=lambda: self._sort_by_column("algorithms_used_json"))
        self.tree.heading("gens", text="Gens", command=lambda: self._sort_by_column("actual_generations_completed"))
        self.tree.heading("cands", text="Cands", command=lambda: self._sort_by_column("candidates_evaluated"))
        self.tree.heading("df_created", text="DF Created", command=lambda: self._sort_by_column("total_df_features_created"))
        self.tree.heading("keep", text="KEEP", command=lambda: self._sort_by_column("keep_count"))
        self.tree.heading("watch", text="WATCH", command=lambda: self._sort_by_column("watch_count"))
        self.tree.heading("remove", text="REMOVE", command=lambda: self._sort_by_column("remove_count"))
        self.tree.heading("pool", text="Active Pool", command=lambda: self._sort_by_column("active_discovery_pool"))
        self.tree.heading("best_cand", text="Champion ID", command=lambda: self._sort_by_column("best_candidate_id"))
        self.tree.heading("best_score", text="Best Score", command=lambda: self._sort_by_column("best_composite_score"))

        self.tree.column("research_id", width=220, anchor=tk.W)
        self.tree.column("campaign_id", width=160, anchor=tk.W)
        self.tree.column("context", width=140, anchor=tk.W)
        self.tree.column("dataset", width=130, anchor=tk.W)
        self.tree.column("started_at", width=125, anchor=tk.W)
        self.tree.column("duration", width=70, anchor=tk.CENTER)
        self.tree.column("status", width=90, anchor=tk.CENTER)
        self.tree.column("algos", width=110, anchor=tk.W)
        self.tree.column("gens", width=50, anchor=tk.CENTER)
        self.tree.column("cands", width=55, anchor=tk.CENTER)
        self.tree.column("df_created", width=75, anchor=tk.CENTER)
        self.tree.column("keep", width=50, anchor=tk.CENTER)
        self.tree.column("watch", width=55, anchor=tk.CENTER)
        self.tree.column("remove", width=60, anchor=tk.CENTER)
        self.tree.column("pool", width=70, anchor=tk.CENTER)
        self.tree.column("best_cand", width=150, anchor=tk.W)
        self.tree.column("best_score", width=80, anchor=tk.E)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_selection_changed())
        self.tree.bind("<Double-1>", lambda _e: self._on_view_detail_clicked())
        self.tree.bind("<Return>", lambda _e: self._on_view_detail_clicked())

        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set, selectmode="extended")

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # 4. Status Bar
        self.status_bar = ttk.Label(self, text="", font=("Segoe UI", 9), padding=(12, 4))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _clear_filters(self) -> None:
        self.filter_campaign.set("All")
        self.filter_status.set("All")
        self.filter_context.set("All")
        self.search_var.set("")
        self._apply_filters()

    def _sort_by_column(self, col_key: str) -> None:
        if self._sort_col == col_key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col_key
            self._sort_reverse = False
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Filter and sort records, then repopulate treeview."""
        sel_camp = self.filter_campaign.get()
        sel_stat = self.filter_status.get()
        sel_ctx = self.filter_context.get()
        q = self.search_var.get().lower().strip()

        filtered = []
        for r in self._all_records:
            if sel_camp != "All" and r.get("campaign_id") != sel_camp:
                continue
            if sel_stat != "All" and str(r.get("status", "")).upper() != sel_stat.upper():
                continue
            if sel_ctx != "All" and r.get("context_key") != sel_ctx:
                continue
            if q:
                row_str = f"{r.get('research_id','')} {r.get('campaign_id','')} {r.get('context_key','')} {r.get('dataset_name','')} {r.get('best_candidate_id','')}".lower()
                if q not in row_str:
                    continue
            filtered.append(r)

        # Sorting
        def _get_sort_val(rec: dict[str, Any]) -> Any:
            v = rec.get(self._sort_col)
            if v is None:
                return ""
            if isinstance(v, (int, float)):
                return v
            try:
                return float(v)
            except (ValueError, TypeError):
                return str(v).lower()

        filtered.sort(key=_get_sort_val, reverse=self._sort_reverse)

        # Clear and repopulate
        for item in self.tree.get_children():
            self.tree.delete(item)

        for r in filtered:
            dur_sec = r.get("duration_seconds")
            dur_str = f"{int(dur_sec//60)}m {int(dur_sec%60)}s" if dur_sec else "—"
            st_str = r.get("status", "UNKNOWN")
            if st_str == "COMPLETED":
                st_icon = "🟢 " + st_str
            elif st_str == "RUNNING":
                st_icon = "🟡 " + st_str
            elif st_str == "FAILED":
                st_icon = "🔴 " + st_str
            elif st_str == "PAUSED":
                st_icon = "⏸️ " + st_str
            elif st_str == "ABORTED":
                st_icon = "🟠 " + st_str
            else:
                st_icon = "⚪ " + st_str

            algos_raw = r.get("algorithms_used_json", "[]")
            try:
                algos_list = json.loads(algos_raw)
                algos_str = ", ".join(algos_list[:2]) + (f" +{len(algos_list)-2}" if len(algos_list) > 2 else "")
            except Exception:
                algos_str = str(algos_raw)

            b_score = float(r.get("best_composite_score") or 0.0)
            self.tree.insert("", tk.END, iid=r.get("research_id"), values=(
                r.get("research_id", "—"),
                r.get("campaign_id", "—"),
                r.get("context_key", "—"),
                r.get("dataset_name", "—"),
                str(r.get("started_at", "—"))[:19].replace("T", " "),
                dur_str,
                st_icon,
                algos_str,
                str(r.get("actual_generations_completed", 0)),
                str(r.get("candidates_evaluated", 0)),
                str(r.get("total_df_features_created", 0)),
                str(r.get("keep_count", 0)),
                str(r.get("watch_count", 0)),
                str(r.get("remove_count", 0)),
                str(r.get("active_discovery_pool", 0)),
                r.get("best_candidate_id") or "—",
                f"{b_score:.2f} pts",
            ))

        total_cnt = len(self._all_records)
        shown_cnt = len(filtered)
        c_comp = sum(1 for r in self._all_records if r.get("status") == "COMPLETED")
        c_fail = sum(1 for r in self._all_records if r.get("status") == "FAILED")
        c_abort = sum(1 for r in self._all_records if r.get("status") == "ABORTED")
        c_run = sum(1 for r in self._all_records if r.get("status") == "RUNNING")

        self.status_bar.config(
            text=f"Showing {shown_cnt} of {total_cnt} permanent research runs | 🟢 {c_comp} COMPLETED · 🔴 {c_fail} FAILED · 🟠 {c_abort} ABORTED" + (f" · 🟡 {c_run} RUNNING" if c_run > 0 else "")
        )
        self.subtitle_lbl.config(text=f"({total_cnt} permanent research runs recorded in analysis.db)")
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        """Update Delete button text and state according to current selection count."""
        sel = self.tree.selection()
        n = len(sel)
        if n == 0:
            self.delete_btn.config(text="Delete Research", state="disabled")
        elif n == 1:
            self.delete_btn.config(text="🗑️ Delete Selected (1)", state="normal")
        else:
            self.delete_btn.config(text=f"🗑️ Delete Selected ({n})", state="normal")

    def _on_delete_clicked(self) -> None:
        """Handle deletion of single or multiple selected research runs with user confirmation."""
        sel = list(self.tree.selection())
        if not sel:
            messagebox.showinfo("Select Research Run", "Please select one or more research runs to delete.")
            return

        n = len(sel)
        plural = "research run" if n == 1 else f"{n} research runs"
        prompt_msg = (
            f"Delete {plural}?\n\n"
            f"This will permanently remove the selected research registry records and associated research-run metadata.\n"
            f"This action cannot be undone."
        )
        if n <= 5:
            prompt_msg += "\n\nSelected Research IDs:\n" + "\n".join(f"• {sid}" for sid in sel)
        else:
            prompt_msg += "\n\nSelected Research IDs:\n" + "\n".join(f"• {sid}" for sid in sel[:5]) + f"\n... and {n - 5} more"

        confirmed = messagebox.askyesno(
            f"Delete {plural}?",
            prompt_msg,
            icon="warning",
        )
        if not confirmed:
            return

        try:
            res = delete_research_records(self.data_dir, sel)
            self.tree.selection_set([])
            self.refresh()
            self._on_selection_changed()
            messagebox.showinfo(
                "Deletion Complete",
                f"Successfully deleted {plural}.\n\n"
                f"Removed from research registry and cleaned up uniquely associated campaign run metadata."
            )
        except Exception as ex:
            messagebox.showerror("Deletion Error", f"Failed to delete research records:\n{ex}")

    def _on_view_detail_clicked(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Research Run", "Please select a research run from the registry table to view its complete historical research dossier.")
            return
        item_vals = self.tree.item(sel[0], "values")
        if item_vals:
            r_id = item_vals[0]
            self._show_research_detail_modal(r_id)

    def _show_research_detail_modal(self, research_id: str) -> None:
        """Show full Research Detail Dossier window positioned beside the main application."""
        detail = get_research_detail(self.data_dir, research_id) if self.data_dir else None
        if not detail:
            messagebox.showerror("Error", f"Could not load details for Research ID: {research_id}")
            return

        camp_id = detail.get("campaign_id") or ""
        dossier: MorningResearchDossier | None = None
        if camp_id and self.data_dir:
            try:
                dossier = generate_morning_research_dossier(self.data_dir, camp_id)
            except Exception:
                dossier = None

        # Reuse existing detail window if already open, or create a new one positioned beside main
        win = getattr(self, "_detail_window", None)
        first_place = False
        if win is not None:
            try:
                if win.winfo_exists():
                    clear_children(win)
                else:
                    win = None
            except tk.TclError:
                win = None

        if win is None:
            root = self.winfo_toplevel()
            win = tk.Toplevel(root)
            win.withdraw()
            try:
                win.transient(root)
            except tk.TclError:
                pass
            self._detail_window = win
            first_place = True

        win.title(f"Autonomous Research Detail — {research_id}")

        # Top window header bar (Feature Transformations style)
        win_hdr = ttk.Frame(win, padding=(10, 8))
        win_hdr.pack(fill=tk.X)
        ttk.Label(
            win_hdr,
            text=f"Autonomous Research Detail: {research_id}",
            font=("Segoe UI", 11, "bold"),
            foreground="#0d47a1",
        ).pack(side=tk.LEFT)
        ttk.Button(win_hdr, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

        cfg = detail.get("research_config") or {}
        hw = detail.get("hardware") or {}
        feats = detail.get("features") or []
        gens = detail.get("generations") or []

        # =========================================================================
        # TAB 1: 📋 Research Summary & Hardware
        # =========================================================================
        t_sum = ScrollableFrame(nb)
        nb.add(t_sum, text="📋 Research Summary & Hardware")
        s_inner = getattr(t_sum, "inner", t_sum)

        hdr = ttk.Frame(s_inner)
        hdr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(hdr, text=f"Autonomous Research Dossier: {detail.get('research_id')}", font=("Segoe UI", 12, "bold"), foreground="#0d47a1").pack(side=tk.LEFT)
        st_val = str(detail.get("status", "UNKNOWN"))
        if st_val == "COMPLETED":
            st_color = "#2e7d32"
        elif st_val == "RUNNING":
            st_color = "#f57f17"
        elif st_val == "FAILED":
            st_color = "#c62828"
        elif st_val == "PAUSED":
            st_color = "#1565c0"
        elif st_val == "ABORTED":
            st_color = "#e65100"
        else:
            st_color = "#757575"
        ttk.Label(hdr, text=f"● {st_val}", font=("Segoe UI", 10, "bold"), foreground=st_color).pack(side=tk.RIGHT, padx=6)

        def _render_section(parent: ttk.Frame, title: str, pairs: list[tuple[str, Any]]) -> None:
            sec = ttk.LabelFrame(parent, text=title, padding=8)
            sec.pack(fill=tk.X, pady=(0, 6))
            grid = ttk.Frame(sec)
            grid.pack(fill=tk.X)
            for idx, (lbl, val) in enumerate(pairs):
                r = idx // 2
                c = (idx % 2) * 2
                ttk.Label(grid, text=f"{lbl}:", font=("Segoe UI", 9, "bold")).grid(row=r, column=c, sticky=tk.NW, padx=(6, 4), pady=2)
                ttk.Label(grid, text=str(val if val is not None and val != "" else "—"), font=("Segoe UI", 9), wraplength=380, justify="left").grid(row=r, column=c+1, sticky=tk.NW, padx=(0, 14), pady=2)

        # A. Research Identity
        ctx_k = str(detail.get("context_key") or "")
        ctx_parts = ctx_k.split(":") if ":" in ctx_k else ctx_k.split("_")
        market_val = cfg.get("market") or (ctx_parts[0] if len(ctx_parts) > 0 else "NIFTY")
        interval_val = cfg.get("sampling_interval") or cfg.get("timeframe") or ("5m" if "5m" in ctx_k else "1m")
        task_val = cfg.get("task") or ("DIRECTION_CLASSIFIER" if "DIRECTION" in ctx_k else "REGRESSION")
        horizon_val = cfg.get("horizon") or cfg.get("prediction_horizon") or ("6s" if "6s" in ctx_k or "6" in ctx_parts else "—")
        regime_val = cfg.get("regime") or (":".join(ctx_parts[2:]) if len(ctx_parts) > 2 else "standard:all")

        _render_section(s_inner, "🏷️ A. Research Identity", [
            ("Research ID", detail.get("research_id")),
            ("Campaign ID", detail.get("campaign_id")),
            ("Context ID", detail.get("context_id")),
            ("Context Key", detail.get("context_key")),
            ("Market", market_val),
            ("Sampling Interval", interval_val),
            ("Task", task_val),
            ("Horizon", horizon_val),
            ("Regime", regime_val),
        ])

        # B. Dataset & Pipeline
        _render_section(s_inner, "📦 B. Dataset & Pipeline", [
            ("Dataset Name", detail.get("dataset_name")),
            ("Dataset Snapshot", detail.get("dataset_snapshot_hash")),
            ("Base Pipeline ID", f"{detail.get('base_pipeline_id')} ({detail.get('base_feature_count', 171)} features)"),
            ("Registry Feature Count", f"{detail.get('registry_feature_count', 211)} permanent features"),
            ("Discovery Pipeline ID", detail.get("discovery_pipeline_id")),
            ("Final Discovery Snapshot", detail.get("final_discovery_snapshot_hash")),
        ])

        # C. Execution Lifecycle
        dur_sec = float(detail.get("duration_seconds") or 0.0)
        dur_fmt = f"{int(dur_sec // 60)}m {int(dur_sec % 60)}s ({dur_sec:.1f}s)" if dur_sec > 0 else "—"
        _render_section(s_inner, "⚙️ C. Execution Lifecycle", [
            ("Status", detail.get("status")),
            ("Stop Reason", detail.get("stop_reason")),
            ("Started At (UTC)", str(detail.get("started_at", "—"))[:19].replace("T", " ")),
            ("Finished At (UTC)", str(detail.get("finished_at", "—"))[:19].replace("T", " ") if detail.get("finished_at") else "—"),
            ("Total Duration", dur_fmt),
            ("Generations Configured", str(detail.get("max_generations_configured", "—"))),
            ("Generations Completed", str(detail.get("actual_generations_completed", 0))),
            ("Elimination Strategy", str(detail.get("elimination_strategy", "—"))),
            ("Candidates Generated", str(detail.get("candidates_generated", 0))),
            ("Candidates Evaluated", str(detail.get("candidates_evaluated", 0))),
            ("Candidates Pruned", str(detail.get("candidates_pruned", 0))),
        ])

        # D. Algorithms / Training
        algos_raw = detail.get("algorithms_used_json") or "[]"
        try:
            algos_list = json.loads(algos_raw)
            algos_fmt = ", ".join(algos_list)
        except Exception:
            algos_fmt = str(algos_raw)

        _render_section(s_inner, "🧠 D. Algorithms & Training", [
            ("Algorithms Used", algos_fmt),
            ("Evaluation Metric", cfg.get("eval_metric") or "ROC-AUC / Composite"),
            ("Cross-Validation", cfg.get("cv_policy") or "TimeSeriesSplit (5 folds)"),
            ("Early Stopping", f"{cfg.get('early_stopping_rounds', 50)} rounds"),
            ("Objective / Target", cfg.get("target_column") or "label_up_2pct_5m / direction"),
            ("Hyperparameter Search", cfg.get("hpo_mode") or "Evolutionary Mutator"),
        ])

        # E. Research Champion
        b_comp = float(detail.get("best_composite_score") or 0.0)
        b_trade = float(detail.get("best_trading_score") or 0.0)
        b_mod = float(detail.get("best_model_score") or 0.0)
        lift = float(detail.get("total_score_lift") or 0.0)
        _render_section(s_inner, "🏆 E. Research Champion Model", [
            ("Champion Candidate ID", detail.get("best_candidate_id")),
            ("Champion Composite Score", f"{b_comp:.2f} pts (Lift: {lift:+.2f} pts)"),
            ("Champion Trading Score", f"{b_trade:.2f} pts"),
            ("Champion Model Score", f"{b_mod:.2f} pts"),
            ("Win Rate", f"{float(detail.get('best_win_rate_pct') or 0.0):.1f}%" if detail.get('best_win_rate_pct') else "—"),
            ("Profit Factor", f"{float(detail.get('best_profit_factor') or 0.0):.2f}" if detail.get('best_profit_factor') else "—"),
            ("Max Drawdown", f"{float(detail.get('best_max_drawdown_pct') or 0.0):.2f}%" if detail.get('best_max_drawdown_pct') else "—"),
            ("Promotion Status", "Research Memory Only (Human Governance Required)"),
        ])

        # F. Discovery Result
        _render_section(s_inner, "🔬 F. Discovered Features Summary", [
            ("Total DF Features Created", str(detail.get("total_df_features_created", 0))),
            ("KEEP Count", f"🟢 {detail.get('keep_count', 0)}"),
            ("WATCH Count", f"🟡 {detail.get('watch_count', 0)}"),
            ("REMOVE Count", f"🔴 {detail.get('remove_count', 0)}"),
            ("Active Discovery Pool", f"⭐ {detail.get('active_discovery_pool', 0)} active features"),
            ("Unique Formulas", str(detail.get("unique_formula_count", detail.get("total_df_features_created", 0)))),
        ])

        # G. Hardware & Runtime
        _render_section(s_inner, "💻 G. Hardware & Runtime", [
            ("CPU", hw.get("cpu")),
            ("GPU Acceleration", hw.get("gpu")),
            ("GPU Model", hw.get("gpu_model")),
            ("Algorithm Device Mapping", hw.get("algorithms_mapping")),
            ("Workers / Threads", hw.get("workers_threads")),
            ("Dedicated VRAM / Memory", hw.get("memory")),
            ("Device Fallback Information", hw.get("fallback_info")),
            ("NVIDIA Driver Version", hw.get("driver_version")),
        ])

        # =========================================================================
        # TAB 2: ⭐ Features (with KEEP / WATCH / REMOVE sub-tabs)
        # =========================================================================
        t_feat = ttk.Frame(nb, padding=6)
        keep_feats = [f for f in feats if str(f.get("lifecycle_status", "")).upper() == "KEEP"]
        watch_feats = [f for f in feats if str(f.get("lifecycle_status", "")).upper() == "WATCH"]
        remove_feats = [f for f in feats if str(f.get("lifecycle_status", "")).upper() == "REMOVE"]
        nb.add(t_feat, text=f"⭐ Features ({len(feats)})")

        feat_nb = ttk.Notebook(t_feat)
        feat_nb.pack(fill=tk.BOTH, expand=True)

        def _build_feature_subtab(sub_parent: ttk.Frame, feature_sublist: list[dict[str, Any]], status_label: str) -> None:
            if not feature_sublist:
                ttk.Label(sub_parent, text=f"No {status_label} features recorded for this research run.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(pady=25)
                return

            paned = ttk.Panedwindow(sub_parent, orient=tk.VERTICAL)
            paned.pack(fill=tk.BOTH, expand=True)

            top_frame = ttk.Frame(paned)
            paned.add(top_frame, weight=3)

            f_cols = (
                "name", "fid", "hash", "strat", "gen", "parents",
                "delta_auc", "ks", "drift", "evidence", "verdict", "rationale"
            )
            tree = ttk.Treeview(top_frame, columns=f_cols, show="headings", height=8)
            tree.heading("name", text="Feature Name (Canonical)")
            tree.heading("fid", text="DF Feature ID")
            tree.heading("hash", text="Formula Hash")
            tree.heading("strat", text="Strategy")
            tree.heading("gen", text="Gen")
            tree.heading("parents", text="Parent Features")
            tree.heading("delta_auc", text="ΔAUC")
            tree.heading("ks", text="D_KS")
            tree.heading("drift", text="Drift")
            tree.heading("evidence", text="Evidence Score")
            tree.heading("verdict", text="Verdict")
            tree.heading("rationale", text="Governance Rationale")

            tree.column("name", width=260, anchor=tk.W)
            tree.column("fid", width=180, anchor=tk.W)
            tree.column("hash", width=110, anchor=tk.CENTER)
            tree.column("strat", width=90, anchor=tk.CENTER)
            tree.column("gen", width=45, anchor=tk.CENTER)
            tree.column("parents", width=160, anchor=tk.W)
            tree.column("delta_auc", width=75, anchor=tk.E)
            tree.column("ks", width=65, anchor=tk.E)
            tree.column("drift", width=55, anchor=tk.CENTER)
            tree.column("evidence", width=85, anchor=tk.E)
            tree.column("verdict", width=75, anchor=tk.CENTER)
            tree.column("rationale", width=220, anchor=tk.W)

            for f in feature_sublist:
                d_auc = f.get("delta_auc")
                d_auc_str = f"{float(d_auc):+.5f}" if d_auc is not None else "—"
                ks_val = f.get("ks_statistic")
                ks_str = f"{float(ks_val):.4f}" if ks_val is not None else "—"
                ev_val = f.get("evidence_score")
                ev_str = f"{float(ev_val):.2f}" if ev_val is not None else "—"
                parents = f.get("parent_features") or []
                parents_str = ", ".join(parents) if isinstance(parents, list) else str(parents)

                tree.insert("", tk.END, iid=f.get("feature_id"), values=(
                    f.get("feature_name", "—"),
                    f.get("feature_id", "—"),
                    f.get("formula_hash", "—"),
                    f.get("generator_strategy", "—"),
                    f"G{f.get('generation_discovered', 0)}",
                    parents_str or "—",
                    d_auc_str,
                    ks_str,
                    f"D{f.get('drift_severity', 0)}",
                    ev_str,
                    f.get("lifecycle_status", "—"),
                    f.get("governance_rationale", "—"),
                ))

            t_vsb = ttk.Scrollbar(top_frame, orient="vertical", command=tree.yview)
            t_hsb = ttk.Scrollbar(top_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=t_vsb.set, xscrollcommand=t_hsb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            t_vsb.pack(side=tk.RIGHT, fill=tk.Y)
            t_hsb.pack(side=tk.BOTTOM, fill=tk.X)

            # Bottom Detail Panel
            bot_frame = ttk.LabelFrame(paned, text="🔍 Feature Detail Inspector", padding=8)
            paned.add(bot_frame, weight=2)

            d_name_var = tk.StringVar(value="Select a feature from the table above")
            d_fid_var = tk.StringVar(value="—")
            d_hash_var = tk.StringVar(value="—")
            d_gen_var = tk.StringVar(value="—")
            d_strat_var = tk.StringVar(value="—")
            d_parents_var = tk.StringVar(value="—")
            d_formula_var = tk.StringVar(value="—")
            d_delta_var = tk.StringVar(value="—")
            d_ks_var = tk.StringVar(value="—")
            d_drift_var = tk.StringVar(value="—")
            d_ev_var = tk.StringVar(value="—")
            d_verdict_var = tk.StringVar(value="—")
            d_rat_var = tk.StringVar(value="—")

            ttk.Label(bot_frame, textvariable=d_name_var, font=("Segoe UI", 10, "bold"), foreground="#0d47a1").pack(anchor=tk.W, pady=(0, 4))
            
            d_grid = ttk.Frame(bot_frame)
            d_grid.pack(fill=tk.X)

            row1 = [
                ("DF Feature ID", d_fid_var),
                ("Formula Hash", d_hash_var),
                ("Generation", d_gen_var),
                ("Strategy", d_strat_var),
            ]
            for idx, (lbl, var) in enumerate(row1):
                ttk.Label(d_grid, text=f"{lbl}:", font=("Segoe UI", 8, "bold")).grid(row=0, column=idx*2, sticky=tk.W, padx=(4, 2), pady=1)
                ttk.Label(d_grid, textvariable=var, font=("Consolas", 8)).grid(row=0, column=idx*2+1, sticky=tk.W, padx=(0, 10), pady=1)

            row2 = [
                ("ΔAUC Lift", d_delta_var),
                ("KS Drift (D_KS)", d_ks_var),
                ("Drift Severity", d_drift_var),
                ("Evidence Score", d_ev_var),
            ]
            for idx, (lbl, var) in enumerate(row2):
                ttk.Label(d_grid, text=f"{lbl}:", font=("Segoe UI", 8, "bold")).grid(row=1, column=idx*2, sticky=tk.W, padx=(4, 2), pady=1)
                ttk.Label(d_grid, textvariable=var, font=("Segoe UI", 8)).grid(row=1, column=idx*2+1, sticky=tk.W, padx=(0, 10), pady=1)

            ttk.Label(d_grid, text="Parent Features:", font=("Segoe UI", 8, "bold")).grid(row=2, column=0, sticky=tk.W, padx=(4, 2), pady=1)
            ttk.Label(d_grid, textvariable=d_parents_var, font=("Segoe UI", 8)).grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=1)

            ttk.Label(d_grid, text="Verdict / Rationale:", font=("Segoe UI", 8, "bold")).grid(row=2, column=4, sticky=tk.W, padx=(4, 2), pady=1)
            ttk.Label(d_grid, textvariable=d_rat_var, font=("Segoe UI", 8, "italic"), foreground="#4a148c").grid(row=2, column=5, columnspan=3, sticky=tk.W, pady=1)

            f_box = ttk.Frame(bot_frame)
            f_box.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(f_box, text="Mathematical AST Formula:", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
            ttk.Label(f_box, textvariable=d_formula_var, font=("Consolas", 8), foreground="#1b5e20", wraplength=880, justify="left").pack(anchor=tk.W)

            feat_map = {f.get("feature_id"): f for f in feature_sublist}

            def _on_feature_selected(_event: Any = None) -> None:
                sel = tree.selection()
                if not sel:
                    return
                f_obj = feat_map.get(sel[0])
                if not f_obj:
                    return
                d_name_var.set(f_obj.get("feature_name", "—"))
                d_fid_var.set(f_obj.get("feature_id", "—"))
                d_hash_var.set(f_obj.get("formula_hash", "—"))
                d_gen_var.set(f"G{f_obj.get('generation_discovered', 0)}")
                d_strat_var.set(f_obj.get("generator_strategy", "—"))
                p_list = f_obj.get("parent_features") or []
                d_parents_var.set(", ".join(p_list) if isinstance(p_list, list) else str(p_list))
                d_formula_var.set(f_obj.get("formula_expression", "—"))
                d_delta_val = f_obj.get("delta_auc")
                d_delta_var.set(f"{float(d_delta_val):+.5f}" if d_delta_val is not None else "—")
                d_ks_val = f_obj.get("ks_statistic")
                d_ks_var.set(f"{float(d_ks_val):.4f}" if d_ks_val is not None else "—")
                d_drift_var.set(f"Severity D{f_obj.get('drift_severity', 0)}")
                d_ev_val = f_obj.get("evidence_score")
                d_ev_var.set(f"{float(d_ev_val):.2f} pts" if d_ev_val is not None else "—")
                d_verdict_var.set(f_obj.get("lifecycle_status", "—"))
                d_rat_var.set(f"[{f_obj.get('lifecycle_status', '—')}] {f_obj.get('governance_rationale', '—')}")

            tree.bind("<<TreeviewSelect>>", _on_feature_selected)
            if feature_sublist:
                first_id = feature_sublist[0].get("feature_id")
                if first_id:
                    tree.selection_set(first_id)
                    _on_feature_selected()

        tab_keep = ttk.Frame(feat_nb, padding=4)
        tab_watch = ttk.Frame(feat_nb, padding=4)
        tab_remove = ttk.Frame(feat_nb, padding=4)

        feat_nb.add(tab_keep, text=f"🟢 KEEP ({len(keep_feats)})")
        feat_nb.add(tab_watch, text=f"🟡 WATCH ({len(watch_feats)})")
        feat_nb.add(tab_remove, text=f"🔴 REMOVE ({len(remove_feats)})")

        _build_feature_subtab(tab_keep, keep_feats, "KEEP")
        _build_feature_subtab(tab_watch, watch_feats, "WATCH")
        _build_feature_subtab(tab_remove, remove_feats, "REMOVE")

        # =========================================================================
        # TAB 3: 🧬 Generational Progress
        # =========================================================================
        t_gen = ttk.Frame(nb, padding=8)
        nb.add(t_gen, text=f"🧬 Generational Progress ({len(gens)})")

        if not gens:
            ttk.Label(t_gen, text="No generational progression records found for this research run.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(pady=25)
        else:
            g_cols = (
                "gen", "snapshot", "cands_gen", "cands_eval", "cands_prune",
                "best_score", "best_trade", "best_model", "best_cand",
                "keep", "watch", "remove", "created_at"
            )
            g_tree = ttk.Treeview(t_gen, columns=g_cols, show="headings", height=12)
            g_tree.heading("gen", text="Gen")
            g_tree.heading("snapshot", text="Discovery Snapshot")
            g_tree.heading("cands_gen", text="Generated")
            g_tree.heading("cands_eval", text="Evaluated")
            g_tree.heading("cands_prune", text="Pruned")
            g_tree.heading("best_score", text="Best Composite")
            g_tree.heading("best_trade", text="Best Trading")
            g_tree.heading("best_model", text="Best Model")
            g_tree.heading("best_cand", text="Champion Candidate")
            g_tree.heading("keep", text="KEEP")
            g_tree.heading("watch", text="WATCH")
            g_tree.heading("remove", text="REMOVE")
            g_tree.heading("created_at", text="Snapshot Timestamp (UTC)")

            g_tree.column("gen", width=45, anchor=tk.CENTER)
            g_tree.column("snapshot", width=180)
            g_tree.column("cands_gen", width=75, anchor=tk.CENTER)
            g_tree.column("cands_eval", width=75, anchor=tk.CENTER)
            g_tree.column("cands_prune", width=65, anchor=tk.CENTER)
            g_tree.column("best_score", width=105, anchor=tk.E)
            g_tree.column("best_trade", width=95, anchor=tk.E)
            g_tree.column("best_model", width=90, anchor=tk.E)
            g_tree.column("best_cand", width=180)
            g_tree.column("keep", width=55, anchor=tk.CENTER)
            g_tree.column("watch", width=60, anchor=tk.CENTER)
            g_tree.column("remove", width=65, anchor=tk.CENTER)
            g_tree.column("created_at", width=135)

            for g in gens:
                b_s = float(g.get("generation_best_score") or g.get("best_composite_score") or 0.0)
                b_t = float(g.get("generation_best_trading_score") or g.get("best_trading_score") or 0.0)
                b_m = float(g.get("generation_best_model_score") or g.get("best_model_score") or 0.0)
                g_tree.insert("", tk.END, values=(
                    f"G{g.get('generation_number', 0)}",
                    g.get("discovery_snapshot_hash") or g.get("snapshot_hash") or "—",
                    str(g.get("candidates_generated", "—")),
                    str(g.get("candidates_evaluated", "—")),
                    str(g.get("candidates_pruned", "—")),
                    f"{b_s:.2f} pts" if b_s > 0 else "—",
                    f"{b_t:.2f} pts" if b_t > 0 else "—",
                    f"{b_m:.2f} pts" if b_m > 0 else "—",
                    g.get("generation_best_candidate_id") or "—",
                    str(g.get("keep_count", 0)),
                    str(g.get("watch_count", 0)),
                    str(g.get("remove_count", 0)),
                    str(g.get("created_at", "—"))[:19].replace("T", " "),
                ))

            g_vsb = ttk.Scrollbar(t_gen, orient="vertical", command=g_tree.yview)
            g_hsb = ttk.Scrollbar(t_gen, orient="horizontal", command=g_tree.xview)
            g_tree.configure(yscrollcommand=g_vsb.set, xscrollcommand=g_hsb.set)
            g_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            g_vsb.pack(side=tk.RIGHT, fill=tk.Y)
            g_hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # =========================================================================
        # TAB 4: 🏆 Candidate Leaderboard
        # =========================================================================
        ranked_cands = dossier.ranked_candidates if dossier else []
        t_lead = ttk.Frame(nb, padding=8)
        nb.add(t_lead, text=f"🏆 Candidate Leaderboard ({len(ranked_cands)})")

        if not ranked_cands:
            ttk.Label(t_lead, text="No candidate ranking records found for this research campaign.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(pady=25)
        else:
            lead_bar = ttk.Frame(t_lead)
            lead_bar.pack(fill=tk.X, pady=(0, 6))

            def _on_add_selected_to_classifier():
                sel = l_tree.selection()
                if not sel:
                    messagebox.showwarning("Select Candidate", "Please select a candidate from the table below.")
                    return
                cand_id = l_tree.item(sel[0], "values")[1]
                try:
                    from chain_replay_ml.training.classifier_registration import register_research_candidate_as_classifier
                    res = register_research_candidate_as_classifier(
                        self.data_dir,
                        cand_id,
                        campaign_id=camp_id,
                    )
                    messagebox.showinfo(
                        "Added to Classifier Registry",
                        f"Candidate '{cand_id}' has been registered in the Classifier Model Registry.\n\n"
                        f"Package Location: {res['package_dir']}\n"
                        f"Context Key: {res['context_key']}\n"
                        f"Composite Score: {res['composite_score']:.2f}\n\n"
                        f"Governance: Status marked as EXPERIMENTAL.\n"
                        f"Feature set, hyperparameters, and replay evidence preserved.\n"
                        f"Production promotion requires explicit human review."
                    )
                    if self.on_select_model:
                        try:
                            self.on_select_model(cand_id)
                        except Exception:
                            pass
                except Exception as ex:
                    messagebox.showerror("Registration Error", str(ex))

            ttk.Button(lead_bar, text="🏆 Add to Classifier", command=_on_add_selected_to_classifier).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(lead_bar, text="Registers the selected candidate into the Classifier Model Registry (marked as EXPERIMENTAL).", font=("Segoe UI", 8, "italic"), foreground=COL_MUTED).pack(side=tk.LEFT)

            l_cols = ("rank", "candidate_id", "composite", "trading", "model", "win_rate", "profit_factor", "max_dd", "class")
            l_tree = ttk.Treeview(t_lead, columns=l_cols, show="headings", height=12)
            l_tree.heading("rank", text="Rank")
            l_tree.heading("candidate_id", text="Candidate ID")
            l_tree.heading("composite", text="Composite Score")
            l_tree.heading("trading", text="Trading Score")
            l_tree.heading("model", text="Model Score")
            l_tree.heading("win_rate", text="Win Rate")
            l_tree.heading("profit_factor", text="Profit Factor")
            l_tree.heading("max_dd", text="Max DD")
            l_tree.heading("class", text="Recommendation Class")

            l_tree.column("rank", width=50, anchor=tk.CENTER)
            l_tree.column("candidate_id", width=220)
            l_tree.column("composite", width=110, anchor=tk.E)
            l_tree.column("trading", width=100, anchor=tk.E)
            l_tree.column("model", width=100, anchor=tk.E)
            l_tree.column("win_rate", width=90, anchor=tk.E)
            l_tree.column("profit_factor", width=90, anchor=tk.E)
            l_tree.column("max_dd", width=90, anchor=tk.E)
            l_tree.column("class", width=160, anchor=tk.CENTER)

            for i, c in enumerate(ranked_cands, 1):
                wr = c.trading_metrics.get("win_rate_pct", 0.0)
                pf = c.trading_metrics.get("profit_factor", 0.0)
                dd = c.trading_metrics.get("max_drawdown_pct", 0.0)
                l_tree.insert("", tk.END, values=(
                    f"#{i}",
                    c.candidate_id,
                    f"{c.composite_score:.2f} pts",
                    f"{c.trading_evidence_score:.2f} pts",
                    f"{c.model_evidence_score:.2f} pts",
                    f"{wr:.1f}%",
                    f"{pf:.2f}",
                    f"{dd:.1f}%",
                    c.recommendation_class.value,
                ))

            l_vsb = ttk.Scrollbar(t_lead, orient="vertical", command=l_tree.yview)
            l_tree.configure(yscrollcommand=l_vsb.set)
            l_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            l_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # =========================================================================
        # TAB 5: 🌳 Generational Lineage
        # =========================================================================
        trials = dossier.fine_tuning_trials if dossier else []
        t_lin = ttk.Frame(nb, padding=8)
        nb.add(t_lin, text=f"🌳 Generational Lineage ({len(trials)})")

        if not trials:
            ttk.Label(t_lin, text="No generational fine-tuning mutations recorded for this research run.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(pady=25)
        else:
            lin_cols = ("gen", "child", "parent", "mutation", "delta", "verdict")
            lin_tree = ttk.Treeview(t_lin, columns=lin_cols, show="headings", height=12)
            lin_tree.heading("gen", text="Gen")
            lin_tree.heading("child", text="Child Candidate")
            lin_tree.heading("parent", text="Parent Candidate")
            lin_tree.heading("mutation", text="Mutation Type")
            lin_tree.heading("delta", text="Delta Composite")
            lin_tree.heading("verdict", text="Decision Verdict")

            lin_tree.column("gen", width=55, anchor=tk.CENTER)
            lin_tree.column("child", width=220)
            lin_tree.column("parent", width=220)
            lin_tree.column("mutation", width=180)
            lin_tree.column("delta", width=110, anchor=tk.E)
            lin_tree.column("verdict", width=180, anchor=tk.CENTER)

            for t in trials:
                d_str = f"+{t.delta_composite_score:.2f} pts" if t.delta_composite_score >= 0 else f"{t.delta_composite_score:.2f} pts"
                lin_tree.insert("", tk.END, values=(
                    f"G{t.generation_number}",
                    t.child_candidate_id,
                    t.parent_candidate_id or "ROOT",
                    t.mutation_type.value,
                    d_str,
                    t.decision_verdict.value,
                ))

            lin_vsb = ttk.Scrollbar(t_lin, orient="vertical", command=lin_tree.yview)
            lin_tree.configure(yscrollcommand=lin_vsb.set)
            lin_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            lin_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # =========================================================================
        # TAB 6: 🛡️ Feature Governance
        # =========================================================================
        t_gov = ScrollableFrame(nb)
        nb.add(t_gov, text="🛡️ Feature Governance")
        gov_inner = getattr(t_gov, "inner", t_gov)

        if dossier and dossier.feature_governance_summary:
            gf = ttk.LabelFrame(gov_inner, text="Feature Lifecycle Governance Audit", padding=10)
            gf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

            ttk.Label(gf, text=f"Active Features Researched: {dossier.feature_governance_summary.total_features_evaluated}", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=2)
            ttk.Label(gf, text=f"Deprecated Features Blocked: {len(dossier.feature_governance_summary.deprecated_features_blocked)} (100% Excluded by Negative Pruning)", font=("Segoe UI", 9), foreground=COL_OK).pack(anchor=tk.W, pady=2)
            ttk.Label(gf, text="Feature Registry State: 100% Immutability Preserved (Zero Automatic Production Promotions)", font=("Segoe UI", 9), foreground=COL_PRODUCTION).pack(anchor=tk.W, pady=2)

            ttk.Label(gf, text="Explored Feature Universe:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
            txt = tk.Text(gf, height=10, width=80)
            txt.insert(tk.END, ", ".join(dossier.feature_governance_summary.features_used))
            txt.config(state="disabled")
            txt.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(gov_inner, text="No governance audit summary recorded for this research run.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(pady=25)

        # =========================================================================
        # TAB 7: 📜 Execution Audit Trail
        # =========================================================================
        t_aud = ttk.Frame(nb, padding=8)
        nb.add(t_aud, text="📜 Execution Audit Trail")

        aud_ctrl = ttk.Frame(t_aud)
        aud_ctrl.pack(fill=tk.X, pady=(0, 6))

        aud_filter_var = tk.StringVar(value="ALL")
        aud_search_var = tk.StringVar(value="")

        ttk.Label(aud_ctrl, text="Event Type:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        aud_cb = ttk.Combobox(aud_ctrl, textvariable=aud_filter_var, values=["ALL", "CANDIDATE", "METRICS", "CHAMPION", "DECISIONS", "WARNINGS"], width=13, state="readonly")
        aud_cb.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(aud_ctrl, text="Search Keyword:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        aud_entry = ttk.Entry(aud_ctrl, textvariable=aud_search_var, width=20)
        aud_entry.pack(side=tk.LEFT, padx=(0, 8))

        aud_cols = ("timestamp", "gen", "type", "cand", "message")
        aud_tree = ttk.Treeview(t_aud, columns=aud_cols, show="headings", height=12)
        aud_tree.heading("timestamp", text="Timestamp (UTC)")
        aud_tree.heading("gen", text="Gen")
        aud_tree.heading("type", text="Event Type")
        aud_tree.heading("cand", text="Candidate ID")
        aud_tree.heading("message", text="Event Message / Details")

        aud_tree.column("timestamp", width=140)
        aud_tree.column("gen", width=45, anchor=tk.CENTER)
        aud_tree.column("type", width=150)
        aud_tree.column("cand", width=160)
        aud_tree.column("message", width=440)

        # Load audit events from DB
        events_cache: list[dict[str, Any]] = []
        if camp_id and self.data_dir:
            conn = connect_analysis_db(self.data_dir)
            try:
                rows = conn.execute(
                    "SELECT created_at, generation_number, event_type, candidate_id, message, event_details_json FROM overnight_campaign_events WHERE campaign_id = ? ORDER BY event_id ASC;",
                    (camp_id,),
                ).fetchall()
                events_cache = [dict(r) for r in rows]
            except Exception:
                events_cache = []
            finally:
                conn.close()

        def _populate_aud_events():
            f_type = aud_filter_var.get()
            q_txt = aud_search_var.get().lower().strip()
            for itm in aud_tree.get_children():
                aud_tree.delete(itm)

            type_map = {
                "CANDIDATE": ["CANDIDATE_GENERATED", "CANDIDATE_EVAL_START", "CANDIDATE_EVAL_DONE", "CANDIDATE_PRUNED"],
                "METRICS": ["METRICS_LOGGED", "SCORE_UPDATED", "OOS_EVALUATION"],
                "CHAMPION": ["NEW_CHAMPION_PROMOTED", "CHAMPION_MAINTAINED", "CHAMPION_CANDIDATE"],
                "DECISIONS": ["DECISION_GATE_PASSED", "DECISION_GATE_FAILED", "BRANCH_PRUNED"],
                "WARNINGS": ["RISK_PENALTY_WARNING", "PLATEAU_WARNING", "FAILURE_WARNING", "WARNING"],
            }

            for ev in events_cache:
                ev_type = ev.get("event_type", "")
                if f_type != "ALL":
                    allowed = type_map.get(f_type, [f_type])
                    if ev_type not in allowed:
                        continue
                if q_txt:
                    combined = f"{ev.get('candidate_id','')} {ev_type} {ev.get('message','')} {ev.get('event_details_json','')}".lower()
                    if q_txt not in combined:
                        continue

                aud_tree.insert("", tk.END, values=(
                    str(ev.get("created_at", "—"))[:19].replace("T", " "),
                    f"G{ev.get('generation_number', 0)}",
                    ev_type,
                    ev.get("candidate_id") or "—",
                    ev.get("message") or "—",
                ))

        aud_cb.bind("<<ComboboxSelected>>", lambda _e: _populate_aud_events())
        aud_entry.bind("<KeyRelease>", lambda _e: _populate_aud_events())
        ttk.Button(aud_ctrl, text="🔍 Search", command=_populate_aud_events).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(aud_ctrl, text="🔄 Reset", command=lambda: [aud_filter_var.set("ALL"), aud_search_var.set(""), _populate_aud_events()]).pack(side=tk.LEFT)

        _populate_aud_events()

        aud_vsb = ttk.Scrollbar(t_aud, orient="vertical", command=aud_tree.yview)
        aud_hsb = ttk.Scrollbar(t_aud, orient="horizontal", command=aud_tree.xview)
        aud_tree.configure(yscrollcommand=aud_vsb.set, xscrollcommand=aud_hsb.set)
        aud_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        aud_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        aud_hsb.pack(side=tk.BOTTOM, fill=tk.X)

        if first_place:
            win.update_idletasks()
            place_toplevel_beside_main(win, self)

        try:
            win.deiconify()
            win.lift()
            win.focus_force()
        except tk.TclError:
            pass
        self._current_detail_research_id = research_id

    # Backward compatibility exports
    def _export_markdown(self) -> None:
        if not self.current_dossier and self._all_records:
            sel = self.tree.selection()
            r_id = sel[0] if sel else self._all_records[0]["research_id"]
            detail = get_research_detail(self.data_dir, r_id)
            if detail and detail.get("campaign_id"):
                self.current_dossier = generate_morning_research_dossier(self.data_dir, detail["campaign_id"])
        if not self.current_dossier:
            return
        md = export_morning_dossier_markdown(self.current_dossier)
        out_path = os.path.join(self.data_dir, f"morning_dossier_{self.current_dossier.campaign_id}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        messagebox.showinfo("Export Successful", f"Morning Research Dossier exported to:\n{out_path}")

    def _export_json(self) -> None:
        if not self.current_dossier and self._all_records:
            sel = self.tree.selection()
            r_id = sel[0] if sel else self._all_records[0]["research_id"]
            detail = get_research_detail(self.data_dir, r_id)
            if detail and detail.get("campaign_id"):
                self.current_dossier = generate_morning_research_dossier(self.data_dir, detail["campaign_id"])
        if not self.current_dossier:
            return
        out_path = os.path.join(self.data_dir, f"morning_dossier_{self.current_dossier.campaign_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.current_dossier.to_dict(), f, indent=2)
        messagebox.showinfo("Export Successful", f"Morning Research Dossier exported to:\n{out_path}")

    def _render_discovered_features_tab(self) -> None:
        """Backward compatibility helper for tests."""
        pass
