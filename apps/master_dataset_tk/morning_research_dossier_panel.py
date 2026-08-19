"""Morning Research Dossier & Model Research Lab Panel (Phase 4F.6).

Provides an interactive GUI presenting overnight campaign summaries, candidate rankings,
generational lineage graphs, feature governance audits, and exportable research reports.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.morning_dossier import (
    MorningResearchDossier,
    export_morning_dossier_markdown,
    generate_morning_research_dossier,
)
from chain_replay_ml.overnight_campaign.persistence import init_campaign_tables
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db

from .build_service import chart_data_dir
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
    """Interactive GUI for viewing overnight research campaign dossiers and candidate lineage."""

    def __init__(self, parent: tk.Widget, *, data_dir: str | None = None) -> None:
        super().__init__(parent)
        self.data_dir = data_dir or chart_data_dir()
        self.selected_campaign_id = tk.StringVar(value="")
        self.selected_context_key = tk.StringVar(value="")
        self.current_dossier: MorningResearchDossier | None = None

        init_analysis_db(self.data_dir)
        init_campaign_tables(self.data_dir)

        self._build_layout()
        self._refresh_campaign_list()

    def _build_layout(self) -> None:
        """Construct the top control toolbar, KPI card area, and notebook tabs."""
        # Top toolbar
        toolbar = ttk.Frame(self, padding=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(toolbar, text="Campaign:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.campaign_combo = ttk.Combobox(toolbar, textvariable=self.selected_campaign_id, width=30, state="readonly")
        self.campaign_combo.pack(side=tk.LEFT, padx=5)
        self.campaign_combo.bind("<<ComboboxSelected>>", lambda e: self.load_selected_campaign())

        ttk.Button(toolbar, text="🔄 Refresh", command=self._refresh_campaign_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="📋 Export Markdown", command=self._export_markdown).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="💾 Export JSON", command=self._export_json).pack(side=tk.RIGHT, padx=5)

        # Main notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_overview = ttk.Frame(self.notebook, padding=10)
        self.tab_leaderboard = ttk.Frame(self.notebook, padding=10)
        self.tab_lineage = ttk.Frame(self.notebook, padding=10)
        self.tab_governance = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_overview, text="🌅 Morning Summary")
        self.notebook.add(self.tab_leaderboard, text="🏆 Candidate Leaderboard")
        self.notebook.add(self.tab_lineage, text="🧬 Generational Lineage")
        self.notebook.add(self.tab_governance, text="🛡️ Feature Governance Audit")

    def _refresh_campaign_list(self) -> None:
        """Query analysis.db for available overnight campaigns."""
        conn = connect_analysis_db(self.data_dir)
        try:
            rows = conn.execute(
                "SELECT campaign_id FROM overnight_campaigns ORDER BY start_time_iso DESC;"
            ).fetchall()
            ids = [r["campaign_id"] for r in rows]
            self.campaign_combo["values"] = ids
            if ids and not self.selected_campaign_id.get():
                self.selected_campaign_id.set(ids[0])
                self.load_selected_campaign()
        finally:
            conn.close()

    def load_selected_campaign(self) -> None:
        """Load and display the dossier for the selected campaign."""
        camp_id = self.selected_campaign_id.get()
        if not camp_id:
            return

        try:
            self.current_dossier = generate_morning_research_dossier(self.data_dir, camp_id)
            self._render_overview_tab()
            self._render_leaderboard_tab()
            self._render_lineage_tab()
            self._render_governance_tab()
        except Exception as ex:
            messagebox.showerror("Error Loading Dossier", str(ex))

    def _render_overview_tab(self) -> None:
        """Render Overview summary cards, KPIs, and recommended next actions."""
        clear_children(self.tab_overview)
        d = self.current_dossier
        if not d:
            ttk.Label(self.tab_overview, text="No campaign selected.").pack(pady=20)
            return

        # Header Title
        ttk.Label(self.tab_overview, text=f"Morning Research Dossier: {d.campaign_id}", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(self.tab_overview, text=f"Context: {d.context_key} | Status: {d.campaign_status.value} | Stop: {d.stop_reason.value}", font=("Segoe UI", 9), foreground=COL_MUTED).pack(anchor=tk.W, pady=(0, 10))

        # KPI Block
        kpi_frame = ttk.LabelFrame(self.tab_overview, text="Executive Research KPIs", padding=10)
        kpi_frame.pack(fill=tk.X, pady=5)

        kpis = [
            ("Top Candidate", d.best_candidate_id or "None"),
            ("Best Composite Score", f"{d.best_composite_score:.2f} / 100.0"),
            ("Total Lift Achieved", f"+{d.total_score_improvement:.2f} pts"),
            ("Trading Score", f"{d.best_trading_score:.2f} / 100.0"),
            ("Statistical Model Score", f"{d.best_model_score:.2f} / 100.0"),
            ("Win Rate (OOS Replay)", f"{d.best_win_rate_pct:.1f}%"),
            ("Profit Factor", f"{d.best_profit_factor:.2f}"),
            ("Max Drawdown", f"{d.best_max_drawdown_pct:.1f}%"),
            ("Generations Completed", str(d.total_generations_completed)),
            ("Candidates Trained / Pruned", f"{d.total_candidates_trained} / {d.total_candidates_pruned}"),
        ]
        kv_block(kpi_frame, kpis, num_cols=2)

        # Recommended Next Actions
        act_frame = ttk.LabelFrame(self.tab_overview, text="Recommended Next Research Actions", padding=10)
        act_frame.pack(fill=tk.X, pady=10)
        for act in d.recommended_next_actions:
            ttk.Label(act_frame, text=f"• {act}", font=("Segoe UI", 9, "bold" if "CRITICAL" in act else "normal"), foreground=COL_PRODUCTION if "CRITICAL" in act else "black").pack(anchor=tk.W, pady=2)

    def _render_leaderboard_tab(self) -> None:
        """Render Candidate Leaderboard Treeview."""
        clear_children(self.tab_leaderboard)
        d = self.current_dossier
        if not d or not d.ranked_candidates:
            ttk.Label(self.tab_leaderboard, text="No candidates ranked.").pack(pady=20)
            return

        cols = ("rank", "candidate_id", "composite", "trading", "model", "win_rate", "profit_factor", "max_dd", "class")
        tree = ttk.Treeview(self.tab_leaderboard, columns=cols, show="headings", height=15)
        tree.heading("rank", text="Rank")
        tree.heading("candidate_id", text="Candidate ID")
        tree.heading("composite", text="Composite Score")
        tree.heading("trading", text="Trading Score")
        tree.heading("model", text="Model Score")
        tree.heading("win_rate", text="Win Rate")
        tree.heading("profit_factor", text="Profit Factor")
        tree.heading("max_dd", text="Max DD")
        tree.heading("class", text="Recommendation Class")

        tree.column("rank", width=50, anchor=tk.CENTER)
        tree.column("candidate_id", width=220)
        tree.column("composite", width=110, anchor=tk.E)
        tree.column("trading", width=100, anchor=tk.E)
        tree.column("model", width=100, anchor=tk.E)
        tree.column("win_rate", width=90, anchor=tk.E)
        tree.column("profit_factor", width=90, anchor=tk.E)
        tree.column("max_dd", width=90, anchor=tk.E)
        tree.column("class", width=160, anchor=tk.CENTER)

        for i, c in enumerate(d.ranked_candidates, 1):
            wr = c.trading_metrics.get("win_rate_pct", 0.0)
            pf = c.trading_metrics.get("profit_factor", 0.0)
            dd = c.trading_metrics.get("max_drawdown_pct", 0.0)
            tree.insert("", tk.END, values=(
                f"#{i}",
                c.candidate_id,
                f"{c.composite_score:.2f}",
                f"{c.trading_evidence_score:.2f}",
                f"{c.model_evidence_score:.2f}",
                f"{wr:.1f}%",
                f"{pf:.2f}",
                f"{dd:.1f}%",
                c.recommendation_class.value,
            ))
        tree.pack(fill=tk.BOTH, expand=True)

    def _render_lineage_tab(self) -> None:
        """Render Generational Lineage & Fine-Tuning Trials Treeview."""
        clear_children(self.tab_lineage)
        d = self.current_dossier
        if not d or not d.fine_tuning_trials:
            ttk.Label(self.tab_lineage, text="No generational fine-tuning trials recorded.").pack(pady=20)
            return

        cols = ("gen", "child", "parent", "mutation", "delta", "verdict")
        tree = ttk.Treeview(self.tab_lineage, columns=cols, show="headings", height=15)
        tree.heading("gen", text="Gen")
        tree.heading("child", text="Child Candidate")
        tree.heading("parent", text="Parent Candidate")
        tree.heading("mutation", text="Mutation Type")
        tree.heading("delta", text="Delta Composite")
        tree.heading("verdict", text="Decision Verdict")

        tree.column("gen", width=60, anchor=tk.CENTER)
        tree.column("child", width=220)
        tree.column("parent", width=220)
        tree.column("mutation", width=180)
        tree.column("delta", width=110, anchor=tk.E)
        tree.column("verdict", width=180, anchor=tk.CENTER)

        for t in d.fine_tuning_trials:
            d_str = f"+{t.delta_composite_score:.2f}" if t.delta_composite_score >= 0 else f"{t.delta_composite_score:.2f}"
            tree.insert("", tk.END, values=(
                f"G{t.generation_number}",
                t.child_candidate_id,
                t.parent_candidate_id,
                t.mutation_type.value,
                d_str,
                t.decision_verdict.value,
            ))
        tree.pack(fill=tk.BOTH, expand=True)

    def _render_governance_tab(self) -> None:
        """Render Feature Governance Audit view."""
        clear_children(self.tab_governance)
        d = self.current_dossier
        if not d:
            return

        gf = ttk.LabelFrame(self.tab_governance, text="Feature Lifecycle Governance Audit", padding=10)
        gf.pack(fill=tk.BOTH, expand=True)

        ttk.Label(gf, text=f"Active Features Researched: {d.feature_governance_summary.total_features_evaluated}", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Label(gf, text=f"Deprecated Features Blocked: {len(d.feature_governance_summary.deprecated_features_blocked)} (100% Excluded by Negative Pruning)", font=("Segoe UI", 9), foreground=COL_OK).pack(anchor=tk.W, pady=2)
        ttk.Label(gf, text="Feature Registry State: 100% Immutability Preserved (Zero Automatic Production Promotions)", font=("Segoe UI", 9), foreground=COL_PRODUCTION).pack(anchor=tk.W, pady=2)

        ttk.Label(gf, text="Explored Feature List:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
        txt = tk.Text(gf, height=10, width=80)
        txt.insert(tk.END, ", ".join(d.feature_governance_summary.features_used))
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True)

    def _export_markdown(self) -> None:
        """Export current dossier as Markdown."""
        if not self.current_dossier:
            return
        md = export_morning_dossier_markdown(self.current_dossier)
        out_path = os.path.join(self.data_dir, f"morning_dossier_{self.current_dossier.campaign_id}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        messagebox.showinfo("Export Successful", f"Morning Research Dossier exported to:\n{out_path}")

    def _export_json(self) -> None:
        """Export current dossier as JSON."""
        if not self.current_dossier:
            return
        out_path = os.path.join(self.data_dir, f"morning_dossier_{self.current_dossier.campaign_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.current_dossier.to_dict(), f, indent=2)
        messagebox.showinfo("Export Successful", f"Morning Research Dossier exported to:\n{out_path}")
