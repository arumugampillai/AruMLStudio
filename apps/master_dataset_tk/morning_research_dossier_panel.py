"""Morning Research Dossier & Model Research Lab Panel (Phase 4F.6).

Provides an interactive GUI presenting overnight campaign summaries, candidate rankings,
generational lineage graphs, feature governance audits, discovered feature intelligence,
feature synergy discoveries, and candidate mutation drill-down analytics.
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
        self.selected_campaign_id = tk.StringVar(value="")
        self.selected_context_key = tk.StringVar(value="")
        self.current_dossier: MorningResearchDossier | None = None
        self._selected_drilldown_candidate = tk.StringVar(value="")


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

        self.tab_overview = ScrollableFrame(self.notebook)
        self.tab_discovered_features = ScrollableFrame(self.notebook)
        self.tab_leaderboard = ttk.Frame(self.notebook, padding=10)
        self.tab_lineage = ttk.Frame(self.notebook, padding=10)
        self.tab_governance = ScrollableFrame(self.notebook)
        self.tab_audit_trail = ttk.Frame(self.notebook, padding=10)

        self._audit_filter_var = tk.StringVar(value="ALL")
        self._audit_search_var = tk.StringVar(value="")
        self._audit_events_cache: list[dict[str, Any]] = []

        self.notebook.add(self.tab_overview, text="🌅 Morning Summary")
        self.notebook.add(self.tab_discovered_features, text="⭐ Discovered Features")
        self.notebook.add(self.tab_leaderboard, text="🏆 Candidate Leaderboard")
        self.notebook.add(self.tab_lineage, text="🧬 Generational Lineage")
        self.notebook.add(self.tab_governance, text="🛡️ Feature Governance Audit")
        self.notebook.add(self.tab_audit_trail, text="📜 Execution Audit Trail")

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
            if self.current_dossier.ranked_candidates:
                self._selected_drilldown_candidate.set(self.current_dossier.ranked_candidates[0].candidate_id)

            self._render_overview_tab()
            self._render_discovered_features_tab()
            self._render_leaderboard_tab()
            self._render_lineage_tab()
            self._render_governance_tab()
            self._render_audit_trail_tab()
        except Exception as ex:
            messagebox.showerror("Error Loading Dossier", str(ex))


    def _render_overview_tab(self) -> None:
        """Render Overview summary cards, KPIs, and recommended next actions."""
        target = getattr(self.tab_overview, "inner", self.tab_overview)
        clear_children(target)
        d = self.current_dossier
        if not d:
            ttk.Label(target, text="No campaign selected.").pack(pady=20)
            return

        # Header Title
        ttk.Label(target, text=f"Morning Research Dossier: {d.campaign_id}", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(target, text=f"Context: {d.context_key} | Status: {d.campaign_status.value} | Stop: {d.stop_reason.value}", font=("Segoe UI", 9), foreground=COL_MUTED).pack(anchor=tk.W, pady=(0, 10))

        # KPI Block using authoritative kv_block(parent, title, items) signature
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
        kv_block(target, "Executive Research KPIs", kpis)

        # Recommended Next Actions
        if d.recommended_next_actions:
            act_frame = ttk.LabelFrame(target, text="Recommended Next Research Actions", padding=10)
            act_frame.pack(fill=tk.X, pady=10)
            for act in d.recommended_next_actions:
                ttk.Label(act_frame, text=f"• {act}", font=("Segoe UI", 9, "bold" if "CRITICAL" in act else "normal"), foreground=COL_PRODUCTION if "CRITICAL" in act else "black").pack(anchor=tk.W, pady=2)

        # Champion Registration Action
        if d.best_candidate_id:
            reg_frame = ttk.LabelFrame(target, text="Model Registry Integration", padding=10)
            reg_frame.pack(fill=tk.X, pady=(0, 10))

            def _on_add_champion_to_classifier():
                try:
                    from chain_replay_ml.training.classifier_registration import register_research_candidate_as_classifier
                    res = register_research_candidate_as_classifier(
                        self.data_dir,
                        d.best_candidate_id,
                        campaign_id=d.campaign_id,
                    )
                    messagebox.showinfo(
                        "Added to Classifier Registry",
                        f"Top Candidate '{d.best_candidate_id}' has been registered in the Classifier Model Registry.\n\n"
                        f"Package Location: {res['package_dir']}\n"
                        f"Context Key: {res['context_key']}\n"
                        f"Composite Score: {res['composite_score']:.2f}\n\n"
                        f"Governance: Status marked as EXPERIMENTAL.\n"
                        f"Lineage, features, hyperparameters, and replay evidence preserved.\n"
                        f"Production promotion requires explicit human review."
                    )
                    if self.on_select_model:
                        try:
                            self.on_select_model(d.best_candidate_id)
                        except Exception:
                            pass
                except Exception as ex:
                    messagebox.showerror("Registration Error", str(ex))


            ttk.Button(
                reg_frame,
                text=f"🏆 Add Winner ({d.best_candidate_id}) to Classifier",
                command=_on_add_champion_to_classifier,
            ).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(
                reg_frame,
                text="Registers this candidate into the Classifier Model Registry without modifying production models.",
                font=("Segoe UI", 9, "italic"),
                foreground=COL_MUTED,
            ).pack(side=tk.LEFT)


    def _render_discovered_features_tab(self) -> None:
        """Render Discovered Feature Intelligence, Synergies, and Mutation Drill-Down."""
        target = getattr(self.tab_discovered_features, "inner", self.tab_discovered_features)
        clear_children(target)
        d = self.current_dossier
        if not d:
            ttk.Label(target, text="No campaign selected.").pack(pady=20)
            return

        # 1. Summary Cards Header
        strong_cnt = sum(1 for f in d.discovered_features if f.status == DiscoveredFeatureStatus.STRONG_DISCOVERED)
        prom_cnt = sum(1 for f in d.discovered_features if f.status == DiscoveredFeatureStatus.PROMISING)
        harm_cnt = sum(1 for f in d.discovered_features if f.status == DiscoveredFeatureStatus.REJECTED_HARMFUL)
        syn_cnt = len(d.discovered_synergies)

        sum_frame = ttk.LabelFrame(target, text="Feature Discovery Empirical Summary", padding=8)
        sum_frame.pack(fill=tk.X, pady=(0, 10))

        s_row = ttk.Frame(sum_frame)
        s_row.pack(fill=tk.X)

        ttk.Label(s_row, text=f"🟢 Strong Features: {strong_cnt}", font=("Segoe UI", 10, "bold"), foreground="#2e7d32").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(s_row, text=f"🟡 Promising: {prom_cnt}", font=("Segoe UI", 10, "bold"), foreground="#f57f17").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(s_row, text=f"🔴 Rejected/Harmful: {harm_cnt}", font=("Segoe UI", 10, "bold"), foreground="#c62828").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(s_row, text=f"⚡ Synergies Discovered: {syn_cnt}", font=("Segoe UI", 10, "bold"), foreground="#1565c0").pack(side=tk.LEFT, padx=(0, 16))

        # Governance Notice
        ttk.Label(
            sum_frame,
            text="⚠️ DISCOVERED — HUMAN REVIEW REQUIRED: Features are not automatically promoted to production.",
            font=("Segoe UI", 8, "italic"),
            foreground=COL_MUTED,
        ).pack(anchor=tk.W, pady=(4, 0))

        # 2. Discovered Features Treeview
        feat_frame = ttk.LabelFrame(target, text="⭐ Discovered Feature Rankings (Ranked by Empirical Candidate Lift)", padding=8)
        feat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ("status", "feature", "tested", "pos_desc", "best_comp", "best_trade", "avg_comp", "best_delta", "p4e_ev", "rec")
        tree = ttk.Treeview(feat_frame, columns=cols, show="headings", height=10)
        tree.heading("status", text="Category")
        tree.heading("feature", text="Feature Name")
        tree.heading("tested", text="Tested")
        tree.heading("pos_desc", text="Pos. Children")
        tree.heading("best_comp", text="Best Score")
        tree.heading("best_trade", text="Best Trading")
        tree.heading("avg_comp", text="Avg Score")
        tree.heading("best_delta", text="Best Δ vs Parent")
        tree.heading("p4e_ev", text="Phase 4E Ev.")
        tree.heading("rec", text="Governance Recommendation")

        tree.column("status", width=140, anchor=tk.CENTER)
        tree.column("feature", width=180)
        tree.column("tested", width=60, anchor=tk.CENTER)
        tree.column("pos_desc", width=90, anchor=tk.CENTER)
        tree.column("best_comp", width=85, anchor=tk.E)
        tree.column("best_trade", width=85, anchor=tk.E)
        tree.column("avg_comp", width=85, anchor=tk.E)
        tree.column("best_delta", width=105, anchor=tk.E)
        tree.column("p4e_ev", width=95, anchor=tk.CENTER)
        tree.column("rec", width=220, anchor=tk.CENTER)

        for f in d.discovered_features:
            cat_icon = "🟢 STRONG" if f.status == DiscoveredFeatureStatus.STRONG_DISCOVERED else ("🟡 PROMISING" if f.status == DiscoveredFeatureStatus.PROMISING else "🔴 REJECTED")
            d_str = f"+{f.best_delta_vs_parent:.2f}" if f.best_delta_vs_parent >= 0 else f"{f.best_delta_vs_parent:.2f}"
            tree.insert("", tk.END, values=(
                cat_icon,
                f.feature_name,
                str(f.times_tested),
                str(f.positive_descendant_count),
                f"{f.best_composite_score:.2f}",
                f"{f.best_trading_score:.2f}",
                f"{f.avg_composite_score:.2f}",
                d_str,
                f.phase4e_evidence_level,
                f.recommendation,
            ))
        tree.pack(fill=tk.BOTH, expand=True)

        # 3. Feature Synergy Discoveries Table
        if d.discovered_synergies:
            syn_frame = ttk.LabelFrame(target, text="⚡ Feature Synergy Discoveries (Pairwise Interactions)", padding=8)
            syn_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            s_cols = ("feat_a", "feat_b", "tested", "d_comp", "d_trade", "cross_reg", "status")
            s_tree = ttk.Treeview(syn_frame, columns=s_cols, show="headings", height=6)
            s_tree.heading("feat_a", text="Feature A")
            s_tree.heading("feat_b", text="Feature B")
            s_tree.heading("tested", text="Times Tested")
            s_tree.heading("d_comp", text="Best Δ Composite")
            s_tree.heading("d_trade", text="Best Δ Trading")
            s_tree.heading("cross_reg", text="Cross-Regime")
            s_tree.heading("status", text="Synergy Status")

            s_tree.column("feat_a", width=180)
            s_tree.column("feat_b", width=180)
            s_tree.column("tested", width=90, anchor=tk.CENTER)
            s_tree.column("d_comp", width=110, anchor=tk.E)
            s_tree.column("d_trade", width=110, anchor=tk.E)
            s_tree.column("cross_reg", width=110, anchor=tk.CENTER)
            s_tree.column("status", width=180, anchor=tk.CENTER)

            for s in d.discovered_synergies[:15]:
                d_c = f"+{s.best_delta_composite:.2f}" if s.best_delta_composite >= 0 else f"{s.best_delta_composite:.2f}"
                d_t = f"+{s.best_delta_trading:.2f}" if s.best_delta_trading >= 0 else f"{s.best_delta_trading:.2f}"
                s_tree.insert("", tk.END, values=(
                    s.feature_a,
                    s.feature_b,
                    str(s.times_tested),
                    d_c,
                    d_t,
                    s.cross_regime_evidence,
                    s.status,
                ))
            s_tree.pack(fill=tk.BOTH, expand=True)

        # 4. Candidate Feature Mutation Drill-Down Panel
        if d.candidate_feature_deltas:
            drill_frame = ttk.LabelFrame(target, text="🔍 Candidate Feature Mutation Drill-Down", padding=8)
            drill_frame.pack(fill=tk.X, pady=(0, 10))

            top_row = ttk.Frame(drill_frame)
            top_row.pack(fill=tk.X, pady=(0, 6))

            ttk.Label(top_row, text="Inspect Candidate:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
            cand_ids = [v.candidate_id for v in d.candidate_feature_deltas]
            cb = ttk.Combobox(top_row, textvariable=self._selected_drilldown_candidate, values=cand_ids, width=32, state="readonly")
            cb.pack(side=tk.LEFT, padx=(0, 10))

            detail_container = ttk.Frame(drill_frame)
            detail_container.pack(fill=tk.X)

            def _update_drill_view(_e=None):
                clear_children(detail_container)
                sel_id = self._selected_drilldown_candidate.get()
                v = next((item for item in d.candidate_feature_deltas if item.candidate_id == sel_id), None)
                if not v:
                    return

                d_comp_str = f"+{v.delta_composite:.2f}" if v.delta_composite >= 0 else f"{v.delta_composite:.2f}"
                d_trade_str = f"+{v.delta_trading:.2f}" if v.delta_trading >= 0 else f"{v.delta_trading:.2f}"
                d_model_str = f"+{v.delta_model:.2f}" if v.delta_model >= 0 else f"{v.delta_model:.2f}"

                # Score row
                sc_row = ttk.Frame(detail_container)
                sc_row.pack(fill=tk.X, pady=(0, 4))
                ttk.Label(sc_row, text=f"Parent Candidate: {v.parent_candidate_id or 'Root Initial Spec'}", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 16))
                ttk.Label(sc_row, text=f"Composite: {v.composite_score:.2f} (Δ: {d_comp_str})", font=("Segoe UI", 9, "bold"), foreground=COL_PRODUCTION).pack(side=tk.LEFT, padx=(0, 16))
                ttk.Label(sc_row, text=f"Trading: {v.trading_score:.2f} (Δ: {d_trade_str})", font=("Segoe UI", 9), foreground="#e65100").pack(side=tk.LEFT, padx=(0, 16))
                ttk.Label(sc_row, text=f"Model: {v.model_score:.2f} (Δ: {d_model_str})", font=("Segoe UI", 9), foreground="#0d47a1").pack(side=tk.LEFT, padx=(0, 16))

                # Added / Removed Features Row
                ft_row = ttk.Frame(detail_container)
                ft_row.pack(fill=tk.X, pady=(2, 4))

                added_str = ", ".join(f"[+] {f}" for f in v.added_features) if v.added_features else "None"
                removed_str = ", ".join(f"[-] {f}" for f in v.removed_features) if v.removed_features else "None"

                ttk.Label(ft_row, text="Added Features:", font=("Segoe UI", 9, "bold"), foreground="#2e7d32").pack(side=tk.LEFT, padx=(0, 4))
                ttk.Label(ft_row, text=added_str, font=("Consolas", 9, "bold"), foreground="#2e7d32").pack(side=tk.LEFT, padx=(0, 16))

                ttk.Label(ft_row, text="Removed Features:", font=("Segoe UI", 9, "bold"), foreground="#c62828").pack(side=tk.LEFT, padx=(0, 4))
                ttk.Label(ft_row, text=removed_str, font=("Consolas", 9), foreground="#c62828").pack(side=tk.LEFT, padx=(0, 16))

                # Metrics row
                met_row = ttk.Frame(detail_container)
                met_row.pack(fill=tk.X, pady=(2, 0))
                ttk.Label(met_row, text=f"OOS Trading Metrics: WinRate: {v.win_rate_pct:.1f}% | Profit Factor: {v.profit_factor:.2f} | Max Drawdown: {v.max_drawdown_pct:.1f}%", font=("Segoe UI", 9), foreground=COL_MUTED).pack(side=tk.LEFT)

            cb.bind("<<ComboboxSelected>>", _update_drill_view)
            _update_drill_view()

    def _render_leaderboard_tab(self) -> None:
        """Render Candidate Leaderboard Treeview with action bar."""
        clear_children(self.tab_leaderboard)
        d = self.current_dossier
        if not d or not d.ranked_candidates:
            ttk.Label(self.tab_leaderboard, text="No candidates ranked.").pack(pady=20)
            return

        # Top Action Bar
        act_bar = ttk.Frame(self.tab_leaderboard)
        act_bar.pack(fill=tk.X, pady=(0, 6))

        def _on_add_selected_to_classifier():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Select Candidate", "Please select a candidate from the table below.")
                return
            cand_id = tree.item(sel[0], "values")[1]
            try:
                from chain_replay_ml.training.classifier_registration import register_research_candidate_as_classifier
                res = register_research_candidate_as_classifier(
                    self.data_dir,
                    cand_id,
                    campaign_id=d.campaign_id,
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


        ttk.Button(
            act_bar,
            text="🏆 Add to Classifier",
            command=_on_add_selected_to_classifier,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            act_bar,
            text="Registers the selected candidate into the Classifier Model Registry (marked as EXPERIMENTAL).",
            font=("Segoe UI", 9, "italic"),
            foreground=COL_MUTED,
        ).pack(side=tk.LEFT)

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
        target = getattr(self.tab_governance, "inner", self.tab_governance)
        clear_children(target)
        d = self.current_dossier
        if not d:
            return

        gf = ttk.LabelFrame(target, text="Feature Lifecycle Governance Audit", padding=10)
        gf.pack(fill=tk.BOTH, expand=True)

        if d.feature_governance_summary:
            ttk.Label(gf, text=f"Active Features Researched: {d.feature_governance_summary.total_features_evaluated}", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=2)
            ttk.Label(gf, text=f"Deprecated Features Blocked: {len(d.feature_governance_summary.deprecated_features_blocked)} (100% Excluded by Negative Pruning)", font=("Segoe UI", 9), foreground=COL_OK).pack(anchor=tk.W, pady=2)
            ttk.Label(gf, text="Feature Registry State: 100% Immutability Preserved (Zero Automatic Production Promotions)", font=("Segoe UI", 9), foreground=COL_PRODUCTION).pack(anchor=tk.W, pady=2)

            ttk.Label(gf, text="Explored Feature List:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
            txt = tk.Text(gf, height=10, width=80)
            txt.insert(tk.END, ", ".join(d.feature_governance_summary.features_used))
            txt.config(state="disabled")
            txt.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(gf, text="No feature governance summary available.", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED).pack(anchor=tk.W, pady=2)

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

    def _render_audit_trail_tab(self) -> None:
        """Render complete chronological execution audit trail for the selected campaign."""
        clear_children(self.tab_audit_trail)
        camp_id = self.selected_campaign_id.get()
        if not camp_id:
            ttk.Label(self.tab_audit_trail, text="No campaign selected.").pack(pady=20)
            return

        # Top filter bar
        ctrl_bar = ttk.Frame(self.tab_audit_trail, padding=(0, 0, 0, 8))
        ctrl_bar.pack(fill=tk.X)

        ttk.Label(ctrl_bar, text="Filter Event Type:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        filter_cb = ttk.Combobox(
            ctrl_bar,
            textvariable=self._audit_filter_var,
            values=["ALL", "CANDIDATE", "METRICS", "CHAMPION", "DECISIONS", "WARNINGS"],
            width=14,
            state="readonly",
        )
        filter_cb.pack(side=tk.LEFT, padx=(0, 12))
        filter_cb.bind("<<ComboboxSelected>>", lambda _e: self._populate_audit_events())

        ttk.Label(ctrl_bar, text="Search Candidate/Keyword:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        search_entry = ttk.Entry(ctrl_bar, textvariable=self._audit_search_var, width=24)
        search_entry.pack(side=tk.LEFT, padx=(0, 8))
        search_entry.bind("<Return>", lambda _e: self._populate_audit_events())

        ttk.Button(ctrl_bar, text="🔍 Search / Filter", command=self._populate_audit_events).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl_bar, text="🔄 Reset Filters", command=self._reset_audit_filters).pack(side=tk.LEFT)

        # Paned Window (Top: Events Treeview, Bottom: JSON Details Pane)
        paned = ttk.PanedWindow(self.tab_audit_trail, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        top_pane = ttk.Frame(paned)
        paned.add(top_pane, weight=3)

        tree_frame = ttk.Frame(top_pane)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        cols = ("timestamp", "generation", "candidate_id", "event_type", "message")
        self._audit_tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            height=12,
            selectmode="browse",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
        )
        tree_scroll_y.config(command=self._audit_tree.yview)
        tree_scroll_x.config(command=self._audit_tree.xview)

        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self._audit_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        headings = (
            ("timestamp", 150, "Timestamp"),
            ("generation", 50, "Gen"),
            ("candidate_id", 200, "Candidate ID"),
            ("event_type", 170, "Event Type"),
            ("message", 450, "Execution Message"),
        )
        for cid, width, text in headings:
            self._audit_tree.heading(cid, text=text)
            self._audit_tree.column(cid, width=width, anchor=tk.CENTER if cid == "generation" else tk.W)

        self._audit_tree.bind("<<TreeviewSelect>>", self._on_audit_event_selected)

        # Bottom Pane: JSON Details Inspector
        bot_pane = ttk.LabelFrame(paned, text="Selected Event Audit Payload (JSON Details)", padding=8)
        paned.add(bot_pane, weight=2)

        self._audit_detail_text = tk.Text(bot_pane, height=8, wrap="none", font=("Consolas", 9))
        txt_scroll_y = ttk.Scrollbar(bot_pane, orient=tk.VERTICAL, command=self._audit_detail_text.yview)
        txt_scroll_x = ttk.Scrollbar(bot_pane, orient=tk.HORIZONTAL, command=self._audit_detail_text.xview)
        self._audit_detail_text.config(yscrollcommand=txt_scroll_y.set, xscrollcommand=txt_scroll_x.set)

        txt_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        txt_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self._audit_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._populate_audit_events()

    def _reset_audit_filters(self) -> None:
        self._audit_filter_var.set("ALL")
        self._audit_search_var.set("")
        self._populate_audit_events()

    def _populate_audit_events(self) -> None:
        camp_id = self.selected_campaign_id.get()
        if not camp_id or not hasattr(self, "_audit_tree"):
            return

        for item in self._audit_tree.get_children():
            self._audit_tree.delete(item)

        filter_val = self._audit_filter_var.get()
        search_val = self._audit_search_var.get().strip()

        from chain_replay_ml.overnight_campaign.persistence import load_campaign_events
        events = load_campaign_events(
            self.data_dir,
            campaign_id=camp_id,
            event_type_filter=filter_val,
            search_query=search_val if search_val else None,
        )
        self._audit_events_cache = events

        for idx, ev in enumerate(events):
            self._audit_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    ev.get("timestamp", ""),
                    f"G{ev.get('generation', 0)}",
                    ev.get("candidate_id", "—"),
                    ev.get("event_type", ""),
                    ev.get("message", ""),
                ),
            )

        if events:
            self._audit_tree.selection_set("0")
            self._on_audit_event_selected(None)
        else:
            self._audit_detail_text.config(state="normal")
            self._audit_detail_text.delete("1.0", tk.END)
            self._audit_detail_text.insert(tk.END, "// No audit events found matching the active filter criteria.")
            self._audit_detail_text.config(state="disabled")

    def _on_audit_event_selected(self, _event: Any) -> None:
        if not hasattr(self, "_audit_tree") or not hasattr(self, "_audit_detail_text"):
            return
        sel = self._audit_tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
            if 0 <= idx < len(self._audit_events_cache):
                ev = self._audit_events_cache[idx]
                payload = {
                    "event_id": ev.get("event_id"),
                    "campaign_id": ev.get("campaign_id"),
                    "timestamp": ev.get("timestamp"),
                    "generation": ev.get("generation"),
                    "candidate_id": ev.get("candidate_id"),
                    "event_type": ev.get("event_type"),
                    "message": ev.get("message"),
                    "details": ev.get("details", {}),
                }
                txt_formatted = json.dumps(payload, indent=2, ensure_ascii=False)
                self._audit_detail_text.config(state="normal")
                self._audit_detail_text.delete("1.0", tk.END)
                self._audit_detail_text.insert(tk.END, txt_formatted)
                self._audit_detail_text.config(state="disabled")
        except Exception:
            pass

