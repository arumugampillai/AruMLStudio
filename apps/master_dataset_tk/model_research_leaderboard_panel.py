"""Model Research Lab Leaderboard Panel (Phase 4D.7).

Provides context-isolated research leaderboard visualization, multi-model robustness
scorecards, Pareto multi-objective frontiers, cross-regime stress evidence, feature
composition governance audits, and immutable champion transition history from `<data_dir>/analysis.db`.

Invariants:
1. Context-Scoped Isolation: Operates strictly within a single `ModelContextKey`.
2. Pure Presentation Layer: Consumes authoritative Phase 4D service APIs (`rank_models_in_context`,
   `get_champion_history_for_context`, etc.) without reproducing ranking mathematics.
3. Human Governance Boundary: Visually distinguishes Production Champion state from
   Research `CHAMPION_CANDIDATE`. Strictly read-only; zero mutation of `analysis.db` or `.active_model.json`
   or `active_model.json`.
4. 16 GB Workstation Safety: Uses database-side filtering and bounded Treeviews.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from chain_replay_ml.model_taxonomy import ModelContextKey, BASELINE_REGIME_CATALOG
from chain_replay_ml.research_memory import (
    get_benchmark_metrics,
    get_benchmark_run,
    get_campaign,
    get_champion_history_for_context,
    get_feature_set_evaluation,
    get_latest_champion_transition,
    get_model_benchmarks_for_context,
    get_regime_evaluations_for_model,
    rank_models_in_context,
)
from chain_replay_ml.research_memory.champion_history import get_champion_for_context
from chain_replay_ml.research_recommendations import (
    RecommendationDossier,
    generate_context_recommendation_dossiers,
)

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


class ModelResearchLeaderboardPanel(ttk.Frame):
    """Context-isolated multi-model research leaderboard and empirical evidence inspector."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        chart_dir: str | None = None,
        on_select_model: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.chart_dir = chart_dir
        self._on_select_model = on_select_model

        # Context selection state
        self._market_var = tk.StringVar(value="NIFTY")
        self._sampling_var = tk.StringVar(value="3s")
        self._task_var = tk.StringVar(value="DIRECTION_CLASSIFIER")
        self._horizon_var = tk.StringVar(value="5m")
        self._regime_var = tk.StringVar(value="R001")
        self._context_key_var = tk.StringVar(value="")

        self._ranked_dossiers: list[dict[str, Any]] = []
        self._selected_dossier: dict[str, Any] | None = None

        self._build_ui()
        self._update_resolved_context_key()

    def _data_dir(self) -> str:
        if not self.chart_dir:
            return ""
        if os.path.exists(os.path.join(self.chart_dir, "analysis.db")):
            return self.chart_dir
        return chart_data_dir(self.chart_dir)

    def _update_resolved_context_key(self) -> None:
        """Resolve dropdown values into canonical ModelContextKey string."""
        interval_str = self._sampling_var.get().replace("s", "").replace("sec", "").strip()
        try:
            interval_sec = int(interval_str)
        except ValueError:
            interval_sec = 3

        # Extract clean regime ID from combo text (e.g. 'R001 - TREND' -> 'R001')
        reg_raw = self._regime_var.get().split()[0].split("-")[0].strip()
        if not reg_raw:
            reg_raw = "R001"

        from chain_replay_ml.model_taxonomy import TaskType

        ctx = ModelContextKey(
            market=self._market_var.get(),
            sampling_interval_sec=interval_sec,
            task_type=TaskType.from_str(self._task_var.get()),
            prediction_horizon=self._horizon_var.get(),
            regime_id=reg_raw,
        )
        self._context_key_var.set(ctx.canonical_key_str())

    def _build_ui(self) -> None:
        # Top Container: Context Selector & Production Status Banner
        top_frame = ttk.Frame(self, padding=(8, 6))
        top_frame.pack(fill="x")

        # 1. Canonical Context Selector Bar
        ctx_box = ttk.LabelFrame(top_frame, text="Canonical Research Context Selector (ModelContextKey)", padding=(8, 6))
        ctx_box.pack(fill="x", pady=(0, 4))

        controls_row = ttk.Frame(ctx_box)
        controls_row.pack(fill="x")

        # Market
        ttk.Label(controls_row, text="Market:").pack(side="left", padx=(2, 2))
        m_cb = ttk.Combobox(controls_row, textvariable=self._market_var, values=["NIFTY", "BANKNIFTY", "FINNIFTY"], width=10, state="readonly")
        m_cb.pack(side="left", padx=(0, 8))
        m_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        # Sampling
        ttk.Label(controls_row, text="Sampling:").pack(side="left", padx=(2, 2))
        s_cb = ttk.Combobox(controls_row, textvariable=self._sampling_var, values=["3s", "1s", "5s", "15s", "1m"], width=6, state="readonly")
        s_cb.pack(side="left", padx=(0, 8))
        s_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        # Task Type
        ttk.Label(controls_row, text="Task:").pack(side="left", padx=(2, 2))
        t_cb = ttk.Combobox(
            controls_row,
            textvariable=self._task_var,
            values=["DIRECTION_CLASSIFIER", "REGRESSION", "TRIPLE_BARRIER", "CONFIDENCE_CLASSIFIER", "VOLATILITY_ESTIMATOR"],
            width=22,
            state="readonly",
        )
        t_cb.pack(side="left", padx=(0, 8))
        t_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        # Horizon
        ttk.Label(controls_row, text="Horizon:").pack(side="left", padx=(2, 2))
        h_cb = ttk.Combobox(controls_row, textvariable=self._horizon_var, values=["5m", "15m", "30m", "1h", "1d"], width=6, state="readonly")
        h_cb.pack(side="left", padx=(0, 8))
        h_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        # Regime
        ttk.Label(controls_row, text="Regime:").pack(side="left", padx=(2, 2))
        reg_vals = ["R001 (Trend)", "R002 (Sideways)", "R003 (High Vol)", "R004 (Low Vol)", "R005 (Breakout)", "R006 (Reversal)", "R007 (Expiry Pinning)"]
        r_cb = ttk.Combobox(controls_row, textvariable=self._regime_var, values=reg_vals, width=18, state="readonly")
        r_cb.pack(side="left", padx=(0, 8))
        r_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        ttk.Button(controls_row, text="Query Leaderboard", command=self.refresh_leaderboard).pack(side="right", padx=4)

        # Context Key Display Label
        key_bar = ttk.Frame(ctx_box, padding=(0, 4))
        key_bar.pack(fill="x")
        ttk.Label(key_bar, text="🎯 Active Context Key:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        ttk.Label(key_bar, textvariable=self._context_key_var, font=("Consolas", 10, "bold"), foreground="#0d47a1").pack(side="left", padx=(6, 0))

        # 2. Production Governance vs Research Candidate Status Banner
        gov_box = ttk.LabelFrame(top_frame, text="Context Governance & Champion Status", padding=(8, 4))
        gov_box.pack(fill="x", pady=(2, 4))

        gov_row = ttk.Frame(gov_box)
        gov_row.pack(fill="x")

        self._prod_champ_var = tk.StringVar(value="👑 Production Champion: None")
        self._prod_chall_var = tk.StringVar(value="⚔️ Production Challenger: None")
        self._cand_champ_var = tk.StringVar(value="🧪 Research Champion Candidate: None")

        ttk.Label(gov_row, textvariable=self._prod_champ_var, font=("TkDefaultFont", 9, "bold"), foreground="#1b5e20").pack(side="left", padx=(0, 16))
        ttk.Label(gov_row, textvariable=self._prod_chall_var, font=("TkDefaultFont", 9), foreground="#e65100").pack(side="left", padx=(0, 16))
        ttk.Label(gov_row, textvariable=self._cand_champ_var, font=("TkDefaultFont", 9, "bold"), foreground="#4a148c").pack(side="left", padx=(0, 16))

        disclaimer_lbl = ttk.Label(
            gov_box,
            text="⚠️ Research Memory is strictly advisory. Candidate promotions require human governance approval.",
            font=("TkDefaultFont", 8, "italic"),
            foreground=COL_MUTED,
        )
        disclaimer_lbl.pack(anchor="w", pady=(2, 0))

        # Split View: Leaderboard Table (Top) & Detail Dossier Notebook (Bottom)
        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        # Top Paned Frame: Leaderboard Table
        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        tree_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)

        cols = (
            "rank",
            "model_name",
            "algorithm",
            "rob_score",
            "pareto_rank",
            "primary_metric",
            "fold_std",
            "ece",
            "regime_deg",
            "exp_ratio",
            "recommendation",
            "features",
            "signature",
        )

        self.leaderboard_tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            selectmode="browse",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
        )
        tree_scroll_y.config(command=self.leaderboard_tree.yview)
        tree_scroll_x.config(command=self.leaderboard_tree.xview)

        tree_scroll_y.pack(side="right", fill="y")
        tree_scroll_x.pack(side="bottom", fill="x")
        self.leaderboard_tree.pack(side="left", fill="both", expand=True)

        headings = (
            ("rank", 50, "Rank #"),
            ("model_name", 180, "Model Name"),
            ("algorithm", 90, "Algorithm"),
            ("rob_score", 95, "Robustness"),
            ("pareto_rank", 80, "Pareto #"),
            ("primary_metric", 120, "Primary Metric"),
            ("fold_std", 85, "Fold Std"),
            ("ece", 80, "ECE (Calib)"),
            ("regime_deg", 95, "Regime Deg %"),
            ("exp_ratio", 85, "Exp Feat %"),
            ("recommendation", 140, "Recommendation"),
            ("features", 65, "Feats"),
            ("signature", 110, "Signature Hash"),
        )

        for cid, width, text in headings:
            self.leaderboard_tree.heading(cid, text=text)
            self.leaderboard_tree.column(cid, width=width, anchor="center" if cid != "model_name" else "w")

        self.leaderboard_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Bottom Paned Frame: Research Evidence Dossier Notebook
        detail_outer = ttk.Frame(paned)
        paned.add(detail_outer, weight=3)

        self._detail_nb = ttk.Notebook(detail_outer)
        self._detail_nb.pack(fill="both", expand=True)

        self._tab_dossier = ScrollableFrame(self._detail_nb)
        self._tab_recommendations = ScrollableFrame(self._detail_nb)
        self._tab_regimes = ScrollableFrame(self._detail_nb)
        self._tab_features = ScrollableFrame(self._detail_nb)
        self._tab_lineage = ScrollableFrame(self._detail_nb)
        self._tab_history = ScrollableFrame(self._detail_nb)

        self._detail_nb.add(self._tab_dossier, text="Robustness Dossier")
        self._detail_nb.add(self._tab_recommendations, text="Research Recommendations")
        self._detail_nb.add(self._tab_regimes, text="Cross-Regime Stress")
        self._detail_nb.add(self._tab_features, text="Feature Composition")
        self._detail_nb.add(self._tab_lineage, text="Research Lineage")
        self._detail_nb.add(self._tab_history, text="Champion History")

    def _on_context_param_changed(self) -> None:
        self._update_resolved_context_key()
        self.refresh_leaderboard()

    def refresh_leaderboard(self) -> None:
        """Query analysis.db using Phase 4D.5 ranking service for the active context key."""
        ctx_key = self._context_key_var.get()
        data_dir = self._data_dir()

        # 1. Update Context Champion & Challenger Banner from analysis.db champion_history
        try:
            prod_doc = get_champion_for_context(data_dir, ctx_key)
            champ_name = prod_doc.get("champion_model_name") if prod_doc else None
            chall_name = prod_doc.get("challenger_model_name") if prod_doc else None
            self._prod_champ_var.set(f"👑 Production Champion: {champ_name or 'None'}")
            self._prod_chall_var.set(f"⚔️ Production Challenger: {chall_name or 'None'}")
        except Exception:
            self._prod_champ_var.set("👑 Production Champion: None")
            self._prod_chall_var.set("⚔️ Production Challenger: None")

        # 2. Query Authoritative Ranking Dossiers from Phase 4D.5
        try:
            dossiers = rank_models_in_context(data_dir, ctx_key)
        except Exception:
            dossiers = []

        self._ranked_dossiers = dossiers

        # Update Research Candidate display
        if dossiers and dossiers[0].get("recommendation_status") == "CHAMPION_CANDIDATE":
            top_cand = dossiers[0]["model_name"]
            score = dossiers[0]["robustness_score"]
            self._cand_champ_var.set(f"🧪 Research Champion Candidate: {top_cand} ({score:.2f} pts)")
        else:
            self._cand_champ_var.set("🧪 Research Champion Candidate: None")

        # Clear tree
        for item in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(item)

        if not dossiers:
            self._render_empty_detail("No benchmarked models found for this context key.")
            self._render_champion_history_tab()
            self._render_recommendations_tab()
            return

        self._item_dossier_map: dict[str, dict[str, Any]] = {}
        for idx, d in enumerate(dossiers, start=1):
            raw = d.get("raw_metrics_summary", {})
            r_score = d.get("robustness_score", 0.0)
            p_rank = d.get("pareto_rank", 1)
            p_metric = f"{raw.get('primary_metric_name', 'metric')}: {raw.get('primary_metric_value', 0.0):.4f}"
            f_std = f"±{raw.get('fold_std', 0.0):.4f}" if raw.get("fold_std") is not None else "—"
            ece = f"{raw.get('expected_calibration_error', 0.0):.4f}" if raw.get("expected_calibration_error") is not None else "—"
            deg = f"{raw.get('avg_regime_degradation_pct', 0.0):.1f}%" if raw.get("avg_regime_degradation_pct") is not None else "—"
            exp_r = f"{float(raw.get('experimental_dependency_ratio', 0.0)) * 100:.1f}%"
            sig = d.get("signature_hash", "")[:12] + "..."

            item_id = f"row_{idx}_{d.get('signature_hash', '')[:8]}"
            self._item_dossier_map[item_id] = d

            self.leaderboard_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    d.get("rank_in_context", idx),
                    d.get("model_name", "—"),
                    d.get("algorithm", "—"),
                    f"{r_score:.2f}",
                    f"Tier {p_rank}",
                    p_metric,
                    f_std,
                    ece,
                    deg,
                    exp_r,
                    d.get("recommendation_status", "—"),
                    raw.get("total_features", 0),
                    sig,
                ),
            )

        # Select first item by default
        children = self.leaderboard_tree.get_children()
        if children:
            self.leaderboard_tree.selection_set(children[0])
            self.leaderboard_tree.see(children[0])
            first_dossier = self._item_dossier_map.get(children[0], dossiers[0])
            self._selected_dossier = first_dossier
            self._load_dossier_detail(first_dossier)

        self._render_champion_history_tab()
        self._render_recommendations_tab()

    def _on_tree_select(self, _event: tk.Event) -> None:
        sel = self.leaderboard_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        d = self._item_dossier_map.get(item_id)
        if d:
            self._selected_dossier = d
            self._load_dossier_detail(d)
            if self._on_select_model and d.get("model_name"):
                self._on_select_model(d["model_name"])

    def _render_empty_detail(self, message: str) -> None:
        for tab in (self._tab_dossier, self._tab_regimes, self._tab_features, self._tab_lineage):
            clear_children(tab.inner)
            ttk.Label(tab.inner, text=message, font=("TkDefaultFont", 10, "italic"), foreground=COL_MUTED).pack(padx=16, pady=16)

    def _load_dossier_detail(self, dossier: dict[str, Any]) -> None:
        """Render multi-tab research evidence for the selected dossier."""
        data_dir = self._data_dir()
        sig_hash = dossier["signature_hash"]
        model_name = dossier["model_name"]

        # 1. Robustness Dossier Tab
        tab1 = self._tab_dossier.inner
        clear_children(tab1)
        section_title(tab1, f"Robustness Dossier: {model_name}")
        section_desc(tab1, f"Canonical Identity: {sig_hash}")

        # Top summary cards
        score = dossier.get("robustness_score", 0.0)
        p_rank = dossier.get("pareto_rank", 1)
        status = dossier.get("recommendation_status", "VALIDATED")

        summary_rows = [
            ("Robustness Score", f"{score:.2f} / 100.00"),
            ("Pareto Optimality Tier", f"Tier {p_rank} ({'Non-Dominated / Optimal' if p_rank == 1 else 'Dominated'})"),
            ("Recommendation Status", status),
            ("Ranking Policy Version", dossier.get("ranking_policy_version", "ROB_POLICY_v1.0")),
            ("Ranking Policy Hash", dossier.get("ranking_policy_hash", "—")),
        ]
        kv_block(tab1, "Summary", summary_rows)

        # Penalties Table
        breakdown = dossier.get("score_breakdown", {})
        penalty_table_rows = [
            ("Base Performance Contribution", f"+{breakdown.get('base_performance_contribution', 0.0):.2f} pts"),
            ("Walk-Forward Variance Penalty", f"{breakdown.get('fold_variance_penalty', 0.0):.2f} pts"),
            ("Worst-Fold Drawdown Penalty", f"{breakdown.get('worst_fold_penalty', 0.0):.2f} pts"),
            ("Calibration Error (ECE) Penalty", f"{breakdown.get('calibration_penalty', 0.0):.2f} pts"),
            ("Cross-Regime Degradation Penalty", f"{breakdown.get('regime_degradation_penalty', 0.0):.2f} pts"),
            ("Experimental Feature Risk Penalty", f"{breakdown.get('experimental_risk_penalty', 0.0):.2f} pts"),
            ("Model Parsimony / Complexity Penalty", f"{breakdown.get('parsimony_penalty', 0.0):.2f} pts"),
            ("Final Clamped Robustness Score", f"{score:.2f} pts"),
        ]
        data_table(
            tab1,
            [("dim", "Evaluation Dimension", 260), ("impact", "Score Contribution / Penalty", 200)],
            penalty_table_rows,
        )

        # Warnings
        warnings = dossier.get("warnings", [])
        if warnings:
            w_box = ttk.LabelFrame(tab1, text="Safety & Quarantine Warnings", padding=8)
            w_box.pack(fill="x", pady=8)
            for w in warnings:
                ttk.Label(w_box, text=f"⚠️ {w}", foreground=COL_WARN, font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=2)

        # 2. Cross-Regime Stress Tab
        tab2 = self._tab_regimes.inner
        clear_children(tab2)
        section_title(tab2, f"Cross-Regime Stress Evidence: {model_name}")
        section_desc(tab2, "Empirical generalization across market regimes (non-native stress slices).")

        try:
            reg_evals = get_regime_evaluations_for_model(data_dir, sig_hash)
        except Exception:
            reg_evals = []

        if reg_evals:
            reg_rows = []
            for r in reg_evals:
                is_nat = "👑 Native" if r.get("is_native_regime") else "🧪 Stress Slice"
                deg_str = f"{r.get('regime_degradation_pct', 0.0):.1f}%" if not r.get("is_native_regime") else "0.0% (Baseline)"
                reg_rows.append((
                    r.get("tested_regime_id", "—"),
                    is_nat,
                    f"{r.get('primary_metric_value', 0.0):.4f}",
                    f"{r.get('baseline_metric_value', 0.0):.4f}",
                    deg_str,
                    str(r.get("sample_count", 0)),
                ))
            data_table(
                tab2,
                [
                    ("reg_id", "Regime ID", 100),
                    ("role", "Role", 110),
                    ("test_m", "Test Metric", 100),
                    ("base_m", "Native Baseline", 110),
                    ("deg", "Degradation %", 110),
                    ("samples", "Samples", 80),
                ],
                reg_rows,
            )
        else:
            ttk.Label(tab2, text="No multi-regime stress evaluations recorded for this experiment.", font=("TkDefaultFont", 9, "italic"), foreground=COL_MUTED).pack(pady=12)

        # 3. Feature Composition Tab
        tab3 = self._tab_features.inner
        clear_children(tab3)
        section_title(tab3, f"Feature Set Governance & Composition: {model_name}")
        section_desc(tab3, "Authoritative classification against Base Pipeline (PL_0001) and Feature Registry.")

        try:
            f_eval = get_feature_set_evaluation(data_dir, sig_hash)
        except Exception:
            f_eval = None

        if f_eval:
            f_summary = [
                ("Total Features", str(f_eval.get("total_features", 0))),
                ("Base Features (PL_0001)", str(f_eval.get("base_feature_count", 0))),
                ("Registry Features", str(f_eval.get("registry_feature_count", 0))),
                ("Experimental Features (PL_0002+)", str(f_eval.get("experimental_feature_count", 0))),
                ("Deprecated Features", str(f_eval.get("deprecated_feature_count", 0))),
                ("Unknown Features", str(f_eval.get("unknown_feature_count", 0))),
                ("Experimental Dependency Ratio", f"{f_eval.get('experimental_dependency_ratio', 0.0) * 100:.1f}%"),
            ]
            kv_block(tab3, "Composition Metrics", f_summary)

            try:
                top_feats = json.loads(f_eval.get("top_10_features_json", "[]"))
                if top_feats:
                    data_table(tab3, [("feat", "Feature Name", 300)], [[f] for f in top_feats])
            except Exception:
                pass
        else:
            raw_feats = dossier.get("raw_metrics_summary", {}).get("total_features", 0)
            kv_block(tab3, "Composition", [("Total Features Count", str(raw_feats)), ("Audit Status", "Pending detailed composition evaluation")])

        # 4. Research Lineage Tab
        tab4 = self._tab_lineage.inner
        clear_children(tab4)
        section_title(tab4, f"End-to-End Research Lineage: {model_name}")
        section_desc(tab4, "Verifiable audit chain from Campaign to Benchmark and Candidate.")

        bm_id = dossier.get("benchmark_id", "—")
        run_id = dossier.get("benchmark_run_id", "—")

        # Try to resolve campaign_id from benchmark_run
        camp_id = None
        if run_id and run_id != "—":
            try:
                b_run = get_benchmark_run(data_dir, run_id)
                if b_run:
                    camp_id = b_run.get("campaign_id")
            except Exception:
                pass

        lineage_rows = [
            ("Campaign ID", camp_id or "Direct / Independent Run"),
            ("Experiment Signature Hash", sig_hash),
            ("Benchmark Run ID", str(run_id)),
            ("Model Benchmark Scorecard ID", str(bm_id)),
            ("Model Name", model_name),
            ("Algorithm", dossier.get("algorithm", "—")),
            ("ModelContextKey", dossier.get("context_key", "—")),
            ("Ranking Policy", dossier.get("ranking_policy_version", "ROB_POLICY_v1.0")),
            ("Ranking Policy Hash", dossier.get("ranking_policy_hash", "—")),
        ]
        kv_block(tab4, "Lineage Chain", lineage_rows)

    def _render_champion_history_tab(self) -> None:
        """Render immutable historical champion promotions for active context key."""
        tab5 = self._tab_history.inner
        clear_children(tab5)
        ctx_key = self._context_key_var.get()
        data_dir = self._data_dir()

        section_title(tab5, f"Champion Transition Audit History: {ctx_key}")
        section_desc(tab5, "Append-only longitudinal record of human-approved model promotions.")

        try:
            history = get_champion_history_for_context(data_dir, ctx_key, limit=50)
        except Exception:
            history = []

        if history:
            h_rows = []
            for h in history:
                h_rows.append((
                    str(h.get("transition_id")),
                    h.get("previous_champion_name") or "None (Initial)",
                    h.get("new_champion_name", "—"),
                    f"{h.get('previous_robustness_score') or 0.0:.2f}",
                    f"{h.get('new_robustness_score', 0.0):.2f}",
                    f"{h.get('score_delta', 0.0):+.2f}",
                    h.get("ranking_policy_version", "—"),
                    h.get("promoted_by", "HUMAN_RESEARCHER"),
                    h.get("promotion_reason", "—"),
                    h.get("transition_timestamp", "—")[:19],
                ))
            data_table(
                tab5,
                [
                    ("id", "ID", 40),
                    ("prev_c", "Previous Champion", 160),
                    ("new_c", "New Champion", 160),
                    ("prev_s", "Prev Score", 80),
                    ("new_s", "New Score", 80),
                    ("delta", "Score Δ", 70),
                    ("pol", "Policy", 100),
                    ("prom_by", "Promoted By", 130),
                    ("reason", "Reason", 200),
                    ("ts", "Timestamp", 140),
                ],
                h_rows,
            )
        else:
            ttk.Label(tab5, text="No champion transitions recorded for this ModelContextKey yet.", font=("TkDefaultFont", 9, "italic"), foreground=COL_MUTED).pack(pady=12)

    def _render_recommendations_tab(self) -> None:
        """Render ranked research recommendation dossiers for the active context key."""
        tab = self._tab_recommendations.inner
        clear_children(tab)
        ctx_key = self._context_key_var.get()
        data_dir = self._data_dir()

        section_title(tab, f"Automated Research Recommendations & Opportunity Agenda: {ctx_key}")
        section_desc(tab, "Deterministic multi-objective research agenda synthesized from Coverage, Vulnerability, Feature Affinity, and Pruning.")

        disclaimer_lbl = ttk.Label(
            tab,
            text="⚠️ RESEARCH RECOMMENDATION — HUMAN DECISION REQUIRED (Strictly advisory; no automated execution or promotion).",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#d84315",
        )
        disclaimer_lbl.pack(anchor="w", pady=(0, 8))

        try:
            dossiers = generate_context_recommendation_dossiers(data_dir, ctx_key)
        except Exception:
            dossiers = []

        if not dossiers:
            ttk.Label(tab, text="No research opportunities identified for this context key.", font=("TkDefaultFont", 9, "italic"), foreground=COL_MUTED).pack(pady=12)
            return

        opp_rows = []
        for idx, d in enumerate(dossiers, start=1):
            feats_str = ", ".join(d.candidate_features[:4]) + ("..." if len(d.candidate_features) > 4 else "")
            opp_rows.append((
                str(idx),
                d.priority_class.value,
                f"{d.priority_score:.2f}",
                d.opportunity_type.value,
                d.evidence_confidence.value,
                d.target_algorithm,
                f"[{feats_str}]",
                d.exclusion_verdict.value,
                d.why_recommended[:80] + ("..." if len(d.why_recommended) > 80 else ""),
            ))

        data_table(
            tab,
            [
                ("rank", "#", 35),
                ("prio_class", "Priority", 95),
                ("score", "Score", 65),
                ("type", "Opportunity Type", 190),
                ("conf", "Confidence", 95),
                ("algo", "Algo", 75),
                ("features", "Candidate Features", 180),
                ("status", "Status", 75),
                ("rationale", "Primary Rationale", 260),
            ],
            opp_rows,
        )

        # Render detail for top recommendation
        top_d = dossiers[0]
        kv_block(
            tab,
            f"Top Priority Opportunity Dossier: {top_d.opportunity_id}",
            [
                ("Opportunity Type", top_d.opportunity_type.value),
                ("Priority Score", f"{top_d.priority_score:.2f} ({top_d.priority_class.value})"),
                ("Evidence Confidence", f"{top_d.evidence_confidence.value} ({top_d.confidence_value:.4f})"),
                ("Target Algorithm", top_d.target_algorithm),
                ("Candidate Features", ", ".join(top_d.candidate_features)),
                ("Why Recommended", top_d.why_recommended),
                ("Missing Evidence", top_d.missing_evidence_summary),
                ("Caution Warnings", "; ".join(top_d.caution_warnings) if top_d.caution_warnings else "None"),
            ],
        )

        breakdown_rows = [
            ("Champion Vulnerability Contribution (30%)", f"{top_d.champion_vulnerability_contrib:.2f} pts"),
            ("Challenger Gap Contribution (25%)", f"{top_d.challenger_gap_contrib:.2f} pts"),
            ("Feature Affinity Contribution (20%)", f"{top_d.feature_affinity_contrib:.2f} pts"),
            ("Coverage Gap Contribution (15%)", f"{top_d.coverage_gap_contrib:.2f} pts"),
            ("Interaction Synergy Contribution (10%)", f"{top_d.interaction_synergy_contrib:.2f} pts"),
            ("Caution Penalty", f"{top_d.caution_penalty:.2f} pts"),
        ]
        kv_block(tab, "Multi-Objective Score Component Decomposition", breakdown_rows)

        steps_rows = [(f"Step {i}", step) for i, step in enumerate(top_d.suggested_next_steps, start=1)]
        kv_block(tab, "Suggested Next Research Directions", steps_rows)

