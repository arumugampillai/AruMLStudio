"""Training Candidate Selection & Model Builder Handoff Dialog (Phase 3B).

Provides user approval, inspection of decision provenance, and selection of
Phase 3A TRAIN_CANDIDATE features for Model Builder preset export.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Sequence

from chain_replay_ml.production_validation.dataset_context import DatasetContext
from chain_replay_ml.production_validation.model_builder_handoff import (
    build_model_builder_training_bundle,
    export_training_candidates_preset,
)
from chain_replay_ml.production_validation.recommendation_policy import (
    RecommendationPolicy,
    load_recommendation_policy,
)
from chain_replay_ml.production_validation.training_decision_engine import (
    TrainingDecisionResult,
    TrainingDecisionState,
)


class DecisionReasonInspectorDialog(tk.Toplevel):
    """Detailed modal popup inspecting Phase 3A decision checks and bullets for a single feature."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        feature_name: str,
        provenance: dict[str, Any],
        context_id: str | None = None,
    ) -> None:
        super().__init__(master)
        self.title(f"Decision Reason Inspector — {feature_name}")
        self.transient(master.winfo_toplevel())
        self.geometry("640x480")
        self.minsize(500, 360)

        main_f = ttk.Frame(self, padding=12)
        main_f.pack(fill="both", expand=True)

        # Header Info
        dec = str(provenance.get("decision") or "UNKNOWN")
        reason = str(provenance.get("primary_reason") or "UNKNOWN")
        badges = " ".join(provenance.get("reason_badges") or [])
        score = float(provenance.get("evidence_score") or 0.0)
        conf = float(provenance.get("evidence_confidence") or 0.0)

        header_box = ttk.LabelFrame(main_f, text="Feature Evaluation Summary", padding=8)
        header_box.pack(fill="x", pady=(0, 10))

        ttk.Label(header_box, text=f"Feature: {feature_name}", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(header_box, text=f"Context ID: {context_id or 'global'}").grid(
            row=0, column=1, sticky="w", padx=15, pady=2
        )

        dec_color = "#107c41" if dec == "TRAIN_CANDIDATE" else ("#d83b01" if dec == "EXCLUDE" else "#b74700")
        dec_lbl = ttk.Label(header_box, text=f"Decision: {dec} {badges}", font=("TkDefaultFont", 9, "bold"), foreground=dec_color)
        dec_lbl.grid(row=1, column=0, sticky="w", pady=2)

        ttk.Label(header_box, text=f"Primary Reason: {reason}").grid(row=1, column=1, sticky="w", padx=15, pady=2)

        metrics_txt = (
            f"Evidence Score: {score:+.1f} | Confidence: {conf * 100:.1f}% | "
            f"Consensus: {provenance.get('dominant_recommendation') or '—'} | "
            f"Freshness: {provenance.get('freshness_label') or '—'}"
        )
        ttk.Label(header_box, text=metrics_txt, foreground="#555").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2
        )

        # Explanation Bullets
        bullets_box = ttk.LabelFrame(main_f, text="Deterministic Policy Evaluation Checks", padding=8)
        bullets_box.pack(fill="both", expand=True, pady=(0, 10))

        txt = tk.Text(bullets_box, wrap="word", font=("Consolas", 9), background="#fafafa", relief="flat")
        sb = ttk.Scrollbar(bullets_box, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        passed = provenance.get("passed_checks") or []
        failed = provenance.get("failed_checks") or []
        all_rules = provenance.get("all_triggered_rules") or []

        txt.insert("end", "=== PASSED CHECKS ===\n", "header")
        for p in passed:
            txt.insert("end", f"  ✓ {p}\n", "passed")
        if not passed:
            txt.insert("end", "  (none)\n")

        txt.insert("end", "\n=== FAILED / WARNING CHECKS ===\n", "header")
        for f in failed:
            txt.insert("end", f"  ✗ {f}\n", "failed")
        if not failed:
            txt.insert("end", "  (none — all checks passed)\n")

        txt.insert("end", "\n=== TRIGGERED RULES (PRECEDENCE ORDER) ===\n", "header")
        for idx, r in enumerate(all_rules, start=1):
            txt.insert("end", f"  {idx}. {r}\n", "rule")

        txt.tag_config("header", font=("Consolas", 9, "bold"), foreground="#003366")
        txt.tag_config("passed", foreground="#107c41")
        txt.tag_config("failed", foreground="#d83b01")
        txt.tag_config("rule", foreground="#004e8c")
        txt.configure(state="disabled")

        btn_f = ttk.Frame(main_f)
        btn_f.pack(fill="x")
        ttk.Button(btn_f, text="Close", command=self.destroy).pack(side="right")


class TrainingCandidateHandoffDialog(tk.Toplevel):
    """Interactive User Approval & Training Candidate Handoff Dialog."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        data_dir: str,
        context_id: str | None = None,
        context: DatasetContext | None = None,
        policy: RecommendationPolicy | None = None,
        on_exported: Callable[[dict[str, Any]], None] | None = None,
        on_open_model_builder: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Training Candidate Selection & Model Builder Handoff")
        self.transient(master.winfo_toplevel())
        self.geometry("980x620")
        self.minsize(800, 480)

        self.chart_dir = chart_dir
        self.data_dir = data_dir
        self.context_id = context_id or (context.context_id if context else None)
        self.context = context
        self.policy = policy or load_recommendation_policy(data_dir, context_id=self.context_id)
        self.on_exported = on_exported
        self.on_open_model_builder = on_open_model_builder

        # State tracking for feature selection
        self._feature_selection_vars: dict[str, tk.BooleanVar] = {}
        self._bundle_cache: dict[str, Any] = {}

        self._build_ui()
        self._load_and_populate()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 1. Top Context & Policy Banner
        header_f = ttk.LabelFrame(main_frame, text="Dataset Context & Active Policy", padding=(10, 8))
        header_f.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self._header_info_var = tk.StringVar(value="Loading context metadata...")
        ttk.Label(header_f, textvariable=self._header_info_var, font=("TkDefaultFont", 9)).pack(side="left")

        # 2. Notebook Tabs for Candidate Categories
        self._notebook = ttk.Notebook(main_frame)
        self._notebook.grid(row=1, column=0, sticky="nsew")

        # Tab 1: Eligible Candidates
        self._tab_candidates = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_candidates, text="🟢 Eligible Candidates (0)")
        self._build_tree_tab(self._tab_candidates, "candidates")

        # Tab 2: Under Review
        self._tab_review = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_review, text="🟡 Under Review (0)")
        self._build_tree_tab(self._tab_review, "review")

        # Tab 3: New / Unseen
        self._tab_unseen = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_unseen, text="⚪ New / Unseen (0)")
        self._build_tree_tab(self._tab_unseen, "unseen")

        # Tab 4: Excluded Features
        self._tab_exclude = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(self._tab_exclude, text="🔴 Excluded Features (0)")
        self._build_tree_tab(self._tab_exclude, "exclude")

        # 3. Bottom Controls & Action Toolbar
        bottom_f = ttk.Frame(main_frame)
        bottom_f.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bottom_f.columnconfigure(1, weight=1)

        sel_controls = ttk.Frame(bottom_f)
        sel_controls.pack(side="left")

        ttk.Button(sel_controls, text="Select All Eligible", command=self._select_all_eligible).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(sel_controls, text="Deselect All", command=self._deselect_all).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(sel_controls, text="Inspect Selected Reason", command=self._on_inspect_selected).pack(
            side="left", padx=(0, 8)
        )

        action_controls = ttk.Frame(bottom_f)
        action_controls.pack(side="right")

        self._export_btn_var = tk.StringVar(value="🚀 Export to Model Builder")
        self._export_btn = ttk.Button(
            action_controls,
            textvariable=self._export_btn_var,
            command=self._on_export_to_model_builder,
        )
        self._export_btn.pack(side="left", padx=(0, 6))

        ttk.Button(action_controls, text="Cancel", command=self.destroy).pack(side="left")

    def _build_tree_tab(self, parent: ttk.Frame, tab_key: str) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        columns = (
            "selected",
            "feature_name",
            "source",
            "reason",
            "score",
            "confidence",
            "consensus",
            "freshness",
            "volatility",
            "generalization",
        )
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")

        tree.heading("selected", text="✓")
        tree.heading("feature_name", text="Feature Name")
        tree.heading("source", text="Source")
        tree.heading("reason", text="Primary Reason / Badges")
        tree.heading("score", text="Score")
        tree.heading("confidence", text="Confidence")
        tree.heading("consensus", text="Consensus")
        tree.heading("freshness", text="Freshness")
        tree.heading("volatility", text="Volatility (σ)")
        tree.heading("generalization", text="Gen (G)")

        tree.column("selected", width=35, anchor="center")
        tree.column("feature_name", width=220, anchor="w")
        tree.column("source", width=75, anchor="center")
        tree.column("reason", width=220, anchor="w")
        tree.column("score", width=65, anchor="e")
        tree.column("confidence", width=75, anchor="e")
        tree.column("consensus", width=80, anchor="center")
        tree.column("freshness", width=80, anchor="center")
        tree.column("volatility", width=85, anchor="e")
        tree.column("generalization", width=70, anchor="e")

        sb_y = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb_y.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")

        # Double click to inspect, Button-1 on checkbox to toggle, Space to toggle
        tree.bind("<Button-1>", lambda e, t=tree, k=tab_key: self._on_tree_click(e, t, k))
        tree.bind("<Double-1>", lambda e, t=tree, k=tab_key: self._on_tree_double_click(t, k))
        tree.bind("<space>", lambda e, t=tree, k=tab_key: self._on_tree_space_toggle(t, k))

        setattr(self, f"_tree_{tab_key}", tree)

    def _load_and_populate(self) -> None:
        self._bundle_cache = build_model_builder_training_bundle(
            self.data_dir,
            context_id=self.context_id,
            context=self.context,
            policy=self.policy,
        )

        rdb = self._bundle_cache.get("recommendation_decision_bundle") or {}
        market = rdb.get("market") or "UNKNOWN"
        interval = rdb.get("sampling_interval_sec") or 0
        window = rdb.get("sliding_window") or "standard"
        p_id = rdb.get("policy_id") or "default"
        p_ver = rdb.get("policy_version") or 1
        cid_txt = self.context_id or "global"

        self._header_info_var.set(
            f"Context: {market} {interval}s ({window}) [{cid_txt}] | "
            f"Active Policy: {p_id} (v{p_ver}) | "
            f"Candidates: {rdb.get('eligible_candidates_count', 0)} | "
            f"Review: {rdb.get('review_count', 0)} | "
            f"Excluded: {rdb.get('excluded_count', 0)}"
        )

        prov_map = rdb.get("feature_provenance") or {}

        # Reset selection vars
        self._feature_selection_vars.clear()

        # Clear trees
        for k in ("candidates", "review", "unseen", "exclude"):
            tree: ttk.Treeview = getattr(self, f"_tree_{k}")
            for row in tree.get_children():
                tree.delete(row)

        c_count = 0
        r_count = 0
        u_count = 0
        e_count = 0

        # Sort features: Promo first, then score
        sorted_features = list(prov_map.keys())
        sorted_features.sort(
            key=lambda fn: (
                0 if "PROMOTION" in str(prov_map[fn].get("primary_reason", "")) else 1,
                -float(prov_map[fn].get("evidence_score") or 0.0),
                fn,
            )
        )

        for fn in sorted_features:
            p = prov_map[fn]
            dec = p.get("decision")
            source = str(p.get("feature_source") or "experimental").upper()[:4]
            reason = str(p.get("primary_reason") or "")
            badges = " ".join(p.get("reason_badges") or [])
            reason_txt = f"{badges} {reason}".strip()

            score = float(p.get("evidence_score") or 0.0)
            conf = float(p.get("evidence_confidence") or 0.0)
            conf_str = f"{conf * 100:.1f}%" if conf > 0 else "—"
            consensus = str(p.get("dominant_recommendation") or "—")
            freshness = str(p.get("freshness_label") or "—")
            vol = p.get("score_volatility")
            vol_str = f"{vol:.1f}" if vol is not None else "—"
            gen = p.get("generalization_score")
            gen_str = f"{gen:.2f}" if gen is not None else "—"

            if dec == TrainingDecisionState.TRAIN_CANDIDATE:
                self._feature_selection_vars[fn] = tk.BooleanVar(value=True)
                sel_char = "☑"
                self._tree_candidates.insert(
                    "",
                    "end",
                    iid=fn,
                    values=(
                        sel_char,
                        fn,
                        f"[{source}]",
                        reason_txt,
                        f"{score:+.1f}",
                        conf_str,
                        consensus,
                        freshness,
                        vol_str,
                        gen_str,
                    ),
                )
                c_count += 1
            elif dec == TrainingDecisionState.REVIEW:
                self._feature_selection_vars[fn] = tk.BooleanVar(value=False)
                sel_char = "☐"
                self._tree_review.insert(
                    "",
                    "end",
                    iid=fn,
                    values=(
                        sel_char,
                        fn,
                        f"[{source}]",
                        reason_txt,
                        f"{score:+.1f}",
                        conf_str,
                        consensus,
                        freshness,
                        vol_str,
                        gen_str,
                    ),
                )
                r_count += 1
            elif dec == TrainingDecisionState.NEW_UNSEEN:
                self._feature_selection_vars[fn] = tk.BooleanVar(value=False)
                sel_char = "☐"
                self._tree_unseen.insert(
                    "",
                    "end",
                    iid=fn,
                    values=(
                        sel_char,
                        fn,
                        f"[{source}]",
                        "NEW / UNSEEN (0 runs)",
                        "0.0",
                        "—",
                        "—",
                        "—",
                        "—",
                        "—",
                    ),
                )
                u_count += 1
            elif dec == TrainingDecisionState.EXCLUDE:
                self._feature_selection_vars[fn] = tk.BooleanVar(value=False)
                sel_char = "🚫"
                self._tree_exclude.insert(
                    "",
                    "end",
                    iid=fn,
                    values=(
                        sel_char,
                        fn,
                        f"[{source}]",
                        reason_txt,
                        f"{score:+.1f}",
                        conf_str,
                        consensus,
                        freshness,
                        vol_str,
                        gen_str,
                    ),
                )
                e_count += 1

        self._notebook.tab(0, text=f"🟢 Eligible Candidates ({c_count})")
        self._notebook.tab(1, text=f"🟡 Under Review ({r_count})")
        self._notebook.tab(2, text=f"⚪ New / Unseen ({u_count})")
        self._notebook.tab(3, text=f"🔴 Excluded Features ({e_count})")

        self._update_export_button_count()

    def toggle_feature_selection(self, feature_name: str, tree: ttk.Treeview | None = None) -> bool:
        """Toggle selection state for an individual feature. Excluded features cannot be selected."""
        fn = str(feature_name).strip()
        rdb = self._bundle_cache.get("recommendation_decision_bundle") or {}
        prov = (rdb.get("feature_provenance") or {}).get(fn) or {}
        if prov.get("decision") == TrainingDecisionState.EXCLUDE:
            return False

        var = self._feature_selection_vars.get(fn)
        if var is None:
            var = tk.BooleanVar(value=False)
            self._feature_selection_vars[fn] = var

        new_val = not var.get()
        var.set(new_val)
        sel_char = "☑" if new_val else "☐"

        # Update Treeview rows across tabs
        for k in ("candidates", "review", "unseen"):
            t: ttk.Treeview | None = getattr(self, f"_tree_{k}", None)
            if t and t.exists(fn):
                vals = list(t.item(fn, "values"))
                if vals:
                    vals[0] = sel_char
                    t.item(fn, values=vals)

        self._update_export_button_count()
        return new_val

    def _on_tree_click(self, event: tk.Event, tree: ttk.Treeview, tab_key: str) -> None:
        """Single click handler for Treeview: toggles checkbox when clicked on checkbox column #1."""
        if tab_key == "exclude":
            return
        col = tree.identify_column(event.x)
        iid = tree.identify_row(event.y)
        if not iid:
            return
        if col == "#1":
            self.toggle_feature_selection(iid, tree=tree)

    def _on_tree_double_click(self, tree: ttk.Treeview, tab_key: str) -> None:
        sel = tree.selection()
        if not sel:
            return
        fn = sel[0]
        rdb = self._bundle_cache.get("recommendation_decision_bundle") or {}
        prov = (rdb.get("feature_provenance") or {}).get(fn)
        if prov:
            DecisionReasonInspectorDialog(self, feature_name=fn, provenance=prov, context_id=self.context_id)

    def _on_tree_space_toggle(self, tree: ttk.Treeview, tab_key: str) -> None:
        if tab_key == "exclude":
            return
        sel = tree.selection()
        if not sel:
            return
        self.toggle_feature_selection(sel[0], tree=tree)

    def _select_all_eligible(self) -> None:
        for fn in self._tree_candidates.get_children():
            var = self._feature_selection_vars.get(fn)
            if var:
                var.set(True)
                vals = list(self._tree_candidates.item(fn, "values"))
                vals[0] = "☑"
                self._tree_candidates.item(fn, values=vals)
        self._update_export_button_count()

    def _deselect_all(self) -> None:
        for var in self._feature_selection_vars.values():
            var.set(False)
        for k in ("candidates", "review", "unseen"):
            tree: ttk.Treeview = getattr(self, f"_tree_{k}")
            for fn in tree.get_children():
                vals = list(tree.item(fn, "values"))
                vals[0] = "☐"
                tree.item(fn, values=vals)
        self._update_export_button_count()

    def _on_inspect_selected(self) -> None:
        for k in ("candidates", "review", "unseen", "exclude"):
            tree: ttk.Treeview = getattr(self, f"_tree_{k}")
            sel = tree.selection()
            if sel:
                self._on_tree_double_click(tree, k)
                return
        messagebox.showinfo("Inspect Reason", "Select a feature row first to inspect its decision checks.", parent=self)

    def _update_export_button_count(self) -> None:
        cnt = sum(1 for v in self._feature_selection_vars.values() if v.get())
        self._export_btn_var.set(f"🚀 Export Selected ({cnt}) to Model Builder")

    def _on_export_to_model_builder(self) -> None:
        selected = [fn for fn, v in self._feature_selection_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning(
                "Export Preset",
                "No features selected. Please select at least one candidate feature to export.",
                parent=self,
            )
            return

        try:
            preset = export_training_candidates_preset(
                self.chart_dir,
                self.data_dir,
                context_id=self.context_id,
                context=self.context,
                policy=self.policy,
                selected_features=selected,
            )
        except Exception as exc:
            messagebox.showerror(
                "Export Preset Failed",
                f"Could not save Model Builder preset:\n{exc}",
                parent=self,
            )
            return

        messagebox.showinfo(
            "Export Successful",
            f"Successfully exported {len(selected)} training candidate(s) to Model Builder preset.\n\n"
            f"Context: {self.context_id or 'global'}\n"
            "Open Model Builder → Create Model to load and train with this preset.",
            parent=self,
        )

        if callable(self.on_exported):
            try:
                self.on_exported(preset)
            except Exception:
                pass

        if callable(self.on_open_model_builder):
            try:
                self.on_open_model_builder()
            except Exception:
                pass

        self.destroy()
