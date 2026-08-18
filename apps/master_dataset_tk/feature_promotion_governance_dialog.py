"""Feature Promotion & Lifecycle Governance Dialog (Phase 3D.4C).

Comprehensive human-in-the-loop governance dialog supporting:
1. REGISTRY_GRADUATION: Graduate qualified experimental feature into Feature Registry.
2. BASE_PIPELINE_PROMOTION: Promote universal graduated feature into Base Pipeline (PL_0001).
3. FEATURE_DEPRECATION: Governed retirement & Base Pipeline eviction of features.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.production_validation.api import (
    compile_feature_evidence_dossier,
    evaluate_base_pipeline_eligibility,
    evaluate_deprecation_prerequisites,
    evaluate_graduation_prerequisites,
    execute_base_pipeline_promotion,
    execute_feature_deprecation,
    execute_registry_graduation,
    get_feature_graduation_audit_log,
    is_feature_in_base_pipeline,
)


class FeaturePromotionGovernanceDialog(tk.Toplevel):
    """Human governance review dialog supporting Registry Graduation, Base Pipeline Promotion, and Deprecation."""

    MODE_REGISTRY_GRADUATION = "REGISTRY_GRADUATION"
    MODE_BASE_PIPELINE_PROMOTION = "BASE_PIPELINE_PROMOTION"
    MODE_FEATURE_DEPRECATION = "FEATURE_DEPRECATION"

    def __init__(
        self,
        master: tk.Misc,
        *,
        data_dir: str,
        feature_name: str,
        mode: str = "REGISTRY_GRADUATION",
        context_id: str | None = None,
        precompiled_dossier: dict[str, Any] | None = None,
        on_decision: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.data_dir = data_dir
        self.feature_name = feature_name
        self.mode = str(mode or self.MODE_REGISTRY_GRADUATION).upper()
        self.context_id = context_id
        self.on_decision = on_decision
        self.governance_result: dict[str, Any] | None = None

        # Title by mode
        if self.mode == self.MODE_BASE_PIPELINE_PROMOTION:
            mode_title = "Base Pipeline Promotion Governance"
        elif self.mode == self.MODE_FEATURE_DEPRECATION:
            mode_title = "Feature Deprecation & Retirement Governance"
        else:
            mode_title = "Feature Registry Graduation Governance"

        self.title(f"{mode_title} — {feature_name}")
        self.transient(master.winfo_toplevel())
        self.geometry("920x760")
        self.minsize(800, 620)

        # 1. Compile or load Dossier & Engine Evaluations
        if precompiled_dossier:
            self.dossier = dict(precompiled_dossier)
            if "prerequisites_evaluation" not in self.dossier:
                self.dossier["prerequisites_evaluation"] = evaluate_graduation_prerequisites(
                    data_dir, feature_name, context_id=context_id, precompiled_dossier=self.dossier
                )
        else:
            try:
                self.dossier = compile_feature_evidence_dossier(data_dir, feature_name, context_id=context_id)
            except Exception:
                self.dossier = {"feature_name": feature_name, "context_id": context_id or "unknown"}

        self.eval_res = self.dossier.get("prerequisites_evaluation") or {}
        self.status = str(self.eval_res.get("status") or "NOT_READY")

        # Specific Mode Evaluations
        self.base_elig_res = evaluate_base_pipeline_eligibility(data_dir, feature_name, precompiled_dossier=self.dossier)
        self.depr_eval_res = evaluate_deprecation_prerequisites(data_dir, feature_name)
        self.in_base_runtime = is_feature_in_base_pipeline(data_dir, feature_name)

        # 2. Build UI layout
        self._build_ui()
        self._validate_form()

    def _build_ui(self) -> None:
        main_container = ttk.Frame(self, padding=12)
        main_container.pack(fill="both", expand=True)

        # Notebook Tabs: 1. Governance Review, 2. Raw Dossier, 3. Audit History
        self.nb = ttk.Notebook(main_container)
        self.nb.pack(fill="both", expand=True, pady=(0, 10))

        tab_review = ttk.Frame(self.nb, padding=8)
        tab_dossier = ttk.Frame(self.nb, padding=8)
        tab_audit = ttk.Frame(self.nb, padding=8)

        tab_title = "  ⚖️ Governance Review  "
        if self.mode == self.MODE_BASE_PIPELINE_PROMOTION:
            tab_title = "  🚀 Base Pipeline Promotion  "
        elif self.mode == self.MODE_FEATURE_DEPRECATION:
            tab_title = "  🛑 Feature Deprecation  "

        self.nb.add(tab_review, text=tab_title)
        self.nb.add(tab_dossier, text="  📊 Full Evidence Dossier  ")
        self.nb.add(tab_audit, text="  📜 Audit Log History  ")

        # Build tabs
        if self.mode == self.MODE_BASE_PIPELINE_PROMOTION:
            self._build_base_pipeline_tab(tab_review)
        elif self.mode == self.MODE_FEATURE_DEPRECATION:
            self._build_deprecation_tab(tab_review)
        else:
            self._build_registry_graduation_tab(tab_review)

        self._build_dossier_tab(tab_dossier)
        self._build_audit_tab(tab_audit)

        # Bottom Action Bar
        self._build_action_bar(main_container)

    # -------------------------------------------------------------------------
    # TAB 1A: REGISTRY GRADUATION MODE
    # -------------------------------------------------------------------------
    def _build_registry_graduation_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(canvas_window, width=event.width))

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 1. Feature Identity & Context Banner
        id_frame = ttk.LabelFrame(scroll_frame, text="Feature Identity & Scope Context", padding=8)
        id_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(id_frame, text=f"Feature Name: {self.feature_name}", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(id_frame, text=f"Source: {self.dossier.get('feature_source', 'experimental').upper()}").grid(
            row=0, column=1, sticky="w", padx=15, pady=2
        )
        ttk.Label(id_frame, text=f"Context ID: {self.dossier.get('context_id', 'unknown')}").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Label(
            id_frame,
            text=f"Market: {self.dossier.get('market', '—')} | Interval: {self.dossier.get('sampling_interval_sec', 0)}s | Project: {self.dossier.get('feature_project_id', 'all')}",
        ).grid(row=1, column=1, sticky="w", padx=15, pady=2)

        # 2. Classification Banner
        class_frame = ttk.LabelFrame(scroll_frame, text="Graduation Classification", padding=8)
        class_frame.pack(fill="x", pady=(0, 8))

        if self.status == "UNIVERSAL_READY":
            status_text = "🟢 UNIVERSAL READY — Qualified across multiple contexts (K >= 2, G >= 0.50)"
            sub_text = "• Universal Feature Registry graduation approved.\n• Eligible for future Base Pipeline Promotion review."
            bg_color = "#e6f4ea"
            fg_color = "#137333"
        elif self.status == "CONTEXT_SCOPED_READY":
            status_text = "🔵 CONTEXT-SCOPED READY — Qualified for specific context only"
            sub_text = f"• Feature Registry graduation permitted with strict context lock: [{self.dossier.get('context_id')}].\n• Universal Base Pipeline promotion is PROHIBITED."
            bg_color = "#e8f0fe"
            fg_color = "#1a73e8"
        else:
            status_text = "🔴 NOT READY FOR GRADUATION — Prerequisites not satisfied"
            failed_str = ", ".join(self.eval_res.get("failed_checks") or ["Criteria unmet"])
            sub_text = f"• Graduation blocked: {failed_str}.\n• Feature remains in experimental validation cycle."
            bg_color = "#fce8e6"
            fg_color = "#c5221f"

        lbl_status = tk.Label(
            class_frame, text=status_text, font=("TkDefaultFont", 10, "bold"), bg=bg_color, fg=fg_color, padx=8, pady=4, anchor="w"
        )
        lbl_status.pack(fill="x", pady=(0, 4))
        lbl_sub = tk.Label(class_frame, text=sub_text, font=("TkDefaultFont", 9), bg=bg_color, fg="#333", padx=8, justify="left", anchor="w")
        lbl_sub.pack(fill="x")

        # 3. Qualification Checklist
        chk_frame = ttk.LabelFrame(scroll_frame, text="Graduation Prerequisites Checklist", padding=8)
        chk_frame.pack(fill="x", pady=(0, 8))

        checks = self.eval_res.get("checks") or {}
        row_idx = 0
        for key, info in checks.items():
            passed = bool(info.get("passed", False))
            mark = "✓ PASS" if passed else "✗ FAIL"
            color = "#107c41" if passed else "#d83b01"
            desc = str(info.get("description") or key)

            ttk.Label(chk_frame, text=mark, font=("TkDefaultFont", 9, "bold"), foreground=color).grid(
                row=row_idx, column=0, sticky="w", padx=(4, 10), pady=2
            )
            ttk.Label(chk_frame, text=desc).grid(row=row_idx, column=1, sticky="w", pady=2)
            row_idx += 1

        # 4. Human-Entered Registry Metadata
        meta_frame = ttk.LabelFrame(scroll_frame, text="Feature Registry Specification (Editable)", padding=8)
        meta_frame.pack(fill="x", pady=(0, 8))

        # Domain
        ttk.Label(meta_frame, text="Domain *:").grid(row=0, column=0, sticky="w", pady=3)
        self.var_domain = tk.StringVar(value="order_flow")
        self.cb_domain = ttk.Combobox(
            meta_frame,
            textvariable=self.var_domain,
            values=["order_flow", "volatility", "statistical", "microstructure", "momentum", "options_greek", "price_action"],
            width=28,
        )
        self.cb_domain.grid(row=0, column=1, sticky="w", padx=8, pady=3)
        self.cb_domain.bind("<<ComboboxSelected>>", lambda e: self._validate_form())
        self.cb_domain.bind("<KeyRelease>", lambda e: self._validate_form())

        # Feature Group
        ttk.Label(meta_frame, text="Feature Group *:").grid(row=0, column=2, sticky="w", padx=(15, 0), pady=3)
        self.var_group = tk.StringVar(value="microstructure")
        self.cb_group = ttk.Combobox(
            meta_frame,
            textvariable=self.var_group,
            values=["microstructure", "spread_analysis", "imbalance", "skew", "volatility_surface", "custom"],
            width=28,
        )
        self.cb_group.grid(row=0, column=3, sticky="w", padx=8, pady=3)
        self.cb_group.bind("<<ComboboxSelected>>", lambda e: self._validate_form())
        self.cb_group.bind("<KeyRelease>", lambda e: self._validate_form())

        # Expected Data Type
        ttk.Label(meta_frame, text="Data Type:").grid(row=1, column=0, sticky="w", pady=3)
        self.var_dtype = tk.StringVar(value="float")
        self.cb_dtype = ttk.Combobox(meta_frame, textvariable=self.var_dtype, values=["float", "int", "bool", "string"], width=28, state="readonly")
        self.cb_dtype.grid(row=1, column=1, sticky="w", padx=8, pady=3)

        # Allowed Contexts
        ttk.Label(meta_frame, text="Allowed Contexts:").grid(row=1, column=2, sticky="w", padx=(15, 0), pady=3)
        init_allowed = "ALL" if self.status == "UNIVERSAL_READY" else str(self.dossier.get("context_id", "ALL"))
        self.var_allowed_ctx = tk.StringVar(value=init_allowed)
        self.ent_allowed_ctx = ttk.Entry(meta_frame, textvariable=self.var_allowed_ctx, width=30)
        self.ent_allowed_ctx.grid(row=1, column=3, sticky="w", padx=8, pady=3)
        if self.status == "CONTEXT_SCOPED_READY":
            self.ent_allowed_ctx.configure(state="readonly")

        # Description
        ttk.Label(meta_frame, text="Description *:").grid(row=2, column=0, sticky="w", pady=3)
        self.var_desc = tk.StringVar(value=f"Production graduated feature for {self.feature_name}")
        self.ent_desc = ttk.Entry(meta_frame, textvariable=self.var_desc, width=70)
        self.ent_desc.grid(row=2, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
        self.ent_desc.bind("<KeyRelease>", lambda e: self._validate_form())

        # Formula
        ttk.Label(meta_frame, text="Formula / Logic:").grid(row=3, column=0, sticky="w", pady=3)
        self.var_formula = tk.StringVar(value=f"calc_{self.feature_name}(ohlcv)")
        self.ent_formula = ttk.Entry(meta_frame, textvariable=self.var_formula, width=70)
        self.ent_formula.grid(row=3, column=1, columnspan=3, sticky="ew", padx=8, pady=3)

        # Reviewer Notes
        ttk.Label(meta_frame, text="Reviewer Notes *:").grid(row=4, column=0, sticky="w", pady=3)
        self.var_notes = tk.StringVar(value="Graduation reviewed and verified via Phase 3D governance dialog.")
        self.ent_notes = ttk.Entry(meta_frame, textvariable=self.var_notes, width=70)
        self.ent_notes.grid(row=4, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
        self.ent_notes.bind("<KeyRelease>", lambda e: self._validate_form())

    # -------------------------------------------------------------------------
    # TAB 1B: BASE PIPELINE PROMOTION MODE
    # -------------------------------------------------------------------------
    def _build_base_pipeline_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(canvas_window, width=event.width))

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Identity & Membership Frame
        id_frame = ttk.LabelFrame(scroll_frame, text="Base Pipeline Candidate Identity", padding=8)
        id_frame.pack(fill="x", pady=(0, 8))

        fid_display = self.base_elig_res.get("feature_id") or "UNREGISTERED"
        in_base_str = "ACTIVE IN PL_0001" if self.in_base_runtime else "NOT IN BASE PIPELINE"
        in_base_color = "#107c41" if self.in_base_runtime else "#555"

        ttk.Label(id_frame, text=f"Feature: {self.feature_name} ({fid_display})", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(id_frame, text=f"Base Pipeline Status: {in_base_str}", foreground=in_base_color, font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=15, pady=2
        )
        ttk.Label(id_frame, text=f"Classification: {self.base_elig_res.get('status')}").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(id_frame, text=f"Primary Context: {self.dossier.get('context_id', 'unknown')}").grid(row=1, column=1, sticky="w", padx=15, pady=2)

        # Base Eligibility Status Banner
        stat_frame = ttk.LabelFrame(scroll_frame, text="Base Pipeline Qualification Standing", padding=8)
        stat_frame.pack(fill="x", pady=(0, 8))

        if self.base_elig_res.get("is_eligible"):
            status_text = "🟢 ELIGIBLE FOR BASE PIPELINE PROMOTION"
            sub_text = "• All 8 Base Pipeline prerequisites passed.\n• Feature is eligible to enter Universal PL_0001 membership."
            bg_color = "#e6f4ea"
            fg_color = "#137333"
        elif self.base_elig_res.get("status") == "CONTEXT_SCOPED_PROHIBITED":
            status_text = "🔵 CONTEXT-SCOPED FEATURE — BASE PIPELINE PROHIBITED"
            sub_text = "• Feature is context-scoped (K < 2 or G < 0.50).\n• Universal Base Pipeline promotion is strictly rejected."
            bg_color = "#e8f0fe"
            fg_color = "#1a73e8"
        elif self.base_elig_res.get("status") == "NOT_GRADUATED":
            status_text = "🔴 NOT GRADUATED — GRADUATION REQUIRED FIRST"
            sub_text = "• Feature has not been graduated into Feature Registry.\n• Run Registry Graduation before Base Pipeline promotion."
            bg_color = "#fce8e6"
            fg_color = "#c5221f"
        else:
            status_text = "🔴 NOT READY FOR BASE PIPELINE PROMOTION"
            failed_str = ", ".join(self.base_elig_res.get("failed_checks") or ["Criteria unmet"])
            sub_text = f"• Base Pipeline promotion blocked: {failed_str}."
            bg_color = "#fce8e6"
            fg_color = "#c5221f"

        lbl_stat = tk.Label(stat_frame, text=status_text, font=("TkDefaultFont", 10, "bold"), bg=bg_color, fg=fg_color, padx=8, pady=4, anchor="w")
        lbl_stat.pack(fill="x", pady=(0, 4))
        lbl_sub = tk.Label(stat_frame, text=sub_text, font=("TkDefaultFont", 9), bg=bg_color, fg="#333", padx=8, justify="left", anchor="w")
        lbl_sub.pack(fill="x")

        # Prerequisites Checklist
        chk_frame = ttk.LabelFrame(scroll_frame, text="Base Pipeline Prerequisites (Stricter Gate)", padding=8)
        chk_frame.pack(fill="x", pady=(0, 8))

        checks = self.base_elig_res.get("checks") or {}
        row_idx = 0
        for key, info in checks.items():
            passed = bool(info.get("passed", False))
            mark = "✓ PASS" if passed else "✗ FAIL"
            color = "#107c41" if passed else "#d83b01"
            desc = str(info.get("description") or key)

            ttk.Label(chk_frame, text=mark, font=("TkDefaultFont", 9, "bold"), foreground=color).grid(
                row=row_idx, column=0, sticky="w", padx=(4, 10), pady=2
            )
            ttk.Label(chk_frame, text=desc).grid(row=row_idx, column=1, sticky="w", pady=2)
            row_idx += 1

        # Governance Sign-Off Frame
        gov_frame = ttk.LabelFrame(scroll_frame, text="Engineering Review & Latency Sign-Off", padding=8)
        gov_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(gov_frame, text="Reviewer Identity *:").grid(row=0, column=0, sticky="w", pady=3)
        self.var_reviewer = tk.StringVar(value="Lead Quantitative Reviewer")
        self.ent_reviewer = ttk.Entry(gov_frame, textvariable=self.var_reviewer, width=32)
        self.ent_reviewer.grid(row=0, column=1, sticky="w", padx=8, pady=3)
        self.ent_reviewer.bind("<KeyRelease>", lambda e: self._validate_form())

        ttk.Label(gov_frame, text="Latency Compliant *:").grid(row=0, column=2, sticky="w", padx=(15, 0), pady=3)
        self.var_latency = tk.StringVar(value="Yes")
        self.cb_latency = ttk.Combobox(gov_frame, textvariable=self.var_latency, values=["Yes", "No"], width=10, state="readonly")
        self.cb_latency.grid(row=0, column=3, sticky="w", padx=8, pady=3)
        self.cb_latency.bind("<<ComboboxSelected>>", lambda e: self._validate_form())

        ttk.Label(gov_frame, text="Reviewer Notes *:").grid(row=1, column=0, sticky="w", pady=3)
        self.var_notes = tk.StringVar(value="Latency budget benchmarked and approved for Base Pipeline membership.")
        self.ent_notes = ttk.Entry(gov_frame, textvariable=self.var_notes, width=70)
        self.ent_notes.grid(row=1, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
        self.ent_notes.bind("<KeyRelease>", lambda e: self._validate_form())

    # -------------------------------------------------------------------------
    # TAB 1C: FEATURE DEPRECATION MODE
    # -------------------------------------------------------------------------
    def _build_deprecation_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(canvas_window, width=event.width))

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Feature Identity & Status
        id_frame = ttk.LabelFrame(scroll_frame, text="Feature Retirement Target", padding=8)
        id_frame.pack(fill="x", pady=(0, 8))

        fid_display = self.depr_eval_res.get("feature_id") or "UNKNOWN"
        in_base_str = "YES (will be demoted from PL_0001)" if self.in_base_runtime else "NO (Registry only)"

        ttk.Label(id_frame, text=f"Feature: {self.feature_name} ({fid_display})", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Label(id_frame, text=f"In Base Pipeline: {in_base_str}", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=15, pady=2
        )
        ttk.Label(id_frame, text=f"Implementation Status: {self.depr_eval_res.get('implementation_status')}").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Label(id_frame, text=f"Deprecation Standing: {self.depr_eval_res.get('status')}").grid(
            row=1, column=1, sticky="w", padx=15, pady=2
        )

        # Warning Banner
        warn_frame = ttk.LabelFrame(scroll_frame, text="Permanent Deprecation Notice", padding=8)
        warn_frame.pack(fill="x", pady=(0, 8))

        warning_text = (
            "⚠ PERMANENT FEATURE DEPRECATION / RETIREMENT\n\n"
            "• The feature will be removed from PL_0001 Base Pipeline membership if present.\n"
            "• The permanent FRxxxx identity will NOT be deleted or reused.\n"
            "• Historical model packages and datasets will remain 100% valid and unaffected.\n"
            "• In Phase 3A, this feature will receive EXCLUDE / DEPRECATED_FEATURE and will be barred from new training candidate presets."
        )
        lbl_warn = tk.Label(
            warn_frame, text=warning_text, font=("TkDefaultFont", 9, "bold"), bg="#fce8e6", fg="#c5221f", justify="left", padx=10, pady=8
        )
        lbl_warn.pack(fill="x")

        # Deprecation Form
        dep_frame = ttk.LabelFrame(scroll_frame, text="Deprecation Review & Governance Sign-Off", padding=8)
        dep_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(dep_frame, text="Reviewer Identity *:").grid(row=0, column=0, sticky="w", pady=3)
        self.var_reviewer = tk.StringVar(value="Chief Architect")
        self.ent_reviewer = ttk.Entry(dep_frame, textvariable=self.var_reviewer, width=32)
        self.ent_reviewer.grid(row=0, column=1, sticky="w", padx=8, pady=3)
        self.ent_reviewer.bind("<KeyRelease>", lambda e: self._validate_form())

        ttk.Label(dep_frame, text="Deprecation Reason *:").grid(row=1, column=0, sticky="w", pady=3)
        self.var_dep_reason = tk.StringVar(value="Replaced by higher-performance microstructure feature")
        self.cb_dep_reason = ttk.Combobox(
            dep_frame,
            textvariable=self.var_dep_reason,
            values=[
                "Replaced by higher-performance microstructure feature",
                "Severe health / degradation alert",
                "Excessive latency / compute budget overhead",
                "Market microstructure regime obsolescence",
                "Duplicate / redundant logic with base feature",
                "Custom retirement reason",
            ],
            width=50,
        )
        self.cb_dep_reason.grid(row=1, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
        self.cb_dep_reason.bind("<<ComboboxSelected>>", lambda e: self._validate_form())
        self.cb_dep_reason.bind("<KeyRelease>", lambda e: self._validate_form())

        ttk.Label(dep_frame, text="Reviewer Notes *:").grid(row=2, column=0, sticky="w", pady=3)
        self.var_notes = tk.StringVar(value="Deprecation audited and approved. Historical models remain intact.")
        self.ent_notes = ttk.Entry(dep_frame, textvariable=self.var_notes, width=70)
        self.ent_notes.grid(row=2, column=1, columnspan=3, sticky="ew", padx=8, pady=3)
        self.ent_notes.bind("<KeyRelease>", lambda e: self._validate_form())

        # Mandatory Acknowledgments
        ack_frame = ttk.Frame(dep_frame)
        ack_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 2))

        self.var_ack_future_exclude = tk.BooleanVar(value=False)
        self.chk_ack_future = ttk.Checkbutton(
            ack_frame,
            text="I confirm that this feature will be excluded from future training candidate generation.",
            variable=self.var_ack_future_exclude,
            command=self._validate_form,
        )
        self.chk_ack_future.pack(anchor="w", pady=2)

        self.var_ack_hist_compat = tk.BooleanVar(value=False)
        self.chk_ack_hist = ttk.Checkbutton(
            ack_frame,
            text="I confirm that historical model compatibility has been reviewed and preserved.",
            variable=self.var_ack_hist_compat,
            command=self._validate_form,
        )
        self.chk_ack_hist.pack(anchor="w", pady=2)

    # -------------------------------------------------------------------------
    # TAB 2: FULL EVIDENCE DOSSIER
    # -------------------------------------------------------------------------
    def _build_dossier_tab(self, parent: ttk.Frame) -> None:
        txt = tk.Text(parent, wrap="word", font=("Consolas", 9), background="#fafafa", relief="flat")
        sb = ttk.Scrollbar(parent, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        lines: list[str] = [
            f"=== EVIDENCE DOSSIER: {self.feature_name} ===",
            f"Context ID: {self.dossier.get('context_id', 'N/A')}",
            f"Market / Interval: {self.dossier.get('market', 'N/A')} / {self.dossier.get('sampling_interval_sec', 'N/A')}s",
            f"Feature Project ID: {self.dossier.get('feature_project_id', 'N/A')}",
            f"Base Pipeline Runtime Membership (PL_0001): {self.in_base_runtime}",
            "",
            "--- CUMULATIVE EVIDENCE METRICS ---",
            f"Total Validation Runs: {self.dossier.get('total_validation_runs', 'N/A')}",
            f"Unique Model Architectures: {self.dossier.get('unique_model_count', 'N/A')}",
            f"Consecutive KEEP Streak: {self.dossier.get('consecutive_keep_count', 'N/A')}",
            f"Keep / Watch / Remove Runs: {self.dossier.get('keep_runs', 'N/A')} / {self.dossier.get('watch_runs', 'N/A')} / {self.dossier.get('remove_runs', 'N/A')}",
            f"Lineage Evidence Score: {self.dossier.get('lineage_evidence_score', 0.0):+.1f}",
            f"Evidence Volume Confidence: {float(self.dossier.get('evidence_confidence', 0.0)):.2%}",
            f"Dominant Model Recommendation: {self.dossier.get('dominant_recommendation', 'N/A')}",
            f"Freshness: {self.dossier.get('freshness', {}).get('display_text', 'N/A') if isinstance(self.dossier.get('freshness'), dict) else 'N/A'}",
            "",
            "--- STABILITY & VOLATILITY ---",
            f"Score Volatility (σ_S): {self.dossier.get('score_volatility', 'N/A')}",
            f"Score Trajectory Range: {self.dossier.get('score_range', 'N/A')}",
            f"Direction Flips: {self.dossier.get('direction_flips', 'N/A')}",
            f"Stability Band: {self.dossier.get('stability_label', 'N/A')}",
            "",
            "--- CROSS-CONTEXT GENERALIZATION ---",
            f"Comparable Context Count (K): {self.dossier.get('comparable_context_count', 'N/A')}",
            f"Generalization Index (G): {self.dossier.get('generalization_score', 'N/A')}",
            "",
            "--- PHASE 3A DECISION STANDING ---",
            f"Phase 3A Training Decision: {self.dossier.get('phase_3a_decision', 'N/A')}",
            f"Phase 3A Primary Reason: {self.dossier.get('phase_3a_primary_reason', 'N/A')}",
            f"Promotion Candidate Standing: {self.dossier.get('is_phase_3a_promotion_qualified', False)}",
            f"Health Status: {self.dossier.get('health_status', 'HEALTHY')}",
            "",
            "--- RECENT VALIDATION RUNS ---",
        ]

        history = self.dossier.get("validation_history") or []
        for idx, h in enumerate(history[-10:]):
            lines.append(
                f"  [{idx + 1}] Model: {h.get('model_name', '—')} | Rec: {h.get('recommendation', '—')} | "
                f"Holdout: #{h.get('holdout_rank', '—')} | Unseen: #{h.get('unseen_rank', '—')} | "
                f"Time: {h.get('run_timestamp', '—')}"
            )
        if not history:
            lines.append("  (No raw runs recorded)")

        txt.insert("end", "\n".join(lines))
        txt.configure(state="disabled")

    # -------------------------------------------------------------------------
    # TAB 3: AUDIT LOG HISTORY (READ-ONLY)
    # -------------------------------------------------------------------------
    def _build_audit_tab(self, parent: ttk.Frame) -> None:
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True)

        cols = ("event_type", "feature", "feature_id", "prev_source", "new_source", "reviewer", "timestamp")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)

        tree.heading("event_type", text="Event Type")
        tree.heading("feature", text="Feature")
        tree.heading("feature_id", text="ID")
        tree.heading("prev_source", text="Previous")
        tree.heading("new_source", text="New Source")
        tree.heading("reviewer", text="Reviewer")
        tree.heading("timestamp", text="Timestamp (UTC)")

        tree.column("event_type", width=180, anchor="w")
        tree.column("feature", width=160, anchor="w")
        tree.column("feature_id", width=80, anchor="center")
        tree.column("prev_source", width=90, anchor="center")
        tree.column("new_source", width=90, anchor="center")
        tree.column("reviewer", width=140, anchor="w")
        tree.column("timestamp", width=150, anchor="w")

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        logs = get_feature_graduation_audit_log(self.data_dir)
        for entry in reversed(logs):
            etype = str(entry.get("event_type") or "GRADUATION")
            fname = str(entry.get("feature_name") or "—")
            fid = str(entry.get("assigned_feature_id") or "—")
            prev_s = str(entry.get("previous_source") or "—")
            new_s = str(entry.get("new_source") or "—")
            rev = str(entry.get("reviewer_information") or "—")
            ts = str(entry.get("timestamp") or "—")[:19].replace("T", " ")

            tree.insert("", "end", values=(etype, fname, fid, prev_s, new_s, rev, ts))

        if not logs:
            ttk.Label(parent, text="No governance events recorded yet in feature_graduation_audit_log.json.", foreground="#666").pack(
                pady=10
            )

    # -------------------------------------------------------------------------
    # ACTION BAR & FORM VALIDATION
    # -------------------------------------------------------------------------
    def _build_action_bar(self, parent: ttk.Frame) -> None:
        btn_bar = ttk.Frame(parent)
        btn_bar.pack(fill="x", pady=(8, 0))

        self.lbl_warning = ttk.Label(btn_bar, text="", foreground="#d83b01", font=("TkDefaultFont", 9, "bold"))
        self.lbl_warning.pack(side="left", padx=4)

        btn_close = ttk.Button(btn_bar, text="Close", command=self.destroy)
        btn_close.pack(side="right", padx=4)

        btn_reject = ttk.Button(btn_bar, text="Reject / Defer", command=self._on_reject)
        btn_reject.pack(side="right", padx=4)

        # Primary Approval Button Label by Mode
        if self.mode == self.MODE_BASE_PIPELINE_PROMOTION:
            btn_text = "Approve Base Pipeline Promotion"
            btn_cmd = self._on_approve_base_promotion
        elif self.mode == self.MODE_FEATURE_DEPRECATION:
            btn_text = "Deprecate Feature"
            btn_cmd = self._on_approve_deprecation
        else:
            btn_text = "Approve Registry Graduation"
            btn_cmd = self._on_approve_registry_graduation

        self.btn_approve = ttk.Button(btn_bar, text=btn_text, command=btn_cmd, style="Accent.TButton")
        self.btn_approve.pack(side="right", padx=4)

    def _validate_form(self) -> None:
        """Validates form requirements according to the active governance mode."""
        errors: list[str] = []

        if self.mode == self.MODE_BASE_PIPELINE_PROMOTION:
            if not self.base_elig_res.get("is_eligible"):
                errors.append(f"Not eligible: {self.base_elig_res.get('status')}")
            if hasattr(self, "var_latency") and self.var_latency.get().strip().lower() != "yes":
                errors.append("Latency budget compliance confirmation is required.")
            if hasattr(self, "var_reviewer") and not self.var_reviewer.get().strip():
                errors.append("Reviewer identity is required.")
            if hasattr(self, "var_notes") and not self.var_notes.get().strip():
                errors.append("Reviewer notes are required.")

        elif self.mode == self.MODE_FEATURE_DEPRECATION:
            if not self.depr_eval_res.get("is_eligible_for_deprecation"):
                errors.append(f"Cannot deprecate: {self.depr_eval_res.get('status')}")
            if hasattr(self, "var_reviewer") and not self.var_reviewer.get().strip():
                errors.append("Reviewer identity is required.")
            if hasattr(self, "var_dep_reason") and not self.var_dep_reason.get().strip():
                errors.append("Deprecation reason is required.")
            if hasattr(self, "var_notes") and not self.var_notes.get().strip():
                errors.append("Reviewer notes are required.")
            if hasattr(self, "var_ack_future_exclude") and not self.var_ack_future_exclude.get():
                errors.append("Acknowledgment of future candidate exclusion is required.")
            if hasattr(self, "var_ack_hist_compat") and not self.var_ack_hist_compat.get():
                errors.append("Confirmation of historical model compatibility is required.")

        else:  # REGISTRY_GRADUATION
            if self.status == "NOT_READY":
                errors.append("Feature is NOT_READY (failed prerequisite checks).")
            if hasattr(self, "var_domain") and not self.var_domain.get().strip():
                errors.append("Domain is required.")
            if hasattr(self, "var_group") and not self.var_group.get().strip():
                errors.append("Feature Group is required.")
            if hasattr(self, "var_desc") and not self.var_desc.get().strip():
                errors.append("Description is required.")
            if hasattr(self, "var_notes") and not self.var_notes.get().strip():
                errors.append("Reviewer notes are required.")

        if errors:
            self.lbl_warning.configure(text=f"⚠ {errors[0]}")
            self.btn_approve.configure(state="disabled")
        else:
            self.lbl_warning.configure(text="")
            self.btn_approve.configure(state="normal")

    # -------------------------------------------------------------------------
    # GOVERNANCE ACTION EXECUTIONS
    # -------------------------------------------------------------------------
    def _build_approval_payload(self) -> dict[str, Any]:
        """Constructs the comprehensive Phase 3D.2/3D.3 approval payload."""
        now_utc = datetime.now(timezone.utc).isoformat()
        is_univ = (self.status == "UNIVERSAL_READY")
        allowed = ["ALL"] if is_univ else [str(self.dossier.get("context_id") or "UNKNOWN")]

        payload: dict[str, Any] = {
            "action": "APPROVE",
            "feature_name": self.feature_name,
            "context_id": self.dossier.get("context_id"),
            "scope_classification": self.status,
            "is_universal_ready": is_univ,
            "is_context_scoped_ready": True,
            "is_base_pipeline_eligible": is_univ,
            "allowed_contexts": allowed,
            "domain": self.var_domain.get().strip(),
            "group": self.var_group.get().strip(),
            "expected_data_type": self.var_dtype.get().strip(),
            "formula": self.var_formula.get().strip(),
            "description": self.var_desc.get().strip(),
            "reviewer_notes": self.var_notes.get().strip(),
            "reviewer": "Reviewer Alice",
            "approved_at": now_utc,
            "dossier_snapshot": {
                "total_validation_runs": self.dossier.get("total_validation_runs"),
                "unique_model_count": self.dossier.get("unique_model_count"),
                "consecutive_keep_count": self.dossier.get("consecutive_keep_count"),
                "lineage_evidence_score": self.dossier.get("lineage_evidence_score"),
                "evidence_confidence": self.dossier.get("evidence_confidence"),
                "score_volatility": self.dossier.get("score_volatility"),
                "generalization_score": self.dossier.get("generalization_score"),
                "comparable_context_count": self.dossier.get("comparable_context_count"),
            },
        }
        return payload

    def _on_approve(self) -> None:
        """Phase 3D.2 in-memory approval payload generator (backward compatibility)."""
        if self.status == "NOT_READY":
            messagebox.showwarning("Graduation Blocked", "This feature is NOT_READY for graduation.")
            return

        payload = self._build_approval_payload()
        self.governance_result = payload

        if self.on_decision:
            try:
                self.on_decision(payload)
            except Exception as exc:
                messagebox.showerror("Callback Error", f"Governance callback error: {exc}")
                return

        messagebox.showinfo(
            "Graduation Decision Recorded",
            f"Governance approval recorded for '{self.feature_name}' ({self.status}).\n\n"
            "Note: In Phase 3D.2, this approval payload is held in memory for Phase 3D.3 registry registration.",
        )
        self.destroy()

    def _on_approve_registry_graduation(self) -> None:
        """Executes Registry Graduation via Phase 3D.3 transactional writer."""
        if self.status == "NOT_READY":
            messagebox.showwarning("Graduation Blocked", "This feature is NOT_READY for graduation.")
            return

        payload = self._build_approval_payload()

        # Confirmation dialog
        if not messagebox.askyesno(
            "Confirm Feature Registry Graduation",
            f"Graduate feature '{self.feature_name}' into Feature Registry with {self.status}?\n\n"
            "This will assign a permanent sequential FRxxxx identity.",
        ):
            return

        exec_res = execute_registry_graduation(self.data_dir, self.feature_name, payload)
        self.governance_result = exec_res

        if exec_res.get("status") == "SUCCESS":
            fid = exec_res.get("assigned_feature_id")
            messagebox.showinfo(
                "Graduation Succeeded",
                f"Feature '{self.feature_name}' successfully graduated as {fid} ({self.status}).\n\n"
                f"Audit Event ID: {exec_res.get('audit_event_id')}",
            )
            if self.on_decision:
                try:
                    self.on_decision(exec_res)
                except Exception:
                    pass
            self.destroy()
        else:
            msg = exec_res.get("message") or f"Graduation failed with status: {exec_res.get('status')}"
            messagebox.showerror("Graduation Failed", msg)

    def _on_approve_base_promotion(self) -> None:
        """Executes Base Pipeline Promotion via Phase 3D.4A transactional engine."""
        if not self.base_elig_res.get("is_eligible"):
            messagebox.showwarning("Promotion Blocked", "This feature is not eligible for Base Pipeline promotion.")
            return

        fid = self.base_elig_res.get("feature_id")
        payload: dict[str, Any] = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": self.var_reviewer.get().strip(),
            "reviewer_notes": self.var_notes.get().strip(),
        }

        # Confirmation dialog
        if not messagebox.askyesno(
            "Confirm Base Pipeline Promotion",
            f"This action will add {fid} ({self.feature_name}) to PL_0001 Base Pipeline membership.\n\n"
            "Base Pipeline features are automatically included in all baseline datasets.\n\n"
            "Proceed with promotion?",
        ):
            return

        exec_res = execute_base_pipeline_promotion(self.data_dir, self.feature_name, payload)
        self.governance_result = exec_res

        if exec_res.get("status") == "SUCCESS":
            messagebox.showinfo(
                "Base Pipeline Promotion Succeeded",
                f"Feature '{self.feature_name}' ({fid}) successfully promoted to Base Pipeline PL_0001.\n\n"
                f"Audit Event ID: {exec_res.get('audit_event_id')}",
            )
            if self.on_decision:
                try:
                    self.on_decision(exec_res)
                except Exception:
                    pass
            self.destroy()
        else:
            msg = exec_res.get("message") or f"Promotion failed with status: {exec_res.get('status')}"
            messagebox.showerror("Promotion Failed", msg)

    def _on_approve_deprecation(self) -> None:
        """Executes Feature Deprecation via Phase 3D.4B transactional engine."""
        if not self.depr_eval_res.get("is_eligible_for_deprecation"):
            messagebox.showwarning("Deprecation Blocked", "This feature cannot be deprecated.")
            return

        fid = self.depr_eval_res.get("feature_id")
        payload: dict[str, Any] = {
            "action": "DEPRECATE",
            "reviewer_information": self.var_reviewer.get().strip(),
            "deprecation_reason": self.var_dep_reason.get().strip(),
            "reviewer_notes": self.var_notes.get().strip(),
        }

        # Strong confirmation dialog
        demote_note = f"• {fid} will be evicted from PL_0001 Base Pipeline.\n" if self.in_base_runtime else ""
        confirm_msg = (
            f"Feature: {self.feature_name}\n"
            f"Feature ID: {fid}\n\n"
            f"{demote_note}"
            "• The permanent FRxxxx identity will NOT be deleted.\n"
            "• The FRxxxx ID will NOT be reused.\n"
            "• Historical model packages and datasets will remain intact.\n"
            "• The feature will receive EXCLUDE / DEPRECATED_FEATURE in Phase 3A.\n\n"
            "Are you sure you want to permanently deprecate this feature?"
        )

        if not messagebox.askyesno("CONFIRM PERMANENT FEATURE DEPRECATION", confirm_msg):
            return

        exec_res = execute_feature_deprecation(self.data_dir, self.feature_name, payload)
        self.governance_result = exec_res

        if exec_res.get("status") == "DEPRECATED":
            messagebox.showinfo(
                "Feature Deprecated",
                f"Feature '{self.feature_name}' ({fid}) has been permanently deprecated.\n\n"
                f"Audit Event ID: {exec_res.get('audit_event_id')}",
            )
            if self.on_decision:
                try:
                    self.on_decision(exec_res)
                except Exception:
                    pass
            self.destroy()
        else:
            msg = exec_res.get("message") or f"Deprecation failed with status: {exec_res.get('status')}"
            messagebox.showerror("Deprecation Failed", msg)

    def _on_reject(self) -> None:
        """Constructs the rejection/deferral governance payload and closes dialog."""
        now_utc = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "action": "REJECT",
            "feature_name": self.feature_name,
            "context_id": self.dossier.get("context_id"),
            "scope_classification": self.status,
            "reviewer_notes": getattr(self, "var_notes", tk.StringVar()).get().strip(),
            "rejected_at": now_utc,
        }
        self.governance_result = payload

        if self.on_decision:
            try:
                self.on_decision(payload)
            except Exception as exc:
                messagebox.showerror("Callback Error", f"Governance callback error: {exc}")
                return

        messagebox.showinfo(
            "Decision Recorded",
            f"Governance review deferred/rejected for '{self.feature_name}'.",
        )
        self.destroy()


