"""Pipeline Feature Registry tab — list pipelines, membership, and rich master/detail discovery feature inspector."""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .pipeline_registry_service import (
    create_pipeline,
    delete_pipeline,
    get_pipeline,
    load_pipelines,
    registry_catalog_features,
    set_pipeline_registry_members,
)

COL_KEEP = "#2e7d32"
COL_WATCH = "#e65100"
COL_REMOVE = "#c62828"
COL_MUTED = "#666666"


def _prepare_modal_dialog(
    win: tk.Toplevel,
    anchor: tk.Misc,
    *,
    width: int | None = None,
    height: int | None = None,
    min_width: int = 320,
    min_height: int = 180,
    padding: int = 16,
) -> None:
    """Center a modal dialog over the Feature Transformations panel."""
    from .fold_replay_widgets import center_toplevel_on_widget

    win.transient(anchor.winfo_toplevel())
    win.update_idletasks()
    ww = max(int(width or 0), int(win.winfo_reqwidth()) + padding, min_width)
    wh = max(int(height or 0), int(win.winfo_reqheight()) + padding, min_height)
    win.minsize(min_width, min_height)
    win.geometry(f"{ww}x{wh}")
    center_toplevel_on_widget(win, anchor)
    win.lift()
    try:
        win.focus_force()
    except tk.TclError:
        pass
    win.grab_set()


class PipelineRegistryFeatureDialog(tk.Toplevel):
    """Select master registry features (by feature_id) for a pipeline."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        selected_ids: list[str],
        on_apply: Callable[[list[str]], None],
    ) -> None:
        super().__init__(master)
        self.title("Select Registry Features")
        _prepare_modal_dialog(self, master, width=640, height=560, min_width=640, min_height=560)
        self._features = registry_catalog_features(chart_dir)
        self._on_apply = on_apply
        self._filter_var = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")
        self._vars: dict[str, tk.BooleanVar] = {}
        selected = {str(x).strip().upper() for x in selected_ids}

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        toolbar = ttk.Frame(top)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="Search").pack(side="left")
        ent = ttk.Entry(toolbar, textvariable=self._filter_var)
        ent.pack(side="left", fill="x", expand=True, padx=(6, 8))
        ent.bind("<KeyRelease>", lambda _e: self._apply_filter())
        ttk.Label(toolbar, textvariable=self._count_var, font=("Segoe UI", 9, "bold")).pack(side="right")

        list_frame = ttk.Frame(top)
        list_frame.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical")
        self._list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=18)
        self._list.configure(yscrollcommand=sb.set)
        sb.config(command=self._list.yview)
        self._list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._rows: list[dict[str, Any]] = []

        for feat in self._features:
            fid = feat["feature_id"]
            var = tk.BooleanVar(value=fid in selected)
            self._vars[fid] = var

        self._rebuild_list()
        self._list.bind("<<ListboxSelect>>", self._on_list_select)

        actions = ttk.Frame(self, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Apply", command=self._apply).pack(side="right")
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

    def _apply_filter(self) -> None:
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        query = self._filter_var.get().strip().lower()
        self._list.delete(0, tk.END)
        self._rows = []
        for feat in self._features:
            fid = str(feat.get("feature_id") or "")
            name = str(feat.get("name") or fid)
            family = str(feat.get("family") or "")
            cat = str(feat.get("category") or "")
            haystack = f"{fid} {name} {family} {cat}".lower()
            if query and query not in haystack:
                continue
            self._rows.append(feat)
            idx = self._list.size()
            label = f"{fid} — {name} [{family}]"
            self._list.insert(tk.END, label)
            if self._vars.get(fid, tk.BooleanVar(value=False)).get():
                self._list.selection_set(idx)
        sel_count = sum(1 for v in self._vars.values() if v.get())
        self._count_var.set(f"Selected: {sel_count} / {len(self._features)}")

    def _on_list_select(self, _event: Any) -> None:
        indices = set(self._list.curselection())
        for idx, feat in enumerate(self._rows):
            fid = feat["feature_id"]
            if idx in indices:
                self._vars[fid].set(True)
            else:
                self._vars[fid].set(False)
        sel_count = sum(1 for v in self._vars.values() if v.get())
        self._count_var.set(f"Selected: {sel_count} / {len(self._features)}")

    def _apply(self) -> None:
        selected_ids = [fid for fid, v in self._vars.items() if v.get()]
        self._on_apply(selected_ids)
        self.destroy()


class CreatePipelineDialog(tk.Toplevel):
    """Create a new pipeline with specified name and type."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_create: Callable[[str, str], None],
    ) -> None:
        super().__init__(master)
        self.title("Create Pipeline")
        _prepare_modal_dialog(self, master, width=440, height=220, min_width=440, min_height=220)
        self.chart_dir = chart_dir
        self._on_create = on_create
        self._name_var = tk.StringVar(value="")
        self._type_var = tk.StringVar(value="manual")

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Pipeline Name:").grid(row=0, column=0, sticky="w", pady=(0, 8))
        name_entry = ttk.Entry(body, textvariable=self._name_var, width=32)
        name_entry.grid(row=0, column=1, sticky="w", pady=(0, 8))
        name_entry.focus_set()

        ttk.Label(body, text="Pipeline Type:").grid(row=1, column=0, sticky="w", pady=(0, 8))
        type_combo = ttk.Combobox(
            body,
            textvariable=self._type_var,
            state="readonly",
            values=["manual", "discovery_experimental"],
            width=24,
        )
        type_combo.grid(row=1, column=1, sticky="w", pady=(0, 8))

        btn_box = ttk.Frame(body)
        btn_box.grid(row=2, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(btn_box, text="Create", command=self._submit).pack(side="right")
        ttk.Button(btn_box, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

    def _submit(self) -> None:
        name = self._name_var.get().strip()
        ptype = self._type_var.get().strip().lower()
        if not name:
            messagebox.showerror("Validation Error", "Pipeline name cannot be empty.", parent=self)
            return
        self._on_create(name, ptype)
        self.destroy()


class PipelineRegistryPanel(ttk.Frame):
    """Pipeline Feature Registry management panel with two-panel master/detail discovery feature inspector."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_pipelines_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self.chart_dir = chart_dir
        self._on_pipelines_changed = on_pipelines_changed
        self._pipelines: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._detail_var = tk.StringVar(value="Select a pipeline")
        self._current_candidates: list[str] = []
        self._current_feat_prov_map: dict[str, dict[str, Any]] = {}
        self._is_discovery_pipeline: bool = False

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="Pipeline Feature Registry", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(header, text="Create Pipeline", command=self._open_create).pack(side="right")
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right", padx=(0, 8))

        # Top Table: Pipelines Overview
        p_cols = ("name", "type", "features", "status")
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="x", pady=(0, 6))
        self._tree = ttk.Treeview(tree_frame, columns=p_cols, show="headings", height=5)
        for c, label, w in (
            ("name", "Pipeline Name", 220),
            ("type", "Type", 140),
            ("features", "Features", 75),
            ("status", "Status", 85),
        ):
            self._tree.heading(c, text=label)
            self._tree.column(c, width=w, anchor="center" if c != "name" else "w")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Bottom Detail Container
        detail_frame = ttk.LabelFrame(self, text="Pipeline Detail & Feature Membership", padding=8)
        detail_frame.pack(fill="both", expand=True, pady=(2, 0))

        # Summary Header
        summary_row = ttk.Frame(detail_frame)
        summary_row.pack(fill="x", pady=(0, 6))
        ttk.Label(summary_row, textvariable=self._detail_var, font=("Segoe UI", 9, "bold"), foreground="#0d47a1").pack(side="left")

        # Action Buttons
        btn_row = ttk.Frame(detail_frame)
        btn_row.pack(fill="x", pady=(0, 6))
        self._select_features_btn = ttk.Button(btn_row, text="Select Master Registry Features", command=self._select_features)
        self._select_features_btn.pack(side="left")
        self._membership_note_lbl = ttk.Label(btn_row, text="(from Master Feature Registry — membership only)", foreground=COL_MUTED)
        self._membership_note_lbl.pack(side="left", padx=(8, 0))
        self._delete_pipeline_btn = ttk.Button(btn_row, text="Delete Pipeline", command=self._delete_pipeline)
        self._delete_pipeline_btn.pack(side="right")

        # Two-Panel Master / Detail Panedwindow
        self._paned = ttk.Panedwindow(detail_frame, orient=tk.HORIZONTAL)
        self._paned.pack(fill="both", expand=True, pady=(2, 0))

        # LEFT PANEL: Feature List / Table
        self._left_frame = ttk.LabelFrame(self._paned, text="Discovered Features", padding=6)
        self._paned.add(self._left_frame, weight=3)

        f_cols = ("name", "verdict", "strategy", "gen", "delta_auc")
        self._feat_tree = ttk.Treeview(self._left_frame, columns=f_cols, show="headings", height=12)
        self._feat_tree.heading("name", text="Feature Name")
        self._feat_tree.heading("verdict", text="Verdict")
        self._feat_tree.heading("strategy", text="Strategy")
        self._feat_tree.heading("gen", text="Gen")
        self._feat_tree.heading("delta_auc", text="ΔAUC")

        self._feat_tree.column("name", width=230, anchor="w")
        self._feat_tree.column("verdict", width=75, anchor="center")
        self._feat_tree.column("strategy", width=85, anchor="center")
        self._feat_tree.column("gen", width=45, anchor="center")
        self._feat_tree.column("delta_auc", width=70, anchor="e")

        self._feat_tree.tag_configure("tag_keep", foreground=COL_KEEP)
        self._feat_tree.tag_configure("tag_watch", foreground=COL_WATCH)
        self._feat_tree.tag_configure("tag_remove", foreground=COL_REMOVE)

        f_vsb = ttk.Scrollbar(self._left_frame, orient="vertical", command=self._feat_tree.yview)
        f_hsb = ttk.Scrollbar(self._left_frame, orient="horizontal", command=self._feat_tree.xview)
        self._feat_tree.configure(yscrollcommand=f_vsb.set, xscrollcommand=f_hsb.set)

        self._feat_tree.pack(side="left", fill="both", expand=True)
        f_vsb.pack(side="right", fill="y")
        f_hsb.pack(side="bottom", fill="x")
        self._feat_tree.bind("<<TreeviewSelect>>", self._on_feat_tree_select)

        # RIGHT PANEL: Structured Feature Details Inspector
        self._right_frame = ttk.LabelFrame(self._paned, text="Selected Feature Details", padding=8)
        self._paned.add(self._right_frame, weight=4)

        # Scrollable container for Right Details
        r_canvas = tk.Canvas(self._right_frame, borderwidth=0, highlightthickness=0)
        r_vsb = ttk.Scrollbar(self._right_frame, orient="vertical", command=r_canvas.yview)
        self._detail_content = ttk.Frame(r_canvas)

        self._detail_content.bind(
            "<Configure>",
            lambda _e: r_canvas.configure(scrollregion=r_canvas.bbox("all")),
        )
        r_canvas.create_window((0, 0), window=self._detail_content, anchor="nw")
        r_canvas.configure(yscrollcommand=r_vsb.set)

        r_canvas.pack(side="left", fill="both", expand=True)
        r_vsb.pack(side="right", fill="y")

        self._build_detail_inspector_widgets()

    def _build_detail_inspector_widgets(self) -> None:
        """Construct the structured card layout inside the right detail inspector."""
        p = self._detail_content

        # 1. Identity Section
        id_sec = ttk.LabelFrame(p, text="📌 Feature Identity", padding=6)
        id_sec.pack(fill="x", pady=(0, 6))

        ttk.Label(id_sec, text="Feature Name:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=1)
        self._d_name_val = ttk.Label(id_sec, text="—", font=("Segoe UI", 9, "bold"), foreground="#0d47a1", wraplength=380, justify="left")
        self._d_name_val.grid(row=0, column=1, sticky="w", pady=1, padx=(6, 0))

        ttk.Label(id_sec, text="Technical ID:", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=1)
        self._d_tech_id_var = tk.StringVar(value="—")
        d_tech_entry = ttk.Entry(id_sec, textvariable=self._d_tech_id_var, state="readonly", width=42, font=("Consolas", 9))
        d_tech_entry.grid(row=1, column=1, sticky="w", pady=1, padx=(6, 0))

        ttk.Label(id_sec, text="Verdict & Strategy:", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=1)
        self._d_verdict_strat_lbl = ttk.Label(id_sec, text="—", font=("Segoe UI", 9))
        self._d_verdict_strat_lbl.grid(row=2, column=1, sticky="w", pady=1, padx=(6, 0))

        # 2. Formula Section
        f_sec = ttk.LabelFrame(p, text="📐 Mathematical AST Formula", padding=6)
        f_sec.pack(fill="x", pady=(0, 6))

        self._formula_text = tk.Text(
            f_sec,
            height=3,
            width=48,
            font=("Consolas", 9),
            wrap="word",
            relief="solid",
            bd=1,
            bg="#f8f9fa",
        )
        self._formula_text.pack(fill="x", expand=True)

        # 3. Evidence & Governance Section
        ev_sec = ttk.LabelFrame(p, text="📊 Evidence & Performance", padding=6)
        ev_sec.pack(fill="x", pady=(0, 6))

        self._d_delta_auc_lbl = ttk.Label(ev_sec, text="Marginal ΔAUC: —")
        self._d_delta_auc_lbl.grid(row=0, column=0, sticky="w", pady=1)

        self._d_drift_lbl = ttk.Label(ev_sec, text="KS Drift (D_KS): —")
        self._d_drift_lbl.grid(row=0, column=1, sticky="w", pady=1, padx=(12, 0))

        self._d_evidence_lbl = ttk.Label(ev_sec, text="Evidence Score: —")
        self._d_evidence_lbl.grid(row=1, column=0, sticky="w", pady=1)

        self._d_folds_lbl = ttk.Label(ev_sec, text="Fold Consistency: —")
        self._d_folds_lbl.grid(row=1, column=1, sticky="w", pady=1, padx=(12, 0))

        ttk.Label(ev_sec, text="Governance Rationale:", font=("Segoe UI", 8, "bold")).grid(row=2, column=0, sticky="nw", pady=(2, 0))
        self._d_rationale_lbl = ttk.Label(ev_sec, text="—", font=("Segoe UI", 8, "italic"), wraplength=360, justify="left")
        self._d_rationale_lbl.grid(row=2, column=1, sticky="w", pady=(2, 0), padx=(12, 0))

        # 4. Provenance & Lineage Section
        prov_sec = ttk.LabelFrame(p, text="🧬 Discovery Provenance & Lineage", padding=6)
        prov_sec.pack(fill="x", pady=(0, 4))

        ttk.Label(prov_sec, text="Source Pipeline:", font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w", pady=1)
        self._d_src_pipeline_lbl = ttk.Label(prov_sec, text="—", font=("Consolas", 8), wraplength=320, justify="left")
        self._d_src_pipeline_lbl.grid(row=0, column=1, sticky="w", pady=1, padx=(6, 0))

        ttk.Label(prov_sec, text="Research ID:", font=("Segoe UI", 8, "bold")).grid(row=1, column=0, sticky="w", pady=1)
        self._d_src_research_lbl = ttk.Label(prov_sec, text="—", font=("Consolas", 8), wraplength=320, justify="left")
        self._d_src_research_lbl.grid(row=1, column=1, sticky="w", pady=1, padx=(6, 0))

        ttk.Label(prov_sec, text="Formula Hash:", font=("Segoe UI", 8, "bold")).grid(row=2, column=0, sticky="w", pady=1)
        self._d_hash_lbl = ttk.Label(prov_sec, text="—", font=("Consolas", 8))
        self._d_hash_lbl.grid(row=2, column=1, sticky="w", pady=1, padx=(6, 0))

        ttk.Label(prov_sec, text="Parent Features:", font=("Segoe UI", 8, "bold")).grid(row=3, column=0, sticky="nw", pady=1)
        self._d_parents_lbl = ttk.Label(prov_sec, text="—", font=("Segoe UI", 8), wraplength=320, justify="left")
        self._d_parents_lbl.grid(row=3, column=1, sticky="w", pady=1, padx=(6, 0))

    def refresh(self) -> None:
        self._pipelines = load_pipelines(self.chart_dir)
        self._tree.delete(*self._tree.get_children())
        for row in self._pipelines:
            pid = row["pipeline_id"]
            self._tree.insert(
                "",
                tk.END,
                iid=pid,
                values=(
                    row.get("name"),
                    row.get("type_label"),
                    row.get("feature_count"),
                    row.get("status_label"),
                ),
            )
        if self._selected_id and self._tree.exists(self._selected_id):
            self._tree.selection_set(self._selected_id)
            self._show_detail(self._selected_id)
        elif self._pipelines:
            first = self._pipelines[0]["pipeline_id"]
            self._tree.selection_set(first)
            self._show_detail(first)

    def _on_select(self, _event: Any = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        self._show_detail(sel[0])

    def _show_detail(self, pipeline_id: str) -> None:
        self._selected_id = pipeline_id
        row = get_pipeline(self.chart_dir, pipeline_id)
        if not row:
            self._detail_var.set("Pipeline not found")
            self._feat_tree.delete(*self._feat_tree.get_children())
            self._current_candidates = []
            self._current_feat_prov_map = {}
            self._clear_detail_inspector("Pipeline not found")
            return

        from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
            is_excluded_pipeline_feature,
        )
        from .build_service import chart_data_dir

        data_dir = chart_data_dir(self.chart_dir)
        candidates = [
            str(n).strip()
            for n in (row.get("candidate_features") or [])
            if str(n).strip() and not is_excluded_pipeline_feature(str(n).strip(), data_dir)
        ]
        self._current_candidates = candidates

        meta = row.get("provenance_metadata") or {}
        feat_prov = meta.get("selected_features_provenance") or []
        feat_prov_map: dict[str, dict[str, Any]] = {}
        for fp in feat_prov:
            if isinstance(fp, dict) and fp.get("feature_id"):
                feat_prov_map[fp["feature_id"]] = fp
        self._current_feat_prov_map = feat_prov_map

        ptype = str(row.get("type") or "").lower()
        self._is_discovery_pipeline = ptype in ("discovery_experimental", "candidate_discovery")

        self._detail_var.set(
            f"ID: {row['pipeline_id']}  ·  Name: {row['name']}  ·  "
            f"Type: {row['type_label']}  ·  Status: {row['status_label']}\n"
            f"Registry Members: {row['registry_feature_count']}  ·  "
            f"Candidate Discovered Features: {len(candidates)}"
        )

        is_base = bool(row.get("is_base")) or ptype == "base"
        try:
            self._delete_pipeline_btn.configure(state="disabled" if is_base else "normal")
            self._select_features_btn.configure(state="disabled" if (is_base or self._is_discovery_pipeline) else "normal")
        except tk.TclError:
            pass

        # Populate Left Treeview
        self._feat_tree.delete(*self._feat_tree.get_children())
        self._left_frame.config(text=f"Discovered Features ({len(candidates)})" if self._is_discovery_pipeline else f"Candidate Features ({len(candidates)})")

        for idx, name in enumerate(candidates):
            prov = feat_prov_map.get(name, {})
            disp = prov.get("display_name") or name
            verdict = prov.get("discovery_verdict") or ("BASE" if is_base else "CANDIDATE")
            strategy = prov.get("generator_strategy") or ("BASE" if is_base else "—")
            gen = f"G{prov['generation_discovered']}" if "generation_discovered" in prov else "—"
            delta_auc_str = f"{prov['marginal_delta_auc']:+.5f}" if "marginal_delta_auc" in prov else "—"

            v_norm = str(verdict).upper()
            if v_norm == "KEEP":
                v_text = "🟢 KEEP"
                tag = "tag_keep"
            elif v_norm == "WATCH":
                v_text = "🟡 WATCH"
                tag = "tag_watch"
            elif v_norm == "REMOVE":
                v_text = "🔴 REMOVE"
                tag = "tag_remove"
            else:
                v_text = v_norm
                tag = ""

            self._feat_tree.insert(
                "",
                tk.END,
                iid=name,
                values=(disp, v_text, strategy, gen, delta_auc_str),
                tags=(tag,) if tag else (),
            )

        if candidates:
            first_item = candidates[0]
            self._feat_tree.selection_set(first_item)
            self._populate_detail_inspector(first_item)
        else:
            self._clear_detail_inspector("No Discovery Features in this Pipeline")

    def _on_feat_tree_select(self, _event: Any = None) -> None:
        sel = self._feat_tree.selection()
        if not sel:
            return
        self._populate_detail_inspector(sel[0])

    def _populate_detail_inspector(self, feature_id: str) -> None:
        """Render the structured right detail inspector card for the selected feature."""
        prov = self._current_feat_prov_map.get(feature_id)

        if prov:
            disp_name = prov.get("display_name") or feature_id
            tech_id = prov.get("feature_id") or feature_id
            verdict = prov.get("discovery_verdict") or "KEEP"
            strategy = prov.get("generator_strategy") or "—"
            gen = prov.get("generation_discovered", 1)
            formula = prov.get("formula_expression") or "—"
            d_auc = float(prov.get("marginal_delta_auc") or 0.0)
            d_ks = float(prov.get("ks_statistic") or 0.0)
            drift_sev = int(prov.get("drift_severity") or 0)
            ev_score = float(prov.get("evidence_score") or 0.0)
            fold_cons = float(prov.get("fold_consistency") or 0.0)
            gov_rat = prov.get("governance_rationale") or "—"
            src_dp = prov.get("source_discovery_pipeline_id") or prov.get("pipeline_id") or "—"
            src_rid = prov.get("source_research_id") or prov.get("research_id") or "—"
            f_hash = prov.get("formula_hash") or "—"
            parents = prov.get("parent_features") or []

            # 1. Identity
            self._d_name_val.config(text=disp_name)
            self._d_tech_id_var.set(tech_id)
            v_badge = "🟢 KEEP" if verdict == "KEEP" else ("🟡 WATCH" if verdict == "WATCH" else "🔴 REMOVE")
            self._d_verdict_strat_lbl.config(text=f"{v_badge}   ·   Strategy: {strategy}   ·   Generation: G{gen}")

            # 2. Formula Area
            self._formula_text.delete("1.0", tk.END)
            self._formula_text.insert("1.0", formula)

            # 3. Evidence & Governance
            sev_label = "0 (Low)" if drift_sev == 0 else ("1 (Moderate)" if drift_sev == 1 else "2 (Severe)")
            self._d_delta_auc_lbl.config(text=f"Marginal ΔAUC: {d_auc:+.5f}")
            self._d_drift_lbl.config(text=f"KS Drift (D_KS): {d_ks:.4f} (Severity {sev_label})")
            self._d_evidence_lbl.config(text=f"Evidence Score: {ev_score:.1f} pts")
            self._d_folds_lbl.config(text=f"Fold Consistency: {fold_cons*100:.0f}%")
            self._d_rationale_lbl.config(text=gov_rat)

            # 4. Provenance
            self._d_src_pipeline_lbl.config(text=src_dp)
            self._d_src_research_lbl.config(text=src_rid)
            self._d_hash_lbl.config(text=f_hash)
            self._d_parents_lbl.config(text=", ".join(parents) if parents else "—")

        else:
            # Fallback for standard Base or non-discovery candidate features
            self._d_name_val.config(text=feature_id)
            self._d_tech_id_var.set(feature_id)
            self._d_verdict_strat_lbl.config(text="Base / Registry Candidate Feature")
            self._formula_text.delete("1.0", tk.END)
            self._formula_text.insert("1.0", f"Base Feature column in analysis dataset: col('{feature_id}')")
            self._d_delta_auc_lbl.config(text="Marginal ΔAUC: Baseline")
            self._d_drift_lbl.config(text="KS Drift (D_KS): Baseline")
            self._d_evidence_lbl.config(text="Evidence Score: Baseline")
            self._d_folds_lbl.config(text="Fold Consistency: 100%")
            self._d_rationale_lbl.config(text="Authoritative pipeline candidate feature.")
            self._d_src_pipeline_lbl.config(text=self._selected_id or "—")
            self._d_src_research_lbl.config(text="Base Pipeline Anchor")
            self._d_hash_lbl.config(text="—")
            self._d_parents_lbl.config(text="Raw Time-Series / Order Flow")

    def _clear_detail_inspector(self, empty_msg: str) -> None:
        """Display clean empty state message."""
        self._d_name_val.config(text=empty_msg)
        self._d_tech_id_var.set("—")
        self._d_verdict_strat_lbl.config(text="—")
        self._formula_text.delete("1.0", tk.END)
        self._formula_text.insert("1.0", empty_msg)
        self._d_delta_auc_lbl.config(text="Marginal ΔAUC: —")
        self._d_drift_lbl.config(text="KS Drift (D_KS): —")
        self._d_evidence_lbl.config(text="Evidence Score: —")
        self._d_folds_lbl.config(text="Fold Consistency: —")
        self._d_rationale_lbl.config(text="—")
        self._d_src_pipeline_lbl.config(text="—")
        self._d_src_research_lbl.config(text="—")
        self._d_hash_lbl.config(text="—")
        self._d_parents_lbl.config(text="—")

    def _open_create(self) -> None:
        CreatePipelineDialog(self, chart_dir=self.chart_dir, on_create=self._create_pipeline)

    def _notify_pipelines_changed(self, pipeline_id: str | None = None) -> None:
        if callable(self._on_pipelines_changed):
            try:
                self._on_pipelines_changed(select_pipeline_id=pipeline_id)
            except TypeError:
                try:
                    self._on_pipelines_changed()
                except Exception:
                    pass
            except Exception:
                pass

    def _create_pipeline(self, name: str, pipeline_type: str) -> None:
        try:
            row = create_pipeline(self.chart_dir, name=name, pipeline_type=pipeline_type)
            self.refresh()
            pid = row.get("pipeline_id")
            if pid:
                self._tree.selection_set(pid)
                self._show_detail(pid)
                self._notify_pipelines_changed(str(pid))
            else:
                self._notify_pipelines_changed()
        except Exception as exc:
            messagebox.showerror("Create Pipeline", str(exc), parent=self)

    def _delete_pipeline(self) -> None:
        pid = self._selected_id
        if not pid:
            messagebox.showinfo("Delete Pipeline", "Select a pipeline first.", parent=self)
            return
        row = get_pipeline(self.chart_dir, pid)
        if not row:
            messagebox.showerror("Delete Pipeline", "Pipeline not found.", parent=self)
            return
        if str(row.get("type") or "") == "base" or row.get("is_base"):
            messagebox.showinfo(
                "Delete Pipeline",
                "The Base pipeline cannot be deleted.",
                parent=self,
            )
            return
        name = str(row.get("name") or pid)
        if not messagebox.askyesno(
            "Delete Pipeline",
            f"Delete pipeline \"{name}\" ({pid})?\n\n"
            "Registry membership and candidate features for this pipeline will be removed.\n"
            "The Master Feature Registry will not be changed.\n\n"
            "This cannot be undone.",
            parent=self,
            icon="warning",
        ):
            return
        try:
            delete_pipeline(self.chart_dir, pid)
        except Exception as exc:
            messagebox.showerror("Delete Pipeline", str(exc), parent=self)
            return
        self._selected_id = None
        self.refresh()
        self._notify_pipelines_changed()

    def _select_features(self) -> None:
        pid = self._selected_id
        if not pid:
            messagebox.showinfo("Select Features", "Select a pipeline first.", parent=self)
            return
        row = get_pipeline(self.chart_dir, pid)
        if not row:
            return
        if str(row.get("type") or "") == "base" or row.get("is_base"):
            messagebox.showinfo(
                "Select Features",
                "The Base pipeline uses the approved feature pool, not registry membership selection.",
                parent=self,
            )
            return

        def _apply(feature_ids: list[str]) -> None:
            try:
                set_pipeline_registry_members(self.chart_dir, pid, feature_ids)
                self.refresh()
                self._show_detail(pid)
            except Exception as exc:
                messagebox.showerror("Select Features", str(exc), parent=self)

        PipelineRegistryFeatureDialog(
            self,
            chart_dir=self.chart_dir,
            selected_ids=list(row.get("registry_feature_ids") or []),
            on_apply=_apply,
        )