"""Feature Analysis Lab — Phase 2 workspace (read-only over analysis datasets).

No feature generation. Loads parquet snapshots and runs research modules that
persist into analysis.db.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .build_service import chart_data_dir


class FeatureAnalysisPanel(ttk.Frame):
    """Analysis tab: pick a dataset, track module status, Correlation views."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_model_builder: Any | None = None,
        defer_dataset_scan: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.chart_dir = chart_dir
        self._on_open_model_builder = on_open_model_builder
        self._data_dir = chart_data_dir(chart_dir)
        self._datasets: list[dict[str, Any]] = []
        self._dataset_id: str | None = None
        self._run_id: str | None = None
        self._module_vars: dict[str, tk.BooleanVar] = {}
        self._module_status_vars: dict[str, tk.StringVar] = {}
        self._busy = False
        self._matrix_features: list[str] = []
        self._catalog_busy = False
        self._gpu_ok_cache: bool | None = None
        self._gpu_probe_inflight = False
        self._progress_last_ui_at: dict[str, float] = {}
        self._progress_last_ui_msg: dict[str, str] = {}

        self._dataset_choice = tk.StringVar(value="")
        self._rows_var = tk.StringVar(value="—")
        self._features_var = tk.StringVar(value="—")
        self._created_var = tk.StringVar(value="—")
        self._hash_var = tk.StringVar(value="—")
        self._status_var = tk.StringVar(value="No dataset")
        self._detail_var = tk.StringVar(
            value="No feature generation occurs here. Load an Analysis Dataset to research."
        )
        self._corr_summary_var = tk.StringVar(
            value="Run Correlation to populate Summary / Top Pairs / Matrix / Clusters."
        )
        self._corr_backend_pref = tk.StringVar(value="auto")
        self._corr_backend_status = tk.StringVar(
            value="Backend Used: — · GPU: checking…"
        )
        self._matrix_search = tk.StringVar(value="")
        self._matrix_cluster = tk.StringVar(value="(all)")
        self._pairs_min_abs = tk.StringVar(value="0.95")
        self._feature_search = tk.StringVar(value="")
        self._profile_text_var = tk.StringVar(value="Select a feature to view its profile.")
        self._all_profile_features: list[str] = []
        self._mi_target = tk.StringVar(value="")
        self._mi_targets: list[str] = []
        self._shap_model = tk.StringVar(value="")
        self._shap_models: list[str] = []
        self._perm_progress = tk.DoubleVar(value=0.0)
        self._perm_progress_text = tk.StringVar(value="")
        self._cancel_token = None
        self._rec_filter = tk.StringVar(value="ALL")
        self._shap_stage_var = tk.StringVar(value="Stage: Model Validation")
        self._shap_warn_var = tk.StringVar(value="")
        self._family_detail_var = tk.StringVar(
            value="Run HCA (Feature Families) after Correlation."
        )
        self._review_family_id = tk.StringVar(value="")
        self._review_rep = tk.StringVar(value="")
        self._review_reason_code = tk.StringVar(value="Interpretability")
        self._review_reason_text = tk.StringVar(value="")
        self._review_status = tk.StringVar(value="For Experiment")
        self._review_summary_var = tk.StringVar(value="")
        self._review_filter = tk.StringVar(value="Needs Review")
        self._families_cache: list[dict[str, Any]] = []
        self._exp_name = tk.StringVar(value="")
        self._exp_notes = tk.StringVar(value="")
        self._exp_focus_family = tk.StringVar(value="")
        self._exp_summary_var = tk.StringVar(value="")
        self._exp_bundle_var = tk.StringVar(value="")
        self._fs_strategy = tk.StringVar(value="hca_corr_perm")
        self._fs_policy = tk.StringVar(value="top_1")
        self._fs_top_n = tk.StringVar(value="4")
        self._fs_corr_thr = tk.StringVar(value="0.95")
        self._fs_perm_thr = tk.StringVar(value="0.001")
        self._fs_preview_var = tk.StringVar(
            value="Choose a Feature Selection Strategy, then Preview / Freeze."
        )
        self._fs_preview_cache: dict[str, Any] | None = None
        self._auto_dashboard_var = tk.StringVar(
            value=(
                "Auto Research\n"
                "Round              —\n"
                "Current Baseline   —\n"
                "Current Score      —\n"
                "Best Ever          —\n"
                "Families Improved  —\n"
                "Remaining Families —"
            )
        )
        self._auto_resume_id: str | None = None
        self._research_level = tk.StringVar(value="balanced")
        self._research_strategy = tk.StringVar(value="greedy")
        self._research_advanced = tk.BooleanVar(value=False)
        self._research_per_round = tk.StringVar(value="10")
        self._champion_card_var = tk.StringVar(
            value="No champion yet — run ▶ Auto Research first."
        )
        self._discovery_banner_var = tk.StringVar(
            value="Discovery  —  Load a dataset and run Stage 1 modules."
        )
        self._experiments_cache: list[dict[str, Any]] = []
        # Research tabs hydrate on demand — never on dataset meta load.
        self._hydrated_tabs: set[str] = set()

        self._build_ui()
        if not defer_dataset_scan:
            self.refresh_datasets()
        else:
            self._status_var.set("Loading catalog…")
            self._detail_var.set("Analysis Lab ready — loading dataset meta…")

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self, padding=(10, 8))
        hdr.pack(fill="x")
        ttk.Label(
            hdr,
            text="Feature Analysis",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        ttk.Button(
            hdr,
            text="Refresh datasets",
            command=lambda: self.refresh_datasets(force_rescan=True),
        ).pack(side="right")

        ttk.Label(
            self,
            text=(
                "Read-only research over exported analysis datasets. "
                "Build datasets in the Auto tab — this tab never regenerates features."
            ),
            foreground="#666",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 6))

        top = ttk.Frame(self, padding=(10, 0))
        top.pack(fill="x")

        left = ttk.LabelFrame(top, text="Analysis Dataset", padding=8)
        left.pack(side="left", fill="y", padx=(0, 8))

        ttk.Label(left, text="Dataset").grid(row=0, column=0, sticky="w")
        self._dataset_combo = ttk.Combobox(
            left,
            textvariable=self._dataset_choice,
            state="readonly",
            width=42,
        )
        self._dataset_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self._dataset_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_dataset_selected()
        )

        meta = ttk.Frame(left)
        meta.grid(row=2, column=0, sticky="ew")
        for i, (label, var) in enumerate(
            (
                ("Rows", self._rows_var),
                ("Features", self._features_var),
                ("Created", self._created_var),
                ("Hash", self._hash_var),
                ("Status", self._status_var),
            )
        ):
            ttk.Label(meta, text=label, foreground="#555").grid(
                row=i, column=0, sticky="w", pady=1
            )
            ttk.Label(meta, textvariable=var).grid(
                row=i, column=1, sticky="w", padx=(12, 0), pady=1
            )

        right = ttk.LabelFrame(top, text="Status", padding=4)
        right.pack(side="left", fill="both", expand=True)

        self._status_nb = ttk.Notebook(right)
        self._status_nb.pack(fill="both", expand=True)

        platform_tab = ttk.Frame(self._status_nb, padding=6)
        analysis_tab = ttk.Frame(self._status_nb, padding=6)
        fs_tab = ttk.Frame(self._status_nb, padding=6)
        self._status_nb.add(platform_tab, text="Platform")
        self._status_nb.add(analysis_tab, text="Analysis Status")
        self._status_nb.add(fs_tab, text="Feature Selection Strategy")

        # Tab 1 — Platform (where am I)
        self._discovery_banner = tk.Label(
            platform_tab,
            textvariable=self._discovery_banner_var,
            justify="left",
            anchor="nw",
            font=("Consolas", 9),
            foreground="#1a4d2e",
            background="#eef6f0",
        )
        self._discovery_banner.pack(fill="both", expand=True)
        ban_btns = ttk.Frame(platform_tab)
        ban_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(
            ban_btns,
            text="Freeze Discovery Bundle",
            command=self._freeze_discovery_bundle,
        ).pack(side="left")
        ttk.Button(
            ban_btns,
            text="Create Experiment",
            command=self._create_experiment_from_review,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            ban_btns,
            text="Open Family Review",
            command=lambda: self._select_research_tab("Family Review"),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            ban_btns,
            text="Open Experiments",
            command=lambda: self._select_research_tab("Experiments"),
        ).pack(side="left", padx=(6, 0))

        # Tab 3 — Feature Selection Strategy (how Final Feature Dataset is built)
        ttk.Label(
            fs_tab,
            text=(
                "Choose how the Final Feature Dataset is built, then hand it "
                "directly to Model Builder."
            ),
            foreground="#555",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        for sid, label in (
            ("hca_corr_perm", "HCA + Correlation + Permutation"),
            ("corr_perm", "Correlation + Permutation Only"),
            ("perm_only", "Permutation Only (Research)"),
            ("corr_only", "Correlation Only (Research)"),
        ):
            ttk.Radiobutton(
                fs_tab,
                text=label,
                value=sid,
                variable=self._fs_strategy,
                command=self._on_fs_strategy_changed,
            ).pack(anchor="w")

        self._fs_hca_fr = ttk.LabelFrame(
            fs_tab, text="Representative Policy", padding=6
        )
        self._fs_hca_fr.pack(fill="x", pady=(8, 0))
        pol_row = ttk.Frame(self._fs_hca_fr)
        pol_row.pack(anchor="w")
        for val, lab in (
            ("top_1", "Top 1"),
            ("top_2", "Top 2"),
            ("top_3", "Top 3"),
            ("top_n", "Top N"),
        ):
            ttk.Radiobutton(
                pol_row,
                text=lab,
                value=val,
                variable=self._fs_policy,
                command=self._on_fs_strategy_changed,
            ).pack(side="left", padx=(0, 8))
        ttk.Label(pol_row, text="N=").pack(side="left")
        ttk.Entry(pol_row, textvariable=self._fs_top_n, width=4).pack(side="left")

        self._fs_flat_fr = ttk.LabelFrame(fs_tab, text="Thresholds", padding=6)
        thr_row = ttk.Frame(self._fs_flat_fr)
        thr_row.pack(anchor="w")
        ttk.Label(thr_row, text="Correlation Threshold").pack(side="left")
        ttk.Entry(thr_row, textvariable=self._fs_corr_thr, width=6).pack(
            side="left", padx=(6, 12)
        )
        ttk.Label(thr_row, text="Permutation Threshold").pack(side="left")
        ttk.Entry(thr_row, textvariable=self._fs_perm_thr, width=8).pack(
            side="left", padx=(6, 0)
        )

        ttk.Label(
            fs_tab,
            textvariable=self._fs_preview_var,
            justify="left",
            foreground="#333",
            font=("Consolas", 9),
        ).pack(anchor="w", pady=(10, 0))
        fs_btns = ttk.Frame(fs_tab)
        fs_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(
            fs_btns,
            text="Preview Final Dataset",
            command=self._preview_feature_selection,
        ).pack(side="left")
        ttk.Button(
            fs_btns,
            text="View Features",
            command=self._view_final_feature_dataset,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            fs_btns,
            text="► Create Model Builder",
            command=self._create_model_builder_from_selection,
        ).pack(side="left", padx=(6, 0))
        self._on_fs_strategy_changed()

        # Tab 2 — Analysis Status (module checklist + run)
        self._mod_frame = ttk.Frame(analysis_tab)
        self._mod_frame.pack(fill="both", expand=True)

        from chain_replay_ml.dataset_builder.analysis_lab_store import (
            ANALYSIS_MODULES,
            MODULE_LABELS,
        )

        for mid in ANALYSIS_MODULES:
            row = ttk.Frame(self._mod_frame)
            row.pack(fill="x", pady=2)
            var = tk.BooleanVar(value=mid == "correlation")
            self._module_vars[mid] = var
            ttk.Checkbutton(
                row,
                variable=var,
                command=self._sync_module_run_controls,
            ).pack(side="left")
            ttk.Label(
                row,
                text=MODULE_LABELS.get(mid, mid),
                width=22,
                anchor="w",
            ).pack(side="left")
            st = tk.StringVar(value="Not Run")
            self._module_status_vars[mid] = st
            ttk.Label(row, textvariable=st, foreground="#444").pack(
                side="left", padx=(8, 0)
            )

        actions = ttk.Frame(analysis_tab)
        actions.pack(fill="x", pady=(10, 0))
        # Inputs appear only when a checked module needs them.
        self._module_inputs = ttk.Frame(actions)
        self._module_inputs.pack(side="left")
        self._mi_target_label = ttk.Label(self._module_inputs, text="Target")
        self._mi_target_combo = ttk.Combobox(
            self._module_inputs,
            textvariable=self._mi_target,
            state="readonly",
            width=18,
            values=[],
        )
        self._mi_target_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._reload_mi_tab()
        )
        self._shap_model_label = ttk.Label(self._module_inputs, text="Model")
        self._shap_model_combo = ttk.Combobox(
            self._module_inputs,
            textvariable=self._shap_model,
            state="readonly",
            width=28,
            values=[],
        )
        self._shap_model_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_shap_model_changed()
        )
        self._run_selected_btn = ttk.Button(
            actions, text="Run Selected", command=self._run_selected
        )
        self._run_selected_btn.pack(side="left", padx=(0, 0))
        self._run_all_btn = ttk.Button(
            actions, text="Run All", command=self._run_all
        )
        self._run_all_btn.pack(side="left", padx=(8, 0))
        self._cancel_btn = ttk.Button(
            actions, text="Cancel", command=self._cancel_running, state="disabled"
        )
        self._cancel_btn.pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="Open analysis.db folder",
            command=self._open_data_dir,
        ).pack(side="right")
        self._sync_module_run_controls()

        prog = ttk.Frame(analysis_tab)
        prog.pack(fill="x", pady=(6, 0))
        self._perm_bar = ttk.Progressbar(
            prog, variable=self._perm_progress, maximum=100, mode="determinate"
        )
        self._perm_bar.pack(fill="x", side="top")
        ttk.Label(
            prog, textvariable=self._perm_progress_text, foreground="#555"
        ).pack(anchor="w")

        # Research workspace (tabs only — Platform lives under Status)
        research = ttk.LabelFrame(self, text="Research", padding=4)
        research.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        self._research_nb = ttk.Notebook(research)
        self._research_nb.pack(fill="both", expand=True)

        corr_box = ttk.Frame(self._research_nb, padding=4)
        explorer_box = ttk.Frame(self._research_nb, padding=4)
        discovery_box = ttk.Frame(self._research_nb, padding=4)
        validation_box = ttk.Frame(self._research_nb, padding=4)
        mi_box = ttk.Frame(self._research_nb, padding=4)
        perm_box = ttk.Frame(self._research_nb, padding=4)
        rec_box = ttk.Frame(self._research_nb, padding=4)
        shap_box = ttk.Frame(self._research_nb, padding=4)
        families_box = ttk.Frame(self._research_nb, padding=4)
        review_box = ttk.Frame(self._research_nb, padding=4)
        experiments_box = ttk.Frame(self._research_nb, padding=4)
        compare_box = ttk.Frame(self._research_nb, padding=4)
        champion_box = ttk.Frame(self._research_nb, padding=4)
        self._research_nb.add(corr_box, text="Correlation")
        self._research_nb.add(families_box, text="Feature Families")
        self._research_nb.add(review_box, text="Family Review")
        self._research_nb.add(mi_box, text="Mutual Information")
        self._research_nb.add(perm_box, text="Permutation")
        self._research_nb.add(explorer_box, text="Feature Explorer")
        self._research_nb.add(discovery_box, text="Discovery Scorecard")
        self._research_nb.add(rec_box, text="Recommendations")
        self._research_nb.add(experiments_box, text="Experiments")
        self._research_nb.add(compare_box, text="Experiment Comparison")
        self._research_nb.add(champion_box, text="Champion")
        self._research_nb.add(shap_box, text="Model Explanation (SHAP)")
        self._research_nb.add(validation_box, text="Model Validation")

        self._corr_nb = ttk.Notebook(corr_box)
        self._corr_nb.pack(fill="both", expand=True)

        self._tab_summary = ttk.Frame(self._corr_nb, padding=8)
        self._tab_pairs = ttk.Frame(self._corr_nb, padding=6)
        self._tab_matrix = ttk.Frame(self._corr_nb, padding=6)
        self._tab_clusters = ttk.Frame(self._corr_nb, padding=6)
        self._tab_insights = ttk.Frame(self._corr_nb, padding=6)
        self._corr_nb.add(self._tab_summary, text="Summary")
        self._corr_nb.add(self._tab_pairs, text="Top Pairs")
        self._corr_nb.add(self._tab_matrix, text="Matrix")
        self._corr_nb.add(self._tab_clusters, text="Clusters")
        self._corr_nb.add(self._tab_insights, text="Insights")

        self._build_summary_tab()
        self._build_pairs_tab()
        self._build_matrix_tab()
        self._build_clusters_tab()
        self._build_insights_tab()
        self._load_correlation_backend_prefs()
        self._refresh_corr_backend_status()
        self._build_feature_families_tab(families_box)
        self._build_family_review_tab(review_box)
        self._build_mi_tab(mi_box)
        self._build_permutation_tab(perm_box)
        self._build_feature_explorer(explorer_box)
        self._build_discovery_scorecard_tab(discovery_box)
        self._build_recommendations_tab(rec_box)
        self._build_experiments_tab(experiments_box)
        self._build_experiment_compare_tab(compare_box)
        self._build_champion_tab(champion_box)
        self._build_shap_explanation_tab(shap_box)
        self._build_validation_scorecard_tab(validation_box)
        self._research_nb.bind(
            "<<NotebookTabChanged>>", self._on_research_tab_changed
        )

        ttk.Label(
            self,
            textvariable=self._detail_var,
            foreground="#555",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(4, 8))

    def _build_summary_tab(self) -> None:
        ttk.Label(
            self._tab_summary,
            text="Correlation Analysis",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        backend_row = ttk.Frame(self._tab_summary)
        backend_row.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(backend_row, text="Backend:").pack(side="left")
        self._corr_backend_combo = ttk.Combobox(
            backend_row,
            textvariable=self._corr_backend_pref,
            values=("auto", "cpu", "gpu"),
            state="readonly",
            width=8,
        )
        self._corr_backend_combo.pack(side="left", padx=(6, 0))
        self._corr_backend_combo.bind(
            "<<ComboboxSelected>>", self._on_corr_backend_changed
        )
        ttk.Label(
            backend_row,
            textvariable=self._corr_backend_status,
            foreground="#555",
        ).pack(side="left", padx=(12, 0))

        ttk.Label(
            self._tab_summary,
            textvariable=self._corr_summary_var,
            justify="left",
            font=("Consolas", 9),
        ).pack(anchor="w", pady=(8, 8))
        btns = ttk.Frame(self._tab_summary)
        btns.pack(anchor="w")
        ttk.Button(
            btns, text="Open Matrix", command=lambda: self._corr_nb.select(self._tab_matrix)
        ).pack(side="left")
        ttk.Button(
            btns,
            text="Open Clusters",
            command=lambda: self._corr_nb.select(self._tab_clusters),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            btns, text="Top Pairs", command=lambda: self._corr_nb.select(self._tab_pairs)
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            btns,
            text="Open Insights",
            command=lambda: self._corr_nb.select(self._tab_insights),
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            btns, text="Refresh views", command=self._refresh_correlation_views
        ).pack(side="left", padx=(12, 0))

    def _analysis_lab_prefs(self) -> dict[str, Any]:
        if not self.chart_dir:
            return {}
        try:
            from .build_config_prefs import load_build_config_prefs

            doc = load_build_config_prefs(self.chart_dir) or {}
            section = doc.get("analysis_lab")
            return section if isinstance(section, dict) else {}
        except Exception:
            return {}

    def _load_correlation_backend_prefs(self) -> None:
        saved = str(self._analysis_lab_prefs().get("correlation_backend") or "").strip().lower()
        if saved in {"auto", "cpu", "gpu"}:
            self._corr_backend_pref.set(saved)

    def _on_corr_backend_changed(self, *_args: Any) -> None:
        pref = str(self._corr_backend_pref.get() or "auto").strip().lower()
        if pref not in {"auto", "cpu", "gpu"}:
            pref = "auto"
            self._corr_backend_pref.set(pref)
        try:
            from .build_config_prefs import save_build_config_prefs

            current = self._analysis_lab_prefs()
            save_build_config_prefs(
                self.chart_dir,
                {"analysis_lab": {**current, "correlation_backend": pref}},
            )
        except Exception:
            pass
        self._refresh_corr_backend_status()

    def _refresh_corr_backend_status(self, compute_meta: dict[str, Any] | None = None) -> None:
        """Update backend status without probing GPU on the Tk thread.

        ``is_gpu_available()`` may import CUDA stacks; keep that off the UI
        thread (see analytics/correlation/gpu_engine.py).
        """
        pref = str(self._corr_backend_pref.get() or "auto").strip().lower() or "auto"

        def _apply(gpu_ok: bool, resolved: str) -> None:
            self._gpu_ok_cache = bool(gpu_ok)
            if compute_meta:
                used = str(compute_meta.get("backend_used") or resolved).upper()
                timing = compute_meta.get("timing") or {}
                total = timing.get("total_sec")
                cpu_t = timing.get("cpu_compute_sec")
                gpu_t = timing.get("gpu_compute_sec")
                xfer = timing.get("gpu_transfer_sec")
                parts = [f"Backend Used: {used}"]
                if total is not None:
                    parts.append(f"total {float(total):.3f}s")
                if used == "GPU":
                    if xfer is not None:
                        parts.append(f"xfer {float(xfer):.3f}s")
                    if gpu_t is not None:
                        parts.append(f"compute {float(gpu_t):.3f}s")
                elif cpu_t is not None:
                    parts.append(f"compute {float(cpu_t):.3f}s")
                fb = timing.get("fallback_reason")
                if fb:
                    parts.append("fell back to CPU")
                parts.append(f"GPU {'available' if gpu_ok else 'N/A (Windows/RAPIDS)'}")
                self._corr_backend_status.set(" · ".join(parts))
                return
            gpu_label = "available" if gpu_ok else "N/A (Windows/RAPIDS)"
            self._corr_backend_status.set(
                f"Will use: {str(resolved).upper()} · preference={pref} · GPU {gpu_label}"
            )

        def _resolve_local(gpu_ok: bool) -> str:
            if pref == "cpu":
                return "cpu"
            if pref == "gpu":
                return "gpu" if gpu_ok else "cpu"
            return "gpu" if gpu_ok else "cpu"

        if self._gpu_ok_cache is not None and (
            compute_meta is not None or not self._gpu_probe_inflight
        ):
            _apply(bool(self._gpu_ok_cache), _resolve_local(bool(self._gpu_ok_cache)))
            return

        # Immediate non-blocking placeholder, then probe on a worker.
        self._corr_backend_status.set(
            f"Will use: {pref.upper()} · preference={pref} · GPU checking…"
        )
        if self._gpu_probe_inflight:
            return
        self._gpu_probe_inflight = True

        def _worker() -> None:
            gpu_ok = False
            resolved = "cpu"
            try:
                from chain_replay_ml.analytics.correlation import (
                    is_gpu_available,
                    resolve_backend,
                )

                gpu_ok = bool(is_gpu_available())
                resolved = str(resolve_backend(pref) or "cpu")
            except Exception:
                gpu_ok = False
                resolved = "cpu"

            def _done() -> None:
                self._gpu_probe_inflight = False
                _apply(gpu_ok, resolved)

            self.after(0, _done)

        threading.Thread(
            target=_worker, name="corr-gpu-probe", daemon=True
        ).start()

    def _build_pairs_tab(self) -> None:
        bar = ttk.Frame(self._tab_pairs)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Min |r|").pack(side="left")
        ttk.Entry(bar, textvariable=self._pairs_min_abs, width=6).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(bar, text="Apply", command=self._reload_top_pairs).pack(
            side="left", padx=(6, 0)
        )
        ttk.Label(
            bar,
            text="Double-click a row for feature name (profile later)",
            foreground="#777",
        ).pack(side="right")

        cols = ("feature_a", "feature_b", "correlation")
        self._pairs_tree = ttk.Treeview(
            self._tab_pairs, columns=cols, show="headings", height=14
        )
        self._pairs_tree.heading("feature_a", text="Feature A")
        self._pairs_tree.heading("feature_b", text="Feature B")
        self._pairs_tree.heading("correlation", text="Correlation")
        self._pairs_tree.column("feature_a", width=260, anchor="w")
        self._pairs_tree.column("feature_b", width=260, anchor="w")
        self._pairs_tree.column("correlation", width=100, anchor="e")
        sb = ttk.Scrollbar(
            self._tab_pairs, orient="vertical", command=self._pairs_tree.yview
        )
        self._pairs_tree.configure(yscrollcommand=sb.set)
        self._pairs_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._pairs_tree.bind("<Double-1>", self._on_pair_double_click)

    def _build_matrix_tab(self) -> None:
        bar = ttk.Frame(self._tab_matrix)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Search").pack(side="left")
        ttk.Entry(bar, textvariable=self._matrix_search, width=28).pack(
            side="left", padx=(6, 0)
        )
        ttk.Label(bar, text="Cluster").pack(side="left", padx=(10, 0))
        self._matrix_cluster_combo = ttk.Combobox(
            bar,
            textvariable=self._matrix_cluster,
            state="readonly",
            width=28,
            values=["(all)"],
        )
        self._matrix_cluster_combo.pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Show", command=self._reload_matrix).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(
            bar,
            text="Shows a focused slice (not full 425×425)",
            foreground="#777",
        ).pack(side="right")

        self._matrix_text = tk.Text(
            self._tab_matrix,
            height=16,
            wrap="none",
            font=("Consolas", 8),
            background="#fafafa",
        )
        xsb = ttk.Scrollbar(
            self._tab_matrix, orient="horizontal", command=self._matrix_text.xview
        )
        ysb = ttk.Scrollbar(
            self._tab_matrix, orient="vertical", command=self._matrix_text.yview
        )
        self._matrix_text.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        self._matrix_text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._matrix_text.tag_configure("hi", foreground="#8b0000")
        self._matrix_text.tag_configure("mid", foreground="#a05a00")
        self._matrix_text.tag_configure("lo", foreground="#333333")
        self._matrix_text.tag_configure("diag", foreground="#006400")

    def _build_clusters_tab(self) -> None:
        self._clusters_text = tk.Text(
            self._tab_clusters,
            height=16,
            wrap="word",
            font=("Consolas", 9),
            background="#fafafa",
        )
        sb = ttk.Scrollbar(
            self._tab_clusters, orient="vertical", command=self._clusters_text.yview
        )
        self._clusters_text.configure(yscrollcommand=sb.set)
        self._clusters_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._clusters_text.tag_configure("head", font=("Segoe UI", 9, "bold"))
        self._clusters_text.tag_configure("meta", foreground="#555")

    def _build_insights_tab(self) -> None:
        note = ttk.Label(
            self._tab_insights,
            text=(
                "Read-only recommendations from correlation clusters. "
                "Never deletes features — investigate, then wait for "
                "MI / Permutation / Discovery Rating."
            ),
            foreground="#555",
            wraplength=720,
            justify="left",
        )
        note.pack(anchor="w", pady=(0, 6))

        table_host = ttk.Frame(self._tab_insights)
        table_host.pack(fill="both", expand=True)

        cols = (
            "cluster",
            "family",
            "members",
            "max_corr",
            "avg_corr",
            "recommendation",
            "reason",
        )
        self._insights_tree = ttk.Treeview(
            table_host,
            columns=cols,
            show="headings",
            height=10,
        )
        headings = {
            "cluster": ("Cluster", 160),
            "family": ("Family", 90),
            "members": ("Members", 70),
            "max_corr": ("Max Corr", 80),
            "avg_corr": ("Avg Corr", 80),
            "recommendation": ("Recommendation", 140),
            "reason": ("Reason", 420),
        }
        for key, (title, width) in headings.items():
            self._insights_tree.heading(key, text=title)
            anchor = "e" if key in {"members", "max_corr", "avg_corr"} else "w"
            self._insights_tree.column(
                key, width=width, anchor=anchor, stretch=(key == "reason")
            )

        ysb = ttk.Scrollbar(table_host, orient="vertical", command=self._insights_tree.yview)
        self._insights_tree.configure(yscrollcommand=ysb.set)
        self._insights_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._insights_tree.tag_configure("dup", foreground="#8b0000")
        self._insights_tree.tag_configure("review", foreground="#a05a00")
        self._insights_tree.tag_configure("keep", foreground="#333333")
        self._insights_tree.bind("<<TreeviewSelect>>", self._on_insight_select)

        self._insight_detail = tk.Text(
            self._tab_insights,
            height=7,
            wrap="word",
            font=("Consolas", 9),
            background="#f7f7f7",
        )
        self._insight_detail.pack(fill="x", pady=(6, 0))
        self._insight_detail.configure(state="disabled")

    # --- dataset selection -------------------------------------------------

    def refresh_datasets(self, force_rescan: bool = False) -> None:
        """Load analysis dataset catalog off the UI thread."""
        if self._catalog_busy:
            return
        self._catalog_busy = True
        self._status_var.set("Loading catalog…")
        self._detail_var.set("Loading analysis dataset meta…")
        data_dir = self._data_dir
        force = bool(force_rescan)

        def _worker() -> None:
            err = ""
            datasets: list[dict[str, Any]] = []
            saved = ""
            try:
                from chain_replay_ml.dataset_builder.analysis_lab_store import (
                    get_selected_analysis_dataset,
                    load_analysis_dataset_catalog,
                )

                datasets = load_analysis_dataset_catalog(
                    data_dir, force_rescan=force
                )
                try:
                    saved = get_selected_analysis_dataset(data_dir).strip()
                except Exception:
                    saved = ""
            except Exception as exc:
                err = str(exc)
                datasets = []

            def _apply() -> None:
                self._catalog_busy = False
                self._apply_dataset_catalog(datasets, saved=saved, error=err)

            self.after(0, _apply)

        threading.Thread(
            target=_worker, name="analysis-catalog-load", daemon=True
        ).start()

    def _apply_dataset_catalog(
        self,
        datasets: list[dict[str, Any]],
        *,
        saved: str = "",
        error: str = "",
    ) -> None:
        if error:
            self._datasets = []
            self._detail_var.set(f"Could not load dataset catalog: {error}")
            self._status_var.set("Catalog error")
            return
        self._datasets = list(datasets or [])
        names = [str(d.get("name") or d.get("dataset_id") or "") for d in self._datasets]
        self._dataset_combo["values"] = names
        if names:
            current = self._dataset_choice.get().strip()
            pick = ""
            if current and current in names:
                pick = current
            elif saved:
                for d in self._datasets:
                    did = str(d.get("dataset_id") or "")
                    dname = str(d.get("name") or "")
                    if saved in (did, dname):
                        pick = dname or did
                        break
            if not pick:
                pick = names[0]
            self._dataset_choice.set(pick)
            self._on_dataset_selected()
        else:
            self._dataset_choice.set("")
            self._clear_meta()
            self._detail_var.set(
                "No analysis datasets found. Build one in the Auto tab first "
                f"(looking under {self._data_dir}/datasets)."
            )

    def _clear_meta(self) -> None:
        self._rows_var.set("—")
        self._features_var.set("—")
        self._created_var.set("—")
        self._hash_var.set("—")
        self._status_var.set("No dataset")
        self._dataset_id = None
        self._run_id = None
        self._hydrated_tabs.clear()
        for st in self._module_status_vars.values():
            st.set("Not Run")
        self._corr_summary_var.set(
            "Open Correlation after selecting a dataset — "
            "or run Correlation if status is Not Run."
        )

    def _selected_dataset(self) -> dict[str, Any] | None:
        name = self._dataset_choice.get().strip()
        for d in self._datasets:
            if str(d.get("name") or "") == name or str(d.get("dataset_id") or "") == name:
                return d
        return None

    def _on_dataset_selected(self) -> None:
        """Load catalog meta + module statuses only — no research compute."""
        from chain_replay_ml.dataset_builder.analysis_lab_store import (
            ensure_analysis_run,
            format_module_status_label,
            module_statuses,
            set_selected_analysis_dataset,
        )

        ds = self._selected_dataset()
        if not ds:
            self._clear_meta()
            return
        self._hydrated_tabs.clear()
        self._dataset_id = str(ds.get("dataset_id") or "")
        try:
            set_selected_analysis_dataset(
                self._data_dir,
                self._dataset_id or str(ds.get("name") or ""),
            )
        except Exception:
            pass
        self._rows_var.set(f"{int(ds.get('rows') or 0):,}")
        self._features_var.set(f"{int(ds.get('features') or 0):,}")
        self._created_var.set(str(ds.get("created_at") or "—"))
        self._hash_var.set(str(ds.get("dataset_hash") or "—"))
        self._status_var.set("Ready")
        try:
            run = ensure_analysis_run(self._data_dir, self._dataset_id)
            self._run_id = str(run.get("run_id") or "")
            for row in module_statuses(self._data_dir, self._run_id):
                mid = str(row.get("module_id") or "")
                if mid in self._module_status_vars:
                    self._module_status_vars[mid].set(
                        format_module_status_label(str(row.get("status") or ""))
                    )
            self._detail_var.set(
                f"Dataset {ds.get('name')} · hash {ds.get('dataset_hash')} · "
                f"run {self._run_id[:8]}… · meta ready (open a research tab to view results)"
            )
            self._reload_discovery_banner()
            self._sync_module_run_controls()
            # Hydrate only the visible research tab (if any results already exist).
            self.after_idle(self._hydrate_current_research_tab)
        except Exception as exc:
            self._detail_var.set(f"Could not open analysis run: {exc}")

    def _on_research_tab_changed(self, _event: Any = None) -> None:
        self._hydrate_current_research_tab()

    def _hydrate_current_research_tab(self) -> None:
        if not hasattr(self, "_research_nb"):
            return
        try:
            tab_id = self._research_nb.select()
            title = str(self._research_nb.tab(tab_id, "text") or "")
        except Exception:
            return
        if title:
            self._hydrate_research_tab(title)

    def _hydrate_research_tab(self, title: str) -> None:
        """Load stored analysis.db views for one tab. Never builds profiles."""
        if not self._run_id or not title:
            return
        if title in self._hydrated_tabs:
            return
        self._hydrated_tabs.add(title)
        try:
            if title == "Correlation":
                self._refresh_correlation_views()
            elif title == "Feature Families":
                self._reload_families_tab()
            elif title == "Family Review":
                self._reload_family_review_tab()
            elif title == "Mutual Information":
                self._reload_mi_targets()
                self._reload_mi_tab()
            elif title == "Permutation":
                self._reload_perm_tab()
            elif title == "Feature Explorer":
                self._ensure_profiles_loaded(async_build=False)
            elif title == "Discovery Scorecard":
                self._ensure_profiles_loaded(async_build=False)
                self._reload_scorecard()
            elif title == "Recommendations":
                self._ensure_profiles_loaded(async_build=False)
                self._reload_recommendations()
            elif title == "Experiments":
                self._reload_experiments_tab()
            elif title == "Experiment Comparison":
                self._reload_experiment_compare_tab()
            elif title == "Champion":
                self._reload_champion_tab()
            elif title == "Model Explanation (SHAP)":
                self._reload_shap_models()
                self._reload_shap_tab()
            elif title == "Model Validation":
                self._ensure_profiles_loaded(async_build=False)
                self._reload_validation_scorecard()
        except Exception as exc:
            self._detail_var.set(f"Could not load {title}: {exc}")

    # --- correlation views -------------------------------------------------

    def _refresh_correlation_views(self) -> None:
        self._reload_summary()
        self._reload_top_pairs()
        self._reload_clusters()
        self._reload_matrix_cluster_choices()
        self._reload_matrix()
        self._reload_insights()
        self._hydrated_tabs.add("Correlation")

    def _reload_summary(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import (
            load_correlation_summary,
        )

        if not self._run_id:
            return
        summary = load_correlation_summary(self._data_dir, self._run_id)
        if not summary:
            self._corr_summary_var.set(
                "Correlation Analysis\n"
                "---------------------------------------\n"
                "Status               Not Run\n\n"
                "Select Correlation and click Run Selected."
            )
            return
        top = str(summary.get("top_cluster") or "—")
        top_n = int(summary.get("top_cluster_size") or 0)
        top_line = f"{top} ({top_n} features)" if top_n else top
        compute = summary.get("compute_backend")
        if not isinstance(compute, dict):
            # summary columns are flattened; meta may live only in summary_json
            raw_json = summary.get("summary_json")
            if isinstance(raw_json, str) and raw_json.strip():
                try:
                    import json

                    doc = json.loads(raw_json)
                    if isinstance(doc, dict) and isinstance(doc.get("compute_backend"), dict):
                        compute = doc["compute_backend"]
                except Exception:
                    compute = None
        backend_lines = ""
        if isinstance(compute, dict) and compute:
            used = str(compute.get("backend_used") or "cpu").upper()
            timing = compute.get("timing") or {}
            total = timing.get("total_sec")
            t_part = f" · {float(total):.3f}s" if total is not None else ""
            backend_lines = f"Backend Used         {used}{t_part}\n"
            self._refresh_corr_backend_status(compute)
        self._corr_summary_var.set(
            "Correlation Analysis\n"
            "---------------------------------------\n"
            f"{backend_lines}"
            f"Features Analysed      {int(summary.get('features_analysed') or 0):,}\n"
            f"Pairs                  {int(summary.get('pairs') or 0):,}\n"
            f"|r| >= 0.95            {int(summary.get('pairs_ge_095') or 0):,} pairs\n"
            f"|r| >= 0.99            {int(summary.get('pairs_ge_099') or 0):,} pairs\n"
            f"Multi-member clusters  {int(summary.get('cluster_count') or 0):,}\n"
            f"Top Cluster\n"
            f"{top_line}"
        )
        # Append insight headline when available
        try:
            from chain_replay_ml.dataset_builder.analysis_correlation_insights import (
                load_correlation_insights,
            )

            insights = load_correlation_insights(self._data_dir, self._run_id)
            n_dup = sum(
                1
                for i in insights
                if str(i.get("recommendation")) == "Duplicate Candidate"
            )
            n_rev = sum(
                1 for i in insights if str(i.get("recommendation")) == "Review"
            )
            extra = (
                f"\n\nInsights\n"
                f"Duplicate candidates   {n_dup}\n"
                f"Review clusters        {n_rev}\n"
                f"(see Insights tab — investigation only, never auto-delete)"
            )
            self._corr_summary_var.set(self._corr_summary_var.get() + extra)
        except Exception:
            pass

    def _reload_top_pairs(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import load_top_pairs

        self._pairs_tree.delete(*self._pairs_tree.get_children())
        if not self._run_id:
            return
        try:
            min_abs = float(self._pairs_min_abs.get() or 0.0)
        except ValueError:
            min_abs = 0.0
        rows = load_top_pairs(
            self._data_dir, self._run_id, limit=500, min_abs=min_abs
        )
        for r in rows:
            self._pairs_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature_a"),
                    r.get("feature_b"),
                    f"{float(r.get('correlation') or 0.0):.4f}",
                ),
            )

    def _on_pair_double_click(self, _event: Any = None) -> None:
        sel = self._pairs_tree.selection()
        if not sel:
            return
        vals = self._pairs_tree.item(sel[0], "values")
        if not vals:
            return
        messagebox.showinfo(
            "Feature Profile",
            f"Feature Profile (coming next):\n\n{vals[0]}\n{vals[1]}\n\n"
            f"correlation = {vals[2]}",
            parent=self,
        )

    def _reload_matrix_cluster_choices(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import load_clusters

        values = ["(all)"]
        if self._run_id:
            for c in load_clusters(self._data_dir, self._run_id):
                if int(c.get("size") or 0) > 1:
                    values.append(str(c.get("cluster")))
        self._matrix_cluster_combo["values"] = values
        if self._matrix_cluster.get() not in values:
            self._matrix_cluster.set("(all)")

    def _reload_matrix(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import (
            list_correlated_features,
            load_clusters,
            load_matrix_slice,
        )

        self._matrix_text.configure(state="normal")
        self._matrix_text.delete("1.0", "end")
        if not self._run_id:
            self._matrix_text.insert("end", "Run Correlation first.\n")
            self._matrix_text.configure(state="disabled")
            return

        needle = self._matrix_search.get().strip().lower()
        cluster_name = self._matrix_cluster.get().strip()
        feats = list_correlated_features(self._data_dir, self._run_id)
        if cluster_name and cluster_name != "(all)":
            members: set[str] = set()
            for c in load_clusters(self._data_dir, self._run_id):
                if str(c.get("cluster")) == cluster_name:
                    members = set(c.get("members") or [])
                    break
            feats = [f for f in feats if f in members]
        if needle:
            feats = [f for f in feats if needle in f.lower()]
        # Cap visible slice for readability
        if len(feats) > 24:
            feats = feats[:24]
            note = f"(showing first 24 of filtered set)\n\n"
        else:
            note = ""
        if not feats:
            self._matrix_text.insert(
                "end", "No features match search / cluster filter.\n"
            )
            self._matrix_text.configure(state="disabled")
            return

        mat = load_matrix_slice(self._data_dir, self._run_id, feats)
        short = [self._short_name(f) for f in feats]
        header = f"{note}{'':16}" + "".join(f"{s:>10}" for s in short) + "\n"
        self._matrix_text.insert("end", header)
        for i, a in enumerate(feats):
            line = f"{short[i]:<16}"
            self._matrix_text.insert("end", line)
            for b in feats:
                if a == b:
                    val = 1.0
                    tag = "diag"
                else:
                    val = float(mat.get(a, {}).get(b, float("nan")))
                    if val != val:  # NaN
                        self._matrix_text.insert("end", f"{'—':>10}")
                        continue
                    ar = abs(val)
                    tag = "hi" if ar >= 0.99 else ("mid" if ar >= 0.95 else "lo")
                self._matrix_text.insert("end", f"{val:>10.2f}", tag)
            self._matrix_text.insert("end", "\n")
        self._matrix_text.configure(state="disabled")

    @staticmethod
    def _short_name(name: str, width: int = 10) -> str:
        n = str(name)
        if len(n) <= width:
            return n
        return n[: width - 1] + "…"

    def _reload_clusters(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import load_clusters

        self._clusters_text.configure(state="normal")
        self._clusters_text.delete("1.0", "end")
        if not self._run_id:
            self._clusters_text.insert("end", "Run Correlation first.\n")
            self._clusters_text.configure(state="disabled")
            return
        clusters = load_clusters(self._data_dir, self._run_id)
        multi = [c for c in clusters if int(c.get("size") or 0) > 1]
        if not multi:
            self._clusters_text.insert(
                "end",
                "No multi-member clusters at |r| >= 0.95.\n"
                "Features are largely orthogonal (or Correlation not run).\n",
            )
            self._clusters_text.configure(state="disabled")
            return
        for c in multi:
            title = str(c.get("cluster") or "Cluster")
            self._clusters_text.insert("end", f"{title}\n", "head")
            self._clusters_text.insert("end", "-" * max(len(title), 12) + "\n")
            for m in c.get("members") or []:
                self._clusters_text.insert("end", f"{m}\n")
            self._clusters_text.insert(
                "end",
                f"\nRepresentative\n{c.get('representative') or '—'}\n",
                "meta",
            )
            high = float(c.get("highest_correlation") or 0.0)
            self._clusters_text.insert(
                "end",
                f"Highest correlation\n{high:.4f}\n\n",
                "meta",
            )
        self._clusters_text.configure(state="disabled")

    def _reload_insights(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation_insights import (
            load_correlation_insights,
            rebuild_insights_from_stored_correlation,
        )

        self._insights_tree.delete(*self._insights_tree.get_children())
        self._insight_detail.configure(state="normal")
        self._insight_detail.delete("1.0", "end")
        self._insight_detail.configure(state="disabled")
        if not self._run_id:
            return
        rows = load_correlation_insights(self._data_dir, self._run_id)
        # Backfill Insights from stored correlation if Correlation ran before Insights existed.
        if not rows:
            try:
                rows = rebuild_insights_from_stored_correlation(
                    self._data_dir, self._run_id
                )
            except Exception as exc:
                self._insight_detail.configure(state="normal")
                self._insight_detail.insert(
                    "end",
                    f"Could not build insights from stored correlation: {exc}\n"
                    "Check Correlation, then click Run Selected (or Refresh views).\n",
                )
                self._insight_detail.configure(state="disabled")
                return
        if not rows:
            self._insight_detail.configure(state="normal")
            self._insight_detail.insert(
                "end",
                "No insights yet. Select Correlation → Run Selected "
                "(or Refresh views after Correlation completes).\n",
            )
            self._insight_detail.configure(state="disabled")
            return
        for r in rows:
            rec = str(r.get("recommendation") or "Keep")
            tag = (
                "dup"
                if rec == "Duplicate Candidate"
                else ("review" if rec == "Review" else "keep")
            )
            member_n = int(r.get("member_count") or 0)
            if not member_n:
                if isinstance(r.get("members"), list):
                    member_n = len(r["members"])
                elif isinstance(r.get("members_list"), list):
                    member_n = len(r["members_list"])
                else:
                    try:
                        member_n = int(r.get("members") or 0)
                    except (TypeError, ValueError):
                        member_n = 0
            max_c = float(r.get("max_correlation") or 0.0)
            avg_c = float(r.get("avg_correlation") or 0.0)
            self._insights_tree.insert(
                "",
                "end",
                iid=str(r.get("cluster") or ""),
                values=(
                    r.get("cluster"),
                    r.get("family"),
                    member_n,
                    f"{max_c:.4f}",
                    f"{avg_c:.4f}",
                    rec,
                    str(r.get("reason") or "")[:180],
                ),
                tags=(tag,),
            )

    def _on_insight_select(self, _event: Any = None) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation_insights import (
            load_correlation_insights,
        )

        sel = self._insights_tree.selection()
        self._insight_detail.configure(state="normal")
        self._insight_detail.delete("1.0", "end")
        if not sel or not self._run_id:
            self._insight_detail.configure(state="disabled")
            return
        cluster = sel[0]
        rows = {
            str(r.get("cluster")): r
            for r in load_correlation_insights(self._data_dir, self._run_id)
        }
        r = rows.get(cluster)
        if not r:
            self._insight_detail.configure(state="disabled")
            return
        members = r.get("members_list") or []
        lines = [
            f"{r.get('cluster')}",
            f"Family: {r.get('family')}",
            f"Members: {int(r.get('members') or 0)}",
            f"Max Corr: {float(r.get('max_correlation') or 0.0):.4f}",
            f"Avg Corr: {float(r.get('avg_correlation') or 0.0):.4f}",
            f"Representative: {r.get('representative') or '—'}",
            f"Recommendation: {r.get('recommendation')}",
            "",
            "Reason:",
            str(r.get("reason") or ""),
            "",
            "Features:",
            *[f"  · {m}" for m in members],
        ]
        if r.get("shap_enrichment"):
            lines.extend(["", "SHAP:", str(r.get("shap_enrichment"))])
        if r.get("vif_enrichment"):
            lines.extend(["", "VIF:", str(r.get("vif_enrichment"))])
        lines.extend(
            [
                "",
                "Note: Correlation Insights never remove features automatically.",
            ]
        )
        self._insight_detail.insert("end", "\n".join(lines))
        self._insight_detail.configure(state="disabled")

    def _build_mi_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            bar,
            text="Top features by Mutual Information vs selected target (no model training).",
            foreground="#555",
        ).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self._reload_mi_tab).pack(side="right")

        cols = ("feature", "mi", "percentile", "category", "interpretation")
        self._mi_tree = ttk.Treeview(parent, columns=cols, show="headings", height=16)
        for key, title, width in (
            ("feature", "Feature", 260),
            ("mi", "MI", 80),
            ("percentile", "Percentile", 90),
            ("category", "Category", 120),
            ("interpretation", "Recommendation", 160),
        ):
            self._mi_tree.heading(key, text=title)
            self._mi_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self._mi_tree.yview)
        self._mi_tree.configure(yscrollcommand=ysb.set)
        self._mi_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._mi_tree.bind("<Double-1>", self._on_mi_double_click)

    def _reload_mi_targets(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
            discover_mi_targets,
        )

        ds = self._selected_dataset()
        if not ds:
            self._mi_targets = []
            self._mi_target_combo["values"] = []
            self._mi_target.set("")
            return
        try:
            self._mi_targets = discover_mi_targets(self._data_dir, ds)
        except Exception:
            self._mi_targets = []
        self._mi_target_combo["values"] = self._mi_targets
        cur = self._mi_target.get()
        if self._mi_targets:
            if cur not in self._mi_targets:
                # Prefer future_ltp_5m when present
                prefer = "future_ltp_5m"
                self._mi_target.set(
                    prefer if prefer in self._mi_targets else self._mi_targets[0]
                )
        else:
            self._mi_target.set("")

    def _reload_shap_models(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_shap import default_shap_model
        from .selection_lists import get_sorted_model_names

        try:
            self._shap_models = get_sorted_model_names(
                self._data_dir, lightweight=True, include_experiments=True
            )
        except Exception:
            self._shap_models = []
        self._shap_model_combo["values"] = self._shap_models
        cur = self._shap_model.get()
        if self._shap_models:
            if cur not in self._shap_models:
                prefer = ""
                try:
                    prefer = default_shap_model(self._data_dir)
                except Exception:
                    prefer = ""
                self._shap_model.set(
                    prefer if prefer in self._shap_models else self._shap_models[0]
                )
        else:
            self._shap_model.set("")

    def _build_permutation_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            bar,
            text=(
                "Permutation Importance — shuffle each predictor and measure "
                "metric degradation (no retraining)."
            ),
            foreground="#555",
        ).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self._reload_perm_tab).pack(
            side="right"
        )

        cols = (
            "rank",
            "feature",
            "delta_rmse",
            "delta_mae",
            "percentile",
            "recommendation",
        )
        self._perm_tree = ttk.Treeview(
            parent, columns=cols, show="headings", height=16
        )
        headings = {
            "rank": ("Rank", 50),
            "feature": ("Feature", 260),
            "delta_rmse": ("Delta RMSE", 90),
            "delta_mae": ("Delta MAE", 90),
            "percentile": ("Percentile", 80),
            "recommendation": ("Recommendation", 120),
        }
        self._perm_sort_col = "rank"
        self._perm_sort_desc = False
        for key, (title, width) in headings.items():
            self._perm_tree.heading(
                key,
                text=title,
                command=lambda c=key: self._sort_perm_tree(c),
            )
            self._perm_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self._perm_tree.yview)
        self._perm_tree.configure(yscrollcommand=ysb.set)
        self._perm_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._perm_tree.bind("<Double-1>", self._on_perm_double_click)
        self._perm_rows: list[dict[str, Any]] = []

    def _sort_perm_tree(self, col: str) -> None:
        if self._perm_sort_col == col:
            self._perm_sort_desc = not self._perm_sort_desc
        else:
            self._perm_sort_col = col
            self._perm_sort_desc = col in {"delta_rmse", "delta_mae", "percentile"}
        self._populate_perm_tree()

    def _populate_perm_tree(self) -> None:
        if not hasattr(self, "_perm_tree"):
            return
        self._perm_tree.delete(*self._perm_tree.get_children())
        rows = list(self._perm_rows)
        col = self._perm_sort_col
        rev = self._perm_sort_desc

        def _key(r: dict[str, Any]) -> Any:
            if col == "feature":
                return str(r.get("feature_name") or "")
            if col == "recommendation":
                return str(r.get("interpretation") or "")
            if col == "rank":
                return int(r.get("importance_rank") or 10**9)
            val = r.get(
                {
                    "delta_rmse": "delta_rmse",
                    "delta_mae": "delta_mae",
                    "percentile": "importance_percentile",
                }.get(col, "importance")
            )
            try:
                return float(val if val is not None else -1e18)
            except (TypeError, ValueError):
                return -1e18

        rows.sort(key=_key, reverse=rev)
        for r in rows:
            drmse = r.get("delta_rmse")
            dmae = r.get("delta_mae")
            # Classification runs store accuracy deltas in importance
            if drmse is None and r.get("delta_accuracy") is not None:
                drmse = r.get("delta_accuracy")
            if dmae is None and r.get("delta_f1") is not None:
                dmae = r.get("delta_f1")
            self._perm_tree.insert(
                "",
                "end",
                values=(
                    r.get("importance_rank"),
                    r.get("feature_name"),
                    f"{float(drmse):+.4f}" if drmse is not None else "—",
                    f"{float(dmae):+.4f}" if dmae is not None else "—",
                    f"{float(r.get('importance_percentile') or 0):.0f}%",
                    r.get("interpretation") or "—",
                ),
            )

    def _reload_perm_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_permutation import (
            load_permutation_results,
        )

        if not hasattr(self, "_perm_tree"):
            return
        self._perm_rows = []
        model = str(self._shap_model.get() or "").strip()
        target = str(self._mi_target.get() or "").strip()
        if self._run_id and model and target:
            try:
                self._perm_rows = load_permutation_results(
                    self._data_dir, self._run_id, model, target, limit=500
                )
            except Exception:
                self._perm_rows = []
        self._populate_perm_tree()

    def _on_perm_double_click(self, _event: Any = None) -> None:
        sel = self._perm_tree.selection()
        if not sel:
            return
        vals = self._perm_tree.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return
        feat = str(vals[1])
        try:
            self._select_research_tab('Feature Explorer')
        except Exception:
            pass
        try:
            names = list(self._feature_list.get(0, "end"))
            if feat in names:
                idx = names.index(feat)
                self._feature_list.selection_clear(0, "end")
                self._feature_list.selection_set(idx)
                self._feature_list.see(idx)
        except Exception:
            pass
        self._show_feature_profile(feat)

    def _cancel_running(self) -> None:
        tok = self._cancel_token
        if tok is not None:
            tok.cancel()
            self._perm_progress_text.set("Cancel requested… finishing current feature")
            self._detail_var.set("Cancel requested — progress is saved for resume.")

    def _reload_mi_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
            load_feature_profile,
        )
        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
            load_mi_results,
        )

        if not hasattr(self, "_mi_tree"):
            return
        self._mi_tree.delete(*self._mi_tree.get_children())
        if not self._run_id or not self._mi_target.get():
            return
        rows = load_mi_results(
            self._data_dir, self._run_id, self._mi_target.get(), limit=200
        )
        for r in rows:
            cat = "—"
            try:
                prof = load_feature_profile(
                    self._data_dir, self._run_id, str(r.get("feature"))
                )
                if prof:
                    cat = str(prof.get("category") or "—")
            except Exception:
                pass
            self._mi_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature"),
                    f"{float(r.get('score') or 0.0):.4f}",
                    f"{float(r.get('percentile') or 0.0):.0f}%",
                    cat,
                    r.get("interpretation") or "—",
                ),
            )

    def _on_mi_double_click(self, _event: Any = None) -> None:
        sel = self._mi_tree.selection()
        if not sel:
            return
        vals = self._mi_tree.item(sel[0], "values")
        if not vals:
            return
        feat = str(vals[0])
        try:
            self._select_research_tab('Feature Explorer')
        except Exception:
            pass
        try:
            names = list(self._feature_list.get(0, "end"))
            if feat in names:
                idx = names.index(feat)
                self._feature_list.selection_clear(0, "end")
                self._feature_list.selection_set(idx)
                self._feature_list.see(idx)
        except Exception:
            pass
        self._show_feature_profile(feat)

    # --- Feature Explorer / Scorecard --------------------------------------

    def _build_feature_explorer(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Search Feature").pack(side="left")
        entry = ttk.Entry(bar, textvariable=self._feature_search, width=36)
        entry.pack(side="left", padx=(6, 0))
        entry.bind("<KeyRelease>", lambda _e: self._filter_feature_list())
        ttk.Button(
            bar, text="Build / Refresh Profiles", command=self._build_profiles_now
        ).pack(side="right")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="Feature List", padding=4)
        left.pack(side="left", fill="y", padx=(0, 6))
        self._feature_list = tk.Listbox(left, width=36, height=22, exportselection=False)
        sb = ttk.Scrollbar(left, orient="vertical", command=self._feature_list.yview)
        self._feature_list.configure(yscrollcommand=sb.set)
        self._feature_list.pack(side="left", fill="y", expand=False)
        sb.pack(side="right", fill="y")
        self._feature_list.bind("<<ListboxSelect>>", self._on_feature_selected)

        right = ttk.LabelFrame(body, text="Feature Profile", padding=6)
        right.pack(side="left", fill="both", expand=True)
        self._profile_text = tk.Text(
            right,
            wrap="word",
            font=("Consolas", 9),
            background="#fafafa",
            height=22,
        )
        psb = ttk.Scrollbar(right, orient="vertical", command=self._profile_text.yview)
        self._profile_text.configure(yscrollcommand=psb.set)
        self._profile_text.pack(side="left", fill="both", expand=True)
        psb.pack(side="right", fill="y")
        self._profile_text.tag_configure("h", font=("Segoe UI", 9, "bold"))
        self._profile_text.tag_configure("muted", foreground="#666")
        self._profile_text.insert("end", self._profile_text_var.get())
        self._profile_text.configure(state="disabled")

    def _select_research_tab(self, title: str) -> None:
        try:
            for i in range(self._research_nb.index("end")):
                if str(self._research_nb.tab(i, "text") or "") == title:
                    self._research_nb.select(i)
                    self._hydrate_research_tab(title)
                    return
        except Exception:
            pass

    def _build_feature_families_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            bar,
            text=(
                "Feature Families (HCA) — organizes similar features. "
                "Lists representative candidates only; does not auto-select."
            ),
            foreground="#555",
            wraplength=640,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(bar, text="Refresh", command=self._reload_families_tab).pack(
            side="right"
        )

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        cols = ("family", "label", "members", "avg_r", "max_r", "candidates")
        self._families_tree = ttk.Treeview(
            left, columns=cols, show="headings", height=14
        )
        headings = {
            "family": ("Family", 80),
            "label": ("Label", 120),
            "members": ("Members", 70),
            "avg_r": ("Avg |r|", 70),
            "max_r": ("Max |r|", 70),
            "candidates": ("Top candidates", 260),
        }
        for key, (title, width) in headings.items():
            self._families_tree.heading(key, text=title)
            self._families_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self._families_tree.yview)
        self._families_tree.configure(yscrollcommand=ysb.set)
        self._families_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._families_tree.bind("<<TreeviewSelect>>", self._on_family_selected)

        right = ttk.LabelFrame(body, text="Family Detail", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._family_detail = tk.Text(
            right, wrap="word", font=("Consolas", 9), height=16, background="#fafafa"
        )
        self._family_detail.pack(fill="both", expand=True)
        self._family_detail.insert("end", self._family_detail_var.get())
        self._family_detail.configure(state="disabled")

    def _build_family_review_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Family Review — choose an Experiment Representative (which "
                "candidate to test first). Discovery suggests defaults and "
                "flags near-ties; it does not declare a permanent winner. "
                "Train + Validation + Experiment Comparison decide that."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            bar, textvariable=self._review_summary_var, foreground="#333"
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(bar, text="Filter").pack(side="left", padx=(8, 4))
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            FILTER_CHOICES,
            MANUAL_STATUSES,
            REVIEW_REASON_CHOICES,
        )

        filt = ttk.Combobox(
            bar,
            textvariable=self._review_filter,
            state="readonly",
            width=16,
            values=list(FILTER_CHOICES),
        )
        filt.pack(side="left")
        filt.bind("<<ComboboxSelected>>", lambda _e: self._reload_family_review_tab())
        ttk.Button(
            bar,
            text="Create Experiment…",
            command=self._create_experiment_from_review,
        ).pack(side="left", padx=(10, 0))

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        cols = (
            "family",
            "label",
            "members",
            "suggested",
            "gap",
            "confidence",
            "status",
            "exp_rep",
        )
        self._review_tree = ttk.Treeview(
            left, columns=cols, show="headings", height=12
        )
        headings = {
            "family": ("Family", 70),
            "label": ("Label", 100),
            "members": ("N", 40),
            "suggested": ("Suggested", 150),
            "gap": ("Gap", 50),
            "confidence": ("Conf", 55),
            "status": ("Status", 130),
            "exp_rep": ("Experiment Rep", 150),
        }
        for key, (title, width) in headings.items():
            self._review_tree.heading(key, text=title)
            self._review_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self._review_tree.yview)
        self._review_tree.configure(yscrollcommand=ysb.set)
        self._review_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._review_tree.bind("<<TreeviewSelect>>", self._on_review_family_selected)

        form = ttk.LabelFrame(body, text="Experiment Representative", padding=8)
        form.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(form, text="Family").grid(row=0, column=0, sticky="w")
        ttk.Label(form, textvariable=self._review_family_id).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(form, text="Experiment Rep").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self._review_rep_combo = ttk.Combobox(
            form,
            textvariable=self._review_rep,
            state="readonly",
            width=28,
            values=[],
        )
        self._review_rep_combo.grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Reason").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            form,
            textvariable=self._review_reason_code,
            state="readonly",
            width=28,
            values=list(REVIEW_REASON_CHOICES),
        ).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(form, text="Notes").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self._review_reason_text, width=30).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Status").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            form,
            textvariable=self._review_status,
            state="readonly",
            width=28,
            values=list(MANUAL_STATUSES),
        ).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Button(
            form, text="Save for Experiment", command=self._save_family_review
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(
            form, text="Refresh", command=self._reload_family_review_tab
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(
            form,
            text=(
                "This pick is for the next experiment only.\n"
                "Create variants on the Experiments tab."
            ),
            foreground="#666",
            justify="left",
            wraplength=220,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _reload_families_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_hca import load_families

        if not hasattr(self, "_families_tree"):
            return
        self._families_tree.delete(*self._families_tree.get_children())
        self._families_cache = []
        if not self._run_id:
            return
        try:
            self._families_cache = load_families(
                self._data_dir, self._run_id, min_size=2
            )
        except Exception as exc:
            self._family_detail_var.set(f"Failed to load families: {exc}")
            return
        for fam in self._families_cache:
            cands = fam.get("candidates") or []
            top = ", ".join(
                str(c.get("feature") or "") for c in cands[:3] if c.get("feature")
            )
            avg = fam.get("avg_corr")
            mx = fam.get("max_corr")
            self._families_tree.insert(
                "",
                "end",
                iid=str(fam.get("family_id")),
                values=(
                    fam.get("family_id"),
                    fam.get("family_label") or "—",
                    fam.get("size"),
                    f"{float(avg):.3f}" if avg is not None else "—",
                    f"{float(mx):.3f}" if mx is not None else "—",
                    top or "—",
                ),
            )

    def _on_family_selected(self, _event: Any = None) -> None:
        if not hasattr(self, "_families_tree"):
            return
        sel = self._families_tree.selection()
        if not sel:
            return
        fid = str(sel[0])
        fam = next(
            (f for f in self._families_cache if str(f.get("family_id")) == fid),
            None,
        )
        if not fam:
            return
        lines = [
            f"Family          {fam.get('family_id')}",
            f"Label           {fam.get('family_label')}",
            f"Members         {fam.get('size')}",
            f"Avg |r|         {fam.get('avg_corr')}",
            f"Max |r|         {fam.get('max_corr')}",
            "",
            "Representative candidates (evidence only — not selected):",
        ]
        for c in fam.get("candidates") or []:
            lines.append(
                f"  #{c.get('candidate_rank')}  {c.get('feature')}  "
                f"avg|r|={float(c.get('avg_corr_to_family') or 0):.4f}"
            )
        lines.append("")
        lines.append("All members:")
        for m in fam.get("members") or []:
            lines.append(f"  {m}")
        lines.append("")
        lines.append(
            "HCA lists candidates only — does not pick a winner.\n"
            "Choose an Experiment Representative on Family Review, then "
            "create experiments to train and compare."
        )
        text = "\n".join(lines)
        self._family_detail.configure(state="normal")
        self._family_detail.delete("1.0", "end")
        self._family_detail.insert("end", text)
        self._family_detail.configure(state="disabled")

    def _reload_discovery_banner(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            platform_workflow_summary,
        )

        if not hasattr(self, "_discovery_banner_var"):
            return
        if not self._run_id:
            self._discovery_banner_var.set(
                "Discovery  —  Load an Analysis Dataset to begin."
            )
            if hasattr(self, "_discovery_banner"):
                self._discovery_banner.configure(
                    foreground="#555", background="#f5f5f5"
                )
            return
        try:
            info = platform_workflow_summary(self._data_dir, self._run_id)
        except Exception as exc:
            self._discovery_banner_var.set(f"Discovery  —  {exc}")
            return
        self._discovery_banner_var.set(str(info.get("banner_text") or ""))
        disc = info.get("discovery") or {}
        if hasattr(self, "_discovery_banner"):
            if info.get("champion_id"):
                self._discovery_banner.configure(
                    foreground="#1a4d2e", background="#e8f5e9"
                )
            elif info.get("discovery_complete"):
                self._discovery_banner.configure(
                    foreground="#1a4d2e", background="#eef6f0"
                )
            elif disc.get("ready_to_create"):
                self._discovery_banner.configure(
                    foreground="#5c4800", background="#fff8e1"
                )
            else:
                self._discovery_banner.configure(
                    foreground="#5c1a1a", background="#fceaea"
                )

    def _fs_selection_config(self) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.analysis_feature_selection import (
            build_selection_config,
        )

        try:
            corr_thr = float(str(self._fs_corr_thr.get() or "0.95"))
        except ValueError:
            corr_thr = 0.95
        try:
            perm_thr = float(str(self._fs_perm_thr.get() or "0.001"))
        except ValueError:
            perm_thr = 0.001
        try:
            top_n = int(str(self._fs_top_n.get() or "4"))
        except ValueError:
            top_n = 4
        return build_selection_config(
            str(self._fs_strategy.get() or "hca_corr_perm"),
            representative_policy=str(self._fs_policy.get() or "top_1"),
            top_n=top_n,
            correlation_threshold=corr_thr,
            permutation_threshold=perm_thr,
        )

    def _on_fs_strategy_changed(self) -> None:
        sid = str(self._fs_strategy.get() or "hca_corr_perm")
        if hasattr(self, "_fs_hca_fr"):
            if sid == "hca_corr_perm":
                self._fs_hca_fr.pack(fill="x", pady=(8, 0))
                self._fs_flat_fr.pack_forget()
            else:
                self._fs_hca_fr.pack_forget()
                self._fs_flat_fr.pack(fill="x", pady=(8, 0))
        self._fs_preview_cache = None
        self._fs_preview_var.set(
            "Strategy updated — click Preview Final Dataset."
        )

    def _format_final_dataset_card(self, prev: dict[str, Any]) -> str:
        """UI card for Final Feature Dataset (shown above action buttons)."""
        from chain_replay_ml.dataset_builder.analysis_feature_selection import (
            STRATEGY_CORR_ONLY,
            STRATEGY_CORR_PERM,
            STRATEGY_HCA,
            STRATEGY_PERM_ONLY,
            STRATEGY_SHORT,
            normalize_strategy,
        )

        sid = normalize_strategy(prev.get("strategy"))
        strat = (
            prev.get("strategy_label")
            or STRATEGY_SHORT.get(sid, sid)
        )
        # Prefer short label style from the mock for Corr+Perm
        if sid == STRATEGY_CORR_PERM:
            strat = "Correlation + Permutation"
        elif sid == STRATEGY_HCA:
            pol = prev.get("representative_policy_label") or ""
            strat = f"HCA + Correlation + Permutation"
            if pol:
                strat = f"{strat} ({pol})"
        lines = [
            "Final Feature Dataset",
            "──────────────────────────────────────────",
            f"Features : {prev.get('n_features') if prev.get('n_features') is not None else '—'}",
            f"Strategy : {strat}",
        ]
        if sid == STRATEGY_HCA and prev.get("representative_policy_label"):
            lines.append(
                f"Representative Policy : {prev.get('representative_policy_label')}"
            )
            if prev.get("n_families") is not None:
                lines.append(f"Families : {prev.get('n_families')}")
        if sid in (STRATEGY_CORR_PERM, STRATEGY_CORR_ONLY) and prev.get(
            "correlation_threshold"
        ) is not None:
            lines.append(
                f"Correlation Threshold : {float(prev['correlation_threshold']):.2f}"
            )
        if sid in (STRATEGY_CORR_PERM, STRATEGY_PERM_ONLY) and prev.get(
            "permutation_threshold"
        ) is not None:
            lines.append(
                f"Permutation Threshold : {float(prev['permutation_threshold']):.6g}"
            )
        return "\n".join(lines)

    def _preview_feature_selection(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_selection import (
            preview_selection,
        )

        if not self._run_id:
            messagebox.showinfo(
                "Feature Selection", "Load an analysis dataset first.", parent=self
            )
            return
        try:
            prev = preview_selection(
                self._data_dir, self._run_id, self._fs_selection_config()
            )
        except Exception as exc:
            messagebox.showerror("Feature Selection", str(exc), parent=self)
            return
        self._fs_preview_cache = prev
        self._fs_preview_var.set(self._format_final_dataset_card(prev))

    def _ensure_final_feature_dataset(self) -> dict[str, Any] | None:
        """Return a Final Feature Dataset preview, building one if needed."""
        prev = self._fs_preview_cache
        if prev and list(prev.get("features") or []):
            return prev
        self._preview_feature_selection()
        prev = self._fs_preview_cache
        if not prev or not list(prev.get("features") or []):
            messagebox.showwarning(
                "Create Model Builder",
                "No Final Feature Dataset yet.\n\n"
                "Run the required Analysis modules, then Preview Final Dataset.",
                parent=self,
            )
            return None
        return prev

    def _create_model_builder_from_selection(self) -> None:
        """Hand Final Feature Dataset → Model Builder (pre-selected features)."""
        from chain_replay_ml.dataset_builder.analysis_feature_selection import (
            STRATEGY_SHORT,
            normalize_strategy,
        )
        from .model_builder.feature_preset import save_feature_preset as save_mb_preset

        prev = self._ensure_final_feature_dataset()
        if not prev:
            return
        features = [
            str(f).strip() for f in (prev.get("features") or []) if str(f).strip()
        ]
        if not features:
            messagebox.showwarning(
                "Create Model Builder",
                "Final Feature Dataset is empty — nothing to send to Model Builder.",
                parent=self,
            )
            return

        ds = None
        name = self._dataset_choice.get().strip()
        for d in self._datasets:
            if str(d.get("name") or "") == name or str(d.get("dataset_id") or "") == name:
                ds = d
                break
        dataset_name = str((ds or {}).get("name") or "").strip() or None
        sid = normalize_strategy(prev.get("strategy"))
        source = (
            f"FeatureSelection:{STRATEGY_SHORT.get(sid, sid)}"
            f":{len(features)}feats"
        )

        from chain_replay_ml.dataset_builder.analysis_feature_selection import (
            build_feature_selection_lineage,
        )

        bundle_id = None
        try:
            from chain_replay_ml.dataset_builder.analysis_family_review import (
                discovery_readiness,
            )

            if self._run_id:
                ready = discovery_readiness(self._data_dir, self._run_id)
                bundle_id = ready.get("latest_discovery_bundle_id")
        except Exception:
            bundle_id = None

        lineage = build_feature_selection_lineage(
            prev,
            source="analysis",
            run_id=self._run_id,
            analysis_dataset=dataset_name,
            discovery_bundle_id=str(bundle_id or "").strip() or None,
        )

        try:
            save_mb_preset(
                self.chart_dir,
                features=features,
                dataset=dataset_name,
                source_model=source,
                analysis_feature_selection=lineage,
            )
        except Exception as exc:
            messagebox.showerror(
                "Create Model Builder",
                f"Could not save Feature Set preset:\n{exc}",
                parent=self,
            )
            return

        self._fs_preview_var.set(
            self._format_final_dataset_card(prev)
            + f"\n\n→ Sending {len(features)} features to Model Builder"
            + (f"\n  Analysis dataset: {dataset_name}" if dataset_name else "")
        )

        cb = self._on_open_model_builder
        if not callable(cb):
            messagebox.showinfo(
                "Create Model Builder",
                f"Saved Feature Set ({len(features)} features).\n\n"
                "Open Model Builder → Create Model to load the preset.\n"
                "(No Model Builder callback is wired in this window.)",
                parent=self,
            )
            return
        try:
            # Pass analysis name as a hint; Model Builder only auto-selects if
            # it matches a training dataset in the catalog.
            cb(
                features=features,
                dataset=dataset_name,
                source_model=source,
                analysis_feature_selection=lineage,
            )
        except TypeError:
            try:
                cb(source, features, dataset_name)
            except Exception as exc:
                messagebox.showerror("Create Model Builder", str(exc), parent=self)
        except Exception as exc:
            messagebox.showerror("Create Model Builder", str(exc), parent=self)

    def _view_final_feature_dataset(self) -> None:
        prev = self._fs_preview_cache
        if not prev:
            self._preview_feature_selection()
            prev = self._fs_preview_cache
        if not prev:
            return
        feats = list(prev.get("features") or [])
        win = tk.Toplevel(self)
        win.title(
            f"Final Feature Dataset · {prev.get('n_features')} features"
        )
        win.geometry("480x520")
        txt = tk.Text(win, wrap="word", font=("Consolas", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert(
            "end",
            (prev.get("summary_text") or "")
            + "\n\n"
            + "\n".join(f"{i}. {f}" for i, f in enumerate(feats, start=1)),
        )
        txt.configure(state="disabled")

    def _freeze_discovery_bundle(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_artifacts import (
            publish_discovery_bundle,
        )

        if not self._run_id:
            messagebox.showinfo(
                "Discovery", "Load an analysis dataset first.", parent=self
            )
            return
        cfg = self._fs_selection_config()
        try:
            art = publish_discovery_bundle(
                self._data_dir,
                self._run_id,
                feature_selection=cfg,
            )
        except Exception as exc:
            messagebox.showerror("Discovery Bundle", str(exc), parent=self)
            return
        card = art.get("card") or {}
        self._detail_var.set(
            f"Frozen {art.get('artifact_ref') or art.get('artifact_id')}"
        )
        self._reload_discovery_banner()
        pl = (art.get("payload") or {}) if isinstance(art, dict) else {}
        n_sel = pl.get("n_selected_features") or card.get("n_selected_features")
        self._fs_preview_var.set(
            f"Frozen Final Feature Dataset\n"
            f"Features: {n_sel}\n"
            f"Strategy: {(pl.get('feature_selection') or {}).get('strategy_label') or cfg.get('strategy_label')}\n"
            f"Bundle: {art.get('artifact_id')}"
        )
        messagebox.showinfo(
            "Discovery Bundle",
            f"Published immutable artifact:\n"
            f"ID: {art.get('artifact_id')}\n"
            f"Fingerprint: {art.get('fingerprint_short') or art.get('fingerprint')}\n"
            f"Selected features: {n_sel}\n"
            f"Strategy: {cfg.get('strategy_label')}\n\n"
            "Create Experiment / Create Model will store this ID + fingerprint.",
            parent=self,
        )

    def _reload_family_review_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            FILTER_NEEDS_REVIEW,
            load_families_with_reviews,
            review_summary,
        )

        if not hasattr(self, "_review_tree"):
            return
        self._review_tree.delete(*self._review_tree.get_children())
        if not self._run_id:
            self._review_summary_var.set("")
            return
        filt = str(self._review_filter.get() or FILTER_NEEDS_REVIEW)
        try:
            rows = load_families_with_reviews(
                self._data_dir,
                self._run_id,
                min_size=2,
                status_filter=filt,
            )
            summary = review_summary(self._data_dir, self._run_id, min_size=2)
        except Exception as exc:
            self._review_summary_var.set(f"Failed to load reviews: {exc}")
            return
        self._review_summary_var.set(
            f"Families {summary.get('n_families', 0)} · "
            f"Suggested Default {summary.get('n_auto_accepted', 0)} · "
            f"Needs Review {summary.get('n_needs_review', 0)} · "
            f"Showing {len(rows)} ({filt})"
        )
        self._families_cache = rows
        for fam in rows:
            gap = fam.get("score_gap")
            gap_s = f"{float(gap):.0f}" if gap is not None else "—"
            exp_rep = (
                fam.get("experiment_representative")
                or fam.get("selected_representative")
                or "—"
            )
            self._review_tree.insert(
                "",
                "end",
                iid=str(fam.get("family_id")),
                values=(
                    fam.get("family_id"),
                    fam.get("family_label") or "—",
                    fam.get("size"),
                    fam.get("suggested_representative") or "—",
                    gap_s,
                    fam.get("confidence") or "—",
                    fam.get("review_status") or "Unreviewed",
                    exp_rep,
                ),
            )
        self._reload_discovery_banner()

    def _on_review_family_selected(self, _event: Any = None) -> None:
        if not hasattr(self, "_review_tree"):
            return
        sel = self._review_tree.selection()
        if not sel:
            return
        fid = str(sel[0])
        fam = next(
            (f for f in self._families_cache if str(f.get("family_id")) == fid),
            None,
        )
        if not fam:
            return
        self._review_family_id.set(fid)
        members = list(fam.get("members") or [])
        self._review_rep_combo["values"] = members
        selected = str(
            fam.get("experiment_representative")
            or fam.get("selected_representative")
            or ""
        ).strip()
        suggested = str(fam.get("suggested_representative") or "").strip()
        if selected and selected in members:
            self._review_rep.set(selected)
        elif suggested and suggested in members:
            self._review_rep.set(suggested)
        elif members:
            self._review_rep.set(members[0])
        else:
            self._review_rep.set("")
        if fam.get("review_reason_code") and fam.get("review_reason_code") != "Auto":
            self._review_reason_code.set(str(fam.get("review_reason_code")))
        else:
            self._review_reason_code.set("Interpretability")
        note = str(fam.get("review_reason_text") or "")
        # Don't dump auto message into notes field for overrides
        if fam.get("decision_source") == "manual":
            self._review_reason_text.set(note)
        else:
            self._review_reason_text.set("")
        self._review_status.set("For Experiment")

    def _save_family_review(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            STATUS_FOR_EXPERIMENT,
            upsert_family_review,
        )

        fid = str(self._review_family_id.get() or "").strip()
        if not self._run_id or not fid:
            messagebox.showinfo(
                "Family Review",
                "Select a family first.",
                parent=self,
            )
            return
        try:
            upsert_family_review(
                self._data_dir,
                self._run_id,
                fid,
                experiment_representative=str(self._review_rep.get() or ""),
                reason_code=str(self._review_reason_code.get() or ""),
                reason_text=str(self._review_reason_text.get() or ""),
                status=str(self._review_status.get() or STATUS_FOR_EXPERIMENT),
            )
        except Exception as exc:
            messagebox.showerror("Family Review", str(exc), parent=self)
            return
        self._detail_var.set(
            f"Experiment Representative saved · {fid} → {self._review_rep.get()}"
        )
        self._reload_family_review_tab()
        self._reload_discovery_banner()
        try:
            # May disappear from Needs Review filter — that's expected
            if fid in self._review_tree.get_children(""):
                self._review_tree.selection_set(fid)
                self._review_tree.see(fid)
        except Exception:
            pass

    def _create_experiment_from_review(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            create_experiment,
        )
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            discovery_readiness,
        )

        if not self._run_id:
            messagebox.showinfo(
                "Experiments", "Load an analysis dataset first.", parent=self
            )
            return
        cfg = self._fs_selection_config()
        try:
            ready = discovery_readiness(
                self._data_dir,
                self._run_id,
                strategy=str(cfg.get("strategy") or ""),
            )
        except Exception:
            ready = {"ready_to_create": True}
        if not ready.get("ready_to_create"):
            messagebox.showinfo(
                "Discovery Incomplete",
                str(ready.get("banner_text") or "Finish Discovery first."),
                parent=self,
            )
            return
        try:
            exp = create_experiment(
                self._data_dir,
                self._run_id,
                name=str(self._exp_name.get() or "").strip(),
                notes=str(self._exp_notes.get() or "").strip()
                or "Feature-set hypothesis from Feature Selection Strategy",
                feature_selection=cfg,
            )
        except Exception as exc:
            messagebox.showerror("Experiments", str(exc), parent=self)
            return
        eid = exp.get("experiment_id")
        self._detail_var.set(
            f"Created {eid} (Ready) — feature-set hypothesis frozen"
        )
        self._reload_discovery_banner()
        self._reload_experiments_tab()
        self._reload_experiment_compare_tab()
        try:
            self._select_research_tab("Experiments")
        except Exception:
            pass

    def _build_discovery_scorecard_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Discovery Scorecard — Corr + MI + Permutation. "
                "REVIEW FAMILY means: pick an Experiment Representative inside "
                "the HCA family (test candidate, not a permanent winner). "
                "Double-click or Open Family."
            ),
            foreground="#555",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(
            bar, text="Open Family", command=self._open_scorecard_family
        ).pack(side="left")
        ttk.Button(
            bar, text="Open in Explorer", command=self._on_scorecard_double_click
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            bar,
            text="Selection shows family context below.",
            foreground="#666",
        ).pack(side="left", padx=(12, 0))

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        cols = (
            "feature",
            "category",
            "family",
            "coverage",
            "recommendation",
            "mi",
            "perm",
            "score",
        )
        self._score_tree = ttk.Treeview(
            left, columns=cols, show="headings", height=12
        )
        headings = {
            "feature": ("Feature", 180),
            "category": ("Category", 90),
            "family": ("HCA Family", 110),
            "coverage": ("Cov%", 55),
            "recommendation": ("Recommendation", 130),
            "mi": ("MI", 85),
            "perm": ("Perm", 95),
            "score": ("Score", 140),
        }
        for key, (title, width) in headings.items():
            self._score_tree.heading(key, text=title)
            self._score_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self._score_tree.yview)
        self._score_tree.configure(yscrollcommand=ysb.set)
        self._score_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._score_tree.bind("<Double-1>", self._on_scorecard_activate)
        self._score_tree.bind("<<TreeviewSelect>>", self._on_scorecard_select)

        right = ttk.LabelFrame(body, text="Family Context", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._score_family_ctx = tk.Text(
            right,
            wrap="word",
            font=("Consolas", 9),
            height=14,
            background="#fafafa",
        )
        self._score_family_ctx.pack(fill="both", expand=True)
        self._score_family_ctx.insert(
            "end",
            "Select a REVIEW FAMILY row to see candidates, scores, and gap.\n"
            "Then Open Family to choose an Experiment Representative.",
        )
        self._score_family_ctx.configure(state="disabled")

    def _build_experiments_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Experiment List = snapshots + Train. Auto Research = live "
                "dashboard (does not hide the table)."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        self._exp_nb = ttk.Notebook(parent)
        self._exp_nb.pack(fill="both", expand=True)

        list_tab = ttk.Frame(self._exp_nb, padding=4)
        auto_tab = ttk.Frame(self._exp_nb, padding=6)
        self._exp_nb.add(list_tab, text="Experiment List")
        self._exp_nb.add(auto_tab, text="Auto Research")

        # --- Experiment List ---
        bundle_fr = ttk.LabelFrame(list_tab, text="Discovery Bundle", padding=4)
        bundle_fr.pack(fill="x", pady=(0, 4))
        self._exp_bundle_text = tk.Label(
            bundle_fr,
            textvariable=self._exp_bundle_var,
            justify="left",
            anchor="w",
            font=("Consolas", 9),
            foreground="#333",
        )
        self._exp_bundle_text.pack(fill="x")

        bar = ttk.Frame(list_tab)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Name").pack(side="left")
        ttk.Entry(bar, textvariable=self._exp_name, width=14).pack(
            side="left", padx=(4, 6)
        )
        ttk.Label(bar, text="Notes").pack(side="left")
        ttk.Entry(bar, textvariable=self._exp_notes, width=20).pack(
            side="left", padx=(4, 6)
        )
        ttk.Button(
            bar,
            text="Create Snapshot",
            command=self._create_experiment_from_review,
        ).pack(side="left")
        ttk.Button(
            bar, text="Refresh", command=self._reload_experiments_tab
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            bar, textvariable=self._exp_summary_var, foreground="#333"
        ).pack(side="left", padx=(10, 0))

        link = ttk.Frame(list_tab)
        link.pack(fill="x", pady=(0, 4))
        ttk.Button(
            link,
            text="Train",
            command=self._train_selected_experiment,
        ).pack(side="left")
        ttk.Button(
            link,
            text="▶ Auto Research",
            command=self._auto_create_and_train,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Continue Auto Research",
            command=self._continue_auto_research,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Promote Champion",
            command=self._promote_champion,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Clone variant…",
            command=self._clone_experiment_variant,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Delete",
            command=self._delete_selected_experiment,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Open Comparison",
            command=self._open_experiment_comparison_table,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            link,
            text="Auto Research tab",
            command=lambda: self._select_experiments_subtab("Auto Research"),
        ).pack(side="left", padx=(8, 0))

        body = ttk.Frame(list_tab)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        cols = ("id", "name", "status", "device", "reps", "holdout", "wf", "validation")
        self._exp_tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        headings = {
            "id": ("Experiment", 80),
            "name": ("Name", 100),
            "status": ("Status", 100),
            "device": ("Device", 70),
            "reps": ("Feature-set (Family → Rep)", 260),
            "holdout": ("Model Holdout", 90),
            "wf": ("Model Walk-fwd", 100),
            "validation": ("Model Valid.", 85),
        }
        for key, (title, width) in headings.items():
            self._exp_tree.heading(key, text=title)
            self._exp_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self._exp_tree.yview)
        self._exp_tree.configure(yscrollcommand=ysb.set)
        self._exp_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._exp_tree.bind("<<TreeviewSelect>>", self._on_experiment_selected)

        right = ttk.LabelFrame(body, text="Experiment Details", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._exp_hypothesis = tk.Text(
            right,
            wrap="word",
            font=("Consolas", 9),
            height=12,
            background="#fafafa",
        )
        self._exp_hypothesis.pack(fill="both", expand=True)
        self._exp_hypothesis.insert(
            "end",
            "Select an experiment to view its full Experiment Details.\n"
            "Includes Final Feature Set, family mapping, parent changes,\n"
            "metrics, and model information (device / path).\n"
            "Lifecycle: Created → Training → Model Produced → Validated → Champion",
        )
        self._exp_hypothesis.configure(state="disabled")

        # --- Auto Research dashboard (own tab — never covers the table) ---
        level_fr = ttk.LabelFrame(auto_tab, text="Research Level", padding=6)
        level_fr.pack(fill="x", pady=(0, 6))
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            LEVEL_BALANCED,
            LEVEL_DEEP,
            LEVEL_HINTS,
            LEVEL_LABELS,
            LEVEL_QUICK,
            LEVEL_TO_STRATEGY,
            PER_ROUND_CHOICES,
            STRATEGY_BEAM,
            STRATEGY_GENETIC,
            STRATEGY_GREEDY,
            STRATEGY_HILL,
            STRATEGY_LABELS,
            STRATEGY_SINGLE_SWAP,
            STRATEGY_TO_LEVEL,
        )

        for lid in (LEVEL_QUICK, LEVEL_BALANCED, LEVEL_DEEP):
            row = ttk.Frame(level_fr)
            row.pack(fill="x", anchor="w")
            ttk.Radiobutton(
                row,
                text=LEVEL_LABELS[lid],
                value=lid,
                variable=self._research_level,
                command=self._on_research_level_changed,
            ).pack(side="left")
            ttk.Label(
                row,
                text=LEVEL_HINTS[lid],
                foreground="#666",
            ).pack(side="left", padx=(10, 0))

        adv_row = ttk.Frame(level_fr)
        adv_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            adv_row,
            text="Advanced — pick exact algorithm",
            variable=self._research_advanced,
            command=self._on_research_advanced_toggled,
        ).pack(anchor="w")

        self._research_advanced_fr = ttk.Frame(level_fr)
        for sid, enabled in (
            (STRATEGY_SINGLE_SWAP, True),
            (STRATEGY_GREEDY, True),
            (STRATEGY_HILL, True),
            (STRATEGY_BEAM, False),
            (STRATEGY_GENETIC, False),
        ):
            label = STRATEGY_LABELS[sid]
            if not enabled:
                label = f"{label} (coming soon)"
            ttk.Radiobutton(
                self._research_advanced_fr,
                text=label,
                value=sid,
                variable=self._research_strategy,
                state="normal" if enabled else "disabled",
                command=self._on_research_strategy_changed,
            ).pack(anchor="w")

        count_row = ttk.Frame(level_fr)
        count_row.pack(fill="x", pady=(8, 0))
        ttk.Label(count_row, text="Neighbours / iteration").pack(side="left")
        self._research_per_round_cb = ttk.Combobox(
            count_row,
            textvariable=self._research_per_round,
            values=[str(n) for n in PER_ROUND_CHOICES],
            width=6,
            state="readonly",
        )
        self._research_per_round_cb.pack(side="left", padx=(8, 0))
        ttk.Label(
            count_row,
            text="(search budget per step)",
            foreground="#666",
        ).pack(side="left", padx=(8, 0))

        # Defaults: Balanced → Greedy; Advanced hidden
        self._research_level.set(LEVEL_BALANCED)
        self._research_strategy.set(LEVEL_TO_STRATEGY[LEVEL_BALANCED])
        self._on_research_advanced_toggled()

        auto_bar = ttk.Frame(auto_tab)
        auto_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            auto_bar,
            text="▶ Auto Research",
            command=self._auto_create_and_train,
        ).pack(side="left")
        ttk.Button(
            auto_bar,
            text="Continue Auto Research",
            command=self._continue_auto_research,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            auto_bar,
            text="Back to Experiment List",
            command=lambda: self._select_experiments_subtab("Experiment List"),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            auto_bar,
            text="Open Champion",
            command=lambda: self._select_research_tab("Champion"),
        ).pack(side="left", padx=(8, 0))

        dash = ttk.LabelFrame(auto_tab, text="Research Statistics", padding=6)
        dash.pack(fill="both", expand=True)
        self._auto_dashboard_text = tk.Text(
            dash,
            wrap="word",
            font=("Consolas", 9),
            height=16,
            background="#e8f1f8",
            foreground="#0d3b66",
            relief="flat",
            padx=6,
            pady=6,
        )
        dash_ysb = ttk.Scrollbar(
            dash, orient="vertical", command=self._auto_dashboard_text.yview
        )
        self._auto_dashboard_text.configure(yscrollcommand=dash_ysb.set)
        self._auto_dashboard_text.pack(side="left", fill="both", expand=True)
        dash_ysb.pack(side="right", fill="y")
        self._auto_dashboard_text.insert("1.0", self._auto_dashboard_var.get())
        self._auto_dashboard_text.configure(state="disabled")

    def _build_champion_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Final destination after Auto Research — Champion Feature Set "
                "ready for production hand-off."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            bar, text="Refresh", command=self._reload_champion_tab
        ).pack(side="left")
        ttk.Button(
            bar, text="Promote Model", command=self._promote_champion
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            bar, text="Export Champion", command=self._export_champion
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            bar,
            text="Open Comparison Table",
            command=self._open_experiment_comparison_table,
        ).pack(side="left", padx=(8, 0))

        card = ttk.LabelFrame(parent, text="Champion Bundle", padding=8)
        card.pack(fill="both", expand=True)
        self._champion_text = tk.Text(
            card,
            wrap="word",
            font=("Consolas", 10),
            height=18,
            background="#e8f5e9",
            foreground="#1a4d2e",
            relief="flat",
            padx=8,
            pady=8,
        )
        ysb = ttk.Scrollbar(card, orient="vertical", command=self._champion_text.yview)
        self._champion_text.configure(yscrollcommand=ysb.set)
        self._champion_text.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._champion_text.insert("1.0", self._champion_card_var.get())
        self._champion_text.configure(state="disabled")

    def _set_champion_card_text(self, text: str) -> None:
        msg = str(text or "")
        self._champion_card_var.set(msg)
        if hasattr(self, "_champion_text"):
            self._champion_text.configure(state="normal")
            self._champion_text.delete("1.0", "end")
            self._champion_text.insert("1.0", msg)
            self._champion_text.configure(state="disabled")

    def _reload_champion_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            format_champion_card,
            _reps_map,
        )
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            format_champion_bundle_card,
            list_experiments,
            load_champion_bundle,
            load_experiment,
        )

        if not self._run_id:
            self._set_champion_card_text("Load an analysis dataset first.")
            return
        try:
            experiments = list_experiments(self._data_dir, self._run_id)
        except Exception as exc:
            self._set_champion_card_text(f"Failed: {exc}")
            return

        parts: list[str] = []
        try:
            bundle = load_champion_bundle(self._data_dir, self._run_id)
        except Exception:
            bundle = None
        if bundle:
            parts.append(format_champion_bundle_card(bundle))

        champ = next(
            (e for e in experiments if int(e.get("is_champion") or 0)),
            None,
        )
        if not champ and self._auto_resume_id:
            champ = load_experiment(self._data_dir, str(self._auto_resume_id))
        if champ:
            baseline = next(
                (
                    e
                    for e in experiments
                    if str(e.get("name") or "") == "Auto-baseline"
                ),
                None,
            )
            baseline_reps = _reps_map(baseline) if baseline else None
            fam_labels = {
                str(r.get("family_id")): str(
                    r.get("family_label") or r.get("family_id")
                )
                for r in (champ.get("family_reps") or [])
            }
            parts.append(
                format_champion_card(
                    experiment=champ,
                    baseline_reps=baseline_reps,
                    fam_labels=fam_labels,
                    research_complete=True,
                )
            )
        if not parts:
            self._set_champion_card_text(
                "No champion yet — run ▶ Auto Research first."
            )
            return
        self._set_champion_card_text("\n\n".join(parts))

    def _export_champion(self) -> None:
        from tkinter import filedialog

        text = str(self._champion_card_var.get() or "").strip()
        if not text or text.startswith("No champion"):
            messagebox.showinfo(
                "Export Champion",
                "No champion card to export yet.",
                parent=self,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Champion Bundle",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
            initialfile="champion_bundle.txt",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        except Exception as exc:
            messagebox.showerror("Export Champion", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Export Champion", f"Saved to:\n{path}", parent=self
        )

    def _select_experiments_subtab(self, title: str) -> None:
        if not hasattr(self, "_exp_nb"):
            return
        try:
            for i in range(self._exp_nb.index("end")):
                if str(self._exp_nb.tab(i, "text") or "") == title:
                    self._exp_nb.select(i)
                    return
        except Exception:
            pass

    def _set_auto_dashboard_text(self, text: str) -> None:
        msg = str(text or "")
        if hasattr(self, "_auto_dashboard_var"):
            self._auto_dashboard_var.set(msg)
        if hasattr(self, "_auto_dashboard_text"):
            self._auto_dashboard_text.configure(state="normal")
            self._auto_dashboard_text.delete("1.0", "end")
            self._auto_dashboard_text.insert("1.0", msg)
            self._auto_dashboard_text.configure(state="disabled")

    def _open_experiment_comparison_summary(self) -> None:
        self._select_research_tab("Experiment Comparison")
        self._reload_experiment_compare_tab()
        self._select_compare_subtab("Summary")

    def _build_experiment_compare_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Summary = Auto Research dashboard + recommendation. "
                "Comparison Table = experiment rows (Holdout / Walk-fwd / "
                "Families Changed). Scores are model metrics, not Discovery."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        self._cmp_nb = ttk.Notebook(parent)
        self._cmp_nb.pack(fill="both", expand=True)

        summary_tab = ttk.Frame(self._cmp_nb, padding=6)
        table_tab = ttk.Frame(self._cmp_nb, padding=6)
        self._cmp_nb.add(summary_tab, text="Summary")
        self._cmp_nb.add(table_tab, text="Comparison Table")

        # --- Summary sub-tab ---
        sum_bar = ttk.Frame(summary_tab)
        sum_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            sum_bar,
            text="Refresh",
            command=self._reload_experiment_compare_tab,
        ).pack(side="left")
        ttk.Button(
            sum_bar,
            text="Open Comparison Table",
            command=lambda: self._select_compare_subtab("Comparison Table"),
        ).pack(side="left", padx=(8, 0))

        rec_fr = ttk.LabelFrame(summary_tab, text="Recommendation", padding=6)
        rec_fr.pack(fill="both", expand=True)
        self._cmp_recommend_var = tk.StringVar(
            value=(
                "Run ▶ Auto Research or Train experiments to see the "
                "dashboard and an explainable recommendation here."
            )
        )
        self._cmp_recommend_text = tk.Text(
            rec_fr,
            wrap="word",
            font=("Consolas", 9),
            height=18,
            background="#eef6f0",
            foreground="#1a4d2e",
            relief="flat",
            padx=6,
            pady=6,
        )
        rec_ysb = ttk.Scrollbar(
            rec_fr, orient="vertical", command=self._cmp_recommend_text.yview
        )
        self._cmp_recommend_text.configure(yscrollcommand=rec_ysb.set)
        self._cmp_recommend_text.pack(side="left", fill="both", expand=True)
        rec_ysb.pack(side="right", fill="y")
        self._cmp_recommend_text.insert("1.0", self._cmp_recommend_var.get())
        self._cmp_recommend_text.configure(state="disabled")

        # --- Comparison Table sub-tab ---
        bar = ttk.Frame(table_tab)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Focus family").pack(side="left")
        self._exp_focus_combo = ttk.Combobox(
            bar,
            textvariable=self._exp_focus_family,
            state="readonly",
            width=28,
            values=[],
        )
        self._exp_focus_combo.pack(side="left", padx=(4, 8))
        self._exp_focus_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._reload_experiment_compare_tab(),
        )
        ttk.Button(
            bar, text="Refresh", command=self._reload_experiment_compare_tab
        ).pack(side="left")
        ttk.Button(
            bar,
            text="Open Summary",
            command=lambda: self._select_compare_subtab("Summary"),
        ).pack(side="left", padx=(8, 0))

        table_body = ttk.Frame(table_tab)
        table_body.pack(fill="both", expand=True)
        cols = (
            "experiment",
            "changed",
            "old_rep",
            "new_rep",
            "n_chg",
            "device",
            "holdout",
            "wf",
            "delta",
            "validation",
            "champion",
        )
        self._cmp_tree = ttk.Treeview(
            table_body, columns=cols, show="headings", height=14
        )
        headings = {
            "experiment": ("Experiment", 75),
            "changed": ("Changed Family", 100),
            "old_rep": ("Old Rep", 100),
            "new_rep": ("New Rep", 100),
            "n_chg": ("Families Changed", 85),
            "device": ("Device", 60),
            "holdout": ("Holdout", 70),
            "wf": ("Walk-fwd", 70),
            "delta": ("Δ vs Base", 70),
            "validation": ("Validation", 70),
            "champion": ("Champion", 90),
        }
        for key, (title, width) in headings.items():
            self._cmp_tree.heading(key, text=title)
            self._cmp_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(
            table_body, orient="vertical", command=self._cmp_tree.yview
        )
        self._cmp_tree.configure(yscrollcommand=ysb.set)
        self._cmp_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

    def _select_compare_subtab(self, title: str) -> None:
        if not hasattr(self, "_cmp_nb"):
            return
        try:
            for i in range(self._cmp_nb.index("end")):
                if str(self._cmp_nb.tab(i, "text") or "") == title:
                    self._cmp_nb.select(i)
                    return
        except Exception:
            pass

    def _open_experiment_comparison_table(self) -> None:
        self._select_research_tab("Experiment Comparison")
        self._reload_experiment_compare_tab()
        self._select_compare_subtab("Comparison Table")

    def _set_compare_recommendation_text(self, text: str) -> None:
        msg = str(text or "")
        if hasattr(self, "_cmp_recommend_var"):
            self._cmp_recommend_var.set(msg)
        if hasattr(self, "_cmp_recommend_text"):
            self._cmp_recommend_text.configure(state="normal")
            self._cmp_recommend_text.delete("1.0", "end")
            self._cmp_recommend_text.insert("1.0", msg)
            self._cmp_recommend_text.configure(state="disabled")

    def _reload_experiments_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            discovery_bundle_card,
            list_experiments,
        )

        if not hasattr(self, "_exp_tree"):
            return
        self._exp_tree.delete(*self._exp_tree.get_children())
        self._experiments_cache = []
        if not self._run_id:
            self._exp_summary_var.set("")
            self._exp_bundle_var.set(
                "Discovery Bundle\n  (load an Analysis Dataset first)"
            )
            return
        try:
            bundle = discovery_bundle_card(self._data_dir, self._run_id)
            self._exp_bundle_var.set(str(bundle.get("card_text") or ""))
            self._experiments_cache = list_experiments(
                self._data_dir, self._run_id
            )
        except Exception as exc:
            self._exp_summary_var.set(f"Failed: {exc}")
            self._exp_bundle_var.set(f"Discovery Bundle\n  Failed: {exc}")
            return
        self._exp_summary_var.set(
            f"{len(self._experiments_cache)} experiment(s) · "
            "Created→Training→Model Produced→Validated→Champion"
        )
        for e in self._experiments_cache:
            reps = ", ".join(
                f"{r.get('family_label') or r.get('family_id')}→"
                f"{r.get('representative')}"
                for r in (e.get("family_reps") or [])[:3]
            )
            hold = e.get("holdout_score")
            wf = e.get("walk_forward_score")
            device = (
                e.get("device_label")
                or e.get("train_device")
                or "—"
            )
            self._exp_tree.insert(
                "",
                "end",
                iid=str(e.get("experiment_id")),
                values=(
                    e.get("experiment_id"),
                    e.get("name") or "—",
                    e.get("status_display") or e.get("status") or "Created",
                    device,
                    reps or "—",
                    f"{float(hold):.4f}" if hold is not None else "—",
                    f"{float(wf):.4f}" if wf is not None else "—",
                    e.get("validation_label") or "—",
                ),
            )

    def _on_experiment_selected(self, _event: Any = None) -> None:
        if not hasattr(self, "_exp_hypothesis"):
            return
        sel = self._exp_tree.selection() if hasattr(self, "_exp_tree") else ()
        self._exp_hypothesis.configure(state="normal")
        self._exp_hypothesis.delete("1.0", "end")
        if not sel:
            self._exp_hypothesis.insert(
                "end",
                "Select an experiment to view its Experiment Details.",
            )
            self._exp_hypothesis.configure(state="disabled")
            return
        eid = str(sel[0])
        text = ""
        try:
            from chain_replay_ml.dataset_builder.analysis_experiments import (
                load_experiment,
            )

            loaded = load_experiment(self._data_dir, eid, verify_bundle=False)
            if loaded:
                text = (
                    loaded.get("details_text")
                    or loaded.get("hypothesis_text")
                    or ""
                )
        except Exception:
            text = ""
        if not text:
            exp = next(
                (
                    e
                    for e in self._experiments_cache
                    if str(e.get("experiment_id")) == eid
                ),
                None,
            )
            text = (exp or {}).get("hypothesis_text") or f"Experiment {eid}"
        self._exp_hypothesis.insert("end", text)
        self._exp_hypothesis.configure(state="disabled")

    def _train_selected_experiment(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            request_train_experiment,
        )

        if not hasattr(self, "_exp_tree"):
            return
        sel = self._exp_tree.selection()
        if not sel:
            messagebox.showinfo(
                "Train",
                "Select a Created experiment (Feature-set Snapshot) first.",
                parent=self,
            )
            return
        eid = str(sel[0])
        self._detail_var.set(
            f"{eid} Training… "
            "(Snapshot→Model→Holdout→Walk-fwd→SHAP→Validation)"
        )
        self.update_idletasks()
        try:
            out = request_train_experiment(
                self._data_dir,
                eid,
                target=str(self._mi_target.get() or ""),
            )
        except Exception as exc:
            messagebox.showerror("Train", str(exc), parent=self)
            self._reload_experiments_tab()
            return
        self._detail_var.set(str(out.get("message") or "").split("\n")[0])
        card = str(out.get("result_card") or out.get("message") or "")
        messagebox.showinfo("Train → Validated", card, parent=self)
        self._reload_experiments_tab()
        self._reload_experiment_compare_tab()
        self._reload_discovery_banner()
        try:
            if eid in self._exp_tree.get_children(""):
                self._exp_tree.selection_set(eid)
                self._on_experiment_selected()
        except Exception:
            pass

    def _promote_champion(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            promote_champion,
        )

        eid = ""
        if hasattr(self, "_exp_tree"):
            sel = self._exp_tree.selection()
            if sel:
                eid = str(sel[0])
        if not eid:
            eid = str(self._auto_resume_id or "")
        if not eid:
            champ = next(
                (
                    e
                    for e in self._experiments_cache
                    if int(e.get("is_champion") or 0)
                ),
                None,
            )
            eid = str(champ["experiment_id"]) if champ else ""
        if not eid:
            messagebox.showinfo(
                "Promote Model",
                "No champion to promote — run Auto Research first, "
                "or select a Validated experiment.",
                parent=self,
            )
            return
        try:
            out = promote_champion(self._data_dir, eid)
        except Exception as exc:
            messagebox.showerror("Promote Model", str(exc), parent=self)
            return
        self._detail_var.set(str(out.get("message") or ""))
        messagebox.showinfo(
            "Promote Model", str(out.get("message") or ""), parent=self
        )
        self._reload_experiments_tab()
        self._reload_experiment_compare_tab()
        self._reload_champion_tab()
        self._reload_discovery_banner()

    def _delete_selected_experiment(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            STATUS_CREATED,
            delete_experiment,
        )

        if not hasattr(self, "_exp_tree"):
            return
        sel = self._exp_tree.selection()
        if not sel:
            messagebox.showinfo(
                "Delete",
                "Select a Created (not trained) experiment to delete.",
                parent=self,
            )
            return
        eid = str(sel[0])
        exp = next(
            (
                e
                for e in self._experiments_cache
                if str(e.get("experiment_id")) == eid
            ),
            None,
        )
        st = str((exp or {}).get("status_display") or (exp or {}).get("status") or "")
        if st != STATUS_CREATED:
            messagebox.showwarning(
                "Delete",
                f"{eid} is {st or 'not Created'} — only not-executed "
                "snapshots can be deleted.",
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "Delete Experiment",
            f"Delete {eid}? This only removes the Created snapshot "
            "(no model was trained).",
            parent=self,
        ):
            return
        try:
            out = delete_experiment(self._data_dir, eid)
        except Exception as exc:
            messagebox.showerror("Delete", str(exc), parent=self)
            return
        self._detail_var.set(str(out.get("message") or f"Deleted {eid}"))
        self._reload_experiments_tab()
        self._reload_experiment_compare_tab()
        self._reload_discovery_banner()

    def _reload_experiment_compare_tab(
        self,
        *,
        highlight_experiment_id: str | None = None,
        recommendation_text: str | None = None,
    ) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            explain_recommendation,
        )
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            compare_experiments,
            list_experiments,
        )
        from chain_replay_ml.dataset_builder.analysis_hca import load_families

        if not hasattr(self, "_cmp_tree"):
            return
        self._cmp_tree.delete(*self._cmp_tree.get_children())
        if not self._run_id:
            return
        try:
            families = load_families(self._data_dir, self._run_id, min_size=2)
        except Exception:
            families = []
        choices = [
            f"{f.get('family_label') or f.get('family_id')}|{f.get('family_id')}"
            for f in families
        ]
        labels = [c.split("|", 1)[0] for c in choices]
        if hasattr(self, "_exp_focus_combo"):
            self._exp_focus_combo["values"] = labels
            self._exp_focus_map = {
                c.split("|", 1)[0]: c.split("|", 1)[1] for c in choices
            }
            cur = str(self._exp_focus_family.get() or "")
            if labels and cur not in labels:
                self._exp_focus_family.set(labels[0])
        focus_id = None
        if hasattr(self, "_exp_focus_map"):
            focus_id = self._exp_focus_map.get(
                str(self._exp_focus_family.get() or "")
            )
        try:
            rows = compare_experiments(
                self._data_dir, self._run_id, focus_family_id=focus_id
            )
        except Exception as exc:
            self._detail_var.set(f"Comparison failed: {exc}")
            return
        highlight = str(highlight_experiment_id or "").strip()
        for r in rows:
            eid = str(r.get("experiment_id") or "")
            is_rec = bool(highlight and eid == highlight)
            if is_rec:
                champ_mark = "★ RECOMMENDED"
            elif int(r.get("is_champion") or 0):
                champ_mark = "★"
            else:
                champ_mark = ""
            self._cmp_tree.insert(
                "",
                "end",
                iid=eid or None,
                values=(
                    eid,
                    r.get("changed_family") or "—",
                    r.get("old_rep") or "—",
                    r.get("new_rep") or "—",
                    r.get("families_changed")
                    if r.get("families_changed") is not None
                    else "—",
                    r.get("device_label") or r.get("train_device") or "—",
                    r.get("holdout"),
                    r.get("walk_forward"),
                    r.get("delta_vs_baseline_txt") or "—",
                    r.get("validation_label"),
                    champ_mark,
                ),
            )
        if highlight:
            try:
                if highlight in self._cmp_tree.get_children(""):
                    self._cmp_tree.selection_set(highlight)
                    self._cmp_tree.see(highlight)
                    self._cmp_tree.focus(highlight)
            except Exception:
                pass
        try:
            self._experiments_cache = list_experiments(
                self._data_dir, self._run_id
            )
        except Exception:
            self._experiments_cache = []
        if hasattr(self, "_cmp_recommend_var") or hasattr(
            self, "_cmp_recommend_text"
        ):
            if recommendation_text:
                self._set_compare_recommendation_text(recommendation_text)
            else:
                try:
                    expl = explain_recommendation(
                        self._experiments_cache,
                        recommended_id=highlight or None,
                    )
                    self._set_compare_recommendation_text(
                        str(expl.get("text") or "No recommendation yet.")
                    )
                except Exception as exc:
                    self._set_compare_recommendation_text(
                        f"Recommendation unavailable: {exc}"
                    )

    def _on_research_level_changed(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            LEVEL_TO_STRATEGY,
        )

        if bool(self._research_advanced.get()):
            return
        level = str(self._research_level.get() or "")
        strat = LEVEL_TO_STRATEGY.get(level)
        if strat:
            self._research_strategy.set(strat)

    def _on_research_strategy_changed(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            STRATEGY_TO_LEVEL,
        )

        if not bool(self._research_advanced.get()):
            return
        strat = str(self._research_strategy.get() or "")
        level = STRATEGY_TO_LEVEL.get(strat)
        if level:
            self._research_level.set(level)

    def _on_research_advanced_toggled(self) -> None:
        show = bool(self._research_advanced.get())
        if not hasattr(self, "_research_advanced_fr"):
            return
        if show:
            self._research_advanced_fr.pack(fill="x", pady=(4, 0))
            self._on_research_strategy_changed()
        else:
            self._research_advanced_fr.pack_forget()
            self._on_research_level_changed()

    def _resolved_research_strategy(self) -> tuple[str, str]:
        """Return (strategy_id, user-facing level/strategy label)."""
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            LEVEL_LABELS,
            STRATEGY_LABELS,
            STRATEGY_TO_LEVEL,
            resolve_research_strategy,
        )

        advanced = bool(self._research_advanced.get())
        strategy = resolve_research_strategy(
            level=str(self._research_level.get() or ""),
            strategy=str(self._research_strategy.get() or ""),
            advanced=advanced,
        )
        if advanced:
            label = STRATEGY_LABELS.get(strategy, strategy)
        else:
            level = STRATEGY_TO_LEVEL.get(strategy) or str(
                self._research_level.get() or ""
            )
            label = LEVEL_LABELS.get(level, level)
        return strategy, label

    def _auto_create_and_train(self) -> None:
        self._start_auto_research(resume_from=None)

    def _continue_auto_research(self) -> None:
        resume = self._auto_resume_id
        if not resume:
            sel = ()
            if hasattr(self, "_exp_tree"):
                sel = self._exp_tree.selection()
            if sel:
                resume = str(sel[0])
            else:
                champ = next(
                    (
                        e
                        for e in self._experiments_cache
                        if int(e.get("is_champion") or 0)
                    ),
                    None,
                )
                resume = str(champ["experiment_id"]) if champ else None
        if not resume:
            messagebox.showinfo(
                "Continue Auto Research",
                "No current best to resume from. Run Auto Research first, "
                "or select an experiment.",
                parent=self,
            )
            return
        self._start_auto_research(resume_from=resume)

    def _start_auto_research(self, *, resume_from: str | None) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            CONTINUABLE_STRATEGIES,
            DEFAULT_MAX_ROUNDS,
            DEFAULT_PER_ROUND,
            LEVEL_DEEP,
            LEVEL_LABELS,
            PER_ROUND_CHOICES,
            STRATEGY_HILL,
            STRATEGY_SINGLE_SWAP,
            STRATEGY_TO_LEVEL,
            auto_create_and_train,
        )

        if not self._run_id:
            messagebox.showinfo(
                "Auto Research",
                "Load an analysis dataset first.",
                parent=self,
            )
            return
        if self._busy:
            messagebox.showinfo(
                "Auto Research",
                "Another analysis job is running.",
                parent=self,
            )
            return
        strategy, level_label = self._resolved_research_strategy()
        if resume_from and strategy not in CONTINUABLE_STRATEGIES:
            # Continue needs an iterative depth
            strategy = STRATEGY_HILL
            self._research_strategy.set(STRATEGY_HILL)
            self._research_level.set(LEVEL_DEEP)
            level_label = LEVEL_LABELS[LEVEL_DEEP]
        try:
            per_round = int(str(self._research_per_round.get() or "").strip())
        except ValueError:
            per_round = DEFAULT_PER_ROUND
        if per_round not in PER_ROUND_CHOICES:
            per_round = DEFAULT_PER_ROUND
            self._research_per_round.set(str(per_round))
        title = "Continue Auto Research" if resume_from else "Auto Research"
        if resume_from:
            body = (
                f"Continue improving from {resume_from}.\n"
                f"Research Level: {level_label}\n"
                f"Up to {per_round} neighbours/iteration × "
                f"{DEFAULT_MAX_ROUNDS} iterations.\n"
                "Stops when improvement falls below threshold.\n\nContinue?"
            )
        elif strategy == STRATEGY_SINGLE_SWAP:
            body = (
                f"Research Level: {level_label}\n"
                f"* Fast pass — up to {per_round} neighbours vs baseline\n"
                "* Pick the best (no stacking)\n"
                "* Best when you want a quick champion check\n\nContinue?"
            )
        elif strategy == STRATEGY_HILL:
            body = (
                f"Research Level: {level_label}\n"
                f"* Thorough search — up to {per_round} neighbours per step\n"
                "* Move to the best improving neighbour, then repeat\n"
                f"* Up to {DEFAULT_MAX_ROUNDS} iterations until converged\n\n"
                "Continue?"
            )
        else:
            body = (
                f"Research Level: {level_label}\n"
                f"* Balanced search — up to {per_round} neighbours/iteration\n"
                f"* Up to {DEFAULT_MAX_ROUNDS} iterations; accepted swaps stack\n"
                "* Stops when an iteration has no meaningful gain\n\nContinue?"
            )
        if not messagebox.askyesno(title, body, parent=self):
            return

        self._busy = True
        try:
            self._cancel_btn.configure(state="normal")
        except Exception:
            pass
        self._perm_progress.set(0.0)
        self._perm_progress_text.set("Auto Research starting...")
        self._detail_var.set("Auto Research starting...")
        self._select_experiments_subtab("Auto Research")
        self.update_idletasks()
        target = str(self._mi_target.get() or "")
        data_dir = self._data_dir
        run_id = self._run_id
        # Keep level in sync for display in Search State
        level = STRATEGY_TO_LEVEL.get(strategy)
        if level and not bool(self._research_advanced.get()):
            self._research_level.set(level)

        def _worker() -> None:
            def on_progress(
                frac: float, msg: str, dashboard: dict | None = None
            ) -> None:
                self.after(
                    0,
                    lambda f=frac, m=msg, d=dashboard: self._on_auto_train_progress(
                        f, m, d
                    ),
                )

            try:
                out = auto_create_and_train(
                    data_dir,
                    run_id,
                    strategy=strategy,
                    max_variants=per_round,
                    max_rounds=DEFAULT_MAX_ROUNDS,
                    target=target,
                    promote=True,
                    resume_from_experiment_id=resume_from,
                    on_progress=on_progress,
                )
                self.after(0, lambda: self._on_auto_train_done(out))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_auto_train_failed(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_auto_train_progress(
        self,
        frac: float,
        msg: str,
        dashboard: dict | None = None,
    ) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiment_auto import (
            format_dashboard,
        )

        try:
            self._perm_progress.set(max(0.0, min(100.0, float(frac) * 100.0)))
        except Exception:
            pass
        self._perm_progress_text.set(str(msg)[:120])
        self._detail_var.set(str(msg))
        if dashboard:
            self._set_auto_dashboard_text(format_dashboard(dashboard))
            self._select_experiments_subtab("Auto Research")

    def _on_auto_train_done(self, out: dict[str, Any]) -> None:
        self._busy = False
        try:
            self._cancel_btn.configure(state="disabled")
        except Exception:
            pass
        try:
            self._perm_progress.set(100.0)
        except Exception:
            pass
        champ = str(out.get("recommended_champion_id") or "")
        self._auto_resume_id = champ or out.get("current_winner_id")
        self._perm_progress_text.set(
            f"Auto Research done · champion {champ or '-'}"
        )
        self._detail_var.set(str(out.get("message") or "Auto Research done"))
        if out.get("dashboard_text"):
            self._set_auto_dashboard_text(str(out.get("dashboard_text")))
        elif out.get("research_statistics_text"):
            self._set_auto_dashboard_text(
                str(out.get("research_statistics_text"))
            )
        if out.get("champion_card"):
            self._set_champion_card_text(str(out.get("champion_card")))
        else:
            self._reload_champion_tab()
        self._reload_experiments_tab()
        self._reload_discovery_banner()
        rec = out.get("recommendation") or {}
        rec_text = str(rec.get("text") or out.get("message") or "")
        if out.get("dashboard_text"):
            rec_text = str(out.get("dashboard_text")) + "\n\n" + rec_text
        self._reload_experiment_compare_tab(
            highlight_experiment_id=champ or None,
            recommendation_text=rec_text,
        )
        # Destination: Champion page (not Comparison)
        self._select_research_tab("Champion")
        errs = list(out.get("train_errors") or []) + list(
            out.get("create_errors") or []
        )
        msg = str(out.get("message") or "")
        card = str(out.get("champion_card") or "")
        if card:
            msg = msg + "\n\n" + card
        elif rec_text:
            msg = msg + "\n\n" + rec_text
        if errs:
            msg = msg + "\n\nIssues:\n" + "\n".join(errs[:8])
        messagebox.showinfo("Auto Research", msg, parent=self)

    def _on_auto_train_failed(self, exc: BaseException) -> None:
        self._busy = False
        try:
            self._cancel_btn.configure(state="disabled")
        except Exception:
            pass
        self._perm_progress_text.set("Auto Research failed")
        self._detail_var.set(f"Auto Research failed: {exc}")
        self._reload_experiments_tab()
        messagebox.showerror("Auto Research", str(exc), parent=self)

    def _clone_experiment_variant(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_experiments import (
            clone_experiment_variant,
            experiment_family_options,
        )

        if not hasattr(self, "_exp_tree"):
            return
        sel = self._exp_tree.selection()
        if not sel:
            messagebox.showinfo(
                "Clone Variant",
                "Select a parent experiment first.",
                parent=self,
            )
            return
        eid = str(sel[0])
        try:
            options = experiment_family_options(self._data_dir, eid)
        except Exception as exc:
            messagebox.showerror("Clone Variant", str(exc), parent=self)
            return
        changeable = [o for o in options if o.get("can_change")]
        if not changeable:
            messagebox.showinfo(
                "Clone Variant",
                f"{eid} has no family with an alternate member — "
                "cannot create a differing variant.",
                parent=self,
            )
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Clone Variant of {eid}")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        ttk.Label(
            dlg,
            text=(
                f"Parent: {eid}\n"
                "Select one or more HCA families, then pick a different "
                "representative for each. Identical snapshots are rejected."
            ),
            justify="left",
            wraplength=520,
        ).pack(anchor="w", padx=10, pady=(10, 6))

        body = ttk.Frame(dlg, padding=8)
        body.pack(fill="both", expand=True)
        left = ttk.LabelFrame(body, text="Families to modify", padding=6)
        left.pack(side="left", fill="both", expand=True)
        fam_list = tk.Listbox(
            left, selectmode="extended", height=14, exportselection=False
        )
        fam_list.pack(side="left", fill="both", expand=True)
        ysb = ttk.Scrollbar(left, orient="vertical", command=fam_list.yview)
        fam_list.configure(yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y")

        idx_to_opt: dict[int, dict[str, Any]] = {}
        for i, opt in enumerate(changeable):
            label = (
                f"{opt.get('family_label') or opt.get('family_id')}  "
                f"[{opt.get('family_id')}]  "
                f"now={opt.get('current_representative')}"
            )
            fam_list.insert("end", label)
            idx_to_opt[i] = opt

        right = ttk.LabelFrame(body, text="New representatives", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        rep_vars: dict[str, tk.StringVar] = {}
        rep_combos: dict[str, ttk.Combobox] = {}
        rows_fr = ttk.Frame(right)
        rows_fr.pack(fill="both", expand=True)

        def _rebuild_rep_rows(_event: Any = None) -> None:
            for child in rows_fr.winfo_children():
                child.destroy()
            rep_vars.clear()
            rep_combos.clear()
            selected = [idx_to_opt[i] for i in fam_list.curselection()]
            if not selected:
                ttk.Label(
                    rows_fr,
                    text="Select one or more families on the left.",
                    foreground="#666",
                ).pack(anchor="w")
                return
            for opt in selected:
                fid = str(opt["family_id"])
                row = ttk.Frame(rows_fr)
                row.pack(fill="x", pady=3)
                ttk.Label(
                    row,
                    text=str(opt.get("family_label") or fid),
                    width=22,
                    anchor="w",
                ).pack(side="left")
                alts = list(opt.get("alternate_members") or [])
                var = tk.StringVar(value=alts[0] if alts else "")
                rep_vars[fid] = var
                cb = ttk.Combobox(
                    row,
                    textvariable=var,
                    values=alts,
                    state="readonly",
                    width=28,
                )
                cb.pack(side="left", padx=(4, 0))
                rep_combos[fid] = cb
                ttk.Label(
                    row,
                    text=f"was {opt.get('current_representative')}",
                    foreground="#777",
                ).pack(side="left", padx=(8, 0))

        fam_list.bind("<<ListboxSelect>>", _rebuild_rep_rows)
        _rebuild_rep_rows()

        name_var = tk.StringVar(value=f"{eid}-var")
        name_row = ttk.Frame(dlg, padding=(10, 0))
        name_row.pack(fill="x")
        ttk.Label(name_row, text="Name").pack(side="left")
        ttk.Entry(name_row, textvariable=name_var, width=24).pack(
            side="left", padx=(6, 0)
        )

        btns = ttk.Frame(dlg, padding=10)
        btns.pack(fill="x")

        def _create() -> None:
            selected = [idx_to_opt[i] for i in fam_list.curselection()]
            if not selected:
                messagebox.showwarning(
                    "Clone Variant",
                    "Select at least one HCA family to modify.",
                    parent=dlg,
                )
                return
            changes: dict[str, str] = {}
            for opt in selected:
                fid = str(opt["family_id"])
                new_rep = str(rep_vars.get(fid, tk.StringVar()).get() or "").strip()
                if not new_rep:
                    messagebox.showwarning(
                        "Clone Variant",
                        f"Choose a new representative for "
                        f"{opt.get('family_label') or fid}.",
                        parent=dlg,
                    )
                    return
                if new_rep == str(opt.get("current_representative") or ""):
                    messagebox.showwarning(
                        "Clone Variant",
                        f"{opt.get('family_label') or fid}: pick a different "
                        "representative than the parent.",
                        parent=dlg,
                    )
                    return
                changes[fid] = new_rep
            try:
                out = clone_experiment_variant(
                    self._data_dir,
                    eid,
                    changes=changes,
                    name=str(name_var.get() or "").strip(),
                )
            except Exception as exc:
                messagebox.showerror("Clone Variant", str(exc), parent=dlg)
                return
            dlg.destroy()
            self._detail_var.set(
                f"Created variant {out.get('experiment_id')} of {eid}"
            )
            self._reload_experiments_tab()
            self._reload_experiment_compare_tab()
            self._reload_discovery_banner()
            try:
                new_id = str(out.get("experiment_id") or "")
                if new_id and new_id in self._exp_tree.get_children(""):
                    self._exp_tree.selection_set(new_id)
                    self._on_experiment_selected()
            except Exception:
                pass

        ttk.Button(btns, text="Create Variant", command=_create).pack(side="left")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(
            side="left", padx=(8, 0)
        )
        dlg.geometry("720x420")
        dlg.focus_set()

    def _build_validation_scorecard_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Model Validation Scorecard — how the trained production model "
                "uses features (SHAP). Holdout / walk-forward / confidence come "
                "from the model package when available."
            ),
            foreground="#555",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))
        cols = (
            "feature",
            "shap",
            "shap_pct",
            "holdout",
            "walk_forward",
            "confidence",
            "note",
        )
        self._val_tree = ttk.Treeview(parent, columns=cols, show="headings", height=16)
        headings = {
            "feature": ("Feature", 220),
            "shap": ("SHAP", 110),
            "shap_pct": ("SHAP %ile", 80),
            "holdout": ("Holdout", 90),
            "walk_forward": ("Walk-forward", 100),
            "confidence": ("Prod confidence", 110),
            "note": ("Note", 220),
        }
        for key, (title, width) in headings.items():
            self._val_tree.heading(key, text=title)
            self._val_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self._val_tree.yview)
        self._val_tree.configure(yscrollcommand=ysb.set)
        self._val_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._val_tree.bind("<Double-1>", self._on_validation_double_click)

    def _build_shap_explanation_tab(self, parent: ttk.Frame) -> None:
        hdr = ttk.Frame(parent)
        hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(
            hdr,
            textvariable=self._shap_stage_var,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        ttk.Button(hdr, text="Refresh", command=self._reload_shap_tab).pack(
            side="right"
        )
        warn = ttk.Label(
            parent,
            textvariable=self._shap_warn_var,
            foreground="#8a4b08",
            wraplength=760,
            justify="left",
        )
        warn.pack(anchor="w", pady=(0, 6))
        ttk.Label(
            parent,
            text=(
                "Model Explanation (SHAP) belongs to Stage 2 — Model Validation. "
                "Use it after selecting a candidate feature set and retraining. "
                "Do not use SHAP as the primary Feature Discovery tool."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        cols = ("rank", "feature", "importance", "percentile")
        self._shap_tree = ttk.Treeview(
            parent, columns=cols, show="headings", height=14
        )
        headings = {
            "rank": ("Rank", 50),
            "feature": ("Feature", 280),
            "importance": ("mean |SHAP|", 110),
            "percentile": ("Percentile", 90),
        }
        for key, (title, width) in headings.items():
            self._shap_tree.heading(key, text=title)
            self._shap_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self._shap_tree.yview)
        self._shap_tree.configure(yscrollcommand=ysb.set)
        self._shap_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._shap_tree.bind("<Double-1>", self._on_shap_double_click)

    def _build_recommendations_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            bar,
            text=(
                "Overall Discovery ratings from Correlation + MI + Permutation "
                "(no SHAP). Ambiguous families go to Family Review."
            ),
            foreground="#555",
            wraplength=640,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(bar, text="Filter").pack(side="left", padx=(8, 4))
        filt = ttk.Combobox(
            bar,
            textvariable=self._rec_filter,
            state="readonly",
            width=14,
            values=("ALL", "KEEP", "REVIEW", "MERGE", "RETIRE"),
        )
        filt.pack(side="left")
        filt.bind("<<ComboboxSelected>>", lambda _e: self._reload_recommendations())

        cols = ("feature", "score", "recommendation", "confidence", "reason")
        self._rec_tree = ttk.Treeview(parent, columns=cols, show="headings", height=16)
        headings = {
            "feature": ("Feature", 220),
            "score": ("Score", 140),
            "recommendation": ("Recommendation", 130),
            "confidence": ("Confidence", 80),
            "reason": ("Reason", 420),
        }
        for key, (title, width) in headings.items():
            self._rec_tree.heading(key, text=title)
            self._rec_tree.column(key, width=width, anchor="w")
        ysb = ttk.Scrollbar(parent, orient="vertical", command=self._rec_tree.yview)
        self._rec_tree.configure(yscrollcommand=ysb.set)
        self._rec_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self._rec_tree.bind("<Double-1>", self._on_recommendation_double_click)

    def _ensure_profiles_loaded(self, *, async_build: bool = False) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
            list_profile_features,
            profiles_exist,
        )

        if not self._run_id:
            return
        if profiles_exist(self._data_dir, self._run_id):
            self._all_profile_features = list_profile_features(
                self._data_dir, self._run_id
            )
            self._filter_feature_list()
            self._reload_scorecard()
            self._reload_recommendations()
            self._reload_validation_scorecard()
            self._reload_shap_tab()
            self._reload_families_tab()
            self._reload_family_review_tab()
            self._reload_experiments_tab()
            self._reload_experiment_compare_tab()
            return
        if async_build:
            self._build_profiles_now()

    def _build_profiles_now(self) -> None:
        if self._busy:
            messagebox.showinfo(
                "Feature Profiles",
                "Another job is running. Wait for it to finish.",
                parent=self,
            )
            return
        ds = self._selected_dataset()
        if not ds or not self._run_id:
            messagebox.showinfo(
                "Feature Profiles",
                "Select an analysis dataset first.",
                parent=self,
            )
            return
        self._busy = True
        self._detail_var.set("Building feature profiles (read-only from parquet + analysis.db)…")

        def _worker() -> None:
            from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
                build_feature_profiles,
            )

            err = ""
            try:
                summary = build_feature_profiles(
                    self._data_dir, self._run_id or "", ds
                )
                msg = f"Feature profiles ready · {summary.get('features'):,} features"
            except Exception as exc:
                err = str(exc)
                msg = f"Feature profile build failed: {exc}"

            def _done() -> None:
                self._busy = False
                self._detail_var.set(msg)
                if not err:
                    self._ensure_profiles_loaded(async_build=False)
                    # Jump to explorer for convenience
                    try:
                        self._select_research_tab("Feature Explorer")
                    except Exception:
                        pass
                else:
                    messagebox.showerror("Feature Profiles", err, parent=self)

            self.after(0, _done)

        threading.Thread(
            target=_worker, name="feature-profiles-build", daemon=True
        ).start()

    def _filter_feature_list(self) -> None:
        if not hasattr(self, "_feature_list"):
            return
        needle = self._feature_search.get().strip().lower()
        self._feature_list.delete(0, "end")
        for name in self._all_profile_features:
            if needle and needle not in name.lower():
                continue
            self._feature_list.insert("end", name)

    def _on_feature_selected(self, _event: Any = None) -> None:
        if not hasattr(self, "_feature_list"):
            return
        sel = self._feature_list.curselection()
        if not sel:
            return
        name = self._feature_list.get(sel[0])
        self._show_feature_profile(str(name))

    def _on_scorecard_double_click(self, _event: Any = None) -> None:
        sel = self._score_tree.selection()
        if not sel:
            return
        vals = self._score_tree.item(sel[0], "values")
        if not vals:
            return
        self._open_feature_in_explorer(str(vals[0] or ""))

    def _on_scorecard_activate(self, _event: Any = None) -> None:
        """Double-click: REVIEW FAMILY → Family Review; else Explorer."""
        sel = self._score_tree.selection()
        if not sel:
            return
        vals = self._score_tree.item(sel[0], "values")
        if not vals:
            return
        feat = str(vals[0] or "")
        rec = str(vals[4] or "") if len(vals) > 4 else ""
        if "REVIEW" in rec.upper() or "FAMILY" in rec.upper():
            self._open_scorecard_family()
        else:
            self._open_feature_in_explorer(feat)

    def _on_scorecard_select(self, _event: Any = None) -> None:
        if not hasattr(self, "_score_family_ctx"):
            return
        sel = self._score_tree.selection()
        text = "Select a feature to see its HCA family context."
        if sel and self._run_id:
            vals = self._score_tree.item(sel[0], "values")
            feat = str(vals[0] or "") if vals else ""
            if feat:
                try:
                    from chain_replay_ml.dataset_builder.analysis_family_review import (
                        format_family_context_text,
                    )

                    text = format_family_context_text(
                        self._data_dir, self._run_id, feat
                    )
                except Exception as exc:
                    text = f"Could not load family context: {exc}"
        self._score_family_ctx.configure(state="normal")
        self._score_family_ctx.delete("1.0", "end")
        self._score_family_ctx.insert("end", text)
        self._score_family_ctx.configure(state="disabled")

    def _open_scorecard_family(self) -> None:
        """Jump to Family Review for the selected scorecard feature's HCA family."""
        if not hasattr(self, "_score_tree") or not self._run_id:
            return
        sel = self._score_tree.selection()
        if not sel:
            messagebox.showinfo(
                "Open Family",
                "Select a Discovery Scorecard feature first.",
                parent=self,
            )
            return
        vals = self._score_tree.item(sel[0], "values")
        feat = str(vals[0] or "") if vals else ""
        if not feat:
            return
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            FILTER_ALL,
            FILTER_NEEDS_REVIEW,
            lookup_family_for_feature,
        )

        fam = lookup_family_for_feature(self._data_dir, self._run_id, feat)
        if not fam:
            messagebox.showinfo(
                "Open Family",
                f"No HCA family for {feat}. Run HCA after Correlation.",
                parent=self,
            )
            return
        fid = str(fam.get("family_id") or "")
        status = str(fam.get("review_status") or "")
        # Show Needs Review queue if ambiguous; else All so Auto Accepted is visible
        from chain_replay_ml.dataset_builder.analysis_family_review import (
            NEEDS_REVIEW_STATUSES,
        )

        if status in NEEDS_REVIEW_STATUSES:
            self._review_filter.set(FILTER_NEEDS_REVIEW)
        else:
            self._review_filter.set(FILTER_ALL)
        self._select_research_tab("Family Review")
        self._reload_family_review_tab()
        try:
            if fid in self._review_tree.get_children(""):
                self._review_tree.selection_set(fid)
                self._review_tree.see(fid)
                self._on_review_family_selected()
        except Exception:
            pass
        self._detail_var.set(
            f"Family Review · {fam.get('family_label') or fid} · from {feat}"
        )

    def _on_recommendation_double_click(self, _event: Any = None) -> None:
        sel = self._rec_tree.selection()
        if not sel:
            return
        vals = self._rec_tree.item(sel[0], "values")
        if not vals:
            return
        self._open_feature_in_explorer(str(vals[0] or ""))

    def _open_feature_in_explorer(self, feat: str) -> None:
        feat = str(feat or "").strip()
        if not feat:
            return
        try:
            self._select_research_tab("Feature Explorer")
        except Exception:
            pass
        try:
            names = list(self._feature_list.get(0, "end"))
            if feat in names:
                idx = names.index(feat)
                self._feature_list.selection_clear(0, "end")
                self._feature_list.selection_set(idx)
                self._feature_list.see(idx)
        except Exception:
            pass
        self._show_feature_profile(feat)

    def _on_validation_double_click(self, _event: Any = None) -> None:
        if not hasattr(self, "_val_tree"):
            return
        sel = self._val_tree.selection()
        if not sel:
            return
        vals = self._val_tree.item(sel[0], "values")
        if vals:
            self._open_feature_in_explorer(str(vals[0] or ""))

    def _on_shap_double_click(self, _event: Any = None) -> None:
        if not hasattr(self, "_shap_tree"):
            return
        sel = self._shap_tree.selection()
        if not sel:
            return
        vals = self._shap_tree.item(sel[0], "values")
        if vals and len(vals) > 1:
            self._open_feature_in_explorer(str(vals[1] or ""))

    def _show_feature_profile(self, feature_name: str) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
            load_feature_profile,
        )
        from chain_replay_ml.dataset_builder.analysis_feature_roles import (
            ROLE_PREDICTOR,
            classify_feature_role,
            role_banner,
        )
        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
            analysis_timeline,
            mi_stars,
        )

        self._profile_text.configure(state="normal")
        self._profile_text.delete("1.0", "end")
        if not self._run_id:
            self._profile_text.insert("end", "No analysis run selected.\n")
            self._profile_text.configure(state="disabled")
            return
        prof = load_feature_profile(self._data_dir, self._run_id, feature_name)
        if not prof:
            self._profile_text.insert(
                "end",
                f"No profile for {feature_name}.\n"
                "Click Build / Refresh Profiles.\n",
            )
            self._profile_text.configure(state="disabled")
            return

        def _pending(val: Any) -> str:
            return "Pending" if val is None else str(val)

        def _fmt(val: Any, digits: int = 4) -> str:
            if val is None:
                return "—"
            try:
                return f"{float(val):.{digits}f}"
            except (TypeError, ValueError):
                return str(val)

        role = str(prof.get("feature_role") or "").strip() or classify_feature_role(
            feature_name
        )
        banner = role_banner(role)

        self._profile_text.insert("end", "Feature Profile\n", "h")
        self._profile_text.insert("end", f"{feature_name}\n\n")

        if banner and role != ROLE_PREDICTOR:
            title, body = banner
            self._profile_text.insert("end", f"{title}\n", "h")
            self._profile_text.insert("end", f"{body}\n\n", "muted")
            # Still show identity + health for context, but skip scoring sections.
            parents = prof.get("parents") or []
            children = prof.get("children") or []
            self._profile_text.insert("end", "Identity\n", "h")
            self._profile_text.insert(
                "end",
                f"Feature Name      {feature_name}\n"
                f"Feature Role      {role}\n"
                f"Category          {prof.get('category') or '—'}\n"
                f"Source            {prof.get('source') or '—'}\n"
                f"Parent Features   {', '.join(parents) if parents else '—'}\n"
                f"Children Features {', '.join(children) if children else '—'}\n\n",
            )
            self._profile_text.insert("end", "Dataset Health\n", "h")
            self._profile_text.insert(
                "end",
                f"Coverage          {_fmt(prof.get('coverage'), 2)}%\n"
                f"NULL %            {_fmt(prof.get('null_pct'), 2)}%\n"
                f"Mean              {_fmt(prof.get('mean'))}\n"
                f"Std Dev           {_fmt(prof.get('std_dev'))}\n\n",
            )
            self._profile_text.insert(
                "end",
                "Excluded from Correlation, MI, SHAP, VIF, Permutation, and Scorecard.\n",
                "muted",
            )
            self._profile_text.configure(state="disabled")
            return

        parents = prof.get("parents") or []
        children = prof.get("children") or []
        top = prof.get("top_correlated") or []
        n_feat = max(
            len(
                [
                    n
                    for n in self._all_profile_features
                    if classify_feature_role(n) == ROLE_PREDICTOR
                ]
            ),
            1,
        )

        # Analysis Timeline
        self._profile_text.insert("end", "Analysis Timeline\n", "h")
        for step in analysis_timeline(
            self._data_dir,
            self._run_id,
            feature_name,
            mi_target=self._mi_target.get() or None,
        ):
            mark = "✓" if step["state"] == "done" else "⏳"
            self._profile_text.insert(
                "end", f"{mark} {step['label']}\n", "muted" if mark == "⏳" else ()
            )
        self._profile_text.insert("end", "\n")

        self._profile_text.insert("end", "Identity\n", "h")
        self._profile_text.insert(
            "end",
            f"Feature Name      {feature_name}\n"
            f"Feature Role      {role or ROLE_PREDICTOR}\n"
            f"Category          {prof.get('category') or '—'}\n"
            f"Source            {prof.get('source') or '—'}\n"
            f"Transformation    {prof.get('transformation') or '—'}\n"
            f"Parent Features   {', '.join(parents) if parents else '—'}\n"
            f"Children Features {', '.join(children) if children else '—'}\n\n",
        )

        self._profile_text.insert("end", "Dataset Health\n", "h")
        self._profile_text.insert(
            "end",
            f"Coverage          {_fmt(prof.get('coverage'), 2)}%\n"
            f"NULL %            {_fmt(prof.get('null_pct'), 2)}%\n"
            f"Warm-up %         {_fmt(prof.get('warmup_pct'), 2)}%\n"
            f"Unique Values     {prof.get('unique_values') if prof.get('unique_values') is not None else '—'}\n"
            f"Mean              {_fmt(prof.get('mean'))}\n"
            f"Std Dev           {_fmt(prof.get('std_dev'))}\n"
            f"Min               {_fmt(prof.get('min_val'))}\n"
            f"Max               {_fmt(prof.get('max_val'))}\n\n",
        )

        self._profile_text.insert("end", "Correlation\n", "h")
        self._profile_text.insert(
            "end",
            f"Correlation Family  {prof.get('cluster_id') or '—'}\n"
            f"Representative      {prof.get('representative') or '—'}\n"
            f"Cluster Members     {prof.get('cluster_members') or '—'}\n"
            "Highest Correlated Features\n",
        )
        if top:
            for item in top:
                self._profile_text.insert(
                    "end",
                    f"  {item.get('feature'):<40} {_fmt(item.get('correlation'))}\n",
                )
        else:
            self._profile_text.insert(
                "end", "  (none — run Correlation first)\n", "muted"
            )

        self._profile_text.insert("end", "\nMutual Information\n", "h")
        mi_score = prof.get("mi_score")
        if mi_score is None:
            self._profile_text.insert(
                "end",
                "Score       Pending\n"
                "Rank        Pending\n"
                "Percentile  Pending\n"
                "Target      —\n"
                "Interpretation  Pending\n\n",
                "muted",
            )
        else:
            rank = prof.get("mi_rank")
            pct = prof.get("mi_percentile")
            rank_s = (
                f"{int(rank)} / {n_feat}"
                if rank is not None
                else "—"
            )
            self._profile_text.insert(
                "end",
                f"Score           {_fmt(mi_score)}\n"
                f"Rank            {rank_s}\n"
                f"Percentile      {_fmt(pct, 0)}%\n"
                f"Target          {prof.get('mi_target') or self._mi_target.get() or '—'}\n"
                f"Interpretation  {prof.get('mi_interpretation') or '—'}\n"
                f"                {mi_stars(pct if pct is not None else None)}\n\n",
            )

        self._profile_text.insert("end", "Model Explanation (SHAP)\n", "h")
        self._profile_text.insert(
            "end",
            "Stage: Model Validation — not used for Feature Discovery recommendations.\n",
            "muted",
        )
        shap_imp = prof.get("shap_importance")
        if shap_imp is None:
            self._profile_text.insert(
                "end",
                "Score (mean |SHAP|)  Pending\n"
                "Rank                 Pending\n"
                "Percentile           Pending\n"
                "Model                —\n\n",
                "muted",
            )
        else:
            s_rank = prof.get("shap_rank")
            s_pct = prof.get("shap_percentile")
            # Rank denominator is model feature count when available
            shap_n = None
            try:
                from chain_replay_ml.dataset_builder.analysis_shap import (
                    load_shap_results,
                )

                mname = str(
                    prof.get("shap_model") or self._shap_model.get() or ""
                )
                if mname and self._run_id:
                    shap_n = len(
                        load_shap_results(self._data_dir, self._run_id, mname)
                    )
            except Exception:
                shap_n = None
            s_rank_s = (
                f"{int(s_rank)} / {shap_n or n_feat}"
                if s_rank is not None
                else "—"
            )
            self._profile_text.insert(
                "end",
                f"Score (mean |SHAP|)  {_fmt(shap_imp)}\n"
                f"Rank                 {s_rank_s}\n"
                f"Percentile           {_fmt(s_pct, 0)}%\n"
                f"Model                {prof.get('shap_model') or self._shap_model.get() or '—'}\n"
                f"                     {mi_stars(s_pct if s_pct is not None else None)}\n\n",
            )

        self._profile_text.insert("end", "Permutation Importance\n", "h")
        perm = prof.get("permutation_importance")
        if perm is None:
            self._profile_text.insert(
                "end",
                "Baseline RMSE     Pending\n"
                "Permuted RMSE     Pending\n"
                "Increase          Pending\n"
                "Rank              Pending\n"
                "Interpretation    Pending\n\n",
                "muted",
            )
        else:
            from chain_replay_ml.dataset_builder.analysis_permutation import (
                permutation_detail,
            )

            p_rank = prof.get("permutation_rank")
            p_pct = prof.get("permutation_percentile")
            base_rmse = prof.get("permutation_baseline_rmse")
            perm_rmse = prof.get("permutation_permuted_rmse")
            d_rmse = prof.get("permutation_delta_rmse")
            if d_rmse is None:
                d_rmse = perm
            task = "regression" if base_rmse is not None else "classification"
            p_rank_s = f"{int(p_rank)} / {n_feat}" if p_rank is not None else "—"
            self._profile_text.insert(
                "end",
                f"Baseline RMSE     {_fmt(base_rmse) if base_rmse is not None else '—'}\n"
                f"Permuted RMSE     {_fmt(perm_rmse) if perm_rmse is not None else '—'}\n"
                f"Increase          {('+' if float(d_rmse) >= 0 else '') + _fmt(d_rmse)}\n"
                f"Rank              {p_rank_s}\n"
                f"Percentile        {_fmt(p_pct, 0)}%\n"
                f"Interpretation    {prof.get('permutation_interpretation') or '—'}\n"
                f"                  {permutation_detail(float(d_rmse), task_type=task)}\n"
                f"                  {mi_stars(p_pct if p_pct is not None else None)}\n\n",
            )

        self._profile_text.insert("end", "Cluster\n", "h")
        self._profile_text.insert(
            "end",
            f"Cluster   {prof.get('cluster_id') or '—'}\n"
            f"Members   {prof.get('cluster_members') or '—'}\n\n",
        )

        self._profile_text.insert("end", "Research Metrics\n", "h")
        self._profile_text.insert(
            "end",
            f"Mutual Information   {_fmt(mi_score) if mi_score is not None else 'Pending'}\n"
            f"SHAP                 {_fmt(shap_imp) if shap_imp is not None else 'Pending'}\n"
            f"Permutation          {_fmt(perm) if perm is not None else 'Pending'}\n"
            f"VIF                  {_pending(prof.get('vif'))}\n\n",
            "muted",
        )

        self._profile_text.insert("end", "Discovery Rating\n", "h")
        rating_score = prof.get("feature_score")
        if rating_score is None:
            rating_score = prof.get("rating_score")
        if rating_score is None:
            self._profile_text.insert(
                "end",
                "Stage          Feature Discovery\n"
                "Score           Pending\n"
                "Stars           Pending\n"
                "Recommendation  Pending\n"
                "Confidence      Pending\n"
                "Reason          Run Discovery Rating after Correlation + MI + Permutation.\n"
                "                (SHAP is Stage 2 — Model Validation only.)\n\n",
                "muted",
            )
        else:
            from chain_replay_ml.dataset_builder.analysis_feature_rating import (
                rating_stars,
            )

            action = (
                prof.get("rating_action")
                or prof.get("recommendation")
                or "—"
            )
            stars = prof.get("rating_stars") or rating_stars(float(rating_score))
            reason = (
                prof.get("rating_reason")
                or prof.get("reason")
                or "—"
            )
            self._profile_text.insert(
                "end",
                f"Stage          Feature Discovery\n"
                f"Score           {float(rating_score):.0f}\n"
                f"Stars           {stars}\n"
                f"Recommendation  {action}\n"
                f"Confidence      {prof.get('rating_confidence') or '—'}\n"
                f"Reason\n{reason}\n\n",
            )

        self._profile_text.insert("end", "Validation Rating\n", "h")
        val_score = prof.get("validation_score")
        if val_score is None:
            self._profile_text.insert(
                "end",
                "Stage          Model Validation\n"
                "Score           Pending\n"
                "Recommendation  Pending\n"
                "Reason          After retrain: Model Explanation (SHAP) "
                "updates Validation Rating (does not change Discovery).\n\n",
                "muted",
            )
        else:
            self._profile_text.insert(
                "end",
                f"Stage          Model Validation\n"
                f"Score           {float(val_score):.0f}\n"
                f"Stars           {prof.get('validation_stars') or '—'}\n"
                f"Recommendation  {prof.get('validation_action') or '—'}\n"
                f"Confidence      {prof.get('validation_confidence') or '—'}\n"
                f"Reason\n{prof.get('validation_reason') or '—'}\n\n",
            )

        self._profile_text.insert("end", "Correlation Insight\n", "h")
        # Prefer MI interpretation when available, else correlation insight
        if prof.get("mi_interpretation") and not prof.get("rating_action"):
            self._profile_text.insert(
                "end",
                f"MI Interpretation  {prof.get('mi_interpretation')}\n"
                f"Correlation        {prof.get('recommendation') or 'Keep'}\n"
                f"Reason\n{prof.get('reason') or '—'}\n\n",
            )
        elif not prof.get("rating_action"):
            self._profile_text.insert(
                "end",
                f"Recommendation  {prof.get('recommendation') or 'Keep'}\n"
                f"Reason\n{prof.get('reason') or '—'}\n\n",
            )
        else:
            self._profile_text.insert(
                "end",
                f"Cluster insight kept separately from Overall Rating "
                f"(see Correlation Insights).\n\n",
                "muted",
            )
        self._profile_text.insert(
            "end",
            "Read-only profile. Never deletes features; "
            "correlation/MI alone never remove features automatically.\n",
            "muted",
        )
        self._profile_text.configure(state="disabled")

    def _reload_scorecard(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
            load_feature_scorecard,
        )
        from chain_replay_ml.dataset_builder.analysis_feature_rating import (
            format_score_cell,
        )
        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
            mi_stars,
        )

        if not hasattr(self, "_score_tree"):
            return
        self._score_tree.delete(*self._score_tree.get_children())
        if not self._run_id:
            return
        for r in load_feature_scorecard(self._data_dir, self._run_id):
            cov = r.get("coverage")
            cov_s = f"{float(cov):.1f}" if cov is not None else "—"
            mi = r.get("mi_score")
            pct = r.get("mi_percentile")
            if mi is None:
                mi_cell = "Pending"
            else:
                mi_cell = f"{float(mi):.3f} {mi_stars(pct)}"
            perm = r.get("permutation_importance")
            perm_pct = r.get("permutation_percentile")
            if perm is None:
                perm_cell = "Pending"
            else:
                sign = "+" if float(perm) >= 0 else ""
                perm_cell = f"{sign}{float(perm):.2f} {mi_stars(perm_pct)}"
            rec = (
                r.get("rating_action")
                or r.get("recommendation")
                or "Keep"
            )
            family = (
                r.get("rating_family_label")
                or r.get("rating_family_id")
                or r.get("cluster_id")
                or "—"
            )
            self._score_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature_name"),
                    r.get("category") or "—",
                    family,
                    cov_s,
                    rec,
                    mi_cell,
                    perm_cell,
                    format_score_cell(r),
                ),
            )

    def _reload_validation_scorecard(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
            load_feature_scorecard,
        )
        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
            mi_stars,
        )

        if not hasattr(self, "_val_tree"):
            return
        self._val_tree.delete(*self._val_tree.get_children())
        if not self._run_id:
            return
        for r in load_feature_scorecard(self._data_dir, self._run_id):
            shap = r.get("shap_importance")
            shap_pct = r.get("shap_percentile")
            if r.get("validation_shap_pct") is not None:
                shap_pct = r.get("validation_shap_pct")
            if shap is None:
                shap_cell = "Pending"
                pct_cell = "Pending"
                note = "Run Model Explanation (SHAP) after retrain"
            else:
                shap_cell = f"{float(shap):.4f} {mi_stars(shap_pct)}"
                pct_cell = (
                    f"{float(shap_pct):.0f}%" if shap_pct is not None else "—"
                )
                note = (
                    str(r.get("validation_action") or "")
                    or "Model Validation"
                )
            self._val_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature_name"),
                    shap_cell,
                    pct_cell,
                    "—",
                    "—",
                    r.get("validation_confidence") or "—",
                    note,
                ),
            )

    def _on_shap_model_changed(self) -> None:
        self._reload_shap_tab()
        self._reload_validation_scorecard()

    def _reload_shap_tab(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_shap import (
            assess_model_feature_selection,
            load_shap_results,
        )

        model = str(self._shap_model.get() or "").strip()
        if hasattr(self, "_shap_tree"):
            self._shap_tree.delete(*self._shap_tree.get_children())

        if not model:
            self._shap_stage_var.set("Stage: Model Validation")
            self._shap_warn_var.set("Select a trained model to explain.")
            return

        ds = self._selected_dataset()
        info = assess_model_feature_selection(
            self._data_dir,
            model,
            dataset=ds,
            run_id=self._run_id or "",
        )
        self._shap_stage_var.set(f"Stage: {info.get('stage') or 'Model Validation'}")
        warn = str(info.get("warning") or "")
        meta = (
            f"Model features: {info.get('model_feature_count')} · "
            f"Analysis predictors: {info.get('predictor_count')}"
        )
        self._shap_warn_var.set(
            (warn + "\n\n" + meta).strip() if warn else meta
        )

        if not self._run_id or not hasattr(self, "_shap_tree"):
            return
        try:
            rows = load_shap_results(self._data_dir, self._run_id, model, limit=300)
        except Exception:
            rows = []
        for r in rows:
            pct = r.get("percentile")
            self._shap_tree.insert(
                "",
                "end",
                values=(
                    r.get("rank"),
                    r.get("feature"),
                    f"{float(r.get('importance') or 0):.6f}",
                    f"{float(pct):.0f}%" if pct is not None else "—",
                ),
            )

    def _reload_recommendations(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_feature_rating import (
            format_score_cell,
            load_feature_ratings,
        )

        if not hasattr(self, "_rec_tree"):
            return
        self._rec_tree.delete(*self._rec_tree.get_children())
        if not self._run_id:
            return
        filt = str(self._rec_filter.get() or "ALL").strip().upper()
        rows = load_feature_ratings(
            self._data_dir,
            self._run_id,
            action_filter=None if filt == "ALL" else filt,
        )
        for r in rows:
            score_cell = format_score_cell(r)
            # Drop action suffix in Score column — shown separately
            score = r.get("feature_score")
            if score is None:
                score = r.get("rating_score")
            if score is not None:
                stars = r.get("rating_stars") or ""
                score_cell = f"{float(score):.0f} {stars}".strip()
            self._rec_tree.insert(
                "",
                "end",
                values=(
                    r.get("feature_name"),
                    score_cell,
                    r.get("rating_action") or r.get("recommendation") or "—",
                    r.get("rating_confidence") or "—",
                    r.get("rating_reason") or r.get("reason") or "—",
                ),
            )

    # --- module execution --------------------------------------------------

    # Analysis Status inputs: only show what checked modules need.
    # Dataset always comes from the Analysis Dataset picker.
    _MODULES_NEED_TARGET = frozenset({"mutual_information", "feature_scorecard"})
    _MODULES_NEED_MODEL = frozenset({"permutation", "shap", "feature_scorecard"})

    def _checked_module_ids(self) -> list[str]:
        return [m for m, v in self._module_vars.items() if v.get()]

    def _module_label(self, module_id: str) -> str:
        try:
            from chain_replay_ml.dataset_builder.analysis_lab_store import MODULE_LABELS

            return str(MODULE_LABELS.get(module_id) or module_id)
        except Exception:
            return str(module_id)

    def _publish_module_progress(
        self,
        module_id: str,
        info: dict[str, Any] | None = None,
        *,
        frac: float | None = None,
        message: str = "",
        elapsed: float | None = None,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        """Update status label, detail timeline, and progress bar for any module."""
        info = dict(info or {})
        if frac is not None:
            info["frac"] = frac
        if message:
            info["message"] = message
        if elapsed is not None:
            info["elapsed"] = elapsed
        if done is not None:
            info["done"] = done
        if total is not None:
            info["total"] = total

        done_i = int(info.get("done") or 0)
        total_i = int(info.get("total") or 0)
        elapsed_f = float(info.get("elapsed") or 0.0)
        msg = str(info.get("message") or "").strip()
        if "frac" in info and info.get("frac") is not None:
            pct = max(0.0, min(100.0, float(info["frac"]) * 100.0))
        elif total_i > 0:
            pct = max(0.0, min(100.0, 100.0 * done_i / total_i))
        else:
            pct = float(self._perm_progress.get() or 0.0)

        label = self._module_label(module_id)
        if total_i > 0:
            status = f"Running · {done_i}/{total_i}"
            bar_text = f"{done_i}/{total_i} · {pct:.0f}%"
        elif pct > 0:
            status = f"Running · {pct:.0f}%"
            bar_text = f"{pct:.0f}% · {elapsed_f:.1f}s"
        else:
            status = f"Running · {elapsed_f:.0f}s"
            bar_text = f"{elapsed_f:.1f}s"

        detail = f"{label} · {elapsed_f:.1f}s"
        if msg:
            detail = f"{detail} · {msg}"

        # Throttle Tk marshaling — flooding after(0) makes the UI feel frozen.
        now = time.perf_counter()
        last_at = float(self._progress_last_ui_at.get(module_id) or 0.0)
        last_msg = str(self._progress_last_ui_msg.get(module_id) or "")
        force = (
            pct <= 0.5
            or pct >= 99.5
            or msg != last_msg
            or "fail" in msg.lower()
            or "done" in msg.lower()
        )
        if not force and (now - last_at) < 0.12:
            return
        self._progress_last_ui_at[module_id] = now
        self._progress_last_ui_msg[module_id] = msg

        def _ui(
            p: float = pct,
            st: str = status,
            bt: str = bar_text,
            det: str = detail,
            mid: str = module_id,
        ) -> None:
            self._perm_progress.set(p)
            self._perm_progress_text.set(bt)
            if mid in self._module_status_vars:
                self._module_status_vars[mid].set(st)
            self._detail_var.set(det)

        self.after(0, _ui)

    def _make_module_progress_cb(self, module_id: str):
        def _cb(info: Any = None, *args: Any) -> None:
            # Normalize SHAP (msg, elapsed) and rating (frac, msg) shapes.
            if isinstance(info, dict):
                self._publish_module_progress(module_id, info)
                return
            if isinstance(info, str) and args:
                # SHAP: progress(msg, elapsed)
                elapsed = float(args[0]) if args else 0.0
                self._publish_module_progress(
                    module_id, message=info, elapsed=elapsed
                )
                return
            if isinstance(info, (int, float)) and args:
                # Feature rating: on_progress(frac, msg)
                self._publish_module_progress(
                    module_id,
                    frac=float(info),
                    message=str(args[0] if args else ""),
                )
                return
            if isinstance(info, str):
                self._publish_module_progress(module_id, message=info)

        return _cb

    def _sync_module_run_controls(self) -> None:
        """Show Target / Model only when a checked module requires them."""
        if not hasattr(self, "_module_inputs"):
            return
        selected = set(self._checked_module_ids())
        need_target = bool(selected & self._MODULES_NEED_TARGET)
        need_model = bool(selected & self._MODULES_NEED_MODEL)

        for w in (
            self._mi_target_label,
            self._mi_target_combo,
            self._shap_model_label,
            self._shap_model_combo,
        ):
            try:
                w.pack_forget()
            except Exception:
                pass

        if need_target:
            self._mi_target_label.pack(side="left")
            self._mi_target_combo.pack(side="left", padx=(4, 8))
            if self._dataset_id and (
                not self._mi_targets or not self._mi_target_combo["values"]
            ):
                self._reload_mi_targets()
        if need_model:
            self._shap_model_label.pack(side="left")
            self._shap_model_combo.pack(side="left", padx=(4, 8))
            if not self._shap_models or not self._shap_model_combo["values"]:
                self._reload_shap_models()

    def _run_selected(self) -> None:
        selected = self._checked_module_ids()
        if not selected:
            messagebox.showinfo(
                "Analysis",
                "Select one or more modules to run.",
                parent=self,
            )
            return
        self._start_modules(selected)

    def _run_all(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_lab_store import ANALYSIS_MODULES

        self._start_modules(list(ANALYSIS_MODULES))

    def _start_modules(self, module_ids: list[str]) -> None:
        if self._busy:
            messagebox.showinfo(
                "Analysis", "A module run is already in progress.", parent=self
            )
            return
        if not self._run_id or not self._dataset_id:
            messagebox.showinfo(
                "Analysis",
                "Select an analysis dataset first.",
                parent=self,
            )
            return

        ds = self._selected_dataset()
        if not ds:
            return

        mi_target = ""
        if "mutual_information" in module_ids or "feature_scorecard" in module_ids:
            if not self._mi_targets:
                self._reload_mi_targets()
            mi_target = str(self._mi_target.get() or "").strip()
            if not mi_target and "mutual_information" in module_ids:
                messagebox.showinfo(
                    "Mutual Information",
                    "Select a Target (e.g. future_ltp_5m) before running.",
                    parent=self,
                )
                return

        shap_model = ""
        if "shap" in module_ids or "permutation" in module_ids or "feature_scorecard" in module_ids:
            if not self._shap_models:
                self._reload_shap_models()
            shap_model = str(self._shap_model.get() or "").strip()
            if not shap_model and (
                "shap" in module_ids or "permutation" in module_ids
            ):
                messagebox.showinfo(
                    "Model required",
                    "Select a Model (trained model) before running "
                    "SHAP or Permutation.",
                    parent=self,
                )
                return

        if "permutation" in module_ids:
            # Target is not shown for Permutation — resolve from the model.
            mi_target = str(self._mi_target.get() or "").strip() or mi_target
            if not mi_target and shap_model:
                try:
                    from chain_replay_ml.training.registry import get_model_summary

                    summary = get_model_summary(self._data_dir, shap_model)
                    mi_target = str(summary.get("target") or "").strip()
                    if mi_target:
                        self._mi_target.set(mi_target)
                except Exception:
                    pass
            if not mi_target:
                messagebox.showinfo(
                    "Permutation",
                    "Could not resolve prediction target from the selected model.",
                    parent=self,
                )
                return

        self._busy = True
        self._detail_var.set("Running analysis modules…")
        done_messages: list[str] = []
        from chain_replay_ml.dataset_builder.analysis_permutation import CancelToken

        self._cancel_token = CancelToken()
        self._progress_last_ui_at.clear()
        self._progress_last_ui_msg.clear()
        try:
            self._cancel_btn.configure(state="normal")
        except Exception:
            pass
        self._perm_progress.set(0.0)
        self._perm_progress_text.set("")

        # Disable run buttons while busy (re-enabled in _done).
        try:
            self._run_selected_btn.configure(state="disabled")
            self._run_all_btn.configure(state="disabled")
        except Exception:
            pass

        def _worker() -> None:
            from chain_replay_ml.dataset_builder.analysis_correlation import (
                run_correlation_analysis,
            )
            from chain_replay_ml.dataset_builder.analysis_lab_store import (
                STATUS_COMPLETED,
                STATUS_FAILED,
                STATUS_RUNNING,
                dependency_blockers,
                format_module_status_label,
                set_module_status,
            )
            from chain_replay_ml.dataset_builder.analysis_mutual_information import (
                run_mutual_information,
            )
            from chain_replay_ml.dataset_builder.analysis_shap import (
                run_shap_analysis,
            )
            from chain_replay_ml.dataset_builder.analysis_permutation import (
                run_permutation_importance,
            )
            from chain_replay_ml.dataset_builder.analysis_feature_rating import (
                run_feature_rating,
            )

            errors: list[str] = []
            for mid in module_ids:
                blockers = dependency_blockers(
                    self._data_dir, self._run_id or "", mid
                )
                if blockers:
                    msg = f"Blocked — needs {', '.join(blockers)} first"
                    set_module_status(
                        self._data_dir,
                        self._run_id or "",
                        mid,
                        STATUS_FAILED,
                        message=msg,
                        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    errors.append(f"{mid}: {msg}")
                    self.after(
                        0,
                        lambda m=mid: self._module_status_vars[m].set("Failed"),
                    )
                    continue

                self.after(
                    0,
                    lambda m=mid: self._module_status_vars[m].set(
                        format_module_status_label(STATUS_RUNNING)
                    ),
                )
                self._publish_module_progress(
                    mid, frac=0.0, message="Starting…", elapsed=0.0
                )
                try:
                    if mid == "correlation":
                        summary = run_correlation_analysis(
                            self._data_dir,
                            self._run_id or "",
                            ds,
                            progress=self._make_module_progress_cb("correlation"),
                            compute_backend=str(
                                self._corr_backend_pref.get() or "auto"
                            ),
                        )
                        compute_meta = summary.get("compute_backend")
                        if isinstance(compute_meta, dict):
                            self.after(
                                0,
                                lambda m=dict(compute_meta): self._refresh_corr_backend_status(
                                    m
                                ),
                            )
                        done_messages.append(
                            f"Correlation · {summary.get('features_analysed')} features · "
                            f"{int(summary.get('pairs') or 0):,} pairs"
                        )
                    elif mid == "hca":
                        from chain_replay_ml.dataset_builder.analysis_hca import (
                            run_hca_analysis,
                        )

                        hca_out = run_hca_analysis(
                            self._data_dir,
                            self._run_id or "",
                            progress=self._make_module_progress_cb("hca"),
                        )
                        done_messages.append(
                            str(hca_out.get("message") or "HCA done")
                        )
                    elif mid == "mutual_information":
                        run_mutual_information(
                            self._data_dir,
                            self._run_id or "",
                            ds,
                            mi_target,
                            progress=self._make_module_progress_cb(
                                "mutual_information"
                            ),
                        )
                        done_messages.append(
                            f"Mutual Information vs {mi_target}"
                        )
                    elif mid == "shap":
                        # Pre-selection warning (does not block)
                        try:
                            from chain_replay_ml.dataset_builder.analysis_shap import (
                                assess_model_feature_selection,
                            )

                            info = assess_model_feature_selection(
                                self._data_dir,
                                shap_model,
                                dataset=ds,
                                run_id=self._run_id or "",
                            )
                            if info.get("pre_selection") and info.get("warning"):
                                done_messages.append(
                                    "Warning: model trained before feature selection"
                                )
                        except Exception:
                            pass

                        shap_out = run_shap_analysis(
                            self._data_dir,
                            self._run_id or "",
                            ds,
                            shap_model,
                            progress=self._make_module_progress_cb("shap"),
                        )
                        done_messages.append(
                            str(
                                shap_out.get("message")
                                or "Model Explanation (SHAP) done"
                            )
                        )
                        # Stage 2 only — writes validation_* (never Discovery fields)
                        try:
                            from chain_replay_ml.dataset_builder.analysis_feature_rating import (
                                STAGE_VALIDATION,
                                run_feature_rating,
                            )

                            rate_out = run_feature_rating(
                                self._data_dir,
                                self._run_id or "",
                                model_name=shap_model,
                                target=mi_target
                                or str(self._mi_target.get() or ""),
                                stage=STAGE_VALIDATION,
                                on_progress=self._make_module_progress_cb("shap"),
                            )
                            done_messages.append(
                                str(
                                    rate_out.get("message")
                                    or "Validation rating updated"
                                )
                            )
                        except Exception as rate_exc:
                            done_messages.append(
                                f"Validation rating skipped: {rate_exc}"
                            )
                    elif mid == "permutation":
                        perm_out = run_permutation_importance(
                            self._data_dir,
                            self._run_id or "",
                            ds,
                            shap_model,
                            mi_target,
                            progress=self._make_module_progress_cb("permutation"),
                            cancel=self._cancel_token,
                        )
                        done_messages.append(
                            str(perm_out.get("message") or "Permutation done")
                        )
                        if perm_out.get("cancelled"):
                            self.after(
                                0,
                                lambda: self._module_status_vars["permutation"].set(
                                    "Failed"
                                ),
                            )
                            continue
                    elif mid == "feature_scorecard":
                        from chain_replay_ml.dataset_builder.analysis_feature_rating import (
                            STAGE_DISCOVERY,
                        )

                        rate_out = run_feature_rating(
                            self._data_dir,
                            self._run_id or "",
                            model_name=shap_model
                            or str(self._shap_model.get() or ""),
                            target=mi_target or str(self._mi_target.get() or ""),
                            stage=STAGE_DISCOVERY,
                            on_progress=self._make_module_progress_cb(
                                "feature_scorecard"
                            ),
                        )
                        done_messages.append(
                            str(rate_out.get("message") or "Feature Rating done")
                        )
                    else:
                        # Scaffold for remaining modules (VIF)
                        started = time.strftime("%Y-%m-%d %H:%M:%S")
                        t0 = time.perf_counter()
                        set_module_status(
                            self._data_dir,
                            self._run_id or "",
                            mid,
                            STATUS_RUNNING,
                            started_at=started,
                            message="Scaffold run",
                        )
                        self._publish_module_progress(
                            mid, frac=0.2, message="Scaffold run…", elapsed=0.0
                        )
                        time.sleep(0.05)
                        set_module_status(
                            self._data_dir,
                            self._run_id or "",
                            mid,
                            STATUS_COMPLETED,
                            started_at=started,
                            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                            elapsed_sec=round(max(time.perf_counter() - t0, 0.0), 3),
                            message="Scaffold completed (compute not implemented yet)",
                        )
                        self._publish_module_progress(
                            mid,
                            frac=1.0,
                            message="Scaffold completed",
                            elapsed=max(time.perf_counter() - t0, 0.0),
                        )
                    try:
                        from chain_replay_ml.dataset_builder.analysis_artifacts import (
                            publish_module_artifact,
                        )

                        publish_module_artifact(
                            self._data_dir, self._run_id or "", mid
                        )
                    except Exception:
                        pass
                    self.after(
                        0,
                        lambda m=mid: self._module_status_vars[m].set(
                            format_module_status_label(STATUS_COMPLETED)
                        ),
                    )
                except Exception as exc:
                    errors.append(f"{mid}: {exc}")
                    set_module_status(
                        self._data_dir,
                        self._run_id or "",
                        mid,
                        STATUS_FAILED,
                        message=str(exc),
                        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    self._publish_module_progress(
                        mid, message=f"Failed · {exc}"
                    )
                    self.after(
                        0, lambda m=mid: self._module_status_vars[m].set("Failed")
                    )

            def _done() -> None:
                self._busy = False
                self._cancel_token = None
                try:
                    self._cancel_btn.configure(state="disabled")
                except Exception:
                    pass
                try:
                    self._run_selected_btn.configure(state="normal")
                    self._run_all_btn.configure(state="normal")
                except Exception:
                    pass
                if errors:
                    self._detail_var.set(
                        "Finished with issues: " + "; ".join(errors[:3])
                    )
                    self._perm_progress_text.set("Finished with issues")
                elif done_messages:
                    self._detail_var.set(" · ".join(done_messages))
                    self._perm_progress.set(100.0)
                    self._perm_progress_text.set("100%")
                else:
                    self._detail_var.set("Module run finished.")
                    self._perm_progress.set(100.0)
                    self._perm_progress_text.set("100%")
                self._on_dataset_selected()
                if "correlation" in module_ids:
                    self._corr_nb.select(self._tab_summary)
                    # Rebuild profiles so Insights/cluster fields stay in sync.
                    # rehydrate_* run inside build_feature_profiles.
                    self.after(100, self._build_profiles_now)
                if "hca" in module_ids:
                    self._select_research_tab("Feature Families")
                    self._reload_families_tab()
                    self._reload_family_review_tab()
                if "mutual_information" in module_ids:
                    try:
                        self._select_research_tab('Mutual Information')
                    except Exception:
                        pass
                    self._reload_mi_tab()
                    if "correlation" not in module_ids:
                        self._reload_scorecard()
                        from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
                            profiles_exist,
                        )
                        from chain_replay_ml.dataset_builder.analysis_mutual_information import (
                            rehydrate_mi_into_profiles,
                        )

                        if not profiles_exist(self._data_dir, self._run_id or ""):
                            self.after(100, self._build_profiles_now)
                        else:
                            rehydrate_mi_into_profiles(
                                self._data_dir,
                                self._run_id or "",
                                target=mi_target or None,
                            )
                            self._ensure_profiles_loaded(async_build=False)
                if "shap" in module_ids and "correlation" not in module_ids:
                    try:
                        self._select_research_tab("Model Validation")
                    except Exception:
                        try:
                            self._select_research_tab("Model Explanation (SHAP)")
                        except Exception:
                            pass
                    self._reload_validation_scorecard()
                    self._reload_shap_tab()
                    # Discovery Scorecard / Recommendations are unchanged by SHAP
                    self._reload_scorecard()
                    self._reload_recommendations()
                    from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
                        profiles_exist,
                    )
                    from chain_replay_ml.dataset_builder.analysis_shap import (
                        rehydrate_shap_into_profiles,
                    )

                    if not profiles_exist(self._data_dir, self._run_id or ""):
                        self.after(100, self._build_profiles_now)
                    else:
                        rehydrate_shap_into_profiles(
                            self._data_dir,
                            self._run_id or "",
                            model_name=shap_model or None,
                        )
                        self._ensure_profiles_loaded(async_build=False)
                    if done_messages and not errors:
                        self._detail_var.set(" · ".join(done_messages))
                if "permutation" in module_ids and "correlation" not in module_ids:
                    try:
                        self._select_research_tab('Permutation')
                    except Exception:
                        pass
                    self._reload_perm_tab()
                    self._reload_scorecard()
                    from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
                        profiles_exist,
                    )
                    from chain_replay_ml.dataset_builder.analysis_permutation import (
                        rehydrate_permutation_into_profiles,
                    )

                    if not profiles_exist(self._data_dir, self._run_id or ""):
                        self.after(100, self._build_profiles_now)
                    else:
                        rehydrate_permutation_into_profiles(
                            self._data_dir,
                            self._run_id or "",
                            model_id=shap_model or None,
                            target=mi_target or None,
                        )
                        self._ensure_profiles_loaded(async_build=False)
                    if done_messages and not errors:
                        self._detail_var.set(" · ".join(done_messages))
                        self._perm_progress.set(100.0)
                if "feature_scorecard" in module_ids:
                    try:
                        from chain_replay_ml.dataset_builder.analysis_family_review import (
                            review_summary,
                        )

                        summ = review_summary(
                            self._data_dir, self._run_id or "", min_size=2
                        )
                        if int(summ.get("n_needs_review") or 0) > 0:
                            self._review_filter.set("Needs Review")
                            self._select_research_tab("Family Review")
                        else:
                            self._select_research_tab("Recommendations")
                    except Exception:
                        self._select_research_tab("Recommendations")
                    self._reload_scorecard()
                    self._reload_recommendations()
                    self._reload_validation_scorecard()
                    self._reload_family_review_tab()
                    self._reload_discovery_banner()
                    self._ensure_profiles_loaded(async_build=False)
                    if done_messages and not errors:
                        self._detail_var.set(" · ".join(done_messages))

            self.after(0, _done)

        threading.Thread(
            target=_worker, name="feature-analysis-run", daemon=True
        ).start()

    def _open_data_dir(self) -> None:
        import os
        import subprocess
        import sys

        path = self._data_dir
        os.makedirs(path, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Open folder", str(exc), parent=self)


__all__ = ["FeatureAnalysisPanel"]
