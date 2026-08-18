"""Feature Studio shell — shared toolbar + tabs for 4.1–4.5 + 5.3 studios.

Controller owns unified Load Artifacts / Compute for the selected model.
Tabs are viewers; Studio Compare stays as an active-tab-only extra.
Feature Intelligence is a separate Model Builder page (not a Studio tab).
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .build_config_prefs import load_build_config_prefs, save_build_config_prefs
from .build_service import chart_data_dir
from .diagnostics_studio_panel import DiagnosticsStudioPanel
from .experiment_planner_studio_panel import ExperimentPlannerStudioPanel
from .feature_distribution_studio_panel import FeatureDistributionStudioPanel
from .feature_drift_studio_panel import FeatureDriftStudioPanel
from .feature_importance_studio_panel import FeatureImportanceStudioPanel
from .feature_studio_pipeline import (
    PIPELINE_ORDER,
    PLANNER_SKIP_MSG,
    PipelineResult,
    STUDIO_LABELS,
    run_compute_pipeline,
    run_load_pipeline,
)
from .multi_model_studio_panel import MultiModelStudioPanel
from .production_validation_panel import ProductionValidationPanel
from .ui_state import get_ui_state_manager


class FeatureStudioPanel(ttk.Frame):
    """Hosts Importance / Distribution / Drift / Compare / Diagnostics / Production Validation / Planner tabs."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_create_model: Callable[[str, list[str], str | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self._model_names: list[str] = []
        self._model_var = tk.StringVar()
        self._model_a_var = tk.StringVar()
        self._model_b_var = tk.StringVar()
        self._filter_var = tk.StringVar()
        self._top_n_var = tk.StringVar(value="20")
        self._top_n_only = tk.BooleanVar(value=False)
        self._loading_prefs = False

        # Per-model artifact cache — invalidated on model change; never reloads on tab switch.
        self._cache_model: str | None = None
        self._cache: PipelineResult | None = None
        self._pipeline_busy = False

        self._readiness_var = tk.StringVar(value="")
        self._readiness = ttk.Label(
            self,
            textvariable=self._readiness_var,
            foreground="#888",
            wraplength=720,
            justify="left",
        )
        self._readiness.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        self._readiness.grid_remove()

        self._ui_state = get_ui_state_manager()
        self._build_toolbar()

        shared = dict(
            model_var=self._model_var,
            filter_var=self._filter_var,
            top_n_var=self._top_n_var,
            top_n_only=self._top_n_only,
        )
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=2, column=0, sticky="nsew")

        self.importance = FeatureImportanceStudioPanel(
            self._notebook, chart_dir=chart_dir, **shared
        )
        self.distribution = FeatureDistributionStudioPanel(
            self._notebook, chart_dir=chart_dir, **shared
        )
        self.drift = FeatureDriftStudioPanel(
            self._notebook, chart_dir=chart_dir, **shared
        )
        self.compare = MultiModelStudioPanel(
            self._notebook,
            chart_dir=chart_dir,
            model_a_var=self._model_a_var,
            model_b_var=self._model_b_var,
            filter_var=self._filter_var,
            top_n_var=self._top_n_var,
            top_n_only=self._top_n_only,
        )
        self.diagnostics = DiagnosticsStudioPanel(
            self._notebook, chart_dir=chart_dir, **shared
        )
        self.production_validation = ProductionValidationPanel(
            self._notebook,
            chart_dir=chart_dir,
            on_create_model=on_create_model,
            **shared,
        )
        self.planner = ExperimentPlannerStudioPanel(
            self._notebook, chart_dir=chart_dir, **shared
        )

        self._tabs: dict[str, ttk.Frame] = {
            "importance": self.importance,
            "distribution": self.distribution,
            "drift": self.drift,
            "compare": self.compare,
            "diagnostics": self.diagnostics,
            "production_validation": self.production_validation,
            "planner": self.planner,
        }
        self._pipeline_panels: dict[str, ttk.Frame] = {
            "importance": self.importance,
            "distribution": self.distribution,
            "drift": self.drift,
            "diagnostics": self.diagnostics,
            "planner": self.planner,
        }
        self._notebook.add(self.importance, text="Importance")
        self._notebook.add(self.distribution, text="Distribution")
        self._notebook.add(self.drift, text="Drift")
        self._notebook.add(self.compare, text="Studio Compare")
        self._notebook.add(self.diagnostics, text="Diagnostics")
        self._notebook.add(self.production_validation, text="Production Validation")
        self._notebook.add(self.planner, text="Experiment Planner")

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._model_var.trace_add("write", lambda *_: self._on_shared_model_changed())
        self._model_a_var.trace_add("write", lambda *_: self._persist_model_prefs())
        self._model_b_var.trace_add("write", lambda *_: self._persist_model_prefs())

        # Model selection already persists via build_config_prefs (feature_studio.*,
        # see _studio_prefs/_persist_model_prefs above) — left as-is. Filter/Top-N and
        # the active tab were not persisted before; wire those onto the shared store.
        self._ui_state.bind_entry(None, "feature_studio.filter", var=self._filter_var, restore=True)
        self._ui_state.bind_entry(None, "feature_studio.top_n", var=self._top_n_var, default="20", restore=True)
        self._ui_state.bind_checkbutton(
            None, "feature_studio.top_n_only", var=self._top_n_only, restore=True
        )
        self._ui_state.bind_notebook(self._notebook, "feature_studio.tab")

    def _studio_prefs(self) -> dict[str, Any]:
        if not self.chart_dir:
            return {}
        doc = load_build_config_prefs(self.chart_dir) or {}
        section = doc.get("feature_studio")
        return section if isinstance(section, dict) else {}

    def _persist_model_prefs(self) -> None:
        if self._loading_prefs or not self.chart_dir:
            return
        patch: dict[str, Any] = {}
        primary = str(self._model_var.get() or "").strip()
        if primary:
            patch["selected_model"] = primary
        model_a = str(self._model_a_var.get() or "").strip()
        if model_a:
            patch["model_a"] = model_a
        model_b = str(self._model_b_var.get() or "").strip()
        if model_b:
            patch["model_b"] = model_b
        if not patch:
            return
        try:
            current = self._studio_prefs()
            save_build_config_prefs(self.chart_dir, {
                "feature_studio": {**current, **patch},
            })
        except Exception:
            pass

    def _pick_restored_name(
        self,
        *,
        current: str,
        names: list[str],
        pref_key: str,
        fallback: str = "",
    ) -> str:
        """Keep current if still listed; else prefs; else fallback / first / empty."""
        if current in names:
            return current
        saved = str(self._studio_prefs().get(pref_key) or "").strip()
        if saved in names:
            return saved
        if fallback and fallback in names:
            return fallback
        return names[0] if names else ""

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 6, 8, 4))
        bar.grid(row=1, column=0, sticky="ew")

        row1 = ttk.Frame(bar)
        row1.pack(fill="x")

        self._model_slot = ttk.Frame(row1)
        self._model_slot.pack(side="left")

        self._single_model_frame = ttk.Frame(self._model_slot)
        ttk.Label(self._single_model_frame, text="Model").pack(side="left")
        self._model_combo = ttk.Combobox(
            self._single_model_frame,
            textvariable=self._model_var,
            width=48,
            state="readonly",
        )
        self._model_combo.pack(side="left", padx=(6, 8))

        self._compare_model_frame = ttk.Frame(self._model_slot)
        ttk.Label(self._compare_model_frame, text="Model A").pack(side="left")
        self._combo_a = ttk.Combobox(
            self._compare_model_frame,
            textvariable=self._model_a_var,
            width=36,
            state="readonly",
        )
        self._combo_a.pack(side="left", padx=(4, 10))
        ttk.Label(self._compare_model_frame, text="Model B").pack(side="left")
        self._combo_b = ttk.Combobox(
            self._compare_model_frame,
            textvariable=self._model_b_var,
            width=36,
            state="readonly",
        )
        self._combo_b.pack(side="left", padx=(4, 8))

        self._single_model_frame.pack(side="left")

        ttk.Button(row1, text="Load Artifacts", command=self._on_load).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="Compute", command=self._on_compute).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="Export CSV", command=self._on_export).pack(
            side="left", padx=2
        )
        ttk.Button(
            row1, text="Evidence DB", command=self._on_open_evidence_viewer
        ).pack(side="left", padx=4)

        row2 = ttk.Frame(bar)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Filter").pack(side="left")
        ttk.Entry(row2, textvariable=self._filter_var, width=18).pack(
            side="left", padx=(6, 0)
        )
        ttk.Checkbutton(
            row2,
            text="Top N only",
            variable=self._top_n_only,
        ).pack(side="left", padx=(12, 4))
        ttk.Entry(row2, textvariable=self._top_n_var, width=5).pack(side="left")

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def _active_panel(self) -> ttk.Frame | None:
        try:
            return self._notebook.nametowidget(self._notebook.select())
        except (tk.TclError, KeyError):
            return None

    def _is_compare_active(self) -> bool:
        return self._active_panel() is self.compare

    def _is_production_validation_active(self) -> bool:
        return self._active_panel() is self.production_validation

    def _sync_toolbar_mode(self) -> None:
        if self._is_compare_active():
            self._single_model_frame.pack_forget()
            self._compare_model_frame.pack(side="left")
        else:
            self._compare_model_frame.pack_forget()
            self._single_model_frame.pack(side="left")

    def _selected_model(self) -> str:
        return str(self._model_var.get() or "").strip()

    def _invalidate_cache(self, *, reason: str = "") -> None:
        self._cache_model = None
        self._cache = None
        msg = reason or "Model changed — click Load Artifacts."
        for key in PIPELINE_ORDER:
            panel = self._pipeline_panels.get(key)
            mark = getattr(panel, "mark_unavailable", None)
            if callable(mark):
                mark(msg)
        # Production Validation is outside the Imp→…→Planner pipeline.
        pv_mark = getattr(self.production_validation, "mark_unavailable", None)
        if callable(pv_mark):
            pv_mark("Model changed — click Resolve Unseen Dataset.")

    def _populate_from_pipeline(self, result: PipelineResult) -> None:
        """Apply loaded payloads to all pipeline tabs (viewers)."""
        name = result.model_name
        for key in PIPELINE_ORDER:
            panel = self._pipeline_panels.get(key)
            if panel is None:
                continue
            stage = result.stages.get(key)
            apply_fn = getattr(panel, "apply_artifacts", None)
            mark = getattr(panel, "mark_unavailable", None)
            if stage is None:
                if callable(mark):
                    mark(f"Unavailable — {STUDIO_LABELS.get(key, key)}.")
                continue
            if stage.skipped:
                if callable(mark):
                    mark(stage.skip_reason or PLANNER_SKIP_MSG)
                continue
            if stage.available and stage.payload is not None:
                if callable(apply_fn):
                    apply_fn(stage.payload, name)
                continue
            # Missing or failed
            if stage.error and callable(mark):
                mark(f"Unavailable — {stage.error}")
            elif callable(mark):
                mark(f"Unavailable — no {STUDIO_LABELS.get(key, key)} artifacts.")

        self._cache_model = name
        self._cache = result
        self._update_readiness(name, pipeline=result)

    def _on_load(self) -> None:
        if self._pipeline_busy:
            return
        # Production Validation is outside the Imp→…→Planner pipeline.
        if self._is_production_validation_active():
            self.production_validation._load_cached_status(quiet=False)
            return
        name = self._selected_model()
        if not name:
            messagebox.showwarning(
                "Feature Studio", "Select a model first.", parent=self
            )
            return

        self._pipeline_busy = True
        self._readiness_var.set(f"{name} — Loading artifacts…")
        self._readiness.grid()
        data_dir = self._data_dir()

        def work() -> None:
            err: str | None = None
            result: PipelineResult | None = None
            try:
                result = run_load_pipeline(data_dir=data_dir, model_name=name)
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._pipeline_busy = False
                if err or result is None:
                    self._readiness_var.set(f"{name} — Load failed: {err or 'unknown'}")
                    messagebox.showerror(
                        "Feature Studio", err or "Load failed", parent=self
                    )
                    return
                self._populate_from_pipeline(result)
                # Studio Compare is active-tab-only extra (not in pipeline).
                if self._is_compare_active():
                    fn = getattr(self.compare, "_on_load", None)
                    if callable(fn):
                        fn()

            self.after(0, done)

        threading.Thread(target=work, name="fs-load", daemon=True).start()

    def _on_compute(self) -> None:
        if self._pipeline_busy:
            return
        # Production Validation has its own Holdout→Unseen compute path.
        if self._is_production_validation_active():
            self.production_validation._on_compute()
            return
        name = self._selected_model()
        if not name:
            messagebox.showwarning(
                "Feature Studio", "Select a model first.", parent=self
            )
            return

        # Compare tab: optional active-tab-only extra after full pipeline.
        also_compare = self._is_compare_active()
        self._pipeline_busy = True
        data_dir = self._data_dir()

        def _progress(event: dict[str, Any]) -> None:
            studio = str(event.get("studio") or "")
            stage = str(event.get("stage") or "")
            label = STUDIO_LABELS.get(studio, studio)
            if stage == "compute":
                text = f"{name} — Computing {label}…"
            elif stage == "load":
                text = f"{name} — Loading {label}…"
            else:
                text = f"{name} — {stage} {label}…"

            def ui() -> None:
                self._readiness_var.set(text)
                self._readiness.grid()

            self.after(0, ui)

        def work() -> None:
            err: str | None = None
            compute_result: PipelineResult | None = None
            load_result: PipelineResult | None = None
            try:
                compute_result = run_compute_pipeline(
                    data_dir=data_dir,
                    model_name=name,
                    progress=_progress,
                )
                # Compute → Persist → Load → Populate
                load_result = run_load_pipeline(
                    data_dir=data_dir,
                    model_name=name,
                    progress=_progress,
                )
                # Preserve planner skip from compute if load found nothing.
                if compute_result is not None and load_result is not None:
                    c_plan = compute_result.stages.get("planner")
                    l_plan = load_result.stages.get("planner")
                    if (
                        c_plan is not None
                        and c_plan.skipped
                        and l_plan is not None
                        and not l_plan.available
                    ):
                        load_result.stages["planner"] = c_plan
                    # Surface compute errors onto load stages that failed to produce artifacts.
                    for key in PIPELINE_ORDER:
                        c_stage = compute_result.stages.get(key)
                        l_stage = load_result.stages.get(key)
                        if (
                            c_stage is not None
                            and c_stage.error
                            and l_stage is not None
                            and not l_stage.available
                            and not l_stage.skipped
                        ):
                            l_stage.error = c_stage.error
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._pipeline_busy = False
                if err or load_result is None:
                    self._readiness_var.set(
                        f"{name} — Compute failed: {err or 'unknown'}"
                    )
                    messagebox.showerror(
                        "Feature Studio", err or "Compute failed", parent=self
                    )
                    return
                self._populate_from_pipeline(load_result)
                skip_bits = []
                if compute_result is not None:
                    for key in PIPELINE_ORDER:
                        stage = compute_result.stages.get(key)
                        if stage and stage.skipped and stage.skip_reason:
                            skip_bits.append(stage.skip_reason)
                        elif stage and stage.error:
                            skip_bits.append(
                                f"{STUDIO_LABELS.get(key, key)}: {stage.error}"
                            )
                if also_compare:
                    fn = getattr(self.compare, "_on_compare", None)
                    if callable(fn):
                        fn()
                if skip_bits:
                    # Non-fatal: pipeline continued; surface a single summary.
                    messagebox.showwarning(
                        "Feature Studio",
                        "Some studios did not complete:\n\n" + "\n".join(skip_bits),
                        parent=self,
                    )

            self.after(0, done)

        threading.Thread(target=work, name="fs-compute", daemon=True).start()

    def _on_export(self) -> None:
        panel = self._active_panel()
        if panel is None:
            return
        fn = getattr(panel, "_on_export", None)
        if callable(fn):
            fn()

    def on_show(self) -> None:
        self._refresh_model_names()
        self._show_active_tab()

    def refresh(self, *, lazy: bool = True) -> None:
        self._refresh_model_names()
        self._show_active_tab(lazy=lazy)

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        for panel in self._tabs.values():
            if hasattr(panel, "chart_dir"):
                panel.chart_dir = chart_dir
        self._invalidate_cache(reason="Chart dir changed — click Load Artifacts.")

    def select_tab(self, key: str) -> None:
        panel = self._tabs.get(key)
        if panel is None:
            return
        try:
            self._notebook.select(panel)
        except tk.TclError:
            return
        self._show_active_tab()

    def open_with_model(self, model_name: str, *, tab: str = "importance") -> None:
        name = str(model_name or "").strip()
        self._refresh_model_names()
        self._model_var.set(name)
        self._update_readiness(name)
        self.select_tab(tab)
        panel = self._tabs.get(tab)
        opener = getattr(panel, "open_with_model", None)
        if callable(opener):
            opener(name)

    def open_with_models(self, model_a: str, model_b: str) -> None:
        self._readiness.grid_remove()
        self.compare.open_with_models(model_a, model_b)
        self._refresh_model_names()
        self.select_tab("compare")

    def _fetch_model_names(self) -> list[str]:
        from .selection_lists import get_sorted_model_names

        return get_sorted_model_names(self._data_dir(), lightweight=False)

    def _refresh_model_names(self) -> None:
        try:
            names = self._fetch_model_names()
        except Exception:
            names = []
        self._model_names = names
        self._model_combo["values"] = names
        self._combo_a["values"] = names
        self._combo_b["values"] = names

        self._loading_prefs = True
        try:
            primary = self._pick_restored_name(
                current=str(self._model_var.get() or "").strip(),
                names=names,
                pref_key="selected_model",
            )
            if self._model_var.get() != primary:
                self._model_var.set(primary)

            model_a = self._pick_restored_name(
                current=str(self._model_a_var.get() or "").strip(),
                names=names,
                pref_key="model_a",
            )
            if self._model_a_var.get() != model_a:
                self._model_a_var.set(model_a)

            model_b = self._pick_restored_name(
                current=str(self._model_b_var.get() or "").strip(),
                names=names,
                pref_key="model_b",
                fallback=names[1] if len(names) > 1 else "",
            )
            if self._model_b_var.get() != model_b:
                self._model_b_var.set(model_b)
        finally:
            self._loading_prefs = False

        for panel in self._tabs.values():
            apply_names = getattr(panel, "apply_model_names", None)
            if callable(apply_names):
                apply_names(names)

    def _update_readiness(
        self, model_name: str, *, pipeline: PipelineResult | None = None
    ) -> None:
        name = str(model_name or "").strip()
        if not name:
            self._readiness.grid_remove()
            return
        marks = ""
        if pipeline is not None:
            marks = pipeline.status_marks()
        elif self._cache is not None and self._cache_model == name:
            marks = self._cache.status_marks()
        try:
            from chain_replay_ml.post_training import (
                format_readiness_line,
                load_feature_studio_status,
            )
            from chain_replay_ml.training.paths import model_package_dir

            package_dir = model_package_dir(chart_data_dir(self.chart_dir), name)
            status = load_feature_studio_status(package_dir)
            line = format_readiness_line(status)
        except Exception as exc:
            line = f"Feature Studio: status unavailable ({exc})"
        if marks:
            self._readiness_var.set(f"{name} — {line}  |  {marks}")
        else:
            self._readiness_var.set(f"{name} — {line}")
        self._readiness.grid()

    def _on_shared_model_changed(self) -> None:
        if not self._loading_prefs:
            self._persist_model_prefs()
        if self._is_compare_active():
            return
        name = self._selected_model()
        if not name:
            self._invalidate_cache(reason="Select a model and Load Artifacts.")
            self._readiness.grid_remove()
            return
        if name != self._cache_model:
            self._invalidate_cache(
                reason="Model changed — click Load Artifacts (no auto-load)."
            )
        self._update_readiness(name)

    def _on_tab_changed(self, _event: Any = None) -> None:
        # Tab switch never reloads artifacts — only sync toolbar / readiness.
        self._show_active_tab()

    def _show_active_tab(self, *, lazy: bool = True) -> None:
        del lazy  # retained for API compat; never triggers artifact reload
        self._sync_toolbar_mode()
        current = self._active_panel()
        if current is None:
            return
        if current is self.compare:
            self._readiness.grid_remove()
        else:
            name = self._selected_model()
            if name:
                self._update_readiness(name)
        # Viewers: on_show is a no-op (no disk load). Compare may refresh labels only.
        on_show = getattr(current, "on_show", None)
        if callable(on_show):
            on_show()

    def _on_open_evidence_viewer(self) -> None:
        from chain_replay_ml.production_validation.dataset_context import (
            resolve_context_from_model_package,
        )
        from .feature_recommendation_viewer import open_feature_recommendation_viewer

        name = self._selected_model()
        ctx = resolve_context_from_model_package(self._data_dir(), name) if name else None
        if ctx:
            open_feature_recommendation_viewer(
                self,
                chart_dir=self.chart_dir,
                initial_market=ctx.market,
                initial_interval_sec=ctx.sampling_interval_sec,
                initial_sliding_window=ctx.sliding_window,
                initial_feature_project_id=ctx.feature_project_id,
            )
        else:
            open_feature_recommendation_viewer(
                self,
                chart_dir=self.chart_dir,
            )
