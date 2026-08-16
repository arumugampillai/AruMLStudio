"""ML Research Studio — Tkinter shell for datasets, models, predictions, and strategy research."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Any

from .create_dataset_panel import CreateDatasetPanel
from .feature_registry_panel import FeatureRegistryPanel
from .master_data_panel import MasterDataPanel
from .create_model_shell_panel import CreateModelShell
from .comparison_panel import ComparisonPanel
from .feature_intelligence_studio_panel import FeatureIntelligenceStudioPanel
from .feature_studio_panel import FeatureStudioPanel
from .model_explorer.panel import ModelExplorerPanel
from .model_registry_panel import ModelRegistryPanel
from .registry_panel import RegistryPanel
from .settings_panel import SettingsPanel
from .nifty_history_bar_panel import NiftyHistoryBarPanel
from .nifty_ormp_overview_panel import NiftyOrmpOverviewPanel
from .fold_replay_panel import FoldReplayPanel
from .research_program_panel import ResearchProgramPanel
from .experiment_planner_panel import ExperimentPlannerPanel
from .strategy_lab_panel import StrategiesPanel
from .project_config import ensure_project_data_dir, resolve_chart_dir, save_project_config
from .build_service import chart_data_dir
from .build_progress_manager import get_build_progress_manager
from .global_status_bar import GlobalBuildStatusBar
from .research_campaign_coordinator import get_research_campaign_coordinator
from .ui_state import get_ui_state_manager

from path_config import CHART_DATA_ROOT as _CHART_DIR
_NAV_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Historical Data",
        [
            ("history.nifty", "Nifty History"),
            ("ormp.overview", "ORMP Overview"),
        ],
    ),
    (
        "Master Dataset",
        [
            ("master.create", "Create"),
            ("master.trading_days", "Master Dataset"),
        ],
    ),
    (
        "Model Builder",
        [
            ("builder.create", "Create Model"),
            ("builder.comparison", "Comparison"),
            ("builder.feature_studio", "Feature Studio"),
            ("builder.feature_intelligence", "Feature Intelligence"),
            ("builder.model_explorer", "Model Explorer"),
        ],
    ),
    (
        "Registry",
        [
            ("registry.datasets", "Dataset Registry"),
            ("registry.features", "Feature Registry"),
            ("registry.models", "Models"),
        ],
    ),
    (
        "Strategy Lab",
        [
            ("strategy.strategies", "Strategies"),
            ("strategy.replay", "Replay"),
            ("strategy.programs", "Research Programs"),
            ("strategy.experiments", "Experiment Planner"),
        ],
    ),
    (
        "Settings",
        [
            ("settings.main", "Settings"),
        ],
    ),
]

_PAGE_TITLES: dict[str, str] = {key: label for _sec, items in _NAV_SECTIONS for key, label in items}

_FEATURE_STUDIO_TAB_ALIASES: dict[str, str] = {
    "builder.importance": "importance",
    "builder.distribution": "distribution",
    "builder.drift": "drift",
    "builder.multi_compare": "compare",
    "builder.diagnostics": "diagnostics",
    "builder.experiment_planner": "planner",
}

# Old standalone comparison nav keys → Comparison shell tabs.
_COMPARISON_TAB_ALIASES: dict[str, str] = {
    "builder.compare": "model",
    "builder.fold_compare": "fold",
    "registry.dataset_compare": "dataset",
}

# Old standalone strategy nav keys → Strategies shell tabs.
_STRATEGY_TAB_ALIASES: dict[str, str] = {
    "strategy.simulation": "simulation",
    "strategy.leaderboard": "leaderboard",
    "predictions.runs": "prediction_runs",
}

# Old standalone Outcome Labels nav key → Create Model shell tab.
_CREATE_MODEL_TAB_ALIASES: dict[str, str] = {
    "builder.ole": "ole",
}

# Any nav key is valid to restore into except pages that only make sense as
# a transient target of another action (none currently excluded).
_UI_STATE_WINDOW_KEY = "app.main_window"
_UI_STATE_LAST_PAGE_KEY = "app.last_page"
_DEFAULT_GEOMETRY = "1260x1010+0+0"


class MLResearchStudioApp(tk.Tk):
    def __init__(self, *, chart_dir: str | None = None) -> None:
        super().__init__()
        self.chart_dir = resolve_chart_dir(cli_chart_dir=chart_dir)
        ensure_project_data_dir(self.chart_dir)
        save_project_config(self.chart_dir)
        self.title("ML Research Studio")
        self._ui_state = get_ui_state_manager()
        self._ui_state.attach_root(self)
        self.geometry(_DEFAULT_GEOMETRY)
        self._current_page = ""
        self._nav_btns: dict[str, ttk.Button] = {}
        self._project_label_var = tk.StringVar(value=self._project_label_text())
        self._progress_manager = get_build_progress_manager()
        self._research_coordinator = get_research_campaign_coordinator()
        self._research_coordinator.bind_ui(self.after)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0, minsize=34)
        self.grid_columnconfigure(0, weight=1)

        self._build_status_bar()
        self._build_shell()
        # Restore the remembered window size/position now that the shell
        # exists — geometry restore also wires ongoing auto-save on resize/move.
        self._ui_state.restore_window(self, _UI_STATE_WINDOW_KEY, default_geometry=_DEFAULT_GEOMETRY)
        self.after(200, self._poll_progress)
        initial_page = str(self._ui_state.get(_UI_STATE_LAST_PAGE_KEY) or "").strip()
        if initial_page in _COMPARISON_TAB_ALIASES:
            pass  # remapped in _show_page to builder.comparison + tab
        elif initial_page in _STRATEGY_TAB_ALIASES:
            pass  # remapped in _show_page to strategy.strategies + tab
        elif initial_page not in self._known_nav_keys():
            initial_page = "master.create"
        self._show_page(initial_page)

    def _known_nav_keys(self) -> set[str]:
        return {key for _sec, items in _NAV_SECTIONS for key, _label in items}

    def _build_status_bar(self) -> None:
        self._status_bar = GlobalBuildStatusBar(
            self,
            manager=self._progress_manager,
            on_details=self._show_build_details,
            on_cancel=self._cancel_active_build,
        )
        self._status_bar.grid(row=1, column=0, sticky="ew")

    def _show_build_details(self) -> None:
        snap = self._progress_manager.snapshot
        if snap.job_kind == "registry_export":
            self._show_page("master.trading_days")
            return
        if snap.job_kind == "research_campaign":
            self._show_page("registry.models")
            return
        self._show_page("master.create")
        try:
            if hasattr(self.create_panel, "progress") and hasattr(self.create_panel.progress, "_notebook"):
                self.create_panel.progress._notebook.select(0)
        except Exception:
            pass

    def _cancel_active_build(self) -> None:
        from tkinter import messagebox

        if self.create_panel.build_running:
            if messagebox.askyesno("Cancel build", "Cancel the running dataset build?"):
                self._progress_manager.request_cancel()
            return
        if self.master_panel.registry_export_running():
            messagebox.showinfo(
                "Cancel export",
                "Registry export cannot be cancelled once started. "
                "It will finish in the background.",
            )

    def _project_label_text(self) -> str:
        name = os.path.basename(self.chart_dir.rstrip("\\/")) or self.chart_dir
        return f"Project: {name}"

    def _sync_window_title(self, *, inventory_updated: str | None = None) -> None:
        section = ""
        label = _PAGE_TITLES.get(self._current_page, self._current_page)
        if self._current_page == "builder.create":
            label = self.create_model_shell.page_title()
        for sec, items in _NAV_SECTIONS:
            if any(k == self._current_page for k, _ in items):
                section = sec
                break
        base = f"ML Research Studio — {section} / {label}" if section else f"ML Research Studio — {label}"
        if inventory_updated:
            base = f"ML Research Studio — inventory {inventory_updated}"
        self.title(f"{base}  [{os.path.basename(self.chart_dir)}]")

    def _build_shell(self) -> None:
        shell = ttk.Frame(self)
        shell.grid(row=0, column=0, sticky="nsew")
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)

        nav_outer = ttk.Frame(body, width=157)
        nav_outer.pack(side="left", fill="y")
        nav_outer.pack_propagate(False)

        canvas = tk.Canvas(nav_outer, highlightthickness=0, width=150)
        scroll = ttk.Scrollbar(nav_outer, orient="vertical", command=canvas.yview)
        nav = ttk.Frame(canvas, padding=(8, 10))
        nav.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=nav, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(nav, text="ML Research Studio", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))

        for section, items in _NAV_SECTIONS:
            ttk.Label(nav, text=section, font=("Segoe UI", 9, "bold"), foreground="#58a6ff").pack(
                anchor="w", pady=(10, 4),
            )
            for key, label in items:
                if key == "builder.create":
                    cmd = self._open_create_model_page
                else:
                    cmd = lambda k=key: self._show_page(k)
                btn = ttk.Button(nav, text=label, command=cmd)
                btn.pack(fill="x", pady=2)
                self._nav_btns[key] = btn

        ttk.Separator(nav, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(nav, textvariable=self._project_label_var, foreground="#58a6ff", wraplength=136, justify="left").pack(anchor="w")
        ttk.Label(nav, text="Standalone\nNo chart server", foreground="#888", justify="left").pack(anchor="w", pady=(6, 0))

        self._content = ttk.Frame(body, padding=(4, 8))
        self._content.pack(side="left", fill="both", expand=True)

        self.create_panel = CreateDatasetPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_data_changed=self._on_master_data_changed,
            on_inventory_loaded=lambda updated: self._sync_window_title(inventory_updated=updated),
        )
        self.master_panel = MasterDataPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_open_create_dataset=lambda: self._show_page("master.create"),
            on_registry_created=self._on_registry_dataset_created,
            on_open_model_builder=self._open_model_builder_from_feature_selection,
        )
        self.registry_panel = RegistryPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_open_builder=self._open_model_builder_dataset,
            on_compare_datasets=self._open_dataset_comparison,
        )
        self.features_panel = FeatureRegistryPanel(self._content, chart_dir=self.chart_dir)
        self.settings_panel = SettingsPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_project_changed=self.set_project_folder,
        )
        self.nifty_history_bar_panel = NiftyHistoryBarPanel(
            self._content,
            chart_dir=self.chart_dir,
        )
        self.ormp_overview_panel = NiftyOrmpOverviewPanel(
            self._content,
            chart_dir=self.chart_dir,
        )

        self.strategies_panel = StrategiesPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_open_model=self._open_model_registry,
            on_open_fold_replay=self._open_fold_replay,
        )
        self.strategy_registry_panel = self.strategies_panel.strategies
        self.prediction_runs_panel = self.strategies_panel.prediction_runs
        self.strategy_simulation_panel = self.strategies_panel.simulation
        self.research_lab_panel = self.strategies_panel.leaderboard
        self.fold_replay_panel = FoldReplayPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_open_experiment_planner=self._open_experiment_planner,
        )
        self.experiment_planner_panel = ExperimentPlannerPanel(
            self._content,
            chart_dir=self.chart_dir,
        )
        self.research_program_panel = ResearchProgramPanel(
            self._content,
            chart_dir=self.chart_dir,
        )

        self.model_registry_panel = ModelRegistryPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_open_prediction_runs=self._open_prediction_runs_for_model,
            on_compare_models=self._open_model_comparison,
            on_open_fold_research=self._open_fold_replay,
            on_lifecycle=self._open_model_builder_lifecycle,
            on_builder_features=self._open_model_builder_with_features,
            on_build_package_classifier=self._open_model_builder_package_classifier,
        )

        self.create_model_shell = CreateModelShell(
            self._content,
            chart_dir=self.chart_dir,
            on_open_registry=self._open_model_registry,
            on_title_changed=self._sync_window_title,
            on_label_run_created=self._on_label_run_created,
            on_open_create_model=self._apply_label_run_to_create_model,
        )
        self.create_model_panel = self.create_model_shell.create
        self.outcome_label_engine_panel = self.create_model_shell.outcome_labels

        self.comparison_panel = ComparisonPanel(
            self._content,
            chart_dir=self.chart_dir,
        )
        self.model_comparison_panel = self.comparison_panel.model
        self.fold_comparison_panel = self.comparison_panel.fold
        self.dataset_comparison_panel = self.comparison_panel.dataset

        self.feature_studio_panel = FeatureStudioPanel(
            self._content,
            chart_dir=self.chart_dir,
            on_create_model=self._open_model_builder_with_features,
        )

        self.feature_intelligence_panel = FeatureIntelligenceStudioPanel(
            self._content,
            chart_dir=self.chart_dir,
        )

        self.model_explorer_panel = ModelExplorerPanel(
            self._content,
            chart_dir=self.chart_dir,
        )

        self._strategy_panels: dict[str, Any] = {
            "strategy.strategies": self.strategies_panel,
            "strategy.replay": self.fold_replay_panel,
            "strategy.programs": self.research_program_panel,
            "strategy.experiments": self.experiment_planner_panel,
        }

        self._pages: dict[str, Any] = {
            "master.create": self.create_panel,
            "master.trading_days": self.master_panel,
            # Aliases (no longer separate nav items)
            "master.validation": self.registry_panel,
            "master.metadata": self.master_panel,
            "registry.datasets": self.registry_panel,
            "registry.features": self.features_panel,
            "registry.models": self.model_registry_panel,
            "builder.create": self.create_model_shell,
            # Backward-compatible alias → Create Model / Outcome Labels tab
            "builder.ole": self.create_model_shell,
            "builder.comparison": self.comparison_panel,
            # Backward-compatible aliases → Comparison shell
            "builder.compare": self.comparison_panel,
            "builder.fold_compare": self.comparison_panel,
            "registry.dataset_compare": self.comparison_panel,
            "builder.feature_studio": self.feature_studio_panel,
            "builder.feature_intelligence": self.feature_intelligence_panel,
            "builder.model_explorer": self.model_explorer_panel,
            # Backward-compatible aliases → Feature Studio shell
            "builder.importance": self.feature_studio_panel,
            "builder.distribution": self.feature_studio_panel,
            "builder.drift": self.feature_studio_panel,
            "builder.multi_compare": self.feature_studio_panel,
            "builder.diagnostics": self.feature_studio_panel,
            "builder.experiment_planner": self.feature_studio_panel,
            "strategy.strategies": self.strategies_panel,
            # Backward-compatible aliases → Strategies shell
            "predictions.runs": self.strategies_panel,
            "strategy.simulation": self.strategies_panel,
            "strategy.leaderboard": self.strategies_panel,
            "strategy.replay": self.fold_replay_panel,
            "strategy.programs": self.research_program_panel,
            "strategy.experiments": self.experiment_planner_panel,
            "settings.main": self.settings_panel,
            "history.nifty": self.nifty_history_bar_panel,
            "ormp.overview": self.ormp_overview_panel,
            # Backward-compatible aliases
            "settings.nifty_history": self.nifty_history_bar_panel,
            "ormp.builds": self.ormp_overview_panel,
            "ormp.dataset_builder": self.ormp_overview_panel,
            "ormp.feature_explorer": self.ormp_overview_panel,
        }
        self._pages.update({k: v for k, v in self._strategy_panels.items() if k not in (
            "strategy.strategies", "strategy.replay", "strategy.programs", "strategy.experiments",
        )})

    def _show_page(self, key: str) -> None:
        if key == "master.validation":
            key = "registry.datasets"
        elif key == "master.metadata":
            key = "master.trading_days"
        studio_tab = _FEATURE_STUDIO_TAB_ALIASES.get(key)
        if studio_tab is not None:
            key = "builder.feature_studio"
        comparison_tab = _COMPARISON_TAB_ALIASES.get(key)
        if comparison_tab is not None:
            key = "builder.comparison"
        strategy_tab = _STRATEGY_TAB_ALIASES.get(key)
        if strategy_tab is not None:
            key = "strategy.strategies"
        create_model_tab = _CREATE_MODEL_TAB_ALIASES.get(key)
        if create_model_tab is not None:
            key = "builder.create"
        if key == self._current_page:
            if comparison_tab is not None:
                self.comparison_panel.select_tab(comparison_tab)
            elif studio_tab is not None:
                self.feature_studio_panel.select_tab(studio_tab)
            elif strategy_tab is not None:
                self.strategies_panel.select_tab(strategy_tab)
            elif create_model_tab is not None:
                self.create_model_shell.select_tab(create_model_tab)
            return
        for page in set(self._pages.values()):
            page.pack_forget()
        page = self._pages.get(key)
        if page is None:
            return
        page.pack(fill="both", expand=True)
        self._current_page = key
        for nav_key, btn in self._nav_btns.items():
            try:
                btn.state(["!pressed"])
            except tk.TclError:
                pass
        if key in self._nav_btns:
            try:
                self._nav_btns[key].state(["pressed"])
            except tk.TclError:
                pass
        if comparison_tab is not None:
            self.comparison_panel.select_tab(comparison_tab)
        elif studio_tab is not None:
            self.feature_studio_panel.select_tab(studio_tab)
        elif strategy_tab is not None:
            self.strategies_panel.select_tab(strategy_tab)
        elif create_model_tab is not None:
            self.create_model_shell.select_tab(create_model_tab)
        else:
            on_show = getattr(page, "on_show", None)
            if callable(on_show):
                on_show()
        self._sync_window_title()
        # Remember the last-visited nav page so relaunching the app opens
        # back up where the user left off (only for real, navigable pages).
        if key in self._nav_btns:
            self._ui_state.set(_UI_STATE_LAST_PAGE_KEY, key, widget=self)

    def _open_create_model_page(self) -> None:
        if self._current_page == "builder.create":
            self.create_model_shell.select_tab("create", from_nav=False)
            self.create_model_panel.begin_new_model()
            self._sync_window_title()
            return
        self._show_create_model_tab()

    def _show_create_model_tab(self) -> None:
        """Navigate to Create Model and ensure the Create Model tab is active."""
        # Switch tab without firing Create Model on_show first; page show (or caller)
        # owns the reset / lifecycle load so we don't wipe a just-prepared form.
        self.create_model_shell.select_tab("create", from_nav=False)
        if self._current_page != "builder.create":
            self._show_page("builder.create")
        else:
            self._sync_window_title()

    def _open_outcome_label_engine(self, prefill: dict[str, Any] | None = None) -> None:
        self._show_page("builder.ole")
        try:
            self.outcome_label_engine_panel.apply_prefill(prefill)
        except Exception:
            pass

    def _on_label_run_created(self, run_id: str) -> None:
        try:
            self.create_model_panel.state.label_run_id = str(run_id or "")
            self.create_model_panel._refresh_label_runs()
        except Exception:
            pass

    def _apply_label_run_to_create_model(self, payload: dict[str, Any] | None = None) -> None:
        """Apply OLE selection onto Create Model (shell already switches to Create tab)."""
        payload = payload or {}
        try:
            ds = str(payload.get("dataset") or "").strip()
            run_id = str(payload.get("label_run_id") or "").strip()
            if ds:
                self.create_model_panel.state.dataset = ds
            if run_id:
                self.create_model_panel.state.label_run_id = run_id
            self.create_model_panel._refresh_label_runs()
            if run_id:
                self.create_model_panel._on_label_run_selected()
        except Exception:
            pass

    def _open_create_model_with_label_run(self, payload: dict[str, Any] | None = None) -> None:
        self._show_create_model_tab()
        self._apply_label_run_to_create_model(payload)

    def set_project_folder(self, chart_dir: str) -> None:
        chart_dir = os.path.abspath(chart_dir)
        ensure_project_data_dir(chart_dir)
        save_project_config(chart_dir)
        self.chart_dir = chart_dir
        self._project_label_var.set(self._project_label_text())
        for page in set(self._pages.values()):
            if hasattr(page, "chart_dir"):
                page.chart_dir = chart_dir
            if hasattr(page, "set_chart_dir"):
                page.set_chart_dir(chart_dir)
            elif hasattr(page, "_data_dir"):
                page._data_dir = chart_data_dir(chart_dir)
            runner = getattr(page, "_runner", None)
            if runner is not None and hasattr(runner, "chart_dir"):
                runner.chart_dir = chart_dir
        if self._current_page == "master.create":
            self.create_panel.load_inventory(lazy=True)
        page = self._pages.get(self._current_page)
        if page is not None:
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                refresh()
            else:
                on_show = getattr(page, "on_show", None)
                if callable(on_show):
                    on_show()
        self._sync_window_title()

    def _on_master_data_changed(self) -> None:
        if self._current_page in ("master.trading_days", "master.metadata"):
            self.master_panel.refresh()
        if self._current_page in ("master.validation", "registry.datasets"):
            self.registry_panel.refresh_registry()

    def _on_registry_dataset_created(self, dataset_name: str) -> None:
        self.registry_panel.refresh_registry()
        self.registry_panel._selected_name = dataset_name

    def _open_model_registry(self, model_name: str) -> None:
        self.model_registry_panel._selected_name = model_name
        self._show_page("registry.models")
        self.model_registry_panel.select_model(model_name)

    def _open_prediction_runs_for_model(self, model_name: str) -> None:
        self._show_page("predictions.runs")
        self.prediction_runs_panel.set_model_filter(model_name)

    def _open_model_comparison(self, model_a: str, model_b: str) -> None:
        panel = self.model_comparison_panel
        panel.open_with_models(model_a, model_b)
        if self._current_page == "builder.comparison":
            self.comparison_panel.select_tab("model")
            panel.activate_pending()
        else:
            self._show_page("builder.compare")  # alias → Comparison / Model

    def _open_dataset_comparison(self, dataset_a: str, dataset_b: str) -> None:
        panel = self.dataset_comparison_panel
        panel.open_with_datasets(dataset_a, dataset_b)
        if self._current_page == "builder.comparison":
            self.comparison_panel.select_tab("dataset")
            panel.activate_pending()
        else:
            self._show_page("registry.dataset_compare")  # alias → Comparison / Dataset

    def _open_model_builder_lifecycle(self, model_name: str, mode: str) -> None:
        self.create_model_panel.prepare_lifecycle_open()
        self._show_create_model_tab()
        self.create_model_panel.load_lifecycle_preset(model_name, mode)

    def _open_model_builder_with_features(
        self,
        model_name: str,
        features: list[str],
        dataset: str | None,
    ) -> None:
        self._show_create_model_tab()
        self.create_model_panel.import_feature_preset(
            features=features,
            dataset=dataset,
            source_model=model_name,
        )

    def _open_model_builder_from_feature_selection(
        self,
        *,
        features: list[str],
        dataset: str | None = None,
        source_model: str | None = None,
        analysis_feature_selection: dict | None = None,
    ) -> None:
        """Research Lab Final Feature Dataset → Create Model (pre-selected)."""
        self._show_create_model_tab()
        self.create_model_panel.import_feature_preset(
            features=list(features or []),
            dataset=dataset,
            source_model=source_model or "FeatureSelection",
            analysis_feature_selection=analysis_feature_selection,
            persist=True,
        )

    def _open_model_builder_dataset(self, dataset_name: str) -> None:
        self._show_create_model_tab()
        self.create_model_panel.open_with_dataset(dataset_name)

    def _open_model_builder_package_classifier(
        self,
        *,
        dataset_name: str,
        target: str,
        features: list[str] | None = None,
        source_model: str | None = None,
        ladder_label: str | None = None,
    ) -> None:
        self._show_create_model_tab()
        self.create_model_panel.open_package_classifier(
            dataset_name=dataset_name,
            target=target,
            features=features,
            source_model=source_model,
            ladder_label=ladder_label,
        )

    def _open_hit_confidence_builder(
        self,
        *,
        dataset_name: str,
        features: list[str] | None = None,
        parent_model: str | None = None,
        lab_db_path: str | None = None,
    ) -> None:
        self.create_model_panel.prepare_lifecycle_open()
        self._show_create_model_tab()
        self.create_model_panel.open_hit_confidence(
            dataset_name=dataset_name,
            features=features,
            parent_model=parent_model,
            lab_db_path=lab_db_path,
        )

    def _open_fold_replay(
        self,
        prediction_run_id: str,
        fold_id: str,
        strategy_run_id: str | None = None,
    ) -> None:
        self.fold_replay_panel.prefill(prediction_run_id, fold_id, strategy_run_id)
        self._show_page("strategy.replay")

    def _open_experiment_planner(
        self,
        report: dict[str, Any] | None = None,
        *,
        proposal_id: str | None = None,
    ) -> None:
        if proposal_id:
            self.experiment_planner_panel.prefill_from_proposal(proposal_id)
        elif report:
            self.experiment_planner_panel.prefill_from_report(report)
        self._show_page("strategy.experiments")

    def _poll_progress(self) -> None:
        from .gil_monitor import MainThreadLagTracker, gil_monitor_enabled

        if gil_monitor_enabled(self.chart_dir) and self.create_panel.needs_progress_poll:
            MainThreadLagTracker.instance().tick()
        self._progress_manager.tick()
        if self.create_panel.needs_progress_poll:
            self.create_panel.poll_progress(
                ui_visible=self._current_page == "master.create",
            )
        if (
            self.master_panel.registry_export_running()
            or self.master_panel.auto_feature_build_running()
            or self._current_page in (
                "master.trading_days",
                "master.metadata",
            )
        ):
            self.master_panel.poll_registry_export()
        if self._current_page in ("master.validation", "registry.datasets"):
            self.registry_panel.poll_jobs()
        if self._current_page == "registry.features":
            self.features_panel.poll_jobs()
        if self._current_page == "strategy.experiments":
            self.experiment_planner_panel.poll_job_progress()
        data_dir = chart_data_dir(self.chart_dir)
        self._research_coordinator.tick(data_dir)
        if self._current_page == "strategy.programs":
            self.research_program_panel.on_coordinator_tick()
        self.after(200, self._poll_progress)


# Backward-compatible alias
MasterDatasetApp = MLResearchStudioApp


def main(*, chart_dir: str | None = None) -> None:
    import multiprocessing

    multiprocessing.freeze_support()
    from path_config import ensure_ml_studio_paths

    ensure_ml_studio_paths()
    resolved = resolve_chart_dir(cli_chart_dir=chart_dir)
    if resolved and resolved not in sys.path:
        sys.path.append(resolved)
    app = MLResearchStudioApp(chart_dir=chart_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
