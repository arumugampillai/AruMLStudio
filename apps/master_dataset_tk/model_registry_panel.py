"""Model Registry — models list with web-parity rich detail tabs (Tk)."""

from __future__ import annotations

import json
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable


from .build_service import chart_data_dir
from .model_registry_detail import (
    render_artifacts,
    render_classification_confusion,
    render_holdout_performance,
    render_lifecycle,
    render_model_metrics,
    render_model_research,
    render_overview,
    render_retrain,
    render_selected_features,
    render_threshold_analysis,
    render_validation_metrics,
    render_walk_forward,
)
from .lazy_panel import LazyLoadMixin
from .model_registry_strategy import ModelRegistryStrategyPanel
from .model_research_leaderboard_panel import ModelResearchLeaderboardPanel
from .model_registry_widgets import ACCENT, COL_MUTED, COL_WARN, ScrollableFrame, fmt_num, fmt_pct, section_desc, section_title
from .ui_state import get_ui_state_manager
from chain_replay_ml.dataset_builder.expected_spec import format_sampling_interval_label


def _fmt_num(v: Any) -> str:
    return fmt_num(v)


def _fmt_pct(v: Any, digits: int = 2) -> str:
    if v is None or v == "" or v == "—":
        return "—"
    try:
        f = float(v)
        if 0.0 < f <= 1.0:
            f = f * 100.0
        return f"{f:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"



def _fmt_roc_auc(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_validation_strategy(v: Any) -> str:
    """Compact validation label for the registry list, e.g. Walk Forward (5 folds) → WF (5)."""
    text = str(v or "").strip()
    if not text or text == "—":
        return "—"
    folds = None
    m = re.search(r"\((\d+)\s*folds?\)", text, flags=re.IGNORECASE)
    if m:
        folds = m.group(1)
    lower = text.lower()
    if "rolling" in lower:
        short = "RW"
    elif "walk" in lower and "forward" in lower:
        short = "WF"
    elif "time" in lower and "series" in lower:
        short = "TS"
    else:
        short = text.split("(")[0].strip() or text
    if folds:
        return f"{short} ({folds})"
    return short


def _fmt_target(v: Any) -> str:
    """Compact target for the registry list, e.g. future_ltp_5m → F-5m."""
    t = str(v or "").strip()
    if not t or t == "—":
        return "—"
    if t.startswith("future_ltp_"):
        hor = t[len("future_ltp_") :]
        return f"F-{hor}" if hor else "F"
    if t.startswith("label_up_"):
        rest = t[len("label_up_") :]
        return f"U-{rest}" if rest else "U"
    if t.startswith("ormp_return_"):
        rest = t[len("ormp_return_") :]
        for suffix in ("_points", "_percent"):
            if rest.endswith(suffix):
                rest = rest[: -len(suffix)]
                break
        return f"OR-{rest}" if rest else "OR"
    if t.startswith("ormp_direction_"):
        rest = t[len("ormp_direction_") :]
        return f"OD-{rest}" if rest else "OD"
    if t == "label_id":
        return "L-id"
    return t


def _fmt_fc(v: Any) -> str:
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def _fmt_interval(v: Any) -> str:
    return format_sampling_interval_label(v) or "—"


def _fmt_size_mb(n: Any) -> str:
    try:
        val = float(n or 0)
    except (TypeError, ValueError):
        return "—"
    if val <= 0:
        return "—"
    return f"{val / (1024 * 1024):.1f} MB"


class ModelRegistryPanel(ttk.Frame, LazyLoadMixin):
    """Models list with detail tabs matching web Model Registry."""

    _TAB_OVERVIEW = "overview"
    _TAB_MODEL_METRICS = "model_metrics"
    _TAB_THRESHOLD_ANALYSIS = "threshold_analysis"
    _TAB_VALIDATION = "validation"
    _TAB_WALK_FORWARD = "walk_forward"
    _TAB_HOLDOUT_PERFORMANCE = "holdout_performance"
    _TAB_CLASSIFICATION = "classification"
    _TAB_PREDICTION_RUNS = "prediction_runs"
    _TAB_FEATURES = "features"
    _TAB_RETRAIN = "retrain"
    _TAB_LIFECYCLE = "lifecycle"
    _TAB_STRATEGY = "strategy"
    _TAB_RESEARCH = "research"
    _TAB_ARTIFACTS = "artifacts"

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_prediction_runs: Callable[[str], None] | None = None,
        on_compare_models: Callable[[str, str], None] | None = None,
        on_open_fold_research: Callable[[str, str, str | None], None] | None = None,
        on_lifecycle: Callable[[str, str], None] | None = None,
        on_builder_features: Callable[[str, list[str], str | None], None] | None = None,
        on_build_package_classifier: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_prediction_runs = on_open_prediction_runs
        self._on_compare_models = on_compare_models
        self._on_open_fold_research = on_open_fold_research
        self._on_lifecycle = on_lifecycle
        self._on_builder_features = on_builder_features
        self._on_build_package_classifier = on_build_package_classifier
        self._rows: list[dict[str, Any]] = []
        self._selected_name: str | None = None
        self._models_family: str = "regression"
        self._detail_doc: dict[str, Any] | None = None
        self._status_var = tk.StringVar(value="")
        self._header_var = tk.StringVar(value="Select a model to view details")
        self._tab_scrolls: dict[str, ScrollableFrame] = {}
        self._tab_ids: dict[str, str] = {}
        self._tabs_rendered: set[str] = set()
        self._detail_visible_tabs: list[str] = []
        self._pred_run_map: dict[str, str] = {}
        self._pred_run_json: str = ""
        self._classification_member_name: str | None = None
        self._classification_member_doc: dict[str, Any] | None = None
        self._classification_member_scrolls: dict[str, ScrollableFrame] = {}
        self._classification_member_status: tk.StringVar | None = None
        self._ui_state = get_ui_state_manager()
        self._build_ui()
        # Detail-tab index is intentionally not persisted: _populate_detail_tabs
        # rebuilds the tab set per model and always re-selects Overview by
        # design, so restoring a stale index would fight that on every load.
        self._ui_state.bind_notebook(self._models_family_nb, "model_registry.family_tab")
        self._restore_selected_model()
        self._lazy_init()
        from .research_campaign_coordinator import get_research_campaign_coordinator

        self._research_coordinator = get_research_campaign_coordinator()
        self._research_coordinator.subscribe(self._on_research_coordinator_update)

    def _data_dir(self) -> str:
        if not self.chart_dir:
            return ""
        if os.path.exists(os.path.join(self.chart_dir, "analysis.db")):
            return self.chart_dir
        return chart_data_dir(self.chart_dir)


    def _restore_selected_model(self) -> None:
        from chain_replay_ml.training.registry import get_active_model

        try:
            self._selected_name = get_active_model(self._data_dir())
        except Exception:
            self._selected_name = None

    def _persist_selected_model(self, model_name: str | None) -> None:
        if not model_name:
            return
        from chain_replay_ml.training.registry import set_active_model

        try:
            set_active_model(self._data_dir(), model_name)
        except Exception:
            pass

    def on_show(self) -> None:
        self.refresh_models(select_first=not bool(self._selected_name), lazy=True)

    def _on_research_coordinator_update(self) -> None:
        if self._selected_name and self._detail_doc:
            try:
                self._load_detail()
            except Exception:
                pass

    def select_model(self, model_name: str) -> None:
        self._selected_name = model_name
        self._persist_selected_model(model_name)
        # Switch list tab to the model's family when known from current rows.
        family = None
        for r in self._rows:
            if str(r.get("model_name") or r.get("name") or "") == model_name:
                from chain_replay_ml.training.registry import resolve_model_registry_family

                family = resolve_model_registry_family(r)
                break
        if family:
            self._set_models_family_tab(family)
        self.refresh_models()
        if model_name in self.models_tree.get_children():
            self.models_tree.selection_set(model_name)
            self.models_tree.see(model_name)
            self._update_compare_button()
            self._load_detail()
            try:
                self._detail_notebook.select(self._tab_frames[self._TAB_OVERVIEW])
            except tk.TclError:
                pass

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Refresh", command=self.refresh_models).pack(side="left", padx=4)
        toolbar_right = ttk.Frame(toolbar)
        toolbar_right.pack(side="right")
        if self._on_open_prediction_runs:
            self._pred_runs_btn = ttk.Button(toolbar_right, text="Open Prediction Runs", command=self._open_prediction_runs)
            self._pred_runs_btn.pack(side="right", padx=4)
        self._research_leaderboard_btn = ttk.Button(
            toolbar_right,
            text="Research Leaderboard",
            command=self._open_research_leaderboard,
        )
        self._research_leaderboard_btn.pack(side="right", padx=4)
        if self._on_compare_models:
            self._compare_btn = ttk.Button(toolbar_right, text="Compare", command=self._open_compare)

        # Phase 4C.4: Faceted Taxonomy & Population Filter Bar
        facet_bar = ttk.LabelFrame(self, text="Taxonomy & Population Filters", padding=(6, 4))
        facet_bar.pack(fill="x", padx=8, pady=(0, 4))

        # Task Type Filter
        ttk.Label(facet_bar, text="Task:").pack(side="left", padx=(2, 2))
        self._filter_task_var = tk.StringVar(value="All Tasks")
        self._task_combo = ttk.Combobox(
            facet_bar,
            textvariable=self._filter_task_var,
            state="readonly",
            width=18,
            values=[
                "All Tasks",
                "DIRECTION_CLASSIFIER",
                "REGIME_CLASSIFIER",
                "REGRESSION",
                "TRIPLE_BARRIER",
                "CONFIDENCE_CLASSIFIER",
                "VOLATILITY_ESTIMATOR",
            ],
        )
        self._task_combo.pack(side="left", padx=(0, 6))
        self._task_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_models_tree())

        # Regime Filter
        ttk.Label(facet_bar, text="Regime:").pack(side="left", padx=(2, 2))
        self._filter_regime_var = tk.StringVar(value="All Regimes")
        self._regime_combo = ttk.Combobox(
            facet_bar,
            textvariable=self._filter_regime_var,
            state="readonly",
            width=22,
            values=self._get_regime_filter_options(),
        )
        self._regime_combo.pack(side="left", padx=(0, 6))
        self._regime_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_models_tree())

        # Population Filter
        ttk.Label(facet_bar, text="Population:").pack(side="left", padx=(2, 2))
        self._filter_pop_var = tk.StringVar(value="All Populations")
        self._pop_combo = ttk.Combobox(
            facet_bar,
            textvariable=self._filter_pop_var,
            state="readonly",
            width=15,
            values=["All Populations", "CHAMPION", "CHALLENGER", "VALIDATED", "EXPERIMENTAL"],
        )
        self._pop_combo.pack(side="left", padx=(0, 6))
        self._pop_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_models_tree())

        # Lifecycle Filter
        ttk.Label(facet_bar, text="Lifecycle:").pack(side="left", padx=(2, 2))
        self._filter_lc_var = tk.StringVar(value="All Statuses")
        self._lc_combo = ttk.Combobox(
            facet_bar,
            textvariable=self._filter_lc_var,
            state="readonly",
            width=12,
            values=["All Statuses", "ACTIVE", "CANDIDATE", "DEGRADED", "DEPRECATED", "RETIRED"],
        )
        self._lc_combo.pack(side="left", padx=(0, 6))
        self._lc_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_models_tree())

        ttk.Button(facet_bar, text="Reset", command=self._reset_filters).pack(side="right", padx=2)

        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=2)

        self._main_paned = paned
        self._table_frame = table_frame

        self._models_family_nb = ttk.Notebook(table_frame)
        self._models_family_nb.pack(fill="x")
        self._models_family_tab_ids: dict[str, int] = {}
        for idx, (key, label) in enumerate(
            (
                ("regression", "Regression"),
                ("classifier", "Classifier"),
                ("triple_barrier", "Triple Barrier"),
            )
        ):
            frame = ttk.Frame(self._models_family_nb)
            self._models_family_nb.add(frame, text=label)
            self._models_family_tab_ids[key] = idx
        self._models_family_nb.bind("<<NotebookTabChanged>>", self._on_models_family_tab_changed)

        self._tree_host = ttk.Frame(table_frame)
        self._tree_host.pack(fill="both", expand=True)
        tree_host = self._tree_host
        cols = (
            "name",
            "task",
            "regime",
            "pop",
            "strategy",
            "dataset",
            "ds_st",
            "label_run",
            "lab_st",
            "target",
            "fc",
            "interval",
            "m1",
            "m2",
            "m3",
            "m4",
            "size",
            "research",
            "delete",
        )
        self.models_tree = ttk.Treeview(
            tree_host,
            columns=cols,
            show="headings",
            height=16,
            selectmode="extended",
        )
        for c, w, label in (
            ("name", 160, "Model"),
            ("task", 75, "Task"),
            ("regime", 90, "Regime"),
            ("pop", 100, "Population"),
            ("strategy", 65, "Valid"),
            ("dataset", 220, "Dataset"),
            ("ds_st", 50, "DS"),
            ("label_run", 100, "Label Run"),
            ("lab_st", 50, "Label"),
            ("target", 56, "Trgt"),
            ("fc", 40, "FC"),
            ("interval", 48, "Interval"),
            ("m1", 64, "MAE"),
            ("m2", 64, "RMSE"),
            ("m3", 64, "Dir %"),
            ("m4", 72, "ROC-AUC"),
            ("size", 52, "Size"),
            ("research", 64, "Research"),
            ("delete", 52, "Delete"),
        ):
            self.models_tree.heading(c, text=label)
            anchor = "center"
            if c in ("name", "dataset", "target", "label_run"):
                anchor = "w"
            self.models_tree.column(c, width=w, anchor=anchor, stretch=(c == "name"))
        self.models_tree.pack(side="left", fill="both", expand=True)
        self._sync_models_metric_columns()
        sb = ttk.Scrollbar(tree_host, orient="vertical", command=self.models_tree.yview)
        sb.pack(side="right", fill="y")
        self.models_tree.configure(yscrollcommand=sb.set)
        self.models_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_models_selection_changed())
        self.models_tree.bind("<Button-1>", self._on_models_tree_click)

        detail_outer = ttk.Frame(paned)
        self._detail_outer = detail_outer
        paned.add(detail_outer, weight=3)

        header = ttk.Frame(detail_outer, padding=(4, 6))
        header.pack(fill="x")
        ttk.Label(header, textvariable=self._header_var, font=("Segoe UI", 11, "bold"), foreground=ACCENT).pack(anchor="w")

        self._detail_notebook = ttk.Notebook(detail_outer)
        self._detail_notebook.pack(fill="both", expand=True)
        self._detail_notebook.bind("<<NotebookTabChanged>>", self._on_detail_tab_changed)

        self._tab_frames: dict[str, ttk.Frame] = {}
        for tab_id, label in (
            (self._TAB_OVERVIEW, "Overview"),
            (self._TAB_MODEL_METRICS, "Model Metrics"),
            (self._TAB_THRESHOLD_ANALYSIS, "Threshold Analysis"),
            (self._TAB_VALIDATION, "Validation Metrics"),
            (self._TAB_WALK_FORWARD, "Walk Forward"),
            (self._TAB_HOLDOUT_PERFORMANCE, "Holdout Performance"),
            (self._TAB_CLASSIFICATION, "Classification"),
            (self._TAB_FEATURES, "Selected Features"),
            (self._TAB_RETRAIN, "Retrain"),
            (self._TAB_LIFECYCLE, "Lifecycle"),
            (self._TAB_PREDICTION_RUNS, "Prediction Runs"),
            (self._TAB_STRATEGY, "Strategy"),
            (self._TAB_RESEARCH, "Research"),
            (self._TAB_ARTIFACTS, "Artifacts"),
        ):
            frame = ttk.Frame(self._detail_notebook)
            self._tab_frames[tab_id] = frame
            self._tab_ids[tab_id] = label

        self._build_prediction_runs_tab()
        self._build_strategy_tab()

        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(anchor="w", padx=10, pady=(0, 4))

    def _models_family_from_notebook(self) -> str:
        try:
            idx = int(self._models_family_nb.index(self._models_family_nb.select()))
        except Exception:
            return self._models_family or "regression"
        for key, tab_idx in self._models_family_tab_ids.items():
            if tab_idx == idx:
                return key
        return "regression"

    def _set_models_family_tab(self, family: str) -> None:
        key = str(family or "regression").strip().lower()
        if key not in self._models_family_tab_ids:
            key = "regression"
        self._models_family = key
        self._sync_models_metric_columns()
        try:
            if self._models_family_from_notebook() != key:
                self._models_family_nb.select(self._models_family_tab_ids[key])
        except tk.TclError:
            pass

    def _on_models_family_tab_changed(self, _event: object | None = None) -> None:
        self._models_family = self._models_family_from_notebook()
        self._sync_models_metric_columns()
        self._populate_models_tree(select_first=True)



    def _sync_models_metric_columns(self) -> None:
        """Swap list metric headers / visible columns for Regression vs Classifier vs Triple Barrier."""
        family = self._models_family or "regression"
        is_tb = family == "triple_barrier"
        is_cls = family == "classifier"

        if is_tb or is_cls:
            headings = (
                ("m1", 72, "Precision"),
                ("m2", 64, "Recall"),
                ("m3", 56, "F1"),
                ("m4", 72, "ROC-AUC"),
            )
            if is_tb:
                # TB: drop Dataset / Target / Label status — Label Run stays.
                display = (
                    "name",
                    "strategy",
                    "ds_st",
                    "label_run",
                    "fc",
                    "interval",
                    "m1",
                    "m2",
                    "m3",
                    "m4",
                    "size",
                    "research",
                    "delete",
                )
            else:
                # Classifier: show dataset, target, fc, interval, m1..m4
                display = (
                    "name",
                    "strategy",
                    "dataset",
                    "target",
                    "fc",
                    "interval",
                    "m1",
                    "m2",
                    "m3",
                    "m4",
                    "size",
                    "research",
                    "delete",
                )
            name_width = 180
        else:
            headings = (
                ("m1", 64, "MAE"),
                ("m2", 64, "RMSE"),
                ("m3", 64, "Dir %"),
                ("m4", 0, ""),
            )
            # Label Run / DS status columns are OLE/TB-focused — hide on Regression.
            display = (
                "name",
                "strategy",
                "dataset",
                "target",
                "fc",
                "interval",
                "m1",
                "m2",
                "m3",
                "size",
                "research",
                "delete",
            )
            name_width = 180

        for cid, width, label in headings:
            self.models_tree.heading(cid, text=label)
            self.models_tree.column(
                cid,
                width=width,
                minwidth=0 if width == 0 else 40,
                anchor="center",
                stretch=False,
            )
        self.models_tree.column("name", width=name_width, anchor="w", stretch=True)
        self.models_tree.heading("task", text="Task")
        self.models_tree.column("task", width=75, anchor="center", stretch=False)
        self.models_tree.heading("regime", text="Regime")
        self.models_tree.column("regime", width=85, anchor="center", stretch=False)
        self.models_tree.heading("pop", text="Population")
        self.models_tree.column("pop", width=105, anchor="center", stretch=False)
        self.models_tree.heading("strategy", text="Valid")
        self.models_tree.column("strategy", width=65, anchor="center", stretch=False)
        self.models_tree.heading("target", text="Trgt")
        self.models_tree.column("target", width=56, anchor="w", stretch=False)
        # Dataset needs room for full names (e.g. analysis_206r_193p_3s_…).
        if not is_tb:
            self.models_tree.column("dataset", width=240, minwidth=180, anchor="w", stretch=False)
        try:
            self.models_tree.configure(displaycolumns=display)
        except tk.TclError:
            pass

    def _get_regime_filter_options(self) -> list[str]:
        options = ["All Regimes"]
        try:
            from chain_replay_ml.model_taxonomy import list_regimes
            data_dir = self._data_dir()
            if data_dir:
                regs = list_regimes(data_dir, include_retired=True)
                for r in regs:
                    options.append(f"{r['regime_id']} — {r['regime_name']}")
        except Exception:
            pass
        if len(options) == 1:
            options.extend([
                "R000 — ALL_REGIMES",
                "R001 — TREND",
                "R002 — SIDEWAYS",
                "R003 — HIGH_VOLATILITY",
                "R004 — LOW_VOLATILITY",
                "R005 — BREAKOUT",
                "R006 — REVERSAL",
                "R007 — EXPIRY_PINNING",
            ])
        return options

    def _reset_filters(self) -> None:
        if hasattr(self, "_filter_task_var"):
            self._filter_task_var.set("All Tasks")
        if hasattr(self, "_filter_regime_var"):
            self._filter_regime_var.set("All Regimes")
        if hasattr(self, "_filter_pop_var"):
            self._filter_pop_var.set("All Populations")
        if hasattr(self, "_filter_lc_var"):
            self._filter_lc_var.set("All Statuses")
        self._populate_models_tree()

    def _fit_dataset_column(self, rows: list[dict[str, Any]]) -> None:
        """Size Dataset column so the longest name fits without ellipsis."""
        if (self._models_family or "regression") == "triple_barrier":
            return
        names = [str(r.get("dataset") or "—") for r in rows]
        if not names:
            return
        try:
            from tkinter import font as tkfont

            measure_font = tkfont.nametofont("TkDefaultFont")
            text_w = max(int(measure_font.measure(n)) for n in names)
        except Exception:
            text_w = max(len(n) for n in names) * 8
        # Padding for cell margins; clamp so extreme names don't dominate the row.
        width = max(200, min(480, text_w + 28))
        self.models_tree.column("dataset", width=width, minwidth=width, anchor="w", stretch=False)

    def _filtered_model_rows(self) -> list[dict[str, Any]]:
        from chain_replay_ml.model_taxonomy import filter_model_records
        from chain_replay_ml.training.registry import resolve_model_registry_family

        family = self._models_family or "regression"
        rows = [r for r in self._rows if resolve_model_registry_family(r) == family]

        # Faceted filter inputs
        t_val = getattr(self, "_filter_task_var", None)
        r_val = getattr(self, "_filter_regime_var", None)
        p_val = getattr(self, "_filter_pop_var", None)
        l_val = getattr(self, "_filter_lc_var", None)

        task_str = t_val.get() if t_val else None
        regime_str = r_val.get() if r_val else None
        pop_str = p_val.get() if p_val else None
        lc_str = l_val.get() if l_val else None

        filtered = filter_model_records(
            rows,
            task_type=task_str,
            regime_id=regime_str,
            population=pop_str,
            lifecycle_status=lc_str,
        )
        return filtered

    def _build_prediction_runs_tab(self) -> None:
        frame = self._tab_frames[self._TAB_PREDICTION_RUNS]
        section_desc(
            frame,
            "Immutable walk-forward validation predictions stored per training run. "
            "Each run captures fold-level metrics and per-row predictions for strategy simulation.",
        )

        runs_header = ttk.Frame(frame, padding=(4, 4, 4, 0))
        runs_header.pack(fill="x")
        ttk.Label(runs_header, text="Prediction Runs", font=("Segoe UI", 10, "bold")).pack(side="left")
        hdr_btns = ttk.Frame(runs_header)
        hdr_btns.pack(side="right")
        ttk.Button(hdr_btns, text="Show JSON", command=self._show_prediction_run_json).pack(side="left", padx=2)
        ttk.Button(hdr_btns, text="Download JSON", command=self._download_prediction_run_json).pack(side="left", padx=2)

        runs_frame = ttk.Frame(frame)
        runs_frame.pack(fill="x", padx=4, pady=4)

        cols = ("run_id", "status", "created", "folds", "rows", "dataset_fp", "feature_hash")
        self._pred_runs_tree = ttk.Treeview(runs_frame, columns=cols, show="headings", height=5)
        for c, w, label in (
            ("run_id", 110, "Run"),
            ("status", 80, "Status"),
            ("created", 130, "Created"),
            ("folds", 50, "Folds"),
            ("rows", 80, "Predictions"),
            ("dataset_fp", 100, "Dataset FP"),
            ("feature_hash", 100, "Feature hash"),
        ):
            self._pred_runs_tree.heading(c, text=label)
            self._pred_runs_tree.column(c, width=w, anchor="center" if c != "run_id" else "w")
        self._pred_runs_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(runs_frame, orient="vertical", command=self._pred_runs_tree.yview)
        sb.pack(side="right", fill="y")
        self._pred_runs_tree.configure(yscrollcommand=sb.set)
        self._pred_runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_prediction_run_detail())

        self._pred_json_frame = ttk.LabelFrame(frame, text="Run JSON", padding=4)
        json_toolbar = ttk.Frame(self._pred_json_frame)
        json_toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(json_toolbar, text="Copy JSON", command=self._copy_prediction_run_json).pack(side="right")
        self._pred_run_detail = scrolledtext.ScrolledText(self._pred_json_frame, height=10, font=("Consolas", 9))
        self._pred_run_detail.pack(fill="both", expand=True)

        fold_cols = ("fold", "val_rows", "mae", "rmse", "direction", "predictions")
        self._pred_folds_tree = ttk.Treeview(frame, columns=fold_cols, show="headings", height=6)
        for c, w, label in (
            ("fold", 50, "Fold"),
            ("val_rows", 80, "Val rows"),
            ("mae", 80, "MAE"),
            ("rmse", 80, "RMSE"),
            ("direction", 90, "Direction %"),
            ("predictions", 90, "Predictions"),
        ):
            self._pred_folds_tree.heading(c, text=label)
            self._pred_folds_tree.column(c, width=w, anchor="center")
        self._pred_folds_tree.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _build_strategy_tab(self) -> None:
        frame = self._tab_frames[self._TAB_STRATEGY]
        self._strategy_panel = ModelRegistryStrategyPanel(
            frame,
            chart_dir=self.chart_dir,
            on_open_fold_research=self._on_open_fold_research,
        )
        self._strategy_panel.pack(fill="both", expand=True)

    def _lazy_holdout_analyze(
        self,
        load: Callable[[], Any],
        apply: Callable[[Any], None],
    ) -> None:
        self.lazy_load(
            load=load,
            apply=apply,
            message="Analyzing holdout performance…",
            show_overlay=False,
            status_var=self._status_var,
        )

    def _rebuild_detail_tabs(self, doc: dict[str, Any]) -> None:
        """Mount tab shells; render Overview now, other tabs on first visit."""
        is_wf = bool(doc.get("is_walk_forward"))
        from .model_registry_detail import _is_classification_model

        is_cls = _is_classification_model(doc)
        package = (
            doc.get("prediction_package")
            if isinstance(doc.get("prediction_package"), dict)
            else {}
        )
        has_package = bool(package)
        visible = [
            self._TAB_OVERVIEW,
            self._TAB_MODEL_METRICS,
        ]
        if is_cls:
            visible.append(self._TAB_THRESHOLD_ANALYSIS)
        visible.append(self._TAB_VALIDATION)
        if is_wf:
            visible.extend([self._TAB_WALK_FORWARD, self._TAB_HOLDOUT_PERFORMANCE])
        if has_package:
            visible.append(self._TAB_CLASSIFICATION)
        visible.extend([
            self._TAB_FEATURES,
            self._TAB_RETRAIN,
            self._TAB_LIFECYCLE,
        ])
        if is_wf:
            visible.append(self._TAB_PREDICTION_RUNS)
        visible.extend([self._TAB_STRATEGY, self._TAB_RESEARCH, self._TAB_ARTIFACTS])
        self._detail_visible_tabs = list(visible)
        self._tabs_rendered = set()

        for tab_id in list(self._tab_ids):
            if tab_id in self._tab_frames:
                try:
                    self._detail_notebook.forget(self._tab_frames[tab_id])
                except tk.TclError:
                    pass

        self._tab_scrolls.clear()
        custom_tabs = (
            self._TAB_PREDICTION_RUNS,
            self._TAB_STRATEGY,
            self._TAB_CLASSIFICATION,
        )
        for tab_id in visible:
            frame = self._tab_frames[tab_id]
            for child in frame.winfo_children():
                if tab_id in (self._TAB_PREDICTION_RUNS, self._TAB_STRATEGY):
                    continue
                child.destroy()

            if tab_id not in custom_tabs:
                scroll = ScrollableFrame(frame)
                scroll.pack(fill="both", expand=True)
                self._tab_scrolls[tab_id] = scroll

            self._detail_notebook.add(frame, text=self._tab_ids[tab_id])

        # Always paint Overview first so switching models feels instant.
        self._render_detail_tab(self._TAB_OVERVIEW)
        try:
            self._detail_notebook.select(self._tab_frames[self._TAB_OVERVIEW])
        except tk.TclError:
            pass

    def _on_detail_tab_changed(self, _event: object | None = None) -> None:
        if not self._detail_doc:
            return
        try:
            current = self._detail_notebook.select()
            frame = self._detail_notebook.nametowidget(current)
        except tk.TclError:
            return
        for tab_id, tab_frame in self._tab_frames.items():
            if tab_frame is frame:
                self._render_detail_tab(tab_id)
                return

    def _render_detail_tab(self, tab_id: str) -> None:
        doc = self._detail_doc
        if doc is None or tab_id in self._tabs_rendered:
            return
        if tab_id not in self._detail_visible_tabs:
            return
        is_wf = bool(doc.get("is_walk_forward"))
        # Pull deferred disk artifacts only when a tab needs them.
        heavy_need = {
            self._TAB_WALK_FORWARD: "walk_forward",
            self._TAB_FEATURES: "features",
            self._TAB_ARTIFACTS: "artifacts",
            self._TAB_LIFECYCLE: "lifecycle",
        }.get(tab_id)
        if heavy_need:
            try:
                from chain_replay_ml.training.registry import enrich_model_detail_heavy

                enrich_model_detail_heavy(self._data_dir(), doc, need=heavy_need)
            except Exception:
                pass
        try:
            if tab_id == self._TAB_OVERVIEW:
                render_overview(
                    self._tab_scrolls[self._TAB_OVERVIEW],
                    doc,
                    on_builder_features=self._on_builder_features,
                    chart_dir=self.chart_dir,
                )
            elif tab_id == self._TAB_MODEL_METRICS:
                render_model_metrics(self._tab_scrolls[self._TAB_MODEL_METRICS], doc)
            elif tab_id == self._TAB_THRESHOLD_ANALYSIS:
                render_threshold_analysis(
                    self._tab_scrolls[self._TAB_THRESHOLD_ANALYSIS],
                    doc,
                    chart_dir=self.chart_dir,
                )
            elif tab_id == self._TAB_VALIDATION:
                render_validation_metrics(self._tab_scrolls[self._TAB_VALIDATION], doc)
            elif tab_id == self._TAB_WALK_FORWARD and is_wf:
                render_walk_forward(self._tab_scrolls[self._TAB_WALK_FORWARD], doc)
            elif tab_id == self._TAB_HOLDOUT_PERFORMANCE and is_wf:
                render_holdout_performance(
                    self._tab_scrolls[self._TAB_HOLDOUT_PERFORMANCE],
                    doc,
                    chart_dir=self.chart_dir,
                    on_analyze=self._lazy_holdout_analyze,
                )
            elif tab_id == self._TAB_CLASSIFICATION:
                self._render_classification_package_tab(doc)
            elif tab_id == self._TAB_FEATURES:
                render_selected_features(
                    self._tab_scrolls[self._TAB_FEATURES],
                    doc,
                    on_builder_features=self._on_builder_features,
                    chart_dir=self.chart_dir,
                    tk_root=self.winfo_toplevel(),
                )
            elif tab_id == self._TAB_RETRAIN:
                render_retrain(
                    self._tab_scrolls[self._TAB_RETRAIN],
                    doc,
                    on_lifecycle=self._on_lifecycle,
                )
            elif tab_id == self._TAB_LIFECYCLE:
                render_lifecycle(self._tab_scrolls[self._TAB_LIFECYCLE], doc)
            elif tab_id == self._TAB_PREDICTION_RUNS and is_wf:
                self._load_model_prediction_runs(doc.get("model_name") or "")
            elif tab_id == self._TAB_STRATEGY:
                model_name = doc.get("model_name") or ""
                if hasattr(self, "_strategy_panel"):
                    self._strategy_panel.load_for_model(model_name)
            elif tab_id == self._TAB_RESEARCH:
                render_model_research(
                    self._tab_scrolls[self._TAB_RESEARCH],
                    chart_dir=self.chart_dir,
                    model_name=doc.get("model_name") or "",
                    on_run_program=self._run_research_program_on_model,
                )
            elif tab_id == self._TAB_ARTIFACTS:
                render_artifacts(self._tab_scrolls[self._TAB_ARTIFACTS], doc)
            else:
                return
        except Exception as exc:
            self._status_var.set(f"Tab render failed: {exc}")
            return
        self._tabs_rendered.add(tab_id)

    def _render_classification_package_tab(self, doc: dict[str, Any]) -> None:
        """Render the package's six optional probability-ladder classifiers."""
        frame = self._tab_frames[self._TAB_CLASSIFICATION]
        for child in frame.winfo_children():
            child.destroy()
        self._classification_member_name = None
        self._classification_member_doc = None
        self._classification_member_scrolls = {}

        package = (
            doc.get("prediction_package")
            if isinstance(doc.get("prediction_package"), dict)
            else {}
        )
        classification = (
            package.get("classification")
            if isinstance(package.get("classification"), dict)
            else {}
        )
        members = [
            item
            for item in (classification.get("members") or [])
            if isinstance(item, dict)
        ]

        header = ttk.Frame(frame, padding=(8, 8, 8, 4))
        header.pack(fill="x")
        available = int(classification.get("available") or 0)
        total = int(classification.get("total") or 6)
        ttk.Label(
            header,
            text=f"Probability Ladder · {available}/{total} classifiers available",
            font=("Segoe UI", 11, "bold"),
            foreground=ACCENT,
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Each classifier is an independent package member trained from the "
                "same dataset and prediction horizon. Partial packages are supported. "
                "Click a missing rung to open Model Builder with that target prefilled."
            ),
            foreground=COL_MUTED,
            wraplength=820,
        ).pack(anchor="w", pady=(2, 6))

        selector = ttk.Frame(header)
        selector.pack(fill="x")
        first_available: dict[str, Any] | None = None
        for member in members:
            label = str(member.get("label") or "—")
            ready = bool(member.get("available") and member.get("model_name"))
            if ready and first_available is None:
                first_available = member
            if ready:
                text = f"{label}   ✅"
                command = lambda m=dict(member): self._load_classification_member(m)
            else:
                text = f"{label}   ❌ Build"
                command = lambda m=dict(member): self._open_build_package_classifier(m)
            button = ttk.Button(selector, text=text, command=command)
            button.pack(side="left", padx=(0, 6), pady=2)

        self._classification_member_status = tk.StringVar(
            value=(
                "Select a classifier."
                if first_available is not None
                else (
                    "No classifiers yet — click +2% / +3% / … to open Model Builder "
                    "with that ladder target prefilled."
                )
            )
        )
        ttk.Label(
            frame,
            textvariable=self._classification_member_status,
            foreground=COL_MUTED,
            padding=(8, 0, 8, 4),
        ).pack(fill="x")

        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        for tab_id, label in (
            ("overview", "Overview"),
            ("metrics", "Metrics"),
            ("confusion", "Confusion Matrix"),
            ("threshold", "Threshold Analysis"),
            ("walk_forward", "Walk Forward"),
            ("holdout", "Holdout"),
            ("features", "Selected Features"),
        ):
            page = ttk.Frame(notebook)
            scroll = ScrollableFrame(page)
            scroll.pack(fill="both", expand=True)
            notebook.add(page, text=label)
            self._classification_member_scrolls[tab_id] = scroll

        if first_available is not None:
            self.after_idle(
                lambda m=dict(first_available): self._load_classification_member(m)
            )

    def _open_build_package_classifier(self, member: dict[str, Any]) -> None:
        """Jump to Model Builder with this missing ladder target prefilled."""
        if not self._on_build_package_classifier:
            messagebox.showinfo(
                "Probability Ladder",
                "Model Builder is not available from this view.",
            )
            return
        doc = self._detail_doc or {}
        package = (
            doc.get("prediction_package")
            if isinstance(doc.get("prediction_package"), dict)
            else {}
        )
        dataset = str(
            package.get("dataset")
            or (doc.get("config") or {}).get("dataset")
            or (doc.get("table_row") or {}).get("dataset")
            or ""
        ).strip()
        target = str(member.get("target") or "").strip()
        if not dataset or not target:
            messagebox.showwarning(
                "Probability Ladder",
                "Could not resolve dataset / target for this ladder slot.",
            )
            return
        features: list[str] = []
        cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
        raw_feats = cfg.get("features") or cfg.get("selected_features") or []
        if isinstance(raw_feats, list):
            features = [str(f) for f in raw_feats if f]
        source_model = str(doc.get("model_name") or "").strip() or None
        try:
            self._on_build_package_classifier(
                dataset_name=dataset,
                target=target,
                features=features or None,
                source_model=source_model,
                ladder_label=str(member.get("label") or ""),
            )
        except Exception as exc:
            messagebox.showerror("Probability Ladder", str(exc))

    def _load_classification_member(self, member: dict[str, Any]) -> None:
        model_name = str(member.get("model_name") or "").strip()
        if not model_name:
            return
        self._classification_member_name = model_name
        if self._classification_member_status is not None:
            self._classification_member_status.set(
                f"Loading {member.get('label') or ''} · {model_name}…"
            )

        def load() -> dict[str, Any]:
            from chain_replay_ml.training.registry import load_model_detail

            return load_model_detail(self._data_dir(), model_name)

        def apply(member_doc: dict[str, Any]) -> None:
            if self._classification_member_name != model_name:
                return
            member_doc["_data_dir"] = self._data_dir()
            self._classification_member_doc = member_doc
            if self._classification_member_status is not None:
                self._classification_member_status.set(
                    f"{member.get('label') or ''} · {model_name}"
                )
            for scroll in self._classification_member_scrolls.values():
                for child in scroll.inner.winfo_children():
                    child.destroy()

            render_overview(
                self._classification_member_scrolls["overview"],
                member_doc,
                on_builder_features=self._on_builder_features,
                chart_dir=self.chart_dir,
            )
            render_model_metrics(
                self._classification_member_scrolls["metrics"],
                member_doc,
            )
            render_classification_confusion(
                self._classification_member_scrolls["confusion"],
                member_doc,
            )
            render_threshold_analysis(
                self._classification_member_scrolls["threshold"],
                member_doc,
                chart_dir=self.chart_dir,
            )
            render_walk_forward(
                self._classification_member_scrolls["walk_forward"],
                member_doc,
            )
            render_holdout_performance(
                self._classification_member_scrolls["holdout"],
                member_doc,
                chart_dir=self.chart_dir,
                on_analyze=self._lazy_holdout_analyze,
            )
            render_selected_features(
                self._classification_member_scrolls["features"],
                member_doc,
                on_builder_features=self._on_builder_features,
                chart_dir=self.chart_dir,
                tk_root=self.winfo_toplevel(),
            )

        def on_error(exc: Exception) -> None:
            if self._classification_member_name != model_name:
                return
            if self._classification_member_status is not None:
                self._classification_member_status.set(
                    f"Could not load {model_name}: {exc}"
                )

        self.lazy_load(
            load=load,
            apply=apply,
            on_error=on_error,
            message=f"Loading classifier {member.get('label') or ''}…",
            show_overlay=False,
            status_var=self._status_var,
        )

    def refresh_models(self, *, select_first: bool = False, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=lambda: self._fetch_models(),
                apply=lambda rows: self._apply_models(rows, select_first=select_first),
                message="Loading models…",
                status_var=self._status_var,
            )
            return
        try:
            rows = self._fetch_models()
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_models(rows, select_first=select_first)

    def _fetch_models(self) -> list[dict[str, Any]]:
        import os
        from chain_replay_ml.training.prediction_packages import package_registry_rows
        from .selection_lists import get_sorted_models

        d_dir = self._data_dir()
        rows = get_sorted_models(d_dir, lightweight=False)
        seen_names = {str(r.get("model_name") or r.get("name") or "") for r in rows}

        candidates: list[str] = []
        if self.chart_dir:
            candidates.append(self.chart_dir)
            candidates.append(chart_data_dir(self.chart_dir))
        for alt_dir in candidates:
            if alt_dir and os.path.isdir(alt_dir) and os.path.abspath(alt_dir) != os.path.abspath(d_dir):
                if os.path.isdir(os.path.join(alt_dir, "models")):
                    extra_rows = get_sorted_models(alt_dir, lightweight=False)
                    for er in extra_rows:
                        m_name = str(er.get("model_name") or er.get("name") or "")
                        if m_name and m_name not in seen_names:
                            rows.append(er)
                            seen_names.add(m_name)

        return package_registry_rows(rows)


    def _apply_models(self, rows: list[dict[str, Any]], *, select_first: bool = False) -> None:
        self._rows = rows
        self._models_family = self._models_family_from_notebook()
        self._populate_models_tree(select_first=select_first)


    def _populate_models_tree(self, *, select_first: bool = False) -> None:
        self._sync_models_metric_columns()
        self.models_tree.delete(*self.models_tree.get_children())
        filtered = self._filtered_model_rows()
        is_tb = (self._models_family or "regression") == "triple_barrier"
        self._fit_dataset_column(filtered)
        from chain_replay_ml.model_taxonomy import format_model_taxonomy_display, get_context_champions_map
        champions_map = get_context_champions_map(self._data_dir())

        for r in filtered:
            name = str(r.get("model_name") or r.get("name") or "")
            if not name:
                continue
            prod = r.get("production_metrics") or {}
            tax = format_model_taxonomy_display(r, champions_map=champions_map)

            def _st(key: str) -> str:
                v = str(r.get(key) or "").strip().lower()
                if v == "available":
                    return "OK"
                if v == "deleted":
                    return "Gone"
                if v in ("n/a", "na", ""):
                    return "—"
                return v[:6]

            if is_tb or self._models_family == "classifier":
                m_metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                m1 = _fmt_pct(prod.get("precision_pct") or prod.get("precision") or m_metrics.get("precision_pct") or m_metrics.get("precision") or r.get("precision_pct") or r.get("precision"))
                m2 = _fmt_pct(prod.get("recall_pct") or prod.get("recall") or m_metrics.get("recall_pct") or m_metrics.get("recall") or r.get("recall_pct") or r.get("recall"))
                m3 = _fmt_pct(prod.get("f1_pct") or prod.get("f1") or m_metrics.get("f1_pct") or m_metrics.get("f1") or r.get("f1_pct") or r.get("f1"))
                m4 = _fmt_roc_auc(prod.get("roc_auc") or m_metrics.get("roc_auc") or r.get("roc_auc"))

            else:
                m1 = _fmt_num(prod.get("mae"))
                m2 = _fmt_num(prod.get("rmse"))
                m3 = _fmt_num(prod.get("directional_accuracy_pct"))
                m4 = ""

            self.models_tree.insert(
                "",
                "end",
                iid=name,
                values=(
                    name,
                    tax["task_type"][:4],
                    tax["regime_id"],
                    tax["population_badge"],
                    _fmt_validation_strategy(r.get("validation_strategy")),
                    r.get("dataset") or "—",
                    _st("dataset_status"),
                    r.get("label_run_id") or "—",
                    _st("label_run_status"),
                    _fmt_target(r.get("target")),
                    _fmt_fc(r.get("feature_count")),
                    _fmt_interval(r.get("sampling_interval_sec")),
                    m1,
                    m2,
                    m3,
                    m4,
                    _fmt_size_mb(r.get("size_bytes")),
                    "Open",
                    "—" if r.get("protected") else "Delete",
                ),
            )
        if self._models_family == "triple_barrier":
            family_label = "Triple Barrier"
        elif self._models_family == "classifier":
            family_label = "Classifier"
        else:
            family_label = "Regression"

        n_family = len(filtered)
        n_all = len(self._rows)
        # Keep tab titles in sync with counts.
        from chain_replay_ml.training.registry import resolve_model_registry_family

        n_reg = sum(1 for r in self._rows if resolve_model_registry_family(r) == "regression")
        n_cls = sum(1 for r in self._rows if resolve_model_registry_family(r) == "classifier")
        n_tb = sum(1 for r in self._rows if resolve_model_registry_family(r) == "triple_barrier")
        try:
            if "regression" in self._models_family_tab_ids:
                self._models_family_nb.tab(
                    self._models_family_tab_ids["regression"],
                    text=f"Regression ({n_reg})",
                )
            if "classifier" in self._models_family_tab_ids:
                self._models_family_nb.tab(
                    self._models_family_tab_ids["classifier"],
                    text=f"Classifier ({n_cls})",
                )
            if "triple_barrier" in self._models_family_tab_ids:
                self._models_family_nb.tab(
                    self._models_family_tab_ids["triple_barrier"],
                    text=f"Triple Barrier ({n_tb})",
                )
        except (KeyError, tk.TclError):
            pass
        self._status_var.set(
            f"{n_family} {family_label} model(s) · {n_all} total · "
            "research Exp_* packages live in Analysis Lab"
        )

        children = self.models_tree.get_children()
        if not children:
            if not (
                self._selected_name
                and any(
                    str(r.get("model_name") or "") == self._selected_name for r in self._rows
                )
            ):
                self._selected_name = None
                self._header_var.set("Select a model to view details")
            self._update_compare_button()
            return
        if not self._selected_name or self._selected_name not in children:
            from chain_replay_ml.training.registry import get_active_model

            stored = None
            try:
                stored = get_active_model(self._data_dir())
            except Exception:
                stored = None
            if stored and stored in children:
                self._selected_name = stored
            elif select_first or not self._selected_name or self._selected_name not in children:
                if self._selected_name not in children:
                    self._selected_name = children[0]
                    self._persist_selected_model(self._selected_name)
        if self._selected_name in children:
            self.models_tree.selection_set(self._selected_name)
            self.models_tree.see(self._selected_name)
            self._update_compare_button()
            self._load_detail()
        else:
            self._update_compare_button()

    def _on_models_selection_changed(self) -> None:
        self._update_compare_button()
        name = self._selected_model()
        if not name:
            return
        self._selected_name = name
        self._persist_selected_model(name)
        self._load_detail()

    def _update_compare_button(self) -> None:
        if self._compare_btn is None:
            return
        sel = self.models_tree.selection()
        if len(sel) == 2:
            self._compare_btn.pack(side="right", padx=4)
        else:
            self._compare_btn.pack_forget()

    def _open_compare(self) -> None:
        sel = self.models_tree.selection()
        if len(sel) != 2 or not self._on_compare_models:
            return
        self._on_compare_models(sel[0], sel[1])

    def _selected_model(self) -> str | None:
        sel = self.models_tree.selection()
        return sel[0] if sel else self._selected_name

    def _on_models_tree_click(self, event: tk.Event) -> str | None:
        if self.models_tree.identify_region(event.x, event.y) != "cell":
            return None
        col = self.models_tree.identify_column(event.x)
        row = self.models_tree.identify_row(event.y)
        if not row:
            return None
        # displaycolumns differ per family — resolve by column id, not hard-coded #N.
        try:
            col_id = self.models_tree.column(col, "id")
        except tk.TclError:
            col_id = ""
        if col_id == "research":
            self.after_idle(lambda n=row: self._open_research_lab(n))
            return "break"
        if col_id == "delete":
            self.after_idle(lambda n=row: self._confirm_delete_model(n))
            return "break"
        return None

    def _open_research_lab(self, model_name: str) -> None:
        """Open Research Lab → Prediction Dataset. Auto-create default lab if missing."""
        from .model_lab_window import open_model_lab_window

        row = next(
            (r for r in self._rows if str(r.get("model_name") or r.get("name") or "") == model_name),
            None,
        )
        try:
            open_model_lab_window(
                self.winfo_toplevel(),
                chart_dir=self.chart_dir,
                model_name=model_name,
                detail_doc=row,
                ensure_lab=True,
                initial_tab="prediction",
            )
        except Exception as exc:
            messagebox.showerror("Research", str(exc), parent=self)

    def _confirm_delete_model(self, model_name: str) -> None:
        row = next(
            (r for r in self._rows if str(r.get("model_name") or r.get("name") or "") == model_name),
            None,
        )
        if row and row.get("protected"):
            reason = str(row.get("protected_reason") or "This model is currently in use.")
            messagebox.showinfo("Delete Model", reason)
            return
        if not messagebox.askyesno(
            "Delete Model",
            f'Permanently delete model "{model_name}"?\n\n'
            "This removes the model folder and all training artifacts from disk.",
            icon="warning",
        ):
            return
        from chain_replay_ml.training.registry import ModelDeleteBlockedError, delete_model

        try:
            result = delete_model(self._data_dir(), model_name)
        except ModelDeleteBlockedError as exc:
            messagebox.showerror("Delete Model", str(exc))
            return
        except FileNotFoundError as exc:
            messagebox.showerror("Delete Model", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Delete Model", str(exc))
            return
        replay_n = int(result.get("replay_sessions_deleted") or 0)
        msg = f'Model "{model_name}" deleted.'
        if replay_n:
            msg += f"\n{replay_n} replay session(s) removed."
        messagebox.showinfo("Delete Model", msg)
        if self._selected_name == model_name:
            self._selected_name = None
            self._detail_doc = None
            self._header_var.set("Select a model to view details")
        self.refresh_models(select_first=bool(not self._selected_name))

    def _load_detail(self) -> None:
        name = self._selected_model()
        if not name:
            return
        self._selected_name = name
        self._header_var.set(f"{name}  ·  loading…")
        data_dir = self._data_dir()

        def _load() -> dict[str, Any]:
            from chain_replay_ml.training.registry import load_model_detail
            from chain_replay_ml.training.prediction_packages import attach_prediction_package

            detail = load_model_detail(data_dir, name)
            return attach_prediction_package(detail, self._rows)

        def _apply(doc: dict[str, Any]) -> None:
            # Selection may have changed while the worker ran.
            if self._selected_model() not in (None, name) and self._selected_name != name:
                return
            self._detail_doc = doc
            doc["_data_dir"] = data_dir
            strat = doc.get("table_row", {}).get("validation_strategy") or _strat_from_doc(doc)

            # Phase 4C.4: Multi-dimensional Taxonomy & Context Champion Header
            try:
                from chain_replay_ml.model_taxonomy import format_model_taxonomy_display
                from chain_replay_ml.research_memory.champion_history import get_champion_for_context

                row = doc.get("table_row") or {}
                tax = format_model_taxonomy_display(row or doc)
                champ_doc = get_champion_for_context(data_dir, tax["context_key"])
                champ_name = str((champ_doc or {}).get("champion_model_name") or (champ_doc or {}).get("current_model_name") or "—")
                chall_name = str((champ_doc or {}).get("challenger_model_name") or "—")
                self._header_var.set(
                    f"{name}  [{tax['population_badge']}]  ·  {tax['task_label']}  ·  {tax['regime_display']}\n"
                    f"Context: {tax['context_key']}  |  Champion: {champ_name}  |  Challenger: {chall_name}  ·  ({strat})"
                )
            except Exception:
                self._header_var.set(f"{name}  ·  {strat}")

            timing = doc.get("_timing") if isinstance(doc.get("_timing"), dict) else {}
            total_ms = timing.get("total_ms")
            if total_ms is not None:
                self._status_var.set(f"Loaded {name} in {total_ms} ms")
            else:
                self._status_var.set(f"Loaded {name}")
            self._rebuild_detail_tabs(doc)

        def _on_error(exc: Exception) -> None:
            self._header_var.set(f"{name}  ·  error")
            self._detail_doc = None
            self._tabs_rendered = set()
            self._status_var.set(f"Load failed: {exc}")
            for tab_id, frame in self._tab_frames.items():
                for child in frame.winfo_children():
                    if tab_id in (self._TAB_PREDICTION_RUNS, self._TAB_STRATEGY):
                        continue
                    child.destroy()
                if tab_id in (self._TAB_PREDICTION_RUNS, self._TAB_STRATEGY):
                    continue
                ttk.Label(frame, text=str(exc), foreground="red").pack(padx=8, pady=8)

        self.lazy_load(
            load=_load,
            apply=_apply,
            on_error=_on_error,
            message=f"Loading {name}…",
            status_var=self._status_var,
            show_overlay=True,
        )

    def _load_model_prediction_runs(self, name: str) -> None:
        from chain_replay_ml.prediction_runs import list_runs

        self._pred_runs_tree.delete(*self._pred_runs_tree.get_children())
        self._pred_folds_tree.delete(*self._pred_folds_tree.get_children())
        self._pred_run_json = ""
        self._pred_run_detail.delete("1.0", "end")
        self._pred_json_frame.pack_forget()
        self._pred_run_map.clear()
        try:
            runs = list_runs(self._data_dir(), name, limit=50)
        except Exception as exc:
            self._pred_run_json = f"Load failed: {exc}"
            return
        if not runs:
            return
        for r in runs:
            rid = str(r.get("run_id") or "")
            short = f"{rid[:8]}…{rid[-4:]}" if len(rid) > 12 else rid
            label = short
            self._pred_run_map[label] = rid
            self._pred_runs_tree.insert(
                "",
                "end",
                iid=label,
                values=(
                    short,
                    r.get("status"),
                    (r.get("created_at") or "")[:19],
                    r.get("fold_count"),
                    r.get("prediction_count"),
                    r.get("dataset_fingerprint") or "—",
                    r.get("feature_snapshot_hash") or "—",
                ),
            )
        children = self._pred_runs_tree.get_children()
        if children:
            self._pred_runs_tree.selection_set(children[0])
            self._pred_runs_tree.see(children[0])
            self._load_prediction_run_detail()

    def _selected_prediction_run_id(self) -> str | None:
        sel = self._pred_runs_tree.selection()
        if not sel:
            return None
        label = sel[0]
        return self._pred_run_map.get(label, label)

    def _fetch_prediction_run_json(self) -> str:
        run_id = self._selected_prediction_run_id()
        if not run_id:
            return ""
        from chain_replay_ml.prediction_runs import get_run_detail

        try:
            run = get_run_detail(self._data_dir(), run_id)
        except Exception as exc:
            return f"Error: {exc}"
        if not run:
            return "Run not found."
        return json.dumps(run, indent=2, default=str)

    def _load_prediction_run_detail(self) -> None:
        self._pred_folds_tree.delete(*self._pred_folds_tree.get_children())
        self._pred_run_json = ""
        self._pred_run_detail.delete("1.0", "end")
        self._pred_json_frame.pack_forget()

        run_id = self._selected_prediction_run_id()
        if not run_id:
            return
        from chain_replay_ml.prediction_runs import get_run_detail

        try:
            run = get_run_detail(self._data_dir(), run_id)
        except Exception as exc:
            self._pred_run_json = f"Error: {exc}"
            return
        if not run:
            self._pred_run_json = "Run not found."
            return
        self._pred_run_json = json.dumps(run, indent=2, default=str)
        for f in run.get("folds") or []:
            if not isinstance(f, dict):
                continue
            self._pred_folds_tree.insert(
                "",
                "end",
                values=(
                    f.get("fold_number"),
                    _fmt_num(f.get("validation_rows")),
                    _fmt_num(f.get("mae")),
                    _fmt_num(f.get("rmse")),
                    _fmt_num(f.get("directional_accuracy_pct")),
                    f.get("prediction_count"),
                ),
            )

    def _show_prediction_run_json(self) -> None:
        if not self._selected_prediction_run_id():
            messagebox.showinfo("Prediction Runs", "Select a prediction run first.")
            return
        if not self._pred_run_json:
            self._pred_run_json = self._fetch_prediction_run_json()
        self._pred_run_detail.delete("1.0", "end")
        self._pred_run_detail.insert("end", self._pred_run_json)
        self._pred_json_frame.pack(fill="both", expand=True, padx=4, pady=4, before=self._pred_folds_tree)

    def _copy_prediction_run_json(self) -> None:
        text = self._pred_run_json or self._pred_run_detail.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Copy JSON", "No JSON to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copy JSON", "Run JSON copied to clipboard.")

    def _download_prediction_run_json(self) -> None:
        if not self._selected_prediction_run_id():
            messagebox.showinfo("Prediction Runs", "Select a prediction run first.")
            return
        if not self._pred_run_json:
            self._pred_run_json = self._fetch_prediction_run_json()
        run_id = self._selected_prediction_run_id() or "prediction_run"
        path = filedialog.asksaveasfilename(
            title="Save prediction run JSON",
            defaultextension=".json",
            initialfile=f"{run_id}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._pred_run_json)
        except OSError as exc:
            messagebox.showerror("Download JSON", str(exc))
            return
        messagebox.showinfo("Download JSON", f"Saved to:\n{path}")

    def _run_research_program_on_model(self, model_name: str, data_dir: str) -> None:
        picked = self._pick_run_research_program(model_name, data_dir)
        if not picked:
            return
        program_id, report_id = picked
        from .research_campaign_coordinator import get_research_campaign_coordinator

        coord = get_research_campaign_coordinator()
        out = coord.start_program_on_model(
            data_dir,
            model_id=model_name,
            program_id=program_id,
            research_report_id=report_id,
        )
        if not out.get("ok"):
            messagebox.showerror("Research Program", out.get("error") or "Failed to start")
            return
        run = out.get("run") or {}
        manifest = run.get("manifest") or {}
        kick = out.get("kick") or {}
        kick_note = ""
        if kick.get("ok"):
            kick_note = "\n\nFirst experiment started — see bottom progress bar."
        elif kick.get("error"):
            kick_note = f"\n\nFirst experiment: {kick.get('error')}"
        messagebox.showinfo(
            "Research Program",
            f"Started on {model_name}.\n\n"
            f"Program: {manifest.get('program_name') or '—'}\n"
            f"Campaigns queued: {manifest.get('total_campaigns') or len(manifest.get('campaigns') or [])}"
            f"{kick_note}",
        )
        if self._selected_name == model_name:
            self._load_detail()

    def _pick_run_research_program(self, model_name: str, data_dir: str) -> tuple[str, str] | None:
        from chain_replay_ml.fold_research import list_research_programs, list_saved_research_reports

        programs = list_research_programs(data_dir)
        if not programs:
            messagebox.showinfo(
                "Research Program",
                "Create a research program first (Strategy Lab → Research Programs).",
            )
            return None

        reports = list_saved_research_reports(data_dir, limit=200)
        model_reports = [
            r for r in reports
            if str(r.get("model_id") or "") == model_name
            or model_name in str(r.get("model_id") or "")
        ]
        if not model_reports:
            model_reports = list(reports)

        win = tk.Toplevel(self.winfo_toplevel())
        win.title("Run Research Program")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.geometry("760x360")

        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text=f"Model: {model_name}",
            font=("Segoe UI", 10, "bold"),
            foreground=ACCENT,
        ).pack(anchor="w")
        ttk.Label(
            body,
            text="Train → Research → Certify → Deploy",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(body, text="Research program:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        prog_labels: list[str] = []
        prog_ids: list[str] = []
        for p in programs:
            stats = p.get("campaign_stats") or {}
            total = int(stats.get("total") or 0)
            prog_labels.append(
                f"{p.get('name')}  ({p.get('program_type') or 'strategy'})  ·  {total} campaigns"
            )
            prog_ids.append(str(p.get("program_id") or ""))

        prog_var = tk.StringVar(value=prog_labels[0] if prog_labels else "")
        prog_combo = ttk.Combobox(
            body,
            textvariable=prog_var,
            values=prog_labels,
            state="readonly",
            width=96,
        )
        prog_combo.pack(fill="x", pady=(4, 4))

        desc_var = tk.StringVar(value=str(programs[0].get("description") or "") if programs else "")
        ttk.Label(body, textvariable=desc_var, foreground=COL_MUTED, wraplength=700).pack(anchor="w", pady=(0, 10))

        ttk.Label(body, text="Baseline research report:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        report_options: list[str] = []
        report_ids: list[str] = []
        for row in model_reports:
            rid = str(row.get("report_id") or "")
            if not rid:
                continue
            created = str(row.get("created_at") or "")[:10]
            report_options.append(
                f"{rid[:16]}…  |  grade={row.get('grade') or '—'}  |  "
                f"trades={row.get('trade_count') or '—'}  |  {created}"
            )
            report_ids.append(rid)

        report_var = tk.StringVar()
        report_combo: ttk.Combobox | None = None
        if report_options:
            report_var.set(report_options[0])
            report_combo = ttk.Combobox(
                body,
                textvariable=report_var,
                values=report_options,
                state="readonly",
                width=96,
                font=("Consolas", 9),
            )
            report_combo.pack(fill="x", pady=(4, 4))
        else:
            ttk.Label(
                body,
                text="No saved reports — run strategy simulation and save a research report first.",
                foreground=COL_WARN,
                wraplength=700,
            ).pack(anchor="w", pady=(4, 4))

        manual_var = tk.StringVar(value=report_ids[0] if report_ids else "")
        manual_row = ttk.Frame(body)
        manual_row.pack(fill="x", pady=(0, 8))
        ttk.Label(manual_row, text="Report ID:").pack(side="left")
        manual_entry = ttk.Entry(manual_row, textvariable=manual_var, width=72, font=("Consolas", 9))
        manual_entry.pack(side="left", padx=(8, 0), fill="x", expand=True)

        def _on_prog_select(_event: object = None) -> None:
            try:
                idx = prog_labels.index(prog_var.get())
            except ValueError:
                return
            desc_var.set(str(programs[idx].get("description") or ""))

        def _on_report_select(_event: object = None) -> None:
            if not report_options:
                return
            try:
                idx = report_options.index(report_var.get())
            except ValueError:
                return
            manual_var.set(report_ids[idx])

        prog_combo.bind("<<ComboboxSelected>>", _on_prog_select)
        if report_combo is not None:
            report_combo.bind("<<ComboboxSelected>>", _on_report_select)

        ttk.Label(
            body,
            text="Each campaign clones from the program, attaches this baseline, and auto-runs with evidence-based stopping.",
            foreground=COL_MUTED,
            wraplength=700,
        ).pack(anchor="w", pady=(4, 0))

        choice: dict[str, tuple[str, str] | None] = {"value": None}

        def _accept() -> None:
            try:
                pidx = prog_labels.index(prog_var.get())
            except ValueError:
                messagebox.showinfo("Research Program", "Select a program.", parent=win)
                return
            report_id = manual_var.get().strip()
            if not report_id:
                messagebox.showinfo("Research Program", "Enter or select a baseline report ID.", parent=win)
                return
            choice["value"] = (prog_ids[pidx], report_id)
            win.destroy()

        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x", pady=(14, 0))
        ttk.Button(btn_row, text="Start Program Run", command=_accept).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 8))

        win.wait_window()
        return choice["value"]

    def _open_prediction_runs(self) -> None:
        name = self._selected_model()
        if name and self._on_open_prediction_runs:
            self._on_open_prediction_runs(name)

    def _ensure_research_leaderboard_window(self) -> tk.Toplevel:
        """Create (once) the Research Leaderboard companion window."""
        win = getattr(self, "_research_leaderboard_win", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    return win
            except tk.TclError:
                pass

        from .model_research_leaderboard_panel import ModelResearchLeaderboardPanel

        root = self.winfo_toplevel()
        win = tk.Toplevel(root)
        win.withdraw()
        win.title("Research Leaderboard")
        try:
            win.transient(root)
        except tk.TclError:
            pass

        hdr = ttk.Frame(win, padding=(10, 8))
        hdr.pack(fill="x")
        ttk.Label(
            hdr,
            text="Autonomous Model Research Leaderboard (Phase 4F)",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        ttk.Button(hdr, text="Close", command=lambda: win.withdraw()).pack(side="right")

        self._leaderboard_panel = ModelResearchLeaderboardPanel(
            win,
            chart_dir=self.chart_dir,
            on_select_model=self.select_model,
        )
        self._leaderboard_panel.pack(fill="both", expand=True)

        self._research_leaderboard_win = win
        return win

    def _open_research_leaderboard(self) -> None:
        """Open Research Leaderboard beside the main app (Companion Window pattern, matching Feature Transformations)."""
        from .fold_replay_widgets import place_toplevel_beside_main

        win = self._ensure_research_leaderboard_window()
        if not getattr(self, "_research_leaderboard_placed", False):
            win.update_idletasks()
            place_toplevel_beside_main(win, self)
            self._research_leaderboard_placed = True
        try:
            win.deiconify()
            win.lift()
            win.focus_force()
        except tk.TclError:
            self._research_leaderboard_win = None
            win = self._ensure_research_leaderboard_window()
            win.update_idletasks()
            place_toplevel_beside_main(win, self)
            self._research_leaderboard_placed = True
            win.deiconify()
            win.lift()
            win.focus_force()

        if hasattr(self, "_leaderboard_panel") and self._leaderboard_panel:
            self._leaderboard_panel.set_chart_dir(self.chart_dir)
            self._leaderboard_panel.refresh_leaderboard()


def _strat_from_doc(doc: dict[str, Any]) -> str:
    strat = doc.get("validation_strategy") or {}
    if isinstance(strat, dict):
        return str(strat.get("label") or "—")
    return str(strat or "—")
