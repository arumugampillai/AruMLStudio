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

from datetime import datetime
import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any, Callable

from chain_replay_ml.model_taxonomy import ModelContextKey, BASELINE_REGIME_CATALOG
from chain_replay_ml.morning_dossier import (
    MorningResearchDossier,
    export_morning_dossier_markdown,
    generate_morning_research_dossier,
)
from chain_replay_ml.overnight_campaign import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignReport,
    OvernightCampaignRunner,
)
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
from chain_replay_ml.discovery_pipeline import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    get_discovery_pipeline_summary,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_pipeline_by_campaign,
    load_discovery_snapshots_for_pipeline,
)
from chain_replay_ml.research_memory.champion_history import get_champion_for_context
from chain_replay_ml.research_recommendations import (
    RecommendationDossier,
    generate_context_recommendation_dossiers,
)

from .build_service import chart_data_dir
from .ui_state import get_ui_state_manager
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
        self._datasets: list[dict[str, Any]] = []
        self._dataset_var = tk.StringVar(value="")
        self._dataset_meta: dict[str, Any] | None = None

        self._market_var = tk.StringVar(value="NIFTY")
        self._sampling_var = tk.StringVar(value="6s")
        self._task_var = tk.StringVar(value="DIRECTION_CLASSIFIER")
        self._horizon_var = tk.StringVar(value="5m")
        self._regime_var = tk.StringVar(value="R001")
        self._context_key_var = tk.StringVar(value="")
        self._dataset_status_var = tk.StringVar(value="")
        self._dataset_available: bool = False

        # Research Campaign Budget Settings (Sensible overnight defaults)
        self._cfg_max_gen = tk.IntVar(value=10)
        self._cfg_max_cands = tk.IntVar(value=100)
        self._cfg_max_hours = tk.DoubleVar(value=8.0)
        self._cfg_plateau_enabled = tk.BooleanVar(value=True)
        self._cfg_plateau_patience = tk.IntVar(value=3)
        self._cfg_plateau_min_lift = tk.DoubleVar(value=0.5)
        self._cfg_min_gen_before_plateau = tk.IntVar(value=3)
        self._elim_strat_var = tk.StringVar(value="NONE")
        self._elim_strat_status_var = tk.StringVar(value="🎯 Elimination Strategy: None")
        self._evidence_db_summary_var = tk.StringVar(value="📊 Evidence DB: Loading...")
        self._campaign_start_ts: float = 0.0

        # Available Algorithm Selection Checkbox variables (All checked by default)
        self._algo_xgb_var = tk.BooleanVar(value=True)
        self._algo_cat_var = tk.BooleanVar(value=True)
        self._algo_lgb_var = tk.BooleanVar(value=True)
        self._algo_rf_var = tk.BooleanVar(value=True)
        self._algo_et_var = tk.BooleanVar(value=True)

        # Lazy loading state per tab
        self._loaded_tab_candidate: dict[str, str] = {}
        self._loaded_context_tabs: set[str] = set()

        # Autonomous Campaign Runner state
        self._active_runner: OvernightCampaignRunner | None = None
        self._is_running: bool = False
        self._worker_thread: threading.Thread | None = None
        self._last_campaign_id: str | None = None

        self._ranked_dossiers: list[dict[str, Any]] = []
        self._selected_dossier: dict[str, Any] | None = None
        self._ui_state = get_ui_state_manager()

        self._build_ui()
        self._refresh_datasets_combo()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._refresh_datasets_combo()

    def _data_dir(self) -> str:
        candidates = []
        if self.chart_dir:
            candidates.append(chart_data_dir(self.chart_dir))
            candidates.append(self.chart_dir)
        candidates.append(chart_data_dir(os.getcwd()))
        candidates.append(os.path.join(os.getcwd(), "data"))
        candidates.append(os.getcwd())

        for c in candidates:
            if not c:
                continue
            ds_dir = os.path.join(c, "datasets") if not c.endswith("datasets") else c
            if os.path.isdir(ds_dir) and any(f.endswith(".json") for f in os.listdir(ds_dir)):
                return c
        for c in candidates:
            if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "analysis.db")):
                return c
        return chart_data_dir(self.chart_dir) if self.chart_dir else chart_data_dir(os.getcwd())

    def _selected_dataset_name(self) -> str:
        from .model_builder.panel import _dataset_name_from_label
        return _dataset_name_from_label(self._dataset_var.get(), self._datasets)

    def _refresh_datasets_combo(self, preferred_name: str | None = None) -> None:
        try:
            from .model_builder import service
            from .model_builder.panel import _dataset_display_label

            candidates = []
            if self.chart_dir:
                candidates.append(chart_data_dir(self.chart_dir))
                candidates.append(self.chart_dir)
            candidates.append(self._data_dir())
            candidates.append(chart_data_dir(os.getcwd()))
            candidates.append(os.path.join(os.getcwd(), "data"))
            candidates.append(os.getcwd())

            all_datasets: list[dict[str, Any]] = []
            for d in dict.fromkeys(c for c in candidates if c):
                try:
                    found = service.list_builder_datasets(d)
                    if found:
                        all_datasets = found
                        break
                except Exception:
                    pass

            self._datasets = all_datasets
            vals = [_dataset_display_label(d) for d in self._datasets if d.get("dataset_name")]
            if hasattr(self, "_dataset_cb"):
                self._dataset_cb["values"] = vals
            names = [d.get("dataset_name") for d in self._datasets if d.get("dataset_name")]
            pick = preferred_name if preferred_name in names else self._selected_dataset_name()
            if not pick and names:
                pick = names[0]
            if pick:
                row = next((d for d in self._datasets if d.get("dataset_name") == pick), None)
                self._dataset_var.set(_dataset_display_label(row) if row else pick)
                self._on_dataset_selected(refresh_leaderboard=True)
            elif not vals:
                self._dataset_var.set("")
                self._dataset_status_var.set("⚠️ No Analysis Datasets found in registry. Build/export a dataset in Dataset Builder first.")
                if hasattr(self, "_lbl_dataset_status"):
                    self._lbl_dataset_status.config(foreground=COL_WARN)
        except Exception:
            pass

    def _on_dataset_selected(self, refresh_leaderboard: bool = True) -> None:
        ds_name = self._selected_dataset_name()
        data_dir = self._data_dir()
        if ds_name and data_dir:
            try:
                from .model_builder import service
                self._dataset_meta = service.load_dataset_metadata_doc(data_dir, ds_name)
            except Exception:
                loaded = None
                candidates = [
                    chart_data_dir(self.chart_dir) if self.chart_dir else "",
                    self.chart_dir or "",
                    chart_data_dir(os.getcwd()),
                    os.path.join(os.getcwd(), "data"),
                    os.getcwd(),
                ]
                for cand in dict.fromkeys(c for c in candidates if c):
                    try:
                        from .model_builder import service
                        loaded = service.load_dataset_metadata_doc(cand, ds_name)
                        if loaded:
                            break
                    except Exception:
                        pass
                self._dataset_meta = loaded
        else:
            self._dataset_meta = None

        meta = (self._dataset_meta or {}).get("metadata") or {}
        market = str(meta.get("market") or meta.get("master_market") or "").strip().upper()
        if not market and ds_name:
            for m in ("BANKNIFTY", "FINNIFTY", "SENSEX", "NIFTY"):
                if m in ds_name.upper():
                    market = m
                    break
        if not market:
            market = "NIFTY"
        self._market_var.set(market)

        int_sec = meta.get("interval_sec") or meta.get("sample_interval_sec")
        if int_sec is None and ds_name:
            import re
            match = re.search(r"_(\d+)s", ds_name)
            if match:
                try:
                    int_sec = int(match.group(1))
                except Exception:
                    int_sec = 6
        int_sec_val = int(int_sec) if int_sec is not None else 6
        self._sampling_var.set(f"{int_sec_val}s")

        self._update_resolved_context_key()
        if refresh_leaderboard:
            self.refresh_leaderboard()

    def _dataset_eligible_features(self) -> list[str]:
        if not self._dataset_meta:
            return []
        meta = (self._dataset_meta or {}).get("metadata") or {}
        cols = (
            meta.get("feature_columns")
            or meta.get("enabled_features")
            or meta.get("selected_features")
            or []
        )
        names = [str(c).strip() for c in cols if str(c).strip()]
        parquet_rel = str(meta.get("output_parquet") or "").strip()
        data_dir = self._data_dir()
        parquet_path = os.path.join(data_dir, parquet_rel) if parquet_rel and not os.path.isabs(parquet_rel) else (
            str(meta.get("parquet_path") or parquet_rel)
        )
        if parquet_path and os.path.isfile(parquet_path):
            try:
                import pyarrow.parquet as pq
                schema_names = set(pq.read_schema(parquet_path).names)
                if schema_names:
                    names = [c for c in names if c in schema_names]
            except Exception:
                pass
        targets = set(meta.get("target_columns") or meta.get("prediction_target_columns") or [])
        meta_skip = {
            "timestamp", "datetime", "date", "time", "token", "symbol", "expiry",
            "option_type", "instrument_type", "day", "trading_day", "open", "high", "low", "close", "ltp"
        }
        eligible = [
            c for c in names
            if c not in targets
            and c.lower() not in meta_skip
            and not c.startswith("target_")
            and not c.startswith("label_")
            and not c.startswith("future_")
        ]
        return list(dict.fromkeys(eligible))

    def _resolve_target_column(self, meta: dict[str, Any]) -> str:
        task = self._task_var.get()
        horizon = self._horizon_var.get()
        targets = list(meta.get("target_columns") or meta.get("prediction_target_columns") or [])
        if task == "DIRECTION_CLASSIFIER":
            preferred = f"label_up_{horizon}"
            if preferred in targets:
                return preferred
            for t in targets:
                if str(t).startswith("label_up") or str(t).startswith("target_up"):
                    return str(t)
        elif task == "REGRESSION":
            preferred = f"future_ret_{horizon}"
            if preferred in targets:
                return preferred
            for t in targets:
                if str(t).startswith("future_ret") or str(t).startswith("future_ltp"):
                    return str(t)
        elif task == "CONFIDENCE_CLASSIFIER":
            for t in targets:
                if str(t) in ("target_reached", "hit"):
                    return str(t)
        if targets:
            return str(targets[0])
        return f"label_up_{horizon}"

    def _update_resolved_context_key(self) -> None:
        """Resolve dataset metadata into canonical ModelContextKey string and update dataset status badge."""
        from chain_replay_ml.model_taxonomy import TaskType

        market = self._market_var.get() or "NIFTY"
        interval_sec = 6
        try:
            interval_sec = int(str(self._sampling_var.get() or "6").replace("s", ""))
        except (TypeError, ValueError):
            interval_sec = 6

        # Extract clean regime ID from combo text (e.g. 'R001 - TREND' -> 'R001')
        reg_raw = self._regime_var.get().split()[0].split("-")[0].strip()
        if not reg_raw:
            reg_raw = "R001"

        ctx = ModelContextKey(
            market=market,
            sampling_interval_sec=interval_sec,
            task_type=TaskType.from_str(self._task_var.get()),
            prediction_horizon=self._horizon_var.get(),
            regime_id=reg_raw,
        )
        self._context_key_var.set(ctx.canonical_key_str())

        ds_name = self._selected_dataset_name()
        if not ds_name or not self._dataset_meta:
            self._dataset_available = False
            self._dataset_status_var.set("⚠️ No Dataset selected — select a dataset from the dropdown.")
            if hasattr(self, "_lbl_dataset_status"):
                self._lbl_dataset_status.config(foreground=COL_WARN)
            return

        meta = (self._dataset_meta or {}).get("metadata") or {}
        row = next((d for d in self._datasets if d.get("dataset_name") == ds_name), None) or {}
        row_count = int(row.get("row_count") or meta.get("row_count") or 0)
        feats = self._dataset_eligible_features()
        feat_count = len(feats) if feats else int(row.get("feature_count") or meta.get("feature_count") or 0)
        target_count = int(row.get("target_count") or len(meta.get("target_columns") or meta.get("prediction_target_columns") or []) or 0)
        days = list(meta.get("days") or (meta.get("trading_day_filter") or {}).get("exported_dates") or [])
        days_count = len(days) if days else (int(meta.get("day_count") or 0) or "—")

        self._dataset_available = True
        self._dataset_status_var.set(
            f"🟢 Dataset: {ds_name} ({row_count:,} rows · {feat_count} features · {target_count} targets · {days_count} days · {interval_sec}s · {market})"
        )
        if hasattr(self, "_lbl_dataset_status"):
            self._lbl_dataset_status.config(foreground=COL_OK)

    def _build_ui(self) -> None:
        # Top Container: Context Selector & Production Status Banner
        top_frame = ttk.Frame(self, padding=(8, 6))
        top_frame.pack(fill="x")

        # 1. Canonical Context Selector Bar
        ctx_box = ttk.LabelFrame(top_frame, text="Canonical Research Context Selector (ModelContextKey)", padding=(8, 6))
        ctx_box.pack(fill="x", pady=(0, 4))

        controls_row = ttk.Frame(ctx_box)
        controls_row.pack(fill="x")

        # Authoritative Dataset Dropdown
        ttk.Label(controls_row, text="Dataset:").pack(side="left", padx=(2, 2))
        self._dataset_cb = ttk.Combobox(
            controls_row,
            textvariable=self._dataset_var,
            width=42,
            state="readonly",
        )
        self._dataset_cb.pack(side="left", padx=(0, 4))
        self._dataset_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_selected())

        ttk.Button(controls_row, text="🔄", width=3, command=self._refresh_datasets_combo).pack(side="left", padx=(0, 8))

        # Task Type
        ttk.Label(controls_row, text="Task:").pack(side="left", padx=(2, 2))
        t_cb = ttk.Combobox(
            controls_row,
            textvariable=self._task_var,
            values=["DIRECTION_CLASSIFIER", "REGRESSION", "TRIPLE_BARRIER", "CONFIDENCE_CLASSIFIER", "VOLATILITY_ESTIMATOR"],
            width=20,
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
        r_cb = ttk.Combobox(controls_row, textvariable=self._regime_var, values=reg_vals, width=16, state="readonly")
        r_cb.pack(side="left", padx=(0, 8))
        r_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_context_param_changed())

        ttk.Button(controls_row, text="Query Leaderboard", command=self.refresh_leaderboard).pack(side="right", padx=4)

        # Context Key Display Label & Real Master Dataset Status
        key_bar = ttk.Frame(ctx_box, padding=(0, 4))
        key_bar.pack(fill="x")
        ttk.Label(key_bar, text="🎯 Active Context Key:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        ttk.Label(key_bar, textvariable=self._context_key_var, font=("Consolas", 10, "bold"), foreground="#0d47a1").pack(side="left", padx=(6, 16))

        self._lbl_dataset_status = ttk.Label(
            key_bar,
            textvariable=self._dataset_status_var,
            font=("Segoe UI", 9, "bold"),
            foreground=COL_OK,
        )
        self._lbl_dataset_status.pack(side="left")


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
        ttk.Label(gov_row, textvariable=self._elim_strat_status_var, font=("TkDefaultFont", 9), foreground="#0d47a1").pack(side="left", padx=(0, 16))

        disclaimer_lbl = ttk.Label(
            gov_box,
            text="⚠️ Research Memory is strictly advisory. Candidate promotions require human governance approval.",
            font=("TkDefaultFont", 8, "italic"),
            foreground=COL_MUTED,
        )
        disclaimer_lbl.pack(anchor="w", pady=(2, 0))

        # 3. Autonomous Research Control & Telemetry Bar (Phase 4F.5)
        run_box = ttk.LabelFrame(top_frame, text="Autonomous Overnight Research Controller (Phase 4F.5)", padding=(8, 4))
        run_box.pack(fill="x", pady=(2, 4))

        # Row 1: Available Algorithms & Action Buttons (Single Horizontal Row)
        algo_row = ttk.Frame(run_box)
        algo_row.pack(fill="x", pady=(0, 2))

        ttk.Label(algo_row, text="Available Algorithms:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        self._cb_algo_xgb = ttk.Checkbutton(algo_row, text="XGBoost", variable=self._algo_xgb_var)
        self._cb_algo_xgb.pack(side="left", padx=(0, 8))
        self._cb_algo_cat = ttk.Checkbutton(algo_row, text="CatBoost", variable=self._algo_cat_var)
        self._cb_algo_cat.pack(side="left", padx=(0, 8))
        self._cb_algo_lgb = ttk.Checkbutton(algo_row, text="LightGBM", variable=self._algo_lgb_var)
        self._cb_algo_lgb.pack(side="left", padx=(0, 8))
        self._cb_algo_rf = ttk.Checkbutton(algo_row, text="Random Forest", variable=self._algo_rf_var)
        self._cb_algo_rf.pack(side="left", padx=(0, 8))
        self._cb_algo_et = ttk.Checkbutton(algo_row, text="Extra Trees", variable=self._algo_et_var)
        self._cb_algo_et.pack(side="left", padx=(0, 14))

        self._btn_start_research = ttk.Button(
            algo_row,
            text="▶ Start Autonomous Research",
            command=self._on_start_autonomous_research,
        )
        self._btn_start_research.pack(side="left", padx=(0, 6))

        self._btn_stop_research = ttk.Button(
            algo_row,
            text="⏹ Stop",
            command=self._on_stop_autonomous_research,
            state="disabled",
        )
        self._btn_stop_research.pack(side="left", padx=(0, 6))

        self._btn_view_dossier = ttk.Button(
            algo_row,
            text="🌅 View Morning Dossier",
            command=self._on_view_morning_dossier,
        )
        self._btn_view_dossier.pack(side="left", padx=(0, 6))

        self._btn_evidence_db = ttk.Button(
            algo_row,
            text="📊 Evidence DB",
            command=self._on_open_evidence_db,
        )
        self._btn_evidence_db.pack(side="left", padx=(0, 6))

        # Row 2: Feature Elimination Strategy Radio Buttons (Positioned directly below Available Algorithms)
        elim_row = ttk.Frame(run_box)
        elim_row.pack(fill="x", pady=(2, 2))

        ttk.Label(elim_row, text="Elimination Strategy:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        for val, lbl in (
            ("NONE", "None"),
            ("SHAP", "SHAP Importance"),
            ("RFE", "Recursive Feature Elimination"),
            ("PERMUTATION", "Permutation Importance"),
        ):
            ttk.Radiobutton(
                elim_row,
                text=lbl,
                value=val,
                variable=self._elim_strat_var,
                command=self._on_elim_strategy_changed,
            ).pack(side="left", padx=(0, 8))

        # Row 3: Budget Parameters & Termination Inputs
        params_row = ttk.Frame(run_box)
        params_row.pack(fill="x", pady=(2, 3))

        ttk.Label(params_row, text="Max Gens:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 3))
        self._spin_max_gen = ttk.Spinbox(params_row, from_=1, to=100, textvariable=self._cfg_max_gen, width=4)
        self._spin_max_gen.pack(side="left", padx=(0, 14))

        ttk.Label(params_row, text="Max Candidates:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 3))
        self._spin_max_cands = ttk.Spinbox(params_row, from_=5, to=2000, increment=10, textvariable=self._cfg_max_cands, width=5)
        self._spin_max_cands.pack(side="left", padx=(0, 14))

        ttk.Label(params_row, text="Max Hours:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 3))
        self._spin_max_hours = ttk.Spinbox(params_row, from_=0.5, to=48.0, increment=0.5, textvariable=self._cfg_max_hours, width=5)
        self._spin_max_hours.pack(side="left", padx=(0, 14))

        ttk.Label(params_row, text="Plateau Patience:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 3))
        self._spin_plateau_patience = ttk.Spinbox(params_row, from_=1, to=10, textvariable=self._cfg_plateau_patience, width=4)
        self._spin_plateau_patience.pack(side="left", padx=(0, 10))

        self._chk_plateau = ttk.Checkbutton(params_row, text="Plateau Halt", variable=self._cfg_plateau_enabled)
        self._chk_plateau.pack(side="left", padx=(4, 0))

        # Wire UI State Persistence (Debounced autosave & restore)
        self._ui_state.bind_checkbutton(self._cb_algo_xgb, "research_leaderboard.algo_xgb", var=self._algo_xgb_var, default=True)
        self._ui_state.bind_checkbutton(self._cb_algo_cat, "research_leaderboard.algo_cat", var=self._algo_cat_var, default=True)
        self._ui_state.bind_checkbutton(self._cb_algo_lgb, "research_leaderboard.algo_lgb", var=self._algo_lgb_var, default=True)
        self._ui_state.bind_checkbutton(self._cb_algo_rf, "research_leaderboard.algo_rf", var=self._algo_rf_var, default=True)
        self._ui_state.bind_checkbutton(self._cb_algo_et, "research_leaderboard.algo_et", var=self._algo_et_var, default=True)

        self._ui_state.bind_radiobutton(self._elim_strat_var, "research_leaderboard.elimination_strategy", default="NONE")
        self._on_elim_strategy_changed()

        self._ui_state.bind_spinbox(self._spin_max_gen, "research_leaderboard.max_gen", var=self._cfg_max_gen, default="10")
        self._ui_state.bind_spinbox(self._spin_max_cands, "research_leaderboard.max_cands", var=self._cfg_max_cands, default="100")
        self._ui_state.bind_spinbox(self._spin_max_hours, "research_leaderboard.max_hours", var=self._cfg_max_hours, default="8.0")
        self._ui_state.bind_spinbox(self._spin_plateau_patience, "research_leaderboard.plateau_patience", var=self._cfg_plateau_patience, default="3")
        self._ui_state.bind_checkbutton(self._chk_plateau, "research_leaderboard.plateau_enabled", var=self._cfg_plateau_enabled, default=True)

        # Row 3: Status & Message
        msg_row = ttk.Frame(run_box)
        msg_row.pack(fill="x", pady=(1, 2))

        ttk.Label(msg_row, text="Status:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 4))
        self._camp_status_var = tk.StringVar(value="IDLE")
        self._lbl_camp_status = ttk.Label(msg_row, textvariable=self._camp_status_var, font=("Segoe UI", 9, "bold"), foreground=COL_MUTED)
        self._lbl_camp_status.pack(side="left", padx=(0, 12))

        self._camp_msg_var = tk.StringVar(value="Ready to start autonomous discovery.")
        ttk.Label(msg_row, textvariable=self._camp_msg_var, font=("Segoe UI", 9), foreground="#333333").pack(side="left")

        # Row 4: Live Multi-Generation Metrics & Telemetry Strip + Evidence DB Summary
        telem_row = ttk.Frame(run_box)
        telem_row.pack(fill="x", pady=(2, 0))

        self._camp_gen_var = tk.StringVar(value="Gen: — / —")
        self._camp_cand_var = tk.StringVar(value="Candidates: 0 / 100 (Pruned: 0)")
        self._camp_runtime_var = tk.StringVar(value="Runtime: 0h 00m / 8.0h")
        self._camp_best_var = tk.StringVar(value="Best: —")
        self._camp_trade_var = tk.StringVar(value="Trading: —")

        ttk.Label(telem_row, textvariable=self._camp_gen_var, font=("Segoe UI", 8, "bold"), foreground="#0d47a1").pack(side="left", padx=(0, 12))
        ttk.Label(telem_row, textvariable=self._camp_cand_var, font=("Segoe UI", 8)).pack(side="left", padx=(0, 12))
        ttk.Label(telem_row, textvariable=self._camp_runtime_var, font=("Segoe UI", 8)).pack(side="left", padx=(0, 12))
        ttk.Label(telem_row, textvariable=self._camp_best_var, font=("Segoe UI", 8, "bold"), foreground=COL_PRODUCTION).pack(side="left", padx=(0, 12))
        ttk.Label(telem_row, textvariable=self._camp_trade_var, font=("Segoe UI", 8), foreground="#e65100").pack(side="left", padx=(0, 12))

        self._lbl_evidence_summary = ttk.Label(
            telem_row,
            textvariable=self._evidence_db_summary_var,
            font=("Segoe UI", 8, "bold"),
            foreground="#004d40",
        )
        self._lbl_evidence_summary.pack(side="right", padx=(8, 4))

        # Split View: Leaderboard Table (Top) & Detail Dossier Notebook (Bottom)
        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        # Top Paned Frame: Leaderboard Table
        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        # Leaderboard Action Bar
        tbl_act_bar = ttk.Frame(table_frame, padding=(0, 2, 0, 4))
        tbl_act_bar.pack(fill="x")

        self._btn_add_to_classifier = ttk.Button(
            tbl_act_bar,
            text="🏆 Add to Classifier",
            command=self._on_add_to_classifier,
        )
        self._btn_add_to_classifier.pack(side="left", padx=(0, 6))

        self._btn_promote_pipeline = ttk.Button(
            tbl_act_bar,
            text="📦 Promote Pipeline",
            command=self._on_promote_pipeline,
        )
        self._btn_promote_pipeline.pack(side="left", padx=(0, 10))

        ttk.Label(
            tbl_act_bar,
            text="Registers the selected candidate into the Classifier Model Registry (marked as EXPERIMENTAL).",
            font=("Segoe UI", 9, "italic"),
            foreground=COL_MUTED,
        ).pack(side="left")

        table_container = ttk.Frame(table_frame)
        table_container.pack(fill="both", expand=True)


        tree_scroll_y = ttk.Scrollbar(table_container, orient=tk.VERTICAL)
        tree_scroll_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)

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
            table_container,
            columns=cols,
            show="headings",
            height=12,
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
        paned.add(detail_outer, weight=4)

        self._detail_nb = ttk.Notebook(detail_outer)
        self._detail_nb.pack(fill="both", expand=True)

        self._tab_dossier = ScrollableFrame(self._detail_nb)
        self._tab_recommendations = ScrollableFrame(self._detail_nb)
        self._tab_regimes = ScrollableFrame(self._detail_nb)
        self._tab_features = ScrollableFrame(self._detail_nb)
        self._tab_lineage = ScrollableFrame(self._detail_nb)
        self._tab_discovered_features = ScrollableFrame(self._detail_nb)
        self._tab_history = ScrollableFrame(self._detail_nb)
        self._tab_audit = ttk.Frame(self._detail_nb, padding=6)

        self._audit_filter_var = tk.StringVar(value="ALL")
        self._audit_search_var = tk.StringVar(value="")
        self._audit_events_cache: list[dict[str, Any]] = []

        self._detail_nb.add(self._tab_dossier, text="Robustness Dossier")
        self._detail_nb.add(self._tab_recommendations, text="Research Recommendations")
        self._detail_nb.add(self._tab_regimes, text="Cross-Regime Stress")
        self._detail_nb.add(self._tab_features, text="Feature Composition")
        self._detail_nb.add(self._tab_lineage, text="Research Lineage")
        self._detail_nb.add(self._tab_discovered_features, text="🔬 Discovered Features")
        self._detail_nb.add(self._tab_history, text="Champion History")
        self._detail_nb.add(self._tab_audit, text="📜 Execution Audit Trail")

        self._detail_nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)


    def _on_context_param_changed(self) -> None:
        self._update_resolved_context_key()
        self.refresh_leaderboard()

    def refresh_leaderboard(self) -> None:
        """Query analysis.db using Phase 4D.5 ranking service for the active context key."""
        if not self._datasets:
            self._refresh_datasets_combo()

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

        # 2. Query Authoritative Ranking Dossiers from Phase 4D.5 / 4F.3
        try:
            dossiers = rank_models_in_context(data_dir, ctx_key)
        except Exception:
            dossiers = []

        if not dossiers:
            try:
                from chain_replay_ml.model_ranking.persistence import load_candidate_rankings_for_context
                cand_scores = load_candidate_rankings_for_context(data_dir, ctx_key)
                for rank_idx, cs in enumerate(cand_scores, start=1):
                    c_id = cs.candidate_id
                    algo = "xgboost"
                    if "_LIG_" in c_id or "lightgbm" in c_id.lower():
                        algo = "lightgbm"
                    elif "_CAT_" in c_id or "catboost" in c_id.lower():
                        algo = "catboost"
                    elif "_RAN_" in c_id or "random_forest" in c_id.lower():
                        algo = "random_forest"
                    elif "_EXT_" in c_id or "extra_trees" in c_id.lower():
                        algo = "extra_trees"

                    m_m = cs.model_metrics or {}
                    t_m = cs.trading_metrics or {}
                    dossiers.append({
                        "rank": rank_idx,
                        "signature_hash": cs.signature_hash,
                        "model_name": cs.candidate_id,
                        "algorithm": algo,
                        "robustness_score": cs.composite_score,
                        "pareto_rank": rank_idx,
                        "is_pareto_optimal": (rank_idx == 1),
                        "recommendation_status": cs.recommendation_class.value,
                        "feature_count": int(m_m.get("total_features", len(self._dataset_eligible_features()) or 382)),
                        "score_breakdown": cs.score_breakdown or {},
                        "raw_metrics_summary": {
                            "primary_metric_name": "ROC_AUC" if "roc_auc" in m_m else "Accuracy",
                            "primary_metric_value": m_m.get("roc_auc", m_m.get("fold_mean", 0.0)),
                            "fold_std": m_m.get("fold_std", 0.0),
                            "expected_calibration_error": m_m.get("expected_calibration_error", 0.0),
                            "avg_regime_degradation_pct": 0.0,
                            "experimental_dependency_ratio": 0.0,
                        },
                        "warnings": list(cs.warnings or []),
                    })
            except Exception:
                pass

        self._ranked_dossiers = dossiers

        # Update Research Candidate display
        if dossiers and dossiers[0].get("recommendation_status") in ("CHAMPION_CANDIDATE", "STRONG_CONTENDER"):
            top_cand = dossiers[0]["model_name"]
            score = dossiers[0]["robustness_score"]
            self._cand_champ_var.set(f"🧪 Research Champion Candidate: {top_cand} ({score:.2f} pts)")
        else:
            self._cand_champ_var.set("🧪 Research Champion Candidate: None")

        # Clear tree
        for item in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(item)

        # Clear lazy loading tab caches
        self._loaded_tab_candidate.clear()
        self._loaded_context_tabs.clear()

        if not dossiers:
            self._render_empty_detail("No benchmarked models found for this context key.")
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
                    idx,
                    d.get("model_name", "—"),
                    d.get("algorithm", "—"),
                    f"{r_score:.2f}",
                    f"#{p_rank}",
                    p_metric,
                    f_std,
                    ece,
                    deg,
                    exp_r,
                    d.get("recommendation_status", "—"),
                    d.get("feature_count", "—"),
                    sig,
                ),
            )

        # Select first item by default and render ONLY the currently active tab
        children = self.leaderboard_tree.get_children()
        if children:
            self.leaderboard_tree.selection_set(children[0])
            first_dossier = self._item_dossier_map.get(children[0], dossiers[0])
            self._selected_dossier = first_dossier
            self._on_notebook_tab_changed()

        self._refresh_evidence_db_summary()

    def _on_tree_select(self, _event: tk.Event) -> None:
        sel = self.leaderboard_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        d = self._item_dossier_map.get(item_id)
        if d:
            self._selected_dossier = d
            # Invalidate candidate-specific tab caches for this new candidate
            self._loaded_tab_candidate.clear()
            # Render only the currently active visible tab!
            self._on_notebook_tab_changed()
            if self._on_select_model and d.get("model_name"):
                self._on_select_model(d["model_name"])

    def _render_empty_detail(self, message: str) -> None:
        for tab in (self._tab_dossier, self._tab_regimes, self._tab_features, self._tab_lineage):
            clear_children(tab.inner)
            ttk.Label(tab.inner, text=message, font=("TkDefaultFont", 10, "italic"), foreground=COL_MUTED).pack(padx=16, pady=16)

    def _on_notebook_tab_changed(self, _event: Any = None) -> None:
        """Lazy-render the currently selected notebook tab on demand."""
        try:
            tab_idx = self._detail_nb.index("current")
        except Exception:
            return

        dossier = self._selected_dossier
        cand_id = dossier.get("model_name") if dossier else ""

        if tab_idx == 0:  # Robustness Dossier
            if dossier and self._loaded_tab_candidate.get("dossier") != cand_id:
                self._render_tab_dossier(dossier)
                self._loaded_tab_candidate["dossier"] = cand_id

        elif tab_idx == 1:  # Research Recommendations
            if "recommendations" not in self._loaded_context_tabs:
                self._render_recommendations_tab()
                self._loaded_context_tabs.add("recommendations")

        elif tab_idx == 2:  # Cross-Regime Stress
            if dossier and self._loaded_tab_candidate.get("regimes") != cand_id:
                self._render_tab_regimes(dossier)
                self._loaded_tab_candidate["regimes"] = cand_id

        elif tab_idx == 3:  # Feature Composition
            if dossier and self._loaded_tab_candidate.get("features") != cand_id:
                self._render_tab_features(dossier)
                self._loaded_tab_candidate["features"] = cand_id

        elif tab_idx == 4:  # Research Lineage
            if dossier and self._loaded_tab_candidate.get("lineage") != cand_id:
                self._render_tab_lineage(dossier)
                self._loaded_tab_candidate["lineage"] = cand_id

        elif tab_idx == 5:  # Discovered Features
            if "discovered" not in self._loaded_context_tabs:
                self._render_tab_discovered_features()
                self._loaded_context_tabs.add("discovered")

        elif tab_idx == 6:  # Champion History
            if "history" not in self._loaded_context_tabs:
                self._render_champion_history_tab()
                self._loaded_context_tabs.add("history")

        elif tab_idx == 7:  # Execution Audit Trail
            if "audit" not in self._loaded_context_tabs:
                self._render_audit_tab()
                self._loaded_context_tabs.add("audit")

    def _render_tab_dossier(self, dossier: dict[str, Any]) -> None:
        """Render Robustness Dossier tab for the selected candidate."""
        sig_hash = dossier.get("signature_hash", "—")
        model_name = dossier.get("model_name", "—")

        tab1 = self._tab_dossier.inner
        clear_children(tab1)
        section_title(tab1, f"Robustness Dossier: {model_name}")
        section_desc(tab1, f"Canonical Identity: {sig_hash}")

        # Top summary cards
        score = dossier.get("robustness_score", 0.0)
        p_rank = dossier.get("pareto_rank", 1)
        status = dossier.get("recommendation_status", "VALIDATED")
        exec_dev = dossier.get("execution_device") or dossier.get("raw_metrics_summary", {}).get("execution_device") or "CPU"
        dev_det = dossier.get("device_details") or dossier.get("raw_metrics_summary", {}).get("device_details") or ""
        dev_label = f"⚡ {exec_dev} ({dev_det})" if dev_det else f"⚡ {exec_dev}"

        summary_rows = [
            ("Robustness Score", f"{score:.2f} / 100.00"),
            ("Pareto Optimality Tier", f"Tier {p_rank} ({'Non-Dominated / Optimal' if p_rank == 1 else 'Dominated'})"),
            ("Execution Device", dev_label),
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

    def _render_tab_regimes(self, dossier: dict[str, Any]) -> None:
        """Render Cross-Regime Stress tab for the selected candidate."""
        data_dir = self._data_dir()
        sig_hash = dossier.get("signature_hash", "")
        model_name = dossier.get("model_name", "—")

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

    def _render_tab_features(self, dossier: dict[str, Any]) -> None:
        """Render Feature Composition tab for the selected candidate."""
        data_dir = self._data_dir()
        sig_hash = dossier.get("signature_hash", "")
        model_name = dossier.get("model_name", "—")

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
            # Query Feature Studio Evidence Store for candidate-specific feature records
            fe_records: list[dict[str, Any]] = []
            try:
                from chain_replay_ml.production_validation.evidence_store import get_connection
                ev_conn = get_connection(data_dir)
                try:
                    cur = ev_conn.execute(
                        "SELECT feature_name, recommendation, relative_imp_drop, drift_severity, evidence_detail_json, run_timestamp FROM recommendation_evidence WHERE model_name = ? ORDER BY holdout_rank ASC",
                        (model_name,),
                    )
                    for r in cur.fetchall():
                        d_json = {}
                        try:
                            d_json = json.loads(r["evidence_detail_json"] or "{}")
                        except Exception:
                            pass
                        fe_records.append({
                            "feature_name": r["feature_name"],
                            "recommendation": r["recommendation"],
                            "ks_stat": r["relative_imp_drop"],
                            "drift_severity": r["drift_severity"],
                            "evidence_score": d_json.get("evidence_score", 0.0),
                            "reason": d_json.get("reason", "—"),
                            "importance": d_json.get("importance", 0.0),
                            "rank": d_json.get("importance_rank", "—"),
                        })
                finally:
                    ev_conn.close()
            except Exception:
                pass

            # Query CandidateSpec for mutation & elimination metadata
            cand_features: list[str] = []
            strat = None
            mut_type = None
            try:
                from chain_replay_ml.overnight_campaign.persistence import load_candidate_specs_for_campaign
                c_specs = load_candidate_specs_for_campaign(data_dir, candidate_id=model_name)
                c_spec = c_specs.get(model_name)
                if c_spec:
                    cand_features = c_spec.get("features", [])
                    strat = c_spec.get("feature_elimination_strategy")
                    mut_type = c_spec.get("mutation_type")
            except Exception:
                pass

            total_cnt = len(fe_records) if fe_records else (len(cand_features) if cand_features else dossier.get("raw_metrics_summary", {}).get("total_features", 0))

            if fe_records:
                keep_cnt = sum(1 for r in fe_records if r["recommendation"] == "KEEP")
                watch_cnt = sum(1 for r in fe_records if r["recommendation"] == "WATCH")
                remove_cnt = sum(1 for r in fe_records if r["recommendation"] == "REMOVE")

                summary_rows = [
                    ("Total Candidate Features", str(total_cnt)),
                    ("Feature Studio: KEEP", f"{keep_cnt} features ({keep_cnt/max(1, total_cnt)*100:.1f}%)"),
                    ("Feature Studio: WATCH", f"{watch_cnt} features ({watch_cnt/max(1, total_cnt)*100:.1f}%)"),
                    ("Feature Studio: REMOVE", f"{remove_cnt} features ({remove_cnt/max(1, total_cnt)*100:.1f}%)"),
                    ("Elimination Strategy", str(strat or "NONE")),
                    ("Mutation Type", str(mut_type or "Candidate Specification")),
                ]
                kv_block(tab3, "Feature Studio Governance", summary_rows)

                table_rows = []
                for r in fe_records:
                    rec = r["recommendation"]
                    badge = f"🟢 {rec}" if rec == "KEEP" else (f"🟡 {rec}" if rec == "WATCH" else f"🔴 {rec}")
                    ks_str = f"{r['ks_stat']:.3f}" if r['ks_stat'] is not None else "—"
                    score_str = f"{r['evidence_score']:.1f}" if r['evidence_score'] is not None else "—"
                    table_rows.append((
                        str(r["rank"]),
                        r["feature_name"],
                        badge,
                        score_str,
                        ks_str,
                        r["reason"],
                    ))

                data_table(
                    tab3,
                    [
                        ("rank", "Rank", 50),
                        ("feat", "Feature Name", 180),
                        ("rec", "Studio Decision", 110),
                        ("ev_score", "Score", 60),
                        ("ks", "KS Drift", 70),
                        ("reason", "Auditable Governance Reason", 320),
                    ],
                    table_rows,
                )
            else:
                rows = [
                    ("Total Features Count", str(total_cnt)),
                    ("Mutation Type", str(mut_type or "Full Baseline / Candidate Spec")),
                    ("Elimination Strategy", str(strat or "NONE")),
                    ("Audit Status", "Candidate Specification verified"),
                ]
                kv_block(tab3, "Composition", rows)
                if cand_features:
                    preview_feats = [[f] for f in cand_features[:30]]
                    data_table(tab3, [("feat", f"Sample Features (Showing {len(preview_feats)} of {len(cand_features)})", 300)], preview_feats)

    def _render_tab_lineage(self, dossier: dict[str, Any]) -> None:
        """Render Research Lineage tab for the selected candidate."""
        data_dir = self._data_dir()
        sig_hash = dossier.get("signature_hash", "")
        model_name = dossier.get("model_name", "—")

        tab4 = self._tab_lineage.inner
        clear_children(tab4)
        section_title(tab4, f"End-to-End Research Lineage: {model_name}")
        section_desc(tab4, "Verifiable audit chain from Campaign to Benchmark and Candidate.")

        bm_id = dossier.get("benchmark_id", "—")
        run_id = dossier.get("benchmark_run_id", "—")

        # Try to resolve campaign_id from benchmark_run or candidate spec
        camp_id = None
        parent_id = None
        mut_desc = None
        elim_strat = None

        if run_id and run_id != "—":
            try:
                b_run = get_benchmark_run(data_dir, run_id)
                if b_run:
                    camp_id = b_run.get("campaign_id")
            except Exception:
                pass

        try:
            from chain_replay_ml.overnight_campaign.persistence import load_candidate_specs_for_campaign
            c_specs = load_candidate_specs_for_campaign(data_dir, candidate_id=model_name)
            c_spec = c_specs.get(model_name)
            if c_spec:
                camp_id = camp_id or c_spec.get("campaign_id")
                parent_id = c_spec.get("parent_candidate_id")
                mut_desc = c_spec.get("mutation_description")
                elim_strat = c_spec.get("feature_elimination_strategy")
        except Exception:
            pass

        lineage_rows = [
            ("Campaign ID", camp_id or "Direct / Independent Run"),
            ("Parent Candidate ID", parent_id or "Root Baseline (Gen 0)"),
            ("Mutation Description", mut_desc or "Baseline Candidate"),
            ("Elimination Strategy", elim_strat or "NONE"),
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

    def _render_tab_discovered_features(self) -> None:
        """Render Autonomous Discovery Pipeline features, generations, snapshots, and telemetry."""
        tab = self._tab_discovered_features.inner
        clear_children(tab)
        ctx_key = self._context_key_var.get()
        data_dir = self._data_dir()

        section_title(tab, f"Autonomous Discovery Pipeline Sandbox: {ctx_key}")
        section_desc(tab, "Isolated research sandbox synthesizing, evaluating, and evolving experimental features across multi-generation campaigns.")

        # Try to resolve latest Discovery Pipeline for this context or campaign
        pipe: DiscoveryPipelineSpec | None = None
        try:
            from chain_replay_ml.research_memory.db import connect_analysis_db
            conn = connect_analysis_db(data_dir)
            try:
                row = conn.execute(
                    """
                    SELECT * FROM discovery_pipelines
                    WHERE context_key = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (ctx_key,),
                ).fetchone()
                if row:
                    pipe = DiscoveryPipelineSpec.from_dict(dict(row))
            finally:
                conn.close()
        except Exception:
            pipe = None

        if not pipe:
            ttk.Label(
                tab,
                text=f"No active Discovery Pipeline found for context '{ctx_key}'.\nRun an Autonomous Research Campaign to discover, evaluate, and evolve novel features.",
                font=("TkDefaultFont", 9, "italic"),
                foreground=COL_MUTED,
                justify="center",
            ).pack(pady=24)
            return

        # 1. Pipeline Summary Cards
        pipe_id = pipe.pipeline_id
        camp_id = pipe.campaign_id
        gen_num = pipe.current_generation
        snap_hash = pipe.current_snapshot_hash or "DP_SNAP_INITIAL"
        active_cnt = pipe.active_features_count
        total_cnt = pipe.total_generated_count

        all_features = load_discovered_features(data_dir, pipe_id)
        keeps = [f for f in all_features if f.lifecycle_status == DiscoveryLifecycleStatus.KEEP]
        watches = [f for f in all_features if f.lifecycle_status == DiscoveryLifecycleStatus.WATCH]
        removes = [f for f in all_features if f.lifecycle_status == DiscoveryLifecycleStatus.REMOVE]

        summary_rows = [
            ("Discovery Pipeline ID", pipe_id),
            ("Base Pipeline Anchor", f"{pipe.base_pipeline_id} ({pipe.base_feature_count} base features)"),
            ("Owning Campaign ID", camp_id),
            ("Current Generation", f"Generation {gen_num}"),
            ("Current Snapshot Hash", snap_hash),
            ("Active Pool Size", f"{active_cnt} surviving features ({len(keeps)} KEEPs + {len(watches)} WATCHes)"),
            ("Cumulative Generated", f"{total_cnt} experimental features synthesized"),
            ("Governance Breakdown", f"🟢 KEEP: {len(keeps)}  |  🟡 WATCH: {len(watches)}  |  🔴 REMOVE: {len(removes)}"),
        ]
        kv_block(tab, "Pipeline Metadata & Status", summary_rows)

        # 2. Discovered Features Table
        if all_features:
            section_title(tab, "Discovered Features Telemetry & Governance Status")
            feat_rows = []
            for idx, f in enumerate(all_features, start=1):
                st_icon = "🟢 KEEP" if f.lifecycle_status == DiscoveryLifecycleStatus.KEEP else ("🟡 WATCH" if f.lifecycle_status == DiscoveryLifecycleStatus.WATCH else "🔴 REMOVE")
                strat_val = f.generator_strategy.value if hasattr(f.generator_strategy, "value") else str(f.generator_strategy)
                d_auc = f.metadata.get("delta_auc", 0.0) if f.metadata else 0.0
                cons = f.metadata.get("fold_consistency", 0.0) if f.metadata else 0.0
                parents_str = ", ".join(f.parent_features) if f.parent_features else "—"

                feat_rows.append((
                    str(idx),
                    f.feature_name,
                    strat_val,
                    st_icon,
                    f"{f.evidence_score:.2f}",
                    f"{d_auc:+.5f}",
                    f"{f.ks_statistic:.4f}",
                    f"{cons*100:.0f}%",
                    f"Gen {f.generation_discovered}",
                    f.formula_expression[:60] + ("..." if len(f.formula_expression) > 60 else ""),
                    parents_str,
                ))

            data_table(
                tab,
                [
                    ("idx", "#", 30),
                    ("feat", "Feature Name", 220),
                    ("strat", "Strategy", 100),
                    ("status", "Status", 95),
                    ("score", "Score", 65),
                    ("delta_auc", "ΔAUC", 85),
                    ("drift", "D_KS Drift", 80),
                    ("cons", "Consistency", 85),
                    ("gen", "Discovered", 80),
                    ("formula", "Formula Expression", 240),
                    ("parents", "Parent Features", 160),
                ],
                feat_rows,
            )

            # 3. Top / Governed Feature Formula Inspector
            top_feature = keeps[0] if keeps else (watches[0] if watches else all_features[0])
            top_meta = top_feature.metadata or {}
            top_delta = top_meta.get("delta_auc", 0.0)
            top_base_auc = top_meta.get("baseline_auc", 0.5)
            top_cons = top_meta.get("fold_consistency", 0.5)
            top_rationale = top_meta.get("governance_rationale", "Evaluated via 5-fold walk-forward cross-validation.")

            inspector_rows = [
                ("Selected Feature", top_feature.feature_name),
                ("Feature Identifier", top_feature.feature_id),
                ("Canonical Formula Hash", top_feature.formula_hash),
                ("Mathematical Expression", top_feature.formula_expression),
                ("Generator Strategy", top_feature.generator_strategy.value if hasattr(top_feature.generator_strategy, "value") else str(top_feature.generator_strategy)),
                ("Parent Input Features", ", ".join(top_feature.parent_features)),
                ("Empirical Marginal Gain", f"ΔAUC = {top_delta:+.5f} (Baseline: {top_base_auc:.4f} → Augmented: {top_base_auc + top_delta:.4f})"),
                ("Distribution Drift Test", f"D_KS = {top_feature.ks_statistic:.4f} (p-value: {top_feature.ks_pvalue:.4f}, Severity: {top_feature.drift_severity})"),
                ("Walk-Forward Stability", f"Fold Consistency = {top_cons*100:.0f}% across 5 expanding folds"),
                ("Evidence Score", f"{top_feature.evidence_score:.2f} / 100.00"),
                ("Governance Verdict", f"{top_feature.lifecycle_status.value} — {top_rationale}"),
            ]
            kv_block(tab, f"Mathematical Provenance & Evidence Inspector: {top_feature.feature_name}", inspector_rows)

        # 4. Cryptographic Snapshots History
        snapshots = load_discovery_snapshots_for_pipeline(data_dir, pipe_id)
        if snapshots:
            section_title(tab, "Cryptographic Generation Snapshots (DP_SNAP_*)")
            snap_rows = []
            for s in snapshots:
                snap_rows.append((
                    f"Gen {s.generation_number}",
                    s.snapshot_hash,
                    str(s.feature_count),
                    str(s.keep_count),
                    str(s.watch_count),
                    str(s.remove_count),
                    s.created_at[:19],
                ))
            data_table(
                tab,
                [
                    ("gen", "Generation", 90),
                    ("hash", "Snapshot Hash", 240),
                    ("count", "Active Features", 110),
                    ("keeps", "KEEPs", 75),
                    ("watches", "WATCHes", 75),
                    ("removes", "REMOVEs", 75),
                    ("created", "Created At", 150),
                ],
                snap_rows,
            )

    def _get_selected_algorithms(self) -> list[str]:
        """Return list of canonical algorithm IDs selected by the researcher."""
        selected: list[str] = []
        if self._algo_xgb_var.get():
            selected.append("xgboost")
        if self._algo_cat_var.get():
            selected.append("catboost")
        if self._algo_lgb_var.get():
            selected.append("lightgbm")
        if self._algo_rf_var.get():
            selected.append("random_forest")
        if self._algo_et_var.get():
            selected.append("extra_trees")
        return selected

    def _elim_strategy_display_name(self, strat: str | None = None) -> str:
        s = str(strat or self._elim_strat_var.get() or "NONE").strip().upper()
        mapping = {
            "NONE": "None",
            "SHAP": "SHAP Importance",
            "RFE": "Recursive Feature Elimination",
            "PERMUTATION": "Permutation Importance",
        }
        return mapping.get(s, s)

    def _on_elim_strategy_changed(self) -> None:
        self._elim_strat_status_var.set(f"🎯 Elimination Strategy: {self._elim_strategy_display_name()}")

    def _on_start_autonomous_research(self) -> None:
        """Start autonomous overnight campaign on a background thread for the active ModelContextKey."""
        if self._is_running:
            messagebox.showwarning("Campaign In Progress", "An autonomous research campaign is already running.")
            return

        ctx_key = self._context_key_var.get()
        if not ctx_key:
            messagebox.showerror("Error", "Please select a valid ModelContextKey first.")
            return

        dataset_name = self._selected_dataset_name()
        if not dataset_name:
            messagebox.showerror("No Dataset Selected", "Please select an authoritative Dataset from the dropdown before starting research.")
            return

        data_dir = self._data_dir()
        if not self._dataset_meta:
            try:
                from .model_builder import service
                self._dataset_meta = service.load_dataset_metadata_doc(data_dir, dataset_name)
            except Exception as exc:
                messagebox.showerror("Dataset Error", f"Could not load metadata for dataset '{dataset_name}':\n{exc}")
                return

        meta = (self._dataset_meta or {}).get("metadata") or {}
        parquet_rel = str(meta.get("output_parquet") or "").strip()
        parquet_path = os.path.join(data_dir, parquet_rel) if parquet_rel and not os.path.isabs(parquet_rel) else (
            str(meta.get("parquet_path") or parquet_rel)
        )
        if parquet_path and not os.path.exists(parquet_path):
            messagebox.showwarning(
                "Dataset Parquet Missing",
                f"Parquet file for dataset '{dataset_name}' not found on disk:\n{parquet_path}\n\n"
                "Please export or rebuild this dataset before training."
            )
            return

        eligible_features = self._dataset_eligible_features()
        if not eligible_features:
            messagebox.showerror(
                "No Features",
                f"Dataset '{dataset_name}' contains no eligible feature columns."
            )
            return

        target_col = self._resolve_target_column(meta)
        dataset_snapshot_hash = str(
            meta.get("dataset_fingerprint")
            or meta.get("schema_hash")
            or meta.get("dataset_hash")
            or "dataset_snapshot_v1"
        )

        # Generate unique campaign ID
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_id = f"CAMP_{ctx_key}_{timestamp_str}"
        self._last_campaign_id = campaign_id
        self._campaign_start_ts = datetime.now().timestamp()

        # Read configured budget from UI controls (with safety minimums)
        max_gen = max(1, self._cfg_max_gen.get())
        max_cands = max(1, self._cfg_max_cands.get())
        max_hours = max(0.1, float(self._cfg_max_hours.get()))
        patience = max(1, self._cfg_plateau_patience.get())
        elim_strat = self._elim_strat_var.get() or "NONE"
        self._elim_strat_status_var.set(f"🎯 Elimination Strategy: {self._elim_strategy_display_name(elim_strat)}")

        selected_algos = self._get_selected_algorithms()
        if not selected_algos:
            messagebox.showerror(
                "No Algorithm Selected",
                "Please select at least one algorithm from Available Algorithms (XGBoost, CatBoost, LightGBM, Random Forest, Extra Trees)."
            )
            return

        # Construct CampaignConfig using active researcher budget and full dataset feature universe
        config = CampaignConfig(
            campaign_id=campaign_id,
            context_keys=[ctx_key],
            max_duration_hours=max_hours,
            max_candidates_total=max_cands,
            max_generations=max_gen,
            plateau_enabled=bool(self._cfg_plateau_enabled.get()),
            plateau_patience_generations=patience,
            plateau_min_lift=float(self._cfg_plateau_min_lift.get()),
            min_generations_before_plateau=int(self._cfg_min_gen_before_plateau.get()),
            dataset_name=dataset_name,
            dataset_path=parquet_path,
            dataset_snapshot_hash=dataset_snapshot_hash,
            dataset_feature_universe=eligible_features,
            target_column=target_col,
            feature_elimination_strategy=elim_strat,
            allowed_algorithms=selected_algos,
        )

        self._active_runner = OvernightCampaignRunner(data_dir=data_dir, config=config)
        self._is_running = True

        self._btn_start_research.config(state="disabled")
        self._btn_stop_research.config(state="normal")
        self._camp_status_var.set("STARTING")
        self._lbl_camp_status.config(foreground=COL_TRAINING)
        self._camp_msg_var.set(
            f"Starting autonomous campaign: {campaign_id} (Dataset: {dataset_name} · {len(eligible_features)} features · Elimination: {self._elim_strategy_display_name(elim_strat)} · {max_gen} Gens, {max_cands} Cands)..."
        )

        def _progress_cb(st: CampaignState, msg: str) -> None:
            self.after(0, lambda s=st, m=msg: self._update_campaign_telemetry(s, m))

        def _worker() -> None:
            try:
                report = self._active_runner.run(progress_callback=_progress_cb)
                self.after(0, lambda r=report: self._on_campaign_completed(r))
            except Exception as ex:
                import traceback
                err_msg = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
                self.after(0, lambda err=err_msg: self._on_campaign_error(err))

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()


    def _on_stop_autonomous_research(self) -> None:
        """Gracefully request campaign cancellation."""
        if self._active_runner and self._is_running:
            self._camp_msg_var.set("Stopping campaign gracefully (finishing active candidate)...")
            self._camp_status_var.set("STOPPING")
            self._lbl_camp_status.config(foreground=COL_WARN)
            self._active_runner.cancel()

    def _update_campaign_telemetry(self, st: CampaignState, msg: str) -> None:
        """Update live telemetry labels on the UI thread."""
        self._camp_status_var.set(st.status.value)
        if st.status == CampaignStatus.RUNNING:
            self._lbl_camp_status.config(foreground=COL_TRAINING)
        elif st.status in (CampaignStatus.COMPLETED,):
            self._lbl_camp_status.config(foreground=COL_OK)
        elif st.status in (CampaignStatus.CAMPAIGN_FAILED, CampaignStatus.CAMPAIGN_STOPPED):
            self._lbl_camp_status.config(foreground=COL_WARN)

        self._camp_msg_var.set(msg)

        max_gen = self._cfg_max_gen.get()
        max_cands = self._cfg_max_cands.get()
        max_hours = self._cfg_max_hours.get()

        elapsed_sec = max(0.0, datetime.now().timestamp() - getattr(self, "_campaign_start_ts", datetime.now().timestamp()))
        elapsed_hours = int(elapsed_sec // 3600)
        elapsed_mins = int((elapsed_sec % 3600) // 60)

        self._camp_gen_var.set(f"Gen: {st.current_generation + 1} / {max_gen}")
        self._camp_cand_var.set(f"Candidates: {st.total_candidates_trained} / {max_cands} (Pruned: {st.total_candidates_pruned})")
        self._camp_runtime_var.set(f"Runtime: {elapsed_hours}h {elapsed_mins:02d}m / {max_hours:.1f}h")
        if st.best_candidate_id:
            lift = st.best_composite_score - st.starting_best_score
            lift_str = f"+{lift:.2f}" if lift >= 0 else f"{lift:.2f}"
            self._camp_best_var.set(f"Best: {st.best_candidate_id} (Score: {st.best_composite_score:.2f}, Lift: {lift_str})")
        if st.best_trading_score > 0:
            self._camp_trade_var.set(f"Trading Score: {st.best_trading_score:.2f} | Model Score: {st.best_model_score:.2f}")

    def _on_campaign_completed(self, report: OvernightCampaignReport) -> None:
        """Handle campaign completion on the UI thread."""
        self._is_running = False
        self._btn_start_research.config(state="normal")
        self._btn_stop_research.config(state="disabled")
        self._camp_status_var.set(report.status.value)
        self._lbl_camp_status.config(foreground=COL_OK if report.status == CampaignStatus.COMPLETED else COL_WARN)
        stop_reason_str = report.stop_reason.value if report.stop_reason else "Completed"
        best_cand_name = report.best_candidate.candidate_id if report.best_candidate else "None"
        self._camp_msg_var.set(f"Campaign finished: {stop_reason_str}. Top Candidate: {best_cand_name} (Score: {report.best_composite_score:.2f})")

        # Refresh Leaderboard automatically
        self.refresh_leaderboard()
        try:
            messagebox.showinfo(
                "Autonomous Research Complete",
                f"Campaign {report.campaign_id} finished.\n\n"
                f"Stop Reason: {stop_reason_str}\n"
                f"Generations Completed: {report.total_generations_completed}\n"
                f"Total Candidates Trained: {report.total_candidates_trained}\n"
                f"Total Candidates Pruned: {report.total_candidates_pruned}\n"
                f"Best Discovered Candidate: {best_cand_name}\n"
                f"Best Composite Score: {report.best_composite_score:.2f} / 100.0 (Lift: +{report.total_score_improvement:.2f})\n\n"
                f"Click '🌅 View Morning Dossier' to review full lineage and governance audit."
            )
        except Exception:
            pass

    def _on_campaign_error(self, err_msg: str) -> None:
        """Handle campaign unexpected error on the UI thread."""
        self._is_running = False
        self._btn_start_research.config(state="normal")
        self._btn_stop_research.config(state="disabled")
        self._camp_status_var.set("ERROR")
        self._lbl_camp_status.config(foreground=COL_WARN)
        self._camp_msg_var.set(f"Campaign error: {err_msg}")
        try:
            messagebox.showerror("Campaign Error", f"An error occurred during autonomous research execution:\n{err_msg}")
        except Exception:
            pass

    def _on_view_morning_dossier(self) -> None:
        """Open a dedicated window displaying the complete Morning Research Dossier for the active campaign or context."""
        camp_id = getattr(self, "_last_campaign_id", None)
        ctx_key = self._context_key_var.get()
        top = tk.Toplevel(self)
        top.title(f"🌅 Morning Research Dossier — {ctx_key}")
        top.geometry("1188x875")
        top.minsize(1000, 750)
        from .morning_research_dossier_panel import MorningResearchDossierPanel
        panel = MorningResearchDossierPanel(
            top,
            data_dir=self._data_dir(),
            on_select_model=self._on_select_model,
        )
        if camp_id:
            panel.selected_campaign_id.set(camp_id)
            panel.load_selected_campaign()
        panel.pack(fill=tk.BOTH, expand=True)

    def _on_open_evidence_db(self) -> None:
        """Open the existing Feature Recommendation Evidence DB / Feature Studio inspector."""
        from .feature_recommendation_viewer import open_feature_recommendation_viewer

        market = self._market_var.get() or "NIFTY"
        try:
            int_sec = int(str(self._sampling_var.get() or "6").replace("s", ""))
        except (TypeError, ValueError):
            int_sec = 6

        open_feature_recommendation_viewer(
            self,
            chart_dir=self.chart_dir or "",
            initial_market=market,
            initial_interval_sec=int_sec,
            initial_sliding_window="standard",
            initial_feature_project_id="all",
            on_changed=self._refresh_evidence_db_summary,
        )

    def _refresh_evidence_db_summary(self) -> None:
        """Query feature_recommendation_evidence.db for compact summary counters."""
        data_dir = self._data_dir()
        if not data_dir:
            self._evidence_db_summary_var.set("📊 Evidence DB: Ready")
            return

        try:
            from chain_replay_ml.production_validation.evidence_store import get_connection
            conn = get_connection(data_dir)
            try:
                # Query recommendation_evidence for total evaluations and verdict distribution
                cur = conn.execute("""
                    SELECT 
                        COUNT(*),
                        COUNT(DISTINCT feature_name),
                        COUNT(DISTINCT model_name),
                        SUM(CASE WHEN recommendation='KEEP' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN recommendation='WATCH' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN recommendation='REMOVE' THEN 1 ELSE 0 END)
                    FROM recommendation_evidence
                """)
                row = cur.fetchone()
                total_ev = int(row[0] or 0)
                uniq_feats = int(row[1] or 0)
                uniq_models = int(row[2] or 0)
                keep_cnt = int(row[3] or 0)
                watch_cnt = int(row[4] or 0)
                rem_cnt = int(row[5] or 0)

                # Query latest/current governance status per unique feature
                cur_uniq = conn.execute("""
                    WITH latest_evals AS (
                        SELECT feature_name, recommendation
                        FROM (
                            SELECT feature_name, recommendation,
                                   ROW_NUMBER() OVER (PARTITION BY feature_name ORDER BY run_timestamp DESC, rowid DESC) as rn
                            FROM recommendation_evidence
                        )
                        WHERE rn = 1
                    )
                    SELECT 
                        SUM(CASE WHEN recommendation='KEEP' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN recommendation='WATCH' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN recommendation='REMOVE' THEN 1 ELSE 0 END)
                    FROM latest_evals
                """)
                uniq_row = cur_uniq.fetchone()
                uniq_keep = int(uniq_row[0] or 0)
                uniq_watch = int(uniq_row[1] or 0)
                uniq_rem = int(uniq_row[2] or 0)

                if total_ev > 0:
                    summary_text = (
                        f"📊 Evidence DB: {total_ev:,} evaluations · {uniq_feats:,} unique features · {uniq_models} models  |  "
                        f"Evaluations: 🟢 {keep_cnt:,} KEEP · 🟡 {watch_cnt:,} WATCH · 🔴 {rem_cnt:,} REMOVE  |  "
                        f"Current Features: 🟢 {uniq_keep:,} KEEP · 🟡 {uniq_watch:,} WATCH · 🔴 {uniq_rem:,} REMOVE"
                    )
                else:
                    summary_text = "📊 Evidence DB: 0 evaluations"
                self._evidence_db_summary_var.set(summary_text)
            finally:
                conn.close()
        except Exception:
            self._evidence_db_summary_var.set("📊 Evidence DB: Ready")


    def _on_add_to_classifier(self) -> None:
        """Register selected candidate model into Classifier Model Registry."""
        d = self._selected_dossier
        if not d or not d.get("model_name"):
            messagebox.showwarning("Select Candidate", "Please select a candidate model from the Leaderboard table first.")
            return

        cand_id = d["model_name"]
        data_dir = self._data_dir()
        if not data_dir:
            return

        try:
            from chain_replay_ml.training.classifier_registration import register_research_candidate_as_classifier
            res = register_research_candidate_as_classifier(
                data_dir,
                cand_id,
                campaign_id=getattr(self, "_last_campaign_id", None),
            )
            messagebox.showinfo(
                "Added to Classifier Registry",
                f"Candidate model '{cand_id}' has been registered in the Classifier Model Registry.\n\n"
                f"Package Location: {res['package_dir']}\n"
                f"Context Key: {res['context_key']}\n"
                f"Composite Score: {res['composite_score']:.2f}\n\n"
                f"Governance: Status marked as EXPERIMENTAL.\n"
                f"Lineage, features, hyperparameters, and research replay evidence preserved.\n"
                f"Production promotion requires explicit human review."
            )
            if self._on_select_model:
                try:
                    self._on_select_model(cand_id)
                except Exception:
                    pass
        except Exception as ex:
            messagebox.showerror("Registration Error", f"Failed to register model as classifier:\n{ex}")

    def _on_promote_pipeline(self) -> None:
        """Promote selected candidate's exact feature set to an authoritative Pipeline Snapshot."""
        d = self._selected_dossier
        if not d or not d.get("model_name"):
            messagebox.showwarning("Select Candidate", "Please select a candidate model from the Leaderboard table first.")
            return

        cand_id = d["model_name"]
        algo = d.get("algorithm", "model")
        data_dir = self._data_dir()
        if not data_dir:
            return

        # 1. Resolve CandidateSpec
        spec = None
        try:
            from chain_replay_ml.overnight_campaign.persistence import load_candidate_specs_for_campaign
            specs = load_candidate_specs_for_campaign(data_dir, candidate_id=cand_id)
            spec_dict = specs.get(cand_id)
            if spec_dict:
                from chain_replay_ml.candidate_generation.generator import create_candidate_spec
                from chain_replay_ml.candidate_generation.types import MutationType
                spec = create_candidate_spec(
                    context_key=spec_dict.get("context_key", self._context_key_var.get()),
                    algorithm=spec_dict.get("algorithm", algo),
                    features=spec_dict.get("features", []),
                    dataset_snapshot_hash=self._dataset_meta.get("metadata", {}).get("dataset_fingerprint", "snapshot_v1") if self._dataset_meta else "snapshot_v1",
                    mutation_type=MutationType(spec_dict.get("mutation_type", "FULL_FEATURE_BASELINE")),
                    campaign_id=getattr(self, "_last_campaign_id", None) or spec_dict.get("campaign_id", "CAMP_PROMOTED"),
                    feature_elimination_strategy=spec_dict.get("feature_elimination_strategy", "NONE"),
                )
        except Exception:
            pass

        if spec is None:
            # Fallback construct spec from dossier features
            from chain_replay_ml.candidate_generation.generator import create_candidate_spec
            feats = list(self._dataset_eligible_features())
            spec = create_candidate_spec(
                context_key=self._context_key_var.get(),
                algorithm=algo,
                features=feats,
                dataset_snapshot_hash=self._dataset_meta.get("metadata", {}).get("dataset_fingerprint", "snapshot_v1") if self._dataset_meta else "snapshot_v1",
                campaign_id=getattr(self, "_last_campaign_id", None) or "CAMP_PROMOTED",
            )

        # 2. Validate Candidate for Promotion
        from chain_replay_ml.dataset_builder.pipeline_promotion_engine import (
            validate_candidate_for_promotion,
            promote_candidate_to_pipeline_snapshot,
        )

        val_report = validate_candidate_for_promotion(data_dir, spec)
        if not val_report.eligible:
            messagebox.showerror(
                "Promotion Blocked",
                f"Candidate '{cand_id}' cannot be promoted to a Pipeline Snapshot due to governance violations:\n\n"
                + "\n".join(f"• {r}" for r in val_report.blocked_reasons)
            )
            return

        # 3. Confirmation Dialog
        warn_text = f"\n⚠️ Warnings ({len(val_report.warnings)}):\n" + "\n".join(f"• {w}" for w in val_report.warnings[:3]) if val_report.warnings else ""
        ds_name = self._selected_dataset_name() or "Active Dataset"

        confirm = messagebox.askyesno(
            "Confirm Pipeline Promotion",
            f"Promote candidate '{cand_id}' ({algo}) to an authoritative Pipeline Snapshot?\n\n"
            f"• Exact Features to Promote: {val_report.feature_count} features\n"
            f"• Feature Studio Governance: {val_report.keep_count} KEEP, {val_report.watch_count} WATCH, {val_report.remove_count} REMOVE\n"
            f"• Mean Evidence Score: {val_report.mean_evidence_score:.1f} pts\n"
            f"• Dataset Source: {ds_name}\n"
            f"{warn_text}\n"
            f"This will allocate a permanent Pipeline ID (PL_XXXX) and link Feature Registry IDs in the pipeline store.",
        )
        if not confirm:
            return

        # 4. Execute Promotion Engine
        try:
            res = promote_candidate_to_pipeline_snapshot(
                data_dir,
                spec,
                campaign_id=getattr(self, "_last_campaign_id", None),
                dataset_name=ds_name,
            )
            self._refresh_evidence_db_summary()
            messagebox.showinfo(
                "Pipeline Promoted Successfully",
                f"Candidate '{cand_id}' has been promoted to a registered Pipeline Snapshot!\n\n"
                f"• Pipeline ID: {res.pipeline_id}\n"
                f"• Snapshot ID: {res.pipeline_snapshot_id}\n"
                f"• Exact Features: {res.feature_count}\n"
                f"• Feature Registry IDs Linked: {res.registry_feature_ids_count}\n"
                f"• Status: {res.status}\n\n"
                f"The promoted pipeline is now available in Model Builder and the Feature Pipeline Registry.",
            )
        except Exception as ex:
            messagebox.showerror("Promotion Error", f"Failed to promote pipeline:\n{ex}")

    def _render_audit_tab(self) -> None:
        """Render complete chronological execution audit trail for the active campaign or context."""
        clear_children(self._tab_audit)

        # Top filter bar
        ctrl_bar = ttk.Frame(self._tab_audit, padding=(0, 0, 0, 6))
        ctrl_bar.pack(fill="x")

        ttk.Label(ctrl_bar, text="Filter Event Type:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 4))
        filter_cb = ttk.Combobox(
            ctrl_bar,
            textvariable=self._audit_filter_var,
            values=["ALL", "CANDIDATE", "METRICS", "CHAMPION", "DECISIONS", "WARNINGS"],
            width=14,
            state="readonly",
        )
        filter_cb.pack(side="left", padx=(0, 12))
        filter_cb.bind("<<ComboboxSelected>>", lambda _e: self._populate_leaderboard_audit_events())

        ttk.Label(ctrl_bar, text="Search Candidate/Keyword:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 4))
        search_entry = ttk.Entry(ctrl_bar, textvariable=self._audit_search_var, width=24)
        search_entry.pack(side="left", padx=(0, 8))
        search_entry.bind("<Return>", lambda _e: self._populate_leaderboard_audit_events())

        ttk.Button(ctrl_bar, text="🔍 Search / Filter", command=self._populate_leaderboard_audit_events).pack(side="left", padx=(0, 6))
        ttk.Button(ctrl_bar, text="🔄 Reset Filters", command=self._reset_leaderboard_audit_filters).pack(side="left")

        # Paned Window (Top: Events Treeview, Bottom: JSON Details Pane)
        paned = ttk.PanedWindow(self._tab_audit, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True)

        top_pane = ttk.Frame(paned)
        paned.add(top_pane, weight=3)

        tree_frame = ttk.Frame(top_pane)
        tree_frame.pack(fill="both", expand=True)

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        cols = ("timestamp", "generation", "candidate_id", "event_type", "message")
        self._audit_tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            height=8,
            selectmode="browse",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
        )
        tree_scroll_y.config(command=self._audit_tree.yview)
        tree_scroll_x.config(command=self._audit_tree.xview)

        tree_scroll_y.pack(side="right", fill="y")
        tree_scroll_x.pack(side="bottom", fill="x")
        self._audit_tree.pack(side="left", fill="both", expand=True)

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

        self._audit_tree.bind("<<TreeviewSelect>>", self._on_leaderboard_audit_selected)

        # Bottom Pane: JSON Details Inspector
        bot_pane = ttk.LabelFrame(paned, text="Selected Event Audit Payload (JSON Details)", padding=6)
        paned.add(bot_pane, weight=2)

        self._audit_detail_text = tk.Text(bot_pane, height=6, wrap="none", font=("Consolas", 9))
        txt_scroll_y = ttk.Scrollbar(bot_pane, orient=tk.VERTICAL, command=self._audit_detail_text.yview)
        txt_scroll_x = ttk.Scrollbar(bot_pane, orient=tk.HORIZONTAL, command=self._audit_detail_text.xview)
        self._audit_detail_text.config(yscrollcommand=txt_scroll_y.set, xscrollcommand=txt_scroll_x.set)

        txt_scroll_y.pack(side="right", fill="y")
        txt_scroll_x.pack(side="bottom", fill="x")
        self._audit_detail_text.pack(side="left", fill="both", expand=True)

        self._populate_leaderboard_audit_events()

    def _reset_leaderboard_audit_filters(self) -> None:
        self._audit_filter_var.set("ALL")
        self._audit_search_var.set("")
        self._populate_leaderboard_audit_events()

    def _populate_leaderboard_audit_events(self) -> None:
        if not hasattr(self, "_audit_tree"):
            return

        for item in self._audit_tree.get_children():
            self._audit_tree.delete(item)

        filter_val = self._audit_filter_var.get()
        search_val = self._audit_search_var.get().strip()

        from chain_replay_ml.overnight_campaign.persistence import load_campaign_events
        camp_id = getattr(self, "_last_campaign_id", None)
        events = load_campaign_events(
            self._data_dir(),
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
            self._on_leaderboard_audit_selected(None)
        else:
            self._audit_detail_text.config(state="normal")
            self._audit_detail_text.delete("1.0", tk.END)
            self._audit_detail_text.insert(tk.END, "// No audit events found matching the active filter criteria.")
            self._audit_detail_text.config(state="disabled")

    def _on_leaderboard_audit_selected(self, _event: Any) -> None:
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





