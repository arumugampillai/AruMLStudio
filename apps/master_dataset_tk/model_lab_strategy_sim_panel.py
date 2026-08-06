"""Research Lab — Strategy Simulator (trading outcomes on Prediction Dataset).

Sibling to Research Dashboard. Consumes the active lab Prediction Dataset —
not prediction runs. Model-evaluation metrics stay out of this tab.
"""

from __future__ import annotations

import csv
import json
import tkinter as tk
from collections import defaultdict
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from .build_service import chart_data_dir
from .model_registry_strategy import (
    _fmt_num,
    _fmt_signed_pnl,
    _pnl_tag,
    _resolve_gross_net_profit,
    _summary_metric_sections,
)
from .model_registry_widgets import (
    COL_MUTED,
    COL_OK,
    COL_WARN,
    SECTION_FONT,
    ScrollableFrame,
    dual_spec_sections,
    fmt_pct,
    fmt_rows,
    fmt_rupee,
    inline_spec_rows,
    section_title,
)

_ALL_DAYS = "All Days"
_CLASSIFIER_DISABLED = "Disabled"
_PROB_DISABLED = "Disabled"
_THRESHOLD_TAB_TEXT = "Prediction Thresholds"

# Registry of "Prediction Thresholds" sub-tabs: (title, builder-method-name-or-None).
# ``None`` means the analysis isn't implemented yet — render a "coming soon" stub.
# Only "Confidence" has a working threshold sweep today (it's the pre-existing
# Classifier Threshold analysis, unchanged). Adding a new family later (e.g. once
# Meta-model threshold tuning exists) means adding one row here plus one builder
# method — no redesign of the tab container itself.
_THRESHOLD_SUBTAB_REGISTRY: tuple[tuple[str, str | None], ...] = (
    ("Probability Ladder", None),
    ("Triple Barrier", "_build_tb_threshold_subtab"),
    ("Confidence", "_build_confidence_threshold_subtab"),
    ("Meta", None),
)


def _format_probability_filter_cell(
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Compact run-list label, e.g. ``2%(0.70)`` / ``>6%(0.55)`` / ``—``."""
    m = metrics if isinstance(metrics, dict) else {}
    meta_d = meta if isinstance(meta, dict) else {}
    pf = m.get("probability_filter") if isinstance(m.get("probability_filter"), dict) else {}
    if not pf:
        pf = meta_d.get("probability_filter") if isinstance(meta_d.get("probability_filter"), dict) else {}

    active = bool(
        m.get("probability_filter_active")
        if m.get("probability_filter_active") is not None
        else pf.get("active")
    )
    label = (
        m.get("probability_filter_label")
        or meta_d.get("classification_filter_label")
        or pf.get("label")
        or ""
    )
    thr = (
        m.get("probability_filter_threshold")
        if m.get("probability_filter_threshold") is not None
        else meta_d.get("classification_filter_threshold")
    )
    if thr is None:
        thr = pf.get("threshold")

    text = str(label or "").strip()
    if text.lower() in ("", "disabled", "none", "off"):
        text = ""
    # "+2% Probability" / "+2%" → "2%"; keep ">6%" as-is.
    if text.lower().endswith(" probability"):
        text = text[: -len(" probability")].strip()
    if text.startswith("+"):
        text = text[1:]

    if not active or not text or thr is None:
        return "—"
    try:
        return f"{text}({float(thr):.2f})"
    except (TypeError, ValueError):
        return text or "—"


def _probability_filter_text(
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Probability part of the run-list Filter cell, or "" when inactive."""
    text = _format_probability_filter_cell(metrics, meta)
    return "" if text == "—" else text


def _tb_filter_text(
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Triple Barrier part of the run-list Filter cell, e.g. ``TB:TP(0.60)``.

    Reads the ``tb_filter``/``tb_filter_active``/``tb_filter_label`` fields
    ``_attach_pipeline_metrics`` (strategy_simulator/service.py) writes onto
    every persisted run's metrics. Old runs saved before the Triple Barrier
    filter existed simply have none of these keys, so this returns "".
    """
    m = metrics if isinstance(metrics, dict) else {}
    meta_d = meta if isinstance(meta, dict) else {}
    tb = m.get("tb_filter") if isinstance(m.get("tb_filter"), dict) else {}
    if not tb:
        tb = meta_d.get("tb_filter") if isinstance(meta_d.get("tb_filter"), dict) else {}

    active = bool(m.get("tb_filter_active") if m.get("tb_filter_active") is not None else tb.get("active"))
    if not active:
        return ""

    label = str(m.get("tb_filter_label") or tb.get("label") or "").strip()
    thr = m.get("tb_filter_threshold") if m.get("tb_filter_threshold") is not None else tb.get("threshold")
    if not label or label.lower() in ("", "disabled", "none", "off") or thr is None:
        return ""
    try:
        return f"TB:{label}({float(thr):.2f})"
    except (TypeError, ValueError):
        return f"TB:{label}"


def _confidence_filter_text(
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Confidence Classifier part of the run-list Filter cell, e.g. ``Conf:target_hit(1)``.

    Reads the ``classifier_filter``/``classifier_active``/``classifier_label``
    fields ``_attach_pipeline_metrics`` writes onto every persisted run's
    metrics, falling back to the raw ``meta`` payload for older runs.
    """
    m = metrics if isinstance(metrics, dict) else {}
    meta_d = meta if isinstance(meta, dict) else {}
    clf = m.get("classifier_filter") if isinstance(m.get("classifier_filter"), dict) else {}
    if not clf:
        clf = meta_d.get("classifier_filter") if isinstance(meta_d.get("classifier_filter"), dict) else {}

    active = bool(m.get("classifier_active") if m.get("classifier_active") is not None else clf.get("active"))
    if not active:
        return ""

    key = str(clf.get("model_key") or meta_d.get("confidence_classifier") or "").strip()
    label = str(m.get("classifier_label") or clf.get("label") or "").strip()
    ident = key or label
    if not ident or ident.lower() in ("", "disabled", "none", "off"):
        return ""

    keep_value = clf.get("keep_value")
    if keep_value is None:
        keep_value = meta_d.get("classifier_keep_value")
    if keep_value is None:
        return f"Conf:{ident}"
    try:
        return f"Conf:{ident}({int(keep_value)})"
    except (TypeError, ValueError):
        return f"Conf:{ident}"


def _format_active_filters_cell(
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Run-list Filter column: every active prediction filter for that run.

    Joins active filters with `` + `` in pipeline order — Confidence Classifier
    → Probability Ladder → Triple Barrier (see the pipeline docstring in
    ``strategy_simulator/service.py``):

        Conf:target_hit(1) + 2%(0.30) + TB:TP(0.60)

    Any filter that was not active for the run is simply omitted, so a
    probability-only run still renders exactly as
    ``_format_probability_filter_cell`` always has (e.g. ``2%(0.30)``), and a
    run with none of them active renders ``"—"``. Old runs persisted before
    the Triple Barrier / Confidence metrics existed have no ``tb_filter`` /
    ``classifier_filter`` data, so they degrade to the probability-only cell
    automatically — no migration needed.
    """
    parts = [
        p
        for p in (
            _confidence_filter_text(metrics, meta),
            _probability_filter_text(metrics, meta),
            _tb_filter_text(metrics, meta),
        )
        if p
    ]
    return " + ".join(parts) if parts else "—"


def _strategy_rule_rows(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Human-readable strategy rule rows for the Charges / Rules panel."""
    entry = cfg.get("entry") if isinstance(cfg.get("entry"), dict) else {}
    stop = cfg.get("stop") if isinstance(cfg.get("stop"), dict) else {}
    target = cfg.get("target") if isinstance(cfg.get("target"), dict) else {}
    hold = cfg.get("hold_time") if isinstance(cfg.get("hold_time"), dict) else {}
    conf = cfg.get("confidence") if isinstance(cfg.get("confidence"), dict) else {}
    pos = cfg.get("position_size") if isinstance(cfg.get("position_size"), dict) else {}
    exe = cfg.get("execution") if isinstance(cfg.get("execution"), dict) else {}
    opt = entry.get("option_types") or []
    opt_txt = ", ".join(str(x) for x in opt) if opt else "—"
    return [
        ("Direction", str(entry.get("direction") or "—")),
        ("Option types", opt_txt),
        ("Premium range", f"{entry.get('premium_min', '—')} – {entry.get('premium_max', '—')}"),
        ("ATM band", str(entry.get("atm_band") if entry.get("atm_band") is not None else "—")),
        ("Expiry", str(entry.get("expiry") or "—")),
        ("Entry cadence", f"{entry.get('entry_cadence_sec', '—')}s"),
        (
            "Min predicted move",
            (
                f"{entry.get('minimum_predicted_move_pct'):g}%"
                if entry.get("minimum_predicted_move_pct") not in (None, "", 0, 0.0)
                else "Off (direction only)"
            ),
        ),
        (
            "Regression",
            "On" if entry.get("use_regression", True) else "Off (without regression)",
        ),
        ("Stop loss", f"{stop.get('stop_loss_pct', '—')}%"),
        (
            "Target",
            (
                "Predicted LTP (entry row)"
                if target.get("use_predicted_ltp") and entry.get("use_regression", True)
                else f"{target.get('target_profit_pct', '—')}%"
            ),
        ),
        ("Max hold", f"{hold.get('max_hold_sec', '—')}s"),
        (
            "Model confidence",
            (
                f"≥ {conf.get('min_signal_strength', 0)}"
                if conf.get("use_model_confidence")
                else "Off"
            ),
        ),
        ("Lots × qty", f"{pos.get('lots', '—')} × {pos.get('qty_per_lot', '—')}"),
        ("Fees mode", str(exe.get("fees_mode") or "—")),
    ]


class ModelLabStrategySimPanel(ttk.Frame):
    """Top-level Research Lab tab: strategy + date range → P&L outcomes."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._model_name = ""
        self._lab_db_path = ""
        self._strat_id_map: dict[str, str] = {}
        self._strat_meta: dict[str, dict[str, Any]] = {}
        self._day_labels: list[str] = []
        self._current_run: dict[str, Any] | None = None
        self._current_trades: list[dict[str, Any]] = []
        self._status_var = tk.StringVar(value="")
        self._dataset_var = tk.StringVar(value="No Prediction Dataset loaded")
        self._loading_prefs = False
        self._saved_strategy_version_id = ""
        self._saved_strategy_id = ""
        self._prob_options: list[dict[str, Any]] = []
        self._prob_threshold: float | None = None
        self._prob_defaults: dict[str, Any] = {}
        self._saved_prob_member = ""
        self._saved_prob_threshold: float | None = None
        self._tb_classes: list[dict[str, Any]] = []
        self._tb_model_name: str | None = None
        self._tb_threshold_defaults: dict[str, Any] = {}
        self._saved_tb_class_label = ""
        self._saved_tb_enabled = False
        self._saved_tb_threshold: float | None = None
        self._last_tb_comparison: dict[str, Any] | None = None
        self._build_ui()
        self._load_ui_prefs()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def refresh_for_lab(self, *, lab_db_path: str | None, model_name: str) -> None:
        self._lab_db_path = str(lab_db_path or "").strip()
        self._model_name = str(model_name or "").strip()
        self._refresh_dataset_banner()
        self._load_strategies()
        self._load_date_range_options()
        self._refresh_classifier_list()
        self._refresh_package_filter_list()
        self._refresh_tb_filter_options()
        self._load_simulation_runs()
        self._refresh_classifier_summary()
        if self._model_name:
            self._status_var.set(
                f"Strategy Simulator · {self._model_name} — trading outcomes only "
                "(model accuracy lives in Research Dashboard)."
            )
        else:
            self._status_var.set("Open a Research Lab to simulate.")

    def refresh_for_model(self, model_name: str) -> None:
        """Backward-compatible refresh when lab path is already set."""
        self.refresh_for_lab(lab_db_path=self._lab_db_path or None, model_name=model_name)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(hdr, text="Strategy Simulator", font=SECTION_FONT).pack(side="left")
        ttk.Label(
            hdr,
            text="How profitable is this strategy on the model's predictions?",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(10, 0))

        outer = ttk.Panedwindow(self, orient=tk.VERTICAL)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        top = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        outer.add(top, weight=2)

        self._build_config(top)
        self._build_runs_list(top)

        detail = ttk.Notebook(outer)
        outer.add(detail, weight=3)
        self._detail_nb = detail
        self._build_overview_tab(detail)
        self._build_pipeline_tab(detail)
        self._build_equity_tab(detail)
        self._build_trades_tab(detail)
        self._build_daily_tab(detail)
        self._build_worst_open_risk_tab(detail)
        self._build_charges_tab(detail)
        self._build_classifier_threshold_tab(detail)

        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(
            anchor="w", padx=8, pady=(0, 4)
        )

        def _sash_top(_e: tk.Event | None = None) -> None:
            w = top.winfo_width()
            if w > 1:
                top.sashpos(0, int(w * 0.42))

        def _sash_outer(_e: tk.Event | None = None) -> None:
            h = outer.winfo_height()
            if h > 1:
                outer.sashpos(0, max(240, int(h * 0.48)))

        top.bind("<Configure>", _sash_top)
        outer.bind("<Configure>", _sash_outer)

    def _build_config(self, parent: ttk.Panedwindow) -> None:
        form = ttk.LabelFrame(parent, text="Configuration", padding=8)
        parent.add(form, weight=2)

        btns = ttk.Frame(form)
        btns.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(btns, text="Run Simulation", command=self._run_simulation).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btns, text="Refresh", command=self._refresh_clicked).pack(side="left")

        ttk.Label(
            form,
            text="Uses the current Prediction Dataset for this model. Strategy rules "
            "(lot size, slippage, confidence, etc.) are loaded from the Strategy Registry.",
            foreground=COL_MUTED,
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            form,
            textvariable=self._dataset_var,
            foreground=COL_MUTED,
            wraplength=380,
        ).pack(anchor="w", pady=(0, 8))

        conf_nb = ttk.Notebook(form)
        conf_nb.pack(fill="both", expand=True)
        strategy_tab = ttk.Frame(conf_nb, padding=4)
        exec_tab = ttk.Frame(conf_nb, padding=4)
        sim_tab = ttk.Frame(conf_nb, padding=4)
        conf_nb.add(strategy_tab, text="Strategy")
        conf_nb.add(exec_tab, text="Execution Rules")
        conf_nb.add(sim_tab, text="Simulation Settings")
        # Strategy tab scrolls so Classifier / Package / TB filters stay reachable.
        self._build_strategy_config_tab(self._scrollable_host(strategy_tab))
        self._build_execution_rules_tab(exec_tab)
        self._build_simulation_settings_tab(sim_tab)

    def _build_strategy_config_tab(self, form: ttk.Frame) -> None:
        strat_row = ttk.Frame(form)
        strat_row.pack(fill="x", pady=2)
        ttk.Label(strat_row, text="Strategy", width=14).pack(side="left")
        self._strat_var = tk.StringVar()
        self._strat_combo = ttk.Combobox(strat_row, textvariable=self._strat_var, state="readonly")
        self._strat_combo.pack(side="left", fill="x", expand=True, padx=(4, 2))
        self._strat_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_strategy_selected())
        ttk.Button(
            strat_row,
            text="↻",
            width=3,
            command=self._refresh_strategy_list,
        ).pack(side="left")
        ttk.Button(
            strat_row,
            text="Edit",
            width=5,
            command=self._edit_selected_strategy,
        ).pack(side="left", padx=(4, 0))

        day_row = ttk.Frame(form)
        day_row.pack(fill="x", pady=2)
        ttk.Label(day_row, text="Date Range", width=14).pack(side="left")
        self._day_var = tk.StringVar(value=_ALL_DAYS)
        self._day_combo = ttk.Combobox(day_row, textvariable=self._day_var, state="readonly")
        self._day_combo.pack(side="left", fill="x", expand=True, padx=4)
        self._day_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_date_range_selected())

        clf_row = ttk.Frame(form)
        clf_row.pack(fill="x", pady=2)
        ttk.Label(clf_row, text="Classifier Filter", width=14).pack(side="left")
        self._classifier_var = tk.StringVar(value=_CLASSIFIER_DISABLED)
        self._classifier_combo = ttk.Combobox(
            clf_row,
            textvariable=self._classifier_var,
            state="readonly",
            values=[_CLASSIFIER_DISABLED],
        )
        self._classifier_combo.pack(side="left", fill="x", expand=True, padx=(4, 2))
        ttk.Button(
            clf_row,
            text="↻",
            width=3,
            command=self._refresh_classifier_list,
        ).pack(side="left")
        self._classifier_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_classifier_summary()
        )

        reg_row = ttk.Frame(form)
        reg_row.pack(fill="x", pady=2)
        ttk.Label(reg_row, text="", width=14).pack(side="left")
        self._without_regression_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            reg_row,
            text="Without regression",
            variable=self._without_regression_var,
            command=self._save_ui_prefs,
        ).pack(side="left", padx=(4, 0))

        summary = ttk.LabelFrame(form, text="Classifier Summary", padding=6)
        summary.pack(fill="x", pady=(4, 2))
        self._clf_summary_vars = {
            "prediction_rows": tk.StringVar(value="—"),
            "rows_kept": tk.StringVar(value="—"),
            "rows_removed": tk.StringVar(value="—"),
            "trades_kept": tk.StringVar(value="—"),
            "trades_removed": tk.StringVar(value="—"),
        }
        for i, (label, key) in enumerate(
            (
                ("Prediction rows", "prediction_rows"),
                ("Rows kept", "rows_kept"),
                ("Rows removed", "rows_removed"),
                ("Executed trades kept", "trades_kept"),
                ("Executed trades removed", "trades_removed"),
            )
        ):
            ttk.Label(summary, text=f"{label}:", foreground=COL_MUTED).grid(
                row=i, column=0, sticky="w", padx=(0, 8), pady=1
            )
            ttk.Label(
                summary,
                textvariable=self._clf_summary_vars[key],
                font=("Consolas", 9),
            ).grid(row=i, column=1, sticky="w", pady=1)

        ttk.Label(
            form,
            text=(
                "Pipeline: Prediction → Classifier → Probability Filter → "
                "Triple Barrier Filter → Strategy Rules → Execution Rules → Trades."
            ),
            foreground=COL_MUTED,
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 4))

        self._build_package_filter_section(form)
        self._build_tb_filter_section(form)

    def _build_simulation_settings_tab(self, root: ttk.Frame) -> None:
        ttk.Label(root, text="Simulation Settings", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            root,
            text=(
                "Capital, sizing, charges, and slippage for this simulation run. "
                "Lot size / qty / charges / slippage are filled from the selected strategy."
            ),
            foreground=COL_MUTED,
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 8))

        self._capital_var = tk.StringVar(value="100000")
        self._lots_var = tk.StringVar(value="1")
        self._qty_var = tk.StringVar(value="65")
        self._commission_var = tk.StringVar(value="Enabled")
        self._slippage_var = tk.StringVar(value="0")

        for label, var, values in (
            ("Capital", self._capital_var, None),
            ("Lot Size", self._lots_var, None),
            ("Qty / Lot", self._qty_var, None),
            ("Charges", self._commission_var, ("Enabled", "Disabled")),
            ("Slippage (ticks)", self._slippage_var, None),
        ):
            row = ttk.Frame(root)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=14, foreground=COL_MUTED).pack(side="left")
            if values:
                ttk.Combobox(row, textvariable=var, values=list(values), state="readonly", width=14).pack(
                    side="left", padx=4
                )
            else:
                ttk.Entry(row, textvariable=var, width=16).pack(side="left", padx=4)
        ttk.Label(
            root,
            text="Charges Enabled = statutory only (zero-brokerage plan). Disabled = ₹0 fees.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
            wraplength=360,
        ).pack(anchor="w", pady=(4, 0))

    def _build_package_filter_section(self, form: ttk.Frame) -> None:
        """Prediction Package Filter — gates entry rows by ladder probability.

        Separate control from Classifier Filter above: that one selects a
        confidence label for strategy metadata, this one thins prediction rows
        before the strategy rules run.
        """
        pkg = ttk.LabelFrame(form, text="Prediction Package Filter", padding=6)
        pkg.pack(fill="x", pady=(4, 2))

        pkg_row = ttk.Frame(pkg)
        pkg_row.pack(fill="x", pady=2)
        ttk.Label(pkg_row, text="Probability", width=12).pack(side="left")
        self._prob_filter_var = tk.StringVar(value=_PROB_DISABLED)
        self._prob_filter_combo = ttk.Combobox(
            pkg_row,
            textvariable=self._prob_filter_var,
            state="readonly",
            values=[_PROB_DISABLED],
        )
        self._prob_filter_combo.pack(side="left", fill="x", expand=True, padx=(4, 2))
        ttk.Button(
            pkg_row,
            text="↻",
            width=3,
            command=self._refresh_package_filter_list,
        ).pack(side="left")
        self._prob_filter_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_package_filter_selected()
        )

        self._prob_threshold_var = tk.StringVar(value="—")
        self._prob_rows_var = tk.StringVar(value="—")
        for label, var in (
            ("Threshold", self._prob_threshold_var),
            ("Rows kept", self._prob_rows_var),
        ):
            row = ttk.Frame(pkg)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label}:", width=12, foreground=COL_MUTED).pack(side="left")
            ttk.Label(row, textvariable=var, font=("Consolas", 9)).pack(side="left", padx=(4, 0))

        self._prob_hint_var = tk.StringVar(
            value="Only trained members of the loaded Prediction Package are listed."
        )
        ttk.Label(
            pkg,
            textvariable=self._prob_hint_var,
            foreground=COL_MUTED,
            wraplength=360,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

    def _build_tb_filter_section(self, form: ttk.Frame) -> None:
        """Triple Barrier Filter — gates entry rows by the persisted TB side-scorer.

        Peer to the Prediction Package Filter above and the Classifier Filter
        higher up: an independent AND predicate, applied only in-memory during
        this simulation. Disabled by default. Reads only the already-persisted
        ``tb_pred_class`` / ``tb_pred_probability`` columns — never runs TB
        inference and never writes to the Prediction Dataset. Class radio
        buttons are labeled from the TB model's metadata / label registry
        (never hardcoded 0/1/2) — see ``resolve_tb_class_options``.
        """
        tb = ttk.LabelFrame(form, text="Triple Barrier Filter", padding=6)
        tb.pack(fill="x", pady=(4, 2))

        self._tb_enabled_var = tk.BooleanVar(value=False)
        self._tb_enabled_cb = ttk.Checkbutton(
            tb,
            text="Enable Triple Barrier Filter",
            variable=self._tb_enabled_var,
            command=self._on_tb_filter_toggled,
        )
        self._tb_enabled_cb.pack(anchor="w")

        self._tb_class_row = ttk.Frame(tb)
        self._tb_class_row.pack(fill="x", pady=(4, 2))
        ttk.Label(self._tb_class_row, text="Prediction", width=12).pack(side="left")
        self._tb_class_var = tk.StringVar(value="")
        self._tb_classes: list[dict[str, Any]] = []
        self._tb_class_radios: list[ttk.Radiobutton] = []
        self._tb_class_radio_host = ttk.Frame(self._tb_class_row)
        self._tb_class_radio_host.pack(side="left", fill="x", expand=True)

        thr_row = ttk.Frame(tb)
        thr_row.pack(fill="x", pady=2)
        ttk.Label(thr_row, text="Minimum Probability", width=18).pack(side="left")
        self._tb_threshold_var = tk.StringVar(value="0.60")
        self._tb_threshold_entry = ttk.Entry(thr_row, textvariable=self._tb_threshold_var, width=8)
        self._tb_threshold_entry.pack(side="left", padx=4)
        self._tb_threshold_var.trace_add("write", lambda *_a: self._on_tb_threshold_changed())

        self._tb_rows_var = tk.StringVar(value="—")
        rows_row = ttk.Frame(tb)
        rows_row.pack(fill="x", pady=1)
        ttk.Label(rows_row, text="Rows kept:", width=18, foreground=COL_MUTED).pack(side="left")
        ttk.Label(rows_row, textvariable=self._tb_rows_var, font=("Consolas", 9)).pack(
            side="left", padx=(4, 0)
        )

        self._tb_hint_var = tk.StringVar(
            value="Disabled by default. Enabling auto-runs a Baseline (TB off) vs Filtered (TB on) comparison."
        )
        ttk.Label(
            tb,
            textvariable=self._tb_hint_var,
            foreground=COL_MUTED,
            wraplength=360,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        self._sync_tb_filter_controls()

    def _sync_tb_filter_controls(self) -> None:
        on = bool(getattr(self, "_tb_enabled_var", None) and self._tb_enabled_var.get())
        has_classes = bool(getattr(self, "_tb_classes", None))
        state = "normal" if (on and has_classes) else "disabled"
        for rb in getattr(self, "_tb_class_radios", []):
            rb.configure(state=state)
        if hasattr(self, "_tb_threshold_entry"):
            self._tb_threshold_entry.configure(state="normal" if on else "disabled")

    def _on_tb_filter_toggled(self) -> None:
        self._sync_tb_filter_controls()
        self._refresh_tb_filter_summary()
        self._save_ui_prefs()

    def _on_tb_threshold_changed(self) -> None:
        self._refresh_tb_filter_summary()
        self._render_tb_threshold_table()
        self._save_ui_prefs()

    def _refresh_tb_filter_options(self) -> None:
        """Populate class radio buttons from TB model metadata / label registry."""
        from chain_replay_ml.strategy_simulator import tb_filter_options

        prev_label = self._tb_class_var.get()
        for rb in getattr(self, "_tb_class_radios", []):
            rb.destroy()
        self._tb_class_radios = []
        self._tb_classes = []
        self._tb_model_name = None
        self._tb_threshold_defaults = {}

        if not self._lab_db_path:
            self._tb_hint_var.set("Open a Research Lab to load Triple Barrier predictions.")
            self._sync_tb_filter_controls()
            self._refresh_tb_filter_summary()
            self._render_tb_threshold_table()
            self._sync_threshold_tab()
            return

        try:
            opts = tb_filter_options(self._data_dir(), self._lab_db_path)
        except Exception as exc:
            self._tb_hint_var.set(f"Triple Barrier options unavailable — {exc}")
            self._sync_tb_filter_controls()
            self._refresh_tb_filter_summary()
            self._render_tb_threshold_table()
            self._sync_threshold_tab()
            return

        self._tb_model_name = opts.get("tb_model_name")
        self._tb_classes = list(opts.get("classes") or [])
        if self._tb_model_name:
            from chain_replay_ml.strategy_simulator import resolve_member_threshold_defaults

            try:
                self._tb_threshold_defaults = resolve_member_threshold_defaults(
                    self._data_dir(), self._tb_model_name
                )
            except Exception:
                self._tb_threshold_defaults = {"model_name": self._tb_model_name, "rows": []}
        for cls in self._tb_classes:
            rb = ttk.Radiobutton(
                self._tb_class_radio_host,
                text=str(cls.get("label")),
                value=str(cls.get("label")),
                variable=self._tb_class_var,
                command=self._on_tb_class_selected,
            )
            rb.pack(side="left", padx=(0, 8))
            self._tb_class_radios.append(rb)

        labels = [str(c.get("label")) for c in self._tb_classes]
        if prev_label in labels:
            self._tb_class_var.set(prev_label)
        elif getattr(self, "_saved_tb_class_label", "") in labels:
            self._tb_class_var.set(self._saved_tb_class_label)
        elif labels:
            self._tb_class_var.set(labels[0])
        else:
            self._tb_class_var.set("")

        note = opts.get("note") or ""
        if opts.get("available"):
            source = str(opts.get("source") or "")
            source_txt = (
                " (default encoding — no Label Run metadata found for this model)"
                if source == "default_triple_barrier_encoding"
                else ""
            )
            self._tb_hint_var.set(
                f"Triple Barrier model: {self._tb_model_name}{source_txt}. "
                "Enabling auto-runs Baseline vs Filtered comparison."
                + (f" {note}" if note else "")
            )
        else:
            self._tb_hint_var.set(note or "No Triple Barrier predictions in this Prediction Dataset.")
            if bool(getattr(self, "_tb_enabled_var", None) and self._tb_enabled_var.get()):
                self._tb_enabled_var.set(False)

        self._sync_tb_filter_controls()
        self._refresh_tb_filter_summary()
        self._render_tb_threshold_table()
        self._sync_threshold_tab()

    def _on_tb_class_selected(self) -> None:
        self._refresh_tb_filter_summary()
        self._save_ui_prefs()

    def _selected_tb_class(self) -> dict[str, Any] | None:
        label = self._tb_class_var.get() if hasattr(self, "_tb_class_var") else ""
        for cls in getattr(self, "_tb_classes", []):
            if str(cls.get("label")) == label:
                return cls
        return None

    def _tb_threshold_value(self) -> float:
        from chain_replay_ml.strategy_simulator import normalize_tb_threshold

        try:
            return normalize_tb_threshold(self._tb_threshold_var.get())
        except Exception:
            return 0.60

    def _tb_filter_kwargs_from_ui(self) -> dict[str, Any]:
        enabled = bool(self._tb_enabled_var.get()) if hasattr(self, "_tb_enabled_var") else False
        cls = self._selected_tb_class()
        class_labels = {
            int(c["class_id"]): str(c["label"]) for c in getattr(self, "_tb_classes", [])
        }
        return {
            "tb_filter_enabled": bool(enabled and cls is not None),
            "tb_class_id": int(cls["class_id"]) if cls is not None else None,
            "tb_class_label": str(cls["label"]) if cls is not None else None,
            "tb_threshold": self._tb_threshold_value(),
            "tb_model_name": getattr(self, "_tb_model_name", None),
            "tb_class_labels": class_labels,
        }

    def _refresh_tb_filter_summary(self, metrics: dict[str, Any] | None = None) -> None:
        if not hasattr(self, "_tb_rows_var"):
            return

        summary = (metrics or {}).get("tb_summary") if metrics else None
        if isinstance(summary, dict) and summary.get("candidate_rows") is not None:
            kept = summary.get("rows_kept")
            total = summary.get("candidate_rows")
            if summary.get("active") and kept is not None and total:
                self._tb_rows_var.set(f"{kept:,} / {total:,} ({100.0 * kept / total:.1f}%)")
            elif summary.get("active"):
                self._tb_rows_var.set(f"{fmt_rows(kept)}" if kept is not None else "—")
            else:
                self._tb_rows_var.set("All rows (filter off)")
            return

        kwargs = self._tb_filter_kwargs_from_ui()
        if not kwargs["tb_filter_enabled"] or not self._lab_db_path:
            self._tb_rows_var.set("All rows (filter off)" if not kwargs["tb_filter_enabled"] else "—")
            return

        from chain_replay_ml.strategy_simulator import tb_row_summary

        try:
            st = tb_row_summary(
                self._lab_db_path,
                class_id=kwargs["tb_class_id"],
                threshold=kwargs["tb_threshold"],
                trading_days=self._selected_trading_days(),
            )
        except Exception as exc:
            self._tb_rows_var.set(f"error — {exc}")
            return
        if not st.get("ok"):
            self._tb_rows_var.set(str(st.get("error") or "unavailable"))
            return
        kept = st.get("rows_kept")
        total = st.get("prediction_rows")
        if kept is not None and total:
            self._tb_rows_var.set(f"{kept:,} / {total:,} ({100.0 * kept / total:.1f}%)")
        else:
            self._tb_rows_var.set("—")

    def _tb_filter_prefs(self) -> dict[str, Any]:
        kwargs = self._tb_filter_kwargs_from_ui()
        self._saved_tb_class_label = kwargs.get("tb_class_label") or ""
        return {
            "enabled": bool(self._tb_enabled_var.get()) if hasattr(self, "_tb_enabled_var") else False,
            "class_label": kwargs.get("tb_class_label"),
            "threshold": kwargs.get("tb_threshold"),
        }

    def _build_execution_rules_tab(self, root: ttk.Frame) -> None:
        ttk.Label(root, text="Execution Rules", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            root,
            text=(
                "Simulator-only constraints applied after Strategy entry signals. "
                "Does not affect Confidence Labels or model training."
            ),
            foreground=COL_MUTED,
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 8))

        self._exec_rules_enabled = tk.BooleanVar(value=False)
        self._exec_max_open_var = tk.StringVar(value="3")
        self._exec_one_per_symbol = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            root,
            text="Enable Execution Rules",
            variable=self._exec_rules_enabled,
            command=self._on_execution_rules_ui_changed,
        ).pack(anchor="w", pady=(0, 8))

        max_row = ttk.Frame(root)
        max_row.pack(fill="x", pady=2)
        ttk.Label(max_row, text="Maximum Open Positions", width=22).pack(side="left")
        self._exec_max_open_entry = ttk.Entry(
            max_row, textvariable=self._exec_max_open_var, width=8
        )
        self._exec_max_open_entry.pack(side="left", padx=4)

        self._exec_one_per_cb = ttk.Checkbutton(
            root,
            text="One Position Per Symbol",
            variable=self._exec_one_per_symbol,
            command=self._save_ui_prefs,
        )
        self._exec_one_per_cb.pack(anchor="w", pady=(8, 0))

        ttk.Label(
            root,
            text=(
                "When enabled: skip candidate signals that would exceed max open "
                "positions, or that target a symbol already held."
            ),
            foreground=COL_MUTED,
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(10, 0))
        self._exec_max_open_var.trace_add("write", lambda *_a: self._save_ui_prefs())
        self._sync_execution_rules_controls()

    def _on_execution_rules_ui_changed(self) -> None:
        self._sync_execution_rules_controls()
        self._save_ui_prefs()

    def _sync_execution_rules_controls(self) -> None:
        on = bool(self._exec_rules_enabled.get())
        state = "normal" if on else "disabled"
        if hasattr(self, "_exec_max_open_entry"):
            self._exec_max_open_entry.configure(state=state)
        if hasattr(self, "_exec_one_per_cb"):
            self._exec_one_per_cb.configure(state=state)

    def _execution_rules_from_ui(self) -> dict[str, Any]:
        try:
            max_open = int(str(self._exec_max_open_var.get() or "0").strip() or "0")
        except ValueError:
            max_open = 3
        return {
            "enabled": bool(self._exec_rules_enabled.get()),
            "max_open_positions": max(0, max_open),
            "one_position_per_symbol": bool(self._exec_one_per_symbol.get()),
        }

    def _load_ui_prefs(self) -> None:
        """Restore Execution Rules (+ strategy id applied when strategies load)."""
        from .strategy_sim_prefs import load_strategy_sim_prefs

        self._loading_prefs = True
        try:
            prefs = load_strategy_sim_prefs(self.chart_dir)
            rules = prefs.get("execution_rules") if isinstance(prefs.get("execution_rules"), dict) else {}
            if rules:
                if "enabled" in rules and hasattr(self, "_exec_rules_enabled"):
                    self._exec_rules_enabled.set(bool(rules.get("enabled")))
                if rules.get("max_open_positions") is not None and hasattr(self, "_exec_max_open_var"):
                    self._exec_max_open_var.set(str(int(rules.get("max_open_positions") or 0)))
                if "one_position_per_symbol" in rules and hasattr(self, "_exec_one_per_symbol"):
                    self._exec_one_per_symbol.set(bool(rules.get("one_position_per_symbol")))
                self._sync_execution_rules_controls()
            self._saved_strategy_version_id = str(prefs.get("strategy_version_id") or "")
            self._saved_strategy_id = str(prefs.get("strategy_id") or "")
            prob = prefs.get("probability_filter")
            if isinstance(prob, dict):
                self._saved_prob_member = str(prob.get("member_key") or "")
                try:
                    thr = prob.get("threshold")
                    self._saved_prob_threshold = float(thr) if thr is not None else None
                except (TypeError, ValueError):
                    self._saved_prob_threshold = None
            tb = prefs.get("triple_barrier_filter")
            if isinstance(tb, dict):
                self._saved_tb_enabled = bool(tb.get("enabled"))
                self._saved_tb_class_label = str(tb.get("class_label") or "")
                try:
                    thr = tb.get("threshold")
                    self._saved_tb_threshold = float(thr) if thr is not None else None
                except (TypeError, ValueError):
                    self._saved_tb_threshold = None
                if hasattr(self, "_tb_enabled_var"):
                    self._tb_enabled_var.set(self._saved_tb_enabled)
                if self._saved_tb_threshold is not None and hasattr(self, "_tb_threshold_var"):
                    self._tb_threshold_var.set(f"{self._saved_tb_threshold:.2f}")
            if "without_regression" in prefs and hasattr(self, "_without_regression_var"):
                self._without_regression_var.set(bool(prefs.get("without_regression")))
        except Exception:
            self._saved_strategy_version_id = ""
            self._saved_strategy_id = ""
        finally:
            self._loading_prefs = False

    def _save_ui_prefs(self) -> None:
        if getattr(self, "_loading_prefs", False):
            return
        from .strategy_sim_prefs import save_strategy_sim_prefs

        ctx = self._selected_strategy_context() if hasattr(self, "_strat_var") else {}
        try:
            save_strategy_sim_prefs(
                self.chart_dir,
                {
                    "strategy_version_id": str(
                        ctx.get("version_id")
                        or self._strat_id_map.get(self._strat_var.get())
                        or getattr(self, "_saved_strategy_version_id", "")
                        or ""
                    ),
                    "strategy_id": str(
                        ctx.get("strategy_id")
                        or getattr(self, "_saved_strategy_id", "")
                        or ""
                    ),
                    "execution_rules": self._execution_rules_from_ui(),
                    "probability_filter": self._probability_filter_prefs(),
                    "triple_barrier_filter": self._tb_filter_prefs(),
                    "without_regression": bool(
                        getattr(self, "_without_regression_var", None)
                        and self._without_regression_var.get()
                    ),
                },
            )
        except Exception:
            pass

    def _probability_filter_prefs(self) -> dict[str, Any]:
        if not self._prob_options:
            # No package loaded yet — keep the restored selection for the next lab.
            return {
                "member_key": self._saved_prob_member,
                "threshold": self._saved_prob_threshold,
            }
        option = self._selected_package_option() if hasattr(self, "_prob_filter_var") else None
        self._saved_prob_member = str((option or {}).get("key") or "")
        self._saved_prob_threshold = self._prob_threshold
        return {
            "member_key": self._saved_prob_member,
            "threshold": self._prob_threshold,
        }

    def _apply_saved_strategy_selection(self, labels: list[str]) -> str | None:
        """Pick combobox label from persisted version/strategy id."""
        saved_vid = str(getattr(self, "_saved_strategy_version_id", "") or "")
        saved_sid = str(getattr(self, "_saved_strategy_id", "") or "")
        if saved_vid:
            for label, vid in self._strat_id_map.items():
                if vid == saved_vid:
                    return label
        if saved_sid:
            for label, meta in self._strat_meta.items():
                if str(meta.get("strategy_id") or "") == saved_sid:
                    return label
        cur = self._strat_var.get()
        if cur in labels:
            return cur
        return labels[0] if labels else None

    def _build_runs_list(self, parent: ttk.Panedwindow) -> None:
        runs_fr = ttk.LabelFrame(parent, text="Strategy Runs", padding=4)
        parent.add(runs_fr, weight=3)
        cols = ("datetime", "strategy", "filter", "trades", "profit", "win", "pf", "dd")
        self._runs_tree = ttk.Treeview(runs_fr, columns=cols, show="headings", height=8)
        for c, w, label in (
            ("datetime", 130, "Datetime"),
            ("strategy", 100, "Strategy"),
            ("filter", 78, "Filter"),
            ("trades", 48, "Executed"),
            ("profit", 68, "Net P&L"),
            ("win", 48, "Win %"),
            ("pf", 42, "PF"),
            ("dd", 68, "Eq Max DD"),
        ):
            self._runs_tree.heading(c, text=label)
            self._runs_tree.column(c, width=w, stretch=True if c in ("strategy", "filter") else False)
        self._runs_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(runs_fr, orient="vertical", command=self._runs_tree.yview)
        sb.pack(side="right", fill="y")
        self._runs_tree.configure(yscrollcommand=sb.set)
        self._runs_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._runs_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._runs_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self._runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_run_detail())

    def _scrollable_host(self, parent: ttk.Frame) -> ttk.Frame:
        """Attach a vertical ScrollableFrame and return its inner content frame."""
        scroll = ScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        return scroll.inner

    def _build_overview_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Overview")
        inner = ttk.Notebook(tab)
        inner.pack(fill="both", expand=True)
        self._overview_nb = inner

        summary_tab = ttk.Frame(inner, padding=4)
        inner.add(summary_tab, text="Summary")
        self._overview_summary_host = self._scrollable_host(summary_tab)
        ttk.Label(
            self._overview_summary_host,
            text="Select a strategy run to see Strategy Evaluation scores.",
            foreground=COL_MUTED,
        ).pack(anchor="w")

        detail_tab = ttk.Frame(inner, padding=4)
        inner.add(detail_tab, text="Detail")
        self._overview_host = self._scrollable_host(detail_tab)
        ttk.Label(
            self._overview_host,
            text="Select a strategy run to see Net Profit, Gross Profit, and trade outcomes.",
            foreground=COL_MUTED,
        ).pack(anchor="w")

    def _build_pipeline_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Pipeline")
        self._pipeline_host = self._scrollable_host(tab)
        ttk.Label(
            self._pipeline_host,
            text="Simulation funnel: predictions → signals → executed trades.",
            foreground=COL_MUTED,
        ).pack(anchor="w")

    def _build_equity_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Equity / Drawdown")
        host = self._scrollable_host(tab)

        self._equity_summary_host = ttk.Frame(host)
        self._equity_summary_host.pack(fill="x", pady=(0, 4))

        self._equity_canvas = tk.Canvas(host, height=220, bg="#1a2a44", highlightthickness=0)
        self._equity_canvas.pack(fill="x", pady=(0, 6))
        self._equity_canvas.bind("<Configure>", lambda _e: self._redraw_equity_canvas())

        table_hdr = ttk.Frame(host)
        table_hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(
            table_hdr,
            text="Equity curve trades (executed)",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Button(
            table_hdr,
            text="Download CSV",
            command=self._download_equity_curve_csv,
        ).pack(side="right")

        tree_fr = ttk.Frame(host)
        tree_fr.pack(fill="x", expand=False, pady=(0, 8))
        cols = ("n", "day", "token", "entry", "exit", "pnl", "cum", "peak", "dd", "mark")
        self._equity_tree = ttk.Treeview(tree_fr, columns=cols, show="headings", height=12)
        for c, w, label in (
            ("n", 40, "#"),
            ("day", 90, "Day"),
            ("token", 70, "Token"),
            ("entry", 75, "Entry"),
            ("exit", 75, "Exit"),
            ("pnl", 70, "Trade P&L"),
            ("cum", 80, "Equity"),
            ("peak", 80, "Peak"),
            ("dd", 70, "Drawdown"),
            ("mark", 80, "Max DD"),
        ):
            self._equity_tree.heading(c, text=label)
            self._equity_tree.column(c, width=w, minwidth=40)
        ysb = ttk.Scrollbar(tree_fr, orient="vertical", command=self._equity_tree.yview)
        xsb = ttk.Scrollbar(tree_fr, orient="horizontal", command=self._equity_tree.xview)
        self._equity_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._equity_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_fr.rowconfigure(0, weight=1)
        tree_fr.columnconfigure(0, weight=1)
        self._equity_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._equity_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._equity_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self._equity_tree.tag_configure("dd_peak", background="#2d4a22")
        self._equity_tree.tag_configure("dd_trough", background="#5a2a2a")
        self._equity_curve_cache: list[dict[str, Any]] = []
        self._equity_episode_cache: dict[str, Any] = {}

    def _build_trades_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Trade List")
        host = self._scrollable_host(tab)
        table_hdr = ttk.Frame(host)
        table_hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(
            table_hdr,
            text="Executed trades for the selected run. Scroll the page or the table.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Button(
            table_hdr,
            text="Download CSV",
            command=self._download_trades_csv,
        ).pack(side="right")
        tree_fr = ttk.Frame(host)
        tree_fr.pack(fill="both", expand=True)
        cols = (
            "day",
            "token",
            "entry_ts",
            "exit_ts",
            "entry",
            "exit",
            "pnl",
            "ret",
            "hold",
            "reason",
            "exit_sample",
        )
        self._trades_tree = ttk.Treeview(tree_fr, columns=cols, show="headings", height=16)
        for c, w, label in (
            ("day", 88, "Day"),
            ("token", 70, "Token"),
            ("entry_ts", 145, "Entry Time"),
            ("exit_ts", 145, "Exit Time"),
            ("entry", 70, "Entry"),
            ("exit", 70, "Exit"),
            ("pnl", 70, "Net P&L"),
            ("ret", 55, "Ret %"),
            ("hold", 80, "Holding Time (s)"),
            ("reason", 90, "Exit Reason"),
            ("exit_sample", 100, "Exit Sample Index"),
        ):
            self._trades_tree.heading(c, text=label)
            self._trades_tree.column(c, width=w, minwidth=50)
        ysb = ttk.Scrollbar(tree_fr, orient="vertical", command=self._trades_tree.yview)
        xsb = ttk.Scrollbar(tree_fr, orient="horizontal", command=self._trades_tree.xview)
        self._trades_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._trades_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_fr.rowconfigure(0, weight=1)
        tree_fr.columnconfigure(0, weight=1)
        self._trades_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._trades_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._trades_tree.tag_configure("pnl_flat", foreground=COL_MUTED)

    def _build_daily_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Daily Performance")
        host = self._scrollable_host(tab)
        ttk.Label(
            host,
            text="Per-day outcomes for the selected run. Scroll the page or the table.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 4))
        tree_fr = ttk.Frame(host)
        tree_fr.pack(fill="both", expand=True)
        cols = ("day", "trades", "wins", "win_pct", "pnl")
        self._daily_tree = ttk.Treeview(tree_fr, columns=cols, show="headings", height=16)
        for c, w, label in (
            ("day", 100, "Day"),
            ("trades", 60, "Trades"),
            ("wins", 50, "Wins"),
            ("win_pct", 60, "Win %"),
            ("pnl", 90, "Net P&L"),
        ):
            self._daily_tree.heading(c, text=label)
            self._daily_tree.column(c, width=w, minwidth=50)
        ysb = ttk.Scrollbar(tree_fr, orient="vertical", command=self._daily_tree.yview)
        xsb = ttk.Scrollbar(tree_fr, orient="horizontal", command=self._daily_tree.xview)
        self._daily_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._daily_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_fr.rowconfigure(0, weight=1)
        tree_fr.columnconfigure(0, weight=1)
        self._daily_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._daily_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._daily_tree.tag_configure("pnl_flat", foreground=COL_MUTED)

    def _build_worst_open_risk_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Worst Open Risk")
        inner = ttk.Notebook(tab)
        inner.pack(fill="both", expand=True)
        self._worst_open_risk_nb = inner

        audit_tab = ttk.Frame(inner, padding=4)
        inner.add(audit_tab, text="Trade Audit")
        self._worst_open_risk_host = self._scrollable_host(audit_tab)
        ttk.Label(
            self._worst_open_risk_host,
            text="Select a strategy run to see the trade behind Max Portfolio DD (Open).",
            foreground=COL_MUTED,
        ).pack(anchor="w")

        replay_tab = ttk.Frame(inner, padding=4)
        inner.add(replay_tab, text="Stop Tick Replay")
        self._stop_tick_replay_host = self._scrollable_host(replay_tab)
        ttk.Label(
            self._stop_tick_replay_host,
            text="Select a strategy run to load stop-path timelines.",
            foreground=COL_MUTED,
        ).pack(anchor="w")

    def _build_charges_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=4)
        nb.add(tab, text="Charges Breakdown")
        self._charges_host = self._scrollable_host(tab)
        ttk.Label(
            self._charges_host,
            text="Select a strategy to see rules and description next to charges.",
            foreground=COL_MUTED,
        ).pack(anchor="w")

    def _build_classifier_threshold_tab(self, nb: ttk.Notebook) -> None:
        """Prediction Thresholds workspace — a notebook of threshold-analysis families.

        Registry-driven (see ``_THRESHOLD_SUBTAB_REGISTRY``) so new families (e.g.
        Meta) can be wired in later without redesigning this container. Only the
        Confidence sub-tab has a working sweep today — it's the pre-existing
        Classifier Threshold analysis, moved here verbatim. Container tab is
        added/removed from ``nb`` only while a package member is selected.
        """
        tab = ttk.Frame(nb, padding=4)
        self._threshold_tab = tab

        inner = ttk.Notebook(tab)
        inner.pack(fill="both", expand=True)
        self._threshold_subtabs_nb = inner

        for title, builder_name in _THRESHOLD_SUBTAB_REGISTRY:
            sub = ttk.Frame(inner, padding=6)
            inner.add(sub, text=title)
            if builder_name:
                getattr(self, builder_name)(sub)
            else:
                self._build_threshold_stub_subtab(sub, title)

    def _build_threshold_stub_subtab(self, host: ttk.Frame, title: str) -> None:
        """Placeholder for a Prediction Thresholds family with no analysis yet."""
        ttk.Label(host, text=f"{title} Thresholds", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            host,
            text="Not available yet — coming soon.",
            foreground=COL_MUTED,
            wraplength=760,
        ).pack(anchor="w", pady=(6, 0))

    def _build_confidence_threshold_subtab(self, tab: ttk.Frame) -> None:
        """Threshold sweep for the selected package member (existing behavior)."""
        ttk.Label(tab, text="Threshold Analysis", font=SECTION_FONT).pack(anchor="w")
        self._threshold_hint_var = tk.StringVar(value="")
        ttk.Label(
            tab,
            textvariable=self._threshold_hint_var,
            foreground=COL_MUTED,
            wraplength=760,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        cols = ("thr", "precision", "recall", "f1", "buy", "tpd", "composite", "selected")
        self._threshold_tree = ttk.Treeview(tab, columns=cols, show="headings", height=10)
        for col, width, label in (
            ("thr", 90, "Threshold"),
            ("precision", 90, "Precision"),
            ("recall", 80, "Recall"),
            ("f1", 80, "F1"),
            ("buy", 100, "BUY Signals"),
            ("tpd", 90, "Trades/Day"),
            ("composite", 100, "Composite"),
            ("selected", 80, "Selected"),
        ):
            self._threshold_tree.heading(col, text=label)
            self._threshold_tree.column(col, width=width, anchor="center")
        self._threshold_tree.pack(fill="both", expand=True, pady=(0, 4))
        self._threshold_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_threshold_picked())
        self._threshold_tree.tag_configure("chosen", foreground=COL_OK)

        ttk.Label(
            tab,
            text=(
                "Click a row to set the operating threshold for this simulation. "
                "● = selected, ★ = recommended. Rows below the threshold are removed "
                "before strategy rules run, so a higher threshold means fewer, "
                "higher-precision entries."
            ),
            foreground=COL_MUTED,
            wraplength=760,
            font=("Segoe UI", 8),
        ).pack(anchor="w")

    def _build_tb_threshold_subtab(self, tab: ttk.Frame) -> None:
        """Threshold sweep for the linked Triple Barrier side-scorer.

        Same pattern as ``_build_confidence_threshold_subtab``: reads the
        analysis persisted by walk-forward training (``member_threshold_analysis``
        / ``resolve_member_threshold_defaults``) — never recomputes, never runs
        inference. Selecting a row here sets the Triple Barrier Filter's
        "Minimum Probability" (``_tb_threshold_var``) — the same field the TB
        filter reads when running simulations.
        """
        ttk.Label(tab, text="Triple Barrier Threshold Analysis", font=SECTION_FONT).pack(
            anchor="w"
        )
        self._tb_threshold_hint_var = tk.StringVar(value="")
        ttk.Label(
            tab,
            textvariable=self._tb_threshold_hint_var,
            foreground=COL_MUTED,
            wraplength=760,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        cols = ("thr", "precision", "recall", "f1", "accuracy", "buy", "tpd", "composite", "selected")
        self._tb_threshold_tree = ttk.Treeview(tab, columns=cols, show="headings", height=10)
        for col, width, label in (
            ("thr", 90, "Threshold"),
            ("precision", 90, "Precision"),
            ("recall", 80, "Recall"),
            ("f1", 80, "F1"),
            ("accuracy", 90, "Accuracy"),
            ("buy", 100, "BUY Signals"),
            ("tpd", 90, "Trades/Day"),
            ("composite", 100, "Composite"),
            ("selected", 80, "Selected"),
        ):
            self._tb_threshold_tree.heading(col, text=label)
            self._tb_threshold_tree.column(col, width=width, anchor="center")
        self._tb_threshold_tree.pack(fill="both", expand=True, pady=(0, 4))
        self._tb_threshold_tree.bind(
            "<<TreeviewSelect>>", lambda _e: self._on_tb_threshold_row_picked()
        )
        self._tb_threshold_tree.tag_configure("chosen", foreground=COL_OK)

        ttk.Label(
            tab,
            text=(
                "Click a row to set the Triple Barrier Filter's Minimum Probability "
                "(above). ● = selected, ★ = recommended. This is the same operating "
                "threshold the Triple Barrier Filter uses when running simulations."
            ),
            foreground=COL_MUTED,
            wraplength=760,
            font=("Segoe UI", 8),
        ).pack(anchor="w")

    def _render_tb_threshold_table(self) -> None:
        """Populate the Triple Barrier threshold tree from persisted metrics.json.

        Empty states are surfaced as a clear hint (no rows) rather than a
        "coming soon" stub: no TB model linked, or no ``threshold_analysis``
        persisted for the linked model.
        """
        if not hasattr(self, "_tb_threshold_tree"):
            return
        self._tb_threshold_tree.delete(*self._tb_threshold_tree.get_children())
        model_name = getattr(self, "_tb_model_name", None)
        if not model_name:
            self._tb_threshold_hint_var.set(
                "No Triple Barrier model linked. Open a Research Lab whose Prediction "
                "Dataset was built with a Triple Barrier side-scorer selected."
            )
            return

        defaults = self._tb_threshold_defaults or {}
        rows = list(defaults.get("rows") or [])
        source = str(defaults.get("source") or "unavailable")
        if not rows:
            self._tb_threshold_hint_var.set(
                f"Triple Barrier model: {model_name}. No threshold analysis found — "
                "metrics.json is missing or has no threshold_analysis block. Train/retrain "
                "this model as binary via walk-forward to populate it."
            )
            return

        recommended = defaults.get("recommended_threshold")
        criterion = str(defaults.get("recommended_criterion") or "")
        self._tb_threshold_hint_var.set(
            f"Triple Barrier model: {model_name} · source: {source}"
            + (
                f" · recommended {float(recommended):.2f} ({criterion})"
                if recommended is not None
                else ""
            )
        )

        selected = self._tb_threshold_value()
        composites = defaults.get("composite_scores") or {}
        for row in rows:
            try:
                thr = round(float(row.get("threshold")), 2)
            except (TypeError, ValueError):
                continue
            buy = row.get("buy_signals")
            if buy is None:
                buy = row.get("predicted_positives")
            tpd = row.get("trades_per_day")
            if tpd is None:
                tpd_txt = "—"
            else:
                try:
                    tpd_f = float(tpd)
                    tpd_txt = (
                        f"{int(round(tpd_f)):,}"
                        if abs(tpd_f - round(tpd_f)) < 1e-9
                        else f"{tpd_f:,.1f}"
                    )
                except (TypeError, ValueError):
                    tpd_txt = "—"
            is_sel = abs(thr - float(selected)) < 1e-9
            score = composites.get(thr)
            is_rec = recommended is not None and abs(thr - float(recommended)) < 1e-9
            self._tb_threshold_tree.insert(
                "",
                "end",
                iid=f"{thr:.2f}",
                tags=("chosen",) if is_sel else (),
                values=(
                    f"{thr:.2f}" + (" ★" if is_rec else ""),
                    fmt_pct(row.get("precision_pct")),
                    fmt_pct(row.get("recall_pct")),
                    fmt_pct(row.get("f1_pct")),
                    fmt_pct(row.get("accuracy_pct")),
                    f"{int(buy):,}" if buy is not None else "—",
                    tpd_txt,
                    f"{float(score):.4f}" if score is not None else "—",
                    "●" if is_sel else "○",
                ),
            )
        iid = f"{float(selected):.2f}"
        if iid in self._tb_threshold_tree.get_children():
            self._tb_threshold_tree.selection_set(iid)

    def _on_tb_threshold_row_picked(self) -> None:
        """Clicking a threshold row sets the TB Filter's Minimum Probability."""
        sel = self._tb_threshold_tree.selection()
        if not sel:
            return
        try:
            thr = float(sel[0])
        except (TypeError, ValueError):
            return
        if abs(thr - self._tb_threshold_value()) < 1e-9:
            return
        self._tb_threshold_var.set(f"{thr:.2f}")

    # ── Loaders ───────────────────────────────────────────────────────────

    def _refresh_clicked(self) -> None:
        self.refresh_for_lab(lab_db_path=self._lab_db_path or None, model_name=self._model_name)

    def _refresh_dataset_banner(self) -> None:
        from chain_replay_ml.model_lab.prediction_builder import prediction_dataset_status

        if not self._lab_db_path:
            self._dataset_var.set("Prediction Dataset: (lab not open)")
            return
        try:
            st = prediction_dataset_status(self._lab_db_path, light=True)
        except Exception as exc:
            self._dataset_var.set(f"Prediction Dataset: error — {exc}")
            return
        n = int(st.get("row_count") or 0)
        days = st.get("trading_days")
        start = st.get("start_day") or ""
        end = st.get("end_day") or ""
        span = f"{start} → {end}" if start and end else ""
        day_txt = f"{days} days" if days is not None else "—"
        self._dataset_var.set(
            f"Prediction Dataset · {n:,} rows · {day_txt}"
            + (f" · {span}" if span else "")
        )

    def _load_date_range_options(self) -> None:
        labels = [_ALL_DAYS]
        self._day_labels = []
        if self._lab_db_path:
            try:
                from chain_replay_ml.model_lab.store import ModelLabStore

                with ModelLabStore(self._lab_db_path) as store:
                    lab = store.read_info()
                    for d in store.list_build_days(lab.lab_uuid):
                        day = str(d.get("trading_day") or "")
                        n = int(d.get("row_count") or 0)
                        if not day or n <= 0:
                            continue
                        labels.append(f"{day} ({n:,} rows)")
                        self._day_labels.append(day)
            except Exception:
                pass
        self._day_combo.configure(values=labels)
        cur = self._day_var.get()
        if cur not in labels:
            self._day_var.set(_ALL_DAYS)

    def _refresh_classifier_list(self) -> None:
        from chain_replay_ml.strategy_simulator.lab_source import classifier_filter_labels

        labels = classifier_filter_labels()
        prev = self._classifier_var.get()
        self._classifier_combo.configure(values=labels)
        if prev in labels:
            self._classifier_var.set(prev)
        else:
            self._classifier_var.set(_CLASSIFIER_DISABLED)
        self._refresh_classifier_summary()

    def _selected_classifier_key(self) -> str | None:
        from chain_replay_ml.strategy_simulator.lab_source import classifier_key_from_label

        key = classifier_key_from_label(self._classifier_var.get())
        return None if key == "disabled" else key

    def _refresh_classifier_summary(self, metrics: dict[str, Any] | None = None) -> None:
        """Update Classifier Summary from SQL preview and/or last run metrics."""
        if not hasattr(self, "_clf_summary_vars"):
            return

        summary = (metrics or {}).get("classifier_summary") if metrics else None
        if isinstance(summary, dict) and summary.get("prediction_rows") is not None:
            self._apply_classifier_summary(summary)
            return

        if not self._lab_db_path:
            self._apply_classifier_summary({})
            return

        from chain_replay_ml.strategy_simulator.lab_source import classifier_row_summary

        try:
            days = self._selected_trading_days()
            st = classifier_row_summary(
                self._lab_db_path,
                confidence_classifier=self._selected_classifier_key(),
                trading_days=days,
            )
        except Exception:
            st = {}
        # Row-level preview; trade counts filled after a simulation / run select.
        preview = {
            "prediction_rows": st.get("prediction_rows"),
            "rows_kept": st.get("rows_kept"),
            "rows_removed": st.get("rows_removed"),
            "rows_kept_pct": st.get("rows_kept_pct"),
            "rows_removed_pct": st.get("rows_removed_pct"),
            "executed_trades_kept": None,
            "executed_trades_removed": None,
            "active": st.get("active"),
        }
        self._apply_classifier_summary(preview)

    def _apply_classifier_summary(self, summary: dict[str, Any]) -> None:
        def _rows(n: Any, pct: Any = None) -> str:
            if n is None:
                return "—"
            try:
                text = f"{int(n):,}"
            except (TypeError, ValueError):
                return "—"
            if pct is not None:
                try:
                    text = f"{text} ({float(pct):.1f}%)"
                except (TypeError, ValueError):
                    pass
            return text

        def _trades(n: Any) -> str:
            if n is None:
                return "— (run simulation)"
            try:
                return f"{int(n):,}"
            except (TypeError, ValueError):
                return "—"

        self._clf_summary_vars["prediction_rows"].set(
            _rows(summary.get("prediction_rows"))
        )
        self._clf_summary_vars["rows_kept"].set(
            _rows(summary.get("rows_kept"), summary.get("rows_kept_pct"))
        )
        self._clf_summary_vars["rows_removed"].set(
            _rows(summary.get("rows_removed"), summary.get("rows_removed_pct"))
        )
        self._clf_summary_vars["trades_kept"].set(
            _trades(summary.get("executed_trades_kept"))
        )
        self._clf_summary_vars["trades_removed"].set(
            _trades(summary.get("executed_trades_removed"))
            if summary.get("active")
            else ("0" if summary.get("prediction_rows") is not None else "—")
        )

    def _on_date_range_selected(self) -> None:
        self._refresh_classifier_summary()
        self._refresh_package_filter_summary()
        self._refresh_tb_filter_summary()

    # ── Prediction Package Filter ─────────────────────────────────────────

    def _refresh_package_filter_list(self) -> None:
        """Populate the probability dropdown from the loaded Prediction Package."""
        from chain_replay_ml.strategy_simulator import (
            probability_filter_labels,
            probability_filter_options,
        )

        prev = self._prob_filter_var.get()
        options: list[dict[str, Any]] = []
        note = ""
        if self._lab_db_path:
            from chain_replay_ml.model_lab.prediction_builder import prediction_dataset_status

            try:
                st = prediction_dataset_status(self._lab_db_path, light=True)
                options = probability_filter_options(
                    self._data_dir(),
                    dataset=str(st.get("parent_dataset") or ""),
                    anchor_target=str(st.get("target_column") or ""),
                    anchor_model_name=str(st.get("parent_model_name") or self._model_name),
                )
            except Exception as exc:
                note = f"Package members unavailable — {exc}"
        else:
            note = "Open a Research Lab to load Prediction Package members."

        self._prob_options = options
        labels = probability_filter_labels(options)
        self._prob_filter_combo.configure(values=labels)
        target = prev if prev in labels else _PROB_DISABLED
        if target == _PROB_DISABLED and self._saved_prob_member:
            for option in options:
                if option.get("key") == self._saved_prob_member:
                    target = str(option["label"])
                    break
        self._prob_filter_var.set(target)

        if not note:
            note = (
                f"{len(options)} trained member(s) in this package."
                if options
                else "No probability members trained for this package yet — "
                "build them from the Classification ladder."
            )
        self._prob_hint_var.set(note)
        self._on_package_filter_selected(keep_threshold=True)

    def _selected_package_option(self) -> dict[str, Any] | None:
        from chain_replay_ml.strategy_simulator import option_from_label

        if not hasattr(self, "_prob_filter_var"):
            return None
        return option_from_label(self._prob_options, self._prob_filter_var.get())

    def _on_package_filter_selected(self, *, keep_threshold: bool = False) -> None:
        """Resolve the member's recommended threshold and sync tab + summary."""
        from chain_replay_ml.strategy_simulator import resolve_member_threshold_defaults

        option = self._selected_package_option()
        if option is None:
            self._prob_defaults = {}
            self._prob_threshold = None
            self._prob_threshold_var.set("—")
        else:
            model_name = str(option.get("model_name") or "")
            if str(self._prob_defaults.get("model_name") or "") != model_name:
                try:
                    self._prob_defaults = resolve_member_threshold_defaults(
                        self._data_dir(), model_name
                    )
                except Exception:
                    self._prob_defaults = {"model_name": model_name, "rows": []}
                keep_threshold = keep_threshold and self._saved_prob_threshold is not None
                self._prob_threshold = None
            if self._prob_threshold is None:
                if keep_threshold and self._saved_prob_threshold is not None:
                    self._prob_threshold = float(self._saved_prob_threshold)
                else:
                    self._prob_threshold = float(
                        self._prob_defaults.get("recommended_threshold") or 0.5
                    )
            self._prob_threshold_var.set(f"{self._prob_threshold:.2f}")

        self._sync_threshold_tab()
        self._render_threshold_table()
        self._refresh_package_filter_summary()
        self._save_ui_prefs()

    def _sync_threshold_tab(self) -> None:
        """Show Prediction Thresholds while a package member or TB model is linked."""
        if not hasattr(self, "_detail_nb") or not hasattr(self, "_threshold_tab"):
            return
        tab = str(self._threshold_tab)
        shown = tab in [str(t) for t in self._detail_nb.tabs()]
        want = self._selected_package_option() is not None or bool(
            getattr(self, "_tb_model_name", None)
        )
        if want and not shown:
            self._detail_nb.add(self._threshold_tab, text=_THRESHOLD_TAB_TEXT)
        elif not want and shown:
            self._detail_nb.forget(self._threshold_tab)

    def _render_threshold_table(self) -> None:
        if not hasattr(self, "_threshold_tree"):
            return
        self._threshold_tree.delete(*self._threshold_tree.get_children())
        option = self._selected_package_option()
        if option is None:
            self._threshold_hint_var.set(
                "Select a Prediction Package Filter member to choose its operating threshold."
            )
            return

        rows = list(self._prob_defaults.get("rows") or [])
        recommended = self._prob_defaults.get("recommended_threshold")
        criterion = str(self._prob_defaults.get("recommended_criterion") or "")
        source = str(self._prob_defaults.get("source") or "unavailable")
        self._threshold_hint_var.set(
            f"{option.get('ladder_label')} · {option.get('model_name')} · "
            f"column {option.get('column')} · source: {source}"
            + (
                f" · recommended {float(recommended):.2f} ({criterion})"
                if recommended is not None
                else ""
            )
        )
        if not rows:
            self._threshold_tree.insert(
                "",
                "end",
                iid="0.50",
                values=("0.50", "—", "—", "—", "—", "—", "—", "●"),
                tags=("chosen",),
            )
            return

        selected = self._prob_threshold
        composites = self._prob_defaults.get("composite_scores") or {}
        for row in rows:
            try:
                thr = round(float(row.get("threshold")), 2)
            except (TypeError, ValueError):
                continue
            buy = row.get("buy_signals")
            if buy is None:
                buy = row.get("predicted_positives")
            tpd = row.get("trades_per_day")
            if tpd is None:
                tpd_txt = "—"
            else:
                try:
                    tpd_f = float(tpd)
                    tpd_txt = (
                        f"{int(round(tpd_f)):,}"
                        if abs(tpd_f - round(tpd_f)) < 1e-9
                        else f"{tpd_f:,.1f}"
                    )
                except (TypeError, ValueError):
                    tpd_txt = "—"
            is_sel = selected is not None and abs(thr - float(selected)) < 1e-9
            score = composites.get(thr)
            is_rec = recommended is not None and abs(thr - float(recommended)) < 1e-9
            self._threshold_tree.insert(
                "",
                "end",
                iid=f"{thr:.2f}",
                tags=("chosen",) if is_sel else (),
                values=(
                    f"{thr:.2f}" + (" ★" if is_rec else ""),
                    fmt_pct(row.get("precision_pct")),
                    fmt_pct(row.get("recall_pct")),
                    fmt_pct(row.get("f1_pct")),
                    f"{int(buy):,}" if buy is not None else "—",
                    tpd_txt,
                    f"{float(score):.4f}" if score is not None else "—",
                    "●" if is_sel else "○",
                ),
            )
        if selected is not None:
            iid = f"{float(selected):.2f}"
            if iid in self._threshold_tree.get_children():
                self._threshold_tree.selection_set(iid)

    def _on_threshold_picked(self) -> None:
        sel = self._threshold_tree.selection()
        if not sel:
            return
        try:
            thr = float(sel[0])
        except (TypeError, ValueError):
            return
        if self._prob_threshold is not None and abs(thr - self._prob_threshold) < 1e-9:
            return
        self._prob_threshold = thr
        self._prob_threshold_var.set(f"{thr:.2f}")
        self._render_threshold_table()
        self._refresh_package_filter_summary()
        self._save_ui_prefs()

    def _refresh_package_filter_summary(self, metrics: dict[str, Any] | None = None) -> None:
        """Row counts for the probability filter — SQL preview or last run."""
        if not hasattr(self, "_prob_rows_var"):
            return

        summary = (metrics or {}).get("probability_summary") if metrics else None
        if isinstance(summary, dict) and summary.get("prediction_rows") is not None:
            self._apply_package_filter_summary(summary)
            return

        option = self._selected_package_option()
        if option is None or not self._lab_db_path:
            self._prob_rows_var.set("—" if option is not None else "All rows (filter off)")
            return

        from chain_replay_ml.strategy_simulator import probability_row_summary

        try:
            st = probability_row_summary(
                self._lab_db_path,
                column=str(option.get("column") or ""),
                threshold=self._prob_threshold,
                trading_days=self._selected_trading_days(),
            )
        except Exception as exc:
            self._prob_rows_var.set(f"error — {exc}")
            return
        if not st.get("ok"):
            self._prob_rows_var.set(str(st.get("error") or "unavailable"))
            return
        self._apply_package_filter_summary(st)

    def _apply_package_filter_summary(self, summary: dict[str, Any]) -> None:
        kept = summary.get("rows_kept")
        total = summary.get("prediction_rows")
        if kept is None or total is None:
            self._prob_rows_var.set("—")
            return
        try:
            pct = summary.get("rows_kept_pct")
            text = f"{int(kept):,} / {int(total):,}"
            if pct is not None:
                text = f"{text} ({float(pct):.1f}%)"
        except (TypeError, ValueError):
            text = "—"
        self._prob_rows_var.set(text)

    def _refresh_strategy_list(self) -> None:
        """Reload Strategy Registry champions into the Strategy dropdown."""
        prev = self._strat_var.get()
        self._load_strategies()
        n = len(self._strat_id_map)
        if n == 0:
            self._status_var.set("No strategies found in Strategy Registry.")
            return
        if prev and prev in self._strat_id_map:
            self._strat_var.set(prev)
            self._on_strategy_selected()
        self._status_var.set(f"Strategy list refreshed — {n} champion version(s).")

    def _load_strategies(self) -> None:
        from chain_replay_ml.strategy_registry import get_strategy_detail, list_strategies

        self._strat_id_map.clear()
        self._strat_meta.clear()
        labels: list[str] = []
        try:
            strategies = list_strategies(self._data_dir())
        except Exception as exc:
            self._status_var.set(f"Strategies: {exc}")
            return
        for s in strategies:
            sid = str(s.get("strategy_id") or "")
            try:
                detail = get_strategy_detail(self._data_dir(), sid)
            except Exception:
                continue
            champ = (detail or {}).get("champion_version") or {}
            vid = str(champ.get("version_id") or "")
            if not vid:
                continue
            label = f"{s.get('display_name')} {s.get('current_version_label')} ({vid[:8]}…)"
            labels.append(label)
            self._strat_id_map[label] = vid
            cfg = champ.get("config") if isinstance(champ.get("config"), dict) else {}
            self._strat_meta[label] = {
                "strategy_id": sid,
                "version_id": vid,
                "version_label": champ.get("version_label") or s.get("current_version_label"),
                "display_name": s.get("display_name") or champ.get("display_name") or cfg.get("name"),
                "description": (
                    champ.get("description")
                    or cfg.get("description")
                    or s.get("description")
                    or ""
                ),
                "config": cfg,
            }
        self._strat_combo.configure(values=labels)
        if labels:
            self._loading_prefs = True
            try:
                chosen = self._apply_saved_strategy_selection(labels)
                if chosen:
                    self._strat_var.set(chosen)
                elif self._strat_var.get() not in labels:
                    self._strat_var.set(labels[0])
            finally:
                self._loading_prefs = False
            self._on_strategy_selected()
        else:
            self._strat_var.set("")

    def _on_strategy_selected(self) -> None:
        champ = self._strat_meta.get(self._strat_var.get()) or {}
        cfg = champ.get("config") if isinstance(champ.get("config"), dict) else {}
        pos = cfg.get("position_size") if isinstance(cfg.get("position_size"), dict) else {}
        exe = cfg.get("execution") if isinstance(cfg.get("execution"), dict) else {}
        if pos.get("lots") is not None:
            self._lots_var.set(str(pos.get("lots")))
        if pos.get("qty_per_lot") is not None:
            self._qty_var.set(str(pos.get("qty_per_lot")))
        fees = str(exe.get("fees_mode") or "rupee_charges")
        self._commission_var.set("Disabled" if fees == "zero" else "Enabled")
        if exe.get("slippage_ticks") is not None:
            self._slippage_var.set(str(exe.get("slippage_ticks")))
        # Keep Charges Breakdown strategy panel in sync with the dropdown.
        self._render_charges(
            getattr(self, "_current_trades", None) or [],
            (getattr(self, "_current_run", None) or {}).get("metrics") or {},
        )
        self._save_ui_prefs()

    def _selected_strategy_context(self) -> dict[str, Any]:
        """Return the currently selected Strategy Registry champion context."""
        label = str(self._strat_var.get() or "")
        meta = dict(self._strat_meta.get(label) or {})
        if not meta and label:
            vid = self._strat_id_map.get(label)
            if vid:
                meta = {"version_id": vid}
        return meta

    def _edit_selected_strategy(self) -> None:
        """Open an editor dialog for the selected strategy (saves a new champion version)."""
        ctx = self._selected_strategy_context()
        sid = str(ctx.get("strategy_id") or "")
        vid = str(ctx.get("version_id") or self._strat_id_map.get(self._strat_var.get()) or "")
        if not sid or not vid:
            messagebox.showinfo(
                "Edit Strategy",
                "Select a strategy first.",
                parent=self,
            )
            return

        cfg = ctx.get("config") if isinstance(ctx.get("config"), dict) else {}
        if not cfg:
            try:
                from chain_replay_ml.strategy_registry import get_strategy_version

                version = get_strategy_version(self._data_dir(), vid) or {}
                cfg = version.get("config") if isinstance(version.get("config"), dict) else {}
                if not ctx.get("description"):
                    ctx["description"] = version.get("description") or ""
                if not ctx.get("display_name"):
                    ctx["display_name"] = version.get("display_name") or ""
            except Exception as exc:
                messagebox.showerror("Edit Strategy", str(exc), parent=self)
                return

        win = tk.Toplevel(self)
        win.title(f"Edit Strategy — {ctx.get('display_name') or 'Strategy'}")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.geometry("720x560")

        hdr = ttk.Frame(win, padding=8)
        hdr.pack(fill="x")
        ttk.Label(
            hdr,
            text=f"{ctx.get('display_name') or 'Strategy'}  ·  {ctx.get('version_label') or vid[:8]}",
            font=SECTION_FONT,
        ).pack(anchor="w")
        ttk.Label(
            hdr,
            text="Edit description / JSON, then Save as new version (immutable + new champion).",
            foreground=COL_MUTED,
            wraplength=680,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 0))

        form = ttk.Frame(win, padding=(8, 0))
        form.pack(fill="x")
        ttk.Label(form, text="Description", width=12).pack(side="left")
        desc_var = tk.StringVar(value=str(ctx.get("description") or cfg.get("description") or ""))
        desc_entry = ttk.Entry(form, textvariable=desc_var)
        desc_entry.pack(side="left", fill="x", expand=True, padx=4)

        editor = scrolledtext.ScrolledText(win, height=22, font=("Consolas", 9))
        editor.pack(fill="both", expand=True, padx=8, pady=8)
        editor.insert("1.0", json.dumps(cfg, indent=2, default=str))

        btns = ttk.Frame(win, padding=8)
        btns.pack(fill="x")

        def _read_cfg() -> dict[str, Any]:
            raw = editor.get("1.0", "end").strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Config must be a JSON object.")
            parsed["description"] = str(desc_var.get() or "").strip()
            if ctx.get("display_name") and not parsed.get("name"):
                parsed["name"] = ctx["display_name"]
            return parsed

        def _validate() -> None:
            from chain_replay_ml.strategy_registry.schema import validate_strategy_config

            try:
                parsed = _read_cfg()
            except json.JSONDecodeError as exc:
                messagebox.showerror("Validate", f"Invalid JSON:\n{exc}", parent=win)
                return
            except ValueError as exc:
                messagebox.showerror("Validate", str(exc), parent=win)
                return
            errors = validate_strategy_config(parsed)
            if errors:
                messagebox.showerror("Validate", "\n".join(errors), parent=win)
                return
            messagebox.showinfo("Validate", "Config is valid.", parent=win)

        def _save() -> None:
            from chain_replay_ml.strategy_registry import create_strategy_version
            from chain_replay_ml.strategy_registry.schema import validate_strategy_config

            try:
                parsed = _read_cfg()
            except json.JSONDecodeError as exc:
                messagebox.showerror("Save", f"Invalid JSON:\n{exc}", parent=win)
                return
            except ValueError as exc:
                messagebox.showerror("Save", str(exc), parent=win)
                return
            errors = validate_strategy_config(parsed)
            if errors:
                messagebox.showerror("Save", "\n".join(errors), parent=win)
                return
            try:
                version = create_strategy_version(
                    self._data_dir(),
                    strategy_id=sid,
                    config=parsed,
                    lifecycle="edit",
                    parent_version_id=vid,
                    set_champion=True,
                )
            except Exception as exc:
                messagebox.showerror("Save", str(exc), parent=win)
                return
            label = version.get("version_label") or "?"
            same = str(version.get("version_id") or "") == str(vid)
            win.destroy()
            self._refresh_strategy_list()
            # Select the new champion in the dropdown when possible.
            new_vid = str(version.get("version_id") or "")
            for combo_label, mapped in self._strat_id_map.items():
                if mapped == new_vid:
                    self._strat_var.set(combo_label)
                    self._on_strategy_selected()
                    break
            if same:
                messagebox.showinfo(
                    "Edit Strategy",
                    f"No changes — still {label} (same config hash).",
                    parent=self,
                )
            else:
                messagebox.showinfo(
                    "Edit Strategy",
                    f"Saved {label} and set as champion.",
                    parent=self,
                )

        ttk.Button(btns, text="Validate", command=_validate).pack(side="left")
        ttk.Button(btns, text="Save as new version", command=_save).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")

    def _selected_trading_days(self) -> list[str] | None:
        label = str(self._day_var.get() or _ALL_DAYS)
        if label == _ALL_DAYS:
            return None
        day = label.split(" ", 1)[0].strip()
        return [day] if day else None

    def _config_overrides_from_ui(self) -> dict[str, Any]:
        try:
            lots = int(float(self._lots_var.get()))
        except (TypeError, ValueError):
            lots = 1
        try:
            qty = int(float(self._qty_var.get()))
        except (TypeError, ValueError):
            qty = 65
        try:
            slip = int(float(self._slippage_var.get()))
        except (TypeError, ValueError):
            slip = 0
        fees_mode = "zero" if self._commission_var.get() == "Disabled" else "rupee_charges"
        use_regression = not bool(
            getattr(self, "_without_regression_var", None)
            and self._without_regression_var.get()
        )
        return {
            "position_size": {"lots": lots, "qty_per_lot": qty},
            "execution": {"fees_mode": fees_mode, "slippage_ticks": slip},
            "entry": {"use_regression": use_regression},
        }

    def _parse_capital(self) -> float | None:
        try:
            return float(str(self._capital_var.get()).replace(",", "").replace("₹", "").strip())
        except (TypeError, ValueError):
            return None

    def _load_simulation_runs(self) -> None:
        from chain_replay_ml.strategy_simulator import list_strategy_runs

        self._runs_tree.delete(*self._runs_tree.get_children())
        if not self._model_name:
            return
        try:
            all_runs = list_strategy_runs(self._data_dir(), limit=200)
        except Exception as exc:
            self._status_var.set(f"Runs: {exc}")
            return
        matched = [r for r in all_runs if str(r.get("model_id") or "") == self._model_name]
        matched.sort(key=lambda r: str(r.get("created_on") or ""), reverse=True)
        for r in matched:
            sid = str(r.get("strategy_run_id") or "")
            m = r.get("metrics") or {}
            meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            _, net = _resolve_gross_net_profit(m)
            profit = net if net is not None else m.get("profit")
            created = str(r.get("created_on") or "")
            if "T" in created:
                created = created.replace("T", " ")[:19]
            else:
                created = created[:19] if created else "—"
            self._runs_tree.insert(
                "",
                "end",
                iid=sid,
                tags=(_pnl_tag(profit),),
                values=(
                    created,
                    (r.get("strategy_id") or "")[:16],
                    _format_active_filters_cell(m, meta),
                    r.get("trade_count"),
                    _fmt_signed_pnl(profit),
                    _fmt_num(m.get("win_rate_pct")),
                    _fmt_num(m.get("profit_factor"), digits=2),
                    fmt_rupee(m.get("account_equity_max_drawdown", m.get("max_drawdown"))),
                ),
            )
        if matched:
            self._status_var.set(f"{len(matched)} strategy run(s) for {self._model_name}")

    # ── Run ───────────────────────────────────────────────────────────────

    def _run_simulation(self) -> None:
        from chain_replay_ml.strategy_simulator import (
            run_strategy_simulation_from_lab_with_tb_comparison,
        )

        version_id = self._strat_id_map.get(self._strat_var.get())
        if not self._lab_db_path:
            messagebox.showinfo(
                "Strategy Simulator",
                "Open a Research Lab with a Prediction Dataset first.",
                parent=self,
            )
            return
        if not version_id:
            messagebox.showinfo("Strategy Simulator", "Select a strategy.", parent=self)
            return
        prob_option = self._selected_package_option() or {}
        tb_kwargs = self._tb_filter_kwargs_from_ui()
        try:
            result = run_strategy_simulation_from_lab_with_tb_comparison(
                self._data_dir(),
                lab_db_path=self._lab_db_path,
                strategy_version_id=version_id,
                trading_days=self._selected_trading_days(),
                config_overrides=self._config_overrides_from_ui(),
                capital=self._parse_capital(),
                confidence_classifier=self._selected_classifier_key(),
                classifier_keep_value=1,
                probability_filter_column=prob_option.get("column"),
                probability_filter_threshold=self._prob_threshold,
                probability_filter_label=prob_option.get("ladder_label"),
                probability_filter_member=prob_option.get("key"),
                execution_rules=self._execution_rules_from_ui(),
                **tb_kwargs,
            )
            self._last_tb_comparison = result.get("comparison")
            self._load_simulation_runs()
            run_payload = result.get("run") or {}
            run_id = str(run_payload.get("strategy_run_id") or "")
            if run_id and run_id in self._runs_tree.get_children():
                self._runs_tree.selection_set(run_id)
                self._runs_tree.see(run_id)
            self._load_run_detail()
            m = run_payload.get("metrics") or {}
            self._refresh_classifier_summary(m)
            self._refresh_package_filter_summary(m)
            self._refresh_tb_filter_summary(m)
            _, net = _resolve_gross_net_profit(m)
            grade = m.get("strategy_grade")
            if grade is None:
                blob = m.get("strategy_score_v1") if isinstance(m.get("strategy_score_v1"), dict) else {}
                grade = blob.get("grade")
            quality = m.get("strategy_score")
            if quality is None:
                blob = m.get("strategy_score_v1") if isinstance(m.get("strategy_score_v1"), dict) else {}
                quality = blob.get("strategy_score")
            evidence = m.get("evidence_score")
            if evidence is None:
                blob = m.get("strategy_score_v1") if isinstance(m.get("strategy_score_v1"), dict) else {}
                evidence = blob.get("evidence_score")
            status = (
                f"Simulation done · {run_payload.get('trade_count') or 0} trades · "
                f"Net P&L {fmt_rupee(net if net is not None else m.get('profit'))} · "
                f"Win {_fmt_num(m.get('win_rate_pct'))}%"
            )
            if grade is not None and quality is not None:
                status += (
                    f" · Grade {grade} · Quality {_fmt_num(quality, digits=1)} · "
                    f"Evidence {_fmt_num(evidence, digits=1)}"
                )
            if result.get("mode") == "comparison":
                status += " · Baseline vs Filtered comparison ready (Pipeline tab)."
            self._status_var.set(status)
        except Exception as exc:
            messagebox.showerror("Strategy Simulator", str(exc), parent=self)

    def _load_run_detail(self) -> None:
        from chain_replay_ml.strategy_simulator import get_strategy_run_trades

        sel = self._runs_tree.selection()
        if not sel:
            return
        run_id = sel[0]
        try:
            doc = get_strategy_run_trades(self._data_dir(), run_id, limit=5000)
        except Exception as exc:
            self._render_overview({})
            self._render_overview_summary({})
            self._render_pipeline({})
            self._render_worst_open_risk({})
            self._render_charges([])
            self._status_var.set(str(exc))
            return
        if not doc.get("ok"):
            return
        self._current_run = doc.get("run") or {}
        self._current_trades = list(doc.get("trades") or [])
        metrics = self._current_run.get("metrics") or {}
        cmp_payload = getattr(self, "_last_tb_comparison", None)
        if cmp_payload is not None:
            filtered_id = str((cmp_payload.get("filtered") or {}).get("strategy_run_id") or "")
            if filtered_id and filtered_id != str(run_id):
                # Selecting a different run invalidates the stale side-by-side.
                self._last_tb_comparison = None
        self._render_overview(metrics)
        self._render_overview_summary(metrics, self._current_trades)
        self._render_pipeline(metrics)
        self._refresh_classifier_summary(metrics)
        self._refresh_package_filter_summary(metrics)
        self._refresh_tb_filter_summary(metrics)
        self._render_trades(self._current_trades)
        self._render_daily(self._current_trades)
        self._render_equity(self._current_trades, metrics)
        self._render_worst_open_risk(metrics)
        self._render_charges(self._current_trades, metrics)

    def _pipeline_counts(self, metrics: dict[str, Any]) -> dict[str, Any]:
        dataset_n = metrics.get("dataset_row_count")
        date_n = metrics.get("date_filtered_count")
        if date_n is None:
            date_n = metrics.get("predictions_loaded")
        evaluated = metrics.get("predictions_evaluated")
        if evaluated is None:
            evaluated = metrics.get("rows_simulated")
        signals = metrics.get("signals_generated")
        if signals is None:
            signals = metrics.get("signals")
        skipped = metrics.get("signals_skipped")
        if skipped is None:
            skipped = metrics.get("skipped_signals")
        executed = metrics.get("executed_trades")
        if executed is None:
            executed = metrics.get("trade_count")
        wins = metrics.get("winning_trades")
        if wins is None:
            wins = metrics.get("wins")
        losses = metrics.get("losing_trades")
        if losses is None:
            losses = metrics.get("losses")
        ignored = metrics.get("ignored_predictions")
        if ignored is None and evaluated is not None and executed is not None:
            try:
                ignored = max(0, int(evaluated) - int(executed))
            except (TypeError, ValueError):
                ignored = None
        clf_label = metrics.get("classifier_label") or "Disabled"
        clf_kept = metrics.get("classifier_kept")
        if clf_kept is None:
            clf_kept = evaluated if not metrics.get("classifier_active") else None
        clf_removed = metrics.get("classifier_removed")
        if clf_removed is None:
            clf_removed = 0
        prob = metrics.get("probability_summary") or {}
        prob_active = bool(metrics.get("probability_filter_active") or prob.get("active"))
        prob_thr = metrics.get("probability_filter_threshold")
        if prob_thr is None:
            prob_thr = prob.get("threshold")
        ss = metrics.get("simulator_summary") or {}
        candidate = metrics.get("candidate_signals")
        if candidate is None:
            candidate = ss.get("candidate_signals")
        skipped_max = metrics.get("skipped_max_positions")
        if skipped_max is None:
            skipped_max = ss.get("skipped_max_positions")
        skipped_sym = metrics.get("skipped_same_symbol")
        if skipped_sym is None:
            skipped_sym = ss.get("skipped_same_symbol")
        return {
            "dataset_n": dataset_n,
            "date_n": date_n if date_n is not None else evaluated,
            "classifier_label": clf_label,
            "classifier_kept": clf_kept,
            "classifier_removed": clf_removed,
            "probability_active": prob_active,
            "probability_label": metrics.get("probability_filter_label") or "Disabled",
            "probability_column": metrics.get("probability_filter_column") or prob.get("column"),
            "probability_threshold": prob_thr,
            "probability_kept": metrics.get("probability_kept", prob.get("rows_kept")),
            "probability_removed": metrics.get("probability_removed", prob.get("rows_removed")),
            "evaluated": evaluated,
            "signals": signals,
            "candidate_signals": candidate,
            "skipped_max_positions": skipped_max,
            "skipped_same_symbol": skipped_sym,
            "skipped": skipped,
            "executed": executed,
            "wins": wins,
            "losses": losses,
            "ignored": ignored,
        }

    def _render_overview_summary(
        self,
        metrics: dict[str, Any],
        trades: list[dict[str, Any]] | None = None,
    ) -> None:
        host = getattr(self, "_overview_summary_host", None)
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        if not metrics:
            ttk.Label(
                host,
                text="Select a strategy run to see Strategy Evaluation scores.",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            return

        from chain_replay_ml.strategy_simulator.scoring import evaluate_strategy_from_run

        score = metrics.get("strategy_score_v1")
        if not isinstance(score, dict) or not score:
            score = evaluate_strategy_from_run(
                metrics,
                trades if trades is not None else getattr(self, "_current_trades", None),
            )

        grade = str(score.get("grade") or "—")
        quality = score.get("strategy_score")
        evidence = score.get("evidence_score")
        reliability = score.get("sample_reliability") or "—"
        comps = score.get("component_scores") or {}
        raw = score.get("raw_metrics") or {}
        sample = score.get("sample_telemetry") or {}

        section_title(host, "STRATEGY EVALUATION")
        ttk.Label(
            host,
            text="Strategy Quality is post-cost edge quality. Evidence is sample confidence, not ML probability.",
            foreground=COL_MUTED,
            wraplength=720,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        dual_spec_sections(
            host,
            [
                (
                    "Strategy Quality",
                    f"{_fmt_num(quality, digits=1)} / 100" if quality is not None else "—",
                ),
                (
                    "Evidence Score",
                    f"{_fmt_num(evidence, digits=1)}%" if evidence is not None else "—",
                ),
            ],
            [
                ("Grade", grade),
                ("Sample Reliability", str(reliability)),
            ],
            left_title="Quality",
            right_title="Grade / Evidence",
            label_width=18,
        )

        section_title(host, "COMPONENT SCORES (0–100)")
        inline_spec_rows(
            host,
            [
                (
                    "Net Profit Factor",
                    f"[ {_fmt_num(comps.get('profit_factor'), digits=1)} / 100 ]",
                ),
                (
                    "RoMaD",
                    f"[ {_fmt_num(comps.get('romad'), digits=1)} / 100 ]",
                ),
                (
                    "Expectancy / Risk",
                    f"[ {_fmt_num(comps.get('expectancy'), digits=1)} / 100 ]",
                ),
                (
                    "Win Rate",
                    f"[ {_fmt_num(comps.get('win_rate'), digits=1)} / 100 ]",
                ),
            ],
            label_width=20,
        )

        section_title(host, "RAW METRICS")
        net_pf_raw = raw.get("net_profit_factor")
        dual_spec_sections(
            host,
            [
                (
                    "Net PF",
                    _fmt_num(net_pf_raw, digits=3) if net_pf_raw is not None else "— (no losers)",
                ),
                (
                    "Win Rate",
                    f"{_fmt_num(raw.get('win_rate'), digits=1)}%"
                    if raw.get("win_rate") is not None
                    else "—",
                ),
            ],
            [
                ("RoMaD", _fmt_num(raw.get("romad"), digits=3)),
                ("Exp/Risk", _fmt_num(raw.get("expectancy_ratio"), digits=4)),
                ("Net Profit", fmt_rupee(raw.get("net_profit")), raw.get("net_profit")),
            ],
            left_title="Edge",
            right_title="Risk / P&L",
            label_width=14,
        )

        section_title(host, "SAMPLE TELEMETRY")
        inline_spec_rows(
            host,
            [
                (
                    "Executed Trades",
                    f"{fmt_rows(sample.get('executed_trades'))} / "
                    f"{fmt_rows(sample.get('target_sample_size'))} target",
                ),
                (
                    "Active Trading Days",
                    f"{fmt_rows(sample.get('active_trading_days'))} / "
                    f"{fmt_rows(sample.get('target_trading_days'))} target",
                ),
                (
                    "Scoring Version",
                    str(score.get("scoring_version") or "v1.0.0"),
                ),
            ],
            label_width=20,
        )

    def _render_overview(self, metrics: dict[str, Any]) -> None:
        for w in self._overview_host.winfo_children():
            w.destroy()
        if not metrics:
            ttk.Label(
                self._overview_host,
                text="Select a strategy run to see Net Profit, Gross Profit, and trade outcomes.",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            return

        counts = self._pipeline_counts(metrics)
        executed = counts["executed"]
        section_title(self._overview_host, "Trading Outcomes")
        ttk.Label(
            self._overview_host,
            text=(
                "P&L, Win Rate, Profit Factor, and Max Drawdown from executed trades "
                "only. Skipped signals (Execution Rules) are excluded."
            ),
            foreground=COL_MUTED,
            wraplength=720,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))
        left, right = _summary_metric_sections(metrics)
        cleaned_left: list[tuple] = []
        for item in left:
            if str(item[0]) == "Trades":
                cleaned_left.append(("Executed Trades", fmt_rows(executed), executed))
            else:
                cleaned_left.append(item)

        dbg = metrics.get("metrics_debug") or metrics.get("simulator_summary") or {}
        debug_rows: list[tuple] = []
        if dbg:
            eq_pts = dbg.get("equity_curve_points")
            if eq_pts is None:
                eq_pts = metrics.get("equity_curve_points")
            match = dbg.get("equity_matches_executed")
            if match is None and eq_pts is not None and executed is not None:
                try:
                    match = int(eq_pts) == int(executed)
                except (TypeError, ValueError):
                    match = None
            debug_rows = [
                ("Candidate signals", fmt_rows(dbg.get("candidate_signals"))),
                ("Executed trades", fmt_rows(dbg.get("executed_trades", executed))),
                ("Skipped (Max Position)", fmt_rows(dbg.get("skipped_max_positions"))),
                ("Skipped (Same Symbol)", fmt_rows(dbg.get("skipped_same_symbol"))),
                ("Equity curve points", fmt_rows(eq_pts)),
                (
                    "Equity == executed",
                    "Yes" if match else ("No" if match is False else "—"),
                ),
                (
                    "Stop Loss / Trade",
                    fmt_rupee(metrics.get("stop_loss_per_trade_rupees")),
                ),
                (
                    "Max Open × Stop",
                    (
                        f"{metrics.get('max_open_positions_for_risk')} × "
                        f"{fmt_rupee(metrics.get('stop_loss_per_trade_rupees'))}"
                    ),
                ),
            ]

        dual_spec_sections(
            self._overview_host,
            cleaned_left,
            right,
            label_width=22,
            extra_rows=debug_rows or None,
            extra_title="Metrics Debug",
            extra_label_width=20,
        )

        audit = metrics.get("outcome_audit")
        if not isinstance(audit, dict) or not audit:
            # Older runs: compute on the fly from loaded trades when available.
            trades = getattr(self, "_current_trades", None) or []
            if trades:
                from chain_replay_ml.strategy_simulator.metrics import compute_outcome_audit

                audit = compute_outcome_audit(list(trades))
        if isinstance(audit, dict) and audit:
            section_title(self._overview_host, "Outcome Audit (target / stop / fees)")
            ttk.Label(
                self._overview_host,
                text=str(audit.get("asymmetry_note") or ""),
                foreground=COL_MUTED,
                wraplength=960,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(0, 6))
            reasons = audit.get("exit_reason_counts") or {}
            audit_left = [
                ("Avg winning trade (net ₹)", fmt_rupee(audit.get("avg_winning_trade_net"))),
                ("Avg losing trade (net ₹)", fmt_rupee(audit.get("avg_losing_trade_net"))),
                (
                    "Gross Profit (before fees)",
                    fmt_rupee(audit.get("gross_profit_before_fees")),
                ),
                (
                    "Gross Loss (before fees)",
                    fmt_rupee(audit.get("gross_loss_before_fees")),
                ),
                (
                    "PF before fees",
                    _fmt_num(audit.get("profit_factor_before_fees"), digits=2),
                ),
                (
                    "PF after fees (UI)",
                    _fmt_num(audit.get("profit_factor_after_fees"), digits=2),
                ),
            ]
            audit_right = [
                ("PF formula", str(audit.get("profit_factor_formula") or "—")),
                ("Net Profit formula", str(audit.get("net_profit_formula") or "—")),
                ("Total fees", fmt_rupee(audit.get("total_fees"))),
                ("Net Profit", fmt_rupee(audit.get("net_profit"))),
                ("Exit: Target", fmt_rows(reasons.get("target"))),
                ("Exit: Stop", fmt_rows(reasons.get("stop"))),
                ("Exit: Hold Time", fmt_rows(reasons.get("max_hold"))),
                ("Exit: End of Path", fmt_rows(reasons.get("end_of_path"))),
            ]
            dual_spec_sections(
                self._overview_host,
                audit_left,
                audit_right,
                left_title="Win / Loss",
                right_title="Formulas / Exits",
                label_width=26,
            )
            fill_left = [
                ("Target exact @ 3%", fmt_rows(audit.get("target_exact"))),
                ("Target above 3%", fmt_rows(audit.get("target_above"))),
                ("Target below 3%", fmt_rows(audit.get("target_below"))),
                (
                    "Target mean return %",
                    _fmt_num(audit.get("target_mean_return_pct"), digits=2),
                ),
            ]
            fill_right = [
                ("Stop exact @ stop ₹", fmt_rows(audit.get("stop_exact"))),
                ("Stop beyond (gap fill)", fmt_rows(audit.get("stop_beyond_gap"))),
                (
                    "Stop trigger sample beyond",
                    fmt_rows(audit.get("stop_trigger_beyond_sample")),
                ),
                (
                    "Stop mean return %",
                    _fmt_num(audit.get("stop_mean_return_pct"), digits=2),
                ),
            ]
            dual_spec_sections(
                self._overview_host,
                fill_left,
                fill_right,
                left_title="Target fills",
                right_title="Stop fills",
                label_width=26,
            )

        self._render_overview_daily_pnl_table()

    def _daily_pnl_rows(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_day: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            day = str(t.get("trading_day") or "—")
            try:
                pnl = float(t.get("net_pnl") or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            by_day[day].append(pnl)
        rows: list[dict[str, Any]] = []
        for day in sorted(by_day.keys()):
            pnls = by_day[day]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            n = len(pnls)
            total = sum(pnls)
            win_pct = (100.0 * wins / n) if n else None
            rows.append({
                "day": day,
                "trades": n,
                "wins": wins,
                "losses": losses,
                "win_pct": win_pct,
                "net_pnl": total,
            })
        return rows

    def _render_overview_daily_pnl_table(self) -> None:
        """Bottom-of-Overview day P&L table when Date Range is All Days."""
        day_label = str(getattr(self, "_day_var", tk.StringVar()).get() or _ALL_DAYS)
        trades = list(getattr(self, "_current_trades", None) or [])
        if day_label != _ALL_DAYS or not trades:
            return
        rows = self._daily_pnl_rows(trades)
        if len(rows) < 1:
            return

        section_title(self._overview_host, "Daily Profit & Loss (All Days)")
        ttk.Label(
            self._overview_host,
            text="Net P&L by trading day for this run. Scroll the Overview panel to see the full table.",
            foreground=COL_MUTED,
            wraplength=960,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 4))

        tree_fr = ttk.Frame(self._overview_host)
        tree_fr.pack(fill="x", expand=False, pady=(0, 8))
        cols = ("day", "trades", "wins", "losses", "win_pct", "pnl")
        tree = ttk.Treeview(tree_fr, columns=cols, show="headings", height=min(14, max(4, len(rows) + 1)))
        for c, w, label in (
            ("day", 110, "Day"),
            ("trades", 70, "Trades"),
            ("wins", 60, "Wins"),
            ("losses", 60, "Losses"),
            ("win_pct", 70, "Win %"),
            ("pnl", 110, "Net P&L"),
        ):
            tree.heading(c, text=label)
            tree.column(c, width=w, minwidth=50, anchor="w")
        ysb = ttk.Scrollbar(tree_fr, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(tree_fr, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_fr.columnconfigure(0, weight=1)
        tree.tag_configure("pnl_pos", foreground=COL_OK)
        tree.tag_configure("pnl_neg", foreground=COL_WARN)
        tree.tag_configure("pnl_flat", foreground=COL_MUTED)

        for row in rows:
            total = row["net_pnl"]
            win_pct = row["win_pct"]
            tree.insert(
                "",
                "end",
                tags=(_pnl_tag(total),),
                values=(
                    row["day"],
                    row["trades"],
                    row["wins"],
                    row["losses"],
                    _fmt_num(win_pct, digits=1) if win_pct is not None else "—",
                    _fmt_signed_pnl(total),
                ),
            )
        # Totals row
        all_pnl = sum(float(r["net_pnl"]) for r in rows)
        all_n = sum(int(r["trades"]) for r in rows)
        all_wins = sum(int(r["wins"]) for r in rows)
        all_losses = sum(int(r["losses"]) for r in rows)
        all_wr = (100.0 * all_wins / all_n) if all_n else None
        tree.insert(
            "",
            "end",
            tags=(_pnl_tag(all_pnl),),
            values=(
                "TOTAL",
                all_n,
                all_wins,
                all_losses,
                _fmt_num(all_wr, digits=1) if all_wr is not None else "—",
                _fmt_signed_pnl(all_pnl),
            ),
        )

    def _render_worst_open_risk(self, metrics: dict[str, Any]) -> None:
        host = getattr(self, "_worst_open_risk_host", None)
        replay_host = getattr(self, "_stop_tick_replay_host", None)
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        if replay_host is not None:
            for w in replay_host.winfo_children():
                w.destroy()

        if not metrics:
            ttk.Label(
                host,
                text="Select a strategy run to see the trade behind Max Portfolio DD (Open).",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            if replay_host is not None:
                ttk.Label(
                    replay_host,
                    text="Select a strategy run to load stop-path timelines.",
                    foreground=COL_MUTED,
                ).pack(anchor="w")
            return

        wt = metrics.get("worst_open_risk_trade") or (
            (metrics.get("portfolio_risk") or {}).get("worst_open_risk_trade")
        )
        section_title(host, "Worst Portfolio DD Trade (Open Risk)")
        ttk.Label(
            host,
            text=(
                "Trade that produced Max Portfolio DD (Open). "
                "Simulator executes on Prediction Dataset samples (~3s) only — "
                "see Stop Tick Replay for raw ticks (reference) vs simulator decisions."
            ),
            foreground=COL_MUTED,
            wraplength=900,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        row = ttk.Frame(host)
        row.pack(fill="both", expand=True)

        summary = ttk.LabelFrame(row, text="Portfolio risk summary", padding=8)
        summary.pack(side="left", fill="both", expand=True, padx=(0, 6))
        inline_spec_rows(
            summary,
            [
                (
                    "Max Portfolio DD (Open)",
                    fmt_rupee(metrics.get("max_portfolio_drawdown_open_risk")),
                ),
                (
                    "Max Theoretical Risk",
                    fmt_rupee(metrics.get("max_theoretical_portfolio_risk")),
                ),
                (
                    "Stop Loss / Trade",
                    fmt_rupee(metrics.get("stop_loss_per_trade_rupees")),
                ),
                (
                    "Max Open Positions",
                    fmt_rows(metrics.get("max_open_positions_for_risk")),
                ),
                (
                    "Observed max concurrent",
                    fmt_rows(metrics.get("observed_max_concurrent_open")),
                ),
            ],
            label_width=26,
        )

        detail = ttk.LabelFrame(row, text="Trade detail", padding=8)
        detail.pack(side="left", fill="both", expand=True, padx=(6, 0))

        if not isinstance(wt, dict) or not wt:
            ttk.Label(
                detail,
                text="No worst-trade audit on this run — re-run the simulation to refresh.",
                foreground=COL_MUTED,
                wraplength=420,
            ).pack(anchor="w")
            if replay_host is not None:
                ttk.Label(
                    replay_host,
                    text="No worst-trade audit on this run — re-run the simulation to refresh.",
                    foreground=COL_MUTED,
                ).pack(anchor="w")
            return

        note = metrics.get("open_risk_incorrect_metric_note") or (
            (metrics.get("portfolio_risk") or {}).get("which_metric_was_wrong")
        )
        if note:
            ttk.Label(
                detail,
                text=str(note),
                foreground=COL_WARN,
                wraplength=420,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(0, 6))

        replay: dict[str, Any] = {}
        try:
            from chain_replay_ml.strategy_simulator.stop_replay import build_stop_replay

            replay = build_stop_replay(
                chart_dir=self.chart_dir,
                trade=wt,
                lab_db_path=self._lab_db_path or None,
                pre_entry_sec=30.0,
            )
        except Exception:
            replay = {}

        left_rows = [
            ("Trade ID", str(wt.get("trade_id") or "—")),
            ("Token", str(wt.get("token") or "—")),
            ("Trading day", str(wt.get("trading_day") or "—")),
            ("Entry time", self._fmt_exit_ts(wt.get("entry_ts"))),
            ("Exit time", self._fmt_exit_ts(wt.get("exit_ts"))),
            ("Entry Price", _fmt_num(wt.get("entry_price"), digits=4)),
            ("Quantity", fmt_rows(wt.get("quantity", wt.get("qty")))),
            ("Position value", fmt_rupee(wt.get("position_value"))),
        ]
        exceeds = (
            "Yes (gap)"
            if wt.get("exceeds_stop_due_to_gap")
            else (
                "Yes — capped"
                if wt.get("stop_cap_applied") or wt.get("exceeds_configured_stop")
                else "No"
            )
        )
        if replay.get("raw_crossed_before_sim_exit"):
            exceeds = "Yes — sampled exit (3s model)"

        right_rows = [
            (
                "Execution Type",
                str(replay.get("execution_type") or "Prediction Sample (~3s)"),
            ),
            ("Stop Loss %", _fmt_num(wt.get("stop_loss_pct"), digits=2)),
            ("Stop Price", _fmt_num(wt.get("stop_price"), digits=4)),
            (
                "Raw Tick Stop Cross (Ref)",
                (
                    f"{replay.get('raw_tick_stop_cross_label') or '—'}  "
                    f"₹{_fmt_num(replay.get('raw_tick_stop_cross_ltp'), digits=4)}"
                    if replay.get("raw_tick_stop_cross_ts") is not None
                    else "—"
                ),
            ),
            (
                "Simulator Exit Sample",
                (
                    f"{replay.get('simulator_exit_sample_label') or self._fmt_exit_ts(wt.get('exit_ts'))}  "
                    f"₹{_fmt_num(replay.get('simulator_exit_sample_ltp', wt.get('exit_price')), digits=4)}"
                ),
            ),
            (
                "Stop Trigger Sample LTP",
                _fmt_num(
                    wt.get("stop_trigger_ltp", wt.get("sample_exit_ltp")),
                    digits=4,
                ),
            ),
            (
                "Lowest live LTP",
                _fmt_num(
                    wt.get("lowest_live_ltp_before_exit", wt.get("lowest_price_reached")),
                    digits=4,
                ),
            ),
            ("Exit Price (fill)", _fmt_num(wt.get("exit_price"), digits=4)),
            ("Expected Stop Loss ₹", fmt_rupee(wt.get("expected_stop_loss_rupees"))),
            (
                "Actual Max Floating ₹",
                fmt_rupee(wt.get("actual_maximum_floating_loss_rupees")),
            ),
            ("Exit Reason", str(wt.get("exit_reason") or "—")),
            ("Exceeds stop?", exceeds),
        ]
        dual_spec_sections(
            detail,
            left_rows,
            right_rows,
            left_title="Entry",
            right_title="Stop / Exit",
            label_width=24,
        )

        if replay.get("raw_crossed_before_sim_exit"):
            ttk.Label(
                host,
                text=(
                    "Portfolio DD note: Raw tick crossed stop before simulator exit. "
                    "Actual execution occurred at the next Prediction Dataset sample "
                    "(~3s execution model) — expected behavior, not a bug."
                ),
                foreground=COL_MUTED,
                wraplength=960,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(8, 0))

        self._render_stop_tick_replay(wt, replay=replay)

    def _render_stop_tick_replay(
        self,
        wt: dict[str, Any],
        *,
        replay: dict[str, Any] | None = None,
    ) -> None:
        """Separate raw-tick (reference) and prediction-sample (executable) timelines."""
        from chain_replay_ml.strategy_simulator.stop_replay import (
            EXECUTION_MODEL_DETAIL,
            EXECUTION_MODEL_LABEL,
            build_stop_replay,
        )

        host = getattr(self, "_stop_tick_replay_host", None)
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()

        section_title(host, "Stop path timelines (entry −30s → exit)")

        if replay is None:
            try:
                replay = build_stop_replay(
                    chart_dir=self.chart_dir,
                    trade=wt,
                    lab_db_path=self._lab_db_path or None,
                    pre_entry_sec=30.0,
                )
            except Exception as exc:
                ttk.Label(
                    host,
                    text=f"Stop replay failed: {exc}",
                    foreground=COL_WARN,
                    wraplength=900,
                ).pack(anchor="w")
                return

        replay = replay or {}
        path_kind = str(replay.get("path_kind") or replay.get("diagnosis") or "no_data")
        # Expected 3s model outcomes stay neutral/green; only real bugs warn.
        diag_color = COL_WARN if path_kind == "stop_bug" else COL_OK
        if path_kind in ("no_ticks_in_window", "no_tick_db", "no_data"):
            diag_color = COL_MUTED

        ttk.Label(
            host,
            text=f"Execution Model: {replay.get('execution_model_label') or EXECUTION_MODEL_LABEL}",
            foreground=COL_OK,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            host,
            text=str(replay.get("execution_model_detail") or EXECUTION_MODEL_DETAIL),
            foreground=COL_MUTED,
            wraplength=960,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 4))

        ttk.Label(
            host,
            text=f"Path note: {replay.get('diagnosis_label') or path_kind}",
            foreground=diag_color,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            host,
            text=str(replay.get("diagnosis_detail") or replay.get("error") or ""),
            foreground=COL_MUTED,
            wraplength=960,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 6))

        meta_bits = [
            f"raw ticks={replay.get('tick_count') or 0}",
            f"sim samples={replay.get('sample_count') or 0}",
            f"ticks between stop→exit={replay.get('ticks_between_stop_and_exit') or 0}",
        ]
        if replay.get("median_sample_interval_sec") is not None:
            meta_bits.append(f"median sample Δt={replay['median_sample_interval_sec']}s")
        if replay.get("raw_tick_stop_cross_ltp") is not None:
            meta_bits.append(
                f"raw stop cross ₹{_fmt_num(replay.get('raw_tick_stop_cross_ltp'), digits=4)}"
            )
        if replay.get("simulator_exit_sample_ltp") is not None:
            meta_bits.append(
                f"sim exit ₹{_fmt_num(replay.get('simulator_exit_sample_ltp'), digits=4)}"
            )
        ttk.Label(
            host,
            text=" · ".join(meta_bits),
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        if not replay.get("ok") and not (replay.get("raw_tick_rows") or replay.get("sim_sample_rows")):
            return

        panes = ttk.Frame(host)
        panes.pack(fill="both", expand=True)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)

        self._fill_timeline_tree(
            panes,
            title="Raw Exchange Tick Timeline (reference only — not executable)",
            rows=list(replay.get("raw_tick_rows") or []),
            mode="raw",
            grid_col=0,
        )
        self._fill_timeline_tree(
            panes,
            title="Prediction Sample Timeline (simulator — executable)",
            rows=list(replay.get("sim_sample_rows") or []),
            mode="sim",
            grid_col=1,
        )

    def _fill_timeline_tree(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        rows: list[dict[str, Any]],
        mode: str,
        grid_col: int,
    ) -> None:
        box = ttk.LabelFrame(parent, text=title, padding=6)
        box.grid(row=0, column=grid_col, sticky="nsew", padx=(0 if grid_col == 0 else 4, 0 if grid_col == 1 else 4))
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        if mode == "raw":
            cols = ("ts", "rel", "ltp", "stop", "state", "note")
            headings = (
                ("ts", 130, "Timestamp"),
                ("rel", 60, "Rel s"),
                ("ltp", 80, "Live LTP"),
                ("stop", 80, "Stop Price"),
                ("state", 80, "State"),
                ("note", 180, "Note"),
            )
        else:
            cols = ("ts", "rel", "ltp", "stop", "state", "decision", "dt")
            headings = (
                ("ts", 130, "Timestamp"),
                ("rel", 60, "Rel s"),
                ("ltp", 80, "Live LTP"),
                ("stop", 80, "Stop Price"),
                ("state", 80, "State"),
                ("decision", 150, "Simulator Decision"),
                ("dt", 55, "Δt s"),
            )

        tree_fr = ttk.Frame(box)
        tree_fr.grid(row=0, column=0, sticky="nsew")
        tree_fr.rowconfigure(0, weight=1)
        tree_fr.columnconfigure(0, weight=1)

        tree = ttk.Treeview(tree_fr, columns=cols, show="headings")
        for c, w, label in headings:
            tree.heading(c, text=label)
            tree.column(c, width=w, minwidth=50, anchor="w")

        ysb = ttk.Scrollbar(tree_fr, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(tree_fr, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")

        tree.tag_configure("cross", foreground=COL_WARN)
        tree.tag_configure("exit", foreground=COL_WARN)
        tree.tag_configure("pre", foreground=COL_MUTED)

        for row in rows:
            state = str(row.get("position_state") or "")
            if mode == "raw":
                note = str(row.get("note") or "")
                tags: list[str] = []
                if note:
                    tags.append("cross")
                elif state == "pre_entry":
                    tags.append("pre")
                tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("time_label") or "—",
                        f"{row.get('rel_sec'):+.3f}" if row.get("rel_sec") is not None else "—",
                        _fmt_num(row.get("live_ltp"), digits=4),
                        _fmt_num(row.get("stop_price"), digits=4),
                        state or "—",
                        note,
                    ),
                    tags=tuple(tags),
                )
            else:
                decision = str(row.get("decision") or "")
                tags = []
                if "Exit" in decision or "Stop" in decision:
                    tags.append("exit")
                elif state == "pre_entry":
                    tags.append("pre")
                dt = row.get("dt_from_prev_sample_sec")
                tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("time_label") or "—",
                        f"{row.get('rel_sec'):+.3f}" if row.get("rel_sec") is not None else "—",
                        _fmt_num(row.get("live_ltp"), digits=4),
                        _fmt_num(row.get("stop_price"), digits=4),
                        state or "—",
                        decision,
                        f"{dt:.1f}" if dt is not None else "",
                    ),
                    tags=tuple(tags),
                )

    def _render_charges(
        self,
        trades: list[dict[str, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        host = getattr(self, "_charges_host", None)
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()

        metrics = metrics or {}
        top = ttk.Frame(host)
        top.pack(fill="both", expand=True)

        # ── Selected strategy rules (left) ─────────────────────────────
        strat_fr = ttk.LabelFrame(top, text="Selected Strategy", padding=8)
        strat_fr.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ctx = self._selected_strategy_context()
        cfg = ctx.get("config") if isinstance(ctx.get("config"), dict) else {}
        if not cfg and not ctx.get("version_id"):
            ttk.Label(
                strat_fr,
                text="Select a strategy in the Strategy tab to see rules.",
                foreground=COL_MUTED,
                wraplength=360,
            ).pack(anchor="w")
        else:
            hdr = ttk.Frame(strat_fr)
            hdr.pack(fill="x", pady=(0, 4))
            name = str(ctx.get("display_name") or cfg.get("name") or "Strategy")
            ver = str(ctx.get("version_label") or (ctx.get("version_id") or "")[:8] or "—")
            ttk.Label(hdr, text=f"{name}  ·  {ver}", font=SECTION_FONT).pack(
                side="left", anchor="w"
            )
            ttk.Button(
                hdr,
                text="Edit",
                width=6,
                command=self._edit_selected_strategy,
            ).pack(side="right")

            desc = str(ctx.get("description") or cfg.get("description") or "").strip()
            ttk.Label(
                strat_fr,
                text=desc or "(no description)",
                foreground=COL_MUTED if not desc else None,
                wraplength=380,
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(0, 8))

            section_title(strat_fr, "Rules")
            inline_spec_rows(strat_fr, _strategy_rule_rows(cfg), label_width=16)

        # ── Charges (right) ────────────────────────────────────────────
        charges_fr = ttk.LabelFrame(top, text="Charges Breakdown", padding=8)
        charges_fr.pack(side="left", fill="both", expand=True, padx=(6, 0))

        if not trades:
            ttk.Label(
                charges_fr,
                text="Select a strategy run to see STT / exchange / SEBI / stamp / GST totals.",
                foreground=COL_MUTED,
                wraplength=360,
            ).pack(anchor="w")
            return

        from chain_replay_ml.capital_simulation import aggregate_trade_charge_breakdown

        br = aggregate_trade_charge_breakdown(list(trades or []))
        ttk.Label(
            charges_fr,
            text=(
                "Zero-brokerage plan: statutory charges only "
                "(STT + Exchange + SEBI + Stamp + GST). Brokerage = ₹0."
            ),
            foreground=COL_MUTED,
            wraplength=400,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        gross, net = _resolve_gross_net_profit(metrics)
        fees_metric = metrics.get("total_fees")
        inline_spec_rows(
            charges_fr,
            [
                ("Executed trades", fmt_rows(br.get("trade_count"))),
                ("Buy notional", fmt_rupee(br.get("buy_notional"))),
                ("Sell notional", fmt_rupee(br.get("sell_notional"))),
                ("Gross P&L", fmt_rupee(gross)),
                ("Total charges", fmt_rupee(br.get("total"))),
                (
                    "Fees on trades (stored)",
                    fmt_rupee(
                        fees_metric if fees_metric is not None else br.get("total_from_trade_fees")
                    ),
                ),
                ("Net P&L", fmt_rupee(net)),
                ("Avg charge / trade", fmt_rupee(br.get("avg_per_trade"))),
                ("Plan", str(br.get("plan") or "zero_brokerage_statutory")),
            ],
            label_width=22,
        )

        components = ttk.LabelFrame(charges_fr, text="Component totals", padding=8)
        components.pack(fill="x", pady=(8, 0))
        total = float(br.get("total") or 0) or 1.0

        def _pct(part: Any) -> str:
            try:
                return f"{100.0 * float(part or 0) / total:.1f}%"
            except (TypeError, ValueError, ZeroDivisionError):
                return "—"

        dual_spec_sections(
            components,
            [
                ("Brokerage", fmt_rupee(br.get("brokerage"))),
                ("STT", fmt_rupee(br.get("stt"))),
                ("Exchange txn", fmt_rupee(br.get("exchange"))),
                ("SEBI", fmt_rupee(br.get("sebi"))),
                ("Stamp duty", fmt_rupee(br.get("stamp"))),
                ("GST", fmt_rupee(br.get("gst"))),
                ("Total", fmt_rupee(br.get("total"))),
            ],
            [
                ("Brokerage %", _pct(br.get("brokerage"))),
                ("STT %", _pct(br.get("stt"))),
                ("Exchange %", _pct(br.get("exchange"))),
                ("SEBI %", _pct(br.get("sebi"))),
                ("Stamp %", _pct(br.get("stamp"))),
                ("GST %", _pct(br.get("gst"))),
                ("Total %", "100.0%"),
            ],
            left_title="₹ amount",
            right_title="Share",
            label_width=14,
        )

        # Daily charges rollup
        by_day: dict[str, float] = defaultdict(float)
        for t in trades:
            day = str(t.get("trading_day") or "—")
            try:
                by_day[day] += float(t.get("fees") or 0)
            except (TypeError, ValueError):
                pass
        if by_day:
            daily_fr = ttk.LabelFrame(host, text="Charges by day", padding=8)
            daily_fr.pack(fill="both", expand=True, pady=(8, 0))
            cols = ("day", "charges")
            tree = ttk.Treeview(daily_fr, columns=cols, show="headings", height=8)
            tree.heading("day", text="Day")
            tree.heading("charges", text="Charges")
            tree.column("day", width=120)
            tree.column("charges", width=120)
            tree.pack(side="left", fill="both", expand=True)
            sb = ttk.Scrollbar(daily_fr, orient="vertical", command=tree.yview)
            sb.pack(side="right", fill="y")
            tree.configure(yscrollcommand=sb.set)
            for day in sorted(by_day.keys()):
                tree.insert("", "end", values=(day, fmt_rupee(by_day[day])))

    def _render_pipeline(self, metrics: dict[str, Any]) -> None:
        for w in self._pipeline_host.winfo_children():
            w.destroy()
        if not metrics:
            ttk.Label(
                self._pipeline_host,
                text="Select a strategy run to see the prediction → trade funnel.",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            return

        counts = self._pipeline_counts(metrics)
        section_title(self._pipeline_host, "Simulation Pipeline")
        ttk.Label(
            self._pipeline_host,
            text=(
                "Pipeline: Prediction → Classifier → Probability Filter → Triple Barrier "
                "Filter → Strategy entry → Execution Rules → Executed trades. Metrics use "
                "executed trades only."
            ),
            foreground=COL_MUTED,
            wraplength=720,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 6))

        section_title(self._pipeline_host, "Simulator Summary")
        eq_pts = metrics.get("equity_curve_points")
        if eq_pts is None:
            eq_pts = (metrics.get("simulator_summary") or {}).get("equity_curve_points")
        match = (metrics.get("simulator_summary") or {}).get("equity_matches_executed")
        if match is None and eq_pts is not None and counts["executed"] is not None:
            try:
                match = int(eq_pts) == int(counts["executed"])
            except (TypeError, ValueError):
                match = None
        inline_spec_rows(
            self._pipeline_host,
            [
                ("Prediction rows", fmt_rows(counts["date_n"]) if counts["date_n"] is not None else "—"),
                ("Classifier kept", fmt_rows(counts["classifier_kept"]) if counts["classifier_kept"] is not None else "—"),
                (
                    "Probability filter",
                    f"{counts['probability_label']} ≥ {float(counts['probability_threshold']):.2f}"
                    if counts["probability_active"] and counts["probability_threshold"] is not None
                    else "Disabled",
                ),
                ("Probability kept", fmt_rows(counts["probability_kept"]) if counts["probability_kept"] is not None else "—"),
                ("Candidate signals", fmt_rows(counts["candidate_signals"]) if counts["candidate_signals"] is not None else "—"),
                ("Skipped (Maximum Positions)", fmt_rows(counts["skipped_max_positions"]) if counts["skipped_max_positions"] is not None else "—"),
                ("Skipped (Same Symbol)", fmt_rows(counts["skipped_same_symbol"]) if counts["skipped_same_symbol"] is not None else "—"),
                ("Executed trades", fmt_rows(counts["executed"]) if counts["executed"] is not None else "—"),
                ("Equity curve points", fmt_rows(eq_pts) if eq_pts is not None else "—"),
                (
                    "Equity points == executed",
                    "Yes" if match else ("No" if match is False else "—"),
                ),
            ],
            label_width=28,
        )

        section_title(self._pipeline_host, "Full Funnel")
        inline_spec_rows(
            self._pipeline_host,
            [
                ("Prediction Dataset", fmt_rows(counts["dataset_n"]) if counts["dataset_n"] is not None else "—"),
                ("Date-filtered Predictions", fmt_rows(counts["date_n"]) if counts["date_n"] is not None else "—"),
                ("Classifier Filter", str(counts["classifier_label"] or "Disabled")),
                ("After Classifier", fmt_rows(counts["classifier_kept"]) if counts["classifier_kept"] is not None else "—"),
                ("Classifier Removed", fmt_rows(counts["classifier_removed"]) if counts["classifier_removed"] is not None else "—"),
                ("Classification Filter", str(counts["probability_label"] or "Disabled")),
                (
                    "Probability Threshold",
                    f"{float(counts['probability_threshold']):.2f}"
                    if counts["probability_threshold"] is not None
                    else "—",
                ),
                ("After Probability Filter", fmt_rows(counts["probability_kept"]) if counts["probability_kept"] is not None else "—"),
                ("Probability Removed", fmt_rows(counts["probability_removed"]) if counts["probability_removed"] is not None else "—"),
                ("Triple Barrier Filter", str((metrics.get("tb_summary") or {}).get("tb_filter_label") or "Disabled")),
                (
                    "After Triple Barrier Filter",
                    fmt_rows((metrics.get("tb_summary") or {}).get("tb_kept"))
                    if (metrics.get("tb_summary") or {}).get("tb_kept") is not None
                    else "—",
                ),
                (
                    "Triple Barrier Removed",
                    fmt_rows((metrics.get("tb_summary") or {}).get("tb_removed"))
                    if (metrics.get("tb_summary") or {}).get("tb_removed") is not None
                    else "—",
                ),
                ("Predictions Evaluated", fmt_rows(counts["evaluated"]) if counts["evaluated"] is not None else "—"),
                ("Signals Generated", fmt_rows(counts["signals"]) if counts["signals"] is not None else "—"),
                ("Candidate Signals", fmt_rows(counts["candidate_signals"]) if counts["candidate_signals"] is not None else "—"),
                ("Signals Skipped", fmt_rows(counts["skipped"]) if counts["skipped"] is not None else "—"),
                ("Executed Trades", fmt_rows(counts["executed"]) if counts["executed"] is not None else "—"),
                ("Winning Trades", fmt_rows(counts["wins"]) if counts["wins"] is not None else "—"),
                ("Losing Trades", fmt_rows(counts["losses"]) if counts["losses"] is not None else "—"),
                ("Ignored Predictions", fmt_rows(counts["ignored"]) if counts["ignored"] is not None else "—"),
            ],
            label_width=24,
        )

        breakdown = []
        if metrics.get("no_signal") is not None:
            breakdown.append(("No entry signal", fmt_rows(metrics.get("no_signal"))))
        if metrics.get("blocked_open") is not None:
            breakdown.append(("Blocked (position open)", fmt_rows(metrics.get("blocked_open"))))
        if metrics.get("skipped_cadence") is not None:
            breakdown.append(("Skipped (entry cadence)", fmt_rows(metrics.get("skipped_cadence"))))
        if metrics.get("skipped_max_positions") is not None:
            breakdown.append(
                ("Skipped (Maximum Positions)", fmt_rows(metrics.get("skipped_max_positions")))
            )
        if metrics.get("skipped_same_symbol") is not None:
            breakdown.append(
                ("Skipped (Same Symbol)", fmt_rows(metrics.get("skipped_same_symbol")))
            )
        if metrics.get("skipped_no_path") is not None:
            breakdown.append(("Skipped (no exit path)", fmt_rows(metrics.get("skipped_no_path"))))
        if breakdown:
            section_title(self._pipeline_host, "Why predictions were not traded")
            inline_spec_rows(self._pipeline_host, breakdown, label_width=28)
        else:
            ttk.Label(
                self._pipeline_host,
                text="Filter breakdown available after re-running the simulation.",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(8, 0))

        self._render_tb_summary_section(metrics)
        self._render_tb_comparison_section()

    def _render_tb_summary_section(self, metrics: dict[str, Any]) -> None:
        tb_summary = metrics.get("tb_summary") if isinstance(metrics.get("tb_summary"), dict) else None
        if not tb_summary or not tb_summary.get("active"):
            return

        section_title(self._pipeline_host, "Triple Barrier Summary")
        rows = [
            ("Triple Barrier Model", str(tb_summary.get("tb_model_name") or "—")),
            ("Prediction Class", str(tb_summary.get("label") or "—")),
            ("Minimum Probability", f"{float(tb_summary.get('threshold')):.2f}" if tb_summary.get("threshold") is not None else "—"),
            ("Candidate Rows", fmt_rows(tb_summary.get("candidate_rows"))),
            ("Trades Filtered (removed)", fmt_rows(tb_summary.get("trades_filtered"))),
            ("Rows Kept", fmt_rows(tb_summary.get("rows_kept"))),
            ("Average TB Probability", _fmt_num(tb_summary.get("avg_tb_probability"), digits=4)),
            (
                "Skipped — Missing Triple Barrier prediction",
                fmt_rows(tb_summary.get("skipped_missing_count")),
            ),
        ]
        inline_spec_rows(self._pipeline_host, rows, label_width=32)

        class_counts = tb_summary.get("class_counts") or {}
        if class_counts:
            ttk.Label(
                self._pipeline_host,
                text="Predicted class counts among candidates:",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(6, 0))
            inline_spec_rows(
                self._pipeline_host,
                [(str(label), fmt_rows(count)) for label, count in class_counts.items()],
                label_width=20,
            )

        dist = tb_summary.get("probability_distribution") or {}
        if dist:
            ttk.Label(
                self._pipeline_host,
                text="TB probability distribution (selected class):",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(6, 0))
            inline_spec_rows(
                self._pipeline_host,
                [(bucket, fmt_rows(count)) for bucket, count in dist.items()],
                label_width=20,
            )

    def _render_tb_comparison_section(self) -> None:
        payload = getattr(self, "_last_tb_comparison", None)
        if not isinstance(payload, dict):
            return

        section_title(self._pipeline_host, "Baseline vs Filtered Comparison")
        ttk.Label(
            self._pipeline_host,
            text="Baseline = Triple Barrier filter off. Filtered = Triple Barrier filter on. All other filters/config identical.",
            foreground=COL_MUTED,
            wraplength=720,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 4))

        labels = {
            "total_trades": "Total Trades",
            "win_rate_pct": "Win Rate %",
            "net_profit": "Net Profit",
            "average_profit": "Average Profit",
            "average_drawdown": "Average Drawdown",
            "max_drawdown": "Max Drawdown",
            "sharpe": "Sharpe",
            "expectancy": "Expectancy",
            "profit_factor": "Profit Factor",
        }
        baseline = payload.get("baseline") or {}
        filtered = payload.get("filtered") or {}
        delta = payload.get("delta") or {}
        order = payload.get("metric_order") or list(labels.keys())

        table = ttk.Frame(self._pipeline_host)
        table.pack(fill="x", pady=(2, 6))
        header = ("Metric", "Baseline", "Filtered", "Delta")
        for col, text in enumerate(header):
            ttk.Label(table, text=text, font=("Segoe UI", 8, "bold")).grid(
                row=0, column=col, sticky="w", padx=(0, 16), pady=(0, 2)
            )
        for r, key in enumerate(order, start=1):
            b_val, f_val, d_val = baseline.get(key), filtered.get(key), delta.get(key)
            if key in ("net_profit", "average_profit", "average_drawdown", "max_drawdown"):
                b_txt = fmt_rupee(b_val) if b_val is not None else "—"
                f_txt = fmt_rupee(f_val) if f_val is not None else "—"
                d_txt = fmt_rupee(d_val) if d_val is not None else "—"
            else:
                b_txt = _fmt_num(b_val) if b_val is not None else "—"
                f_txt = _fmt_num(f_val) if f_val is not None else "—"
                d_txt = _fmt_num(d_val) if d_val is not None else "—"
            ttk.Label(table, text=labels.get(key, key)).grid(row=r, column=0, sticky="w", padx=(0, 16))
            ttk.Label(table, text=b_txt, font=("Consolas", 9)).grid(row=r, column=1, sticky="w", padx=(0, 16))
            ttk.Label(table, text=f_txt, font=("Consolas", 9)).grid(row=r, column=2, sticky="w", padx=(0, 16))
            ttk.Label(table, text=d_txt, font=("Consolas", 9)).grid(row=r, column=3, sticky="w")

    def _render_trades(self, trades: list[dict[str, Any]]) -> None:
        self._trades_tree.delete(*self._trades_tree.get_children())
        for t in trades:
            pnl = t.get("net_pnl")
            meta = t.get("meta") if isinstance(t.get("meta"), dict) else {}
            exit_sample = t.get("exit_sample_index")
            if exit_sample is None:
                exit_sample = meta.get("exit_sample_index")
            hold = t.get("holding_seconds")
            try:
                hold_disp = f"{float(hold):.3f}" if hold is not None else "—"
            except (TypeError, ValueError):
                hold_disp = str(hold) if hold is not None else "—"
            self._trades_tree.insert(
                "",
                "end",
                tags=(_pnl_tag(pnl),),
                values=(
                    t.get("trading_day"),
                    t.get("token"),
                    self._fmt_exit_ts(t.get("entry_ts")),
                    self._fmt_exit_ts(t.get("exit_ts")),
                    _fmt_num(t.get("entry_price"), digits=4),
                    _fmt_num(t.get("exit_price"), digits=4),
                    _fmt_signed_pnl(pnl),
                    _fmt_num(t.get("return_pct")),
                    hold_disp,
                    t.get("exit_reason") or "—",
                    exit_sample if exit_sample is not None else "—",
                ),
            )

    def _download_trades_csv(self) -> None:
        trades = list(getattr(self, "_current_trades", None) or [])
        if not trades:
            messagebox.showinfo(
                "Download CSV",
                "No trades to download. Select a strategy run first.",
                parent=self,
            )
            return
        run = getattr(self, "_current_run", None) or {}
        run_id = str(run.get("strategy_run_id") or "strategy_run")[:12]
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Download Trade List CSV",
            defaultextension=".csv",
            initialfile=f"trade_list_{run_id}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        fieldnames = [
            "trading_day",
            "token",
            "strike",
            "option_type",
            "direction",
            "trade_id",
            "entry_ts",
            "exit_ts",
            "entry_time_ist",
            "exit_time_ist",
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "qty",
            "gross_pnl",
            "fees",
            "net_pnl",
            "return_pct",
            "holding_seconds",
            "exit_reason",
            "exit_sample_index",
            "exit_row_index",
            "stop_trigger_ltp",
            "target_trigger_ltp",
            "sample_exit_ltp",
            "gap_beyond_stop",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for t in trades:
                    meta = t.get("meta") if isinstance(t.get("meta"), dict) else {}

                    def _pick(key: str) -> Any:
                        v = t.get(key)
                        return v if v is not None else meta.get(key)

                    writer.writerow({
                        "trading_day": t.get("trading_day"),
                        "token": t.get("token"),
                        "strike": t.get("strike"),
                        "option_type": t.get("option_type"),
                        "direction": _pick("direction"),
                        "trade_id": t.get("trade_id"),
                        "entry_ts": t.get("entry_ts"),
                        "exit_ts": t.get("exit_ts"),
                        "entry_time_ist": self._fmt_exit_ts(t.get("entry_ts")),
                        "exit_time_ist": self._fmt_exit_ts(t.get("exit_ts")),
                        "entry_price": t.get("entry_price"),
                        "exit_price": t.get("exit_price"),
                        "stop_price": _pick("stop_price"),
                        "target_price": _pick("target_price"),
                        "qty": t.get("qty"),
                        "gross_pnl": t.get("gross_pnl"),
                        "fees": t.get("fees"),
                        "net_pnl": t.get("net_pnl"),
                        "return_pct": t.get("return_pct"),
                        "holding_seconds": t.get("holding_seconds"),
                        "exit_reason": t.get("exit_reason"),
                        "exit_sample_index": _pick("exit_sample_index"),
                        "exit_row_index": _pick("exit_row_index"),
                        "stop_trigger_ltp": _pick("stop_trigger_ltp"),
                        "target_trigger_ltp": _pick("target_trigger_ltp"),
                        "sample_exit_ltp": _pick("sample_exit_ltp"),
                        "gap_beyond_stop": _pick("gap_beyond_stop"),
                    })
        except OSError as exc:
            messagebox.showerror("Download CSV", str(exc), parent=self)
            return
        self._status_var.set(f"Trade list CSV saved: {path}")
        messagebox.showinfo("Download CSV", f"Saved {len(trades)} trades to:\n{path}", parent=self)

    def _render_daily(self, trades: list[dict[str, Any]]) -> None:
        self._daily_tree.delete(*self._daily_tree.get_children())
        for row in self._daily_pnl_rows(trades):
            total = row["net_pnl"]
            win_pct = row["win_pct"]
            self._daily_tree.insert(
                "",
                "end",
                tags=(_pnl_tag(total),),
                values=(
                    row["day"],
                    row["trades"],
                    row["wins"],
                    _fmt_num(win_pct, digits=1) if win_pct is not None else "—",
                    _fmt_signed_pnl(total),
                ),
            )

    @staticmethod
    def _fmt_exit_ts(ts: Any) -> str:
        if ts is None or ts == "":
            return "—"
        try:
            val = float(ts)
        except (TypeError, ValueError):
            return str(ts)
        # Real lab timestamps are unix epoch; unit tests use small synthetic values.
        if val >= 1_000_000_000:
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                return datetime.fromtimestamp(val, tz=ZoneInfo("Asia/Kolkata")).strftime(
                    "%Y-%m-%d %H:%M:%S IST"
                )
            except (OSError, OverflowError, ValueError):
                return str(val)
        return str(val)

    def _render_equity(
        self,
        trades: list[dict[str, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        from chain_replay_ml.strategy_simulator.metrics import (
            annotate_equity_curve_max_dd,
            build_equity_curve,
            compute_max_drawdown_episode,
        )

        metrics = metrics or {}
        episode = metrics.get("max_drawdown_episode")
        if not isinstance(episode, dict) or not episode:
            episode = compute_max_drawdown_episode(list(trades or []))
        curve = annotate_equity_curve_max_dd(build_equity_curve(list(trades or [])), episode)
        self._equity_curve_cache = curve
        self._equity_episode_cache = dict(episode)

        for w in self._equity_summary_host.winfo_children():
            w.destroy()
        section_title(self._equity_summary_host, "Account Equity Max Drawdown (closed-trade equity)")
        ttk.Label(
            self._equity_summary_host,
            text=(
                "Peak-to-trough on cumulative realized net P&L after each executed trade closes. "
                "Separate from Max Portfolio DD (Open Risk) and Max Theoretical Portfolio Risk."
            ),
            foreground=COL_MUTED,
            wraplength=720,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 4))
        pr = (metrics or {}).get("portfolio_risk") or {}
        theo = pr.get("max_theoretical_portfolio_risk") or {}
        open_r = pr.get("max_portfolio_drawdown_open_risk") or {}
        inline_spec_rows(
            self._equity_summary_host,
            [
                ("Account Equity Max DD", fmt_rupee(episode.get("max_drawdown"))),
                ("Peak Equity", fmt_rupee(episode.get("peak_equity"))),
                ("Trough Equity", fmt_rupee(episode.get("trough_equity"))),
                (
                    "Peak at",
                    (
                        f"#{episode.get('peak_point')} · "
                        f"{episode.get('peak_trading_day') or '—'} · "
                        f"{self._fmt_exit_ts(episode.get('peak_exit_ts'))}"
                        if episode.get("peak_point") is not None
                        else "—"
                    ),
                ),
                (
                    "Trough at",
                    (
                        f"#{episode.get('trough_point')} · "
                        f"{episode.get('trough_trading_day') or '—'} · "
                        f"{self._fmt_exit_ts(episode.get('trough_exit_ts'))}"
                        if episode.get("trough_point") is not None
                        else "—"
                    ),
                ),
                (
                    "Max Portfolio DD (Open Risk)",
                    fmt_rupee(
                        (metrics or {}).get(
                            "max_portfolio_drawdown_open_risk",
                            open_r.get("max_portfolio_drawdown_open_risk"),
                        )
                    ),
                ),
                (
                    "Open risk method",
                    str(open_r.get("method") or "simultaneous unrealized gross"),
                ),
                (
                    "Max Theoretical Portfolio Risk",
                    fmt_rupee(
                        (metrics or {}).get(
                            "max_theoretical_portfolio_risk",
                            theo.get("max_theoretical_portfolio_risk"),
                        )
                    ),
                ),
                (
                    "Theoretical formula",
                    str(
                        theo.get("formula")
                        or (
                            f"{(metrics or {}).get('max_open_positions_for_risk')} × "
                            f"{fmt_rupee((metrics or {}).get('stop_loss_per_trade_rupees'))}"
                        )
                    ),
                ),
                ("Equity curve points", fmt_rows(len(curve))),
                ("Executed trades", fmt_rows(len(trades or []))),
            ],
            label_width=30,
        )

        self._equity_tree.delete(*self._equity_tree.get_children())
        for point in curve:
            pnl = point.get("net_pnl")
            dd = float(point.get("drawdown") or 0)
            mark = ""
            tags = [_pnl_tag(pnl)]
            if point.get("is_max_dd_peak"):
                mark = "PEAK"
                tags.append("dd_peak")
            if point.get("is_max_dd_trough"):
                mark = "TROUGH" if not mark else "PEAK+TROUGH"
                tags.append("dd_trough")
            self._equity_tree.insert(
                "",
                "end",
                tags=tuple(tags),
                values=(
                    point.get("point"),
                    point.get("trading_day"),
                    point.get("token") or "—",
                    _fmt_num(point.get("entry_price"), digits=4),
                    _fmt_num(point.get("exit_price"), digits=4),
                    _fmt_signed_pnl(pnl),
                    _fmt_signed_pnl(point.get("equity")),
                    _fmt_signed_pnl(point.get("peak")),
                    _fmt_signed_pnl(-dd) if dd else "0",
                    mark,
                ),
            )
        self._redraw_equity_canvas()

    def _download_equity_curve_csv(self) -> None:
        curve = list(getattr(self, "_equity_curve_cache", None) or [])
        if not curve:
            messagebox.showinfo(
                "Download CSV",
                "No equity curve rows to download. Select a strategy run first.",
                parent=self,
            )
            return
        run = getattr(self, "_current_run", None) or {}
        run_id = str(run.get("strategy_run_id") or "strategy_run")[:12]
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Download Equity Curve CSV",
            defaultextension=".csv",
            initialfile=f"equity_curve_{run_id}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        fieldnames = [
            "point",
            "trading_day",
            "token",
            "trade_id",
            "entry_ts",
            "exit_ts",
            "entry_price",
            "exit_price",
            "exit_reason",
            "qty",
            "net_pnl",
            "equity",
            "peak",
            "drawdown",
            "is_max_dd_peak",
            "is_max_dd_trough",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in curve:
                    writer.writerow({k: row.get(k) for k in fieldnames})
        except OSError as exc:
            messagebox.showerror("Download CSV", str(exc), parent=self)
            return
        self._status_var.set(f"Equity curve CSV saved: {path}")
        messagebox.showinfo("Download CSV", f"Saved {len(curve)} rows to:\n{path}", parent=self)

    def _redraw_equity_canvas(self) -> None:
        canvas = getattr(self, "_equity_canvas", None)
        if canvas is None:
            return
        curve = getattr(self, "_equity_curve_cache", None) or []
        episode = getattr(self, "_equity_episode_cache", None) or {}
        canvas.delete("all")
        w = max(int(canvas.winfo_width() or 0), 200)
        h = max(int(canvas.winfo_height() or 0), 160)
        pad = 28
        canvas.create_rectangle(0, 0, w, h, fill="#1a2a44", outline="")
        canvas.create_text(
            pad,
            12,
            text="Equity curve (executed trades) — green=Max DD peak, red=Max DD trough",
            anchor="w",
            fill="#aab",
            font=("Segoe UI", 9, "bold"),
        )
        if not curve:
            canvas.create_text(w / 2, h / 2, text="No executed trades", fill="#666", font=("Segoe UI", 9))
            return

        equities = [float(p.get("equity") or 0) for p in curve]
        points = [0.0] + equities  # start at flat zero before first close
        lo = min(min(points), 0.0)
        hi = max(max(points), 0.0)
        if hi == lo:
            hi = lo + 1.0
        inner_w = w - pad * 2
        inner_h = h - pad * 2 - 8

        def _xy(i: int, v: float) -> tuple[float, float]:
            x = pad + (inner_w * i / max(len(points) - 1, 1))
            y = pad + 8 + inner_h * (hi - v) / (hi - lo)
            return x, y

        zero_y = _xy(0, 0.0)[1]
        canvas.create_line(pad, zero_y, w - pad, zero_y, fill="#334", dash=(3, 3))

        coords: list[float] = []
        for i, v in enumerate(points):
            x, y = _xy(i, v)
            coords.extend([x, y])
        canvas.create_line(*coords, fill="#58a6ff", width=2)

        peak_pt = episode.get("peak_point")
        trough_pt = episode.get("trough_point")
        # points[0] is equity before trades; curve point k maps to index k
        if peak_pt is not None:
            try:
                idx = int(peak_pt)
                x, y = _xy(idx, float(episode.get("peak_equity") or points[idx]))
                canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#3fb950", outline="#fff")
                canvas.create_text(x + 8, y - 8, text="Peak", anchor="w", fill="#3fb950", font=("Segoe UI", 8))
            except (TypeError, ValueError, IndexError):
                pass
        if trough_pt is not None:
            try:
                idx = int(trough_pt)
                x, y = _xy(idx, float(episode.get("trough_equity") or points[idx]))
                canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#f85149", outline="#fff")
                canvas.create_text(x + 8, y + 10, text="Trough", anchor="w", fill="#f85149", font=("Segoe UI", 8))
                # Shade the drawdown span peak → trough on the curve
                if peak_pt is not None:
                    pidx = int(peak_pt)
                    if 0 <= pidx < len(points) and 0 <= idx < len(points) and pidx <= idx:
                        span: list[float] = []
                        for j in range(pidx, idx + 1):
                            sx, sy = _xy(j, points[j])
                            span.extend([sx, sy])
                        if len(span) >= 4:
                            canvas.create_line(*span, fill="#f85149", width=3)
            except (TypeError, ValueError, IndexError):
                pass
