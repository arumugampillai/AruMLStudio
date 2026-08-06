"""Strike Prediction Dashboard — per-strike LTP / confidence / error / gap charts."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .fold_replay_widgets import draw_line_chart
from .model_registry_widgets import COL_MUTED, SECTION_FONT

# Chart colors (dark canvas — white Actual LTP readable)
COL_ACTUAL = "#ffffff"
COL_PREDICTED = "#58a6ff"
COL_PRED_EMA = "#1e3a8a"
COL_CONF = "#f59e0b"
COL_CONF_EMA = "#ef4444"
COL_CONF_EMA_10 = "#fb7185"
COL_ERROR = "#a78bfa"
COL_ERROR_EMA = "#7c3aed"
COL_GAP = "#38bdf8"
COL_REGR = "#58a6ff"
COL_REGR_EMA = "#1e3a8a"
COL_REGR_SLOPE = "#f472b6"
CHART_FILL = "#0d1117"

TAB_SPECS: tuple[tuple[str, str], ...] = (
    ("ltp", "Actual vs Prediction"),
    ("conf", "Confidence"),
    ("err", "Prediction Error"),
    ("gap", "Prediction Gap"),
    ("regr", "Regression Trend"),
    ("conf_pred", "Confidence vs Prediction"),
)


class ModelLabStrikeDashboardPanel(ttk.Frame):
    """Research Lab → Strike Dashboard: six tabs (error tab is stats-first)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_status: Callable[[str], None] | None = None,
        get_day_filter: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_status = on_status or (lambda _s: None)
        self._get_day_filter = get_day_filter or (lambda: None)
        self._lab_db_path = ""
        self._model_name = ""
        self._load_gen = 0
        self._loading = False
        self._strike_labels: list[str] = []
        self._day_options: list[str] = []
        self._last_bundle: dict[str, Any] | None = None
        self._cursor_index: int | None = None
        self._cursor_ts: Any = None
        self._chart_layouts: dict[str, dict[str, Any]] = {}
        self._build_ui()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def refresh_for_lab(
        self,
        *,
        lab_db_path: str | None,
        model_name: str = "",
        trading_day: str | None = None,
    ) -> None:
        self._lab_db_path = str(lab_db_path or "").strip()
        self._model_name = str(model_name or "").strip()
        if trading_day is not None:
            day = str(trading_day or "").strip()
            self._day_var.set(day if day else "All days")
        elif self._get_day_filter:
            day = self._get_day_filter()
            if day:
                self._day_var.set(str(day))
        self._reload_filters_then_charts()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(hdr, text="Strike Prediction Dashboard", font=SECTION_FONT).pack(
            side="left"
        )
        ttk.Label(
            hdr,
            text="Actual vs predicted · confidence · error stats · gap · regression",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(10, 0))

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))

        ttk.Label(toolbar, text="Day:").pack(side="left")
        self._day_var = tk.StringVar(value="All days")
        self._day_combo = ttk.Combobox(
            toolbar, textvariable=self._day_var, state="readonly", width=14
        )
        self._day_combo.pack(side="left", padx=(4, 10))
        self._day_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_day_changed())

        ttk.Label(toolbar, text="Strike:").pack(side="left")
        self._strike_var = tk.StringVar(value="")
        self._strike_combo = ttk.Combobox(
            toolbar, textvariable=self._strike_var, state="readonly", width=16
        )
        self._strike_combo.pack(side="left", padx=(4, 10))
        self._strike_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._load_charts_async()
        )

        ttk.Label(toolbar, text="EMA:").pack(side="left")
        self._ema_var = tk.StringVar(value="5")
        self._ema_combo = ttk.Combobox(
            toolbar,
            textvariable=self._ema_var,
            values=("5", "10"),
            state="readonly",
            width=4,
        )
        self._ema_combo.pack(side="left", padx=(4, 10))
        self._ema_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._load_charts_async()
        )

        ttk.Button(toolbar, text="Refresh", command=self._reload_filters_then_charts).pack(
            side="left", padx=(0, 8)
        )
        self._status_var = tk.StringVar(value="Open a Research Lab to begin.")
        ttk.Label(toolbar, textvariable=self._status_var, foreground=COL_MUTED).pack(
            side="left", padx=(4, 0)
        )

        # Always-visible compact summaries (any chart tab).
        self._build_global_ltp_strip()

        detail = ttk.Frame(self)
        detail.pack(fill="x", pady=(0, 6))
        self._detail_var = tk.StringVar(
            value="Hover a chart for linked crosshair details."
        )
        ttk.Label(
            detail,
            textvariable=self._detail_var,
            foreground=COL_MUTED,
            wraplength=960,
            justify="left",
        ).pack(anchor="w")

        self._chart_notebook = ttk.Notebook(self)
        self._chart_notebook.pack(fill="both", expand=True)
        self._chart_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._canvases: dict[str, tk.Canvas] = {}
        self._err_source_indices: list[int] = []
        self._show_err_chart_var = tk.BooleanVar(value=False)
        for key, title in TAB_SPECS:
            tab = ttk.Frame(self._chart_notebook, padding=2)
            self._chart_notebook.add(tab, text=title)
            if key == "err":
                self._build_error_tab(tab)
            elif key == "ltp":
                self._build_ltp_tab(tab)
            else:
                cv = tk.Canvas(tab, background=CHART_FILL, highlightthickness=0)
                cv.pack(fill="both", expand=True)
                cv.bind("<Configure>", lambda _e, k=key: self._redraw_one(k))
                cv.bind("<Motion>", lambda e, k=key: self._on_chart_motion(k, e))
                cv.bind("<Leave>", lambda _e: None)
                self._canvases[key] = cv

    def _build_global_ltp_strip(self) -> None:
        """One-line Actual / Predicted / Premium summaries above the chart notebook."""
        strip = ttk.Frame(self)
        strip.pack(fill="x", pady=(0, 6))
        for col in range(3):
            strip.columnconfigure(col, weight=1, uniform="ltp_strip")

        self._ltp_strip_actual_var = tk.StringVar(value="Min — · Max — · Mean — · Med —")
        self._ltp_strip_predicted_var = tk.StringVar(
            value="Min — · Max — · Mean — · Med —"
        )
        self._ltp_strip_premium_var = tk.StringVar(value="Range — · Spread —")

        actual_fr = ttk.LabelFrame(strip, text="Actual LTP", padding=(6, 2))
        actual_fr.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Label(actual_fr, textvariable=self._ltp_strip_actual_var).pack(anchor="w")

        predicted_fr = ttk.LabelFrame(strip, text="Predicted LTP", padding=(6, 2))
        predicted_fr.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(predicted_fr, textvariable=self._ltp_strip_predicted_var).pack(
            anchor="w"
        )

        premium_fr = ttk.LabelFrame(strip, text="Premium Range", padding=(6, 2))
        premium_fr.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        ttk.Label(premium_fr, textvariable=self._ltp_strip_premium_var).pack(anchor="w")

    def _build_ltp_tab(self, tab: ttk.Frame) -> None:
        """Actual vs Prediction: three summary blocks in one row + chart."""
        outer = ttk.Frame(tab)
        outer.pack(fill="both", expand=True)

        summary_row = ttk.Frame(outer)
        summary_row.pack(fill="x", pady=(0, 6))
        for col in range(3):
            summary_row.columnconfigure(col, weight=1, uniform="ltp_sum")

        actual_fr = ttk.LabelFrame(summary_row, text="Actual LTP", padding=6)
        actual_fr.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        predicted_fr = ttk.LabelFrame(summary_row, text="Predicted LTP", padding=6)
        predicted_fr.grid(row=0, column=1, sticky="nsew", padx=4)
        premium_fr = ttk.LabelFrame(summary_row, text="Premium Range", padding=6)
        premium_fr.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        self._ltp_actual_vars: dict[str, tk.StringVar] = {
            "min": tk.StringVar(value="Min: —"),
            "max": tk.StringVar(value="Max: —"),
            "mean": tk.StringVar(value="Mean: —"),
            "median": tk.StringVar(value="Median: —"),
        }
        self._ltp_predicted_vars: dict[str, tk.StringVar] = {
            "min": tk.StringVar(value="Min: —"),
            "max": tk.StringVar(value="Max: —"),
            "mean": tk.StringVar(value="Mean: —"),
            "median": tk.StringVar(value="Median: —"),
        }
        self._ltp_premium_vars: dict[str, tk.StringVar] = {
            "range": tk.StringVar(value="Range: —"),
            "spread": tk.StringVar(value="Spread: —"),
        }
        for i, key in enumerate(("min", "max", "mean", "median")):
            ttk.Label(actual_fr, textvariable=self._ltp_actual_vars[key]).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 14), pady=1
            )
            ttk.Label(predicted_fr, textvariable=self._ltp_predicted_vars[key]).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 14), pady=1
            )
        ttk.Label(premium_fr, textvariable=self._ltp_premium_vars["range"]).pack(
            anchor="w"
        )
        ttk.Label(premium_fr, textvariable=self._ltp_premium_vars["spread"]).pack(
            anchor="w"
        )

        cv = tk.Canvas(outer, background=CHART_FILL, highlightthickness=0)
        cv.pack(fill="both", expand=True)
        cv.bind("<Configure>", lambda _e: self._redraw_one("ltp"))
        cv.bind("<Motion>", lambda e: self._on_chart_motion("ltp", e))
        cv.bind("<Leave>", lambda _e: None)
        self._canvases["ltp"] = cv

    @staticmethod
    def _block_line(block: dict[str, Any], *, fmt: Callable[[Any], str]) -> str:
        if not int(block.get("n") or 0):
            return "Min — · Max — · Mean — · Med —"
        return (
            f"Min {fmt(block.get('min'))} · Max {fmt(block.get('max'))} · "
            f"Mean {fmt(block.get('mean'))} · Med {fmt(block.get('median'))}"
        )

    def _refresh_ltp_stats_ui(self) -> None:
        b = self._last_bundle
        summary = (b or {}).get("ltp_summary") if b else None
        empty_block = {"n": 0}

        if not isinstance(summary, dict):
            for key, default in (
                ("min", "Min: —"),
                ("max", "Max: —"),
                ("mean", "Mean: —"),
                ("median", "Median: —"),
            ):
                self._ltp_actual_vars[key].set(default)
                self._ltp_predicted_vars[key].set(default)
            self._ltp_premium_vars["range"].set("Range: —")
            self._ltp_premium_vars["spread"].set("Spread: —")
            self._ltp_strip_actual_var.set(self._block_line(empty_block, fmt=self._fmt_num))
            self._ltp_strip_predicted_var.set(
                self._block_line(empty_block, fmt=self._fmt_num)
            )
            self._ltp_strip_premium_var.set("Range — · Spread —")
            return

        actual = summary.get("actual") if isinstance(summary.get("actual"), dict) else {}
        predicted = (
            summary.get("predicted")
            if isinstance(summary.get("predicted"), dict)
            else {}
        )
        premium = (
            summary.get("premium_range")
            if isinstance(summary.get("premium_range"), dict)
            else {}
        )

        def _set_block(vars_map: dict[str, tk.StringVar], block: dict[str, Any]) -> None:
            if not int(block.get("n") or 0):
                for key, default in (
                    ("min", "Min: —"),
                    ("max", "Max: —"),
                    ("mean", "Mean: —"),
                    ("median", "Median: —"),
                ):
                    vars_map[key].set(default)
                return
            vars_map["min"].set(f"Min: {self._fmt_num(block.get('min'))}")
            vars_map["max"].set(f"Max: {self._fmt_num(block.get('max'))}")
            vars_map["mean"].set(f"Mean: {self._fmt_num(block.get('mean'))}")
            vars_map["median"].set(f"Median: {self._fmt_num(block.get('median'))}")

        _set_block(self._ltp_actual_vars, actual)
        _set_block(self._ltp_predicted_vars, predicted)

        prem_min = premium.get("min")
        prem_max = premium.get("max")
        spread = premium.get("spread")
        if prem_min is not None and prem_max is not None:
            self._ltp_premium_vars["range"].set(
                f"Range: ₹{self._fmt_num(prem_min)} → ₹{self._fmt_num(prem_max)}"
            )
        else:
            self._ltp_premium_vars["range"].set("Range: —")
        if spread is not None:
            self._ltp_premium_vars["spread"].set(f"Spread: ₹{self._fmt_num(spread)}")
        else:
            self._ltp_premium_vars["spread"].set("Spread: —")

        # Global strip (visible on every chart tab).
        self._ltp_strip_actual_var.set(self._block_line(actual, fmt=self._fmt_num))
        self._ltp_strip_predicted_var.set(self._block_line(predicted, fmt=self._fmt_num))
        if prem_min is not None and prem_max is not None:
            prem_txt = (
                f"₹{self._fmt_num(prem_min)} → ₹{self._fmt_num(prem_max)}"
                f" · Spread ₹{self._fmt_num(spread) if spread is not None else '—'}"
            )
            self._ltp_strip_premium_var.set(prem_txt)
        else:
            self._ltp_strip_premium_var.set("Range — · Spread —")

    def _build_error_tab(self, tab: ttk.Frame) -> None:
        """Stats-first Prediction Error layout; chart optional / collapsed."""
        outer = ttk.Frame(tab)
        outer.pack(fill="both", expand=True)

        # Summary + trend flags in one row (summary wider).
        top_row = ttk.Frame(outer)
        top_row.pack(fill="x", pady=(0, 6))
        top_row.columnconfigure(0, weight=3)
        top_row.columnconfigure(1, weight=1)

        summary = ttk.LabelFrame(top_row, text="Error summary", padding=6)
        summary.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._err_summary_vars: dict[str, tk.StringVar] = {
            "mean": tk.StringVar(value="Mean (bias): —"),
            "mae": tk.StringVar(value="MAE: —"),
            "rmse": tk.StringVar(value="RMSE: —"),
            "opt": tk.StringVar(value="% optimistic: —"),
            "pes": tk.StringVar(value="% pessimistic: —"),
            "std": tk.StringVar(value="Std: —"),
            "latest": tk.StringVar(value="Latest error: —"),
            "ema": tk.StringVar(value="Error EMA: —"),
            "n": tk.StringVar(value="n: —"),
        }
        grid = ttk.Frame(summary)
        grid.pack(fill="x")
        order = (
            "mean",
            "mae",
            "rmse",
            "opt",
            "pes",
            "std",
            "latest",
            "ema",
            "n",
        )
        for i, key in enumerate(order):
            ttk.Label(grid, textvariable=self._err_summary_vars[key]).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 18), pady=1
            )

        trends = ttk.LabelFrame(top_row, text="Trend flags", padding=6)
        trends.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._err_trend_mag_var = tk.StringVar(value="Error —")
        self._err_trend_bias_var = tk.StringVar(value="Bias —")
        ttk.Label(trends, textvariable=self._err_trend_mag_var).pack(anchor="w")
        ttk.Label(trends, textvariable=self._err_trend_bias_var).pack(anchor="w")

        # Nested: Error Quantiles | Recent Samples
        self._err_sub_notebook = ttk.Notebook(outer)
        self._err_sub_notebook.pack(fill="both", expand=True)

        q_tab = ttk.Frame(self._err_sub_notebook, padding=4)
        self._err_sub_notebook.add(q_tab, text="Error Quantiles")
        self._err_q_tree = ttk.Treeview(
            q_tab,
            columns=("q", "v", "n"),
            show="headings",
            height=12,
            selectmode="none",
        )
        self._err_q_tree.heading("q", text="Quantile")
        self._err_q_tree.heading("v", text="Error ₹")
        self._err_q_tree.heading("n", text="Samples")
        self._err_q_tree.column("q", width=80, anchor="w")
        self._err_q_tree.column("v", width=100, anchor="e")
        self._err_q_tree.column("n", width=70, anchor="e")
        q_scroll = ttk.Scrollbar(q_tab, orient="vertical", command=self._err_q_tree.yview)
        self._err_q_tree.configure(yscrollcommand=q_scroll.set)
        self._err_q_tree.pack(side="left", fill="both", expand=True)
        q_scroll.pack(side="right", fill="y")

        r_tab = ttk.Frame(self._err_sub_notebook, padding=4)
        self._err_sub_notebook.add(r_tab, text="Recent Samples")
        self._err_recent_tree = ttk.Treeview(
            r_tab,
            columns=(
                "ts",
                "cur",
                "fact",
                "fpred",
                "adelta",
                "pdelta",
                "err",
                "ema",
                "conf",
                "conf_ema",
            ),
            show="headings",
            height=12,
            selectmode="browse",
        )
        for col, text, w, anchor in (
            ("ts", "Timestamp", 90, "w"),
            ("cur", "Current LTP", 78, "e"),
            ("fact", "Future Actual", 82, "e"),
            ("fpred", "Future Pred", 78, "e"),
            ("adelta", "Actual Δ₹", 72, "e"),
            ("pdelta", "Pred Δ₹", 68, "e"),
            ("err", "Error", 60, "e"),
            ("ema", "Error EMA", 72, "e"),
            ("conf", "Conf", 58, "e"),
            ("conf_ema", "Conf EMA", 72, "e"),
        ):
            self._err_recent_tree.heading(col, text=text)
            self._err_recent_tree.column(col, width=w, anchor=anchor, minwidth=50)
        recent_scroll = ttk.Scrollbar(
            r_tab, orient="vertical", command=self._err_recent_tree.yview
        )
        self._err_recent_tree.configure(yscrollcommand=recent_scroll.set)
        self._err_recent_tree.pack(side="left", fill="both", expand=True)
        recent_scroll.pack(side="right", fill="y")
        self._err_recent_tree.bind("<<TreeviewSelect>>", self._on_err_recent_select)

        # Optional chart (below nested notebook)
        chart_bar = ttk.Frame(outer)
        chart_bar.pack(fill="x", pady=(6, 2))
        ttk.Checkbutton(
            chart_bar,
            text="Show chart (downsampled)",
            variable=self._show_err_chart_var,
            command=self._on_err_chart_toggle,
        ).pack(side="left")
        self._err_chart_hint_var = tk.StringVar(
            value="Chart off — use summary / quantiles / recent samples."
        )
        ttk.Label(
            chart_bar, textvariable=self._err_chart_hint_var, foreground=COL_MUTED
        ).pack(side="left", padx=(10, 0))

        self._err_chart_frame = ttk.Frame(outer)
        # Not packed until checkbox is on
        cv = tk.Canvas(
            self._err_chart_frame, background=CHART_FILL, highlightthickness=0, height=220
        )
        cv.pack(fill="both", expand=True)
        cv.bind("<Configure>", lambda _e: self._redraw_one("err"))
        cv.bind("<Motion>", lambda e: self._on_chart_motion("err", e))
        cv.bind("<Leave>", lambda _e: None)
        self._canvases["err"] = cv

    def _signed_rupee(self, v: Any, *, digits: int = 2) -> str:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "—"
        if fv != fv:
            return "—"
        if fv > 0:
            return f"+{fv:.{digits}f}"
        if fv < 0:
            return f"−{abs(fv):.{digits}f}"
        return f"{fv:.{digits}f}"

    def _on_err_chart_toggle(self) -> None:
        if self._show_err_chart_var.get():
            self._err_chart_frame.pack(fill="both", expand=True, pady=(0, 2))
            self._redraw_one("err")
        else:
            self._err_chart_frame.pack_forget()
            self._err_chart_hint_var.set(
                "Chart off — use summary / quantiles / recent samples."
            )

    def _on_err_recent_select(self, _event: Any = None) -> None:
        sel = self._err_recent_tree.selection()
        if not sel:
            return
        item = sel[0]
        try:
            idx = int(self._err_recent_tree.item(item, "tags")[0])
        except (IndexError, TypeError, ValueError):
            return
        b = self._last_bundle
        if not b:
            return
        timestamps = b.get("timestamps") or []
        self._cursor_index = idx
        self._cursor_ts = timestamps[idx] if idx < len(timestamps) else None
        self._update_detail_strip()
        if self._show_err_chart_var.get():
            self._redraw_one("err")

    def _refresh_error_stats_ui(self) -> None:
        b = self._last_bundle
        summary = (b or {}).get("error_summary") if b else None
        if not isinstance(summary, dict) or not int(summary.get("n") or 0):
            for key, default in (
                ("mean", "Mean (bias): —"),
                ("mae", "MAE: —"),
                ("rmse", "RMSE: —"),
                ("opt", "% optimistic: —"),
                ("pes", "% pessimistic: —"),
                ("std", "Std: —"),
                ("latest", "Latest error: —"),
                ("ema", "Error EMA: —"),
                ("n", "n: —"),
            ):
                self._err_summary_vars[key].set(default)
            self._err_trend_mag_var.set("Error —")
            self._err_trend_bias_var.set("Bias —")
            for tree in (self._err_q_tree, self._err_recent_tree):
                for iid in tree.get_children():
                    tree.delete(iid)
            self._err_chart_hint_var.set("No error samples for this strike.")
            return

        span = int((b or {}).get("ema_span") or 5)
        self._err_summary_vars["mean"].set(
            f"Mean (bias): {self._signed_rupee(summary.get('mean_error'))} ₹"
        )
        self._err_summary_vars["mae"].set(
            f"MAE: {self._fmt_num(summary.get('mae'))} ₹"
        )
        self._err_summary_vars["rmse"].set(
            f"RMSE: {self._fmt_num(summary.get('rmse'))} ₹"
        )
        opt = summary.get("pct_optimistic")
        pes = summary.get("pct_pessimistic")
        self._err_summary_vars["opt"].set(
            f"% optimistic: {self._fmt_num(opt, digits=1)}%"
            if opt is not None
            else "% optimistic: —"
        )
        self._err_summary_vars["pes"].set(
            f"% pessimistic: {self._fmt_num(pes, digits=1)}%"
            if pes is not None
            else "% pessimistic: —"
        )
        self._err_summary_vars["std"].set(
            f"Std: {self._fmt_num(summary.get('std_error'))} ₹"
        )
        self._err_summary_vars["latest"].set(
            f"Latest error: {self._signed_rupee(summary.get('latest_error'))} ₹"
        )
        self._err_summary_vars["ema"].set(
            f"Error EMA({span}): {self._signed_rupee(summary.get('latest_error_ema'))} ₹"
        )
        self._err_summary_vars["n"].set(f"n: {int(summary.get('n') or 0):,}")

        trends = summary.get("trends") if isinstance(summary.get("trends"), dict) else {}
        mag = trends.get("magnitude_label") or "Error —"
        bias = trends.get("bias_label") or "Bias —"
        earlier_mae = trends.get("earlier_mae")
        later_mae = trends.get("later_mae")
        if earlier_mae is not None and later_mae is not None:
            mag = (
                f"{mag}  (MAE {self._fmt_num(earlier_mae)} → {self._fmt_num(later_mae)})"
            )
        earlier_mean = trends.get("earlier_mean")
        later_mean = trends.get("later_mean")
        if earlier_mean is not None and later_mean is not None:
            bias = (
                f"{bias}  "
                f"(mean {self._signed_rupee(earlier_mean)} → "
                f"{self._signed_rupee(later_mean)})"
            )
        self._err_trend_mag_var.set(mag)
        self._err_trend_bias_var.set(bias)

        for iid in self._err_q_tree.get_children():
            self._err_q_tree.delete(iid)
        q = summary.get("quantiles") if isinstance(summary.get("quantiles"), dict) else {}
        for i in range(1, 100):
            key = f"p{i:02d}"
            entry = q.get(key)
            if isinstance(entry, dict):
                err_v = entry.get("error")
                samples = entry.get("samples", 0)
            else:
                err_v = entry
                samples = 0
            self._err_q_tree.insert(
                "",
                "end",
                values=(
                    f"P{i:02d}",
                    self._signed_rupee(err_v),
                    "" if samples is None else str(int(samples)),
                ),
            )

        for iid in self._err_recent_tree.get_children():
            self._err_recent_tree.delete(iid)
        rows = (b or {}).get("error_recent_rows") or []
        for row in rows:
            self._err_recent_tree.insert(
                "",
                "end",
                values=(
                    self._fmt_ts(row.get("timestamp")),
                    self._fmt_num(row.get("current_ltp")),
                    self._fmt_num(row.get("future_actual", row.get("actual"))),
                    self._fmt_num(row.get("future_pred", row.get("predicted"))),
                    self._signed_rupee(row.get("actual_delta")),
                    self._signed_rupee(row.get("pred_delta")),
                    self._signed_rupee(row.get("error")),
                    self._signed_rupee(row.get("error_ema")),
                    self._fmt_num(row.get("confidence"), digits=3),
                    self._fmt_num(row.get("confidence_ema"), digits=3),
                ),
                tags=(str(int(row.get("index") or 0)),),
            )

        ds = (b or {}).get("error_downsampled") or {}
        n_src = int(ds.get("n_source") or summary.get("n") or 0)
        n_ds = int(ds.get("n_downsampled") or 0)
        if self._show_err_chart_var.get():
            self._err_chart_hint_var.set(
                f"Downsampled {n_src:,} → {n_ds:,} pts (bucket mean)."
            )
        else:
            self._err_chart_hint_var.set(
                f"{n_src:,} samples — enable chart for downsampled view (~{n_ds} pts)."
            )

    def _selected_day(self) -> str | None:
        day = str(self._day_var.get() or "").strip()
        if not day or day == "All days":
            return None
        return day

    def _ema_span(self) -> int:
        try:
            return 10 if int(self._ema_var.get()) == 10 else 5
        except (TypeError, ValueError):
            return 5

    def _on_day_changed(self) -> None:
        self._load_strike_options_async(then_charts=True)

    def _reload_filters_then_charts(self) -> None:
        if not self._lab_db_path:
            self._status_var.set("No Research Lab open")
            self._day_combo["values"] = ("All days",)
            self._day_var.set("All days")
            self._strike_combo["values"] = ()
            self._strike_var.set("")
            self._last_bundle = None
            self._cursor_index = None
            self._cursor_ts = None
            self._update_detail_strip()
            self._refresh_error_stats_ui()
            self._refresh_ltp_stats_ui()
            for k in self._canvases:
                self._redraw_one(k)
            return
        self._load_day_options_async()

    def _load_day_options_async(self) -> None:
        lab = self._lab_db_path
        gen = self._load_gen = self._load_gen + 1
        self._status_var.set("Loading days…")

        def work() -> None:
            days: list[str] = []
            err = ""
            try:
                from chain_replay_ml.model_lab.store import ModelLabStore

                with ModelLabStore(lab) as store:
                    counts = store.prediction_row_counts_by_day()
                days = sorted(counts.keys())
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if gen != self._load_gen:
                    return
                if err:
                    self._status_var.set(f"Days: {err}")
                self._day_options = days
                values = ["All days", *days]
                self._day_combo["values"] = values
                cur = str(self._day_var.get() or "")
                if cur not in values:
                    pref = None
                    try:
                        pref = self._get_day_filter()
                    except Exception:
                        pref = None
                    if pref and str(pref) in values:
                        self._day_var.set(str(pref))
                    else:
                        self._day_var.set("All days")
                self._load_strike_options_async(then_charts=True)

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _load_strike_options_async(self, *, then_charts: bool = False) -> None:
        lab = self._lab_db_path
        day = self._selected_day()
        gen = self._load_gen = self._load_gen + 1
        self._status_var.set("Loading strikes…")

        def work() -> None:
            labels: list[str] = []
            err = ""
            try:
                from chain_replay_ml.model_lab.store import ModelLabStore
                from chain_replay_ml.model_lab.strike_prediction_dashboard import (
                    distinct_strikes,
                )

                where = ""
                args: list[Any] = []
                if day:
                    where = '"trading_day" = ?'
                    args = [day]
                with ModelLabStore(lab) as store:
                    cols, rows = store.query_predictions(
                        columns=["strike", "option_type"],
                        where_sql=where,
                        where_args=args,
                        order_by="strike ASC",
                        limit=50000,
                        data_dir=self._data_dir(),
                    )
                idx = {c: i for i, c in enumerate(cols)}
                strikes = [r[idx["strike"]] for r in rows] if "strike" in idx else []
                ots = (
                    [r[idx["option_type"]] for r in rows]
                    if "option_type" in idx
                    else None
                )
                labels = distinct_strikes(strikes, ots)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if gen != self._load_gen:
                    return
                if err:
                    self._status_var.set(f"Strikes: {err}")
                self._strike_labels = labels
                self._strike_combo["values"] = labels
                cur = str(self._strike_var.get() or "")
                if labels and cur not in labels:
                    self._strike_var.set(labels[len(labels) // 2])
                elif not labels:
                    self._strike_var.set("")
                if then_charts:
                    self._load_charts_async()
                else:
                    self._status_var.set(
                        f"{len(labels)} strike(s)" + (f" · {day}" if day else "")
                    )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _load_charts_async(self) -> None:
        lab = self._lab_db_path
        strike_label = str(self._strike_var.get() or "").strip()
        day = self._selected_day()
        ema_span = self._ema_span()
        model_name = self._model_name
        data_dir = self._data_dir()
        if not lab:
            self._status_var.set("No Research Lab open")
            return
        if not strike_label:
            self._status_var.set("No strikes in filtered prediction data")
            self._last_bundle = None
            self._cursor_index = None
            self._cursor_ts = None
            self._update_detail_strip()
            self._refresh_error_stats_ui()
            self._refresh_ltp_stats_ui()
            for k in self._canvases:
                self._redraw_one(k)
            return

        gen = self._load_gen = self._load_gen + 1
        self._loading = True
        self._status_var.set(f"Loading {strike_label}…")

        def work() -> None:
            bundle: dict[str, Any] | None = None
            err = ""
            try:
                from chain_replay_ml.model_lab.store import ModelLabStore
                from chain_replay_ml.model_lab.strike_prediction_dashboard import (
                    DASHBOARD_QUERY_COLUMNS,
                    build_strike_chart_bundle,
                    parse_strike_label,
                    resolve_confidence_column,
                )
                from chain_replay_ml.strategy_simulator import (
                    get_strategy_run_trades,
                    list_strategy_runs,
                )

                strike, opt = parse_strike_label(strike_label)
                clauses: list[str] = []
                args: list[Any] = []
                if day:
                    clauses.append('"trading_day" = ?')
                    args.append(day)
                if strike is not None:
                    clauses.append('"strike" = ?')
                    args.append(strike)
                if opt:
                    clauses.append('UPPER("option_type") = ?')
                    args.append(opt)

                with ModelLabStore(lab) as store:
                    available = set(store.list_prediction_columns())
                    conf_col = resolve_confidence_column(available)
                    want = [c for c in DASHBOARD_QUERY_COLUMNS if c in available]
                    for extra in (
                        "confidence_target_hit_pred",
                        "confidence_rr_1_1_pred",
                        "confidence_trade_winner_pred",
                    ):
                        if extra in available and extra not in want:
                            want.append(extra)
                    if conf_col and conf_col not in want and conf_col in available:
                        want.append(conf_col)
                    cols, rows = store.query_predictions(
                        columns=want,
                        where_sql=" AND ".join(clauses),
                        where_args=args,
                        order_by="timestamp ASC",
                        limit=20000,
                        data_dir=data_dir,
                    )

                trades: list[dict[str, Any]] = []
                try:
                    runs = list_strategy_runs(data_dir, limit=80)
                    for run in runs:
                        if model_name and str(run.get("model_id") or "") != model_name:
                            continue
                        rid = str(run.get("strategy_run_id") or "")
                        if not rid:
                            continue
                        detail = get_strategy_run_trades(data_dir, rid, limit=5000)
                        if detail.get("ok"):
                            trades.extend(list(detail.get("trades") or []))
                        if len(trades) >= 8000:
                            break
                except Exception:
                    trades = []

                bundle = build_strike_chart_bundle(
                    cols,
                    rows,
                    ema_span=ema_span,
                    confidence_column=conf_col,
                    trades=trades,
                    strike=strike,
                    option_type=opt,
                    trading_day=day,
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if gen != self._load_gen:
                    return
                self._loading = False
                if err:
                    self._status_var.set(f"Error: {err}")
                    self._last_bundle = None
                    self._cursor_index = None
                    self._cursor_ts = None
                else:
                    self._last_bundle = bundle
                    self._restore_cursor_from_bundle()
                    n = int((bundle or {}).get("row_count") or 0)
                    conf = (bundle or {}).get("confidence_column") or "—"
                    self._status_var.set(
                        f"{strike_label} · {n:,} rows · conf={conf} · EMA({ema_span})"
                    )
                self._update_detail_strip()
                self._refresh_error_stats_ui()
                self._refresh_ltp_stats_ui()
                for k in self._canvases:
                    self._redraw_one(k)

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    # ── Crosshair ─────────────────────────────────────────────────────────

    def _restore_cursor_from_bundle(self) -> None:
        b = self._last_bundle
        if not b:
            self._cursor_index = None
            self._cursor_ts = None
            return
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            index_for_timestamp,
        )

        n = int(b.get("row_count") or 0)
        if n <= 0:
            self._cursor_index = None
            self._cursor_ts = None
            return
        if self._cursor_ts is not None:
            idx = index_for_timestamp(b.get("timestamps") or [], self._cursor_ts)
            if idx is not None:
                self._cursor_index = idx
                return
        if self._cursor_index is not None and 0 <= self._cursor_index < n:
            ts = (b.get("timestamps") or [None] * n)[self._cursor_index]
            self._cursor_ts = ts
            return
        self._cursor_index = None
        self._cursor_ts = None

    def _on_tab_changed(self, _event: Any = None) -> None:
        self._update_detail_strip()
        # Redraw visible tab so crosshair appears at shared index.
        try:
            tab_id = self._chart_notebook.select()
            tab = self._chart_notebook.nametowidget(tab_id)
            for key, cv in self._canvases.items():
                if cv.master is tab:
                    self._redraw_one(key)
                    break
        except Exception:
            pass

    def _on_chart_motion(self, key: str, event: Any) -> None:
        b = self._last_bundle
        if not b or not b.get("row_count"):
            return
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            series_index_from_x,
        )

        layout = self._chart_layouts.get(key) or getattr(
            self._canvases.get(key), "_chart_layout", None
        )
        if not layout:
            return
        idx = series_index_from_x(
            float(event.x),
            pad=float(layout.get("pad") or 28),
            inner_w=float(layout.get("inner_w") or 1),
            n_points=int(layout.get("n_points") or b.get("row_count") or 0),
        )
        if idx is None:
            return
        # Error chart is downsampled — map bucket index back to source row.
        if key == "err":
            src = layout.get("source_indices") or self._err_source_indices
            if src and 0 <= idx < len(src):
                idx = int(src[idx])
        if idx == self._cursor_index:
            return
        timestamps = b.get("timestamps") or []
        self._cursor_index = idx
        self._cursor_ts = timestamps[idx] if idx < len(timestamps) else None
        self._update_detail_strip()
        self._redraw_one(key)

    def _fmt_num(self, v: Any, *, digits: int = 2) -> str:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "—"
        if fv != fv:  # NaN
            return "—"
        return f"{fv:.{digits}f}"

    def _fmt_ts(self, ts: Any) -> str:
        if ts is None:
            return "—"
        try:
            return f"{float(ts):.3f}"
        except (TypeError, ValueError):
            return str(ts)

    def _update_detail_strip(self) -> None:
        b = self._last_bundle
        if not b or self._cursor_index is None:
            self._detail_var.set("Hover a chart for linked crosshair details.")
            return
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            crosshair_detail_at_index,
        )

        d = crosshair_detail_at_index(b, int(self._cursor_index))
        span = int(b.get("ema_span") or 5)
        self._detail_var.set(
            " · ".join(
                [
                    f"t={self._fmt_ts(d.get('timestamp'))}",
                    f"Actual={self._fmt_num(d.get('actual_ltp'))}",
                    f"Pred={self._fmt_num(d.get('predicted_ltp'))}",
                    f"PredEMA({span})={self._fmt_num(d.get('predicted_ema'))}",
                    f"Conf={self._fmt_num(d.get('confidence'), digits=3)}",
                    f"ConfEMA5={self._fmt_num(d.get('confidence_ema_5'), digits=3)}",
                    f"ConfEMA10={self._fmt_num(d.get('confidence_ema_10'), digits=3)}",
                    f"Err={self._fmt_num(d.get('error'))}",
                    f"Gap={self._fmt_num(d.get('gap'))}",
                    f"Slope={self._fmt_num(d.get('regression_slope'), digits=4)}",
                ]
            )
        )

    # ── Draw ──────────────────────────────────────────────────────────────

    def _finite_list(self, vals: list[Any] | None) -> list[float]:
        out: list[float] = []
        for v in vals or []:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                out.append(float("nan"))
                continue
            out.append(fv)
        return out

    def _redraw_one(self, key: str) -> None:
        cv = self._canvases.get(key)
        if cv is None:
            return
        b = self._last_bundle
        cursor = self._cursor_index
        if not b or not b.get("row_count"):
            layout = draw_line_chart(
                cv,
                [],
                title="",
                fill=CHART_FILL,
                empty_message="No prediction data",
            )
            self._chart_layouts[key] = layout
            return

        span = int(b.get("ema_span") or 5)
        if key == "ltp":
            layout = draw_line_chart(
                cv,
                series=[
                    ("Actual", self._finite_list(b.get("actual_ltp")), COL_ACTUAL),
                    ("Predicted", self._finite_list(b.get("predicted_ltp")), COL_PREDICTED),
                    (
                        f"Pred EMA({span})",
                        self._finite_list(b.get("predicted_ema")),
                        COL_PRED_EMA,
                    ),
                ],
                fill=CHART_FILL,
                include_zero=False,
                cursor_index=cursor,
            )
        elif key == "conf":
            layout = draw_line_chart(
                cv,
                series=[
                    ("Confidence", self._finite_list(b.get("confidence")), COL_CONF),
                    ("EMA(5)", self._finite_list(b.get("confidence_ema_5")), COL_CONF_EMA),
                    (
                        "EMA(10)",
                        self._finite_list(b.get("confidence_ema_10")),
                        COL_CONF_EMA_10,
                    ),
                ],
                fill=CHART_FILL,
                cursor_index=cursor,
            )
        elif key == "err":
            if not self._show_err_chart_var.get():
                # Canvas not visible — skip expensive redraw.
                self._chart_layouts[key] = self._chart_layouts.get(key) or {}
                return
            err_ds = b.get("error_downsampled") or {}
            ema_ds = b.get("error_ema_downsampled") or {}
            src_idxs = list(err_ds.get("source_indices") or [])
            self._err_source_indices = src_idxs
            # Map shared full-series cursor onto downsampled x-index.
            ds_cursor: int | None = None
            if cursor is not None and src_idxs:
                best_i = 0
                best_d = abs(int(src_idxs[0]) - int(cursor))
                for i, si in enumerate(src_idxs):
                    d = abs(int(si) - int(cursor))
                    if d < best_d:
                        best_d = d
                        best_i = i
                ds_cursor = best_i
            layout = draw_line_chart(
                cv,
                series=[
                    (
                        "Error (ds)",
                        self._finite_list(err_ds.get("values")),
                        COL_ERROR,
                    ),
                    (
                        f"EMA({span}) (ds)",
                        self._finite_list(ema_ds.get("values")),
                        COL_ERROR_EMA,
                    ),
                ],
                fill=CHART_FILL,
                cursor_index=ds_cursor,
            )
            layout["source_indices"] = src_idxs
            n_src = int(err_ds.get("n_source") or 0)
            n_ds = int(err_ds.get("n_downsampled") or 0)
            self._err_chart_hint_var.set(
                f"Downsampled {n_src:,} → {n_ds:,} pts (bucket mean)."
            )
        elif key == "gap":
            layout = draw_line_chart(
                cv,
                series=[
                    (
                        "Gap (PredEMA−Actual)",
                        self._finite_list(b.get("gap")),
                        COL_GAP,
                    ),
                ],
                fill=CHART_FILL,
                cursor_index=cursor,
            )
        elif key == "regr":
            # Slope on right axis — magnitude differs from LTP scale.
            layout = draw_line_chart(
                cv,
                series=[
                    (
                        "Regression",
                        self._finite_list(b.get("regression")),
                        COL_REGR,
                    ),
                    (
                        f"Reg EMA({span})",
                        self._finite_list(b.get("regression_ema")),
                        COL_REGR_EMA,
                    ),
                ],
                secondary_series=[
                    (
                        "EMA Slope",
                        self._finite_list(b.get("regression_ema_slope")),
                        COL_REGR_SLOPE,
                    ),
                ],
                fill=CHART_FILL,
                include_zero=False,
                cursor_index=cursor,
            )
        elif key == "conf_pred":
            # Dual y-axis: Regression EMA (left) + Confidence EMA (right).
            layout = draw_line_chart(
                cv,
                series=[
                    (
                        f"Reg EMA({span})",
                        self._finite_list(b.get("regression_ema")),
                        COL_REGR_EMA,
                    ),
                ],
                secondary_series=[
                    (
                        f"Conf EMA({span})",
                        self._finite_list(b.get("confidence_ema")),
                        COL_CONF_EMA,
                    ),
                ],
                fill=CHART_FILL,
                include_zero=False,
                cursor_index=cursor,
            )
        else:
            layout = draw_line_chart(
                cv, [], fill=CHART_FILL, empty_message="Unknown chart"
            )
        self._chart_layouts[key] = layout
