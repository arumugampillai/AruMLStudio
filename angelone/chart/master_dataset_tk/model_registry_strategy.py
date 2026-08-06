"""Model-scoped Strategy tab — Simulation + Leaderboard for one model."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, dual_spec_sections, fmt_rupee, fmt_rows

_LEADERBOARD_MODES = (
    ("best_composite", "Best Composite"),
    ("highest_profit", "Highest Profit"),
    ("highest_win_rate", "Highest Win Rate"),
    ("highest_profit_factor", "Highest Profit Factor"),
    ("lowest_drawdown", "Lowest Drawdown"),
)


def _fmt_num(v: Any, *, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_signed_pnl(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if n == 0:
        return "0"
    return f"{n:+.0f}"


def _pnl_tag(v: Any) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "pnl_flat"
    if n > 0:
        return "pnl_pos"
    if n < 0:
        return "pnl_neg"
    return "pnl_flat"


def _resolve_gross_net_profit(metrics: dict[str, Any]) -> tuple[Any, Any]:
    net = metrics.get("net_profit")
    if net is None:
        net = metrics.get("profit")
    gross = metrics.get("gross_pnl_total")
    if gross is None:
        fees = metrics.get("total_fees")
        if net is not None and fees is not None:
            gross = float(net) + float(fees)
    return gross, net


def _summary_metric_sections(metrics: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    n = int(metrics.get("trade_count") or 0)
    gross_profit, net_profit = _resolve_gross_net_profit(metrics)
    avg_trade = metrics.get("avg_trade_pnl")
    if avg_trade is None and n > 0 and net_profit is not None:
        avg_trade = float(net_profit) / n
    expectancy = metrics.get("expectancy")
    if expectancy is None and n > 0 and net_profit is not None:
        expectancy = float(net_profit) / n
    win = metrics.get("win_rate_pct")
    win_label = f"{_fmt_num(win)}%" if win is not None else "—"
    pf = metrics.get("profit_factor")
    wins = metrics.get("wins")
    losses = metrics.get("losses")
    fees = metrics.get("total_fees")
    performance = [
        ("Gross Profit", fmt_rupee(gross_profit), gross_profit),
        ("Net Profit", fmt_rupee(net_profit), net_profit),
        ("Trades", fmt_rows(n) if n else "0"),
        ("Win Rate", win_label),
        ("Wins", fmt_rows(int(wins)) if wins is not None else "—"),
        ("Losses", fmt_rows(int(losses)) if losses is not None else "—"),
    ]
    risk = [
        ("Profit Factor", _fmt_num(pf, digits=2) if pf is not None else "—"),
        (
            "Account Equity Max DD",
            fmt_rupee(metrics.get("account_equity_max_drawdown", metrics.get("max_drawdown"))),
        ),
        (
            "Max Portfolio DD (Open)",
            fmt_rupee(metrics.get("max_portfolio_drawdown_open_risk")),
        ),
        (
            "Max Theoretical Risk",
            fmt_rupee(metrics.get("max_theoretical_portfolio_risk")),
        ),
        ("Avg Trade", fmt_rupee(avg_trade), avg_trade),
        ("Expectancy", fmt_rupee(expectancy), expectancy),
        ("Total Fees", fmt_rupee(fees) if fees is not None else "—"),
    ]
    return performance, risk


class ModelRegistryStrategyPanel(ttk.Frame):
    """Strategy simulations and leaderboard filtered to the selected model."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_fold_research: Callable[[str, str, str | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_fold_research = on_open_fold_research
        self._model_name = ""
        self._pred_id_map: dict[str, str] = {}
        self._fold_id_map: dict[str, str | None] = {}
        self._strat_id_map: dict[str, str] = {}
        self._current_run: dict[str, Any] | None = None
        self._fold_row_map: dict[str, dict[str, Any]] = {}
        self._status_var = tk.StringVar(value="")
        self._build_ui()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self._sim_tab = ttk.Frame(self._notebook, padding=4)
        self._lb_tab = ttk.Frame(self._notebook, padding=4)
        self._notebook.add(self._sim_tab, text="Simulation")
        self._notebook.add(self._lb_tab, text="Leaderboard")
        self._build_simulation_tab()
        self._build_leaderboard_tab()
        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(anchor="w", padx=8, pady=(0, 4))

    def _build_simulation_tab(self) -> None:
        outer = ttk.Panedwindow(self._sim_tab, orient=tk.VERTICAL)
        outer.pack(fill="both", expand=True)

        row1 = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        outer.add(row1, weight=2)

        form = ttk.LabelFrame(row1, text="Run Simulation", padding=8)
        row1.add(form, weight=2)

        pred_row = ttk.Frame(form)
        pred_row.pack(fill="x", pady=2)
        ttk.Label(pred_row, text="Prediction Run", width=14).pack(side="left")
        self._pred_var = tk.StringVar()
        self._pred_combo = ttk.Combobox(pred_row, textvariable=self._pred_var, state="readonly")
        self._pred_combo.pack(side="left", fill="x", expand=True, padx=4)
        self._pred_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_prediction_selected())

        strat_row = ttk.Frame(form)
        strat_row.pack(fill="x", pady=2)
        ttk.Label(strat_row, text="Strategy Version", width=14).pack(side="left")
        self._strat_var = tk.StringVar()
        self._strat_combo = ttk.Combobox(strat_row, textvariable=self._strat_var, state="readonly")
        self._strat_combo.pack(side="left", fill="x", expand=True, padx=4)

        fold_row = ttk.Frame(form)
        fold_row.pack(fill="x", pady=2)
        ttk.Label(fold_row, text="Fold", width=14).pack(side="left")
        self._fold_var = tk.StringVar(value="(all folds)")
        self._fold_combo = ttk.Combobox(fold_row, textvariable=self._fold_var, state="readonly")
        self._fold_combo.pack(side="left", fill="x", expand=True, padx=4)

        btns = ttk.Frame(form)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Run Simulation", command=self._run_simulation).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Refresh", command=self.refresh).pack(side="left")

        runs_fr = ttk.LabelFrame(row1, text="Strategy Runs", padding=4)
        row1.add(runs_fr, weight=3)
        cols = ("run", "prediction", "strategy", "trades", "profit", "win", "pf", "status")
        self._runs_tree = ttk.Treeview(runs_fr, columns=cols, show="headings", height=8)
        for c, w, label in (
            ("run", 90, "Strategy Run"),
            ("prediction", 90, "Prediction Run"),
            ("strategy", 110, "Strategy"),
            ("trades", 50, "Trades"),
            ("profit", 65, "Profit"),
            ("win", 50, "Win %"),
            ("pf", 45, "PF"),
            ("status", 65, "Status"),
        ):
            self._runs_tree.heading(c, text=label)
            self._runs_tree.column(c, width=w)
        self._runs_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(runs_fr, orient="vertical", command=self._runs_tree.yview)
        sb.pack(side="right", fill="y")
        self._runs_tree.configure(yscrollcommand=sb.set)
        self._runs_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._runs_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._runs_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self._runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_run_detail())

        detail_nb = ttk.Notebook(outer)
        outer.add(detail_nb, weight=3)
        summary_tab = ttk.Frame(detail_nb, padding=4)
        trades_tab = ttk.Frame(detail_nb, padding=4)
        fold_tab = ttk.Frame(detail_nb, padding=4)
        metrics_tab = ttk.Frame(detail_nb, padding=4)
        detail_nb.add(summary_tab, text="Summary")
        detail_nb.add(trades_tab, text="Trades")
        detail_nb.add(fold_tab, text="Fold wise P&L")
        detail_nb.add(metrics_tab, text="Metrics")

        self._summary_host = ttk.Frame(summary_tab)
        self._summary_host.pack(fill="both", expand=True)

        self._trades_tree = ttk.Treeview(
            trades_tab,
            columns=("day", "token", "entry", "exit", "pnl", "ret", "hold", "reason"),
            show="headings",
        )
        for c, w, label in (
            ("day", 88, "Day"),
            ("token", 65, "Token"),
            ("entry", 65, "Entry"),
            ("exit", 65, "Exit"),
            ("pnl", 65, "Net PnL"),
            ("ret", 55, "Ret %"),
            ("hold", 50, "Hold s"),
            ("reason", 65, "Exit"),
        ):
            self._trades_tree.heading(c, text=label)
            self._trades_tree.column(c, width=w)
        self._trades_tree.pack(side="left", fill="both", expand=True)
        tsb = ttk.Scrollbar(trades_tab, orient="vertical", command=self._trades_tree.yview)
        tsb.pack(side="right", fill="y")
        self._trades_tree.configure(yscrollcommand=tsb.set)
        self._trades_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._trades_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._trades_tree.tag_configure("pnl_flat", foreground=COL_MUTED)

        fold_cols = ("fold", "fold_id", "trades", "profit", "win", "pf", "max_dd", "wins", "losses")
        self._fold_pnl_tree = ttk.Treeview(fold_tab, columns=fold_cols, show="headings")
        for c, w, label in (
            ("fold", 44, "Fold"),
            ("fold_id", 72, "Fold ID"),
            ("trades", 50, "Trades"),
            ("profit", 72, "Net P&L"),
            ("win", 52, "Win %"),
            ("pf", 48, "PF"),
            ("max_dd", 68, "Max DD"),
            ("wins", 44, "Wins"),
            ("losses", 50, "Losses"),
        ):
            self._fold_pnl_tree.heading(c, text=label)
            self._fold_pnl_tree.column(c, width=w)
        self._fold_pnl_tree.pack(side="left", fill="both", expand=True)
        fsb = ttk.Scrollbar(fold_tab, orient="vertical", command=self._fold_pnl_tree.yview)
        fsb.pack(side="right", fill="y")
        self._fold_pnl_tree.configure(yscrollcommand=fsb.set)
        self._fold_pnl_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._fold_pnl_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._fold_pnl_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self._fold_pnl_tree.bind("<Double-1>", self._on_fold_double_click)
        ttk.Label(
            fold_tab,
            text="Double-click a fold to open Fold Research — trade timeline, equity curve, entry/exit reasons, prediction error, feature drift.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
            wraplength=720,
        ).pack(anchor="w", padx=6, pady=(4, 2))

        self._metrics_text = scrolledtext.ScrolledText(metrics_tab, font=("Consolas", 9))
        self._metrics_text.pack(fill="both", expand=True)

        def _set_sim_row1_ratio(_event: tk.Event | None = None) -> None:
            width = row1.winfo_width()
            if width > 1:
                row1.sashpos(0, int(width * 0.4))

        def _set_sim_outer_ratio(_event: tk.Event | None = None) -> None:
            height = outer.winfo_height()
            if height > 1:
                outer.sashpos(0, int(height * 0.42))

        row1.bind("<Configure>", _set_sim_row1_ratio)
        outer.bind("<Configure>", _set_sim_outer_ratio)

    def _build_leaderboard_tab(self) -> None:
        bar = ttk.Frame(self._lb_tab)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Rank by").pack(side="left")
        self._lb_mode_var = tk.StringVar(value="best_composite")
        mode_cb = ttk.Combobox(
            bar,
            textvariable=self._lb_mode_var,
            values=[m[0] for m in _LEADERBOARD_MODES],
            width=22,
            state="readonly",
        )
        mode_cb.pack(side="left", padx=6)
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._load_leaderboard())
        ttk.Button(bar, text="Refresh", command=self._load_leaderboard).pack(side="right")

        lb_cols = ("rank", "strategy", "profit", "win", "pf", "dd", "trades", "composite")
        self._lb_tree = ttk.Treeview(self._lb_tab, columns=lb_cols, show="headings", height=12)
        for c, w, label in (
            ("rank", 36, "#"),
            ("strategy", 130, "Strategy"),
            ("profit", 70, "Profit"),
            ("win", 55, "Win %"),
            ("pf", 50, "PF"),
            ("dd", 60, "Max DD"),
            ("trades", 55, "Trades"),
            ("composite", 70, "Composite"),
        ):
            self._lb_tree.heading(c, text=label)
            self._lb_tree.column(c, width=w)
        self._lb_tree.pack(fill="both", expand=True)
        self._lb_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._lb_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._lb_tree.tag_configure("pnl_flat", foreground=COL_MUTED)

    def load_for_model(self, model_name: str) -> None:
        self._model_name = str(model_name or "").strip()
        self._load_prediction_runs()
        self._load_strategies()
        self._load_simulation_runs()
        self._load_leaderboard()

    def refresh(self) -> None:
        if self._model_name:
            self.load_for_model(self._model_name)

    def _load_prediction_runs(self) -> None:
        from chain_replay_ml.prediction_runs import list_runs

        self._pred_id_map.clear()
        labels: list[str] = []
        if not self._model_name:
            self._pred_combo.configure(values=[])
            self._pred_var.set("")
            return
        try:
            runs = list_runs(self._data_dir(), self._model_name, limit=50)
        except Exception as exc:
            self._status_var.set(f"Prediction runs: {exc}")
            return
        for r in runs:
            rid = str(r.get("run_id") or "")
            short = f"{rid[:8]}…" if len(rid) > 8 else rid
            label = f"{short} · {r.get('fold_count', 0)} folds · {r.get('prediction_count', 0)} rows"
            labels.append(label)
            self._pred_id_map[label] = rid
        self._pred_combo.configure(values=labels)
        if labels:
            self._pred_var.set(labels[0])
            self._on_prediction_selected()
        else:
            self._pred_var.set("")
            self._fold_combo.configure(values=["(all folds)"])
            self._fold_var.set("(all folds)")

    def _load_strategies(self) -> None:
        from chain_replay_ml.strategy_registry import get_strategy_detail, list_strategies

        self._strat_id_map.clear()
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
        self._strat_combo.configure(values=labels)
        if labels:
            self._strat_var.set(labels[0])

    def _on_prediction_selected(self) -> None:
        from chain_replay_ml.prediction_runs import get_run_detail

        rid = self._pred_id_map.get(self._pred_var.get())
        if not rid:
            self._fold_combo.configure(values=["(all folds)"])
            self._fold_var.set("(all folds)")
            return
        try:
            detail = get_run_detail(self._data_dir(), rid)
        except Exception:
            return
        folds = ["(all folds)"]
        self._fold_id_map = {"(all folds)": None}
        for f in (detail or {}).get("folds") or []:
            fn = f.get("fold_number")
            fid = str(f.get("fold_id") or "")
            lbl = f"Fold {fn} ({fid[:8]}…)"
            folds.append(lbl)
            self._fold_id_map[lbl] = fid
        self._fold_combo.configure(values=folds)
        self._fold_var.set("(all folds)")

    def _model_prediction_run_ids(self) -> set[str]:
        return set(self._pred_id_map.values())

    def _load_simulation_runs(self) -> None:
        from chain_replay_ml.strategy_simulator import list_strategy_runs

        self._runs_tree.delete(*self._runs_tree.get_children())
        if not self._model_name:
            return
        pred_ids = self._model_prediction_run_ids()
        try:
            all_runs = list_strategy_runs(self._data_dir(), limit=200)
        except Exception as exc:
            self._status_var.set(f"Runs: {exc}")
            return
        shown = 0
        for r in all_runs:
            prid = str(r.get("prediction_run_id") or "")
            model_id = str(r.get("model_id") or "")
            model_match = model_id == self._model_name
            pred_match = bool(pred_ids) and prid in pred_ids
            if not model_match and not pred_match:
                continue
            sid = str(r.get("strategy_run_id") or "")
            m = r.get("metrics") or {}
            profit = m.get("profit")
            self._runs_tree.insert(
                "",
                "end",
                iid=sid,
                tags=(_pnl_tag(profit),),
                values=(
                    sid[:10] + "…" if len(sid) > 10 else sid,
                    prid[:10] + "…" if len(prid) > 10 else prid,
                    (r.get("strategy_id") or "")[:14],
                    r.get("trade_count"),
                    m.get("profit"),
                    m.get("win_rate_pct"),
                    m.get("profit_factor"),
                    r.get("status"),
                ),
            )
            shown += 1
        self._status_var.set(f"{shown} strategy run(s) for {self._model_name}")

    def _run_simulation(self) -> None:
        from chain_replay_ml.strategy_simulator import run_strategy_simulation

        pred_id = self._pred_id_map.get(self._pred_var.get())
        version_id = self._strat_id_map.get(self._strat_var.get())
        fold_id = self._fold_id_map.get(self._fold_var.get())
        if not pred_id or not version_id:
            messagebox.showinfo("Simulation", "Select a prediction run and strategy version for this model.")
            return
        try:
            detail = run_strategy_simulation(
                self._data_dir(),
                prediction_run_id=pred_id,
                strategy_version_id=version_id,
                fold_id=fold_id,
            )
            self._load_simulation_runs()
            run_id = detail["run"]["strategy_run_id"]
            if run_id in self._runs_tree.get_children():
                self._runs_tree.selection_set(run_id)
            self._load_run_detail()
            self._load_leaderboard()
            m = detail["run"].get("metrics") or {}
            messagebox.showinfo(
                "Simulation",
                f"Done — {detail['run'].get('trade_count')} trades\n"
                f"Profit: {m.get('profit')} | Win%: {m.get('win_rate_pct')}",
            )
        except Exception as exc:
            messagebox.showerror("Simulation", str(exc))

    def _load_run_detail(self) -> None:
        from chain_replay_ml.strategy_simulator import get_strategy_run_trades

        sel = self._runs_tree.selection()
        if not sel:
            return
        run_id = sel[0]
        try:
            doc = get_strategy_run_trades(self._data_dir(), run_id, limit=300)
        except Exception as exc:
            self._metrics_text.delete("1.0", "end")
            self._metrics_text.insert("end", str(exc))
            self._fold_pnl_tree.delete(*self._fold_pnl_tree.get_children())
            self._render_summary({})
            return
        if not doc.get("ok"):
            return
        self._current_run = doc.get("run") or {}
        metrics = self._current_run.get("metrics") or {}
        self._render_summary(metrics)
        self._metrics_text.delete("1.0", "end")
        self._metrics_text.insert("end", json.dumps(metrics, indent=2, default=str))
        self._render_fold_pnl(metrics)
        self._trades_tree.delete(*self._trades_tree.get_children())
        for t in doc.get("trades") or []:
            pnl = t.get("net_pnl")
            self._trades_tree.insert(
                "",
                "end",
                tags=(_pnl_tag(pnl),),
                values=(
                    t.get("trading_day"),
                    t.get("token"),
                    t.get("entry_price"),
                    t.get("exit_price"),
                    t.get("net_pnl"),
                    t.get("return_pct"),
                    t.get("holding_seconds"),
                    t.get("exit_reason"),
                ),
            )

    def _render_summary(self, metrics: dict[str, Any]) -> None:
        for w in self._summary_host.winfo_children():
            w.destroy()
        if not metrics:
            ttk.Label(self._summary_host, text="Select a strategy run to view summary.", foreground=COL_MUTED).pack(anchor="w")
            return
        left, right = _summary_metric_sections(metrics)
        dual_spec_sections(self._summary_host, left, right, label_width=26)

    def _render_fold_pnl(self, metrics: dict[str, Any]) -> None:
        self._fold_pnl_tree.delete(*self._fold_pnl_tree.get_children())
        self._fold_row_map.clear()
        rows = metrics.get("fold_metrics") or []
        if not isinstance(rows, list):
            return

        def _fold_sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
            fn = row.get("fold_number")
            if fn is not None:
                try:
                    return (0, int(fn))
                except (TypeError, ValueError):
                    pass
            return (1, str(row.get("fold_id") or ""))

        for row in sorted((r for r in rows if isinstance(r, dict)), key=_fold_sort_key):
            fid = str(row.get("fold_id") or "")
            fid_short = f"{fid[:8]}…" if len(fid) > 8 else (fid or "—")
            fn = row.get("fold_number")
            fold_label = f"Fold {fn}" if fn is not None else "Fold —"
            profit = row.get("profit")
            iid = fid or f"fold_{fn or len(self._fold_row_map)}"
            self._fold_row_map[iid] = row
            self._fold_pnl_tree.insert(
                "",
                "end",
                iid=iid,
                tags=(_pnl_tag(profit),),
                values=(
                    fold_label,
                    fid_short,
                    row.get("trade_count") if row.get("trade_count") is not None else "—",
                    _fmt_signed_pnl(profit),
                    _fmt_num(row.get("win_rate_pct")),
                    _fmt_num(row.get("profit_factor"), digits=4),
                    _fmt_num(row.get("max_drawdown")),
                    row.get("wins") if row.get("wins") is not None else "—",
                    row.get("losses") if row.get("losses") is not None else "—",
                ),
            )

    def _on_fold_double_click(self, _event: tk.Event) -> None:
        if not self._on_open_fold_research:
            return
        sel = self._fold_pnl_tree.selection()
        if not sel:
            item = self._fold_pnl_tree.identify_row(_event.y)
            if item:
                self._fold_pnl_tree.selection_set(item)
                sel = (item,)
        if not sel:
            return
        row = self._fold_row_map.get(sel[0]) or {}
        fold_id = str(row.get("fold_id") or "")
        if not fold_id:
            return
        run = self._current_run or {}
        pred_id = str(run.get("prediction_run_id") or "")
        strat_id = str(run.get("strategy_run_id") or "") or None
        if not pred_id:
            messagebox.showinfo("Fold Research", "No prediction run linked to this strategy run.")
            return
        self._on_open_fold_research(pred_id, fold_id, strat_id)

    def _load_leaderboard(self) -> None:
        from chain_replay_ml.research_lab import get_leaderboard

        self._lb_tree.delete(*self._lb_tree.get_children())
        if not self._model_name:
            return
        try:
            doc = get_leaderboard(
                self._data_dir(),
                mode=self._lb_mode_var.get(),
                filters={"model_id": self._model_name},
                limit=50,
            )
        except Exception as exc:
            self._status_var.set(f"Leaderboard: {exc}")
            return
        rows = doc.get("leaderboard") or []
        for r in rows:
            profit = r.get("profit")
            self._lb_tree.insert(
                "",
                "end",
                tags=(_pnl_tag(profit),),
                values=(
                    r.get("rank"),
                    r.get("strategy_name"),
                    r.get("profit"),
                    r.get("win_rate_pct"),
                    r.get("profit_factor"),
                    r.get("max_drawdown"),
                    r.get("trade_count"),
                    r.get("composite_score"),
                ),
            )
        if not rows:
            self._status_var.set(f"No strategy runs for {self._model_name} yet — run a simulation.")
