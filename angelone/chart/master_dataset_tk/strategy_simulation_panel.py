"""Strategy Simulation panel — run strategy on prediction rows (Phase 3 Tk)."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN


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


class StrategySimulationPanel(ttk.Frame, LazyLoadMixin):
    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._runs: list[dict[str, Any]] = []
        self._strategies: list[dict[str, Any]] = []
        self._prediction_runs: list[dict[str, Any]] = []
        self._status_var = tk.StringVar(value="")
        self._build_ui()
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh_all(lazy=True)

    def refresh_all(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_simulation_data,
                apply=self._apply_simulation_data,
                message="Loading simulation data…",
                status_var=self._status_var,
            )
            return
        self._apply_simulation_data(self._fetch_simulation_data())

    def _fetch_simulation_data(self) -> dict[str, Any]:
        from chain_replay_ml.prediction_runs import list_all_runs
        from chain_replay_ml.strategy_registry import list_strategies
        from chain_replay_ml.strategy_simulator import list_strategy_runs

        return {
            "prediction_runs": list_all_runs(self._data_dir(), limit=100),
            "strategies": list_strategies(self._data_dir()),
            "runs": list_strategy_runs(self._data_dir(), limit=100),
        }

    def _apply_simulation_data(self, bundle: dict[str, Any]) -> None:
        self._prediction_runs = list(bundle.get("prediction_runs") or [])
        self._strategies = list(bundle.get("strategies") or [])
        self._runs = list(bundle.get("runs") or [])
        self._apply_source_combos()
        self._apply_runs_tree()

    def _apply_source_combos(self) -> None:
        pred_labels = []
        self._pred_id_map = {}
        for r in self._prediction_runs:
            rid = str(r.get("run_id") or "")
            label = f"{rid[:8]}… — {r.get('model_id')} ({r.get('target')})"
            pred_labels.append(label)
            self._pred_id_map[label] = rid
        self._pred_combo.configure(values=pred_labels)
        if pred_labels:
            self._pred_combo.set(pred_labels[0])
            self._on_prediction_selected()

        self._strat_id_map = {}
        strat_labels = []
        for s in self._strategies:
            from chain_replay_ml.strategy_registry import get_strategy_detail

            detail = get_strategy_detail(self._data_dir(), s["strategy_id"])
            champ = (detail or {}).get("champion_version") or {}
            vid = str(champ.get("version_id") or "")
            label = f"{s.get('display_name')} {s.get('current_version_label')} ({vid[:8]}…)"
            strat_labels.append(label)
            self._strat_id_map[label] = vid
        self._strat_combo.configure(values=strat_labels)
        if strat_labels:
            self._strat_combo.set(strat_labels[0])

    def _apply_runs_tree(self) -> None:
        self.runs_tree.delete(*self.runs_tree.get_children())
        for r in self._runs:
            sid = str(r.get("strategy_run_id") or "")
            m = r.get("metrics") or {}
            profit = m.get("profit")
            self.runs_tree.insert(
                "",
                "end",
                iid=sid,
                tags=(_pnl_tag(profit),),
                values=(
                    sid[:10] + "…",
                    (r.get("prediction_run_id") or "")[:10] + "…",
                    (r.get("strategy_id") or "")[:14],
                    r.get("trade_count"),
                    m.get("profit"),
                    m.get("win_rate_pct"),
                    m.get("profit_factor"),
                    r.get("status"),
                ),
            )
        self._status_var.set(f"{len(self._runs)} strategy run(s)")

    def _build_ui(self) -> None:
        form = ttk.LabelFrame(self, text="Run Simulation", padding=8)
        form.pack(fill="x", padx=8, pady=8)

        row1 = ttk.Frame(form)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Prediction Run", width=14).pack(side="left")
        self._pred_var = tk.StringVar()
        self._pred_combo = ttk.Combobox(row1, textvariable=self._pred_var, width=48, state="readonly")
        self._pred_combo.pack(side="left", padx=4)

        row2 = ttk.Frame(form)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Strategy Version", width=14).pack(side="left")
        self._strat_var = tk.StringVar()
        self._strat_combo = ttk.Combobox(row2, textvariable=self._strat_var, width=48, state="readonly")
        self._strat_combo.pack(side="left", padx=4)

        row3 = ttk.Frame(form)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="Fold (optional)", width=14).pack(side="left")
        self._fold_var = tk.StringVar(value="(all folds)")
        self._fold_combo = ttk.Combobox(row3, textvariable=self._fold_var, width=48, state="readonly")
        self._fold_combo.pack(side="left", padx=4)

        btn_row = ttk.Frame(form)
        btn_row.pack(fill="x", pady=8)
        ttk.Button(btn_row, text="Run Simulation", command=self._run_simulation).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Refresh", command=lambda: self.refresh_all(lazy=True)).pack(side="left", padx=2)

        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        top = ttk.Frame(paned)
        paned.add(top, weight=2)
        cols = ("run", "prediction", "strategy", "trades", "profit", "win", "pf", "status")
        self.runs_tree = ttk.Treeview(top, columns=cols, show="headings", height=8)
        for c, w, label in (
            ("run", 100, "Strategy Run"),
            ("prediction", 100, "Prediction Run"),
            ("strategy", 120, "Strategy"),
            ("trades", 50, "Trades"),
            ("profit", 70, "Profit"),
            ("win", 50, "Win %"),
            ("pf", 50, "PF"),
            ("status", 70, "Status"),
        ):
            self.runs_tree.heading(c, text=label)
            self.runs_tree.column(c, width=w)
        self.runs_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(top, orient="vertical", command=self.runs_tree.yview)
        sb.pack(side="right", fill="y")
        self.runs_tree.configure(yscrollcommand=sb.set)
        self.runs_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self.runs_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self.runs_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self.runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._load_run_detail())

        bottom = ttk.Frame(paned)
        paned.add(bottom, weight=3)
        self._detail_notebook = ttk.Notebook(bottom)
        self._detail_notebook.pack(fill="both", expand=True)

        self._metrics_text = scrolledtext.ScrolledText(self._detail_notebook, height=8, font=("Consolas", 9))
        self._trades_tree = ttk.Treeview(
            self._detail_notebook,
            columns=("day", "token", "entry", "exit", "pnl", "ret", "hold", "reason"),
            show="headings",
            height=10,
        )
        for c, w, label in (
            ("day", 90, "Day"),
            ("token", 70, "Token"),
            ("entry", 70, "Entry"),
            ("exit", 70, "Exit"),
            ("pnl", 70, "Net PnL"),
            ("ret", 60, "Ret %"),
            ("hold", 50, "Hold s"),
            ("reason", 70, "Exit"),
        ):
            self._trades_tree.heading(c, text=label)
            self._trades_tree.column(c, width=w)
        self._trades_tree.tag_configure("pnl_pos", foreground=COL_OK)
        self._trades_tree.tag_configure("pnl_neg", foreground=COL_WARN)
        self._trades_tree.tag_configure("pnl_flat", foreground=COL_MUTED)
        self._detail_notebook.add(self._metrics_text, text="Metrics")
        self._detail_notebook.add(self._trades_tree, text="Trades")

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))
        self._pred_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_prediction_selected())

    def _load_sources(self) -> None:
        self.refresh_all(lazy=False)

    def _on_prediction_selected(self) -> None:
        from chain_replay_ml.prediction_runs import get_run_detail

        label = self._pred_var.get()
        rid = getattr(self, "_pred_id_map", {}).get(label)
        if not rid:
            self._fold_combo.configure(values=["(all folds)"])
            self._fold_var.set("(all folds)")
            return
        try:
            detail = get_run_detail(self._data_dir(), rid)
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
        except Exception:
            pass

    def refresh_runs(self, *, lazy: bool = False) -> None:
        if lazy:
            self.refresh_all(lazy=True)
            return
        from chain_replay_ml.strategy_simulator import list_strategy_runs

        try:
            self._runs = list_strategy_runs(self._data_dir(), limit=100)
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_runs_tree()

    def _run_simulation(self) -> None:
        from chain_replay_ml.strategy_simulator import run_strategy_simulation

        pred_label = self._pred_var.get()
        strat_label = self._strat_var.get()
        pred_id = getattr(self, "_pred_id_map", {}).get(pred_label)
        version_id = getattr(self, "_strat_id_map", {}).get(strat_label)
        fold_id = getattr(self, "_fold_id_map", {}).get(self._fold_var.get())
        if not pred_id or not version_id:
            messagebox.showinfo("Simulation", "Select prediction run and strategy version.")
            return
        try:
            detail = run_strategy_simulation(
                self._data_dir(),
                prediction_run_id=pred_id,
                strategy_version_id=version_id,
                fold_id=fold_id,
            )
            self.refresh_runs()
            run_id = detail["run"]["strategy_run_id"]
            if run_id in self.runs_tree.get_children():
                self.runs_tree.selection_set(run_id)
            self._load_run_detail()
            m = detail["run"].get("metrics") or {}
            messagebox.showinfo(
                "Simulation",
                f"Done — {detail['run'].get('trade_count')} trades\n"
                f"Profit: {m.get('profit')} | Win%: {m.get('win_rate_pct')} | PF: {m.get('profit_factor')}",
            )
        except Exception as exc:
            messagebox.showerror("Simulation", str(exc))

    def _load_run_detail(self) -> None:
        from chain_replay_ml.strategy_simulator import get_strategy_run_trades

        sel = self.runs_tree.selection()
        if not sel:
            return
        run_id = sel[0]
        try:
            doc = get_strategy_run_trades(self._data_dir(), run_id, limit=500)
        except Exception as exc:
            self._metrics_text.delete("1.0", "end")
            self._metrics_text.insert("end", str(exc))
            return
        if not doc.get("ok"):
            return
        run = doc.get("run") or {}
        metrics = run.get("metrics") or {}
        self._metrics_text.delete("1.0", "end")
        self._metrics_text.insert("end", json.dumps(metrics, indent=2, default=str))

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

    def prefill(self, *, prediction_run_id: str | None = None, strategy_version_id: str | None = None) -> None:
        self._load_sources()
        if prediction_run_id:
            for label, rid in getattr(self, "_pred_id_map", {}).items():
                if rid == prediction_run_id:
                    self._pred_var.set(label)
                    self._on_prediction_selected()
                    break
        if strategy_version_id:
            for label, vid in getattr(self, "_strat_id_map", {}).items():
                if vid == strategy_version_id:
                    self._strat_var.set(label)
                    break
