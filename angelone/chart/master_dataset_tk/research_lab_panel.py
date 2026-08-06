"""Research Lab panel — matrix, leaderboard, compare, sessions (Phase 4 Tk)."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin

LEADERBOARD_MODES = [
    ("best_composite", "Best Composite"),
    ("highest_profit", "Highest Profit"),
    ("highest_win_rate", "Highest Win Rate"),
    ("highest_profit_factor", "Highest Profit Factor"),
    ("lowest_drawdown", "Lowest Drawdown"),
    ("most_stable", "Most Stable"),
    ("most_trades", "Most Trades"),
]


class ResearchLabPanel(ttk.Frame, LazyLoadMixin):
    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._matrix_rows: list[dict[str, Any]] = []
        self._status_var = tk.StringVar(value="")
        self._mode_var = tk.StringVar(value="best_composite")
        self._build_ui()
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh_all(lazy=True)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Refresh", command=lambda: self.refresh_all(lazy=True)).pack(side="left", padx=2)
        ttk.Label(toolbar, text="Leaderboard").pack(side="left", padx=(12, 4))
        mode_cb = ttk.Combobox(
            toolbar,
            textvariable=self._mode_var,
            values=[m[0] for m in LEADERBOARD_MODES],
            width=22,
            state="readonly",
        )
        mode_cb.pack(side="left", padx=4)
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._load_leaderboard())
        ttk.Button(toolbar, text="New Session", command=self._create_session).pack(side="right", padx=2)

        self._summary_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._summary_var, foreground="#58a6ff").pack(anchor="w", padx=10)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self._tab_matrix = ttk.Frame(self._notebook)
        self._tab_leaderboard = ttk.Frame(self._notebook)
        self._tab_grid = ttk.Frame(self._notebook)
        self._tab_compare = ttk.Frame(self._notebook)
        self._tab_sessions = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_matrix, text="Matrix")
        self._notebook.add(self._tab_leaderboard, text="Leaderboard")
        self._notebook.add(self._tab_grid, text="Model × Strategy")
        self._notebook.add(self._tab_compare, text="Compare")
        self._notebook.add(self._tab_sessions, text="Sessions")

        cols = (
            "model", "strategy", "profit", "win", "pf", "dd", "trades", "composite", "pred_mae",
        )
        self.matrix_tree = ttk.Treeview(self._tab_matrix, columns=cols, show="headings", height=12)
        for c, w, label in (
            ("model", 140, "Model"),
            ("strategy", 120, "Strategy"),
            ("profit", 70, "Profit"),
            ("win", 55, "Win %"),
            ("pf", 50, "PF"),
            ("dd", 60, "Max DD"),
            ("trades", 50, "Trades"),
            ("composite", 70, "Composite"),
            ("pred_mae", 60, "Pred MAE"),
        ):
            self.matrix_tree.heading(c, text=label)
            self.matrix_tree.column(c, width=w)
        self.matrix_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.matrix_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_matrix_select())

        lb_cols = ("rank", "model", "strategy", "profit", "win", "pf", "dd", "composite")
        self.leaderboard_tree = ttk.Treeview(self._tab_leaderboard, columns=lb_cols, show="headings", height=14)
        for c, w, label in (
            ("rank", 40, "#"),
            ("model", 140, "Model"),
            ("strategy", 120, "Strategy"),
            ("profit", 70, "Profit"),
            ("win", 55, "Win %"),
            ("pf", 50, "PF"),
            ("dd", 60, "Max DD"),
            ("composite", 70, "Composite"),
        ):
            self.leaderboard_tree.heading(c, text=label)
            self.leaderboard_tree.column(c, width=w)
        self.leaderboard_tree.pack(fill="both", expand=True, padx=4, pady=4)

        self._grid_text = scrolledtext.ScrolledText(self._tab_grid, height=16, font=("Consolas", 9))
        self._grid_text.pack(fill="both", expand=True, padx=4, pady=4)

        cmp_row = ttk.Frame(self._tab_compare, padding=4)
        cmp_row.pack(fill="x")
        self._cmp_a = tk.StringVar()
        self._cmp_b = tk.StringVar()
        ttk.Label(cmp_row, text="Run A").pack(side="left")
        ttk.Entry(cmp_row, textvariable=self._cmp_a, width=30).pack(side="left", padx=4)
        ttk.Label(cmp_row, text="Run B").pack(side="left")
        ttk.Entry(cmp_row, textvariable=self._cmp_b, width=30).pack(side="left", padx=4)
        ttk.Button(cmp_row, text="Compare", command=self._compare_runs).pack(side="left", padx=4)
        self._compare_text = scrolledtext.ScrolledText(self._tab_compare, height=14, font=("Consolas", 9))
        self._compare_text.pack(fill="both", expand=True, padx=4, pady=4)

        sess_row = ttk.Frame(self._tab_sessions, padding=4)
        sess_row.pack(fill="x")
        ttk.Button(sess_row, text="Refresh Sessions", command=self._load_sessions).pack(side="left", padx=2)
        self.sessions_tree = ttk.Treeview(
            self._tab_sessions,
            columns=("title", "runs", "preds", "updated"),
            show="headings",
            height=8,
        )
        for c, w, label in (
            ("title", 200, "Title"),
            ("runs", 60, "Runs"),
            ("preds", 60, "Pred Runs"),
            ("updated", 140, "Updated"),
        ):
            self.sessions_tree.heading(c, text=label)
            self.sessions_tree.column(c, width=w)
        self.sessions_tree.pack(fill="both", expand=True, padx=4, pady=4)

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))

    def refresh_all(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_lab_data,
                apply=self._apply_lab_data,
                message="Loading research lab…",
                status_var=self._status_var,
            )
            return
        self._apply_lab_data(self._fetch_lab_data())

    def _fetch_lab_data(self) -> dict[str, Any]:
        from chain_replay_ml.research_lab import (
            build_research_matrix,
            get_leaderboard,
            get_matrix,
            get_summary,
            list_research_sessions,
        )

        mode = self._mode_var.get()
        return {
            "summary": get_summary(self._data_dir()),
            "matrix": build_research_matrix(self._data_dir(), limit=200),
            "leaderboard": get_leaderboard(self._data_dir(), mode=mode, limit=25),
            "grid": get_matrix(self._data_dir()),
            "sessions": list_research_sessions(self._data_dir()),
        }

    def _apply_lab_data(self, bundle: dict[str, Any]) -> None:
        summary = bundle.get("summary") or {}
        self._summary_var.set(
            f"{summary.get('strategy_run_count', 0)} strategy runs · "
            f"{summary.get('model_count', 0)} models · "
            f"{summary.get('strategy_count', 0)} strategies"
        )

        self.matrix_tree.delete(*self.matrix_tree.get_children())
        matrix_doc = bundle.get("matrix") or {}
        self._matrix_rows = matrix_doc.get("rows") or []
        for r in self._matrix_rows:
            sid = str(r.get("strategy_run_id") or "")
            self.matrix_tree.insert(
                "",
                "end",
                iid=sid,
                values=(
                    r.get("model_id"),
                    r.get("strategy_name"),
                    r.get("profit"),
                    r.get("win_rate_pct"),
                    r.get("profit_factor"),
                    r.get("max_drawdown"),
                    r.get("trade_count"),
                    r.get("composite_score"),
                    r.get("prediction_mae"),
                ),
            )

        self.leaderboard_tree.delete(*self.leaderboard_tree.get_children())
        leaderboard_doc = bundle.get("leaderboard") or {}
        for r in leaderboard_doc.get("leaderboard") or []:
            self.leaderboard_tree.insert(
                "",
                "end",
                values=(
                    r.get("rank"),
                    r.get("model_id"),
                    r.get("strategy_name"),
                    r.get("profit"),
                    r.get("win_rate_pct"),
                    r.get("profit_factor"),
                    r.get("max_drawdown"),
                    r.get("composite_score"),
                ),
            )

        self._grid_text.delete("1.0", "end")
        grid_doc = bundle.get("grid") or {}
        lines = ["Model × Strategy (composite score / profit)", ""]
        models = grid_doc.get("models") or []
        strategies = grid_doc.get("strategies") or []
        grid = grid_doc.get("grid") or {}
        header = "Model".ljust(24) + "".join(s[:12].ljust(14) for s in strategies)
        lines.append(header)
        for model in models:
            cells = []
            for strat in strategies:
                cell = grid.get(model, {}).get(strat)
                if cell:
                    val = cell.get("profit")
                    comp = cell.get("composite_score")
                    cells.append(f"{val}/{comp}"[:12].ljust(14))
                else:
                    cells.append("—".ljust(14))
            lines.append(model[:24].ljust(24) + "".join(cells))
        self._grid_text.insert("end", "\n".join(lines))

        self.sessions_tree.delete(*self.sessions_tree.get_children())
        sessions_doc = bundle.get("sessions") or {}
        for s in sessions_doc.get("sessions") or []:
            sid = str(s.get("session_id") or "")
            self.sessions_tree.insert(
                "",
                "end",
                iid=sid,
                values=(
                    s.get("title"),
                    len(s.get("strategy_run_ids") or []),
                    len(s.get("prediction_run_ids") or []),
                    (s.get("updated_on") or "")[:19],
                ),
            )
        self._status_var.set(f"Matrix: {len(self._matrix_rows)} row(s)")

    def _load_leaderboard(self) -> None:
        self.refresh_all(lazy=True)

    def _load_sessions(self) -> None:
        from chain_replay_ml.research_lab import list_research_sessions

        try:
            doc = list_research_sessions(self._data_dir())
            self.sessions_tree.delete(*self.sessions_tree.get_children())
            for s in doc.get("sessions") or []:
                sid = str(s.get("session_id") or "")
                self.sessions_tree.insert(
                    "",
                    "end",
                    iid=sid,
                    values=(
                        s.get("title"),
                        len(s.get("strategy_run_ids") or []),
                        len(s.get("prediction_run_ids") or []),
                        (s.get("updated_on") or "")[:19],
                    ),
                )
        except Exception as exc:
            self._status_var.set(f"Sessions error: {exc}")

    def _on_matrix_select(self) -> None:
        sel = self.matrix_tree.selection()
        if not sel:
            return
        self._cmp_a.set(sel[0])

    def _compare_runs(self) -> None:
        from chain_replay_ml.research_lab import compare_strategy_runs

        a = self._cmp_a.get().strip()
        b = self._cmp_b.get().strip()
        if not a or not b:
            messagebox.showinfo("Compare", "Enter two strategy run IDs.")
            return
        self._compare_text.delete("1.0", "end")
        try:
            doc = compare_strategy_runs(self._data_dir(), [a, b])
            self._compare_text.insert("end", json.dumps(doc, indent=2, default=str))
        except Exception as exc:
            self._compare_text.insert("end", str(exc))

    def _create_session(self) -> None:
        from chain_replay_ml.research_lab import create_research_session, update_research_session

        title = simpledialog.askstring("Research Session", "Session title:", parent=self)
        if not title or not title.strip():
            return
        try:
            doc = create_research_session(self._data_dir(), title=title.strip())
            sid = doc["session"]["session_id"]
            run_ids = [str(r.get("strategy_run_id")) for r in self._matrix_rows if r.get("strategy_run_id")]
            if run_ids:
                update_research_session(self._data_dir(), sid, {"strategy_run_ids": run_ids[:20]})
            self.refresh_all(lazy=False)
            messagebox.showinfo("Session", f"Created: {title.strip()}")
        except Exception as exc:
            messagebox.showerror("Session", str(exc))
