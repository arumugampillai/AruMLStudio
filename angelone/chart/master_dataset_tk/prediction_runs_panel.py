"""Prediction Runs — browse runs, folds, rows, and compare (Phase 1 Tk)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .ui_state import get_ui_state_manager


def _fmt_num(v: Any) -> str:
    try:
        return f"{float(v):,.4f}"
    except (TypeError, ValueError):
        return "—"


def _short_id(run_id: str) -> str:
    if len(run_id) <= 14:
        return run_id
    return f"{run_id[:8]}…{run_id[-4:]}"


class PredictionRunsPanel(ttk.Frame, LazyLoadMixin):
    """Runs (+ fold metrics) | Prediction Rows | Compare Runs."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_model: Callable[[str], None] | None = None,
        on_open_fold_replay: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_model = on_open_model
        self._on_open_fold_replay = on_open_fold_replay
        self._runs: list[dict[str, Any]] = []
        self._selected_run_id: str | None = None
        self._selected_fold_id: str | None = None
        self._ui_state = get_ui_state_manager()
        self._model_filter = tk.StringVar(
            value=str(self._ui_state.get("prediction_runs.model_filter") or "(all models)")
        )
        self._status_var = tk.StringVar(value="")
        self._compare_a = tk.StringVar(value="")
        self._compare_b = tk.StringVar(value="")
        self._build_ui()
        self._ui_state.bind_combobox(
            self._model_combo, "prediction_runs.model_filter", var=self._model_filter, restore=False
        )
        self._ui_state.bind_notebook(self._notebook, "prediction_runs.tab")
        self._ui_state.bind_notebook(self._run_detail_notebook, "prediction_runs.detail_tab")
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh_runs(lazy=True)

    def set_model_filter(self, model_name: str | None) -> None:
        if model_name:
            self._model_filter.set(model_name)
        else:
            self._model_filter.set("(all models)")
        self.refresh_runs()

    def select_run(self, run_id: str) -> None:
        self._selected_run_id = run_id
        self._notebook.select(0)
        self.refresh_runs()
        self._load_fold_metrics()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Model filter").pack(side="left")
        self._model_combo = ttk.Combobox(
            toolbar, textvariable=self._model_filter, width=36, state="readonly",
        )
        self._model_combo.pack(side="left", padx=6)
        self._model_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_runs())
        ttk.Button(toolbar, text="Refresh", command=self.refresh_runs).pack(side="left", padx=4)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self._tab_runs = ttk.Frame(self._notebook)
        self._tab_rows = ttk.Frame(self._notebook)
        self._tab_compare = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_runs, text="Runs")
        self._notebook.add(self._tab_rows, text="Prediction Rows")
        self._notebook.add(self._tab_compare, text="Compare Runs")

        self._build_runs_tab()
        self._build_rows_tab()
        self._build_compare_tab()

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))

    def _build_runs_tab(self) -> None:
        self._tab_runs.rowconfigure(0, weight=3)
        self._tab_runs.rowconfigure(1, weight=1)
        self._tab_runs.columnconfigure(0, weight=1)

        runs_frame = ttk.Frame(self._tab_runs)
        runs_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))

        table_frame = ttk.Frame(runs_frame)
        table_frame.pack(fill="both", expand=True)
        cols = ("run_id", "model", "status", "created", "folds", "predictions", "target")
        self.runs_tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        for c, w, label in (
            ("run_id", 120, "Run ID"),
            ("model", 160, "Model"),
            ("status", 80, "Status"),
            ("created", 140, "Created"),
            ("folds", 50, "Folds"),
            ("predictions", 80, "Rows"),
            ("target", 120, "Target"),
        ):
            self.runs_tree.heading(c, text=label)
            self.runs_tree.column(c, width=w, anchor="center" if c not in ("run_id", "model", "target") else "w")
        self.runs_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.runs_tree.yview)
        sb.pack(side="right", fill="y")
        self.runs_tree.configure(yscrollcommand=sb.set)
        self.runs_tree.bind("<<TreeviewSelect>>", self._on_run_selected)

        btn_row = ttk.Frame(runs_frame)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="Open Fold Replay", command=self._open_fold_replay).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Open model", command=self._open_selected_model).pack(side="left", padx=2)

        detail_frame = ttk.Frame(self._tab_runs, padding=2)
        detail_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

        self._run_detail_notebook = ttk.Notebook(detail_frame)
        self._run_detail_notebook.pack(fill="both", expand=True)

        tab_fold_metrics = ttk.Frame(self._run_detail_notebook, padding=4)
        tab_run_detail = ttk.Frame(self._run_detail_notebook, padding=4)
        self._run_detail_notebook.add(tab_fold_metrics, text="Fold Metrics")
        self._run_detail_notebook.add(tab_run_detail, text="Run Details")

        self._run_detail_text = scrolledtext.ScrolledText(tab_run_detail, font=("Consolas", 9))
        self._run_detail_text.pack(fill="both", expand=True)

        fold_table = ttk.Frame(tab_fold_metrics)
        fold_table.pack(fill="both", expand=True)
        fold_cols = ("fold", "val_rows", "mae", "rmse", "direction", "predictions")
        self.folds_tree = ttk.Treeview(fold_table, columns=fold_cols, show="headings")
        for c, w, label in (
            ("fold", 60, "Fold"),
            ("val_rows", 80, "Val rows"),
            ("mae", 80, "MAE"),
            ("rmse", 80, "RMSE"),
            ("direction", 90, "Direction %"),
            ("predictions", 90, "Predictions"),
        ):
            self.folds_tree.heading(c, text=label)
            self.folds_tree.column(c, width=w, anchor="center")
        self.folds_tree.pack(side="left", fill="both", expand=True)
        fold_sb = ttk.Scrollbar(fold_table, orient="vertical", command=self.folds_tree.yview)
        fold_sb.pack(side="right", fill="y")
        self.folds_tree.configure(yscrollcommand=fold_sb.set)
        self.folds_tree.bind("<<TreeviewSelect>>", self._on_fold_selected)

    def _build_rows_tab(self) -> None:
        top = ttk.Frame(self._tab_rows, padding=4)
        top.pack(fill="x")
        ttk.Button(top, text="Load rows", command=self._load_prediction_rows).pack(side="left", padx=4)
        self._rows_info = tk.StringVar(value="Select a run and fold, then load rows.")
        ttk.Label(top, textvariable=self._rows_info).pack(side="left", padx=8)

        cols = (
            "idx", "day", "token", "strike", "type", "spot",
            "ltp", "predicted", "actual", "error", "direction",
        )
        self.rows_tree = ttk.Treeview(self._tab_rows, columns=cols, show="headings", height=16)
        for c, w, label in (
            ("idx", 40, "#"),
            ("day", 90, "Day"),
            ("token", 70, "Token"),
            ("strike", 60, "Strike"),
            ("type", 40, "Type"),
            ("spot", 70, "Spot"),
            ("ltp", 70, "LTP"),
            ("predicted", 80, "Predicted"),
            ("actual", 80, "Actual"),
            ("error", 70, "Error"),
            ("direction", 50, "Dir"),
        ):
            self.rows_tree.heading(c, text=label)
            self.rows_tree.column(c, width=w, anchor="center")
        self.rows_tree.pack(fill="both", expand=True, padx=8, pady=4)

    def _build_compare_tab(self) -> None:
        form = ttk.Frame(self._tab_compare, padding=8)
        form.pack(fill="x")
        ttk.Label(form, text="Run A").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(form, textvariable=self._compare_a, width=40).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(form, text="Run B").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(form, textvariable=self._compare_b, width=40).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(form, text="Compare", command=self._run_compare).grid(row=2, column=1, sticky="e", padx=4, pady=8)

        self._compare_text = scrolledtext.ScrolledText(self._tab_compare, height=20, font=("Consolas", 9))
        self._compare_text.pack(fill="both", expand=True, padx=8, pady=4)

    def _load_model_names(self) -> list[str]:
        try:
            from .selection_lists import get_sorted_model_names

            return get_sorted_model_names(self._data_dir(), lightweight=True)
        except Exception:
            return []

    def refresh_runs(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_runs,
                apply=self._apply_runs,
                message="Loading prediction runs…",
                status_var=self._status_var,
            )
            return
        try:
            bundle = self._fetch_runs()
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_runs(bundle)

    def _fetch_runs(self) -> dict[str, Any]:
        from chain_replay_ml.prediction_runs import list_all_runs, list_runs
        from .selection_lists import get_sorted_model_names

        model_names = get_sorted_model_names(self._data_dir(), lightweight=True)
        filt = self._model_filter.get()
        if filt and filt != "(all models)":
            runs = list_runs(self._data_dir(), filt, limit=100)
        else:
            runs = list_all_runs(self._data_dir(), limit=100)
        return {"models": model_names, "runs": runs}

    def _apply_runs(self, bundle: dict[str, Any]) -> None:
        values = ["(all models)"] + list(bundle.get("models") or [])
        self._model_combo.configure(values=values)
        self._runs = list(bundle.get("runs") or [])
        self.runs_tree.delete(*self.runs_tree.get_children())
        for r in self._runs:
            rid = str(r.get("run_id") or "")
            self.runs_tree.insert(
                "",
                "end",
                iid=rid,
                values=(
                    _short_id(rid),
                    r.get("model_id") or "—",
                    r.get("status") or "—",
                    (r.get("created_at") or "—")[:19],
                    r.get("fold_count") or "—",
                    r.get("prediction_count") or "—",
                    r.get("target") or "—",
                ),
            )
        self._status_var.set(f"{len(self._runs)} prediction run(s)")
        if self._selected_run_id and self._selected_run_id in self.runs_tree.get_children():
            self.runs_tree.selection_set(self._selected_run_id)
            self._load_fold_metrics()

    def _on_run_selected(self, _event: object = None) -> None:
        sel = self.runs_tree.selection()
        if not sel:
            return
        self._selected_run_id = sel[0]
        self._selected_fold_id = None
        self._load_fold_metrics()

    def _on_fold_selected(self, _event: object = None) -> None:
        sel = self.folds_tree.selection()
        if not sel:
            return
        self._selected_fold_id = sel[0]

    def _load_fold_metrics(self) -> None:
        from chain_replay_ml.prediction_runs import get_run_detail

        self.folds_tree.delete(*self.folds_tree.get_children())
        self._run_detail_text.delete("1.0", "end")
        rid = self._selected_run_id
        if not rid:
            return
        try:
            detail = get_run_detail(self._data_dir(), rid)
        except Exception as exc:
            self._run_detail_text.insert("end", f"Error: {exc}")
            return
        if not detail:
            self._run_detail_text.insert("end", "Run not found.")
            return

        lines = [
            f"run_id: {detail.get('run_id')}",
            f"model_id: {detail.get('model_id')}",
            f"status: {detail.get('status')}",
            f"dataset: {detail.get('dataset_name')}",
            f"target: {detail.get('target')}",
            f"predictions stored: {detail.get('prediction_count_stored')}",
            f"dataset_fp: {detail.get('dataset_fingerprint')}",
            f"feature_hash: {detail.get('feature_snapshot_hash')}",
            f"wf_hash: {detail.get('walk_forward_config_hash')}",
        ]
        self._run_detail_text.insert("end", "\n".join(lines))
        try:
            self._run_detail_notebook.select(0)
        except tk.TclError:
            pass

        for f in detail.get("folds") or []:
            fid = str(f.get("fold_id") or "")
            self.folds_tree.insert(
                "",
                "end",
                iid=fid,
                values=(
                    f.get("fold_number"),
                    f.get("validation_rows") or "—",
                    _fmt_num(f.get("mae")),
                    _fmt_num(f.get("rmse")),
                    _fmt_num(f.get("directional_accuracy_pct")),
                    f.get("prediction_count") or "—",
                ),
            )
        folds = detail.get("folds") or []
        if folds and not self._selected_fold_id:
            self._selected_fold_id = str(folds[0].get("fold_id") or "")
            if self._selected_fold_id:
                self.folds_tree.selection_set(self._selected_fold_id)

    def _load_prediction_rows(self) -> None:
        from chain_replay_ml.prediction_runs import get_fold_rows

        self.rows_tree.delete(*self.rows_tree.get_children())
        rid = self._selected_run_id
        fid = self._selected_fold_id
        if not rid or not fid:
            messagebox.showinfo("Prediction Rows", "Select a run and fold first.")
            return
        try:
            doc = get_fold_rows(self._data_dir(), rid, fid, limit=500)
        except Exception as exc:
            messagebox.showerror("Prediction Rows", str(exc))
            return
        if not doc.get("ok"):
            messagebox.showerror("Prediction Rows", doc.get("error") or "Failed")
            return
        total = doc.get("total") or 0
        self._rows_info.set(f"Showing {len(doc.get('rows') or [])} of {total} rows (fold {fid[:12]}…)")
        for row in doc.get("rows") or []:
            self.rows_tree.insert(
                "",
                "end",
                values=(
                    row.get("row_index"),
                    row.get("trading_day") or "—",
                    row.get("token") or "—",
                    row.get("strike") or "—",
                    row.get("option_type") or "—",
                    _fmt_num(row.get("spot")),
                    _fmt_num(row.get("ltp")),
                    _fmt_num(row.get("predicted_ltp")),
                    _fmt_num(row.get("actual_ltp")),
                    _fmt_num(row.get("prediction_error")),
                    row.get("direction_correct") if row.get("direction_correct") is not None else "—",
                ),
            )

    def _run_compare(self) -> None:
        from chain_replay_ml.prediction_runs import compare_runs

        a = self._compare_a.get().strip()
        b = self._compare_b.get().strip()
        if not a or not b:
            messagebox.showinfo("Compare", "Enter both run IDs.")
            return
        self._compare_text.delete("1.0", "end")
        try:
            result = compare_runs(self._data_dir(), a, b)
        except Exception as exc:
            self._compare_text.insert("end", f"Error: {exc}")
            return
        if not result.get("ok"):
            self._compare_text.insert("end", result.get("error") or "Compare failed")
            return
        lines = [
            f"Run A: {result['run_a'].get('model_id')} ({a})",
            f"Run B: {result['run_b'].get('model_id')} ({b})",
            "",
            f"{'Fold':<6} {'A MAE':>10} {'B MAE':>10} {'A RMSE':>10} {'B RMSE':>10} {'A Dir%':>10} {'B Dir%':>10}",
        ]
        for c in result.get("fold_comparisons") or []:
            fa = c.get("run_a") or {}
            fb = c.get("run_b") or {}
            lines.append(
                f"{c.get('fold_number'):<6} "
                f"{_fmt_num(fa.get('mae')):>10} {_fmt_num(fb.get('mae')):>10} "
                f"{_fmt_num(fa.get('rmse')):>10} {_fmt_num(fb.get('rmse')):>10} "
                f"{_fmt_num(fa.get('directional_accuracy_pct')):>10} {_fmt_num(fb.get('directional_accuracy_pct')):>10}"
            )
        self._compare_text.insert("end", "\n".join(lines))

    def _open_fold_replay(self) -> None:
        if not self._selected_run_id or not self._selected_fold_id or not self._on_open_fold_replay:
            messagebox.showinfo("Fold Replay", "Select a run and fold first.")
            return
        self._on_open_fold_replay(self._selected_run_id, self._selected_fold_id)

    def _open_selected_model(self) -> None:
        if not self._selected_run_id or not self._on_open_model:
            return
        run = next((r for r in self._runs if r.get("run_id") == self._selected_run_id), None)
        model = run.get("model_id") if run else None
        if model:
            self._on_open_model(str(model))

    def prefill_compare(self, run_a: str, run_b: str = "") -> None:
        self._compare_a.set(run_a)
        if run_b:
            self._compare_b.set(run_b)
        self._notebook.select(2)
