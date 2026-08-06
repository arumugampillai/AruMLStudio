"""Outcome Label Engine UI — Create Label Run (Phase X).

Writes ``data/label_runs/{id}.parquet`` + ``_meta.json`` only.
Never mutates Feature Dataset parquet under ``data/datasets/``.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .model_builder import service as mb_service
from .ui_state import UIStateManager, get_ui_state_manager

# Migrated from a panel-local ``ole_ui_prefs.json`` file to the shared
# UIStateManager store (one JSON file for the whole app). ``load_ole_prefs``/
# ``save_ole_prefs`` are kept as thin wrappers around it so the rest of this
# module's restore/persist logic is unchanged; the whole prefs dict is saved
# under one project-scoped key so switching chart/data folders can't mix up
# two different projects' Outcome Label Engine selections.
_OLE_PREFS_KEY = "outcome_label_engine.prefs"


def _prefs_key(chart_dir: str) -> str:
    return UIStateManager.scope_key(chart_dir, _OLE_PREFS_KEY)


def load_ole_prefs(chart_dir: str) -> dict[str, Any]:
    doc = get_ui_state_manager().get(_prefs_key(chart_dir))
    return doc if isinstance(doc, dict) else {}


def save_ole_prefs(chart_dir: str, doc: dict[str, Any]) -> None:
    get_ui_state_manager().set(_prefs_key(chart_dir), doc)


def _fmt_rows(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


class OutcomeLabelEnginePanel(ttk.Frame):
    """Dedicated OLE surface: Feature Dataset + strategy → Create Label Run."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_label_run_created: Callable[[str], None] | None = None,
        on_open_create_model: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._data_dir = chart_data_dir(chart_dir)
        self._on_label_run_created = on_label_run_created
        self._on_open_create_model = on_open_create_model
        self._busy = False
        self._suppress_persist = False
        self._datasets: list[dict[str, Any]] = []
        self._dataset_by_name: dict[str, dict[str, Any]] = {}
        self._build()
        self._restore_prefs()

    def page_title(self) -> str:
        return "Outcome Label Engine"

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._data_dir = chart_data_dir(chart_dir)
        self._restore_prefs()
        self.refresh()

    def show(self) -> None:
        self.lift()
        self.tkraise()
        self.refresh()

    def on_show(self) -> None:
        self.refresh()

    def apply_prefill(self, prefill: dict[str, Any] | None) -> None:
        prefill = prefill or {}
        self._suppress_persist = True
        try:
            ds = str(prefill.get("dataset") or "").strip()
            if ds:
                self._dataset_var.set(ds)
            strat = str(prefill.get("strategy") or "").strip()
            if strat:
                self._strategy_var.set(strat)
            self._on_strategy_changed()
            params = prefill.get("params") if isinstance(prefill.get("params"), dict) else {}
            if params:
                if "holding_seconds" in params:
                    self._hold_var.set(str(params.get("holding_seconds")))
                if "tp_value" in params:
                    self._tp_var.set(str(params.get("tp_value")))
                if "sl_value" in params:
                    self._sl_var.set(str(params.get("sl_value")))
                if params.get("barrier_type"):
                    self._barrier_var.set(str(params.get("barrier_type")))
            col = str(prefill.get("target_column") or prefill.get("target") or "").strip()
            if col:
                self._column_var.set(col)
        finally:
            self._suppress_persist = False
        self.refresh()
        self._persist_prefs()

    def _build(self) -> None:
        head = ttk.Frame(self)
        head.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(head, text="Outcome Label Engine", font=("Segoe UI", 14, "bold")).pack(
            side="left"
        )
        ttk.Label(
            head,
            text="Create Label Runs only — Feature Datasets stay immutable",
            foreground="#888",
        ).pack(side="left", padx=(12, 0))
        self._counts_var = tk.StringVar(value="")
        ttk.Label(head, textvariable=self._counts_var, foreground="#58a6ff").pack(
            side="right"
        )

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        form = ttk.LabelFrame(body, text="Create Label Run", padding=10)
        form.pack(fill="x")

        row = ttk.Frame(form)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Feature Dataset", width=18).pack(side="left")
        self._dataset_var = tk.StringVar()
        self._dataset_cb = ttk.Combobox(row, textvariable=self._dataset_var, state="readonly", width=48)
        self._dataset_cb.pack(side="left", fill="x", expand=True)
        self._dataset_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_changed(persist=True))
        self._dataset_meta_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self._dataset_meta_var, foreground="#888").pack(
            side="left", padx=(8, 0)
        )

        row = ttk.Frame(form)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Mode", width=18).pack(side="left")
        self._mode_var = tk.StringVar(value="promote")
        ttk.Radiobutton(
            row, text="Promote column (Fixed Horizon / existing target)",
            variable=self._mode_var, value="promote", command=self._on_mode_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            row, text="Triple Barrier (sample-grid LTP paths from Feature Dataset)",
            variable=self._mode_var, value="triple_barrier", command=self._on_mode_changed,
        ).pack(side="left", padx=(12, 0))

        self._promote_frame = ttk.Frame(form)
        self._promote_frame.pack(fill="x", pady=3)
        ttk.Label(self._promote_frame, text="Target column", width=18).pack(side="left")
        self._column_var = tk.StringVar()
        self._column_cb = ttk.Combobox(
            self._promote_frame, textvariable=self._column_var, state="readonly", width=40
        )
        self._column_cb.pack(side="left")
        self._column_cb.bind("<<ComboboxSelected>>", lambda _e: self._persist_prefs())
        self._column_count_var = tk.StringVar(value="")
        ttk.Label(self._promote_frame, textvariable=self._column_count_var, foreground="#888").pack(
            side="left", padx=(8, 0)
        )

        self._tb_frame = ttk.Frame(form)
        row = ttk.Frame(self._tb_frame)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Strategy", width=18).pack(side="left")
        self._strategy_var = tk.StringVar(value="triple_barrier")
        ttk.Entry(row, textvariable=self._strategy_var, width=24, state="readonly").pack(side="left")

        row = ttk.Frame(self._tb_frame)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Barrier type", width=18).pack(side="left")
        self._barrier_var = tk.StringVar(value="percentage")
        ttk.Combobox(
            row, textvariable=self._barrier_var, values=["percentage", "points"],
            state="readonly", width=16,
        ).pack(side="left")
        ttk.Label(row, text="Hold (s)").pack(side="left", padx=(12, 4))
        self._hold_var = tk.StringVar(value="300")
        ttk.Entry(row, textvariable=self._hold_var, width=8).pack(side="left")
        ttk.Label(row, text="TP").pack(side="left", padx=(12, 4))
        self._tp_var = tk.StringVar(value="20")
        ttk.Entry(row, textvariable=self._tp_var, width=8).pack(side="left")
        ttk.Label(row, text="SL").pack(side="left", padx=(12, 4))
        self._sl_var = tk.StringVar(value="10")
        ttk.Entry(row, textvariable=self._sl_var, width=8).pack(side="left")
        for var in (self._barrier_var, self._hold_var, self._tp_var, self._sl_var):
            var.trace_add("write", lambda *_a: self._persist_prefs())

        btns = ttk.Frame(form)
        btns.pack(fill="x", pady=(10, 0))
        self._create_btn = ttk.Button(btns, text="Create Label Run", command=self._on_create)
        self._create_btn.pack(side="left")
        ttk.Button(btns, text="Refresh Registry", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(
            btns, text="Return to Create Model", command=self._return_to_create_model
        ).pack(side="left", padx=6)

        self._status = ttk.Label(form, text="", foreground="#888")
        self._status.pack(anchor="w", pady=(8, 0))

        reg = ttk.LabelFrame(body, text="Label Registry", padding=10)
        reg.pack(fill="both", expand=True, pady=(12, 0))
        self._registry_title_var = tk.StringVar(value="Label Registry")
        # LabelFrame text is static at create; keep a subtitle for count.
        self._registry_count = ttk.Label(reg, textvariable=self._registry_title_var, foreground="#58a6ff")
        self._registry_count.pack(anchor="w", pady=(0, 4))
        cols = ("run_id", "strategy", "dataset", "rows", "created")
        self._tree = ttk.Treeview(reg, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (36, 16, 28, 10, 22)):
            self._tree.heading(c, text=c.replace("_", " ").title())
            self._tree.column(c, width=w * 8, stretch=True)
        self._tree.pack(fill="both", expand=True)
        self._on_mode_changed()

    def _sync_strategy_to_mode(self) -> None:
        """Keep readonly Strategy field consistent with Mode (never stale prefs)."""
        if self._mode_var.get().strip() == "triple_barrier":
            self._strategy_var.set("triple_barrier")
        else:
            self._strategy_var.set("fixed_horizon")

    def _restore_prefs(self) -> None:
        prefs = load_ole_prefs(self.chart_dir)
        if not prefs:
            return
        self._suppress_persist = True
        try:
            if prefs.get("mode"):
                self._mode_var.set(str(prefs.get("mode")))
            if prefs.get("dataset"):
                self._dataset_var.set(str(prefs.get("dataset")))
            if prefs.get("target_column"):
                self._column_var.set(str(prefs.get("target_column")))
            # Strategy is derived from mode — ignore stale prefs.strategy.
            if prefs.get("barrier_type"):
                self._barrier_var.set(str(prefs.get("barrier_type")))
            if prefs.get("holding_seconds") is not None:
                self._hold_var.set(str(prefs.get("holding_seconds")))
            if prefs.get("tp_value") is not None:
                self._tp_var.set(str(prefs.get("tp_value")))
            if prefs.get("sl_value") is not None:
                self._sl_var.set(str(prefs.get("sl_value")))
            self._on_mode_changed()
        finally:
            self._suppress_persist = False

    def _persist_prefs(self) -> None:
        if self._suppress_persist:
            return
        try:
            save_ole_prefs(
                self.chart_dir,
                {
                    "dataset": self._selected_dataset_name(),
                    "mode": self._mode_var.get().strip(),
                    "target_column": self._column_var.get().strip(),
                    "strategy": self._strategy_var.get().strip(),
                    "barrier_type": self._barrier_var.get().strip(),
                    "holding_seconds": self._hold_var.get().strip(),
                    "tp_value": self._tp_var.get().strip(),
                    "sl_value": self._sl_var.get().strip(),
                },
            )
        except Exception:
            pass

    def refresh(self) -> None:
        prefs = load_ole_prefs(self.chart_dir)
        saved_ds = str(prefs.get("dataset") or self._dataset_var.get() or "").strip()
        saved_col = str(prefs.get("target_column") or self._column_var.get() or "").strip()
        try:
            datasets = mb_service.list_builder_datasets(self._data_dir)
        except Exception:
            datasets = []
        self._datasets = list(datasets)
        self._dataset_by_name = {
            str(d.get("dataset_name") or ""): d
            for d in self._datasets
            if d.get("dataset_name")
        }
        names = list(self._dataset_by_name.keys())
        display: list[str] = []
        display_to_name: dict[str, str] = {}
        name_to_display: dict[str, str] = {}
        for n in names:
            row = self._dataset_by_name.get(n) or {}
            label = f"{n}  ({_fmt_rows(row.get('row_count'))} rows)"
            display.append(label)
            display_to_name[label] = n
            name_to_display[n] = label
        self._display_to_name = display_to_name
        self._name_to_display = name_to_display
        self._dataset_cb["values"] = display

        pick = ""
        if saved_ds and saved_ds in name_to_display:
            pick = name_to_display[saved_ds]
        elif display:
            pick = display[0]
        self._suppress_persist = True
        try:
            self._dataset_var.set(pick)
            if saved_col:
                self._column_var.set(saved_col)
            self._on_dataset_changed(persist=False)
        finally:
            self._suppress_persist = False

        n_ds = len(names)
        self._counts_var.set(f"{n_ds} dataset{'s' if n_ds != 1 else ''}")
        self._reload_registry()
        self._persist_prefs()

    def _selected_dataset_name(self) -> str:
        raw = self._dataset_var.get().strip()
        if hasattr(self, "_display_to_name") and raw in self._display_to_name:
            return self._display_to_name[raw]
        # Already a bare name.
        if raw in self._dataset_by_name:
            return raw
        return raw

    def _on_dataset_changed(self, *, persist: bool = True) -> None:
        ds = self._selected_dataset_name()
        row = self._dataset_by_name.get(ds) or {}
        rows_n = row.get("row_count")
        feats_n = row.get("feature_count")
        tgts_n = row.get("target_count")
        bits = []
        if rows_n is not None:
            bits.append(f"{_fmt_rows(rows_n)} rows")
        if feats_n is not None:
            bits.append(f"{_fmt_rows(feats_n)} features")
        if tgts_n is not None:
            bits.append(f"{_fmt_rows(tgts_n)} targets")
        self._dataset_meta_var.set(" · ".join(bits) if bits else "")

        cols: list[str] = []
        if ds:
            try:
                doc = mb_service.load_dataset_metadata_doc(self._data_dir, ds) or {}
                meta = doc.get("metadata") or doc
                raw = meta.get("prediction_target_columns") or meta.get("target_columns") or []
                cols = [str(c) for c in raw]
            except Exception:
                cols = []
        prev_col = self._column_var.get().strip()
        self._column_cb["values"] = cols
        self._column_count_var.set(f"{len(cols)} column{'s' if len(cols) != 1 else ''}")
        if prev_col and prev_col in cols:
            self._column_var.set(prev_col)
        elif cols:
            self._column_var.set(cols[0])
        else:
            self._column_var.set("")
        if persist:
            self._persist_prefs()

    def _on_mode_changed(self) -> None:
        self._sync_strategy_to_mode()
        if self._mode_var.get() == "promote":
            self._promote_frame.pack(fill="x", pady=3)
            self._tb_frame.pack_forget()
        else:
            self._promote_frame.pack_forget()
            self._tb_frame.pack(fill="x", pady=3)
        self._persist_prefs()

    def _on_strategy_changed(self) -> None:
        if self._strategy_var.get() == "triple_barrier":
            self._mode_var.set("triple_barrier")
        else:
            self._mode_var.set("promote")
        self._on_mode_changed()

    def _reload_registry(self) -> None:
        for i in self._tree.get_children():
            self._tree.delete(i)
        n = 0
        try:
            from chain_replay_ml.label_runs import list_label_runs

            runs = list_label_runs(self._data_dir)
            n = len(runs)
            for r in runs:
                created = str(r.created_at or "")[:19]
                self._tree.insert(
                    "",
                    "end",
                    values=(r.run_id, r.strategy, r.dataset_id, _fmt_rows(r.rows), created),
                )
            self._registry_title_var.set(
                f"Label Registry — {n} run{'s' if n != 1 else ''}"
            )
            # Keep header counts in sync.
            n_ds = len(self._dataset_by_name)
            self._counts_var.set(
                f"{n_ds} dataset{'s' if n_ds != 1 else ''} · {n} label run{'s' if n != 1 else ''}"
            )
        except Exception as exc:
            self._status.configure(text=f"Registry error: {exc}")
            self._registry_title_var.set("Label Registry")

    def _on_create(self) -> None:
        if self._busy:
            return
        ds = self._selected_dataset_name()
        if not ds:
            messagebox.showwarning("OLE", "Select a Feature Dataset.", parent=self)
            return
        self._sync_strategy_to_mode()
        self._persist_prefs()
        mode = self._mode_var.get()
        hold_s = self._hold_var.get().strip()
        tp_s = self._tp_var.get().strip()
        sl_s = self._sl_var.get().strip()
        barrier = self._barrier_var.get().strip() or "percentage"
        self._busy = True
        self._create_btn.configure(state="disabled")
        self._status.configure(text="Creating Label Run…")

        def work() -> None:
            err: str | None = None
            result: dict[str, Any] | None = None
            try:
                if mode == "promote":
                    from chain_replay_ml.label_runs import promote_feature_column_to_label_run

                    col = self._column_var.get().strip()
                    if not col:
                        raise ValueError("Select a target column to promote")
                    result = promote_feature_column_to_label_run(
                        self._data_dir,
                        ds,
                        col,
                        strategy="fixed_horizon",
                        parameters={"source_column": col},
                    )
                else:
                    from chain_replay_ml.label_runs import create_triple_barrier_label_run

                    try:
                        holding_seconds = int(float(hold_s))
                        tp_value = float(tp_s)
                        sl_value = float(sl_s)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Hold / TP / SL must be numeric for Triple Barrier"
                        ) from exc
                    if holding_seconds <= 0:
                        raise ValueError("Holding seconds must be > 0")
                    if tp_value <= 0 or sl_value <= 0:
                        raise ValueError("TP and SL must be > 0")
                    result = create_triple_barrier_label_run(
                        self._data_dir,
                        ds,
                        barrier_type=barrier,
                        holding_seconds=holding_seconds,
                        tp_value=tp_value,
                        sl_value=sl_value,
                    )
            except Exception as exc:
                err = str(exc)

            def done() -> None:
                self._busy = False
                self._create_btn.configure(state="normal")
                if err:
                    self._status.configure(text=f"Failed: {err}")
                    messagebox.showerror("Create Label Run", err, parent=self)
                    return
                run_id = str((result or {}).get("run_id") or "")
                self._status.configure(text=f"Created Label Run: {run_id}")
                self._reload_registry()
                self._persist_prefs()
                if self._on_label_run_created and run_id:
                    self._on_label_run_created(run_id)
                messagebox.showinfo(
                    "Create Label Run",
                    f"Label Run created:\n{run_id}\n\nFeature dataset was not modified.",
                    parent=self,
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _return_to_create_model(self) -> None:
        self._persist_prefs()
        sel = self._tree.selection()
        run_id = None
        if sel:
            vals = self._tree.item(sel[0], "values")
            if vals:
                run_id = vals[0]
        payload = {
            "dataset": self._selected_dataset_name(),
            "label_run_id": run_id,
        }
        if self._on_open_create_model:
            self._on_open_create_model(payload)
