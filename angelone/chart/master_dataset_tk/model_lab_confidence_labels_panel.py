"""Confidence Labels page — Replay-Based outcome labels for Confidence Models.

User-facing name is \"Confidence Labels\" (not Label Runs). Run history and
staleness live here; Strategy Simulator stays unchanged.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, SECTION_FONT


class ModelLabConfidenceLabelsPanel(ttk.Frame):
    """Confidence → Labels: history, detail, build, staleness."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_status = on_status or (lambda _s: None)
        self._lab_db_path = ""
        self._model_name = ""
        self._runs: list[dict[str, Any]] = []
        self._selected: dict[str, Any] | None = None
        self._replay_targets: list[dict[str, Any]] = []
        self._building = False
        self._build_ui()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def refresh_for_lab(self, *, lab_db_path: str | None, model_name: str = "") -> None:
        self._lab_db_path = str(lab_db_path or "").strip()
        self._model_name = str(model_name or "").strip()
        self._reload()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        ttk.Label(self, text="Confidence Labels", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Replay-Based labels for Confidence Models · "
                "built once from a strategy, then reused for many targets"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 8))

        # Staleness banner
        status_box = ttk.LabelFrame(self, text="Latest Labels", padding=8)
        status_box.pack(fill="x", pady=(0, 8))
        top = ttk.Frame(status_box)
        top.pack(fill="x")
        self._status_var = tk.StringVar(value="Not built")
        self._status_lbl = ttk.Label(
            top, textvariable=self._status_var, font=("Segoe UI", 10, "bold")
        )
        self._status_lbl.pack(side="left")
        self._rebuild_btn = ttk.Button(
            top, text="Rebuild Labels", command=self._on_build_labels, state="disabled"
        )
        self._rebuild_btn.pack(side="right")
        self._reason_var = tk.StringVar(value="")
        ttk.Label(
            status_box,
            textvariable=self._reason_var,
            foreground=COL_MUTED,
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3, minsize=360)
        body.columnconfigure(1, weight=2, minsize=280)

        # Run history
        hist = ttk.LabelFrame(body, text="Run History", padding=8)
        hist.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        cols = ("run", "strategy", "status", "rows", "targets", "created")
        self._runs_tree = ttk.Treeview(
            hist, columns=cols, show="headings", height=10, selectmode="browse"
        )
        headings = {
            "run": ("Run", 90),
            "strategy": ("Strategy", 140),
            "status": ("Status", 90),
            "rows": ("Rows", 90),
            "targets": ("Replay Targets", 100),
            "created": ("Created", 140),
        }
        for c, (title, w) in headings.items():
            self._runs_tree.heading(c, text=title)
            self._runs_tree.column(c, width=w, anchor="center" if c != "strategy" else "w")
        self._runs_tree.pack(fill="both", expand=True)
        self._runs_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_run_selected())

        # Detail
        detail = ttk.LabelFrame(body, text="Selected Run", padding=8)
        detail.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._detail_text = tk.Text(
            detail, height=18, wrap="word", font=("Consolas", 9), relief="flat"
        )
        self._detail_text.pack(fill="both", expand=True)
        self._detail_text.insert("1.0", "Select a run to inspect strategy, targets, and outcomes.")
        self._detail_text.configure(state="disabled")

        # Actions
        actions = ttk.LabelFrame(self, text="Build Labels", padding=8)
        actions.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(actions)
        row.pack(fill="x")
        ttk.Label(row, text="Strategy:").pack(side="left")
        self._strategy_var = tk.StringVar(value="")
        self._strategy_combo = ttk.Combobox(
            row, textvariable=self._strategy_var, state="readonly", width=42
        )
        self._strategy_combo.pack(side="left", padx=(6, 8), fill="x", expand=True)
        self._strat_id_map: dict[str, str] = {}
        ttk.Button(row, text="↻", width=3, command=self._load_strategies).pack(
            side="left", padx=(0, 8)
        )
        self._build_btn = ttk.Button(
            row, text="Build Confidence Labels", command=self._on_build_labels
        )
        self._build_btn.pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Refresh", command=self._reload).pack(side="left")

        self._progress = ttk.Progressbar(actions, mode="indeterminate")
        self._progress.pack(fill="x", pady=(8, 0))
        self._action_var = tk.StringVar(value="")
        ttk.Label(actions, textvariable=self._action_var, foreground=COL_MUTED).pack(
            anchor="w", pady=(4, 0)
        )

    # ── Data ──────────────────────────────────────────────────────────────

    def _reload(self) -> None:
        self._load_strategies()
        if not self._lab_db_path:
            self._status_var.set("No Research Lab open")
            self._reason_var.set("Open a Research Lab with a Prediction Dataset.")
            self._status_lbl.configure(foreground=COL_MUTED)
            self._clear_runs()
            self._set_detail("Open a Research Lab to manage Confidence Labels.")
            self._build_btn.configure(state="disabled")
            self._rebuild_btn.configure(state="disabled")
            return

        from chain_replay_ml.model_lab.confidence import confidence_labels_status

        try:
            st = confidence_labels_status(self._lab_db_path, data_dir=self._data_dir())
        except Exception as exc:
            self._status_var.set("Error")
            self._reason_var.set(str(exc))
            self._status_lbl.configure(foreground=COL_WARN)
            return

        self._runs = list(st.get("runs") or [])
        self._replay_targets = list(st.get("replay_targets") or [])
        stale = st.get("staleness") or {}
        self._apply_staleness(stale)
        self._fill_runs(stale)
        self._build_btn.configure(state="normal")

        # Auto-select latest
        if self._runs_tree.get_children():
            first = self._runs_tree.get_children()[0]
            self._runs_tree.selection_set(first)
            self._on_run_selected()
        else:
            self._set_detail(
                "No Confidence Labels yet.\n\n"
                "Choose a strategy and click Build Confidence Labels.\n"
                "This runs the shared replay engine once and derives all "
                "Replay-Based targets (Trade Winner, Profit ≥ X, …)."
            )

    def _apply_staleness(self, stale: dict[str, Any]) -> None:
        status = str(stale.get("status") or "missing")
        reasons = list(stale.get("reasons") or [])
        if status == "up_to_date":
            self._status_var.set("🟢 Up to date")
            self._status_lbl.configure(foreground=COL_OK)
            self._reason_var.set(
                "Latest labels match the current Prediction Dataset and strategy."
            )
            self._rebuild_btn.configure(state="disabled")
        elif status == "out_of_date":
            self._status_var.set("🟠 Out of date")
            self._status_lbl.configure(foreground=COL_WARN)
            reason_block = "\n".join(f"Reason: {r}" for r in reasons) if reasons else (
                "Reason: Labels are stale."
            )
            self._reason_var.set(reason_block)
            self._rebuild_btn.configure(state="normal")
        else:
            self._status_var.set(str(stale.get("status_display") or "Not built"))
            self._status_lbl.configure(foreground=COL_MUTED)
            self._reason_var.set(
                reasons[0]
                if reasons
                else "Build Replay-Based labels before training those models."
            )
            self._rebuild_btn.configure(state="normal" if self._lab_db_path else "disabled")

    def _clear_runs(self) -> None:
        for iid in self._runs_tree.get_children():
            self._runs_tree.delete(iid)
        self._runs = []
        self._selected = None

    def _fill_runs(self, stale: dict[str, Any]) -> None:
        self._clear_runs()
        latest_stale = not bool(stale.get("up_to_date")) if stale.get("has_run") else False
        for i, run in enumerate(self._runs):
            rid = str(run.get("label_run_id") or "")
            is_latest = bool(run.get("is_latest")) or i == 0
            if is_latest:
                label = "Latest"
            elif i == 1:
                label = "Previous"
            else:
                label = f"#{i + 1}"
            strat = (
                f"{run.get('strategy_display_name') or 'Strategy'} "
                f"{run.get('strategy_version_label') or ''}"
            ).strip()
            if is_latest:
                status = "Out of date" if latest_stale else "Complete"
            else:
                status = "Complete"
            rows = int(run.get("prediction_row_count") or run.get("rows_loaded") or 0)
            n_tgt = len(run.get("binary_columns") or [])
            created = str(run.get("created_at") or "—")
            if "T" in created:
                created = created.replace("T", " ")[:19]
            self._runs_tree.insert(
                "",
                "end",
                iid=rid,
                values=(label, strat, status, f"{rows:,}", str(n_tgt), created),
            )

    def _on_run_selected(self) -> None:
        sel = self._runs_tree.selection()
        if not sel:
            return
        rid = sel[0]
        run = next((r for r in self._runs if str(r.get("label_run_id")) == rid), None)
        if not run:
            return
        self._selected = run
        self._render_detail(run)

    def _render_detail(self, run: dict[str, Any]) -> None:
        from chain_replay_ml.model_lab.confidence import assess_label_run_staleness
        from chain_replay_ml.model_lab.target_spec import REPLAY_TARGET_SPECS, TARGET_SPEC_BY_KEY

        stale = assess_label_run_staleness(
            self._lab_db_path, data_dir=self._data_dir(), meta=run
        )
        summary = run.get("outcome_summary") or {}
        if not summary and self._lab_db_path:
            from chain_replay_ml.model_lab.confidence import load_replay_outcome_frames
            from chain_replay_ml.model_lab.confidence_label_builder import (
                outcome_summary_from_frame,
            )

            loaded = load_replay_outcome_frames(
                self._lab_db_path, label_run_id=str(run.get("label_run_id") or "")
            )
            if loaded.get("ok"):
                summary = outcome_summary_from_frame(loaded.get("outcomes"))

        lines: list[str] = []
        lines.append("Strategy")
        lines.append("---------")
        lines.append(
            f"Name:     {run.get('strategy_display_name') or '—'} "
            f"{run.get('strategy_version_label') or ''}".rstrip()
        )
        lines.append(f"Version:  {run.get('strategy_version_id') or '—'}")
        lines.append(f"Config:   {run.get('strategy_config_hash') or '—'}")
        lines.append(f"Mode:     {run.get('replay_mode') or '—'}")
        lines.append(f"Created:  {run.get('created_at') or '—'}")
        lines.append(f"Pred hash:{run.get('prediction_dataset_hash') or '—'}")
        lines.append("")
        if stale.get("has_run"):
            lines.append(
                f"Status:   {stale.get('status_display')}"
                + ("" if stale.get("up_to_date") else "  ← rebuild recommended")
            )
            for reason in stale.get("reasons") or []:
                lines.append(f"  · {reason}")
            lines.append("")

        lines.append("Replay-Based Targets")
        lines.append("--------------------")
        bin_cols = set(run.get("binary_columns") or [])
        rates = run.get("positive_rates") or {}
        for spec in REPLAY_TARGET_SPECS:
            mark = "✓" if spec.column in bin_cols else "·"
            rate = rates.get(spec.column)
            rate_s = f"  ({100 * rate:.1f}% pos)" if isinstance(rate, (int, float)) else ""
            lines.append(f"{mark} {spec.label}{rate_s}")
        # Also show any unknown columns
        known = {t.column for t in REPLAY_TARGET_SPECS}
        for col in sorted(bin_cols - known):
            label = (TARGET_SPEC_BY_KEY.get(col) or type("X", (), {"label": col})).label
            lines.append(f"✓ {label}")

        lines.append("")
        lines.append("Continuous Outcomes")
        lines.append("-------------------")
        lines.append(f"Rows Processed:    {int(summary.get('rows_processed') or run.get('rows_loaded') or 0):,}")
        avg_net = summary.get("avg_net_pnl")
        avg_ret = summary.get("avg_return_pct")
        avg_hold = summary.get("avg_holding_seconds")
        lines.append(
            f"Average Net P&L:   {avg_net:,.2f}" if avg_net is not None else "Average Net P&L:   —"
        )
        lines.append(
            f"Average Return %:  {avg_ret:.2f}%" if avg_ret is not None else "Average Return %:  —"
        )
        lines.append(
            f"Average Hold Time: {avg_hold:.1f}s" if avg_hold is not None else "Average Hold Time: —"
        )

        self._set_detail("\n".join(lines))

    def _set_detail(self, text: str) -> None:
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("1.0", text)
        self._detail_text.configure(state="disabled")

    def _load_strategies(self) -> None:
        from chain_replay_ml.strategy_registry import get_strategy_detail, list_strategies

        self._strat_id_map.clear()
        labels: list[str] = []
        try:
            strategies = list_strategies(self._data_dir())
        except Exception:
            strategies = []
        for s in strategies:
            sid = str(s.get("strategy_id") or "")
            if not sid:
                continue
            try:
                detail = get_strategy_detail(self._data_dir(), sid)
            except Exception:
                detail = None
            champ = (detail or {}).get("champion_version") or {}
            vid = str(champ.get("version_id") or "")
            if not vid:
                continue
            name = str(
                champ.get("display_name")
                or s.get("display_name")
                or "Strategy"
            )
            ver = str(champ.get("version_label") or "")
            label = f"{name} {ver}".strip()
            labels.append(label)
            self._strat_id_map[label] = vid
        self._strategy_combo.configure(values=labels)
        if labels and self._strategy_var.get() not in labels:
            self._strategy_var.set(labels[0])
        elif not labels:
            self._strategy_var.set("")

    def _selected_strategy_version_id(self) -> str:
        return self._strat_id_map.get(self._strategy_var.get(), "")

    def _on_build_labels(self) -> None:
        if self._building:
            return
        if not self._lab_db_path:
            messagebox.showinfo(
                "Confidence Labels",
                "Open a Research Lab with a Prediction Dataset first.",
                parent=self,
            )
            return
        version_id = self._selected_strategy_version_id()
        if not version_id:
            messagebox.showinfo(
                "Confidence Labels",
                "Select a strategy to replay for label generation.",
                parent=self,
            )
            return

        self._building = True
        self._build_btn.configure(state="disabled")
        self._rebuild_btn.configure(state="disabled")
        self._progress.start(12)
        self._action_var.set("Building Confidence Labels (one replay pass)…")
        self._on_status("Confidence Labels · building…")

        lab_path = self._lab_db_path
        data_dir = self._data_dir()

        def _worker() -> None:
            from chain_replay_ml.model_lab.confidence import run_confidence_label_builder

            try:
                result = run_confidence_label_builder(
                    lab_path,
                    data_dir=data_dir,
                    strategy_version_id=version_id,
                )
                self.after(0, lambda: self._on_build_done(result, None))
            except Exception as exc:
                self.after(0, lambda: self._on_build_done(None, exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_build_done(self, result: dict[str, Any] | None, error: Exception | None) -> None:
        self._building = False
        self._progress.stop()
        self._build_btn.configure(state="normal")
        if error is not None:
            self._action_var.set(f"Failed: {error}")
            self._on_status("Confidence Labels · failed")
            messagebox.showerror("Confidence Labels", str(error), parent=self)
            self._reload()
            return
        if not result or not result.get("ok"):
            err = (result or {}).get("error") or "Label build failed."
            self._action_var.set(err)
            self._on_status("Confidence Labels · failed")
            messagebox.showerror("Confidence Labels", str(err), parent=self)
            self._reload()
            return

        n = int(result.get("rows_loaded") or 0)
        n_tgt = len(result.get("binary_columns") or [])
        self._action_var.set(
            f"Complete · {n:,} rows · {n_tgt} Replay-Based targets"
        )
        self._on_status("Confidence Labels · up to date")
        messagebox.showinfo(
            "Confidence Labels",
            (
                f"Built Replay-Based labels\n\n"
                f"Rows: {n:,}\n"
                f"Targets: {n_tgt}\n"
                f"Strategy: {result.get('strategy_display_name') or ''} "
                f"{result.get('strategy_version_label') or ''}\n\n"
                f"Rebuild the Confidence Dataset to include these labels for training."
            ),
            parent=self,
        )
        self._reload()
