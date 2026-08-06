"""Model training progress — Tk Create Model training panel."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from .training_dashboard_summary import build_config_summary_rows, format_config_summary, format_live_dashboard_section
from .runner import ModelTrainingRunner

_BASE_STEPS = (
    ("preparing_dataset", "Preparing dataset"),
    ("preparing_matrix", "Preparing matrix"),
)

_TAIL_STEPS = (
    ("evaluation", "Evaluation"),
    ("saving", "Saving package"),
)

_POST_TRAINING_STEPS = (
    ("post_training_importance", "Feature Importance"),
    ("post_training_distribution", "Feature Distribution"),
    ("post_training_drift", "Feature Drift"),
)

_WF_SUBSTAGES = (
    ("train_model", "Train fold model"),
    ("validation", "Validate fold"),
    ("shap_importance", "SHAP importance"),
)


def _fmt_num(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return "—"
    try:
        n = float(v)
        if digits == 0:
            return str(int(round(n)))
        return f"{n:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_dur(sec: Any) -> str:
    if sec is None:
        return "—"
    try:
        total = max(0, int(float(sec)))
    except (TypeError, ValueError):
        return "—"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if m:
        return f"{m:02d}:{s:02d}"
    return f"{s}s"


def _fmt_pct_improve(baseline: Any, current: Any) -> str:
    try:
        b, c = float(baseline), float(current)
        if b == 0:
            return "—"
        pct = ((c - b) / abs(b)) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _block_bar(pct: float, blocks: int = 20) -> str:
    filled = round(min(100, max(0, pct)) / 100 * blocks)
    return "█" * filled + "░" * (blocks - filled)


class ModelTrainingPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_done = on_done
        self._on_back = on_back
        self._runner = ModelTrainingRunner(chart_dir)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._config: dict[str, Any] = {}
        self._config_summary_rows: list[tuple[str, str]] = []
        self._model_name = ""
        self._train_steps: list[tuple[str, str]] = []
        self._step_widgets: dict[str, tuple[ttk.Label, ttk.Progressbar]] = {}
        self._progress: dict[str, Any] = {
            "steps": {},
            "dashboard": {},
            "wf": {"fold_results": [], "fs_complete": None},
            "hpo": {},
            "training_meta": {},
            "metrics": {},
        }
        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Model Training", font=("Segoe UI", 12, "bold")).pack(side="left")
        self._title_var = tk.StringVar(value="—")
        ttk.Label(header, textvariable=self._title_var, foreground="#58a6ff").pack(side="left", padx=12)
        self._status_var = tk.StringVar(value="Waiting to start…")
        ttk.Label(header, textvariable=self._status_var, foreground="#888").pack(side="left", padx=8)
        ttk.Button(header, text="← Back to Builder", command=self._go_back).pack(side="right", padx=4)
        self._cancel_btn = ttk.Button(header, text="Cancel", command=self._cancel, state="disabled")
        self._cancel_btn.pack(side="right")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body, padding=4)
        body.add(left, weight=3)
        right = ttk.Frame(body, padding=4)
        body.add(right, weight=2)

        self._steps_host = ttk.LabelFrame(left, text="Training Steps", padding=8)
        self._steps_host.pack(fill="x", pady=(0, 8))

        self._hpo_frame = ttk.LabelFrame(left, text="Hyperparameter Optimization", padding=8)
        self._hpo_stats_var = tk.StringVar(value="")
        ttk.Label(self._hpo_frame, textvariable=self._hpo_stats_var, justify="left", wraplength=520).pack(anchor="w")
        self._hpo_compare_var = tk.StringVar(value="")
        ttk.Label(self._hpo_frame, textvariable=self._hpo_compare_var, justify="left", wraplength=520).pack(anchor="w", pady=(6, 0))

        self._wf_frame = ttk.LabelFrame(left, text="Walk-Forward Pipeline", padding=8)
        wf_top = ttk.Frame(self._wf_frame)
        wf_top.pack(fill="x")
        self._wf_title_var = tk.StringVar(value="Walk-Forward Pipeline")
        ttk.Label(wf_top, textvariable=self._wf_title_var, font=("Segoe UI", 9, "bold")).pack(side="left")
        self._wf_pct_var = tk.StringVar(value="0%")
        ttk.Label(wf_top, textvariable=self._wf_pct_var, foreground="#58a6ff", font=("Segoe UI", 10, "bold")).pack(side="right")
        self._wf_bar_var = tk.StringVar(value=_block_bar(0))
        ttk.Label(self._wf_frame, textvariable=self._wf_bar_var, font=("Consolas", 10), foreground="#58a6ff").pack(anchor="w", pady=(4, 0))
        self._wf_overall = ttk.Progressbar(self._wf_frame, mode="determinate", maximum=100)
        self._wf_overall.pack(fill="x", pady=4)
        self._wf_detail_var = tk.StringVar(value="")
        ttk.Label(self._wf_frame, textvariable=self._wf_detail_var, justify="left", wraplength=520).pack(anchor="w")
        self._wf_fs_var = tk.StringVar(value="")
        ttk.Label(self._wf_frame, textvariable=self._wf_fs_var, justify="left", wraplength=520, foreground="#ffb74d").pack(anchor="w", pady=(6, 0))
        self._wf_folds_tree = ttk.Treeview(self._wf_frame, columns=(), show="headings", height=5)
        self._wf_folds_tree.pack(fill="x", pady=(6, 0))
        self._configure_wf_fold_columns(classification=False)

        log_frame = ttk.LabelFrame(left, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, pady=8)
        self._log = tk.Text(log_frame, height=12, font=("Consolas", 9), wrap="word")
        self._log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.configure(yscrollcommand=sb.set, state="disabled")

        metrics_nb = ttk.Notebook(right)
        metrics_nb.pack(fill="both", expand=True)
        self._metrics_nb = metrics_nb

        dash_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(dash_tab, text="Dashboard")
        self._dash_text = scrolledtext.ScrolledText(
            dash_tab,
            height=22,
            font=("Segoe UI", 9),
            wrap="word",
            state="disabled",
        )
        self._dash_text.pack(fill="both", expand=True)

        wf_val_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(wf_val_tab, text="WF Validation")
        self._wf_val_tab = wf_val_tab
        self._wf_val_metrics_var = tk.StringVar(
            value="Available after walk-forward completes…"
        )
        ttk.Label(
            wf_val_tab,
            textvariable=self._wf_val_metrics_var,
            justify="left",
            wraplength=320,
        ).pack(anchor="w")
        self._wf_val_hint_var = tk.StringVar(
            value=(
                "Mean MAE / RMSE / Dir % across walk-forward fold validations.\n"
                "Not holdout test — see Test Metrics after Evaluation."
            )
        )
        ttk.Label(
            wf_val_tab,
            textvariable=self._wf_val_hint_var,
            foreground="#888",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", pady=(10, 0))

        test_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(test_tab, text="Test Metrics")
        self._test_metrics_var = tk.StringVar(value="Available after evaluation…")
        ttk.Label(test_tab, textvariable=self._test_metrics_var, justify="left", wraplength=300).pack(anchor="w")

        val_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(val_tab, text="Validation")
        self._val_metrics_var = tk.StringVar(value="Available during training…")
        ttk.Label(val_tab, textvariable=self._val_metrics_var, justify="left", wraplength=300).pack(anchor="w")

        chart_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(chart_tab, text="Loss Curve")
        self._loss_canvas = tk.Canvas(chart_tab, height=180, bg="#1a1f2b", highlightthickness=0)
        self._loss_canvas.pack(fill="both", expand=True)
        self._loss_empty = ttk.Label(chart_tab, text="Available after training completes…", foreground="#888")
        self._loss_empty.place(relx=0.5, rely=0.5, anchor="center")

        imp_tab = ttk.Frame(metrics_nb, padding=6)
        metrics_nb.add(imp_tab, text="Importance")
        imp_cols = ("feature", "importance")
        self._imp_tree = ttk.Treeview(imp_tab, columns=imp_cols, show="headings", height=14)
        self._imp_tree.heading("feature", text="Feature")
        self._imp_tree.heading("importance", text="Importance")
        self._imp_tree.column("feature", width=180)
        self._imp_tree.column("importance", width=90)
        self._imp_tree.pack(fill="both", expand=True)

        self._completion = ttk.LabelFrame(right, text="Complete", padding=8)
        self._completion_var = tk.StringVar(value="")
        ttk.Label(self._completion, textvariable=self._completion_var, wraplength=300, justify="left").pack(anchor="w")
        self._open_registry_btn = ttk.Button(
            self._completion,
            text="Open Model Registry",
            command=self._open_registry,
            state="disabled",
        )
        self._open_registry_btn.pack(anchor="w", pady=(8, 0))
        self._completion.pack_forget()

    def _configure_steps(self, config: dict[str, Any]) -> None:
        steps = list(_BASE_STEPS)
        split = config.get("split") or {}
        global_hpo = split.get("hyperparameter_optimization") or {}
        hpo_on = bool(global_hpo.get("enabled"))
        is_wf = split.get("strategy") == "walk_forward"
        if is_wf:
            steps.append(("walk_forward", "Walk-forward pipeline"))
            if hpo_on or (split.get("walk_forward") or {}).get("hyperparameter_optimization", {}).get("enabled"):
                steps.append(("hyperparameter_optimization", "Hyperparameter optimization"))
        elif hpo_on:
            steps.append(("hyperparameter_optimization", "Hyperparameter optimization"))
        train_label = "Final model training" if is_wf else "Training model"
        steps.append(("training", train_label))
        steps.extend(_TAIL_STEPS)
        pt = config.get("post_training") if isinstance(config.get("post_training"), dict) else {}
        pt_enabled = bool(pt.get("enabled", True))
        if pt_enabled:
            for step_id, label in _POST_TRAINING_STEPS:
                key = step_id.replace("post_training_", "", 1)
                if bool(pt.get(key, True)):
                    steps.append((step_id, label))
        self._train_steps = steps
        for w in self._steps_host.winfo_children():
            w.destroy()
        self._step_widgets.clear()
        for step_id, label in steps:
            row = ttk.Frame(self._steps_host)
            row.pack(fill="x", pady=2)
            lbl = ttk.Label(row, text=label, width=24)
            lbl.pack(side="left")
            bar = ttk.Progressbar(row, mode="determinate", maximum=100, length=200)
            bar.pack(side="left", fill="x", expand=True, padx=8)
            self._step_widgets[step_id] = (lbl, bar)

    def _is_classification(self) -> bool:
        pred = str(self._config.get("prediction_type") or "").strip().lower()
        return pred in ("binary", "classification", "multiclass")

    def _configure_wf_fold_columns(self, *, classification: bool) -> None:
        """Swap Walk-Forward fold table headers for regression vs classification."""
        tree = self._wf_folds_tree
        # Clear existing column config
        tree.configure(columns=())
        if classification:
            cols = (
                ("fold", 44, "Fold"),
                ("accuracy", 72, "Accuracy"),
                ("precision", 72, "Precision"),
                ("recall", 64, "Recall"),
                ("f1", 56, "F1"),
                ("auc", 56, "AUC"),
                ("composite", 72, "Composite"),
                ("features", 56, "Feats"),
            )
        else:
            cols = (
                ("fold", 44, "Fold"),
                ("mae", 72, "MAE"),
                ("rmse", 72, "RMSE"),
                ("dir", 56, "Dir %"),
                ("composite", 72, "Composite"),
                ("features", 56, "Feats"),
            )
        names = tuple(c[0] for c in cols)
        tree.configure(columns=names)
        for c, w, label in cols:
            tree.heading(c, text=label)
            tree.column(c, width=w, anchor="center")

    def start(self, config: dict[str, Any]) -> None:
        if self._runner.running:
            messagebox.showwarning("Training", "A training job is already running.")
            return
        self._config = dict(config)
        self._config_summary_rows = build_config_summary_rows(self._config)
        self._model_name = str(config.get("model_name") or "")
        self._title_var.set(self._model_name or "—")
        self._progress = {
            "steps": {},
            "dashboard": {},
            "wf": {"fold_results": [], "fs_complete": None},
            "hpo": self._init_hpo_state(config),
            "training_meta": {},
            "metrics": {},
            "walk_forward": {},
        }
        self._configure_steps(config)
        self._configure_wf_fold_columns(classification=self._is_classification())
        if self._is_classification():
            self._wf_val_hint_var.set(
                "Mean Accuracy / Precision / Recall / F1 / AUC across walk-forward folds.\n"
                "Not holdout test — see Test Metrics after Evaluation."
            )
        else:
            self._wf_val_hint_var.set(
                "Mean MAE / RMSE / Dir % across walk-forward fold validations.\n"
                "Not holdout test — see Test Metrics after Evaluation."
            )
        self._log_clear()
        self._completion.pack_forget()
        self._open_registry_btn.configure(state="disabled")
        self._hpo_frame.pack_forget()
        self._wf_frame.pack_forget()
        self._status_var.set("Starting…")
        self._render_dashboard()
        self._wf_val_metrics_var.set("Available after walk-forward completes…")
        self._test_metrics_var.set("Available after evaluation…")
        self._val_metrics_var.set("Available during training…")
        self._hpo_stats_var.set("")
        self._hpo_compare_var.set("")
        self._wf_folds_tree.delete(*self._wf_folds_tree.get_children())
        self._imp_tree.delete(*self._imp_tree.get_children())
        self._loss_canvas.delete("all")
        self._loss_empty.place(relx=0.5, rely=0.5, anchor="center")
        for _sid, (lbl, bar) in self._step_widgets.items():
            lbl.configure(foreground="#666")
            bar.configure(value=0)
        lc_mode = (config.get("lifecycle") or {}).get("mode")
        if lc_mode == "complete_optimization":
            self._hpo_frame.pack(fill="x", pady=(0, 8), before=self._wf_frame)
            self._render_hpo_panel()
        self._cancel_btn.configure(state="normal")
        self._runner.start(
            config,
            on_progress=self._on_progress,
            on_done=self._on_train_done,
        )
        self.after(150, self._poll_queue)

    def _init_hpo_state(self, config: dict[str, Any]) -> dict[str, Any]:
        lc = config.get("lifecycle") or {}
        src = lc.get("source_metrics") or {}
        split = config.get("split") or {}
        hpo = split.get("hyperparameter_optimization") or {}
        wf_hpo = (split.get("walk_forward") or {}).get("hyperparameter_optimization") or {}
        n_trials = hpo.get("n_trials") or wf_hpo.get("n_trials") or 25
        return {
            "baseline_composite": src.get("composite_score"),
            "baseline_mae": src.get("mae"),
            "baseline_dir": src.get("directional_accuracy_pct"),
            "current_best": None,
            "best_trial": None,
            "trial": 0,
            "n_trials": n_trials,
            "best_mae": None,
            "best_dir": None,
        }

    def _on_progress(self, payload: dict[str, Any]) -> None:
        self._queue.put(("progress", payload))

    def _on_train_done(self, result: dict[str, Any]) -> None:
        self._queue.put(("done", result))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    self._apply_progress(payload)
                elif kind == "done":
                    self._apply_done(payload)
        except queue.Empty:
            pass
        if self._runner.running:
            self.after(150, self._poll_queue)
        else:
            self._cancel_btn.configure(state="disabled")

    def _apply_progress(self, msg: dict[str, Any]) -> None:
        if msg.get("elapsed_sec") is not None:
            self._progress["dashboard"]["elapsed_sec"] = msg["elapsed_sec"]
        if msg.get("phase") == "dashboard":
            self._progress["dashboard"].update(msg)
            self._render_dashboard()
            return

        stage = str(msg.get("stage") or "")
        step = str(msg.get("step") or "")
        # Post-training emits stage=post_training_* (no step key).
        if not step and stage.startswith("post_training"):
            step = stage
        status = str(msg.get("status") or "").lower()
        if status == "started":
            status = "running"
        detail = str(msg.get("detail") or msg.get("message") or "")
        if detail:
            self._log_append(detail)

        if step:
            self._progress["steps"][step] = status or "running"
            if step in ("walk_forward", "hyperparameter_optimization"):
                self._progress["wf"].update(msg)
                if msg.get("fold_result"):
                    fr = msg["fold_result"]
                    rows = self._progress["wf"]["fold_results"]
                    idx = next((i for i, r in enumerate(rows) if r.get("fold") == fr.get("fold")), -1)
                    if idx >= 0:
                        rows[idx] = fr
                    else:
                        rows.append(fr)
                    rows.sort(key=lambda r: r.get("fold") or 0)
                if msg.get("fs_complete"):
                    self._progress["wf"]["fs_complete"] = dict(msg)
                if step == "hyperparameter_optimization":
                    self._update_hpo_from_msg(msg)
            if step == "training" and msg.get("training_meta"):
                self._progress["training_meta"].update(msg["training_meta"])
            if step == "evaluation" and msg.get("metrics"):
                self._progress["metrics"] = msg["metrics"]
            if status == "done" and step == "walk_forward":
                if msg.get("walk_forward"):
                    self._progress["walk_forward"] = msg["walk_forward"]
                if msg.get("feature_selection"):
                    self._progress["feature_selection"] = msg["feature_selection"]
                self._render_wf_validation_metrics(msg)
                self._render_dashboard()
                try:
                    self._metrics_nb.select(self._wf_val_tab)
                except tk.TclError:
                    pass
            if status == "done" and step == "hyperparameter_optimization" and msg.get("hyperparameter_optimization"):
                self._update_hpo_from_done(msg["hyperparameter_optimization"])
            if status == "done" and step == "training":
                if msg.get("validation_loss_curve"):
                    self._render_loss_curve(msg["validation_loss_curve"])
                self._render_validation_metrics(msg.get("training_meta"))
            if status == "done" and step == "evaluation":
                self._render_test_metrics(msg.get("metrics"))
                # Keep WF Validation panel; Test Metrics is holdout-only.
                if self._progress.get("walk_forward"):
                    self._render_wf_validation_metrics({})
            if msg.get("feature_importance"):
                self._render_importance(msg.get("feature_importance_top20") or msg["feature_importance"])

        if step in self._step_widgets:
            lbl, bar = self._step_widgets[step]
            pct = self._step_progress_pct(step, status, msg)
            if status == "running":
                lbl.configure(foreground="#1976D2")
                bar.configure(value=max(pct, 5))
            elif status in ("done", "pass", "ok", "completed"):
                lbl.configure(foreground="#2E7D32")
                bar.configure(value=100)
            elif status in ("fail", "error", "failed", "partial"):
                lbl.configure(foreground="#C62828")
                if status == "partial":
                    bar.configure(value=max(pct, 50))

        if step == "walk_forward" and status == "running":
            self._wf_frame.pack(fill="x", pady=(0, 8), before=self._log.master)
            self._status_var.set(
                f"Walk-forward · Fold {msg.get('fold', '…')}/{msg.get('n_folds', '…')} · "
                f"{str(msg.get('wf_stage', '')).replace('_', ' ')}"
            )
        elif step == "hyperparameter_optimization" and status == "running":
            self._wf_frame.pack(fill="x", pady=(0, 8), before=self._log.master)
            self._status_var.set(
                f"HPO trial {msg.get('trial', '…')}/{msg.get('n_trials', '…')}"
                + (f" · score {_fmt_num(msg.get('display_score'))}" if msg.get("display_score") is not None else "")
            )
        elif step == "training" and status == "running":
            trees = msg.get("current_tree")
            total = msg.get("trees_total")
            if trees is not None and total:
                self._progress["dashboard"]["train_progress_pct"] = int((trees / total) * 100)
            self._status_var.set(f"Training · tree {trees or '…'}/{total or '…'}")
        elif step.startswith("post_training") and status == "running":
            label = {
                "post_training": "Feature Studio",
                "post_training_importance": "Feature Importance",
                "post_training_distribution": "Feature Distribution",
                "post_training_drift": "Feature Drift",
            }.get(step, step.replace("post_training_", "").replace("_", " ").title())
            self._status_var.set(f"Post-training · {label}")
        elif step == "post_training" and status in ("done", "completed", "partial", "failed"):
            self._status_var.set(f"Post-training · {status}")

        self._progress["dashboard"].update({k: v for k, v in msg.items() if k not in ("phase",)})
        self._render_wf_panel()
        self._render_hpo_panel()
        self._render_dashboard()

    def _step_progress_pct(self, step: str, status: str, msg: dict[str, Any]) -> int:
        if status == "done":
            return 100
        if status != "running":
            return 0
        if step == "walk_forward":
            pct = msg.get("wf_overall_pct")
            return int(pct) if pct is not None else 20
        if step == "hyperparameter_optimization":
            pct = msg.get("hpo_overall_pct")
            return int(pct) if pct is not None else 15
        if step == "training":
            pct = self._progress["dashboard"].get("train_progress_pct")
            return int(pct) if pct is not None else 15
        if step == "saving":
            return 92
        return 40

    def _update_hpo_from_msg(self, msg: dict[str, Any]) -> None:
        hpo = self._progress["hpo"]
        if msg.get("baseline_composite_score") is not None:
            hpo["baseline_composite"] = msg["baseline_composite_score"]
        if msg.get("best_display_score") is not None:
            hpo["current_best"] = msg["best_display_score"]
        if msg.get("best_trial") is not None:
            hpo["best_trial"] = msg["best_trial"]
        if msg.get("trial") is not None:
            hpo["trial"] = msg["trial"]
        if msg.get("n_trials") is not None:
            hpo["n_trials"] = msg["n_trials"]
        if msg.get("best_mean_mae") is not None:
            hpo["best_mae"] = msg["best_mean_mae"]
        if msg.get("best_mean_directional_accuracy_pct") is not None:
            hpo["best_dir"] = msg["best_mean_directional_accuracy_pct"]

    def _update_hpo_from_done(self, hpo_doc: dict[str, Any]) -> None:
        hpo = self._progress["hpo"]
        baseline = hpo_doc.get("baseline_evaluation") or {}
        best = hpo_doc.get("best_evaluation") or {}
        if baseline.get("display_score") is not None:
            hpo["baseline_composite"] = baseline["display_score"]
        if best.get("display_score") is not None:
            hpo["current_best"] = best["display_score"]
        if best.get("mean_mae") is not None:
            hpo["best_mae"] = best["mean_mae"]
        if best.get("mean_directional_accuracy_pct") is not None:
            hpo["best_dir"] = best["mean_directional_accuracy_pct"]
        if hpo_doc.get("best_trial") is not None:
            hpo["best_trial"] = int(hpo_doc["best_trial"]) + 1
        if hpo_doc.get("n_trials") is not None:
            hpo["n_trials"] = hpo_doc["n_trials"]
            hpo["trial"] = hpo_doc["n_trials"]

    def _render_hpo_panel(self) -> None:
        lc_mode = (self._config.get("lifecycle") or {}).get("mode")
        if lc_mode != "complete_optimization":
            return
        h = self._progress["hpo"]
        lines = [
            f"Baseline composite: {_fmt_num(h.get('baseline_composite'))}",
            f"Best trial: {h.get('best_trial') if h.get('best_trial') is not None else '—'}",
            f"Current best: {_fmt_num(h.get('current_best'))}",
            f"Improvement: {_fmt_pct_improve(h.get('baseline_composite'), h.get('current_best'))}",
            f"Trial: {h.get('trial', 0)} / {h.get('n_trials', '—')}",
        ]
        self._hpo_stats_var.set("\n".join(lines))
        compare = (
            f"Source — MAE {_fmt_num(h.get('baseline_mae'), 2)} · "
            f"Dir {_fmt_num(h.get('baseline_dir'), 2)}% · "
            f"Composite {_fmt_num(h.get('baseline_composite'))}\n"
            f"Best — MAE {_fmt_num(h.get('best_mae'), 2)} · "
            f"Dir {_fmt_num(h.get('best_dir'), 2)}% · "
            f"Composite {_fmt_num(h.get('current_best'))}"
        )
        self._hpo_compare_var.set(compare)

    def _render_wf_panel(self) -> None:
        wf = self._progress["wf"]
        hpo_running = self._progress["steps"].get("hyperparameter_optimization") == "running"
        wf_running = self._progress["steps"].get("walk_forward") == "running"
        if not (hpo_running or wf_running or wf.get("fold_results") or wf.get("fs_complete")):
            return
        self._wf_frame.pack(fill="x", pady=(0, 8), before=self._log.master)
        is_hpo = hpo_running
        pct = wf.get("hpo_overall_pct") if is_hpo else wf.get("wf_overall_pct", 0)
        pct = int(pct or 0)
        self._wf_title_var.set("Hyperparameter Optimization" if is_hpo else "Walk-Forward Pipeline")
        self._wf_pct_var.set(f"{pct}%")
        self._wf_bar_var.set(_block_bar(pct))
        self._wf_overall.configure(value=pct)

        phase = wf.get("wf_phase") or ("hyperparameter_optimization" if is_hpo else "fold_validation")
        fold_txt = "—"
        if not is_hpo and phase != "feature_selection" and wf.get("fold") is not None:
            fold_txt = f"Fold {wf.get('fold')} / {wf.get('n_folds', '—')}"
        substage = self._wf_substage_text(wf.get("wf_stage"), phase)
        d = self._progress["dashboard"]
        elapsed = d.get("elapsed_sec")
        detail_lines = [
            f"Current fold: {fold_txt}",
            f"Current phase: {phase.replace('_', ' ').title()}",
            f"Sub-stage: {substage}",
            f"Features: {wf.get('current_features') or wf.get('feature_count') or '—'}",
            f"Trial: {wf.get('current_trial', '—')} / {wf.get('n_trials', '—')}" if is_hpo else "",
            f"Elapsed: {_fmt_dur(elapsed)}",
        ]
        self._wf_detail_var.set("\n".join(line for line in detail_lines if line))

        if phase == "feature_selection" and wf.get("wf_stage") == "feature_selection" and not wf.get("fs_complete"):
            removed = wf.get("removed_features") or []
            fs_lines = [
                f"Iteration {wf.get('current_iteration', '—')} · evaluating {wf.get('current_features', '—')} features",
            ]
            if wf.get("best_features_count") is not None:
                fs_lines.append(f"Best so far {wf['best_features_count']} features · score {_fmt_num(wf.get('best_display_score'))}")
            if removed:
                fs_lines.append(f"Removing {len(removed)} feature(s): {', '.join(str(x) for x in removed[:5])}")
            fs_lines.append(f"Validation RMSE {_fmt_num(wf.get('validation_rmse'))}")
            self._wf_fs_var.set("\n".join(fs_lines))
        elif wf.get("fs_complete") or self._progress["wf"].get("fs_complete"):
            fs = wf.get("fs_complete") or self._progress["wf"].get("fs_complete") or {}
            self._wf_fs_var.set(
                f"Feature selection complete · started {fs.get('started_features', '—')} → "
                f"selected {fs.get('selected_features', fs.get('finished_features', '—'))} features"
            )
        else:
            self._wf_fs_var.set("")

        self._render_wf_fold_table(wf)

    def _wf_n_folds(self, wf: dict[str, Any]) -> int:
        raw = wf.get("n_folds")
        if raw is None:
            wf_cfg = (self._config.get("split") or {}).get("walk_forward") or {}
            raw = wf_cfg.get("n_folds")
        try:
            n_folds = int(raw)
        except (TypeError, ValueError):
            n_folds = 0
        if n_folds <= 0:
            results = wf.get("fold_results") or []
            fold_nums = [int(r.get("fold")) for r in results if r.get("fold") is not None]
            n_folds = max(fold_nums) if fold_nums else 0
        return max(0, n_folds)

    def _metric_from_fold(self, fr: dict[str, Any], *keys: str) -> Any:
        """Pull a metric from fold_result top-level or nested metrics dict."""
        metrics = fr.get("metrics") if isinstance(fr.get("metrics"), dict) else {}
        for key in keys:
            if fr.get(key) is not None:
                return fr.get(key)
            if metrics.get(key) is not None:
                return metrics.get(key)
        return None

    def _render_wf_fold_table(self, wf: dict[str, Any]) -> None:
        n_folds = self._wf_n_folds(wf)
        results_by_fold: dict[int, dict[str, Any]] = {}
        for row in wf.get("fold_results") or []:
            fold_num = row.get("fold")
            if fold_num is None:
                continue
            try:
                results_by_fold[int(fold_num)] = row
            except (TypeError, ValueError):
                continue

        self._wf_folds_tree.configure(height=max(min(n_folds, 12), 3) if n_folds else 3)
        self._wf_folds_tree.delete(*self._wf_folds_tree.get_children())
        cls = self._is_classification()
        for fold_num in range(1, n_folds + 1):
            fr = results_by_fold.get(fold_num) or {}
            has_result = bool(fr)
            feats = fr.get("feature_count") if fr.get("feature_count") is not None else "—"
            if cls:
                self._wf_folds_tree.insert(
                    "",
                    "end",
                    values=(
                        fold_num,
                        f"{_fmt_num(self._metric_from_fold(fr, 'accuracy_pct'), 1)}%" if has_result else "—",
                        f"{_fmt_num(self._metric_from_fold(fr, 'precision_pct'), 1)}%" if has_result else "—",
                        f"{_fmt_num(self._metric_from_fold(fr, 'recall_pct'), 1)}%" if has_result else "—",
                        f"{_fmt_num(self._metric_from_fold(fr, 'f1_pct'), 1)}%" if has_result else "—",
                        _fmt_num(self._metric_from_fold(fr, "roc_auc"), 2) if has_result else "—",
                        _fmt_num(self._metric_from_fold(fr, "composite_score")) if has_result else "—",
                        feats,
                    ),
                )
            else:
                mae = self._metric_from_fold(fr, "mae")
                self._wf_folds_tree.insert(
                    "",
                    "end",
                    values=(
                        fold_num,
                        _fmt_num(mae) if has_result else "—",
                        _fmt_num(self._metric_from_fold(fr, "rmse")) if has_result else "—",
                        _fmt_num(self._metric_from_fold(fr, "directional_accuracy_pct"), 1) if has_result else "—",
                        _fmt_num(self._metric_from_fold(fr, "composite_score")) if has_result else "—",
                        feats,
                    ),
                )

    def _wf_substage_text(self, stage: str | None, phase: str) -> str:
        if phase == "feature_selection":
            return "Feature elimination in progress"
        if phase == "hyperparameter_optimization":
            return "Optuna trial evaluation"
        idx = next((i for i, (k, _) in enumerate(_WF_SUBSTAGES) if k == stage), -1)
        parts = []
        for i, (_, label) in enumerate(_WF_SUBSTAGES):
            if idx >= 0 and i < idx:
                parts.append(f"✓ {label}")
            elif idx >= 0 and i == idx:
                parts.append(f"⟳ {label}")
            else:
                parts.append(f"□ {label}")
        return " · ".join(parts)

    def _set_dashboard_text(self, text: str) -> None:
        self._dash_text.configure(state="normal")
        self._dash_text.delete("1.0", "end")
        self._dash_text.insert("end", text)
        self._dash_text.configure(state="disabled")

    def _render_dashboard(self) -> None:
        d = self._progress["dashboard"]
        done = self._progress["steps"].get("saving") == "done"
        wf_agg = self._wf_aggregate_for_ui()
        ram = "—"
        if d.get("ram_used_gb") is not None and d.get("ram_total_gb") is not None:
            ram = f"{_fmt_num(d['ram_used_gb'], 1)} / {_fmt_num(d['ram_total_gb'], 1)} GB"
        elif d.get("memory_mb") is not None:
            ram = f"{d['memory_mb']} MB"
        tree_line = "—"
        if d.get("current_tree") is not None:
            tree_line = f"{d['current_tree']} / {d.get('trees_total', '—')}"
        elif done and d.get("final_tree") is not None:
            tree_line = f"{d['final_tree']} / {d.get('trees_total', '—')}"
        x_shape = "—"
        if d.get("x_shape"):
            shape = d["x_shape"]
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                x_shape = f"{int(shape[0]):,} × {int(shape[1])}"
        if self._is_classification():
            live_rows = [
                ("Status", "Complete" if done else "Running"),
                ("Elapsed", _fmt_dur(d.get("elapsed_sec"))),
                ("CPU", f"{d.get('cpu_percent')}%" if d.get("cpu_percent") is not None else "—"),
                ("RAM", ram),
                ("GPU", f"{_fmt_num(d.get('gpu_percent'), 0)}%" if d.get("gpu_percent") is not None else "Unavailable"),
                ("Phase", d.get("training_phase") or ("Complete" if done else "—")),
                ("Trial", f"{d.get('current_trial')} / {d.get('total_trials')}" if d.get("current_trial") is not None else "—"),
                ("Trees", tree_line),
                ("Matrix shape", x_shape),
                ("Best iteration", d.get("best_iteration") or self._progress["training_meta"].get("best_iteration") or "—"),
                ("WF Accuracy", f"{_fmt_num(wf_agg.get('mean_accuracy_pct'), 2)}%"),
                ("WF Precision", f"{_fmt_num(wf_agg.get('mean_precision_pct'), 2)}%"),
                ("WF Recall", f"{_fmt_num(wf_agg.get('mean_recall_pct'), 2)}%"),
                ("WF F1", f"{_fmt_num(wf_agg.get('mean_f1_pct'), 2)}%"),
                ("WF AUC", _fmt_num(wf_agg.get("mean_roc_auc"), 3)),
                ("Test Accuracy", f"{_fmt_num(d.get('test_accuracy_pct'), 2)}%" if done else "—"),
            ]
        else:
            live_rows = [
                ("Status", "Complete" if done else "Running"),
                ("Elapsed", _fmt_dur(d.get("elapsed_sec"))),
                ("CPU", f"{d.get('cpu_percent')}%" if d.get("cpu_percent") is not None else "—"),
                ("RAM", ram),
                ("GPU", f"{_fmt_num(d.get('gpu_percent'), 0)}%" if d.get("gpu_percent") is not None else "Unavailable"),
                ("Phase", d.get("training_phase") or ("Complete" if done else "—")),
                ("Trial", f"{d.get('current_trial')} / {d.get('total_trials')}" if d.get("current_trial") is not None else "—"),
                ("Trees", tree_line),
                ("Matrix shape", x_shape),
                ("Best iteration", d.get("best_iteration") or self._progress["training_meta"].get("best_iteration") or "—"),
                ("Val RMSE", _fmt_num(d.get("validation_rmse"))),
                ("Train RMSE", _fmt_num(d.get("train_rmse"))),
                ("WF MAE", _fmt_num(wf_agg.get("mean_mae"))),
                ("WF RMSE", _fmt_num(wf_agg.get("mean_rmse"))),
                ("WF Dir %", _fmt_num(wf_agg.get("mean_directional_accuracy_pct"), 2)),
                ("Test RMSE", _fmt_num(d.get("test_rmse")) if done else "—"),
            ]
        prem_live = self._premium_filter_live_rows(d)
        if prem_live:
            # Insert after Phase so filter status is visible while the run is active.
            insert_at = next((i for i, (k, _) in enumerate(live_rows) if k == "Phase"), 0) + 1
            for offset, row in enumerate(prem_live):
                live_rows.insert(insert_at + offset, row)
        text = format_config_summary(self._config) + format_live_dashboard_section(live_rows)
        self._set_dashboard_text(text)

    def _premium_filter_live_rows(self, d: dict[str, Any]) -> list[tuple[str, str]]:
        """Live Premium Selection stats emitted after dataset load."""
        prem = d.get("premium_selection")
        if not isinstance(prem, dict):
            return []
        rows: list[tuple[str, str]] = []
        lo = prem.get("premium_min")
        hi = prem.get("premium_max")
        if lo is not None and hi is not None:
            rows.append(("Premium filter", f"LTP {lo:g}–{hi:g}"))
        before = prem.get("rows_before")
        after = prem.get("rows_after")
        dropped = prem.get("rows_dropped")
        if before is not None and after is not None:
            try:
                rows.append(
                    (
                        "Premium rows",
                        f"{int(after):,} kept / {int(before):,} "
                        f"({int(dropped or max(0, int(before) - int(after))):,} dropped)",
                    ),
                )
            except (TypeError, ValueError):
                rows.append(("Premium rows", f"{after} kept / {before}"))
        return rows

    def _wf_aggregate_for_ui(self, msg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Resolve walk-forward aggregate from progress event and/or fold table."""
        msg = msg or {}
        agg = dict(self._progress.get("walk_forward") or {})
        if msg.get("walk_forward") and isinstance(msg["walk_forward"], dict):
            agg = {**agg, **msg["walk_forward"]}
        for key in (
            "mean_mae",
            "mean_rmse",
            "mean_directional_accuracy_pct",
            "mean_accuracy_pct",
            "mean_precision_pct",
            "mean_recall_pct",
            "mean_f1_pct",
            "mean_roc_auc",
            "mean_composite_score",
            "n_folds",
            "std_mae",
            "std_rmse",
            "std_directional_accuracy_pct",
            "std_accuracy_pct",
            "std_f1_pct",
            "std_roc_auc",
        ):
            if msg.get(key) is not None and agg.get(key) is None:
                agg[key] = msg.get(key)
        # Fallback: mean of live fold rows if aggregate missing.
        folds = list((self._progress.get("wf") or {}).get("fold_results") or [])
        if folds:
            def _mean(key: str) -> float | None:
                vals = []
                for r in folds:
                    v = r.get(key)
                    if v is None and key == "directional_accuracy_pct":
                        v = r.get("dir")
                    if v is None:
                        m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                        v = m.get(key)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        continue
                return sum(vals) / len(vals) if vals else None

            agg.setdefault("n_folds", len(folds))
            agg.setdefault("mean_mae", _mean("mae"))
            agg.setdefault("mean_rmse", _mean("rmse"))
            agg.setdefault("mean_directional_accuracy_pct", _mean("directional_accuracy_pct"))
            agg.setdefault("mean_accuracy_pct", _mean("accuracy_pct"))
            agg.setdefault("mean_precision_pct", _mean("precision_pct"))
            agg.setdefault("mean_recall_pct", _mean("recall_pct"))
            agg.setdefault("mean_f1_pct", _mean("f1_pct"))
            agg.setdefault("mean_roc_auc", _mean("roc_auc"))
            agg.setdefault("mean_composite_score", _mean("composite_score"))
        return agg

    def _render_wf_validation_metrics(self, msg: dict[str, Any] | None = None) -> None:
        """Show walk-forward fold-validation averages (before holdout Evaluation)."""
        agg = self._wf_aggregate_for_ui(msg)
        self._progress["walk_forward"] = agg
        cls = self._is_classification()
        has_cls = agg.get("mean_accuracy_pct") is not None or agg.get("mean_f1_pct") is not None
        has_reg = agg.get("mean_mae") is not None or agg.get("mean_rmse") is not None
        if cls and not has_cls and not has_reg:
            self._wf_val_metrics_var.set("Available after walk-forward completes…")
            return
        if not cls and not has_reg:
            self._wf_val_metrics_var.set("Available after walk-forward completes…")
            return
        n_folds = agg.get("n_folds") or len((self._progress.get("wf") or {}).get("fold_results") or [])
        if cls:
            lines = [
                "Walk-forward validation (fold means)",
                f"Folds: {_fmt_num(n_folds, 0)}",
                "",
                f"Mean Accuracy: {_fmt_num(agg.get('mean_accuracy_pct'), 2)}%",
                f"Mean Precision: {_fmt_num(agg.get('mean_precision_pct'), 2)}%",
                f"Mean Recall: {_fmt_num(agg.get('mean_recall_pct'), 2)}%",
                f"Mean F1: {_fmt_num(agg.get('mean_f1_pct'), 2)}%",
                f"Mean AUC: {_fmt_num(agg.get('mean_roc_auc'), 3)}",
                f"Mean Composite: {_fmt_num(agg.get('mean_composite_score'))}",
            ]
            if agg.get("std_accuracy_pct") is not None or agg.get("std_f1_pct") is not None:
                lines.extend(
                    [
                        "",
                        f"Std Accuracy: {_fmt_num(agg.get('std_accuracy_pct'), 2)}%",
                        f"Std F1: {_fmt_num(agg.get('std_f1_pct'), 2)}%",
                        f"Std AUC: {_fmt_num(agg.get('std_roc_auc'), 3)}",
                    ]
                )
        else:
            lines = [
                "Walk-forward validation (fold means)",
                f"Folds: {_fmt_num(n_folds, 0)}",
                "",
                f"Mean MAE: {_fmt_num(agg.get('mean_mae'))}",
                f"Mean RMSE: {_fmt_num(agg.get('mean_rmse'))}",
                f"Mean Dir %: {_fmt_num(agg.get('mean_directional_accuracy_pct'), 2)}%",
                f"Mean Composite: {_fmt_num(agg.get('mean_composite_score'))}",
            ]
            if agg.get("std_rmse") is not None or agg.get("std_mae") is not None:
                lines.extend(
                    [
                        "",
                        f"Std MAE: {_fmt_num(agg.get('std_mae'))}",
                        f"Std RMSE: {_fmt_num(agg.get('std_rmse'))}",
                        f"Std Dir %: {_fmt_num(agg.get('std_directional_accuracy_pct'), 2)}%",
                    ]
                )
        fs = self._progress.get("feature_selection") or {}
        if fs.get("n_selected") is not None or fs.get("selected_count") is not None:
            lines.extend(
                [
                    "",
                    f"Features after selection: "
                    f"{_fmt_num(fs.get('n_selected') or fs.get('selected_count'), 0)}",
                ]
            )
        lines.extend(
            [
                "",
                "Per-fold detail is in the Metrics table on the left.",
                "Holdout Test Metrics appears after Evaluation.",
            ]
        )
        self._wf_val_metrics_var.set("\n".join(lines))

    def _render_validation_metrics(self, training_meta: dict[str, Any] | None) -> None:
        meta = training_meta or self._progress["training_meta"]
        val = (self._progress["metrics"] or {}).get("validation") or {}
        if self._is_classification():
            lines = [
                f"Validation Accuracy: {_fmt_num(val.get('accuracy_pct'), 2)}%",
                f"Validation Precision: {_fmt_num(val.get('precision_pct'), 2)}%",
                f"Validation Recall: {_fmt_num(val.get('recall_pct'), 2)}%",
                f"Validation F1: {_fmt_num(val.get('f1_pct'), 2)}%",
                f"Validation AUC: {_fmt_num(val.get('roc_auc'), 3)}",
                f"Best iteration: {_fmt_num(meta.get('best_iteration'), 0)}",
                f"Early stopping rounds: {_fmt_num(meta.get('early_stopping_rounds'), 0)}",
            ]
        else:
            lines = [
                f"Best validation RMSE: {_fmt_num(meta.get('best_validation_rmse') or val.get('rmse'))}",
                f"Validation MAE: {_fmt_num(val.get('mae'))}",
                f"Validation R²: {_fmt_num(val.get('r2'))}",
                f"Directional accuracy: {_fmt_num(val.get('directional_accuracy_pct'), 2)}%",
                f"Best iteration: {_fmt_num(meta.get('best_iteration'), 0)}",
                f"Early stopping rounds: {_fmt_num(meta.get('early_stopping_rounds'), 0)}",
                f"Training RMSE: {_fmt_num(meta.get('train_rmse'))}",
            ]
        self._val_metrics_var.set("\n".join(lines))

    def _render_test_metrics(self, metrics: dict[str, Any] | None) -> None:
        test = (metrics or {}).get("test") or {}
        if not test:
            return
        if self._is_classification():
            lines = [
                f"Accuracy: {_fmt_num(test.get('accuracy_pct'), 2)}%",
                f"Precision: {_fmt_num(test.get('precision_pct'), 2)}%",
                f"Recall: {_fmt_num(test.get('recall_pct'), 2)}%",
                f"F1: {_fmt_num(test.get('f1_pct'), 2)}%",
                f"AUC: {_fmt_num(test.get('roc_auc'), 3)}",
                f"PR-AUC: {_fmt_num(test.get('pr_auc'), 3)}",
                f"Brier: {_fmt_num(test.get('brier_score'), 4)}",
            ]
        else:
            lines = [
                f"RMSE: {_fmt_num(test.get('rmse'))}",
                f"MAE: {_fmt_num(test.get('mae'))}",
                f"MAPE: {_fmt_num(test.get('mape'))}",
                f"R²: {_fmt_num(test.get('r2'))}",
                f"Directional accuracy: {_fmt_num(test.get('directional_accuracy_pct'), 2)}%",
                f"Median error: {_fmt_num(test.get('median_error'))}",
                f"Max error: {_fmt_num(test.get('max_error'))}",
            ]
        self._test_metrics_var.set("\n".join(lines))

    def _render_loss_curve(self, curve: list[dict[str, Any]]) -> None:
        if not curve:
            return
        self._loss_empty.place_forget()
        c = self._loss_canvas
        c.update_idletasks()
        w = max(c.winfo_width(), 280)
        h = max(c.winfo_height(), 160)
        c.delete("all")
        pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 28
        inner_w = w - pad_l - pad_r
        inner_h = h - pad_t - pad_b
        vals = []
        for row in curve:
            if row.get("train_rmse") is not None:
                vals.append(float(row["train_rmse"]))
            if row.get("validation_rmse") is not None:
                vals.append(float(row["validation_rmse"]))
        if not vals:
            return
        min_y, max_y = min(vals), max(vals)
        span_y = max_y - min_y or 1.0
        max_x = max(int(row.get("iteration") or 0) for row in curve) or 1

        def x_at(i: int) -> float:
            return pad_l + ((i - 1) / max(max_x - 1, 1)) * inner_w

        def y_at(v: float) -> float:
            return pad_t + inner_h - ((v - min_y) / span_y) * inner_h

        c.create_line(pad_l, pad_t + inner_h, w - pad_r, pad_t + inner_h, fill="#2a3142")
        c.create_line(pad_l, pad_t, pad_l, pad_t + inner_h, fill="#2a3142")
        for key, color in (("train_rmse", "#1976d2"), ("validation_rmse", "#ffb74d")):
            pts = []
            for row in curve:
                if row.get(key) is None:
                    continue
                pts.extend((x_at(int(row.get("iteration") or 1)), y_at(float(row[key]))))
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)
        best_iter = self._progress["training_meta"].get("best_iteration")
        if best_iter:
            bx = x_at(int(best_iter))
            c.create_line(bx, pad_t, bx, pad_t + inner_h, fill="#81c784", dash=(4, 3))

    def _render_importance(self, rows: list[dict[str, Any]]) -> None:
        self._imp_tree.delete(*self._imp_tree.get_children())
        for row in rows[:30]:
            feat = row.get("feature") or row.get("name") or ""
            imp = row.get("importance") or row.get("gain") or row.get("value")
            self._imp_tree.insert("", "end", values=(feat, _fmt_num(imp, 4)))

    def _apply_done(self, result: dict[str, Any]) -> None:
        self._cancel_btn.configure(state="disabled")
        if result.get("cancelled"):
            self._log_append("Training cancelled.")
            self._status_var.set("Cancelled")
            return
        if result.get("blocked"):
            self._log_append("Training blocked by validation.")
            validation = result.get("validation") or {}
            checks = validation.get("checks") or []
            for c in checks:
                if not c.get("passed"):
                    self._log_append(f"✗ {c.get('label')}: {c.get('detail', '')}")
            missing = list(validation.get("missing_features") or [])
            if missing:
                self._log_append(
                    f"Missing from dataset ({len(missing)}): "
                    + ", ".join(missing[:40])
                    + (f", … +{len(missing) - 40} more" if len(missing) > 40 else "")
                )
            messagebox.showerror("Training Blocked", "Pre-training validation failed. See log.")
            self._status_var.set("Blocked")
            return
        if not result.get("ok"):
            err = result.get("error") or result.get("detail") or "Training failed"
            self._log_append(str(err))
            messagebox.showerror("Training Failed", str(err))
            self._status_var.set("Failed")
            return
        name = result.get("model_name") or self._model_name
        self._model_name = str(name or "")
        lc_mode = (self._config.get("lifecycle") or {}).get("mode")
        pt = result.get("post_training") if isinstance(result.get("post_training"), dict) else None
        pt_line = self._format_post_training_summary(pt)
        if lc_mode == "complete_optimization":
            h = self._progress["hpo"]
            completion = (
                "Optimization complete\n"
                f"Model: {self._model_name}\n"
                f"Composite {_fmt_num(h.get('baseline_composite'))} → {_fmt_num(h.get('current_best'))}"
            )
        else:
            completion = f"Training complete\nModel: {self._model_name}"
        if pt_line:
            completion = f"{completion}\n{pt_line}"
        self._completion_var.set(completion)
        self._completion.pack(fill="x", pady=8)
        self._open_registry_btn.configure(state="normal")
        self._status_var.set("Complete")
        self._log_append(f"Done — saved {self._model_name}")
        if pt_line:
            self._log_append(pt_line)
            self._mark_post_training_steps(pt)
        if result.get("metrics"):
            metrics = result["metrics"]
            self._progress["metrics"] = metrics
            if metrics.get("walk_forward"):
                self._progress["walk_forward"] = dict(metrics["walk_forward"])
                self._render_wf_validation_metrics({})
            self._render_test_metrics(metrics)
            self._render_dashboard()
        if result.get("feature_importance"):
            self._render_importance(result.get("feature_importance_top20") or result["feature_importance"])
        if result.get("validation_loss_curve"):
            self._render_loss_curve(result["validation_loss_curve"])
        if self._on_done:
            self._on_done(result)

    @staticmethod
    def _format_post_training_summary(pt: dict[str, Any] | None) -> str:
        if not pt:
            return ""
        status = str(pt.get("status") or "").strip() or "unknown"
        if status == "skipped":
            return "Feature Studio: skipped"
        parts: list[str] = []
        for key, label in (
            ("importance", "Importance"),
            ("distribution", "Distribution"),
            ("drift", "Drift"),
        ):
            st = str(pt.get(key) or "").strip()
            stages = pt.get("stages") if isinstance(pt.get("stages"), dict) else {}
            if not st and isinstance(stages.get(key), dict):
                st = str(stages[key].get("status") or "").strip()
            glyph = {"completed": "✓", "failed": "✗", "skipped": "–"}.get(st, "·")
            parts.append(f"{glyph} {label}")
        dur = pt.get("duration_sec")
        dur_s = f" · {float(dur):.1f}s" if isinstance(dur, (int, float)) else ""
        return f"Feature Studio ({status}){dur_s}: {' · '.join(parts)}"

    def _mark_post_training_steps(self, pt: dict[str, Any] | None) -> None:
        if not pt:
            return
        stages = pt.get("stages") if isinstance(pt.get("stages"), dict) else {}
        for key in ("importance", "distribution", "drift"):
            step = f"post_training_{key}"
            if step not in self._step_widgets:
                continue
            st = str(pt.get(key) or "").strip().lower()
            if not st and isinstance(stages.get(key), dict):
                st = str(stages[key].get("status") or "").strip().lower()
            lbl, bar = self._step_widgets[step]
            if st == "completed":
                lbl.configure(foreground="#2E7D32")
                bar.configure(value=100)
            elif st in ("failed", "error"):
                lbl.configure(foreground="#C62828")
                bar.configure(value=100)
            elif st == "skipped":
                lbl.configure(foreground="#888")
                bar.configure(value=0)

    def _cancel(self) -> None:
        if self._runner.running:
            self._runner.cancel()
            self._log_append("Cancel requested…")

    def _go_back(self) -> None:
        if self._runner.running:
            if not messagebox.askyesno("Training", "Training is running. Cancel and go back?"):
                return
            self._runner.cancel()
        if self._on_back:
            self._on_back()

    def _open_registry(self) -> None:
        if self._on_done and self._model_name:
            self._on_done({"ok": True, "model_name": self._model_name, "open_registry": True})

    def _log_append(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text.rstrip() + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_clear(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
