"""Fold Comparison panel — compare two walk-forward folds of one model."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .build_service import chart_data_dir
from .fold_replay_widgets import draw_bucket_bars
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import (
    ACCENT,
    COL_HOLDOUT,
    COL_MUTED,
    COL_OK,
    COL_WARN,
    ScrollableFrame,
    clear_children,
    data_table,
    fmt_num,
    fmt_pct,
    fmt_rows,
    fmt_rupee,
    fmt_val,
    section_desc,
    section_title,
    BODY_FONT,
)
from .ui_state import get_ui_state_manager


class FoldComparisonPanel(ttk.Frame, LazyLoadMixin):
    """Compare Fold A vs Fold B for a single walk-forward model."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._model_names: list[str] = []
        self._fold_ids: list[int] = []
        self._current_doc: dict[str, Any] | None = None
        self._fold_pick_step = 0  # 0 → next table click sets Fold A, 1 → Fold B
        self._status_var = tk.StringVar(
            value="Select a walk-forward model, pick two folds, then Compare."
        )
        self._model_var = tk.StringVar()
        self._fold_a_var = tk.StringVar()
        self._fold_b_var = tk.StringVar()
        self._tab_scrolls: dict[str, ScrollableFrame] = {}
        self._last_report: dict[str, Any] | None = None
        self._ui_state = get_ui_state_manager()
        self._build_ui()
        self._ui_state.bind_notebook(self._notebook, "fold_comparison.tab")
        self._lazy_init()
        self._update_compare_enabled()
        self._update_csv_enabled()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def refresh(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_model_names,
                apply=self._apply_model_names,
                message="Loading models…",
                status_var=self._status_var,
            )
            return
        try:
            names = self._fetch_model_names()
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_model_names(names)

    def _fetch_model_names(self) -> list[str]:
        from .selection_lists import get_sorted_models

        rows = get_sorted_models(self._data_dir(), lightweight=False)
        # Prefer walk-forward models (fold comparison only makes sense for WF).
        rows = [r for r in rows if r.get("is_walk_forward") is not False]
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("model_name") or row.get("name") or "").strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def _apply_model_names(self, names: list[str]) -> None:
        self._model_names = names
        self._model_combo["values"] = names
        if self._model_var.get() not in names:
            saved = self._ui_state.get("fold_comparison.model")
            self._model_var.set(saved if saved in names else (names[0] if names else ""))
        self._status_var.set(f"{len(names)} walk-forward model(s) available.")
        self._reload_folds()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Model:").pack(side="left", padx=(0, 4))
        self._model_combo = ttk.Combobox(
            toolbar,
            textvariable=self._model_var,
            width=48,
            state="readonly",
        )
        self._model_combo.pack(side="left", padx=(0, 12))
        self._model_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: (self._ui_state.set("fold_comparison.model", self._model_var.get()), self._reload_folds()),
        )

        ttk.Label(toolbar, text="Fold A:").pack(side="left", padx=(0, 4))
        self._fold_a_combo = ttk.Combobox(
            toolbar,
            textvariable=self._fold_a_var,
            width=8,
            state="readonly",
        )
        self._fold_a_combo.pack(side="left", padx=(0, 12))
        self._fold_a_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_compare_enabled())

        ttk.Label(toolbar, text="Fold B:").pack(side="left", padx=(0, 4))
        self._fold_b_combo = ttk.Combobox(
            toolbar,
            textvariable=self._fold_b_var,
            width=8,
            state="readonly",
        )
        self._fold_b_combo.pack(side="left", padx=(0, 12))
        self._fold_b_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_compare_enabled())

        self._compare_btn = ttk.Button(toolbar, text="Compare", command=self._run_compare)
        self._compare_btn.pack(side="left", padx=4)
        self._csv_btn = ttk.Button(toolbar, text="Download CSV", command=self._download_csv)
        self._csv_btn.pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh models", command=self.refresh).pack(side="left", padx=4)

        # Registry-style Fold Metrics for the selected model (before compare)
        self._metrics_panel = ttk.Frame(self, padding=(8, 0, 8, 4))
        self._metrics_panel.pack(fill="x")
        self._render_fold_metrics_placeholder("Select a model to load fold metrics.")

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        for tab_id, label in (
            ("summary", "Summary"),
            ("distributions", "Distributions"),
            ("prediction_errors", "Prediction Errors"),
            ("bands", "Premium Bands"),
            ("importance", "Feature Importance"),
            ("errors", "Error Metrics"),
            ("predictions", "Prediction Metrics"),
        ):
            frame = ttk.Frame(self._notebook)
            scroll = ScrollableFrame(frame)
            scroll.pack(fill="both", expand=True)
            self._tab_scrolls[tab_id] = scroll
            self._notebook.add(frame, text=label)
            self._render_placeholder(
                scroll,
                f"Select two different folds and click Compare to view {label.lower()}.",
            )

        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(
            anchor="w", padx=10, pady=(0, 6)
        )

    def _render_placeholder(self, scroll: ScrollableFrame, text: str) -> None:
        clear_children(scroll.inner)
        ttk.Label(scroll.inner, text=text, foreground=COL_MUTED, wraplength=900, justify="left").pack(
            anchor="w", pady=8
        )

    def _render_fold_metrics_placeholder(self, text: str) -> None:
        clear_children(self._metrics_panel)
        section_title(self._metrics_panel, "Fold Metrics", color=ACCENT)
        ttk.Label(self._metrics_panel, text=text, foreground=COL_MUTED).pack(anchor="w", pady=(2, 4))

    def _render_model_fold_metrics(self, doc: dict[str, Any]) -> None:
        """Same Fold Metrics table as Model Registry → Walk Forward (+ val rows/days)."""
        from chain_replay_ml.training.fold_comparison import model_fold_metrics_table

        clear_children(self._metrics_panel)
        section_title(self._metrics_panel, "Fold Metrics", color=ACCENT)
        table = model_fold_metrics_table(doc, data_dir=self._data_dir())
        section_desc(self._metrics_panel, table.get("source_label") or "—")
        if table.get("hit_note"):
            ttk.Label(
                self._metrics_panel,
                text=str(table["hit_note"]),
                foreground=COL_MUTED,
                wraplength=980,
            ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            self._metrics_panel,
            text="Tip: click a row to set Fold A, click another to set Fold B.",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 4))

        fold_rows = []
        is_cls = bool(table.get("is_classification"))
        for r in table.get("rows") or []:
            if not isinstance(r, dict):
                continue
            if is_cls:
                fold_rows.append((
                    r.get("fold", "—"),
                    fmt_pct(r.get("accuracy_pct")) if r.get("accuracy_pct") is not None else "—",
                    fmt_pct(r.get("precision_pct")) if r.get("precision_pct") is not None else "—",
                    fmt_pct(r.get("recall_pct")) if r.get("recall_pct") is not None else "—",
                    fmt_pct(r.get("f1_pct")) if r.get("f1_pct") is not None else "—",
                    fmt_num(r.get("roc_auc"), 3) if r.get("roc_auc") is not None else "—",
                    fmt_num(r.get("composite_score"), 4),
                    fmt_val(r.get("feature_count") if r.get("feature_count") is not None else r.get("trees_trained")),
                ))
            else:
                fold_rows.append((
                    r.get("fold", "—"),
                    fmt_rows(r.get("validation_rows")),
                    r.get("validation_days_label") or "—",
                    fmt_pct(r.get("endpoint_hit_pct") if r.get("endpoint_hit_pct") is not None else r.get("target_hit_pct"))
                    if (r.get("endpoint_hit_pct") if r.get("endpoint_hit_pct") is not None else r.get("target_hit_pct")) is not None
                    else "—",
                    fmt_num(r.get("rmse"), 4),
                    fmt_num(r.get("mae"), 4),
                    fmt_num(r.get("r2"), 4),
                    fmt_num(r.get("mape"), 4),
                    fmt_pct(r.get("directional_accuracy_pct"))
                    if r.get("directional_accuracy_pct") is not None else "—",
                    fmt_num(r.get("composite_score"), 4),
                    fmt_val(r.get("trees_trained")),
                ))
        if not fold_rows:
            ttk.Label(self._metrics_panel, text="Not available", foreground=COL_MUTED).pack(anchor="w")
            return

        if is_cls:
            cols = [
                ("fold", "Fold", 40),
                ("acc", "Accuracy", 70),
                ("prec", "Precision", 70),
                ("rec", "Recall", 60),
                ("f1", "F1", 55),
                ("auc", "AUC", 55),
                ("comp", "Composite", 70),
                ("feats", "Feats", 50),
            ]
        else:
            cols = [
                ("fold", "Fold", 40),
                ("vrows", "Val Rows", 65),
                ("vdays", "Validation Days", 130),
                ("hit", "Endpoint Hit %", 95),
                ("rmse", "RMSE", 60),
                ("mae", "MAE", 60),
                ("r2", "R²", 50),
                ("mape", "MAPE", 55),
                ("dir", "Direction", 70),
                ("comp", "Composite", 70),
                ("trees", "Trees", 50),
            ]
        tree = data_table(
            self._metrics_panel,
            cols,
            fold_rows,
            height=min(12, len(fold_rows) + 1),
        )
        tree.bind("<<TreeviewSelect>>", lambda _e: self._on_fold_metrics_select(tree))

    def _on_fold_metrics_select(self, tree: ttk.Treeview) -> None:
        sel = tree.selection()
        if not sel:
            return
        values = tree.item(sel[0], "values")
        if not values:
            return
        fold_txt = str(values[0]).strip()
        try:
            fold_id = int(float(fold_txt))
        except (TypeError, ValueError):
            return
        if str(fold_id) not in (self._fold_a_combo["values"] or ()):
            return

        a = self._fold_a_var.get().strip()
        b = self._fold_b_var.get().strip()
        if not a or self._fold_pick_step == 0:
            self._fold_a_var.set(str(fold_id))
            self._fold_pick_step = 1
            self._status_var.set(f"Fold A = {fold_id}. Select Fold B (or click another row).")
        elif not b or str(fold_id) != a:
            self._fold_b_var.set(str(fold_id))
            self._fold_pick_step = 0
            self._status_var.set(f"Fold A = {a or self._fold_a_var.get()} · Fold B = {fold_id}. Compare enabled.")
        self._update_compare_enabled()

    def _update_compare_enabled(self) -> None:
        ready = False
        try:
            a = int(self._fold_a_var.get().strip())
            b = int(self._fold_b_var.get().strip())
            ready = a != b
        except (TypeError, ValueError):
            ready = False
        state = ["!disabled"] if ready else ["disabled"]
        try:
            self._compare_btn.state(state)
        except tk.TclError:
            pass
        if not ready:
            if self._fold_a_var.get() and self._fold_b_var.get() and self._fold_a_var.get() == self._fold_b_var.get():
                self._status_var.set("Choose two different folds to enable Compare.")

    def _update_csv_enabled(self) -> None:
        ready = bool(self._last_report and self._last_report.get("ok"))
        state = ["!disabled"] if ready else ["disabled"]
        try:
            self._csv_btn.state(state)
        except (tk.TclError, AttributeError):
            pass

    def _reload_folds(self) -> None:
        model = self._model_var.get().strip()
        self._fold_a_var.set("")
        self._fold_b_var.set("")
        self._fold_pick_step = 0
        self._update_compare_enabled()
        if not model:
            self._fold_ids = []
            self._current_doc = None
            self._fold_a_combo["values"] = []
            self._fold_b_combo["values"] = []
            self._render_fold_metrics_placeholder("Select a model to load fold metrics.")
            return

        def load() -> dict[str, Any]:
            from chain_replay_ml.training.fold_comparison import list_fold_ids, list_fold_ids_on_disk
            from chain_replay_ml.training.registry import load_model_detail

            doc = load_model_detail(self._data_dir(), model) or {}
            ids = list_fold_ids(doc)
            if not ids:
                ids = list_fold_ids_on_disk(self._data_dir(), model)
            return {"doc": doc, "ids": ids}

        def apply(payload: dict[str, Any]) -> None:
            doc = payload.get("doc") if isinstance(payload.get("doc"), dict) else {}
            ids = list(payload.get("ids") or [])
            self._current_doc = doc
            self._fold_ids = ids
            values = [str(i) for i in ids]
            self._fold_a_combo["values"] = values
            self._fold_b_combo["values"] = values
            # Leave folds unselected until the user picks two (enables Compare).
            self._fold_a_var.set("")
            self._fold_b_var.set("")
            self._fold_pick_step = 0
            self._last_report = None
            self._update_csv_enabled()
            if doc.get("is_walk_forward") or ids:
                self._render_model_fold_metrics(doc)
            else:
                self._render_fold_metrics_placeholder("This model has no walk-forward fold metrics.")
            self._status_var.set(
                f"{model}: {len(ids)} fold(s). Select Fold A and Fold B to enable Compare."
                if ids
                else f"{model}: no folds found."
            )
            self._update_compare_enabled()
            # Reset comparison tabs until Compare is clicked
            for tab_id, label in (
                ("summary", "Summary"),
                ("distributions", "Distributions"),
                ("prediction_errors", "Prediction Errors"),
                ("bands", "Premium Bands"),
                ("importance", "Feature Importance"),
                ("errors", "Error Metrics"),
                ("predictions", "Prediction Metrics"),
            ):
                self._render_placeholder(
                    self._tab_scrolls[tab_id],
                    f"Select two different folds and click Compare to view {label.lower()}.",
                )

        self.lazy_load(
            load=load,
            apply=apply,
            message="Loading folds…",
            status_var=self._status_var,
        )

    def _validate_selection(self) -> tuple[str, int, int] | None:
        model = self._model_var.get().strip()
        if not model:
            messagebox.showwarning("Fold Comparison", "Select a model.")
            return None
        try:
            fold_a = int(self._fold_a_var.get().strip())
            fold_b = int(self._fold_b_var.get().strip())
        except (TypeError, ValueError):
            messagebox.showwarning("Fold Comparison", "Select Fold A and Fold B.")
            return None
        if fold_a == fold_b:
            messagebox.showwarning("Fold Comparison", "Choose two different folds.")
            return None
        return model, fold_a, fold_b

    def _run_compare(self) -> None:
        selection = self._validate_selection()
        if not selection:
            return
        model, fold_a, fold_b = selection
        self._status_var.set(f"Comparing {model} Fold {fold_a} vs Fold {fold_b}…")
        self.lazy_load(
            load=lambda: self._load_comparison(model, fold_a, fold_b),
            apply=self._render_comparison,
            message=f"Comparing Fold {fold_a} vs Fold {fold_b}…",
            status_var=self._status_var,
            show_overlay=True,
        )

    def _load_comparison(self, model: str, fold_a: int, fold_b: int) -> dict[str, Any]:
        from chain_replay_ml.training.fold_comparison import build_fold_comparison
        from chain_replay_ml.training.registry import load_model_detail

        doc = load_model_detail(self._data_dir(), model)
        if not doc:
            raise ValueError(f"Model not found: {model}")
        return build_fold_comparison(self._data_dir(), doc, fold_a, fold_b)

    def _render_comparison(self, report: dict[str, Any]) -> None:
        self._last_report = report if isinstance(report, dict) else None
        self._update_csv_enabled()
        if not report.get("ok"):
            for scroll in self._tab_scrolls.values():
                self._render_placeholder(scroll, report.get("error") or "Comparison failed.")
            self._status_var.set(report.get("error") or "Comparison failed.")
            return

        label_a = str(report.get("label_a") or "Fold A")
        label_b = str(report.get("label_b") or "Fold B")
        self._render_summary_tab(report, label_a, label_b)
        self._render_distributions_tab(report, label_a, label_b)
        self._render_prediction_errors_tab(report, label_a, label_b)
        self._render_bands_tab(report, label_a, label_b)
        self._render_importance_tab(report, label_a, label_b)
        self._render_errors_tab(report, label_a, label_b)
        self._render_predictions_tab(report)
        self._status_var.set(
            f"Compared {report.get('model_name')} {label_a} vs {label_b} "
            f"(metrics source: {report.get('source') or '—'})."
        )

    def _download_csv(self) -> None:
        from tkinter import filedialog

        from chain_replay_ml.training.fold_comparison import build_fold_comparison_csv

        report = self._last_report
        if not report or not report.get("ok"):
            messagebox.showinfo("Download CSV", "Run Compare first.")
            return
        model = str(report.get("model_name") or "model").strip() or "model"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
        fa = report.get("fold_a")
        fb = report.get("fold_b")
        initial = f"{safe}_fold_{fa}_vs_{fb}_comparison.csv"
        path = filedialog.asksaveasfilename(
            title="Save Fold Comparison CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=initial,
        )
        if not path:
            return
        try:
            text = build_fold_comparison_csv(report)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        except OSError as exc:
            messagebox.showerror("Download CSV", f"Could not save:\n{exc}")
            return
        self._status_var.set(f"Saved comparison CSV → {path}")

    def _render_context_card(self, parent: tk.Misc, ctx: dict[str, Any], title: str) -> None:
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        if not ctx.get("available"):
            ttk.Label(box, text=ctx.get("message") or "Context unavailable", foreground=COL_WARN).pack(anchor="w")
            return
        regime = ctx.get("market_regime") or "—"
        ttk.Label(box, text=str(regime), font=("Segoe UI", 10, "bold"), foreground=COL_HOLDOUT, wraplength=420).pack(
            anchor="w", pady=(0, 6)
        )
        for row in ctx.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("key") == "market_regime":
                continue
            line = ttk.Frame(box)
            line.pack(fill="x", pady=1)
            ttk.Label(line, text=str(row.get("label") or ""), foreground=COL_MUTED, width=22, anchor="w").pack(
                side="left"
            )
            ttk.Label(line, text=str(row.get("value") or "—"), font=BODY_FONT, wraplength=280, justify="left").pack(
                side="left", fill="x", expand=True
            )

    def _render_summary_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["summary"]
        clear_children(scroll.inner)
        parent = scroll.inner

        diagnosis = report.get("diagnosis") if isinstance(report.get("diagnosis"), dict) else {}
        diag = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}

        # --- Why? (primary — user should not need to compare tables) ---
        section_title(parent, "Why?", color=COL_HOLDOUT)
        section_desc(parent, "Every metric below is explained by market/regime/feature drift — not raw table hunting.")
        ttk.Label(
            parent,
            text=str(diagnosis.get("headline") or "—"),
            font=("Segoe UI", 12, "bold"),
            foreground=COL_WARN if diagnosis.get("worse_label") else COL_MUTED,
            wraplength=960,
            justify="left",
        ).pack(anchor="w", pady=(2, 6))
        for reason in diagnosis.get("reasons") or []:
            ttk.Label(
                parent,
                text=f"✓ {reason}",
                foreground=COL_MUTED,
                wraplength=960,
                justify="left",
                font=("Segoe UI", 10),
            ).pack(anchor="w")

        # Metric cards with Why attached
        cards = diagnosis.get("metric_cards") or []
        if cards:
            section_title(parent, f"{diagnosis.get('worse_label') or 'Weaker fold'} — metrics with Why", color=ACCENT)
            cards_row = ttk.Frame(parent)
            cards_row.pack(fill="x", pady=(4, 10))
            for card in cards:
                if not isinstance(card, dict):
                    continue
                frame = ttk.LabelFrame(cards_row, text=str(card.get("title") or ""), padding=8)
                frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
                ttk.Label(
                    frame,
                    text=str(card.get("value_display") or "—"),
                    font=("Segoe UI", 14, "bold"),
                    foreground=COL_WARN,
                ).pack(anchor="w")
                why_items = card.get("why") or []
                if why_items:
                    ttk.Label(frame, text="Why?", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(
                        anchor="w", pady=(6, 2)
                    )
                    for w in why_items[:5]:
                        ttk.Label(frame, text=f"✓ {w}", foreground=COL_MUTED, wraplength=360, justify="left").pack(
                            anchor="w"
                        )
                else:
                    ttk.Label(frame, text="No large gap vs peer fold.", foreground=COL_MUTED).pack(anchor="w", pady=4)

        # What is unique (outlier vs training)
        unique = diagnosis.get("what_is_unique") if isinstance(diagnosis.get("what_is_unique"), dict) else {}
        if unique.get("available"):
            section_title(parent, f"What is unique about {unique.get('fold_label') or 'this fold'}?", color=COL_HOLDOUT)
            section_desc(parent, "Vs that fold’s own training distribution (z-score).")
            for i, row in enumerate(unique.get("rows") or [], start=1):
                if not isinstance(row, dict):
                    continue
                ttk.Label(
                    parent,
                    text=f"{i}. {row.get('emoji') or ''} {row.get('feature')}    {row.get('display')}".strip(),
                    font=BODY_FONT,
                ).pack(anchor="w")

        # Fold Context cards
        section_title(parent, "Fold Context", color=ACCENT)
        section_desc(parent, "Click a fold in metrics above, then Compare — context answers what kind of market it was.")
        ctx_row = ttk.Frame(parent)
        ctx_row.pack(fill="x", pady=(4, 10))
        self._render_context_card(ctx_row, diag.get("context_a") if isinstance(diag.get("context_a"), dict) else {}, label_a)
        self._render_context_card(ctx_row, diag.get("context_b") if isinstance(diag.get("context_b"), dict) else {}, label_b)

        # Compact summary metrics (secondary)
        section_title(parent, "Summary metrics", color=ACCENT)
        section_desc(parent, f"Side-by-side numbers — {report.get('model_name') or ''} (details live in Why? above)")
        self._metric_table(parent, report.get("summary_metrics") or [], label_a, label_b)

        vw = report.get("validation_window") if isinstance(report.get("validation_window"), dict) else {}
        wa = vw.get("fold_a") if isinstance(vw.get("fold_a"), dict) else {}
        wb = vw.get("fold_b") if isinstance(vw.get("fold_b"), dict) else {}
        window_rows = [
            ("Validation rows", fmt_rows(wa.get("validation_rows")), fmt_rows(wb.get("validation_rows"))),
            (
                "Trading days",
                ", ".join(wa.get("trading_days") or []) or "—",
                ", ".join(wb.get("trading_days") or []) or "—",
            ),
        ]
        data_table(
            parent,
            [("field", "Field", 160), ("a", label_a, 260), ("b", label_b, 260)],
            window_rows,
            height=len(window_rows) + 1,
        )

    def _render_distributions_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["distributions"]
        clear_children(scroll.inner)
        parent = scroll.inner
        diag = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
        dist = diag.get("distribution_shift") if isinstance(diag.get("distribution_shift"), dict) else {}

        section_title(parent, "Feature Distribution Shift", color=COL_HOLDOUT)
        section_desc(
            parent,
            f"Input means: {label_b} − {label_a}. This is more actionable than importance alone — "
            "it shows how the market changed.",
        )
        if not dist.get("available"):
            ttk.Label(
                parent,
                text=dist.get("message") or diag.get("message") or "Distribution comparison not available.",
                foreground=COL_WARN,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", pady=8)
            return

        table_rows = []
        for r in dist.get("rows") or []:
            if not isinstance(r, dict):
                continue
            table_rows.append((
                f"{r.get('severity_emoji') or ''} {r.get('feature') or '—'}".strip(),
                fmt_val(r.get("fold_a")),
                fmt_val(r.get("fold_b")),
                r.get("display_delta") or "—",
                r.get("severity") or "—",
            ))
        data_table(
            parent,
            [
                ("feature", "Feature", 220),
                ("a", label_a, 100),
                ("b", label_b, 100),
                ("d", "Δ", 80),
                ("sev", "Shift", 80),
            ],
            table_rows,
            height=min(18, len(table_rows) + 1),
        )

        # Outliers vs training for both folds
        for key, lbl in (("outliers_a", label_a), ("outliers_b", label_b)):
            out = diag.get(key) if isinstance(diag.get(key), dict) else {}
            section_title(parent, f"Most unusual vs training — {lbl}", color=ACCENT)
            if not out.get("available"):
                ttk.Label(parent, text=out.get("message") or "—", foreground=COL_MUTED).pack(anchor="w", pady=4)
                continue
            orows = []
            for r in out.get("rows") or []:
                if not isinstance(r, dict):
                    continue
                orows.append((
                    f"{r.get('severity_emoji') or ''} {r.get('feature')}".strip(),
                    fmt_val(r.get("fold_mean")),
                    fmt_val(r.get("train_mean")),
                    r.get("display") or "—",
                    fmt_pct(r.get("percentile")) if r.get("percentile") is not None else "—",
                ))
            data_table(
                parent,
                [
                    ("feature", "Feature", 220),
                    ("fm", "Fold mean", 100),
                    ("tm", "Train mean", 100),
                    ("z", "Z-score", 80),
                    ("p", "Percentile", 90),
                ],
                orows,
                height=min(10, len(orows) + 1),
            )

    def _render_prediction_errors_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["prediction_errors"]
        clear_children(scroll.inner)
        parent = scroll.inner

        hist = report.get("error_histograms") if isinstance(report.get("error_histograms"), dict) else {}
        section_title(parent, "Error Histogram", color=COL_HOLDOUT)
        if not hist.get("available"):
            ttk.Label(
                parent,
                text=hist.get("message") or "Prediction error histogram not available.",
                foreground=COL_WARN,
                font=("Segoe UI", 11, "bold"),
                wraplength=900,
            ).pack(anchor="w", pady=8)
            return

        section_desc(
            parent,
            f"Distribution of absolute prediction error ({hist.get('unit') or 'Rs'}). "
            "Production model scored on each fold validation window.",
        )
        if hist.get("insight"):
            ttk.Label(
                parent,
                text=str(hist["insight"]),
                foreground=COL_WARN,
                wraplength=900,
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(4, 8))

        row = ttk.Frame(parent)
        row.pack(fill="both", expand=True, pady=(4, 8))

        for side_key, color in (("fold_a", "#1B7F3A"), ("fold_b", "#C0392B")):
            block = hist.get(side_key) if isinstance(hist.get(side_key), dict) else {}
            col = ttk.Frame(row)
            col.pack(side="left", fill="both", expand=True, padx=(0, 12) if side_key == "fold_a" else (12, 0))

            lbl = str(block.get("label") or (label_a if side_key == "fold_a" else label_b))
            days = block.get("trading_days") or []
            days_txt = ", ".join(str(d) for d in days) if days else "—"
            section_title(col, lbl, color=ACCENT)
            ttk.Label(
                col,
                text=f"Validation days: {days_txt}  ·  Rows: {fmt_rows(block.get('total'))}",
                foreground=COL_MUTED,
            ).pack(anchor="w", pady=(0, 4))
            stats = (
                f"Mean {fmt_rupee(block.get('mean_abs_error'))}  ·  "
                f"Median {fmt_rupee(block.get('median_abs_error'))}  ·  "
                f"Max {fmt_rupee(block.get('max_abs_error'))}"
            )
            ttk.Label(col, text=stats, foreground=COL_MUTED).pack(anchor="w", pady=(0, 6))

            bucket_rows = []
            for r in block.get("rows") or []:
                if not isinstance(r, dict):
                    continue
                bucket_rows.append((
                    r.get("bucket") or "—",
                    fmt_rows(r.get("count")),
                    f"{float(r['pct']):.1f}%" if r.get("pct") is not None else "—",
                ))
            if bucket_rows:
                data_table(
                    col,
                    [("bucket", "Error (Rs)", 70), ("count", "Count", 70), ("pct", "Share", 70)],
                    bucket_rows,
                    height=len(bucket_rows) + 1,
                )

            canvas = tk.Canvas(col, height=150, bg="#1a2a44", highlightthickness=0)
            canvas.pack(fill="x", pady=(6, 0))
            bars = [
                (str(r.get("bucket") or ""), int(r.get("count") or 0))
                for r in (block.get("rows") or [])
                if isinstance(r, dict)
            ]

            def _draw(c: tk.Canvas = canvas, b: list = bars, t: str = lbl, clr: str = color) -> None:
                draw_bucket_bars(c, b, title=f"{t} error distribution", color=clr)

            canvas.bind("<Configure>", lambda _e, fn=_draw: fn())
            col.after(50, _draw)

    def _render_bands_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["bands"]
        clear_children(scroll.inner)
        parent = scroll.inner
        section_title(parent, "Premium Band Comparison", color=ACCENT)
        section_desc(parent, "Per-band MAE and direction accuracy on each fold’s validation window.")
        bands = report.get("premium_bands") or []
        if not bands:
            ttk.Label(parent, text="Premium band metrics not available.", foreground=COL_MUTED).pack(anchor="w")
            return
        rows = []
        for b in bands:
            if not isinstance(b, dict):
                continue
            rows.append((
                b.get("band_label") or b.get("band") or "—",
                fmt_rows(b.get("samples_a")),
                fmt_rows(b.get("samples_b")),
                fmt_rupee(b.get("mae_a")),
                fmt_rupee(b.get("mae_b")),
                b.get("mae_winner") or "—",
                fmt_pct(b.get("dir_a")) if b.get("dir_a") is not None else "—",
                fmt_pct(b.get("dir_b")) if b.get("dir_b") is not None else "—",
                b.get("dir_winner") or "—",
            ))
        data_table(
            parent,
            [
                ("band", "Band", 70),
                ("na", f"Rows {label_a}", 80),
                ("nb", f"Rows {label_b}", 80),
                ("ma", f"MAE {label_a}", 80),
                ("mb", f"MAE {label_b}", 80),
                ("mw", "MAE Winner", 90),
                ("da", f"Dir {label_a}", 80),
                ("db", f"Dir {label_b}", 80),
                ("dw", "Dir Winner", 90),
            ],
            rows,
            height=min(12, len(rows) + 1),
        )

    def _render_importance_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["importance"]
        clear_children(scroll.inner)
        parent = scroll.inner
        section_title(parent, "Feature Importance Δ", color=ACCENT)
        imp = report.get("feature_importance") if isinstance(report.get("feature_importance"), dict) else {}
        if not imp.get("available"):
            ttk.Label(
                parent,
                text=imp.get("message") or "Not saved",
                foreground=COL_WARN,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", pady=8)
            section_desc(parent, "Importance CSVs are written under walk_forward/fold_XX/ during initial WF training.")
            return

        all_rows = [r for r in (imp.get("rows") or []) if isinstance(r, dict)]
        section_desc(
            parent,
            f"Δ = {label_b} − {label_a} importance %. "
            f"Positive ⇒ {label_b} relies more; negative ⇒ {label_a} relies more.",
        )

        # --- Largest Feature Shifts ---
        section_title(parent, "Largest Feature Shifts", color=COL_HOLDOUT)
        shifts = imp.get("largest_shifts") or all_rows[:5]
        shift_frame = ttk.Frame(parent)
        shift_frame.pack(fill="x", pady=(2, 8))
        if not shifts:
            ttk.Label(shift_frame, text="No shifts available.", foreground=COL_MUTED).pack(anchor="w")
        else:
            for row in shifts[:5]:
                if not isinstance(row, dict) or row.get("delta") is None:
                    continue
                delta = float(row["delta"])
                arrow = row.get("arrow") or ("↓" if delta < 0 else "↑")
                color = "#1B7F3A" if delta < 0 else ("#C0392B" if delta > 0 else COL_MUTED)
                line = ttk.Frame(shift_frame)
                line.pack(fill="x", pady=1)
                ttk.Label(
                    line,
                    text=f"{arrow}  {row.get('feature') or '—'}",
                    font=BODY_FONT,
                ).pack(side="left")
                ttk.Label(
                    line,
                    text=f"{delta:+.2f}%",
                    font=("Segoe UI", 10, "bold"),
                    foreground=color,
                ).pack(side="right")

        # --- Legend ---
        legend = ttk.Frame(parent)
        legend.pack(fill="x", pady=(0, 6))
        ttk.Label(
            legend,
            text=f"🟢  {label_a} relies much more     🔴  {label_b} relies much more     ⚪  Nearly identical",
            foreground=COL_MUTED,
            wraplength=900,
        ).pack(anchor="w")

        # --- Search / filter ---
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(4, 6))
        ttk.Label(controls, text="Search Feature:").pack(side="left", padx=(0, 4))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(controls, textvariable=search_var, width=36)
        search_entry.pack(side="left", padx=(0, 12))
        only_large_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Show only |Δ| > 5%",
            variable=only_large_var,
        ).pack(side="left", padx=(0, 12))
        count_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=count_var, foreground=COL_MUTED).pack(side="left")

        # --- Table ---
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True, pady=(4, 8))

        cols = ("feat", "a", "b", "delta", "sig")
        tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            height=18,
        )
        tree.heading("feat", text="Feature")
        tree.heading("a", text=label_a)
        tree.heading("b", text=label_b)
        tree.heading("delta", text="Δ")
        tree.heading("sig", text="Signal")
        tree.column("feat", width=280, anchor="w")
        tree.column("a", width=90, anchor="center")
        tree.column("b", width=90, anchor="center")
        tree.column("delta", width=90, anchor="center")
        tree.column("sig", width=70, anchor="center")

        # Row colors by signal
        tree.tag_configure("fold_a", foreground="#1B7F3A")
        tree.tag_configure("fold_b", foreground="#C0392B")
        tree.tag_configure("similar", foreground="#6B7280")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        def _populate() -> None:
            tree.delete(*tree.get_children())
            query = search_var.get().strip().casefold()
            only_large = bool(only_large_var.get())
            shown = 0
            for r in all_rows:
                feat = str(r.get("feature") or "")
                abs_d = r.get("abs_delta")
                if abs_d is None and r.get("delta") is not None:
                    try:
                        abs_d = abs(float(r["delta"]))
                    except (TypeError, ValueError):
                        abs_d = None
                if query and query not in feat.casefold():
                    continue
                if only_large and (abs_d is None or abs_d <= 5.0):
                    continue
                code = str(r.get("signal") or "similar")
                tag = code if code in ("fold_a", "fold_b", "similar") else "similar"
                tree.insert(
                    "",
                    "end",
                    values=(
                        feat or "—",
                        f"{float(r['fold_a']):.2f}%" if r.get("fold_a") is not None else "—",
                        f"{float(r['fold_b']):.2f}%" if r.get("fold_b") is not None else "—",
                        r.get("delta_display") or "—",
                        r.get("signal_emoji") or "⚪",
                    ),
                    tags=(tag,),
                )
                shown += 1
            count_var.set(f"Showing {shown} of {len(all_rows)} features")

        search_var.trace_add("write", lambda *_: _populate())
        only_large_var.trace_add("write", lambda *_: _populate())
        _populate()


    def _render_errors_tab(self, report: dict[str, Any], label_a: str, label_b: str) -> None:
        scroll = self._tab_scrolls["errors"]
        clear_children(scroll.inner)
        parent = scroll.inner
        section_title(parent, "Error Metrics", color=ACCENT)
        section_desc(parent, "MAE, RMSE, bias, and P95 absolute error on each fold validation window.")
        self._metric_table(parent, report.get("error_metrics") or [], label_a, label_b)

    def _render_predictions_tab(self, report: dict[str, Any]) -> None:
        scroll = self._tab_scrolls["predictions"]
        clear_children(scroll.inner)
        parent = scroll.inner
        section_title(parent, "Prediction Metrics", color=ACCENT)
        pred = report.get("prediction_metrics") if isinstance(report.get("prediction_metrics"), dict) else {}
        if not pred.get("available"):
            ttk.Label(
                parent,
                text=pred.get("message") or "Prediction data not available for this fold.",
                foreground=COL_WARN,
                font=("Segoe UI", 11, "bold"),
                wraplength=900,
            ).pack(anchor="w", pady=8)
            section_desc(
                parent,
                "Hit Rate, drawdown, and time-to-target require saved fold prediction rows. "
                "They are shown here only when that data exists.",
            )
            return
        label_a = str(report.get("label_a") or "Fold A")
        label_b = str(report.get("label_b") or "Fold B")
        self._metric_table(parent, pred.get("rows") or [], label_a, label_b)

    def _metric_table(
        self,
        parent: tk.Misc,
        rows: list[tuple[Any, ...]],
        label_a: str,
        label_b: str,
    ) -> None:
        table_rows = []
        for row in rows:
            if not row or len(row) < 5:
                continue
            label, va, vb, delta, winner = row[0], row[1], row[2], row[3], row[4]
            table_rows.append((
                label,
                self._fmt_metric(label, va),
                self._fmt_metric(label, vb),
                fmt_val(delta) if delta is not None else "—",
                winner or "—",
            ))
        if not table_rows:
            ttk.Label(parent, text="No metrics available.", foreground=COL_MUTED).pack(anchor="w")
            return
        data_table(
            parent,
            [
                ("metric", "Metric", 120),
                ("a", label_a, 110),
                ("b", label_b, 110),
                ("delta", "Δ", 100),
                ("winner", "Winner", 100),
            ],
            table_rows,
            height=min(12, len(table_rows) + 1),
        )

    @staticmethod
    def _fmt_metric(label: str, value: Any) -> str:
        if value is None or value == "":
            return "—"
        key = str(label).lower()
        if "direction" in key or "composite" in key or "%" in key:
            if "composite" in key:
                return fmt_num(value, 4)
            return fmt_pct(value) if _is_number(value) else fmt_val(value)
        if key in ("mae", "rmse", "bias", "p95"):
            return fmt_rupee(value)
        return fmt_val(value)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
