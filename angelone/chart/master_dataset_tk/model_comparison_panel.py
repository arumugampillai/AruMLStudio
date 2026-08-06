"""Model Comparison panel — side-by-side model metrics in Tk."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_comparison import (
    MetricRow,
    SummaryRow,
    build_metric_comparison,
    build_core_model_metrics_comparison,
    build_feature_set_comparison,
    build_premium_model_metrics_rows,
    build_premium_band_comparison,
    premium_metrics_available,
    build_summary_comparison,
    build_validation_comparison,
    build_walk_forward_comparison,
    model_display_label,
)
from .model_registry_widgets import (
    ACCENT,
    COL_HOLDOUT,
    COL_MUTED,
    COL_OK,
    COL_PRODUCTION,
    COL_TRAINING,
    ScrollableFrame,
    clear_children,
    data_table,
    fmt_num,
    fmt_pct,
    fmt_rupee,
    fmt_val,
    section_desc,
    section_title,
)
from .ui_state import get_ui_state_manager


class ModelComparisonPanel(ttk.Frame, LazyLoadMixin):
    """Compare two trained models across summary, metrics, validation, and walk-forward."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._model_names: list[str] = []
        self._pending_auto_compare: tuple[str, str] | None = None
        self._status_var = tk.StringVar(value="Select two models and click Compare.")
        self._model_a_var = tk.StringVar()
        self._model_b_var = tk.StringVar()
        self._tab_scrolls: dict[str, ScrollableFrame] = {}
        self._ui_state = get_ui_state_manager()
        self._build_ui()
        self._ui_state.bind_combobox(
            self._model_a_combo, "model_comparison.model_a", var=self._model_a_var, restore=False
        )
        self._ui_state.bind_combobox(
            self._model_b_combo, "model_comparison.model_b", var=self._model_b_var, restore=False
        )
        self._ui_state.bind_notebook(self._notebook, "model_comparison.tab")
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def open_with_models(self, model_a: str, model_b: str) -> None:
        """Pre-select two models; comparison runs after model names load."""
        self._pending_auto_compare = (model_a, model_b)
        self._model_a_var.set(model_a)
        self._model_b_var.set(model_b)

    def activate_pending(self) -> None:
        """Run a pending comparison when the panel is already visible."""
        if not self._pending_auto_compare:
            return
        if not self._model_names:
            self.refresh(lazy=True)
            return
        self._try_auto_compare()

    def prepare_comparison(self, model_a: str, model_b: str) -> None:
        """Backward-compatible alias for registry navigation."""
        self.open_with_models(model_a, model_b)

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
        from .selection_lists import get_sorted_model_names

        return get_sorted_model_names(self._data_dir(), lightweight=False)

    def _apply_model_names(self, names: list[str]) -> None:
        self._model_names = names
        self._model_a_combo["values"] = names
        self._model_b_combo["values"] = names
        if self._model_a_var.get() not in names:
            saved_a = self._ui_state.get("model_comparison.model_a")
            self._model_a_var.set(saved_a if saved_a in names else (names[0] if names else ""))
        if self._model_b_var.get() not in names:
            saved_b = self._ui_state.get("model_comparison.model_b")
            if saved_b in names:
                self._model_b_var.set(saved_b)
            else:
                self._model_b_var.set(names[1] if len(names) > 1 else (names[0] if names else ""))
        self._status_var.set(f"{len(names)} model(s) available.")
        self._try_auto_compare()

    def _try_auto_compare(self) -> bool:
        pending = self._pending_auto_compare
        if not pending or not self._model_names:
            return False
        model_a, model_b = pending
        self._pending_auto_compare = None
        if model_a not in self._model_names or model_b not in self._model_names:
            self._status_var.set("One or both selected models are no longer available.")
            return True
        self._model_a_var.set(model_a)
        self._model_b_var.set(model_b)
        self.after(0, self._run_compare)
        return True

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Model A:").pack(side="left", padx=(0, 4))
        self._model_a_combo = ttk.Combobox(
            toolbar,
            textvariable=self._model_a_var,
            width=42,
            state="readonly",
        )
        self._model_a_combo.pack(side="left", padx=(0, 12))

        ttk.Label(toolbar, text="Model B:").pack(side="left", padx=(0, 4))
        self._model_b_combo = ttk.Combobox(
            toolbar,
            textvariable=self._model_b_var,
            width=42,
            state="readonly",
        )
        self._model_b_combo.pack(side="left", padx=(0, 12))

        ttk.Button(toolbar, text="Compare", command=self._run_compare).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh models", command=self.refresh).pack(side="left", padx=4)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        for tab_id, label in (
            ("summary", "Summary"),
            ("model_metrics", "Model Metrics"),
            ("validation", "Validation Metrics"),
            ("walk_forward", "Walk Forward"),
            ("features", "Features"),
        ):
            frame = ttk.Frame(self._notebook)
            scroll = ScrollableFrame(frame)
            scroll.pack(fill="both", expand=True)
            self._tab_scrolls[tab_id] = scroll
            self._notebook.add(frame, text=label)
            self._render_placeholder(scroll, f"Choose two models and click Compare to view {label.lower()}.")

        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(anchor="w", padx=10, pady=(0, 6))

    def _render_placeholder(self, scroll: ScrollableFrame, text: str) -> None:
        clear_children(scroll.inner)
        ttk.Label(scroll.inner, text=text, foreground=COL_MUTED, wraplength=900, justify="left").pack(anchor="w", pady=8)

    def _validate_selection(self) -> tuple[str, str] | None:
        name_a = self._model_a_var.get().strip()
        name_b = self._model_b_var.get().strip()
        if not name_a or not name_b:
            messagebox.showwarning("Model Comparison", "Select both Model A and Model B.")
            return None
        if name_a == name_b:
            messagebox.showwarning("Model Comparison", "Choose two different models to compare.")
            return None
        if name_a not in self._model_names or name_b not in self._model_names:
            messagebox.showwarning(
                "Model Comparison",
                "One or both selected models are no longer available. Refresh the list.",
            )
            return None
        return name_a, name_b

    def _run_compare(self) -> None:
        selection = self._validate_selection()
        if not selection:
            return
        name_a, name_b = selection
        self._status_var.set(f"Loading {name_a} and {name_b}…")
        self.lazy_load(
            load=lambda: self._load_docs(name_a, name_b),
            apply=self._render_comparison,
            message=f"Comparing {name_a} vs {name_b}…",
            status_var=self._status_var,
            show_overlay=True,
        )

    def _load_docs(self, name_a: str, name_b: str) -> tuple[dict[str, Any], dict[str, Any]]:
        from chain_replay_ml.training.registry import load_model_detail

        doc_a = load_model_detail(self._data_dir(), name_a)
        doc_b = load_model_detail(self._data_dir(), name_b)
        if not doc_a:
            raise ValueError(f"Model not found: {name_a}")
        if not doc_b:
            raise ValueError(f"Model not found: {name_b}")
        data_dir = self._data_dir()
        doc_a["_data_dir"] = data_dir
        doc_b["_data_dir"] = data_dir
        return doc_a, doc_b

    def _render_comparison(self, docs: tuple[dict[str, Any], dict[str, Any]]) -> None:
        doc_a, doc_b = docs
        label_a = model_display_label(doc_a)
        label_b = model_display_label(doc_b)

        self._render_summary_tab(doc_a, doc_b, label_a, label_b)
        self._render_model_metrics_tab(doc_a, doc_b, label_a, label_b)
        self._render_validation_tab(doc_a, doc_b, label_a, label_b)
        self._render_walk_forward_tab(doc_a, doc_b, label_a, label_b)
        self._render_features_tab(doc_a, doc_b, label_a, label_b)

        self._status_var.set(f"Compared {label_a} vs {label_b}.")

    def _render_features_tab(
        self,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        scroll = self._tab_scrolls["features"]
        clear_children(scroll.inner)
        parent = scroll.inner
        cmp = build_feature_set_comparison(doc_a, doc_b)

        section_title(parent, "Feature Set Comparison")
        section_desc(
            parent,
            f"{label_a}: {cmp['count_a']} features · {label_b}: {cmp['count_b']} features · "
            f"overlap {cmp['overlap_pct']}%",
        )
        self._summary_table(
            parent,
            [
                ("Model A feature count", cmp["count_a"], "—"),
                ("Model B feature count", "—", cmp["count_b"]),
                ("Common features", cmp["common_count"], cmp["common_count"]),
                ("Only in Model A", cmp["only_a_count"], "—"),
                ("Only in Model B", "—", cmp["only_b_count"]),
                ("Set overlap %", f"{cmp['overlap_pct']}%", f"{cmp['overlap_pct']}%"),
            ],
            label_a,
            label_b,
            show_status=False,
            compact=True,
        )

        # Sub-tabs: Common / A only / B only
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, pady=(10, 0))

        def _fill_list(tab: ttk.Frame, title: str, names: list[str], empty_hint: str) -> None:
            section_desc(tab, f"{title} ({len(names)})")
            if not names:
                ttk.Label(tab, text=empty_hint, foreground=COL_MUTED).pack(anchor="w", pady=6)
                return
            rows = [(str(i + 1), name) for i, name in enumerate(names)]
            data_table(
                tab,
                [
                    ("#", "#", 48),
                    ("feature", "Feature", 420),
                ],
                rows,
                height=min(22, max(6, len(rows) + 1)),
            )

        tab_common = ttk.Frame(nb, padding=6)
        tab_a = ttk.Frame(nb, padding=6)
        tab_b = ttk.Frame(nb, padding=6)
        nb.add(tab_common, text=f"Common ({cmp['common_count']})")
        nb.add(tab_a, text=f"Only in A ({cmp['only_a_count']})")
        nb.add(tab_b, text=f"Only in B ({cmp['only_b_count']})")
        _fill_list(tab_common, "Common features", list(cmp["common"]), "No shared features.")
        _fill_list(
            tab_a,
            f"In {label_a} but not {label_b}",
            list(cmp["only_a"]),
            f"No A-only features — {label_a}'s set is covered by {label_b}.",
        )
        _fill_list(
            tab_b,
            f"In {label_b} but not {label_a}",
            list(cmp["only_b"]),
            f"No B-only features — {label_b}'s set is covered by {label_a}.",
        )

    def _render_summary_tab(
        self,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        scroll = self._tab_scrolls["summary"]
        clear_children(scroll.inner)
        parent = scroll.inner
        section_title(parent, "Model Summary")

        groups = {title: rows for title, rows in build_summary_comparison(doc_a, doc_b)}
        top_row = ttk.Frame(parent)
        top_row.pack(fill="both", expand=True)
        for col, title, pad in (
            (ttk.Frame(top_row, padding=(0, 4)), "Validation Strategy", (0, 6)),
            (ttk.Frame(top_row, padding=(0, 4)), "Dataset", (6, 0)),
        ):
            col.pack(side="left", fill="both", expand=True, padx=pad)
            rows = groups.get(title) or []
            if not rows:
                continue
            section_title(col, title, color=ACCENT)
            self._summary_table(col, rows, label_a, label_b, show_status=True, compact=True)

        details_rows = groups.get("Model Details") or []
        ops_rows = groups.get("Operational Performance") or []
        if ops_rows:
            section_title(parent, "Operational Performance", color=ACCENT)
            self._summary_table(parent, ops_rows, label_a, label_b, model_col_a="Model A", model_col_b="Model B")
        if details_rows:
            section_title(parent, "Model Details", color=ACCENT)
            self._summary_table(parent, details_rows, label_a, label_b, model_col_a="A model", model_col_b="B model")

        self._render_production_metrics_block(parent, doc_a, doc_b, label_a, label_b)

    def _render_model_metrics_tab(
        self,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        scroll = self._tab_scrolls["model_metrics"]
        clear_children(scroll.inner)
        self._render_production_metrics_block(scroll.inner, doc_a, doc_b, label_a, label_b)

    def _render_production_metrics_block(
        self,
        parent: tk.Misc,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        metrics_row = ttk.Frame(parent)
        metrics_row.pack(fill="both", expand=True)
        left = ttk.Frame(metrics_row, padding=(0, 4))
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = ttk.Frame(metrics_row, padding=(0, 4))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        section_title(left, "Model Metrics")
        self._compact_metric_table(left, build_core_model_metrics_comparison(doc_a, doc_b), label_a, label_b)

        section_title(right, "Premium Model Metrics")
        premium_notes: list[str] = []
        if not premium_metrics_available(doc_a):
            premium_notes.append(f"Model A ({label_a}): no premium metrics saved — retrain to populate")
        if not premium_metrics_available(doc_b):
            premium_notes.append(f"Model B ({label_b}): no premium metrics saved — retrain to populate")
        if premium_notes:
            section_desc(right, " ".join(premium_notes))
        self._ab_value_table(right, build_premium_model_metrics_rows(doc_a, doc_b))

        band_rows = build_premium_band_comparison(doc_a, doc_b)
        if band_rows:
            section_title(parent, "Premium Band Performance")
            if premium_notes:
                section_desc(parent, "Band values show — when premium metrics were not computed for that model.")
            self._premium_band_table(parent, band_rows, label_a, label_b)

    def _render_validation_tab(
        self,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        scroll = self._tab_scrolls["validation"]
        clear_children(scroll.inner)
        parent = scroll.inner
        sections = build_validation_comparison(doc_a, doc_b)
        for section_key, title, color, desc in (
            ("training_validation", "Training Validation", COL_TRAINING, "Metrics used during training and early stopping."),
            ("production", "Production Performance", COL_PRODUCTION, "Deployed champion model after final retrain."),
            ("holdout_test", "Holdout Test", COL_HOLDOUT, "Final holdout test set — generalization on unseen data."),
        ):
            section_title(parent, title, color=color)
            section_desc(parent, desc)
            self._metric_table(parent, sections[section_key], label_a, label_b, use_model_aliases=True)

    def _render_walk_forward_tab(
        self,
        doc_a: dict[str, Any],
        doc_b: dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> None:
        scroll = self._tab_scrolls["walk_forward"]
        clear_children(scroll.inner)
        parent = scroll.inner

        if not doc_a.get("is_walk_forward") and not doc_b.get("is_walk_forward"):
            section_desc(parent, "Neither model is a walk-forward model.")
            return

        wf = build_walk_forward_comparison(doc_a, doc_b)
        section_title(parent, "Walk Forward Configuration")
        self._summary_table(
            parent,
            wf["config"],
            label_a,
            label_b,
            model_col_a="Model A",
            model_col_b="Model B",
        )

        folds = wf.get("folds") or []
        if not folds:
            section_desc(parent, "No per-fold metrics available for either model.")
            return

        section_title(parent, "Per-Fold Metrics")
        for fold_block in folds:
            fold_id = fold_block.get("fold", "—")
            section_title(parent, f"Fold {fold_id}", color=ACCENT)
            self._metric_table(
                parent,
                fold_block.get("metrics") or [],
                label_a,
                label_b,
                use_model_aliases=True,
            )

    def _summary_table(
        self,
        parent: tk.Misc,
        rows: list[SummaryRow],
        label_a: str,
        label_b: str,
        *,
        show_status: bool = False,
        compact: bool = False,
        model_col_a: str | None = None,
        model_col_b: str | None = None,
    ) -> None:
        if show_status:
            table_rows = [
                (field, fmt_val(val_a), fmt_val(val_b), self._summary_status(val_a, val_b))
                for field, val_a, val_b in rows
            ]
            field_w, val_w, status_w = (140, 120, 72) if compact else (200, 200, 90)
            col_a = model_col_a or ("Model A" if compact else label_a)
            col_b = model_col_b or ("Model B" if compact else label_b)
            columns = [
                ("field", "Field", field_w),
                ("a", col_a, val_w),
                ("b", col_b, val_w),
                ("status", "Status", status_w),
            ]
        else:
            table_rows = [(field, fmt_val(val_a), fmt_val(val_b)) for field, val_a, val_b in rows]
            col_a = model_col_a or label_a
            col_b = model_col_b or label_b
            columns = [
                ("field", "Field", 200),
                ("a", col_a, 220),
                ("b", col_b, 220),
            ]
        data_table(
            parent,
            columns,
            table_rows,
            height=min(20, len(table_rows) + 1),
        )

    @staticmethod
    def _summary_values_match(val_a: Any, val_b: Any) -> bool:
        def _norm(v: Any) -> Any:
            if v is None or v == "":
                return None
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                fv = float(v)
                return None if fv != fv else fv
            text = str(v).strip()
            return text if text else None

        a = _norm(val_a)
        b = _norm(val_b)
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) < 1e-12
        return a == b

    @classmethod
    def _summary_status(cls, val_a: Any, val_b: Any) -> str:
        return "Same" if cls._summary_values_match(val_a, val_b) else "Different"

    def _metric_table(
        self,
        parent: tk.Misc,
        rows: list[MetricRow],
        label_a: str,
        label_b: str,
        *,
        use_model_aliases: bool = True,
    ) -> None:
        col_a = "Model A" if use_model_aliases else label_a
        col_b = "Model B" if use_model_aliases else label_b
        table_rows = [
            (
                label,
                self._fmt_metric_value(label, val_a),
                self._fmt_metric_value(label, val_b),
                delta or "—",
                self._winner_label(winner, label_a, label_b, use_model_aliases=use_model_aliases),
            )
            for label, val_a, val_b, delta, winner in rows
        ]
        tree = data_table(
            parent,
            [
                ("metric", "Metric", 180),
                ("a", col_a, 110),
                ("b", col_b, 110),
                ("delta", "Delta (B−A)", 100),
                ("winner", "Winner", 80),
            ],
            table_rows,
            height=min(14, len(table_rows) + 1),
        )
        self._tag_winners(tree)

    def _ab_value_table(
        self,
        parent: tk.Misc,
        rows: list[SummaryRow],
    ) -> None:
        table_rows = [
            (
                label,
                self._fmt_metric_value(label, val_a),
                self._fmt_metric_value(label, val_b),
            )
            for label, val_a, val_b in rows
        ]
        data_table(
            parent,
            [
                ("metric", "Metrics", 180),
                ("a", "A model value", 120),
                ("b", "B model value", 120),
            ],
            table_rows,
            height=min(8, len(table_rows) + 1),
        )

    def _premium_band_table(
        self,
        parent: tk.Misc,
        rows: list[tuple[str, str, str, str, str, str, str]],
        label_a: str,
        label_b: str,
    ) -> None:
        table_rows = [
            (
                band,
                mae,
                rmse,
                mae_pct,
                rmse_pct,
                direction,
                self._winner_label(winner, label_a, label_b, use_model_aliases=True),
            )
            for band, mae, rmse, mae_pct, rmse_pct, direction, winner in rows
        ]
        tree = data_table(
            parent,
            [
                ("band", "Band", 72),
                ("mae", "MAE A/B", 96),
                ("rmse", "RMSE A/B", 96),
                ("mae_pct", "MAE% A/B", 96),
                ("rmse_pct", "RMSE% A/B", 100),
                ("dir", "Direction A/B", 108),
                ("winner", "Winner", 80),
            ],
            table_rows,
            height=min(10, len(table_rows) + 1),
        )
        self._tag_winners(tree)

    def _compact_metric_table(
        self,
        parent: tk.Misc,
        rows: list[MetricRow],
        label_a: str,
        label_b: str,
    ) -> None:
        table_rows = [
            (
                label,
                self._fmt_metric_value(label, val_a),
                self._fmt_metric_value(label, val_b),
                self._winner_label(winner, label_a, label_b, use_model_aliases=True),
            )
            for label, val_a, val_b, _delta, winner in rows
        ]
        tree = data_table(
            parent,
            [
                ("metric", "Metrics", 160),
                ("a", "A model value", 120),
                ("b", "B model value", 120),
                ("winner", "Winner", 90),
            ],
            table_rows,
            height=min(6, len(table_rows) + 1),
        )
        self._tag_winners(tree)

    def _fmt_metric_value(self, label: str, value: Any) -> str:
        if value is None or value == "":
            return "—"
        label_l = label.lower()
        if label_l in ("mae", "rmse") or ("mae" in label_l or "rmse" in label_l or "error" in label_l or "bias" in label_l):
            if "premium" in label_l or "%" in label:
                return fmt_pct(value)
            if "₹" in label:
                return fmt_rupee(value)
            try:
                float(value)
                if "percentile" in label_l or "median" in label_l or label_l in ("mae", "rmse"):
                    return fmt_rupee(value)
                return fmt_num(value)
            except (TypeError, ValueError):
                return fmt_val(value)
        if "direction" in label_l or "%" in label:
            return fmt_pct(value)
        if "r²" in label_l or "composite" in label_l or "mape" in label_l:
            return fmt_num(value)
        return fmt_val(value)

    @staticmethod
    def _winner_label(
        winner: str | None,
        label_a: str,
        label_b: str,
        *,
        use_model_aliases: bool = True,
    ) -> str:
        if winner == "A":
            return "Model A" if use_model_aliases else label_a
        if winner == "B":
            return "Model B" if use_model_aliases else label_b
        if winner == "Tie":
            return "Tie"
        return "—"

    @staticmethod
    def _tag_winners(tree: ttk.Treeview) -> None:
        try:
            tree.tag_configure("winner", foreground=COL_OK)
        except tk.TclError:
            return
