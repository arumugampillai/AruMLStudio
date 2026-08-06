"""Dataset Comparison panel — side-by-side dataset metadata and feature sets in Tk."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .build_service import chart_data_dir
from .dataset_comparison import (
    SummaryRow,
    build_feature_set_comparison,
    build_summary_comparison,
    dataset_display_label,
    load_dataset_compare_doc,
)
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import (
    ACCENT,
    COL_MUTED,
    ScrollableFrame,
    clear_children,
    data_table,
    fmt_val,
    section_desc,
    section_title,
)
from .ui_state import get_ui_state_manager


class DatasetComparisonPanel(ttk.Frame, LazyLoadMixin):
    """Compare two datasets across summary and feature-set tabs (metadata-only)."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._dataset_names: list[str] = []
        self._pending_auto_compare: tuple[str, str] | None = None
        self._status_var = tk.StringVar(value="Select two datasets and click Compare.")
        self._dataset_a_var = tk.StringVar()
        self._dataset_b_var = tk.StringVar()
        self._tab_scrolls: dict[str, ScrollableFrame] = {}
        self._ui_state = get_ui_state_manager()
        self._build_ui()
        self._ui_state.bind_combobox(
            self._dataset_a_combo, "dataset_comparison.dataset_a", var=self._dataset_a_var, restore=False
        )
        self._ui_state.bind_combobox(
            self._dataset_b_combo, "dataset_comparison.dataset_b", var=self._dataset_b_var, restore=False
        )
        self._ui_state.bind_notebook(self._notebook, "dataset_comparison.tab")
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def open_with_datasets(self, dataset_a: str, dataset_b: str) -> None:
        """Pre-select two datasets; comparison runs after names load."""
        self._pending_auto_compare = (dataset_a, dataset_b)
        self._dataset_a_var.set(dataset_a)
        self._dataset_b_var.set(dataset_b)

    def activate_pending(self) -> None:
        """Run a pending comparison when the panel is already visible."""
        if not self._pending_auto_compare:
            return
        if not self._dataset_names:
            self.refresh(lazy=True)
            return
        self._try_auto_compare()

    def prepare_comparison(self, dataset_a: str, dataset_b: str) -> None:
        """Backward-compatible alias for registry navigation."""
        self.open_with_datasets(dataset_a, dataset_b)

    def refresh(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_dataset_names,
                apply=self._apply_dataset_names,
                message="Loading datasets…",
                status_var=self._status_var,
            )
            return
        try:
            names = self._fetch_dataset_names()
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_dataset_names(names)

    def _fetch_dataset_names(self) -> list[str]:
        from .selection_lists import get_sorted_dataset_names

        return get_sorted_dataset_names(self._data_dir())

    def _apply_dataset_names(self, names: list[str]) -> None:
        self._dataset_names = names
        self._dataset_a_combo["values"] = names
        self._dataset_b_combo["values"] = names
        if self._dataset_a_var.get() not in names:
            saved_a = self._ui_state.get("dataset_comparison.dataset_a")
            self._dataset_a_var.set(saved_a if saved_a in names else (names[0] if names else ""))
        if self._dataset_b_var.get() not in names:
            saved_b = self._ui_state.get("dataset_comparison.dataset_b")
            if saved_b in names:
                self._dataset_b_var.set(saved_b)
            else:
                self._dataset_b_var.set(names[1] if len(names) > 1 else (names[0] if names else ""))
        self._status_var.set(f"{len(names)} dataset(s) available.")
        self._try_auto_compare()

    def _try_auto_compare(self) -> bool:
        pending = self._pending_auto_compare
        if not pending or not self._dataset_names:
            return False
        dataset_a, dataset_b = pending
        self._pending_auto_compare = None
        if dataset_a not in self._dataset_names or dataset_b not in self._dataset_names:
            self._status_var.set("One or both selected datasets are no longer available.")
            return True
        self._dataset_a_var.set(dataset_a)
        self._dataset_b_var.set(dataset_b)
        self.after(0, self._run_compare)
        return True

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Dataset A:").pack(side="left", padx=(0, 4))
        self._dataset_a_combo = ttk.Combobox(
            toolbar,
            textvariable=self._dataset_a_var,
            width=42,
            state="readonly",
        )
        self._dataset_a_combo.pack(side="left", padx=(0, 12))

        ttk.Label(toolbar, text="Dataset B:").pack(side="left", padx=(0, 4))
        self._dataset_b_combo = ttk.Combobox(
            toolbar,
            textvariable=self._dataset_b_var,
            width=42,
            state="readonly",
        )
        self._dataset_b_combo.pack(side="left", padx=(0, 12))

        ttk.Button(toolbar, text="Compare", command=self._run_compare).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh datasets", command=self.refresh).pack(side="left", padx=4)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)

        for tab_id, label in (
            ("summary", "Summary"),
            ("common", "Common features"),
            ("only_a", "Only in A"),
            ("only_b", "Only in B"),
        ):
            frame = ttk.Frame(self._notebook)
            scroll = ScrollableFrame(frame)
            scroll.pack(fill="both", expand=True)
            self._tab_scrolls[tab_id] = scroll
            self._notebook.add(frame, text=label)
            self._render_placeholder(scroll, f"Choose two datasets and click Compare to view {label.lower()}.")

        ttk.Label(self, textvariable=self._status_var, foreground=COL_MUTED).pack(anchor="w", padx=10, pady=(0, 6))

    def _render_placeholder(self, scroll: ScrollableFrame, text: str) -> None:
        clear_children(scroll.inner)
        ttk.Label(scroll.inner, text=text, foreground=COL_MUTED, wraplength=900, justify="left").pack(anchor="w", pady=8)

    def _validate_selection(self) -> tuple[str, str] | None:
        name_a = self._dataset_a_var.get().strip()
        name_b = self._dataset_b_var.get().strip()
        if not name_a or not name_b:
            messagebox.showwarning("Dataset Comparison", "Select both Dataset A and Dataset B.")
            return None
        if name_a == name_b:
            messagebox.showwarning("Dataset Comparison", "Choose two different datasets to compare.")
            return None
        if name_a not in self._dataset_names or name_b not in self._dataset_names:
            messagebox.showwarning(
                "Dataset Comparison",
                "One or both selected datasets are no longer available. Refresh the list.",
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

    def _load_docs(self, name_a: str, name_b: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        data_dir = self._data_dir()
        doc_a = load_dataset_compare_doc(data_dir, name_a)
        doc_b = load_dataset_compare_doc(data_dir, name_b)
        feat_cmp = build_feature_set_comparison(doc_a, doc_b)
        return doc_a, doc_b, feat_cmp

    def _render_comparison(self, payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
        doc_a, doc_b, feat_cmp = payload
        label_a = dataset_display_label(doc_a)
        label_b = dataset_display_label(doc_b)

        self._render_summary_tab(doc_a, doc_b, label_a, label_b)
        self._render_feature_list_tab("common", list(feat_cmp["common"]), label_a, label_b, feat_cmp["common_count"])
        self._render_feature_list_tab(
            "only_a",
            list(feat_cmp["only_a"]),
            label_a,
            label_b,
            feat_cmp["only_a_count"],
            empty_hint=f"No A-only features — {label_a}'s set is covered by {label_b}.",
        )
        self._render_feature_list_tab(
            "only_b",
            list(feat_cmp["only_b"]),
            label_a,
            label_b,
            feat_cmp["only_b_count"],
            empty_hint=f"No B-only features — {label_b}'s set is covered by {label_a}.",
        )

        self._status_var.set(
            f"Compared {label_a} vs {label_b}: "
            f"{feat_cmp['common_count']} common, "
            f"{feat_cmp['only_a_count']} only in A, "
            f"{feat_cmp['only_b_count']} only in B "
            f"({feat_cmp['overlap_pct']}% overlap)."
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
        section_title(parent, "Dataset Summary")
        section_desc(parent, f"{label_a} vs {label_b} — metadata and expected spec only (no parquet reads).")

        for title, rows in build_summary_comparison(doc_a, doc_b):
            section_title(parent, title, color=ACCENT)
            self._summary_table(parent, rows, label_a, label_b, show_status=True, compact=True)

    def _render_feature_list_tab(
        self,
        tab_id: str,
        names: list[str],
        label_a: str,
        label_b: str,
        count: int,
        *,
        empty_hint: str = "No shared features.",
    ) -> None:
        scroll = self._tab_scrolls[tab_id]
        clear_children(scroll.inner)
        parent = scroll.inner

        if tab_id == "common":
            title = "Common features"
            desc = f"Features present in both {label_a} and {label_b}."
        elif tab_id == "only_a":
            title = f"In {label_a} but not {label_b}"
            desc = f"Features implemented in Dataset A only ({count})."
        else:
            title = f"In {label_b} but not {label_a}"
            desc = f"Features implemented in Dataset B only ({count})."

        section_title(parent, title)
        section_desc(parent, desc)

        if not names:
            ttk.Label(parent, text=empty_hint, foreground=COL_MUTED).pack(anchor="w", pady=6)
            return

        rows = [(str(i + 1), name) for i, name in enumerate(names)]
        data_table(
            parent,
            [
                ("#", "#", 48),
                ("feature", "Feature", 420),
            ],
            rows,
            height=min(24, max(6, len(rows) + 1)),
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
    ) -> None:
        if show_status:
            table_rows = [
                (field, fmt_val(val_a), fmt_val(val_b), self._summary_status(val_a, val_b))
                for field, val_a, val_b in rows
            ]
            field_w, val_w, status_w = (160, 140, 72) if compact else (200, 200, 90)
            columns = [
                ("field", "Field", field_w),
                ("a", "Dataset A" if compact else label_a, val_w),
                ("b", "Dataset B" if compact else label_b, val_w),
                ("status", "Status", status_w),
            ]
        else:
            table_rows = [(field, fmt_val(val_a), fmt_val(val_b)) for field, val_a, val_b in rows]
            columns = [
                ("field", "Field", 200),
                ("a", label_a, 220),
                ("b", label_b, 220),
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
