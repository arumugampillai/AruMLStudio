"""Build summary dialog — configuration, warm-up, and policy review before build."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Callable

from . import feature_policy_format as fmt


class BuildSummaryDialog(tk.Toplevel):
    """Pre-build summary: configuration, classification, warm-up budget, checks."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        preview: dict[str, Any],
        title: str = "Build Summary",
        on_proceed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.transient(master.winfo_toplevel())
        self.grab_set()
        self._on_proceed = on_proceed
        self._preview = preview

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        hdr = ttk.Frame(body)
        hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(
            hdr,
            text="Review build configuration, warm-up, and policy before building",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        failed = any(c.get("status") == "fail" for c in (preview.get("checks") or []))
        self._status_var = tk.StringVar(
            value="Blocked — fix selection before build" if failed else "Ready to build",
        )
        ttk.Label(
            hdr,
            textvariable=self._status_var,
            foreground="#C62828" if failed else "#2E7D32",
        ).pack(anchor="w")

        cfg = preview.get("build_config") or {}
        if cfg:
            config_row = ttk.LabelFrame(body, text="Build configuration", padding=6)
            config_row.pack(fill="x", pady=(0, 6))
            grid = ttk.Frame(config_row)
            grid.pack(fill="x")
            rows = [
                ("Sampling interval", cfg.get("sampling_label") or cfg.get("sampling_interval_sec")),
                ("Strike selection", cfg.get("strike_label")),
                ("Target labels", cfg.get("target_labels_text") or ", ".join(cfg.get("target_labels") or [])),
            ]
            for row_i, (label, value) in enumerate(rows):
                ttk.Label(grid, text=f"{label}:", foreground="#555").grid(
                    row=row_i, column=0, sticky="w", padx=(0, 8), pady=1,
                )
                ttk.Label(grid, text=str(value or "—"), foreground="#222").grid(
                    row=row_i, column=1, sticky="w", pady=1,
                )

        btns = ttk.Frame(body)
        btns.pack(side="bottom", fill="x", pady=(8, 0))
        self._build_btn: ttk.Button | None = None
        if on_proceed is not None:
            self._build_btn = ttk.Button(btns, text="Build", command=self._proceed)
            self._build_btn.pack(side="right", padx=(4, 0))
            if failed:
                self._build_btn.configure(state="disabled")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")

        paned = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, pady=6)

        left = ttk.LabelFrame(paned, text="Warm-up features", padding=4)
        paned.add(left, weight=2)
        filter_row = ttk.Frame(left)
        filter_row.pack(fill="x", pady=(0, 4))
        ttk.Label(filter_row, text="Category").pack(side="left")
        self._category_var = tk.StringVar(value="all|All")
        self._category_combo = ttk.Combobox(
            filter_row, textvariable=self._category_var, width=16, state="readonly",
        )
        self._category_combo["values"] = [
            f"{key}|{label}" for key, label in fmt.CATEGORY_FILTER_OPTIONS
        ]
        self._category_combo.pack(side="left", padx=(4, 0))
        self._category_var.trace_add("write", lambda *_a: self._render_warmup_table())
        cols = ("feature", "samples", "time", "cat", "inherited")
        self._tree = ttk.Treeview(left, columns=cols, show="headings", height=10)
        for col, w, title in (
            ("feature", 180, "Feature"),
            ("samples", 72, "Samples"),
            ("time", 72, "Time"),
            ("cat", 72, "Category"),
            ("inherited", 96, "Inherited"),
        ):
            self._tree.heading(col, text=title)
            self._tree.column(col, width=w)
        self._tree.tag_configure("warmup_low", foreground="#2e7d32")
        self._tree.tag_configure("warmup_mid", foreground="#e65100")
        self._tree.tag_configure("warmup_high", foreground="#c62828")
        self._tree.pack(fill="both", expand=True)
        self._render_warmup_table()

        right = ttk.LabelFrame(paned, text="Summary", padding=4)
        paned.add(right, weight=3)
        self._text = scrolledtext.ScrolledText(right, wrap="word", font=("Consolas", 9), height=12)
        self._text.pack(fill="both", expand=True)
        self._text.insert("1.0", fmt.format_build_summary_preview(preview))
        self._text.configure(state="disabled")

        self.minsize(680, 520)
        self.geometry("860x620")
        self.update_idletasks()

    def _render_warmup_table(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        cat_key = fmt.category_filter_key(self._category_var.get())
        for row in self._preview.get("warmup_preview") or []:
            row_cat = str(row.get("category") or "").lower()
            if cat_key != "all" and row_cat != cat_key:
                continue
            samples = int(row.get("samples") or 0)
            icon = fmt.warmup_tier_icon(samples)
            inh = row.get("inherited_from") or ""
            inh_txt = f"← {inh}" if inh else ("yes" if row.get("inherited") else "—")
            tag = fmt.warmup_tier(samples)
            self._tree.insert(
                "",
                "end",
                values=(
                    row.get("name"),
                    f"{icon} {samples}",
                    fmt.format_warmup_duration(samples, sampling_interval_sec=float(
                        self._preview.get("sampling_interval_sec") or 3
                    )),
                    fmt.category_label(row_cat),
                    inh_txt,
                ),
                tags=(f"warmup_{tag}",),
            )

    def _proceed(self) -> None:
        if self._on_proceed:
            self._on_proceed()
        self.destroy()


# Backward-compatible aliases
BuildValidationDialog = BuildSummaryDialog


def show_build_summary_dialog(
    master: tk.Misc,
    *,
    feature_names: list[str],
    sampling_interval_sec: float,
    sliding_stride_sec: float | None = None,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
    gap_max_sec: float | None = None,
    estimated_rows: int | None = None,
    estimated_sessions: int | None = None,
    on_proceed: Callable[[], None] | None = None,
) -> BuildSummaryDialog | None:
    try:
        from chain_replay_ml.dataset_builder.build_summary import build_summary_preview
    except ImportError:
        return None
    preview = build_summary_preview(
        feature_names,
        sampling_interval_sec=sampling_interval_sec,
        sliding_stride_sec=sliding_stride_sec,
        strike_selection=strike_selection,
        gap_policy=gap_policy,
        prediction_targets=prediction_targets,
        gap_max_sec=gap_max_sec,
        estimated_rows=estimated_rows,
        estimated_sessions=estimated_sessions,
    )
    return BuildSummaryDialog(master, preview=preview, on_proceed=on_proceed)


def show_build_validation_dialog(
    master: tk.Misc,
    *,
    feature_names: list[str],
    sampling_interval_sec: float,
    sliding_stride_sec: float | None = None,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
    gap_max_sec: float | None = None,
    estimated_rows: int | None = None,
    estimated_sessions: int | None = None,
    on_proceed: Callable[[], None] | None = None,
) -> BuildSummaryDialog | None:
    return show_build_summary_dialog(
        master,
        feature_names=feature_names,
        sampling_interval_sec=sampling_interval_sec,
        sliding_stride_sec=sliding_stride_sec,
        strike_selection=strike_selection,
        gap_policy=gap_policy,
        prediction_targets=prediction_targets,
        gap_max_sec=gap_max_sec,
        estimated_rows=estimated_rows,
        estimated_sessions=estimated_sessions,
        on_proceed=on_proceed,
    )
