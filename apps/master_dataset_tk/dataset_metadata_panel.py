"""Dataset Registry metadata — structured companion panel (Day Metadata pattern)."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from . import registry_service as svc
from .day_metadata_panel import _fmt_dur, _fmt_int, _fmt_num, _fmt_pct
from .fold_replay_widgets import place_toplevel_beside_main
from .ui_util import open_path


def _feature_coverage_rows(chart_dir: str, dataset_name: str) -> list[dict[str, Any]]:
    """Per-feature coverage rows for the Coverage table (parquet scan)."""
    import pandas as pd

    from chain_replay_ml.dataset_builder.audit_diagnostics import audit_feature_coverage
    from chain_replay_ml.dataset_builder.feature_sources_catalog import (
        classify_dataset_feature_source,
        dataset_base_pipeline_export_feature_names,
        dataset_feature_source_label,
        dataset_registry_export_feature_names,
    )
    from chain_replay_ml.dataset_builder.schema_registry import columns_map, load_schema_registry

    data_dir = svc.data_dir_for(chart_dir)
    doc = svc.load_dataset_metadata(chart_dir, dataset_name)
    meta = doc.get("metadata") or {}
    cols = [str(c).strip() for c in (meta.get("feature_columns") or []) if str(c).strip()]
    if not cols:
        return []
    registry_names = dataset_registry_export_feature_names(meta, data_dir=data_dir)
    base_names = dataset_base_pipeline_export_feature_names(meta, data_dir=data_dir)
    pq_path = str((doc.get("source_dataset") or {}).get("path") or "").strip()
    if not pq_path or not os.path.isfile(pq_path):
        return []
    try:
        df = pd.read_parquet(pq_path, columns=cols)
    except Exception:
        df = pd.read_parquet(pq_path)
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return []
        df = df[cols]
    audit_rows = audit_feature_coverage(df, cols).get("features") or []
    cmap = columns_map(load_schema_registry())
    out: list[dict[str, Any]] = []
    for row in audit_rows:
        feat = str(row.get("feature") or "")
        col = cmap.get(feat) or {}
        family = str(col.get("group_id") or col.get("group") or "Other")
        st = str(row.get("status") or "")
        if st == "pass":
            status = "Healthy"
        elif st == "warn":
            status = "Partial"
        else:
            status = "Unexpected Empty"
        cov = float(row.get("coverage_pct") or 0.0)
        missing = int(row.get("missing_count") or 0)
        total = int(row.get("row_count") or 0)
        bucket = classify_dataset_feature_source(
            feat,
            data_dir=data_dir,
            registry_names=registry_names,
            base_pipeline_names=base_names,
        )
        out.append(
            {
                "feature": feat,
                "feature_family": family,
                "source": dataset_feature_source_label(bucket),
                "source_bucket": bucket,
                "status": status,
                "availability": "Available" if cov > 0 else "Missing",
                "coverage_pct": cov,
                "reason": (
                    "Fully populated"
                    if missing == 0
                    else f"{missing:,} missing / {total:,} rows"
                ),
            }
        )
    return out


def _day_blocks(meta: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = meta.get("sources") or meta.get("days") or []
    return [b for b in blocks if isinstance(b, dict)]


def _audit_finding_lines(findings: Any, limit: int = 12) -> list[str]:
    """Flatten audit findings (list or {warnings, information} dict)."""
    items: list[Any] = []
    if isinstance(findings, dict):
        items.extend(findings.get("warnings") or [])
        items.extend(findings.get("information") or [])
        for key, val in findings.items():
            if key in ("warnings", "information"):
                continue
            if isinstance(val, list):
                items.extend(val)
    elif isinstance(findings, list):
        items = list(findings)
    lines: list[str] = []
    for raw in items[:limit]:
        if isinstance(raw, dict):
            title = raw.get("title") or raw.get("message") or raw.get("root_cause")
            rec = raw.get("recommendation")
            text = str(title or raw)
            if rec:
                text = f"{text} — {rec}"
            lines.append(text)
        else:
            lines.append(str(raw))
    return lines


def _string_list_lines(items: Any, limit: int = 12) -> list[str]:
    if isinstance(items, list):
        return [str(x) for x in items[:limit]]
    if items:
        return [str(items)]
    return []


class DatasetMetadataPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        dataset_name: str,
        metadata_path: str | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self.dataset_name = str(dataset_name or "").strip()
        self.metadata_path = str(metadata_path or "").strip() or None
        self._load_gen = 0
        self._day_blocks: list[dict[str, Any]] = []
        self._columns_cache: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self._meta_doc: dict[str, Any] = {}

        self._status = tk.StringVar(value="")
        self._day_var = tk.StringVar(value="")
        self._filter = tk.StringVar(value="all")
        self._source_filter = tk.StringVar(value="feature_registry")
        self._family = tk.StringVar(value="All")
        self._detail_var = tk.StringVar(value="Select a feature")
        self._score_detail = tk.StringVar(value="")
        self._gap_summary = tk.StringVar(value="")
        self._reg_detail = tk.StringVar(value="")
        self._audit_detail = tk.StringVar(value="")

        self._fact_vars: dict[str, tk.StringVar] = {}
        self._health_vars: dict[str, tk.StringVar] = {}
        self._source_vars: dict[str, tk.StringVar] = {}
        self._build_vars: dict[str, tk.StringVar] = {}
        self._quality_vars: dict[str, tk.StringVar] = {}
        self._audit_vars: dict[str, tk.StringVar] = {}
        self._validation_vars: dict[str, tk.StringVar] = {}
        self._reg_vars: dict[str, tk.StringVar] = {}

        self._build_ui()
        self.after_idle(self.refresh)

    def _kv_grid(
        self,
        parent: ttk.Frame,
        fields: tuple[tuple[str, str], ...],
        store: dict[str, tk.StringVar],
        *,
        cols: int = 4,
    ) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, cols)
            cell = ttk.Frame(grid, padding=4)
            cell.grid(row=r, column=c, sticky="nw", padx=6, pady=2)
            ttk.Label(cell, text=label, foreground="#888").pack(anchor="w")
            var = tk.StringVar(value="—")
            store[key] = var
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 10, "bold")).pack(anchor="w")

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Dataset Metadata", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, text=self.dataset_name, foreground="#444").pack(side="left", padx=(12, 0))
        ttk.Label(top, text="Trading day").pack(side="left", padx=(16, 4))
        self._day_cb = ttk.Combobox(
            top, textvariable=self._day_var, width=14, state="readonly"
        )
        self._day_cb.pack(side="left")
        self._day_cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_day_facts())
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=8)
        ttk.Label(top, textvariable=self._status, foreground="#888").pack(side="right")

        main_nb = ttk.Notebook(self)
        main_nb.pack(fill="both", expand=True, padx=8, pady=4)

        data_tab = ttk.Frame(main_nb, padding=4)
        main_nb.add(data_tab, text="Data Overview")

        row1 = ttk.Frame(data_tab)
        row1.pack(fill="x")
        row1.columnconfigure(0, weight=3)
        row1.columnconfigure(1, weight=1)

        facts = ttk.LabelFrame(row1, text="Data Overview", padding=8)
        facts.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._kv_grid(
            facts,
            (
                ("trading_day", "Trading Day"),
                ("total_rows", "Rows"),
                ("total_columns", "Columns"),
                ("meta_columns", "Meta Columns"),
                ("feature_count", "Features"),
                ("target_count", "Targets"),
                ("market", "Market"),
                ("sampling", "Sampling"),
                ("trading_duration_sec", "Trading Duration"),
            ),
            self._fact_vars,
            cols=3,
        )

        ds_stats = ttk.LabelFrame(row1, text="Dataset Statistics", padding=8)
        ds_stats.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._kv_grid(
            ds_stats,
            (
                ("audit_status", "Audit"),
                ("validation_status", "Validation"),
                ("confidence", "Confidence"),
                ("training_ready", "Training"),
            ),
            self._fact_vars,
            cols=2,
        )

        row2 = ttk.Frame(data_tab)
        row2.pack(fill="x", pady=(8, 0))
        row2.columnconfigure(0, weight=3)
        row2.columnconfigure(1, weight=1)

        health = ttk.LabelFrame(row2, text="Health Assessment (derived)", padding=8)
        health.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._kv_grid(
            health,
            (
                ("average_coverage", "Overall Coverage"),
                ("unexpected_nulls", "Unexpected Nulls"),
                ("expected_nulls", "Expected Nulls"),
                ("invalid_timestamps", "Invalid Timestamps"),
                ("registry_match", "Registry / Spec"),
                ("complete_features", "Complete"),
                ("partial_features", "Partial"),
                ("sparse_features", "Sparse"),
                ("healthy_features", "Healthy"),
                ("health_score", "Health Score"),
            ),
            self._health_vars,
            cols=4,
        )
        ttk.Label(
            health, textvariable=self._score_detail, foreground="#666", justify="left"
        ).pack(anchor="w", pady=(4, 0))

        sources = ttk.LabelFrame(row2, text="Feature Sources", padding=8)
        sources.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._kv_grid(
            sources,
            (
                ("feature_registry", "Feature Registry"),
                ("base_pipeline", "Base Pipeline"),
                ("other_pipeline", "Other Pipeline"),
                ("feature_total", "Total Features"),
            ),
            self._source_vars,
            cols=2,
        )

        detail_nb = ttk.Notebook(data_tab)
        detail_nb.pack(fill="both", expand=True, pady=(8, 0))

        cov_tab = ttk.Frame(detail_nb, padding=4)
        detail_nb.add(cov_tab, text="Coverage")
        source_row = ttk.Frame(cov_tab)
        source_row.pack(fill="x", pady=(0, 4))
        ttk.Label(source_row, text="Feature source").pack(side="left", padx=(0, 6))
        for value, label in (
            ("feature_registry", "Feature Registry"),
            ("base_pipeline", "Base Pipeline"),
            ("other_pipeline", "Other Pipeline"),
        ):
            ttk.Radiobutton(
                source_row,
                text=label,
                value=value,
                variable=self._source_filter,
                command=self._render_columns,
            ).pack(side="left", padx=4)
        filt = ttk.Frame(cov_tab)
        filt.pack(fill="x", pady=(0, 4))
        for value, label in (
            ("all", "All"),
            ("problems", "Problems"),
            ("coverage_lt_99", "Coverage <99%"),
            ("Healthy", "Healthy"),
        ):
            ttk.Radiobutton(
                filt, text=label, value=value, variable=self._filter, command=self._render_columns
            ).pack(side="left", padx=3)
        ttk.Label(filt, text="Category").pack(side="left", padx=(12, 4))
        self._family_cb = ttk.Combobox(
            filt,
            textvariable=self._family,
            values=[
                "All",
                "Base",
                "Derived",
                "Market Structure",
                "Greeks",
                "Volatility",
                "Order Book",
                "Prediction",
                "Target",
                "Meta",
                "Other",
            ],
            width=16,
            state="readonly",
        )
        self._family_cb.pack(side="left")
        self._family_cb.bind("<<ComboboxSelected>>", lambda _e: self._render_columns())

        cov_split = ttk.Panedwindow(cov_tab, orient=tk.HORIZONTAL)
        cov_split.pack(fill="both", expand=True)
        cov_left = ttk.Frame(cov_split)
        cov_split.add(cov_left, weight=3)
        cov_right = ttk.LabelFrame(cov_split, text="Feature Catalog", padding=8)
        cov_split.add(cov_right, weight=1)

        cols = ("feature", "family", "source", "status", "availability", "coverage", "reason")
        self._cov_tree = ttk.Treeview(cov_left, columns=cols, show="headings", height=14)
        for c, (txt, w) in {
            "feature": ("Feature", 130),
            "family": ("Category", 100),
            "source": ("Source", 100),
            "status": ("Status", 110),
            "availability": ("Availability", 90),
            "coverage": ("Coverage", 70),
            "reason": ("Reason", 180),
        }.items():
            self._cov_tree.heading(c, text=txt)
            self._cov_tree.column(
                c, width=w, anchor="w" if c in ("feature", "reason") else "center"
            )
        sb = ttk.Scrollbar(cov_left, orient="vertical", command=self._cov_tree.yview)
        self._cov_tree.configure(yscrollcommand=sb.set)
        self._cov_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._cov_tree.bind("<<TreeviewSelect>>", self._on_feature_select)
        self._cov_tree.tag_configure("unexpected", foreground="#C62828")
        self._cov_tree.tag_configure("expected", foreground="#546E7A")
        self._cov_tree.tag_configure("ok", foreground="#2E7D32")
        self._row_by_iid: dict[str, dict[str, Any]] = {}
        ttk.Label(cov_right, textvariable=self._detail_var, justify="left", wraplength=260).pack(
            anchor="nw"
        )

        quality_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(quality_tab, text="Integrity")
        self._kv_grid(
            quality_tab,
            (
                ("duplicate_rows", "Duplicate Rows"),
                ("missing_targets", "Missing Targets"),
                ("invalid_strike_rows", "Invalid Strike Rows"),
                ("invalid_timestamps", "Invalid Timestamps"),
                ("expected_nulls", "Expected Nulls"),
                ("unexpected_nulls", "Unexpected Nulls"),
            ),
            self._quality_vars,
            cols=3,
        )

        audit_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(audit_tab, text="Audit")
        self._kv_grid(
            audit_tab,
            (
                ("status", "Status"),
                ("audited_at", "Audited At"),
                ("warnings", "Warnings"),
                ("critical_issues", "Critical Issues"),
                ("training_recommendation", "Training Recommendation"),
                ("dataset_health_pct", "Dataset Health %"),
            ),
            self._audit_vars,
            cols=3,
        )
        ttk.Label(
            audit_tab, textvariable=self._audit_detail, justify="left", wraplength=720
        ).pack(anchor="w", pady=(12, 0))

        val_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(val_tab, text="Validation")
        self._kv_grid(
            val_tab,
            (
                ("status", "Status"),
                ("label", "Label"),
                ("validated_at", "Validated At"),
            ),
            self._validation_vars,
            cols=3,
        )

        reg_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(reg_tab, text="Registry")
        self._kv_grid(
            reg_tab,
            (
                ("expected", "Expected Features"),
                ("implemented", "Implemented"),
                ("coverage_pct", "Coverage %"),
                ("spec_hash_match", "Spec Hash Match"),
                ("formula_validation", "Formula Validation"),
                ("replay_validation", "Replay Validation"),
            ),
            self._reg_vars,
            cols=3,
        )
        ttk.Label(reg_tab, textvariable=self._reg_detail, justify="left", wraplength=720).pack(
            anchor="w", pady=(12, 0)
        )

        build_tab = ttk.Frame(main_nb, padding=4)
        main_nb.add(build_tab, text="Build Info")
        build_scroll = ttk.Frame(build_tab)
        build_scroll.pack(fill="both", expand=True)

        build_cfg = ttk.LabelFrame(build_scroll, text="Build configuration", padding=8)
        build_cfg.pack(fill="x", pady=(0, 8))
        self._kv_grid(
            build_cfg,
            (
                ("market", "Market"),
                ("sampling", "Sampling interval"),
                ("strikes", "Strike selection"),
                ("feature_count", "Features"),
                ("target_count", "Targets"),
                ("builder_version", "Builder version"),
                ("dataset_version", "Dataset version"),
                ("created_at", "Created at"),
            ),
            self._build_vars,
            cols=4,
        )

        filters = ttk.LabelFrame(build_scroll, text="Filters & lineage", padding=8)
        filters.pack(fill="x", pady=(0, 8))
        self._filter_text = scrolledtext.ScrolledText(
            filters, height=8, wrap="word", font=("Consolas", 9)
        )
        self._filter_text.pack(fill="both", expand=True)
        self._filter_text.configure(state="disabled")

        actions = ttk.Frame(self, padding=(8, 0, 8, 8))
        actions.pack(fill="x")
        self._open_file_btn = ttk.Button(
            actions, text="Open metadata file", command=self._open_metadata_file, state="disabled"
        )
        self._open_file_btn.pack(side="left")
        self._open_csv_btn = ttk.Button(
            actions, text="Open CSV", command=self._open_csv, state="disabled"
        )
        self._open_csv_btn.pack(side="left", padx=(4, 0))
        self._open_csv_folder_btn = ttk.Button(
            actions, text="Open CSV folder", command=self._open_csv_folder, state="disabled"
        )
        self._open_csv_folder_btn.pack(side="left", padx=(4, 0))
        self._delete_csv_btn = ttk.Button(
            actions, text="Delete CSV", command=self._delete_csv, state="disabled"
        )
        self._delete_csv_btn.pack(side="left", padx=(4, 0))

    def refresh(self) -> None:
        if not self.dataset_name:
            self._status.set("No dataset")
            return
        self._load_gen += 1
        gen = self._load_gen
        self._status.set("Loading…")

        def worker() -> None:
            err = ""
            summary: dict[str, Any] = {}
            meta_doc: dict[str, Any] = {}
            coverage_rows: list[dict[str, Any]] = []
            try:
                summary = svc.load_dataset_summary(self.chart_dir, self.dataset_name)
                meta_doc = svc.load_dataset_metadata(self.chart_dir, self.dataset_name)
                coverage_rows = _feature_coverage_rows(self.chart_dir, self.dataset_name)
            except Exception as exc:
                err = str(exc)

            def apply() -> None:
                if gen != self._load_gen:
                    return
                if err:
                    self._status.set(f"Load failed: {err}")
                    self._clear_all()
                    return
                self._summary = summary
                self._meta_doc = meta_doc
                self._columns_cache = coverage_rows
                self._day_blocks = _day_blocks(meta_doc.get("metadata") or {})
                days = [
                    str(b.get("trading_day") or "").strip()
                    for b in self._day_blocks
                    if str(b.get("trading_day") or "").strip()
                ]
                if not days:
                    labels = str(
                        (summary.get("dataset") or {}).get("trading_day_labels") or ""
                    )
                    days = [d.strip() for d in labels.split(",") if d.strip()]
                self._day_cb.configure(values=days)
                if days:
                    current = self._day_var.get().strip()
                    pick = current if current in days else days[0]
                    self._day_var.set(pick)
                else:
                    self._day_var.set("")
                self._fill_overview(summary, meta_doc)
                self._fill_health(summary, coverage_rows)
                self._fill_source_counts(coverage_rows)
                self._fill_quality(summary)
                self._fill_audit(summary)
                self._fill_validation(summary)
                self._fill_registry(summary)
                self._fill_build_info(summary)
                self._apply_day_facts()
                self._render_columns()
                self._apply_csv_button_states(meta_doc)
                self._status.set(f"Loaded {self.dataset_name}")

            self.after(0, apply)

        threading.Thread(target=worker, daemon=True, name=f"ds-meta-{self.dataset_name}").start()

    def _clear_all(self) -> None:
        for store in (
            self._fact_vars,
            self._health_vars,
            self._source_vars,
            self._build_vars,
            self._quality_vars,
            self._audit_vars,
            self._validation_vars,
            self._reg_vars,
        ):
            for v in store.values():
                v.set("—")
        self._score_detail.set("")
        self._reg_detail.set("")
        self._audit_detail.set("")
        self._columns_cache = []
        self._cov_tree.delete(*self._cov_tree.get_children())
        self._filter_text.configure(state="normal")
        self._filter_text.delete("1.0", tk.END)
        self._filter_text.configure(state="disabled")

    def _fill_overview(self, summary: dict[str, Any], meta_doc: dict[str, Any]) -> None:
        ds = summary.get("dataset") or {}
        sampling = summary.get("sampling") or {}
        audit = summary.get("audit") or {}
        validation = summary.get("validation") or {}
        conf = summary.get("dataset_confidence") or {}
        readiness = summary.get("model_training_readiness") or {}
        self._fact_vars["total_rows"].set(_fmt_int(ds.get("rows")))
        self._fact_vars["total_columns"].set(_fmt_int(ds.get("total_columns")))
        self._fact_vars["meta_columns"].set(_fmt_int(ds.get("metadata_columns")))
        self._fact_vars["feature_count"].set(_fmt_int(ds.get("features")))
        self._fact_vars["target_count"].set(_fmt_int(ds.get("targets")))
        self._fact_vars["market"].set(str(ds.get("market") or "—"))
        self._fact_vars["sampling"].set(str(sampling.get("interval_label") or "—"))
        self._fact_vars["audit_status"].set(str(audit.get("status") or "—"))
        self._fact_vars["validation_status"].set(str(validation.get("label") or "—"))
        pct = conf.get("pct")
        self._fact_vars["confidence"].set(
            f"{_fmt_num(pct, 1)}%" if pct is not None else "—"
        )
        self._fact_vars["training_ready"].set(
            str(readiness.get("label") or audit.get("training_recommendation_display") or "—")
        )

    def _apply_day_facts(self) -> None:
        day = self._day_var.get().strip()
        block = next(
            (b for b in self._day_blocks if str(b.get("trading_day") or "") == day),
            None,
        )
        if block:
            self._fact_vars["trading_day"].set(day)
            self._fact_vars["total_rows"].set(_fmt_int(block.get("rows")))
            dur = block.get("trading_duration_sec") or block.get("duration_sec")
            self._fact_vars["trading_duration_sec"].set(_fmt_dur(dur))
        else:
            ds = self._summary.get("dataset") or {}
            labels = str(ds.get("trading_day_labels") or "—")
            self._fact_vars["trading_day"].set(labels if labels else "—")
            self._fact_vars["total_rows"].set(_fmt_int(ds.get("rows")))
            self._fact_vars["trading_duration_sec"].set("—")

    def _fill_health(self, summary: dict[str, Any], coverage_rows: list[dict[str, Any]]) -> None:
        quality = summary.get("quality") or {}
        feat = summary.get("feature_summary") or {}
        pipeline = summary.get("pipeline") or {}
        conf = summary.get("dataset_confidence") or {}

        healthy = sum(1 for r in coverage_rows if r.get("status") == "Healthy")
        partial = sum(1 for r in coverage_rows if r.get("status") == "Partial")
        sparse = sum(
            1
            for r in coverage_rows
            if r.get("status") not in ("Healthy", "Partial") and float(r.get("coverage_pct") or 0) > 0
        )
        complete = healthy

        spec_ok = pipeline.get("fingerprint_match")
        if spec_ok is True:
            reg_match = "OK"
        elif spec_ok is False:
            reg_match = "Spec mismatch"
        else:
            reg_match = "—"

        self._health_vars["average_coverage"].set(_fmt_pct(feat.get("coverage_pct")))
        self._health_vars["unexpected_nulls"].set(_fmt_int(quality.get("unexpected_nulls")))
        self._health_vars["expected_nulls"].set(_fmt_int(quality.get("expected_nulls")))
        self._health_vars["invalid_timestamps"].set(_fmt_int(quality.get("invalid_timestamps")))
        self._health_vars["registry_match"].set(reg_match)
        self._health_vars["complete_features"].set(_fmt_int(complete))
        self._health_vars["partial_features"].set(_fmt_int(partial))
        self._health_vars["sparse_features"].set(_fmt_int(sparse))
        self._health_vars["healthy_features"].set(_fmt_int(healthy))
        hs = conf.get("pct")
        self._health_vars["health_score"].set(
            f"{_fmt_num(hs, 1)}/100" if hs is not None else "—"
        )
        comps = conf.get("components") or []
        if comps:
            parts = [
                f"{c.get('label') or c.get('id')}: {_fmt_num(c.get('score_pct'), 1)}"
                for c in comps[:6]
            ]
            self._score_detail.set("Derived summary · " + " · ".join(parts))
        else:
            self._score_detail.set("Run Audit for full investigation detail.")

    def _fill_source_counts(self, coverage_rows: list[dict[str, Any]]) -> None:
        from chain_replay_ml.dataset_builder.feature_sources_catalog import (
            DATASET_SOURCE_BASE_PIPELINE,
            DATASET_SOURCE_FEATURE_REGISTRY,
            DATASET_SOURCE_OTHER_PIPELINE,
            dataset_base_pipeline_export_feature_names,
            dataset_registry_export_feature_names,
            other_pipeline_feature_names_from_metadata,
        )

        counts = {
            DATASET_SOURCE_FEATURE_REGISTRY: 0,
            DATASET_SOURCE_BASE_PIPELINE: 0,
            DATASET_SOURCE_OTHER_PIPELINE: 0,
        }
        for row in coverage_rows:
            bucket = str(row.get("source_bucket") or DATASET_SOURCE_OTHER_PIPELINE)
            if bucket in counts:
                counts[bucket] += 1
            else:
                counts[DATASET_SOURCE_OTHER_PIPELINE] += 1
        total = len(coverage_rows)
        meta = (self._meta_doc or {}).get("metadata") or {}
        data_dir = svc.data_dir_for(self.chart_dir)
        reg_catalogue = len(dataset_registry_export_feature_names(meta, data_dir=data_dir))
        base_catalogue = len(dataset_base_pipeline_export_feature_names(meta, data_dir=data_dir))
        other_catalogue = len(other_pipeline_feature_names_from_metadata(meta))

        reg_n = counts[DATASET_SOURCE_FEATURE_REGISTRY]
        if reg_catalogue and reg_n != reg_catalogue:
            self._source_vars["feature_registry"].set(
                f"{_fmt_int(reg_n)} / {_fmt_int(reg_catalogue)}"
            )
        else:
            self._source_vars["feature_registry"].set(_fmt_int(reg_n))

        base_n = counts[DATASET_SOURCE_BASE_PIPELINE]
        if base_catalogue:
            self._source_vars["base_pipeline"].set(
                f"{_fmt_int(base_n)} / {_fmt_int(base_catalogue)}"
            )
        else:
            self._source_vars["base_pipeline"].set(_fmt_int(base_n))

        other_n = counts[DATASET_SOURCE_OTHER_PIPELINE]
        if other_catalogue and other_n != other_catalogue:
            self._source_vars["other_pipeline"].set(
                f"{_fmt_int(other_n)} / {_fmt_int(other_catalogue)}"
            )
        else:
            self._source_vars["other_pipeline"].set(_fmt_int(other_n))

        self._source_vars["feature_total"].set(_fmt_int(total))

    def _fill_quality(self, summary: dict[str, Any]) -> None:
        q = summary.get("quality") or {}
        for key in self._quality_vars:
            self._quality_vars[key].set(_fmt_int(q.get(key)))

    def _fill_audit(self, summary: dict[str, Any]) -> None:
        audit = summary.get("audit") or {}
        for key in ("status", "audited_at", "warnings", "critical_issues"):
            val = audit.get(key)
            self._audit_vars[key].set(
                _fmt_int(val) if key in ("warnings", "critical_issues") else str(val or "—")
            )
        self._audit_vars["training_recommendation"].set(
            str(audit.get("training_recommendation_display") or audit.get("training_recommendation") or "—")
        )
        self._audit_vars["dataset_health_pct"].set(
            _fmt_pct(audit.get("dataset_health_pct"))
            if audit.get("dataset_health_pct") is not None
            else "—"
        )
        findings = audit.get("findings")
        blocking = audit.get("blocking_issues")
        lines: list[str] = []
        blocking_lines = _string_list_lines(blocking, limit=12)
        if blocking_lines:
            lines.append("Blocking issues:\n  " + "\n  ".join(blocking_lines))
        finding_lines = _audit_finding_lines(findings, limit=12)
        if finding_lines:
            lines.append("Findings:\n  " + "\n  ".join(finding_lines))
        self._audit_detail.set("\n\n".join(lines) if lines else "No audit findings recorded.")

    def _fill_validation(self, summary: dict[str, Any]) -> None:
        val = summary.get("validation") or {}
        for key in self._validation_vars:
            self._validation_vars[key].set(str(val.get(key) or "—"))

    def _fill_registry(self, summary: dict[str, Any]) -> None:
        feat = summary.get("feature_summary") or {}
        pipeline = summary.get("pipeline") or {}
        self._reg_vars["expected"].set(_fmt_int(feat.get("expected")))
        self._reg_vars["implemented"].set(_fmt_int(feat.get("implemented")))
        self._reg_vars["coverage_pct"].set(_fmt_pct(feat.get("coverage_pct")))
        match = pipeline.get("fingerprint_match")
        self._reg_vars["spec_hash_match"].set(
            "Match" if match is True else ("Mismatch" if match is False else "—")
        )
        self._reg_vars["formula_validation"].set(str(feat.get("formula_display") or "—"))
        self._reg_vars["replay_validation"].set(str(feat.get("replay_display") or "—"))
        groups = summary.get("feature_group_coverage") or []
        if groups:
            lines = [str(g.get("display") or g.get("label") or g) for g in groups[:20]]
            self._reg_detail.set("Feature groups:\n  " + "\n  ".join(lines))
        else:
            self._reg_detail.set("Registry feature groups — see Build Info for filters.")

    def _fill_build_info(self, summary: dict[str, Any]) -> None:
        ds = summary.get("dataset") or {}
        sampling = summary.get("sampling") or {}
        pipeline = summary.get("pipeline") or {}
        meta = self._meta_doc.get("metadata") or {}
        self._build_vars["market"].set(str(ds.get("market") or "—"))
        self._build_vars["sampling"].set(str(sampling.get("interval_label") or "—"))
        self._build_vars["strikes"].set(str(sampling.get("strikes_label") or "—"))
        self._build_vars["feature_count"].set(_fmt_int(ds.get("features")))
        self._build_vars["target_count"].set(_fmt_int(ds.get("targets")))
        self._build_vars["builder_version"].set(str(pipeline.get("builder_version") or "—"))
        self._build_vars["dataset_version"].set(str(pipeline.get("dataset_version") or "—"))
        self._build_vars["created_at"].set(str(meta.get("created_at") or "—")[:19])

        lines: list[str] = []
        for row in summary.get("filter_summary") or []:
            if isinstance(row, dict):
                lines.append(f"{row.get('label') or '—'}: {row.get('value') or '—'}")
        lineage = summary.get("dataset_lineage") or {}
        if isinstance(lineage, dict) and lineage:
            lines.append("")
            lines.append("Lineage")
            for k, v in lineage.items():
                lines.append(f"  {k}: {v}")
        pipe = summary.get("pipeline") or {}
        if pipe.get("git_commit"):
            lines.append(f"  git_commit: {pipe.get('git_commit')}")
        self._filter_text.configure(state="normal")
        self._filter_text.delete("1.0", tk.END)
        self._filter_text.insert("1.0", "\n".join(lines) if lines else "No filter summary recorded.")
        self._filter_text.configure(state="disabled")

    def _render_columns(self) -> None:
        tree = self._cov_tree
        tree.delete(*tree.get_children())
        mode = str(self._filter.get() or "all")
        source_mode = str(self._source_filter.get() or "feature_registry")
        family = str(self._family.get() or "All")
        self._row_by_iid = {}
        for row in self._columns_cache:
            if str(row.get("source_bucket") or "other_pipeline") != source_mode:
                continue
            status = str(row.get("status") or "")
            fam = str(row.get("feature_family") or "")
            cov = float(row.get("coverage_pct") or 0.0)
            if family != "All" and fam != family:
                continue
            if mode == "problems" and status == "Healthy":
                continue
            if mode == "coverage_lt_99" and cov >= 99.0:
                continue
            if mode == "Healthy" and status != "Healthy":
                continue
            tag = "ok"
            if status == "Unexpected Empty":
                tag = "unexpected"
            elif status == "Partial":
                tag = "expected"
            iid = tree.insert(
                "",
                "end",
                values=(
                    row.get("feature"),
                    fam,
                    row.get("source") or "",
                    status,
                    row.get("availability") or "",
                    _fmt_pct(cov),
                    row.get("reason") or "",
                ),
                tags=(tag,),
            )
            self._row_by_iid[iid] = row

    def _on_feature_select(self, _event: tk.Event | None = None) -> None:
        sel = self._cov_tree.selection()
        if not sel:
            self._detail_var.set("Select a feature")
            return
        row = self._row_by_iid.get(sel[0]) or {}
        self._detail_var.set(
            "\n".join(
                [
                    str(row.get("feature") or "—"),
                    f"Source: {row.get('source') or '—'}",
                    f"Category: {row.get('feature_family') or '—'}",
                    f"Status: {row.get('status') or '—'}",
                    f"Coverage: {_fmt_pct(row.get('coverage_pct'))}",
                    str(row.get("reason") or ""),
                ]
            )
        )

    def _apply_csv_button_states(self, data: dict[str, Any]) -> None:
        csv_info = data.get("csv_export") or {}
        has_csv = csv_info.get("status") == "Generated"
        state = "normal" if has_csv else "disabled"
        self._open_csv_btn.configure(state=state)
        self._open_csv_folder_btn.configure(state=state)
        self._delete_csv_btn.configure(state=state)
        meta_path = self.metadata_path
        if not meta_path:
            meta_path = self._metadata_path_from_row(data)
        self._open_file_btn.configure(
            state="normal" if meta_path and os.path.isfile(meta_path) else "disabled"
        )

    def _metadata_path_from_row(self, data: dict[str, Any]) -> str | None:
        name = str(data.get("dataset_name") or self.dataset_name)
        from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

        data_dir = svc.data_dir_for(self.chart_dir)
        safe = _safe_filename(name)
        path = os.path.join(datasets_dir(data_dir), f"{safe}.json")
        return path if os.path.isfile(path) else None

    def _open_metadata_file(self) -> None:
        path = self.metadata_path
        if not path and self._meta_doc:
            path = self._metadata_path_from_row(self._meta_doc)
        if path and os.path.isfile(path):
            open_path(path)

    def _open_csv(self) -> None:
        info = (self._meta_doc or {}).get("csv_export") or {}
        try:
            if not info:
                info = svc.load_dataset_metadata(self.chart_dir, self.dataset_name).get(
                    "csv_export"
                ) or {}
        except Exception as exc:
            messagebox.showerror("Open CSV", str(exc), parent=self.winfo_toplevel())
            return
        path = info.get("csv_path")
        if not path or not os.path.isfile(path):
            messagebox.showinfo("Open CSV", "No CSV export found.", parent=self.winfo_toplevel())
            self.refresh()
            return
        open_path(path)

    def _open_csv_folder(self) -> None:
        info = (self._meta_doc or {}).get("csv_export") or {}
        try:
            if not info:
                info = svc.load_dataset_metadata(self.chart_dir, self.dataset_name).get(
                    "csv_export"
                ) or {}
        except Exception as exc:
            messagebox.showerror("Open CSV folder", str(exc), parent=self.winfo_toplevel())
            return
        path = info.get("csv_path")
        if not path:
            messagebox.showinfo(
                "Open CSV folder", "No CSV export found.", parent=self.winfo_toplevel()
            )
            return
        open_path(os.path.dirname(path))

    def _delete_csv(self) -> None:
        parent = self.winfo_toplevel()
        info = (self._meta_doc or {}).get("csv_export") or {}
        if info.get("status") != "Generated":
            messagebox.showinfo(
                "Delete CSV", "No CSV export exists for this dataset.", parent=parent
            )
            return
        if not messagebox.askyesno(
            "Delete CSV",
            f'Delete the CSV export for "{self.dataset_name}"?\n\n'
            "The Parquet dataset and registry entry will not be removed.",
            parent=parent,
        ):
            return
        try:
            svc.delete_registry_csv(self.chart_dir, self.dataset_name)
        except Exception as exc:
            messagebox.showerror("Delete CSV", str(exc), parent=parent)
            return
        messagebox.showinfo("Delete CSV", "CSV export deleted.", parent=parent)
        self.refresh()


def open_dataset_metadata_window(
    master: tk.Misc,
    *,
    chart_dir: str,
    dataset_name: str,
    metadata_path: str | None = None,
) -> tk.Toplevel:
    """Open dataset metadata beside the main window (same pattern as Day Metadata)."""
    win = tk.Toplevel(master)
    name = str(dataset_name or "").strip()
    win.title(f"Dataset Metadata — {name}")
    win.transient(master.winfo_toplevel())
    panel = DatasetMetadataPanel(
        win,
        chart_dir=chart_dir,
        dataset_name=name,
        metadata_path=metadata_path,
    )
    panel.pack(fill="both", expand=True)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win
