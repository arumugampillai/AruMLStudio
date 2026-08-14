"""Day Metadata panel — Data Overview tab + Build Info tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .fold_replay_widgets import place_toplevel_beside_main


def _fmt_int(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_clock(ts: Any) -> str:
    try:
        from chain_replay_ml.dataset_builder.day_metadata import format_clock

        return format_clock(float(ts) if ts is not None else None)
    except Exception:
        return "—"


def _fmt_dur(sec: Any) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "—"
    if s < 120:
        return f"{s:.0f} sec"
    return f"{s / 60:.1f} min"


class DayMetadataPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        db_path: str,
        trading_days: list[str],
        on_rebuild: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.db_path = db_path
        self._days = list(trading_days)
        self._on_rebuild = on_rebuild
        self._filter = tk.StringVar(value="all")
        self._family = tk.StringVar(value="All")
        self._day_var = tk.StringVar(value=self._days[0] if self._days else "")
        self._status = tk.StringVar(value="")
        self._columns_cache: list[dict[str, Any]] = []
        self._ov: dict[str, Any] = {}
        self._fact_vars: dict[str, tk.StringVar] = {}
        self._health_vars: dict[str, tk.StringVar] = {}
        self._build_vars: dict[str, tk.StringVar] = {}
        self._build_ui()
        if self._days:
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
        ttk.Label(top, text="Day Metadata", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, text="Trading day").pack(side="left", padx=(16, 4))
        self._day_cb = ttk.Combobox(
            top, textvariable=self._day_var, values=self._days, width=14, state="readonly"
        )
        self._day_cb.pack(side="left")
        self._day_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=8)
        ttk.Button(top, text="Rebuild metadata…", command=self._rebuild).pack(side="left")
        ttk.Label(top, textvariable=self._status, foreground="#888").pack(side="right")

        main_nb = ttk.Notebook(self)
        main_nb.pack(fill="both", expand=True, padx=8, pady=4)

        # Tab 1 — Data Overview (day facts + detail sub-tabs)
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
                ("first_timestamp", "First Timestamp"),
                ("last_timestamp", "Last Timestamp"),
                ("token_count", "Token Count"),
                ("expiry", "Expiry"),
                ("trading_duration_sec", "Trading Duration"),
            ),
            self._fact_vars,
            cols=3,
        )

        day_stats = ttk.LabelFrame(row1, text="Day Statistics", padding=8)
        day_stats.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._kv_grid(
            day_stats,
            (
                ("spot_min", "Spot Min"),
                ("spot_max", "Spot Max"),
                ("avg_iv", "Average IV"),
                ("avg_spread", "Average Spread"),
            ),
            self._fact_vars,
            cols=2,
        )

        row2 = ttk.Frame(data_tab)
        row2.pack(fill="x", pady=(8, 0))
        row2.columnconfigure(0, weight=1)

        health = ttk.LabelFrame(row2, text="Health Assessment (derived)", padding=8)
        health.grid(row=0, column=0, sticky="nsew")
        self._kv_grid(
            health,
            (
                ("average_coverage", "Overall Coverage"),
                ("gap_events", "Gap Events"),
                ("missing_timestamps", "Missing Samples"),
                ("warmup_features", "Warm-up Features"),
                ("registry_match", "Registry Match"),
                ("expected_empty_features", "Expected Empty"),
                ("unexpected_empty_features", "Unexpected Empty"),
                ("complete_features", "Complete"),
                ("partial_features", "Partial"),
                ("sparse_features", "Sparse"),
                ("healthy_features", "Healthy"),
                ("health_score", "Health Score"),
            ),
            self._health_vars,
            cols=4,
        )
        self._score_detail = tk.StringVar(value="")
        ttk.Label(
            health, textvariable=self._score_detail, foreground="#666", justify="left"
        ).pack(anchor="w", pady=(4, 0))

        detail_nb = ttk.Notebook(data_tab)
        detail_nb.pack(fill="both", expand=True, pady=(8, 0))

        # Coverage
        cov_tab = ttk.Frame(detail_nb, padding=4)
        detail_nb.add(cov_tab, text="Coverage")
        filt = ttk.Frame(cov_tab)
        filt.pack(fill="x", pady=(0, 4))
        for value, label in (
            ("all", "All"),
            ("problems", "Problems"),
            ("unexpected_empty", "Unexpected Empty"),
            ("expected_empty", "Expected Empty"),
            ("coverage_lt_99", "Coverage <99%"),
            ("warmup", "Warm-up"),
            ("Healthy", "Healthy"),
        ):
            ttk.Radiobutton(
                filt, text=label, value=value, variable=self._filter, command=self._render_columns
            ).pack(side="left", padx=3)
        ttk.Label(filt, text="Category").pack(side="left", padx=(12, 4))
        self._family_cb = ttk.Combobox(
            filt,
            textvariable=self._family,
            values=["All", "Base", "Derived", "Market Structure", "Greeks", "Volatility",
                    "Order Book", "Prediction", "Target", "Meta", "Other"],
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

        cols = (
            "feature", "family", "source", "status", "availability",
            "coverage", "reason",
        )
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
            self._cov_tree.column(c, width=w, anchor="w" if c in ("feature", "reason") else "center")
        sb = ttk.Scrollbar(cov_left, orient="vertical", command=self._cov_tree.yview)
        self._cov_tree.configure(yscrollcommand=sb.set)
        self._cov_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._cov_tree.bind("<<TreeviewSelect>>", self._on_feature_select)
        self._cov_tree.tag_configure("unexpected", foreground="#C62828")
        self._cov_tree.tag_configure("expected", foreground="#546E7A")
        self._cov_tree.tag_configure("ok", foreground="#2E7D32")

        self._detail_var = tk.StringVar(value="Select a feature")
        ttk.Label(cov_right, textvariable=self._detail_var, justify="left", wraplength=260).pack(
            anchor="nw"
        )

        # Gaps
        gap_tab = ttk.Frame(detail_nb, padding=4)
        detail_nb.add(gap_tab, text="Gap Analysis")
        self._gap_summary = tk.StringVar(value="")
        ttk.Label(gap_tab, textvariable=self._gap_summary, justify="left").pack(anchor="w", pady=(0, 6))
        gcols = ("start", "end", "duration", "missing", "token", "action")
        self._gap_tree = ttk.Treeview(gap_tab, columns=gcols, show="headings", height=12)
        for c, (txt, w) in {
            "start": ("Start", 90),
            "end": ("End", 90),
            "duration": ("Duration", 80),
            "missing": ("Missing Samples", 110),
            "token": ("Token", 90),
            "action": ("Action", 140),
        }.items():
            self._gap_tree.heading(c, text=txt)
            self._gap_tree.column(c, width=w, anchor="center")
        gsb = ttk.Scrollbar(gap_tab, orient="vertical", command=self._gap_tree.yview)
        self._gap_tree.configure(yscrollcommand=gsb.set)
        self._gap_tree.pack(side="left", fill="both", expand=True)
        gsb.pack(side="right", fill="y")

        # Timestamps
        ts_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(ts_tab, text="Timestamp Quality")
        self._ts_vars: dict[str, tk.StringVar] = {}
        self._kv_grid(
            ts_tab,
            (
                ("first_timestamp", "First Timestamp"),
                ("last_timestamp", "Last Timestamp"),
                ("sampling_interval_sec", "Expected Interval"),
                ("observed_interval_sec", "Observed Interval"),
                ("duplicate_timestamps", "Duplicate Timestamps"),
                ("out_of_order_timestamps", "Out-of-order"),
                ("missing_timestamps", "Missing Timestamps"),
                ("actual_unique_timestamps", "Actual Unique Timestamps"),
                ("expected_samples", "Expected Samples"),
                ("largest_gap_sec", "Largest Continuous Gap"),
            ),
            self._ts_vars,
            cols=3,
        )

        # Registry
        reg_tab = ttk.Frame(detail_nb, padding=8)
        detail_nb.add(reg_tab, text="Registry")
        self._reg_vars: dict[str, tk.StringVar] = {}
        self._kv_grid(
            reg_tab,
            (
                ("registry_expected", "Expected Features"),
                ("registry_found", "Found"),
                ("registry_missing", "Missing"),
                ("registry_unexpected", "Unexpected"),
                ("duplicate_columns", "Duplicate Columns"),
            ),
            self._reg_vars,
            cols=3,
        )
        self._reg_detail = tk.StringVar(value="")
        ttk.Label(reg_tab, textvariable=self._reg_detail, justify="left", wraplength=720).pack(
            anchor="w", pady=(12, 0)
        )

        # Tab 2 — Build Info (dataset-level configuration)
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
                ("sliding_stride", "Sliding stride"),
                ("feature_window", "Feature window"),
                ("strike_selection", "Strike selection"),
                ("gap_policy", "Gap policy"),
                ("target_labels", "Target labels"),
                ("feature_count", "Features"),
                ("target_count", "Targets"),
                ("atm_band", "ATM band"),
                ("lookback_policy", "Lookback policy"),
            ),
            self._build_vars,
            cols=3,
        )

        feat_sel = ttk.LabelFrame(build_scroll, text="Feature selection", padding=8)
        feat_sel.pack(fill="x", pady=(0, 8))
        self._kv_grid(
            feat_sel,
            (
                ("feature_profile", "Profile"),
                ("feature_groups", "Enabled groups"),
            ),
            self._build_vars,
            cols=2,
        )
        feat_text_frame = ttk.Frame(feat_sel)
        feat_text_frame.pack(fill="both", expand=True, pady=(8, 0))
        feat_sb = ttk.Scrollbar(feat_text_frame, orient="vertical")
        self._feature_list_text = tk.Text(
            feat_text_frame,
            height=10,
            wrap="none",
            font=("Consolas", 9),
            yscrollcommand=feat_sb.set,
        )
        feat_sb.config(command=self._feature_list_text.yview)
        self._feature_list_text.pack(side="left", fill="both", expand=True)
        feat_sb.pack(side="right", fill="y")
        self._feature_list_text.configure(state="disabled")

        provenance = ttk.LabelFrame(build_scroll, text="Build provenance", padding=8)
        provenance.pack(fill="x")
        self._kv_grid(
            provenance,
            (
                ("dataset_version", "Dataset version"),
                ("registry_version", "Registry version"),
                ("feature_engine_version", "Feature engine"),
                ("gap_policy_version", "Gap policy version"),
                ("builder_version", "Builder version"),
                ("schema_hash", "Schema hash"),
                ("created_at", "Created at"),
                ("updated_at", "Updated at"),
            ),
            self._build_vars,
            cols=4,
        )

    def refresh(self) -> None:
        day = str(self._day_var.get() or "").strip()
        if not day:
            self._status.set("Select a trading day")
            return
        try:
            from chain_replay_ml.dataset_builder.day_metadata import (
                load_column_metadata,
                load_day_overview,
                load_gap_metadata,
            )
            from chain_replay_ml.dataset_builder.master_store import MasterStore

            store = MasterStore(self.db_path)
            store.open()
            try:
                overview = load_day_overview(store.conn, day)
                columns = load_column_metadata(store.conn, day)
                gaps = load_gap_metadata(store.conn, day)
            finally:
                store.close()
        except Exception as exc:
            self._status.set(f"Load failed: {exc}")
            return

        if not overview:
            self._status.set("No metadata — click Rebuild metadata…")
            self._ov = {}
            self._columns_cache = []
            self._clear_vars()
            self._fill_build_info()
            self._render_columns()
            self._render_gaps({})
            return

        self._status.set(f"Loaded {day}")
        self._ov = overview
        self._columns_cache = columns
        self._fill_facts(overview)
        self._fill_health(overview)
        self._fill_timestamps(overview)
        self._fill_registry(overview)
        self._fill_build_info(day_overview=overview)
        self._render_columns()
        self._render_gaps(gaps)

    def _clear_vars(self) -> None:
        for d in (
            self._fact_vars,
            self._health_vars,
            self._build_vars,
            self._ts_vars,
            self._reg_vars,
        ):
            for v in d.values():
                v.set("—")
        self._score_detail.set("")
        self._reg_detail.set("")
        self._gap_summary.set("")
        if hasattr(self, "_feature_list_text"):
            self._feature_list_text.configure(state="normal")
            self._feature_list_text.delete("1.0", tk.END)
            self._feature_list_text.configure(state="disabled")

    def _fill_facts(self, ov: dict[str, Any]) -> None:
        self._fact_vars["trading_day"].set(str(ov.get("trading_day") or "—"))
        self._fact_vars["total_rows"].set(_fmt_int(ov.get("total_rows")))
        self._fact_vars["total_columns"].set(_fmt_int(ov.get("total_columns")))
        self._fact_vars["meta_columns"].set(_fmt_int(ov.get("meta_columns")))
        self._fact_vars["first_timestamp"].set(_fmt_clock(ov.get("first_timestamp")))
        self._fact_vars["last_timestamp"].set(_fmt_clock(ov.get("last_timestamp")))
        self._fact_vars["token_count"].set(_fmt_int(ov.get("token_count")))
        self._fact_vars["expiry"].set(str(ov.get("expiry") or "—"))
        self._fact_vars["trading_duration_sec"].set(_fmt_dur(ov.get("trading_duration_sec")))
        self._fact_vars["spot_min"].set(_fmt_num(ov.get("spot_min"), 2))
        self._fact_vars["spot_max"].set(_fmt_num(ov.get("spot_max"), 2))
        self._fact_vars["avg_iv"].set(_fmt_num(ov.get("avg_iv"), 4))
        self._fact_vars["avg_spread"].set(_fmt_num(ov.get("avg_spread"), 4))

    def _fill_health(self, ov: dict[str, Any]) -> None:
        missing = int(ov.get("registry_missing") or 0)
        unexpected = int(ov.get("registry_unexpected") or 0)
        match = "OK" if missing == 0 and unexpected == 0 else f"{missing} missing / {unexpected} unexpected"
        self._health_vars["average_coverage"].set(_fmt_pct(ov.get("average_coverage")))
        self._health_vars["gap_events"].set(_fmt_int(ov.get("gap_events")))
        self._health_vars["missing_timestamps"].set(_fmt_int(ov.get("missing_timestamps")))
        self._health_vars["warmup_features"].set(_fmt_int(ov.get("warmup_features")))
        self._health_vars["registry_match"].set(match)
        self._health_vars["expected_empty_features"].set(_fmt_int(ov.get("expected_empty_features")))
        self._health_vars["unexpected_empty_features"].set(_fmt_int(ov.get("unexpected_empty_features")))
        self._health_vars["complete_features"].set(_fmt_int(ov.get("complete_features")))
        self._health_vars["partial_features"].set(_fmt_int(ov.get("partial_features")))
        self._health_vars["sparse_features"].set(_fmt_int(ov.get("sparse_features")))
        self._health_vars["healthy_features"].set(_fmt_int(ov.get("healthy_features")))
        hs = ov.get("health_score")
        self._health_vars["health_score"].set(f"{_fmt_num(hs, 1)}/100" if hs is not None else "—")
        comps = ov.get("health_components") or {}
        self._score_detail.set(
            "Derived summary · Coverage {c} · Gap {g} · Registry {r} · Timestamp {t} · Null {n}".format(
                c=_fmt_num(comps.get("coverage"), 1),
                g=_fmt_num(comps.get("gap_quality"), 1),
                r=_fmt_num(comps.get("registry"), 1),
                t=_fmt_num(comps.get("timestamp_quality"), 1),
                n=_fmt_num(comps.get("null_quality"), 1),
            )
        )

    def _fill_build_info(self, *, day_overview: dict[str, Any] | None = None) -> None:
        if not hasattr(self, "_build_vars"):
            return
        try:
            from chain_replay_ml.dataset_builder.master_store import MasterStore

            from .day_build_info import format_feature_names_text, load_master_build_info

            store = MasterStore(self.db_path)
            store.open()
            try:
                info = load_master_build_info(store)
            finally:
                store.close()
        except Exception as exc:
            for v in self._build_vars.values():
                v.set("—")
            if hasattr(self, "_feature_list_text"):
                self._feature_list_text.configure(state="normal")
                self._feature_list_text.delete("1.0", tk.END)
                self._feature_list_text.insert("1.0", f"Build info unavailable: {exc}")
                self._feature_list_text.configure(state="disabled")
            return

        for key, val in (info.get("kv_fields") or {}).items():
            if key in self._build_vars:
                self._build_vars[key].set(str(val))
        if day_overview:
            meta_at = str(
                day_overview.get("metadata_generated_at")
                or day_overview.get("updated_at")
                or ""
            )[:19]
            if meta_at and self._build_vars.get("updated_at"):
                self._build_vars["updated_at"].set(meta_at)
            imported = str(day_overview.get("imported_at") or "")[:19]
            if imported and self._build_vars.get("created_at"):
                if self._build_vars["created_at"].get() == "—":
                    self._build_vars["created_at"].set(imported)
        if hasattr(self, "_feature_list_text"):
            self._feature_list_text.configure(state="normal")
            self._feature_list_text.delete("1.0", tk.END)
            self._feature_list_text.insert("1.0", format_feature_names_text(info))
            self._feature_list_text.configure(state="disabled")

    def _fill_timestamps(self, ov: dict[str, Any]) -> None:
        self._ts_vars["first_timestamp"].set(_fmt_clock(ov.get("first_timestamp")))
        self._ts_vars["last_timestamp"].set(_fmt_clock(ov.get("last_timestamp")))
        self._ts_vars["sampling_interval_sec"].set(f"{_fmt_num(ov.get('sampling_interval_sec'), 1)} sec")
        self._ts_vars["observed_interval_sec"].set(f"{_fmt_num(ov.get('observed_interval_sec'), 2)} sec")
        self._ts_vars["duplicate_timestamps"].set(_fmt_int(ov.get("duplicate_timestamps")))
        self._ts_vars["out_of_order_timestamps"].set(_fmt_int(ov.get("out_of_order_timestamps")))
        self._ts_vars["missing_timestamps"].set(_fmt_int(ov.get("missing_timestamps")))
        self._ts_vars["actual_unique_timestamps"].set(_fmt_int(ov.get("actual_unique_timestamps")))
        self._ts_vars["expected_samples"].set(_fmt_int(ov.get("expected_samples")))
        self._ts_vars["largest_gap_sec"].set(_fmt_dur(ov.get("largest_gap_sec")))

    def _fill_registry(self, ov: dict[str, Any]) -> None:
        for key in (
            "registry_expected",
            "registry_found",
            "registry_missing",
            "registry_unexpected",
            "duplicate_columns",
        ):
            self._reg_vars[key].set(_fmt_int(ov.get(key)))
        comps = ov.get("health_components") or {}
        missing = comps.get("registry_missing_names") or []
        unexpected = comps.get("registry_unexpected_names") or []
        lines = []
        if missing:
            lines.append("Missing Features\n  " + ", ".join(missing[:40]))
        if unexpected:
            lines.append("Unexpected Columns\n  " + ", ".join(unexpected[:40]))
        self._reg_detail.set("\n\n".join(lines) if lines else "Registry matches the selected feature set.")

    def _render_columns(self) -> None:
        tree = self._cov_tree
        tree.delete(*tree.get_children())
        mode = str(self._filter.get() or "all")
        family = str(self._family.get() or "All")
        self._row_by_iid: dict[str, dict[str, Any]] = {}
        for row in self._columns_cache:
            status = str(row.get("status") or "")
            fam = str(row.get("feature_family") or row.get("column_type") or "")
            cov = float(row.get("coverage_pct") or 0.0)
            expected_empty = bool(row.get("expected_empty"))
            if family != "All" and fam != family:
                continue
            if mode == "problems" and status in (
                "Healthy",
                "Expected Empty",
                "Constant",
                "Warm-up",
            ):
                continue
            if mode == "unexpected_empty" and status != "Unexpected Empty":
                continue
            if mode == "expected_empty" and not (
                expected_empty or status in ("Expected Empty",)
            ):
                continue
            if mode == "coverage_lt_99" and (cov >= 99.0 or expected_empty):
                continue
            if mode == "warmup" and status != "Warm-up":
                continue
            if mode == "Healthy" and status != mode:
                continue
            tag = "ok"
            if status == "Unexpected Empty":
                tag = "unexpected"
            elif expected_empty or status == "Expected Empty":
                tag = "expected"
            elif status == "Healthy":
                tag = "ok"
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
                    row.get("reason") or row.get("notes") or "",
                ),
                tags=(tag,),
            )
            self._row_by_iid[iid] = row

    def _on_feature_select(self, _event: Any = None) -> None:
        sel = self._cov_tree.selection()
        if not sel:
            self._detail_var.set("Select a feature")
            return
        row = getattr(self, "_row_by_iid", {}).get(sel[0]) or {}
        expected = "Yes" if row.get("expected_empty") else "No"
        req = "Yes" if row.get("required_flag") else "No"
        self._detail_var.set(
            f"Feature\n{row.get('feature')}\n\n"
            f"Category\n{row.get('feature_family') or row.get('column_type')}\n\n"
            f"Source\n{row.get('source') or '—'}\n\n"
            f"Coverage\n{_fmt_pct(row.get('coverage_pct'))}\n\n"
            f"Status\n{row.get('status')}\n\n"
            f"Availability\n{row.get('availability')}\n\n"
            f"Reason\n{row.get('reason') or row.get('notes') or '—'}\n\n"
            f"Required\n{req}\n\n"
            f"Expected empty\n{expected}\n\n"
            f"First valid\n{_fmt_clock(row.get('first_valid_ts'))}"
        )

    def _render_gaps(self, gaps: list[dict[str, Any]]) -> None:
        ov = self._ov
        self._gap_tree.delete(*self._gap_tree.get_children())
        self._gap_summary.set(
            f"Policy {_fmt_num(ov.get('gap_policy_sec'), 0)} sec · "
            f"Expected {_fmt_int(ov.get('expected_samples'))} · "
            f"Actual {_fmt_int(ov.get('actual_unique_timestamps'))} · "
            f"Missing {_fmt_int(ov.get('missing_timestamps'))} · "
            f"Triggered {_fmt_int(ov.get('gap_triggered') or ov.get('gap_events'))} · "
            f"Ignored {_fmt_int(ov.get('gap_ignored'))} · "
            f"Filled {_fmt_int(ov.get('gap_filled'))} · "
            f"Largest {_fmt_dur(ov.get('largest_gap_sec'))} · "
            f"Rows affected {_fmt_int(ov.get('rows_affected_by_gaps'))} · "
            f"Coverage loss {_fmt_pct(ov.get('coverage_loss_pct'), 3)}"
        )
        for g in gaps:
            self._gap_tree.insert(
                "",
                "end",
                values=(
                    _fmt_clock(g.get("start_ts")),
                    _fmt_clock(g.get("end_ts")),
                    _fmt_dur(g.get("gap_seconds")),
                    _fmt_int(g.get("missing_samples")),
                    str(g.get("token") or "")[:12],
                    g.get("action") or "",
                ),
            )

    def _rebuild(self) -> None:
        day = str(self._day_var.get() or "").strip()
        if not day:
            return
        if not messagebox.askyesno(
            "Rebuild metadata",
            f"Rescan samples for {day} and rebuild day metadata?",
        ):
            return
        if callable(self._on_rebuild):
            try:
                self._on_rebuild(day)
                self.refresh()
            except Exception as exc:
                messagebox.showerror("Rebuild metadata", str(exc))


def open_day_metadata_window(
    master: tk.Misc,
    *,
    db_path: str,
    trading_days: list[str],
    on_rebuild: Callable[[str], None] | None = None,
) -> tk.Toplevel:
    win = tk.Toplevel(master)
    days = list(trading_days) or []
    title_day = days[0] if len(days) == 1 else f"{len(days)} days"
    win.title(f"Day Metadata — {title_day}")
    win.transient(master.winfo_toplevel())
    panel = DayMetadataPanel(
        win, db_path=db_path, trading_days=days, on_rebuild=on_rebuild
    )
    panel.pack(fill="both", expand=True)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win
