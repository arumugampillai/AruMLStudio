"""Auto Feature Transformation — Phase 1A workspace.

Feature sources + Analysis Dataset build + live monitor.
Manual Feature Transformations remain unchanged elsewhere.
"""

from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    FEATURE_SOURCE_PIPELINE,
    FEATURE_SOURCE_REGISTRY,
    feature_sources_catalog,
)
from chain_replay_ml.dataset_builder.master_naming import (
    master_db_filename,
    resolve_master_db_path,
)

from .build_progress_manager import get_build_progress_manager
from .build_config_prefs import (
    apply_auto_feature_transform_prefs,
    auto_feature_transform_prefs_snapshot,
    load_auto_feature_transform_prefs,
    save_auto_feature_transform_prefs,
)
from .build_service import AnalysisDatasetRunner, chart_data_dir


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    total = max(0, int(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_bytes(n: int | None) -> str:
    if n is None or n < 0:
        return "—"
    mb = n / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def _fmt_int(n: int | float | None) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


class AutoFeatureTransformPanel(ttk.Frame):
    """Phase 1A Auto Feature Transformation page."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_pipelines_changed: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self.chart_dir = chart_dir
        self._on_pipelines_changed = on_pipelines_changed
        self._runner = AnalysisDatasetRunner(chart_dir)
        self._progress_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._catalog: dict[str, Any] | None = None
        self._last_summary: dict[str, Any] | None = None
        self._building = False
        self._log_seen = 0

        self._market_var = tk.StringVar(value="NIFTY")
        self._interval_var = tk.StringVar(value="3")
        self._include_registry = tk.BooleanVar(value=True)
        self._include_pipeline = tk.BooleanVar(value=True)
        self._day_scope_var = tk.StringVar(value="all")
        self._selected_days: set[str] = set()
        self._selected_days_hint_var = tk.StringVar(value="All trading days in the master database")
        self._no_null_data = tk.BooleanVar(value=False)
        self._pipeline_no_null_report = tk.BooleanVar(value=False)
        self._prem_en_var = tk.BooleanVar(value=True)
        self._prem_min_var = tk.StringVar(value="15")
        self._prem_max_var = tk.StringVar(value="40")
        self._target_pipeline_var = tk.StringVar(value="")
        self._target_pipeline_mode = tk.StringVar(value="existing")
        self._new_pipeline_preview_var = tk.StringVar(value="")
        self._build_pipeline_var = tk.StringVar(value="")
        self._pipeline_label_to_id: dict[str, str] = {}
        self._build_pipeline_label_to_id: dict[str, str] = {}
        self._feature_project_display_var = tk.StringVar(value="—")

        from .auto_candidate_generation import (
            DEFAULT_HORIZONS_SEC,
            INTERACTION_OP_LABELS,
            TRANSFORM_LABELS,
            default_candidate_generation_prefs,
        )

        cg = default_candidate_generation_prefs()
        self._candidate_source_mode = tk.StringVar(value=str(cg["source"]))
        self._transform_vars = {
            key: tk.BooleanVar(value=bool(cg["transformations"].get(key)))
            for key, _ in TRANSFORM_LABELS
        }
        self._horizon_vars = {
            int(h): tk.BooleanVar(value=int(h) in cg["horizons_sec"])
            for h in DEFAULT_HORIZONS_SEC
        }
        self._interaction_op_vars = {
            key: tk.BooleanVar(value=bool(cg["interaction_ops"].get(key)))
            for key, _ in INTERACTION_OP_LABELS
        }
        self._gen_generated = tk.StringVar(value="0")
        self._gen_skipped = tk.StringVar(value="0")
        self._gen_errors = tk.StringVar(value="0")

        self._reg_status = tk.StringVar(value="—")
        self._pipe_status = tk.StringVar(value="—")
        self._analysis_status = tk.StringVar(value="Not Built")
        self._analysis_detail = tk.StringVar(value="Last Build: Never")

        self._mon_status = tk.StringVar(value="Idle")
        self._mon_elapsed = tk.StringVar(value="—")
        self._mon_eta = tk.StringVar(value="—")
        self._mon_percent = tk.StringVar(value="0%")
        self._mon_reg = tk.StringVar(value="—")
        self._mon_pipe = tk.StringVar(value="—")
        self._mon_overall = tk.StringVar(value="—")
        self._mon_current = tk.StringVar(value="—")
        self._mon_source = tk.StringVar(value="—")
        self._mon_speed = tk.StringVar(value="—")
        self._mon_rows = tk.StringVar(value="—")
        self._mon_day = tk.StringVar(value="—")
        self._mon_mode = tk.StringVar(value="—")
        self._mon_features = tk.StringVar(value="—")
        self._mon_token = tk.StringVar(value="—")
        self._mon_wave = tk.StringVar(value="—")
        self._stage_vars = {
            "registry": tk.StringVar(value="○ Registry Features"),
            "pipeline": tk.StringVar(value="○ Pipeline Features"),
            "no_null": tk.StringVar(value="○ No-Null Filter"),
            "premium": tk.StringVar(value="○ Premium Filter"),
            "finalize": tk.StringVar(value="○ Dataset Finalization"),
        }

        self._summary_var = tk.StringVar(value="Build an analysis dataset to see the completion summary.")

        self._prefs_loading = False
        self._build_ui()
        self._apply_saved_build_prefs()
        self._bind_build_pref_traces()
        self.refresh()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._runner.chart_dir = chart_dir
        self._apply_saved_build_prefs()
        self.refresh()

    def on_show(self) -> None:
        self.refresh()

    def _apply_saved_build_prefs(self) -> None:
        if not self.chart_dir:
            return
        applied = apply_auto_feature_transform_prefs(
            load_auto_feature_transform_prefs(self.chart_dir)
        )
        self._prefs_loading = True
        try:
            self._market_var.set(str(applied["market"]))
            self._interval_var.set(str(applied["interval_sec"]))
            self._include_registry.set(bool(applied["include_registry"]))
            self._include_pipeline.set(bool(applied["include_pipeline"]))
            self._day_scope_var.set(str(applied.get("day_scope") or "all"))
            self._selected_days = set(applied.get("selected_days") or [])
            self._no_null_data.set(bool(applied["no_null_data"]))
            self._pipeline_no_null_report.set(bool(applied["pipeline_no_null_report"]))
            self._prem_en_var.set(bool(applied["premium_enabled"]))
            self._prem_min_var.set(str(applied["premium_min"]))
            self._prem_max_var.set(str(applied["premium_max"]))
            saved_pid = str(applied.get("target_pipeline_id") or "").strip().upper()
            if saved_pid:
                self._target_pipeline_var.set(saved_pid)
            mode = str(applied.get("target_pipeline_mode") or "existing").strip().lower()
            self._target_pipeline_mode.set("create_new" if mode == "create_new" else "existing")
            saved_build_pid = str(applied.get("build_pipeline_id") or "").strip().upper()
            if saved_build_pid:
                self._build_pipeline_var.set(saved_build_pid)
            cg = applied.get("candidate_generation") or {}
            if isinstance(cg, dict):
                self._candidate_source_mode.set(str(cg.get("source") or "registry"))
                transforms = cg.get("transformations") if isinstance(cg.get("transformations"), dict) else {}
                for key, var in self._transform_vars.items():
                    if key in transforms:
                        var.set(bool(transforms[key]))
                horizons = cg.get("horizons_sec") or []
                if isinstance(horizons, list):
                    hset = {int(h) for h in horizons if int(h) > 0}
                    for h, var in self._horizon_vars.items():
                        var.set(h in hset)
                ops = cg.get("interaction_ops") if isinstance(cg.get("interaction_ops"), dict) else {}
                for key, var in self._interaction_op_vars.items():
                    if key in ops:
                        var.set(bool(ops[key]))
        finally:
            self._prefs_loading = False
        self._update_day_scope_state()

    def _persist_build_prefs(self, *_args: Any) -> None:
        if self._prefs_loading or not self.chart_dir:
            return
        try:
            interval = int(float(str(self._interval_var.get() or "3").strip() or "3"))
        except (TypeError, ValueError):
            interval = 3
        day_scope = str(self._day_scope_var.get() or "all")
        snapshot = auto_feature_transform_prefs_snapshot(
            market=str(self._market_var.get() or "NIFTY"),
            interval_sec=interval,
            include_registry=bool(self._include_registry.get()),
            include_pipeline=bool(self._include_pipeline.get()),
            all_days=(day_scope == "all"),
            day_scope=day_scope,
            selected_days=sorted(self._selected_days),
            no_null_data=bool(self._no_null_data.get()),
            pipeline_no_null_report=bool(self._pipeline_no_null_report.get()),
            premium_enabled=bool(self._prem_en_var.get()),
            premium_min=str(self._prem_min_var.get() or "15"),
            premium_max=str(self._prem_max_var.get() or "40"),
            target_pipeline_id=str(self._target_pipeline_var.get() or "").strip().upper(),
            target_pipeline_mode=str(self._target_pipeline_mode.get() or "existing"),
            build_pipeline_id=str(self._build_pipeline_var.get() or "").strip().upper(),
        )
        from .auto_candidate_generation import candidate_generation_prefs_snapshot

        snapshot["candidate_generation"] = candidate_generation_prefs_snapshot(
            source=str(self._candidate_source_mode.get() or "registry"),
            transformations={k: bool(v.get()) for k, v in self._transform_vars.items()},
            horizons_sec=[h for h, v in self._horizon_vars.items() if v.get()],
            interaction_ops={k: bool(v.get()) for k, v in self._interaction_op_vars.items()},
        )
        try:
            save_auto_feature_transform_prefs(self.chart_dir, snapshot)
        except Exception:
            pass

    def _bind_build_pref_traces(self) -> None:
        for var in (
            self._market_var,
            self._interval_var,
            self._include_registry,
            self._include_pipeline,
            self._day_scope_var,
            self._no_null_data,
            self._pipeline_no_null_report,
            self._prem_en_var,
            self._prem_min_var,
            self._prem_max_var,
            self._target_pipeline_var,
            self._target_pipeline_mode,
            self._build_pipeline_var,
            self._candidate_source_mode,
        ):
            try:
                var.trace_add("write", self._persist_build_prefs)
            except Exception:
                pass
        for var in (
            *self._transform_vars.values(),
            *self._horizon_vars.values(),
            *self._interaction_op_vars.values(),
        ):
            try:
                var.trace_add("write", self._persist_build_prefs)
            except Exception:
                pass

    def refresh(self) -> None:
        try:
            data_dir = chart_data_dir(self.chart_dir)
            project_id = self._resolve_master_feature_project_id()
            self._feature_project_display_var.set(project_id)
            self._catalog = feature_sources_catalog(
                data_dir=data_dir,
                feature_project_id=project_id,
            )
        except Exception as exc:
            self._catalog = None
            self._feature_project_display_var.set("—")
            messagebox.showerror("Feature Sources", str(exc), parent=self)
            return
        self._render_status_cards()
        self._render_source_trees()
        self.refresh_target_pipelines()
        self.refresh_build_pipelines()
        self._sync_target_pipeline_mode()

    def _resolve_master_feature_project_id(self) -> str:
        from chain_replay_ml.dataset_builder.master_feature_project import (
            MasterFeatureProjectError,
            resolve_master_feature_project_id_for_path,
        )

        path = self._master_db_path()
        if not os.path.isfile(path):
            raise MasterFeatureProjectError(
                "Master database not found for the current Market / Interval."
            )
        return resolve_master_feature_project_id_for_path(
            path,
            chart_data_dir(self.chart_dir),
        )

    def _refresh_new_pipeline_preview(self) -> None:
        if not self.chart_dir:
            self._new_pipeline_preview_var.set("")
            return
        try:
            from .pipeline_registry_service import peek_next_pipeline_identity

            info = peek_next_pipeline_identity(self.chart_dir)
            name = str(info.get("name") or "").strip()
            pid = str(info.get("pipeline_id") or "").strip().upper()
            if name and pid:
                self._new_pipeline_preview_var.set(f"{name} ({pid})")
            else:
                self._new_pipeline_preview_var.set(name or pid or "")
        except Exception:
            self._new_pipeline_preview_var.set("")

    def _sync_target_pipeline_mode(self) -> None:
        if not hasattr(self, "_target_pipeline_cb"):
            return
        mode = str(self._target_pipeline_mode.get() or "existing").strip().lower()
        if mode == "create_new":
            self._refresh_new_pipeline_preview()
            try:
                self._target_pipeline_cb.configure(state="disabled")
            except tk.TclError:
                pass
            if hasattr(self, "_new_pipeline_name_entry"):
                try:
                    self._new_pipeline_name_entry.configure(state="readonly")
                except tk.TclError:
                    pass
        else:
            try:
                self._target_pipeline_cb.configure(state="readonly")
            except tk.TclError:
                pass

    def _on_target_pipeline_mode_changed(self) -> None:
        self._sync_target_pipeline_mode()
        self._persist_build_prefs()

    def _experimental_pipeline_labels(
        self,
        *,
        select_pipeline_id: str | None = None,
        current_var: tk.StringVar,
    ) -> tuple[list[str], dict[str, str]]:
        from .pipeline_registry_service import get_experimental_pipelines

        rows = get_experimental_pipelines(self.chart_dir)
        labels: list[str] = []
        id_by_label: dict[str, str] = {}
        for row in rows:
            pid = str(row.get("pipeline_id") or "")
            name = str(row.get("name") or pid)
            label = f"{name} ({pid})"
            labels.append(label)
            id_by_label[label] = pid
        want = str(select_pipeline_id or current_var.get() or "").strip().upper()
        if want:
            for label, pid in id_by_label.items():
                if pid == want:
                    current_var.set(pid)
                    return labels, id_by_label
        if labels:
            current_var.set(id_by_label[labels[0]])
        return labels, id_by_label

    def refresh_target_pipelines(self, *, select_pipeline_id: str | None = None) -> None:
        if not hasattr(self, "_target_pipeline_cb"):
            return
        labels, id_by_label = self._experimental_pipeline_labels(
            select_pipeline_id=select_pipeline_id,
            current_var=self._target_pipeline_var,
        )
        self._pipeline_label_to_id = id_by_label
        self._target_pipeline_cb["values"] = labels
        want = str(select_pipeline_id or self._target_pipeline_var.get() or "").strip().upper()
        if want:
            for label, pid in id_by_label.items():
                if pid == want:
                    self._target_pipeline_cb.set(label)
                    self._target_pipeline_var.set(pid)
                    return
        if labels:
            self._target_pipeline_cb.set(labels[0])
            self._target_pipeline_var.set(id_by_label[labels[0]])
        self._refresh_new_pipeline_preview()

    def refresh_build_pipelines(self, *, select_pipeline_id: str | None = None) -> None:
        if not hasattr(self, "_build_pipeline_cb"):
            return
        labels, id_by_label = self._experimental_pipeline_labels(
            select_pipeline_id=select_pipeline_id,
            current_var=self._build_pipeline_var,
        )
        self._build_pipeline_label_to_id = id_by_label
        self._build_pipeline_cb["values"] = labels
        want = str(select_pipeline_id or self._build_pipeline_var.get() or "").strip().upper()
        if want:
            for label, pid in id_by_label.items():
                if pid == want:
                    self._build_pipeline_cb.set(label)
                    self._build_pipeline_var.set(pid)
                    break
        elif labels:
            self._build_pipeline_cb.set(labels[0])
            self._build_pipeline_var.set(id_by_label[labels[0]])
        self._sync_build_pipeline_selector_state()

    def poll_progress(self) -> None:
        try:
            while True:
                msg = self._progress_q.get_nowait()
                self._handle_progress(msg)
        except queue.Empty:
            pass

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        title = ttk.Frame(self)
        title.pack(fill="x", pady=(0, 6))
        ttk.Label(
            title,
            text="Feature Auto Transformation",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        ttk.Label(
            title,
            text="Phase 1A — Feature Sources → Analysis Dataset",
            foreground="#666",
        ).pack(side="left", padx=(12, 0))

        self._build_status_cards()

        main_nb = ttk.Notebook(self)
        main_nb.pack(fill="both", expand=True, pady=(8, 0))

        monitor_tab = ttk.Frame(main_nb, padding=4)
        candidate_tab = ttk.Frame(main_nb, padding=4)
        main_nb.add(monitor_tab, text="Live Build Monitor")
        main_nb.add(candidate_tab, text="Auto Candidate Generation")

        monitor_body = ttk.Panedwindow(monitor_tab, orient=tk.HORIZONTAL)
        monitor_body.pack(fill="both", expand=True)
        left = ttk.Frame(monitor_body, padding=(0, 0, 6, 0))
        right = ttk.Frame(monitor_body, padding=(6, 0, 0, 0))
        monitor_body.add(left, weight=1)
        monitor_body.add(right, weight=1)

        left_nb = ttk.Notebook(left)
        left_nb.pack(fill="both", expand=True)
        build_tab = ttk.Frame(left_nb, padding=4)
        sources_tab = ttk.Frame(left_nb, padding=4)
        build_tab.columnconfigure(0, weight=1)
        build_tab.rowconfigure(0, weight=1)
        sources_tab.columnconfigure(0, weight=1)
        sources_tab.rowconfigure(0, weight=1)
        left_nb.add(build_tab, text="Build Configuration")
        left_nb.add(sources_tab, text="Feature Sources")
        self._build_build_panel(build_tab)
        self._build_sources_panel(sources_tab)
        self._build_monitor_panel(right)
        self._build_summary_panel(right)

        self._build_candidate_generation_panel(candidate_tab)

    def _build_status_cards(self) -> None:
        cards = ttk.Frame(self)
        cards.pack(fill="x", pady=(0, 4))
        self._pipe_card: ttk.LabelFrame | None = None
        self._reg_card: ttk.LabelFrame | None = None
        for col, spec in enumerate((
            ("Registry Features", self._reg_status, None, True, "Click to Select Features", self._open_registry_selection),
            ("Pipeline Features", self._pipe_status, None, True, "Click to view / delete features", self._open_pipeline_manager),
            ("Analysis Dataset", self._analysis_status, self._analysis_detail, False, None, None),
        )):
            title, main_var, sub_var, clickable, hint_text, click_cmd = spec
            cell = ttk.LabelFrame(cards, text=title, padding=8)
            cell.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            cards.grid_columnconfigure(col, weight=1)
            main = ttk.Label(cell, textvariable=main_var, font=("Segoe UI", 10, "bold"))
            main.pack(anchor="w")
            if sub_var is not None:
                ttk.Label(cell, textvariable=sub_var, foreground="#666").pack(anchor="w", pady=(2, 0))
            if clickable and hint_text and click_cmd:
                if title.startswith("Registry"):
                    self._reg_card = cell
                elif title.startswith("Pipeline"):
                    self._pipe_card = cell
                hint = ttk.Label(
                    cell,
                    text=hint_text,
                    foreground="#1565c0",
                    cursor="hand2",
                )
                hint.pack(anchor="w", pady=(4, 0))
                for widget in (cell, main, hint):
                    widget.bind("<Button-1>", lambda _e, cmd=click_cmd: cmd())
                    try:
                        widget.configure(cursor="hand2")
                    except tk.TclError:
                        pass
    def _build_sources_panel(self, parent: tk.Misc) -> None:
        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(0, 4))
        ttk.Label(tools, text="Feature Project:").pack(side="left")
        ttk.Label(
            tools,
            textvariable=self._feature_project_display_var,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            tools,
            text="(from Master Dataset)",
            foreground="#666",
        ).pack(side="left", padx=(8, 0))

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        reg_tab = ttk.Frame(nb, padding=4)
        pipe_tab = ttk.Frame(nb, padding=4)
        nb.add(reg_tab, text="Registry Features")
        nb.add(pipe_tab, text="Pipeline Features")

        self._reg_tree = self._make_source_tree(reg_tab)
        pipe_tools = ttk.Frame(pipe_tab)
        pipe_tools.pack(fill="x", pady=(0, 4))
        ttk.Button(
            pipe_tools,
            text="Manage / Delete…",
            command=self._open_pipeline_manager,
        ).pack(side="right")
        self._pipe_tree = self._make_source_tree(pipe_tab)

    def _make_source_tree(self, parent: tk.Misc) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=("count",), show="tree headings", height=14)
        tree.heading("#0", text="Group / Feature")
        tree.heading("count", text="#")
        tree.column("#0", width=280, stretch=True)
        tree.column("count", width=48, anchor="e")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    def _build_build_panel(self, parent: tk.Misc) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        scroll_wrap = ttk.Frame(parent)
        scroll_wrap.grid(row=0, column=0, sticky="nsew")
        scroll_wrap.columnconfigure(0, weight=1)
        scroll_wrap.rowconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_wrap, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(0, 0, 4, 0))
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        def _sync_scroll(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event: tk.Event) -> None:
            canvas.itemconfigure(inner_id, width=max(1, int(event.width)))

        inner.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _sync_width)

        def _on_wheel(event: tk.Event) -> str | None:
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            return "break"

        for w in (canvas, inner):
            w.bind("<Enter>", lambda _e, widget=w: widget.bind_all("<MouseWheel>", _on_wheel), add="+")
            w.bind("<Leave>", lambda _e, widget=w: widget.unbind_all("<MouseWheel>"), add="+")
            w.bind("<Button-4>", _on_wheel, add="+")
            w.bind("<Button-5>", _on_wheel, add="+")

        row = ttk.Frame(inner)
        row.pack(fill="x")
        ttk.Label(row, text="Market").pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self._market_var,
            values=("NIFTY", "BANKNIFTY", "SENSEX"),
            width=12,
            state="readonly",
        ).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Interval (s)").pack(side="left")
        ttk.Entry(row, textvariable=self._interval_var, width=6).pack(side="left", padx=4)

        src = ttk.Frame(inner)
        src.pack(fill="x", pady=(8, 0))
        ttk.Label(src, text="Feature Sources", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Checkbutton(
            src,
            text="Registry Features",
            variable=self._include_registry,
        ).pack(anchor="w", pady=1)
        ttk.Checkbutton(
            src,
            text="Pipeline Features",
            variable=self._include_pipeline,
            command=self._sync_build_pipeline_selector_state,
        ).pack(anchor="w", pady=1)
        pipe_sel = ttk.Frame(src)
        pipe_sel.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(pipe_sel, text="Pipeline:").pack(side="left")
        self._build_pipeline_cb = ttk.Combobox(
            pipe_sel,
            width=32,
            state="readonly",
        )
        self._build_pipeline_cb.pack(side="left", padx=(8, 0))
        self._build_pipeline_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_build_pipeline_selected(),
        )
        day_scope_box = ttk.Frame(src)
        day_scope_box.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(day_scope_box, text="Trading days", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        day_scope_btns = ttk.Frame(day_scope_box)
        day_scope_btns.pack(anchor="w", pady=(2, 0))
        ttk.Radiobutton(
            day_scope_btns,
            text="All days",
            value="all",
            variable=self._day_scope_var,
            command=self._on_day_scope_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            day_scope_btns,
            text="Selected days",
            value="selected",
            variable=self._day_scope_var,
            command=self._on_day_scope_changed,
        ).pack(side="left", padx=(10, 0))
        self._btn_choose_days = ttk.Button(
            day_scope_btns,
            text="Choose days…",
            command=self._open_day_selection,
        )
        self._btn_choose_days.pack(side="left", padx=(10, 0))
        ttk.Label(
            day_scope_box,
            textvariable=self._selected_days_hint_var,
            foreground="#666",
            wraplength=360,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Checkbutton(
            src,
            text="No null data",
            variable=self._no_null_data,
        ).pack(anchor="w", pady=1)
        ttk.Label(
            src,
            text="Drop 100% NULL columns, then keep complete rows. "
                 "Runs only after Feature Transformations finish.",
            foreground="#666",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(
            src,
            text="Run No-Null Analysis",
            variable=self._pipeline_no_null_report,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            src,
            text=(
                "Diagnostics only → Activity Log: top NULL pipeline features, "
                "exclusive impact, dependency chains."
            ),
            foreground="#666",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 2))

        prem_row = ttk.Frame(src)
        prem_row.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Checkbutton(
            prem_row,
            text="LTP",
            variable=self._prem_en_var,
        ).pack(side="left")
        ttk.Entry(prem_row, textvariable=self._prem_min_var, width=5).pack(side="left", padx=(8, 0))
        ttk.Label(prem_row, text="–").pack(side="left", padx=2)
        ttk.Entry(prem_row, textvariable=self._prem_max_var, width=5).pack(side="left")
        ttk.Label(
            src,
            text="Same as Master Dataset premium filter. Applied after No null data.",
            foreground="#666",
            wraplength=360,
        ).pack(anchor="w", pady=(0, 2))

        actions = ttk.Frame(inner)
        actions.pack(fill="x", pady=(12, 4))
        self._btn_build = ttk.Button(
            actions,
            text="Create Analysis Dataset",
            command=self._start_build,
        )
        self._btn_build.pack(side="left")
        self._btn_cancel = ttk.Button(
            actions,
            text="Cancel",
            command=self._cancel_build,
            state="disabled",
        )
        self._btn_cancel.pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Refresh Catalogues", command=self.refresh).pack(side="right")

    def _on_target_pipeline_selected(self) -> None:
        label = str(self._target_pipeline_cb.get() or "")
        pid = self._pipeline_label_to_id.get(label)
        if pid:
            self._target_pipeline_var.set(pid)
            self._persist_build_prefs()

    def _on_build_pipeline_selected(self) -> None:
        label = str(self._build_pipeline_cb.get() or "")
        pid = self._build_pipeline_label_to_id.get(label)
        if pid:
            self._build_pipeline_var.set(pid)
            self._persist_build_prefs()

    def _sync_build_pipeline_selector_state(self) -> None:
        if not hasattr(self, "_build_pipeline_cb"):
            return
        enabled = bool(self._include_pipeline.get())
        try:
            self._build_pipeline_cb.configure(state="readonly" if enabled else "disabled")
        except tk.TclError:
            pass

    def _candidate_prefs_snapshot(self) -> dict[str, Any]:
        from .auto_candidate_generation import candidate_generation_prefs_snapshot

        return candidate_generation_prefs_snapshot(
            source=str(self._candidate_source_mode.get() or "registry"),
            transformations={k: bool(v.get()) for k, v in self._transform_vars.items()},
            horizons_sec=[h for h, v in self._horizon_vars.items() if v.get()],
            interaction_ops={k: bool(v.get()) for k, v in self._interaction_op_vars.items()},
        )

    def _build_candidate_generation_panel(self, parent: tk.Misc) -> None:
        from .auto_candidate_generation import (
            DEFAULT_HORIZONS_SEC,
            INTERACTION_OP_LABELS,
            SOURCE_OPTIONS,
            TRANSFORM_LABELS,
        )

        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scroll.config(command=canvas.yview)
        canvas.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill="both", expand=True)
        inner = ttk.Frame(canvas, padding=(0, 0, 8, 0))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        ttk.Label(inner, text="AUTO CANDIDATE GENERATION", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 8)
        )

        tp_box = ttk.LabelFrame(inner, text="Target Pipeline", padding=6)
        tp_box.pack(fill="x", pady=(0, 10))

        existing_row = ttk.Frame(tp_box)
        existing_row.pack(fill="x")
        ttk.Radiobutton(
            existing_row,
            text="Existing Pipeline",
            variable=self._target_pipeline_mode,
            value="existing",
            command=self._on_target_pipeline_mode_changed,
        ).pack(anchor="w")
        self._target_pipeline_cb = ttk.Combobox(
            existing_row,
            width=32,
            state="readonly",
        )
        self._target_pipeline_cb.pack(anchor="w", padx=(20, 0), pady=(4, 0))
        self._target_pipeline_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_target_pipeline_selected(),
        )

        create_row = ttk.Frame(tp_box)
        create_row.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(
            create_row,
            text="Create New Auto Pipeline",
            variable=self._target_pipeline_mode,
            value="create_new",
            command=self._on_target_pipeline_mode_changed,
        ).pack(anchor="w")
        name_row = ttk.Frame(create_row)
        name_row.pack(anchor="w", padx=(20, 0), pady=(4, 0))
        ttk.Label(name_row, text="Name:").pack(side="left")
        self._new_pipeline_name_entry = ttk.Entry(
            name_row,
            textvariable=self._new_pipeline_preview_var,
            width=30,
            state="readonly",
        )
        self._new_pipeline_name_entry.pack(side="left", padx=(8, 0))

        src_box = ttk.LabelFrame(inner, text="Source Features", padding=6)
        src_box.pack(fill="x", pady=(10, 0))
        for mode, label in SOURCE_OPTIONS:
            ttk.Radiobutton(
                src_box,
                text=label,
                variable=self._candidate_source_mode,
                value=mode,
            ).pack(anchor="w")

        tx_box = ttk.LabelFrame(inner, text="Transformations", padding=6)
        tx_box.pack(fill="x", pady=(10, 0))
        tx_grid = ttk.Frame(tx_box)
        tx_grid.pack(fill="x")
        for idx, (key, label) in enumerate(TRANSFORM_LABELS):
            ttk.Checkbutton(
                tx_grid,
                text=label,
                variable=self._transform_vars[key],
            ).grid(row=idx // 3, column=idx % 3, sticky="w", padx=(0, 12), pady=2)

        hz_box = ttk.LabelFrame(inner, text="Horizons", padding=6)
        hz_box.pack(fill="x", pady=(10, 0))
        hz_grid = ttk.Frame(hz_box)
        hz_grid.pack(fill="x")
        for idx, horizon in enumerate(DEFAULT_HORIZONS_SEC):
            ttk.Checkbutton(
                hz_grid,
                text=f"{horizon}s",
                variable=self._horizon_vars[int(horizon)],
            ).grid(row=0, column=idx, sticky="w", padx=(0, 10))

        ix_box = ttk.LabelFrame(inner, text="Interaction Operations", padding=6)
        ix_box.pack(fill="x", pady=(10, 0))
        ix_grid = ttk.Frame(ix_box)
        ix_grid.pack(fill="x")
        for idx, (key, label) in enumerate(INTERACTION_OP_LABELS):
            ttk.Checkbutton(
                ix_grid,
                text=label,
                variable=self._interaction_op_vars[key],
            ).grid(row=idx // 3, column=idx % 3, sticky="w", padx=(0, 12), pady=2)

        ttk.Button(
            inner,
            text="Generate Candidates",
            command=self._generate_candidates,
        ).pack(anchor="w", pady=(14, 0))

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill="x", pady=(14, 10))

        prog_box = ttk.LabelFrame(inner, text="Generation Progress", padding=6)
        prog_box.pack(fill="x")
        for label, var in (
            ("Generated", self._gen_generated),
            ("Skipped", self._gen_skipped),
            ("Errors", self._gen_errors),
        ):
            row = ttk.Frame(prog_box)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label}:").pack(side="left")
            ttk.Label(row, textvariable=var).pack(side="left", padx=(6, 0))

    def _generate_candidates(self) -> None:
        from .pipeline_registry_service import (
            add_pipeline_candidates,
            create_pipeline,
            get_pipeline,
            peek_next_pipeline_identity,
        )

        created_pipeline: dict[str, Any] | None = None
        mode = str(self._target_pipeline_mode.get() or "existing").strip().lower()
        if mode == "create_new":
            preview = peek_next_pipeline_identity(self.chart_dir)
            try:
                created_pipeline = create_pipeline(
                    self.chart_dir,
                    name=str(preview.get("name") or "").strip() or None,
                    pipeline_type="auto",
                )
            except Exception as exc:
                messagebox.showerror("Generate Candidates", str(exc), parent=self)
                return
            pid = str(created_pipeline.get("pipeline_id") or "").strip().upper()
            if not pid:
                messagebox.showerror(
                    "Generate Candidates",
                    "Failed to create a new Auto pipeline.",
                    parent=self,
                )
                return
            self._target_pipeline_var.set(pid)
            self._target_pipeline_mode.set("existing")
            self.refresh_target_pipelines(select_pipeline_id=pid)
            self._sync_target_pipeline_mode()
            if callable(self._on_pipelines_changed):
                try:
                    self._on_pipelines_changed(select_pipeline_id=pid)
                except TypeError:
                    try:
                        self._on_pipelines_changed()
                    except Exception:
                        pass
                except Exception:
                    pass
            self._append_log(
                f"Created Auto pipeline: {created_pipeline.get('name') or pid} ({pid})"
            )
        else:
            pid = str(self._target_pipeline_var.get() or "").strip().upper()
            if not pid:
                messagebox.showwarning(
                    "Generate Candidates",
                    "Select a target pipeline first.",
                    parent=self,
                )
                return

        row = get_pipeline(self.chart_dir, pid)
        if not row:
            messagebox.showerror("Generate Candidates", f"Pipeline {pid} not found.", parent=self)
            return
        if str(row.get("type") or "") == "base" or row.get("is_base"):
            messagebox.showinfo(
                "Generate Candidates",
                "The Base pipeline is the approved feature pool and cannot receive generated candidates.",
                parent=self,
            )
            return

        from .auto_candidate_generation import generate_pipeline_candidate_names

        try:
            interval = max(1, int(str(self._interval_var.get() or "3").strip()))
        except (TypeError, ValueError):
            interval = 3

        prefs = self._candidate_prefs_snapshot()
        report = generate_pipeline_candidate_names(
            chart_dir=self.chart_dir,
            pipeline_id=pid,
            interval_sec=interval,
            candidate_prefs=prefs,
        )
        new_names = list(report.new_names)
        skipped = list(report.duplicate_names)

        self._gen_generated.set(str(report.candidates_generated))
        self._gen_skipped.set(str(report.candidates_rejected_duplicates))
        self._gen_errors.set(str(report.candidates_rejected_policy + len(report.errors)))

        self._append_log("Auto Candidate Generation")
        self._append_log(f"  Target pipeline: {report.target_pipeline_id}")
        self._append_log(f"  Source feature count: {report.source_feature_count}")
        self._append_log(f"  Selected transformations: {report.selected_transformations}")
        self._append_log(f"  Selected horizons: {report.selected_horizons}")
        self._append_log(f"  Selected interaction operations: {report.selected_interaction_ops}")
        self._append_log(f"  Candidate combinations estimated: {report.combinations_estimated}")
        self._append_log(f"  Candidates generated: {report.candidates_generated}")
        self._append_log(f"  Candidates rejected by policy: {report.candidates_rejected_policy}")
        self._append_log(f"  Candidates rejected as duplicates: {report.candidates_rejected_duplicates}")
        self._append_log(f"  Candidates finally added: {report.candidates_added}")

        if report.errors:
            messagebox.showerror(
                "Generate Candidates",
                "\n".join(report.errors),
                parent=self,
            )
            if not new_names:
                return

        if not new_names:
            messagebox.showwarning(
                "Generate Candidates",
                "No new candidate features to add (all outputs already exist or none were produced).",
                parent=self,
            )
            return

        try:
            updated = add_pipeline_candidates(self.chart_dir, pid, new_names, replace=False)
            from .auto_candidate_generation import (
                build_auto_candidate_transformation_config,
                resolve_source_features,
            )
            from .pipeline_registry_service import update_pipeline_transformation_config

            source_feats = resolve_source_features(self.chart_dir, pid, prefs.get("source", "registry"))
            from .build_service import chart_data_dir

            data_dir = chart_data_dir(self.chart_dir)
            tx_config = build_auto_candidate_transformation_config(
                features=source_feats,
                interval_sec=interval,
                candidate_prefs=prefs,
                data_dir=data_dir,
            )
            update_pipeline_transformation_config(self.chart_dir, pid, tx_config)
        except Exception as exc:
            messagebox.showerror("Generate Candidates", str(exc), parent=self)
            return
        n_new = len(new_names)
        total = int((updated or {}).get("candidate_count") or 0)
        pipeline_label = str((updated or {}).get("name") or pid)
        created_note = ""
        if created_pipeline:
            created_note = (
                f"Created Auto pipeline {created_pipeline.get('name') or pid} ({pid}).\n"
            )
        messagebox.showinfo(
            "Generate Candidates",
            f"{created_note}"
            f"Added {n_new} candidate feature name(s) to {pipeline_label}.\n"
            f"Skipped {len(skipped)} existing name(s).\n"
            f"Pipeline now has {total} candidate feature(s).",
            parent=self,
        )
        self.refresh_target_pipelines(select_pipeline_id=pid)
        if callable(self._on_pipelines_changed):
            try:
                self._on_pipelines_changed(select_pipeline_id=pid)
            except TypeError:
                try:
                    self._on_pipelines_changed()
                except Exception:
                    pass
            except Exception:
                pass

    def _build_monitor_panel(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Live Build Monitor", padding=6)
        box.pack(fill="both", expand=True)

        top = ttk.Frame(box)
        top.pack(fill="x")
        for label, var in (
            ("Status", self._mon_status),
            ("Elapsed", self._mon_elapsed),
            ("ETA", self._mon_eta),
            ("Progress", self._mon_percent),
        ):
            cell = ttk.Frame(top)
            cell.pack(side="left", padx=(0, 14))
            ttk.Label(cell, text=label, foreground="#888").pack(anchor="w")
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self._progress = ttk.Progressbar(box, mode="determinate", maximum=100)
        self._progress.pack(fill="x", pady=(8, 6))

        stages = ttk.Frame(box)
        stages.pack(fill="x", pady=(0, 6))
        ttk.Label(stages, text="Stages", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for key in ("registry", "pipeline", "no_null", "premium", "finalize"):
            ttk.Label(stages, textvariable=self._stage_vars[key]).pack(anchor="w")

        src = ttk.LabelFrame(box, text="Feature Sources", padding=4)
        src.pack(fill="x", pady=(0, 6))
        ttk.Label(src, textvariable=self._mon_reg).pack(anchor="w")
        ttk.Label(src, textvariable=self._mon_pipe).pack(anchor="w")
        ttk.Separator(src, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(src, textvariable=self._mon_overall, font=("Segoe UI", 9, "bold")).pack(anchor="w")

        cur = ttk.LabelFrame(box, text="Current Day", padding=4)
        cur.pack(fill="x", pady=(0, 6))
        day_grid = ttk.Frame(cur)
        day_grid.pack(fill="x")
        for r, (label, var) in enumerate(
            (
                ("Overall", self._mon_day),
                ("Current", self._mon_current),
                ("Mode", self._mon_mode),
                ("Rows", self._mon_rows),
                ("Features", self._mon_features),
                ("Token", self._mon_token),
                ("Wave", self._mon_wave),
            )
        ):
            ttk.Label(day_grid, text=label, foreground="#888", width=10, anchor="w").grid(
                row=r, column=0, sticky="w", pady=1
            )
            ttk.Label(day_grid, textvariable=var, font=("Segoe UI", 9, "bold")).grid(
                row=r, column=1, sticky="w", pady=1
            )
        ttk.Label(cur, textvariable=self._mon_source, foreground="#666").pack(anchor="w", pady=(4, 0))
        metrics = ttk.Frame(cur)
        metrics.pack(fill="x", pady=(2, 0))
        ttk.Label(metrics, textvariable=self._mon_speed).pack(side="left", padx=(0, 12))

        ttk.Label(box, text="Activity Log", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._log = scrolledtext.ScrolledText(box, height=10, wrap="word", state="disabled")
        self._log.pack(fill="both", expand=True, pady=(2, 0))

    def _build_summary_panel(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Completion Summary", padding=6)
        box.pack(fill="x", pady=(8, 0))
        ttk.Label(box, textvariable=self._summary_var, justify="left", wraplength=420).pack(anchor="w")

    # -------------------------------------------------------------- render
    def _render_status_cards(self) -> None:
        cat = self._catalog or {}
        sources = {s["id"]: s for s in (cat.get("sources") or [])}
        reg = sources.get(FEATURE_SOURCE_REGISTRY) or {}
        pipe = sources.get(FEATURE_SOURCE_PIPELINE) or {}
        reg_n = int(reg.get("total") or 0)
        pipe_n = int(pipe.get("total") or 0)
        try:
            from chain_replay_ml.dataset_builder.registry_features_prefs import (
                registry_export_selection_summary,
            )

            reg_sel = registry_export_selection_summary(chart_data_dir(self.chart_dir))
            sel_n = int(reg_sel.get("selected") or reg_n)
            retired_reg = int(reg.get("retired_count") or 0)
            reg_line = f"{sel_n} / {reg_n} Features" if sel_n != reg_n else f"{reg_n} Features"
            if retired_reg:
                reg_line = f"{reg_line} ({retired_reg} retired)"
        except Exception:
            reg_line = f"{reg_n} Features"
        self._reg_status.set(f"{reg_line}\n✓ Ready" if reg.get("ready") else reg_line)
        retired_n = int(pipe.get("retired_count") or 0)
        if pipe.get("ready"):
            pipe_txt = f"{pipe_n} Features\n✓ Ready"
            if retired_n:
                pipe_txt += f"\n({retired_n} deleted)"
            self._pipe_status.set(pipe_txt)
        else:
            self._pipe_status.set(f"{pipe_n} Features")
        if self._last_summary and self._last_summary.get("status") == "completed":
            s = self._last_summary
            self._analysis_status.set(
                f"{_fmt_int(s.get('feature_count'))} Features\n"
                f"Rows {_fmt_int(s.get('row_count'))}\nStatus Ready"
            )
            self._analysis_detail.set(f"Built {s.get('built_at_display') or '—'}")
        else:
            self._analysis_status.set("Not Built")
            self._analysis_detail.set("Last Build: Never")

    def _render_source_trees(self) -> None:
        cat = self._catalog or {}
        sources = {s["id"]: s for s in (cat.get("sources") or [])}
        self._fill_tree(self._reg_tree, sources.get(FEATURE_SOURCE_REGISTRY))
        self._fill_tree(self._pipe_tree, sources.get(FEATURE_SOURCE_PIPELINE))

    def _fill_tree(self, tree: ttk.Treeview, source: dict[str, Any] | None) -> None:
        tree.delete(*tree.get_children())
        if not source:
            return
        for group in source.get("groups") or []:
            gid = tree.insert(
                "",
                "end",
                text=f"{group.get('label') or group.get('id')}",
                values=(group.get("count") or 0,),
                open=False,
            )
            for feat in group.get("features") or []:
                tree.insert(gid, "end", text=str(feat), values=("",))

    # --------------------------------------------------------------- build
    def _open_pipeline_manager(self) -> None:
        from .pipeline_features_manager import open_pipeline_features_manager

        open_pipeline_features_manager(
            self,
            data_dir=chart_data_dir(self.chart_dir),
            on_changed=self.refresh,
        )

    def _open_registry_selection(self) -> None:
        from .registry_features_manager import open_registry_features_selection

        open_registry_features_selection(
            self,
            data_dir=chart_data_dir(self.chart_dir),
            feature_project_id=self._resolve_master_feature_project_id(),
            on_changed=self.refresh,
        )

    def _master_db_path(self) -> str:
        data_dir = chart_data_dir(self.chart_dir)
        market = str(self._market_var.get() or "NIFTY").upper()
        try:
            interval = int(float(self._interval_var.get() or 3))
        except (TypeError, ValueError):
            interval = 3
        return resolve_master_db_path(
            data_dir,
            market=market,
            sampling_interval_sec=interval,
        )

    def _resolve_master_trading_days(self, path: str) -> list[str]:
        """Sorted trading-day keys for the current master DB (drives Choose days…)."""
        try:
            from chain_replay_ml.dataset_builder.trading_day_filter import master_trading_days

            return master_trading_days(path)
        except Exception:
            return []

    def _update_day_scope_state(self) -> None:
        scope = str(self._day_scope_var.get() or "all")
        if scope == "selected":
            self._btn_choose_days.configure(state="normal")
            n = len(self._selected_days)
            self._selected_days_hint_var.set(
                f"{n} trading day(s) selected" if n else "No days selected — click Choose days…"
            )
        else:
            self._btn_choose_days.configure(state="disabled")
            self._selected_days_hint_var.set("All trading days in the master database")

    def _on_day_scope_changed(self) -> None:
        self._update_day_scope_state()
        self._persist_build_prefs()

    def _open_day_selection(self) -> None:
        path = self._master_db_path()
        days = self._resolve_master_trading_days(path)
        if not days:
            messagebox.showwarning(
                "Selected days",
                "No trading days found in the master database for the "
                "current Market / Interval.",
                parent=self,
            )
            return
        from .day_selection_dialog import select_trading_days

        result = select_trading_days(
            self,
            days=days,
            initial=self._selected_days,
            title="Select Trading Days",
        )
        if result is None:
            return
        self._selected_days = set(result)
        self._day_scope_var.set("selected")
        self._update_day_scope_state()
        self._persist_build_prefs()

    def _start_build(self) -> None:
        if self._building or self._runner.running:
            messagebox.showinfo("Build", "An analysis dataset build is already running.", parent=self)
            return
        if not self._include_registry.get() and not self._include_pipeline.get():
            messagebox.showwarning("Build", "Select at least one feature source.", parent=self)
            return
        if self._include_pipeline.get():
            build_pid = str(self._build_pipeline_var.get() or "").strip().upper()
            if not build_pid:
                messagebox.showwarning(
                    "Build",
                    "Pipeline Features is enabled, but no experimental pipeline is selected.",
                    parent=self,
                )
                return
            from .pipeline_registry_service import get_pipeline, is_base_pipeline

            if is_base_pipeline(self.chart_dir, build_pid):
                messagebox.showwarning(
                    "Build",
                    "The Base pipeline cannot be used for experimental dataset creation.\n"
                    "Select an experimental pipeline (Auto/Manual).",
                    parent=self,
                )
                return
            pipe_row = get_pipeline(self.chart_dir, build_pid)
            if not pipe_row:
                messagebox.showerror("Build", f"Pipeline {build_pid} not found.", parent=self)
                return
            if int(pipe_row.get("candidate_count") or 0) <= 0:
                messagebox.showwarning(
                    "Build",
                    f"Pipeline {pipe_row.get('name') or build_pid} has no candidate features.",
                    parent=self,
                )
                return
        data_dir = chart_data_dir(self.chart_dir)
        if self._include_registry.get():
            try:
                from chain_replay_ml.dataset_builder.registry_features_prefs import (
                    resolve_registry_export_features,
                )

                if not resolve_registry_export_features(data_dir):
                    messagebox.showwarning(
                        "Build",
                        "No Registry Features are selected for export.\n\n"
                        "Use “Click to Select Features” on the Registry Features card.",
                        parent=self,
                    )
                    return
            except Exception:
                pass
        if not self._include_pipeline.get():
            pipe_n = 0
            try:
                totals = (self._catalog or {}).get("totals") or {}
                pipe_n = int(totals.get(FEATURE_SOURCE_PIPELINE) or 0)
            except Exception:
                pipe_n = 0
            if pipe_n <= 0:
                try:
                    from chain_replay_ml.dataset_builder.feature_sources_catalog import (
                        pipeline_feature_names,
                    )

                    pipe_n = len(pipeline_feature_names(data_dir=chart_data_dir(self.chart_dir)))
                except Exception:
                    pipe_n = 0
            if pipe_n > 0:
                if not messagebox.askyesno(
                    "Pipeline Features off",
                    f"Pipeline Features is unchecked, but the catalogue has "
                    f"{pipe_n} pipeline features ready.\n\n"
                    "This build will include Registry only "
                    "(transformation pipeline Enabled: 0).\n\n"
                    "Continue without Pipeline Features?",
                    parent=self,
                ):
                    return
        path = self._master_db_path()
        if not os.path.isfile(path):
            messagebox.showerror(
                "Build",
                f"Master database not found:\n{os.path.basename(path)}\n\n"
                f"Expected: {master_db_filename(market=self._market_var.get(), sampling_interval_sec=int(float(self._interval_var.get() or 3)))}",
                parent=self,
            )
            return

        day_scope = str(self._day_scope_var.get() or "all")
        if day_scope == "selected" and not self._selected_days:
            messagebox.showwarning(
                "Build",
                "“Selected days” is chosen, but no trading days are selected.\n\n"
                "Click “Choose days…” to pick at least one trading day, "
                "or switch to “All days”.",
                parent=self,
            )
            return

        from chain_replay_ml.dataset_builder.trading_day_filter import resolve_day_scope_filter

        master_days = self._resolve_master_trading_days(path)
        all_days_flag, explicit_days, day_filter_meta = resolve_day_scope_filter(
            scope=day_scope,
            selected_days=self._selected_days,
            master_days=master_days,
        )
        if day_scope == "selected" and not explicit_days:
            messagebox.showwarning(
                "Build",
                "None of the selected trading days were found in the master "
                "database for the current Market / Interval.\n\n"
                "Re-open “Choose days…” to pick valid trading days.",
                parent=self,
            )
            return

        try:
            interval = int(float(self._interval_var.get() or 3))
        except (TypeError, ValueError):
            messagebox.showerror("Build", "Interval must be a number.", parent=self)
            return

        prem_en = bool(self._prem_en_var.get())
        prem_lo: float | None = None
        prem_hi: float | None = None
        if prem_en:
            try:
                prem_lo = float(str(self._prem_min_var.get()).strip())
                prem_hi = float(str(self._prem_max_var.get()).strip())
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Build",
                    "LTP premium min/max must be numbers (same as Master Dataset).",
                    parent=self,
                )
                return
            if prem_lo > prem_hi:
                prem_lo, prem_hi = prem_hi, prem_lo

        self._building = True
        self._log_seen = 0
        self._clear_log()
        self._btn_build.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._mon_status.set("Running")
        self._progress["value"] = 0
        self._summary_var.set("Build in progress…")

        mgr = get_build_progress_manager()
        mgr.begin_job("analysis_dataset_build", title="Analysis Dataset", cancel_fn=self._runner.cancel)

        try:
            feature_project_id = self._resolve_master_feature_project_id()
        except Exception as exc:
            messagebox.showerror("Build", str(exc), parent=self)
            return

        kwargs = {
            "market": str(self._market_var.get() or "NIFTY").upper(),
            "interval_sec": interval,
            "include_registry": bool(self._include_registry.get()),
            "include_pipeline": bool(self._include_pipeline.get()),
            "all_days": all_days_flag,
            "no_null_data": bool(self._no_null_data.get()),
            "pipeline_no_null_report": bool(self._pipeline_no_null_report.get()),
            "premium_enabled": prem_en,
            "premium_min": prem_lo,
            "premium_max": prem_hi,
            "master_db_path": path,
            "feature_project_id": feature_project_id,
        }
        if self._include_pipeline.get():
            kwargs["pipeline_id"] = str(self._build_pipeline_var.get() or "").strip().upper()
        if not all_days_flag:
            kwargs["selected_days"] = explicit_days
        if day_filter_meta.get("selected_dates"):
            kwargs["trading_day_filter"] = day_filter_meta

        def _on_progress(payload: dict[str, Any]) -> None:
            self._progress_q.put({"type": "progress", "payload": payload})
            try:
                mgr.publish({
                    "status": payload.get("status") or "running",
                    "job_kind": "analysis_dataset_build",
                    "message": payload.get("message") or payload.get("current_feature") or "",
                    "percent": payload.get("percent"),
                    "stage_name": payload.get("current_source") or payload.get("stage"),
                })
            except Exception:
                pass

        def _on_done(result: dict[str, Any]) -> None:
            self._progress_q.put({"type": "done", "result": result})

        try:
            self._runner.start(
                export_kwargs=kwargs,
                on_done=_on_done,
                on_progress=_on_progress,
            )
        except Exception as exc:
            self._building = False
            self._btn_build.configure(state="normal")
            self._btn_cancel.configure(state="disabled")
            messagebox.showerror("Build", str(exc), parent=self)

    def _cancel_build(self) -> None:
        if self._runner.running:
            self._runner.cancel()
            self._mon_status.set("Cancelling")

    def _handle_progress(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "progress":
            self._apply_progress(msg.get("payload") or {})
        elif kind == "done":
            self._finish_build(msg.get("result") or {})

    def _apply_progress(self, p: dict[str, Any]) -> None:
        status = str(p.get("status") or "running")
        self._mon_status.set(status.title())
        self._mon_elapsed.set(_fmt_duration(p.get("elapsed_sec")))
        self._mon_eta.set(_fmt_duration(p.get("eta_sec")))
        pct = float(p.get("percent") or 0.0)
        self._mon_percent.set(f"{pct:.0f}%")
        self._progress["value"] = max(0.0, min(100.0, pct))

        self._mon_reg.set(
            f"Registry     {int(p.get('registry_done') or 0)} / {int(p.get('registry_total') or 0)}"
        )
        self._mon_pipe.set(
            f"Pipeline     {int(p.get('pipeline_done') or 0)} / {int(p.get('pipeline_total') or 0)}"
        )
        self._mon_overall.set(
            f"Overall Progress     {int(p.get('overall_done') or 0)} / {int(p.get('overall_total') or 0)}"
        )
        feat = str(p.get("current_feature") or "").strip()
        day = str(p.get("day") or "").strip()
        day_i = p.get("day_index")
        day_tot = p.get("day_total")
        if day_i is not None and day_tot:
            self._mon_day.set(f"Day {int(day_i)} / {int(day_tot)}")
        else:
            self._mon_day.set("—")
        self._mon_current.set(day or feat or str(p.get("message") or "—"))
        mode = str(p.get("mode") or "").strip()
        self._mon_mode.set(mode.title() if mode else "—")
        self._mon_source.set(str(p.get("current_source") or "—"))
        cps = p.get("columns_per_sec")
        self._mon_speed.set(f"Columns/s  {cps:.1f}" if isinstance(cps, (int, float)) else "Columns/s  —")
        rows = p.get("rows") if p.get("rows") is not None else p.get("rows_processed")
        self._mon_rows.set(_fmt_int(rows) if rows is not None else "—")
        feats = p.get("features")
        self._mon_features.set(_fmt_int(feats) if feats is not None else "—")
        tok_i, tok_t = p.get("token_index"), p.get("token_total")
        if tok_i is not None and tok_t:
            self._mon_token.set(f"{int(tok_i)} / {int(tok_t)}")
        else:
            self._mon_token.set("—")
        wave_i, wave_t = p.get("wave_index"), p.get("wave_total")
        if wave_i is not None and wave_t:
            self._mon_wave.set(f"{int(wave_i)} / {int(wave_t)}")
        else:
            self._mon_wave.set("—")

        seen_stages: set[str] = set()
        for st in p.get("stages") or []:
            sid = str(st.get("id") or "")
            if sid not in self._stage_vars:
                continue
            seen_stages.add(sid)
            mark = {"done": "✓", "running": "▶", "pending": "○"}.get(str(st.get("status")), "○")
            self._stage_vars[sid].set(f"{mark} {st.get('label') or sid}")
        if "no_null" not in seen_stages and "no_null" in self._stage_vars:
            self._stage_vars["no_null"].set("○ No-Null Filter (off)")
        if "premium" not in seen_stages and "premium" in self._stage_vars:
            self._stage_vars["premium"].set("○ Premium Filter (off)")

        lines = p.get("log_lines") or []
        if isinstance(lines, list) and len(lines) > self._log_seen:
            for line in lines[self._log_seen:]:
                self._append_log(str(line))
            self._log_seen = len(lines)

    def _finish_build(self, result: dict[str, Any]) -> None:
        self._building = False
        self._btn_build.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        mgr = get_build_progress_manager()
        status = str(result.get("status") or "failed")
        if status == "completed":
            self._last_summary = result
            self._mon_status.set("Completed")
            self._progress["value"] = 100
            self._render_status_cards()
            missing = int(result.get("pipeline_missing_count") or 0)
            missing_note = ""
            if missing:
                missing_note = f"\nPipeline not created: {missing} (see activity log / missing list)"
            no_null_note = ""
            if result.get("no_null_data"):
                no_null_note = (
                    f"\nNo-Null            on · dropped cols "
                    f"{_fmt_int(result.get('no_null_dropped_count'))}"
                )
            if result.get("pipeline_no_null_report_enabled") or result.get(
                "pipeline_no_null_report"
            ):
                no_null_note += "\nNo-Null Analysis   see Activity Log"
            prem_note = ""
            if result.get("premium_enabled"):
                pmin = result.get("premium_min")
                pmax = result.get("premium_max")
                prem_rep = result.get("premium_report") or {}
                kept = prem_rep.get("rows_after")
                before = prem_rep.get("rows_before")
                band = f"₹{pmin:g}–₹{pmax:g}" if pmin is not None and pmax is not None else "on"
                if kept is not None and before is not None:
                    prem_note = f"\nPremium            {band} · kept {_fmt_int(kept)} / {_fmt_int(before)}"
                else:
                    prem_note = f"\nPremium            {band}"
            xform_note = ""
            ts = result.get("transformation_summary")
            if isinstance(ts, dict) and ts:
                from chain_replay_ml.dataset_builder.transformations.day_pipeline_support import (
                    format_transformation_summary_text,
                )

                xform_note = "\n\n" + format_transformation_summary_text(ts)
            self._summary_var.set(
                "Build Completed\n"
                f"Registry          {_fmt_int(result.get('registry_present'))} / {_fmt_int(result.get('registry_total'))}\n"
                f"Pipeline          {_fmt_int(result.get('pipeline_present'))} / {_fmt_int(result.get('pipeline_total'))}\n"
                f"----------------------------\n"
                f"Total Columns     {_fmt_int(result.get('feature_count'))}\n"
                f"Rows              {_fmt_int(result.get('row_count'))}\n"
                f"Build Time        {_fmt_duration(result.get('build_time_sec'))}\n"
                f"Output Size       {_fmt_bytes(result.get('output_size_bytes'))}\n"
                f"Dataset           {result.get('dataset_name') or '—'}"
                f"{no_null_note}"
                f"{prem_note}"
                f"{missing_note}"
                f"{xform_note}"
            )
            try:
                mgr.publish({"status": "completed", "job_kind": "analysis_dataset_build", "percent": 100})
            except Exception:
                pass
            # Register snapshot into Phase 2 analysis.db (fingerprint + run scope).
            try:
                from chain_replay_ml.dataset_builder.analysis_lab_store import (
                    register_dataset,
                )

                pq = (
                    str(result.get("parquet_path") or "").strip()
                    or str(result.get("output_parquet") or "").strip()
                )
                if pq:
                    import os

                    abs_pq = pq
                    if not os.path.isabs(abs_pq):
                        abs_pq = os.path.join(chart_data_dir(self.chart_dir), abs_pq)
                    if os.path.isfile(abs_pq):
                        register_dataset(
                            chart_data_dir(self.chart_dir),
                            abs_pq,
                            name=str(result.get("dataset_name") or "") or None,
                            relative_path=str(result.get("output_parquet") or "") or None,
                        )
                        self._append_log(
                            f"Analysis Lab  registered {result.get('dataset_name')} "
                            f"in analysis.db"
                        )
            except Exception as exc:
                self._append_log(f"Analysis Lab  register skipped: {exc}")
            messagebox.showinfo(
                "Analysis Dataset",
                f"Created {result.get('dataset_name')}\n"
                f"Features: {_fmt_int(result.get('feature_count'))} · "
                f"Rows: {_fmt_int(result.get('row_count'))}",
                parent=self,
            )
        else:
            self._mon_status.set("Failed")
            err = str(result.get("error") or "Build failed")
            self._summary_var.set(f"Build failed\n{err}")
            self._append_log(f"ERROR  {err}")
            try:
                mgr.publish({"status": "failed", "job_kind": "analysis_dataset_build", "message": err})
            except Exception:
                pass
            messagebox.showerror("Analysis Dataset", err, parent=self)

    def _clear_log(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _append_log(self, line: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", line.rstrip() + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")
