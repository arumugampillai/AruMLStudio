"""Dataset Registry page — mirrors web registry tab (standalone)."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from . import registry_format as fmt
from . import registry_service as svc
from .lazy_panel import LazyLoadMixin
from .ui_state import get_ui_state_manager
from .ui_util import open_path


class RegistryPanel(ttk.Frame, LazyLoadMixin):
    """Golden regression, registry table, and detail tabs (Summary/Schema/Audit/Compare/Merge)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_builder: Callable[[str], None] | None = None,
        on_compare_datasets: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_builder = on_open_builder
        self._on_compare_datasets = on_compare_datasets
        self._rows: list[dict[str, Any]] = []
        self._selected_name: str | None = None
        self._job_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._busy = False
        self._merge_job_id: str | None = None
        self._summary_gen = 0
        self._status_var = tk.StringVar(value="")
        self._build_ui()
        self._ui_state = get_ui_state_manager()
        # Compare-B is repopulated on every refresh; restore now so the
        # first `refresh_combobox()` call preserves the saved value if it
        # still exists (else falls back to the newest dataset).
        self._ui_state.bind_combobox(
            self._compare_b_combo, "registry.compare_b", var=self._compare_b_var
        )
        self._ui_state.bind_notebook(self._notebook, "registry.detail_tab")
        self._lazy_init()

    def on_show(self) -> None:
        self.refresh_all(lazy=True)

    def refresh_all(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_registry_bundle,
                apply=self._apply_registry_bundle,
                message="Loading dataset registry…",
                status_var=self._status_var,
            )
            return
        self._apply_registry_bundle(self._fetch_registry_bundle())

    def _fetch_registry_bundle(self) -> dict[str, Any]:
        rows = svc.list_registry_datasets(self.chart_dir)
        golden = svc.golden_status(self.chart_dir)
        schema = svc.load_schema_viewer()
        return {"rows": rows, "golden": golden, "schema": schema}

    def _apply_registry_bundle(self, bundle: dict[str, Any]) -> None:
        self._rows = list(bundle.get("rows") or [])
        self._apply_registry_rows()
        status = bundle.get("golden") or {}
        self._set_text(self._golden_text, fmt.format_golden_status(status))
        can_update = bool(status.get("manifest_update_allowed") and status.get("dataset_exists"))
        self._golden_manifest_btn.configure(state="normal" if can_update else "disabled")
        schema = bundle.get("schema")
        try:
            self._set_text(self._schema_text, fmt.format_schema_view(schema))
        except Exception as exc:
            self._set_text(self._schema_text, f"Error loading schema:\n{exc}")

    def _apply_registry_rows(self) -> None:
        from .selection_lists import refresh_combobox

        self._tree.delete(*self._tree.get_children())
        names: list[str] = []
        for row in self._rows:
            iid = row["dataset_name"]
            names.append(iid)
            draft = " [draft]" if row.get("is_draft") else ""
            self._tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    iid + draft,
                    row.get("market") or "—",
                    fmt.fmt_selection(row),
                    fmt.fmt_num(row.get("day_count")),
                    fmt.fmt_num(row.get("row_count")),
                    fmt.fmt_num(row.get("feature_count")),
                    fmt.fmt_num(row.get("target_count")),
                    fmt.fmt_audit_cell(row),
                    fmt.fmt_readiness(row),
                    fmt.fmt_files(row),
                    fmt.fmt_created(row),
                ),
            )
        refresh_combobox(self._compare_b_combo, names, var=self._compare_b_var)
        if self._selected_name and self._selected_name in names:
            self._tree.selection_set(self._selected_name)
            self._load_summary(self._selected_name, show_tab=False)
        elif names:
            name = names[0]
            self._selected_name = name
            self._compare_a_var.set(name)
            self._tree.selection_set(name)
            self._load_summary(name, show_tab=False)
        self._update_compare_button()
        self._status_var.set(f"{len(self._rows)} dataset(s) in registry")

    def poll_jobs(self) -> None:
        try:
            while True:
                msg = self._job_queue.get_nowait()
                self._handle_job_message(msg)
        except queue.Empty:
            pass
        if self._merge_job_id:
            self._poll_merge_job()

    def _build_ui(self) -> None:
        self._build_golden_section()
        self._build_toolbar()

        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=2)
        self._build_table(table_frame)

        detail_frame = ttk.LabelFrame(paned, text="Dataset detail", padding=4)
        paned.add(detail_frame, weight=3)
        self._build_detail(detail_frame)

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))

    def _build_golden_section(self) -> None:
        frame = ttk.LabelFrame(self, text="Golden Dataset Regression", padding=6)
        frame.pack(fill="x", padx=8, pady=(8, 4))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(0, 4))
        self._golden_fast_btn = ttk.Button(btn_row, text="Run Fast", command=lambda: self._run_golden("fast"))
        self._golden_fast_btn.pack(side="left", padx=(0, 4))
        self._golden_full_btn = ttk.Button(btn_row, text="Run Full", command=lambda: self._run_golden("full"))
        self._golden_full_btn.pack(side="left", padx=4)
        self._golden_manifest_btn = ttk.Button(
            btn_row, text="Update Manifest", command=self._update_golden_manifest, state="disabled"
        )
        self._golden_manifest_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Refresh", command=self._load_golden_status).pack(side="left", padx=4)

        self._golden_text = scrolledtext.ScrolledText(frame, height=7, wrap="word", font=("Consolas", 9))
        self._golden_text.pack(fill="x")
        self._golden_text.configure(state="disabled")

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(fill="x")

        actions = (
            ("Summary", self._show_summary),
            ("Audit", self._run_audit_for_selected),
            ("Cached audit", self._view_cached_audit),
            ("Validate", self._open_validate_dialog),
            ("Append RR Labels", self._open_rr_enrich_dialog),
            ("Train", self._open_train),
            ("Metadata", self._open_metadata),
            ("Generate CSV", self._generate_csv),
            ("Delete CSV", self._delete_csv),
            ("Delete", self._delete_selected),
        )
        for label, cmd in actions:
            ttk.Button(bar, text=label, command=cmd).pack(side="left", padx=2)
        toolbar_right = ttk.Frame(bar)
        toolbar_right.pack(side="right")
        self._compare_btn: ttk.Button | None = None
        if self._on_compare_datasets:
            self._compare_btn = ttk.Button(toolbar_right, text="Compare", command=self._open_compare)
        ttk.Button(toolbar_right, text="Refresh", command=self.refresh_registry).pack(side="right", padx=4)
        ttk.Button(toolbar_right, text="Open datasets folder", command=self._open_datasets_folder).pack(side="right", padx=4)
        ttk.Button(toolbar_right, text="Save Summary PDF", command=self._save_summary_pdf).pack(side="right", padx=4)

    def _build_table(self, parent: ttk.Frame) -> None:
        cols = (
            ("dataset", "Dataset", 160),
            ("market", "Market", 72),
            ("selection", "Selection", 100),
            ("days", "Days", 48),
            ("rows", "Rows", 72),
            ("features", "Features", 64),
            ("targets", "Targets", 56),
            ("audit", "Audit", 100),
            ("readiness", "Readiness", 88),
            ("files", "Files", 56),
            ("created", "Created", 120),
        )
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True)
        self._tree = ttk.Treeview(
            tree_frame,
            columns=[c[0] for c in cols],
            show="headings",
            selectmode="extended",
            height=8,
        )
        for col_id, heading, width in cols:
            self._tree.heading(col_id, text=heading)
            self._tree.column(col_id, width=width, minwidth=40)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-1>", lambda _e: self._show_summary())

    def _build_detail(self, parent: ttk.Frame) -> None:
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill="both", expand=True)

        self._summary_text = self._make_text_tab("Summary")
        self._schema_text = self._make_text_tab("Schema")
        self._audit_text = self._make_text_tab("Audit")
        self._compare_frame = ttk.Frame(self._notebook, padding=4)
        self._notebook.add(self._compare_frame, text="Compare")
        self._merge_frame = ttk.Frame(self._notebook, padding=4)
        self._notebook.add(self._merge_frame, text="Merge")
        self._build_compare_tab()
        self._build_merge_tab()

    def _make_text_tab(self, title: str) -> scrolledtext.ScrolledText:
        frame = ttk.Frame(self._notebook)
        self._notebook.add(frame, text=title)
        txt = scrolledtext.ScrolledText(frame, wrap="word", font=("Consolas", 9))
        txt.pack(fill="both", expand=True)
        txt.configure(state="disabled")
        return txt

    def _build_compare_tab(self) -> None:
        row = ttk.Frame(self._compare_frame)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Dataset A:").pack(side="left")
        self._compare_a_var = tk.StringVar()
        ttk.Label(row, textvariable=self._compare_a_var, width=24).pack(side="left", padx=4)
        ttk.Label(row, text="Dataset B:").pack(side="left", padx=(12, 0))
        self._compare_b_var = tk.StringVar()
        self._compare_b_combo = ttk.Combobox(row, textvariable=self._compare_b_var, width=28, state="readonly")
        self._compare_b_combo.pack(side="left", padx=4)
        ttk.Button(row, text="Compare", command=self._run_compare).pack(side="left", padx=8)
        self._compare_text = scrolledtext.ScrolledText(self._compare_frame, wrap="word", font=("Consolas", 9))
        self._compare_text.pack(fill="both", expand=True)
        self._compare_text.configure(state="disabled")

    def _build_merge_tab(self) -> None:
        top = ttk.Frame(self._merge_frame)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="Load merge plan", command=self._load_merge_plan).pack(side="left")
        ttk.Button(top, text="Select all candidates", command=self._merge_select_all).pack(side="left", padx=4)
        ttk.Button(top, text="Clear selection", command=self._merge_clear).pack(side="left", padx=4)
        self._merge_start_btn = ttk.Button(top, text="Start merge", command=self._start_merge)
        self._merge_start_btn.pack(side="left", padx=8)

        body = ttk.Panedwindow(self._merge_frame, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)
        list_frame = ttk.Frame(body)
        body.add(list_frame, weight=1)
        self._merge_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, font=("Consolas", 9))
        merge_sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._merge_list.yview)
        self._merge_list.configure(yscrollcommand=merge_sb.set)
        self._merge_list.pack(side="left", fill="both", expand=True)
        merge_sb.pack(side="right", fill="y")
        self._merge_plan_text = scrolledtext.ScrolledText(body, wrap="word", font=("Consolas", 9), width=40)
        body.add(self._merge_plan_text, weight=1)
        self._merge_plan_text.configure(state="disabled")
        self._merge_progress = scrolledtext.ScrolledText(self._merge_frame, height=4, wrap="word", font=("Consolas", 9))
        self._merge_progress.pack(fill="x", pady=(4, 0))
        self._merge_progress.configure(state="disabled")
        self._merge_candidates: list[str] = []

    def _set_text(self, widget: scrolledtext.ScrolledText, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _append_text(self, widget: scrolledtext.ScrolledText, line: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, line + "\n")
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _selected_row(self) -> dict[str, Any] | None:
        if not self._selected_name:
            return None
        for row in self._rows:
            if row.get("dataset_name") == self._selected_name:
                return row
        return None

    def _require_selection(self) -> str | None:
        name = self._selected_name
        if not name:
            messagebox.showinfo("Dataset Registry", "Select a dataset from the table first.")
            return None
        return name

    def refresh_registry(self, *, lazy: bool = True) -> None:
        if lazy:
            self.refresh_all(lazy=True)
            return
        try:
            self._rows = svc.list_registry_datasets(self.chart_dir)
        except Exception as exc:
            messagebox.showerror("Dataset Registry", f"Could not load registry:\n{exc}")
            self._rows = []
        self._apply_registry_rows()

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        sel = self._tree.selection()
        self._update_compare_button()
        if not sel:
            return
        if len(sel) != 1:
            return
        name = sel[0]
        if name == self._selected_name:
            return
        self._selected_name = name
        self._compare_a_var.set(self._selected_name)
        self._load_summary(name, show_tab=True)

    def _update_compare_button(self) -> None:
        if self._compare_btn is None:
            return
        sel = self._tree.selection()
        if len(sel) == 2:
            self._compare_btn.pack(side="right", padx=4)
        else:
            self._compare_btn.pack_forget()

    def _open_compare(self) -> None:
        sel = self._tree.selection()
        if len(sel) != 2 or not self._on_compare_datasets:
            return
        self._on_compare_datasets(sel[0], sel[1])

    def _load_golden_status(self) -> None:
        try:
            status = svc.golden_status(self.chart_dir)
            self._set_text(self._golden_text, fmt.format_golden_status(status))
            can_update = bool(status.get("manifest_update_allowed") and status.get("dataset_exists"))
            self._golden_manifest_btn.configure(state="normal" if can_update else "disabled")
        except Exception as exc:
            self._set_text(self._golden_text, f"Error loading golden status:\n{exc}")

    def _run_golden(self, mode: str) -> None:
        if self._busy:
            messagebox.showwarning("Golden Regression", "Another job is already running.")
            return
        self._set_busy(True)
        self._golden_fast_btn.configure(state="disabled")
        self._golden_full_btn.configure(state="disabled")
        self._set_text(self._golden_text, f"Running {mode} regression…")

        def worker() -> None:
            try:
                result = svc.run_golden(self.chart_dir, mode)
                self._job_queue.put({"kind": "golden_done", "result": result})
            except Exception as exc:
                self._job_queue.put({"kind": "error", "target": "golden", "message": str(exc)})

        threading.Thread(target=worker, daemon=True, name="golden-regression").start()

    def _update_golden_manifest(self) -> None:
        if not messagebox.askyesno(
            "Update Manifest",
            "Capture the current golden dataset as the manifest reference?",
        ):
            return
        try:
            svc.update_golden_manifest(self.chart_dir)
            messagebox.showinfo("Update Manifest", "Manifest updated.")
            self._load_golden_status()
        except Exception as exc:
            messagebox.showerror("Update Manifest", str(exc))

    def _load_schema_tab(self) -> None:
        try:
            data = svc.load_schema_viewer()
            self._set_text(self._schema_text, fmt.format_schema_view(data))
        except Exception as exc:
            self._set_text(self._schema_text, f"Error loading schema:\n{exc}")

    def _load_summary(self, name: str, *, show_tab: bool = True) -> None:
        if show_tab:
            self._notebook.select(0)
        self._set_text(self._summary_text, f"Loading summary for {name}…")
        self._summary_gen += 1
        gen = self._summary_gen

        def worker() -> None:
            try:
                data = svc.load_dataset_summary(self.chart_dir, name)
                self._job_queue.put({"kind": "summary_done", "gen": gen, "data": data})
            except Exception as exc:
                self._job_queue.put({"kind": "summary_error", "gen": gen, "message": str(exc)})

        threading.Thread(target=worker, daemon=True, name=f"summary-{name}").start()

    def _show_summary(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._load_summary(name, show_tab=True)

    def _run_audit_for_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._notebook.select(2)
        self._set_text(self._audit_text, f"Running audit for {name}…\n")
        self._run_background(
            f"audit:{name}",
            lambda on_progress=None: svc.run_audit(self.chart_dir, name, on_progress=on_progress),
            on_progress=lambda p: self._append_text(
                self._audit_text,
                p.get("message") or p.get("stage") or str(p.get("event") or p),
            ),
            on_done=lambda data: self._set_text(self._audit_text, fmt.format_audit_report(data)),
            refresh_on_done=True,
        )

    def _view_cached_audit(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._notebook.select(2)
        try:
            data = svc.load_dataset_metadata(self.chart_dir, name)
            cache = data.get("audit_cache")
            if not cache:
                messagebox.showinfo("Cached audit", "No audit cache — run Audit first.")
                return
            report = dict(cache)
            report.setdefault("dataset_name", name)
            self._set_text(self._audit_text, fmt.format_audit_report(report))
        except Exception as exc:
            messagebox.showerror("Cached audit", str(exc))

    def _open_validate_dialog(self) -> None:
        name = self._require_selection()
        if not name:
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Validate — {name}")
        dlg.geometry("520x420")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        opts = ttk.Frame(dlg, padding=8)
        opts.pack(fill="x")
        ttk.Label(opts, text="Sample rows:").pack(side="left")
        sample_var = tk.IntVar(value=100)
        for n in (50, 100, 500):
            ttk.Radiobutton(opts, text=str(n), variable=sample_var, value=n).pack(side="left", padx=4)
        ttk.Label(opts, text="Tolerance:").pack(side="left", padx=(16, 4))
        tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(opts, textvariable=tol_var, width=10).pack(side="left")

        body = scrolledtext.ScrolledText(dlg, wrap="word", font=("Consolas", 9))
        body.pack(fill="both", expand=True, padx=8, pady=4)

        def start() -> None:
            try:
                tolerance = float(tol_var.get())
            except ValueError:
                messagebox.showerror("Validate", "Invalid tolerance.", parent=dlg)
                return
            body.configure(state="normal")
            body.delete("1.0", tk.END)
            body.insert("1.0", f"Validating {name}…\n")
            body.configure(state="disabled")
            start_btn.configure(state="disabled")

            def worker() -> None:
                def on_progress(p: dict[str, Any]) -> None:
                    self._job_queue.put({
                        "kind": "validate_progress",
                        "dlg": dlg,
                        "body": body,
                        "line": p.get("message") or p.get("stage") or str(p),
                    })

                try:
                    report = svc.run_validation(
                        self.chart_dir,
                        name,
                        n_sample=sample_var.get(),
                        tolerance=tolerance,
                        on_progress=on_progress,
                    )
                    self._job_queue.put({
                        "kind": "validate_done",
                        "dlg": dlg,
                        "body": body,
                        "start_btn": start_btn,
                        "report": report,
                    })
                except Exception as exc:
                    self._job_queue.put({
                        "kind": "validate_error",
                        "dlg": dlg,
                        "body": body,
                        "start_btn": start_btn,
                        "message": str(exc),
                    })

            threading.Thread(target=worker, daemon=True, name="dataset-validate").start()

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack(fill="x")
        start_btn = ttk.Button(btn_row, text="Start validation", command=start)
        start_btn.pack(side="left")
        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="right")

    def _open_rr_enrich_dialog(self) -> None:
        """Append RR classifier labels from a Prediction Lab onto the selected dataset."""
        name = self._require_selection()
        if not name:
            return
        try:
            labs = svc.list_rr_enrichment_labs(self.chart_dir, name)
        except Exception as exc:
            messagebox.showerror("Append RR Labels", str(exc))
            return
        if not labs:
            messagebox.showinfo(
                "Append RR Labels",
                "No Research Lab prediction datasets with RR labels found.\n\n"
                "Build a Prediction Dataset (with RR labels) first, then retry.",
            )
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Append RR Labels — {name}")
        dlg.geometry("640x420")
        dlg.transient(self.winfo_toplevel())
        ttk.Label(
            dlg,
            text=(
                f"Enrich training dataset \"{name}\" with RR hit labels\n"
                f"(rr_1_1 / rr_2_3 / rr_1_2 / rr_1_3 / rr_1_4)\n"
                "from Seen prediction rows only. Unseen rows are ignored (no leakage).\n"
                "Features are not recomputed. A new dataset is saved."
            ),
            justify="left",
            wraplength=600,
        ).pack(anchor="w", padx=12, pady=(12, 8))

        ttk.Label(dlg, text="Prediction Lab (source of RR labels):").pack(anchor="w", padx=12)
        lab_var = tk.StringVar()
        labels = [str(lab.get("label") or lab.get("db_path")) for lab in labs]
        path_by_label = {
            str(lab.get("label") or lab.get("db_path")): str(lab.get("db_path"))
            for lab in labs
        }
        combo = ttk.Combobox(dlg, textvariable=lab_var, values=labels, state="readonly", width=80)
        combo.pack(fill="x", padx=12, pady=4)
        if labels:
            lab_var.set(labels[0])

        status = tk.StringVar(value="Select a lab and click Enrich.")
        ttk.Label(dlg, textvariable=status, foreground="#666").pack(anchor="w", padx=12, pady=8)

        result_box = scrolledtext.ScrolledText(dlg, wrap="word", font=("Consolas", 9), height=12)
        result_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        result_box.configure(state="disabled")

        def _set_result(text: str) -> None:
            result_box.configure(state="normal")
            result_box.delete("1.0", "end")
            result_box.insert("1.0", text)
            result_box.configure(state="disabled")

        def start() -> None:
            label = lab_var.get().strip()
            lab_path = path_by_label.get(label)
            if not lab_path:
                messagebox.showinfo("Append RR Labels", "Select a Prediction Lab.")
                return
            start_btn.configure(state="disabled")
            status.set("Enriching… matching rows and writing new dataset…")
            self.update_idletasks()

            local_q: queue.Queue[dict[str, Any]] = queue.Queue()

            def worker() -> None:
                try:
                    result = svc.enrich_dataset_with_rr_labels(
                        self.chart_dir, name, lab_path
                    )
                    from chain_replay_ml.model_lab.rr_dataset_enrich import (
                        format_rr_enrichment_report,
                    )

                    report_text = format_rr_enrichment_report(result)
                    local_q.put(
                        {
                            "ok": bool(result.get("ok")),
                            "result": result,
                            "report": report_text,
                        }
                    )
                except Exception as exc:
                    local_q.put(
                        {
                            "ok": False,
                            "error": str(exc),
                            "report": f"RR Dataset Enrichment\n\nFailed:\n{exc}",
                        }
                    )

            def on_poll() -> None:
                try:
                    msg = local_q.get_nowait()
                except queue.Empty:
                    dlg.after(200, on_poll)
                    return
                start_btn.configure(state="normal")
                _set_result(str(msg.get("report") or ""))
                if msg.get("ok"):
                    status.set("Completed — new dataset registered.")
                    self.refresh_all(lazy=False)
                    saved = (msg.get("result") or {}).get("dataset_name")
                    messagebox.showinfo(
                        "Append RR Labels",
                        f"Saved as {saved}\n\nSee report in the dialog.",
                        parent=dlg,
                    )
                else:
                    status.set("Failed — see report.")
                    err = msg.get("error") or "Validation failed"
                    messagebox.showerror("Append RR Labels", err, parent=dlg)

            threading.Thread(target=worker, daemon=True, name="rr-enrich").start()
            dlg.after(200, on_poll)

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack(fill="x")
        start_btn = ttk.Button(btn_row, text="Enrich Dataset", command=start)
        start_btn.pack(side="left")
        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="right")

    def _open_train(self) -> None:
        row = self._selected_row()
        name = self._require_selection()
        if not name:
            return
        if not svc.training_allowed(row):
            messagebox.showwarning(
                "Train",
                "Training is not allowed until audit/validation readiness passes.",
            )
            return
        if self._on_open_builder:
            self._on_open_builder(name)
            return
        messagebox.showinfo(
            "Train",
            f"Open Create Model from the main navigation to train on dataset \"{name}\".",
        )

    def _show_compare_tab(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._compare_a_var.set(name)
        self._notebook.select(3)

    def _run_compare(self) -> None:
        a = self._compare_a_var.get().strip()
        b = self._compare_b_var.get().strip()
        if not a or not b:
            messagebox.showinfo("Compare", "Select datasets A and B.")
            return
        if a == b:
            messagebox.showinfo("Compare", "Choose two different datasets.")
            return
        self._set_text(self._compare_text, "Comparing…")
        self._run_background(
            f"compare:{a}:{b}",
            lambda: svc.compare_registry_datasets(self.chart_dir, a, b),
            on_done=lambda data: self._set_text(self._compare_text, fmt.format_compare(data)),
        )

    def _open_metadata(self) -> None:
        name = self._require_selection()
        if not name:
            return
        meta_path = None
        for row in self._rows:
            if row.get("dataset_name") == name:
                meta_path = row.get("metadata_path")
                break
        from .dataset_metadata_panel import open_dataset_metadata_window

        open_dataset_metadata_window(
            self,
            chart_dir=self.chart_dir,
            dataset_name=name,
            metadata_path=str(meta_path or "") or None,
        )

    def _generate_csv(self) -> None:
        name = self._require_selection()
        if not name:
            return
        replace = False
        try:
            info = svc.load_dataset_metadata(self.chart_dir, name).get("csv_export") or {}
        except Exception as exc:
            messagebox.showerror("Generate CSV", str(exc))
            return
        if info.get("status") == "Generated":
            if not messagebox.askyesno(
                "Regenerate CSV",
                f'A CSV export already exists for "{name}".\n\n'
                "Regenerate it? Only the CSV file will be replaced; the Parquet dataset stays unchanged.",
            ):
                return
            replace = True

        def worker() -> dict[str, Any]:
            return svc.generate_registry_csv(self.chart_dir, name, replace=replace)

        def on_done(_result: Any) -> None:
            messagebox.showinfo("Generate CSV", f'CSV export created for "{name}".')

        self._run_background(
            f"csv:{name}",
            lambda: worker(),
            on_done=on_done,
            refresh_on_done=True,
        )

    def _delete_csv(
        self,
        *,
        dataset_name: str | None = None,
        on_done: Callable[[], None] | None = None,
        parent: tk.Misc | None = None,
    ) -> None:
        name = dataset_name or self._require_selection()
        if not name:
            return
        try:
            info = svc.load_dataset_metadata(self.chart_dir, name).get("csv_export") or {}
        except Exception as exc:
            messagebox.showerror("Delete CSV", str(exc), parent=parent)
            return
        if info.get("status") != "Generated":
            messagebox.showinfo(
                "Delete CSV",
                "No CSV export exists for this dataset.",
                parent=parent,
            )
            return
        if not messagebox.askyesno(
            "Delete CSV",
            f'Delete the CSV export for "{name}"?\n\n'
            "The Parquet dataset and registry entry will not be removed.",
            parent=parent,
        ):
            return
        try:
            svc.delete_registry_csv(self.chart_dir, name)
        except Exception as exc:
            messagebox.showerror("Delete CSV", str(exc), parent=parent)
            return
        if on_done:
            on_done()
        else:
            messagebox.showinfo("Delete CSV", "CSV export deleted.", parent=parent)
        self.refresh_registry()

    def _delete_selected(self) -> None:
        name = self._require_selection()
        if not name:
            return
        if not messagebox.askyesno(
            "Delete dataset",
            f'Delete dataset "{name}"?\n\n'
            "Removes .parquet, .json, .expected.json, audit cache, validation, and investigation history.",
        ):
            return
        try:
            svc.delete_registry_dataset(self.chart_dir, name)
            self._selected_name = None
            self.refresh_registry()
            messagebox.showinfo("Delete", f'Dataset "{name}" deleted.')
        except Exception as exc:
            messagebox.showerror("Delete", str(exc))

    def _open_datasets_folder(self) -> None:
        from chain_replay_ml.dataset_builder.writer import datasets_dir

        path = datasets_dir(chart_data_dir(self.chart_dir))
        open_path(path)

    def _save_summary_pdf(self) -> None:
        name = self._require_selection()
        if not name:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"{name}_summary.pdf",
        )
        if not path:
            return
        try:
            svc.save_summary_pdf(self.chart_dir, name, path)
            messagebox.showinfo("PDF", f"Saved to\n{path}")
            open_path(path)
        except Exception as exc:
            messagebox.showerror("PDF", str(exc))

    def _load_merge_plan(self) -> None:
        name = self._require_selection()
        if not name:
            return
        self._notebook.select(4)
        try:
            plan = svc.merge_plan(self.chart_dir, name)
        except Exception as exc:
            messagebox.showerror("Merge", str(exc))
            return
        self._merge_candidates = [c["name"] for c in (plan.get("merge_candidates") or [])]
        self._merge_list.delete(0, tk.END)
        for c in plan.get("merge_candidates") or []:
            tag = " [new]" if c.get("is_new_since_build") else ""
            self._merge_list.insert(tk.END, f"{c['name']} ({c.get('group', '')}){tag}")
        self._set_text(self._merge_plan_text, fmt.format_merge_plan(plan))

    def _merge_select_all(self) -> None:
        self._merge_list.select_set(0, tk.END)

    def _merge_clear(self) -> None:
        self._merge_list.selection_clear(0, tk.END)

    def _start_merge(self) -> None:
        name = self._require_selection()
        if not name:
            return
        indices = self._merge_list.curselection()
        if not indices:
            messagebox.showinfo("Merge", "Select features to merge.")
            return
        features = [self._merge_candidates[i] for i in indices if i < len(self._merge_candidates)]
        if not features:
            return
        if not messagebox.askyesno("Merge", f"Merge {len(features)} feature(s) into {name}?"):
            return
        try:
            job = svc.start_merge(self.chart_dir, name, features)
            self._merge_job_id = str(job.get("job_id") or "")
            self._set_text(self._merge_progress, f"Merge started: {self._merge_job_id}\n")
            self._merge_start_btn.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Merge", str(exc))

    def _poll_merge_job(self) -> None:
        if not self._merge_job_id:
            return
        try:
            job = svc.merge_job_status(self._merge_job_id)
        except Exception:
            return
        if not job:
            return
        status = str(job.get("status") or "")
        msg = job.get("message") or job.get("stage") or status
        self._set_text(self._merge_progress, f"Merge {self._merge_job_id}: {msg}\n{fmt.format_json(job)}")
        if status in ("done", "completed", "failed", "error", "cancelled"):
            self._merge_job_id = None
            self._merge_start_btn.configure(state="normal")
            if status in ("done", "completed"):
                self.refresh_registry()
                self._load_merge_plan()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            self._golden_fast_btn.configure(state="normal")
            self._golden_full_btn.configure(state="normal")

    def _run_background(
        self,
        tag: str,
        fn: Callable[..., Any],
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_done: Callable[[Any], None] | None = None,
        refresh_on_done: bool = False,
    ) -> None:
        if self._busy:
            messagebox.showwarning("Dataset Registry", "Another job is already running.")
            return
        self._set_busy(True)
        self._status_var.set(f"Running {tag}…")

        def worker() -> None:
            try:
                if on_progress is not None:
                    def progress_cb(p: dict[str, Any]) -> None:
                        self._job_queue.put({
                            "kind": "job_progress",
                            "tag": tag,
                            "on_progress": on_progress,
                            "payload": p,
                        })

                    result = fn(on_progress=progress_cb)
                else:
                    result = fn()
                self._job_queue.put({
                    "kind": "job_done",
                    "tag": tag,
                    "result": result,
                    "refresh": refresh_on_done,
                    "on_done": on_done,
                })
            except Exception as exc:
                self._job_queue.put({"kind": "job_error", "tag": tag, "message": str(exc)})

        threading.Thread(target=worker, daemon=True, name=f"registry-{tag}").start()

    def _handle_job_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("kind")
        if kind == "summary_done":
            if msg.get("gen") != self._summary_gen:
                return
            data = msg.get("data") or {}
            self._set_text(self._summary_text, fmt.format_summary(data))
            return
        if kind == "summary_error":
            if msg.get("gen") != self._summary_gen:
                return
            self._set_text(self._summary_text, f"Error loading summary:\n{msg.get('message')}")
            return
        if kind == "golden_done":
            self._set_busy(False)
            result = msg.get("result") or {}
            text = fmt.format_golden_status(result) if isinstance(result, dict) else str(result)
            self._set_text(self._golden_text, text)
            self._load_golden_status()
            self._status_var.set("Golden regression finished")
            return
        if kind == "error" and msg.get("target") == "golden":
            self._set_busy(False)
            self._set_text(self._golden_text, f"Error:\n{msg.get('message')}")
            return
        if kind == "validate_progress":
            body = msg.get("body")
            if body and body.winfo_exists():
                self._append_text(body, str(msg.get("line")))
            return
        if kind == "validate_done":
            body = msg.get("body")
            start_btn = msg.get("start_btn")
            if start_btn:
                start_btn.configure(state="normal")
            report = msg.get("report") or {}
            if body and body.winfo_exists():
                self._set_text(body, fmt.format_validation_report(report))
            self.refresh_registry()
            return
        if kind == "validate_error":
            body = msg.get("body")
            start_btn = msg.get("start_btn")
            if start_btn:
                start_btn.configure(state="normal")
            if body and body.winfo_exists():
                self._set_text(body, f"Error:\n{msg.get('message')}")
            return
        if kind == "job_progress":
            cb = msg.get("on_progress")
            payload = msg.get("payload") or {}
            if callable(cb):
                cb(payload)
            return
        if kind == "job_done":
            self._set_busy(False)
            on_done = msg.get("on_done")
            if callable(on_done):
                on_done(msg.get("result"))
            if msg.get("refresh"):
                self.refresh_registry()
            self._status_var.set(f"Finished {msg.get('tag')}")
            return
        if kind == "job_error":
            self._set_busy(False)
            messagebox.showerror("Dataset Registry", f"{msg.get('tag')}: {msg.get('message')}")
            self._status_var.set("Job failed")
