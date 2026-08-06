"""Feature migration UI — add registry columns to an existing master SQLite DB."""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_migration_engine import run_feature_migration_job

from .build_progress_manager import get_build_progress_manager


def _can_resume(progress: dict[str, Any] | None) -> bool:
    if not isinstance(progress, dict):
        return False
    status = str(progress.get("status") or "")
    if status not in ("computing", "preparing", "validated", "validation_failed"):
        return False
    pending = progress.get("pending_days") or []
    return bool(pending) or status in ("computing", "preparing")


class FeatureMigrationPanel(ttk.Frame):
    """Analyze / migrate / validate / commit missing registry features."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        resolve_db_path: Callable[[], str],
        db_exists: Callable[[], bool],
        on_migrated: Callable[[], None] | None = None,
        is_external_busy: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(master, padding=6)
        self._resolve_db_path = resolve_db_path
        self._db_exists = db_exists
        self._on_migrated = on_migrated
        self._is_external_busy = is_external_busy or (lambda: False)

        self._analysis: dict[str, Any] | None = None
        self._progress: dict[str, Any] | None = None
        self._validation: dict[str, Any] | None = None
        self._selected: set[str] = set()
        self._running = False
        self._worker_gen = 0
        self._analyzed_db_path: str | None = None
        self._analyze_stale = True
        self._last_ui_progress_at = 0.0

        self._status_var = tk.StringVar(value="Open this tab and click Analyze (or it runs once when selected)")
        self._current_var = tk.StringVar(value="—")
        self._registry_var = tk.StringVar(value="—")
        self._missing_var = tk.StringVar(value="—")
        self._selected_var = tk.StringVar(value="0")
        self._resume_var = tk.StringVar(value="")

        self._build_ui()

    @property
    def running(self) -> bool:
        return self._running

    def _build_ui(self) -> None:
        stats = ttk.Frame(self)
        stats.pack(fill="x", pady=(0, 6))
        for col, (label, var) in enumerate((
            ("Current", self._current_var),
            ("Registry", self._registry_var),
            ("Missing", self._missing_var),
            ("Selected", self._selected_var),
        )):
            cell = ttk.Frame(stats, padding=2)
            cell.grid(row=0, column=col, sticky="w", padx=(0, 10))
            ttk.Label(cell, text=label, foreground="#888").pack(anchor="w")
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(0, 4))
        self._btn_analyze = ttk.Button(actions, text="Analyze", command=self._analyze)
        self._btn_analyze.pack(side="left", padx=(0, 4))
        self._btn_continue = ttk.Button(actions, text="Continue", command=lambda: self._start(resume=True))
        self._btn_continue.pack(side="left", padx=(0, 4))
        self._btn_start = ttk.Button(actions, text="Migrate", command=lambda: self._start(resume=False))
        self._btn_start.pack(side="left", padx=(0, 4))
        self._btn_validate = ttk.Button(actions, text="Validate", command=self._validate)
        self._btn_validate.pack(side="left", padx=(0, 4))
        self._btn_commit = ttk.Button(actions, text="Commit", command=self._commit)
        self._btn_commit.pack(side="left", padx=(0, 4))
        self._btn_rollback = ttk.Button(actions, text="Rollback", command=self._rollback)
        self._btn_rollback.pack(side="left")

        ttk.Label(self, textvariable=self._resume_var, foreground="#C68A00", wraplength=320).pack(anchor="w", pady=(0, 4))

        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, pady=(0, 4))
        cols = ("sel", "feature", "group", "status")
        self._tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=7)
        for c, w, txt in (
            ("sel", 28, "✓"),
            ("feature", 150, "Feature"),
            ("group", 80, "Group"),
            ("status", 110, "Status"),
        ):
            self._tree.heading(c, text=txt)
            self._tree.column(c, width=w, anchor="center" if c == "sel" else "w")
        self._tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        sb.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.bind("<Button-1>", self._on_tree_click)

        sel_row = ttk.Frame(self)
        sel_row.pack(fill="x", pady=(0, 4))
        ttk.Button(sel_row, text="Select all", command=self._select_all).pack(side="left", padx=(0, 4))
        ttk.Button(sel_row, text="Clear", command=self._clear_selection).pack(side="left")

        ttk.Label(self, text="Progress", foreground="#888").pack(anchor="w")
        self._progress_text = scrolledtext.ScrolledText(self, height=5, font=("Consolas", 8), wrap="word")
        self._progress_text.pack(fill="x", pady=(0, 4))
        self._progress_text.configure(state="disabled")

        ttk.Label(self, textvariable=self._status_var, foreground="#666", wraplength=320).pack(anchor="w")
        self._update_buttons()

    def refresh(self) -> None:
        """
        React to master DB / market / interval change — do NOT re-analyze.

        Full analyze is expensive on large SQLite masters (multi‑GB). Interval
        switches used to block the UI behind auto-analyze; now we only mark
        stale and clear. Analyze runs when the user opens this tab or clicks Analyze.
        """
        # Cancel any in-flight silent analyze for the previous DB.
        self._worker_gen += 1
        self._running = False
        self._analyze_stale = True
        self._analyzed_db_path = None
        self._analysis = None
        self._progress = None
        self._validation = None
        self._selected.clear()
        self._current_var.set("—")
        self._registry_var.set("—")
        self._missing_var.set("—")
        self._selected_var.set("0")
        self._resume_var.set("")
        self._progress_text.configure(state="normal")
        self._progress_text.delete("1.0", "end")
        self._progress_text.configure(state="disabled")
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        if not self._db_exists():
            self._status_var.set("No master DB for this market/interval — add chain days first")
        else:
            self._status_var.set(
                "Master/interval changed — open this tab to Analyze (not auto-run on switch)"
            )
        self._update_buttons()

    def on_tab_selected(self) -> None:
        """Lazy analyze only when Feature Migration tab is shown and data is stale."""
        if self._running:
            return
        if not self._db_exists():
            self.refresh()
            return
        path = self._db_path()
        if self._analyze_stale or self._analyzed_db_path != path or self._analysis is None:
            self._analyze(silent=True)

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        for btn in (
            self._btn_analyze,
            self._btn_continue,
            self._btn_start,
            self._btn_validate,
            self._btn_commit,
            self._btn_rollback,
        ):
            btn.configure(state=state)
        self._update_buttons()

    def _db_path(self) -> str:
        return self._resolve_db_path()

    def _run_bg(
        self,
        work: Callable[[], Any],
        *,
        on_ok: Callable[[Any], None],
        on_err: Callable[[Exception], None] | None = None,
        silent: bool = False,
    ) -> None:
        if self._running or self._is_external_busy():
            if not silent:
                messagebox.showwarning("Feature Migration", "Another operation is already running.")
            return
        self._worker_gen += 1
        generation = self._worker_gen
        self._set_running(True)

        def worker() -> None:
            err: Exception | None = None
            result: Any = None
            try:
                result = work()
            except Exception as exc:
                err = exc

            def finish() -> None:
                if generation != self._worker_gen:
                    return
                self._set_running(False)
                if err is not None:
                    if on_err:
                        on_err(err)
                    else:
                        self._status_var.set(str(err))
                        if not silent:
                            messagebox.showerror("Feature Migration", str(err))
                    return
                on_ok(result)

            try:
                self.after(0, finish)
            except tk.TclError:
                pass

        threading.Thread(target=worker, name="feature-migration-ui", daemon=True).start()

    def _analyze(self, *, silent: bool = False) -> None:
        if not self._db_exists():
            self._status_var.set("No master DB — add chain days first")
            return

        path = self._db_path()

        def work() -> dict[str, Any]:
            return run_feature_migration_job(path, "analyze")

        def apply(data: dict[str, Any]) -> None:
            self._apply_analysis(data)
            self._analyzed_db_path = path
            self._analyze_stale = False
            # Status text is set inside _apply_analysis from missing_by_reason.

        self._run_bg(work, on_ok=apply, silent=silent)

    def _apply_analysis(self, data: dict[str, Any]) -> None:
        self._analysis = data
        self._progress = data.get("migration_progress") if isinstance(data.get("migration_progress"), dict) else None
        if self._progress and self._progress.get("validation"):
            self._validation = self._progress.get("validation")
        prog_feats = list((self._progress or {}).get("features") or [])
        missing = [
            f["name"]
            for f in (data.get("missing_features") or [])
            if isinstance(f, dict) and f.get("name")
        ]
        missing_set = set(missing)
        if prog_feats and _can_resume(self._progress):
            # Keep interrupted job selection, but drop features no longer missing.
            self._selected = {f for f in prog_feats if f in missing_set} or set(missing)
        else:
            # Always align selection to current missing list (avoids stale 49-feature picks).
            self._selected = set(missing)

        self._current_var.set(str(data.get("current_feature_count") or "—"))
        self._registry_var.set(str(data.get("registry_feature_count") or "—"))
        self._missing_var.set(str(data.get("missing_count") or "—"))
        self._selected_var.set(str(len(self._selected)))
        self._render_feature_list(data)
        self._render_progress(self._progress)
        self._update_resume_hint()
        self._update_buttons()
        by_reason = data.get("missing_by_reason") or {}
        n_miss = int(data.get("missing_count") or 0)
        if n_miss <= 0:
            self._status_var.set("Analysis ready — master schema matches Feature Registry")
        else:
            parts = []
            if by_reason.get("missing_column"):
                parts.append(f"{by_reason['missing_column']} need ADD COLUMN")
            if by_reason.get("not_in_schema"):
                parts.append(f"{by_reason['not_in_schema']} in DB but not in build_schema")
            if by_reason.get("unpopulated"):
                parts.append(f"{by_reason['unpopulated']} unpopulated")
            detail = "; ".join(parts) if parts else f"{n_miss} missing"
            self._status_var.set(f"Analysis ready — {detail}. Select features and Migrate")

    def _render_feature_list(self, data: dict[str, Any]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        reason_label = {
            "missing_column": "needs column",
            "not_in_schema": "schema lag",
            "unpopulated": "unpopulated",
        }
        for row in data.get("missing_features") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if not name:
                continue
            sel = "☑" if name in self._selected else "☐"
            status = reason_label.get(str(row.get("reason") or ""), str(row.get("reason") or "—"))
            self._tree.insert(
                "",
                "end",
                iid=name,
                values=(sel, name, row.get("group") or "—", status),
            )

    def _on_tree_click(self, event: tk.Event) -> None:
        if self._tree.identify_column(event.x) != "#1":
            return
        iid = self._tree.identify_row(event.y)
        if not iid:
            return
        if iid in self._selected:
            self._selected.discard(iid)
        else:
            self._selected.add(iid)
        self._selected_var.set(str(len(self._selected)))
        sel = "☑" if iid in self._selected else "☐"
        vals = list(self._tree.item(iid, "values"))
        if vals:
            vals[0] = sel
            self._tree.item(iid, values=vals)
        self._update_buttons()

    def _select_all(self) -> None:
        for iid in self._tree.get_children():
            self._selected.add(str(iid))
            vals = list(self._tree.item(iid, "values"))
            if vals:
                vals[0] = "☑"
                self._tree.item(iid, values=vals)
        self._selected_var.set(str(len(self._selected)))
        self._update_buttons()

    def _clear_selection(self) -> None:
        self._selected.clear()
        for iid in self._tree.get_children():
            vals = list(self._tree.item(iid, "values"))
            if vals:
                vals[0] = "☐"
                self._tree.item(iid, values=vals)
        self._selected_var.set("0")
        self._update_buttons()

    def _start(self, *, resume: bool) -> None:
        if not self._db_exists():
            messagebox.showwarning("Feature Migration", "No master DB for this market/interval.")
            return
        features = sorted(self._selected)
        if resume:
            features = list((self._progress or {}).get("features") or features)
            if not features:
                messagebox.showwarning("Feature Migration", "Nothing to continue — run Analyze first.")
                return
        else:
            # Prefer only currently missing features from last Analyze.
            missing = {
                str(f.get("name") or "")
                for f in ((self._analysis or {}).get("missing_features") or [])
                if isinstance(f, dict)
            }
            missing.discard("")
            if missing:
                features = sorted(f for f in features if f in missing)
            if not features:
                messagebox.showwarning(
                    "Feature Migration",
                    "No missing features selected.\n\n"
                    "Click Analyze, then select features with status "
                    "\"schema lag\" / \"needs column\", then Migrate.",
                )
                return
        if _can_resume(self._progress) and not resume:
            if not messagebox.askyesno(
                "Feature Migration",
                "A partial migration exists. Restart from scratch?\n\n"
                "Choose No and use Continue to keep progress.",
            ):
                return

        def work() -> dict[str, Any]:
            mgr = get_build_progress_manager()
            mgr.begin_job(
                "feature_migration",
                title="Feature Migration",
                cancel_fn=None,
            )
            mgr.publish({
                "status": "running",
                "job_kind": "feature_migration",
                "job_title": "Feature Migration",
                "stage_name": "Prepare",
                "message": "Starting migration…",
                "percent": 1.0,
                "elapsed_sec": 0.0,
                "pipeline": {"total_elapsed_sec": 0.0, "total_elapsed_label": "00:00"},
            })

            out = run_feature_migration_job(
                self._db_path(),
                "start",
                features=features,
                resume=resume,
            )
            pending = list((out.get("progress") or {}).get("pending_days") or out.get("pending_days") or [])
            total_days = len(pending) + len(
                (out.get("progress") or {}).get("completed_days") or []
            )
            self._publish_migration_progress(
                out.get("progress") if isinstance(out.get("progress"), dict) else {},
                force=True,
            )

            def on_compute_progress(prog: dict[str, Any]) -> None:
                self._publish_migration_progress(prog)

            while pending:
                day = pending[0]
                out = run_feature_migration_job(
                    self._db_path(),
                    "compute",
                    trading_day=day,
                    on_progress=on_compute_progress,
                )
                prog = out.get("progress") if isinstance(out.get("progress"), dict) else {}
                pending = list(prog.get("pending_days") or [])
                self._publish_migration_progress(prog, force=True)

            mgr.publish({
                "status": "running",
                "job_kind": "feature_migration",
                "stage_name": "Validate",
                "message": "Validating migration…",
                "percent": 97.0,
            })
            val = run_feature_migration_job(self._db_path(), "validate")
            return {
                "start": out,
                "validation": val.get("validation"),
                "analyze": run_feature_migration_job(self._db_path(), "analyze"),
                "total_days": total_days,
            }

        def apply(bundle: dict[str, Any]) -> None:
            self._validation = bundle.get("validation") if isinstance(bundle.get("validation"), dict) else None
            analysis = bundle.get("analyze")
            if isinstance(analysis, dict):
                self._apply_analysis(analysis)
            passed = bool(self._validation and self._validation.get("passed"))
            self._status_var.set(
                "Validation passed — ready to Commit" if passed else "Compute finished — review validation"
            )
            get_build_progress_manager().publish({
                "status": "completed" if passed else "failed",
                "_done": True,
                "job_kind": "feature_migration",
                "message": (
                    "Migration validated — click Commit"
                    if passed
                    else "Migration validation failed"
                ),
                "percent": 100.0 if passed else 0.0,
            })
            if passed:
                messagebox.showinfo("Feature Migration", "Validation passed. Click Commit to merge into master DB.")

        def on_err(exc: Exception) -> None:
            self._status_var.set(str(exc))
            get_build_progress_manager().publish({
                "status": "failed",
                "_done": True,
                "job_kind": "feature_migration",
                "message": str(exc),
            })
            try:
                st = run_feature_migration_job(self._db_path(), "status")
                if isinstance(st.get("progress"), dict):
                    self._progress = st["progress"]
                    self._render_progress(self._progress)
            except Exception:
                pass
            messagebox.showerror("Feature Migration", str(exc))

        self._status_var.set("Migrating features (backup → compute → validate)…")
        self._run_bg(work, on_ok=apply, on_err=on_err)

    def _publish_migration_progress(
        self,
        progress: dict[str, Any] | None,
        *,
        force: bool = False,
    ) -> None:
        """Update panel Progress box + global status bar (throttled from worker thread)."""
        if not isinstance(progress, dict):
            return
        now = time.monotonic()
        if not force and (now - self._last_ui_progress_at) < 0.35:
            return
        self._last_ui_progress_at = now
        populate = progress.get("populate") if isinstance(progress.get("populate"), dict) else {}
        completed = list(progress.get("completed_days") or [])
        pending = list(progress.get("pending_days") or [])
        days_done = int(populate.get("days_done") if populate.get("days_done") is not None else len(completed))
        days_total = int(
            populate.get("days_total")
            if populate.get("days_total") is not None
            else (days_done + len(pending))
        )
        day_pct = float(populate.get("day_pct") or 0.0)
        overall_pct = float(populate.get("pct") or 0.0)
        day = str(populate.get("trading_day") or progress.get("trading_day") or "—")
        day_done = int(populate.get("day_rows_done") or 0)
        day_total = int(populate.get("day_rows_total") or 0)
        rows_done = int(populate.get("rows_done") or 0)
        rows_total = int(populate.get("rows_total") or 0)
        elapsed = float(populate.get("elapsed_sec") or 0.0)
        msg = (
            f"Days {days_done}/{days_total} ({overall_pct:.0f}%) · "
            f"{day} {day_pct:.0f}%"
        )
        if day_total > 0:
            msg += f" ({day_done:,}/{day_total:,})"

        # Thread-safe publish; UI widgets updated on main thread.
        get_build_progress_manager().publish({
            "status": "running",
            "job_kind": "feature_migration",
            "job_title": "Feature Migration",
            "stage_name": msg,
            "message": msg,
            "trading_day": day,
            "source_day_index": days_done + (1 if pending else 0),
            "source_day_total": max(days_total, 1),
            "rows": day_done if day_total else rows_done,
            "total": day_total if day_total else rows_total,
            "percent": overall_pct,
            "elapsed_sec": elapsed,
            "pipeline": {
                "total_elapsed_sec": elapsed,
                "total_elapsed_label": f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}",
            },
        })

        snapshot = dict(progress)

        def _ui() -> None:
            self._progress = snapshot
            self._render_progress(snapshot)
            self._status_var.set(msg)

        try:
            self.after(0, _ui)
        except tk.TclError:
            pass

    def _validate(self) -> None:
        def work() -> dict[str, Any]:
            val = run_feature_migration_job(self._db_path(), "validate")
            return {"validation": val.get("validation"), "analyze": run_feature_migration_job(self._db_path(), "analyze")}

        def apply(bundle: dict[str, Any]) -> None:
            self._validation = bundle.get("validation") if isinstance(bundle.get("validation"), dict) else None
            analysis = bundle.get("analyze")
            if isinstance(analysis, dict):
                self._apply_analysis(analysis)
            passed = bool(self._validation and self._validation.get("passed"))
            self._status_var.set("Validation passed" if passed else "Validation failed")

        self._run_bg(work, on_ok=apply)

    def _commit(self) -> None:
        if not self._validation or not self._validation.get("passed"):
            messagebox.showwarning("Feature Migration", "Run Validate first (must pass).")
            return
        if not messagebox.askyesno(
            "Commit migration",
            "Merge validated features into master samples?\n\nThis updates the master DB.",
        ):
            return

        def work() -> dict[str, Any]:
            return run_feature_migration_job(self._db_path(), "commit")

        def apply(data: dict[str, Any]) -> None:
            merged = data.get("features_merged") or []
            self._status_var.set(f"Committed {len(merged)} feature(s)")
            self._validation = None
            self._progress = None
            messagebox.showinfo("Feature Migration", f"Committed {len(merged)} feature(s) to master DB.")
            if self._on_migrated:
                self._on_migrated()
            self._analyze(silent=True)

        self._run_bg(work, on_ok=apply)

    def _rollback(self) -> None:
        if not messagebox.askyesno("Rollback", "Discard temp migration and restore from backup if needed?"):
            return

        def work() -> dict[str, Any]:
            return run_feature_migration_job(self._db_path(), "rollback")

        def apply(_data: dict[str, Any]) -> None:
            self._validation = None
            self._progress = None
            self._status_var.set("Migration rolled back")
            self._analyze(silent=True)

        self._run_bg(work, on_ok=apply)

    def _render_progress(self, progress: dict[str, Any] | None) -> None:
        self._progress = progress
        lines: list[str] = []
        if isinstance(progress, dict):
            for step in progress.get("timeline") or []:
                if not isinstance(step, dict):
                    continue
                mark = {"done": "✓", "running": "…", "error": "✗", "pending": "○"}.get(
                    str(step.get("status") or ""), "·"
                )
                detail = step.get("detail") or ""
                lines.append(f"{mark} {step.get('label') or step.get('id')}: {detail}".rstrip())
            populate = progress.get("populate")
            if isinstance(populate, dict):
                days_done = populate.get("days_done")
                days_total = populate.get("days_total")
                day_pct = populate.get("day_pct")
                overall = populate.get("pct")
                day = populate.get("trading_day") or "—"
                if days_total is not None:
                    lines.append(
                        f"Days {days_done or 0}/{days_total} · overall {overall if overall is not None else '—'}%"
                    )
                lines.append(
                    f"Current day {day}: {day_pct if day_pct is not None else '—'}% "
                    f"({int(populate.get('day_rows_done') or 0):,}/"
                    f"{int(populate.get('day_rows_total') or 0):,} rows)"
                )
                if populate.get("rows_pct") is not None:
                    lines.append(
                        f"Rows overall {populate.get('rows_pct')}% · "
                        f"ETA {populate.get('eta_sec') if populate.get('eta_sec') is not None else '—'}s"
                    )
        text = "\n".join(lines) if lines else "—"
        self._progress_text.configure(state="normal")
        self._progress_text.delete("1.0", tk.END)
        self._progress_text.insert("1.0", text)
        self._progress_text.configure(state="disabled")

    def _update_resume_hint(self) -> None:
        prog = self._progress
        if _can_resume(prog):
            pending = len((prog or {}).get("pending_days") or [])
            done = len((prog or {}).get("completed_days") or [])
            self._resume_var.set(f"Interrupted job · {done} day(s) done · {pending} remaining — use Continue")
        else:
            self._resume_var.set("")

    def _update_buttons(self) -> None:
        if self._running:
            return
        exists = self._db_exists()
        analysis = self._analysis or {}
        temp = bool(analysis.get("temp_table_exists"))
        status = str((self._progress or {}).get("status") or analysis.get("migration_status") or "idle")
        validated = status == "validated" or bool(self._validation and self._validation.get("passed"))
        can_resume = _can_resume(self._progress)

        self._btn_analyze.configure(state="normal" if exists else "disabled")
        self._btn_continue.configure(state="normal" if exists and can_resume else "disabled")
        self._btn_start.configure(state="normal" if exists and self._selected else "disabled")
        self._btn_validate.configure(state="normal" if exists and temp else "disabled")
        self._btn_commit.configure(state="normal" if exists and validated else "disabled")
        self._btn_rollback.configure(state="normal" if exists and (temp or status not in ("idle", "")) else "disabled")
