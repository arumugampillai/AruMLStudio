"""Create Dataset page — insert chain days into master SQLite (standalone)."""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .build_config_prefs import load_chain_source_prefs, save_chain_source_prefs
from .build_service import MasterBuildRunner, MasterDebugFeatureRunner, chart_data_dir
from .build_progress_manager import get_build_progress_manager
from .config_panel import BuildConfigPanel
from .inventory import filter_inventory_rows, load_chain_inventory_rows
from .preview_util import (
    day_in_master,
    day_row_count,
    estimate_rows_for_sources,
    read_master_status,
)
from .lazy_panel import LazyLoadMixin
from .progress_panel import MasterBuildProgressPanel


def master_db_basename(market: str, interval_sec: int) -> str:
    from chain_replay_ml.dataset_builder.master_naming import master_db_filename

    return master_db_filename(market=market, sampling_interval_sec=interval_sec)


class CreateDatasetPanel(ttk.Frame, LazyLoadMixin):
    """Port chain-source days into master DB — mirrors Create Dataset master-only build."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_data_changed: Callable[[], None] | None = None,
        on_inventory_loaded: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_data_changed = on_data_changed
        self._on_inventory_loaded = on_inventory_loaded
        self.inventory_rows: list[dict[str, Any]] = []
        self.selected_ids: set[str] = set()
        self.progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._runner = MasterBuildRunner(self.chart_dir)
        self._debug_feature_runner = MasterDebugFeatureRunner(self.chart_dir)
        self._market_var = tk.StringVar(value="NIFTY")
        self._master_status: dict[str, Any] | None = None
        self._adding_row_id: str | None = None
        self._debug_panel_win: tk.Toplevel | None = None
        self._last_live_tick_at = 0.0
        self._build_ui_active = False
        self._build_ui()
        self._lazy_init()

    @property
    def build_running(self) -> bool:
        return self._runner.running or self._debug_feature_runner.running

    @property
    def needs_progress_poll(self) -> bool:
        """True while a build is running or awaiting final queue drain."""
        return self.build_running or self._build_ui_active

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        if hasattr(self, "config_panel"):
            self.config_panel.set_chart_dir(chart_dir)
        if hasattr(self, "_runner") and hasattr(self._runner, "chart_dir"):
            self._runner.chart_dir = chart_dir
        if hasattr(self, "_debug_feature_runner"):
            self._debug_feature_runner.chart_dir = chart_dir
        if self._debug_panel_win is not None and self._debug_panel_win.winfo_exists():
            panel = getattr(self._debug_panel_win, "_debug_panel", None)
            if panel is not None:
                panel.set_chart_dir(chart_dir)
        if self.inventory_rows:
            self._restore_chain_source_selection()
            self._refresh_master_status()
            self._refresh_table()
            self._update_preview()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Market").pack(side="left")
        market_cb = ttk.Combobox(
            top,
            textvariable=self._market_var,
            values=["NIFTY", "BANKNIFTY", "SENSEX", "BOTH"],
            width=12,
            state="readonly",
        )
        market_cb.pack(side="left", padx=(4, 12))
        market_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_market_changed())

        ttk.Button(top, text="Reload inventory", command=lambda: self.load_inventory(lazy=True, force=True)).pack(side="left", padx=4)
        ttk.Button(top, text="Open master DB folder", command=self.open_master_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Open Debug", command=self._open_debug_panel).pack(side="left", padx=4)

        self.build_btn = ttk.Button(top, text="Build selected", command=self._start_build_selected)
        self.build_btn.pack(side="right", padx=4)
        self.debug_build_btn = ttk.Button(top, text="Debug Build", command=self._start_debug_build)
        self.debug_build_btn.pack(side="right", padx=4)
        self.cancel_btn = ttk.Button(top, text="Cancel", command=self._cancel_build, state="disabled")
        self.cancel_btn.pack(side="right")

        hint = (
            "Master SQLite insert uses your strike selection (ATM band, gap, targets). "
            "Use per-row Add or bulk insert."
        )
        ttk.Label(self, text=hint, wraplength=900, foreground="#666").pack(anchor="w", padx=10)

        path_row = ttk.Frame(self, padding=(10, 2))
        path_row.pack(fill="x")
        self.master_path_var = tk.StringVar(value="—")
        ttk.Label(path_row, textvariable=self.master_path_var, foreground="#58a6ff").pack(side="left")
        self.preview_var = tk.StringVar(value="")
        ttk.Label(path_row, textvariable=self.preview_var, foreground="#aaa").pack(side="right", padx=8)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)
        self._config_paned = paned
        self._config_paned_split_done = False
        paned.bind("<Configure>", self._apply_config_paned_split, add="+")

        self.config_panel = BuildConfigPanel(
            paned,
            chart_dir=self.chart_dir,
            on_change=self._on_config_changed,
            on_build=self._start_build_selected,
        )
        paned.add(self.config_panel, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        table_frame = ttk.LabelFrame(right, text="Chain sources", padding=4)
        table_frame.pack(fill="both", expand=True)

        cols = ("sel", "date", "market", "expiry", "ticks", "spot", "master", "action")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        for c, w, txt in (
            ("sel", 32, "✓"),
            ("date", 96, "Date"),
            ("market", 72, "Market"),
            ("expiry", 88, "Expiry"),
            ("ticks", 80, "Ticks"),
            ("spot", 44, "Spot"),
            ("master", 130, "Master Dataset"),
            ("action", 56, ""),
        ):
            self.tree.heading(c, text=txt)
            anchor = "w" if c in ("expiry", "master") else "center"
            self.tree.column(c, width=w, anchor=anchor)
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Button-1>", self._on_tree_click)

        self.progress = MasterBuildProgressPanel(self)
        self.progress.pack(fill="both", expand=True, padx=8, pady=4)

    def _apply_config_paned_split(self, _event: tk.Event | None = None) -> None:
        if self._config_paned_split_done:
            return
        paned = getattr(self, "_config_paned", None)
        if paned is None:
            return
        width = paned.winfo_width()
        if width <= 1:
            return
        # ~44% width for build config (25% wider than prior 35% split).
        paned.sashpos(0, max(275, int(width * 0.44)))
        self._config_paned_split_done = True

    def on_show(self) -> None:
        if hasattr(self, "config_panel"):
            self.config_panel.refresh_registry_exclusions()
        self.load_inventory(lazy=True, force=False)

    def load_inventory(self, *, lazy: bool = False, force: bool = False) -> str | None:
        if lazy:
            def apply_bundle(bundle: dict[str, Any]) -> None:
                self._apply_inventory(bundle)
                if not force:
                    self._schedule_inventory_refresh()

            self.lazy_load(
                load=lambda: self._fetch_inventory(force=force),
                apply=apply_bundle,
                message="Refreshing inventory…" if force else "Loading chain sources…",
                status_var=self.preview_var,
                show_overlay=force,
            )
            return None
        try:
            bundle = self._fetch_inventory(force=force)
        except Exception as exc:
            messagebox.showerror("Inventory", f"Could not load inventory:\n{exc}")
            return None
        return self._apply_inventory(bundle)

    def _bundle_from_cache(self) -> dict[str, Any]:
        rows, meta = load_chain_inventory_rows(self.chart_dir)
        market = self._market_var.get()
        master_status = None
        if market != "BOTH":
            master_status = read_master_status(
                self.chart_dir,
                market=market,
                interval_sec=self._interval_sec(),
            )
        return {"rows": rows, "meta": meta, "master_status": master_status}

    def _fetch_inventory(self, *, force: bool = False) -> dict[str, Any]:
        from .market_db_service import load_inventory, load_inventory_disk_only

        if force:
            load_inventory(self.chart_dir, force=True)
        else:
            if not load_inventory_disk_only(self.chart_dir):
                load_inventory(self.chart_dir, force=False)
        return self._bundle_from_cache()

    def _schedule_inventory_refresh(self) -> None:
        """Rescan tick DBs in background after showing cached inventory."""
        generation = getattr(self, "_lazy_generation", 0)

        def worker() -> None:
            err: Exception | None = None
            try:
                from .market_db_service import load_inventory

                load_inventory(self.chart_dir, force=False)
            except Exception as exc:
                err = exc

            def finish() -> None:
                if generation != getattr(self, "_lazy_generation", 0):
                    return
                if err is not None:
                    self.preview_var.set(f"Inventory refresh failed: {err}")
                    return
                try:
                    updated = self._apply_inventory(self._bundle_from_cache())
                except Exception as exc:
                    self.preview_var.set(f"Inventory refresh failed: {exc}")
                    return
                if updated:
                    self.preview_var.set(f"Inventory updated · {updated}")

            try:
                self.after(0, finish)
            except tk.TclError:
                pass

        threading.Thread(
            target=worker,
            name="create-dataset-inventory-refresh",
            daemon=True,
        ).start()

    def _apply_inventory(self, bundle: dict[str, Any]) -> str | None:
        self.inventory_rows = list(bundle.get("rows") or [])
        self._master_status = bundle.get("master_status")
        self._restore_chain_source_selection()
        if self._master_status is None and self._market_var.get() != "BOTH":
            self._refresh_master_status()
        selectable = self._selectable_source_ids()
        self.selected_ids &= selectable
        self._refresh_table()
        self._update_preview()
        self._persist_chain_source_selection()
        if hasattr(self, "config_panel"):
            self.config_panel.refresh_registry_exclusions()
        self._on_config_changed()
        meta = bundle.get("meta") or {}
        updated = str(meta.get("last_updated") or "unknown")
        if self._on_inventory_loaded is not None:
            self._on_inventory_loaded(updated)
        return updated

    def _interval_sec(self) -> int:
        return self.config_panel.interval_sec()

    def _refresh_master_status(self) -> None:
        market = self._market_var.get()
        if market == "BOTH":
            self._master_status = None
            return
        self._master_status = read_master_status(
            self.chart_dir,
            market=market,
            interval_sec=self._interval_sec(),
        )

    def _on_market_changed(self) -> None:
        self.selected_ids &= self._selectable_source_ids()
        self._refresh_master_status()
        self._refresh_table()
        self._update_preview()
        self._persist_chain_source_selection()

    def _visible_inventory_rows(self) -> list[dict[str, Any]]:
        rows = filter_inventory_rows(self.inventory_rows, self._market_var.get())
        # Hide thin chain days — only show sources with more than 10L ticks.
        return [
            r for r in rows
            if int(r.get("chain_ticks") or 0) > 1_000_000
        ]

    def _selectable_source_ids(self) -> set[str]:
        return {
            str(r["source_id"])
            for r in self._visible_inventory_rows()
            if r.get("spot_available") is not False
        }

    def _restore_chain_source_selection(self) -> None:
        prefs = load_chain_source_prefs(self.chart_dir)
        market = str(prefs.get("market") or "").strip().upper()
        if market in ("NIFTY", "BANKNIFTY", "SENSEX", "BOTH"):
            self._market_var.set(market)
        selectable = self._selectable_source_ids()
        saved = prefs.get("selected_source_ids")
        if isinstance(saved, list):
            restored = {str(s) for s in saved if str(s) in selectable}
            if restored:
                self.selected_ids = restored
                return
        if not self.selected_ids:
            self.selected_ids = set(selectable)

    def _persist_chain_source_selection(self) -> None:
        if not self.chart_dir:
            return
        save_chain_source_prefs(
            self.chart_dir,
            market=self._market_var.get(),
            selected_source_ids=sorted(self.selected_ids),
        )

    def _on_config_changed(self) -> None:
        if not hasattr(self, "tree"):
            return
        self._refresh_master_status()
        self._update_preview()
        self._refresh_table()

    def _update_master_path_hint(self) -> None:
        market = self._market_var.get()
        if market == "BOTH":
            self.master_path_var.set("Select a single market (not BOTH)")
            return
        from chain_replay_ml.dataset_builder.master_naming import resolve_master_db_path

        abs_path = resolve_master_db_path(
            chart_data_dir(self.chart_dir),
            market=market,
            sampling_interval_sec=self._interval_sec(),
        )
        self.master_path_var.set(abs_path)

    def _update_preview(self) -> None:
        self._update_master_path_hint()
        sources = self._selected_sources()
        est = estimate_rows_for_sources(
            sources,
            interval_sec=self._interval_sec(),
            stride_sec=self.config_panel.sliding_stride_sec(),
            horizons_sec=self.config_panel.horizons_sec(),
            atm_band=self.config_panel.resolved_atm_band(),
        )
        self.config_panel.update_summary(estimated_rows=est if sources else None)
        if sources:
            self.preview_var.set(f"Selected: {len(sources)} day(s) · ~{est:,} new rows")
        else:
            st = self._master_status or {}
            total = int(st.get("row_count") or 0)
            days = len(st.get("days_in_master") or [])
            if total:
                self.preview_var.set(f"Master DB: {days} days · {total:,} rows")
            else:
                self.preview_var.set("No days selected")

    def _master_cell(self, row: dict[str, Any]) -> tuple[str, str]:
        td = row.get("trading_day") or ""
        if day_in_master(self._master_status, td):
            n = day_row_count(self._master_status, td)
            label = f"✓ In master ({n:,})" if n else "✓ In master"
            return label, "—"
        if self._adding_row_id == row.get("source_id"):
            return "Adding…", "—"
        if self.build_running:
            return "—", "—"
        return "—", "Add"

    def _refresh_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self._visible_inventory_rows():
            if row.get("spot_available") is False:
                continue
            sid = row["source_id"]
            tick = f"{int(row.get('chain_ticks') or 0):,}"
            sel = "☑" if sid in self.selected_ids else "☐"
            spot = "✓" if row.get("spot_available") else "—"
            master_lbl, action = self._master_cell(row)
            self.tree.insert(
                "",
                "end",
                iid=sid,
                values=(
                    sel, row.get("trading_day"), row.get("market"), row.get("expiry"),
                    tick, spot, master_lbl, action,
                ),
            )

    def _on_tree_click(self, event: tk.Event) -> None:
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        if col == "#1":
            if iid in self.selected_ids:
                self.selected_ids.discard(iid)
            else:
                self.selected_ids.add(iid)
            self._refresh_table()
            self._update_preview()
            self._persist_chain_source_selection()
            return
        if col == "#8":
            row = next((r for r in self.inventory_rows if r.get("source_id") == iid), None)
            if row:
                self._start_build_sources([row], single_row_id=iid)

    def _selected_sources(self) -> list[dict[str, Any]]:
        visible = {str(r["source_id"]) for r in self._visible_inventory_rows()}
        return self._sources_payload([
            r for r in self.inventory_rows
            if r.get("source_id") in self.selected_ids and str(r.get("source_id")) in visible
        ])

    @staticmethod
    def _sources_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            doc: dict[str, Any] = {
                "source_id": row["source_id"],
                "trading_day": row["trading_day"],
                "market": row["market"],
                "expiry": row["expiry"],
                "date": row.get("date") or row["trading_day"],
            }
            if row.get("db_path"):
                doc["db_path"] = row["db_path"]
            if row.get("db_file"):
                doc["db_file"] = row["db_file"]
            out.append(doc)
        return out

    def _open_debug_panel(self) -> None:
        if self._debug_panel_win is not None and self._debug_panel_win.winfo_exists():
            self._debug_panel_win.lift()
            self._debug_panel_win.focus_force()
            return
        from .debug_side_panel import open_debug_panel

        self._debug_panel_win = open_debug_panel(
            self.winfo_toplevel(),
            chart_dir=self.chart_dir,
            get_market=lambda: self._market_var.get(),
            get_interval_sec=self._interval_sec,
        )
        self._debug_panel_win.bind("<Destroy>", lambda _e: setattr(self, "_debug_panel_win", None))

    def _set_build_ui_active(self, active: bool) -> None:
        self._build_ui_active = active
        self.build_btn.configure(state="disabled" if active else "normal")
        self.debug_build_btn.configure(state="disabled" if active else "normal")
        self.cancel_btn.configure(state="normal" if active else "disabled")
        if hasattr(self, "config_panel"):
            self.config_panel.set_build_active(active)

    def _start_debug_build(self) -> None:
        """Load ticks + generate features in memory (no master DB write)."""
        if self.build_running:
            return
        market = self._market_var.get()
        if market == "BOTH":
            messagebox.showwarning("Market", "Select NIFTY or BANKNIFTY (not BOTH).")
            return
        sources = self._selected_sources()
        if not sources:
            messagebox.showwarning("Sources", "Select at least one chain source.")
            return
        features = self.config_panel.resolved_build_features()
        if not features:
            messagebox.showwarning("Features", "Select at least one feature domain / feature.")
            return
        self._set_build_ui_active(True)
        self.progress.reset()
        self._begin_build_job(title="Debug Build")

        def on_progress(payload: dict[str, Any]) -> None:
            self.progress_queue.put(payload)
            self._publish_build_progress(payload)

        def on_done(result: dict[str, Any]) -> None:
            done_payload = {**result, "_done": True}
            self.progress_queue.put(done_payload)
            self._publish_build_progress(done_payload)

        try:
            self._debug_feature_runner.start(
                sources=sources,
                interval_sec=self._interval_sec(),
                feature_selection=self.config_panel.feature_selection(),
                prediction_targets=self.config_panel.prediction_targets(),
                strike_selection=self.config_panel.strike_selection(),
                gap_policy=self.config_panel.gap_policy(),
                on_progress=on_progress,
                on_done=on_done,
            )
        except Exception as exc:
            self._set_build_ui_active(False)
            messagebox.showerror("Debug Build", str(exc))

    def _start_build_selected(self) -> None:
        sources = self._selected_sources()
        if not sources:
            messagebox.showwarning("Sources", "Select at least one chain day.")
            return
        self._start_build_sources(sources)

    def _start_build_sources(
        self,
        rows: list[dict[str, Any]],
        *,
        single_row_id: str | None = None,
    ) -> None:
        if self.build_running:
            return
        market = self._market_var.get()
        if market == "BOTH":
            messagebox.showwarning("Market", "Select NIFTY or BANKNIFTY (not BOTH).")
            return

        sources = self._sources_payload(rows)
        pending = [s for s in sources if not day_in_master(self._master_status, s["trading_day"])]
        if not pending:
            messagebox.showinfo("Master", "Selected day(s) are already in the master DB.")
            return

        est = estimate_rows_for_sources(
            pending,
            interval_sec=self._interval_sec(),
            stride_sec=self.config_panel.sliding_stride_sec(),
            horizons_sec=self.config_panel.horizons_sec(),
            atm_band=self.config_panel.resolved_atm_band(),
        )
        self._pending_build = {
            "pending": pending,
            "single_row_id": single_row_id,
            "estimated_rows": est,
        }
        features = self.config_panel.resolved_build_features()
        from .build_validation_panel import show_build_summary_dialog

        dlg = show_build_summary_dialog(
            self,
            feature_names=features,
            sampling_interval_sec=float(self._interval_sec()),
            sliding_stride_sec=float(self.config_panel.sliding_stride_sec()),
            strike_selection=self.config_panel.resolved_strike_selection(),
            gap_policy=self.config_panel.gap_policy(),
            prediction_targets=self.config_panel.prediction_targets(),
            estimated_rows=est,
            estimated_sessions=len(pending),
            on_proceed=self._confirm_and_run_build,
        )
        if dlg is None:
            self._confirm_and_run_build()

    def _confirm_and_run_build(self) -> None:
        ctx = getattr(self, "_pending_build", None) or {}
        pending = ctx.get("pending") or []
        single_row_id = ctx.get("single_row_id")
        est = int(ctx.get("estimated_rows") or 0)
        if not pending:
            return
        mode = "low-memory" if self.config_panel.low_memory() else "standard"
        strike_lbl = self.config_panel.resolved_strike_summary_label()
        gap_lbl = self.config_panel.resolved_gap_summary_label()
        if not messagebox.askyesno(
            "Confirm insert",
            f"Insert {len(pending)} day(s) into master DB?\n\n"
            f"~{est:,} rows · {strike_lbl}\n"
            f"Gap reset: {gap_lbl}\n"
            f"Mode: {mode}",
        ):
            return

        self._adding_row_id = single_row_id
        self._set_build_ui_active(True)
        self.progress.reset()
        self._begin_build_job(title="Building Dataset")

        def on_progress(payload: dict[str, Any]) -> None:
            self.progress_queue.put(payload)
            self._publish_build_progress(payload)

        def on_done(result: dict[str, Any]) -> None:
            done_payload = {**result, "_done": True}
            self.progress_queue.put(done_payload)
            self._publish_build_progress(done_payload)

        try:
            self._runner.start(
                sources=pending,
                interval_sec=self._interval_sec(),
                sampling=self.config_panel.sampling_config(),
                feature_selection=self.config_panel.feature_selection(),
                prediction_targets=self.config_panel.prediction_targets(),
                strike_selection=self.config_panel.strike_selection(),
                gap_policy=self.config_panel.gap_policy(),
                low_memory=self.config_panel.low_memory(),
                build_profiler=self.config_panel.build_profiler(),
                on_progress=on_progress,
                on_done=on_done,
            )
        except Exception as exc:
            self._adding_row_id = None
            self._set_build_ui_active(False)
            messagebox.showerror("Build", str(exc))
            return
        self.after(0, self._refresh_table)

    def open_master_folder(self) -> None:
        from .ui_util import open_path
        from chain_replay_ml.dataset_builder.master_naming import (
            resolve_master_datasets_dir,
            resolve_master_db_path,
        )

        market = self._market_var.get()
        data_dir = chart_data_dir(self.chart_dir)
        if market == "BOTH":
            folder = resolve_master_datasets_dir(data_dir)
        else:
            folder = os.path.dirname(
                resolve_master_db_path(
                    data_dir,
                    market=market,
                    sampling_interval_sec=self._interval_sec(),
                )
            )
        try:
            open_path(folder)
        except Exception as exc:
            messagebox.showerror("Open folder", str(exc))

    def _cancel_build(self) -> None:
        if messagebox.askyesno("Cancel", "Cancel the running build?"):
            self._cancel_builders()

    def _cancel_builders(self) -> None:
        self._runner.cancel()
        self._debug_feature_runner.cancel()

    def _publish_build_progress(self, payload: dict[str, Any]) -> None:
        get_build_progress_manager().publish(payload)

    def _begin_build_job(self, *, title: str = "Building Dataset") -> None:
        get_build_progress_manager().begin_job(
            "dataset_build",
            title=title,
            cancel_fn=self._cancel_builders,
        )

    def poll_progress(self, *, ui_visible: bool = True) -> bool:
        """Drain progress queue. Returns True when a build finished."""
        self._runner.drain_ipc()
        done = False
        latest: dict[str, Any] | None = None
        try:
            while True:
                payload = self.progress_queue.get_nowait()
                latest = payload
        except queue.Empty:
            pass
        if latest is not None:
            if latest.pop("_done", False):
                done = True
            if ui_visible or done:
                self.progress.render(latest)
            else:
                self.progress._last_payload = latest
                self.progress._received_at = time.time()
                self.progress._payload = latest
        if ui_visible:
            try:
                if (self.progress._last_payload or {}).get("status") == "running":
                    now = time.monotonic()
                    if now - self._last_live_tick_at >= 0.5:
                        self._last_live_tick_at = now
                        self.progress.render_live_tick()
            except Exception as exc:
                self.progress.error_var.set(f"Progress display error: {exc}")
        if done:
            self._adding_row_id = None
            self._set_build_ui_active(False)
            status = (self.progress._payload or {}).get("status")
            self._refresh_master_status()
            self._refresh_table()
            self._update_preview()
            if self._on_data_changed:
                self._on_data_changed()
            if status == "completed":
                payload = self.progress._payload or {}
                if not payload.get("debug_load") and not payload.get("debug_features"):
                    messagebox.showinfo("Done", self.progress.completion_var.get() or "Build completed.")
            elif status == "failed":
                messagebox.showerror(
                    "Build failed",
                    (self.progress._payload or {}).get("error") or "Unknown error",
                )
            elif status == "cancelled":
                messagebox.showwarning("Cancelled", "Build was cancelled.")
        return done
