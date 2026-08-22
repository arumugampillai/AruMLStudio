"""Safe Data Migration Assistant Dialog for ML Research Studio (Doc 17, Phase 4)."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.core.data_root import DataRootService, normalize_storage_path
from chain_replay_ml.core.migration_service import DataMigrationService, MigrationPlan


class MigrationAssistantDialog(tk.Toplevel):
    """Modal dialog for executing safe 5-stage Data Root migration."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        target_data_root: str = r"D:\data",
        initial_source_dir: str | None = None,
        on_completed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("📦 Safe Data Migration Assistant — ML Research Studio")
        self.geometry("920x680")
        self.minsize(800, 550)
        self.transient(parent.winfo_toplevel())
        self.grab_set()

        self.target_data_root = target_data_root
        self.initial_source_dir = initial_source_dir or os.path.join(os.getcwd(), "data")
        self.on_completed = on_completed

        self.migration_svc = DataMigrationService(target_data_root=self.target_data_root)
        self.current_plan: MigrationPlan | None = None

        self._source_var = tk.StringVar(value=self.initial_source_dir)
        self._target_var = tk.StringVar(value=self.target_data_root)
        self._status_msg_var = tk.StringVar(value="Ready. Select a legacy source folder and click 'Run Pre-Flight Scan'.")

        self._build_ui()
        self.center_on_parent()

    def center_on_parent(self) -> None:
        self.update_idletasks()
        p = self.master.winfo_toplevel()
        pw, ph = p.winfo_width(), p.winfo_height()
        px, py = p.winfo_rootx(), p.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        x = max(0, px + (pw - w) // 2)
        y = max(0, py + (ph - h) // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        # Header
        ttk.Label(wrap, text="📦 Safe Data Migration Assistant", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            wrap,
            text="Consolidate legacy databases, registries, parquets, models, and predictions into the canonical Data Root with SHA-256 verification and zero data loss.",
            foreground="#666",
            wraplength=880,
        ).pack(anchor="w", pady=(2, 12))

        # Paths Frame
        path_box = ttk.LabelFrame(wrap, text="1. Source & Target Roots", padding=10)
        path_box.pack(fill="x", pady=(0, 10))

        # Source Row
        s_row = ttk.Frame(path_box)
        s_row.pack(fill="x", pady=2)
        ttk.Label(s_row, text="Legacy Source Folder:", width=20, font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Entry(s_row, textvariable=self._source_var, width=65).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(s_row, text="Browse…", command=self._browse_source).pack(side="left")

        # Target Row
        t_row = ttk.Frame(path_box)
        t_row.pack(fill="x", pady=2)
        ttk.Label(t_row, text="Canonical Data Root:", width=20, font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Entry(t_row, textvariable=self._target_var, state="readonly", width=65).pack(side="left", fill="x", expand=True, padx=4)

        # Actions Row
        scan_row = ttk.Frame(path_box)
        scan_row.pack(fill="x", pady=(8, 2))
        self.scan_btn = ttk.Button(scan_row, text="🔍 Run Pre-Flight Scan & Dry-Run", command=self._run_scan)
        self.scan_btn.pack(side="left")
        self.target_free_lbl = ttk.Label(scan_row, text="", foreground="#007acc", font=("Segoe UI", 9))
        self.target_free_lbl.pack(side="right")

        # Plan Summary Card
        self.summary_box = ttk.LabelFrame(wrap, text="2. Pre-Flight Inspection Summary", padding=10)
        self.summary_box.pack(fill="x", pady=(0, 10))

        self.sum_lbl = ttk.Label(
            self.summary_box,
            text="No scan executed yet. Click 'Run Pre-Flight Scan' above.",
            font=("Segoe UI", 9),
            wraplength=880,
        )
        self.sum_lbl.pack(anchor="w")

        # Items Table
        tree_box = ttk.LabelFrame(wrap, text="3. Discovered Migration Items", padding=8)
        tree_box.pack(fill="both", expand=True, pady=(0, 10))

        cols = ("category", "name", "size", "status", "action")
        self.tree = ttk.Treeview(tree_box, columns=cols, show="headings", height=8)
        self.tree.heading("category", text="Category")
        self.tree.heading("name", text="Artifact Name")
        self.tree.heading("size", text="Size")
        self.tree.heading("status", text="Destination Status")
        self.tree.heading("action", text="Migration Action")

        self.tree.column("category", width=140, anchor="w")
        self.tree.column("name", width=250, anchor="w")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("status", width=180, anchor="w")
        self.tree.column("action", width=180, anchor="w")

        tree_sb = ttk.Scrollbar(tree_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_sb.pack(side="right", fill="y")

        # Progress & Action Bar
        bottom_box = ttk.Frame(wrap)
        bottom_box.pack(fill="x", pady=(0, 4))

        self.prog_bar = ttk.Progressbar(bottom_box, mode="determinate")
        self.prog_bar.pack(fill="x", pady=(0, 6))

        b_row = ttk.Frame(bottom_box)
        b_row.pack(fill="x")

        self.status_lbl = ttk.Label(b_row, textvariable=self._status_msg_var, foreground="#333", font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", fill="x", expand=True)

        self.exec_btn = ttk.Button(
            b_row,
            text="🚀 Execute Safe Migration",
            command=self._confirm_and_execute,
            state="disabled",
        )
        self.exec_btn.pack(side="right", padx=(8, 0))

        ttk.Button(b_row, text="Close", command=self.destroy).pack(side="right")

    def _browse_source(self) -> None:
        init = self._source_var.get().strip() or os.getcwd()
        picked = filedialog.askdirectory(parent=self, title="Select Legacy Source Directory", initialdir=init)
        if picked:
            self._source_var.set(normalize_storage_path(picked))
            self._run_scan()

    def _run_scan(self) -> None:
        src = self._source_var.get().strip()
        if not src or not os.path.isdir(src):
            messagebox.showwarning("Invalid Source", f"Selected source directory does not exist:\n{src}", parent=self)
            return

        self._status_msg_var.set("Scanning source and computing checksums…")
        self.update_idletasks()

        self.current_plan = self.migration_svc.build_plan(src)
        p = self.current_plan

        # Update Summary
        s_text = (
            f"• Source: {p.source_dir}\n"
            f"• Target Data Root: {p.target_data_root}\n"
            f"• Total Items Discovered: {len(p.items)} ({p.total_bytes / (1024*1024):.2f} MB)\n"
            f"• Ready for Migration: {p.ready_count} | Already Identical: {p.identical_count} | Conflicts: {p.conflict_count}\n"
            f"• Target Drive Free Space: {p.disk_free_bytes / (1024**3):.2f} GB"
        )
        if p.issues:
            s_text += "\n⚠️ WARNINGS: " + "; ".join(p.issues)
            self.sum_lbl.config(foreground="#b00")
        else:
            self.sum_lbl.config(foreground="#006600")
        self.sum_lbl.config(text=s_text)

        # Populate Tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        for it in p.items:
            sz_str = f"{it.size_bytes / (1024*1024):.2f} MB" if it.size_bytes >= 1024*1024 else f"{it.size_bytes / 1024:.1f} KB"
            act_str = "Copy & Verify" if it.status == "ready" else ("Preserve Identical" if it.status == "identical" else "SKIP (Conflict)")
            self.tree.insert("", "end", values=(it.category, it.name, sz_str, it.status, act_str))

        if p.is_safe_to_execute and len(p.items) > 0:
            self.exec_btn.config(state="normal")
            self._status_msg_var.set(f"Pre-flight scan passed. {p.ready_count} item(s) ready to migrate.")
        else:
            self.exec_btn.config(state="disabled")
            self._status_msg_var.set("Scan complete. Nothing to migrate or execution blocked.")

    def _confirm_and_execute(self) -> None:
        if not self.current_plan:
            return
        p = self.current_plan
        msg = (
            f"Ready to migrate data to {p.target_data_root}?\n\n"
            f"• Items to copy: {p.ready_count}\n"
            f"• Items already verified: {p.identical_count}\n"
            f"• Total Data Size: {p.total_bytes / (1024*1024):.2f} MB\n\n"
            f"Source files will remain untouched for safe rollback."
        )
        if not messagebox.askyesno("Confirm Safe Migration", msg, parent=self):
            return

        self.scan_btn.config(state="disabled")
        self.exec_btn.config(state="disabled")
        self.prog_bar["value"] = 0

        def _worker() -> None:
            def _on_prog(m: str, cur: int, tot: int) -> None:
                pct = int((cur / max(tot, 1)) * 100)
                self.after(0, lambda: self._update_prog(m, pct))

            res = self.migration_svc.execute_migration(self.current_plan, on_progress=_on_prog)
            self.after(0, lambda: self._on_done(res))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_prog(self, msg: str, pct: int) -> None:
        self._status_msg_var.set(msg)
        self.prog_bar["value"] = pct

    def _on_done(self, res: dict[str, Any]) -> None:
        self.scan_btn.config(state="normal")
        self.prog_bar["value"] = 100
        if res.get("success"):
            self._status_msg_var.set("✅ Migration successfully completed and verified!")
            messagebox.showinfo(
                "Migration Complete",
                f"{res.get('message')}\n\nManifest recorded at:\n{res.get('manifest_path')}",
                parent=self,
            )
            if self.on_completed:
                self.on_completed(self.target_data_root)
            self.destroy()
        else:
            self._status_msg_var.set("❌ Migration encountered issues.")
            err_msg = "; ".join(res.get("errors", []))
            messagebox.showerror("Migration Error", f"Migration issues encountered:\n{err_msg}", parent=self)
            self.exec_btn.config(state="normal")
