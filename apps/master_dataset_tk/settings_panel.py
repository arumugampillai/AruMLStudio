"""Settings page — Canonical Data Root, Storage Layout, and Migration Assistant (Doc 17, Phase 4)."""

from __future__ import annotations

import os
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from chain_replay_ml.core.data_root import (
    DEFAULT_CANONICAL_DATA_ROOT,
    DataRootService,
    get_data_root_service,
    normalize_storage_path,
    resolve_data_root,
    save_data_root,
)
from .migration_dialog import MigrationAssistantDialog
from .project_config import (
    bundled_chart_dir,
    config_path,
    normalize_chart_dir,
    resolve_tick_data_dir,
    save_tick_data_dir,
)
from .ui_util import open_path


class SettingsPanel(ttk.Frame):
    """Authoritative Settings UI for managing the canonical Data Root and Storage Layout."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str | None = None,
        on_project_changed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir or resolve_data_root()
        self._on_project_changed = on_project_changed

        self._data_root_var = tk.StringVar(value=resolve_data_root())
        self._tick_data_var = tk.StringVar(value=self._configured_tick_data_dir())

        self._build_ui()

    def _data_root_service(self) -> DataRootService:
        return get_data_root_service(self._data_root_var.get().strip() or DEFAULT_CANONICAL_DATA_ROOT)

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._data_root_var.set(resolve_data_root())
        self._tick_data_var.set(self._configured_tick_data_dir())
        self._refresh_health_card()
        if hasattr(self, "_disable_gil_var"):
            self._disable_gil_var.set(self._load_disable_gil_monitor())

    def _configured_tick_data_dir(self) -> str:
        from .project_config import load_project_config

        saved = str(load_project_config().get("tick_data_dir") or "").strip()
        if saved:
            return normalize_chart_dir(saved)
        from tick_data_paths import DEFAULT_TICK_DATA_DIR

        return DEFAULT_TICK_DATA_DIR

    def _resolved_tick_data_dir(self) -> str:
        return resolve_tick_data_dir(self.chart_dir)

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        # Header
        ttk.Label(wrap, text="⚙️ Storage & Data Root Settings", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            wrap,
            text="ML Research Studio uses a Single Authoritative Data Root. All databases, registries, datasets, models, and predictions resolve beneath this root.",
            foreground="#666",
            wraplength=840,
        ).pack(anchor="w", pady=(0, 14))

        # 1. Primary Data Root Section
        root_box = ttk.LabelFrame(wrap, text="📁 1. Application Data Root (Canonical Ground Truth)", padding=12)
        root_box.pack(fill="x", pady=(0, 14))

        ttk.Label(
            root_box,
            text="Primary directory owning all persistent application data (Doc 17 specification):",
            foreground="#555",
        ).pack(anchor="w", pady=(0, 6))

        r_row = ttk.Frame(root_box)
        r_row.pack(fill="x", pady=(0, 8))
        ttk.Entry(r_row, textvariable=self._data_root_var, font=("Segoe UI", 10, "bold"), width=70).pack(side="left", fill="x", expand=True)
        ttk.Button(r_row, text="Browse…", command=self._browse_data_root).pack(side="left", padx=(6, 0))

        btn_row = ttk.Frame(root_box)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="📂 Open Data Root", command=lambda: open_path(self._data_root_var.get().strip())).pack(side="left")
        ttk.Button(btn_row, text="💾 Save & Apply Data Root", command=self._apply_data_root).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="📦 Open Data Migration Assistant", command=self._open_migration_assistant).pack(side="right")

        ttk.Label(
            root_box,
            text=f"Configuration file: {config_path()}",
            foreground="#888",
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(8, 0))

        # 2. Storage Layout & Telemetry Card
        self.layout_box = ttk.LabelFrame(wrap, text="📊 2. Canonical Storage Layout & Health Status", padding=12)
        self.layout_box.pack(fill="both", expand=True, pady=(0, 14))

        self._health_status_lbl = ttk.Label(self.layout_box, text="", font=("Segoe UI", 10, "bold"))
        self._health_status_lbl.pack(anchor="w", pady=(0, 8))

        # Treeview showing canonical directory hierarchy
        cols = ("category", "canonical_path", "status", "size_desc")
        self.layout_tree = ttk.Treeview(self.layout_box, columns=cols, show="headings", height=9)
        self.layout_tree.heading("category", text="Category")
        self.layout_tree.heading("canonical_path", text="Canonical Subdirectory")
        self.layout_tree.heading("status", text="Existence")
        self.layout_tree.heading("size_desc", text="Tracked Contents")

        self.layout_tree.column("category", width=140, anchor="w")
        self.layout_tree.column("canonical_path", width=380, anchor="w")
        self.layout_tree.column("status", width=90, anchor="center")
        self.layout_tree.column("size_desc", width=220, anchor="w")

        tree_sb = ttk.Scrollbar(self.layout_box, orient="vertical", command=self.layout_tree.yview)
        self.layout_tree.configure(yscrollcommand=tree_sb.set)
        self.layout_tree.pack(side="left", fill="both", expand=True)
        tree_sb.pack(side="right", fill="y")

        # 3. Operations & External Overrides
        bottom_fr = ttk.Frame(wrap)
        bottom_fr.pack(fill="x", pady=(0, 6))

        ttk.Button(bottom_fr, text="🔄 Verify Storage Integrity", command=self._verify_storage_integrity).pack(side="left")

        # Diagnostics Frame
        diag_fr = ttk.LabelFrame(wrap, text="⚙️ 3. Diagnostics", padding=10)
        diag_fr.pack(fill="x", pady=(8, 0))
        self._disable_gil_var = tk.BooleanVar(value=self._load_disable_gil_monitor())
        ttk.Checkbutton(
            diag_fr,
            text="Disable GIL / event-loop diagnostics during builds (recommended)",
            variable=self._disable_gil_var,
            command=self._save_gil_monitor_pref,
        ).pack(anchor="w")

        self._refresh_health_card()

    def _refresh_health_card(self) -> None:
        svc = self._data_root_service()
        val = svc.validate_layout()

        # Update Health Label
        if val["root_exists"]:
            self._health_status_lbl.config(
                text=f"🟢 Storage Status: HEALTHY · Disk Free: {val['disk_free_gb']} GB / {val['disk_total_gb']} GB",
                foreground="#006600",
            )
        else:
            self._health_status_lbl.config(
                text=f"🟡 Storage Status: ROOT DIRECTORY NOT FOUND ({svc.data_root})",
                foreground="#b00",
            )

        for item in self.layout_tree.get_children():
            self.layout_tree.delete(item)

        categories = [
            ("Databases", svc.get_database_path("analysis"), "analysis.db, evidence.db, historic_bars.db"),
            ("Registries", svc.get_registry_path("pipeline"), "pipeline_registry, feature_registry"),
            ("Datasets (Master)", svc.get_datasets_dir("master"), "master_dataset_nifty_*.db"),
            ("Datasets (Analysis)", svc.get_datasets_dir("analysis"), "analysis_*.parquet & json schemas"),
            ("Datasets (Labels)", svc.get_datasets_dir("labels"), "triple_barrier_run_*.parquet"),
            ("Datasets (Exports)", svc.get_datasets_dir("exports"), "chain_NIFTY_*.json exports"),
            ("Models (Candidates)", svc.get_models_dir("candidates"), "CAND_* validated model packages"),
            ("Models (Research)", svc.get_models_dir("research"), "Exp_* research models & reports"),
            ("Models (Production)", svc.get_models_dir("production"), "Active production model weights"),
            ("Predictions (Datasets)", svc.get_predictions_dir("datasets"), "model_lab_*.db evaluation stores"),
            ("Predictions (Artifacts)", svc.get_predictions_dir("artifacts"), "prediction metadata & manifests"),
            ("Ticks (Market Feeds)", svc.get_ticks_dir(), "angel_market_YYYY-MM-DD.db"),
            ("Application Logs", svc.get_logs_dir(), "worker & campaign execution logs"),
            ("Cache", svc.get_cache_dir(), "OpenAPIScripMaster.json"),
        ]

        for cat_name, path, desc in categories:
            parent_dir = path if not path.endswith((".db", ".json")) else os.path.dirname(path)
            exists = "🟢 YES" if os.path.exists(path if path.endswith((".db", ".json")) else parent_dir) else "⚪ Missing"
            self.layout_tree.insert("", "end", values=(cat_name, parent_dir, exists, desc))

    def _browse_data_root(self) -> None:
        init = self._data_root_var.get().strip() or DEFAULT_CANONICAL_DATA_ROOT
        if not os.path.isdir(init):
            init = os.path.splitdrive(init)[0] or "C:\\"
        picked = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Select Canonical Application Data Root", initialdir=init)
        if picked:
            self._data_root_var.set(normalize_storage_path(picked))
            self._refresh_health_card()

    def _apply_data_root(self) -> None:
        raw = self._data_root_var.get().strip()
        if not raw:
            messagebox.showinfo("Data Root", "Choose a directory first.", parent=self.winfo_toplevel())
            return
        root = normalize_storage_path(raw)
        save_data_root(root)
        self.chart_dir = root
        self._data_root_var.set(root)
        self._refresh_health_card()
        if self._on_project_changed:
            self._on_project_changed(root)
        messagebox.showinfo(
            "Data Root Saved",
            f"Canonical Data Root successfully updated to:\n{root}\n\nAll application subsystems will now resolve from this root.",
            parent=self.winfo_toplevel(),
        )

    def _open_migration_assistant(self) -> None:
        MigrationAssistantDialog(
            self.winfo_toplevel(),
            target_data_root=self._data_root_var.get().strip() or DEFAULT_CANONICAL_DATA_ROOT,
            on_completed=lambda r: self._on_migration_completed(r),
        )

    def _on_migration_completed(self, new_root: str) -> None:
        self._data_root_var.set(new_root)
        self._refresh_health_card()
        if self._on_project_changed:
            self._on_project_changed(new_root)

    def _verify_storage_integrity(self) -> None:
        svc = self._data_root_service()
        val = svc.validate_layout()
        if not val["root_exists"]:
            messagebox.showwarning("Storage Integrity", f"Data Root does not exist:\n{svc.data_root}", parent=self.winfo_toplevel())
            return

        db_checks: list[str] = []
        for name, db_type in [("Analysis DB", "analysis"), ("Evidence DB", "feature_evidence"), ("Historic Bars DB", "angel_historic")]:
            p = svc.get_database_path(db_type)
            if os.path.isfile(p):
                try:
                    conn = sqlite3.connect(p, timeout=5.0)
                    cur = conn.cursor()
                    cur.execute("PRAGMA integrity_check;")
                    res = cur.fetchone()[0]
                    conn.close()
                    db_checks.append(f"• {name}: {res} ({os.path.getsize(p)/(1024*1024):.1f} MB)")
                except Exception as e:
                    db_checks.append(f"• {name}: ERROR ({e})")
            else:
                db_checks.append(f"• {name}: Not found at {p}")

        msg = (
            f"Storage Integrity Verification for {svc.data_root}:\n\n"
            f"• Disk Space: {val['disk_free_gb']} GB Free / {val['disk_total_gb']} GB Total\n\n"
            + "\n".join(db_checks)
        )
        messagebox.showinfo("Storage Integrity Report", msg, parent=self.winfo_toplevel())

    def _load_disable_gil_monitor(self) -> bool:
        from .build_config_prefs import load_build_config_prefs

        studio = (load_build_config_prefs(self.chart_dir) or {}).get("studio") or {}
        return bool(studio.get("disable_gil_monitor", True))

    def _save_gil_monitor_pref(self) -> None:
        from .build_config_prefs import load_build_config_prefs, save_build_config_prefs

        existing = load_build_config_prefs(self.chart_dir) or {}
        studio = dict(existing.get("studio") or {})
        studio["disable_gil_monitor"] = bool(self._disable_gil_var.get())
        save_build_config_prefs(self.chart_dir, {"studio": studio})
