"""Strategy Registry panel — create, clone, lifecycle, champion (Phase 2 Tk)."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .ui_state import get_ui_state_manager


class StrategyRegistryPanel(ttk.Frame, LazyLoadMixin):
    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._rows: list[dict[str, Any]] = []
        self._detail: dict[str, Any] | None = None
        self._selected_id: str | None = None
        self._editing_version_id: str | None = None
        self._status_var = tk.StringVar(value="")
        self._ui_state = get_ui_state_manager()
        self._build_ui()
        self._ui_state.bind_notebook(self._notebook, "strategy_registry.tab")
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="left", padx=2)
        ttk.Button(toolbar, text="New strategy", command=self._create_from_template).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Clone", command=self._clone_selected).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Set champion", command=self._set_champion).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Archive", command=self._archive_selected).pack(side="left", padx=2)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, weight=2)
        cols = ("name", "version", "hash", "status", "updated")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14)
        for c, w, label in (
            ("name", 180, "Strategy"),
            ("version", 50, "Champion"),
            ("hash", 90, "Hash"),
            ("status", 70, "Status"),
            ("updated", 140, "Updated"),
        ):
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="center" if c not in ("name",) else "w")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._load_detail())

        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill="both", expand=True)

        self._tab_overview = ttk.Frame(self._notebook)
        self._tab_versions = ttk.Frame(self._notebook)
        self._tab_config = ttk.Frame(self._notebook)
        self._tab_compare = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_overview, text="Overview")
        self._notebook.add(self._tab_versions, text="Lifecycle")
        self._notebook.add(self._tab_config, text="Champion Config")
        self._notebook.add(self._tab_compare, text="Compare")

        self._overview_text = scrolledtext.ScrolledText(self._tab_overview, height=10, font=("Consolas", 9))
        self._overview_text.pack(fill="both", expand=True, padx=4, pady=4)

        self._versions_tree = ttk.Treeview(
            self._tab_versions,
            columns=("ver", "lifecycle", "hash", "champion", "created"),
            show="headings",
            height=12,
        )
        for c, w, label in (
            ("ver", 50, "Ver"),
            ("lifecycle", 100, "Lifecycle"),
            ("hash", 90, "Hash"),
            ("champion", 70, "Champion"),
            ("created", 140, "Created"),
        ):
            self._versions_tree.heading(c, text=label)
            self._versions_tree.column(c, width=w)
        self._versions_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._versions_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_version_config())

        config_toolbar = ttk.Frame(self._tab_config, padding=(4, 4, 4, 0))
        config_toolbar.pack(fill="x")
        ttk.Button(config_toolbar, text="Save as new version", command=self._save_edited_config).pack(side="left", padx=2)
        ttk.Button(config_toolbar, text="Reload", command=self._reload_config_editor).pack(side="left", padx=2)
        ttk.Button(config_toolbar, text="Validate", command=self._validate_config_editor).pack(side="left", padx=2)
        ttk.Label(
            config_toolbar,
            text="Edit JSON, then Save as new version (immutable version + new champion).",
            foreground="#888",
        ).pack(side="left", padx=(12, 0))

        self._config_text = scrolledtext.ScrolledText(self._tab_config, height=16, font=("Consolas", 9))
        self._config_text.pack(fill="both", expand=True, padx=4, pady=4)

        compare_row = ttk.Frame(self._tab_compare, padding=4)
        compare_row.pack(fill="x")
        self._cmp_a = tk.StringVar()
        self._cmp_b = tk.StringVar()
        ttk.Label(compare_row, text="Version A").pack(side="left")
        ttk.Entry(compare_row, textvariable=self._cmp_a, width=28).pack(side="left", padx=4)
        ttk.Label(compare_row, text="Version B").pack(side="left")
        ttk.Entry(compare_row, textvariable=self._cmp_b, width=28).pack(side="left", padx=4)
        ttk.Button(compare_row, text="Compare", command=self._compare_versions).pack(side="left", padx=4)
        self._compare_text = scrolledtext.ScrolledText(self._tab_compare, height=14, font=("Consolas", 9))
        self._compare_text.pack(fill="both", expand=True, padx=4, pady=4)

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10, pady=(0, 4))

    def refresh(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_strategies,
                apply=self._apply_strategies,
                message="Loading strategies…",
                status_var=self._status_var,
            )
            return
        try:
            rows = self._fetch_strategies()
        except Exception as exc:
            self._status_var.set(f"Load failed: {exc}")
            return
        self._apply_strategies(rows)

    def _fetch_strategies(self) -> list[dict[str, Any]]:
        from chain_replay_ml.strategy_registry import list_strategies

        return list_strategies(self._data_dir())

    def _apply_strategies(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.tree.delete(*self.tree.get_children())
        for r in self._rows:
            sid = str(r.get("strategy_id") or "")
            self.tree.insert(
                "",
                "end",
                iid=sid,
                values=(
                    r.get("display_name"),
                    r.get("current_version_label"),
                    r.get("champion_config_hash"),
                    r.get("status"),
                    (r.get("updated_on") or "")[:19],
                ),
            )
        self._status_var.set(f"{len(self._rows)} strateg{'y' if len(self._rows) == 1 else 'ies'}")
        if self._selected_id and self._selected_id in self.tree.get_children():
            self.tree.selection_set(self._selected_id)
            self._load_detail()

    def _selected_strategy_id(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else self._selected_id

    def _load_detail(self) -> None:
        from chain_replay_ml.strategy_registry import get_strategy_detail

        sid = self._selected_strategy_id()
        if not sid:
            return
        self._selected_id = sid
        try:
            self._detail = get_strategy_detail(self._data_dir(), sid)
        except Exception as exc:
            messagebox.showerror("Strategy", str(exc))
            return
        if not self._detail:
            return

        profile = self._detail.get("profile") or {}
        champion = self._detail.get("champion_version") or {}
        lines = [
            f"strategy_id: {profile.get('strategy_id')}",
            f"display_name: {profile.get('display_name')}",
            f"description: {profile.get('description')}",
            f"champion: {profile.get('current_version_label')} ({profile.get('current_version_id')})",
            f"config_hash: {profile.get('champion_config_hash')}",
            f"status: {profile.get('status')}",
            f"versions: {self._detail.get('version_count')}",
            "",
            "Champion rules summary:",
            f"  target: {champion.get('config', {}).get('target')}",
            f"  stop: {champion.get('config', {}).get('stop')}",
            f"  hold: {champion.get('config', {}).get('hold_time')}",
            f"  entry: {champion.get('config', {}).get('entry')}",
        ]
        try:
            from chain_replay_ml.fold_research import get_strategy_filter_knowledge

            cfg = champion.get("config") or {}
            kb_hints = get_strategy_filter_knowledge(self._data_dir(), cfg)
            if kb_hints:
                lines.extend(["", "Knowledge Base:"])
                for h in kb_hints:
                    lines.append(
                        f"  • {h.get('filter_label')}: {h.get('experiments')} experiments, "
                        f"{h.get('confidence_pct')}% confidence ({h.get('status')})",
                    )
        except Exception:
            pass
        self._overview_text.delete("1.0", "end")
        self._overview_text.insert("end", "\n".join(lines))

        self._versions_tree.delete(*self._versions_tree.get_children())
        for v in self._detail.get("versions") or []:
            vid = str(v.get("version_id") or "")
            self._versions_tree.insert(
                "",
                "end",
                iid=vid,
                values=(
                    v.get("version_label"),
                    v.get("lifecycle_label") or v.get("lifecycle"),
                    v.get("config_hash"),
                    "yes" if v.get("is_champion") else "",
                    (v.get("created_on") or "")[:19],
                ),
            )

        cfg = champion.get("config") or {}
        self._editing_version_id = str(champion.get("version_id") or "") or None
        self._set_config_editor(cfg)

    def _set_config_editor(self, cfg: dict[str, Any]) -> None:
        self._config_text.delete("1.0", "end")
        self._config_text.insert("end", json.dumps(cfg, indent=2, default=str))

    def _config_from_editor(self) -> dict[str, Any]:
        raw = self._config_text.get("1.0", "end").strip()
        if not raw:
            raise ValueError("Config is empty")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Config must be a JSON object")
        return parsed

    def _reload_config_editor(self) -> None:
        if not self._detail:
            return
        vid = self._editing_version_id
        if vid:
            for v in self._detail.get("versions") or []:
                if v.get("version_id") == vid:
                    self._set_config_editor(v.get("config") or {})
                    return
        champion = self._detail.get("champion_version") or {}
        self._editing_version_id = str(champion.get("version_id") or "") or None
        self._set_config_editor(champion.get("config") or {})

    def _validate_config_editor(self) -> None:
        from chain_replay_ml.strategy_registry.schema import validate_strategy_config

        try:
            cfg = self._config_from_editor()
        except (json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("Validate", str(exc))
            return
        errors = validate_strategy_config(cfg)
        if errors:
            messagebox.showerror("Validate", "\n".join(errors))
            return
        messagebox.showinfo("Validate", "Config is valid.")

    def _save_edited_config(self) -> None:
        from chain_replay_ml.strategy_registry import create_strategy, create_strategy_version

        try:
            cfg = self._config_from_editor()
        except json.JSONDecodeError as exc:
            messagebox.showerror("Save", f"Invalid JSON:\n{exc}")
            return
        except ValueError as exc:
            messagebox.showerror("Save", str(exc))
            return

        sid = self._selected_strategy_id()
        if not sid or not self._detail:
            name = str(cfg.get("name") or "").strip()
            if not name:
                name = simpledialog.askstring("New strategy", "Strategy name:", parent=self) or ""
                name = name.strip()
                if not name:
                    return
                cfg["name"] = name
            try:
                detail = create_strategy(
                    self._data_dir(),
                    display_name=name,
                    description=str(cfg.get("description") or ""),
                    config=cfg,
                    lifecycle="new_strategy",
                )
            except Exception as exc:
                messagebox.showerror("Save", str(exc))
                return
            self._selected_id = detail["profile"]["strategy_id"]
            self.refresh()
            self._notebook.select(self._tab_config)
            messagebox.showinfo(
                "Save",
                f"Created {name} ({detail['profile']['current_version_label']}).",
            )
            return

        parent = self._editing_version_id or (self._detail.get("champion_version") or {}).get("version_id")
        try:
            version = create_strategy_version(
                self._data_dir(),
                strategy_id=sid,
                config=cfg,
                lifecycle="edit",
                parent_version_id=parent,
                set_champion=True,
            )
        except Exception as exc:
            messagebox.showerror("Save", str(exc))
            return

        label = version.get("version_label") or "?"
        same_version = str(version.get("version_id") or "") == str(parent or "")
        self._selected_id = sid
        self.refresh()
        self._notebook.select(self._tab_versions)
        vid = str(version.get("version_id") or "")
        if vid and vid in self._versions_tree.get_children():
            self._versions_tree.selection_set(vid)
            self._versions_tree.see(vid)
            self._show_version_config()
        if same_version:
            messagebox.showinfo("Save", f"No changes — still {label} (same config hash).")
        else:
            messagebox.showinfo("Save", f"Saved {label} and set as champion.")

    def _show_version_config(self) -> None:
        if not self._detail:
            return
        sel = self._versions_tree.selection()
        if not sel:
            return
        vid = sel[0]
        for v in self._detail.get("versions") or []:
            if v.get("version_id") == vid:
                self._editing_version_id = vid
                self._set_config_editor(v.get("config") or {})
                break

    def _create_from_template(self) -> None:
        from chain_replay_ml.strategy_registry import create_strategy, get_default_template

        name = simpledialog.askstring("Create Strategy", "Strategy name:", parent=self)
        if not name or not name.strip():
            return
        try:
            detail = create_strategy(
                self._data_dir(),
                display_name=name.strip(),
                config=get_default_template(),
            )
            self._selected_id = detail["profile"]["strategy_id"]
            self.refresh()
            self._notebook.select(self._tab_config)
            messagebox.showinfo(
                "Strategy",
                f"Created {name.strip()} ({detail['profile']['current_version_label']}).\n\n"
                "Edit the JSON in Champion Config, then click Save as new version.",
            )
        except Exception as exc:
            messagebox.showerror("Create Strategy", str(exc))

    def _clone_selected(self) -> None:
        from chain_replay_ml.strategy_registry import clone_strategy_version

        if not self._detail or not self._detail.get("champion_version"):
            messagebox.showinfo("Clone", "Select a strategy first.")
            return
        src = self._detail["champion_version"]["version_id"]
        name = simpledialog.askstring(
            "Clone Strategy",
            "New strategy family name (leave blank to version within same family):",
            parent=self,
        )
        if name is None:
            return
        try:
            version = clone_strategy_version(
                self._data_dir(),
                source_version_id=src,
                display_name=name.strip() or None,
            )
            self._selected_id = version.get("strategy_id")
            self.refresh()
            self._notebook.select(self._tab_versions)
            vid = str(version.get("version_id") or "")
            if vid and vid in self._versions_tree.get_children():
                self._versions_tree.selection_set(vid)
                self._versions_tree.see(vid)
            label = version.get("version_label") or "?"
            messagebox.showinfo(
                "Clone",
                f"Created {label}.\n\nEdit Champion Config with your experiment changes,\n"
                "then run Simulation on the new champion.",
            )
        except Exception as exc:
            messagebox.showerror("Clone", str(exc))

    def _set_champion(self) -> None:
        from chain_replay_ml.strategy_registry import set_champion_version

        sid = self._selected_strategy_id()
        sel = self._versions_tree.selection()
        if not sid or not sel:
            messagebox.showinfo("Champion", "Select a strategy and a version.")
            return
        try:
            set_champion_version(self._data_dir(), sid, sel[0])
            self.refresh()
            messagebox.showinfo("Champion", "Champion version updated.")
        except Exception as exc:
            messagebox.showerror("Champion", str(exc))

    def _archive_selected(self) -> None:
        from chain_replay_ml.strategy_registry import archive_strategy

        sid = self._selected_strategy_id()
        if not sid:
            return
        if not messagebox.askyesno("Archive", "Archive this strategy family?"):
            return
        try:
            archive_strategy(self._data_dir(), sid)
            self._selected_id = None
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Archive", str(exc))

    def _compare_versions(self) -> None:
        from chain_replay_ml.strategy_registry import compare_strategy_versions

        a = self._cmp_a.get().strip()
        b = self._cmp_b.get().strip()
        if not a or not b:
            sel = self._versions_tree.selection()
            versions = self._detail.get("versions") or [] if self._detail else []
            if len(versions) >= 2:
                a = versions[1]["version_id"]
                b = versions[0]["version_id"]
            elif sel:
                a = sel[0]
        if not a or not b:
            messagebox.showinfo("Compare", "Enter two version IDs or select versions.")
            return
        self._compare_text.delete("1.0", "end")
        try:
            result = compare_strategy_versions(self._data_dir(), a, b)
        except Exception as exc:
            self._compare_text.insert("end", str(exc))
            return
        if not result.get("ok"):
            self._compare_text.insert("end", result.get("error") or "Failed")
            return
        lines = [
            f"same_hash: {result.get('same_hash')}",
            f"changes: {len(result.get('changes') or [])}",
            "",
        ]
        for ch in result.get("changes") or []:
            lines.append(f"{ch['path']}: {ch['a']} -> {ch['b']}")
        self._compare_text.insert("end", "\n".join(lines))
