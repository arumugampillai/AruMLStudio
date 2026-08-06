"""Dialog: Recommended Features for Retirement (Feature Registry)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from . import feature_registry_service as svc


class RecommendedRetirementDialog(tk.Toplevel):
    """Show REMOVE recommendations; retire or ignore with explicit confirmation."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Recommended Features for Retirement")
        self.transient(master.winfo_toplevel())
        self.geometry("720x480")
        self._chart_dir = chart_dir
        self._on_changed = on_changed
        self._status_var = tk.StringVar(value="")
        self._iid_to_name: dict[str, str] = {}
        self._catalog_by_name: dict[str, dict[str, Any]] = {}

        self._build_ui()
        self._reload()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        top.columnconfigure(0, weight=1)
        top.rowconfigure(1, weight=1)

        ttk.Label(
            top,
            text=(
                "From Production Validation history. Retirement uses the registry "
                "Disable path — features can be re-enabled later. Nothing is retired "
                "without confirmation."
            ),
            wraplength=680,
            foreground="#555",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        tree_frame = ttk.Frame(top)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("feature", "domain", "remove_runs", "last_date"),
            show="headings",
            selectmode="extended",
            height=16,
        )
        self._tree.heading("feature", text="Feature Name")
        self._tree.heading("domain", text="Domain")
        self._tree.heading("remove_runs", text="REMOVE Runs")
        self._tree.heading("last_date", text="Last Date")
        self._tree.column("feature", width=240, anchor="w", stretch=True)
        self._tree.column("domain", width=160, anchor="w")
        self._tree.column("remove_runs", width=100, anchor="e")
        self._tree.column("last_date", width=140, anchor="w")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        ttk.Label(top, textvariable=self._status_var, foreground="#555").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        actions = ttk.Frame(top)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(actions, text="Unselect All", command=self._unselect_all).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(
            actions,
            text="Ignore Recommendation",
            command=self._ignore_selected,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(actions, text="Close", command=self.destroy).pack(side="right")
        ttk.Button(
            actions,
            text="Retire Selected",
            command=self._retire_selected,
        ).pack(side="right", padx=(0, 6))

    def _format_last_date(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "—"
        if "T" in text:
            return text.split("T", 1)[0]
        return text[:19]

    def _domain_for(self, name: str, row: dict[str, Any]) -> str:
        domain = str(row.get("domain") or "").strip()
        if domain:
            return domain
        feat = self._catalog_by_name.get(name) or {}
        return str(
            feat.get("primary_domain_label")
            or feat.get("domain")
            or feat.get("group_filter")
            or "—"
        )

    def _feature_is_active(self, feat: dict[str, Any]) -> bool:
        return bool(feat.get("registry_active", True))

    def _reload(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._iid_to_name.clear()
        self._catalog_by_name.clear()

        try:
            catalog = svc.load_catalog(self._chart_dir)
        except Exception as exc:
            self._status_var.set(f"Could not load Feature Registry: {exc}")
            return

        for feat in catalog.get("features") or []:
            if not isinstance(feat, dict):
                continue
            name = str(feat.get("name") or "").strip()
            if name:
                self._catalog_by_name[name] = feat

        try:
            from chain_replay_ml.production_validation import recommended_for_removal

            rows = recommended_for_removal(svc.data_dir_for(self._chart_dir))
        except Exception as exc:
            self._status_var.set(f"Could not load recommendations: {exc}")
            return

        shown = 0
        skipped_missing = 0
        skipped_inactive = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("feature_name") or "").strip()
            if not name:
                continue
            feat = self._catalog_by_name.get(name)
            if feat is None:
                skipped_missing += 1
                continue
            if not self._feature_is_active(feat):
                skipped_inactive += 1
                continue
            iid = self._tree.insert(
                "",
                "end",
                values=(
                    name,
                    self._domain_for(name, row),
                    str(int(row.get("remove_runs") or 0)),
                    self._format_last_date(row.get("last_date")),
                ),
            )
            self._iid_to_name[iid] = name
            shown += 1

        parts = [f"{shown} recommendation(s)"]
        if skipped_inactive:
            parts.append(f"{skipped_inactive} already disabled")
        if skipped_missing:
            parts.append(f"{skipped_missing} not in registry catalogue")
        self._status_var.set(" · ".join(parts))

    def _selected_names(self) -> list[str]:
        out: list[str] = []
        for iid in self._tree.selection():
            name = self._iid_to_name.get(iid)
            if name:
                out.append(name)
        return sorted(set(out))

    def _select_all(self) -> None:
        self._tree.selection_set(tuple(self._iid_to_name.keys()))

    def _unselect_all(self) -> None:
        self._tree.selection_remove(self._tree.selection())

    def _notify_changed(self) -> None:
        if self._on_changed:
            try:
                self._on_changed()
            except Exception:
                pass

    def _retire_selected(self) -> None:
        names = self._selected_names()
        if not names:
            messagebox.showinfo(
                "Recommended Features for Retirement",
                "Select one or more features to retire.",
                parent=self,
            )
            return
        preview = "\n".join(f"  • {n}" for n in names[:20])
        extra = f"\n  … and {len(names) - 20} more" if len(names) > 20 else ""
        ok = messagebox.askyesno(
            "Retire Selected?",
            f"Retire {len(names)} Feature Registry feature(s)?\n\n"
            f"{preview}{extra}\n\n"
            "They will be disabled in the registry (can be re-enabled later).\n"
            "This does not delete features permanently.",
            parent=self,
        )
        if not ok:
            return

        errors: list[str] = []
        retired = 0
        for name in names:
            feat = self._catalog_by_name.get(name) or {}
            home_group = str(feat.get("group_id") or "") or None
            try:
                svc.set_feature_registry_active(
                    self._chart_dir,
                    name,
                    active=False,
                    home_group_id=home_group,
                )
                retired += 1
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        self._reload()
        self._notify_changed()
        if errors:
            messagebox.showwarning(
                "Recommended Features for Retirement",
                f"Retired {retired} feature(s).\n\n"
                f"{len(errors)} failed:\n" + "\n".join(errors[:10]),
                parent=self,
            )
        else:
            messagebox.showinfo(
                "Recommended Features for Retirement",
                f"Retired {retired} feature(s).",
                parent=self,
            )

    def _ignore_selected(self) -> None:
        names = self._selected_names()
        if not names:
            messagebox.showinfo(
                "Recommended Features for Retirement",
                "Select one or more recommendations to ignore.",
                parent=self,
            )
            return
        ok = messagebox.askyesno(
            "Ignore Recommendation?",
            f"Ignore retirement recommendations for {len(names)} feature(s)?\n\n"
            "History is kept; they will no longer appear in this list.",
            parent=self,
        )
        if not ok:
            return
        try:
            from chain_replay_ml.production_validation import ignore_recommendation

            data_dir = svc.data_dir_for(self._chart_dir)
            for name in names:
                ignore_recommendation(
                    data_dir,
                    name,
                    scope="registry",
                    reason="Ignored from Feature Registry retirement dialog",
                )
        except Exception as exc:
            messagebox.showerror(
                "Recommended Features for Retirement", str(exc), parent=self
            )
            return
        self._reload()
        messagebox.showinfo(
            "Recommended Features for Retirement",
            f"Ignored {len(names)} recommendation(s).",
            parent=self,
        )


def open_recommended_retirement_dialog(
    master: tk.Misc,
    *,
    chart_dir: str,
    on_changed: Callable[[], None] | None = None,
) -> RecommendedRetirementDialog:
    return RecommendedRetirementDialog(
        master, chart_dir=chart_dir, on_changed=on_changed
    )


__all__ = [
    "RecommendedRetirementDialog",
    "open_recommended_retirement_dialog",
]
