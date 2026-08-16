"""Scrollable feature/group picker — shared by build tab and side panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry

from .feature_selection_engine import (
    all_group_ids,
    all_registry_feature_names,
    default_feature_config,
    disable_feature_group,
    enable_feature_group,
    enforce_mandatory_features,
    export_feature_columns,
    group_display_state,
    group_feature_count,
    group_meta,
    is_default_feature_selection,
    normalize_enabled_groups,
    read_feature_config,
    set_group_features_enabled,
    sync_enabled_groups_from_features,
    total_active_registry_features,
    active_registry_feature_names,
)


class FeatureSelectionPicker:
    """Group + per-feature checkboxes with optional search."""

    def __init__(
        self,
        registry: dict[str, Any],
        *,
        profile_var: tk.StringVar | None = None,
        on_change: Callable[[], None] | None = None,
        chart_dir: str | None = None,
        feature_project_id: str | None = None,
    ) -> None:
        self._registry = registry
        self._profile_var = profile_var
        self._on_change = on_change
        self._chart_dir = chart_dir
        self._feature_project_id = str(feature_project_id or "all").strip().lower()
        self._schema_columns = (load_schema_registry() or {}).get("columns") or {}
        self._enabled_groups: set[str] = set()
        self._enabled_features: set[str] = set()
        self._search_var = tk.StringVar(value="")
        self._mount_parent: ttk.Frame | None = None
        self._groups_inner: ttk.Frame | None = None
        self._stats_var = tk.StringVar(value="")
        self._canvas: tk.Canvas | None = None
        self._always_expand_features = True
        self._unlocked_groups: set[str] = set()
        self._exclude_features: set[str] = set()

    def set_feature_project(
        self,
        project_id: str | None,
        chart_dir: str | None = None,
    ) -> None:
        if chart_dir is not None:
            self._chart_dir = chart_dir
        self._feature_project_id = str(project_id or "all").strip().lower()
        active_names = set(self.active_project_features())
        # Reconcile enabled features to new project
        matching = self._enabled_features & active_names
        if matching:
            self._enabled_features = matching
        else:
            self._enabled_features = set(active_names)
        self._unlocked_groups.clear()
        self._rebuild_groups()
        self._update_stats()

    def set_excluded_features(self, names: set[str] | list[str] | None) -> None:
        self._exclude_features = {str(n) for n in (names or [])}
        if self._enabled_features:
            self._enabled_features -= self._exclude_features
        self._rebuild_groups()
        self._update_stats()

    def excluded_features(self) -> frozenset[str]:
        return frozenset(self._exclude_features)

    def _project_source(self) -> dict[str, Any]:
        if self._chart_dir:
            from .build_service import chart_data_dir
            from chain_replay_ml.dataset_builder.feature_project_organization import (
                project_registry_feature_source,
            )

            data_dir = chart_data_dir(self._chart_dir)
            pid = str(self._feature_project_id or "all").strip().lower()
            return project_registry_feature_source(data_dir=data_dir, project_id=pid)
        from chain_replay_ml.dataset_builder.feature_project_organization import canonical_registry_groups
        from chain_replay_ml.dataset_builder.feature_ownership import canonical_registry_features

        return {
            "groups": canonical_registry_groups(),
            "features": sorted(canonical_registry_features()),
            "project_id": "all",
        }

    def active_project_features(self) -> list[str]:
        source = self._project_source()
        all_feats = source.get("features") or []
        retired = self._exclude_features
        return [f for f in all_feats if f not in retired]

    def active_feature_total(self) -> int:
        return len(self.active_project_features())

    def apply_config(self, cfg: dict[str, Any]) -> None:
        self._enabled_groups = normalize_enabled_groups(
            self._registry,
            {str(g) for g in (cfg.get("enabledGroups") or [])},
        )
        feats = cfg.get("enabledFeatures") or []
        if feats:
            self._enabled_features = {str(f) for f in feats}
        else:
            self._enabled_features = set(
                export_feature_columns(self._registry, self._enabled_groups),
            )
        if self._exclude_features:
            self._enabled_features -= self._exclude_features
        active_set = set(self.active_project_features())
        if active_set:
            self._enabled_features &= active_set
        if self._profile_var is not None:
            profile = str(cfg.get("profile") or "default")
            if profile != "custom" and not is_default_feature_selection(
                self._registry,
                self._enabled_groups,
                sorted(self._enabled_features),
                exclude_features=self._exclude_features,
            ):
                profile = "custom"
            self._profile_var.set(profile)
        enforce_mandatory_features(self._registry, self._enabled_groups, self._enabled_features)
        self._unlocked_groups.clear()
        self._enabled_groups = sync_enabled_groups_from_features(
            self._registry, self._enabled_features,
        )
        self._rebuild_groups()
        self._update_stats()

    def get_config(self) -> dict[str, Any]:
        profile = "custom"
        if self._profile_var is not None:
            profile = self._profile_var.get()
        return read_feature_config(
            self._registry,
            profile=profile,
            enabled_groups=set(self._enabled_groups),
            enabled_features=set(self._enabled_features),
        )

    def stats_text(self) -> str:
        groups = self._domain_rows()
        active_names = self.active_project_features()
        selected_groups = 0
        for gid, _, feats in groups:
            if feats and all(f in self._enabled_features for f in feats):
                selected_groups += 1
        selected_features = len([f for f in active_names if f in self._enabled_features])
        return (
            f"Groups {selected_groups} / {len(groups)}"
            f"   ·   Features {selected_features} / {len(active_names)}"
        )

    def mount(
        self,
        parent: ttk.Frame,
        *,
        show_search: bool = True,
        always_expand_features: bool = True,
        canvas_height: int = 420,
    ) -> None:
        self.unmount()
        self._mount_parent = parent
        self._always_expand_features = always_expand_features

        if show_search:
            search_row = ttk.Frame(parent)
            search_row.pack(fill="x", pady=(0, 6))
            ttk.Label(search_row, text="Search").pack(side="left")
            entry = ttk.Entry(search_row, textvariable=self._search_var, width=28)
            entry.pack(side="left", padx=(6, 0), fill="x", expand=True)
            entry.bind("<KeyRelease>", lambda _e: self._rebuild_groups())

        stats_row = ttk.Frame(parent)
        stats_row.pack(fill="x", pady=(0, 4))
        ttk.Label(stats_row, textvariable=self._stats_var, foreground="#666").pack(side="left")
        ttk.Button(stats_row, text="Select all", command=self._select_all).pack(side="right")

        groups_outer = ttk.LabelFrame(parent, text="Feature domains", padding=2)
        groups_outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(groups_outer, height=canvas_height, highlightthickness=0)
        scroll = ttk.Scrollbar(groups_outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._canvas = canvas
        self._groups_inner = inner
        self._update_stats()
        self._rebuild_groups()

    def unmount(self) -> None:
        if self._mount_parent is not None:
            for child in self._mount_parent.winfo_children():
                child.destroy()
        self._mount_parent = None
        self._groups_inner = None
        self._canvas = None

    def _is_custom(self) -> bool:
        if self._profile_var is None:
            return True
        return self._profile_var.get() == "custom"

    def _notify(self) -> None:
        self._update_stats()
        if self._on_change:
            self._on_change()

    def _update_stats(self) -> None:
        self._stats_var.set(self.stats_text())

    def _select_all(self) -> None:
        if self._profile_var is not None:
            self._profile_var.set("custom")
        self._enabled_groups = normalize_enabled_groups(
            self._registry, set(all_group_ids(self._registry)),
        )
        self._enabled_features = set(
            active_registry_feature_names(
                self._registry, exclude=self._exclude_features,
            ),
        )
        self._enforce_selection()
        self._enabled_groups = sync_enabled_groups_from_features(
            self._registry, self._enabled_features,
        )
        if self._profile_var is not None and is_default_feature_selection(
            self._registry,
            self._enabled_groups,
            sorted(self._enabled_features),
            exclude_features=self._exclude_features,
        ):
            self._profile_var.set("default")
        self._rebuild_groups()
        self._notify()

    def _display_name(self, feature_name: str) -> str:
        col = self._schema_columns.get(feature_name) or {}
        return str(col.get("display_name") or feature_name)

    def _group_editable(self, group_id: str, state: dict[str, Any]) -> bool:
        if group_id in self._unlocked_groups:
            return True
        return state["kind"] not in ("mandatory", "locked")

    def _unlock_group(self, group_id: str) -> None:
        state = group_display_state(self._registry, group_id, self._enabled_groups)
        if state["kind"] not in ("mandatory", "locked"):
            return
        self._unlocked_groups.add(group_id)
        if self._profile_var is not None:
            self._profile_var.set("custom")
        self._rebuild_groups()
        self._notify()

    def _bind_unlock(self, widget: tk.Misc, group_id: str) -> None:
        widget.bind("<Double-Button-1>", lambda _e, g=group_id: self._unlock_group(g))

    def _enforce_selection(self) -> None:
        enforce_mandatory_features(
            self._registry,
            self._enabled_groups,
            self._enabled_features,
            except_groups=self._unlocked_groups,
        )

    def _matches_search(self, query: str, *parts: str) -> bool:
        if not query:
            return True
        q = query.lower()
        return any(q in str(p or "").lower() for p in parts)

    def _feature_plugin_group(self, feature_name: str) -> str | None:
        for gid in all_group_ids(self._registry):
            feats = group_meta(self._registry, gid).get("features") or []
            if feature_name in {str(f) for f in feats}:
                return gid
        return None

    def _domain_rows(self) -> list[tuple[str, str, list[str]]]:
        """Ordered (group_id, label, active feature names) for the picker."""
        source = self._project_source()
        groups = source.get("groups") or []
        retired = self._exclude_features
        rows: list[tuple[str, str, list[str]]] = []
        for g in groups:
            gid = str(g.get("id") or "")
            label = str(g.get("label") or gid)
            feats = [str(f) for f in (g.get("features") or []) if str(f) not in retired]
            if not feats:
                continue
            rows.append((gid, label, feats))
        return rows

    def _domain_lock_state(self, feats: list[str]) -> dict[str, Any]:
        """Aggregate plugin-group lock state across features in a domain."""
        kinds: list[str] = []
        required_by: list[str] = []
        plugin_ids: list[str] = []
        for feat in feats:
            gid = self._feature_plugin_group(feat)
            if not gid:
                continue
            plugin_ids.append(gid)
            state = group_display_state(self._registry, gid, self._enabled_groups)
            kinds.append(str(state.get("kind") or "normal"))
            if state.get("required_by"):
                required_by.append(str(state["required_by"]))
        if any(k == "mandatory" for k in kinds):
            kind = "mandatory"
        elif any(k == "locked" for k in kinds):
            kind = "locked"
        else:
            kind = "normal"
        # Domain editable if every contributing plugin group is editable / unlocked.
        editable = True
        for gid in dict.fromkeys(plugin_ids):
            state = group_display_state(self._registry, gid, self._enabled_groups)
            if not self._group_editable(gid, state):
                editable = False
                break
        return {
            "kind": kind,
            "editable": editable,
            "required_by": ", ".join(dict.fromkeys(required_by)),
            "plugin_ids": list(dict.fromkeys(plugin_ids)),
        }

    def _unlock_domain(self, domain_id: str, feats: list[str]) -> None:
        lock = self._domain_lock_state(feats)
        if lock["kind"] not in ("mandatory", "locked"):
            return
        for gid in lock.get("plugin_ids") or []:
            self._unlocked_groups.add(str(gid))
        if self._profile_var is not None:
            self._profile_var.set("custom")
        self._rebuild_groups()
        self._notify()

    def _rebuild_groups(self) -> None:
        inner = self._groups_inner
        if inner is None:
            return
        for child in inner.winfo_children():
            child.destroy()

        is_custom = self._is_custom()
        query = self._search_var.get().strip()

        for domain_id, label, feats in self._domain_rows():
            count = len(feats)
            lock = self._domain_lock_state(feats)
            visible_feats = [
                f for f in feats
                if self._matches_search(query, f, self._display_name(f), label, domain_id)
            ]
            domain_matches = self._matches_search(query, label, domain_id)
            if query and not domain_matches and not visible_feats:
                continue

            selected_in_domain = sum(1 for f in feats if f in self._enabled_features)
            editable = bool(lock.get("editable"))
            domain_locked = not editable
            show_feats = visible_feats if query else feats
            if self._always_expand_features and not query:
                show_feats = feats

            block = ttk.Frame(inner)
            block.pack(fill="x", anchor="w", pady=2)

            hdr = ttk.Frame(block)
            hdr.pack(fill="x", anchor="w")

            if lock["kind"] == "mandatory" and not editable:
                lbl = ttk.Label(
                    hdr,
                    text=f"🔒 {label} ({count}) — Mandatory (double-click to unlock)",
                    foreground="#444",
                    cursor="hand2",
                )
                lbl.pack(side="left", anchor="w")
                lbl.bind(
                    "<Double-Button-1>",
                    lambda _e, d=domain_id, fs=list(feats): self._unlock_domain(d, fs),
                )
            elif lock["kind"] == "locked" and not editable:
                lbl = ttk.Label(
                    hdr,
                    text=(
                        f"🔒 {label} ({count}) — Required by: "
                        f"{lock.get('required_by', '')} (double-click to unlock)"
                    ),
                    foreground="#444",
                    wraplength=360,
                    cursor="hand2",
                )
                lbl.pack(side="left", anchor="w", fill="x", expand=True)
                lbl.bind(
                    "<Double-Button-1>",
                    lambda _e, d=domain_id, fs=list(feats): self._unlock_domain(d, fs),
                )
            elif not is_custom:
                var = tk.BooleanVar(value=True)
                ttk.Checkbutton(
                    hdr,
                    text=f"{label} ({count})",
                    variable=var,
                    state="disabled",
                ).pack(side="left", anchor="w")
            else:
                var = tk.BooleanVar(value=selected_in_domain == count and count > 0)
                if selected_in_domain not in (0, count):
                    var.set(True)
                cb = ttk.Checkbutton(
                    hdr,
                    text=f"{label} ({count})",
                    variable=var,
                    command=lambda d=domain_id, fs=list(feats), v=var: self._on_domain_toggle(d, fs, v),
                )
                cb.pack(side="left", anchor="w")
                if domain_locked:
                    cb.state(["disabled"])

            if show_feats:
                feat_frame = ttk.Frame(block, padding=(16, 0, 0, 0))
                feat_frame.pack(fill="x", anchor="w")
                for feat in show_feats:
                    fvar = tk.BooleanVar(value=feat in self._enabled_features)
                    plugin_gid = self._feature_plugin_group(feat) or ""
                    feat_group_locked = False
                    if plugin_gid:
                        st = group_display_state(
                            self._registry, plugin_gid, self._enabled_groups,
                        )
                        feat_group_locked = not self._group_editable(plugin_gid, st)
                    feat_disabled = (not is_custom) or feat_group_locked
                    cb = ttk.Checkbutton(
                        feat_frame,
                        text=self._display_name(feat),
                        variable=fvar,
                        state="disabled" if feat_disabled else "normal",
                        command=lambda g=plugin_gid, f=feat, v=fvar: self._on_feature_toggle(g, f, v),
                    )
                    cb.pack(anchor="w")

        if self._canvas is not None:
            self._canvas.update_idletasks()
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_domain_toggle(
        self,
        domain_id: str,
        feats: list[str],
        var: tk.BooleanVar,
    ) -> None:
        enabled = bool(var.get())
        for feat in feats:
            plugin_gid = self._feature_plugin_group(feat)
            if plugin_gid:
                st = group_display_state(self._registry, plugin_gid, self._enabled_groups)
                if not self._group_editable(plugin_gid, st) and not enabled:
                    continue
            if enabled:
                self._enabled_features.add(feat)
            else:
                self._enabled_features.discard(feat)
        self._enforce_selection()
        self._enabled_groups = sync_enabled_groups_from_features(
            self._registry, self._enabled_features,
        )
        if self._profile_var is not None and not is_default_feature_selection(
            self._registry,
            self._enabled_groups,
            sorted(self._enabled_features),
            exclude_features=self._exclude_features,
        ):
            self._profile_var.set("custom")
        self._rebuild_groups()
        self._notify()

    def _on_group_toggle(self, group_id: str, var: tk.BooleanVar) -> None:
        if var.get():
            self._enabled_groups = enable_feature_group(
                self._registry, self._enabled_groups, group_id,
            )
            for feat in group_meta(self._registry, group_id).get("features") or []:
                self._enabled_features.add(str(feat))
        else:
            self._enabled_groups = disable_feature_group(
                self._registry, self._enabled_groups, group_id,
                except_groups=self._unlocked_groups,
            )
            self._enabled_groups, self._enabled_features = set_group_features_enabled(
                self._registry,
                self._enabled_groups,
                self._enabled_features,
                group_id,
                enabled=False,
                except_groups=self._unlocked_groups,
            )
        self._enforce_selection()
        self._enabled_groups = sync_enabled_groups_from_features(
            self._registry, self._enabled_features,
        )
        if self._profile_var is not None and not is_default_feature_selection(
            self._registry,
            self._enabled_groups,
            sorted(self._enabled_features),
            exclude_features=self._exclude_features,
        ):
            self._profile_var.set("custom")
        self._rebuild_groups()
        self._notify()

    def _on_feature_toggle(self, group_id: str, feature_name: str, var: tk.BooleanVar) -> None:
        if var.get():
            self._enabled_features.add(feature_name)
        else:
            self._enabled_features.discard(feature_name)
        self._enforce_selection()
        self._enabled_groups = sync_enabled_groups_from_features(
            self._registry, self._enabled_features,
        )
        if self._profile_var is not None and not is_default_feature_selection(
            self._registry,
            self._enabled_groups,
            sorted(self._enabled_features),
            exclude_features=self._exclude_features,
        ):
            self._profile_var.set("custom")
        self._rebuild_groups()
        self._notify()
