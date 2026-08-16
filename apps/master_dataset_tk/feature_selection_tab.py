"""Feature selection tab — profile summary + side panel picker for build config."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .feature_selection_engine import (
    default_feature_config,
    is_default_feature_selection,
    profile_label,
)
from .feature_selection_picker import FeatureSelectionPicker
from .feature_selection_side_panel import open_feature_selection_panel


class FeatureSelectionTab:
    def __init__(
        self,
        parent: ttk.Frame,
        registry: dict[str, Any],
        *,
        chart_dir: str | None = None,
        feature_project_id: str | None = None,
        on_change: Callable[[], None] | None = None,
        get_sampling_interval_sec: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry
        self._chart_dir = chart_dir
        self._feature_project_id = str(feature_project_id or "all").strip().lower()
        self._on_change = on_change
        self._get_sampling_interval_sec = get_sampling_interval_sec
        self._host = parent
        self._profile_var = tk.StringVar(value="default")
        self._stats_var = tk.StringVar(value="")
        self._warmup_var = tk.StringVar(value="")
        self._default_profile_label = tk.StringVar(value="")
        self._picker = FeatureSelectionPicker(
            registry,
            profile_var=self._profile_var,
            on_change=self._notify,
            chart_dir=chart_dir,
            feature_project_id=self._feature_project_id,
        )
        self._panel_win: tk.Toplevel | None = None
        self._build(parent)
        self.apply_config(default_feature_config(registry))
        self.sync_registry_exclusions()

    def set_chart_dir(self, chart_dir: str | None) -> None:
        self._chart_dir = chart_dir
        self._picker.set_feature_project(self._feature_project_id, chart_dir)
        self.sync_registry_exclusions()

    def set_feature_project_id(self, pid: str) -> None:
        self._feature_project_id = str(pid or "all").strip().lower()
        self._picker.set_feature_project(self._feature_project_id, self._chart_dir)
        self._refresh_profile_labels()
        self._update_stats()
        self.refresh_warmup()

    def sync_registry_exclusions(self) -> None:
        if not self._chart_dir:
            self._picker.set_excluded_features([])
        else:
            from . import feature_registry_service as fr_svc

            self._picker.set_excluded_features(fr_svc.disabled_registry_features(self._chart_dir))
        self._refresh_profile_labels()
        self._update_stats()

    def refresh_warmup(self) -> None:
        self._update_warmup()

    def _active_feature_total(self) -> int:
        return self._picker.active_feature_total()

    def _refresh_profile_labels(self) -> None:
        total = self._active_feature_total()
        self._default_profile_label.set(f"{profile_label(self._registry, 'default')} ({total})")

    def apply_config(self, cfg: dict[str, Any]) -> None:
        self._picker.apply_config(cfg)
        self._update_stats()

    def get_config(self) -> dict[str, Any]:
        cfg = self._picker.get_config()
        if self._chart_dir:
            from . import feature_registry_service as fr_svc

            cfg = fr_svc.sanitize_feature_selection(
                self._chart_dir, cfg, self._registry,
            )
        return cfg

    def _notify(self) -> None:
        self._update_stats()
        if self._on_change:
            self._on_change()

    def _build(self, parent: ttk.Frame) -> None:
        profile_row = ttk.LabelFrame(parent, text="Feature profile", padding=4)
        profile_row.pack(fill="x", pady=(0, 4))
        ttk.Radiobutton(
            profile_row,
            textvariable=self._default_profile_label,
            variable=self._profile_var,
            value="default",
            command=self._on_profile_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            profile_row,
            text="Custom",
            variable=self._profile_var,
            value="custom",
            command=self._on_profile_changed,
        ).pack(anchor="w")

        stats_row = ttk.Frame(parent)
        stats_row.pack(fill="x", pady=(4, 0))
        ttk.Label(stats_row, textvariable=self._stats_var, foreground="#666").pack(side="left")
        ttk.Button(
            stats_row,
            text="Selection…",
            command=self._open_selection_panel,
        ).pack(side="right")

        ttk.Label(
            parent,
            textvariable=self._warmup_var,
            foreground="#555",
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(
            parent,
            text="Use Selection to pick domains and features in a panel beside the build view.",
            foreground="#888",
            font=("TkDefaultFont", 8),
            wraplength=340,
        ).pack(anchor="w", pady=(6, 0))
        self._refresh_profile_labels()

    def _update_stats(self) -> None:
        self._stats_var.set(self._picker.stats_text())
        self._update_warmup()

    def _resolved_build_features(self) -> list[str]:
        from chain_replay_ml.dataset_builder.feature_plugins import (
            resolve_implemented_features_for_selection,
        )

        from .build_service import chart_data_dir

        data_dir = chart_data_dir(self._chart_dir) if self._chart_dir else None
        _, implemented, _, _ = resolve_implemented_features_for_selection(
            self.get_config(), self._registry, data_dir=data_dir,
        )
        return list(implemented)

    def _sampling_interval_sec(self) -> float:
        if self._get_sampling_interval_sec is not None:
            try:
                return max(0.001, float(self._get_sampling_interval_sec()))
            except (TypeError, ValueError):
                pass
        return 10.0

    def _update_warmup(self) -> None:
        from .feature_policy_format import format_required_warmup_label

        self._warmup_var.set(
            format_required_warmup_label(
                self._resolved_build_features(),
                sampling_interval_sec=self._sampling_interval_sec(),
            ),
        )

    def _on_profile_changed(self) -> None:
        if self._profile_var.get() != "custom":
            self._picker.apply_config(default_feature_config(self._registry))
        self._notify()

    def _open_selection_panel(self) -> None:
        if self._panel_win is not None and self._panel_win.winfo_exists():
            self._panel_win.lift()
            self._panel_win.focus_force()
            return

        if self._profile_var.get() != "custom":
            self._profile_var.set("custom")

        def on_apply(cfg: dict[str, Any]) -> None:
            self.apply_config(cfg)
            if not is_default_feature_selection(
                self._registry,
                set(cfg.get("enabledGroups") or []),
                list(cfg.get("enabledFeatures") or []),
                exclude_features=self._picker.excluded_features(),
            ):
                self._profile_var.set("custom")
            self._notify()

        self._panel_win = open_feature_selection_panel(
            self._host.winfo_toplevel(),
            registry=self._registry,
            profile_var=self._profile_var,
            initial_config=self.get_config(),
            excluded_features=self._picker.excluded_features(),
            chart_dir=self._chart_dir,
            feature_project_id=self._feature_project_id,
            on_apply=on_apply,
        )
        self._panel_win.bind("<Destroy>", lambda _e: setattr(self, "_panel_win", None))
