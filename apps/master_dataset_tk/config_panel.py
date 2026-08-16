"""Build configuration panel — sampling, targets, feature selection."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_plugins import horizon_label
from chain_replay_ml.dataset_builder.feature_plugins import resolve_implemented_features_for_selection
from chain_replay_ml.dataset_builder.master_defaults import default_master_prediction_targets
from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry

from chain_replay_ml.dataset_builder.gap_policy import (
    default_gap_policy,
    gap_max_sec_from_policy,
    gap_summary_label,
    normalize_gap_policy,
)

from .build_config_prefs import load_build_config_prefs, save_build_config_prefs
from .feature_selection_engine import default_feature_config, normalize_enabled_groups
from .feature_selection_tab import FeatureSelectionTab
from .gap_policy_tab import GapPolicyTab
from .strike_selection_engine import (
    default_strike_config,
    normalize_strike_config,
)
from .strike_selection_tab import StrikeSelectionTab
from .target_horizons import (
    DEFAULT_HORIZON_SEC,
    TARGET_TYPE_LABEL,
    default_horizon_selection,
    target_horizon_columns,
)


class BuildConfigPanel(ttk.LabelFrame):
  def __init__(
      self,
      master: tk.Misc,
      *,
      chart_dir: str | None = None,
      on_change: Callable[[], None] | None = None,
      on_build: Callable[[], None] | None = None,
      **kwargs: Any,
  ) -> None:
      super().__init__(master, text="Build configuration", padding=6, **kwargs)
      self._chart_dir = chart_dir
      self._on_change = on_change
      self._on_build = on_build
      self._registry = _load_feature_registry()
      self._horizon_vars: dict[int, tk.BooleanVar] = {}
      self._feature_tab: FeatureSelectionTab | None = None
      self._strike_tab: StrikeSelectionTab | None = None
      self._gap_tab: GapPolicyTab | None = None
      self._master_status: dict[str, Any] | None = None
      self._feature_project_locked = False

      self.interval_var = tk.IntVar(value=10)
      self.stride_var = tk.IntVar(value=10)
      self.stride_error_var = tk.StringVar(value="")
      self.low_memory_var = tk.BooleanVar(value=False)
      self.build_profiler_var = tk.BooleanVar(value=True)
      from chain_replay_ml.dataset_builder.feature_project_organization import RESERVED_ALL_PROJECT_ID

      self._feature_project_var = tk.StringVar(value=RESERVED_ALL_PROJECT_ID)
      self._feature_project_label_to_id: dict[str, str] = {}

      self._build_ui()
      self._apply_saved_prefs()
      self._update_stride_validation()
      self.update_summary()

  def _notify(self) -> None:
      if self._feature_tab is not None:
          self._feature_tab.refresh_warmup()
      self.update_summary()
      self._persist_prefs()
      if self._on_change:
          self._on_change()

  def set_chart_dir(self, chart_dir: str | None) -> None:
      self._chart_dir = chart_dir
      if self._feature_tab is not None:
          self._feature_tab.set_chart_dir(chart_dir)
      self._apply_saved_prefs()
      self.refresh_registry_exclusions()

  def refresh_registry_exclusions(self) -> None:
      if self._feature_tab is not None:
          self._feature_tab.sync_registry_exclusions()
      self.update_summary()

  def _apply_saved_prefs(self) -> None:
      if not self._chart_dir or self._feature_tab is None:
          return
      prefs = load_build_config_prefs(self._chart_dir) or {}
      build = prefs.get("build") if isinstance(prefs.get("build"), dict) else prefs
      if not isinstance(build, dict):
          return
      try:
          interval = int(build.get("sampling_interval_sec") or 0)
          if interval >= 3:
              self.interval_var.set(interval)
          try:
              stride = int(build.get("sliding_stride_sec") or 0)
              if stride >= 1:
                  self.stride_var.set(stride)
              else:
                  self.stride_var.set(interval if interval >= 3 else 10)
          except (TypeError, ValueError):
              pass
      except (TypeError, ValueError):
          pass
      if "low_memory" in build:
          self.low_memory_var.set(bool(build.get("low_memory")))
      if "build_profiler" in build:
          self.build_profiler_var.set(bool(build.get("build_profiler")))
      horizons = build.get("horizons_sec")
      if isinstance(horizons, list) and self._horizon_vars:
          hz_set = {int(h) for h in horizons if str(h).isdigit() or isinstance(h, int)}
          if hz_set:
              for h, var in self._horizon_vars.items():
                  var.set(h in hz_set)
          else:
              defaults = default_horizon_selection()
              for h, var in self._horizon_vars.items():
                  var.set(defaults.get(h, True))
      feat_cfg = default_feature_config(self._registry)
      saved_groups = build.get("enabled_groups")
      if isinstance(saved_groups, list):
          feat_cfg["enabledGroups"] = sorted(
              normalize_enabled_groups(self._registry, [str(g) for g in saved_groups]),
          )
      saved_features = build.get("enabled_features")
      if isinstance(saved_features, list) and saved_features:
          feat_cfg["enabledFeatures"] = [str(f) for f in saved_features]
      profile = build.get("feature_profile")
      if isinstance(profile, str) and profile:
          feat_cfg["profile"] = profile
      self._feature_tab.apply_config(feat_cfg)
      strike_cfg = default_strike_config()
      saved_strike = build.get("strike_selection")
      if isinstance(saved_strike, dict):
          strike_cfg = normalize_strike_config(saved_strike)
      if self._strike_tab is not None:
          self._strike_tab.apply_config(strike_cfg)
      gap_cfg = default_gap_policy()
      saved_gap = build.get("gap_policy")
      if isinstance(saved_gap, dict):
          gap_cfg = normalize_gap_policy(saved_gap)
      if self._gap_tab is not None:
          self._gap_tab.apply_config(gap_cfg)
      saved_project = build.get("feature_project_id")
      if isinstance(saved_project, str) and saved_project.strip() and not self._feature_project_locked:
          self._set_feature_project_combo_value(saved_project.strip().lower())
      else:
          self._populate_feature_project_combo()
          if self._feature_tab is not None:
              self._feature_tab.set_feature_project_id(self.feature_project_id())

  def refresh_master_project(self, master_status: dict[str, Any] | None) -> None:
      """Bind project dropdown to an existing master DB when present."""
      self._master_status = master_status
      locked_pid: str | None = None
      if isinstance(master_status, dict) and master_status.get("exists"):
          cfg = master_status.get("master_config")
          if isinstance(cfg, dict):
              raw = cfg.get("feature_project_id")
              if raw is not None and str(raw).strip():
                  locked_pid = str(raw).strip().lower()
          if not locked_pid:
              meta = master_status.get("master_meta")
              if isinstance(meta, dict):
                  raw = meta.get("feature_project_id")
                  if raw is not None and str(raw).strip():
                      locked_pid = str(raw).strip().lower()
          if not locked_pid and int(master_status.get("row_count") or 0) > 0:
              locked_pid = str(master_status.get("feature_project_id") or "").strip().lower() or None
      self._feature_project_locked = bool(locked_pid)
      self._populate_feature_project_combo()
      if locked_pid:
          self._set_feature_project_combo_value(locked_pid)
      elif self._feature_tab is not None:
          self._feature_tab.set_feature_project_id(self.feature_project_id())
      if hasattr(self, "_feature_project_cb"):
          self._feature_project_cb.configure(
              state="disabled" if self._feature_project_locked else "readonly",
          )

  def _set_feature_project_combo_value(self, project_id: str) -> None:
      pid = str(project_id or "").strip().lower()
      for label, mapped in self._feature_project_label_to_id.items():
          if mapped == pid:
              self._feature_project_var.set(label)
              if self._feature_tab is not None:
                  self._feature_tab.set_feature_project_id(pid)
              return
      self._feature_project_var.set(pid)
      if self._feature_tab is not None:
          self._feature_tab.set_feature_project_id(pid)

  def _populate_feature_project_combo(self) -> None:
      from chain_replay_ml.dataset_builder.feature_project_organization import (
          RESERVED_ALL_PROJECT_ID,
          is_reserved_all_project_id,
      )

      labels: list[str] = [RESERVED_ALL_PROJECT_ID]
      self._feature_project_label_to_id = {RESERVED_ALL_PROJECT_ID: RESERVED_ALL_PROJECT_ID}
      if self._chart_dir:
          from . import feature_registry_service as svc

          for p in svc.list_projects(self._chart_dir):
              pid = str(p.get("id") or "").strip()
              if not pid or is_reserved_all_project_id(pid):
                  continue
              name = str(p.get("label") or pid)
              label = f"{name} ({pid})"
              labels.append(label)
              self._feature_project_label_to_id[label] = pid
      if hasattr(self, "_feature_project_cb"):
          self._feature_project_cb["values"] = labels

  def feature_project_id(self) -> str:
      from chain_replay_ml.dataset_builder.feature_project_organization import RESERVED_ALL_PROJECT_ID

      label = str(self._feature_project_var.get() or "").strip()
      pid = self._feature_project_label_to_id.get(label, label).strip().lower()
      return pid or RESERVED_ALL_PROJECT_ID

  def _on_feature_project_selected(self) -> None:
      if self._feature_project_locked:
          return
      pid = self.feature_project_id()
      if self._feature_tab is not None:
          self._feature_tab.set_feature_project_id(pid)
      self._notify()

  def _persist_prefs(self) -> None:
      if not self._chart_dir or self._feature_tab is None:
          return
      feat = self._feature_tab.get_config()
      strike = self.strike_selection()
      gap = self.gap_policy()
      save_build_config_prefs(self._chart_dir, {
          "build": {
              "sampling_interval_sec": self.interval_sec(),
              "sliding_stride_sec": self.sliding_stride_sec(),
              "low_memory": self.low_memory(),
              "build_profiler": self.build_profiler(),
              "horizons_sec": self.horizons_sec(),
              "target_type": "future_ltp",
              "feature_profile": feat.get("profile"),
              "enabled_groups": feat.get("enabledGroups") or [],
              "enabled_features": feat.get("enabledFeatures") or [],
              "strike_selection": strike,
              "gap_policy": gap,
              "feature_project_id": self.feature_project_id(),
          },
      })

  def _build_ui(self) -> None:
      project_row = ttk.Frame(self)
      project_row.pack(fill="x", pady=(0, 6))
      ttk.Label(project_row, text="Feature Project").pack(side="left")
      self._feature_project_cb = ttk.Combobox(
          project_row,
          textvariable=self._feature_project_var,
          width=34,
          state="readonly",
      )
      self._feature_project_cb.pack(side="left", padx=(6, 12))
      self._feature_project_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_feature_project_selected())
      ttk.Label(
          project_row,
          text="Project identity for this master dataset (locked after first build).",
          foreground="#666",
      ).pack(side="left")

      row0 = ttk.Frame(self)
      row0.pack(fill="x", pady=(0, 6))
      ttk.Label(row0, text="Sampling interval (s)").pack(side="left")
      spin = ttk.Spinbox(
          row0, from_=3, to=60, textvariable=self.interval_var, width=5,
          command=self._on_sampling_changed,
      )
      spin.pack(side="left", padx=(6, 12))
      spin.bind("<KeyRelease>", lambda _e: self._on_sampling_changed())

      ttk.Label(row0, text="Sliding stride (s)").pack(side="left")
      stride_spin = ttk.Spinbox(
          row0, from_=1, to=60, textvariable=self.stride_var, width=5,
          command=self._on_sampling_changed,
      )
      stride_spin.pack(side="left", padx=(6, 12))
      stride_spin.bind("<KeyRelease>", lambda _e: self._on_sampling_changed())
      ttk.Label(
          row0,
          textvariable=self.stride_error_var,
          foreground="#C62828",
      ).pack(side="left", padx=(4, 0))

      row0_checks = ttk.Frame(self)
      row0_checks.pack(fill="x", pady=(0, 6))
      ttk.Checkbutton(
          row0_checks,
          text="Low memory (reserved)",
          variable=self.low_memory_var,
          command=self._notify,
      ).pack(side="left")
      ttk.Checkbutton(
          row0_checks,
          text="Build profiler (recommended)",
          variable=self.build_profiler_var,
          command=self._notify,
      ).pack(side="left", padx=(8, 0))

      notebook = ttk.Notebook(self)
      notebook.pack(fill="both", expand=True, pady=4)

      features_tab = ttk.Frame(notebook, padding=4)
      notebook.add(features_tab, text="Feature selection")
      self._feature_tab = FeatureSelectionTab(
          features_tab,
          self._registry,
          chart_dir=self._chart_dir,
          feature_project_id=self.feature_project_id(),
          on_change=self._notify,
          get_sampling_interval_sec=lambda: float(self.interval_sec()),
      )

      targets_tab = ttk.Frame(notebook, padding=4)
      notebook.add(targets_tab, text="Target labels")
      self._build_target_labels_tab(targets_tab)

      strike_tab = ttk.Frame(notebook, padding=4)
      notebook.add(strike_tab, text="Strike selection")
      self._strike_tab = StrikeSelectionTab(strike_tab, on_change=self._notify)

      gap_tab = ttk.Frame(notebook, padding=4)
      notebook.add(gap_tab, text="Gap policy")
      self._gap_tab = GapPolicyTab(gap_tab, on_change=self._notify)

      btn_row = ttk.Frame(self)
      btn_row.pack(fill="x", pady=4)
      self._build_btn = ttk.Button(btn_row, text="Build", command=self._on_build_click)
      self._build_btn.pack(side="left", padx=2)
      ttk.Button(btn_row, text="Build Summary", command=self._open_build_summary).pack(side="right", padx=2)
      ttk.Button(btn_row, text="Feature Policy…", command=self._open_feature_policy).pack(side="right", padx=2)

      self.summary_var = tk.StringVar(value="")
      ttk.Label(self, textvariable=self.summary_var, foreground="#888").pack(anchor="w")

  def _build_target_labels_tab(self, parent: ttk.Frame) -> None:
      type_row = ttk.Frame(parent)
      type_row.pack(fill="x", pady=(0, 6))
      ttk.Label(type_row, text="Target type:").pack(side="left")
      ttk.Label(type_row, text=TARGET_TYPE_LABEL, foreground="#333").pack(side="left", padx=(6, 0))

      cols = ttk.Frame(parent)
      cols.pack(fill="both", expand=True)

      sec_rows, min_rows = target_horizon_columns()
      defaults = default_horizon_selection()

      sec_col = ttk.LabelFrame(cols, text="Seconds", padding=4)
      sec_col.pack(side="left", fill="both", expand=True, padx=(0, 4))
      for row in sec_rows:
          sec = int(row["sec"])
          var = tk.BooleanVar(value=defaults.get(sec, True))
          self._horizon_vars[sec] = var
          ttk.Checkbutton(
              sec_col,
              text=row["display_name"],
              variable=var,
              command=self._on_horizon_toggle,
          ).pack(anchor="w")

      min_col = ttk.LabelFrame(cols, text="Minutes", padding=4)
      min_col.pack(side="left", fill="both", expand=True, padx=(4, 0))
      for row in min_rows:
          sec = int(row["sec"])
          var = tk.BooleanVar(value=defaults.get(sec, True))
          self._horizon_vars[sec] = var
          ttk.Checkbutton(
              min_col,
              text=row["display_name"],
              variable=var,
              command=self._on_horizon_toggle,
          ).pack(anchor="w")

  def _on_horizon_toggle(self) -> None:
      if not self.horizons_sec():
          messagebox.showwarning(
              "Target labels",
              "At least one prediction horizon must remain selected.",
              parent=self.winfo_toplevel(),
          )
          defaults = default_horizon_selection()
          for h, var in self._horizon_vars.items():
              var.set(defaults.get(h, True))
      self._notify()

  def _on_sampling_changed(self) -> None:
      interval = self.interval_sec()
      try:
          stride = int(self.stride_var.get() or interval)
      except (TypeError, ValueError):
          stride = interval
      if stride > interval:
          self.stride_var.set(interval)
      self._update_stride_validation()
      self._notify()

  def _update_stride_validation(self) -> None:
      err = self.stride_validation_error()
      self.stride_error_var.set(err or "")
      self._refresh_build_button_state()

  def stride_validation_error(self) -> str | None:
      from chain_replay_ml.dataset_builder.sliding_stride_policy import validate_sliding_stride

      return validate_sliding_stride(self.interval_sec(), self.sliding_stride_sec())

  def sliding_stride_sec(self) -> int:
      try:
          return max(1, int(self.stride_var.get() or self.interval_sec()))
      except (TypeError, ValueError):
          return self.interval_sec()

  def interval_sec(self) -> int:
      try:
          return max(3, int(self.interval_var.get() or 10))
      except (TypeError, ValueError):
          return 10

  def low_memory(self) -> bool:
      return bool(self.low_memory_var.get())

  def build_profiler(self) -> bool:
      return bool(self.build_profiler_var.get())

  def horizons_sec(self) -> list[int]:
      out = [h for h, var in self._horizon_vars.items() if var.get()]
      return sorted(out) if out else list(DEFAULT_HORIZON_SEC)

  def feature_selection(self) -> dict[str, Any]:
      if self._feature_tab is None:
          return default_feature_config(self._registry)
      return self._feature_tab.get_config()

  def strike_selection(self) -> dict[str, Any]:
      if self._strike_tab is None:
          return default_strike_config()
      return self._strike_tab.get_config()

  def gap_policy(self) -> dict[str, Any]:
      if self._gap_tab is None:
          return default_gap_policy()
      return self._gap_tab.get_config()

  def resolved_gap_max_sec(self) -> float:
      return gap_max_sec_from_policy(self.gap_policy())

  def resolved_gap_summary_label(self) -> str:
      return gap_summary_label(self.gap_policy())

  def resolved_strike_selection(self) -> dict[str, Any]:
      """Strike selection applied to master SQLite insert."""
      from .strike_selection_engine import strike_selection_for_master

      return strike_selection_for_master(self.strike_selection())

  def resolved_atm_band(self) -> int:
      from .strike_selection_engine import atm_band_from_strike_config

      return atm_band_from_strike_config(self.resolved_strike_selection())

  def resolved_strike_summary_label(self) -> str:
      from chain_replay_ml.dataset_builder.expected_spec import format_strike_selection_label

      return format_strike_selection_label(self.resolved_strike_selection()) or "—"

  def prediction_targets(self) -> dict[str, Any]:
      base = dict(default_master_prediction_targets())
      base["horizonsSec"] = self.horizons_sec()
      base["applied"] = True
      return base

  def sampling_config(self) -> dict[str, Any]:
      return {
          "configVersion": 1,
          "trainingIntervalSec": self.interval_sec(),
          "slidingStrideSec": self.sliding_stride_sec(),
          "samplingMethod": "fixed_interval",
          "applied": True,
      }

  def column_counts(self) -> tuple[int, int]:
      from .build_service import chart_data_dir

      data_dir = chart_data_dir(self._chart_dir) if self._chart_dir else None
      _, implemented, _, _ = resolve_implemented_features_for_selection(
          self.feature_selection(), self._registry, data_dir=data_dir,
      )
      return len(implemented), len(self.horizons_sec())

  def resolved_build_features(self) -> list[str]:
      """Feature columns that would be built — excludes registry-disabled features."""
      from .build_service import chart_data_dir

      data_dir = chart_data_dir(self._chart_dir) if self._chart_dir else None
      _, implemented, _, _ = resolve_implemented_features_for_selection(
          self.feature_selection(), self._registry, data_dir=data_dir,
      )
      return list(implemented)

  def update_summary(self, *, estimated_rows: int | None = None) -> None:
      feat_n, tgt_n = self.column_counts()
      tgt_labels = ", ".join(horizon_label(h) for h in self.horizons_sec())
      strike_lbl = self.resolved_strike_summary_label()
      gap_lbl = self.resolved_gap_summary_label()
      parts = [
          f"{feat_n} features",
          f"{tgt_n} targets ({tgt_labels})",
          f"Strikes: {strike_lbl}",
          f"Gap: {gap_lbl}",
      ]
      if estimated_rows is not None:
          parts.append(f"~{estimated_rows:,} rows")
      if self.low_memory():
          parts.append("low-memory mode")
      if self.build_profiler():
          parts.append("profiler on")
      self.summary_var.set(" · ".join(parts))

  def _refresh_build_button_state(self) -> None:
      if hasattr(self, "_build_btn"):
          blocked = self.stride_validation_error() is not None
          self._build_btn.configure(state="disabled" if blocked else "normal")

  def set_build_active(self, active: bool) -> None:
      if hasattr(self, "_build_btn"):
          blocked = active or self.stride_validation_error() is not None
          self._build_btn.configure(state="disabled" if blocked else "normal")

  def _on_build_click(self) -> None:
      err = self.stride_validation_error()
      if err:
          messagebox.showerror("Sliding stride", err, parent=self.winfo_toplevel())
          return
      if self._on_build:
          self._on_build()

  def _open_feature_policy(self) -> None:
      from .feature_policy_panel import open_feature_policy_window

      features = self.resolved_build_features()
      open_feature_policy_window(
          self,
          title="Dataset Feature Policy",
          feature_names=features,
          sampling_interval_sec=float(self.interval_sec()),
          gap_max_sec=float(self.resolved_gap_max_sec() or 20.0),
          chart_dir=self._chart_dir,
      )

  def _open_build_summary(self, *, on_proceed: Callable[[], None] | None = None) -> None:
      from .build_validation_panel import show_build_summary_dialog

      show_build_summary_dialog(
          self,
          feature_names=self.resolved_build_features(),
          sampling_interval_sec=float(self.interval_sec()),
          sliding_stride_sec=float(self.sliding_stride_sec()),
          strike_selection=self.resolved_strike_selection(),
          gap_policy=self.gap_policy(),
          prediction_targets=self.prediction_targets(),
          on_proceed=on_proceed if on_proceed is not None else self._on_build_click,
      )

  def _open_build_validation(self, *, on_proceed: Callable[[], None] | None = None) -> None:
      self._open_build_summary(on_proceed=on_proceed)
