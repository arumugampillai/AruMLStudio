"""Model Builder lifecycle UI — banner, locked fields, retrain compatibility."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from ..model_registry_widgets import ACCENT, COL_MUTED, COL_OK, COL_WARN, fmt_num
from . import service

_MODE_LABELS = {
    "retrain": "Retrain",
    "complete_optimization": "Complete Optimization",
    "feature_optimization": "Feature Optimization",
}

_MODE_HINTS = {
    "retrain": "Choose a compatible dataset (same target, strike band, sampling). Adjust features and walk-forward split if needed. Feature elimination stays None.",
    "complete_optimization": "Only hyperparameters will be optimized. Dataset, target, algorithm, validation, and features are locked.",
    "feature_optimization": "Adjust prediction type, features, and walk-forward elimination. Dataset, target, algorithm, and validation stay locked.",
}


def _target_label(name: str) -> str:
    n = str(name or "")
    if n.startswith("future_ltp_"):
        return f"Future LTP {n[len('future_ltp_'):]}"
    return n.replace("_", " ").title()


_WF_FEAT_LABELS = {
    "none": "None",
    "shap": "SHAP",
    "rfe": "RFE",
    "permutation": "Permutation",
}


def _wf_feat_label(val: str) -> str:
    return _WF_FEAT_LABELS.get(str(val or ""), str(val or "—"))


class LifecycleUiManager:
    """Sync lifecycle banner, locked displays, and retrain compatibility."""

    def __init__(
        self,
        *,
        data_dir: str,
        state: Any,
        on_change: Callable[[], None],
    ) -> None:
        self._data_dir = data_dir
        self.state = state
        self._on_change = on_change
        self.lifecycle_feature_snapshot: list[str] = []
        self.features_inspect_expanded = False
        self.retrain_compatible_datasets: list[dict[str, Any]] = []
        self.retrain_compatibility: dict[str, Any] | None = None
        self.retrain_required_profile: dict[str, Any] | None = None
        self._widgets: dict[str, Any] = {}

    def bind_widgets(self, widgets: dict[str, Any]) -> None:
        self._widgets = widgets

    def get_mode(self) -> str | None:
        if not (self.state.lifecycle or {}).get("source_model"):
            return None
        mode_var = self._widgets.get("lifecycle_mode_var")
        if mode_var is not None:
            return str(mode_var.get() or self.state.lifecycle_mode or "")
        return self.state.lifecycle_mode

    def uses_feature_snapshot(self) -> bool:
        mode = self.get_mode()
        if not mode:
            return False
        if mode in ("retrain", "complete_optimization"):
            return True
        if mode == "feature_optimization":
            feat_mode_var = self._widgets.get("lifecycle_feature_mode_var")
            if feat_mode_var is not None:
                return str(feat_mode_var.get()) == "locked"
            return self.state.lifecycle_feature_mode == "locked"
        return False

    def features_ui_locked(self) -> bool:
        """True when the feature tree must be read-only (complete optimization / locked feature opt)."""
        mode = self.get_mode()
        if mode == "complete_optimization":
            return True
        if mode == "feature_optimization":
            return self.uses_feature_snapshot()
        return False

    def training_features(self) -> list[str]:
        if self.features_ui_locked() and self.lifecycle_feature_snapshot:
            return sorted(self.lifecycle_feature_snapshot)
        return sorted(self.state.features)

    def is_retrain_mode(self) -> bool:
        return self.get_mode() == "retrain"

    def is_retrain_compatible(self) -> bool:
        if not self.is_retrain_mode():
            return True
        return (self.retrain_compatibility or {}).get("compatible") is True

    def clear(self) -> None:
        self.lifecycle_feature_snapshot = []
        self.features_inspect_expanded = False
        self.retrain_compatible_datasets = []
        self.retrain_compatibility = None
        self.retrain_required_profile = None
        self.state.lifecycle = None
        self.state.lifecycle_mode = None
        self.sync_ui()

    def apply_preset(self, preset_doc: dict[str, Any]) -> None:
        tc = preset_doc.get("training_config") or {}
        if not tc:
            raise ValueError("Missing training_config in lifecycle preset")
        mode = str(preset_doc.get("mode") or tc.get("lifecycle", {}).get("mode") or "complete_optimization")
        lc = dict(tc.get("lifecycle") or {})
        lc["mode"] = mode
        self.state.lifecycle = lc
        self.state.lifecycle_mode = mode

        feats = list(tc.get("features") or [])
        snap = list(lc.get("feature_snapshot") or feats)
        self.lifecycle_feature_snapshot = snap
        self.state.set_lifecycle_feature_snapshot(snap)
        self.state.features = set(snap)
        self.features_inspect_expanded = mode == "feature_optimization"

        if tc.get("dataset"):
            self.state.dataset = str(tc["dataset"])
        if tc.get("target"):
            self.state.target = str(tc["target"])
        if tc.get("prediction_type"):
            self.state.prediction_type = str(tc["prediction_type"])
        if tc.get("algorithm"):
            self.state.algorithm = str(tc["algorithm"])
        if tc.get("model_name"):
            self.state.model_name = str(tc["model_name"])
            self.state.model_name_manual = True
        if tc.get("model_version"):
            self.state.model_version = str(tc["model_version"])
        if tc.get("model_description"):
            self.state.model_description = str(tc["model_description"])

        split = tc.get("split") or {}
        if split.get("train") is not None:
            self.state.split_train = int(split["train"])
        if split.get("validation") is not None:
            self.state.split_val = int(split["validation"])
        if split.get("test") is not None:
            self.state.split_test = int(split["test"])
        vs = split.get("validation_strategy_ui") or split.get("strategy")
        if vs:
            key = str(vs)
            if key == "walk_forward" and not key.endswith("_split"):
                self.state.validation_strategy = "walk_forward"
            elif key == "rolling":
                self.state.validation_strategy = "rolling_window"
            elif key in ("time_series", "time_series_split"):
                self.state.validation_strategy = "time_series_split"
            else:
                self.state.validation_strategy = key

        wf = split.get("walk_forward") or {}
        if wf.get("n_folds") is not None:
            self.state.wf_folds = int(wf["n_folds"])
        if wf.get("window_mode"):
            self.state.wf_window_mode = str(wf["window_mode"])
        if wf.get("fold_placement"):
            self.state.wf_fold_placement = str(wf["fold_placement"])
        if wf.get("train_window_size") is not None:
            self.state.wf_train_window = int(wf["train_window_size"])
        if wf.get("validation_window_size") is not None:
            self.state.wf_val_window = int(wf["validation_window_size"])
        fs = "none" if mode == "retrain" else str(wf.get("feature_selection_method") or "rfe")
        self.state.wf_feature_selection = fs
        if wf.get("optimization_metric"):
            self.state.wf_opt_metric = str(wf["optimization_metric"])

        hpo = split.get("hyperparameter_optimization") or wf.get("hyperparameter_optimization") or {}
        self.state.global_hpo_enabled = bool(hpo.get("enabled"))
        if hpo.get("n_trials") is not None:
            self.state.global_hpo_trials = int(hpo["n_trials"])
        self.state.wf_hpo_enabled = bool(hpo.get("enabled")) if mode != "retrain" else False
        trials_var = self._widgets.get("lifecycle_hpo_trials_var")
        n_trials = int(hpo.get("n_trials") or 25)
        if trials_var is not None:
            trials_var.set(n_trials)

        params = tc.get("parameters") or lc.get("baseline_parameters") or {}
        for src, attr in (
            ("learning_rate", "xgb_lr"),
            ("n_estimators", "xgb_trees"),
            ("max_depth", "xgb_depth"),
            ("early_stopping_rounds", "xgb_early_stop"),
            ("random_seed", "xgb_seed"),
            ("subsample", "xgb_subsample"),
            ("colsample_bytree", "xgb_colsample"),
            ("min_child_weight", "xgb_min_child"),
            ("reg_alpha", "xgb_reg_alpha"),
            ("reg_lambda", "xgb_reg_lambda"),
        ):
            if params.get(src) is not None:
                setattr(self.state, attr, params[src])

        self.retrain_required_profile = lc.get("retrain_profile")
        mode_var = self._widgets.get("lifecycle_mode_var")
        if mode_var is not None:
            mode_var.set(mode)
        feat_mode_var = self._widgets.get("lifecycle_feature_mode_var")
        if feat_mode_var is not None and mode == "feature_optimization":
            feat_mode_var.set("optimize")

    def load_retrain_datasets(self, source_model: str, preferred: str | None = None) -> list[str]:
        doc = service.retrain_compatible(self._data_dir, source_model)
        self.retrain_required_profile = doc.get("required_profile") or self.retrain_required_profile
        self.retrain_compatible_datasets = list(doc.get("compatible_datasets") or [])
        names = [str(d.get("dataset_name") or "") for d in self.retrain_compatible_datasets if d.get("dataset_name")]
        pick = preferred if preferred in names else str(doc.get("default_dataset") or (names[0] if names else ""))
        if pick:
            cached = next((d for d in self.retrain_compatible_datasets if d.get("dataset_name") == pick), None)
            self.retrain_compatibility = (cached or {}).get("compatibility")
        return names

    def refresh_retrain_compatibility(self, dataset_name: str) -> None:
        lc = self.state.lifecycle or {}
        source = str(lc.get("source_model") or "")
        if not self.is_retrain_mode() or not source or not dataset_name:
            self.retrain_compatibility = None
            self._render_compat_panel()
            return
        cached = next(
            (d for d in self.retrain_compatible_datasets if d.get("dataset_name") == dataset_name),
            None,
        )
        if cached and cached.get("compatibility"):
            self.retrain_compatibility = cached["compatibility"]
        else:
            try:
                doc = service.retrain_compatibility(self._data_dir, source, dataset_name)
                self.retrain_compatibility = doc.get("compatibility")
                self.retrain_required_profile = doc.get("required_profile") or self.retrain_required_profile
            except Exception:
                self.retrain_compatibility = {"compatible": False, "score_pct": 0, "checks": []}
        self._render_compat_panel()

    def sync_ui(self) -> None:
        mode = self.get_mode()
        lc = self.state.lifecycle or {}
        banner = self._widgets.get("banner_frame")
        summary = self._widgets.get("summary_frame")
        if banner is not None:
            if mode and lc.get("source_model"):
                banner.pack(fill="x", padx=8, pady=(0, 4), before=self._widgets.get("paned"))
                self._render_banner(mode, lc)
            else:
                banner.pack_forget()
        if summary is not None:
            if mode and lc.get("source_model"):
                summary.pack(fill="x", padx=8, pady=(0, 4), before=self._widgets.get("paned"))
                self._render_pretrain_summary(mode, lc)
            else:
                summary.pack_forget()

        lifecycle_mode_panel = self._widgets.get("lifecycle_mode_panel")
        if lifecycle_mode_panel is not None:
            if mode:
                lifecycle_mode_panel.pack(fill="x", pady=6, before=self._widgets.get("train_btn"))
            else:
                lifecycle_mode_panel.pack_forget()

        self._set_locked("dataset", self._is_dataset_locked(mode))
        self._set_locked("target", bool(mode))
        self._set_locked("prediction_type", mode in ("retrain", "complete_optimization"))
        self._set_locked("algorithm", bool(mode))
        self._sync_data_split_sections(mode)
        self._set_locked("xgb", mode in ("retrain", "complete_optimization"))

        hpo_panel = self._widgets.get("lifecycle_hpo_panel")
        if hpo_panel is not None:
            if mode == "complete_optimization":
                hpo_panel.pack(fill="x", pady=6, before=self._widgets.get("xgb_editable"))
            else:
                hpo_panel.pack_forget()

        self._sync_lifecycle_training_controls(mode)

        compat = self._widgets.get("compat_frame")
        if compat is not None:
            if self.is_retrain_mode():
                compat.pack(fill="x", pady=6)
                self._render_compat_panel()
            else:
                compat.pack_forget()

        feat_summary = self._widgets.get("feat_lifecycle_summary")
        feat_mode_panel = self._widgets.get("feat_mode_panel")
        feat_controls = self._widgets.get("feat_controls")
        feat_details = self._widgets.get("feat_details")
        locked = self.features_ui_locked()
        if feat_summary is not None:
            if mode and locked and not self.features_inspect_expanded:
                feat_summary.pack(fill="x", pady=4)
                self._render_feature_summary(lc)
            else:
                feat_summary.pack_forget()
        if feat_mode_panel is not None:
            if mode == "feature_optimization":
                feat_mode_panel.pack(fill="x", pady=4)
            else:
                feat_mode_panel.pack_forget()
        if feat_controls is not None and feat_details is not None:
            if mode and locked and not self.features_inspect_expanded:
                feat_controls.pack_forget()
                feat_details.pack_forget()
            elif locked and self.features_inspect_expanded:
                feat_controls.pack_forget()
                feat_details.pack(fill="both", expand=True, pady=4)
            else:
                feat_controls.pack(fill="x", pady=4)
                show_var = self._widgets.get("show_feat_var")
                if show_var and show_var.get():
                    feat_details.pack(fill="both", expand=True, pady=4)
                else:
                    feat_details.pack_forget()

        name_entry = self._widgets.get("name_entry")
        version_entry = self._widgets.get("version_entry")
        if name_entry is not None:
            name_entry.configure(state="disabled" if mode else "normal")
        if version_entry is not None:
            version_entry.configure(state="disabled" if mode else "normal")

        ds_cb = self._widgets.get("dataset_cb")
        if ds_cb is not None:
            ds_cb.configure(state="readonly" if mode else "readonly")

        global_hpo = self._widgets.get("global_hpo_frame")
        if global_hpo is not None:
            if mode:
                global_hpo.pack_forget()
            else:
                global_hpo.pack(fill="x", pady=4)

        self._update_locked_values()
        self._render_champion_grid()
        apply_vis = self._widgets.get("apply_val_strategy_visibility")
        if apply_vis:
            apply_vis()
        self._on_change()

    def _sync_data_split_sections(self, mode: str | None) -> None:
        locked_fr = self._widgets.get("data_split_locked")
        editable_fr = self._widgets.get("data_split_editable")
        feat_opt = mode == "feature_optimization"
        retrain = mode == "retrain"
        core_locked = bool(mode) and not feat_opt and not retrain
        if feat_opt:
            if locked_fr is not None:
                locked_fr.pack(fill="x", pady=(0, 4))
            if editable_fr is not None:
                editable_fr.pack(fill="x")
            self._set_var(
                "split_locked_sub",
                "Validation strategy and walk-forward folds stay fixed. Adjust feature elimination and optimization metric below.",
            )
            self._pack_split_locked_only(False)
            wfe_hint = self._widgets.get("lifecycle_wfe_hint")
            if wfe_hint is not None:
                wfe_hint.pack(anchor="w", pady=(0, 6))
        elif retrain:
            if locked_fr is not None:
                locked_fr.pack_forget()
            if editable_fr is not None:
                editable_fr.pack(fill="x")
            self._pack_split_locked_only(True)
            wfe_hint = self._widgets.get("lifecycle_wfe_hint")
            if wfe_hint is not None:
                wfe_hint.configure(
                    text=(
                        "Data split is editable for retrain (e.g. distributed folds on a new compatible dataset). "
                        "Feature elimination is fixed to None — the locked feature snapshot from the source model is used."
                    ),
                )
                wfe_hint.pack(anchor="w", pady=(0, 6))
        elif core_locked:
            if locked_fr is not None:
                locked_fr.pack(fill="x", pady=(0, 4))
            if editable_fr is not None:
                editable_fr.pack_forget()
            self._set_var(
                "split_locked_sub",
                "Validation strategy and walk-forward configuration are fixed for this lifecycle run.",
            )
            self._pack_split_locked_only(False)
            wfe_hint = self._widgets.get("lifecycle_wfe_hint")
            if wfe_hint is not None:
                wfe_hint.pack_forget()
        else:
            if locked_fr is not None:
                locked_fr.pack_forget()
            if editable_fr is not None:
                editable_fr.pack(fill="x")
            wfe_hint = self._widgets.get("lifecycle_wfe_hint")
            if wfe_hint is not None:
                wfe_hint.pack_forget()
            self._pack_split_locked_only(True)

        elim_locked = self._widgets.get("wf_feat_elim_locked")
        elim_edit = self._widgets.get("wf_feat_elim_editable")
        if elim_locked is not None:
            if retrain:
                elim_locked.pack(fill="x", pady=(8, 4))
            else:
                elim_locked.pack_forget()
        if elim_edit is not None:
            if retrain:
                elim_edit.pack_forget()
            else:
                elim_edit.pack(fill="x")
        if retrain:
            feat_var = self._widgets.get("wf_feat_sel_var")
            if feat_var is not None:
                feat_var.set("none")
            self.state.wf_feature_selection = "none"
        elif mode != "feature_optimization":
            wfe_hint = self._widgets.get("lifecycle_wfe_hint")
            if wfe_hint is not None and mode != "retrain":
                wfe_hint.pack_forget()

    def _pack_split_locked_only(self, show: bool) -> None:
        for key in ("split_tss_panel", "wf_folds_grid", "wf_preview_panel", "wf_hpo_section"):
            fr = self._widgets.get(key)
            if fr is None:
                continue
            if show:
                fr.pack(fill="x")
            else:
                fr.pack_forget()

    def _sync_lifecycle_training_controls(self, mode: str | None) -> None:
        if not mode or not self.state.lifecycle:
            return
        global_hpo = self._widgets.get("global_hpo_var")
        wf_hpo = self._widgets.get("wf_hpo_var")
        feat_sel = self._widgets.get("wf_feat_sel_var")
        lifecycle_trials = self._widgets.get("lifecycle_hpo_trials_var")
        if mode == "complete_optimization":
            if global_hpo is not None:
                global_hpo.set(True)
            if feat_sel is not None:
                feat_sel.set("none")
            trials = int(lifecycle_trials.get()) if lifecycle_trials is not None else self.state.global_hpo_trials
            self.state.global_hpo_enabled = True
            self.state.global_hpo_resume = False
            self.state.global_hpo_trials = trials
            self.state.wf_feature_selection = "none"
            if self.state.validation_strategy in ("walk_forward", "rolling_window"):
                self.state.wf_hpo_enabled = True
                self.state.wf_hpo_trials = trials
            else:
                self.state.wf_hpo_enabled = False
        elif mode == "feature_optimization":
            if global_hpo is not None:
                global_hpo.set(False)
            if wf_hpo is not None:
                wf_hpo.set(False)
            self.state.global_hpo_enabled = False
            self.state.wf_hpo_enabled = False
        elif mode == "retrain":
            if global_hpo is not None:
                global_hpo.set(False)
            if wf_hpo is not None:
                wf_hpo.set(False)
            self.state.global_hpo_enabled = False
            self.state.wf_hpo_enabled = False
            self.state.wf_feature_selection = "none"

    def _render_champion_grid(self) -> None:
        host = self._widgets.get("xgb_champion_grid")
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        mode = self.get_mode()
        if mode not in ("retrain", "complete_optimization"):
            return
        lc = self.state.lifecycle or {}
        params = lc.get("baseline_parameters") or {}
        rows = (
            ("Learning Rate", params.get("learning_rate", self.state.xgb_lr)),
            ("Max Depth", params.get("max_depth", self.state.xgb_depth)),
            ("Trees", params.get("n_estimators", self.state.xgb_trees)),
            ("Subsample", params.get("subsample", self.state.xgb_subsample)),
            ("Colsample", params.get("colsample_bytree", self.state.xgb_colsample)),
        )
        grid = ttk.Frame(host)
        grid.pack(fill="x")
        for i, (label, val) in enumerate(rows):
            cell = ttk.Frame(grid)
            cell.grid(row=i // 3, column=i % 3, padx=6, pady=2, sticky="w")
            ttk.Label(cell, text=label, foreground=COL_MUTED, font=("Segoe UI", 8)).pack(anchor="w")
            ttk.Label(cell, text=str(val) if val is not None else "—", font=("Segoe UI", 9, "bold")).pack(anchor="w")

    def _is_dataset_locked(self, mode: str | None) -> bool:
        return mode in ("complete_optimization", "feature_optimization")

    def _set_locked(self, section: str, locked: bool) -> None:
        locked_fr = self._widgets.get(f"{section}_locked")
        editable_fr = self._widgets.get(f"{section}_editable")
        if locked_fr is not None:
            if locked:
                locked_fr.pack(fill="x", pady=(0, 4))
            else:
                locked_fr.pack_forget()
        if editable_fr is not None:
            if locked:
                editable_fr.pack_forget()
            else:
                editable_fr.pack(fill="x")

    def _update_locked_values(self) -> None:
        lc = self.state.lifecycle or {}
        ds_val = self._widgets.get("dataset_var")
        self._set_var("dataset_locked_val", ds_val.get() if ds_val else self.state.dataset)
        self._set_var("target_locked_val", _target_label(self.state.target))
        self._set_var("pred_locked_val", self.state.prediction_type.replace("_", " ").title())
        algo = self.state.algorithm
        self._set_var("algo_locked_val", algo.upper() if algo else "—")
        self._set_var("split_locked_val", self._format_split_locked())
        params = lc.get("baseline_parameters") or {}
        lines = [
            f"LR {params.get('learning_rate', self.state.xgb_lr)}",
            f"Depth {params.get('max_depth', self.state.xgb_depth)}",
            f"Trees {params.get('n_estimators', self.state.xgb_trees)}",
        ]
        self._set_var("xgb_locked_val", " · ".join(str(x) for x in lines))

    def _set_var(self, key: str, value: str) -> None:
        var = self._widgets.get(key)
        if var is not None:
            var.set(value)

    def _format_split_locked(self) -> str:
        vs = self.state.validation_strategy
        if vs in ("walk_forward", "rolling_window"):
            label = "Rolling Window" if vs == "rolling_window" else "Walk Forward"
            return (
                f"{label} · {self.state.wf_folds} folds · {self.state.wf_window_mode} window · "
                f"{self.state.wf_fold_placement} placement · "
                f"feature elim {_wf_feat_label(self.state.wf_feature_selection)}"
            )
        return f"Time Series Split · {self.state.split_train}% / {self.state.split_val}% / {self.state.split_test}%"

    def _render_banner(self, mode: str, lc: dict[str, Any]) -> None:
        text = self._widgets.get("banner_text")
        if text is None:
            return
        src = lc.get("source_model") or "—"
        ver = lc.get("source_model_version") or ""
        family = lc.get("family_model_name") or ""
        next_ver = lc.get("next_version_label") or ""
        lines = [
            f"Lifecycle Mode — {_MODE_LABELS.get(mode, mode)}",
            f"Source: {src}" + (f" (v{ver})" if ver else ""),
        ]
        if family:
            lines.append(f"Model family: {family}")
        if next_ver:
            lines.append(f"Next version: {next_ver}")
        profile = self.retrain_required_profile or lc.get("retrain_profile") or {}
        if mode == "retrain" and profile.get("source_dataset_missing"):
            src_ds = profile.get("source_dataset") or "original dataset"
            lines.append(
                f"Note: {src_ds} is no longer available — choose a compatible dataset below."
            )
        lines.append(_MODE_HINTS.get(mode, ""))
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("end", "\n".join(lines))
        text.configure(state="disabled")

    def _render_pretrain_summary(self, mode: str, lc: dict[str, Any]) -> None:
        text = self._widgets.get("summary_text")
        if text is None:
            return
        diff = lc.get("lifecycle_diff") or {"same": [], "changes": []}
        feat_count = len(self.lifecycle_feature_snapshot)
        lines = [f"Lifecycle Summary — {_MODE_LABELS.get(mode, mode)}", ""]
        src_metrics = lc.get("source_metrics") or {}
        if src_metrics.get("composite_score") is not None:
            lines.append(
                f"Source metrics: composite {fmt_num(src_metrics.get('composite_score'))} · "
                f"MAE {fmt_num(src_metrics.get('mae'), 2)} · "
                f"dir {fmt_num(src_metrics.get('directional_accuracy_pct'), 2)}%"
            )
        if mode == "retrain":
            lines.extend([
                "",
                "Editable: Dataset (compatible only), Data split / walk-forward",
                "",
                "Locked:",
            ])
            for item in (
                "Target", "Strike selection", "Sampling", "Prediction type", "Algorithm",
                "Feature snapshot", "Feature elimination (None only)", "Hyperparameters",
            ):
                lines.append(f"  · {item}")
        else:
            same = diff.get("same") or []
            changes = diff.get("changes") or []
            if same:
                lines.extend(["", "Will remain same:"])
                for item in same:
                    label = item
                    if str(item).startswith("Feature Set") and feat_count:
                        label = f"Feature Set ({feat_count})"
                    lines.append(f"  ✓ {label}")
            if changes:
                lines.extend(["", "Will change:"])
                for item in changes:
                    lines.append(f"  → {item}")
        if lc.get("feature_snapshot_hash"):
            lines.append(f"\nSnapshot hash: {lc.get('feature_snapshot_hash')}")
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("end", "\n".join(lines))
        text.configure(state="disabled")

    def _render_feature_summary(self, lc: dict[str, Any]) -> None:
        host = self._widgets.get("feat_summary_grid")
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        n = len(self.lifecycle_feature_snapshot)
        ttk.Label(host, text=f"🔒 Locked — {n} features from source model", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Label(host, text=f"Method: {lc.get('selection_method') or 'Source snapshot'}", foreground=COL_MUTED).pack(anchor="w")
        btn_row = ttk.Frame(host)
        btn_row.pack(anchor="w", pady=4)
        ttk.Button(btn_row, text="Show Features", command=self._expand_features).pack(side="left")

    def _expand_features(self) -> None:
        self.features_inspect_expanded = True
        self.sync_ui()

    def _render_compat_panel(self) -> None:
        host = self._widgets.get("compat_checks")
        footer = self._widgets.get("compat_footer")
        chips = self._widgets.get("compat_chips")
        if host is None:
            return
        for w in host.winfo_children():
            w.destroy()
        if chips is not None:
            for w in chips.winfo_children():
                w.destroy()
        profile = self.retrain_required_profile or (self.state.lifecycle or {}).get("retrain_profile") or {}
        if chips is not None and profile:
            chip_texts = []
            if profile.get("target"):
                chip_texts.append(f"✓ {_target_label(profile['target'])}")
            if profile.get("strike_selection_label"):
                chip_texts.append(f"✓ {profile['strike_selection_label']}")
            if profile.get("sampling_interval_label"):
                chip_texts.append(f"✓ {profile['sampling_interval_label']}")
            if profile.get("prediction_type"):
                chip_texts.append(f"✓ {profile['prediction_type']}")
            for txt in chip_texts:
                ttk.Label(chips, text=txt, foreground=COL_OK).pack(side="left", padx=4)
        compat = self.retrain_compatibility or {}
        for check in compat.get("checks") or []:
            icon = "✓" if check.get("passed") else "✗"
            color = COL_OK if check.get("passed") else COL_WARN
            row = ttk.Frame(host)
            row.pack(fill="x")
            ttk.Label(row, text=f"{icon} {check.get('label', '')}", foreground=color).pack(anchor="w")
            if not check.get("passed") and check.get("expected") is not None:
                ttk.Label(
                    row,
                    text=f"Expected: {check.get('expected')} · Got: {check.get('actual', '—')}",
                    foreground=COL_MUTED,
                    font=("Segoe UI", 8),
                ).pack(anchor="w", padx=12)
        if footer is not None:
            ok = compat.get("compatible") is True
            footer.configure(
                text=(
                    f"Compatibility {compat.get('score_pct', 100 if ok else 0)}% · "
                    f"{'Retraining allowed' if ok else 'Retraining disabled'}"
                ),
                foreground=COL_OK if ok else COL_WARN,
            )
