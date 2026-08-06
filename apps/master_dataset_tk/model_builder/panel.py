"""Create Model panel — Tk parity with web Model Builder (new model flow)."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from ..build_service import chart_data_dir
from ..model_registry_widgets import ACCENT, COL_MUTED, ScrollableFrame, fmt_num, fmt_rows
from . import service
from .feature_preset import apply_feature_preset, clear_feature_preset, load_feature_preset, save_feature_preset
from .feature_tree import FeatureSelectionTree
from .lifecycle_ui import LifecycleUiManager
from .runner import ModelTrainingRunner
from .state import ModelBuilderState, _strip_lifecycle_from_saved, load_persisted_state, save_persisted_state
from ..lazy_panel import LazyLoadMixin
from .training_panel import ModelTrainingPanel
from .wf_preview import compute_walk_forward_preview_plan

def _algorithms_catalog() -> list[tuple[str, str]]:
    try:
        from chain_replay_ml.training.trainers import supported_algorithms

        return list(supported_algorithms())
    except Exception:
        return [
            ("xgboost", "XGBoost"),
            ("lightgbm", "LightGBM"),
            ("catboost", "CatBoost"),
            ("random_forest", "Random Forest"),
            ("extra_trees", "Extra Trees"),
        ]


_ALGORITHMS = tuple(_algorithms_catalog())
_VAL_STRATEGIES = (
    ("time_series_split", "Time Series Split"),
    ("walk_forward", "Walk Forward"),
    ("rolling_window", "Rolling Window"),
)
_WF_FEATURE_SEL = (
    ("none", "None"),
    ("shap", "SHAP Importance"),
    ("rfe", "Recursive Feature Elimination"),
    ("permutation", "Permutation Importance"),
)
_WF_OPT_METRICS = (
    ("composite", "Composite"),
    ("rmse", "Lowest RMSE"),
    ("mae", "Lowest MAE"),
    ("directional_accuracy", "Highest Direction %"),
    ("accuracy", "Highest Accuracy"),
    ("f1", "Highest F1"),
    ("auto", "Auto"),
)
_REGISTRY_TOP_OPTS = ("25", "50", "75", "100", "125", "all")
_ALGO_PARAM_TITLES = {
    "xgboost": "7 — XGBoost Parameters",
    "lightgbm": "7 — LightGBM Parameters",
    "catboost": "7 — CatBoost Parameters",
    "random_forest": "7 — Random Forest Parameters",
    "extra_trees": "7 — Extra Trees Parameters",
}
_HPO_ALGORITHMS = frozenset({"xgboost"})


def _target_label(name: str) -> str:
    n = str(name or "")
    if n == "label_id":
        return "Label id (encoded class)"
    if n == "label_name":
        return "Label name"
    if n.startswith("future_ltp_"):
        hor = n[len("future_ltp_") :]
        return f"Future LTP {hor}"
    if n in ("target_reached", "hit"):
        return "Hit (target reached)"
    if n.startswith("ormp_return_") and n.endswith("_points"):
        mid = n[len("ormp_return_") : -len("_points")]
        return f"ORMP return {mid} (points)"
    if n.startswith("ormp_return_") and n.endswith("_percent"):
        mid = n[len("ormp_return_") : -len("_percent")]
        return f"ORMP return {mid} (percent)"
    if n.startswith("ormp_direction_"):
        mid = n[len("ormp_direction_") :]
        return f"ORMP direction {mid}"
    if n.startswith("label_up_"):
        # label_up_2pct_5m → Up ≥2% (5m); label_up_gt6pct_5m → Up >6% (5m)
        body = n[len("label_up_") :]
        if body.startswith("gt") and body.endswith("_5m"):
            pct = body[len("gt") : -len("_5m")].replace("pct", "")
            return f"Up >{pct}% (5m)"
        if body.endswith("_5m"):
            pct = body[: -len("_5m")].replace("pct", "")
            return f"Up ≥{pct}% (5m)"
        return n.replace("_", " ")
    return n.replace("_", " ").title()


def _prediction_target_columns(meta: dict[str, Any], expected: dict[str, Any] | None = None) -> list[str]:
    cols = list(
        meta.get("prediction_target_columns")
        or meta.get("target_columns")
        or meta.get("targets")
        or []
    )
    if not cols and expected:
        cols = list(
            expected.get("prediction_target_columns")
            or expected.get("target_column_names")
            or []
        )
    if not cols:
        raw = meta.get("prediction_targets") or []
        if isinstance(raw, list):
            from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name

            for h in raw:
                hs = str(h)
                if hs.startswith("future") or hs.startswith("label_up_"):
                    cols.append(hs)
                    continue
                try:
                    sec = int(h)
                except (TypeError, ValueError):
                    continue
                cols.append(horizon_column_name(sec))
    if not cols and isinstance(meta.get("columns"), list):
        cols = [
            str(c)
            for c in meta["columns"]
            if str(c).startswith("future_ltp")
            or str(c).startswith("label_up_")
            or str(c) in ("target_reached", "hit")
        ]
    return list(dict.fromkeys(str(c) for c in cols if c))


def _dataset_display_label(row: dict[str, Any]) -> str:
    name = str(row.get("dataset_name") or "").strip()
    if not name:
        return ""
    count = row.get("row_count")
    suffix = ""
    if row.get("needs_parquet"):
        suffix = " · needs Parquet export"
    if count is None:
        return f"{name}{suffix}"
    try:
        base = f"{name} ({fmt_rows(int(count))})"
    except (TypeError, ValueError):
        base = name
    return f"{base}{suffix}"


def _dataset_name_from_label(value: str, datasets: list[dict[str, Any]]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for row in datasets:
        name = str(row.get("dataset_name") or "")
        if text == name or text == _dataset_display_label(row):
            return name
    if " (" in text:
        return text.rsplit(" (", 1)[0].strip()
    return text


class CreateModelPanel(ttk.Frame, LazyLoadMixin):
    """Full in-app Model Builder for new models (lifecycle modes: phase 2)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_registry: Callable[[str], None] | None = None,
        on_title_changed: Callable[[], None] | None = None,
        on_open_outcome_label_engine: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_registry = on_open_registry
        self._on_title_changed = on_title_changed
        self._on_open_outcome_label_engine = on_open_outcome_label_engine
        self._data_dir = chart_data_dir(chart_dir)
        self.state = ModelBuilderState()
        self._datasets: list[dict[str, Any]] = []
        self._dataset_meta: dict[str, Any] | None = None
        self._schema: dict[str, Any] | None = None
        self._models: list[dict[str, Any]] = []
        self._fr_projects: list[dict[str, Any]] = []
        self._feature_vars: dict[str, tk.BooleanVar] = {}
        self._group_vars: dict[str, tk.BooleanVar] = {}
        self._target_var = tk.StringVar(value="")
        self._label_strategy_var = tk.StringVar(value="fixed_horizon")
        self._label_strategy_param_vars: dict[str, tk.Variable] = {}
        self._building_features = False
        self._pending_feature_preset: dict[str, Any] | None = None
        self._pending_hit_confidence: dict[str, Any] | None = None
        self._ignore_name_trace = False
        self._skip_reset_on_show = False
        self._panel_title_var = tk.StringVar(value="Create Model")
        self._lifecycle = LifecycleUiManager(
            data_dir=self._data_dir,
            state=self.state,
            on_change=lambda: self._on_change(),
        )
        self._build_ui()
        self._lazy_init()
        self._show_builder()

    def page_title(self) -> str:
        lc = self.state.lifecycle or {}
        if lc.get("source_model"):
            return str(lc.get("family_model_name") or lc.get("source_model") or "Create Model")
        return "Create Model"

    def prepare_lifecycle_open(self) -> None:
        """Skip fresh-model reset when navigating from Model Registry lifecycle actions."""
        self._skip_reset_on_show = True

    def on_show(self) -> None:
        if not self._skip_reset_on_show:
            self._reset_lifecycle_for_new_model()
        self._skip_reset_on_show = False
        stored = load_feature_preset(self.chart_dir)
        if stored:
            self._pending_feature_preset = stored
        self._load_catalog(lazy=True)
        self._sync_panel_title()

    def begin_new_model(self) -> None:
        """Fresh new-model flow when Create Model is clicked while already on this page."""
        self._show_builder()
        self._reset_lifecycle_for_new_model()
        stored = load_feature_preset(self.chart_dir)
        if stored:
            self._pending_feature_preset = stored
        self._load_catalog(lazy=True)

    def _reset_lifecycle_for_new_model(self) -> None:
        """Sidebar Create Model opens a fresh new-model flow (not a prior retrain session)."""
        self._lifecycle.clear()
        self.state.lifecycle_feature_mode = "optimize"
        self.state.lifecycle_mode = None
        self.state.model_name = ""
        self.state.model_name_manual = False
        self.state.package_anchor = None
        # Keep radio default for the next lifecycle session; get_mode() ignores it with no source model.
        self._lifecycle_mode_var.set("complete_optimization")
        self._lifecycle_feature_mode_var.set("optimize")
        self._set_model_name_field("")
        self._render_preset_banner()
        self._knowledge_banner.pack_forget()
        self._sync_panel_title()

    def _sync_panel_title(self) -> None:
        self._panel_title_var.set(self.page_title())
        if self._on_title_changed:
            self._on_title_changed()

    def _set_model_name_field(self, value: str) -> None:
        self._ignore_name_trace = True
        try:
            self._name_var.set(value)
        finally:
            self._ignore_name_trace = False

    def _refresh_auto_model_name(self, *, force: bool = False) -> None:
        if self.state.lifecycle and self.state.lifecycle_mode:
            family = (self.state.lifecycle or {}).get("family_model_name") or (self.state.lifecycle or {}).get("source_model")
            if family:
                self._set_model_name_field(str(family))
            ver = (self.state.lifecycle or {}).get("next_version_label")
            if ver:
                self._version_var.set(str(ver))
            return
        if not force and self.state.model_name_manual:
            return
        suggested = self.state.suggest_model_name()
        if suggested:
            self.state.model_name_manual = False
            self._set_model_name_field(suggested)
            self.state.model_name = suggested

    def _dataset_combo_values(self) -> list[str]:
        return [_dataset_display_label(d) for d in self._datasets if d.get("dataset_name")]

    def _set_dataset_combo(self, dataset_name: str) -> None:
        row = next((d for d in self._datasets if d.get("dataset_name") == dataset_name), None)
        self._dataset_var.set(_dataset_display_label(row) if row else dataset_name)

    def _selected_dataset_name(self) -> str:
        return _dataset_name_from_label(self._dataset_var.get(), self._datasets)

    def _refresh_dataset_combo(self, datasets: list[dict[str, Any]] | None = None) -> None:
        if datasets is not None:
            self._datasets = datasets
        self._dataset_cb["values"] = self._dataset_combo_values()

    def import_feature_preset(
        self,
        *,
        features: list[str],
        dataset: str | None = None,
        source_model: str | None = None,
        analysis_feature_selection: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> None:
        """Apply a feature preset from Registry or dataset workflows."""
        if persist:
            save_feature_preset(
                self.chart_dir,
                features=features,
                dataset=dataset,
                source_model=source_model,
                analysis_feature_selection=analysis_feature_selection,
            )
        self._pending_feature_preset = {
            "features": list(features),
            "dataset": dataset,
            "source_model": source_model,
            "analysis_feature_selection": (
                dict(analysis_feature_selection)
                if isinstance(analysis_feature_selection, dict)
                else None
            ),
        }
        if isinstance(analysis_feature_selection, dict) and analysis_feature_selection:
            self.state.analysis_feature_selection = dict(analysis_feature_selection)
        self._show_builder()
        if not self._datasets:
            # Catalog apply will call _try_apply_pending_feature_preset
            self._load_catalog(lazy=True)
            self._update_feature_knowledge_hint(features)
            return
        names = [d.get("dataset_name") for d in self._datasets]
        # Only auto-select when the name is a real Model Builder dataset
        if dataset and dataset in names:
            self._set_dataset_combo(dataset)
            self._on_dataset_changed()
        elif self.state.dataset and self.state.dataset in names:
            self._try_apply_pending_feature_preset()
        else:
            self._render_preset_banner(
                pending=True,
                requested=len(features),
                preset_dataset=dataset,
            )
            # Still try apply against current/empty selection
            self._try_apply_pending_feature_preset()
        self._update_feature_knowledge_hint(features)

    def _update_feature_knowledge_hint(self, features: list[str]) -> None:
        try:
            from chain_replay_ml.fold_research import get_feature_knowledge

            hints = get_feature_knowledge(self._data_dir, features)
        except Exception:
            self._knowledge_banner.pack_forget()
            return
        if not hints:
            self._knowledge_banner.pack_forget()
            return
        lines = []
        for h in hints[:4]:
            gain = h.get("average_pf_gain")
            gain_txt = f" · avg PF +{gain}" if gain is not None and gain > 0 else ""
            lines.append(
                f"• {h.get('feature')}: evaluated in {h.get('evaluated_experiments')} experiments{gain_txt} "
                f"({h.get('status')}, {h.get('confidence_pct')}% confidence)",
            )
        self._knowledge_banner_var.set("\n".join(lines))
        self._pack_builder_banner(self._knowledge_banner)

    def _pack_builder_banner(self, banner: tk.Misc) -> None:
        """Pack a top banner safely even when Lifecycle Mode is not shown."""
        kwargs: dict[str, Any] = {"fill": "x", "padx": 8, "pady": (0, 4)}
        for sibling in (self._lifecycle_banner, getattr(self, "_main_paned", None)):
            if sibling is None:
                continue
            try:
                if sibling.winfo_manager():
                    kwargs["before"] = sibling
                    break
            except tk.TclError:
                continue
        banner.pack(**kwargs)

    def open_with_dataset(self, dataset_name: str) -> None:
        from chain_replay_ml.dataset_builder.writer import _safe_filename

        self._show_builder()
        if hasattr(self, "cancel_lazy_load"):
            self.cancel_lazy_load()
        self._load_catalog(lazy=False)
        safe = _safe_filename(dataset_name)
        names = [d.get("dataset_name") for d in self._datasets]
        pick = safe if safe in names else (dataset_name if dataset_name in names else "")
        if pick:
            self._set_dataset_combo(pick)
            self._on_dataset_changed()
            row = next((d for d in self._datasets if d.get("dataset_name") == pick), None)
            if row and row.get("needs_parquet"):
                messagebox.showwarning(
                    "Parquet required",
                    f"Dataset \"{pick}\" is registered but has no Parquet file yet.\n\n"
                    "Open Master Dataset, select your days, and use "
                    "\"Export to registry\" before training.",
                )
        elif dataset_name:
            self.state.dataset = safe or dataset_name
            self._set_dataset_combo(safe or dataset_name)
            messagebox.showwarning(
                "Dataset not found",
                f"\"{dataset_name}\" is not available for training yet.\n\n"
                "If you just exported it, click Refresh datasets.\n"
                "Otherwise export from Master Dataset to the registry "
                "(creates Parquet + metadata).",
            )

    def open_package_classifier(
        self,
        *,
        dataset_name: str,
        target: str,
        features: list[str] | None = None,
        source_model: str | None = None,
        ladder_label: str | None = None,
    ) -> None:
        """Prefill Model Builder for a Probability Ladder classifier (+2% … >6%)."""
        from chain_replay_ml.dataset_builder.writer import _safe_filename

        target_name = str(target or "").strip()
        if not target_name:
            messagebox.showwarning(
                "Probability Ladder",
                "Missing classifier target.",
                parent=self.winfo_toplevel(),
            )
            return

        self._show_builder()
        self._reset_lifecycle_for_new_model()
        if hasattr(self, "cancel_lazy_load"):
            self.cancel_lazy_load()
        self._pending_hit_confidence = None

        feat_list = [str(f) for f in (features or []) if f]
        if feat_list:
            self._pending_feature_preset = {
                "features": feat_list,
                "dataset": dataset_name,
                "source_model": source_model,
            }
        else:
            self._pending_feature_preset = None

        self._load_catalog(lazy=False)
        safe = _safe_filename(dataset_name)
        names = [d.get("dataset_name") for d in self._datasets]
        pick = safe if safe in names else (dataset_name if dataset_name in names else "")
        if not pick:
            messagebox.showwarning(
                "Probability Ladder",
                f'Dataset "{dataset_name}" is not available for training yet.\n\n'
                "Export it from Master Dataset first, then retry.",
                parent=self.winfo_toplevel(),
            )
            return

        self.state.prediction_type = "binary"
        self.state.target = target_name
        self.state.algorithm = self._default_algorithm_for_prediction("binary")
        self.state.wf_opt_metric = "f1"
        self.state.dataset = pick
        self.state.package_anchor = str(source_model or "").strip() or None
        if feat_list:
            self.state.features = set(feat_list)
        self._pred_type_var.set("binary")
        self._algo_var.set(self.state.algorithm)
        self._wf_opt_var.set("f1")
        self._set_dataset_combo(pick)
        self._on_dataset_changed(persist=False)

        # Dataset load / saved-state sync can overwrite — reassert ladder setup.
        def _apply_ladder_target() -> None:
            self.state.prediction_type = "binary"
            self.state.target = target_name
            self.state.algorithm = self._default_algorithm_for_prediction("binary")
            self.state.wf_opt_metric = "f1"
            self.state.package_anchor = str(source_model or "").strip() or None
            if feat_list:
                self.state.features = set(feat_list)
            self._pred_type_var.set("binary")
            self._algo_var.set(self.state.algorithm)
            self._wf_opt_var.set("f1")
            self._sync_algorithm_availability()
            self._render_targets()
            available = self._available_targets()
            if target_name not in available:
                messagebox.showwarning(
                    "Probability Ladder",
                    f'Target "{target_name}" is not in dataset "{pick}".\n\n'
                    "Re-export that dataset from Master Dataset so classification "
                    "labels (label_up_*_5m) are generated, then open this chip again.",
                    parent=self.winfo_toplevel(),
                )
                return
            self._target_var.set(target_name)
            self.state.target = target_name
            self._ensure_prediction_target_compat()
            if feat_list:
                self.state.features = set(feat_list)
                self._render_feature_groups(notify=False)
                self._feat_count_var.set(f"{self._training_feature_count()} features selected")
            self._on_change()

        try:
            if feat_list:
                self._try_apply_pending_feature_preset()
        except tk.TclError:
            # Banner packing must never block target prefill.
            pass
        _apply_ladder_target()

    def open_hit_confidence(
        self,
        *,
        dataset_name: str,
        features: list[str] | None = None,
        parent_model: str | None = None,
        lab_db_path: str | None = None,
        model_name_hint: str | None = None,
    ) -> None:
        """Prefill Model Builder for Research Lab Hit Confidence training."""
        from chain_replay_ml.dataset_builder.writer import _safe_filename

        self._show_builder()
        self._reset_lifecycle_for_new_model()
        if hasattr(self, "cancel_lazy_load"):
            self.cancel_lazy_load()
        self._pending_feature_preset = None

        if model_name_hint:
            model_name = str(model_name_hint)
        elif parent_model:
            safe_parent = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in parent_model)[:40]
            model_name = f"hit_confidence_{safe_parent}"
        else:
            model_name = "hit_confidence"
        description = (
            f"Hit Confidence classifier for {parent_model or 'parent model'}"
            + (f"\nlab={lab_db_path}" if lab_db_path else "")
        )
        self._pending_hit_confidence = {
            "dataset": str(dataset_name or "").strip(),
            "features": list(features or []),
            "model_name": model_name,
            "description": description,
            "parent_model": parent_model,
        }
        self._load_catalog(lazy=False)
        # Sync catalog apply runs finish; if save-path skipped it, finish now.
        if self._pending_hit_confidence:
            self._finish_pending_hit_confidence()

    def _finish_pending_hit_confidence(self) -> None:
        from chain_replay_ml.dataset_builder.writer import _safe_filename

        pending = getattr(self, "_pending_hit_confidence", None)
        self._pending_hit_confidence = None
        if not isinstance(pending, dict):
            return
        raw_ds = str(pending.get("dataset") or "").strip()
        safe = _safe_filename(raw_ds)
        names = [d.get("dataset_name") for d in self._datasets]
        pick = safe if safe in names else (raw_ds if raw_ds in names else "")
        if not pick:
            # Dataset may exist on disk but was filtered from picker — try force-load.
            try:
                meta_doc = service.load_dataset_metadata_doc(self._data_dir, safe or raw_ds)
                meta = meta_doc.get("metadata") or {}
                pick = str(meta_doc.get("dataset_name") or safe or raw_ds)
                self._datasets = [
                    d for d in self._datasets if d.get("dataset_name") != pick
                ] + [{
                    "dataset_name": pick,
                    "row_count": int(meta.get("row_count") or 0),
                    "feature_count": int(meta.get("feature_count") or 0),
                    "target_count": int(
                        meta.get("target_count")
                        or len(meta.get("prediction_target_columns") or [])
                        or 0
                    ),
                    "has_parquet": True,
                    "needs_parquet": False,
                    "is_draft": False,
                }]
                self._refresh_dataset_combo()
            except Exception as exc:
                messagebox.showwarning(
                    "Confidence Model",
                    f"Exported dataset \"{raw_ds}\" is not in the training catalog yet.\n\n{exc}",
                    parent=self.winfo_toplevel(),
                )
                return

        feat_list = [str(f) for f in (pending.get("features") or []) if f]
        self.state.prediction_type = "binary"
        self.state.target = "target_reached"
        self.state.algorithm = self._default_algorithm_for_prediction("binary")
        self.state.wf_opt_metric = "f1"
        self.state.model_name = str(pending.get("model_name") or "hit_confidence")
        self.state.model_name_manual = True
        self.state.model_description = str(pending.get("description") or "")
        self.state.features = set(feat_list)
        self.state.dataset = pick

        self._pred_type_var.set("binary")
        self._algo_var.set(self.state.algorithm)
        self._wf_opt_var.set("f1")
        self._set_model_name_field(self.state.model_name)
        if hasattr(self, "_desc_text"):
            self._desc_text.delete("1.0", "end")
            self._desc_text.insert("1.0", self.state.model_description)
        self._set_dataset_combo(pick)
        self._on_dataset_changed(persist=False)

        # Dataset change / saved-state sync can wipe binary + features — reassert.
        self.state.prediction_type = "binary"
        self.state.target = "target_reached"
        self.state.algorithm = self._default_algorithm_for_prediction("binary")
        self.state.wf_opt_metric = "f1"
        if feat_list:
            self.state.features = set(feat_list)
        elif not self.state.features:
            self.state.features = set(self._dataset_feature_names())
        self._pred_type_var.set("binary")
        self._algo_var.set(self.state.algorithm)
        self._wf_opt_var.set("f1")
        self._sync_algorithm_availability()
        self._render_targets()
        if "target_reached" in self._available_targets():
            self._target_var.set("target_reached")
            self.state.target = "target_reached"
        elif "hit" in self._available_targets():
            self._target_var.set("hit")
            self.state.target = "hit"
        self._render_feature_groups(notify=False)
        self._feat_count_var.set(f"{self._training_feature_count()} features selected")
        self._on_change()

    def _default_algorithm_for_prediction(self, prediction_type: str) -> str:
        """Keep current algorithm if capable; otherwise first capable from registry."""
        from chain_replay_ml.training.algorithm_capabilities import (
            algorithm_supports_prediction_type,
            algorithms_for_prediction_type,
        )

        current = str(getattr(self.state, "algorithm", "") or self._algo_var.get() or "").strip()
        if current and algorithm_supports_prediction_type(current, prediction_type):
            return current
        supported = algorithms_for_prediction_type(prediction_type)
        if supported:
            return supported[0][0]
        return "xgboost"

    def _sync_algorithm_availability(self) -> None:
        """Enable/disable algorithm radios from the centralized capability registry."""
        from chain_replay_ml.training.algorithm_capabilities import (
            algorithm_supports_prediction_type,
        )

        pred = str(self._pred_type_var.get() or self.state.prediction_type or "regression")
        radios = getattr(self, "_algo_radios", {}) or {}
        for algo_id, radio in radios.items():
            ok = algorithm_supports_prediction_type(algo_id, pred)
            try:
                radio.configure(state="normal" if ok else "disabled")
            except tk.TclError:
                pass
        current = str(self._algo_var.get() or "")
        if current and not algorithm_supports_prediction_type(current, pred):
            pick = self._default_algorithm_for_prediction(pred)
            self._algo_var.set(pick)
            self.state.algorithm = pick
            self._sync_algorithm_panels()

    def _on_prediction_type_changed(self) -> None:
        self.state.prediction_type = self._pred_type_var.get()
        targets = self._available_targets()
        is_tb = str(self.state.label_strategy_id or "").strip().lower() == "triple_barrier"
        if self.state.prediction_type == "binary":
            from chain_replay_ml.training.target_kinds import is_binary_hit_target, is_label_up_target

            binary_targets = [t for t in targets if is_binary_hit_target(t)]
            if is_tb:
                # Triple Barrier uses OLE label_id — do not require label_up_* columns.
                self._target_var.set("label_id")
                self.state.target = "label_id"
                if self._wf_opt_var.get() in ("rmse", "mae", "directional_accuracy"):
                    self._wf_opt_var.set("f1")
            elif not binary_targets:
                messagebox.showinfo(
                    "Binary classification",
                    "This dataset has no binary label columns "
                    "(target_reached / hit / label_up_*).\n"
                    "Staying on regression.",
                )
                self._pred_type_var.set("regression")
                self.state.prediction_type = "regression"
            else:
                if "target_reached" in binary_targets:
                    self._target_var.set("target_reached")
                elif "hit" in binary_targets:
                    self._target_var.set("hit")
                else:
                    self._target_var.set(
                        next(t for t in binary_targets if is_label_up_target(t))
                    )
                if self._wf_opt_var.get() in ("rmse", "mae", "directional_accuracy"):
                    self._wf_opt_var.set("f1")
        self._sync_algorithm_availability()
        self._render_label_strategies()
        self._render_targets()
        self._on_change()

    @staticmethod
    def _prediction_type_for_target(target: str) -> str | None:
        from chain_replay_ml.training.target_kinds import prediction_type_for_target

        return prediction_type_for_target(target)

    def _ensure_prediction_target_compat(self) -> None:
        """Keep prediction type + target paired so Train is not blocked by validation."""
        needed = self._prediction_type_for_target(self.state.target)
        if not needed:
            return
        current = str(self.state.prediction_type or "").strip().lower()
        if needed == "regression" and current != "regression":
            self.state.prediction_type = "regression"
            self._pred_type_var.set("regression")
        elif needed == "binary" and current not in ("binary", "classification"):
            self.state.prediction_type = "binary"
            self._pred_type_var.set("binary")
        elif needed == "classification" and current not in ("binary", "classification"):
            self.state.prediction_type = "classification"
            self._pred_type_var.set("classification")
        self._sync_algorithm_availability()

    def _on_algorithm_changed(self) -> None:
        self._sync_algorithm_panels()
        self._on_change()

    def _sync_algorithm_panels(self) -> None:
        algo = self._algo_var.get() or "xgboost"
        self._params_frame.configure(text=_ALGO_PARAM_TITLES.get(algo, "7 — Model Parameters"))
        self._trees_label_var.set(
            "Iterations" if algo == "catboost"
            else ("Trees (estimators)" if algo in ("random_forest", "extra_trees") else "Trees")
        )
        self._lgb_extra.pack_forget()
        self._cat_extra.pack_forget()
        if algo == "lightgbm":
            self._lgb_extra.pack(fill="x", pady=4)
        elif algo == "catboost":
            self._cat_extra.pack(fill="x", pady=4)
        if algo in _HPO_ALGORITHMS:
            self._hpo_algo_hint.pack_forget()
        else:
            self._hpo_algo_hint.pack(anchor="w", pady=(4, 0))
            self._global_hpo_var.set(False)
            self._wf_hpo_var.set(False)
        if self._lifecycle.get_mode():
            return
        if algo in _HPO_ALGORITHMS:
            self._global_hpo_frame.pack(fill="x", pady=4)
        else:
            self._global_hpo_frame.pack_forget()

    def _try_apply_pending_feature_preset(self) -> None:
        preset = self._pending_feature_preset or load_feature_preset(self.chart_dir)
        if not preset:
            self._render_preset_banner()
            return
        ds = self._dataset_var.get() or self.state.dataset
        result = apply_feature_preset(
            preset,
            dataset_name=ds,
            dataset_feature_names=self._dataset_feature_names(),
        )
        if self._lifecycle.uses_feature_snapshot():
            self._render_preset_banner(skipped="Lifecycle feature snapshot is locked.")
            return
        if result.get("applied"):
            self.state.features = set(result["features"])
            afs = (
                result.get("analysis_feature_selection")
                or preset.get("analysis_feature_selection")
            )
            if isinstance(afs, dict) and afs:
                self.state.analysis_feature_selection = dict(afs)
            self._pending_feature_preset = None
            clear_feature_preset(self.chart_dir)
            self._render_feature_groups()
            self._update_feature_knowledge_hint(list(result["features"]))
            src = preset.get("source_model")
            self._preset_banner_var.set(
                f"Applied {result['applied_count']} feature(s) from preset"
                + (f" (model: {src})" if src else "")
            )
            self._pack_builder_banner(self._preset_banner)
        elif result.get("pending"):
            self._render_preset_banner(
                pending=True,
                requested=result.get("requested_count", 0),
                preset_dataset=result.get("dataset"),
            )
        else:
            self._render_preset_banner()
        self._lifecycle.sync_ui()

    def _render_preset_banner(
        self,
        *,
        pending: bool = False,
        requested: int = 0,
        preset_dataset: str | None = None,
        skipped: str | None = None,
    ) -> None:
        if skipped:
            self._preset_banner_var.set(skipped)
            self._pack_builder_banner(self._preset_banner)
            return
        if pending and requested:
            ds_note = f" for dataset {preset_dataset}" if preset_dataset else ""
            self._preset_banner_var.set(
                f"Pending feature preset: {requested} feature(s){ds_note}. "
                "Select a compatible dataset or merge missing features."
            )
            self._pack_builder_banner(self._preset_banner)
            return
        self._preset_banner.pack_forget()


    def _show_builder(self) -> None:
        self._training_panel.pack_forget()
        self._builder_outer.pack(fill="both", expand=True)

    def _show_training(self, config: dict[str, Any]) -> None:
        self._builder_outer.pack_forget()
        self._training_panel.pack(fill="both", expand=True)
        self._training_panel.start(config)

    def _build_ui(self) -> None:
        self._builder_outer = ttk.Frame(self)
        self._training_panel = ModelTrainingPanel(
            self,
            chart_dir=self.chart_dir,
            on_done=self._training_finished,
            on_back=self._show_builder,
        )

        toolbar = ttk.Frame(self._builder_outer, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, textvariable=self._panel_title_var, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Refresh datasets", command=self._load_catalog).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Save config", command=self._save_config).pack(side="right", padx=4)
        # Pack order (side=right): Outcome Labels sits immediately after New Model visually.
        ttk.Button(toolbar, text="Outcome Labels", command=self._open_outcome_label_engine).pack(
            side="right", padx=4
        )
        ttk.Button(toolbar, text="New Model", command=self._clear_lifecycle).pack(side="right", padx=4)

        self._preset_banner = ttk.LabelFrame(self._builder_outer, text="Feature Preset", padding=6)
        self._preset_banner_var = tk.StringVar(value="")
        ttk.Label(self._preset_banner, textvariable=self._preset_banner_var, wraplength=720, justify="left").pack(anchor="w")

        self._knowledge_banner = ttk.LabelFrame(self._builder_outer, text="Knowledge Base", padding=6)
        self._knowledge_banner_var = tk.StringVar(value="")
        ttk.Label(self._knowledge_banner, textvariable=self._knowledge_banner_var, wraplength=720, justify="left", foreground=COL_MUTED).pack(anchor="w")

        self._lifecycle_banner = ttk.LabelFrame(self._builder_outer, text="Lifecycle Mode", padding=6)
        self._lifecycle_banner_text = scrolledtext.ScrolledText(
            self._lifecycle_banner, height=4, font=("Segoe UI", 9), wrap="word", state="disabled",
        )
        self._lifecycle_banner_text.pack(fill="x")

        self._lifecycle_summary = ttk.LabelFrame(self._builder_outer, text="Lifecycle Summary", padding=6)
        self._lifecycle_summary_text = scrolledtext.ScrolledText(
            self._lifecycle_summary, height=6, font=("Segoe UI", 9), wrap="word", state="disabled",
        )
        self._lifecycle_summary_text.pack(fill="x")

        paned = ttk.Panedwindow(self._builder_outer, orient=tk.HORIZONTAL)
        self._main_paned = paned
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        left_scroll = ScrollableFrame(paned)
        paned.add(left_scroll, weight=3)
        right_scroll = ScrollableFrame(paned)
        paned.add(right_scroll, weight=2)
        left = left_scroll.inner
        right = right_scroll.inner

        # 1 — Dataset
        ds_frame = ttk.LabelFrame(left, text="1 — Dataset", padding=8)
        ds_frame.pack(fill="x", pady=4)
        self._ds_locked = ttk.Frame(ds_frame)
        ttk.Label(self._ds_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        self._dataset_locked_var = tk.StringVar()
        ttk.Label(self._ds_locked, textvariable=self._dataset_locked_var, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._ds_editable = ttk.Frame(ds_frame)
        self._ds_editable.pack(fill="x")
        ttk.Label(self._ds_editable, text="Compatible datasets only", foreground=COL_MUTED).pack(anchor="w")
        self._dataset_var = tk.StringVar()
        self._dataset_cb = ttk.Combobox(self._ds_editable, textvariable=self._dataset_var, state="readonly", width=48)
        self._dataset_cb.pack(fill="x")
        self._dataset_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_dataset_changed())
        self._compat_frame = ttk.LabelFrame(self._ds_editable, text="Dataset Compatibility", padding=6)
        self._compat_chips = ttk.Frame(self._compat_frame)
        self._compat_chips.pack(fill="x")
        self._compat_checks = ttk.Frame(self._compat_frame)
        self._compat_checks.pack(fill="x", pady=4)
        self._compat_footer = ttk.Label(self._compat_frame, text="", wraplength=480)
        self._compat_footer.pack(anchor="w")
        stats = ttk.Frame(self._ds_editable)
        stats.pack(fill="x", pady=(8, 0))
        self._stat_rows = tk.StringVar(value="—")
        self._stat_feats = tk.StringVar(value="—")
        self._stat_targets = tk.StringVar(value="—")
        for var, lbl in ((self._stat_rows, "Rows"), (self._stat_feats, "Features"), (self._stat_targets, "Targets")):
            box = ttk.Frame(stats)
            box.pack(side="left", expand=True, fill="x", padx=2)
            ttk.Label(box, textvariable=var, font=("Segoe UI", 11, "bold"), foreground=ACCENT).pack()
            ttk.Label(box, text=lbl, foreground=COL_MUTED, font=("Segoe UI", 8)).pack()

        # 2 — Target (regression + classification side by side)
        tgt_frame = ttk.LabelFrame(left, text="2 — Target", padding=8)
        tgt_frame.pack(fill="x", pady=4)
        self._target_locked = ttk.Frame(tgt_frame)
        ttk.Label(self._target_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        self._target_locked_var = tk.StringVar()
        ttk.Label(
            self._target_locked,
            textvariable=self._target_locked_var,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        self._target_editable = ttk.Frame(tgt_frame)
        self._target_editable.pack(fill="x")

        strat_panel = ttk.LabelFrame(self._target_editable, text="Label Strategy", padding=6)
        strat_panel.pack(fill="x", pady=(0, 6))
        ttk.Label(
            strat_panel,
            text="From Outcome Label Engine registry (display_name / description)",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        self._label_strategy_list = ttk.Frame(strat_panel)
        self._label_strategy_list.pack(fill="x", pady=(4, 0))
        self._label_strategy_params_frame = ttk.Frame(strat_panel)
        self._label_strategy_params_frame.pack(fill="x", pady=(6, 0))

        # Phase X — Label Run picker (one Label Run per model).
        run_panel = ttk.LabelFrame(self._target_editable, text="Label Run", padding=6)
        run_panel.pack(fill="x", pady=(0, 6))
        ttk.Label(
            run_panel,
            text="Feature Dataset + Label Run → training join. Strategy params come from the run.",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        run_row = ttk.Frame(run_panel)
        run_row.pack(fill="x", pady=(4, 0))
        self._label_run_var = tk.StringVar(value="")
        self._label_run_combo = ttk.Combobox(
            run_row, textvariable=self._label_run_var, state="readonly", width=56
        )
        self._label_run_combo.pack(side="left", fill="x", expand=True)
        self._label_run_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_label_run_selected())
        ttk.Button(run_row, text="Refresh", command=self._refresh_label_runs).pack(side="left", padx=(6, 0))
        ttk.Button(
            run_row, text="Open Outcome Label Engine", command=self._open_outcome_label_engine
        ).pack(side="left", padx=(6, 0))
        self._label_run_hint = ttk.Label(run_panel, text="", foreground=COL_MUTED, font=("Segoe UI", 8))
        self._label_run_hint.pack(anchor="w", pady=(4, 0))
        self._label_run_options: list[dict[str, Any]] = []

        tgt_row = ttk.Frame(self._target_editable)
        tgt_row.pack(fill="x")
        tgt_row.columnconfigure(0, weight=1, uniform="target_kind")
        tgt_row.columnconfigure(1, weight=1, uniform="target_kind")

        reg_panel = ttk.LabelFrame(tgt_row, text="Regression Target", padding=6)
        reg_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._reg_target_panel = reg_panel
        self._reg_target_frame = ttk.Frame(reg_panel)
        self._reg_target_frame.pack(fill="both", expand=True)

        clf_panel = ttk.LabelFrame(tgt_row, text="Classification Target", padding=6)
        clf_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._clf_target_panel = clf_panel
        self._clf_target_frame = ttk.Frame(clf_panel)
        self._clf_target_frame.pack(fill="both", expand=True)

        # Keep a combined host alias for any code that still expects `_target_frame`.
        self._target_frame = self._reg_target_frame

        self._render_label_strategies()

        # 3–4 — Prediction Type + Algorithm (same row)
        type_algo_row = ttk.Frame(left)
        type_algo_row.pack(fill="x", pady=4)
        type_algo_row.columnconfigure(0, weight=1, uniform="type_algo")
        type_algo_row.columnconfigure(1, weight=1, uniform="type_algo")

        pred_frame = ttk.LabelFrame(type_algo_row, text="3 — Prediction Type", padding=8)
        pred_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._pred_locked = ttk.Frame(pred_frame)
        ttk.Label(self._pred_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        self._pred_locked_var = tk.StringVar()
        ttk.Label(self._pred_locked, textvariable=self._pred_locked_var, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._pred_editable = ttk.Frame(pred_frame)
        self._pred_editable.pack(fill="x")
        self._pred_type_var = tk.StringVar(value="regression")
        ttk.Radiobutton(
            self._pred_editable,
            text="Regression",
            variable=self._pred_type_var,
            value="regression",
            command=self._on_prediction_type_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            self._pred_editable,
            text="Binary Classification (Hit)",
            variable=self._pred_type_var,
            value="binary",
            command=self._on_prediction_type_changed,
        ).pack(anchor="w")
        row = ttk.Frame(self._pred_editable)
        row.pack(fill="x", anchor="w")
        ttk.Radiobutton(
            row,
            text="Multi-Class Classification",
            variable=self._pred_type_var,
            value="multiclass",
            state="disabled",
        ).pack(side="left")
        ttk.Label(row, text="Soon", foreground=COL_MUTED, font=("Segoe UI", 8)).pack(side="left", padx=6)

        algo_frame = ttk.LabelFrame(type_algo_row, text="4 — Algorithm", padding=8)
        algo_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._algo_locked = ttk.Frame(algo_frame)
        ttk.Label(self._algo_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        self._algo_locked_var = tk.StringVar()
        ttk.Label(self._algo_locked, textvariable=self._algo_locked_var, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._algo_editable = ttk.Frame(algo_frame)
        self._algo_editable.pack(fill="x")
        self._algo_var = tk.StringVar(value="xgboost")
        self._algo_radios: dict[str, ttk.Radiobutton] = {}
        for val, label in _ALGORITHMS:
            rb = ttk.Radiobutton(
                self._algo_editable,
                text=label,
                variable=self._algo_var,
                value=val,
                command=self._on_algorithm_changed,
            )
            rb.pack(anchor="w")
            self._algo_radios[val] = rb
        self._sync_algorithm_availability()

        # 5 — Feature Selection (Features + Premium Selection tabs)
        feat_frame = ttk.LabelFrame(left, text="5 — Feature Selection", padding=8)
        feat_frame.pack(fill="x", pady=4)
        hdr = ttk.Frame(feat_frame)
        hdr.pack(fill="x")
        self._feat_count_var = tk.StringVar(value="0 features selected")
        ttk.Label(hdr, textvariable=self._feat_count_var, font=("Segoe UI", 9, "bold")).pack(side="left")
        self._feat_lifecycle_summary = ttk.Frame(feat_frame)
        self._feat_summary_grid = ttk.Frame(self._feat_lifecycle_summary)
        self._feat_summary_grid.pack(fill="x")
        self._feat_mode_panel = ttk.LabelFrame(feat_frame, text="Feature Mode", padding=6)
        self._lifecycle_feature_mode_var = tk.StringVar(value="optimize")
        ttk.Radiobutton(
            self._feat_mode_panel,
            text="Keep Existing Features",
            variable=self._lifecycle_feature_mode_var,
            value="locked",
            command=self._on_lifecycle_feature_mode,
        ).pack(anchor="w")
        ttk.Radiobutton(
            self._feat_mode_panel,
            text="Optimize Features",
            variable=self._lifecycle_feature_mode_var,
            value="optimize",
            command=self._on_lifecycle_feature_mode,
        ).pack(anchor="w")

        feat_nb = ttk.Notebook(feat_frame)
        feat_nb.pack(fill="both", expand=True, pady=(4, 0))
        features_tab = ttk.Frame(feat_nb, padding=4)
        premium_tab = ttk.Frame(feat_nb, padding=4)
        feat_nb.add(features_tab, text="Features")
        feat_nb.add(premium_tab, text="Premium Selection")
        self._feat_notebook = feat_nb

        self._feat_controls = ttk.Frame(features_tab)
        self._feat_controls.pack(fill="x", pady=4)
        project_row = ttk.Frame(self._feat_controls)
        project_row.pack(fill="x", pady=(0, 4))
        self._feat_project_enabled_var = tk.BooleanVar(value=bool(self.state.feature_project_enabled))
        ttk.Checkbutton(
            project_row,
            text="Feature project",
            variable=self._feat_project_enabled_var,
            command=self._on_feature_project_enabled_toggled,
        ).pack(side="left")
        self._feat_project_var = tk.StringVar(value="all")
        self._feat_project_cb = ttk.Combobox(
            project_row,
            textvariable=self._feat_project_var,
            width=28,
            state="readonly",
        )
        self._feat_project_cb.pack(side="left", padx=4)
        self._feat_project_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_feature_project_changed())
        self._feat_project_hint = tk.StringVar(value="")
        ttk.Label(project_row, textvariable=self._feat_project_hint, foreground=COL_MUTED).pack(side="left", padx=4)
        self._sync_feature_project_controls()
        self._show_feat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._feat_controls, text="Show Features", variable=self._show_feat_var, command=self._toggle_feature_details).pack(anchor="w")
        ttk.Button(self._feat_controls, text="Policy Panel…", command=self._open_policy_panel).pack(anchor="w", pady=(2, 0))
        self._skip_audit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._feat_controls,
            text="Skip audit & validation",
            variable=self._skip_audit_var,
            command=self._on_change,
        ).pack(anchor="w")

        pt_frame = ttk.LabelFrame(self._feat_controls, text="After training (Feature Studio)", padding=4)
        pt_frame.pack(fill="x", pady=(6, 2))
        self._pt_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            pt_frame,
            text="Auto-run Feature Studio",
            variable=self._pt_enabled_var,
            command=self._on_post_training_toggle,
        ).pack(anchor="w")
        pt_stages = ttk.Frame(pt_frame)
        pt_stages.pack(fill="x", padx=(12, 0))
        self._pt_importance_var = tk.BooleanVar(value=True)
        self._pt_distribution_var = tk.BooleanVar(value=True)
        self._pt_drift_var = tk.BooleanVar(value=True)
        self._pt_stage_cbs: list[ttk.Checkbutton] = []
        for text, var in (
            ("Importance", self._pt_importance_var),
            ("Distribution", self._pt_distribution_var),
            ("Drift", self._pt_drift_var),
        ):
            cb = ttk.Checkbutton(pt_stages, text=text, variable=var, command=self._on_change)
            cb.pack(side="left", padx=(0, 8))
            self._pt_stage_cbs.append(cb)

        self._registry_auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._feat_controls,
            text="Registry Auto Features",
            variable=self._registry_auto_var,
            command=self._on_registry_auto_toggle,
        ).pack(anchor="w")
        auto_row = ttk.Frame(self._feat_controls)
        auto_row.pack(fill="x")
        ttk.Label(auto_row, text="Keep top:").pack(side="left")
        self._registry_top_var = tk.StringVar(value="75")
        ttk.Combobox(
            auto_row,
            textvariable=self._registry_top_var,
            values=_REGISTRY_TOP_OPTS,
            width=6,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Button(auto_row, text="Apply Registry Auto", command=self._apply_registry_auto).pack(side="left", padx=4)
        self._registry_auto_status = tk.StringVar(value="")
        ttk.Label(features_tab, textvariable=self._registry_auto_status, foreground=COL_MUTED, wraplength=480).pack(anchor="w")

        model_row = ttk.Frame(features_tab)
        model_row.pack(fill="x", pady=4)
        ttk.Label(model_row, text="Feature from model:").pack(side="left")
        self._model_feat_var = tk.StringVar()
        self._model_feat_cb = ttk.Combobox(model_row, textvariable=self._model_feat_var, width=36, state="readonly")
        self._model_feat_cb.pack(side="left", padx=4)
        ttk.Button(model_row, text="Apply", command=self._apply_model_features).pack(side="left")

        self._feat_details = ttk.Frame(features_tab)
        tool = ttk.Frame(self._feat_details)
        tool.pack(fill="x")
        for txt, cmd in (
            ("All", self._features_all),
            ("Clear", self._features_clear),
            ("Expand All", self._features_expand_all),
            ("Collapse All", self._features_collapse_all),
        ):
            ttk.Button(tool, text=txt, command=cmd).pack(side="left", padx=2)
        self._feat_search_var = tk.StringVar()
        self._feat_search_var.trace_add("write", lambda *_a: self._render_feature_groups())
        search_row = ttk.Frame(self._feat_details)
        search_row.pack(fill="x", pady=4)
        ttk.Entry(search_row, textvariable=self._feat_search_var).pack(side="left", fill="x", expand=True)
        self._feat_groups_host = ttk.Frame(self._feat_details)
        self._feat_groups_host.pack(fill="both", expand=True)
        self._feature_tree = FeatureSelectionTree(
            self._feat_groups_host,
            on_structure_change=self._on_feature_tree_structure_change,
            on_selection_change=self._on_feature_tree_selection_change,
            on_preview=self._show_feature_policy,
        )

        self._build_premium_selection_tab(premium_tab)

        # 6–7 — Data Split, then algorithm parameters (stacked)
        split_params_row = ttk.Frame(left)
        split_params_row.pack(fill="x", pady=4)
        split_params_row.columnconfigure(0, weight=1)

        split_frame = ttk.LabelFrame(split_params_row, text="6 — Data Split", padding=8)
        split_frame.grid(row=0, column=0, sticky="ew")
        self._split_locked = ttk.Frame(split_frame)
        ttk.Label(self._split_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        self._split_locked_var = tk.StringVar()
        ttk.Label(self._split_locked, textvariable=self._split_locked_var, font=("Segoe UI", 9, "bold"), wraplength=520).pack(anchor="w")
        self._split_locked_sub_var = tk.StringVar()
        ttk.Label(self._split_locked, textvariable=self._split_locked_sub_var, foreground=COL_MUTED, wraplength=520, font=("Segoe UI", 8)).pack(anchor="w")
        self._split_editable = ttk.Frame(split_frame)
        self._split_editable.pack(fill="x")
        self._lifecycle_wfe_hint = ttk.Label(
            self._split_editable,
            text="Walk-forward split settings stay locked from the source model. Adjust elimination method and optimization metric below.",
            foreground=COL_MUTED,
            wraplength=520,
        )
        self._split_tss_panel = ttk.Frame(self._split_editable)
        self._split_tss_panel.pack(fill="x")
        pct_row = ttk.Frame(self._split_tss_panel)
        pct_row.pack(fill="x", pady=6)
        self._split_train_var = tk.IntVar(value=70)
        self._split_val_var = tk.IntVar(value=15)
        self._split_test_var = tk.IntVar(value=15)
        for label, var in (("Train %", self._split_train_var), ("Val %", self._split_val_var), ("Test %", self._split_test_var)):
            col = ttk.Frame(pct_row)
            col.pack(side="left", padx=4)
            ttk.Label(col, text=label).pack()
            sp = ttk.Spinbox(col, from_=5, to=90, textvariable=var, width=5, command=self._on_split_changed)
            sp.pack()
            sp.bind("<FocusOut>", lambda _e: self._on_split_changed())
        ttk.Label(self._split_tss_panel, text="Validation Strategy", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self._val_strat_var = tk.StringVar(value="walk_forward")
        for val, label in _VAL_STRATEGIES:
            ttk.Radiobutton(
                self._split_tss_panel,
                text=label,
                variable=self._val_strat_var,
                value=val,
                command=self._on_val_strategy_changed,
            ).pack(anchor="w")
        self._wf_panel = ttk.Frame(self._split_editable)
        self._wf_folds_grid = ttk.Frame(self._wf_panel)
        self._wf_folds_grid.pack(fill="x")
        wf_grid = ttk.Frame(self._wf_folds_grid)
        wf_grid.pack(fill="x")
        self._wf_folds_var = tk.IntVar(value=5)
        self._wf_mode_var = tk.StringVar(value="expanding")
        self._wf_train_win_var = tk.IntVar(value=5000)
        self._wf_val_win_var = tk.IntVar(value=1000)
        for label, var, frm, to in (
            ("Folds", self._wf_folds_var, 2, 50),
            ("Train window", self._wf_train_win_var, 100, 500000),
            ("Val window", self._wf_val_win_var, 50, 100000),
            ("Test holdout %", self._split_test_var, 5, 50),
        ):
            col = ttk.Frame(wf_grid)
            col.pack(side="left", padx=4)
            ttk.Label(col, text=label).pack()
            sp = ttk.Spinbox(col, from_=frm, to=to, textvariable=var, width=8, command=self._on_wf_preview_inputs_changed)
            sp.pack()
            sp.bind("<KeyRelease>", lambda _e: self._on_wf_preview_inputs_changed())
            sp.bind("<FocusOut>", lambda _e: self._on_wf_preview_inputs_changed())
        mode_col = ttk.Frame(wf_grid)
        mode_col.pack(side="left", padx=4)
        ttk.Label(mode_col, text="Window mode").pack()
        self._wf_mode_cb = ttk.Combobox(
            mode_col, textvariable=self._wf_mode_var, values=("expanding", "rolling"), width=10, state="readonly",
        )
        self._wf_mode_cb.pack()
        self._wf_mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_wf_preview_inputs_changed())
        placement_col = ttk.Frame(wf_grid)
        placement_col.pack(side="left", padx=4)
        ttk.Label(placement_col, text="Fold placement").pack()
        self._wf_placement_var = tk.StringVar(value="distributed")
        self._wf_placement_cb = ttk.Combobox(
            placement_col,
            textvariable=self._wf_placement_var,
            values=("anchored", "distributed"),
            width=12,
            state="readonly",
        )
        self._wf_placement_cb.pack()
        self._wf_placement_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_wf_preview_inputs_changed())
        self._wf_preview_panel = ttk.LabelFrame(self._wf_panel, text="Walk-Forward Preview", padding=6)
        self._wf_preview_panel.pack(fill="x", pady=(8, 4))
        self._wf_preview_summary_var = tk.StringVar(value="Select a dataset and walk-forward settings to preview fold ranges.")
        ttk.Label(
            self._wf_preview_panel,
            textvariable=self._wf_preview_summary_var,
            justify="left",
            wraplength=720,
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        self._wf_preview_error_var = tk.StringVar(value="")
        ttk.Label(
            self._wf_preview_panel,
            textvariable=self._wf_preview_error_var,
            foreground="#c62828",
            wraplength=720,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 4))
        preview_cols = (
            ("fold", "Fold", 44),
            ("train_start", "Train Start", 84),
            ("train_end", "Train End", 84),
            ("val_start", "Val Start", 84),
            ("val_end", "Val End", 84),
            ("train_rows", "Train Rows", 80),
            ("val_rows", "Val Rows", 72),
        )
        preview_tree_wrap = ttk.Frame(self._wf_preview_panel)
        preview_tree_wrap.pack(fill="x")
        preview_tree_wrap.columnconfigure(0, weight=1)
        preview_tree_wrap.rowconfigure(0, weight=1)
        self._wf_preview_tree = ttk.Treeview(
            preview_tree_wrap,
            columns=[c[0] for c in preview_cols],
            show="headings",
            height=5,
        )
        for col_id, label, width in preview_cols:
            self._wf_preview_tree.heading(col_id, text=label)
            self._wf_preview_tree.column(col_id, width=width, anchor="e" if col_id != "fold" else "center", stretch=False)
        preview_v_scroll = ttk.Scrollbar(preview_tree_wrap, orient="vertical", command=self._wf_preview_tree.yview)
        preview_h_scroll = ttk.Scrollbar(preview_tree_wrap, orient="horizontal", command=self._wf_preview_tree.xview)
        self._wf_preview_tree.configure(yscrollcommand=preview_v_scroll.set, xscrollcommand=preview_h_scroll.set)
        self._wf_preview_tree.grid(row=0, column=0, sticky="nsew")
        preview_v_scroll.grid(row=0, column=1, sticky="ns")
        preview_h_scroll.grid(row=1, column=0, sticky="ew")
        self._wf_elim_locked = ttk.Frame(self._wf_panel)
        ttk.Label(self._wf_elim_locked, text="🔒 Locked", foreground=COL_MUTED).pack(anchor="w")
        ttk.Label(
            self._wf_elim_locked,
            text="None (Skipped) — retrain uses the locked feature snapshot",
            font=("Segoe UI", 9, "bold"),
            wraplength=520,
        ).pack(anchor="w")
        self._wf_feat_elim_editable = ttk.Frame(self._wf_panel)
        self._wf_feat_elim_editable.pack(fill="x")
        ttk.Label(self._wf_feat_elim_editable, text="Feature elimination", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self._wf_feat_sel_var = tk.StringVar(value="rfe")
        for val, label in _WF_FEATURE_SEL:
            ttk.Radiobutton(self._wf_feat_elim_editable, text=label, variable=self._wf_feat_sel_var, value=val, command=self._on_change).pack(anchor="w")
        ttk.Label(self._wf_panel, text="Optimization metric", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self._wf_opt_var = tk.StringVar(value="composite")
        for val, label in _WF_OPT_METRICS:
            ttk.Radiobutton(self._wf_panel, text=label, variable=self._wf_opt_var, value=val, command=self._on_change).pack(anchor="w")
        self._wf_hpo_section = ttk.Frame(self._wf_panel)
        self._wf_hpo_section.pack(fill="x", pady=4)
        hpo_row = ttk.Frame(self._wf_hpo_section)
        hpo_row.pack(fill="x")
        self._wf_hpo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hpo_row, text="Enable Optuna HPO", variable=self._wf_hpo_var, command=self._on_change).pack(side="left")
        self._wf_hpo_trials_var = tk.IntVar(value=25)
        ttk.Label(hpo_row, text="Trials").pack(side="left", padx=(12, 2))
        ttk.Spinbox(hpo_row, from_=5, to=200, increment=5, textvariable=self._wf_hpo_trials_var, width=6, command=self._on_change).pack(side="left")

        # 7 — Algorithm parameters (XGBoost / LightGBM / CatBoost)
        self._params_frame = ttk.LabelFrame(split_params_row, text=_ALGO_PARAM_TITLES["xgboost"], padding=8)
        self._params_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        xgb_frame = self._params_frame
        self._xgb_locked = ttk.Frame(xgb_frame)
        ttk.Label(self._xgb_locked, text="🔒 Champion Parameters", foreground=COL_MUTED).pack(anchor="w")
        self._xgb_champion_grid = ttk.Frame(self._xgb_locked)
        self._xgb_champion_grid.pack(fill="x", pady=2)
        self._xgb_locked_var = tk.StringVar()
        ttk.Label(self._xgb_locked, textvariable=self._xgb_locked_var, font=("Segoe UI", 9, "bold"), wraplength=520).pack(anchor="w")
        self._lifecycle_hpo_panel = ttk.LabelFrame(xgb_frame, text="Hyperparameter Optimization", padding=6)
        ttk.Label(self._lifecycle_hpo_panel, text="Baseline: Champion Parameters", foreground=COL_MUTED).pack(anchor="w")
        hpo_trials_row = ttk.Frame(self._lifecycle_hpo_panel)
        hpo_trials_row.pack(fill="x", pady=4)
        ttk.Label(hpo_trials_row, text="Trials").pack(side="left")
        self._lifecycle_hpo_trials_var = tk.IntVar(value=25)
        ttk.Spinbox(
            hpo_trials_row,
            from_=5,
            to=200,
            increment=5,
            textvariable=self._lifecycle_hpo_trials_var,
            width=6,
            command=self._on_lifecycle_hpo_trials,
        ).pack(side="left", padx=6)
        ttk.Label(
            self._lifecycle_hpo_panel,
            text="Optuna explores around champion parameters. Feature set stays fixed.",
            foreground=COL_MUTED,
            wraplength=520,
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        self._xgb_editable = ttk.Frame(xgb_frame)
        self._xgb_editable.pack(fill="x")
        grid = ttk.Frame(self._xgb_editable)
        grid.pack(fill="x")
        self._xgb_lr_var = tk.DoubleVar(value=0.05)
        self._xgb_trees_var = tk.IntVar(value=1000)
        self._xgb_depth_var = tk.IntVar(value=6)
        self._xgb_stop_var = tk.IntVar(value=100)
        self._xgb_seed_var = tk.IntVar(value=42)
        self._trees_label_var = tk.StringVar(value="Trees")
        for label_key, var, col in (
            ("Learning rate", self._xgb_lr_var, 0),
            ("trees", self._xgb_trees_var, 1),
            ("Max depth", self._xgb_depth_var, 2),
            ("Early stop", self._xgb_stop_var, 0),
            ("Seed", self._xgb_seed_var, 1),
        ):
            cell = ttk.Frame(grid)
            cell.grid(row=col // 3, column=col % 3, padx=4, pady=2, sticky="w")
            if label_key == "trees":
                ttk.Label(cell, textvariable=self._trees_label_var).pack(anchor="w")
            else:
                ttk.Label(cell, text=label_key).pack(anchor="w")
            ttk.Entry(cell, textvariable=var, width=10).pack()
        self._lgb_extra = ttk.Frame(self._xgb_editable)
        lgb_row = ttk.Frame(self._lgb_extra)
        lgb_row.pack(fill="x", pady=4)
        ttk.Label(lgb_row, text="Num leaves").pack(side="left")
        self._lgb_leaves_var = tk.IntVar(value=31)
        ttk.Spinbox(lgb_row, from_=8, to=256, textvariable=self._lgb_leaves_var, width=8, command=self._on_change).pack(side="left", padx=6)
        self._cat_extra = ttk.Frame(self._xgb_editable)
        cat_row = ttk.Frame(self._cat_extra)
        cat_row.pack(fill="x", pady=4)
        ttk.Label(cat_row, text="Device").pack(side="left")
        self._cat_device_var = tk.StringVar(value="CPU")
        ttk.Combobox(
            cat_row,
            textvariable=self._cat_device_var,
            values=("CPU", "GPU"),
            width=8,
            state="readonly",
        ).pack(side="left", padx=6)
        self._hpo_algo_hint = ttk.Label(
            self._xgb_editable,
            text="Optuna HPO is available for XGBoost only.",
            foreground=COL_MUTED,
            wraplength=520,
        )
        self._show_adv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._xgb_editable, text="Show advanced parameters", variable=self._show_adv_var, command=self._on_advanced_toggle).pack(anchor="w", pady=4)
        self._xgb_adv = ttk.Frame(self._xgb_editable)
        self._xgb_sub_var = tk.DoubleVar(value=0.8)
        self._xgb_col_var = tk.DoubleVar(value=0.8)
        self._xgb_child_var = tk.IntVar(value=1)
        self._xgb_alpha_var = tk.DoubleVar(value=0.0)
        self._xgb_lambda_var = tk.DoubleVar(value=1.0)
        adv_grid = ttk.Frame(self._xgb_adv)
        adv_grid.pack(fill="x")
        for i, (label, var) in enumerate(
            (
                ("Subsample", self._xgb_sub_var),
                ("Colsample", self._xgb_col_var),
                ("Min child", self._xgb_child_var),
                ("Reg alpha", self._xgb_alpha_var),
                ("Reg lambda", self._xgb_lambda_var),
            )
        ):
            cell = ttk.Frame(adv_grid)
            cell.grid(row=i // 3, column=i % 3, padx=4, pady=2, sticky="w")
            ttk.Label(cell, text=label).pack(anchor="w")
            ttk.Entry(cell, textvariable=var, width=10).pack()

        # 8 — Model Information
        info_frame = ttk.LabelFrame(left, text="8 — Model Information", padding=8)
        info_frame.pack(fill="x", pady=4)
        ttk.Label(info_frame, text="Version").pack(anchor="w")
        self._version_var = tk.StringVar(value="1.0")
        self._version_entry = ttk.Entry(info_frame, textvariable=self._version_var)
        self._version_entry.pack(fill="x", pady=(0, 6))
        ttk.Label(info_frame, text="Description").pack(anchor="w")
        self._desc_text = tk.Text(info_frame, height=3, font=("Segoe UI", 9))
        self._desc_text.pack(fill="x")

        # 9 — Feature Policy Preview (right)
        preview_frame = ttk.LabelFrame(right, text="9 — Feature Policy Preview", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=4)
        self._policy_preview_text = scrolledtext.ScrolledText(
            preview_frame, height=10, font=("Consolas", 9), state="disabled", wrap="word",
        )
        self._policy_preview_text.pack(fill="both", expand=True)
        self._preview_feature: str | None = None
        self._policy_features_by_name: dict[str, dict[str, Any]] = {}
        self._fr_projects: list[dict[str, Any]] = []
        self._set_policy_preview(None)

        # 10 — Configuration Summary (right)
        sum_frame = ttk.LabelFrame(right, text="10 — Configuration Summary", padding=8)
        sum_frame.pack(fill="both", expand=True, pady=4)
        self._summary_text = scrolledtext.ScrolledText(sum_frame, height=14, font=("Segoe UI", 9), state="disabled")
        self._summary_text.pack(fill="both", expand=True)
        ttk.Label(sum_frame, text="Model Name").pack(anchor="w")
        self._name_var = tk.StringVar()
        self._name_entry = ttk.Entry(sum_frame, textvariable=self._name_var)
        self._name_entry.pack(fill="x", pady=(0, 6))
        self._name_var.trace_add("write", lambda *_a: self._on_name_edited())
        self._lifecycle_mode_panel = ttk.LabelFrame(sum_frame, text="Lifecycle Mode", padding=6)
        self._lifecycle_mode_var = tk.StringVar(value="complete_optimization")
        for val, label in (
            ("retrain", "Retrain"),
            ("complete_optimization", "Complete Optimization"),
            ("feature_optimization", "Feature Optimization"),
        ):
            ttk.Radiobutton(
                self._lifecycle_mode_panel,
                text=label,
                variable=self._lifecycle_mode_var,
                value=val,
                command=self._on_lifecycle_mode_changed,
            ).pack(anchor="w")
        self._global_hpo_frame = ttk.Frame(sum_frame)
        ghpo = self._global_hpo_frame
        ghpo.pack(fill="x", pady=4)
        self._global_hpo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ghpo, text="Global HPO (non-WF)", variable=self._global_hpo_var, command=self._on_change).pack(side="left")
        self._global_hpo_trials_var = tk.IntVar(value=25)
        ttk.Spinbox(ghpo, from_=5, to=200, increment=5, textvariable=self._global_hpo_trials_var, width=6, command=self._on_change).pack(side="left", padx=4)
        self._train_btn = ttk.Button(sum_frame, text="Train Model", command=self._start_training, state="disabled")
        self._train_btn.pack(fill="x", pady=(8, 0))
        self._gate_text = scrolledtext.ScrolledText(
            sum_frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            foreground="#C62828",
        )
        self._gate_text.pack(anchor="w", fill="both", expand=False, pady=(6, 0))
        self._gate_text.configure(state="disabled")

        self._bind_lifecycle_widgets()
        self._apply_val_strategy_visibility()
        self._sync_algorithm_panels()
        self._sync_algorithm_availability()

    def _set_gate_text(self, text: str) -> None:
        if not hasattr(self, "_gate_text"):
            return
        self._gate_text.configure(state="normal")
        self._gate_text.delete("1.0", "end")
        if text:
            self._gate_text.insert("1.0", text)
        self._gate_text.configure(state="disabled")

    def _bind_lifecycle_widgets(self) -> None:
        self._lifecycle.bind_widgets({
            "paned": self._main_paned,
            "banner_frame": self._lifecycle_banner,
            "banner_text": self._lifecycle_banner_text,
            "summary_frame": self._lifecycle_summary,
            "summary_text": self._lifecycle_summary_text,
            "lifecycle_mode_panel": self._lifecycle_mode_panel,
            "lifecycle_mode_var": self._lifecycle_mode_var,
            "lifecycle_feature_mode_var": self._lifecycle_feature_mode_var,
            "train_btn": self._train_btn,
            "name_entry": self._name_entry,
            "version_entry": self._version_entry,
            "dataset_var": self._dataset_var,
            "dataset_cb": self._dataset_cb,
            "dataset_locked_val": self._dataset_locked_var,
            "target_locked_val": self._target_locked_var,
            "pred_locked_val": self._pred_locked_var,
            "algo_locked_val": self._algo_locked_var,
            "split_locked_val": self._split_locked_var,
            "split_locked_sub": self._split_locked_sub_var,
            "xgb_locked_val": self._xgb_locked_var,
            "xgb_champion_grid": self._xgb_champion_grid,
            "dataset_locked": self._ds_locked,
            "dataset_editable": self._ds_editable,
            "target_locked": self._target_locked,
            "target_editable": self._target_editable,
            "prediction_type_locked": self._pred_locked,
            "prediction_type_editable": self._pred_editable,
            "algorithm_locked": self._algo_locked,
            "algorithm_editable": self._algo_editable,
            "data_split_locked": self._split_locked,
            "data_split_editable": self._split_editable,
            "split_tss_panel": self._split_tss_panel,
            "wf_folds_grid": self._wf_folds_grid,
            "wf_preview_panel": self._wf_preview_panel,
            "wf_hpo_section": self._wf_hpo_section,
            "lifecycle_wfe_hint": self._lifecycle_wfe_hint,
            "wf_feat_elim_locked": self._wf_elim_locked,
            "wf_feat_elim_editable": self._wf_feat_elim_editable,
            "lifecycle_hpo_panel": self._lifecycle_hpo_panel,
            "lifecycle_hpo_trials_var": self._lifecycle_hpo_trials_var,
            "global_hpo_var": self._global_hpo_var,
            "wf_hpo_var": self._wf_hpo_var,
            "wf_feat_sel_var": self._wf_feat_sel_var,
            "xgb_locked": self._xgb_locked,
            "xgb_editable": self._xgb_editable,
            "compat_frame": self._compat_frame,
            "compat_chips": self._compat_chips,
            "compat_checks": self._compat_checks,
            "compat_footer": self._compat_footer,
            "feat_lifecycle_summary": self._feat_lifecycle_summary,
            "feat_summary_grid": self._feat_summary_grid,
            "feat_mode_panel": self._feat_mode_panel,
            "feat_controls": self._feat_controls,
            "feat_details": self._feat_details,
            "show_feat_var": self._show_feat_var,
            "global_hpo_frame": self._global_hpo_frame,
            "apply_val_strategy_visibility": self._apply_val_strategy_visibility,
        })

    def _clear_lifecycle(self) -> None:
        self._reset_lifecycle_for_new_model()
        self.state.model_name_manual = False
        self._load_catalog()

    def _on_lifecycle_mode_changed(self) -> None:
        mode = self._lifecycle_mode_var.get()
        self.state.lifecycle_mode = mode
        if self.state.lifecycle:
            self.state.lifecycle = {**self.state.lifecycle, "mode": mode}
        lc = self.state.lifecycle or {}
        if lc.get("source_model") and self._lifecycle.is_retrain_mode():
            self._lifecycle.load_retrain_datasets(str(lc["source_model"]), self.state.dataset)
            self._refresh_dataset_combo(list(self._lifecycle.retrain_compatible_datasets))
            names = [d.get("dataset_name") for d in self._datasets]
            if names:
                pick = self.state.dataset if self.state.dataset in names else names[0]
                self._set_dataset_combo(pick)
                self._on_dataset_changed()
        self._lifecycle.sync_ui()

    def _on_lifecycle_feature_mode(self) -> None:
        self.state.lifecycle_feature_mode = self._lifecycle_feature_mode_var.get()
        self._lifecycle.sync_ui()

    def _on_lifecycle_hpo_trials(self) -> None:
        trials = int(self._lifecycle_hpo_trials_var.get())
        self._global_hpo_trials_var.set(trials)
        self._wf_hpo_trials_var.set(trials)
        self._on_change()

    def _training_feature_count(self) -> int:
        return len(self._lifecycle.training_features())

    # ── catalog ──────────────────────────────────────────────────────────

    def _load_catalog(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_catalog_bundle,
                apply=self._apply_catalog_bundle,
                message="Loading model builder…",
            )
            return
        try:
            bundle = self._fetch_catalog_bundle()
        except Exception as exc:
            messagebox.showerror("Create Model", f"Load failed: {exc}")
            return
        self._apply_catalog_bundle(bundle)

    def _fetch_catalog_bundle(self) -> dict[str, Any]:
        from .. import feature_registry_service as fr_svc
        from chain_replay_ml.feature_policy import load_feature_policy_registry

        policy_features: dict[str, Any] = {}
        projects: list[dict[str, Any]] = []
        try:
            catalog = fr_svc.load_catalog(self.chart_dir)
            projects = list(catalog.get("projects") or [])
            for feat in catalog.get("features") or []:
                name = feat.get("name")
                if name:
                    policy_features[str(name)] = feat
            if policy_features:
                reg = load_feature_policy_registry(feature_names=list(policy_features.keys()))
                for name, meta in reg.features.items():
                    if name.startswith("__roll."):
                        continue
                    existing = policy_features.get(name) or {}
                    policy_features[name] = {**existing, **meta.as_dict(), "name": name}
        except Exception:
            pass
        models: list[dict[str, Any]] = []
        try:
            models = service.list_models_light(self._data_dir)
        except Exception:
            models = []
        saved: dict[str, Any] | None = None
        try:
            saved = load_persisted_state(self.chart_dir)
        except Exception:
            saved = None
        datasets: list[dict[str, Any]] = []
        try:
            datasets = service.list_builder_datasets(self._data_dir)
        except Exception:
            datasets = []
        schema: dict[str, Any] = {}
        try:
            schema = service.load_schema_registry() or {}
        except Exception:
            schema = {}
        return {
            "datasets": datasets,
            "schema": schema,
            "models": models,
            "policy_features": policy_features,
            "projects": projects,
            "saved": saved,
        }

    def _apply_catalog_bundle(self, bundle: dict[str, Any]) -> None:
        self._datasets = bundle.get("datasets") or []
        self._schema = bundle.get("schema") or {}
        self._models = bundle.get("models") or []
        self._policy_features_by_name = dict(bundle.get("policy_features") or {})
        self._fr_projects = list(bundle.get("projects") or [])
        names = [d["dataset_name"] for d in self._datasets if d.get("dataset_name")]
        self._refresh_dataset_combo()
        saved = _strip_lifecycle_from_saved(bundle.get("saved"))
        had_saved_features = False
        if saved and not self.state.lifecycle:
            self.state.apply_saved_dict(saved)
            had_saved_features = bool(saved.get("features"))
        self._populate_feature_project_combo()
        self._sync_widgets_from_state()
        pick = self.state.dataset if self.state.dataset in names else (names[0] if names else "")
        if pick:
            # Prefer a dataset whose metadata actually loads (skip empty/corrupt JSON)
            candidates = [pick] + [n for n in names if n != pick]
            chosen = ""
            for cand in candidates:
                try:
                    service.load_dataset_metadata_doc(self._data_dir, cand)
                    chosen = cand
                    break
                except Exception:
                    continue
            if chosen:
                self._set_dataset_combo(chosen)
                if had_saved_features:
                    self._pending_feature_preset = None
                try:
                    self._on_dataset_changed(persist=False)
                except Exception as exc:
                    messagebox.showwarning(
                        "Create Model",
                        f"Dataset load warning:\n{exc}",
                        parent=self.winfo_toplevel(),
                    )
            elif names:
                messagebox.showwarning(
                    "Create Model",
                    "No dataset metadata could be loaded "
                    "(empty or invalid .json under datasets/).\n"
                    "Rebuild/export a dataset, then select it in Model Builder.",
                    parent=self.winfo_toplevel(),
                )
        self._sync_widgets_from_state()
        if self.state.lifecycle_mode:
            self._lifecycle_mode_var.set(self.state.lifecycle_mode)
        if self.state.lifecycle_feature_mode:
            self._lifecycle_feature_mode_var.set(self.state.lifecycle_feature_mode)
        self._apply_val_strategy_visibility()
        self._toggle_xgb_advanced()
        self._render_feature_groups(notify=False)
        self._lifecycle.sync_ui()
        if not self.state.lifecycle:
            self._refresh_auto_model_name(force=True)
        self._update_summary()
        self._save_config(quiet=True)
        model_names = [m.get("model_name") or m.get("name") for m in self._models]
        self._model_feat_cb["values"] = [n for n in model_names if n]
        if getattr(self, "_pending_hit_confidence", None):
            self._finish_pending_hit_confidence()
        # Feature Selection / Registry handoff — apply even if dataset meta failed
        if self._pending_feature_preset or load_feature_preset(self.chart_dir):
            self._try_apply_pending_feature_preset()

    def _on_dataset_changed(self, *, persist: bool = True) -> None:
        ds = self._selected_dataset_name()
        self.state.dataset = ds
        row = next((d for d in self._datasets if d.get("dataset_name") == ds), None)
        try:
            self._dataset_meta = (
                service.load_dataset_metadata_doc(self._data_dir, ds) if ds else None
            )
        except Exception as exc:
            # Don't abort catalog/preset handoff on a single corrupt metadata file
            self._dataset_meta = None
            messagebox.showwarning(
                "Dataset",
                f"Could not load metadata for {ds or '(none)'}:\n{exc}",
                parent=self.winfo_toplevel(),
            )
        if row:
            self._stat_rows.set(fmt_rows(row.get("row_count")))
            # Prefer parquet-synced feature names over stale JSON feature_count
            # (No-Null can drop columns that metadata still advertises).
            synced_feats = self._dataset_feature_names()
            self._stat_feats.set(
                fmt_rows(len(synced_feats) if synced_feats else row.get("feature_count"))
            )
            self._stat_targets.set(fmt_rows(row.get("target_count")))
        meta = (self._dataset_meta or {}).get("metadata") or {}
        pred = str(meta.get("prediction_type") or "").strip().lower()
        if pred in ("binary", "classification") and not getattr(self, "_pending_hit_confidence", None):
            # Only auto-switch when user picks a Hit dataset manually (pending path sets its own).
            if self._pred_type_var.get() != "binary":
                self.state.prediction_type = "binary"
                self._pred_type_var.set("binary")
        self._render_targets()
        self._refresh_label_runs()
        preserve = self._lifecycle.features_ui_locked() and self._lifecycle.lifecycle_feature_snapshot
        pending_preset = self._pending_feature_preset or load_feature_preset(self.chart_dir)
        if preserve:
            self.state.features = set(self._lifecycle.lifecycle_feature_snapshot)
        elif pending_preset:
            pass
        elif not self.state.features:
            self.state.features = set(self._visible_dataset_feature_names())
        else:
            self._prune_features_to_project()
        # Always drop ghost columns absent from parquet (e.g. 100% NULL VWAPs).
        if self.state.features and not preserve:
            self._prune_features_to_project()
        self._update_feature_project_hint()
        self._render_feature_groups(notify=False)
        self._lifecycle.refresh_retrain_compatibility(ds)
        self._lifecycle.sync_ui()
        self._try_apply_pending_feature_preset()
        self._on_change(persist=persist)

    def _dataset_feature_names(self) -> list[str]:
        meta = (self._dataset_meta or {}).get("metadata") or {}
        cols = (
            meta.get("feature_columns")
            or meta.get("enabled_features")
            or meta.get("selected_features")
        )
        names: list[str] = []
        if isinstance(cols, list) and cols:
            names = [str(c) for c in cols if str(c).strip()]
        if not names:
            return []
        # Intersect with parquet so Create Model never selects dropped columns.
        parquet_rel = str(meta.get("output_parquet") or "").strip()
        if not parquet_rel:
            return names
        parquet_path = (
            parquet_rel
            if os.path.isabs(parquet_rel)
            else os.path.join(self._data_dir, parquet_rel)
        )
        try:
            from chain_replay_ml.training.dataset_loader import parquet_column_names

            available = parquet_column_names(parquet_path)
        except Exception:
            available = None
        if not available:
            return names
        return [c for c in names if c in available]

    def _feature_project_enabled(self) -> bool:
        if hasattr(self, "_feat_project_enabled_var"):
            return bool(self._feat_project_enabled_var.get())
        return bool(self.state.feature_project_enabled)

    def _sync_feature_project_controls(self) -> None:
        if not hasattr(self, "_feat_project_cb"):
            return
        enabled = self._feature_project_enabled()
        try:
            self._feat_project_cb.configure(state="readonly" if enabled else "disabled")
        except tk.TclError:
            pass

    def _feature_project_key(self, combo_val: str | None = None) -> str:
        raw = combo_val if combo_val is not None else (
            self._feat_project_var.get()
            if hasattr(self, "_feat_project_var")
            else self.state.feature_registry_project
        )
        text = str(raw or "all").strip()
        if not text:
            return "all"
        return text.split("|", 1)[0]

    def _current_feature_project(self) -> dict[str, Any] | None:
        if not self._feature_project_enabled():
            return None
        key = self._feature_project_key()
        if key in ("", "all"):
            return None
        for proj in self._fr_projects:
            if str(proj.get("id") or "") == key:
                return proj
        return None

    def _project_allowed_feature_names(self) -> set[str] | None:
        """Return allowlist when Feature project is enabled, else None (full dataset)."""
        if not self._feature_project_enabled():
            return None
        proj = self._current_feature_project()
        if not proj:
            # Enabled + "all" → registry ∩ dataset (no extra project allowlist).
            return None
        if "feature_names" in proj or "enabled_features" in proj:
            return {
                str(n)
                for n in (proj.get("feature_names") or proj.get("enabled_features") or [])
                if str(n).strip()
            }
        group_ids = {str(g) for g in (proj.get("group_ids") or []) if str(g).strip()}
        if not group_ids:
            return None
        reg = (self._schema or {}).get("groups") or {}
        names: set[str] = set()
        for gid in group_ids:
            for feat in (reg.get(gid) or {}).get("features") or []:
                names.add(str(feat))
        return names

    def _use_full_dataset_features(self) -> bool:
        """True when Feature project is off — expose every dataset column."""
        return not self._feature_project_enabled()

    def _visible_dataset_feature_names(self) -> list[str]:
        dataset_names = self._dataset_feature_names()
        if self._use_full_dataset_features():
            return list(dataset_names)
        allowed = self._project_allowed_feature_names()
        if allowed is None:
            # Enabled + "all": registry ∩ dataset
            reg_names = {
                str(f)
                for g in ((self._schema or {}).get("groups") or {}).values()
                for f in (g.get("features") or [])
            }
            if not reg_names:
                return list(dataset_names)
            return [n for n in dataset_names if n in reg_names]
        return [n for n in dataset_names if n in allowed]

    def _prune_features_to_project(self) -> None:
        visible = set(self._visible_dataset_feature_names())
        if not visible:
            return
        self.state.features = {f for f in self.state.features if f in visible}

    def _populate_feature_project_combo(self) -> None:
        if not hasattr(self, "_feat_project_cb"):
            return
        values = ["all"] + [
            f"{p.get('id')}|{p.get('label') or p.get('id')}"
            for p in self._fr_projects
        ]
        self._feat_project_cb["values"] = values
        saved_key = self._feature_project_key(self.state.feature_registry_project)
        pick = "all"
        for val in values:
            if self._feature_project_key(val) == saved_key:
                pick = val
                break
        self._feat_project_var.set(pick)
        self.state.feature_registry_project = self._feature_project_key(pick)
        if hasattr(self, "_feat_project_enabled_var"):
            self._feat_project_enabled_var.set(bool(self.state.feature_project_enabled))
        self._sync_feature_project_controls()
        self._update_feature_project_hint()

    def _update_feature_project_hint(self) -> None:
        if not hasattr(self, "_feat_project_hint"):
            return
        if not self._feature_project_enabled():
            n = len(self._dataset_feature_names())
            self._feat_project_hint.set(f"Dataset features ({n})")
            return
        proj = self._current_feature_project()
        if not proj:
            n = len(self._visible_dataset_feature_names())
            self._feat_project_hint.set(f"All registry features (dataset ∩ schema) · {n}")
            return
        names = self._project_allowed_feature_names() or set()
        in_ds = len(set(self._visible_dataset_feature_names()))
        label = str(proj.get("label") or proj.get("id") or "")
        self._feat_project_hint.set(f"{label}: {in_ds}/{len(names)} in dataset")

    def _feature_project_summary_label(self) -> str:
        if not self._feature_project_enabled():
            return "off (dataset features)"
        proj = self._current_feature_project()
        if not proj:
            return "all"
        return str(proj.get("label") or proj.get("id") or "all")

    def _premium_selection_summary_label(self) -> str:
        if not self.state.premium_selection_enabled:
            return "off"
        lo = float(self.state.premium_min)
        hi = float(self.state.premium_max)
        if lo > hi:
            lo, hi = hi, lo
        return f"LTP {lo:g}–{hi:g}"

    def _on_feature_project_enabled_toggled(self) -> None:
        self.state.feature_project_enabled = self._feature_project_enabled()
        self._sync_feature_project_controls()
        if self._lifecycle.features_ui_locked() and self._lifecycle.uses_feature_snapshot():
            self._update_feature_project_hint()
            self._render_feature_groups(notify=False)
            self._on_change()
            return
        visible = self._visible_dataset_feature_names()
        self.state.features = set(visible)
        self._update_feature_project_hint()
        if not self._show_feat_var.get() and visible:
            self._show_feat_var.set(True)
            self._toggle_feature_details()
        else:
            self._render_feature_groups()
        self._on_change()

    def _on_feature_project_changed(self) -> None:
        if self._lifecycle.features_ui_locked() and self._lifecycle.uses_feature_snapshot():
            # Keep combo in sync with locked snapshot context but do not rewrite features.
            self.state.feature_registry_project = self._feature_project_key()
            self._update_feature_project_hint()
            self._render_feature_groups(notify=False)
            return
        if not self._feature_project_enabled():
            self.state.feature_registry_project = self._feature_project_key()
            self._update_feature_project_hint()
            return
        key = self._feature_project_key()
        self.state.feature_registry_project = key
        visible = self._visible_dataset_feature_names()
        self.state.features = set(visible)
        self._update_feature_project_hint()
        if not self._show_feat_var.get():
            self._show_feat_var.set(True)
            self._toggle_feature_details()
        else:
            self._render_feature_groups()
        self._on_change()

    def _feature_groups(self) -> list[dict[str, Any]]:
        reg = (self._schema or {}).get("groups") or {}
        order = (self._schema or {}).get("groupOrder") or (self._dataset_meta or {}).get("metadata", {}).get("feature_groups") or []
        dataset_names = self._dataset_feature_names()
        allowed = set(dataset_names) if dataset_names else None
        project_names = self._project_allowed_feature_names()
        full_dataset = self._use_full_dataset_features()
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        covered: set[str] = set()

        def add(gid: str) -> None:
            if gid in seen:
                return
            seen.add(gid)
            block = reg.get(gid) or {}
            registry_feats = list(block.get("features") or [])
            feats = list(registry_feats)
            if allowed is not None:
                feats = [f for f in feats if f in allowed]
            if project_names is not None:
                feats = [f for f in feats if f in project_names]
                if not feats:
                    return
            # Enabled + "all": only show registry features present in dataset.
            if not full_dataset and project_names is None and not feats and not registry_feats:
                return
            if registry_feats or (full_dataset and feats):
                covered.update(feats)
                out.append({
                    "id": gid,
                    "label": block.get("label") or gid,
                    "features": feats,
                    "registry_features": registry_feats or list(feats),
                    "total_features": len(registry_feats) if registry_feats else len(feats),
                })

        for gid in order:
            add(str(gid))
        for gid in reg:
            add(str(gid))

        if full_dataset and allowed is not None:
            extras = sorted(n for n in allowed if n not in covered)
            if extras:
                out.append({
                    "id": "__dataset_extras",
                    "label": "Dataset (not in registry)",
                    "features": extras,
                    "registry_features": extras,
                    "total_features": len(extras),
                })
        return out

    def _sampling_interval_sec(self) -> float:
        meta = (self._dataset_meta or {}).get("metadata") or {}
        cfg = meta.get("dataset_configuration") or {}
        for key in ("sampling_interval_sec", "feature_grid_step_sec"):
            try:
                val = cfg.get(key) or meta.get(key)
                if val is not None:
                    return max(0.001, float(val))
            except (TypeError, ValueError):
                pass
        return 10.0

    def _refresh_policy_feature_cache(self) -> None:
        self._policy_features_by_name = {}
        try:
            from .. import feature_registry_service as fr_svc
            from chain_replay_ml.feature_policy import load_feature_policy_registry

            catalog = fr_svc.load_catalog(self.chart_dir)
            for feat in catalog.get("features") or []:
                name = feat.get("name")
                if name:
                    self._policy_features_by_name[str(name)] = feat
            if self._policy_features_by_name:
                reg = load_feature_policy_registry(
                    feature_names=list(self._policy_features_by_name.keys()),
                )
                for name, meta in reg.features.items():
                    if name.startswith("__roll."):
                        continue
                    existing = self._policy_features_by_name.get(name) or {}
                    self._policy_features_by_name[name] = {**existing, **meta.as_dict(), "name": name}
        except Exception:
            pass

    def _set_policy_preview(self, fname: str | None) -> None:
        from .. import feature_policy_format as pol_fmt
        from chain_replay_ml.feature_policy import DEFAULT_GAP_MAX_SEC

        self._preview_feature = fname
        if not fname:
            text = (
                "Click a feature name in section 5 to preview policy "
                "(category, lifecycle, warm-up, dependencies)."
            )
        else:
            feat = self._policy_features_by_name.get(fname) or {"name": fname}
            if not feat.get("feature_category") and not feat.get("policy"):
                try:
                    from chain_replay_ml.feature_policy import load_feature_policy_registry

                    meta = load_feature_policy_registry(feature_names=[fname]).get(fname)
                    if meta:
                        feat = {**feat, **meta.as_dict()}
                        self._policy_features_by_name[fname] = feat
                except Exception:
                    pass
            text = pol_fmt.format_feature_policy_detail(
                feat,
                sampling_interval_sec=self._sampling_interval_sec(),
                gap_max_sec=DEFAULT_GAP_MAX_SEC,
                features_by_name=self._policy_features_by_name,
                used_by_index=pol_fmt.build_used_by_index(
                    list(self._policy_features_by_name.keys()),
                    self._policy_features_by_name,
                ),
            )
        self._policy_preview_text.configure(state="normal")
        self._policy_preview_text.delete("1.0", "end")
        self._policy_preview_text.insert("end", text)
        self._policy_preview_text.configure(state="disabled")

    def _show_feature_policy(self, fname: str) -> None:
        for group in self._feature_groups():
            if fname in group.get("features", []):
                if self._feature_tree is not None:
                    self._feature_tree.ensure_expanded(str(group["id"]))
                break
        self._set_policy_preview(fname)
        if self._show_feat_var.get():
            self._render_feature_groups(notify=False)

    def _on_feature_tree_structure_change(self) -> None:
        self._render_feature_groups(notify=False)

    def _on_feature_tree_selection_change(self, event: tuple[Any, ...]) -> None:
        if event[0] == "group":
            feats = list(event[2])
            if event[3]:
                self.state.features.update(feats)
            else:
                self.state.features.difference_update(feats)
        elif event[0] == "feature":
            if event[2]:
                self.state.features.add(str(event[1]))
            else:
                self.state.features.discard(str(event[1]))
        self._render_feature_groups()

    def _open_policy_panel(self) -> None:
        from ..feature_policy_panel import open_feature_policy_window

        features = sorted(self.state.features) if self.state.features else sorted(self._dataset_feature_names())
        open_feature_policy_window(
            self,
            title="Model Feature Policy",
            feature_names=features,
            features_by_name=self._policy_features_by_name,
            sampling_interval_sec=self._sampling_interval_sec(),
            select_feature=self._preview_feature,
        )

    def _render_label_strategies(self) -> None:
        """Populate Label Strategy radios from OLE registry metadata (no strategy-name branches)."""
        from chain_replay_ml.outcome_label_engine import (
            default_params_for_strategy,
            default_strategy_id_for_prediction_type,
            ensure_builtin_strategies,
            strategy_selector_rows,
        )

        ensure_builtin_strategies()
        for w in self._label_strategy_list.winfo_children():
            w.destroy()
        pred = str(self._pred_type_var.get() if hasattr(self, "_pred_type_var") else self.state.prediction_type or "regression")
        rows = strategy_selector_rows(pred)
        if not rows:
            rows = strategy_selector_rows(None)
        current = str(self.state.label_strategy_id or "").strip()
        ids = {r["strategy_id"] for r in rows}
        if current not in ids:
            current = default_strategy_id_for_prediction_type(pred)
            if current not in ids and rows:
                current = rows[0]["strategy_id"]
        self.state.label_strategy_id = current
        self._label_strategy_var.set(current)
        if not self.state.label_strategy_params:
            try:
                self.state.label_strategy_params = default_params_for_strategy(current)
            except Exception:
                self.state.label_strategy_params = {}

        for row in rows:
            sid = row["strategy_id"]
            block = ttk.Frame(self._label_strategy_list)
            block.pack(anchor="w", fill="x", pady=1)
            ttk.Radiobutton(
                block,
                text=row["display_name"],
                variable=self._label_strategy_var,
                value=sid,
                command=self._on_label_strategy_changed,
            ).pack(anchor="w")
            ttk.Label(
                block,
                text=row["description"],
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", padx=(22, 0))

        self._render_label_strategy_params()

    def _render_label_strategy_params(self) -> None:
        """Generic schema form from ``strategy.get_config_schema()``.

        Triple Barrier params are owned by the Label Run — do not edit them here.
        """
        from chain_replay_ml.outcome_label_engine import (
            config_schema_fields,
            default_params_for_strategy,
            merge_strategy_params,
        )

        for w in self._label_strategy_params_frame.winfo_children():
            w.destroy()
        self._label_strategy_param_vars.clear()
        sid = str(self._label_strategy_var.get() or self.state.label_strategy_id or "").strip()
        if not sid:
            return
        # Phase X: TB strategy params live on the Label Run, not Create Model.
        if sid == "triple_barrier":
            ttk.Label(
                self._label_strategy_params_frame,
                text="TP / SL / holding come from the selected Label Run (Outcome Label Engine).",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
                wraplength=520,
            ).pack(anchor="w")
            return
        try:
            fields = config_schema_fields(sid)
            defaults = default_params_for_strategy(sid)
            params = merge_strategy_params(sid, self.state.label_strategy_params or defaults)
        except Exception:
            ttk.Label(
                self._label_strategy_params_frame,
                text="Label strategy schema unavailable.",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            return
        self.state.label_strategy_params = dict(params)
        if not fields:
            return
        ttk.Label(
            self._label_strategy_params_frame,
            text="Strategy parameters",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        for name, spec in fields:
            ftype = str(spec.get("type") or "str")
            # Skip complex list fields in Phase 4 UI (defaults still applied in params).
            if ftype in ("int_list", "float_list"):
                continue
            row = ttk.Frame(self._label_strategy_params_frame)
            row.pack(fill="x", pady=1)
            label_text = str(spec.get("label") or name.replace("_", " "))
            ttk.Label(row, text=label_text, width=18).pack(side="left")
            val = params.get(name, spec.get("default"))
            if ftype == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(row, variable=var, command=self._on_label_strategy_params_changed).pack(
                    side="left"
                )
            elif ftype == "enum":
                from chain_replay_ml.outcome_label_engine.types import normalize_enum_choices

                choices = normalize_enum_choices(spec.get("choices"))
                current = str(val if val is not None else spec.get("default") or "")
                enabled_values = {c["value"] for c in choices if c.get("enabled", True)}
                if current not in enabled_values and enabled_values:
                    current = next(iter(enabled_values))
                var = tk.StringVar(value=current)
                choice_col = ttk.Frame(row)
                choice_col.pack(side="left", fill="x")
                for choice in choices:
                    rb = ttk.Radiobutton(
                        choice_col,
                        text=choice["label"],
                        variable=var,
                        value=choice["value"],
                        command=self._on_label_strategy_enum_changed,
                    )
                    if not choice.get("enabled", True):
                        rb.state(["disabled"])
                    rb.pack(anchor="w")
            elif ftype == "int":
                var = tk.StringVar(value="" if val is None else str(int(val)))
                ttk.Entry(row, textvariable=var, width=12).pack(side="left")
                var.trace_add("write", lambda *_a: self._on_label_strategy_params_changed())
            elif ftype == "float":
                var = tk.StringVar(value="" if val is None else str(val))
                ttk.Entry(row, textvariable=var, width=12).pack(side="left")
                var.trace_add("write", lambda *_a: self._on_label_strategy_params_changed())
                # Unit hint for barrier values (generic: schema help or barrier_type).
                unit = ""
                if name in ("tp_value", "sl_value"):
                    bt = str(params.get("barrier_type") or "")
                    unit = "%" if bt == "percentage" else ("pts" if bt == "points" else "")
                if unit:
                    ttk.Label(row, text=unit, foreground=COL_MUTED).pack(side="left", padx=(4, 0))
            else:
                var = tk.StringVar(value="" if val is None else str(val))
                ttk.Entry(row, textvariable=var, width=18).pack(side="left")
                var.trace_add("write", lambda *_a: self._on_label_strategy_params_changed())
            self._label_strategy_param_vars[name] = var
            help_text = str(spec.get("help") or "").strip()
            if help_text and ftype == "enum":
                ttk.Label(
                    self._label_strategy_params_frame,
                    text=help_text,
                    foreground=COL_MUTED,
                    font=("Segoe UI", 8),
                ).pack(anchor="w", padx=(22, 0))

    def _collect_label_strategy_params(self) -> dict[str, Any]:
        from chain_replay_ml.outcome_label_engine import (
            config_schema_fields,
            merge_strategy_params,
        )

        sid = str(self._label_strategy_var.get() or self.state.label_strategy_id or "").strip()
        if not sid:
            return dict(self.state.label_strategy_params or {})
        # TB form is hidden — keep params stamped from the selected Label Run.
        if sid == "triple_barrier":
            return dict(self.state.label_strategy_params or {})
        raw: dict[str, Any] = {}
        try:
            fields = dict(config_schema_fields(sid))
        except Exception:
            return dict(self.state.label_strategy_params or {})
        for name, var in self._label_strategy_param_vars.items():
            spec = fields.get(name) or {}
            ftype = str(spec.get("type") or "str")
            if ftype == "bool":
                raw[name] = bool(var.get())
                continue
            text = str(var.get()).strip()
            if text == "":
                continue
            if ftype == "int":
                try:
                    raw[name] = int(float(text))
                except (TypeError, ValueError):
                    pass
            elif ftype == "float":
                try:
                    raw[name] = float(text)
                except (TypeError, ValueError):
                    pass
            else:
                # str + enum
                raw[name] = text
        try:
            return merge_strategy_params(sid, raw)
        except Exception:
            return raw

    def _on_label_strategy_changed(self) -> None:
        sid = str(self._label_strategy_var.get() or "").strip()
        self.state.label_strategy_id = sid
        from chain_replay_ml.outcome_label_engine import default_params_for_strategy

        try:
            self.state.label_strategy_params = default_params_for_strategy(sid)
        except Exception:
            self.state.label_strategy_params = {}
        if sid == "triple_barrier":
            # TB is classification; force binary so strategy stays visible.
            if str(self._pred_type_var.get() or "") == "regression":
                self._pred_type_var.set("binary")
                self.state.prediction_type = "binary"
            self._target_var.set("label_id")
            self.state.target = "label_id"
            if self._wf_opt_var.get() in ("rmse", "mae", "directional_accuracy"):
                self._wf_opt_var.set("f1")
        self._render_label_strategy_params()
        self._render_targets()
        # Auto-name: Triple Barrier → TB_tp_20_sl_10_… (unless user typed a manual name).
        self._refresh_auto_model_name(force=not self.state.model_name_manual)
        self._on_change()

    def _on_label_strategy_enum_changed(self) -> None:
        """Re-render after enum change so unit hints / dependents stay in sync."""
        self.state.label_strategy_params = self._collect_label_strategy_params()
        self._render_label_strategy_params()
        self._refresh_auto_model_name(force=not self.state.model_name_manual)
        self._on_change()

    def _on_label_strategy_params_changed(self) -> None:
        self.state.label_strategy_params = self._collect_label_strategy_params()
        self._refresh_auto_model_name(force=not self.state.model_name_manual)
        self._on_change()

    def _refresh_label_runs(self) -> None:
        """Populate Label Run combo for the selected Feature Dataset."""
        ds = self._selected_dataset_name()
        options: list[str] = [""]
        self._label_run_options = []
        rows = []
        try:
            from chain_replay_ml.label_runs import list_label_runs

            rows = list_label_runs(self._data_dir, dataset_id=ds) if ds else []
            self._label_run_options = [r.to_dict() for r in rows]
            options.extend(r.summary_line() for r in rows)
        except Exception:
            rows = []
        if hasattr(self, "_label_run_combo"):
            self._label_run_combo["values"] = options
            current = str(self.state.label_run_id or "").strip()
            selected = ""
            for line in options:
                if current and line.startswith(current):
                    selected = line
                    break
            self._label_run_var.set(selected)
        if hasattr(self, "_label_run_hint"):
            if not rows and ds:
                self._label_run_hint.configure(
                    text="No Label Run found for this dataset. Open Outcome Label Engine to Create Label Run."
                )
            elif not ds:
                self._label_run_hint.configure(text="Select a Feature Dataset first.")
            else:
                self._label_run_hint.configure(text=f"{len(rows)} Label Run(s) for {ds}")

    def _on_label_run_selected(self) -> None:
        line = str(self._label_run_var.get() or "").strip()
        run_id = line.split("  ·  ", 1)[0].strip() if line else ""
        self.state.label_run_id = run_id
        if not run_id:
            self._on_change()
            return
        try:
            from chain_replay_ml.label_runs import get_label_run

            rec = get_label_run(self._data_dir, run_id)
            if rec.strategy:
                self.state.label_strategy_id = rec.strategy
                self._label_strategy_var.set(rec.strategy)
            if rec.parameters:
                self.state.label_strategy_params = dict(rec.parameters)
            if rec.primary_target:
                self.state.target = rec.primary_target
                self._target_var.set(rec.primary_target)
            if str(rec.strategy).lower() == "triple_barrier":
                self.state.prediction_type = "binary"
                self._pred_type_var.set("binary")
            self._render_label_strategies()
            self._render_targets()
        except Exception:
            pass
        self._on_change()

    def _open_outcome_label_engine(self) -> None:
        prefill = {
            "dataset": self._selected_dataset_name(),
            "strategy": str(self._label_strategy_var.get() or self.state.label_strategy_id or ""),
            "params": dict(self.state.label_strategy_params or {}),
        }
        if self._on_open_outcome_label_engine:
            self._on_open_outcome_label_engine(prefill)
            return
        messagebox.showinfo(
            "Outcome Label Engine",
            "Open Outcome Labels from Create Model "
            "(Outcome Labels tab or toolbar button).\n"
            f"Prefill dataset: {prefill.get('dataset') or '—'}",
            parent=self.winfo_toplevel(),
        )

    def _available_targets(self) -> list[str]:
        meta = (self._dataset_meta or {}).get("metadata") or {}
        expected = (self._dataset_meta or {}).get("expected_spec") or {}
        cols = _prediction_target_columns(meta, expected)
        cols_set = set(cols)
        reg_targets = list((self._schema or {}).get("targets") or {})
        names = reg_targets if reg_targets else cols
        if cols_set:
            ordered = [n for n in names if n in cols_set]
            # Keep metadata-only targets (e.g. label_up_*) not in schema catalog.
            extras = [c for c in cols if c not in ordered]
            names = ordered + extras if ordered else cols
        return names

    def _render_targets(self) -> None:
        for host in (self._reg_target_frame, self._clf_target_frame):
            for w in host.winfo_children():
                w.destroy()
        sid = str(self._label_strategy_var.get() or self.state.label_strategy_id or "").strip()
        is_tb = sid == "triple_barrier"
        if is_tb:
            ttk.Label(
                self._reg_target_frame,
                text="Disabled for Triple Barrier.",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            ttk.Label(
                self._clf_target_frame,
                text="Disabled — OLE target is label_id\n(TP / SL / TIME).",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            self._target_var.set("label_id")
            self.state.target = "label_id"
            if str(self._pred_type_var.get() or "") == "regression":
                self._pred_type_var.set("binary")
                self.state.prediction_type = "binary"
            return

        targets = self._available_targets()
        if not targets:
            ttk.Label(
                self._reg_target_frame, text="No targets in dataset.", foreground=COL_MUTED
            ).pack(anchor="w")
            ttk.Label(
                self._clf_target_frame, text="No targets in dataset.", foreground=COL_MUTED
            ).pack(anchor="w")
            return
        from chain_replay_ml.training.target_kinds import (
            is_binary_hit_target,
            is_classification_target,
            is_label_up_target,
            is_regression_target,
        )

        classification_targets = [t for t in targets if is_classification_target(t)]
        regression_targets = [t for t in targets if is_regression_target(t)]
        # Unknown metadata targets land with classification so they stay selectable.
        other_targets = [
            t for t in targets
            if t not in regression_targets and t not in classification_targets
        ]
        classification_targets = classification_targets + other_targets
        binary_targets = [t for t in classification_targets if is_binary_hit_target(t)]

        pred = self._pred_type_var.get()
        if pred == "binary" and not classification_targets and regression_targets:
            self._pred_type_var.set("regression")
            self.state.prediction_type = "regression"
            pred = "regression"
        elif pred == "regression" and not regression_targets and classification_targets:
            # Prefer binary for hit / label_up targets; classification for ORMP direction-only.
            if binary_targets:
                self._pred_type_var.set("binary")
                self.state.prediction_type = "binary"
                pred = "binary"
            else:
                self._pred_type_var.set("classification")
                self.state.prediction_type = "classification"
                pred = "classification"
        if pred in ("binary", "classification"):
            pool = classification_targets or targets
            preferred = "target_reached" if "target_reached" in pool else (
                "hit" if "hit" in pool else (
                    next((t for t in pool if is_label_up_target(t)), pool[0])
                )
            )
        else:
            pool = regression_targets or targets
            preferred = (
                self.state.target if self.state.target in pool else (
                    "future_ltp_5m" if "future_ltp_5m" in pool else pool[-1]
                )
            )
        # Prefer OLE strategy primary_target when present (no strategy-name branches).
        try:
            from chain_replay_ml.outcome_label_engine import preferred_target_column

            ole_pref = preferred_target_column(sid, pool) if sid else None
            if ole_pref:
                preferred = ole_pref
        except Exception:
            pass
        current = str(self._target_var.get() or self.state.target or "").strip()
        if current in pool:
            default = current
        elif current in targets:
            # Keep an already-picked target even if prediction type just flipped.
            default = current
        else:
            default = preferred if preferred in pool else pool[0]
        self._target_var.set(default)
        self.state.target = default
        self._ensure_prediction_target_compat()

        def _fill(host: ttk.Frame, names: list[str], empty_text: str) -> None:
            if not names:
                ttk.Label(host, text=empty_text, foreground=COL_MUTED).pack(anchor="w")
                return
            for name in names:
                ttk.Radiobutton(
                    host,
                    text=_target_label(name),
                    variable=self._target_var,
                    value=name,
                    command=self._on_target_changed,
                ).pack(anchor="w")

        _fill(self._reg_target_frame, regression_targets, "No regression targets.")
        _fill(self._clf_target_frame, classification_targets, "No classification targets.")

    def _on_target_changed(self) -> None:
        self.state.target = self._target_var.get()
        self._ensure_prediction_target_compat()
        self._on_change()

    def _build_premium_selection_tab(self, parent: ttk.Frame) -> None:
        """Train-time LTP band filter — same semantics as Master Dataset premium filter."""
        ttk.Label(
            parent,
            text=(
                "Optional: keep only training rows whose current premium (LTP) "
                "falls in this range. Applied when the dataset is loaded for training."
            ),
            foreground=COL_MUTED,
            wraplength=480,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self._prem_sel_enabled_var = tk.BooleanVar(value=bool(self.state.premium_selection_enabled))
        ttk.Checkbutton(
            parent,
            text="Filter training rows by premium (LTP)",
            variable=self._prem_sel_enabled_var,
            command=self._on_premium_selection_changed,
        ).pack(anchor="w")

        range_box = ttk.LabelFrame(parent, text="Premium Range", padding=8)
        range_box.pack(fill="x", pady=(8, 0))
        self._prem_sel_min_var = tk.StringVar(value=str(self.state.premium_min))
        self._prem_sel_max_var = tk.StringVar(value=str(self.state.premium_max))

        row1 = ttk.Frame(range_box)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Minimum LTP", width=14).pack(side="left")
        e1 = ttk.Entry(row1, textvariable=self._prem_sel_min_var, width=10)
        e1.pack(side="left")
        e1.bind("<KeyRelease>", lambda _e: self._on_premium_selection_changed())

        row2 = ttk.Frame(range_box)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Maximum LTP", width=14).pack(side="left")
        e2 = ttk.Entry(row2, textvariable=self._prem_sel_max_var, width=10)
        e2.pack(side="left")
        e2.bind("<KeyRelease>", lambda _e: self._on_premium_selection_changed())

        self._prem_sel_hint_var = tk.StringVar(value="")
        ttk.Label(
            parent,
            textvariable=self._prem_sel_hint_var,
            foreground=COL_MUTED,
            wraplength=480,
        ).pack(anchor="w", pady=(6, 0))
        self._sync_premium_selection_hint()

    def _on_premium_selection_changed(self) -> None:
        self._sync_premium_selection_hint()
        self._on_change()

    def _sync_premium_selection_hint(self) -> None:
        if not hasattr(self, "_prem_sel_hint_var"):
            return
        if not self._prem_sel_enabled_var.get():
            self._prem_sel_hint_var.set("Off — all dataset rows are used for training.")
            return
        try:
            lo = float(str(self._prem_sel_min_var.get()).strip())
            hi = float(str(self._prem_sel_max_var.get()).strip())
            if lo > hi:
                lo, hi = hi, lo
            self._prem_sel_hint_var.set(f"On — keep rows where {lo:g} ≤ LTP ≤ {hi:g}.")
        except (TypeError, ValueError):
            self._prem_sel_hint_var.set("Enter numeric Minimum and Maximum LTP values.")

    def _toggle_feature_details(self) -> None:
        if self._show_feat_var.get():
            self._feat_details.pack(fill="both", expand=True, pady=4)
            self._render_feature_groups()
        else:
            self._feat_details.pack_forget()

    def _toggle_xgb_advanced(self) -> None:
        if self._show_adv_var.get():
            self._xgb_adv.pack(fill="x")
        else:
            self._xgb_adv.pack_forget()

    def _on_advanced_toggle(self) -> None:
        self._toggle_xgb_advanced()
        self._on_change()

    def _apply_val_strategy_visibility(self) -> None:
        strat = self._val_strat_var.get()
        if strat in ("walk_forward", "rolling_window"):
            self._wf_panel.pack(fill="x", pady=6)
        else:
            self._wf_panel.pack_forget()

    def _on_val_strategy_changed(self) -> None:
        self._apply_val_strategy_visibility()
        self._on_wf_preview_inputs_changed()

    def _on_split_changed(self) -> None:
        total = self._split_train_var.get() + self._split_val_var.get() + self._split_test_var.get()
        if total != 100:
            self._split_test_var.set(max(5, 100 - self._split_train_var.get() - self._split_val_var.get()))
        self._on_wf_preview_inputs_changed()

    def _safe_int_var(self, var: tk.IntVar, default: int) -> int:
        try:
            return int(var.get())
        except (tk.TclError, ValueError, TypeError):
            return default

    def _dataset_row_count(self) -> int | None:
        ds = self.state.dataset or self._selected_dataset_name()
        row = next((d for d in self._datasets if d.get("dataset_name") == ds), None)
        if row and row.get("row_count") is not None:
            try:
                count = int(row.get("row_count"))
                if count > 0:
                    return count
            except (TypeError, ValueError):
                pass
        meta = (self._dataset_meta or {}).get("metadata") or {}
        if meta.get("row_count") is not None:
            try:
                count = int(meta.get("row_count"))
                if count > 0:
                    return count
            except (TypeError, ValueError):
                pass
        return None

    def _on_wf_preview_inputs_changed(self) -> None:
        self._on_change()

    def _refresh_wf_preview(self) -> None:
        if self._val_strat_var.get() not in ("walk_forward", "rolling_window"):
            self._wf_preview_error_var.set("")
            self._wf_preview_summary_var.set("")
            self._wf_preview_tree.delete(*self._wf_preview_tree.get_children())
            return

        plan = compute_walk_forward_preview_plan(
            row_count=self._dataset_row_count(),
            n_folds=self._safe_int_var(self._wf_folds_var, self.state.wf_folds),
            train_window=self._safe_int_var(self._wf_train_win_var, self.state.wf_train_window),
            val_window=self._safe_int_var(self._wf_val_win_var, self.state.wf_val_window),
            window_mode=str(self._wf_mode_var.get() or self.state.wf_window_mode),
            fold_placement=str(self._wf_placement_var.get() or self.state.wf_fold_placement),
            test_holdout_pct=self._safe_int_var(self._split_test_var, self.state.split_test),
            validation_strategy=str(self._val_strat_var.get() or self.state.validation_strategy),
        )
        self._wf_preview_tree.delete(*self._wf_preview_tree.get_children())
        if not plan.get("ok"):
            self._wf_preview_summary_var.set("")
            self._wf_preview_error_var.set(str(plan.get("error") or "Could not compute walk-forward preview."))
            return

        summary = plan.get("summary") or {}
        self._wf_preview_error_var.set("")
        self._wf_preview_summary_var.set(
            " · ".join([
                f"Total rows: {fmt_rows(summary.get('total_rows'))}",
                f"Walk-forward region: rows {fmt_num(summary.get('walk_forward_region_start'), 0)}–"
                f"{fmt_num(summary.get('walk_forward_region_end'), 0)} "
                f"({fmt_rows(summary.get('walk_forward_region_rows'))} rows)",
                f"Test holdout: rows {fmt_num(summary.get('test_holdout_start'), 0)}–"
                f"{fmt_num(summary.get('test_holdout_end'), 0)} "
                f"({fmt_rows(summary.get('test_holdout_rows'))} rows, {summary.get('test_holdout_pct')}%)",
                f"Placement: {summary.get('fold_placement')}",
                f"Window: {summary.get('window_mode')}",
                f"Train/Val window: {fmt_rows(summary.get('train_window'))}/"
                f"{fmt_rows(summary.get('validation_window'))}",
                f"Folds: {summary.get('n_folds')}",
            ])
        )
        for fold in plan.get("folds") or []:
            self._wf_preview_tree.insert(
                "",
                "end",
                values=(
                    fold.get("fold"),
                    fmt_num(fold.get("train_start"), 0),
                    fmt_num(fold.get("train_end"), 0),
                    fmt_num(fold.get("val_start"), 0),
                    fmt_num(fold.get("val_end"), 0),
                    fmt_rows(fold.get("train_rows")),
                    fmt_rows(fold.get("val_rows")),
                ),
            )
        n_folds = int(summary.get("n_folds") or 0)
        self._wf_preview_tree.configure(height=max(min(n_folds, 12), 3) if n_folds else 3)

    def _on_name_edited(self) -> None:
        if self._ignore_name_trace:
            return
        self.state.model_name_manual = bool(self._name_var.get().strip())

    def _on_registry_auto_toggle(self) -> None:
        self.state.registry_auto_enabled = self._registry_auto_var.get()
        self._on_change()

    def _on_post_training_toggle(self) -> None:
        self._sync_post_training_stage_states()
        self._on_change()

    def _sync_post_training_stage_states(self) -> None:
        enabled = bool(self._pt_enabled_var.get()) if hasattr(self, "_pt_enabled_var") else True
        state = "normal" if enabled else "disabled"
        for cb in getattr(self, "_pt_stage_cbs", []):
            try:
                cb.configure(state=state)
            except tk.TclError:
                pass

    def _apply_registry_auto(self) -> None:
        top = self._registry_top_var.get()
        try:
            doc = service.registry_auto_features(self._data_dir, top=top)
        except Exception as exc:
            messagebox.showerror("Registry Auto", str(exc))
            return
        feats = [str(f) for f in (doc.get("features") or [])]
        allowed = set(self._visible_dataset_feature_names())
        applied = {f for f in feats if not allowed or f in allowed}
        self.state.features = applied
        self._registry_auto_status.set(
            f"Applied {len(applied)} features from {doc.get('model_count', '—')} models"
        )
        self._render_feature_groups()
        self._on_change()

    def _apply_model_features(self) -> None:
        model = self._model_feat_var.get().strip()
        if not model:
            return
        try:
            doc = service.model_summary(self._data_dir, model)
        except Exception as exc:
            messagebox.showerror("Model features", str(exc))
            return
        feats = doc.get("selected_features") or []
        allowed = set(self._visible_dataset_feature_names())
        applied = {str(f) for f in feats if str(f) in allowed or not allowed}
        if not applied:
            messagebox.showwarning(
                "Model features",
                "No overlapping features with current dataset"
                + (" / feature project." if self._current_feature_project() else "."),
            )
            return
        self.state.features = applied
        self._render_feature_groups()
        self._on_change()

    def _features_all(self) -> None:
        if self._lifecycle.uses_feature_snapshot():
            return
        self.state.features = {
            f for g in self._feature_groups() for f in g["features"]
        } or set(self._visible_dataset_feature_names())
        self._render_feature_groups()
        self._on_change()

    def _features_clear(self) -> None:
        if self._lifecycle.uses_feature_snapshot():
            return
        self.state.features.clear()
        self._render_feature_groups()
        self._on_change()

    def _features_expand_all(self) -> None:
        if self._feature_tree is not None:
            self._feature_tree.expand_all([str(g["id"]) for g in self._feature_groups()])
        self._render_feature_groups(notify=False)

    def _features_collapse_all(self) -> None:
        if self._feature_tree is not None:
            self._feature_tree.collapse_all()
        self._render_feature_groups(notify=False)

    def _render_feature_groups(self, *, notify: bool = True) -> None:
        if self._building_features or self._feature_tree is None:
            return
        if self._lifecycle.features_ui_locked() and not self._lifecycle.features_inspect_expanded:
            self._feat_count_var.set(f"{self._training_feature_count()} features selected")
            return
        if not self._show_feat_var.get():
            self._feat_count_var.set(f"{self._training_feature_count()} features selected")
            if notify:
                self._on_change()
            return
        self._building_features = True
        read_only = self._lifecycle.features_ui_locked()
        inspect_only = read_only and self._lifecycle.features_inspect_expanded
        self._feature_tree.render(
            groups=self._feature_groups(),
            selected=set(self.state.features),
            columns=(self._schema or {}).get("columns") or {},
            query=self._feat_search_var.get(),
            preview_feature=self._preview_feature,
            read_only=read_only and not inspect_only,
            inspect_only=inspect_only,
        )
        self._feat_count_var.set(f"{self._training_feature_count()} features selected")
        self._building_features = False
        if notify:
            self._on_change()

    def _sync_state_from_widgets(self) -> None:
        self.state.dataset = self._selected_dataset_name()
        self.state.target = self._target_var.get()
        self.state.prediction_type = self._pred_type_var.get()
        self.state.label_strategy_id = str(
            self._label_strategy_var.get() or self.state.label_strategy_id or "fixed_horizon"
        )
        self.state.label_strategy_params = self._collect_label_strategy_params()
        run_line = str(self._label_run_var.get() or "").strip()
        self.state.label_run_id = run_line.split("  ·  ", 1)[0].strip() if run_line else ""
        self._ensure_prediction_target_compat()
        self.state.algorithm = self._algo_var.get()
        self.state.split_train = int(self._split_train_var.get())
        self.state.split_val = int(self._split_val_var.get())
        self.state.split_test = int(self._split_test_var.get())
        self.state.validation_strategy = self._val_strat_var.get()
        self.state.wf_folds = int(self._wf_folds_var.get())
        self.state.wf_window_mode = self._wf_mode_var.get()
        self.state.wf_fold_placement = self._wf_placement_var.get()
        self.state.wf_train_window = int(self._wf_train_win_var.get())
        self.state.wf_val_window = int(self._wf_val_win_var.get())
        self.state.wf_feature_selection = self._wf_feat_sel_var.get()
        if self._lifecycle.is_retrain_mode():
            self.state.wf_feature_selection = "none"
        self.state.wf_opt_metric = self._wf_opt_var.get()
        self.state.wf_hpo_enabled = self._wf_hpo_var.get()
        self.state.wf_hpo_trials = int(self._wf_hpo_trials_var.get())
        self.state.global_hpo_enabled = self._global_hpo_var.get()
        self.state.global_hpo_trials = int(self._global_hpo_trials_var.get())
        self.state.xgb_lr = float(self._xgb_lr_var.get())
        self.state.xgb_trees = int(self._xgb_trees_var.get())
        self.state.xgb_depth = int(self._xgb_depth_var.get())
        self.state.xgb_early_stop = int(self._xgb_stop_var.get())
        self.state.xgb_seed = int(self._xgb_seed_var.get())
        self.state.xgb_subsample = float(self._xgb_sub_var.get())
        self.state.xgb_colsample = float(self._xgb_col_var.get())
        self.state.xgb_min_child = int(self._xgb_child_var.get())
        self.state.xgb_reg_alpha = float(self._xgb_alpha_var.get())
        self.state.xgb_reg_lambda = float(self._xgb_lambda_var.get())
        self.state.lgb_num_leaves = int(self._lgb_leaves_var.get())
        self.state.catboost_device = str(self._cat_device_var.get() or "CPU")
        self.state.model_version = self._version_var.get()
        self.state.model_description = self._desc_text.get("1.0", "end").strip()
        self.state.model_name = self._name_var.get().strip()
        self.state.skip_audit_validation = self._skip_audit_var.get()
        self.state.registry_auto_enabled = self._registry_auto_var.get()
        self.state.registry_auto_top = self._registry_top_var.get()
        self.state.show_features = self._show_feat_var.get()
        self.state.show_advanced_params = self._show_adv_var.get()
        if hasattr(self, "_pt_enabled_var"):
            self.state.post_training_enabled = bool(self._pt_enabled_var.get())
            self.state.post_training_importance = bool(self._pt_importance_var.get())
            self.state.post_training_distribution = bool(self._pt_distribution_var.get())
            self.state.post_training_drift = bool(self._pt_drift_var.get())
        if hasattr(self, "_prem_sel_enabled_var"):
            self.state.premium_selection_enabled = bool(self._prem_sel_enabled_var.get())
            try:
                self.state.premium_min = float(str(self._prem_sel_min_var.get()).strip())
            except (TypeError, ValueError):
                pass
            try:
                self.state.premium_max = float(str(self._prem_sel_max_var.get()).strip())
            except (TypeError, ValueError):
                pass
        if hasattr(self, "_feat_project_var"):
            if hasattr(self, "_feat_project_enabled_var"):
                self.state.feature_project_enabled = bool(self._feat_project_enabled_var.get())
            self.state.feature_registry_project = self._feature_project_key()
            self._prune_features_to_project()
        if self.state.lifecycle:
            self.state.lifecycle_mode = self._lifecycle_mode_var.get() or self.state.lifecycle_mode
            self.state.lifecycle_feature_mode = self._lifecycle_feature_mode_var.get()

    def _sync_widgets_from_state(self) -> None:
        self._pred_type_var.set(self.state.prediction_type)
        self._label_strategy_var.set(self.state.label_strategy_id or "fixed_horizon")
        self._render_label_strategies()
        self._algo_var.set(self.state.algorithm)
        self._split_train_var.set(self.state.split_train)
        self._split_val_var.set(self.state.split_val)
        self._split_test_var.set(self.state.split_test)
        self._val_strat_var.set(self.state.validation_strategy)
        self._wf_folds_var.set(self.state.wf_folds)
        self._wf_mode_var.set(self.state.wf_window_mode)
        self._wf_placement_var.set(self.state.wf_fold_placement)
        self._wf_train_win_var.set(self.state.wf_train_window)
        self._wf_val_win_var.set(self.state.wf_val_window)
        if self._lifecycle.is_retrain_mode():
            self._wf_feat_sel_var.set("none")
        else:
            self._wf_feat_sel_var.set(self.state.wf_feature_selection)
        self._wf_opt_var.set(self.state.wf_opt_metric)
        self._wf_hpo_var.set(self.state.wf_hpo_enabled)
        self._wf_hpo_trials_var.set(self.state.wf_hpo_trials)
        self._global_hpo_var.set(self.state.global_hpo_enabled)
        self._global_hpo_trials_var.set(self.state.global_hpo_trials)
        self._xgb_lr_var.set(self.state.xgb_lr)
        self._xgb_trees_var.set(self.state.xgb_trees)
        self._xgb_depth_var.set(self.state.xgb_depth)
        self._xgb_stop_var.set(self.state.xgb_early_stop)
        self._xgb_seed_var.set(self.state.xgb_seed)
        self._xgb_sub_var.set(self.state.xgb_subsample)
        self._xgb_col_var.set(self.state.xgb_colsample)
        self._xgb_child_var.set(self.state.xgb_min_child)
        self._xgb_alpha_var.set(self.state.xgb_reg_alpha)
        self._xgb_lambda_var.set(self.state.xgb_reg_lambda)
        self._lgb_leaves_var.set(self.state.lgb_num_leaves)
        self._cat_device_var.set(self.state.catboost_device)
        self._version_var.set(self.state.model_version)
        self._desc_text.delete("1.0", "end")
        if self.state.model_description:
            self._desc_text.insert("end", self.state.model_description)
        self._skip_audit_var.set(self.state.skip_audit_validation)
        self._registry_auto_var.set(self.state.registry_auto_enabled)
        self._registry_top_var.set(str(self.state.registry_auto_top))
        if hasattr(self, "_pt_enabled_var"):
            self._pt_enabled_var.set(bool(self.state.post_training_enabled))
            self._pt_importance_var.set(bool(self.state.post_training_importance))
            self._pt_distribution_var.set(bool(self.state.post_training_distribution))
            self._pt_drift_var.set(bool(self.state.post_training_drift))
            self._sync_post_training_stage_states()
        if self.state.target:
            self._target_var.set(self.state.target)
        self._show_feat_var.set(self.state.show_features)
        self._show_adv_var.set(self.state.show_advanced_params)
        if hasattr(self, "_prem_sel_enabled_var"):
            self._prem_sel_enabled_var.set(bool(self.state.premium_selection_enabled))
            self._prem_sel_min_var.set(str(self.state.premium_min))
            self._prem_sel_max_var.set(str(self.state.premium_max))
            self._sync_premium_selection_hint()
        self._populate_feature_project_combo()
        self._toggle_feature_details()
        self._sync_algorithm_panels()
        self._sync_algorithm_availability()

    def _on_change(self, *, persist: bool = True) -> None:
        self._sync_state_from_widgets()
        if self._lifecycle.uses_feature_snapshot():
            self.state.set_lifecycle_feature_snapshot(self._lifecycle.lifecycle_feature_snapshot)
        self._refresh_auto_model_name()
        self._sync_panel_title()
        self._refresh_wf_preview()
        self._update_summary()
        if persist:
            self._save_config(quiet=True)

    def _update_summary(self) -> None:
        cfg = self.state.build_training_config()
        strat = str(cfg.get("label_strategy") or "fixed_horizon")
        if strat == "triple_barrier":
            from chain_replay_ml.training.naming import barrier_params_slug

            params = cfg.get("label_strategy_params") or {}
            unit = "%" if str(params.get("barrier_type") or "") == "percentage" else "pts"
            barriers = barrier_params_slug(params)
            target_line = f"Target: Triple Barrier label_id ({barriers.replace('_', ' ')} {unit})"
        else:
            target_line = f"Target: {_target_label(cfg.get('target', ''))}"
        lines = [
            f"Dataset: {cfg.get('dataset') or '—'}",
            target_line,
            f"Label strategy: {strat}",
            f"Algorithm: {cfg.get('algorithm')}",
            f"Features: {self._training_feature_count()}",
            f"Feature project: {self._feature_project_summary_label()}",
            f"Premium: {self._premium_selection_summary_label()}",
            f"Validation: {self.state.validation_strategy}",
        ]
        split = cfg.get("split") or {}
        if split.get("strategy") == "walk_forward":
            wf = split.get("walk_forward") or {}
            lines.append(f"WF folds: {wf.get('n_folds')} · elim: {wf.get('feature_selection_method')}")
        else:
            lines.append(f"Split: {split.get('train')}% / {split.get('validation')}% / {split.get('test')}%")
        lines.append(f"Model: {self._name_var.get() or '—'}")
        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("end", "\n".join(lines))
        self._summary_text.configure(state="disabled")
        ready = bool(cfg.get("dataset") and cfg.get("target") and cfg.get("features"))
        if self._lifecycle.is_retrain_mode() and not self._lifecycle.is_retrain_compatible():
            ready = False
        self._train_btn.configure(state="normal" if ready else "disabled")
        mode = self.state.lifecycle_mode
        train_labels = {
            "retrain": "Start Retrain",
            "complete_optimization": "Start Optimization",
            "feature_optimization": "Start Feature Optimization",
        }
        self._train_btn.configure(text=train_labels.get(mode or "", "Train Model"))
        if ready:
            try:
                result = service.validate_config(self._data_dir, cfg)
                gate_lines = []
                for c in result.get("checks") or []:
                    icon = "✓" if c.get("passed") else "✗"
                    label = str(c.get("label") or "")
                    detail = str(c.get("detail") or "").strip()
                    if c.get("passed"):
                        gate_lines.append(f"{icon} {label}")
                    elif detail:
                        # Failed checks: show why (esp. missing feature names on retrain).
                        if "\n" in detail:
                            gate_lines.append(f"{icon} {label}")
                            gate_lines.append(detail)
                        else:
                            gate_lines.append(f"{icon} {label} — {detail}")
                    else:
                        gate_lines.append(f"{icon} {label}")
                self._set_gate_text("\n".join(gate_lines))
                if result.get("blocked"):
                    self._train_btn.configure(state="disabled")
            except Exception as exc:
                self._set_gate_text(str(exc))
        else:
            self._set_gate_text("")

    def _save_config(self, *, quiet: bool = False) -> None:
        self._sync_state_from_widgets()
        try:
            save_persisted_state(self.chart_dir, self.state)
            if not quiet:
                messagebox.showinfo("Config", "Configuration saved.")
        except OSError as exc:
            if not quiet:
                messagebox.showerror("Config", str(exc))

    def _start_training(self) -> None:
        self._sync_state_from_widgets()
        row = next((d for d in self._datasets if d.get("dataset_name") == self.state.dataset), None)
        if row and row.get("needs_parquet"):
            messagebox.showerror(
                "Parquet required",
                f"Dataset \"{self.state.dataset}\" has no Parquet file.\n\n"
                "Export from Master Dataset to registry first.",
            )
            return
        if self._lifecycle.is_retrain_mode() and not self._lifecycle.is_retrain_compatible():
            messagebox.showerror("Retrain", "Select a compatible dataset before retraining.")
            return
        cfg = self.state.build_training_config()
        try:
            result = service.validate_config(self._data_dir, cfg)
        except Exception as exc:
            messagebox.showerror("Validation", str(exc))
            return
        if result.get("blocked"):
            missing = list(result.get("missing_features") or [])
            if missing:
                preview = "\n".join(f"  • {name}" for name in missing[:25])
                extra = len(missing) - 25
                if extra > 0:
                    preview += f"\n  … +{extra} more"
                messagebox.showerror(
                    "Training Blocked",
                    f"{len(missing)} model feature(s) are missing from the selected dataset.\n\n"
                    f"{preview}\n\n"
                    "Pick a dataset that includes these features, or merge them into the dataset.",
                )
            else:
                messagebox.showerror(
                    "Training Blocked",
                    "Pre-training validation failed. See summary checks for details.",
                )
            return
        final_cfg = result.get("config") or cfg
        if not final_cfg.get("model_name"):
            final_cfg["model_name"] = self.state.suggest_model_name()
        final_cfg = self._enrich_training_config(final_cfg)
        self._show_training(final_cfg)

    def _enrich_training_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Attach dataset stats, build snapshot, and walk-forward preview for the training dashboard."""
        enriched = dict(cfg)
        row = next((d for d in self._datasets if d.get("dataset_name") == cfg.get("dataset")), None)
        dataset_meta: dict[str, Any] = {}
        if row:
            enriched["dataset_stats"] = {
                "row_count": row.get("row_count"),
                "feature_count": row.get("feature_count"),
                "target_count": row.get("target_count"),
                "sampling_interval_sec": row.get("sampling_interval_sec"),
            }
        try:
            from chain_replay_ml.dataset_builder.dataset_summary import build_dataset_build_snapshot
            from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
            import json
            import os

            safe = _safe_filename(str(cfg.get("dataset") or ""))
            meta_path = os.path.join(datasets_dir(self._data_dir()), f"{safe}.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as fh:
                    dataset_meta = json.load(fh)
        except Exception:
            dataset_meta = {}

        if dataset_meta:
            enriched["dataset_build_snapshot"] = build_dataset_build_snapshot(
                dataset_meta,
                dataset_name=str(cfg.get("dataset") or ""),
            )
            stats = dict(enriched.get("dataset_stats") or {})
            snap = enriched["dataset_build_snapshot"]
            stats.setdefault("row_count", snap.get("row_count"))
            stats.setdefault("feature_count", snap.get("feature_count"))
            sampling = snap.get("sampling") if isinstance(snap.get("sampling"), dict) else {}
            if sampling.get("interval_sec") is not None:
                stats.setdefault("sampling_interval_sec", sampling.get("interval_sec"))
            enriched["dataset_stats"] = stats

        split = cfg.get("split") or {}
        if split.get("strategy") == "walk_forward":
            wf = split.get("walk_forward") or {}
            plan = compute_walk_forward_preview_plan(
                row_count=self._dataset_row_count(),
                n_folds=int(wf.get("n_folds") or self.state.wf_folds),
                train_window=int(wf.get("train_window_size") or self.state.wf_train_window),
                val_window=int(wf.get("validation_window_size") or self.state.wf_val_window),
                window_mode=str(wf.get("window_mode") or self.state.wf_window_mode),
                fold_placement=str(wf.get("fold_placement") or self.state.wf_fold_placement),
                test_holdout_pct=int(split.get("test") or self.state.split_test),
                validation_strategy=str(split.get("validation_strategy_ui") or self.state.validation_strategy),
            )
            if plan.get("ok"):
                enriched["walk_forward_preview"] = plan
        return enriched

    def _training_finished(self, result: dict[str, Any]) -> None:
        if result.get("ok"):
            name = str(result.get("model_name") or "")
            if name and self._on_open_registry:
                self._on_open_registry(name)
                return
        if result.get("open_registry") and self._on_open_registry:
            name = str(result.get("model_name") or "")
            if name:
                self._on_open_registry(name)
                return
        self._show_builder()
        self._load_catalog()

    def load_lifecycle_preset(self, model_name: str, mode: str) -> None:
        """Open from Model Registry Retrain tab — full lifecycle preset."""
        try:
            doc = service.lifecycle_preset(self._data_dir, model_name, mode)
        except Exception as exc:
            messagebox.showerror("Lifecycle", str(exc))
            return
        self._lifecycle.apply_preset({**doc, "mode": mode})
        self.state.set_lifecycle_feature_snapshot(self._lifecycle.lifecycle_feature_snapshot)
        self._lifecycle_mode_var.set(mode)
        if self._lifecycle.is_retrain_mode():
            names = self._lifecycle.load_retrain_datasets(model_name, self.state.dataset)
            self._refresh_dataset_combo(list(self._lifecycle.retrain_compatible_datasets))
            if not names:
                messagebox.showinfo(
                    "Retrain",
                    "The original training dataset is no longer available and no compatible "
                    "datasets were found.\n\n"
                    "Register a dataset with the same target, strike selection, and sampling "
                    "interval, then try again.",
                )
            pick = self.state.dataset if self.state.dataset in names else (names[0] if names else "")
            if pick:
                self._set_dataset_combo(pick)
        self._sync_widgets_from_state()
        if self.state.dataset and not self._lifecycle.is_retrain_mode():
            self._on_dataset_changed()
        elif self._lifecycle.is_retrain_mode() and self._dataset_var.get():
            self._on_dataset_changed()
        else:
            self._render_feature_groups()
            self._lifecycle.sync_ui()
        self._show_builder()
        self._on_change()
        self._sync_panel_title()
