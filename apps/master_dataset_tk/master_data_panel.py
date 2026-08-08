"""Master Data page — mirrors web Master Data DB detail view (standalone)."""

from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_config_prefs import (
    apply_master_data_prefs,
    load_build_config_prefs,
    load_master_data_prefs,
    master_data_prefs_snapshot,
    save_build_config_prefs,
    save_master_data_prefs,
)
from .build_progress_manager import get_build_progress_manager
from .build_service import MasterRegistryExportRunner, chart_data_dir
from .feature_migration_panel import FeatureMigrationPanel
from .lazy_panel import LazyLoadMixin
from .master_data_transform_norm_regime_math import NormRegimeMathTransformMixin
from .ui_util import open_path


def effective_master_day_scope(
    selected_days: set[str],
    all_master_days: list[str],
) -> tuple[bool, list[str]]:
    """Map checkbox selection to preview scope — all days in DB vs explicit subset."""
    master = [str(d).strip() for d in all_master_days if str(d).strip()]
    selected = {str(d).strip() for d in selected_days if str(d).strip()}
    if not selected:
        return False, []
    if master and len(selected) >= len(master) and all(d in selected for d in master):
        return True, []
    return False, sorted(selected)


def preview_scope_day_count(
    *,
    all_days: bool,
    selected_days: list[str],
    trading_day: str | None,
    master_day_count: int = 0,
) -> int:
    """Selected trading-day count for master data panel labels."""
    if all_days:
        return max(int(master_day_count), len(selected_days), 0)
    days = [str(d).strip() for d in selected_days if str(d).strip()]
    if days:
        return len(days)
    if str(trading_day or "").strip():
        return 1
    return 0


def preview_day_count_from_preview(
    preview: dict[str, Any] | None,
    *,
    master_day_count: int = 0,
) -> int:
    if not preview:
        return 0
    return preview_scope_day_count(
        all_days=bool(preview.get("all_days")),
        selected_days=list(preview.get("selected_days") or []),
        trading_day=preview.get("trading_day"),
        master_day_count=master_day_count,
    )


def preview_display_match_count(preview: dict[str, Any] | None, *, no_null_data: bool) -> int:
    """Row count shown after Apply filters — all rows vs fully populated rows."""
    if not preview:
        return 0
    if no_null_data:
        if preview.get("no_null_match_count") is not None:
            return int(preview.get("no_null_match_count") or 0)
    return int(preview.get("match_count") or 0)


def empty_filter_match_message(
    preview: dict[str, Any] | None,
    *,
    no_null_data: bool,
) -> str:
    """Explain why Matching rows is 0 (ATM/LTP vs No null data)."""
    filtered = int((preview or {}).get("match_count") or 0)
    if no_null_data and filtered > 0:
        dropped = (preview or {}).get("no_null_dropped_columns") or []
        drop_note = (
            f" ({len(dropped)} all-null feature columns ignored)"
            if dropped
            else ""
        )
        return (
            f"ATM / LTP filters match {filtered:,} rows, but No null data leaves "
            f"0 complete rows{drop_note}.\n\n"
            "Every remaining row has at least one null feature value.\n\n"
            "Uncheck “No null data” to create the dataset from the filtered rows, "
            "or rebuild/fix features so rows are fully populated."
        )
    if no_null_data:
        return (
            "No complete rows match the current filters "
            "(ATM, LTP, and No null data).\n\n"
            "Try widening ATM / LTP, or uncheck “No null data”."
        )
    return (
        "No rows match the current ATM / LTP filters.\n\n"
        "Widen the ATM band or LTP range and click Apply filters."
    )


def build_export_preview_dict(
    *,
    scope: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Merge scope + filter kwargs into a sample_preview-shaped dict for registry export."""
    preview: dict[str, Any] = {}
    if scope.get("all_days"):
        preview["all_days"] = True
    elif scope.get("preview_day"):
        preview["trading_day"] = str(scope["preview_day"])
    elif scope.get("selected_days"):
        preview["selected_days"] = [str(d) for d in scope["selected_days"]]
    atm = filters.get("atm_band_filter")
    if atm is not None:
        preview["atm_band_filter"] = int(atm)
    pmin = filters.get("premium_min")
    pmax = filters.get("premium_max")
    if pmin is not None and pmax is not None:
        preview["premium_min"] = float(pmin)
        preview["premium_max"] = float(pmax)
    dmin = filters.get("delta_min")
    dmax = filters.get("delta_max")
    if dmin is not None and dmax is not None:
        preview["delta_min"] = float(dmin)
        preview["delta_max"] = float(dmax)
    return preview


def _optional_float(raw: Any) -> float | None:
    """Parse a Tk entry value; blank / invalid → None (never raises)."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _preview_float_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def preview_matches_export_settings(
    cached: dict[str, Any] | None,
    export_preview: dict[str, Any],
    *,
    no_null_data: bool,
) -> bool:
    """True when a cached sample_preview reflects current export scope and filters."""
    if not cached or cached.get("token"):
        return False
    if bool(cached.get("all_days")) != bool(export_preview.get("all_days")):
        return False
    if not cached.get("all_days"):
        cached_day = str(cached.get("trading_day") or "")
        export_day = str(export_preview.get("trading_day") or "")
        cached_sel = sorted(str(d) for d in (cached.get("selected_days") or []))
        export_sel = sorted(str(d) for d in (export_preview.get("selected_days") or []))
        if cached_day != export_day or cached_sel != export_sel:
            return False
    if cached.get("atm_band_filter") != export_preview.get("atm_band_filter"):
        return False
    for key in ("premium_min", "premium_max", "delta_min", "delta_max"):
        if not _preview_float_equal(cached.get(key), export_preview.get(key)):
            return False
    if no_null_data and cached.get("no_null_match_count") is None:
        return False
    return cached.get("match_count") is not None


def build_registry_export_kwargs(
    *,
    market: str,
    interval_sec: int,
    preview: dict[str, Any],
    feature_count: int,
    dataset_name: str | None = None,
    audit_validation_required: bool = False,
    no_null_data: bool = False,
    trading_day_filter: dict[str, Any] | None = None,
    transformation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build kwargs for create_master_registry_dataset from a sample_preview dict."""
    prem_en = preview.get("premium_min") is not None and preview.get("premium_max") is not None
    delta_en = preview.get("delta_min") is not None and preview.get("delta_max") is not None
    kwargs: dict[str, Any] = {
        "market": market,
        "interval_sec": interval_sec,
        "all_days": bool(preview.get("all_days")),
        "audit_validation_required": audit_validation_required,
        "premium_enabled": prem_en,
        "delta_enabled": delta_en,
    }
    if dataset_name:
        kwargs["dataset_name"] = dataset_name
    selected = [str(d) for d in (preview.get("selected_days") or []) if str(d).strip()]
    if kwargs["all_days"]:
        pass
    elif preview.get("trading_day"):
        kwargs["trading_day"] = str(preview["trading_day"])
    elif selected:
        if len(selected) == 1:
            kwargs["trading_day"] = selected[0]
        else:
            kwargs["selected_days"] = selected
    if preview.get("token"):
        kwargs["token"] = str(preview["token"])
    if preview.get("atm_band_filter") is not None:
        kwargs["atm_band_filter"] = int(preview["atm_band_filter"])
    if prem_en:
        kwargs["premium_min"] = float(preview["premium_min"])
        kwargs["premium_max"] = float(preview["premium_max"])
    if delta_en:
        kwargs["delta_min"] = float(preview["delta_min"])
        kwargs["delta_max"] = float(preview["delta_max"])
    if no_null_data:
        kwargs["no_null_data"] = True
    if isinstance(trading_day_filter, dict) and trading_day_filter:
        kwargs["trading_day_filter"] = {
            "mode": str(trading_day_filter.get("mode") or "all"),
            "selected_days": int(trading_day_filter.get("selected_days") or 0),
            "exported_days": int(trading_day_filter.get("exported_days") or 0),
            "selected_dates": [
                str(d).strip() for d in (trading_day_filter.get("selected_dates") or []) if str(d).strip()
            ],
            "exported_dates": [
                str(d).strip() for d in (trading_day_filter.get("exported_dates") or []) if str(d).strip()
            ],
            "excluded_dates": [
                str(d).strip() for d in (trading_day_filter.get("excluded_dates") or []) if str(d).strip()
            ],
            "expiry_dates": [
                str(d).strip() for d in (trading_day_filter.get("expiry_dates") or []) if str(d).strip()
            ],
        }
    if isinstance(transformation_config, dict) and transformation_config:
        kwargs["transformation_config"] = transformation_config
    return kwargs


def build_sample_csv_kwargs(
    *,
    market: str,
    interval_sec: int,
    preview: dict[str, Any],
    no_null_data: bool = False,
) -> dict[str, Any]:
    """Build kwargs for build_master_sample_csv_bytes from a sample_preview dict."""
    prem_en = preview.get("premium_min") is not None and preview.get("premium_max") is not None
    kwargs: dict[str, Any] = {
        "market": market,
        "interval_sec": interval_sec,
        "all_days": bool(preview.get("all_days")),
    }
    selected = [str(d) for d in (preview.get("selected_days") or []) if str(d).strip()]
    if kwargs["all_days"]:
        pass
    elif preview.get("trading_day"):
        kwargs["trading_day"] = str(preview["trading_day"])
    elif selected:
        if len(selected) == 1:
            kwargs["trading_day"] = selected[0]
        else:
            kwargs["selected_days"] = selected
    if preview.get("token"):
        kwargs["token"] = str(preview["token"])
    if preview.get("atm_band_filter") is not None:
        kwargs["atm_band_filter"] = int(preview["atm_band_filter"])
    if prem_en:
        kwargs["premium_min"] = float(preview["premium_min"])
        kwargs["premium_max"] = float(preview["premium_max"])
    delta_en = preview.get("delta_min") is not None and preview.get("delta_max") is not None
    if delta_en:
        kwargs["delta_min"] = float(preview["delta_min"])
        kwargs["delta_max"] = float(preview["delta_max"])
    if no_null_data:
        kwargs["no_null_data"] = True
    return kwargs


INTERVALS = (3, 6, 9, 10, 15, 30, 60)


def _fmt_num(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_bytes(n: Any) -> str:
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "—"
    if v >= 1_073_741_824:
        return f"{v / 1_073_741_824:.2f} GB"
    if v >= 1_048_576:
        return f"{v / 1_048_576:.1f} MB"
    if v >= 1024:
        return f"{v / 1024:.1f} KB"
    return f"{int(v)} B"


def _fmt_ts(epoch: Any) -> str:
    try:
        v = float(epoch)
    except (TypeError, ValueError):
        return "—"
    import datetime as dt

    return dt.datetime.fromtimestamp(v).strftime("%H:%M:%S")


class MasterDataPanel(NormRegimeMathTransformMixin, ttk.Frame, LazyLoadMixin):
    """Browse master SQLite DB — overview, days, filters, preview, metadata."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_create_dataset: Callable[[], None] | None = None,
        on_registry_created: Callable[[str], None] | None = None,
        on_open_model_builder: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_create_dataset = on_open_create_dataset
        self._on_registry_created = on_registry_created
        self._on_open_model_builder = on_open_model_builder
        self._detail: dict[str, Any] | None = None
        self._selected_day: str | None = None
        self._selected_days: set[str] = set()
        self._preview_detail: dict[str, Any] | None = None
        self._preview_token: str | None = None
        self._preview_drill_day: str | None = None
        self._registry_runner = MasterRegistryExportRunner(chart_dir)
        self._registry_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._registry_progress_queue: queue.Queue[str] = queue.Queue()
        self._market_var = tk.StringVar(value="NIFTY")
        self._interval_var = tk.IntVar(value=10)
        self._status_var = tk.StringVar(value="")
        self._days_selected_hint_var = tk.StringVar(value="0 trading days selected")
        self._registry_rows_var = tk.StringVar(value="—")
        self._registry_scope_var = tk.StringVar(value="—")
        self._registry_features_var = tk.StringVar(value="—")
        self._registry_name_var = tk.StringVar(value="—")
        self._registry_progress_var = tk.StringVar(value="")
        self._audit_required_var = tk.BooleanVar(value=False)
        self._no_null_data_var = tk.BooleanVar(value=False)
        self._no_null_filter_report_var = tk.BooleanVar(value=False)
        self._trading_day_filter_var = tk.StringVar(value="all")
        self._lag_enabled_var = tk.BooleanVar(value=False)
        self._difference_enabled_var = tk.BooleanVar(value=False)
        self._return_enabled_var = tk.BooleanVar(value=False)
        self._rolling_enabled_var = tk.BooleanVar(value=False)
        self._exponential_rolling_enabled_var = tk.BooleanVar(value=False)
        self._ohlc_aggregation_enabled_var = tk.BooleanVar(value=False)
        self._interaction_enabled_var = tk.BooleanVar(value=False)
        self._normalization_enabled_var = tk.BooleanVar(value=False)
        self._regime_enabled_var = tk.BooleanVar(value=False)
        self._math_enabled_var = tk.BooleanVar(value=False)
        self._interaction_op_var = tk.StringVar(value="multiply")
        self._interaction_feat_a_var = tk.StringVar(value="")
        self._interaction_feat_b_var = tk.StringVar(value="")
        self._interaction_source_a_var = tk.StringVar(value="Master Features")
        self._interaction_source_b_var = tk.StringVar(value="Master Features")
        self._interaction_source_a_ids: dict[str, str] = {"Master Features": "master"}
        self._interaction_source_b_ids: dict[str, str] = {"Master Features": "master"}
        self._interaction_output_var = tk.StringVar(value="")
        self._interaction_search_a_var = tk.StringVar(value="")
        self._interaction_search_b_var = tk.StringVar(value="")
        self._interaction_bulk_search_a_var = tk.StringVar(value="")
        self._interaction_bulk_search_b_var = tk.StringVar(value="")
        self._interaction_feat_a_cols_full: list[str] = []
        self._interaction_feat_b_cols_full: list[str] = []
        self._interaction_bulk_a_meta: list[dict[str, Any]] = []
        self._interaction_bulk_b_meta: list[dict[str, Any]] = []
        self._interaction_pairs: list[dict[str, Any]] = []
        self._interaction_pipeline_pairs: list[dict[str, Any]] = []
        self._interaction_bulk_a_vars: dict[str, tk.BooleanVar] = {}
        self._interaction_bulk_b_vars: dict[str, tk.BooleanVar] = {}
        self._interaction_available: list[str] = []
        self._interaction_columns_by_source: dict[str, list[str]] = {}
        self._lag_feature_vars: dict[str, tk.BooleanVar] = {}
        self._lag_seconds_vars: dict[int, tk.BooleanVar] = {}
        self._lag_seconds_buttons: dict[int, ttk.Checkbutton] = {}
        self._difference_feature_vars: dict[str, tk.BooleanVar] = {}
        self._difference_seconds_vars: dict[int, tk.BooleanVar] = {}
        self._difference_preview_var = tk.StringVar(value="")
        self._difference_warn_var = tk.StringVar(value="")
        self._pending_difference_features: list[str] | None = None
        self._return_feature_vars: dict[str, tk.BooleanVar] = {}
        self._return_seconds_vars: dict[int, tk.BooleanVar] = {}
        self._return_preview_var = tk.StringVar(value="")
        self._return_warn_var = tk.StringVar(value="")
        self._pending_return_features: list[str] | None = None
        self._rolling_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_rolling_features: list[str] | None = None
        self._rolling_window_vars: dict[int, tk.BooleanVar] = {}
        self._rolling_op_vars: dict[str, tk.BooleanVar] = {}
        self._exponential_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_exponential_features: list[str] | None = None
        self._exponential_period_vars: dict[int, tk.BooleanVar] = {}
        self._exponential_op_vars: dict[str, tk.BooleanVar] = {}
        self._ohlc_tf_vars: dict[str, tk.BooleanVar] = {}
        self._ohlc_output_vars: dict[str, tk.BooleanVar] = {}
        self._ohlc_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_ohlc_features: list[str] | None = None
        self._lag_preview_var = tk.StringVar(value="")
        self._feature_preview_var = tk.StringVar(value="")
        self._rolling_preview_var = tk.StringVar(value="")
        self._exponential_rolling_preview_var = tk.StringVar(value="")
        self._ohlc_aggregation_preview_var = tk.StringVar(value="")
        self._normalization_preview_var = tk.StringVar(value="")
        self._regime_preview_var = tk.StringVar(value="")
        self._math_preview_var = tk.StringVar(value="")
        self._normalization_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_normalization_features: list[str] | None = None
        self._normalization_window_vars: dict[int, tk.BooleanVar] = {}
        self._normalization_method_vars: dict[str, tk.BooleanVar] = {}
        self._regime_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_regime_features: list[str] | None = None
        self._regime_window_vars: dict[int, tk.BooleanVar] = {}
        self._regime_method_vars: dict[str, tk.BooleanVar] = {}
        self._math_feature_vars: dict[str, tk.BooleanVar] = {}
        self._pending_math_features: list[str] | None = None
        self._math_op_vars: dict[str, tk.BooleanVar] = {}
        self._math_clip_min_var = tk.StringVar(value="0")
        self._math_clip_max_var = tk.StringVar(value="")
        self._regime_threshold_var = tk.StringVar(value="0")
        self._regime_low_var = tk.StringVar(value="-1")
        self._regime_high_var = tk.StringVar(value="1")
        self._regime_n_bins_var = tk.StringVar(value="5")
        self._lag_warn_var = tk.StringVar(value="")
        self._lag_search_var = tk.StringVar(value="")
        self._lag_category_expanded: dict[str, bool] = {}
        self._lag_available_features: list[str] = []
        self._pending_lag_features: list[str] | None = None
        self._day_selection_explicit = False
        self._loading_prefs = False
        self._preview_busy = False
        self._export_count_busy = False
        self._preview_load_id = 0
        self._feature_xform_win = None
        self._feature_xform_placed = False
        self._xform_ui_built = False
        self._xform_populate_pending = False
        self._build_ui()
        self._lazy_init()
        self._bind_pref_traces()
        self._load_prefs()

    def _on_market_changed(self) -> None:
        if self._loading_prefs:
            return
        self._save_prefs()
        self.refresh()

    def _bind_pref_traces(self) -> None:
        for var in (
            self._market_var,
            self._atm_var,
            self._prem_en_var,
            self._prem_min_var,
            self._prem_max_var,
            self._audit_required_var,
            self._no_null_data_var,
            self._no_null_filter_report_var,
            self._trading_day_filter_var,
        ):
            var.trace_add("write", lambda *_a: self._save_prefs())
        for var in (
            self._atm_var,
            self._prem_en_var,
            self._prem_min_var,
            self._prem_max_var,
            self._trading_day_filter_var,
        ):
            var.trace_add("write", lambda *_a: self._update_registry_export_panel())

    def _on_interval_changed(self) -> None:
        self._save_prefs()
        self._rebuild_lag_horizon_checks()
        self._refresh_lag_seconds_labels()
        try:
            self._rebuild_simple_horizon_checks(
                seconds_host_attr="_difference_seconds_host",
                seconds_vars_attr="_difference_seconds_vars",
                on_change=self._on_difference_settings_changed,
            )
            self._rebuild_simple_horizon_checks(
                seconds_host_attr="_return_seconds_host",
                seconds_vars_attr="_return_seconds_vars",
                on_change=self._on_return_settings_changed,
            )
        except Exception:
            pass
        self._rebuild_ohlc_timeframe_checks()
        self._update_ohlc_interval_hint()
        self.refresh()

    def _pending_features_from_prefs(
        self,
        applied: dict[str, Any],
        raw_prefs: dict[str, Any],
        key: str,
        *,
        fallback_key: str | None = None,
    ) -> list[str] | None:
        """Load feature list from prefs; fall back for older prefs without the key."""
        if key in raw_prefs:
            return [str(f) for f in (applied.get(key) or []) if str(f).strip()]
        if fallback_key and fallback_key in raw_prefs:
            return [str(f) for f in (applied.get(fallback_key) or []) if str(f).strip()]
        return None

    def _apply_seconds_vars_from_prefs(
        self,
        applied: dict[str, Any],
        raw_prefs: dict[str, Any],
        key: str,
        seconds_vars_attr: str,
        *,
        fallback_key: str | None = None,
    ) -> None:
        vars_map = getattr(self, seconds_vars_attr, None) or {}
        if not vars_map:
            return
        src_key = (
            key
            if key in raw_prefs
            else (fallback_key if fallback_key and fallback_key in raw_prefs else None)
        )
        if src_key is None:
            return
        wanted: set[int] = set()
        for s in applied.get(src_key) or []:
            try:
                wanted.add(int(s))
            except (TypeError, ValueError):
                continue
        for sec, var in vars_map.items():
            var.set(int(sec) in wanted)

    def _load_prefs(self) -> None:
        if not self.chart_dir:
            return
        build = load_build_config_prefs(self.chart_dir) or {}
        build_section = build.get("build") if isinstance(build.get("build"), dict) else {}
        applied = apply_master_data_prefs(
            load_master_data_prefs(self.chart_dir),
            build_prefs=build_section,
            allowed_intervals=INTERVALS,
        )
        self._loading_prefs = True
        try:
            self._market_var.set(applied["market"])
            self._interval_var.set(int(applied["interval_sec"]))
            if hasattr(self, "_rebuild_lag_horizon_checks"):
                self._rebuild_lag_horizon_checks()
            try:
                self._rebuild_simple_horizon_checks(
                    seconds_host_attr="_difference_seconds_host",
                    seconds_vars_attr="_difference_seconds_vars",
                    on_change=self._on_difference_settings_changed,
                )
                self._rebuild_simple_horizon_checks(
                    seconds_host_attr="_return_seconds_host",
                    seconds_vars_attr="_return_seconds_vars",
                    on_change=self._on_return_settings_changed,
                )
            except Exception:
                pass
            if hasattr(self, "_rebuild_ohlc_timeframe_checks"):
                self._rebuild_ohlc_timeframe_checks()
                self._update_ohlc_interval_hint()
            self._selected_days = set(applied["selected_days"])
            self._day_selection_explicit = bool(applied["day_selection_explicit"])
            self._atm_var.set(applied["atm_band"])
            self._prem_en_var.set(bool(applied["premium_enabled"]))
            self._prem_min_var.set(applied["premium_min"])
            self._prem_max_var.set(applied["premium_max"])
            self._audit_required_var.set(bool(applied["audit_required"]))
            self._no_null_data_var.set(bool(applied.get("no_null_data", False)))
            self._no_null_filter_report_var.set(
                bool(applied.get("no_null_filter_report", False))
            )
            self._trading_day_filter_var.set(str(applied.get("trading_day_filter") or "all"))
            self._lag_enabled_var.set(bool(applied.get("lag_enabled", False)))
            self._difference_enabled_var.set(bool(applied.get("difference_enabled", False)))
            self._return_enabled_var.set(bool(applied.get("return_enabled", False)))
            self._rolling_enabled_var.set(bool(applied.get("rolling_enabled", False)))
            self._exponential_rolling_enabled_var.set(
                bool(applied.get("exponential_rolling_enabled", False))
            )
            self._ohlc_aggregation_enabled_var.set(
                bool(applied.get("ohlc_aggregation_enabled", False))
            )
            self._interaction_enabled_var.set(bool(applied.get("interaction_enabled", False)))
            self._normalization_enabled_var.set(
                bool(applied.get("normalization_enabled", False))
            )
            self._regime_enabled_var.set(bool(applied.get("regime_enabled", False)))
            self._math_enabled_var.set(bool(applied.get("math_enabled", False)))
            self._regime_threshold_var.set(str(applied.get("regime_threshold", "0")))
            self._regime_low_var.set(str(applied.get("regime_low", "-1")))
            self._regime_high_var.set(str(applied.get("regime_high", "1")))
            self._regime_n_bins_var.set(str(applied.get("regime_n_bins", "5")))
            self._math_clip_min_var.set(str(applied.get("math_clip_min", "0")))
            self._math_clip_max_var.set(str(applied.get("math_clip_max", "")))
            raw_pairs = applied.get("interaction_pairs")
            if isinstance(raw_pairs, list):
                self._interaction_pairs = [
                    dict(p) for p in raw_pairs if isinstance(p, dict)
                ]
            else:
                self._interaction_pairs = []
            raw_pipeline = applied.get("interaction_pipeline_pairs")
            if isinstance(raw_pipeline, list):
                self._interaction_pipeline_pairs = [
                    dict(p) for p in raw_pipeline if isinstance(p, dict)
                ]
            elif self._interaction_pairs:
                # Legacy: pairs were applied directly to the pipeline.
                self._interaction_pipeline_pairs = [
                    dict(p) for p in self._interaction_pairs
                ]
            else:
                self._interaction_pipeline_pairs = []
            raw_prefs = load_master_data_prefs(self.chart_dir) or {}
            if "lag_seconds" in raw_prefs and self._lag_seconds_vars:
                wanted = set()
                for s in (applied.get("lag_seconds") or []):
                    try:
                        wanted.add(int(s))
                    except (TypeError, ValueError):
                        continue
                for sec, var in self._lag_seconds_vars.items():
                    var.set(sec in wanted)
            if "rolling_windows" in raw_prefs and self._rolling_window_vars:
                wanted_wins = set()
                for w in (applied.get("rolling_windows") or []):
                    try:
                        wanted_wins.add(int(w))
                    except (TypeError, ValueError):
                        continue
                for win, var in self._rolling_window_vars.items():
                    var.set(win in wanted_wins)
            if "rolling_operations" in raw_prefs and self._rolling_op_vars:
                wanted_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("rolling_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._rolling_op_vars.items():
                    var.set(op in wanted_ops)
            if "exponential_rolling_periods" in raw_prefs and self._exponential_period_vars:
                wanted_pers = set()
                for p in (applied.get("exponential_rolling_periods") or []):
                    try:
                        wanted_pers.add(int(p))
                    except (TypeError, ValueError):
                        continue
                for per, var in self._exponential_period_vars.items():
                    var.set(per in wanted_pers)
            if "exponential_rolling_operations" in raw_prefs and self._exponential_op_vars:
                wanted_exp_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("exponential_rolling_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._exponential_op_vars.items():
                    var.set(op in wanted_exp_ops)
            if "ohlc_aggregation_timeframes" in raw_prefs and self._ohlc_tf_vars:
                wanted_tfs = {
                    str(tf).strip().lower()
                    for tf in (applied.get("ohlc_aggregation_timeframes") or [])
                    if str(tf).strip()
                }
                for tf, var in self._ohlc_tf_vars.items():
                    var.set(tf in wanted_tfs)
            if "ohlc_aggregation_outputs" in raw_prefs and self._ohlc_output_vars:
                wanted_outs = {
                    str(o).strip().lower()
                    for o in (applied.get("ohlc_aggregation_outputs") or [])
                    if str(o).strip()
                }
                for fld, var in self._ohlc_output_vars.items():
                    var.set(fld in wanted_outs)
            if "normalization_windows" in raw_prefs and self._normalization_window_vars:
                wanted_norm_w = {
                    int(w)
                    for w in (applied.get("normalization_windows") or [])
                    if str(w).strip()
                }
                for win, var in self._normalization_window_vars.items():
                    var.set(win in wanted_norm_w)
            if "normalization_methods" in raw_prefs and self._normalization_method_vars:
                wanted_norm_m = {
                    str(m).strip().lower()
                    for m in (applied.get("normalization_methods") or [])
                    if str(m).strip()
                }
                for meth, var in self._normalization_method_vars.items():
                    var.set(meth in wanted_norm_m)
            if "regime_windows" in raw_prefs and self._regime_window_vars:
                wanted_reg_w = {
                    int(w)
                    for w in (applied.get("regime_windows") or [])
                    if str(w).strip()
                }
                for win, var in self._regime_window_vars.items():
                    var.set(win in wanted_reg_w)
            if "regime_methods" in raw_prefs and self._regime_method_vars:
                wanted_reg_m = {
                    str(m).strip().lower()
                    for m in (applied.get("regime_methods") or [])
                    if str(m).strip()
                }
                for meth, var in self._regime_method_vars.items():
                    var.set(meth in wanted_reg_m)
            if "math_operations" in raw_prefs and self._math_op_vars:
                wanted_math_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("math_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._math_op_vars.items():
                    var.set(op in wanted_math_ops)
            self._pending_lag_features = self._pending_features_from_prefs(
                applied, raw_prefs, "lag_features"
            )
            self._pending_difference_features = self._pending_features_from_prefs(
                applied, raw_prefs, "difference_features", fallback_key="lag_features"
            )
            self._pending_return_features = self._pending_features_from_prefs(
                applied, raw_prefs, "return_features", fallback_key="lag_features"
            )
            self._pending_rolling_features = self._pending_features_from_prefs(
                applied, raw_prefs, "rolling_features", fallback_key="lag_features"
            )
            self._pending_exponential_features = self._pending_features_from_prefs(
                applied,
                raw_prefs,
                "exponential_rolling_features",
                fallback_key="lag_features",
            )
            self._pending_ohlc_features = self._pending_features_from_prefs(
                applied, raw_prefs, "ohlc_aggregation_features"
            )
            self._pending_normalization_features = self._pending_features_from_prefs(
                applied, raw_prefs, "normalization_features", fallback_key="lag_features"
            )
            self._pending_regime_features = self._pending_features_from_prefs(
                applied, raw_prefs, "regime_features", fallback_key="lag_features"
            )
            self._pending_math_features = self._pending_features_from_prefs(
                applied, raw_prefs, "math_features", fallback_key="lag_features"
            )
        finally:
            self._loading_prefs = False
        self._refresh_lag_feature_checkboxes()
        pending = getattr(self, "_pending_lag_features", None)
        if isinstance(pending, list):
            wanted = set(pending)
            for name, var in self._lag_feature_vars.items():
                var.set(name in wanted)
            self._pending_lag_features = None
            self._rebuild_lag_feature_list()
        pending_ohlc = getattr(self, "_pending_ohlc_features", None)
        if isinstance(pending_ohlc, list):
            wanted_ohlc = set(pending_ohlc)
            for name, var in self._ohlc_feature_vars.items():
                var.set(name in wanted_ohlc)
            self._pending_ohlc_features = None
            self._rebuild_ohlc_feature_list()
        # Diff/Return horizons may have been rebuilt during feature refresh.
        self._apply_seconds_vars_from_prefs(
            applied,
            raw_prefs,
            "difference_seconds",
            "_difference_seconds_vars",
            fallback_key="lag_seconds",
        )
        self._apply_seconds_vars_from_prefs(
            applied,
            raw_prefs,
            "return_seconds",
            "_return_seconds_vars",
            fallback_key="lag_seconds",
        )
        self._refresh_lag_seconds_labels()
        self._update_lag_preview()
        self._sync_lag_body_state()
        self._sync_enable_body("_difference_body", self._difference_enabled_var)
        self._sync_enable_body("_return_body", self._return_enabled_var)
        self._sync_rolling_body_state()
        self._sync_exponential_rolling_body_state()
        self._sync_ohlc_aggregation_body_state()
        self._refresh_interaction_feature_lists()
        self._rebuild_interaction_pairs_list()
        self._sync_interaction_body_state()

    def _save_prefs(self) -> None:
        if not self.chart_dir or self._loading_prefs:
            return
        # Until Feature Transformations UI is built, keep prior list prefs on disk
        # so we do not overwrite them with empty selections.
        existing = load_master_data_prefs(self.chart_dir) or {}
        xform_ready = bool(getattr(self, "_xform_ui_built", False))

        def _list_or_existing(key: str, current: list[Any]) -> list[Any]:
            if xform_ready:
                return current
            raw = existing.get(key)
            return list(raw) if isinstance(raw, list) else current

        snapshot = master_data_prefs_snapshot(
            market=self._market_var.get(),
            interval_sec=self._interval_sec(),
            selected_days=self._selected_days,
            day_selection_explicit=self._day_selection_explicit,
            atm_band=self._atm_var.get(),
            premium_enabled=self._prem_en_var.get(),
            premium_min=self._prem_min_var.get(),
            premium_max=self._prem_max_var.get(),
            audit_required=self._audit_required_var.get(),
            no_null_data=self._no_null_data_var.get(),
            no_null_filter_report=self._no_null_filter_report_var.get(),
            trading_day_filter=self._trading_day_filter_var.get(),
            lag_enabled=self._lag_enabled_var.get(),
            lag_features=_list_or_existing("lag_features", self._selected_lag_features()),
            lag_seconds=_list_or_existing("lag_seconds", self._selected_lag_seconds()),
            difference_enabled=self._difference_enabled_var.get(),
            difference_features=_list_or_existing(
                "difference_features", self._selected_difference_features()
            ),
            difference_seconds=_list_or_existing(
                "difference_seconds", self._selected_difference_seconds()
            ),
            return_enabled=self._return_enabled_var.get(),
            return_features=_list_or_existing(
                "return_features", self._selected_return_features()
            ),
            return_seconds=_list_or_existing(
                "return_seconds", self._selected_return_seconds()
            ),
            rolling_enabled=self._rolling_enabled_var.get(),
            rolling_features=_list_or_existing(
                "rolling_features", self._selected_rolling_features()
            ),
            rolling_windows=_list_or_existing(
                "rolling_windows", self._selected_rolling_windows()
            ),
            rolling_operations=_list_or_existing(
                "rolling_operations", self._selected_rolling_operations()
            ),
            exponential_rolling_enabled=self._exponential_rolling_enabled_var.get(),
            exponential_rolling_features=_list_or_existing(
                "exponential_rolling_features", self._selected_exponential_features()
            ),
            exponential_rolling_periods=_list_or_existing(
                "exponential_rolling_periods", self._selected_exponential_periods()
            ),
            exponential_rolling_operations=_list_or_existing(
                "exponential_rolling_operations",
                self._selected_exponential_operations(),
            ),
            ohlc_aggregation_enabled=self._ohlc_aggregation_enabled_var.get(),
            ohlc_aggregation_features=_list_or_existing(
                "ohlc_aggregation_features", self._selected_ohlc_features()
            ),
            ohlc_aggregation_timeframes=_list_or_existing(
                "ohlc_aggregation_timeframes", self._selected_ohlc_timeframes()
            ),
            ohlc_aggregation_outputs=_list_or_existing(
                "ohlc_aggregation_outputs", self._selected_ohlc_outputs()
            ),
            interaction_enabled=self._interaction_enabled_var.get(),
            interaction_pairs=list(self._interaction_pairs),
            interaction_pipeline_pairs=list(self._interaction_pipeline_pairs),
            normalization_enabled=self._normalization_enabled_var.get(),
            normalization_features=_list_or_existing(
                "normalization_features", self._selected_normalization_features()
            ),
            normalization_methods=_list_or_existing(
                "normalization_methods", self._selected_normalization_methods()
            ),
            normalization_windows=_list_or_existing(
                "normalization_windows", self._selected_normalization_windows()
            ),
            regime_enabled=self._regime_enabled_var.get(),
            regime_features=_list_or_existing(
                "regime_features", self._selected_regime_features()
            ),
            regime_methods=_list_or_existing(
                "regime_methods", self._selected_regime_methods()
            ),
            regime_windows=_list_or_existing(
                "regime_windows", self._selected_regime_windows()
            ),
            regime_threshold=self._regime_threshold_var.get(),
            regime_low=self._regime_low_var.get(),
            regime_high=self._regime_high_var.get(),
            regime_n_bins=self._regime_n_bins_var.get(),
            math_enabled=self._math_enabled_var.get(),
            math_features=_list_or_existing(
                "math_features", self._selected_math_features()
            ),
            math_operations=_list_or_existing(
                "math_operations", self._selected_math_operations()
            ),
            math_clip_min=self._math_clip_min_var.get(),
            math_clip_max=self._math_clip_max_var.get(),
        )
        save_master_data_prefs(self.chart_dir, snapshot)
        save_build_config_prefs(self.chart_dir, {
            "build": {"sampling_interval_sec": self._interval_sec()},
        })

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Index").pack(side="left")
        ttk.Combobox(
            toolbar,
            textvariable=self._market_var,
            values=["NIFTY", "BANKNIFTY", "SENSEX"],
            width=12,
            state="readonly",
        ).pack(side="left", padx=(4, 12))
        self._market_var.trace_add("write", lambda *_: self._on_market_changed())

        ttk.Label(toolbar, text="Interval").pack(side="left", padx=(8, 4))
        interval_row = ttk.Frame(toolbar)
        interval_row.pack(side="left")
        for sec in INTERVALS:
            label = f"{sec}s" if sec < 60 else "1m"
            ttk.Radiobutton(
                interval_row,
                text=label,
                variable=self._interval_var,
                value=sec,
                command=self._on_interval_changed,
            ).pack(side="left", padx=2)

        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Open folder", command=self._open_folder).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Delete all", command=self._delete_all).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Add chain days →", command=self._open_chain_insert).pack(side="right", padx=4)

        ttk.Label(self, textvariable=self._status_var, foreground="#888").pack(anchor="w", padx=10)

        outer = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(outer)
        outer.add(left, weight=3)
        right = ttk.Frame(outer)
        outer.add(right, weight=2)

        self._build_overview(left)
        self._build_days_section(left)
        self._build_preview_section(left)
        self._build_sidebar(right)
        # Feature Transformations companion window is created lazily on first open.
        # Eager Toplevel() during panel __init__ can leave the main Tk window
        # invisible / unmapped on Windows (PyCharm run).

    def _ensure_feature_transformations_window(self) -> tk.Toplevel:
        """Create (once) the Feature Transformations companion window."""
        win = getattr(self, "_feature_xform_win", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    return win
            except tk.TclError:
                pass

        from .auto_feature_transform_panel import AutoFeatureTransformPanel
        from .model_registry_widgets import ScrollableFrame

        # Parent to the root Tk window — not this Frame — so withdraw/deiconify
        # never interferes with the main app shell.
        root = self.winfo_toplevel()
        win = tk.Toplevel(root)
        win.withdraw()
        win.title("Feature Transformations")
        try:
            win.transient(root)
        except tk.TclError:
            pass

        hdr = ttk.Frame(win, padding=(10, 8))
        hdr.pack(fill="x")
        ttk.Label(
            hdr,
            text="Feature Transformations",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        ttk.Button(hdr, text="Close", command=lambda: win.withdraw()).pack(side="right")

        outer_nb = ttk.Notebook(win)
        outer_nb.pack(fill="both", expand=True, padx=4, pady=(0, 8))
        self._feature_xform_outer_nb = outer_nb

        manual_tab = ttk.Frame(outer_nb)
        auto_tab = ttk.Frame(outer_nb)
        analysis_tab = ttk.Frame(outer_nb)
        outer_nb.add(manual_tab, text="Manual")
        outer_nb.add(auto_tab, text="Auto")
        outer_nb.add(analysis_tab, text="Analysis")

        xform_scroll = ScrollableFrame(manual_tab)
        xform_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 8))
        self._xform_scroll = xform_scroll
        first_build = not getattr(self, "_xform_ui_built", False)
        self._build_feature_transforms_notebook(xform_scroll.inner, populate=False)
        self._build_interaction_builder_section(xform_scroll.inner, populate=False)
        self._xform_ui_built = True

        # Phase 1A Auto workspace — second tab; Manual tab unchanged.
        self._auto_feature_panel = AutoFeatureTransformPanel(auto_tab, chart_dir=self.chart_dir)
        self._auto_feature_panel.pack(fill="both", expand=True)

        # Phase 2 Analysis Lab — build lazily on first Analysis-tab select.
        # Eager FeatureAnalysisPanel (Experiments / HCA / Auto Research UI +
        # dataset scan) was freezing the first Feature Transformations open.
        self._analysis_tab_host = analysis_tab
        self._feature_analysis_panel = None
        self._analysis_panel_placeholder = ttk.Label(
            analysis_tab,
            text="Analysis Lab loads when you open this tab…",
            foreground="#666",
            padding=12,
        )
        self._analysis_panel_placeholder.pack(anchor="w")
        outer_nb.bind("<<NotebookTabChanged>>", self._on_feature_xform_tab_changed)

        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        self._feature_xform_win = win
        self._feature_xform_placed = False
        self._xform_populate_pending = bool(first_build)
        return win

    def _on_feature_xform_tab_changed(self, _event: Any = None) -> None:
        """Lazy-create Analysis Lab only when its tab is selected."""
        nb = getattr(self, "_feature_xform_outer_nb", None)
        if nb is None:
            return
        try:
            tab_id = nb.select()
            tab_text = nb.tab(tab_id, "text")
        except tk.TclError:
            return
        if str(tab_text) != "Analysis":
            return
        self._ensure_feature_analysis_panel()

    def _ensure_feature_analysis_panel(self) -> None:
        if getattr(self, "_feature_analysis_panel", None) is not None:
            return
        host = getattr(self, "_analysis_tab_host", None)
        if host is None:
            return
        from .feature_analysis_panel import FeatureAnalysisPanel

        ph = getattr(self, "_analysis_panel_placeholder", None)
        if ph is not None:
            try:
                ph.destroy()
            except tk.TclError:
                pass
            self._analysis_panel_placeholder = None
        loading = ttk.Label(
            host,
            text="Loading Analysis Lab…",
            foreground="#666",
            padding=12,
        )
        loading.pack(anchor="w")
        host.update_idletasks()

        panel = FeatureAnalysisPanel(
            host,
            chart_dir=self.chart_dir,
            on_open_model_builder=self._on_open_model_builder,
            defer_dataset_scan=True,
        )
        try:
            loading.destroy()
        except tk.TclError:
            pass
        panel.pack(fill="both", expand=True)
        self._feature_analysis_panel = panel
        # Dataset scan after paint — avoids blocking tab switch
        host.after(30, panel.refresh_datasets)

    def _populate_feature_transformations_content(self) -> None:
        """Heavy feature lists — run after the window is visible to avoid UI freeze."""
        if not getattr(self, "_xform_populate_pending", False):
            return
        self._xform_populate_pending = False
        try:
            if hasattr(self, "_lag_preview_var"):
                self._lag_preview_var.set("Loading features…")
            if hasattr(self, "_feature_preview_var"):
                self._feature_preview_var.set("Loading…")
            self.update_idletasks()
            # Yield once so the Toplevel can paint before BooleanVar / checklist work.
            self.after(10, self._safe_hydrate_transform_widgets)
        except Exception as exc:
            try:
                if hasattr(self, "_lag_preview_var"):
                    self._lag_preview_var.set(f"Failed to load features: {exc}")
            except tk.TclError:
                pass

    def _safe_hydrate_transform_widgets(self) -> None:
        try:
            self._hydrate_transform_widgets_from_prefs()
        except Exception as exc:
            try:
                if hasattr(self, "_lag_preview_var"):
                    self._lag_preview_var.set(f"Failed to load features: {exc}")
            except tk.TclError:
                pass

    def _hydrate_transform_widgets_from_prefs(self) -> None:
        """Apply saved transform prefs after the companion panel is first built."""
        if not self.chart_dir:
            self._refresh_lag_feature_checkboxes()
            self._update_lag_preview()
            return
        applied = apply_master_data_prefs(
            load_master_data_prefs(self.chart_dir),
            allowed_intervals=INTERVALS,
        )
        raw_prefs = load_master_data_prefs(self.chart_dir) or {}
        pending: list[str] | None = None
        pending_ohlc: list[str] | None = None
        was_loading = self._loading_prefs
        self._loading_prefs = True
        try:
            if "lag_seconds" in raw_prefs and self._lag_seconds_vars:
                wanted = set()
                for s in (applied.get("lag_seconds") or []):
                    try:
                        wanted.add(int(s))
                    except (TypeError, ValueError):
                        continue
                for sec, var in self._lag_seconds_vars.items():
                    var.set(sec in wanted)
            if "rolling_windows" in raw_prefs and self._rolling_window_vars:
                wanted_wins = {
                    int(w)
                    for w in (applied.get("rolling_windows") or [])
                    if str(w).strip()
                }
                for win, var in self._rolling_window_vars.items():
                    var.set(win in wanted_wins)
            if "rolling_operations" in raw_prefs and self._rolling_op_vars:
                wanted_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("rolling_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._rolling_op_vars.items():
                    var.set(op in wanted_ops)
            if "exponential_rolling_periods" in raw_prefs and self._exponential_period_vars:
                wanted_pers = {
                    int(p)
                    for p in (applied.get("exponential_rolling_periods") or [])
                    if str(p).strip()
                }
                for per, var in self._exponential_period_vars.items():
                    var.set(per in wanted_pers)
            if "exponential_rolling_operations" in raw_prefs and self._exponential_op_vars:
                wanted_exp_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("exponential_rolling_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._exponential_op_vars.items():
                    var.set(op in wanted_exp_ops)
            if "ohlc_aggregation_timeframes" in raw_prefs and self._ohlc_tf_vars:
                wanted_tfs = {
                    str(tf).strip().lower()
                    for tf in (applied.get("ohlc_aggregation_timeframes") or [])
                    if str(tf).strip()
                }
                for tf, var in self._ohlc_tf_vars.items():
                    var.set(tf in wanted_tfs)
            if "ohlc_aggregation_outputs" in raw_prefs and self._ohlc_output_vars:
                wanted_outs = {
                    str(o).strip().lower()
                    for o in (applied.get("ohlc_aggregation_outputs") or [])
                    if str(o).strip()
                }
                for fld, var in self._ohlc_output_vars.items():
                    var.set(fld in wanted_outs)
            if "normalization_windows" in raw_prefs and self._normalization_window_vars:
                wanted_norm_w = {
                    int(w)
                    for w in (applied.get("normalization_windows") or [])
                    if str(w).strip()
                }
                for win, var in self._normalization_window_vars.items():
                    var.set(win in wanted_norm_w)
            if "normalization_methods" in raw_prefs and self._normalization_method_vars:
                wanted_norm_m = {
                    str(m).strip().lower()
                    for m in (applied.get("normalization_methods") or [])
                    if str(m).strip()
                }
                for meth, var in self._normalization_method_vars.items():
                    var.set(meth in wanted_norm_m)
            if "regime_windows" in raw_prefs and self._regime_window_vars:
                wanted_reg_w = {
                    int(w)
                    for w in (applied.get("regime_windows") or [])
                    if str(w).strip()
                }
                for win, var in self._regime_window_vars.items():
                    var.set(win in wanted_reg_w)
            if "regime_methods" in raw_prefs and self._regime_method_vars:
                wanted_reg_m = {
                    str(m).strip().lower()
                    for m in (applied.get("regime_methods") or [])
                    if str(m).strip()
                }
                for meth, var in self._regime_method_vars.items():
                    var.set(meth in wanted_reg_m)
            if "math_operations" in raw_prefs and self._math_op_vars:
                wanted_math_ops = {
                    str(op).strip().lower()
                    for op in (applied.get("math_operations") or [])
                    if str(op).strip()
                }
                for op, var in self._math_op_vars.items():
                    var.set(op in wanted_math_ops)
            pending = self._pending_features_from_prefs(
                applied, raw_prefs, "lag_features"
            )
            self._pending_difference_features = self._pending_features_from_prefs(
                applied, raw_prefs, "difference_features", fallback_key="lag_features"
            )
            self._pending_return_features = self._pending_features_from_prefs(
                applied, raw_prefs, "return_features", fallback_key="lag_features"
            )
            self._pending_rolling_features = self._pending_features_from_prefs(
                applied, raw_prefs, "rolling_features", fallback_key="lag_features"
            )
            self._pending_exponential_features = self._pending_features_from_prefs(
                applied,
                raw_prefs,
                "exponential_rolling_features",
                fallback_key="lag_features",
            )
            pending_ohlc = self._pending_features_from_prefs(
                applied, raw_prefs, "ohlc_aggregation_features"
            )
            self._pending_normalization_features = self._pending_features_from_prefs(
                applied, raw_prefs, "normalization_features", fallback_key="lag_features"
            )
            self._pending_regime_features = self._pending_features_from_prefs(
                applied, raw_prefs, "regime_features", fallback_key="lag_features"
            )
            self._pending_math_features = self._pending_features_from_prefs(
                applied, raw_prefs, "math_features", fallback_key="lag_features"
            )
        finally:
            self._loading_prefs = was_loading
        self._refresh_lag_feature_checkboxes()
        if isinstance(pending, list):
            wanted = set(pending)
            for name, var in self._lag_feature_vars.items():
                var.set(name in wanted)
            self._rebuild_lag_feature_list()
        if isinstance(pending_ohlc, list):
            wanted_ohlc = set(pending_ohlc)
            for name, var in self._ohlc_feature_vars.items():
                var.set(name in wanted_ohlc)
            self._rebuild_ohlc_feature_list()
        self._apply_seconds_vars_from_prefs(
            applied,
            raw_prefs,
            "difference_seconds",
            "_difference_seconds_vars",
            fallback_key="lag_seconds",
        )
        self._apply_seconds_vars_from_prefs(
            applied,
            raw_prefs,
            "return_seconds",
            "_return_seconds_vars",
            fallback_key="lag_seconds",
        )
        self._refresh_lag_seconds_labels()
        self._update_lag_preview()
        self._sync_lag_body_state()
        self._sync_enable_body("_difference_body", self._difference_enabled_var)
        self._sync_enable_body("_return_body", self._return_enabled_var)
        self._sync_rolling_body_state()
        self._sync_exponential_rolling_body_state()
        self._sync_ohlc_aggregation_body_state()
        self._sync_normalization_body_state()
        self._sync_regime_body_state()
        self._sync_math_body_state()
        self._refresh_interaction_feature_lists()
        self._rebuild_interaction_pairs_list()
        self._sync_interaction_body_state()

    def _open_feature_transformations(self) -> None:
        """Open Feature Transformations beside the main app (Feature Policy pattern)."""
        from .fold_replay_widgets import place_toplevel_beside_main

        win = self._ensure_feature_transformations_window()
        if not getattr(self, "_feature_xform_placed", False):
            win.update_idletasks()
            place_toplevel_beside_main(win, self)
            self._feature_xform_placed = True
        try:
            win.deiconify()
            win.lift()
            win.focus_force()
        except tk.TclError:
            self._feature_xform_win = None
            self._xform_ui_built = False
            self._xform_populate_pending = True
            win = self._ensure_feature_transformations_window()
            win.update_idletasks()
            place_toplevel_beside_main(win, self)
            self._feature_xform_placed = True
            win.deiconify()
            win.lift()
            win.focus_force()
        # Populate feature checklists after the window is on screen.
        if getattr(self, "_xform_populate_pending", False):
            self.after(50, self._populate_feature_transformations_content)

    def _open_day_metadata(self) -> None:
        """Open Day Metadata beside the main app for selected trading day(s)."""
        days = sorted(self._selected_days) if self._selected_days else []
        if not days and self._selected_day:
            days = [self._selected_day]
        if not days:
            days = list(self._master_day_keys())
        if not days:
            messagebox.showwarning("Day Metadata", "No trading days in this master database.")
            return
        path = self._master_db_path()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Day Metadata", "Master DB file does not exist.")
            return

        def _rebuild(day: str) -> None:
            from chain_replay_ml.dataset_builder.day_metadata import (
                DEFAULT_META_COLUMNS,
                feature_family_map_from_registry,
                rebuild_day_metadata_from_samples,
            )
            from chain_replay_ml.dataset_builder.gap_policy import GAP_POLICY_VERSION
            from chain_replay_ml.dataset_builder.master_store import MasterStore

            store = MasterStore(path)
            store.open()
            try:
                schema = store.get_meta("build_schema") or {}
                features = list(schema.get("feature_columns") or [])
                if not features:
                    cols = [
                        str(r[1])
                        for r in store.conn.execute("PRAGMA table_info(samples)").fetchall()
                    ]
                    meta = set(DEFAULT_META_COLUMNS)
                    features = [c for c in cols if c not in meta]
                cfg = store.get_meta("master_config") or {}
                gp = cfg.get("gap_policy") if isinstance(cfg.get("gap_policy"), dict) else {}
                try:
                    from chain_replay_ml.dataset_builder.gap_policy import gap_max_sec_from_policy

                    gap = float(gap_max_sec_from_policy(gp))
                except Exception:
                    gap = float(gp.get("gapMaxSec") or gp.get("gap_max_sec") or 20)
                step = float(cfg.get("sampling_interval_sec") or 3)
                build_ver = str(cfg.get("dataset_name") or os.path.basename(path))
                family_by_name: dict[str, str] = {}
                expectation_by_name: dict[str, dict] = {}
                registry_version = ""
                try:
                    from chain_replay_ml.dataset_builder.feature_expectation import (
                        build_expectation_index,
                    )
                    from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry

                    reg = _load_feature_registry()
                    family_by_name = feature_family_map_from_registry(reg)
                    expectation_by_name = build_expectation_index(reg)
                    registry_version = str((reg or {}).get("version") or "")
                except Exception:
                    pass
                rebuild_day_metadata_from_samples(
                    store.conn,
                    day,
                    registry_features=features,
                    meta_columns=DEFAULT_META_COLUMNS,
                    gap_max_sec=gap,
                    sampling_interval_sec=step,
                    build_version=build_ver,
                    family_by_name=family_by_name,
                    expectation_by_name=expectation_by_name,
                    ingestion={
                        "dataset_version": build_ver,
                        "registry_version": registry_version,
                        "feature_engine_version": registry_version,
                        "gap_policy_version": str(GAP_POLICY_VERSION),
                    },
                )
            finally:
                store.close()

        from .day_metadata_panel import open_day_metadata_window

        open_day_metadata_window(
            self,
            db_path=path,
            trading_days=days,
            on_rebuild=_rebuild,
        )

    def _build_overview(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Database Overview", padding=6)
        frame.pack(fill="x", pady=(0, 6))
        self._overview_labels: dict[str, ttk.Label] = {}
        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        for i, key in enumerate((
            "file_status", "total_rows", "trading_days", "features",
            "targets", "columns", "db_size", "wal_size", "journal",
        )):
            row, col = divmod(i, 3)
            cell = ttk.Frame(grid, padding=4)
            cell.grid(row=row, column=col, sticky="w", padx=4, pady=2)
            ttk.Label(cell, text=key.replace("_", " ").title(), foreground="#888").pack(anchor="w")
            val = ttk.Label(cell, text="—", font=("Segoe UI", 10, "bold"))
            val.pack(anchor="w")
            self._overview_labels[key] = val
        self._path_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._path_var, foreground="#58a6ff", wraplength=680).pack(
            anchor="w", pady=(6, 0),
        )

    def _build_days_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Trading Days in Master DB", padding=4)
        frame.pack(fill="both", expand=True, pady=4)

        cols = ("sel", "day", "samples", "tokens", "expiries", "time_range")
        self.days_tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c, w, txt in (
            ("sel", 32, "✓"),
            ("day", 100, "Trading Day"),
            ("samples", 90, "Samples"),
            ("tokens", 70, "Tokens"),
            ("expiries", 70, "Expiries"),
            ("time_range", 200, "Time range"),
        ):
            self.days_tree.heading(c, text=txt)
            anchor = "w" if c == "day" else "center"
            self.days_tree.column(c, width=w, anchor=anchor)
        self.days_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.days_tree.yview)
        sb.pack(side="right", fill="y")
        self.days_tree.configure(yscrollcommand=sb.set)
        self.days_tree.bind("<<TreeviewSelect>>", self._on_day_selected)
        self.days_tree.bind("<Button-1>", self._on_days_tree_click)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Select all", command=self._select_all_days).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self._clear_day_selection).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Delete selected day", command=self._delete_day).pack(side="left")
        ttk.Button(
            btn_row,
            text="Feature Transformations…",
            command=self._open_feature_transformations,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            btn_row,
            text="Day Metadata…",
            command=self._open_day_metadata,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(btn_row, textvariable=self._days_selected_hint_var, foreground="#888").pack(side="right", padx=4)

        self._build_trading_day_filter_section(parent)
        self._build_registry_export_section(parent)

    def _build_trading_day_filter_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Trading Day Filter", padding=6)
        frame.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        for value, label in (
            ("all", "All selected days"),
            ("exclude_expiry", "Exclude expiry days"),
            ("expiry_only", "Expiry days only"),
        ):
            ttk.Radiobutton(
                row,
                text=label,
                variable=self._trading_day_filter_var,
                value=value,
                command=self._on_trading_day_filter_changed,
            ).pack(side="left", padx=(0, 16))

    def _build_registry_export_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Create Dataset from Master DB", padding=6)
        frame.pack(fill="x", pady=(4, 0))

        stats = ttk.Frame(frame)
        stats.pack(fill="x", pady=(0, 4))
        for col, (label, var) in enumerate((
            ("Matching rows", self._registry_rows_var),
            ("Trading days", self._registry_scope_var),
            ("Features", self._registry_features_var),
            ("Registry name", self._registry_name_var),
        )):
            cell = ttk.Frame(stats, padding=4)
            cell.grid(row=0, column=col, sticky="w", padx=(0, 12))
            ttk.Label(cell, text=label, foreground="#888").pack(anchor="w")
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        gate_row = ttk.Frame(frame)
        gate_row.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            gate_row,
            text="Audit & validation required before training",
            variable=self._audit_required_var,
        ).pack(side="left")

        action_row = ttk.Frame(frame)
        action_row.pack(fill="x")
        self._create_registry_btn = ttk.Button(
            action_row,
            text="Create Dataset →",
            command=self._create_registry_dataset,
        )
        self._create_registry_btn.pack(side="left")
        ttk.Label(
            action_row,
            text="Uses current filter settings · Feature Transformations… configures packaging",
            foreground="#888",
        ).pack(side="left", padx=(8, 0))
        ttk.Label(action_row, textvariable=self._registry_progress_var, foreground="#888").pack(
            side="right", padx=4,
        )
        self._update_registry_export_panel()

    def _build_lag_transform_section(
        self, parent: ttk.Frame, *, populate: bool = True
    ) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            PRESET_SELECT_DYNAMIC,
            PRESET_SELECT_GREEKS,
            PRESET_SELECT_OI,
            PRESET_SELECT_PRICE,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="x", expand=False, anchor="nw")
        self._lag_frame = frame

        # Companion panel UI — Lag is the first transform section.
        enable_row = ttk.Frame(frame)
        enable_row.pack(anchor="w", fill="x")
        ttk.Checkbutton(
            enable_row,
            text="Lag",
            variable=self._lag_enabled_var,
            command=self._on_lag_settings_changed,
        ).pack(side="left")
        self._lag_horizon_policy_var = tk.StringVar(value="")
        ttk.Label(
            frame,
            textvariable=self._lag_horizon_policy_var,
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._lag_body = body

        search_row = ttk.Frame(body)
        search_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_row, text="Search Features").pack(side="left")
        search_entry = ttk.Entry(search_row, textvariable=self._lag_search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._lag_search_var.trace_add("write", lambda *_: self._rebuild_lag_feature_list())

        preset_row = ttk.Frame(body)
        preset_row.pack(fill="x", pady=(0, 4))
        for label, preset in (
            ("Select Dynamic", PRESET_SELECT_DYNAMIC),
            ("Select Price", PRESET_SELECT_PRICE),
            ("Select Greeks", PRESET_SELECT_GREEKS),
            ("Select OI", PRESET_SELECT_OI),
        ):
            ttk.Button(
                preset_row,
                text=label,
                command=lambda p=preset: self._apply_lag_preset(p),
            ).pack(side="left", padx=(0, 4))
        ttk.Button(preset_row, text="Clear", command=self._clear_lag_features).pack(
            side="left", padx=(0, 4),
        )
        ttk.Button(preset_row, text="Invert", command=self._invert_lag_features).pack(
            side="left",
        )

        mid = ttk.Frame(body)
        mid.pack(fill="x", expand=False)

        # Fixed widths leave room for the shared Feature Preview beside the notebook.
        # Input Features ~50% narrower; stage Preview ~25% narrower.
        _feat_w = 230
        _preview_w = 195

        feat_box = ttk.LabelFrame(mid, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 6))
        canvas = tk.Canvas(feat_box, highlightthickness=0, height=160, width=_feat_w)
        scroll = ttk.Scrollbar(feat_box, orient="vertical", command=canvas.yview)
        self._lag_feat_host = ttk.Frame(canvas)
        self._lag_feat_host.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._lag_feat_host, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="y", expand=False)
        scroll.pack(side="right", fill="y")
        self._lag_feat_canvas = canvas

        # Horizons + Preview share one row next to Features.
        right_row = ttk.Frame(mid)
        right_row.pack(side="left", fill="both", expand=True)

        self._lag_seconds_box = ttk.LabelFrame(right_row, text="Horizons", padding=4)
        self._lag_seconds_box.pack(side="left", fill="y", expand=False, padx=(0, 6))
        lag_canvas_row = ttk.Frame(self._lag_seconds_box)
        lag_canvas_row.pack(fill="both", expand=True)
        # Fixed height avoids Configure-driven resize loops that freeze Tk.
        lag_canvas = tk.Canvas(
            lag_canvas_row, highlightthickness=0, width=160, height=160
        )
        lag_scroll = ttk.Scrollbar(
            lag_canvas_row, orient="vertical", command=lag_canvas.yview
        )
        self._lag_seconds_host = ttk.Frame(lag_canvas)
        self._lag_seconds_host.bind(
            "<Configure>",
            lambda e: lag_canvas.configure(scrollregion=lag_canvas.bbox("all")),
        )
        lag_canvas.create_window((0, 0), window=self._lag_seconds_host, anchor="nw")
        lag_canvas.configure(yscrollcommand=lag_scroll.set)
        lag_canvas.pack(side="left", fill="both", expand=True)
        lag_scroll.pack(side="right", fill="y")
        self._lag_seconds_canvas = lag_canvas
        self._lag_seconds_vars = {}
        self._lag_seconds_buttons = {}
        self._rebuild_lag_horizon_checks()

        ttk.Label(
            self._lag_seconds_box,
            textvariable=self._lag_warn_var,
            foreground="#b45309",
            wraplength=160,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        lag_preview_box = ttk.LabelFrame(right_row, text="Preview", padding=4)
        lag_preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            lag_preview_box,
            textvariable=self._lag_preview_var,
            foreground="#555",
            justify="left",
            wraplength=_preview_w,
            width=28,
        ).pack(anchor="nw", fill="y", expand=False)

        self._lag_preview_box = lag_preview_box

        if populate:
            self._refresh_lag_feature_checkboxes()
            self._refresh_lag_seconds_labels()
            self._update_lag_preview()
            self._sync_lag_body_state()
        else:
            ttk.Label(
                self._lag_feat_host,
                text="Loading features…",
                foreground="#888",
            ).pack(anchor="w")
            self._lag_preview_var.set("Loading…")
            self._feature_preview_var.set("Loading…")

    def _build_simple_time_shift_tab(
        self,
        parent: ttk.Frame,
        *,
        kind: str,
        title: str,
        enable_var: tk.BooleanVar,
        feature_vars_attr: str,
        seconds_vars_attr: str,
        feat_host_attr: str,
        seconds_host_attr: str,
        body_attr: str,
        preview_var: tk.StringVar,
        warn_var: tk.StringVar,
        on_change,
    ) -> None:
        """Difference / Return tab: enable + input features + horizons + preview."""
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        ttk.Checkbutton(
            frame,
            text=title,
            variable=enable_var,
            command=on_change,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Own Input Features and Horizons (independent of Lag).",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, pady=(4, 0))
        setattr(self, body_attr, body)

        # Input Features ~50% narrower; stage Preview ~25% narrower.
        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        setattr(
            self,
            feat_host_attr,
            self._make_scroll_host(feat_box, height=160, width=230),
        )
        setattr(self, feature_vars_attr, {})

        sec_box = ttk.LabelFrame(body, text="Horizons", padding=4)
        sec_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        setattr(
            self,
            seconds_host_attr,
            self._make_scroll_host(sec_box, height=160, width=160),
        )
        setattr(self, seconds_vars_attr, {})
        self._rebuild_simple_horizon_checks(
            seconds_host_attr=seconds_host_attr,
            seconds_vars_attr=seconds_vars_attr,
            on_change=on_change,
        )
        ttk.Label(
            sec_box,
            textvariable=warn_var,
            foreground="#b45309",
            wraplength=160,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._refresh_flat_feature_checkboxes(
            feat_host_attr=feat_host_attr,
            feature_vars_attr=feature_vars_attr,
            pending_attr=f"_pending_{kind}_features",
            on_change=on_change,
        )
        self._sync_enable_body(body_attr, enable_var)

    def _rebuild_simple_horizon_checks(
        self,
        *,
        seconds_host_attr: str,
        seconds_vars_attr: str,
        on_change,
    ) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            default_lag_seconds_for_interval,
            lag_seconds_label,
        )

        host = getattr(self, seconds_host_attr, None)
        if host is None:
            return
        for child in host.winfo_children():
            child.destroy()
        interval = self._interval_sec()
        prev_map: dict[int, tk.BooleanVar] = getattr(self, seconds_vars_attr) or {}
        prev = {sec: bool(var.get()) for sec, var in prev_map.items()}
        new_vars: dict[int, tk.BooleanVar] = {}
        try:
            horizons = list(default_lag_seconds_for_interval(interval))
        except Exception:
            horizons = list(default_lag_seconds_for_interval(3))
        had_prev = bool(prev)
        for sec in horizons:
            selected = prev.get(int(sec), True) if had_prev else True
            if had_prev and int(sec) not in prev:
                selected = True
            var = tk.BooleanVar(value=selected)
            new_vars[int(sec)] = var
            ttk.Checkbutton(
                host,
                text=lag_seconds_label(int(sec), interval),
                variable=var,
                command=on_change,
            ).pack(anchor="w")
        setattr(self, seconds_vars_attr, new_vars)

    def _refresh_flat_feature_checkboxes(
        self,
        *,
        feat_host_attr: str,
        feature_vars_attr: str,
        pending_attr: str,
        on_change,
        default_fn=None,
    ) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            default_selected_lag_features,
            group_features_by_category,
        )

        host = getattr(self, feat_host_attr, None)
        if host is None:
            return
        available = self._laggable_feature_names()
        previous = {
            name: bool(var.get())
            for name, var in (getattr(self, feature_vars_attr) or {}).items()
        }
        pending = getattr(self, pending_attr, None)
        var_map: dict[str, tk.BooleanVar] = {}
        defaults_fn = default_fn or default_selected_lag_features
        defaults = set(defaults_fn(available)) if available else set()
        for name in available:
            if isinstance(pending, list):
                checked = name in set(pending)
            elif name in previous:
                checked = previous[name]
            else:
                checked = name in defaults
            var_map[name] = tk.BooleanVar(value=checked)
        setattr(self, feature_vars_attr, var_map)
        if isinstance(pending, list):
            setattr(self, pending_attr, None)
        for child in host.winfo_children():
            child.destroy()
        if not available:
            ttk.Label(
                host,
                text="No feature columns in master schema yet.",
                foreground="#888",
            ).pack(anchor="w")
            return
        # Per-stage category expand/collapse tracking.
        expanded_attr = f"{feat_host_attr}_category_expanded"
        if not hasattr(self, expanded_attr):
            setattr(self, expanded_attr, {})
        expanded_map: dict[str, bool] = getattr(self, expanded_attr)
        grouped = group_features_by_category(available)
        for cat, names in grouped.items():
            expanded_map.setdefault(cat, False)
            is_expanded = expanded_map.get(cat, False)
            marker = "▼" if is_expanded else "▶"
            selected_n = sum(
                1 for n in names
                if n in var_map and var_map[n].get()
            )
            header = ttk.Frame(host)
            header.pack(fill="x", anchor="w", pady=(4, 0))
            ttk.Button(
                header,
                text=f"{marker} {cat}  ({selected_n}/{len(names)})",
                width=36,
                command=lambda c=cat, fh=feat_host_attr, fv=feature_vars_attr,
                               pa=pending_attr, oc=on_change, df=default_fn:
                    self._toggle_flat_category(c, fh, fv, pa, oc, df),
            ).pack(side="left", anchor="w")
            if not is_expanded:
                continue
            for name in names:
                var = var_map.get(name)
                if var is None:
                    var = tk.BooleanVar(value=False)
                    var_map[name] = var
                ttk.Checkbutton(
                    host,
                    text=f"    {name}",
                    variable=var,
                    command=on_change,
                ).pack(anchor="w")

    def _toggle_flat_category(
        self,
        category: str,
        feat_host_attr: str,
        feature_vars_attr: str,
        pending_attr: str,
        on_change,
        default_fn,
    ) -> None:
        expanded_attr = f"{feat_host_attr}_category_expanded"
        expanded_map: dict[str, bool] = getattr(self, expanded_attr, {})
        expanded_map[category] = not expanded_map.get(category, False)
        setattr(self, expanded_attr, expanded_map)
        self._refresh_flat_feature_checkboxes(
            feat_host_attr=feat_host_attr,
            feature_vars_attr=feature_vars_attr,
            pending_attr=pending_attr,
            on_change=on_change,
            default_fn=default_fn,
        )

    def _selected_from_feature_vars(self, feature_vars_attr: str) -> list[str]:
        return [
            name
            for name, var in (getattr(self, feature_vars_attr) or {}).items()
            if bool(var.get())
        ]

    def _selected_from_seconds_vars(self, seconds_vars_attr: str) -> list[int]:
        return sorted(
            int(sec)
            for sec, var in (getattr(self, seconds_vars_attr) or {}).items()
            if bool(var.get())
        )

    def _sync_enable_body(self, body_attr: str, enable_var: tk.BooleanVar) -> None:
        body = getattr(self, body_attr, None)
        if body is None:
            return
        state = "normal" if enable_var.get() else "disabled"

        def _walk(widget: tk.Misc) -> None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _walk(child)

        _walk(body)

    def _build_feature_transforms_notebook(
        self, parent: ttk.Frame, *, populate: bool = True
    ) -> None:
        """Lag / Diff / Return / Rolling Statistics / Exp / OHLC / Norm / Regime / Math.

        Feature Preview sits beside the notebook so it stays visible for every tab.
        """
        host = ttk.Frame(parent)
        host.pack(fill="both", expand=True, anchor="nw")

        row = ttk.Frame(host)
        row.pack(fill="both", expand=True)

        # Pack Feature Preview on the right first so it always keeps reserved width.
        feature_preview_box = ttk.LabelFrame(row, text="Feature Preview", padding=4)
        feature_preview_box.pack(side="right", fill="y", expand=False, padx=(6, 0))
        ttk.Label(
            feature_preview_box,
            textvariable=self._feature_preview_var,
            foreground="#555",
            justify="left",
            wraplength=240,
            width=34,
        ).pack(anchor="nw", fill="y", expand=False)
        self._feature_preview_box = feature_preview_box

        nb_host = ttk.Frame(row)
        nb_host.pack(side="left", fill="both", expand=True)
        nb = ttk.Notebook(nb_host)
        nb.pack(fill="both", expand=True)
        self._feature_transforms_nb = nb

        lag_tab = ttk.Frame(nb, padding=6)
        diff_tab = ttk.Frame(nb, padding=6)
        ret_tab = ttk.Frame(nb, padding=6)
        rolling_tab = ttk.Frame(nb, padding=6)
        exp_tab = ttk.Frame(nb, padding=6)
        ohlc_tab = ttk.Frame(nb, padding=6)
        norm_tab = ttk.Frame(nb, padding=6)
        regime_tab = ttk.Frame(nb, padding=6)
        math_tab = ttk.Frame(nb, padding=6)
        nb.add(lag_tab, text="Lag")
        nb.add(diff_tab, text="Difference")
        nb.add(ret_tab, text="Return")
        nb.add(rolling_tab, text="Rolling Statistics")
        nb.add(exp_tab, text="Exponential Rolling")
        nb.add(ohlc_tab, text="OHLC Aggregation")
        nb.add(norm_tab, text="Normalization")
        nb.add(regime_tab, text="Regime / Bucket")
        nb.add(math_tab, text="Math (Unary)")

        self._build_lag_transform_section(lag_tab, populate=populate)
        self._build_simple_time_shift_tab(
            diff_tab,
            kind="difference",
            title="Difference  (feature − lag)",
            enable_var=self._difference_enabled_var,
            feature_vars_attr="_difference_feature_vars",
            seconds_vars_attr="_difference_seconds_vars",
            feat_host_attr="_difference_feat_host",
            seconds_host_attr="_difference_seconds_host",
            body_attr="_difference_body",
            preview_var=self._difference_preview_var,
            warn_var=self._difference_warn_var,
            on_change=self._on_difference_settings_changed,
        )
        self._build_simple_time_shift_tab(
            ret_tab,
            kind="return",
            title="Return  ((feature − lag) / lag)",
            enable_var=self._return_enabled_var,
            feature_vars_attr="_return_feature_vars",
            seconds_vars_attr="_return_seconds_vars",
            feat_host_attr="_return_feat_host",
            seconds_host_attr="_return_seconds_host",
            body_attr="_return_body",
            preview_var=self._return_preview_var,
            warn_var=self._return_warn_var,
            on_change=self._on_return_settings_changed,
        )
        self._build_rolling_transform_section(rolling_tab)
        self._build_exponential_rolling_transform_section(exp_tab)
        self._build_ohlc_aggregation_transform_section(ohlc_tab)
        self._build_normalization_transform_section(norm_tab)
        self._build_regime_transform_section(regime_tab)
        self._build_math_transform_section(math_tab)

    def _build_window_transforms_notebook(self, parent: ttk.Frame) -> None:
        """Backward-compatible alias — unified notebook owns all transform tabs."""
        self._build_feature_transforms_notebook(parent, populate=True)

    def _sync_horizons_canvas_height(self, *_args: object) -> None:
        """No-op kept for callers; height is fixed to avoid Configure loops."""
        return

    def _update_lag_horizon_policy_hint(self) -> None:
        if not hasattr(self, "_lag_horizon_policy_var"):
            return
        try:
            from chain_replay_ml.dataset_builder.transformations.horizon_policy import (
                format_horizon_policy_summary,
            )

            self._lag_horizon_policy_var.set(
                format_horizon_policy_summary(self._interval_sec())
            )
        except Exception as exc:
            self._lag_horizon_policy_var.set(f"Horizon policy unavailable: {exc}")

    def _rebuild_lag_horizon_checks(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            default_lag_seconds_for_interval,
            lag_seconds_label,
        )

        if not hasattr(self, "_lag_seconds_host"):
            return
        for child in self._lag_seconds_host.winfo_children():
            child.destroy()
        interval = self._interval_sec()
        prev = {
            sec: bool(var.get())
            for sec, var in (self._lag_seconds_vars or {}).items()
        }
        self._lag_seconds_vars = {}
        self._lag_seconds_buttons = {}
        try:
            horizons = list(default_lag_seconds_for_interval(interval))
        except Exception:
            horizons = list(default_lag_seconds_for_interval(3))
        # If prior prefs had a selection, keep intersection; else default all on.
        had_prev = bool(prev)
        for sec in horizons:
            selected = prev.get(int(sec), True) if had_prev else True
            if had_prev and int(sec) not in prev:
                selected = True
            var = tk.BooleanVar(value=selected)
            self._lag_seconds_vars[int(sec)] = var
            btn = ttk.Checkbutton(
                self._lag_seconds_host,
                text=lag_seconds_label(int(sec), interval),
                variable=var,
                command=self._on_lag_settings_changed,
            )
            btn.pack(anchor="w")
            self._lag_seconds_buttons[int(sec)] = btn
        self._update_lag_horizon_policy_hint()

    def _build_rolling_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
            DEFAULT_ROLLING_OPS,
            DEFAULT_ROLLING_WINDOWS,
            OP_DISPLAY_LABELS,
            ROLLING_OPS,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._rolling_frame = frame

        ttk.Checkbutton(
            frame,
            text="Rolling Statistics",
            variable=self._rolling_enabled_var,
            command=self._on_rolling_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Window aggregations (mean / std / min / max / median).",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._rolling_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        self._rolling_feat_host = self._make_scroll_host(feat_box, height=160, width=230)
        self._rolling_feature_vars = {}
        self._refresh_flat_feature_checkboxes(
            feat_host_attr="_rolling_feat_host",
            feature_vars_attr="_rolling_feature_vars",
            pending_attr="_pending_rolling_features",
            on_change=self._on_rolling_settings_changed,
        )

        win_box = ttk.LabelFrame(body, text="Windows (row)", padding=4)
        win_box.pack(side="left", fill="y", padx=(0, 8))
        self._rolling_window_vars = {}
        for win in DEFAULT_ROLLING_WINDOWS:
            var = tk.BooleanVar(value=True)
            self._rolling_window_vars[int(win)] = var
            ttk.Checkbutton(
                win_box,
                text=str(int(win)),
                variable=var,
                command=self._on_rolling_settings_changed,
            ).pack(anchor="w")

        op_box = ttk.LabelFrame(body, text="Operations", padding=4)
        op_box.pack(side="left", fill="y", padx=(0, 8))
        self._rolling_op_vars = {}
        default_ops = set(DEFAULT_ROLLING_OPS)
        for op in ROLLING_OPS:
            var = tk.BooleanVar(value=op in default_ops)
            self._rolling_op_vars[str(op)] = var
            ttk.Checkbutton(
                op_box,
                text=OP_DISPLAY_LABELS.get(op, op),
                variable=var,
                command=self._on_rolling_settings_changed,
            ).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._rolling_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._update_lag_preview()
        self._sync_rolling_body_state()

    def _build_exponential_rolling_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.exponential_rolling import (
            EXPONENTIAL_ROLLING_OPS,
        )
        from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
            DEFAULT_EMA_PERIODS,
            OP_DISPLAY_LABELS,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._exponential_rolling_frame = frame

        ttk.Checkbutton(
            frame,
            text="Exponential Rolling",
            variable=self._exponential_rolling_enabled_var,
            command=self._on_exponential_rolling_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Own Input Features (independent of Lag / Diff / Return).",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._exponential_rolling_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        self._exponential_feat_host = self._make_scroll_host(
            feat_box, height=160, width=230
        )
        self._exponential_feature_vars = {}
        self._refresh_flat_feature_checkboxes(
            feat_host_attr="_exponential_feat_host",
            feature_vars_attr="_exponential_feature_vars",
            pending_attr="_pending_exponential_features",
            on_change=self._on_exponential_rolling_settings_changed,
        )

        per_box = ttk.LabelFrame(body, text="Periods (span)", padding=4)
        per_box.pack(side="left", fill="y", padx=(0, 8))
        self._exponential_period_vars = {}
        for per in DEFAULT_EMA_PERIODS:
            var = tk.BooleanVar(value=True)
            self._exponential_period_vars[int(per)] = var
            ttk.Checkbutton(
                per_box,
                text=str(int(per)),
                variable=var,
                command=self._on_exponential_rolling_settings_changed,
            ).pack(anchor="w")

        op_box = ttk.LabelFrame(body, text="Operations", padding=4)
        op_box.pack(side="left", fill="y", padx=(0, 8))
        self._exponential_op_vars = {}
        for op in EXPONENTIAL_ROLLING_OPS:
            # Default EMA on for backward-compatible prefs without operations key.
            var = tk.BooleanVar(value=(op == "ema"))
            self._exponential_op_vars[str(op)] = var
            ttk.Checkbutton(
                op_box,
                text=OP_DISPLAY_LABELS.get(op, op),
                variable=var,
                command=self._on_exponential_rolling_settings_changed,
            ).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._exponential_rolling_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._update_lag_preview()
        self._sync_exponential_rolling_body_state()

    def _build_ohlc_aggregation_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            FIELD_DISPLAY_LABELS,
            OHLC_FIELDS,
            timeframe_display_label,
        )
        from chain_replay_ml.dataset_builder.transformations.ohlc_history_profiles import (
            available_ohlc_timeframes,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._ohlc_aggregation_frame = frame

        ttk.Checkbutton(
            frame,
            text="OHLC Aggregation",
            variable=self._ohlc_aggregation_enabled_var,
            command=self._on_ohlc_aggregation_settings_changed,
        ).pack(anchor="w")
        self._ohlc_interval_hint_var = tk.StringVar(value="")
        ttk.Label(
            frame,
            textvariable=self._ohlc_interval_hint_var,
            foreground="#666",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, pady=(4, 0))
        self._ohlc_aggregation_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        ttk.Label(
            feat_box,
            text="OHLC only — independent of Lag/Rolling features",
            foreground="#666",
        ).pack(anchor="w")
        self._ohlc_feat_host = self._make_scroll_host(feat_box, height=160, width=230)
        self._ohlc_feature_vars = {}

        self._ohlc_tf_box = ttk.LabelFrame(body, text="Timeframes", padding=4)
        self._ohlc_tf_box.pack(side="left", fill="y", padx=(0, 8))
        self._ohlc_tf_vars = {}
        self._rebuild_ohlc_timeframe_checks()

        out_box = ttk.LabelFrame(body, text="Outputs", padding=4)
        out_box.pack(side="left", fill="y", padx=(0, 8))
        self._ohlc_output_vars = {}
        for fld in OHLC_FIELDS:
            var = tk.BooleanVar(value=True)
            self._ohlc_output_vars[str(fld)] = var
            ttk.Checkbutton(
                out_box,
                text=FIELD_DISPLAY_LABELS.get(fld, fld),
                variable=var,
                command=self._on_ohlc_aggregation_settings_changed,
            ).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._ohlc_aggregation_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._update_ohlc_interval_hint()
        self._refresh_ohlc_feature_checkboxes()
        self._update_lag_preview()
        self._sync_ohlc_aggregation_body_state()

    def _update_ohlc_interval_hint(self) -> None:
        if not hasattr(self, "_ohlc_interval_hint_var"):
            return
        from chain_replay_ml.dataset_builder.transformations.ohlc_history_profiles import (
            format_ohlc_approximation_hint,
            format_ohlc_unavailable_hint,
        )

        interval = self._interval_sec()
        lines = [
            f"Independent Input Features (not Lag list). History profile for {interval}s sampling "
            "(edit ohlc_history_profiles.json to extend)."
        ]
        approx = format_ohlc_approximation_hint(interval)
        if approx:
            lines.append(approx)
        unavailable = format_ohlc_unavailable_hint(interval)
        if unavailable:
            lines.append(unavailable)
        self._ohlc_interval_hint_var.set("\n".join(lines))

    def _rebuild_ohlc_timeframe_checks(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            timeframe_display_label,
        )
        from chain_replay_ml.dataset_builder.transformations.ohlc_history_profiles import (
            available_ohlc_timeframes,
        )

        if not hasattr(self, "_ohlc_tf_box"):
            return
        for child in self._ohlc_tf_box.winfo_children():
            child.destroy()
        interval = self._interval_sec()
        prev = {
            tf: bool(var.get())
            for tf, var in (self._ohlc_tf_vars or {}).items()
        }
        self._ohlc_tf_vars = {}
        try:
            keys = list(available_ohlc_timeframes(interval))
        except Exception:
            keys = list(available_ohlc_timeframes(3))
        for tf in keys:
            # Keep prior selection when still available; default new keys on.
            var = tk.BooleanVar(value=prev.get(tf, True))
            self._ohlc_tf_vars[str(tf)] = var
            ttk.Checkbutton(
                self._ohlc_tf_box,
                text=timeframe_display_label(tf, sample_interval_sec=interval),
                variable=var,
                command=self._on_ohlc_aggregation_settings_changed,
            ).pack(anchor="w")

    def _build_interaction_builder_section(
        self, parent: ttk.Frame, *, populate: bool = True
    ) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            OP_CHOICES,
            OP_DISPLAY_LABELS,
        )

        box = ttk.LabelFrame(parent, text="Interaction Builder", padding=6)
        box.pack(fill="both", expand=True, pady=(10, 0))
        self._interaction_frame = box

        ttk.Checkbutton(
            box,
            text="Interaction",
            variable=self._interaction_enabled_var,
            command=self._on_interaction_settings_changed,
        ).pack(anchor="w")

        body = ttk.Frame(box)
        body.pack(fill="both", expand=True, pady=(4, 0))
        self._interaction_body = body

        notebook = ttk.Notebook(body)
        notebook.pack(fill="both", expand=True)
        single_tab = ttk.Frame(notebook, padding=4)
        bulk_tab = ttk.Frame(notebook, padding=4)
        notebook.add(single_tab, text="Single Interaction")
        notebook.add(bulk_tab, text="Bulk Interaction")

        # --- Tab 1: Single Interaction ---
        src_a_row = ttk.Frame(single_tab)
        src_a_row.pack(fill="x", pady=(0, 4))
        ttk.Label(src_a_row, text="Feature A Source").pack(side="left")
        self._interaction_source_a_combo = ttk.Combobox(
            src_a_row,
            textvariable=self._interaction_source_a_var,
            values=["Master Features"],
            state="readonly",
            width=22,
        )
        self._interaction_source_a_combo.pack(side="left", padx=(6, 0))
        self._interaction_source_a_combo.bind(
            "<<ComboboxSelected>>", lambda *_: self._on_interaction_source_a_changed()
        )

        search_a_row = ttk.Frame(single_tab)
        search_a_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_a_row, text="Feature A Search", width=16).pack(side="left")
        ent_a = ttk.Entry(search_a_row, textvariable=self._interaction_search_a_var)
        ent_a.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ent_a.bind("<KeyRelease>", lambda _e: self._filter_interaction_single_features("a"))

        feat_a_row = ttk.Frame(single_tab)
        feat_a_row.pack(fill="x", pady=(0, 6))
        ttk.Label(feat_a_row, text="Feature A", width=16).pack(side="left")
        self._interaction_feat_a_combo = ttk.Combobox(
            feat_a_row,
            textvariable=self._interaction_feat_a_var,
            values=[],
            width=40,
        )
        self._interaction_feat_a_combo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._interaction_feat_a_combo.bind(
            "<<ComboboxSelected>>", lambda *_: self._on_interaction_builder_changed()
        )
        self._interaction_feat_a_var.trace_add(
            "write", lambda *_: self._on_interaction_builder_changed()
        )

        src_b_row = ttk.Frame(single_tab)
        src_b_row.pack(fill="x", pady=(0, 4))
        ttk.Label(src_b_row, text="Feature B Source").pack(side="left")
        self._interaction_source_b_combo = ttk.Combobox(
            src_b_row,
            textvariable=self._interaction_source_b_var,
            values=["Master Features"],
            state="readonly",
            width=22,
        )
        self._interaction_source_b_combo.pack(side="left", padx=(6, 0))
        self._interaction_source_b_combo.bind(
            "<<ComboboxSelected>>", lambda *_: self._on_interaction_source_b_changed()
        )

        search_b_row = ttk.Frame(single_tab)
        search_b_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_b_row, text="Feature B Search", width=16).pack(side="left")
        ent_b = ttk.Entry(search_b_row, textvariable=self._interaction_search_b_var)
        ent_b.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ent_b.bind("<KeyRelease>", lambda _e: self._filter_interaction_single_features("b"))

        feat_b_row = ttk.Frame(single_tab)
        feat_b_row.pack(fill="x", pady=(0, 6))
        ttk.Label(feat_b_row, text="Feature B", width=16).pack(side="left")
        self._interaction_feat_b_combo = ttk.Combobox(
            feat_b_row,
            textvariable=self._interaction_feat_b_var,
            values=[],
            width=40,
        )
        self._interaction_feat_b_combo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._interaction_feat_b_combo.bind(
            "<<ComboboxSelected>>", lambda *_: self._on_interaction_builder_changed()
        )
        self._interaction_feat_b_var.trace_add(
            "write", lambda *_: self._on_interaction_builder_changed()
        )

        op_row = ttk.Frame(single_tab)
        op_row.pack(fill="x", pady=(0, 4))
        ttk.Label(op_row, text="Operation", width=16).pack(side="left")
        op_labels = [OP_DISPLAY_LABELS[k] for k in OP_CHOICES]
        self._interaction_op_labels = {OP_DISPLAY_LABELS[k]: k for k in OP_CHOICES}
        self._interaction_op_display = tk.StringVar(value=OP_DISPLAY_LABELS["multiply"])
        op_combo = ttk.Combobox(
            op_row,
            textvariable=self._interaction_op_display,
            values=op_labels,
            state="readonly",
            width=18,
        )
        op_combo.pack(side="left", padx=(4, 0))
        op_combo.bind("<<ComboboxSelected>>", lambda *_: self._on_interaction_builder_changed())

        out_row = ttk.Frame(single_tab)
        out_row.pack(fill="x", pady=(0, 6))
        ttk.Label(out_row, text="Output Preview", width=16).pack(side="left")
        ttk.Label(
            out_row,
            textvariable=self._interaction_output_var,
            foreground="#333",
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        ttk.Button(single_tab, text="Add", command=self._add_interaction_pair).pack(
            anchor="w", pady=(4, 0)
        )

        # --- Tab 2: Bulk Interaction ---
        bulk_search = ttk.Frame(bulk_tab)
        bulk_search.pack(fill="x", pady=(0, 4))
        bulk_search.columnconfigure(1, weight=1)
        bulk_search.columnconfigure(3, weight=1)
        ttk.Label(bulk_search, text="Feature A Search").grid(row=0, column=0, sticky="w")
        ent_ba = ttk.Entry(bulk_search, textvariable=self._interaction_bulk_search_a_var)
        ent_ba.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ent_ba.bind("<KeyRelease>", lambda _e: self._apply_interaction_bulk_filter("a"))
        ttk.Label(bulk_search, text="Feature B Search").grid(row=0, column=2, sticky="w")
        ent_bb = ttk.Entry(bulk_search, textvariable=self._interaction_bulk_search_b_var)
        ent_bb.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        ent_bb.bind("<KeyRelease>", lambda _e: self._apply_interaction_bulk_filter("b"))

        bulk_cols = ttk.Frame(bulk_tab)
        bulk_cols.pack(fill="both", expand=True, pady=(4, 4))
        bulk_cols.columnconfigure(0, weight=15, uniform="ix_bulk")
        bulk_cols.columnconfigure(1, weight=17, uniform="ix_bulk")
        left_bulk = ttk.Frame(bulk_cols)
        left_bulk.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        right_bulk = ttk.Frame(bulk_cols)
        right_bulk.grid(row=0, column=1, sticky="nsew")
        bulk_cols.rowconfigure(0, weight=1)
        ttk.Label(left_bulk, text="Feature A Set").pack(anchor="w")
        ttk.Label(right_bulk, text="Feature B Set").pack(anchor="w")
        self._interaction_bulk_a_host = self._make_scroll_host(left_bulk, height=180)
        self._interaction_bulk_b_host = self._make_scroll_host(right_bulk, height=180)

        bulk_ctrl = ttk.LabelFrame(bulk_tab, text="Bulk Pair Generation", padding=4)
        bulk_ctrl.pack(fill="x", pady=(4, 4))
        ttk.Button(
            bulk_ctrl,
            text="Generate Bulk Pairs",
            command=self._generate_bulk_interaction_pairs,
        ).pack(side="left")
        ttk.Button(
            bulk_ctrl,
            text="Clear",
            command=self._clear_interaction_pairs,
        ).pack(side="left", padx=(6, 0))

        pairs_frame = ttk.LabelFrame(body, text="Configured Pairs (with lineage)", padding=4)
        pairs_frame.pack(fill="both", expand=True, pady=(6, 0))
        pairs_toolbar = ttk.Frame(pairs_frame)
        pairs_toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(
            pairs_toolbar,
            text="Remove selected",
            command=self._remove_interaction_pair,
        ).pack(side="left")
        ttk.Button(
            pairs_toolbar,
            text="Add to Pipeline",
            command=self._add_interaction_pairs_to_pipeline,
        ).pack(side="left", padx=(6, 0))
        self._interaction_pairs_list = tk.Listbox(pairs_frame, height=8, font=("Consolas", 9))
        self._interaction_pairs_list.pack(fill="both", expand=True)

        self._interaction_lineage_var = tk.StringVar(value="")
        ttk.Label(
            body,
            textvariable=self._interaction_lineage_var,
            foreground="#555",
            justify="left",
            font=("Consolas", 9),
        ).pack(anchor="w", pady=(4, 0))

        if populate:
            self._refresh_interaction_feature_lists()
            self._rebuild_interaction_pairs_list()
            self._sync_interaction_body_state()
        else:
            self._rebuild_interaction_pairs_list()
            self._sync_interaction_body_state()

    def _build_preview_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Sample Preview", padding=4)
        frame.pack(fill="both", expand=True, pady=4)

        filter_row = ttk.Frame(frame)
        filter_row.pack(fill="x", pady=(0, 4))

        ttk.Label(filter_row, text="ATM ±").pack(side="left", padx=(0, 2))
        self._atm_var = tk.StringVar(value="10")
        ttk.Combobox(
            filter_row, textvariable=self._atm_var, values=["10", "15", ""], width=6, state="readonly",
        ).pack(side="left")

        self._prem_en_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(filter_row, text="LTP", variable=self._prem_en_var).pack(side="left", padx=(12, 2))
        self._prem_min_var = tk.StringVar(value="15")
        self._prem_max_var = tk.StringVar(value="40")
        ttk.Entry(filter_row, textvariable=self._prem_min_var, width=5).pack(side="left")
        ttk.Label(filter_row, text="–").pack(side="left")
        ttk.Entry(filter_row, textvariable=self._prem_max_var, width=5).pack(side="left")

        self._apply_filters_btn = ttk.Button(
            filter_row,
            text="Apply filters",
            command=self._apply_preview,
        )
        self._apply_filters_btn.pack(side="right", padx=(4, 0))
        ttk.Checkbutton(
            filter_row,
            text="No null data",
            variable=self._no_null_data_var,
            command=self._on_no_null_data_toggled,
        ).pack(side="right", padx=(8, 0))
        self._download_csv_btn = ttk.Button(
            filter_row,
            text="⬇ CSV",
            command=self._download_preview_csv,
            state="disabled",
        )
        self._download_csv_btn.pack(side="right", padx=(4, 0))

        hint_row = ttk.Frame(frame)
        hint_row.pack(fill="x", pady=(0, 2))
        self._preview_hint_var = tk.StringVar(value="")
        ttk.Label(hint_row, textvariable=self._preview_hint_var, foreground="#888", wraplength=520).pack(
            side="left", anchor="w",
        )
        self._clear_token_btn = ttk.Button(
            hint_row,
            text="Clear token filter",
            command=self._clear_token_filter,
            state="disabled",
        )
        self._clear_token_btn.pack(side="right", padx=(4, 0))
        ttk.Checkbutton(
            hint_row,
            text="No-Null Data Filter Report",
            variable=self._no_null_filter_report_var,
        ).pack(side="right", padx=(8, 0))

        prev_cols = ("trading_day", "timestamp", "token", "symbol", "ltp", "spot")
        self.preview_tree = ttk.Treeview(frame, columns=prev_cols, show="headings", height=6)
        for c, txt in zip(prev_cols, ("Day", "Time", "Token", "Symbol", "LTP", "Spot")):
            self.preview_tree.heading(c, text=txt)
            self.preview_tree.column(c, width=90 if c != "symbol" else 140)
        self.preview_tree.pack(fill="both", expand=True, pady=4)
        self.preview_tree.bind("<Double-1>", self._on_preview_row_double_click)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        self._sidebar_nb = notebook

        # Feature Transformations opens as a companion panel (button next to
        # Delete selected day), same pattern as Feature Policy on Create Dataset.

        # Non-migration tab so interval switches don't auto-analyze a multi-GB DB.
        builder_tab = ttk.Frame(notebook, padding=6)
        notebook.add(builder_tab, text="Builder Progress")
        self._builder_tab = builder_tab
        notebook.select(builder_tab)
        self._builder_text = scrolledtext.ScrolledText(
            builder_tab,
            font=("Consolas", 9),
            wrap="word",
            height=16,
        )
        self._builder_text.pack(anchor="nw", fill="both", expand=True)
        self._set_builder_text("—")
        self._registry_export_log: list[str] = []
        self._registry_export_t0: float | None = None

        mig_tab = ttk.Frame(notebook, padding=4)
        notebook.add(mig_tab, text="Feature Migration")
        self._migration_panel = FeatureMigrationPanel(
            mig_tab,
            resolve_db_path=self._master_db_path,
            db_exists=lambda: os.path.isfile(self._master_db_path()),
            on_migrated=lambda: self.refresh(lazy=False),
            is_external_busy=lambda: self._registry_runner.running or self._export_count_busy,
        )
        self._migration_panel.pack(fill="both", expand=True)

        meta_tab = ttk.Frame(notebook, padding=4)
        notebook.add(meta_tab, text="Dataset Meta")
        self._meta_text = scrolledtext.ScrolledText(meta_tab, font=("Consolas", 9), wrap="word")
        self._meta_text.pack(fill="both", expand=True)

        health_tab = ttk.Frame(notebook, padding=4)
        notebook.add(health_tab, text="Feature Health")
        self._health_text = scrolledtext.ScrolledText(health_tab, font=("Consolas", 9), wrap="word")
        self._health_text.pack(fill="both", expand=True)

        tables_tab = ttk.Frame(notebook, padding=4)
        notebook.add(tables_tab, text="SQLite Tables")
        self._tables_tree = ttk.Treeview(tables_tab, columns=("rows", "cols"), show="headings")
        self._tables_tree.heading("#0", text="Table")
        self._tables_tree.heading("rows", text="Rows")
        self._tables_tree.heading("cols", text="Cols")
        self._tables_tree.column("#0", width=160)
        self._tables_tree.column("rows", width=70, anchor="e")
        self._tables_tree.column("cols", width=50, anchor="e")
        self._tables_tree.pack(fill="both", expand=True)

        notebook.bind("<<NotebookTabChanged>>", self._on_sidebar_tab_changed)

    def _on_sidebar_tab_changed(self, _event: object | None = None) -> None:
        try:
            current = self._sidebar_nb.select()
            tab_text = self._sidebar_nb.tab(current, "text")
        except tk.TclError:
            return
        if tab_text == "Feature Migration" and hasattr(self, "_migration_panel"):
            self._migration_panel.on_tab_selected()

    def on_show(self) -> None:
        self._load_prefs()
        self.refresh(lazy=True)

    def set_chart_dir(self, chart_dir: str) -> None:
        self.cancel_lazy_load()
        self.chart_dir = chart_dir
        self._registry_runner.chart_dir = chart_dir
        auto = getattr(self, "_auto_feature_panel", None)
        if auto is not None:
            auto.set_chart_dir(chart_dir)
        self._load_prefs()

    def registry_export_running(self) -> bool:
        return self._registry_runner.running

    def auto_feature_build_running(self) -> bool:
        auto = getattr(self, "_auto_feature_panel", None)
        return bool(auto is not None and getattr(auto, "_building", False))

    def poll_auto_feature_progress(self) -> None:
        auto = getattr(self, "_auto_feature_panel", None)
        if auto is not None:
            auto.poll_progress()

    def poll_registry_export(self) -> bool:
        """Drain registry export progress + completion queue. Returns True when export finished."""
        self.poll_auto_feature_progress()
        self._drain_registry_progress()
        done = False
        latest: dict[str, Any] | None = None
        try:
            while True:
                latest = self._registry_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return False
        done = True
        self._set_registry_export_active(False)
        status = latest.get("status")
        mgr = get_build_progress_manager()
        if status == "completed":
            name = str(latest.get("dataset_name") or "")
            rows = int(latest.get("row_count") or 0)
            self._registry_progress_var.set(f"Created {name} · {rows:,} rows")
            self._append_builder_log(f"DONE: Created {name} with {rows:,} rows")
            mgr.publish({
                "status": "completed",
                "_done": True,
                "job_kind": "registry_export",
                "message": f"Created {name} with {rows:,} rows in Dataset Registry.",
            })
            messagebox.showinfo(
                "Dataset created",
                f"Created {name} with {rows:,} rows in Dataset Registry.",
            )
            if callable(self._on_registry_created) and name:
                self._on_registry_created(name)
        else:
            err = str(latest.get("error") or "Registry export failed")
            self._registry_progress_var.set(err)
            self._append_builder_log(f"FAILED: {err}")
            mgr.publish({
                "status": "failed",
                "_done": True,
                "job_kind": "registry_export",
                "message": err,
            })
            messagebox.showerror("Create Dataset", err)
        return done

    def _set_builder_text(self, text: str) -> None:
        if not hasattr(self, "_builder_text"):
            return
        self._builder_text.configure(state="normal")
        self._builder_text.delete("1.0", "end")
        self._builder_text.insert("1.0", text or "—")
        self._builder_text.configure(state="disabled")

    def _append_builder_log(self, msg: str) -> None:
        """Append a Create Dataset / export status line to Builder Progress."""
        import time

        line = str(msg or "").rstrip()
        if not line:
            return
        # Parquet write emits often; keep Builder Progress readable.
        low = line.lower()
        if low.startswith("writing registry parquet"):
            now = time.time()
            last = float(getattr(self, "_last_parquet_log_at", 0.0) or 0.0)
            if now - last < 3.0 and self._registry_export_log:
                # Replace the previous write-progress line in-place.
                self._registry_export_log[-1] = line
                if hasattr(self, "_builder_text"):
                    self._builder_text.configure(state="normal")
                    self._builder_text.delete("end-2l", "end-1l")
                    self._builder_text.insert("end", line + "\n")
                    self._builder_text.see("end")
                    self._builder_text.configure(state="disabled")
                return
            self._last_parquet_log_at = now
        self._registry_export_log.append(line)
        # Cap log growth for multi-GB exports with frequent parquet heartbeats.
        if len(self._registry_export_log) > 400:
            self._registry_export_log = self._registry_export_log[-300:]
        if not hasattr(self, "_builder_text"):
            return
        self._builder_text.configure(state="normal")
        self._builder_text.insert("end", line + "\n")
        self._builder_text.see("end")
        self._builder_text.configure(state="disabled")

    def _clear_builder_export_log(self, header: str) -> None:
        self._registry_export_log = []
        self._set_builder_text("")
        self._append_builder_log(header)

    def _show_builder_progress_tab(self) -> None:
        if hasattr(self, "_sidebar_nb"):
            try:
                self._sidebar_nb.select(self._builder_tab)
            except Exception:
                pass

    @staticmethod
    def _registry_stage_name(msg: str) -> str:
        m = (msg or "").lower()
        if "no-null" in m or "non-null" in m:
            return "No-Null Filter"
        if "writing" in m or "parquet" in m or "write parquet" in m:
            return "Write Parquet"
        if "metadata" in m or "opening" in m or "prepar" in m or "creat" in m:
            return "Prepare Export"
        return "Export"

    def _publish_registry_progress(self, msg: str, cur: int, tot: int) -> None:
        import time

        total = max(0, int(tot or 0))
        current = max(0, int(cur or 0))
        if total > 0:
            percent = round(100.0 * current / total, 1)
        elif current > 0:
            # Indeterminate write: keep bar moving lightly without inventing a total.
            percent = min(90.0, 15.0 + (current % 10_000) / 10_000.0 * 5.0)
        else:
            stage = self._registry_stage_name(msg)
            percent = {
                "Prepare Export": 3.0,
                "No-Null Filter": 8.0,
                "Write Parquet": 15.0,
            }.get(stage, 5.0)
        if self._registry_export_t0 is None:
            self._registry_export_t0 = time.time()
        elapsed = max(0.0, time.time() - self._registry_export_t0)
        short_msg = str(msg or "").strip()
        if "\n" in short_msg:
            # Keep status-bar task label short; full text goes to Builder Progress.
            first = short_msg.splitlines()[0].strip()
            short_msg = first or short_msg[:80]
        stage = self._registry_stage_name(msg)
        # Prefer the live status line in the global bar when it is concise.
        task = short_msg if short_msg and len(short_msg) <= 72 else stage
        get_build_progress_manager().publish({
            "status": "running",
            "job_kind": "registry_export",
            "job_title": "Exporting Dataset",
            "stage_name": task,
            "message": short_msg,
            "rows": current,
            "total": total,
            "percent": percent,
            "elapsed_sec": elapsed,
            "pipeline": {
                "total_elapsed_sec": elapsed,
                "total_elapsed_label": (
                    f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
                ),
            },
        })

    def _drain_registry_progress(self) -> None:
        try:
            while True:
                msg = self._registry_progress_queue.get_nowait()
                if msg:
                    text = str(msg)
                    self._registry_progress_var.set(
                        text.splitlines()[0] if "\n" in text else text
                    )
                    self._append_builder_log(text)
        except queue.Empty:
            pass

    def _interval_sec(self) -> int:
        return int(self._interval_var.get() or 10)

    def _master_day_keys(self) -> list[str]:
        d = self._detail or {}
        return [
            str(day.get("trading_day") or "")
            for day in (d.get("day_details") or [])
            if str(day.get("trading_day") or "").strip()
        ]

    def _day_meta_by_key(self) -> dict[str, dict[str, Any]]:
        d = self._detail or {}
        out: dict[str, dict[str, Any]] = {}
        for day in d.get("day_details") or []:
            td = str(day.get("trading_day") or "").strip()
            if td:
                out[td] = day
        return out

    def _trading_day_filter_scope(self) -> tuple[bool, list[str], dict[str, Any]]:
        """Checkbox selection after Trading Day Filter → export scope + registry meta."""
        from chain_replay_ml.dataset_builder.trading_day_filter import (
            MODE_ALL,
            apply_trading_day_filter,
            normalize_mode,
        )

        mode = normalize_mode(self._trading_day_filter_var.get())
        checked = sorted(str(d) for d in self._selected_days if str(d).strip())
        exported, meta = apply_trading_day_filter(checked, self._day_meta_by_key(), mode)
        master = self._master_day_keys()
        if not exported:
            return False, [], meta
        all_days, _ = effective_master_day_scope(set(exported), master)
        # Entire DB only when every master day exports and no expiry filter excluded days.
        if all_days and mode == MODE_ALL:
            return True, [], meta
        return False, exported, meta

    def _sync_day_selection(self) -> None:
        master_days = self._master_day_keys()
        self._selected_days &= set(master_days)
        if master_days and not self._selected_days and not self._day_selection_explicit:
            self._selected_days = set(master_days)
        if self._selected_day and self._selected_day not in master_days:
            self._selected_day = None
        self._update_days_selected_hint()

    def _preview_scope_kwargs(self) -> dict[str, Any]:
        if self._preview_drill_day and self._preview_token:
            return {"preview_day": self._preview_drill_day}
        all_days, selected, _meta = self._trading_day_filter_scope()
        if not all_days and not selected:
            return {}
        if all_days:
            return {"all_days": True}
        if len(selected) == 1:
            return {"preview_day": selected[0]}
        return {"selected_days": selected}

    def _export_filter_kwargs(self) -> dict[str, Any]:
        """ATM/LTP filters from UI — always used for registry export."""
        atm_raw = str(self._atm_var.get() or "").strip()
        try:
            atm = int(atm_raw) if atm_raw else None
        except ValueError:
            atm = None
        prem_en = bool(self._prem_en_var.get())
        pmin = _optional_float(self._prem_min_var.get()) if prem_en else None
        pmax = _optional_float(self._prem_max_var.get()) if prem_en else None
        # Incomplete LTP range (user mid-edit) → treat as unset, do not crash Tk traces
        if pmin is None or pmax is None:
            pmin = pmax = None
        return {
            "atm_band_filter": atm,
            "premium_min": pmin,
            "premium_max": pmax,
            "delta_min": None,
            "delta_max": None,
        }

    def _preview_filter_kwargs(self) -> dict[str, Any]:
        """ATM/LTP filters — skipped during token drill-down (show all rows for token+day)."""
        if self._preview_token:
            return {}
        return self._export_filter_kwargs()

    def _build_export_preview_dict(self) -> dict[str, Any]:
        scope = self._preview_scope_kwargs()
        return build_export_preview_dict(
            scope=scope or {},
            filters=self._export_filter_kwargs(),
        )

    def _on_trading_day_filter_changed(self) -> None:
        self._update_days_selected_hint()
        self._clear_preview()
        self._update_registry_export_panel()

    def _empty_scope_warning(self, *, title: str) -> None:
        """Warn when preview/export has no days after checkbox + Trading Day Filter."""
        from chain_replay_ml.dataset_builder.trading_day_filter import trading_day_filter_label

        _all, _selected, filter_meta = self._trading_day_filter_scope()
        mode = str(filter_meta.get("mode") or "all")
        checked_n = int(filter_meta.get("selected_days") or len(self._selected_days) or 0)
        if checked_n > 0:
            messagebox.showwarning(
                title,
                f"No trading days remain after filter "
                f"“{trading_day_filter_label(mode)}”.\n\n"
                f"You have {checked_n} day(s) checked, but they were all removed "
                f"by this filter (e.g. expiry days when Exclude expiry days is on).\n\n"
                f"Switch Trading Day Filter to “All selected days” or "
                f"“Expiry days only”, or check non-expiry days.",
            )
        else:
            messagebox.showwarning(title, "Select at least one trading day (checkbox).")

    def _update_days_selected_hint(self) -> None:
        master_days = self._master_day_keys()
        n = len(self._selected_days)
        if not master_days:
            self._days_selected_hint_var.set("No trading days in master DB")
            return
        all_days_raw, _ = effective_master_day_scope(self._selected_days, master_days)
        all_days, exported, filter_meta = self._trading_day_filter_scope()
        mode = str(filter_meta.get("mode") or "all")
        from chain_replay_ml.dataset_builder.trading_day_filter import (
            MODE_ALL,
            trading_day_filter_label,
        )

        if all_days_raw and mode == MODE_ALL:
            self._days_selected_hint_var.set(
                f"All {len(master_days)} trading days selected (entire master DB)"
            )
            return
        if not n:
            self._days_selected_hint_var.set("0 trading days selected")
            return
        if mode == MODE_ALL:
            self._days_selected_hint_var.set(f"{n} trading day(s) selected")
            return
        remain = len(exported)
        label = trading_day_filter_label(mode)
        if remain:
            self._days_selected_hint_var.set(
                f"{n} checked · {remain} after “{label}”"
            )
        else:
            self._days_selected_hint_var.set(
                f"{n} checked · 0 after “{label}” (nothing to preview/export)"
            )

    def _load_detail(self, *, include_preview: bool = False) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.master_status import read_master_dataset_detail

        scope = self._preview_scope_kwargs() if include_preview else {}
        filters = self._preview_filter_kwargs() if include_preview else {}
        preview_limit = 5000 if self._preview_token else 12

        return read_master_dataset_detail(
            self._data_dir(),
            market=self._market_var.get(),
            interval_sec=self._interval_sec(),
            preview_limit=preview_limit,
            preview_token=self._preview_token,
            include_preview=include_preview and bool(scope),
            **scope,
            **filters,
        )

    def refresh(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=lambda: self._load_detail(include_preview=False),
                apply=self._apply_refresh,
                message="Loading trading days…",
                status_var=self._status_var,
            )
            return
        try:
            detail = self._load_detail(include_preview=False)
        except Exception as exc:
            self._status_var.set(f"Error: {exc}")
            return
        self._apply_refresh(detail)

    def _apply_refresh(self, detail: dict[str, Any]) -> None:
        self._detail = detail
        d = detail
        self._status_var.set(
            f"Ready — {self._market_var.get()} · {self._interval_sec()}s"
            if d.get("exists")
            else f"No master DB for {self._interval_sec()}s — add days from Create Dataset"
        )
        self._render_overview(d)
        self._sync_day_selection()
        self._render_days(d)
        self._render_builder(d)
        self._render_meta(d)
        self._render_feature_health(d)
        self._render_tables(d)
        self._clear_preview()
        self._update_registry_export_panel()
        self._refresh_lag_feature_checkboxes()
        if hasattr(self, "_migration_panel"):
            # Cheap reset only — never auto-analyze on interval/market switch.
            self._migration_panel.refresh()
            # If the user is viewing Feature Migration, refresh analyze for the new DB.
            # (Default tab is Builder Progress so 3s↔10s switches stay light.)
            try:
                current = self._sidebar_nb.select()
                if self._sidebar_nb.tab(current, "text") == "Feature Migration":
                    self._migration_panel.on_tab_selected()
            except (tk.TclError, AttributeError):
                pass

    def _proposed_registry_name(self, feature_count: int) -> str:
        from chain_replay_ml.dataset_builder.master_registry_export import master_registry_dataset_name

        fc = int(feature_count) if feature_count > 0 else 0
        return master_registry_dataset_name(
            feature_count=fc,
            interval_sec=self._interval_sec(),
        )

    def _registry_retired_features(self) -> set[str]:
        from . import feature_registry_service as fr_svc

        return fr_svc.disabled_registry_features(self.chart_dir)

    def _master_feature_columns(self) -> list[str]:
        d = self._preview_detail or self._detail or {}
        schema = d.get("build_schema") if isinstance(d.get("build_schema"), dict) else {}
        return [str(c) for c in (schema.get("feature_columns") or d.get("feature_columns") or [])]

    def _laggable_feature_names(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import filter_laggable_features

        return filter_laggable_features(
            self._master_feature_columns(),
            registry_only=True,
            exclude_names=self._registry_retired_features(),
        )

    def _refresh_lag_feature_checkboxes(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import default_selected_lag_features

        if not hasattr(self, "_lag_feat_host"):
            return
        available = self._laggable_feature_names()
        previous = {
            name: bool(var.get())
            for name, var in self._lag_feature_vars.items()
        }
        self._lag_available_features = list(available)
        self._lag_feature_vars = {}
        if not available:
            self._rebuild_lag_feature_list()
            self._update_lag_preview()
            self._refresh_dependent_transform_feature_lists()
            self._refresh_interaction_feature_lists()
            return
        defaults = set(default_selected_lag_features(available))
        for name in available:
            checked = previous[name] if name in previous else (name in defaults)
            self._lag_feature_vars[name] = tk.BooleanVar(value=checked)
        self._rebuild_lag_feature_list()
        self._update_lag_preview()
        self._refresh_dependent_transform_feature_lists()
        self._refresh_interaction_feature_lists()

    def _refresh_dependent_transform_feature_lists(self) -> None:
        """Refresh Diff/Return/Rolling/Exp/OHLC independent input feature checklists."""
        for kwargs in (
            {
                "feat_host_attr": "_difference_feat_host",
                "feature_vars_attr": "_difference_feature_vars",
                "pending_attr": "_pending_difference_features",
                "on_change": self._on_difference_settings_changed,
            },
            {
                "feat_host_attr": "_return_feat_host",
                "feature_vars_attr": "_return_feature_vars",
                "pending_attr": "_pending_return_features",
                "on_change": self._on_return_settings_changed,
            },
            {
                "feat_host_attr": "_rolling_feat_host",
                "feature_vars_attr": "_rolling_feature_vars",
                "pending_attr": "_pending_rolling_features",
                "on_change": self._on_rolling_settings_changed,
            },
            {
                "feat_host_attr": "_exponential_feat_host",
                "feature_vars_attr": "_exponential_feature_vars",
                "pending_attr": "_pending_exponential_features",
                "on_change": self._on_exponential_rolling_settings_changed,
            },
            {
                "feat_host_attr": "_normalization_feat_host",
                "feature_vars_attr": "_normalization_feature_vars",
                "pending_attr": "_pending_normalization_features",
                "on_change": self._on_normalization_settings_changed,
            },
            {
                "feat_host_attr": "_regime_feat_host",
                "feature_vars_attr": "_regime_feature_vars",
                "pending_attr": "_pending_regime_features",
                "on_change": self._on_regime_settings_changed,
            },
            {
                "feat_host_attr": "_math_feat_host",
                "feature_vars_attr": "_math_feature_vars",
                "pending_attr": "_pending_math_features",
                "on_change": self._on_math_settings_changed,
            },
        ):
            try:
                self._refresh_flat_feature_checkboxes(**kwargs)
            except Exception:
                pass
        try:
            self._refresh_ohlc_feature_checkboxes()
        except Exception:
            pass
        # Rebuild Diff/Return horizons when interval may have changed.
        try:
            self._rebuild_simple_horizon_checks(
                seconds_host_attr="_difference_seconds_host",
                seconds_vars_attr="_difference_seconds_vars",
                on_change=self._on_difference_settings_changed,
            )
            self._rebuild_simple_horizon_checks(
                seconds_host_attr="_return_seconds_host",
                seconds_vars_attr="_return_seconds_vars",
                on_change=self._on_return_settings_changed,
            )
        except Exception:
            pass

    def _rebuild_lag_feature_list(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            filter_features_by_search,
            group_features_by_category,
        )

        if not hasattr(self, "_lag_feat_host"):
            return
        for child in self._lag_feat_host.winfo_children():
            child.destroy()
        available = list(self._lag_available_features)
        if not available:
            ttk.Label(
                self._lag_feat_host,
                text="No feature columns in master schema yet.",
                foreground="#888",
            ).pack(anchor="w")
            return
        query = self._lag_search_var.get() if hasattr(self, "_lag_search_var") else ""
        visible = filter_features_by_search(available, query)
        if not visible:
            ttk.Label(
                self._lag_feat_host,
                text=f"No features match “{query.strip()}”.",
                foreground="#888",
            ).pack(anchor="w")
            return
        searching = bool(str(query or "").strip())
        grouped = group_features_by_category(visible)
        for cat, names in grouped.items():
            # Collapse by default so first open does not create hundreds of widgets.
            expanded = True if searching else self._lag_category_expanded.get(cat, False)
            self._lag_category_expanded.setdefault(cat, False)
            header = ttk.Frame(self._lag_feat_host)
            header.pack(fill="x", anchor="w", pady=(4, 0))
            marker = "▼" if expanded else "▶"
            selected_n = sum(
                1 for n in names
                if n in self._lag_feature_vars and self._lag_feature_vars[n].get()
            )
            ttk.Button(
                header,
                text=f"{marker} {cat}  ({selected_n}/{len(names)})",
                width=36,
                command=lambda c=cat: self._toggle_lag_category(c),
            ).pack(side="left", anchor="w")
            if not expanded:
                continue
            for name in names:
                var = self._lag_feature_vars.get(name)
                if var is None:
                    var = tk.BooleanVar(value=False)
                    self._lag_feature_vars[name] = var
                ttk.Checkbutton(
                    self._lag_feat_host,
                    text=f"    {name}",
                    variable=var,
                    command=self._on_lag_settings_changed,
                ).pack(anchor="w")

    def _toggle_lag_category(self, category: str) -> None:
        cur = self._lag_category_expanded.get(category, True)
        self._lag_category_expanded[category] = not cur
        self._rebuild_lag_feature_list()

    def _apply_lag_preset(self, preset: str) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import features_for_preset

        targets = set(features_for_preset(preset, self._lag_available_features))
        for name, var in self._lag_feature_vars.items():
            var.set(name in targets)
        self._rebuild_lag_feature_list()
        self._on_lag_settings_changed()

    def _clear_lag_features(self) -> None:
        for var in self._lag_feature_vars.values():
            var.set(False)
        self._rebuild_lag_feature_list()
        self._on_lag_settings_changed()

    def _invert_lag_features(self) -> None:
        for var in self._lag_feature_vars.values():
            var.set(not bool(var.get()))
        self._rebuild_lag_feature_list()
        self._on_lag_settings_changed()

    def _selected_lag_features(self) -> list[str]:
        return [n for n, v in self._lag_feature_vars.items() if v.get()]

    def _selected_lag_seconds(self) -> list[int]:
        return sorted(sec for sec, var in self._lag_seconds_vars.items() if var.get())

    def _selected_difference_features(self) -> list[str]:
        return self._selected_from_feature_vars("_difference_feature_vars")

    def _selected_difference_seconds(self) -> list[int]:
        return self._selected_from_seconds_vars("_difference_seconds_vars")

    def _selected_return_features(self) -> list[str]:
        return self._selected_from_feature_vars("_return_feature_vars")

    def _selected_return_seconds(self) -> list[int]:
        return self._selected_from_seconds_vars("_return_seconds_vars")

    def _selected_rolling_features(self) -> list[str]:
        return self._selected_from_feature_vars("_rolling_feature_vars")

    def _selected_exponential_features(self) -> list[str]:
        return self._selected_from_feature_vars("_exponential_feature_vars")

    def _selected_rolling_windows(self) -> list[int]:
        return sorted(win for win, var in self._rolling_window_vars.items() if var.get())

    def _selected_rolling_operations(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.rolling_ui import ROLLING_OPS

        return [op for op in ROLLING_OPS if self._rolling_op_vars.get(op) and self._rolling_op_vars[op].get()]

    def _selected_exponential_periods(self) -> list[int]:
        return sorted(
            per for per, var in self._exponential_period_vars.items() if var.get()
        )

    def _selected_exponential_operations(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.exponential_rolling import (
            EXPONENTIAL_ROLLING_OPS,
        )

        return [
            op
            for op in EXPONENTIAL_ROLLING_OPS
            if self._exponential_op_vars.get(op) and self._exponential_op_vars[op].get()
        ]

    def _selected_ohlc_features(self) -> list[str]:
        return [
            name
            for name, var in self._ohlc_feature_vars.items()
            if bool(var.get())
        ]

    def _refresh_ohlc_feature_checkboxes(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            default_selected_ohlc_features,
        )

        if not hasattr(self, "_ohlc_feat_host"):
            return
        available = self._laggable_feature_names()
        previous = {
            name: bool(var.get())
            for name, var in self._ohlc_feature_vars.items()
        }
        pending = getattr(self, "_pending_ohlc_features", None)
        self._ohlc_feature_vars = {}
        if not available:
            self._rebuild_ohlc_feature_list()
            return
        defaults = set(default_selected_ohlc_features(available))
        for name in available:
            if isinstance(pending, list):
                checked = name in set(pending)
            elif name in previous:
                checked = previous[name]
            else:
                checked = name in defaults
            self._ohlc_feature_vars[name] = tk.BooleanVar(value=checked)
        self._rebuild_ohlc_feature_list()

    def _rebuild_ohlc_feature_list(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            group_features_by_category,
        )

        if not hasattr(self, "_ohlc_feat_host"):
            return
        for child in self._ohlc_feat_host.winfo_children():
            child.destroy()
        available = list(self._ohlc_feature_vars.keys())
        if not available:
            ttk.Label(
                self._ohlc_feat_host,
                text="No feature columns in master schema yet.",
                foreground="#888",
            ).pack(anchor="w")
            return
        if not hasattr(self, "_ohlc_category_expanded"):
            self._ohlc_category_expanded: dict[str, bool] = {}
        grouped = group_features_by_category(available)
        for cat, names in grouped.items():
            self._ohlc_category_expanded.setdefault(cat, False)
            is_expanded = self._ohlc_category_expanded.get(cat, False)
            marker = "▼" if is_expanded else "▶"
            selected_n = sum(
                1 for n in names
                if n in self._ohlc_feature_vars and self._ohlc_feature_vars[n].get()
            )
            header = ttk.Frame(self._ohlc_feat_host)
            header.pack(fill="x", anchor="w", pady=(4, 0))
            ttk.Button(
                header,
                text=f"{marker} {cat}  ({selected_n}/{len(names)})",
                width=36,
                command=lambda c=cat: self._toggle_ohlc_category(c),
            ).pack(side="left", anchor="w")
            if not is_expanded:
                continue
            for name in names:
                var = self._ohlc_feature_vars.get(name)
                if var is None:
                    var = tk.BooleanVar(value=False)
                    self._ohlc_feature_vars[name] = var
                ttk.Checkbutton(
                    self._ohlc_feat_host,
                    text=f"    {name}",
                    variable=var,
                    command=self._on_ohlc_aggregation_settings_changed,
                ).pack(anchor="w")

    def _toggle_ohlc_category(self, category: str) -> None:
        cur = self._ohlc_category_expanded.get(category, False)
        self._ohlc_category_expanded[category] = not cur
        self._rebuild_ohlc_feature_list()

    def _selected_ohlc_timeframes(self) -> list[str]:
        return [
            tf
            for tf, var in self._ohlc_tf_vars.items()
            if var.get()
        ]

    def _selected_ohlc_outputs(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            OHLC_FIELDS,
        )

        return [
            fld
            for fld in OHLC_FIELDS
            if self._ohlc_output_vars.get(fld) and self._ohlc_output_vars[fld].get()
        ]

    def _lag_estimated_rows(self) -> int | None:
        cached = (self._preview_detail or {}).get("sample_preview")
        if not cached:
            return None
        export_preview = self._build_export_preview_dict()
        no_null = bool(self._no_null_data_var.get())
        if not preview_matches_export_settings(cached, export_preview, no_null_data=no_null):
            return None
        return preview_display_match_count(cached, no_null_data=no_null)

    def _lag_warmup_seconds(self) -> float:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import resolve_warmup_seconds

        return resolve_warmup_seconds(
            self._preview_detail or self._detail,
            sample_interval_sec=self._interval_sec(),
        )

    def _time_shift_enabled(self) -> bool:
        return bool(self._lag_enabled_var.get())

    def _time_shift_transform_count(self) -> int:
        return 1 if self._lag_enabled_var.get() else 0

    def _lag_preview_counts(self) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import lag_preview_counts

        return lag_preview_counts(
            enabled=bool(self._lag_enabled_var.get()),
            selected_features=self._selected_lag_features(),
            lag_seconds=self._selected_lag_seconds(),
            current_columns=self._registry_feature_count(),
            estimated_rows=self._lag_estimated_rows(),
            transform_count=1,
        )

    def _simple_time_shift_preview_counts(
        self,
        *,
        enabled: bool,
        features: list[str],
        seconds: list[int],
    ) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import lag_preview_counts

        return lag_preview_counts(
            enabled=enabled,
            selected_features=features,
            lag_seconds=seconds,
            current_columns=self._registry_feature_count(),
            estimated_rows=self._lag_estimated_rows(),
            transform_count=1,
        )

    def _refresh_lag_seconds_labels(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            lag_seconds_is_valid_multiple,
            lag_seconds_label,
            resolve_warmup_seconds,
        )

        if not getattr(self, "_lag_seconds_buttons", None):
            return
        interval = self._interval_sec()
        warm = resolve_warmup_seconds(
            self._preview_detail or self._detail,
            sample_interval_sec=interval,
        )
        for sec, btn in self._lag_seconds_buttons.items():
            label = lag_seconds_label(sec, interval)
            valid = lag_seconds_is_valid_multiple(sec, interval)
            exceeds = warm > 0 and sec > warm
            if exceeds:
                label = f"{label}  ⚠"
            try:
                btn.configure(text=label)
            except tk.TclError:
                pass
            var = self._lag_seconds_vars.get(sec)
            if not valid:
                if var is not None:
                    var.set(False)
                try:
                    btn.configure(state="disabled")
                except tk.TclError:
                    pass
            elif exceeds:
                # Allow selection but keep warning marker; still enabled when a transform is on.
                try:
                    if self._time_shift_enabled():
                        btn.configure(state="normal")
                except tk.TclError:
                    pass
            else:
                try:
                    if self._time_shift_enabled():
                        btn.configure(state="normal")
                except tk.TclError:
                    pass

    def _update_lag_preview(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            format_pipeline_ledger_text,
            pipeline_feature_ledger,
        )
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            format_lag_preview_text,
            lag_warmup_warning,
        )
        from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
            exponential_rolling_column_count,
            format_exponential_rolling_preview_text,
            preview_exponential_rolling_columns,
        )
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            format_ohlc_aggregation_preview_text,
            ohlc_aggregation_column_count,
            preview_ohlc_aggregation_columns,
        )
        from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
            format_rolling_preview_text,
            preview_rolling_columns,
            rolling_column_count,
        )

        if not hasattr(self, "_lag_preview_var"):
            return
        c = self._lag_preview_counts()
        warn = lag_warmup_warning(
            enabled=bool(self._lag_enabled_var.get()),
            lag_seconds=self._selected_lag_seconds(),
            warmup_seconds=self._lag_warmup_seconds(),
        )
        if hasattr(self, "_lag_warn_var"):
            self._lag_warn_var.set(warn or "")
        self._lag_preview_var.set(
            format_lag_preview_text(
                c,
                enabled=bool(self._lag_enabled_var.get()),
                warmup_warning=None,  # shown separately above Lag Seconds
            )
        )
        # Difference / Return previews (independent feature × horizon sets).
        if hasattr(self, "_difference_preview_var"):
            diff_on = bool(self._difference_enabled_var.get())
            diff_c = self._simple_time_shift_preview_counts(
                enabled=diff_on,
                features=self._selected_difference_features(),
                seconds=self._selected_difference_seconds(),
            )
            self._difference_preview_var.set(
                format_lag_preview_text(diff_c, enabled=diff_on, warmup_warning=None)
            )
            dw = lag_warmup_warning(
                enabled=diff_on,
                lag_seconds=self._selected_difference_seconds(),
                warmup_seconds=self._lag_warmup_seconds(),
            )
            self._difference_warn_var.set(dw or "")
        if hasattr(self, "_return_preview_var"):
            ret_on = bool(self._return_enabled_var.get())
            ret_c = self._simple_time_shift_preview_counts(
                enabled=ret_on,
                features=self._selected_return_features(),
                seconds=self._selected_return_seconds(),
            )
            self._return_preview_var.set(
                format_lag_preview_text(ret_c, enabled=ret_on, warmup_warning=None)
            )
            rw = lag_warmup_warning(
                enabled=ret_on,
                lag_seconds=self._selected_return_seconds(),
                warmup_seconds=self._lag_warmup_seconds(),
            )
            self._return_warn_var.set(rw or "")

        roll_feats = self._selected_rolling_features()
        wins = self._selected_rolling_windows()
        ops = self._selected_rolling_operations()
        rolling_on = bool(self._rolling_enabled_var.get())
        roll_n = rolling_column_count(
            enabled=rolling_on,
            features=roll_feats,
            windows=wins,
            operations=ops,
        )
        if hasattr(self, "_rolling_preview_var"):
            samples = preview_rolling_columns(
                features=roll_feats,
                windows=wins,
                operations=ops,
                limit=4,
            ) if rolling_on else []
            self._rolling_preview_var.set(
                format_rolling_preview_text(
                    enabled=rolling_on,
                    feature_count=len(roll_feats) if rolling_on else 0,
                    window_count=len(wins) if rolling_on else 0,
                    operation_count=len(ops) if rolling_on else 0,
                    columns_to_add=roll_n,
                    sample_names=samples,
                )
            )
        exp_feats = self._selected_exponential_features()
        pers = self._selected_exponential_periods()
        exp_ops = self._selected_exponential_operations()
        exp_on = bool(self._exponential_rolling_enabled_var.get())
        exp_n = exponential_rolling_column_count(
            enabled=exp_on,
            features=exp_feats,
            periods=pers,
            operations=exp_ops,
        )
        if hasattr(self, "_exponential_rolling_preview_var"):
            exp_samples = preview_exponential_rolling_columns(
                features=exp_feats,
                periods=pers,
                operations=exp_ops,
                limit=4,
            ) if exp_on else []
            self._exponential_rolling_preview_var.set(
                format_exponential_rolling_preview_text(
                    enabled=exp_on,
                    feature_count=len(exp_feats) if exp_on else 0,
                    period_count=len(pers) if exp_on else 0,
                    operation_count=len(exp_ops) if exp_on else 0,
                    columns_to_add=exp_n,
                    sample_names=exp_samples,
                )
            )
        ohlc_feats = self._selected_ohlc_features()
        ohlc_tfs = self._selected_ohlc_timeframes()
        ohlc_outs = self._selected_ohlc_outputs()
        ohlc_on = bool(self._ohlc_aggregation_enabled_var.get())
        interval = self._interval_sec()
        ohlc_n = ohlc_aggregation_column_count(
            enabled=ohlc_on,
            features=ohlc_feats,
            timeframes=ohlc_tfs,
            outputs=ohlc_outs,
            sample_interval_sec=interval,
        )
        if hasattr(self, "_ohlc_aggregation_preview_var"):
            ohlc_samples = preview_ohlc_aggregation_columns(
                features=ohlc_feats,
                timeframes=ohlc_tfs,
                outputs=ohlc_outs,
                sample_interval_sec=interval,
                limit=4,
            ) if ohlc_on else []
            self._ohlc_aggregation_preview_var.set(
                format_ohlc_aggregation_preview_text(
                    enabled=ohlc_on,
                    feature_count=len(ohlc_feats) if ohlc_on else 0,
                    timeframe_count=len(ohlc_tfs) if ohlc_on else 0,
                    output_count=len(ohlc_outs) if ohlc_on else 0,
                    columns_to_add=ohlc_n,
                    sample_names=ohlc_samples,
                    sample_interval_sec=interval,
                )
            )
        ix_n = (
            len(self._interaction_pipeline_pairs)
            if self._interaction_enabled_var.get()
            else 0
        )
        norm_feats = self._selected_normalization_features()
        norm_meths = self._selected_normalization_methods()
        norm_wins = self._selected_normalization_windows()
        norm_on = bool(self._normalization_enabled_var.get())
        from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
            format_normalization_preview_text,
            normalization_column_count,
            preview_normalization_columns,
        )

        norm_n = normalization_column_count(
            enabled=norm_on,
            features=norm_feats,
            methods=norm_meths,
            windows=norm_wins,
        )
        if hasattr(self, "_normalization_preview_var"):
            norm_samples = preview_normalization_columns(
                features=norm_feats,
                methods=norm_meths,
                windows=norm_wins,
                limit=4,
            ) if norm_on else []
            self._normalization_preview_var.set(
                format_normalization_preview_text(
                    enabled=norm_on,
                    feature_count=len(norm_feats) if norm_on else 0,
                    method_count=len(norm_meths) if norm_on else 0,
                    window_count=len(norm_wins) if norm_on else 0,
                    columns_to_add=norm_n,
                    sample_names=norm_samples,
                )
            )
        reg_feats = self._selected_regime_features()
        reg_meths = self._selected_regime_methods()
        reg_wins = self._selected_regime_windows()
        reg_on = bool(self._regime_enabled_var.get())
        rp = self._regime_params()
        from chain_replay_ml.dataset_builder.transformations.regime_ui import (
            format_regime_preview_text,
            preview_regime_columns,
            regime_column_count,
        )

        reg_n = regime_column_count(
            enabled=reg_on,
            features=reg_feats,
            methods=reg_meths,
            windows=reg_wins,
        )
        if hasattr(self, "_regime_preview_var"):
            reg_samples = preview_regime_columns(
                features=reg_feats,
                methods=reg_meths,
                windows=reg_wins,
                threshold=float(rp["threshold"]),
                low=float(rp["low"]),
                high=float(rp["high"]),
                n_bins=int(rp["n_bins"]),
                limit=4,
            ) if reg_on else []
            self._regime_preview_var.set(
                format_regime_preview_text(
                    enabled=reg_on,
                    feature_count=len(reg_feats) if reg_on else 0,
                    method_count=len(reg_meths) if reg_on else 0,
                    window_count=len(reg_wins) if reg_on else 0,
                    columns_to_add=reg_n,
                    sample_names=reg_samples,
                )
            )
        math_feats = self._selected_math_features()
        math_ops = self._selected_math_operations()
        math_on = bool(self._math_enabled_var.get())
        clip_min, clip_max = self._math_clip_bounds()
        from chain_replay_ml.dataset_builder.transformations.math_ui import (
            format_math_preview_text,
            math_column_count,
            preview_math_columns,
        )

        math_n = math_column_count(
            enabled=math_on,
            features=math_feats,
            operations=math_ops,
        )
        if hasattr(self, "_math_preview_var"):
            math_samples = preview_math_columns(
                features=math_feats,
                operations=math_ops,
                clip_min=clip_min,
                clip_max=clip_max,
                limit=4,
            ) if math_on else []
            self._math_preview_var.set(
                format_math_preview_text(
                    enabled=math_on,
                    feature_count=len(math_feats) if math_on else 0,
                    operation_count=len(math_ops) if math_on else 0,
                    columns_to_add=math_n,
                    sample_names=math_samples,
                )
            )
        lag_on = bool(self._lag_enabled_var.get())
        diff_on = bool(self._difference_enabled_var.get())
        ret_on = bool(self._return_enabled_var.get())
        lag_n = (
            len(self._selected_lag_features()) * len(self._selected_lag_seconds())
            if lag_on else 0
        )
        diff_n = (
            len(self._selected_difference_features()) * len(self._selected_difference_seconds())
            if diff_on else 0
        )
        ret_n = (
            len(self._selected_return_features()) * len(self._selected_return_seconds())
            if ret_on else 0
        )
        ledger = pipeline_feature_ledger(
            master_count=self._registry_feature_count(),
            lag_enabled=lag_on,
            difference_enabled=diff_on,
            return_enabled=ret_on,
            rolling_count=roll_n,
            exponential_rolling_count=exp_n,
            ohlc_aggregation_count=ohlc_n,
            interaction_count=ix_n,
            math_count=math_n,
            normalization_count=norm_n,
            regime_count=reg_n,
            selected_features=self._selected_lag_features(),
            lag_seconds=self._selected_lag_seconds(),
            lag_count=lag_n,
            difference_count=diff_n,
            return_count=ret_n,
        )
        if hasattr(self, "_feature_preview_var"):
            self._feature_preview_var.set(format_pipeline_ledger_text(ledger))

    def _sync_lag_body_state(self) -> None:
        if not hasattr(self, "_lag_body"):
            return
        state = "normal" if self._lag_enabled_var.get() else "disabled"

        def _walk(widget: tk.Misc) -> None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _walk(child)

        _walk(self._lag_body)
        # Re-apply interval/warm-up disable rules after bulk enable.
        if self._lag_enabled_var.get():
            self._refresh_lag_seconds_labels()

    def _on_lag_settings_changed(self) -> None:
        self._sync_lag_body_state()
        self._update_lag_preview()
        self._refresh_interaction_feature_lists()
        self._save_prefs()

    def _on_difference_settings_changed(self) -> None:
        self._sync_enable_body("_difference_body", self._difference_enabled_var)
        self._update_lag_preview()
        self._refresh_interaction_feature_lists()
        self._save_prefs()

    def _on_return_settings_changed(self) -> None:
        self._sync_enable_body("_return_body", self._return_enabled_var)
        self._update_lag_preview()
        self._refresh_interaction_feature_lists()
        self._save_prefs()

    def _sync_rolling_body_state(self) -> None:
        if not hasattr(self, "_rolling_body"):
            return
        state = "normal" if self._rolling_enabled_var.get() else "disabled"

        def _walk(widget: tk.Misc) -> None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _walk(child)

        _walk(self._rolling_body)

    def _on_rolling_settings_changed(self) -> None:
        self._sync_rolling_body_state()
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()

    def _sync_exponential_rolling_body_state(self) -> None:
        if not hasattr(self, "_exponential_rolling_body"):
            return
        state = "normal" if self._exponential_rolling_enabled_var.get() else "disabled"

        def _walk(widget: tk.Misc) -> None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _walk(child)

        _walk(self._exponential_rolling_body)

    def _on_exponential_rolling_settings_changed(self) -> None:
        self._sync_exponential_rolling_body_state()
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()

    def _sync_ohlc_aggregation_body_state(self) -> None:
        if not hasattr(self, "_ohlc_aggregation_body"):
            return
        state = "normal" if self._ohlc_aggregation_enabled_var.get() else "disabled"

        def _walk(widget: tk.Misc) -> None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _walk(child)

        _walk(self._ohlc_aggregation_body)

    def _on_ohlc_aggregation_settings_changed(self) -> None:
        self._sync_ohlc_aggregation_body_state()
        self._update_lag_preview()
        self._refresh_interaction_feature_lists()
        self._save_prefs()

    def _make_scroll_host(
        self, parent: ttk.Frame, *, height: int = 100, width: int | None = None
    ) -> ttk.Frame:
        wrap = ttk.Frame(parent)
        if width is not None:
            wrap.pack(fill="y", expand=False)
        else:
            wrap.pack(fill="both", expand=True)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        canvas_kwargs: dict[str, Any] = {
            "height": height,
            "highlightthickness": 0,
            "borderwidth": 0,
        }
        if width is not None:
            canvas_kwargs["width"] = int(width)
        canvas = tk.Canvas(wrap, **canvas_kwargs)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        host = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=host, anchor="nw")

        def _on_host_configure(_event: tk.Event, c: tk.Canvas = canvas) -> None:
            c.configure(scrollregion=c.bbox("all"))

        def _on_canvas_configure(event: tk.Event, c: tk.Canvas = canvas, wid: int = win_id) -> None:
            # Keep inner frame width in sync so the scrollbar stays usable.
            c.itemconfigure(wid, width=max(1, int(event.width)))

        host.bind("<Configure>", _on_host_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        return host

    def _interaction_op_id(self) -> str:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            OP_DISPLAY_LABELS,
        )

        label = str(getattr(self, "_interaction_op_display", tk.StringVar(value="Multiply")).get())
        mapping = getattr(self, "_interaction_op_labels", None) or {
            v: k for k, v in OP_DISPLAY_LABELS.items()
        }
        return str(mapping.get(label) or "multiply")

    def _on_interaction_builder_changed(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction import (
            interaction_column_name,
        )

        a = str(self._interaction_feat_a_var.get() or "").strip()
        b = str(self._interaction_feat_b_var.get() or "").strip()
        if a and b:
            try:
                self._interaction_output_var.set(
                    interaction_column_name(a, b, self._interaction_op_id())
                )
            except Exception:
                self._interaction_output_var.set("")
        else:
            self._interaction_output_var.set("")

    def _on_interaction_settings_changed(self) -> None:
        """Main Interaction enable/disable toggle."""
        self._sync_interaction_body_state()
        self._update_lag_preview()
        self._save_prefs()

    def _on_interaction_bulk_selection_changed(self) -> None:
        """Bulk A/B checkbox toggles — do not re-walk widget states."""
        self._update_lag_preview()
        self._save_prefs()

    def _sync_interaction_body_state(self) -> None:
        if not hasattr(self, "_interaction_body"):
            return
        enabled = bool(self._interaction_enabled_var.get())

        def _apply(widget: tk.Misc) -> None:
            # Never disable Canvas/Frame/Label — disabling a Canvas freezes all
            # embedded bulk checkboxes in both Feature A and Feature B sets.
            cls = widget.winfo_class()
            try:
                if cls == "Canvas":
                    # Undo any prior disable from older sync logic.
                    widget.configure(state="normal")
                elif cls in ("TCombobox", "Combobox"):
                    widget.configure(state="readonly" if enabled else "disabled")
                elif cls in ("TButton", "Button", "TCheckbutton", "Checkbutton", "TEntry", "Entry"):
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _apply(child)

        _apply(self._interaction_body)

    def _refresh_interaction_feature_lists(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            available_interaction_features_from_config,
            columns_for_interaction_source,
            group_features_by_source,
            interaction_source_choices,
        )

        if not hasattr(self, "_interaction_feat_a_combo"):
            return
        master = self._laggable_feature_names()
        try:
            cfg = self._build_transformation_config()
        except Exception:
            cfg = {"transformations": []}
        interval = self._interval_sec()
        avail = available_interaction_features_from_config(
            cfg,
            master_features=master,
            sample_interval_sec=interval,
        )
        self._interaction_available = list(avail)

        choices = interaction_source_choices(
            cfg,
            master_features=master,
            sample_interval_sec=interval,
        )
        label_to_id = {label: sid for sid, label in choices}
        labels = [label for _sid, label in choices]
        self._interaction_source_a_ids = dict(label_to_id)
        self._interaction_source_b_ids = dict(label_to_id)
        by_source: dict[str, list[str]] = {}
        for sid, _label in choices:
            by_source[sid] = columns_for_interaction_source(
                cfg,
                sid,
                master_features=master,
                sample_interval_sec=interval,
            )
        self._interaction_columns_by_source = by_source

        for combo, var, id_map in (
            (
                getattr(self, "_interaction_source_a_combo", None),
                self._interaction_source_a_var,
                self._interaction_source_a_ids,
            ),
            (
                getattr(self, "_interaction_source_b_combo", None),
                self._interaction_source_b_var,
                self._interaction_source_b_ids,
            ),
        ):
            if combo is None:
                continue
            try:
                combo.configure(values=labels)
            except tk.TclError:
                pass
            cur = str(var.get() or "")
            if cur not in label_to_id and labels:
                var.set(labels[0])

        self._apply_interaction_source_feature_list("a")
        self._apply_interaction_source_feature_list("b")

        ix_outs = {
            str(p.get("output") or "")
            for p in self._interaction_pipeline_pairs
            if isinstance(p, dict)
        }
        # Bulk checkbox hosts must stay small — Master + Interaction only.
        bulk_names = list(dict.fromkeys([*master, *sorted(ix_outs)]))
        grouped = group_features_by_source(
            bulk_names,
            master_features=set(master),
            interaction_outputs=ix_outs,
        )
        self._rebuild_interaction_bulk_lists(grouped)
        self._on_interaction_builder_changed()

    def _interaction_selected_source_id(self, side: str) -> str:
        from chain_replay_ml.dataset_builder.transformations.describe import MASTER_STAGE_ID

        if side == "a":
            label = str(self._interaction_source_a_var.get() or "")
            return self._interaction_source_a_ids.get(label, MASTER_STAGE_ID)
        label = str(self._interaction_source_b_var.get() or "")
        return self._interaction_source_b_ids.get(label, MASTER_STAGE_ID)

    def _apply_interaction_source_feature_list(self, side: str) -> None:
        sid = self._interaction_selected_source_id(side)
        cols = list(self._interaction_columns_by_source.get(sid) or [])
        if side == "a":
            combo = getattr(self, "_interaction_feat_a_combo", None)
            var = self._interaction_feat_a_var
            self._interaction_feat_a_cols_full = list(cols)
        else:
            combo = getattr(self, "_interaction_feat_b_combo", None)
            var = self._interaction_feat_b_var
            self._interaction_feat_b_cols_full = list(cols)
        if combo is None:
            return
        self._filter_interaction_single_features(side)
        full = self._interaction_feat_a_cols_full if side == "a" else self._interaction_feat_b_cols_full
        cur = str(var.get() or "").strip()
        if cur and cur not in full:
            var.set(full[0] if full else "")
        elif not cur and full:
            var.set(full[0])

    def _filter_interaction_single_features(self, side: str) -> None:
        if side == "a":
            combo = getattr(self, "_interaction_feat_a_combo", None)
            full = list(self._interaction_feat_a_cols_full)
            needle = str(self._interaction_search_a_var.get() or "").strip().lower()
        else:
            combo = getattr(self, "_interaction_feat_b_combo", None)
            full = list(self._interaction_feat_b_cols_full)
            needle = str(self._interaction_search_b_var.get() or "").strip().lower()
        if combo is None:
            return
        if needle:
            visible = [c for c in full if needle in str(c).lower()]
        else:
            visible = list(full)
        try:
            combo.configure(values=visible)
        except tk.TclError:
            pass

    def _apply_interaction_bulk_filter(self, side: str) -> None:
        if side == "a":
            meta = getattr(self, "_interaction_bulk_a_meta", [])
            needle = str(self._interaction_bulk_search_a_var.get() or "").strip().lower()
        else:
            meta = getattr(self, "_interaction_bulk_b_meta", [])
            needle = str(self._interaction_bulk_search_b_var.get() or "").strip().lower()
        for block in meta:
            group_label = block.get("group_label")
            items = block.get("items") or []
            any_visible = False
            for item in items:
                name = str(item.get("name") or "")
                cb = item.get("checkbox")
                if cb is None:
                    continue
                show = not needle or needle in name.lower()
                if show:
                    cb.pack(anchor="w")
                    any_visible = True
                else:
                    cb.pack_forget()
            if group_label is not None:
                if any_visible:
                    group_label.pack(anchor="w", pady=(4, 0))
                else:
                    group_label.pack_forget()

    def _on_interaction_source_a_changed(self) -> None:
        self._apply_interaction_source_feature_list("a")
        self._on_interaction_builder_changed()

    def _on_interaction_source_b_changed(self) -> None:
        self._apply_interaction_source_feature_list("b")
        self._on_interaction_builder_changed()

    def _rebuild_interaction_bulk_lists(self, grouped: dict[str, list[str]]) -> None:
        for host_name, var_map_name, meta_name, side in (
            ("_interaction_bulk_a_host", "_interaction_bulk_a_vars", "_interaction_bulk_a_meta", "a"),
            ("_interaction_bulk_b_host", "_interaction_bulk_b_vars", "_interaction_bulk_b_meta", "b"),
        ):
            host = getattr(self, host_name, None)
            if host is None:
                continue
            for child in host.winfo_children():
                child.destroy()
            previous = {
                k: bool(v.get())
                for k, v in getattr(self, var_map_name, {}).items()
            }
            var_map: dict[str, tk.BooleanVar] = {}
            meta_blocks: list[dict[str, Any]] = []
            for group, names in grouped.items():
                group_label = ttk.Label(host, text=group, foreground="#666")
                group_label.pack(anchor="w", pady=(4, 0))
                items: list[dict[str, Any]] = []
                for name in names:
                    var = tk.BooleanVar(value=bool(previous.get(name, False)))
                    var_map[name] = var
                    cb = ttk.Checkbutton(
                        host,
                        text=f"  {name}",
                        variable=var,
                        command=self._on_interaction_bulk_selection_changed,
                    )
                    cb.pack(anchor="w")
                    items.append({"name": name, "checkbox": cb})
                meta_blocks.append({"group_label": group_label, "items": items})
            setattr(self, var_map_name, var_map)
            setattr(self, meta_name, meta_blocks)
            self._apply_interaction_bulk_filter(side)
        self._sync_interaction_body_state()

    def _add_interaction_pair(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction import (
            interaction_column_name,
            normalize_interaction_pair,
        )

        a = str(self._interaction_feat_a_var.get() or "").strip()
        b = str(self._interaction_feat_b_var.get() or "").strip()
        if not a or not b:
            messagebox.showinfo("Interaction", "Select Feature A and Feature B.", parent=self)
            return
        op = self._interaction_op_id()
        pair = normalize_interaction_pair({
            "left": a,
            "right": b,
            "op": op,
            "output": interaction_column_name(a, b, op),
        })
        existing = {str(p.get("output") or "") for p in self._interaction_pairs}
        if pair["output"] in existing:
            messagebox.showinfo(
                "Interaction",
                f"Pair already configured:\n{pair['output']}",
                parent=self,
            )
            return
        self._interaction_pairs.append(pair)
        self._rebuild_interaction_pairs_list()
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()

    def _remove_interaction_pair(self) -> None:
        if not hasattr(self, "_interaction_pairs_list"):
            return
        sel = self._interaction_pairs_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        # Listbox may show multi-line entries — map by stored index tags.
        # We store one row per pair.
        if 0 <= idx < len(self._interaction_pairs):
            del self._interaction_pairs[idx]
            self._rebuild_interaction_pairs_list()
            self._refresh_interaction_feature_lists()
            self._update_lag_preview()
            self._save_prefs()

    def _clear_interaction_pairs(self) -> None:
        self._interaction_pairs = []
        self._rebuild_interaction_pairs_list()
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()

    def _add_interaction_pairs_to_pipeline(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            available_interaction_features_from_config,
            register_staged_interaction_pairs_to_pipeline,
        )

        if not self._interaction_pairs:
            messagebox.showinfo(
                "Interaction",
                "No configured pairs to add. Use Single or Bulk Interaction first.",
                parent=self,
            )
            return
        try:
            cfg = self._build_transformation_config_through_interaction()
        except Exception as exc:
            messagebox.showerror("Add to Pipeline", str(exc), parent=self)
            return
        avail = available_interaction_features_from_config(
            cfg,
            master_features=self._laggable_feature_names(),
            sample_interval_sec=self._interval_sec(),
        )
        pipeline, counts, errors = register_staged_interaction_pairs_to_pipeline(
            list(self._interaction_pairs),
            list(self._interaction_pipeline_pairs),
            available_features=avail,
        )
        self._interaction_pipeline_pairs = pipeline
        if int(counts.get("added") or 0) > 0:
            self._interaction_enabled_var.set(True)
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()
        summary = (
            f"Added: {counts.get('added', 0)}\n"
            f"Skipped (already exists): {counts.get('skipped', 0)}\n"
            f"Failed: {counts.get('failed', 0)}"
        )
        if errors:
            detail = "\n".join(errors[:5])
            if len(errors) > 5:
                detail += f"\n… +{len(errors) - 5} more"
            summary = f"{summary}\n\n{detail}"
        messagebox.showinfo("Add to Pipeline", summary, parent=self)

    def _generate_bulk_interaction_pairs(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            BULK_CONFIRM_THRESHOLD,
            bulk_interaction_pairs,
        )

        a_sel = [n for n, v in self._interaction_bulk_a_vars.items() if v.get()]
        b_sel = [n for n, v in self._interaction_bulk_b_vars.items() if v.get()]
        if not a_sel or not b_sel:
            messagebox.showinfo(
                "Interaction",
                "Select at least one Feature A and one Feature B for bulk generate.",
                parent=self,
            )
            return
        new_pairs = bulk_interaction_pairs(a_sel, b_sel, op=self._interaction_op_id())
        if len(new_pairs) > BULK_CONFIRM_THRESHOLD:
            if not messagebox.askyesno(
                "Interaction",
                f"This will add {len(new_pairs)} interaction pairs. Continue?",
                parent=self,
            ):
                return
        existing = {str(p.get("output") or "") for p in self._interaction_pairs}
        added = 0
        for pair in new_pairs:
            if pair["output"] in existing:
                continue
            self._interaction_pairs.append(pair)
            existing.add(pair["output"])
            added += 1
        self._rebuild_interaction_pairs_list()
        self._refresh_interaction_feature_lists()
        self._update_lag_preview()
        self._save_prefs()
        messagebox.showinfo("Interaction", f"Added {added} pair(s).", parent=self)

    def _rebuild_interaction_pairs_list(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            format_pair_chain_entry,
            format_lineage_tree,
        )

        if not hasattr(self, "_interaction_pairs_list"):
            return
        lb = self._interaction_pairs_list
        lb.delete(0, tk.END)
        parent_map = {
            str(p.get("output") or ""): p
            for p in self._interaction_pairs
            if isinstance(p, dict) and p.get("output")
        }
        for i, pair in enumerate(self._interaction_pairs, start=1):
            block = format_pair_chain_entry(i, pair)
            # Listbox is single-line; use compact display + lineage below.
            out = str(pair.get("output") or "")
            left = str(pair.get("left") or "")
            right = str(pair.get("right") or "")
            op = str(pair.get("op") or "multiply")
            lb.insert(tk.END, f"{i}. {left}  [{op}]  {right}  →  {out}")
        if self._interaction_pairs:
            last = self._interaction_pairs[-1]
            self._interaction_lineage_var.set(
                format_lineage_tree(str(last.get("output") or ""), parent_map=parent_map)
            )
        else:
            self._interaction_lineage_var.set("")

    def _build_transformation_config(self) -> dict[str, Any]:
        # Materialize transform widgets (withdrawn) so prefs/selections are available.
        any_enabled = (
            bool(self._lag_enabled_var.get())
            or bool(self._difference_enabled_var.get())
            or bool(self._return_enabled_var.get())
            or bool(self._rolling_enabled_var.get())
            or bool(self._exponential_rolling_enabled_var.get())
            or bool(self._ohlc_aggregation_enabled_var.get())
            or bool(self._interaction_enabled_var.get())
            or bool(self._normalization_enabled_var.get())
            or bool(self._regime_enabled_var.get())
            or bool(self._math_enabled_var.get())
        )
        if any_enabled and not getattr(self, "_xform_ui_built", False):
            try:
                win = self._ensure_feature_transformations_window()
                win.withdraw()
            except tk.TclError:
                pass
        return self._merge_post_interaction_transforms(
            self._build_transformation_config_through_interaction()
        )

    def _build_transformation_config_through_interaction(self) -> dict[str, Any]:
        """Pipeline config through Interaction (inputs for Math / Norm / Regime)."""
        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            merge_interaction_into_config,
        )
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            build_lag_transformation_config,
        )
        from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
            merge_exponential_rolling_into_config,
        )
        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            merge_ohlc_aggregation_into_config,
        )
        from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
            merge_rolling_into_config,
        )

        base = build_lag_transformation_config(
            enabled=bool(self._lag_enabled_var.get()),
            features=self._selected_lag_features(),
            lag_seconds=self._selected_lag_seconds(),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
            difference_enabled=bool(self._difference_enabled_var.get()),
            difference_features=self._selected_difference_features(),
            difference_lag_seconds=self._selected_difference_seconds(),
            return_enabled=bool(self._return_enabled_var.get()),
            return_features=self._selected_return_features(),
            return_lag_seconds=self._selected_return_seconds(),
        )
        with_rolling = merge_rolling_into_config(
            base,
            enabled=bool(self._rolling_enabled_var.get()),
            features=self._selected_rolling_features(),
            windows=self._selected_rolling_windows(),
            operations=self._selected_rolling_operations(),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
        with_exp = merge_exponential_rolling_into_config(
            with_rolling,
            enabled=bool(self._exponential_rolling_enabled_var.get()),
            features=self._selected_exponential_features(),
            periods=self._selected_exponential_periods(),
            operations=self._selected_exponential_operations(),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
        with_ohlc = merge_ohlc_aggregation_into_config(
            with_exp,
            enabled=bool(self._ohlc_aggregation_enabled_var.get()),
            features=self._selected_ohlc_features(),
            timeframes=self._selected_ohlc_timeframes(),
            outputs=self._selected_ohlc_outputs(),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
        return merge_interaction_into_config(
            with_ohlc,
            enabled=bool(self._interaction_enabled_var.get()),
            pairs=list(self._interaction_pipeline_pairs),
        )

    def _registry_feature_count(self) -> int:
        """Canonical Feature Registry count present in this Master (not raw DB width).

        Overview still shows the on-disk Master ``feature_count`` (may be stale).
        Create Dataset / Feature Transformation Preview use this filtered count.
        """
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            registry_feature_count_from_master,
        )

        d = self._preview_detail or self._detail or {}
        schema = d.get("build_schema") if isinstance(d.get("build_schema"), dict) else {}
        fallback = int(
            d.get("feature_count")
            or schema.get("feature_count")
            or 0
        )
        return registry_feature_count_from_master(
            self._master_feature_columns(),
            fallback_feature_count=fallback,
            exclude_names=self._registry_retired_features(),
        )

    def _update_registry_export_panel(self) -> None:
        cached = (self._preview_detail or {}).get("sample_preview")
        busy = self._registry_runner.running or self._export_count_busy
        scope = self._preview_scope_kwargs()
        if not scope:
            self._registry_rows_var.set("—")
            self._registry_scope_var.set("—")
            self._registry_features_var.set("—")
            self._registry_name_var.set("—")
            if hasattr(self, "_create_registry_btn"):
                self._create_registry_btn.configure(state="disabled")
            return

        fc = self._registry_feature_count()
        export_preview = self._build_export_preview_dict()
        no_null = bool(self._no_null_data_var.get())
        all_days, selected, filter_meta = self._trading_day_filter_scope()
        day_count = preview_scope_day_count(
            all_days=all_days,
            selected_days=selected,
            trading_day=None,
            master_day_count=len(self._master_day_keys()),
        )
        selected_n = int(filter_meta.get("selected_days") or 0)
        exported_n = int(filter_meta.get("exported_days") or 0)
        if filter_meta.get("mode") not in (None, "all") and selected_n != exported_n:
            self._registry_scope_var.set(f"{_fmt_num(exported_n)} / {_fmt_num(selected_n)}")
        else:
            self._registry_scope_var.set(_fmt_num(day_count))
        self._registry_features_var.set(str(fc) if fc else "—")
        self._registry_name_var.set(self._proposed_registry_name(fc))

        # Matching rows = optional Apply Filters estimate only (never required for Create).
        has_estimate = bool(
            cached and preview_matches_export_settings(cached, export_preview, no_null_data=no_null)
        )
        if has_estimate:
            match = preview_display_match_count(cached, no_null_data=no_null)
            self._registry_rows_var.set(_fmt_num(match))
        else:
            self._registry_rows_var.set("—")

        db_exists = bool(self._detail and self._detail.get("exists"))
        enabled = (
            db_exists
            and not busy
            and bool(all_days or selected)
        )
        if hasattr(self, "_create_registry_btn"):
            self._create_registry_btn.configure(state="normal" if enabled else "disabled")
        self._update_lag_preview()

    def _set_registry_export_active(self, active: bool) -> None:
        if hasattr(self, "_create_registry_btn"):
            self._create_registry_btn.configure(state="disabled" if active else "normal")
        self._update_registry_export_panel()

    def _render_overview(self, d: dict[str, Any]) -> None:
        sizes = d.get("file_sizes") or {}
        exists = d.get("exists")
        self._overview_labels["file_status"].configure(
            text="exists" if exists else "not created",
            foreground="#81c784" if exists else "#ef5350",
        )
        self._overview_labels["total_rows"].configure(text=_fmt_num(d.get("row_count")))
        self._overview_labels["trading_days"].configure(
            text=_fmt_num(len(d.get("days_in_master") or [])),
        )
        self._overview_labels["features"].configure(text=_fmt_num(d.get("feature_count")))
        self._overview_labels["targets"].configure(text=_fmt_num(d.get("target_count")))
        self._overview_labels["columns"].configure(text=_fmt_num(d.get("samples_column_count")))
        self._overview_labels["db_size"].configure(text=_fmt_bytes(sizes.get("db_bytes")))
        self._overview_labels["wal_size"].configure(text=_fmt_bytes(sizes.get("wal_bytes")))
        sqlite = d.get("sqlite") or {}
        self._overview_labels["journal"].configure(text=str(sqlite.get("journal_mode") or "—"))
        self._path_var.set(d.get("master_db_abs") or "—")

    def _render_days(self, d: dict[str, Any]) -> None:
        for item in self.days_tree.get_children():
            self.days_tree.delete(item)
        for day in d.get("day_details") or []:
            td = day.get("trading_day") or ""
            tmin = _fmt_ts(day.get("timestamp_min"))
            tmax = _fmt_ts(day.get("timestamp_max"))
            sel = "☑" if td in self._selected_days else "☐"
            exp_count = day.get("expiry_count")
            if day.get("is_expiry_day"):
                expiries_txt = f"✓ {_fmt_num(exp_count)}" if exp_count is not None else "✓"
            else:
                expiries_txt = _fmt_num(exp_count)
            self.days_tree.insert(
                "",
                "end",
                iid=td,
                values=(
                    sel,
                    td,
                    _fmt_num(day.get("row_count")),
                    _fmt_num(day.get("token_count")),
                    expiries_txt,
                    f"{tmin} → {tmax}",
                ),
            )
        if self._selected_day and self._selected_day in self.days_tree.get_children():
            self.days_tree.selection_set(self._selected_day)
        self._update_days_selected_hint()

    def _render_builder(self, d: dict[str, Any]) -> None:
        # Don't overwrite live Create Dataset log while export is running.
        if self._registry_runner.running:
            return
        p = d.get("builder_progress")
        if not p:
            self._set_builder_text("No builder progress (DB not created)")
            return
        lines = [
            f"Status: {p.get('status') or 'idle'}",
            f"Current: {p.get('current_day') or '—'}",
            f"Last done: {p.get('last_completed_day') or '—'}",
            f"Days: {p.get('days_done', '—')} / {p.get('days_total', '—')}",
        ]
        if p.get("error_message"):
            lines.append(f"Error: {p['error_message']}")
        if self._registry_export_log:
            lines.append("")
            lines.append("--- Last Create Dataset log ---")
            lines.extend(self._registry_export_log[-40:])
        self._set_builder_text("\n".join(lines))

    def _render_meta(self, d: dict[str, Any]) -> None:
        self._meta_text.configure(state="normal")
        self._meta_text.delete("1.0", "end")
        parts: list[str] = []
        keys = d.get("dataset_meta_keys") or []
        if keys:
            parts.append("Keys: " + ", ".join(k.get("key", "") for k in keys))
        cfg = d.get("master_config")
        if cfg:
            parts.append("\n--- master_config ---\n" + json.dumps(cfg, indent=2))
        schema = d.get("build_schema")
        if isinstance(schema, dict):
            slim = dict(schema)
            feats = slim.get("feature_columns")
            if isinstance(feats, list) and len(feats) > 12:
                slim["feature_columns"] = feats[:12] + [f"… +{len(feats) - 12} more"]
            parts.append("\n--- build_schema ---\n" + json.dumps(slim, indent=2))
        if not parts:
            parts.append("No dataset_meta yet — add days from Create Dataset.")
        self._meta_text.insert("1.0", "\n".join(parts))
        self._meta_text.configure(state="disabled")

    def _render_feature_health(self, d: dict[str, Any]) -> None:
        from . import feature_policy_format as pol_fmt

        report = d.get("feature_policy_report")
        if not report and isinstance(d.get("feature_policy"), dict):
            try:
                from chain_replay_ml.feature_policy import build_policy_report

                report = build_policy_report(d["feature_policy"])
            except Exception:
                report = None
        text = pol_fmt.format_feature_health_report(report) if report else (
            "No feature health report yet — rebuild or add days after policy engine is enabled."
        )
        self._health_text.configure(state="normal")
        self._health_text.delete("1.0", "end")
        self._health_text.insert("1.0", text)
        self._health_text.configure(state="disabled")

    def _render_tables(self, d: dict[str, Any]) -> None:
        for item in self._tables_tree.get_children():
            self._tables_tree.delete(item)
        for t in (d.get("sqlite") or {}).get("tables") or []:
            self._tables_tree.insert(
                "",
                "end",
                text=t.get("name") or "",
                values=(_fmt_num(t.get("row_count")), _fmt_num(len(t.get("columns") or []))),
            )

    def _on_day_selected(self, _event: tk.Event | None = None) -> None:
        sel = self.days_tree.selection()
        self._selected_day = sel[0] if sel else None

    def _on_days_tree_click(self, event: tk.Event) -> None:
        region = self.days_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.days_tree.identify_column(event.x)
        iid = self.days_tree.identify_row(event.y)
        if not iid or col != "#1":
            return
        if iid in self._selected_days:
            self._selected_days.discard(iid)
        else:
            self._selected_days.add(iid)
        self._day_selection_explicit = True
        self._render_days(self._detail or {})
        self._clear_preview()
        self._update_registry_export_panel()
        self._save_prefs()

    def _select_all_days(self) -> None:
        self._selected_days = set(self._master_day_keys())
        self._day_selection_explicit = True
        self._render_days(self._detail or {})
        self._clear_preview()
        self._update_registry_export_panel()
        self._save_prefs()

    def _clear_day_selection(self) -> None:
        self._selected_days.clear()
        self._day_selection_explicit = True
        self._render_days(self._detail or {})
        self._clear_preview()
        self._update_registry_export_panel()
        self._save_prefs()

    def _open_chain_insert(self) -> None:
        if callable(self._on_open_create_dataset):
            self._on_open_create_dataset()

    def _clear_preview(self) -> None:
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        self._preview_detail = None
        self._preview_token = None
        self._preview_drill_day = None
        self._preview_hint_var.set("")
        self._clear_token_btn.configure(state="disabled")
        self._download_csv_btn.configure(state="disabled")
        self._update_registry_export_panel()

    def _clear_token_filter(self) -> None:
        self._preview_token = None
        self._preview_drill_day = None
        self._clear_token_btn.configure(state="disabled")
        if self._preview_scope_kwargs():
            self._apply_preview()
        else:
            self._clear_preview()

    def _on_preview_row_double_click(self, event: tk.Event) -> None:
        region = self.preview_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        iid = self.preview_tree.identify_row(event.y)
        if not iid:
            return
        vals = self.preview_tree.item(iid, "values")
        if len(vals) < 3:
            return
        trading_day = str(vals[0] or "").strip()
        token = str(vals[2] or "").strip()
        if not trading_day or not token:
            return
        self._preview_drill_day = trading_day
        self._preview_token = token
        self._clear_token_btn.configure(state="normal")
        self._apply_preview()

    def _download_preview_csv(self) -> None:
        preview = (self._preview_detail or {}).get("sample_preview") or {}
        no_null = bool(self._no_null_data_var.get())
        match = preview_display_match_count(preview, no_null_data=no_null)
        if match <= 0:
            messagebox.showwarning("Download CSV", "No rows match the current filter.")
            return
        from chain_replay_ml.dataset_builder.master_status import (
            MasterSampleCsvError,
            build_master_sample_csv_bytes,
        )

        kwargs = build_sample_csv_kwargs(
            market=self._market_var.get(),
            interval_sec=self._interval_sec(),
            preview=preview,
            no_null_data=no_null,
        )
        try:
            default_name, csv_bytes, row_count = build_master_sample_csv_bytes(
                self._data_dir(),
                **kwargs,
            )
        except MasterSampleCsvError as exc:
            messagebox.showerror("Download CSV", str(exc.detail))
            return
        except Exception as exc:
            messagebox.showerror("Download CSV", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save sample CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "wb") as fh:
                fh.write(csv_bytes)
            messagebox.showinfo("Download CSV", f"Saved {row_count:,} rows to\n{path}")
        except OSError as exc:
            messagebox.showerror("Download CSV", str(exc))

    def _create_registry_dataset(self) -> None:
        if self._registry_runner.running or self._export_count_busy:
            return
        if not self._detail or not self._detail.get("exists"):
            messagebox.showwarning("Create Dataset", "No master DB for this market/interval.")
            return
        all_days, selected, filter_meta = self._trading_day_filter_scope()
        if not all_days and not selected:
            self._empty_scope_warning(title="Create Dataset")
            return

        # Snapshot current UI selections — Create does not depend on Apply Filters.
        export_preview = self._build_export_preview_dict()
        no_null = bool(self._no_null_data_var.get())
        cached = (self._preview_detail or {}).get("sample_preview")
        estimated: int | None = None
        if cached and preview_matches_export_settings(cached, export_preview, no_null_data=no_null):
            estimated = preview_display_match_count(cached, no_null_data=no_null)
        self._confirm_and_start_registry_export(
            export_preview,
            filter_meta,
            estimated_rows=estimated,
        )

    def _confirm_and_start_registry_export(
        self,
        export_preview: dict[str, Any],
        filter_meta: dict[str, Any] | None = None,
        *,
        estimated_rows: int | None = None,
    ) -> None:
        from chain_replay_ml.dataset_builder.trading_day_filter import trading_day_filter_label

        fc = self._registry_feature_count()
        proposed = self._proposed_registry_name(fc)
        all_days, selected, scope_meta = self._trading_day_filter_scope()
        meta = filter_meta or scope_meta
        day_count = preview_scope_day_count(
            all_days=all_days,
            selected_days=selected,
            trading_day=None,
            master_day_count=len(self._master_day_keys()),
        )
        no_null = bool(self._no_null_data_var.get())
        atm = str(self._atm_var.get() or "").strip() or "all"
        atm_line = f"ATM Band     : ±{atm}" if atm != "all" else "ATM Band     : all"
        if bool(self._prem_en_var.get()):
            pmin = str(self._prem_min_var.get() or "").strip() or "?"
            pmax = str(self._prem_max_var.get() or "").strip() or "?"
            ltp_line = f"LTP Range    : ₹{pmin} - ₹{pmax}"
        else:
            ltp_line = "LTP Range    : off"
        null_line = f"No Null Data : {'Yes' if no_null else 'No'}"
        filter_note = ""
        mode = str(meta.get("mode") or "all")
        if mode != "all":
            filter_note = (
                f"\nTrading day filter: {trading_day_filter_label(mode)} "
                f"({meta.get('exported_days')}/{meta.get('selected_days')} days)"
            )
        estimate_note = ""
        if estimated_rows is not None and estimated_rows > 0:
            estimate_note = f"\n\nEstimated rows (previous preview): ~{estimated_rows:,}"
        elif estimated_rows == 0:
            estimate_note = (
                "\n\nEstimated rows (previous preview): 0 — "
                "create will fail if filters still match nothing."
            )

        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            validate_time_shift_settings,
        )

        lag_err = validate_time_shift_settings(
            lag_enabled=bool(self._lag_enabled_var.get()),
            difference_enabled=bool(self._difference_enabled_var.get()),
            return_enabled=bool(self._return_enabled_var.get()),
            features=self._selected_lag_features(),
            lag_seconds=self._selected_lag_seconds(),
            sample_interval_sec=self._interval_sec(),
            available_features=self._laggable_feature_names(),
            difference_features=self._selected_difference_features(),
            difference_lag_seconds=self._selected_difference_seconds(),
            return_features=self._selected_return_features(),
            return_lag_seconds=self._selected_return_seconds(),
        )
        if lag_err:
            messagebox.showerror("Create Dataset — Transformations", lag_err)
            return

        from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
            validate_rolling_settings,
        )

        rolling_err = validate_rolling_settings(
            enabled=bool(self._rolling_enabled_var.get()),
            features=self._selected_rolling_features(),
            windows=self._selected_rolling_windows(),
            operations=self._selected_rolling_operations(),
            available_features=self._laggable_feature_names(),
        )
        if rolling_err:
            messagebox.showerror("Create Dataset — Rolling", rolling_err)
            return

        from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
            validate_exponential_rolling_settings,
        )

        exp_err = validate_exponential_rolling_settings(
            enabled=bool(self._exponential_rolling_enabled_var.get()),
            features=self._selected_exponential_features(),
            periods=self._selected_exponential_periods(),
            operations=self._selected_exponential_operations(),
            available_features=self._laggable_feature_names(),
        )
        if exp_err:
            messagebox.showerror("Create Dataset — Exponential Rolling", exp_err)
            return

        from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
            validate_ohlc_aggregation_settings,
        )

        ohlc_err = validate_ohlc_aggregation_settings(
            enabled=bool(self._ohlc_aggregation_enabled_var.get()),
            features=self._selected_ohlc_features(),
            timeframes=self._selected_ohlc_timeframes(),
            outputs=self._selected_ohlc_outputs(),
            available_features=self._laggable_feature_names(),
            sample_interval_sec=self._interval_sec(),
        )
        if ohlc_err:
            messagebox.showerror("Create Dataset — OHLC Aggregation", ohlc_err)
            return

        from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
            available_interaction_features_from_config,
            validate_interaction_for_export,
        )

        ix_cfg = self._build_transformation_config_through_interaction()
        ix_avail = available_interaction_features_from_config(
            ix_cfg,
            master_features=self._laggable_feature_names(),
            sample_interval_sec=self._interval_sec(),
        )
        ix_err = validate_interaction_for_export(
            enabled=bool(self._interaction_enabled_var.get()),
            pairs=list(self._interaction_pipeline_pairs),
            available_features=ix_avail,
        )
        if ix_err:
            messagebox.showerror("Create Dataset — Interaction", ix_err)
            return

        from chain_replay_ml.dataset_builder.transformations.math_ui import (
            validate_math_settings,
        )
        from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
            validate_normalization_settings,
        )
        from chain_replay_ml.dataset_builder.transformations.regime_ui import (
            validate_regime_settings,
        )

        math_err = validate_math_settings(
            enabled=bool(self._math_enabled_var.get()),
            features=self._selected_math_features(),
            operations=self._selected_math_operations(),
            available_features=ix_avail,
        )
        if math_err:
            messagebox.showerror("Create Dataset — Math (Unary)", math_err)
            return
        norm_err = validate_normalization_settings(
            enabled=bool(self._normalization_enabled_var.get()),
            features=self._selected_normalization_features(),
            methods=self._selected_normalization_methods(),
            windows=self._selected_normalization_windows(),
            available_features=ix_avail,
        )
        if norm_err:
            messagebox.showerror("Create Dataset — Normalization", norm_err)
            return
        regime_err = validate_regime_settings(
            enabled=bool(self._regime_enabled_var.get()),
            features=self._selected_regime_features(),
            methods=self._selected_regime_methods(),
            windows=self._selected_regime_windows(),
            available_features=ix_avail,
        )
        if regime_err:
            messagebox.showerror("Create Dataset — Regime / Bucket", regime_err)
            return

        lag_counts = self._lag_preview_counts()
        feature_count_block = (
            f"\n\nFeature Count\n"
            f"Current columns : {lag_counts['current_columns']}\n"
            f"Lag columns     : {lag_counts['lag_columns']}\n"
            f"Final columns   : {lag_counts['final_columns']}"
        )

        if not messagebox.askyesno(
            "Create Dataset",
            f"Create Dataset\n\n"
            f"Trading Days : {_fmt_num(day_count)}\n"
            f"{atm_line}\n"
            f"{ltp_line}\n"
            f"{null_line}"
            f"{filter_note}"
            f"{feature_count_block}\n\n"
            f"Name: {proposed}\n\n"
            f"Create dataset using these settings?"
            f"{estimate_note}",
        ):
            return

        export_kwargs = build_registry_export_kwargs(
            market=self._market_var.get(),
            interval_sec=self._interval_sec(),
            preview=export_preview,
            feature_count=fc,
            dataset_name=proposed,
            audit_validation_required=self._audit_required_var.get(),
            no_null_data=no_null,
            trading_day_filter=meta,
            transformation_config=self._build_transformation_config(),
        )
        self._registry_progress_var.set(f"Creating {proposed}…")
        self._set_registry_export_active(True)
        self._registry_export_t0 = None
        self._show_builder_progress_tab()
        self._clear_builder_export_log(
            f"Create Dataset started: {proposed}\n"
            f"(No-Null={'ON' if no_null else 'OFF'}; "
            "status lines append here while the job runs)"
        )
        get_build_progress_manager().begin_job(
            "registry_export",
            title="Exporting Dataset",
            cancel_fn=None,
        )
        progress_total = int(estimated_rows) if estimated_rows and estimated_rows > 0 else 0
        get_build_progress_manager().publish({
            "status": "running",
            "job_kind": "registry_export",
            "job_title": "Exporting Dataset",
            "stage_name": "Prepare Export",
            "message": f"Creating {proposed}…",
            "rows": 0,
            "total": progress_total,
            "percent": 2.0,
            "elapsed_sec": 0.0,
            "pipeline": {
                "total_elapsed_sec": 0.0,
                "total_elapsed_label": "00:00",
            },
        })

        def on_progress(msg: str, cur: int, tot: int) -> None:
            self._registry_progress_queue.put(str(msg))
            self._publish_registry_progress(msg, cur, tot)

        def on_done(result: dict[str, Any]) -> None:
            self._registry_queue.put(result)

        try:
            self._registry_runner.start(
                export_kwargs=export_kwargs,
                on_done=on_done,
                on_progress=on_progress,
            )
            self._append_builder_log("Worker thread started — waiting for stage updates…")
        except Exception as exc:
            self._set_registry_export_active(False)
            self._append_builder_log(f"FAILED to start: {exc}")
            messagebox.showerror("Create Dataset", str(exc))

    def _preview_kwargs(self) -> dict[str, Any]:
        scope = self._preview_scope_kwargs()
        filters = self._preview_filter_kwargs()
        preview_limit = 5000 if self._preview_token else 12
        return {
            "data_dir": self._data_dir(),
            "market": self._market_var.get(),
            "interval_sec": self._interval_sec(),
            "preview_limit": preview_limit,
            "preview_token": self._preview_token,
            "no_null_data": bool(self._no_null_data_var.get()),
            **scope,
            **filters,
        }

    def _set_preview_loading(self, active: bool) -> None:
        if hasattr(self, "_apply_filters_btn"):
            self._apply_filters_btn.configure(state="disabled" if active else "normal")
        if active:
            self._preview_hint_var.set("Applying filters… (UI stays responsive)")
        else:
            # Clear the applying hint when idle; render path sets the real count text.
            if (self._preview_hint_var.get() or "").startswith("Applying filters"):
                self._preview_hint_var.set("")
        self.configure(cursor="watch" if active else "")
        self._update_registry_export_panel()

    def _on_no_null_data_toggled(self) -> None:
        preview = (self._preview_detail or {}).get("sample_preview")
        if self._no_null_data_var.get() and preview and preview.get("no_null_match_count") is None:
            self._apply_preview()
            return
        if preview:
            self._render_preview_result(preview)
        self._update_registry_export_panel()

    def _apply_preview(self) -> None:
        if not self._preview_scope_kwargs():
            self._empty_scope_warning(title="Preview")
            return
        if self._preview_busy:
            return
        self._preview_busy = True
        self._preview_load_id += 1
        load_id = self._preview_load_id
        kwargs = self._preview_kwargs()
        run_report = bool(self._no_null_filter_report_var.get())
        report_kwargs = self._no_null_report_kwargs() if run_report else None
        self._set_preview_loading(True)
        if run_report:
            self._show_builder_progress_tab()
            self._clear_builder_export_log(
                "No-Null Data Filter Report — starting (diagnostics only)…"
            )

        def worker() -> None:
            err: str | None = None
            preview: dict[str, Any] | None = None
            report_text: str | None = None
            try:
                if run_report and report_kwargs is not None:
                    from chain_replay_ml.dataset_builder.no_null_filter_report import (
                        build_no_null_filter_report_text,
                    )

                    def _prog(msg: str) -> None:
                        self.after(0, lambda m=msg: self._append_builder_log(m))

                    report_text = build_no_null_filter_report_text(
                        on_progress=_prog,
                        **report_kwargs,
                    )
                from chain_replay_ml.dataset_builder.master_status import (
                    read_master_sample_preview,
                )

                preview = read_master_sample_preview(**kwargs)
            except Exception as exc:
                err = str(exc)
            self.after(
                0,
                lambda: self._on_preview_loaded(
                    load_id, preview, err, report_text=report_text
                ),
            )

        threading.Thread(target=worker, daemon=True, name="master-preview").start()

    def _no_null_report_kwargs(self) -> dict[str, Any]:
        """Filter scope for the diagnostic No-Null report (read-only)."""
        scope = self._preview_scope_kwargs()
        filters = self._export_filter_kwargs()
        kwargs: dict[str, Any] = {
            "db_path": self._master_db_path(),
            "chart_dir": self.chart_dir,
            "atm_band_filter": filters.get("atm_band_filter"),
            "premium_min": filters.get("premium_min"),
            "premium_max": filters.get("premium_max"),
            "delta_min": filters.get("delta_min"),
            "delta_max": filters.get("delta_max"),
            "token": self._preview_token,
        }
        if scope.get("all_days"):
            kwargs["all_days"] = True
        elif scope.get("preview_day"):
            kwargs["trading_day"] = str(scope["preview_day"])
        elif scope.get("selected_days"):
            kwargs["selected_days"] = list(scope["selected_days"])
        return kwargs

    def _on_preview_loaded(
        self,
        load_id: int,
        preview: dict[str, Any] | None,
        err: str | None,
        report_text: str | None = None,
    ) -> None:
        if load_id != self._preview_load_id:
            return
        self._preview_busy = False
        self._set_preview_loading(False)
        if report_text:
            self._show_builder_progress_tab()
            # Replace live progress crumbs with the full diagnostic report.
            self._registry_export_log = [
                ln for ln in str(report_text).splitlines() if ln is not None
            ]
            self._registry_export_log.append(
                "No-Null Data Filter Report — complete (dataset unchanged)."
            )
            self._set_builder_text("\n".join(self._registry_export_log) + "\n")
        if err:
            if report_text:
                self._append_builder_log(f"Preview failed after report: {err}")
            messagebox.showerror("Preview", err)
            return
        base = dict(self._detail or {})
        base["sample_preview"] = preview
        self._detail = base
        self._preview_detail = base
        self._render_preview_result(preview or {})

    def _render_preview_result(self, prev: dict[str, Any]) -> None:
        rows = prev.get("rows") or []
        no_null = bool(self._no_null_data_var.get())
        match = preview_display_match_count(prev, no_null_data=no_null)
        total = int(prev.get("match_count") or 0)
        day_count = preview_day_count_from_preview(
            prev,
            master_day_count=len(self._master_day_keys()),
        )
        token_note = f" · token {prev['token']}" if prev.get("token") else ""
        count_note = f"{_fmt_num(match)} matches"
        if no_null and prev.get("no_null_match_count") is not None:
            nn = prev.get("no_null_report") or {}
            if nn:
                count_note = (
                    f"{_fmt_num(match)} complete rows "
                    f"(of {_fmt_num(total)} filtered; "
                    f"cols {nn.get('columns_before')}→{nn.get('columns_after')}, "
                    f"dropped {nn.get('empty_columns_removed', 0)} empty cols, "
                    f"{nn.get('incomplete_rows_removed', 0)} incomplete rows; "
                    f"NULL cells={nn.get('remaining_null_cells', 0)})"
                )
            else:
                dropped = prev.get("no_null_dropped_columns") or []
                drop_note = f", {len(dropped)} all-null cols ignored" if dropped else ""
                count_note = (
                    f"{_fmt_num(match)} complete rows "
                    f"(of {_fmt_num(total)} filtered{drop_note})"
                )
        self._preview_hint_var.set(
            f"{_fmt_num(day_count)} trading days{token_note} · {len(rows)} shown of {count_note}",
        )
        self._clear_token_btn.configure(state="normal" if prev.get("token") else "disabled")
        self._download_csv_btn.configure(state="normal" if match > 0 else "disabled")

        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        for row in rows:
            self.preview_tree.insert(
                "",
                "end",
                values=(
                    row.get("trading_day", ""),
                    _fmt_ts(row.get("timestamp")),
                    row.get("token", ""),
                    row.get("symbol", ""),
                    row.get("ltp", ""),
                    row.get("spot", ""),
                ),
            )
        self._update_registry_export_panel()

    def _master_db_path(self) -> str:
        from chain_replay_ml.dataset_builder.master_naming import resolve_master_db_path

        return resolve_master_db_path(
            self._data_dir(),
            market=self._market_var.get(),
            sampling_interval_sec=self._interval_sec(),
        )

    def _open_folder(self) -> None:
        try:
            open_path(os.path.dirname(self._master_db_path()))
        except Exception as exc:
            messagebox.showerror("Open folder", str(exc))

    def _delete_day(self) -> None:
        if not self._selected_day:
            messagebox.showwarning("Delete", "Select a trading day first.")
            return
        td = self._selected_day
        if not messagebox.askyesno("Delete day", f"Delete all samples for {td} from master DB?"):
            return
        path = self._master_db_path()
        if not os.path.isfile(path):
            messagebox.showerror("Delete", "Master DB file does not exist.")
            return
        try:
            from chain_replay_ml.dataset_builder.master_store import MasterStore

            store = MasterStore(path)
            store.open()
            try:
                deleted = store.delete_day(td)
            finally:
                store.close()
            messagebox.showinfo("Delete", f"Removed {deleted:,} rows for {td}.")
            self._selected_day = None
            self._selected_days.discard(td)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Delete", str(exc))

    def _delete_all(self) -> None:
        from chain_replay_ml.dataset_builder.master_status import (
            delete_master_database_files,
            related_master_artifact_paths,
        )

        path = self._master_db_path()
        related = related_master_artifact_paths(path)
        if not any(os.path.isfile(p) for p in related):
            messagebox.showinfo("Delete", "No master DB file for this market/interval.")
            return
        if not messagebox.askyesno(
            "Delete all",
            "Delete entire master database for a fresh start?\n\n"
            f"{path}\n\n"
            "This removes the DB, WAL/SHM, and related backups "
            "(e.g. .pre_rebuild_*.bak). Meta, days, and builder progress "
            "are cleared. This cannot be undone.",
        ):
            return
        try:
            result = delete_master_database_files(path)
            if result.still_exists:
                raise RuntimeError("; ".join(result.errors) or "File still exists")
            self._selected_day = None
            self._selected_days.clear()
            self._detail = {}
            self._preview_detail = None
            n = len(result.removed)
            messagebox.showinfo(
                "Delete",
                f"Master database wiped ({n} file(s) removed).\n"
                "DB size is 0 — add days to start fresh.",
            )
            self.refresh(lazy=False)
        except Exception as exc:
            messagebox.showerror("Delete", str(exc))
