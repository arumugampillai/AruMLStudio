"""OHLC Aggregation — completed-candle history from fixed sample-interval bars.

Pipeline-only. Emits Open/High/Low/Close for completed candles only.
Derived body / range / return / ratios belong in Interaction, not here.

History depth is read from interval-specific profiles
(``ohlc_history_profiles.json``) — product configuration, not warm-up.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .ohlc_history_profiles import (
    OhlcTimeframeSpec,
    available_ohlc_timeframes,
    default_ohlc_history_profiles,
    get_ohlc_interval_profile,
    parse_history_overrides,
    resolve_timeframe_spec,
    timeframe_specs_metadata,
)
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
)

OHLC_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
_FIELD_ALIASES: dict[str, str] = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
}

DEFAULT_OHLC_FIELDS: tuple[str, ...] = OHLC_FIELDS

# Backward-compatible aliases: 3s profile timeframe map (lazy via property helpers).
def _legacy_timeframes() -> dict[str, OhlcTimeframeSpec]:
    return dict(get_ohlc_interval_profile(3).timeframes)


def _default_timeframe_keys(sample_interval_sec: float = 3.0) -> tuple[str, ...]:
    try:
        return available_ohlc_timeframes(sample_interval_sec)
    except Exception:
        return available_ohlc_timeframes(3)


# Module-level name kept for older imports/tests; reflects the 3s default profile.
OHLC_TIMEFRAMES: dict[str, OhlcTimeframeSpec] = {}
DEFAULT_OHLC_TIMEFRAMES: tuple[str, ...] = ()


def _ensure_legacy_exports() -> None:
    global OHLC_TIMEFRAMES, DEFAULT_OHLC_TIMEFRAMES
    if OHLC_TIMEFRAMES:
        return
    OHLC_TIMEFRAMES = _legacy_timeframes()
    DEFAULT_OHLC_TIMEFRAMES = tuple(OHLC_TIMEFRAMES.keys())


def normalize_ohlc_field(field: str) -> str:
    key = str(field or "").strip().lower()
    if key not in _FIELD_ALIASES:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid output field={field!r}. Supported: {list(OHLC_FIELDS)}."
        )
    return _FIELD_ALIASES[key]


def normalize_ohlc_timeframe(
    tf: str,
    *,
    sample_interval_sec: float | int | None = None,
) -> str:
    """Normalize a timeframe key; optionally require it in the interval profile."""
    key = str(tf or "").strip().lower()
    if not key:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid timeframe={tf!r}."
        )
    if sample_interval_sec is None:
        # Accept any known key across all profiles (for naming helpers / UI).
        known: set[str] = set()
        for profile in default_ohlc_history_profiles().values():
            known.update(profile.timeframes)
        if key not in known:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Invalid timeframe={tf!r}. Known: {sorted(known)}."
            )
        return key
    get_ohlc_interval_profile(sample_interval_sec).get(key)
    return key


def ohlc_aggregation_column_name(
    feature: str,
    timeframe: str,
    history_index: int,
    field: str,
    *,
    sample_interval_sec: float | int | None = None,
    history_len: int | None = None,
) -> str:
    """``{feature}_{tf}_{index}_{field}`` e.g. ``spot_ltp_3m_1_close``."""
    feat = str(feature or "").strip()
    interval = 3 if sample_interval_sec is None else sample_interval_sec
    tf = normalize_ohlc_timeframe(timeframe, sample_interval_sec=interval)
    fld = normalize_ohlc_field(field)
    idx = int(history_index)
    if history_len is None:
        max_hist = resolve_timeframe_spec(interval, tf).history
    else:
        max_hist = int(history_len)
    if idx <= 0 or idx > max_hist:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid history index={history_index!r} for {tf} "
            f"at {interval}s (valid 1..{max_hist})."
        )
    return f"{feat}_{tf}_{idx}_{fld}"


def period_rows_for_timeframe(
    timeframe: str,
    sample_interval_sec: float,
    *,
    history_overrides: dict[str, Any] | None = None,
) -> int:
    """Rows per candle from the interval profile (exact multiple required)."""
    interval = float(sample_interval_sec)
    if interval <= 0:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid sample_interval_sec={sample_interval_sec!r}."
        )
    spec = resolve_timeframe_spec(
        interval,
        timeframe,
        history_overrides=history_overrides,
    )
    seconds = int(spec.seconds)
    rows_f = seconds / interval
    rows = int(round(rows_f))
    if abs(rows_f - rows) > 1e-9 or rows <= 0:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Timeframe {timeframe} ({seconds}s) is not an exact multiple of "
            f"sample_interval_sec={interval}."
        )
    return rows


def _parse_features(params: dict[str, Any]) -> list[str]:
    raw = params.get("features")
    if raw is None:
        raw = params.get("feature")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    out: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _parse_timeframes(
    params: dict[str, Any],
    *,
    sample_interval_sec: float,
) -> list[str]:
    raw = params.get("timeframes")
    if raw is None:
        raw = params.get("timeframe")
    if raw is None:
        return list(available_ohlc_timeframes(sample_interval_sec))
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        key = normalize_ohlc_timeframe(
            str(item), sample_interval_sec=sample_interval_sec
        )
        if key not in out:
            out.append(key)
    return out


def _parse_fields(params: dict[str, Any]) -> list[str]:
    raw = params.get("outputs")
    if raw is None:
        raw = params.get("fields")
    if raw is None:
        return list(DEFAULT_OHLC_FIELDS)
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        fld = normalize_ohlc_field(str(item))
        if fld not in out:
            out.append(fld)
    return out or list(DEFAULT_OHLC_FIELDS)


def _candle_ohlc_arrays(values: np.ndarray, period_rows: int) -> tuple[np.ndarray, ...]:
    """Compute OHLC for each fully completed candle (vectorized)."""
    n = int(values.shape[0])
    n_candles = n // period_rows
    if n_candles <= 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty
    block = values[: n_candles * period_rows].reshape(n_candles, period_rows)
    opens = block[:, 0].astype(float, copy=False)
    closes = block[:, -1].astype(float, copy=False)
    with np.errstate(all="ignore"):
        highs = np.nanmax(block, axis=1)
        lows = np.nanmin(block, axis=1)
    return opens, highs, lows, closes


def build_ohlc_history_columns(
    values: np.ndarray | pd.Series,
    *,
    period_rows: int,
    history_len: int,
    fields: list[str],
) -> dict[tuple[int, str], np.ndarray]:
    """Map each row to completed-candle history (1 = newest). Partials → NaN."""
    vals = np.asarray(values, dtype=float)
    n = int(vals.shape[0])
    opens, highs, lows, closes = _candle_ohlc_arrays(vals, period_rows)
    field_map = {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    }
    idx = np.arange(n, dtype=np.int64)
    completed = (idx + 1) // int(period_rows)
    out: dict[tuple[int, str], np.ndarray] = {}
    for h in range(1, int(history_len) + 1):
        cidx = completed - h
        valid = cidx >= 0
        for fld in fields:
            col = np.full(n, np.nan, dtype=float)
            src = field_map[fld]
            if src.size and bool(valid.any()):
                col[valid] = src[cidx[valid]]
            out[(h, fld)] = col
    return out


def _apply_feature_timeframe(
    series: pd.Series,
    *,
    period_rows: int,
    history_len: int,
    fields: list[str],
) -> dict[tuple[int, str], np.ndarray]:
    return build_ohlc_history_columns(
        series.to_numpy(dtype=float, copy=False),
        period_rows=period_rows,
        history_len=history_len,
        fields=fields,
    )


@register_transformation
class OhlcAggregationTransformation(FeatureTransformation):
    """Completed OHLC candle history (pipeline engineering only)."""

    id = "ohlc_aggregation"
    name = "OHLC Aggregation"
    order = 44  # After Exponential Rolling (42); before Interaction (50)
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def describe(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream=None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ):
        from .describe import MASTER_STAGE_ID, OutputDescriptor, make_stage_descriptor
        from .ohlc_history_profiles import resolve_timeframe_spec

        del upstream, master_features
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        interval = float(sample_interval_sec) if sample_interval_sec is not None else float(
            params.get("sample_interval_sec") or 3.0
        )
        outputs: list[OutputDescriptor] = []
        try:
            features = _parse_features(params)
            fields = _parse_fields(params)
            timeframes = _parse_timeframes(params, sample_interval_sec=interval)
            for feat in features:
                for tf_raw in timeframes:
                    try:
                        spec = resolve_timeframe_spec(interval, tf_raw)
                    except Exception:
                        continue
                    for h in range(1, int(spec.history) + 1):
                        for fld in fields:
                            outputs.append(
                                OutputDescriptor(
                                    name=ohlc_aggregation_column_name(
                                        feat,
                                        spec.key,
                                        h,
                                        fld,
                                        sample_interval_sec=interval,
                                        history_len=spec.history,
                                    ),
                                    kind="ohlc_aggregation",
                                    source_feature=feat,
                                    op=str(fld),
                                    meta={"timeframe": spec.key, "history_index": h},
                                )
                            )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=[MASTER_STAGE_ID],
            notes="Planned OHLC history columns from features × timeframes × fields.",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(
            self.id, getattr(self, "params", None), context.config
        )
        features = _parse_features(params)
        fields = _parse_fields(params)
        interval = resolve_sample_interval(context, params)
        if interval <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                "sample_interval_sec is missing or invalid."
            )
        overrides = parse_history_overrides(params)
        timeframes = _parse_timeframes(params, sample_interval_sec=interval)
        if not features or not timeframes or not fields:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                "Requires non-empty features, timeframes, and outputs."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )

        specs: dict[str, OhlcTimeframeSpec] = {
            tf: resolve_timeframe_spec(
                interval, tf, history_overrides=overrides
            )
            for tf in timeframes
        }
        tf_periods: dict[str, int] = {
            tf: period_rows_for_timeframe(
                tf, interval, history_overrides=overrides
            )
            for tf in timeframes
        }

        planned_names: list[str] = []
        seen: set[str] = set()
        for feat in features:
            for tf in timeframes:
                hist = specs[tf].history
                for h in range(1, hist + 1):
                    for fld in fields:
                        col = ohlc_aggregation_column_name(
                            feat,
                            tf,
                            h,
                            fld,
                            sample_interval_sec=interval,
                            history_len=hist,
                        )
                        if col in seen:
                            raise LagConfigError(
                                "OHLC Aggregation Transformation\n"
                                f"Duplicate output name: {col}"
                            )
                        if col in df.columns:
                            raise LagConfigError(
                                "OHLC Aggregation Transformation\n"
                                f"Output column already exists: {col}"
                            )
                        seen.add(col)
                        planned_names.append(col)

        persist_sample_interval(context, self.id, interval)
        tf_meta = timeframe_specs_metadata(
            interval, timeframes, history_overrides=overrides
        )
        _persist_timeframe_specs(context, self.id, tf_meta)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        plans: list[dict[str, Any]] = []
        created: list[str] = []
        for feat, tf in ((f, t) for f in features for t in timeframes):
            period = tf_periods[tf]
            hist_len = specs[tf].history

            def _name_fn(
                h: int,
                fld: str,
                *,
                _feat=feat,
                _tf=tf,
                _hist=hist_len,
            ) -> str:
                return ohlc_aggregation_column_name(
                    _feat,
                    _tf,
                    h,
                    fld,
                    sample_interval_sec=interval,
                    history_len=_hist,
                )

            for h in range(1, hist_len + 1):
                for fld in fields:
                    created.append(_name_fn(h, fld))
            plans.append(
                {
                    "feature": feat,
                    "period_rows": period,
                    "history_len": hist_len,
                    "fields": list(fields),
                    "name_fn": _name_fn,
                }
            )

        total = len(plans)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for plan in plans:
                feat = plan["feature"]
                period = int(plan["period_rows"])
                hist_len = int(plan["history_len"])
                if partition_by:
                    grouped = local.groupby(partition_by, sort=False, group_keys=False)
                    for h in range(1, hist_len + 1):
                        for fld in fields:
                            local[plan["name_fn"](h, fld)] = np.nan
                    for _, gdf in grouped:
                        hist_map = _apply_feature_timeframe(
                            gdf[feat],
                            period_rows=period,
                            history_len=hist_len,
                            fields=fields,
                        )
                        positions = gdf.index
                        for h in range(1, hist_len + 1):
                            for fld in fields:
                                name = plan["name_fn"](h, fld)
                                local.loc[positions, name] = hist_map[(h, fld)]
                else:
                    hist_map = _apply_feature_timeframe(
                        local[feat],
                        period_rows=period,
                        history_len=hist_len,
                        fields=fields,
                    )
                    for h in range(1, hist_len + 1):
                        for fld in fields:
                            local[plan["name_fn"](h, fld)] = hist_map[(h, fld)]
            return local

        from .polars_ops import apply_ohlc_history_via_polars

        out = apply_ohlc_history_via_polars(
            df,
            plans=plans,
            partition_by=partition_by,
            build_history_columns=build_ohlc_history_columns,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(
            f"OHLC Aggregation: {total}/{total} feature×timeframe",
            total,
            total,
        )

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        profile = get_ohlc_interval_profile(interval)
        approx_msgs = [
            (
                f"{spec.timeframe_label}: requested {spec.nominal_duration_sec}s → "
                f"actual {spec.actual_duration_sec}s "
                f"({spec.sample_count(interval)} samples)"
            )
            for spec in (specs[tf] for tf in timeframes)
            if spec.is_approximate
        ]
        context.log("OHLC Aggregation Transformation")
        context.log(f"Features Selected : {len(features)}")
        context.log(f"Sample Interval   : {interval}")
        context.log(f"History Profile   : {interval}s → {list(profile.timeframes)}")
        context.log(f"Timeframes        : {timeframes}")
        context.log(
            "History Depths    : "
            + ", ".join(f"{tf}={specs[tf].history}" for tf in timeframes)
        )
        context.log(
            "Duration Metadata : "
            + ", ".join(
                f"{m['timeframe_label']}="
                f"{m['actual_duration_sec']}s×{m['sample_count']}"
                + ("~" if m["is_approximate"] else "")
                for m in tf_meta
            )
        )
        for msg in approx_msgs:
            context.log(f"Approximate TF    : {msg}")
        context.log(f"Outputs           : {fields}")
        context.log(f"Columns Created   : {len(created)}")
        context.log(f"Elapsed           : {elapsed:.2f} s")
        _ = planned_names
        messages = [
            f"Features Selected : {len(features)}",
            f"Timeframes : {timeframes}",
            f"History : "
            + ", ".join(f"{tf}×{specs[tf].history}" for tf in timeframes),
            f"Outputs : {fields}",
            f"Columns Created : {len(created)}",
            f"sample_interval_sec={interval}",
            f"partition_by={partition_by or []}",
            f"timeframe_specs={tf_meta}",
        ]
        messages.extend(f"Approximate : {m}" for m in approx_msgs)
        messages.append("frame_backend=polars_ohlc_history")
        return TransformationResult(
            frame=out,
            created_columns=created,
            elapsed_sec=elapsed,
            rows_processed=int(len(out)),
            transformation_id=self.id,
            transformation_name=self.name,
            messages=messages,
        )


def _persist_timeframe_specs(
    context: Any,
    transform_id: str,
    timeframe_specs: list[dict[str, Any]],
) -> None:
    """Stamp resolved timeframe metadata into pipeline config for dataset exports."""
    for entry in (getattr(context, "config", None) or {}).get("transformations") or []:
        if not isinstance(entry, dict) or str(entry.get("id") or "") != transform_id:
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            params = {}
            entry["params"] = params
        params["timeframe_specs"] = list(timeframe_specs)
        break


# Populate legacy module exports used by UI/tests.
_ensure_legacy_exports()
