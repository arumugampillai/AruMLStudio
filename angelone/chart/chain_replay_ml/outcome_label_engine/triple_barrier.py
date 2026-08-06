"""Triple Barrier labeling — Prediction Dataset sample-grid paths (Phase 3)."""

from __future__ import annotations

import bisect
from typing import Any

from .registry import register_strategy
from .types import (
    SOURCE_PREDICTION,
    LabelBatchResult,
    LabelSourceContext,
    LabelStrategyConfig,
    StrategyCapabilities,
    StrategyMetadata,
    TargetDefinitions,
    validate_config_against_schema,
)

STRATEGY_ID = "triple_barrier"
STRATEGY_VERSION = "1.1"

LABEL_TP = "TP"
LABEL_SL = "SL"
LABEL_TIME = "TIME"

BARRIER_PERCENTAGE = "percentage"
BARRIER_POINTS = "points"
BARRIER_ATR = "atr"
BARRIER_VOL_ADJUSTED = "vol_adjusted"

BARRIER_TYPES_ENABLED = frozenset({BARRIER_PERCENTAGE, BARRIER_POINTS})

LABEL_ENCODING: dict[str, int] = {
    LABEL_TP: 0,
    LABEL_SL: 1,
    LABEL_TIME: 2,
}

_METADATA = StrategyMetadata(
    strategy_id=STRATEGY_ID,
    version=STRATEGY_VERSION,
    display_name="Triple Barrier",
    description="First TP / SL / Timeout",
    category="Classification",
)

_CAPABILITIES = StrategyCapabilities(
    strategy_id=STRATEGY_ID,
    supported_sources=frozenset({SOURCE_PREDICTION}),
    supported_problem_types=frozenset({"binary_classification", "multiclass"}),
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mark_price(row: dict[str, Any]) -> float | None:
    """Live mark at this timestamp — never use actual/future horizon LTP."""
    return _num(row.get("ltp")) if _num(row.get("ltp")) is not None else _num(row.get("current_ltp"))


def _entry_price(row: dict[str, Any]) -> float | None:
    for key in ("entry_price", "current_ltp", "ltp"):
        v = _num(row.get(key))
        if v is not None:
            return v
    return None


def group_path_series(
    path_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], list[float]]]:
    """Group mark-path rows by (trading_day, token) with parallel timestamps."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(
        path_rows,
        key=lambda r: (_num(r.get("timestamp")) or 0.0, int(r.get("row_index") or 0)),
    ):
        key = (str(row.get("trading_day") or ""), str(row.get("token") or ""))
        by_key.setdefault(key, []).append(row)
    out: dict[tuple[str, str], tuple[list[dict[str, Any]], list[float]]] = {}
    for key, series in by_key.items():
        ts_list = [_num(r.get("timestamp")) or 0.0 for r in series]
        out[key] = (series, ts_list)
    return out


def forward_path_after_entry(
    series: list[dict[str, Any]],
    ts_list: list[float],
    entry_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Samples strictly after the entry row on the mark path."""
    if not series:
        return []
    pid = entry_row.get("prediction_id")
    if pid is not None:
        for i, r in enumerate(series):
            if r.get("prediction_id") == pid:
                return series[i + 1 :]
    entry_ts = _num(entry_row.get("timestamp"))
    if entry_ts is None:
        return []
    entry_ri = int(entry_row.get("row_index") or 0)
    idx = bisect.bisect_left(ts_list, entry_ts)
    while idx < len(series) and ts_list[idx] == entry_ts:
        if int(series[idx].get("row_index") or 0) <= entry_ri:
            idx += 1
            continue
        break
    return series[idx:]


def resolve_session_close_ts(
    *,
    trading_day: str | None,
    explicit_close_ts: float | None,
) -> float | None:
    if explicit_close_ts is not None:
        return float(explicit_close_ts)
    if not trading_day:
        return None
    try:
        from storage.chain_replay_export import ist_market_session_bounds

        _open_ts, close_ts = ist_market_session_bounds(str(trading_day))
        return float(close_ts)
    except Exception:
        return None


def _invalid_row(
    sample: dict[str, Any],
    *,
    reason: str,
    entry_price: float | None = None,
    tp_price: float | None = None,
    sl_price: float | None = None,
) -> dict[str, Any]:
    out = {
        "prediction_id": sample.get("prediction_id"),
        "trading_day": sample.get("trading_day"),
        "token": sample.get("token"),
        "timestamp": sample.get("timestamp"),
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "exit_price": None,
        "exit_timestamp": None,
        "holding_seconds": None,
        "exit_reason": None,
        "realized_return": None,
        "label_name": None,
        "label_id": None,
        "is_valid": False,
        "invalid_reason": reason,
    }
    return out


def _valid_row(
    sample: dict[str, Any],
    *,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    exit_price: float,
    exit_timestamp: float,
    holding_seconds: float,
    exit_reason: str,
    label_name: str,
) -> dict[str, Any]:
    return {
        "prediction_id": sample.get("prediction_id"),
        "trading_day": sample.get("trading_day"),
        "token": sample.get("token"),
        "timestamp": sample.get("timestamp"),
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "exit_price": exit_price,
        "exit_timestamp": exit_timestamp,
        "holding_seconds": holding_seconds,
        "exit_reason": exit_reason,
        "realized_return": (exit_price - entry_price) / entry_price if entry_price else None,
        "label_name": label_name,
        "label_id": LABEL_ENCODING[label_name],
        "is_valid": True,
        "invalid_reason": None,
    }


def resolve_barrier_prices(
    entry_price: float,
    *,
    barrier_type: str,
    tp_value: float,
    sl_value: float,
) -> tuple[float, float]:
    """Map schema params → absolute TP/SL prices (long).

    percentage: TP = entry * (1 + tp%/100), SL = entry * (1 - sl%/100)
    points:     TP = entry + tp, SL = entry - sl
    """
    kind = str(barrier_type or BARRIER_POINTS).strip().lower()
    tp = float(tp_value)
    sl = float(sl_value)
    if kind == BARRIER_PERCENTAGE:
        return entry_price * (1.0 + tp / 100.0), entry_price * (1.0 - sl / 100.0)
    if kind == BARRIER_POINTS:
        return entry_price + tp, entry_price - sl
    raise ValueError(f"barrier_type {barrier_type!r} is not implemented yet")


def label_triple_barrier_sample(
    sample: dict[str, Any],
    *,
    forward_rows: list[dict[str, Any]] | None = None,
    holding_seconds: float,
    barrier_type: str = BARRIER_POINTS,
    tp_value: float | None = None,
    sl_value: float | None = None,
    tp_points: float | None = None,
    sl_points: float | None = None,
    truncate_at_close: bool = True,
    session_close_ts: float | None = None,
    max_path_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Label one entry from a discrete forward mark path (no tick DB).

    Barrier mode comes from ``barrier_type`` (percentage | points; atr/vol later).
    Legacy ``tp_points`` / ``sl_points`` map to points mode when ``tp_value`` omitted.
    First observed sample that breaches TP or SL wins; else TIME at hold/close.
    """
    entry_ts = _num(sample.get("timestamp"))
    entry_price = _entry_price(sample)
    if entry_ts is None:
        return _invalid_row(sample, reason="missing_entry_timestamp")
    if entry_price is None or entry_price <= 0:
        return _invalid_row(sample, reason="missing_entry_price", entry_price=entry_price)

    kind = str(barrier_type or BARRIER_POINTS).strip().lower()
    # Legacy aliases keep direct callers / older tests working.
    resolved_tp = tp_value if tp_value is not None else tp_points
    resolved_sl = sl_value if sl_value is not None else sl_points
    if resolved_tp is None or resolved_sl is None:
        return _invalid_row(
            sample,
            reason="missing_barrier_values",
            entry_price=entry_price,
        )
    if kind not in BARRIER_TYPES_ENABLED:
        return _invalid_row(
            sample,
            reason=f"barrier_type_unimplemented:{kind}",
            entry_price=entry_price,
        )
    try:
        tp_price, sl_price = resolve_barrier_prices(
            entry_price,
            barrier_type=kind,
            tp_value=float(resolved_tp),
            sl_value=float(resolved_sl),
        )
    except (TypeError, ValueError):
        return _invalid_row(
            sample,
            reason="invalid_barrier_values",
            entry_price=entry_price,
        )

    close_ts = resolve_session_close_ts(
        trading_day=str(sample.get("trading_day") or "") or None,
        explicit_close_ts=_num(sample.get("session_close_ts"))
        if session_close_ts is None
        else session_close_ts,
    )
    if truncate_at_close and close_ts is not None and entry_ts >= close_ts:
        return _invalid_row(
            sample,
            reason="session_closed_before_entry",
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )

    hold_end = entry_ts + float(holding_seconds)
    if truncate_at_close and close_ts is not None:
        hold_end = min(hold_end, close_ts)

    # Prefer explicit path on the sample; else use provided forward_rows.
    path = sample.get("path")
    if path is not None:
        forward = list(path)
    elif forward_rows is not None:
        forward = list(forward_rows)
    else:
        forward = []

    # Restrict to marks at/before hold_end (observed resolution only).
    window: list[tuple[float, float]] = []
    prev_ts: float | None = entry_ts
    for row in forward:
        ts = _num(row.get("timestamp"))
        mark = _mark_price(row) if isinstance(row, dict) else None
        if isinstance(row, dict) and mark is None and "ltp" in row:
            mark = _num(row.get("ltp"))
        if ts is None or mark is None:
            continue
        if ts <= entry_ts:
            continue
        if ts > hold_end:
            break
        if max_path_gap_sec is not None and prev_ts is not None:
            if (ts - prev_ts) > float(max_path_gap_sec):
                return _invalid_row(
                    sample,
                    reason="stale_path",
                    entry_price=entry_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                )
        window.append((ts, float(mark)))
        prev_ts = ts

    if not window:
        return _invalid_row(
            sample,
            reason="empty_path",
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )

    for ts, mark in window:
        if mark >= tp_price:
            return _valid_row(
                sample,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                exit_price=mark,
                exit_timestamp=ts,
                holding_seconds=ts - entry_ts,
                exit_reason=LABEL_TP,
                label_name=LABEL_TP,
            )
        if mark <= sl_price:
            return _valid_row(
                sample,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                exit_price=mark,
                exit_timestamp=ts,
                holding_seconds=ts - entry_ts,
                exit_reason=LABEL_SL,
                label_name=LABEL_SL,
            )

    # TIME: last observed mark within the effective holding window.
    exit_ts, exit_px = window[-1]
    return _valid_row(
        sample,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        exit_price=exit_px,
        exit_timestamp=exit_ts,
        holding_seconds=exit_ts - entry_ts,
        exit_reason=LABEL_TIME,
        label_name=LABEL_TIME,
    )


class TripleBarrierStrategy:
    """Prediction-path Triple Barrier (configurable barrier type; ATR later)."""

    @staticmethod
    def normalize_config_params(params: dict[str, Any]) -> dict[str, Any]:
        """Map legacy ``tp_points`` / ``sl_points`` into schema fields."""
        raw = dict(params or {})
        if "tp_value" not in raw and "tp_points" in raw:
            raw["tp_value"] = raw.pop("tp_points")
            raw.setdefault("barrier_type", BARRIER_POINTS)
        if "sl_value" not in raw and "sl_points" in raw:
            raw["sl_value"] = raw.pop("sl_points")
            raw.setdefault("barrier_type", BARRIER_POINTS)
        return raw

    @property
    def metadata(self) -> StrategyMetadata:
        return _METADATA

    @property
    def capabilities(self) -> StrategyCapabilities:
        return _CAPABILITIES

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "barrier_type": {
                "type": "enum",
                "label": "Barrier Type",
                "default": BARRIER_PERCENTAGE,
                "choices": [
                    {"value": BARRIER_PERCENTAGE, "label": "Percentage"},
                    {"value": BARRIER_POINTS, "label": "Points"},
                    {
                        "value": BARRIER_ATR,
                        "label": "ATR (future)",
                        "enabled": False,
                    },
                    {
                        "value": BARRIER_VOL_ADJUSTED,
                        "label": "Volatility-adjusted (future)",
                        "enabled": False,
                    },
                ],
            },
            "tp_value": {
                "type": "float",
                "label": "TP",
                "default": 20.0,
                "help": "Percent when Barrier Type=Percentage; premium points when Points",
            },
            "sl_value": {
                "type": "float",
                "label": "SL",
                "default": 10.0,
                "help": "Percent when Barrier Type=Percentage; premium points when Points",
            },
            "holding_seconds": {"type": "int", "label": "Holding seconds", "default": 300},
            "truncate_at_close": {
                "type": "bool",
                "label": "Truncate at close",
                "default": True,
            },
            "max_path_gap_sec": {
                "type": "float",
                "label": "Max path gap sec",
                "default": None,
            },
        }

    def get_target_definitions(self) -> TargetDefinitions:
        return TargetDefinitions(
            primary_target="label_id",
            display_target="label_name",
            label_encoding=dict(LABEL_ENCODING),
        )

    def build_labels(
        self,
        source: LabelSourceContext,
        samples: Any,
        config: LabelStrategyConfig,
    ) -> LabelBatchResult:
        params = validate_config_against_schema(
            self.normalize_config_params(dict(config.params)),
            self.get_config_schema(),
        )
        holding_seconds = float(params["holding_seconds"])
        barrier_type = str(params["barrier_type"])
        tp_value = float(params["tp_value"])
        sl_value = float(params["sl_value"])
        truncate_at_close = bool(params.get("truncate_at_close", True))
        max_path_gap_sec = params.get("max_path_gap_sec")
        if max_path_gap_sec is not None:
            max_path_gap_sec = float(max_path_gap_sec)

        session_close_ts = _num((source.handles or {}).get("session_close_ts"))
        rows_in = list(samples or [])

        # If samples already carry explicit paths, label independently.
        # Otherwise treat the day cohort as a sample-grid and build forward paths.
        has_explicit_paths = any(
            isinstance(r, dict) and r.get("path") is not None for r in rows_in
        )
        path_by_key: dict[tuple[str, str], tuple[list[dict[str, Any]], list[float]]] | None
        if has_explicit_paths:
            path_by_key = None
        else:
            path_by_key = group_path_series(rows_in)

        out_rows: list[dict[str, Any]] = []
        for sample in rows_in:
            if not isinstance(sample, dict):
                continue
            # Prefer per-sample session_close_ts; else run/source handle.
            sample_close = _num(sample.get("session_close_ts"))
            close_for_row = (
                sample_close if sample_close is not None else session_close_ts
            )
            forward: list[dict[str, Any]] | None = None
            if path_by_key is not None:
                key = (
                    str(sample.get("trading_day") or ""),
                    str(sample.get("token") or ""),
                )
                series_ts = path_by_key.get(key)
                if series_ts is not None:
                    series, ts_list = series_ts
                    forward = forward_path_after_entry(series, ts_list, sample)
                else:
                    forward = []
            labeled = label_triple_barrier_sample(
                sample,
                forward_rows=forward,
                holding_seconds=holding_seconds,
                barrier_type=barrier_type,
                tp_value=tp_value,
                sl_value=sl_value,
                truncate_at_close=truncate_at_close,
                session_close_ts=close_for_row,
                max_path_gap_sec=max_path_gap_sec,
            )
            out_rows.append(labeled)

        defs = self.get_target_definitions()
        return LabelBatchResult(
            rows=out_rows,
            target_columns=["label_id", "label_name"],
            target_definitions=defs,
            metadata={
                "strategy": STRATEGY_ID,
                "day": source.day,
                "source_kind": source.source_kind,
                "barrier_type": barrier_type,
                "tp_value": tp_value,
                "sl_value": sl_value,
                "holding_seconds": holding_seconds,
                "truncate_at_close": truncate_at_close,
            },
        )


_TRIPLE_BARRIER = TripleBarrierStrategy()


def get_triple_barrier_strategy() -> TripleBarrierStrategy:
    return _TRIPLE_BARRIER


def register_triple_barrier_strategy(*, replace: bool = True) -> TripleBarrierStrategy:
    register_strategy(_TRIPLE_BARRIER, replace=replace)
    return _TRIPLE_BARRIER
