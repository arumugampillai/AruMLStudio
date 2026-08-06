"""Delta-range strike selection: live preview, build stats, and dataset audit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name
from chain_replay_ml.features_atm_band import (
    compute_delta_at_ts,
    delta_matches_selection,
    find_atm_strike,
)

from .day_context import DayContext, SourceSpec, load_day_context
from .stages import LOOKBACK_START_SEC, build_sample_timestamps


def _chart_dir_from_data(data_dir: str) -> str:
    return os.path.dirname(data_dir) if os.path.basename(data_dir) == "data" else data_dir


def normalize_delta_bounds(delta_min: float, delta_max: float) -> tuple[float, float]:
    lo = float(delta_min)
    hi = float(delta_max)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def expected_rule_text(delta_type: str, delta_min: float, delta_max: float) -> str:
    lo, hi = normalize_delta_bounds(delta_min, delta_max)
    dt = str(delta_type or "absolute").lower()
    if dt == "ce":
        return f"{lo:.2f} ≤ CE Delta ≤ {hi:.2f}"
    if dt == "pe":
        return f"{(-hi):.2f} ≤ PE Delta ≤ {(-lo):.2f}"
    return f"{lo:.2f} ≤ |Delta| ≤ {hi:.2f}"


def rejection_rules_text(delta_type: str, delta_min: float, delta_max: float) -> list[str]:
    lo, hi = normalize_delta_bounds(delta_min, delta_max)
    dt = str(delta_type or "absolute").lower()
    if dt == "ce":
        return [f"CE Delta < {lo:.2f}", f"CE Delta > {hi:.2f}"]
    if dt == "pe":
        return [f"PE Delta < {(-hi):.2f}", f"PE Delta > {(-lo):.2f}"]
    return [f"|Delta| < {lo:.2f}", f"|Delta| > {hi:.2f}"]


def audit_delta_value(
    delta: float,
    option_type: str,
    *,
    delta_type: str,
    delta_min: float,
    delta_max: float,
) -> bool:
    return delta_matches_selection(
        float(delta),
        option_type,
        delta_type=delta_type,
        delta_min=delta_min,
        delta_max=delta_max,
    )


def _strike_selection_params(strike_selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_type": str(strike_selection.get("deltaType") or "absolute").lower(),
        "delta_min": float(strike_selection.get("deltaMin") or 0.15),
        "delta_max": float(strike_selection.get("deltaMax") or 0.50),
    }


def preview_rows_at_ts(
    ctx: DayContext,
    ts: float,
    strike_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    """All chain strikes at ``ts`` with delta and selected flag."""
    params = _strike_selection_params(strike_selection)
    delta_type = params["delta_type"]
    dmin = params["delta_min"]
    dmax = params["delta_max"]
    index_key = normalize_index_name(ctx.source.market)
    step = STRIKE_STEP.get(index_key, 50)
    spot = ctx.index_tl.ltp_rupees_at(ts)
    atm = find_atm_strike(float(spot or 0), step) if spot else None

    rows: list[dict[str, Any]] = []
    for (strike_r, opt_type), (_tok, _sym, opt_tl) in sorted(
        ctx.strike_mapping.items(),
        key=lambda kv: (kv[0][0], kv[0][1]),
    ):
        if delta_type == "ce" and opt_type != "CE":
            continue
        if delta_type == "pe" and opt_type != "PE":
            continue
        delta = compute_delta_at_ts(
            ts=ts,
            index_timeline=ctx.index_tl,
            option_timeline=opt_tl,
            option_type=opt_type,
            strike_rupees=strike_r,
            expiry_ts=ctx.expiry_ts,
        )
        selected = (
            delta is not None
            and audit_delta_value(
                delta,
                opt_type,
                delta_type=delta_type,
                delta_min=dmin,
                delta_max=dmax,
            )
        )
        strike_label = f"{int(round(strike_r))} {opt_type}"
        rows.append({
            "strike": float(strike_r),
            "option_type": opt_type,
            "strike_label": strike_label,
            "delta": round(float(delta), 4) if delta is not None else None,
            "selected": bool(selected),
            "distance_from_atm": int(round(strike_r)) - int(atm) if atm is not None else None,
        })
    return rows


def build_delta_range_preview(
    data_dir: str,
    *,
    strike_selection: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one replay day and return strike/delta preview table."""
    chart_dir = _chart_dir_from_data(data_dir)
    if not source or not source.get("trading_day"):
        raise ValueError("A dataset source (trading_day, market, expiry) is required for preview")

    spec = SourceSpec(
        source_id=str(source.get("source_id") or ""),
        trading_day=str(source["trading_day"]),
        market=str(source.get("market") or "NIFTY"),
        expiry=str(source.get("expiry") or ""),
        date_label=str(source.get("date") or ""),
    )
    ctx = load_day_context(chart_dir, spec)
    timestamps = build_sample_timestamps(ctx, step_sec=10, max_horizon_sec=0)
    ts = timestamps[len(timestamps) // 2] if timestamps else (ctx.open_ts + LOOKBACK_START_SEC + 600.0)
    spot = ctx.index_tl.ltp_rupees_at(ts)

    params = _strike_selection_params(strike_selection)
    rows = preview_rows_at_ts(ctx, ts, strike_selection)
    selected_n = sum(1 for r in rows if r["selected"])

    return {
        "trading_day": spec.trading_day,
        "market": spec.market,
        "expiry": spec.expiry,
        "preview_ts": float(ts),
        "spot": float(spot) if spot is not None else None,
        "delta_type": params["delta_type"],
        "delta_min": params["delta_min"],
        "delta_max": params["delta_max"],
        "expected_rule": expected_rule_text(params["delta_type"], params["delta_min"], params["delta_max"]),
        "rejection_rules": rejection_rules_text(params["delta_type"], params["delta_min"], params["delta_max"]),
        "estimated_selected_min": max(0, selected_n - 4),
        "estimated_selected_max": selected_n + 4,
        "selected_at_preview": selected_n,
        "rows": rows,
    }


@dataclass
class DeltaRangeBuildStats:
    rows_processed: int = 0
    rows_selected: int = 0
    min_delta_seen: float | None = None
    max_delta_seen: float | None = None
    strikes_per_sample: list[int] = field(default_factory=list)

    def record_candidate(self) -> None:
        self.rows_processed += 1

    def record_selected(self, delta: float | None) -> None:
        self.rows_selected += 1
        if delta is None:
            return
        val = float(delta)
        if self.min_delta_seen is None or val < self.min_delta_seen:
            self.min_delta_seen = val
        if self.max_delta_seen is None or val > self.max_delta_seen:
            self.max_delta_seen = val

    def record_timestamp(self, selected_count: int) -> None:
        self.strikes_per_sample.append(int(selected_count))

    def merge(self, other: DeltaRangeBuildStats) -> None:
        self.rows_processed += other.rows_processed
        self.rows_selected += other.rows_selected
        self.strikes_per_sample.extend(other.strikes_per_sample)
        if other.min_delta_seen is not None:
            if self.min_delta_seen is None or other.min_delta_seen < self.min_delta_seen:
                self.min_delta_seen = other.min_delta_seen
        if other.max_delta_seen is not None:
            if self.max_delta_seen is None or other.max_delta_seen > self.max_delta_seen:
                self.max_delta_seen = other.max_delta_seen

    def to_dict(self, *, strike_selection: dict[str, Any] | None = None) -> dict[str, Any]:
        sample_count = len(self.strikes_per_sample)
        strikes_sum = sum(self.strikes_per_sample)
        avg_strikes = round(strikes_sum / sample_count, 1) if sample_count else 0.0
        out: dict[str, Any] = {
            "selection_mode": "Delta Range",
            "rows_processed": int(self.rows_processed),
            "rows_selected": int(self.rows_selected),
            "average_strikes_per_sample": avg_strikes,
            "sample_count": int(sample_count),
            "strikes_per_sample_sum": int(strikes_sum),
            "min_delta_seen": self.min_delta_seen,
            "max_delta_seen": self.max_delta_seen,
        }
        if strike_selection:
            params = _strike_selection_params(strike_selection)
            out.update({
                "delta_type": params["delta_type"],
                "delta_min": params["delta_min"],
                "delta_max": params["delta_max"],
                "expected_rule": expected_rule_text(
                    params["delta_type"], params["delta_min"], params["delta_max"],
                ),
            })
        return out


def merge_delta_range_stats_dict(
    acc: dict[str, Any] | None,
    part: dict[str, Any],
) -> dict[str, Any]:
    if not acc:
        return dict(part)
    merged = dict(acc)
    merged["rows_processed"] = int(acc.get("rows_processed") or 0) + int(part.get("rows_processed") or 0)
    merged["rows_selected"] = int(acc.get("rows_selected") or 0) + int(part.get("rows_selected") or 0)
    merged["sample_count"] = int(acc.get("sample_count") or 0) + int(part.get("sample_count") or 0)
    merged["strikes_per_sample_sum"] = int(acc.get("strikes_per_sample_sum") or 0) + int(
        part.get("strikes_per_sample_sum") or 0
    )
    sc = merged["sample_count"]
    merged["average_strikes_per_sample"] = (
        round(merged["strikes_per_sample_sum"] / sc, 1) if sc else 0.0
    )
    for key in ("min_delta_seen", "max_delta_seen"):
        a = acc.get(key)
        b = part.get(key)
        if a is None:
            merged[key] = b
        elif b is None:
            merged[key] = a
        elif key == "min_delta_seen":
            merged[key] = min(float(a), float(b))
        else:
            merged[key] = max(float(a), float(b))
    return merged


def collect_delta_candidates_for_timestamp(
    *,
    ts: float,
    ctx: DayContext,
    strike_selection: dict[str, Any],
    stats: DeltaRangeBuildStats,
) -> list[tuple[float, str, str, str, Any, int]]:
    """Mirror ``select_option_entries_for_timestamp`` delta_range path with stats."""
    from chain_replay_ml.features_atm_band import find_atm_strike, select_option_entries_for_timestamp

    params = _strike_selection_params(strike_selection)
    delta_type = params["delta_type"]
    dmin = params["delta_min"]
    dmax = params["delta_max"]
    index_key = normalize_index_name(ctx.source.market)
    step = STRIKE_STEP.get(index_key, 50)
    spot = ctx.index_tl.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return []
    atm = find_atm_strike(float(spot), step)

    for (strike_r, opt_type), (_tok, _sym, opt_tl) in ctx.strike_mapping.items():
        if delta_type == "ce" and opt_type != "CE":
            continue
        if delta_type == "pe" and opt_type != "PE":
            continue
        stats.record_candidate()
        delta = compute_delta_at_ts(
            ts=ts,
            index_timeline=ctx.index_tl,
            option_timeline=opt_tl,
            option_type=opt_type,
            strike_rupees=strike_r,
            expiry_ts=ctx.expiry_ts,
        )
        if delta is not None and audit_delta_value(
            delta, opt_type, delta_type=delta_type, delta_min=dmin, delta_max=dmax,
        ):
            stats.record_selected(delta)

    entries = select_option_entries_for_timestamp(
        ts=ts,
        spot=float(spot),
        strike_step=step,
        index_timeline=ctx.index_tl,
        strike_mapping=ctx.strike_mapping,
        expiry_ts=ctx.expiry_ts,
        strike_selection=strike_selection,
    )
    stats.record_timestamp(len(entries))
    return entries


def audit_delta_range_dataset(
    df: pd.DataFrame,
    strike_meta: dict[str, Any],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Validate every row's delta lies in the configured band."""
    mode = str(strike_meta.get("mode") or "").upper()
    if mode != "DELTA_RANGE":
        return {"applicable": False}

    delta_type = str(strike_meta.get("delta_type") or "absolute").lower()
    dmin = float(strike_meta.get("delta_min") or 0.15)
    dmax = float(strike_meta.get("delta_max") or 0.50)
    expected = expected_rule_text(delta_type, dmin, dmax)

    if "delta" not in df.columns:
        return {
            "applicable": True,
            "status": "warn",
            "expected_rule": expected,
            "minimum_delta": None,
            "maximum_delta": None,
            "violations_count": 0,
            "violations": [],
            "message": "Delta column not present in dataset",
        }

    work = df[["delta", "strike", "option_type"]].copy()
    work["delta_num"] = pd.to_numeric(work["delta"], errors="coerce")
    valid = work["delta_num"].notna()
    min_delta = float(work.loc[valid, "delta_num"].min()) if valid.any() else None
    max_delta = float(work.loc[valid, "delta_num"].max()) if valid.any() else None

    violations: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        delta = row["delta_num"]
        if pd.isna(delta):
            continue
        opt = str(row.get("option_type") or "").upper()
        if audit_delta_value(
            float(delta), opt, delta_type=delta_type, delta_min=dmin, delta_max=dmax,
        ):
            continue
        strike = row.get("strike")
        try:
            strike_i = int(round(float(strike)))
        except (TypeError, ValueError):
            strike_i = strike
        violations.append({
            "strike_label": f"{strike_i} {opt}",
            "strike": strike,
            "option_type": opt,
            "delta": round(float(delta), 4),
        })
        if len(violations) >= sample_limit:
            break

    violations_count = 0
    if valid.any():
        mask = work.loc[valid].apply(
            lambda r: not audit_delta_value(
                float(r["delta_num"]),
                str(r.get("option_type") or ""),
                delta_type=delta_type,
                delta_min=dmin,
                delta_max=dmax,
            ),
            axis=1,
        )
        violations_count = int(mask.sum())

    status = "pass" if violations_count == 0 else "fail"
    return {
        "applicable": True,
        "status": status,
        "expected_rule": expected,
        "delta_type": delta_type,
        "delta_min": dmin,
        "delta_max": dmax,
        "minimum_delta": round(min_delta, 4) if min_delta is not None else None,
        "maximum_delta": round(max_delta, 4) if max_delta is not None else None,
        "violations_count": violations_count,
        "violations": violations,
    }
