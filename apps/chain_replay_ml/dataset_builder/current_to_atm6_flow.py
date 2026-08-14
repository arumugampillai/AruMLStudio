"""Current-strike to ATM+6 wing flow feature — volume/OI participation toward ATM."""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml.features_atm_band import compute_pct_change
from chain_replay_ml.ticks import TickTimeline

LOOKBACK_SEC = 60.0
NUM_STRIKES = 7

CURRENT_TO_ATM6_FLOW_FEATURE = "current_to_atm6_flow_delta_ltp_to_spot_ratio"

CURRENT_TO_ATM6_FLOW_FEATURES: frozenset[str] = frozenset({CURRENT_TO_ATM6_FLOW_FEATURE})


def active_current_to_atm6_flow_features(active: Iterable[str] | None) -> frozenset[str]:
    if not active:
        return CURRENT_TO_ATM6_FLOW_FEATURES
    wanted = frozenset(active)
    return CURRENT_TO_ATM6_FLOW_FEATURES & wanted


def needs_current_to_atm6_flow(active: Iterable[str] | None) -> bool:
    return bool(active_current_to_atm6_flow_features(active))


def strikes_toward_atm(
    current_strike: float,
    *,
    step: int,
    option_type: str,
) -> list[float]:
    """Seven strikes from current toward ATM: CE → lower, PE → higher."""
    if step <= 0:
        return []
    direction = -1 if str(option_type).upper() == "CE" else 1
    return [float(current_strike) + direction * i * float(step) for i in range(NUM_STRIKES)]


def _pct_change_at(
    timeline: TickTimeline,
    ts: float,
    *,
    attr: str,
    lookback_sec: float,
) -> float | None:
    if attr == "volume":
        cur = timeline.volume_at(ts)
        past = timeline.volume_at(ts - lookback_sec)
    elif attr == "oi":
        cur = timeline.oi_at(ts)
        past = timeline.oi_at(ts - lookback_sec)
    else:
        return None
    return compute_pct_change(
        float(cur) if cur is not None else None,
        float(past) if past is not None else None,
    )


def _avg_change_pct_across_strikes(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    ts: float,
    *,
    current_strike: float,
    step: int,
    option_type: str,
    attr: str,
    lookback_sec: float = LOOKBACK_SEC,
) -> float | None:
    values: list[float] = []
    opt = str(option_type).upper()
    for strike in strikes_toward_atm(current_strike, step=step, option_type=opt):
        entry = strike_mapping.get((strike, opt))
        if not entry:
            return None
        _, _, tl = entry
        pct = _pct_change_at(tl, ts, attr=attr, lookback_sec=lookback_sec)
        if pct is None:
            return None
        values.append(float(pct))
    if len(values) != NUM_STRIKES:
        return None
    return sum(values) / float(len(values))


def compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
    *,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    ts: float,
    current_strike: float,
    step: int,
    option_type: str,
    delta: float | None,
    ltp: float | None,
    spot: float | None,
    lookback_sec: float = LOOKBACK_SEC,
) -> float | None:
    """Flow strength (avg vol% + avg OI% over 7 strikes toward ATM) × |delta| × LTP / spot."""
    if step <= 0 or delta is None or ltp is None or ltp <= 0 or spot is None or spot <= 0:
        return None

    delta_abs = abs(float(delta))

    vol_avg = _avg_change_pct_across_strikes(
        strike_mapping,
        ts,
        current_strike=current_strike,
        step=step,
        option_type=option_type,
        attr="volume",
        lookback_sec=lookback_sec,
    )
    oi_avg = _avg_change_pct_across_strikes(
        strike_mapping,
        ts,
        current_strike=current_strike,
        step=step,
        option_type=option_type,
        attr="oi",
        lookback_sec=lookback_sec,
    )
    if vol_avg is None or oi_avg is None:
        return None

    flow_strength = (vol_avg + oi_avg) / 2.0
    return float(flow_strength * delta_abs * float(ltp) / float(spot))


def enrich_current_to_atm6_flow_features(
    raw: dict[str, Any],
    *,
    ts: float,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    strike_rupees: float,
    strike_step: int,
    option_type: str,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_current_to_atm6_flow_features(active_features)
    out = dict(raw)
    if CURRENT_TO_ATM6_FLOW_FEATURE not in wanted:
        return out

    out[CURRENT_TO_ATM6_FLOW_FEATURE] = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
        strike_mapping=strike_mapping,
        ts=ts,
        current_strike=float(strike_rupees),
        step=int(strike_step),
        option_type=str(option_type).upper(),
        delta=out.get("delta"),
        ltp=out.get("ltp"),
        spot=out.get("spot"),
    )
    return out


def enrich_current_to_atm6_flow_dataframe(
    df,
    *,
    chart_dir: str,
    market: str = "NIFTY",
    feature_grid_step_sec: int = 3,
    column: str = CURRENT_TO_ATM6_FLOW_FEATURE,
):
    """Batch-compute ``current_to_atm6_flow_delta_ltp_to_spot_ratio`` on an export frame."""
    import pandas as pd

    from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name

    from .chain_maps import precompute_chain_maps
    from .day_context import DayContext, SourceSpec, load_day_context

    out = df
    if column in out.columns:
        return out

    required = {"trading_day", "timestamp", "strike", "option_type", "delta", "ltp", "spot"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns for {column}: {sorted(missing)}"
        )

    index_key = normalize_index_name(market)
    strike_step = STRIKE_STEP.get(index_key, 50)
    ctx_cache: dict[str, DayContext | None] = {}
    col_values = pd.Series(index=out.index, dtype="float64")

    for day, day_df in out.groupby("trading_day", sort=False):
        day_str = str(day)
        if day_str not in ctx_cache:
            try:
                ctx_cache[day_str] = load_day_context(
                    chart_dir,
                    SourceSpec(
                        source_id=day_str,
                        trading_day=day_str,
                        market=str(market or "NIFTY").upper(),
                    ),
                    feature_grid_step_sec=int(feature_grid_step_sec),
                )
            except Exception:
                ctx_cache[day_str] = None
        ctx = ctx_cache.get(day_str)
        if ctx is None:
            continue

        for idx, row in day_df.iterrows():
            try:
                delta = row.get("delta")
                delta_f = float(delta) if delta is not None else None
            except (TypeError, ValueError):
                delta_f = None
            try:
                ltp_f = float(row.get("ltp")) if row.get("ltp") is not None else None
            except (TypeError, ValueError):
                ltp_f = None
            try:
                spot_f = float(row.get("spot")) if row.get("spot") is not None else None
            except (TypeError, ValueError):
                spot_f = None
            val = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
                strike_mapping=ctx.strike_mapping,
                ts=float(row["timestamp"]),
                current_strike=float(row["strike"]),
                step=strike_step,
                option_type=str(row["option_type"]),
                delta=delta_f,
                ltp=ltp_f,
                spot=spot_f,
            )
            col_values.at[idx] = val

    out = out.copy()
    out[column] = col_values
    return out
