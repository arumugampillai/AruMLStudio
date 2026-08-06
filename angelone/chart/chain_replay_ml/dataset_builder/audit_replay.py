"""Replay builder state to a dataset row for audit / RCA."""

from __future__ import annotations

from typing import Any

import pandas as pd

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name

from .audit_diagnostics import _horizon_sec_from_column
from .chain_maps import precompute_chain_maps
from .day_context import SourceSpec, load_day_context
from .extended_features import OptionFeatureState
from .formula_recalc import _recompute_row


def replay_row_context(
    df: pd.DataFrame,
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
    chart_dir: str,
    row: pd.Series,
) -> dict[str, Any]:
    """Replay token rows up to ``row`` and return builder state at that timestamp."""
    day = str(row["trading_day"])
    token = str(row["token"])
    target_ts = float(row["timestamp"])
    strike = float(row["strike"])
    opt_type = str(row["option_type"])

    day_info = next((d for d in (meta_doc.get("days") or []) if str(d.get("trading_day")) == day), None)
    if not day_info:
        raise ValueError(f"No day metadata for {day}")

    ctx = load_day_context(
        chart_dir,
        SourceSpec(
            source_id=str(day_info.get("source_id") or day),
            trading_day=day,
            market=str(day_info.get("market") or "NIFTY"),
            expiry=str(day_info.get("expiry") or ""),
        ),
    )
    index_key = normalize_index_name(ctx.source.market)
    strike_step = STRIKE_STEP.get(index_key, 50)
    enabled_groups = list(expected_doc.get("feature_groups") or [])
    target_columns = list(
        expected_doc.get("prediction_target_columns")
        or (expected_doc.get("expected") or {}).get("target_column_names")
        or []
    )
    horizons_sec = [_horizon_sec_from_column(c) for c in target_columns if _horizon_sec_from_column(c) > 0]

    all_day_ts = sorted(df.loc[df["trading_day"] == day, "timestamp"].unique())
    chain_maps = precompute_chain_maps(
        index_tl=ctx.index_tl,
        strike_mapping=ctx.strike_mapping,
        timestamps=all_day_ts,
        strike_step=strike_step,
    )

    entry = ctx.strike_mapping.get((strike, opt_type))
    if not entry:
        raise ValueError(f"No strike mapping for {strike} {opt_type}")
    _tok, _sym, opt_tl = entry

    opt_state = OptionFeatureState()
    token_df = df[(df["trading_day"] == day) & (df["token"] == token)].sort_values("timestamp")
    recomputed: dict[str, Any] = {}
    for _, r in token_df.iterrows():
        strike_r = float(r["strike"])
        otype = str(r["option_type"])
        ent = ctx.strike_mapping.get((strike_r, otype))
        if not ent:
            continue
        _t, _s, otl = ent
        ts = float(r["timestamp"])
        recomputed = _recompute_row(
            ctx=ctx,
            chain_maps=chain_maps,
            opt_state=opt_state,
            strike_step=strike_step,
            ts=ts,
            strike=strike_r,
            option_type=otype,
            token=token,
            opt_tl=otl,
            enabled_groups=enabled_groups,
            horizons_sec=horizons_sec,
        )
        if abs(ts - target_ts) < 0.01:
            return {
                "ctx": ctx,
                "opt_state": opt_state,
                "row": r,
                "recomputed": recomputed,
                "opt_tl": opt_tl,
                "strike": strike,
                "option_type": opt_type,
                "ts": ts,
            }
    raise ValueError("Could not replay builder state to target row")
