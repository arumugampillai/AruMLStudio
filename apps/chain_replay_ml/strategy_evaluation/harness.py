"""Pure, deterministic simulation replay harness (Phase 4F.1).

Evaluates model predictions by converting them into controlled, deterministic 1-position trades.
Strictly separates ML model evaluation from production strategy and averaging logic.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd

from .types import (
    EvaluationTrade,
    ExitReason,
    StrategyEvaluationPolicy,
    TradeDirection,
)


def run_deterministic_replay(
    df: pd.DataFrame,
    policy: StrategyEvaluationPolicy | None = None,
    *,
    price_col: str = "ltp",
    prob_col: str = "predicted_prob",
    pred_col: str = "predicted_class",
    regime_col: str = "regime_id",
    ts_col: str = "ts",
    fold_col: str = "fold_index",
) -> list[EvaluationTrade]:
    """Execute pure, deterministic trade replay on chronologically sorted out-of-fold predictions.
    
    Invariants:
    1. Single Position Only: Never averages down, compounds, or adds lots.
    2. Fixed Hypotheses: Evaluates fixed target/stop/time rules defined in `policy`.
    3. Zero Look-Ahead: Decides entries and exits strictly at current step; evaluates forward paths bar-by-bar.
    4. Deterministic Tie-Breaking: If both target and stop are reached in the same bar, conservative STOP_HIT applies.
    """
    pol = policy or StrategyEvaluationPolicy()
    if df is None or len(df) == 0:
        return []

    # Ensure required columns exist
    if price_col not in df.columns:
        # Fallback to spot if ltp missing
        price_col = "spot" if "spot" in df.columns else df.columns[0]

    prices = df[price_col].to_numpy(dtype=float)
    probs = df[prob_col].to_numpy(dtype=float) if prob_col in df.columns else np.ones(len(df))
    preds = df[pred_col].to_numpy() if pred_col in df.columns else np.ones(len(df))
    timestamps = df[ts_col].to_numpy() if ts_col in df.columns else np.arange(len(df))
    regimes = df[regime_col].to_numpy(dtype=str) if regime_col in df.columns else np.array(["R000"] * len(df))
    folds = df[fold_col].to_numpy() if fold_col in df.columns else np.zeros(len(df), dtype=int)

    trades: list[EvaluationTrade] = []
    active_trade: EvaluationTrade | None = None
    cooldown_remaining: int = 0
    trade_counter: int = 0

    n_rows = len(df)
    target_pct = float(pol.target_return_pct)
    stop_pct = float(pol.stop_loss_pct)
    max_bars = int(pol.max_holding_bars)
    cooldown_bars = int(pol.cooldown_bars)
    min_conf = float(pol.min_confidence_threshold)

    for i in range(n_rows):
        curr_price = float(prices[i])
        curr_ts = int(timestamps[i]) if hasattr(timestamps[i], "__int__") else i
        curr_regime = str(regimes[i])
        curr_fold = int(folds[i]) if hasattr(folds[i], "__int__") else None

        # Skip invalid or non-positive price rows
        if not np.isfinite(curr_price) or curr_price <= 0:
            continue

        # 1. Manage Active Trade
        if active_trade is not None:
            active_trade.holding_bars += 1
            entry_p = active_trade.entry_price

            # Calculate price return from entry
            if active_trade.direction == TradeDirection.LONG:
                ret_pct = ((curr_price - entry_p) / entry_p) * 100.0
            else:
                ret_pct = ((entry_p - curr_price) / entry_p) * 100.0

            # Track MFE and MAE
            if ret_pct > active_trade.mfe_pct:
                active_trade.mfe_pct = float(ret_pct)
            if ret_pct < active_trade.mae_pct:
                active_trade.mae_pct = float(ret_pct)

            # Check Exit Conditions
            is_exit = False
            exit_reason = ExitReason.NONE

            # Rule A: Target Hit
            if ret_pct >= target_pct:
                is_exit = True
                exit_reason = ExitReason.TARGET_HIT
            # Rule B: Stop Hit
            elif ret_pct <= -stop_pct:
                is_exit = True
                exit_reason = ExitReason.STOP_HIT
            # Rule C: Time Horizon Expired
            elif active_trade.holding_bars >= max_bars:
                is_exit = True
                exit_reason = ExitReason.TIME_EXPIRED
            # Rule D: End of Dataset / Last Row
            elif i == n_rows - 1:
                is_exit = True
                exit_reason = ExitReason.SESSION_CLOSE

            if is_exit:
                active_trade.exit_index = i
                active_trade.exit_ts = curr_ts
                active_trade.exit_price = curr_price
                active_trade.exit_reason = exit_reason
                active_trade.realized_return_pct = round(float(ret_pct), 4)
                active_trade.is_win = bool(ret_pct > 0.05)
                active_trade.is_loss = bool(ret_pct < -0.05)
                active_trade.is_scratch = bool(abs(ret_pct) <= 0.05)

                trades.append(active_trade)
                active_trade = None
                cooldown_remaining = cooldown_bars
                continue

        # 2. Check Cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # 3. Check Entry Condition (Flat State)
        if active_trade is None and i < n_rows - 1:
            conf = float(probs[i]) if np.isfinite(probs[i]) else 0.0
            pred = preds[i]

            # Signal Trigger Predicate
            should_enter = False
            trade_dir = TradeDirection.LONG

            # Check if prediction signal matches confidence threshold
            if conf >= min_conf:
                # Determine direction from prediction label or class
                if isinstance(pred, (int, np.integer, float, np.floating)):
                    if pred > 0:
                        should_enter = True
                        trade_dir = TradeDirection.LONG
                    elif pred < 0:
                        should_enter = True
                        trade_dir = TradeDirection.SHORT
                elif isinstance(pred, str):
                    p_lower = pred.lower().strip()
                    if p_lower in ("up", "long", "call", "1", "+1", "bullish", "hit", "target"):
                        should_enter = True
                        trade_dir = TradeDirection.LONG
                    elif p_lower in ("down", "short", "put", "-1", "bearish"):
                        should_enter = True
                        trade_dir = TradeDirection.SHORT

            # Apply Direction Mode Filtering
            if should_enter:
                if pol.direction_mode == "LONG_ONLY" and trade_dir != TradeDirection.LONG:
                    should_enter = False
                elif pol.direction_mode == "SHORT_ONLY" and trade_dir != TradeDirection.SHORT:
                    should_enter = False

            if should_enter:
                trade_counter += 1
                active_trade = EvaluationTrade(
                    trade_id=f"EV_{trade_counter:05d}",
                    entry_index=i,
                    entry_ts=curr_ts,
                    entry_price=curr_price,
                    direction=trade_dir,
                    mfe_pct=0.0,
                    mae_pct=0.0,
                    holding_bars=0,
                    regime_id=curr_regime,
                    fold_index=curr_fold,
                    entry_confidence=round(conf, 4),
                )

    return trades
