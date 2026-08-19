"""Trading evidence metrics computation and aggregation (Phase 4F.1).

Computes comprehensive empirical trading telemetry:
1. Signal & Outcome Quality: Win Rate %, Profit Factor proxy, Net Return %
2. Excursion Dynamics: MFE %, MAE %, Efficiency Ratio
3. Risk & Drawdowns: Closed-trade Max Drawdown %, Drawdown Duration, Loss Streaks
4. Cross-Regime & Session distributions
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .types import (
    EvaluationTrade,
    ExitReason,
    RegimeTradeMetrics,
    SessionTimeMetrics,
    TradingEvidenceDossier,
)


def compute_trading_evidence(
    trades: list[EvaluationTrade],
    *,
    policy_id: str = "EVAL_POLICY_BASELINE_v1.0",
    context_key: str = "CONTEXT_KEY",
    model_name: str = "MODEL_NAME",
    total_evaluated_rows: int = 0,
    total_signals_generated: int = 0,
) -> TradingEvidenceDossier:
    """Aggregate simulated evaluation trades into an explainable TradingEvidenceDossier."""
    n_trades = len(trades)
    if n_trades == 0:
        return TradingEvidenceDossier(
            policy_id=policy_id,
            context_key=context_key,
            model_name=model_name,
            total_evaluated_rows=total_evaluated_rows,
            total_signals_generated=total_signals_generated,
            total_trades_executed=0,
            winning_trades=0,
            losing_trades=0,
            scratch_trades=0,
            win_rate_pct=0.0,
            loss_rate_pct=0.0,
            gross_profit_pct=0.0,
            gross_loss_pct=0.0,
            profit_factor=0.0,
            net_return_pct=0.0,
            mean_trade_return_pct=0.0,
            mean_mfe_pct=0.0,
            max_mfe_pct=0.0,
            mean_mae_pct=0.0,
            max_mae_pct=0.0,
            mfe_mae_efficiency_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            mean_holding_bars=0.0,
            mean_time_to_target_bars=None,
            mean_time_to_stop_bars=None,
            target_hit_count=0,
            stop_hit_count=0,
            time_expired_count=0,
            session_close_count=0,
            regime_breakdown={},
            time_of_day_breakdown={},
            trades=[],
        )

    returns = np.array([t.realized_return_pct for t in trades], dtype=float)
    mfes = np.array([t.mfe_pct for t in trades], dtype=float)
    maes = np.array([t.mae_pct for t in trades], dtype=float)
    holding_bars_arr = np.array([t.holding_bars for t in trades], dtype=int)

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if t.is_loss]
    scratches = [t for t in trades if t.is_scratch]

    n_wins = len(wins)
    n_losses = len(losses)
    n_scratches = len(scratches)

    win_rate = round(float(n_wins / n_trades) * 100.0, 2)
    loss_rate = round(float(n_losses / n_trades) * 100.0, 2)

    gross_profit = float(returns[returns > 0].sum()) if (returns > 0).any() else 0.0
    gross_loss = abs(float(returns[returns < 0].sum())) if (returns < 0).any() else 0.0

    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    net_return = round(float(returns.sum()), 2)
    mean_trade_ret = round(float(returns.mean()), 4)

    mean_mfe = round(float(mfes.mean()), 2)
    max_mfe = round(float(mfes.max()), 2)
    mean_mae = round(float(maes.mean()), 2)
    max_mae = round(float(maes.min()), 2)  # Most negative adverse move

    abs_mean_mae = abs(mean_mae) if abs(mean_mae) > 1e-4 else 1e-4
    efficiency_ratio = round(mean_mfe / abs_mean_mae, 2)

    # Calculate Cumulative Equity Curve & Peak-to-Trough Drawdown
    equity = np.cumsum(returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = round(float(drawdowns.max()) if len(drawdowns) > 0 else 0.0, 2)

    # Drawdown Duration (consecutive bars in drawdown)
    dd_duration = 0
    max_dd_duration = 0
    for dd in drawdowns:
        if dd > 1e-4:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Consecutive Streaks
    max_cons_wins = 0
    max_cons_losses = 0
    curr_wins = 0
    curr_losses = 0

    for t in trades:
        if t.is_win:
            curr_wins += 1
            curr_losses = 0
            max_cons_wins = max(max_cons_wins, curr_wins)
        elif t.is_loss:
            curr_losses += 1
            curr_wins = 0
            max_cons_losses = max(max_cons_losses, curr_losses)
        else:
            curr_wins = 0
            curr_losses = 0

    # Exit Reason Categorization
    target_hits = [t for t in trades if t.exit_reason == ExitReason.TARGET_HIT]
    stop_hits = [t for t in trades if t.exit_reason == ExitReason.STOP_HIT]
    time_expired = [t for t in trades if t.exit_reason == ExitReason.TIME_EXPIRED]
    session_closes = [t for t in trades if t.exit_reason == ExitReason.SESSION_CLOSE]

    mean_target_bars = round(float(np.mean([t.holding_bars for t in target_hits])), 1) if target_hits else None
    mean_stop_bars = round(float(np.mean([t.holding_bars for t in stop_hits])), 1) if stop_hits else None

    # Cross-Regime Slicing
    regime_breakdown: dict[str, RegimeTradeMetrics] = {}
    unique_regimes = sorted(list({t.regime_id for t in trades}))
    for reg in unique_regimes:
        r_trades = [t for t in trades if t.regime_id == reg]
        r_count = len(r_trades)
        r_wins = len([t for t in r_trades if t.is_win])
        r_losses = len([t for t in r_trades if t.is_loss])
        r_rets = np.array([t.realized_return_pct for t in r_trades])
        r_mfes = np.array([t.mfe_pct for t in r_trades])
        r_maes = np.array([t.mae_pct for t in r_trades])

        r_gp = float(r_rets[r_rets > 0].sum()) if (r_rets > 0).any() else 0.0
        r_gl = abs(float(r_rets[r_rets < 0].sum())) if (r_rets < 0).any() else 0.0
        r_pf = round(r_gp / r_gl, 2) if r_gl > 0 else (99.0 if r_gp > 0 else 0.0)

        regime_breakdown[reg] = RegimeTradeMetrics(
            regime_id=reg,
            trade_count=r_count,
            win_count=r_wins,
            loss_count=r_losses,
            win_rate_pct=round((r_wins / r_count) * 100.0, 2) if r_count > 0 else 0.0,
            profit_factor=r_pf,
            net_return_pct=round(float(r_rets.sum()), 2),
            mean_mfe_pct=round(float(r_mfes.mean()), 2) if r_count > 0 else 0.0,
            mean_mae_pct=round(float(r_maes.mean()), 2) if r_count > 0 else 0.0,
        )

    # Time-of-Day Slicing
    time_of_day_breakdown: dict[str, SessionTimeMetrics] = {}
    # Simple split into Morning (first 33% of trades), Midday (middle 33%), Afternoon (last 34%)
    if n_trades >= 3:
        n_third = n_trades // 3
        windows = [
            ("Morning (09:15-11:30)", trades[:n_third]),
            ("Midday (11:30-13:30)", trades[n_third:2 * n_third]),
            ("Afternoon (13:30-15:30)", trades[2 * n_third:]),
        ]
        for w_name, w_trades in windows:
            w_count = len(w_trades)
            w_wins = len([t for t in w_trades if t.is_win])
            w_rets = np.array([t.realized_return_pct for t in w_trades])
            time_of_day_breakdown[w_name] = SessionTimeMetrics(
                window_name=w_name,
                trade_count=w_count,
                win_count=w_wins,
                win_rate_pct=round((w_wins / w_count) * 100.0, 2) if w_count > 0 else 0.0,
                net_return_pct=round(float(w_rets.sum()), 2) if w_count > 0 else 0.0,
            )

    return TradingEvidenceDossier(
        policy_id=policy_id,
        context_key=context_key,
        model_name=model_name,
        total_evaluated_rows=total_evaluated_rows,
        total_signals_generated=total_signals_generated or n_trades,
        total_trades_executed=n_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        scratch_trades=n_scratches,
        win_rate_pct=win_rate,
        loss_rate_pct=loss_rate,
        gross_profit_pct=round(gross_profit, 2),
        gross_loss_pct=round(gross_loss, 2),
        profit_factor=profit_factor,
        net_return_pct=net_return,
        mean_trade_return_pct=mean_trade_ret,
        mean_mfe_pct=mean_mfe,
        max_mfe_pct=max_mfe,
        mean_mae_pct=mean_mae,
        max_mae_pct=max_mae,
        mfe_mae_efficiency_ratio=efficiency_ratio,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_bars=max_dd_duration,
        max_consecutive_losses=max_cons_losses,
        max_consecutive_wins=max_cons_wins,
        mean_holding_bars=round(float(holding_bars_arr.mean()), 1) if n_trades > 0 else 0.0,
        mean_time_to_target_bars=mean_target_bars,
        mean_time_to_stop_bars=mean_stop_bars,
        target_hit_count=len(target_hits),
        stop_hit_count=len(stop_hits),
        time_expired_count=len(time_expired),
        session_close_count=len(session_closes),
        regime_breakdown=regime_breakdown,
        time_of_day_breakdown=time_of_day_breakdown,
        trades=trades,
    )
