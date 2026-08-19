"""Data types and schemas for the Deterministic Strategy Evaluation Harness (Phase 4F.1).

Defines:
1. ExitReason: Enumeration of deterministic trade exit reasons.
2. TradeDirection: Enumeration of trade directions.
3. StrategyEvaluationPolicy: Research-configurable baseline evaluation parameters.
4. EvaluationTrade: Record of a single simulated trade.
5. RegimeTradeMetrics: Per-regime slice of trading metrics.
6. SessionTimeMetrics: Per-session-window slice of trading metrics.
7. TradingEvidenceDossier: Complete trading-level evidence telemetry dossier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExitReason(str, Enum):
    """Deterministic trade exit causes."""
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    TIME_EXPIRED = "TIME_EXPIRED"
    SESSION_CLOSE = "SESSION_CLOSE"
    NONE = "NONE"


class TradeDirection(str, Enum):
    """Trade direction for evaluation."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class StrategyEvaluationPolicy:
    """Research-configurable baseline evaluation parameters.
    
    All numeric parameters (+2.0%, -2.0%, 0.55, 60 bars) are baseline research candidate
    hypotheses, NOT authoritative production constants.
    """
    policy_id: str = "EVAL_POLICY_BASELINE_v1.0"
    min_confidence_threshold: float = 0.55       # Minimum P(Signal) required to enter
    target_return_pct: float = 2.0              # Target return hypothesis (+2.0%)
    stop_loss_pct: float = 2.0                  # Stop loss hypothesis (-2.0%)
    max_holding_bars: int = 60                  # Maximum holding duration in bars (e.g. 5m on 5s bars)
    cooldown_bars: int = 5                      # Mandatory bars to wait after trade exit before next entry
    direction_mode: str = "SIGNAL_DIRECTION"    # SIGNAL_DIRECTION, LONG_ONLY, SHORT_ONLY
    allow_multiple_open: bool = False           # Strictly 1 trade at a time (no averaging or compounding)
    session_cutoff_hhmm: str = "15:15"          # Intraday deterministic session cutoff

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationTrade:
    """Record of a single simulated evaluation trade."""
    trade_id: str
    entry_index: int
    entry_ts: int
    entry_price: float
    direction: TradeDirection
    exit_index: int | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_reason: ExitReason = ExitReason.NONE
    realized_return_pct: float = 0.0
    is_win: bool = False
    is_loss: bool = False
    is_scratch: bool = False
    mfe_pct: float = 0.0                        # Maximum Favorable Excursion (%)
    mae_pct: float = 0.0                        # Maximum Adverse Excursion (%)
    holding_bars: int = 0
    regime_id: str = "R000"
    fold_index: int | None = None
    entry_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["exit_reason"] = self.exit_reason.value
        return d


@dataclass
class RegimeTradeMetrics:
    """Trading metrics sliced by market regime."""
    regime_id: str
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    net_return_pct: float = 0.0
    mean_mfe_pct: float = 0.0
    mean_mae_pct: float = 0.0


@dataclass
class SessionTimeMetrics:
    """Trading metrics sliced by intraday time window."""
    window_name: str                            # Morning (09:15-11:30), Midday (11:30-13:30), Afternoon (13:30-15:30)
    trade_count: int = 0
    win_count: int = 0
    win_rate_pct: float = 0.0
    net_return_pct: float = 0.0


@dataclass
class TradingEvidenceDossier:
    """Complete trading-level evidence telemetry produced by the Strategy Evaluation Harness."""
    policy_id: str
    context_key: str
    model_name: str
    total_evaluated_rows: int
    total_signals_generated: int
    total_trades_executed: int
    winning_trades: int
    losing_trades: int
    scratch_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    gross_profit_pct: float
    gross_loss_pct: float
    profit_factor: float
    net_return_pct: float
    mean_trade_return_pct: float
    mean_mfe_pct: float
    max_mfe_pct: float
    mean_mae_pct: float
    max_mae_pct: float
    mfe_mae_efficiency_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    max_consecutive_losses: int
    max_consecutive_wins: int
    mean_holding_bars: float
    mean_time_to_target_bars: float | None
    mean_time_to_stop_bars: float | None
    target_hit_count: int
    stop_hit_count: int
    time_expired_count: int
    session_close_count: int
    regime_breakdown: dict[str, RegimeTradeMetrics] = field(default_factory=dict)
    time_of_day_breakdown: dict[str, SessionTimeMetrics] = field(default_factory=dict)
    trades: list[EvaluationTrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regime_breakdown"] = {k: asdict(v) for k, v in self.regime_breakdown.items()}
        d["time_of_day_breakdown"] = {k: asdict(v) for k, v in self.time_of_day_breakdown.items()}
        d["trades"] = [t.to_dict() if hasattr(t, "to_dict") else t for t in self.trades]
        return d
