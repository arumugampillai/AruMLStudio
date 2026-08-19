"""Phase 4F.1: Deterministic Strategy Evaluation Harness & Trading Telemetry Engine."""

from .evaluator import evaluate_model_predictions
from .harness import run_deterministic_replay
from .metrics import compute_trading_evidence
from .persistence import (
    get_trading_evidence_for_benchmark,
    persist_trading_evidence,
)
from .types import (
    EvaluationTrade,
    ExitReason,
    RegimeTradeMetrics,
    SessionTimeMetrics,
    StrategyEvaluationPolicy,
    TradeDirection,
    TradingEvidenceDossier,
)

__all__ = [
    "EvaluationTrade",
    "ExitReason",
    "RegimeTradeMetrics",
    "SessionTimeMetrics",
    "StrategyEvaluationPolicy",
    "TradeDirection",
    "TradingEvidenceDossier",
    "compute_trading_evidence",
    "evaluate_model_predictions",
    "get_trading_evidence_for_benchmark",
    "persist_trading_evidence",
    "run_deterministic_replay",
]
