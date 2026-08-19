"""High-level Strategy Evaluation service (Phase 4F.1).

Public entry point for evaluating out-of-sample model predictions against the deterministic strategy harness.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from .harness import run_deterministic_replay
from .metrics import compute_trading_evidence
from .persistence import persist_trading_evidence
from .types import (
    EvaluationTrade,
    ExitReason,
    StrategyEvaluationPolicy,
    TradeDirection,
    TradingEvidenceDossier,
)


def evaluate_model_predictions(
    df_predictions: pd.DataFrame,
    policy: StrategyEvaluationPolicy | None = None,
    *,
    context_key: str = "CONTEXT_KEY",
    model_name: str = "MODEL_NAME",
    data_dir: str | None = None,
    benchmark_id: int | None = None,
    fold_index: int | None = None,
    price_col: str = "ltp",
    prob_col: str = "predicted_prob",
    pred_col: str = "predicted_class",
    regime_col: str = "regime_id",
    ts_col: str = "ts",
) -> TradingEvidenceDossier:
    """Run full out-of-sample strategy evaluation and optionally persist to analysis.db."""
    pol = policy or StrategyEvaluationPolicy()
    
    # 1. Run Replay Simulation
    trades = run_deterministic_replay(
        df_predictions,
        pol,
        price_col=price_col,
        prob_col=prob_col,
        pred_col=pred_col,
        regime_col=regime_col,
        ts_col=ts_col,
    )

    # 2. Compute Trading Evidence Dossier
    total_rows = len(df_predictions) if df_predictions is not None else 0
    dossier = compute_trading_evidence(
        trades,
        policy_id=pol.policy_id,
        context_key=context_key,
        model_name=model_name,
        total_evaluated_rows=total_rows,
    )

    # 3. Persist to analysis.db if requested
    if data_dir and benchmark_id is not None:
        persist_trading_evidence(data_dir, benchmark_id, dossier, fold_index=fold_index)

    return dossier
