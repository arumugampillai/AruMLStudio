"""Persistence bridge for Strategy Evaluation evidence in analysis.db (Phase 4F.1).

Directly persists granular trading metrics into the `benchmark_metrics` table in `<data_dir>/analysis.db`
with `metric_stage = 'TRADING_EVALUATION'`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Any

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import TradingEvidenceDossier


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_trading_evidence(
    data_dir: str,
    benchmark_id: int,
    dossier: TradingEvidenceDossier,
    *,
    fold_index: int | None = None,
) -> int:
    """Persist granular trading-level evidence metrics into analysis.db benchmark_metrics table.
    
    Returns:
        The count of scalar metrics written.
    """
    init_analysis_db(data_dir)
    now = _utc_now_iso()

    metric_tuples: list[tuple[str, float, str]] = [
        ("WIN_RATE_PCT", float(dossier.win_rate_pct), "PERCENTAGE"),
        ("PROFIT_FACTOR", float(dossier.profit_factor), "RATIO"),
        ("NET_RETURN_PCT", float(dossier.net_return_pct), "PERCENTAGE"),
        ("MAX_DRAWDOWN_PCT", float(dossier.max_drawdown_pct), "PERCENTAGE"),
        ("MEAN_MFE_PCT", float(dossier.mean_mfe_pct), "PERCENTAGE"),
        ("MAX_MFE_PCT", float(dossier.max_mfe_pct), "PERCENTAGE"),
        ("MEAN_MAE_PCT", float(dossier.mean_mae_pct), "PERCENTAGE"),
        ("MAX_MAE_PCT", float(dossier.max_mae_pct), "PERCENTAGE"),
        ("MFE_MAE_EFFICIENCY", float(dossier.mfe_mae_efficiency_ratio), "RATIO"),
        ("MAX_CONSECUTIVE_LOSSES", float(dossier.max_consecutive_losses), "COUNT"),
        ("TOTAL_TRADES_EXECUTED", float(dossier.total_trades_executed), "COUNT"),
        ("TOTAL_SIGNALS_GENERATED", float(dossier.total_signals_generated), "COUNT"),
        ("MEAN_HOLDING_BARS", float(dossier.mean_holding_bars), "COUNT"),
        ("TARGET_HIT_COUNT", float(dossier.target_hit_count), "COUNT"),
        ("STOP_HIT_COUNT", float(dossier.stop_hit_count), "COUNT"),
    ]

    # Add cross-regime win rates if available
    for reg_id, reg_m in dossier.regime_breakdown.items():
        metric_tuples.append((f"REGIME_{reg_id}_WIN_RATE_PCT", float(reg_m.win_rate_pct), "PERCENTAGE"))
        metric_tuples.append((f"REGIME_{reg_id}_NET_RETURN_PCT", float(reg_m.net_return_pct), "PERCENTAGE"))

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            for name, val, m_type in metric_tuples:
                conn.execute(
                    """
                    INSERT INTO benchmark_metrics (
                        benchmark_id, metric_name, metric_stage, fold_index,
                        metric_value, metric_type, created_at
                    ) VALUES (?, ?, 'TRADING_EVALUATION', ?, ?, ?, ?);
                    """,
                    (int(benchmark_id), str(name), fold_index, float(val), str(m_type), now),
                )
        return len(metric_tuples)
    finally:
        conn.close()


def get_trading_evidence_for_benchmark(
    data_dir: str,
    benchmark_id: int,
) -> dict[str, float]:
    """Retrieve all TRADING_EVALUATION stage metrics for a benchmark scorecard."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT metric_name, metric_value FROM benchmark_metrics
            WHERE benchmark_id = ? AND metric_stage = 'TRADING_EVALUATION'
            ORDER BY metric_id ASC;
            """,
            (int(benchmark_id),),
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    finally:
        conn.close()
