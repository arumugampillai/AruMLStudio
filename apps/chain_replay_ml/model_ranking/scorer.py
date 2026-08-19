"""Mathematical evidence scoring engine for Phase 4F.3.

Combines statistical model robustness (Phase 4D) and strategy replay telemetry (Phase 4F.1)
into a unified Pareto composite score.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from chain_replay_ml.research_memory.ranking import (
    RobustnessRankingPolicy,
    compute_robustness_score,
)
from .types import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    RANK_POLICY_v1_0,
    RecommendationClass,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_win_rate(win_rate_pct: float) -> float:
    """Normalize Win Rate (40% baseline -> 0.0, 70%+ -> 1.0)."""
    try:
        val = float(win_rate_pct)
    except (ValueError, TypeError):
        return 0.0
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, min(1.0, (val - 40.0) / (70.0 - 40.0)))


def normalize_profit_factor(profit_factor: float) -> float:
    """Normalize Profit Factor (0.80 baseline -> 0.0, 2.50+ -> 1.0)."""
    try:
        val = float(profit_factor)
    except (ValueError, TypeError):
        return 0.0
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, min(1.0, (val - 0.80) / (2.50 - 0.80)))


def normalize_mfe_mae_ratio(mfe_mae_ratio: float) -> float:
    """Normalize MFE/MAE efficiency capture ratio (0.50 baseline -> 0.0, 2.00+ -> 1.0)."""
    try:
        val = float(mfe_mae_ratio)
    except (ValueError, TypeError):
        return 0.0
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, min(1.0, (val - 0.50) / (2.00 - 0.50)))


def compute_trading_evidence_score(
    *,
    win_rate_pct: float,
    profit_factor: float,
    mfe_mae_ratio: float,
    max_drawdown_pct: float = 0.0,
    max_consecutive_losses: int = 0,
    total_trades: int = 0,
    regime_spread_pct: float = 0.0,
    policy: CandidateRankingPolicy = RANK_POLICY_v1_0,
) -> tuple[float, float, float, dict[str, float], list[str]]:
    """Compute normalized trading evidence score, risk penalties, and volume confidence.
    
    Returns:
        (trading_evidence_score, risk_penalty, volume_confidence, breakdown_dict, warnings)
    """
    warnings: list[str] = []

    # Check for NaN / Inf in inputs
    for name, val in [("win_rate_pct", win_rate_pct), ("profit_factor", profit_factor), ("mfe_mae_ratio", mfe_mae_ratio)]:
        if val is None or math.isnan(float(val)) or math.isinf(float(val)):
            return 0.0, 0.0, 0.0, {}, ["REJECTED_INVALID_TRADING_METRICS"]

    # 1. Sub-component normalizations
    norm_wr = normalize_win_rate(win_rate_pct)
    norm_pf = normalize_profit_factor(profit_factor)
    norm_eff = normalize_mfe_mae_ratio(mfe_mae_ratio)

    raw_trading = 100.0 * (
        (policy.alpha_win_rate * norm_wr)
        + (policy.alpha_profit_factor * norm_pf)
        + (policy.alpha_mfe_mae * norm_eff)
    )

    # 2. Risk Penalties
    p_dd = policy.lambda_drawdown * max(0.0, float(max_drawdown_pct) - policy.tau_safe_drawdown)
    p_streak = policy.lambda_loss_streak * max(0, int(max_consecutive_losses) - policy.tau_safe_loss_streak)
    p_reg = policy.lambda_reg_spread * max(0.0, float(regime_spread_pct) - policy.tau_safe_reg_spread)

    total_risk_penalty = p_dd + p_streak + p_reg

    if p_dd > 0:
        warnings.append(f"HIGH_DRAWDOWN_PENALTY_{max_drawdown_pct:.1f}%")
    if p_streak > 0:
        warnings.append(f"LOSS_STREAK_PENALTY_{max_consecutive_losses}_TRADES")
    if p_reg > 0:
        warnings.append(f"REGIME_FRAGILITY_SPREAD_PENALTY_{regime_spread_pct:.1f}%")

    # 3. Volume Confidence Adjustment
    t_count = max(0, int(total_trades))
    if t_count <= 0:
        vol_conf = 0.0
        warnings.append("ZERO_TRADES_EXECUTED")
    else:
        vol_conf = min(1.0, math.sqrt(t_count / max(1, policy.min_trade_volume)))
        if t_count < policy.min_trade_volume:
            warnings.append(f"LOW_TRADE_VOLUME_{t_count}_TRADES")

    # Net trading evidence score
    net_trade_score = max(0.0, min(100.0, (raw_trading - total_risk_penalty) * vol_conf))

    breakdown = {
        "raw_trading_score": round(raw_trading, 4),
        "normalized_win_rate": round(norm_wr, 4),
        "normalized_profit_factor": round(norm_pf, 4),
        "normalized_mfe_mae": round(norm_eff, 4),
        "drawdown_penalty": round(-p_dd, 4),
        "loss_streak_penalty": round(-p_streak, 4),
        "regime_spread_penalty": round(-p_reg, 4),
        "total_risk_penalty": round(-total_risk_penalty, 4),
        "volume_confidence_multiplier": round(vol_conf, 4),
        "net_trading_score": round(net_trade_score, 4),
    }

    return net_trade_score, total_risk_penalty, vol_conf, breakdown, warnings


def compute_composite_candidate_score(
    model_evidence_score: float,
    trading_evidence_score: float,
    *,
    policy: CandidateRankingPolicy = RANK_POLICY_v1_0,
) -> float:
    """Compute overall Pareto composite research score."""
    m_score = max(0.0, min(100.0, float(model_evidence_score)))
    t_score = max(0.0, min(100.0, float(trading_evidence_score)))

    composite = (policy.w_model * m_score) + (policy.w_trade * t_score)
    return max(0.0, min(100.0, round(composite, 4)))


def evaluate_candidate_evidence(
    *,
    candidate_id: str,
    signature_hash: str,
    context_key: str,
    model_metrics: dict[str, Any],
    trading_metrics: dict[str, Any],
    parent_candidate_id: str | None = None,
    parent_composite_score: float | None = None,
    opportunity_id: str | None = None,
    policy: CandidateRankingPolicy = RANK_POLICY_v1_0,
    champion_composite_score: float | None = None,
) -> CandidateEvidenceScore:
    """Evaluate and package full evidence score for a single candidate."""
    # 1. Model Robustness Score (Reuse Phase 4D compute_robustness_score)
    primary_metric = str(model_metrics.get("primary_metric", "roc_auc"))
    primary_val = float(model_metrics.get("primary_metric_value", model_metrics.get("roc_auc", 0.50)))
    fold_mean = float(model_metrics.get("fold_mean", primary_val))
    fold_std = float(model_metrics.get("fold_std", 0.0))
    ece = float(model_metrics.get("expected_calibration_error", 0.0)) if "expected_calibration_error" in model_metrics else None
    worst_fold_dd = float(model_metrics.get("worst_fold_drawdown", 0.0)) if "worst_fold_drawdown" in model_metrics else None
    exp_ratio = float(model_metrics.get("experimental_dependency_ratio", 0.0))
    dep_count = int(model_metrics.get("deprecated_feature_count", 0))
    total_feats = int(model_metrics.get("total_features", 30))

    m_score, m_breakdown, m_warnings = compute_robustness_score(
        primary_metric_name=primary_metric,
        primary_metric_value=primary_val,
        fold_mean=fold_mean,
        fold_std=fold_std,
        worst_fold_drawdown=worst_fold_dd,
        ece=ece,
        total_features=total_feats,
        experimental_dependency_ratio=exp_ratio,
        deprecated_feature_count=dep_count,
    )

    # 2. Trading Evidence Score (Phase 4F.1 metrics)
    wr = float(trading_metrics.get("win_rate_pct", 0.0))
    pf = float(trading_metrics.get("profit_factor", 0.0))
    mfe_mae = float(trading_metrics.get("mfe_mae_efficiency_ratio", trading_metrics.get("mfe_mae_ratio", 1.0)))
    max_dd = float(trading_metrics.get("max_drawdown_pct", 0.0))
    max_losses = int(trading_metrics.get("max_consecutive_losses", 0))
    t_count = int(trading_metrics.get("total_trades_executed", trading_metrics.get("total_trades", 0)))
    reg_spread = float(trading_metrics.get("regime_win_rate_spread_pct", 0.0))

    t_score, risk_pen, vol_conf, t_breakdown, t_warnings = compute_trading_evidence_score(
        win_rate_pct=wr,
        profit_factor=pf,
        mfe_mae_ratio=mfe_mae,
        max_drawdown_pct=max_dd,
        max_consecutive_losses=max_losses,
        total_trades=t_count,
        regime_spread_pct=reg_spread,
        policy=policy,
    )

    # 3. Composite Score
    comp_score = compute_composite_candidate_score(m_score, t_score, policy=policy)

    # 4. Recommendation Classification
    if risk_pen > 20.0 or comp_score < 50.0:
        rec_class = RecommendationClass.REJECTED
    elif champion_composite_score is not None and comp_score >= (champion_composite_score + policy.champion_beat_margin) and comp_score >= policy.champion_min_score:
        rec_class = RecommendationClass.CHAMPION_CANDIDATE
    elif comp_score >= 70.0:
        rec_class = RecommendationClass.STRONG_CONTENDER
    else:
        rec_class = RecommendationClass.BENCHMARK_ONLY

    # Delta vs parent
    delta_parent = round(comp_score - parent_composite_score, 4) if parent_composite_score is not None else None

    # Merge breakdowns
    score_breakdown = {
        "composite_score": comp_score,
        "model_evidence_score": round(m_score, 4),
        "trading_evidence_score": round(t_score, 4),
        "risk_penalty": round(risk_pen, 4),
        "volume_confidence": round(vol_conf, 4),
        **{f"model_{k}": v for k, v in m_breakdown.items()},
        **{f"trade_{k}": v for k, v in t_breakdown.items()},
    }

    all_warnings = list(m_warnings) + list(t_warnings)

    clean_model_metrics = {k: float(v) for k, v in model_metrics.items() if isinstance(v, (int, float))}
    clean_trading_metrics = {k: float(v) for k, v in trading_metrics.items() if isinstance(v, (int, float))}

    return CandidateEvidenceScore(
        candidate_id=candidate_id,
        signature_hash=signature_hash,
        context_key=context_key,
        composite_score=comp_score,
        model_evidence_score=round(m_score, 4),
        trading_evidence_score=round(t_score, 4),
        risk_penalty=round(risk_pen, 4),
        volume_confidence=round(vol_conf, 4),
        recommendation_class=rec_class,
        model_metrics=clean_model_metrics,
        trading_metrics=clean_trading_metrics,
        score_breakdown=score_breakdown,
        warnings=all_warnings,
        parent_candidate_id=parent_candidate_id,
        delta_vs_parent=delta_parent,
        opportunity_id=opportunity_id,
        evaluated_at=_utc_now_iso(),
    )
