"""Strategy quality / evidence scoring (freeze v1.0.0).

Decouples edge quality (0–100 + letter grade) from sample confidence
(Evidence Score). Net Profit Factor is always post-cost.

See ``docs/antigravity-doc/strategy_simulator_score_freeze_v1.md``.
"""

from __future__ import annotations

from typing import Any

SCORING_VERSION = "v1.0.0"

# Evidence Score → Sample Reliability (display-only; does not alter grade).
# High: >= 70 | Medium: >= 40 | Low: < 40
_EVIDENCE_HIGH = 70.0
_EVIDENCE_MEDIUM = 40.0


def get_strategy_grade(score: float) -> str:
    """
    Maps a normalized strategy quality score (0-100) to a letter grade.

    Grading Scale:
        >= 90.0 : A+ (Exceptional Edge - Production Candidate)
        >= 80.0 : A  (Strong Edge - Ready for Forward/Paper Testing)
        >= 70.0 : B  (Viable Edge - Requires Fine-Tuning)
        >= 60.0 : C  (Weak Edge - High Friction Sensitivity)
        <  60.0 : F  (Failed - Non-Viable Strategy)
    """
    if score >= 90.0:
        return "A+"
    elif score >= 80.0:
        return "A"
    elif score >= 70.0:
        return "B"
    elif score >= 60.0:
        return "C"
    else:
        return "F"


def sample_reliability_label(evidence_score: float) -> str:
    """Map Evidence Score (0–100) to High / Medium / Low (display only)."""
    if evidence_score >= _EVIDENCE_HIGH:
        return "High"
    if evidence_score >= _EVIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def count_active_trading_days(trades: list[dict[str, Any]] | None) -> int:
    """Distinct ``trading_day`` values among executed trades (empty → 0)."""
    days: set[str] = set()
    for t in trades or []:
        day = t.get("trading_day")
        if day is None:
            continue
        text = str(day).strip()
        if not text or text == "—":
            continue
        days.add(text)
    return len(days)


def evaluate_strategy(
    executed_trades: int,
    active_trading_days: int,
    net_profit: float,
    net_pf: float | None,
    max_dd: float,
    expectancy: float,
    stop_loss: float,
    win_rate: float,
    *,
    # ---------- Strategy Configurable Targets ----------
    target_net_pf: float = 1.50,
    target_romad: float = 2.00,
    target_exp_ratio: float = 0.10,
    minimum_win_rate: float = 0.30,
    target_win_rate: float = 0.60,
    # ---------- Evidence Configurable Targets ----------
    target_sample_size: int = 500,
    target_trading_days: int = 60,
) -> dict[str, Any]:
    """
    Evaluates algorithmic trading strategies by decoupling performance quality from
    sample size reliability.

    Strategy Quality Score (0-100)
    ------------------------------
    - 35% Net Profit Factor (Post-cost trade quality)
    - 35% RoMaD (Return relative to max drawdown)
    - 20% Expectancy / Risk (Average edge per trade relative to stop loss)
    - 10% Win Rate (Consistency & execution stability)

    Evidence Score (0-100%)
    -----------------------
    - 50% Executed Trades vs. Target Sample Size
    - 50% Active Trading Days vs. Target Trading Days

    Edge guards (freeze §13 resolutions):
    - ``net_pf is None`` (no losers): component = 100 when net_profit > 0 else 0.
    - ``max_dd <= 0``: RoMaD raw = 0 (component 0) — no special-case perfect score.
    """

    # 0. INPUT SANITIZATION & EDGE GUARDS
    if win_rate > 1.0:
        win_rate = win_rate / 100.0
    if minimum_win_rate > 1.0:
        minimum_win_rate = minimum_win_rate / 100.0
    if target_win_rate > 1.0:
        target_win_rate = target_win_rate / 100.0

    executed_trades = max(0, int(executed_trades or 0))
    active_trading_days = max(0, int(active_trading_days or 0))
    net_profit = float(net_profit or 0.0)
    max_dd = float(max_dd or 0.0)
    expectancy = float(expectancy or 0.0)
    stop_loss = float(stop_loss or 0.0)

    # 1. RAW METRICS DERIVATION
    # Max DD = 0 → RoMaD 0 (freeze formula; no perfect-cap special case).
    romad = (net_profit / max_dd) if max_dd > 0 else 0.0
    exp_ratio = (expectancy / stop_loss) if stop_loss > 0 else 0.0

    # 2. NORMALIZED COMPONENT SCORES (0.0 TO 100.0)

    # A. Net Profit Factor (35% Weight)
    # None = undefined PF (no losing side). Freeze §13 Q3.
    if net_pf is None:
        s_pf = 100.0 if net_profit > 0 else 0.0
        net_pf_raw: float | None = None
    else:
        net_pf_f = float(net_pf)
        net_pf_raw = net_pf_f
        pf_range = max(0.01, target_net_pf - 1.00)
        s_pf = max(
            0.0,
            min(
                100.0,
                ((net_pf_f - 1.00) / pf_range) * 100.0,
            ),
        )

    # B. RoMaD (35% Weight)
    s_romad = max(
        0.0,
        min(
            100.0,
            (romad / target_romad) * 100.0,
        ),
    )

    # C. Expectancy / Risk Ratio (20% Weight)
    s_exp = max(
        0.0,
        min(
            100.0,
            (exp_ratio / target_exp_ratio) * 100.0,
        ),
    )

    # D. Win Rate (10% Weight)
    wr_range = max(0.01, target_win_rate - minimum_win_rate)
    s_win = max(
        0.0,
        min(
            100.0,
            ((win_rate - minimum_win_rate) / wr_range) * 100.0,
        ),
    )

    # 3. STRATEGY QUALITY SCORE AGGREGATION
    strategy_score = (
        0.35 * s_pf +
        0.35 * s_romad +
        0.20 * s_exp +
        0.10 * s_win
    )
    strategy_score = round(strategy_score, 1)

    # 4. EVIDENCE SCORE (SAMPLE CONFIDENCE)
    trade_confidence = (
        min(1.0, executed_trades / target_sample_size)
        if target_sample_size > 0 else 1.0
    )
    day_confidence = (
        min(1.0, active_trading_days / target_trading_days)
        if target_trading_days > 0 else 1.0
    )
    evidence_score = (
        0.50 * trade_confidence +
        0.50 * day_confidence
    )
    evidence_score = round(evidence_score * 100.0, 1)

    # 5. RETURN TELEMETRY PAYLOAD
    return {
        "strategy_score": strategy_score,
        "grade": get_strategy_grade(strategy_score),
        "evidence_score": evidence_score,
        "sample_reliability": sample_reliability_label(evidence_score),
        "component_scores": {
            "profit_factor": round(s_pf, 1),
            "romad": round(s_romad, 1),
            "expectancy": round(s_exp, 1),
            "win_rate": round(s_win, 1),
        },
        "raw_metrics": {
            "net_profit_factor": (
                round(net_pf_raw, 3) if net_pf_raw is not None else None
            ),
            "romad": round(romad, 3),
            "expectancy_ratio": round(exp_ratio, 4),
            "win_rate": round(win_rate * 100, 2),
            "net_profit": round(net_profit, 2),
        },
        "sample_telemetry": {
            "executed_trades": executed_trades,
            "active_trading_days": active_trading_days,
            "target_sample_size": target_sample_size,
            "target_trading_days": target_trading_days,
        },
        "scoring_version": SCORING_VERSION,
    }


def evaluate_strategy_from_run(
    metrics: dict[str, Any] | None,
    trades: list[dict[str, Any]] | None = None,
    **target_overrides: Any,
) -> dict[str, Any]:
    """Map simulator run metrics + trades → ``evaluate_strategy`` args.

    Field mapping follows freeze §8. ``active_trading_days`` is taken from
    metrics when present, otherwise derived from unique trade ``trading_day``.
    """
    m = metrics if isinstance(metrics, dict) else {}
    trade_list = list(trades or [])

    executed = m.get("executed_trades")
    if executed is None:
        executed = m.get("trade_count")
    if executed is None:
        ss = m.get("simulator_summary") or {}
        executed = ss.get("executed_trades")
    try:
        executed_i = int(executed or 0)
    except (TypeError, ValueError):
        executed_i = 0

    active_days = m.get("active_trading_days")
    if active_days is None:
        active_days = count_active_trading_days(trade_list)
    try:
        active_days_i = int(active_days or 0)
    except (TypeError, ValueError):
        active_days_i = 0

    net_profit = m.get("net_profit")
    if net_profit is None:
        net_profit = m.get("profit")
    try:
        net_profit_f = float(net_profit if net_profit is not None else 0.0)
    except (TypeError, ValueError):
        net_profit_f = 0.0

    net_pf = m.get("profit_factor")
    if net_pf is None:
        audit = m.get("outcome_audit") if isinstance(m.get("outcome_audit"), dict) else {}
        net_pf = audit.get("profit_factor_after_fees")
    net_pf_f: float | None
    if net_pf is None:
        net_pf_f = None
    else:
        try:
            net_pf_f = float(net_pf)
        except (TypeError, ValueError):
            net_pf_f = None

    max_dd = m.get("account_equity_max_drawdown")
    if max_dd is None:
        max_dd = m.get("max_drawdown")
    try:
        max_dd_f = float(max_dd if max_dd is not None else 0.0)
    except (TypeError, ValueError):
        max_dd_f = 0.0

    expectancy = m.get("expectancy")
    if expectancy is None and executed_i > 0:
        expectancy = net_profit_f / executed_i
    try:
        expectancy_f = float(expectancy if expectancy is not None else 0.0)
    except (TypeError, ValueError):
        expectancy_f = 0.0

    stop_loss = m.get("stop_loss_per_trade_rupees")
    try:
        stop_loss_f = float(stop_loss if stop_loss is not None else 0.0)
    except (TypeError, ValueError):
        stop_loss_f = 0.0

    win_rate = m.get("win_rate_pct")
    if win_rate is None:
        win_rate = m.get("win_rate")
    try:
        win_rate_f = float(win_rate if win_rate is not None else 0.0)
    except (TypeError, ValueError):
        win_rate_f = 0.0

    return evaluate_strategy(
        executed_trades=executed_i,
        active_trading_days=active_days_i,
        net_profit=net_profit_f,
        net_pf=net_pf_f,
        max_dd=max_dd_f,
        expectancy=expectancy_f,
        stop_loss=stop_loss_f,
        win_rate=win_rate_f,
        **target_overrides,
    )


def attach_strategy_score(
    metrics: dict[str, Any],
    trades: list[dict[str, Any]] | None = None,
    **target_overrides: Any,
) -> dict[str, Any]:
    """Compute and nest score blob on metrics; also set ``active_trading_days``."""
    out = dict(metrics or {})
    trades_list = list(trades or [])
    if out.get("active_trading_days") is None:
        out["active_trading_days"] = count_active_trading_days(trades_list)
    score = evaluate_strategy_from_run(out, trades_list, **target_overrides)
    out["strategy_score_v1"] = score
    out["strategy_score"] = score.get("strategy_score")
    out["strategy_grade"] = score.get("grade")
    out["evidence_score"] = score.get("evidence_score")
    return out
