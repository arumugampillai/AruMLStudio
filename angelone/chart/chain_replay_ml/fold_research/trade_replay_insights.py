"""Trade replay insights — classification, prediction failure, research conclusion."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def confidence_tier(pct: float | None) -> str:
    if pct is None:
        return "Unknown"
    if pct >= 85:
        return "Very High"
    if pct >= 70:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"


def build_prediction_failure_analysis(
    decision: dict[str, Any],
    since_entry: dict[str, Any],
    feature_alerts: list[dict[str, Any]],
    *,
    exit_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred = (decision or {}).get("prediction") or {}
    expected_pct = _num(pred.get("prediction_pct"))
    ltp = _num(pred.get("current_ltp"))
    actual_ltp = _num(pred.get("actual_ltp"))
    actual_pct = None
    if ltp and actual_ltp and ltp > 0:
        actual_pct = round((actual_ltp - ltp) / ltp * 100.0, 2)
    difference_pct = None
    if expected_pct is not None and actual_pct is not None:
        difference_pct = round(expected_pct - actual_pct, 2)

    contributors: list[str] = []
    metrics = (since_entry or {}).get("metrics") or {}
    entry = (since_entry or {}).get("entry") or {}

    theta = _num(entry.get("theta"))
    if theta is not None and abs(theta) > 0.45:
        contributors.append("Theta decay")
    spot_chg = metrics.get("spot")
    if spot_chg is not None and abs(spot_chg) < 0.08:
        contributors.append("Spot stagnation")
    elif spot_chg is not None and spot_chg < -0.1:
        contributors.append("Spot reversal")
    iv_chg = metrics.get("iv")
    if iv_chg is not None and iv_chg < -1.0:
        contributors.append("IV contraction")
    prem_chg = metrics.get("premium")
    if prem_chg is not None and prem_chg < -3:
        contributors.append("Premium decay")

    for alert in sorted(feature_alerts or [], key=lambda a: {"HIGH": 0, "MEDIUM": 1}.get(str(a.get("severity")), 2)):
        label = str(alert.get("label") or "")
        if label and label not in contributors:
            contributors.append(label.replace("_", " ").title())

    model_wrong = exit_analysis and exit_analysis.get("prediction_correct") in (0, False, "0")
    failed = model_wrong or (difference_pct is not None and abs(difference_pct) > 5)

    return {
        "failed": failed,
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "difference_pct": difference_pct,
        "contributors": contributors[:5],
        "largest_contributor": contributors[0] if contributors else None,
    }


def classify_trade(
    trade: dict[str, Any],
    verdict: dict[str, Any],
    exit_analysis: dict[str, Any],
    max_opportunity: dict[str, Any],
    since_entry: dict[str, Any],
    regime_badges: list[str],
) -> dict[str, Any]:
    net_pnl = _num(trade.get("net_pnl"))
    model_v = (verdict.get("model") or {}).get("verdict", "")
    strat_v = (verdict.get("strategy") or {}).get("verdict", "")
    max_possible = _num(max_opportunity.get("maximum_possible"))
    captured = _num(max_opportunity.get("captured_profit"))
    eff = _num(max_opportunity.get("capture_efficiency_pct"))
    hold = float(trade.get("holding_seconds") or 0)

    tags: list[str] = []
    primary = "Noise Trade"

    if net_pnl is not None and net_pnl > 0 and model_v == "Correct" and strat_v == "Succeeded":
        primary = "Perfect Trade"
    elif net_pnl is not None and net_pnl > 0 and model_v == "Wrong":
        primary = "Lucky Winner"
    elif model_v == "Wrong" and strat_v != "Succeeded":
        primary = "Model Failure"
    elif model_v == "Correct" and strat_v == "Failed":
        primary = "Strategy Failure"
    elif eff is not None and eff < 30 and max_possible and max_possible > 0:
        primary = "Execution Failure"
    elif max_possible is not None and max_possible <= 0:
        primary = "Market Regime Failure"
    elif max_possible and captured is not None and max_possible > captured + 0.5:
        primary = "Missed Opportunity"

    badge_set = {str(b).lower() for b in regime_badges}
    if "range" in badge_set or "high theta" in badge_set or "theta zone" in badge_set:
        if primary == "Noise Trade":
            primary = "Market Regime Failure"

    if hold < 8 and abs(net_pnl or 0) < 1:
        tags.append("Noise Trade")

    tags.append(primary)
    return {"primary": primary, "tags": list(dict.fromkeys(tags))}


def build_research_conclusion(
    trade: dict[str, Any],
    decision: dict[str, Any],
    classification: dict[str, Any],
    prediction_failure: dict[str, Any],
    since_entry: dict[str, Any],
    max_opportunity: dict[str, Any],
    counterfactuals: dict[str, Any],
    regime_badges: list[str],
) -> dict[str, Any]:
    pred = (decision or {}).get("prediction") or {}
    entry = (since_entry or {}).get("entry") or {}
    premium = _num(entry.get("premium")) or _num(trade.get("entry_price"))
    conf = _num(pred.get("confidence_pct")) or _num(pred.get("probability_success_pct"))

    root_cause = prediction_failure.get("largest_contributor")
    if not root_cause:
        root_cause = classification.get("primary") or "Unclear"

    recommendations: list[str] = []
    if premium is not None:
        recommendations.append(f"Avoid entries when Premium < ₹{max(1, int(premium))}")
    badge_set = {str(b).lower() for b in regime_badges}
    if "high theta" in badge_set or "theta zone" in badge_set:
        recommendations.append("High Theta")
    if "range" in badge_set:
        recommendations.append("Range regime")
    if conf is not None and conf < 70:
        recommendations.append("Confidence < 70%")
    elif conf is None:
        recommendations.append("Raise confidence threshold when model confidence is enabled")

    best = None
    actual = _num(trade.get("net_pnl"))
    for sc in (counterfactuals or {}).get("scenarios") or []:
        if sc.get("is_actual"):
            continue
        p = _num(sc.get("profit"))
        if p is not None and (best is None or p > best):
            best = p
    expected_improvement = None
    if best is not None and actual is not None and best > actual:
        delta = best - actual
        if actual != 0:
            pf_hint = abs(delta / max(abs(actual), 0.01))
            expected_improvement = f"+{min(pf_hint * 0.4, 2.5):.1f} PF (est.)"
        else:
            expected_improvement = f"+₹{delta:.0f} per similar trade (est.)"

    return {
        "root_cause": root_cause,
        "recommendations": recommendations[:5],
        "expected_improvement": expected_improvement,
        "classification": classification.get("primary"),
    }
