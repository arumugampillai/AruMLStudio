"""AI-style trade explanation narrative (rule-based research assistant)."""

from __future__ import annotations

from typing import Any


def explain_trade_narrative(replay_doc: dict[str, Any]) -> dict[str, Any]:
    """Produce a paragraph explanation for Explain This Trade."""
    trade = replay_doc.get("trade") or {}
    since = replay_doc.get("since_entry") or {}
    verdict = replay_doc.get("trade_verdict") or {}
    badges = replay_doc.get("regime_badges") or []
    similar = replay_doc.get("similar_trades") or []
    counter = replay_doc.get("counterfactuals") or {}
    metrics = since.get("metrics") or {}

    spot_chg = metrics.get("spot")
    prem_chg = metrics.get("premium")
    iv_chg = metrics.get("iv")

    parts: list[str] = []

    if spot_chg is not None and prem_chg is not None:
        if spot_chg > 0.1 and prem_chg < -3:
            parts.append(
                f"Spot remained bullish (+{spot_chg:.2f}%), but premium lost {abs(prem_chg):.1f}% "
                "because option premium did not follow spot."
            )
        elif spot_chg < -0.1:
            parts.append(f"Spot reversed ({spot_chg:.2f}%) during the hold window.")
        else:
            parts.append(f"Spot moved {spot_chg:+.2f}% while premium moved {prem_chg:+.2f}%.")

    if iv_chg is not None and iv_chg < -1:
        parts.append(f"IV contracted {iv_chg:.2f}%, pressuring premium.")
    theta = (since.get("entry") or {}).get("theta")
    if theta is not None and abs(float(theta)) > 0.45:
        parts.append("Theta decay accelerated relative to the move.")

    model_v = (verdict.get("model") or {}).get("verdict", "")
    strat_v = (verdict.get("strategy") or {}).get("verdict", "")
    if model_v == "Wrong":
        parts.append("The model overestimated continuation.")
    elif model_v == "Correct" and strat_v == "Failed":
        parts.append("The model direction was correct, but strategy execution underperformed.")

    regime_txt = ", ".join(badges[:4])
    if regime_txt:
        parts.append(f"Regime context: {regime_txt}.")

    if similar:
        fails = sum(1 for s in similar if (_num(s.get("net_pnl")) or 0) < 0)
        if fails >= 2:
            parts.append(f"{fails} of {len(similar)} similar trades in this fold also lost.")
        elif fails == 0 and similar:
            parts.append("Similar trades in this fold were mostly profitable — this one diverged.")

    best = counter.get("best_label")
    scenarios = counter.get("scenarios") or []
    if best and best != "Actual":
        alt = next((s for s in scenarios if s.get("label") == best), None)
        if alt and alt.get("profit") is not None:
            parts.append(
                f"Counterfactual replay suggests '{best}' would have yielded "
                f"₹{float(alt['profit']):.2f} vs actual ₹{float(trade.get('net_pnl') or 0):.2f}."
            )

    if "High Theta" in badges or "Theta Zone" in badges:
        parts.append("Consider raising confidence threshold or avoiding High Theta + Range combinations.")

    narrative = " ".join(parts) if parts else "Insufficient context to generate a detailed explanation."
    return {
        "narrative": narrative,
        "token": trade.get("token"),
        "trade_id": trade.get("trade_id"),
    }


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
