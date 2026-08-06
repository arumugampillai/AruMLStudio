"""Aggregate prediction-run research across all walk-forward folds."""

from __future__ import annotations

from collections import Counter
from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics

from .regime_iv import load_validation_feature_map
from .trade_replay import (
    _build_exit_analysis,
    _build_maximum_opportunity,
    _build_regime_badges,
    _build_since_entry,
    _build_trade_verdict,
    _find_entry_row,
    _find_exit_row,
    _num,
    explain_entry_decision,
)
from .trade_replay_insights import (
    build_prediction_failure_analysis,
    build_research_conclusion,
    classify_trade,
)

_ROOT_BUCKETS = (
    "Theta decay",
    "Wrong direction",
    "Low confidence",
    "Premium too low",
    "Strategy execution",
    "Range regime",
    "Other",
)


def _premium_threshold(trades: list[dict[str, Any]]) -> float:
    prices = [_num(t.get("entry_price")) for t in trades]
    prices = [p for p in prices if p is not None]
    if not prices:
        return 25.0
    prices.sort()
    idx = max(0, int(len(prices) * 0.25) - 1)
    return max(20.0, round(prices[idx]))


def _bucket_root_cause(
    *,
    trade: dict[str, Any],
    classification: dict[str, Any],
    prediction_failure: dict[str, Any],
    exit_analysis: dict[str, Any],
    decision: dict[str, Any],
    regime_badges: list[str],
    premium_threshold: float,
) -> str:
    contributors = [str(c) for c in (prediction_failure.get("contributors") or [])]
    badges_lower = [str(b).lower() for b in regime_badges]
    pred = (decision.get("prediction") or {})
    conf = _num(pred.get("confidence_pct")) or _num(pred.get("probability_success_pct"))
    entry_price = _num(trade.get("entry_price"))

    if any("theta" in c.lower() for c in contributors) or any(
        k in badges_lower for k in ("high theta", "theta zone")
    ):
        return "Theta decay"
    if (
        classification.get("primary") == "Model Failure"
        or exit_analysis.get("prediction_correct") in (0, False, "0")
        or any("direction" in c.lower() or "reversal" in c.lower() for c in contributors)
    ):
        return "Wrong direction"
    if conf is not None and conf < 70:
        return "Low confidence"
    if entry_price is not None and entry_price < premium_threshold:
        return "Premium too low"
    if classification.get("primary") in ("Strategy Failure", "Execution Failure", "Missed Opportunity"):
        return "Strategy execution"
    if classification.get("primary") == "Market Regime Failure" or "range" in badges_lower:
        return "Range regime"
    if contributors:
        first = contributors[0].lower()
        if "theta" in first:
            return "Theta decay"
        if "stagnation" in first or "reversal" in first:
            return "Wrong direction"
    return "Other"


def _analyze_losing_trade(
    trade: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    feature_rows: dict[int, dict[str, Any]] | None,
    direction: str,
    premium_threshold: float,
) -> dict[str, Any]:
    entry_row = _find_entry_row(trade, rows)
    exit_row = _find_exit_row(trade, rows)
    decision = explain_entry_decision(entry_row or {}, {}) if entry_row else {}

    exit_reason = str(trade.get("exit_reason") or "")
    stop_hit = exit_reason == "stop"
    target_hit = exit_reason == "target"
    premium_path = [
        {"timestamp": trade.get("entry_ts"), "value": trade.get("entry_price")},
        {"timestamp": trade.get("exit_ts"), "value": trade.get("exit_price")},
    ]
    exit_analysis = _build_exit_analysis(
        trade,
        entry_row,
        target_price=_num(trade.get("target_price")),
        stop_price=_num(trade.get("stop_price")),
        stop_hit=stop_hit,
        target_hit=target_hit,
        premium_path=premium_path,
        direction=direction,
    )
    max_opportunity = _build_maximum_opportunity(
        entry_price=_num(trade.get("entry_price")),
        exit_price=_num(trade.get("exit_price")),
        premium_path=premium_path,
        net_pnl=_num(trade.get("net_pnl")),
        direction=direction,
    )
    since_entry = _build_since_entry(entry_row, exit_row, feature_rows=feature_rows, direction=direction)
    regime_badges = _build_regime_badges(entry_row, rows, feature_rows, direction=direction)
    trade_verdict = _build_trade_verdict(
        trade,
        exit_analysis,
        since_entry,
        decision,
        stop_hit=stop_hit,
        target_hit=target_hit,
        max_opportunity=max_opportunity,
    )
    prediction_failure = build_prediction_failure_analysis(
        decision, since_entry, [], exit_analysis=exit_analysis,
    )
    classification = classify_trade(
        trade, trade_verdict, exit_analysis, max_opportunity, since_entry, regime_badges,
    )
    research_conclusion = build_research_conclusion(
        trade,
        decision,
        classification,
        prediction_failure,
        since_entry,
        max_opportunity,
        {},
        regime_badges,
    )
    bucket = _bucket_root_cause(
        trade=trade,
        classification=classification,
        prediction_failure=prediction_failure,
        exit_analysis=exit_analysis,
        decision=decision,
        regime_badges=regime_badges,
        premium_threshold=premium_threshold,
    )
    return {
        "root_cause_bucket": bucket,
        "research_conclusion": research_conclusion,
        "classification": classification,
        "entry_row": entry_row,
        "regime_badges": regime_badges,
        "decision": decision,
    }


def _normalize_recommendation(text: str, *, premium_threshold: float) -> str:
    raw = str(text or "").strip()
    low = raw.lower()
    if "premium" in low and "<" in low:
        return f"Avoid premium below ₹{int(premium_threshold)}"
    if "high theta" in low or "theta" in low:
        return "Filter high theta trades"
    if "range" in low:
        return "Skip range regime"
    if "confidence" in low:
        return "Increase confidence threshold to 70%"
    return raw


def _derive_filters(recommendations: list[dict[str, Any]], *, premium_threshold: float) -> dict[str, Any]:
    texts = {r.get("text") for r in recommendations}
    filters: dict[str, Any] = {}
    if any("premium below" in str(t).lower() for t in texts):
        filters["min_premium"] = premium_threshold
    if any("confidence" in str(t).lower() for t in texts):
        filters["min_confidence"] = 70.0
    if any("theta" in str(t).lower() for t in texts):
        filters["max_abs_theta"] = 0.45
    if any("range" in str(t).lower() for t in texts):
        filters["skip_range"] = True
    return filters


def _entry_features(entry_row: dict[str, Any] | None, feature_rows: dict[int, dict[str, Any]] | None) -> dict[str, Any]:
    if not entry_row or not feature_rows:
        return {}
    ri = entry_row.get("row_index")
    if ri is None:
        return {}
    return dict(feature_rows.get(int(ri)) or {})


def _trade_passes_filters(
    trade: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    feature_rows: dict[int, dict[str, Any]] | None,
    filters: dict[str, Any],
) -> bool:
    if not filters:
        return True
    entry_row = _find_entry_row(trade, rows)
    entry_price = _num(trade.get("entry_price"))
    if filters.get("min_premium") is not None and entry_price is not None:
        if entry_price < float(filters["min_premium"]):
            return False
    pred = (explain_entry_decision(entry_row or {}, {}).get("prediction") or {}) if entry_row else {}
    conf = _num(pred.get("confidence_pct")) or _num(pred.get("probability_success_pct"))
    if filters.get("min_confidence") is not None and conf is not None:
        if conf < float(filters["min_confidence"]):
            return False
    feats = _entry_features(entry_row, feature_rows)
    theta = _num(feats.get("theta"))
    if filters.get("max_abs_theta") is not None and theta is not None:
        if abs(theta) > float(filters["max_abs_theta"]):
            return False
    if filters.get("skip_range"):
        badges = _build_regime_badges(entry_row, rows, feature_rows, direction="long")
        if any("range" in str(b).lower() for b in badges):
            return False
    return True


def _pct_distribution(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for label in _ROOT_BUCKETS:
        count = counter.get(label, 0)
        if count <= 0:
            continue
        rows.append({
            "label": label,
            "count": count,
            "pct": round(count / total * 100.0, 1),
        })
    extra = [k for k in counter if k not in _ROOT_BUCKETS]
    for label in sorted(extra):
        count = counter[label]
        rows.append({"label": label, "count": count, "pct": round(count / total * 100.0, 1)})
    rows.sort(key=lambda r: r["pct"], reverse=True)
    return rows


def build_prediction_run_summary(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    from .service import _list_strategy_runs_for_fold, _load_fold_trades

    with PredictionRunStore(data_dir) as store:
        pred_run = store.get_run(prediction_run_id)
        if not pred_run:
            return {"ok": False, "error": "prediction run not found"}
        folds = sorted(store.list_folds(prediction_run_id), key=lambda f: int(f.get("fold_number") or 0))

    fold_summaries: list[dict[str, Any]] = []
    selected_run = strategy_run_id
    enriched: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = str(fold.get("fold_id") or "")
        with PredictionRunStore(data_dir) as store:
            rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)

        run_id = selected_run
        if not run_id:
            available = _list_strategy_runs_for_fold(data_dir, prediction_run_id, fold_id)
            run_id = available[0].get("strategy_run_id") if available else None
        if not run_id:
            fold_summaries.append({
                "fold_id": fold_id,
                "fold_number": fold.get("fold_number"),
                "trade_count": 0,
            })
            continue

        trades = _load_fold_trades(data_dir, str(run_id), fold_id)
        feature_map = load_validation_feature_map(data_dir, run=pred_run, fold=fold)
        for t in trades:
            enriched.append({
                "trade": dict(t),
                "fold_id": fold_id,
                "rows": rows,
                "feature_rows": feature_map,
            })
        fold_summaries.append({
            "fold_id": fold_id,
            "fold_number": fold.get("fold_number"),
            "trade_count": len(trades),
            "profit": compute_trade_metrics(trades).get("profit") if trades else None,
        })
        if not selected_run and run_id:
            selected_run = str(run_id)

    all_trades = [e["trade"] for e in enriched]

    if not all_trades:
        return {
            "ok": True,
            "prediction_run": pred_run,
            "strategy_run_id": selected_run,
            "fold_count": len(folds),
            "folds": fold_summaries,
            "trade_count": 0,
            "losing_trade_count": 0,
            "baseline_metrics": compute_trade_metrics([]),
            "root_causes": [],
            "recommendations": [],
            "estimated_improvement": None,
            "note": "No strategy trades found for this prediction run.",
        }

    baseline = compute_trade_metrics(all_trades)
    losers = [e for e in enriched if float(e["trade"].get("net_pnl") or 0) <= 0]
    premium_threshold = _premium_threshold(all_trades)

    root_counter: Counter[str] = Counter()
    rec_counter: Counter[str] = Counter()
    direction = "long"

    for item in losers:
        trade = item["trade"]
        rows = item["rows"]
        feature_rows = item["feature_rows"]
        try:
            insight = _analyze_losing_trade(
                trade,
                rows,
                feature_rows=feature_rows,
                direction=direction,
                premium_threshold=premium_threshold,
            )
        except Exception:
            root_counter["Other"] += 1
            continue
        root_counter[insight["root_cause_bucket"]] += 1
        rc = insight.get("research_conclusion") or {}
        for rec in rc.get("recommendations") or []:
            norm = _normalize_recommendation(str(rec), premium_threshold=premium_threshold)
            if norm:
                rec_counter[norm] += 1

    root_causes = _pct_distribution(root_counter, len(losers))
    recommendations = [
        {"text": text, "count": count, "selected": True}
        for text, count in rec_counter.most_common(6)
    ]

    filters = _derive_filters(recommendations, premium_threshold=premium_threshold)
    filtered: list[dict[str, Any]] = []
    for item in enriched:
        if _trade_passes_filters(
            item["trade"],
            item["rows"],
            feature_rows=item["feature_rows"],
            filters=filters,
        ):
            filtered.append(item["trade"])

    filtered_metrics = compute_trade_metrics(filtered)
    estimated_improvement = None
    if filters and filtered:
        estimated_improvement = {
            "filters_applied": filters,
            "trades_before": baseline.get("trade_count"),
            "trades_after": filtered_metrics.get("trade_count"),
            "profit_factor_before": baseline.get("profit_factor"),
            "profit_factor_after": filtered_metrics.get("profit_factor"),
            "win_rate_before_pct": baseline.get("win_rate_pct"),
            "win_rate_after_pct": filtered_metrics.get("win_rate_pct"),
            "profit_before": baseline.get("profit"),
            "profit_after": filtered_metrics.get("profit"),
        }

    return {
        "ok": True,
        "prediction_run": pred_run,
        "strategy_run_id": selected_run,
        "fold_count": len(folds),
        "folds": fold_summaries,
        "trade_count": baseline.get("trade_count"),
        "losing_trade_count": len(losers),
        "premium_threshold": premium_threshold,
        "baseline_metrics": baseline,
        "root_causes": root_causes,
        "recommendations": recommendations,
        "estimated_improvement": estimated_improvement,
    }
