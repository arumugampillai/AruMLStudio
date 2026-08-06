"""Research Report — six-section synthesis for an entire prediction run."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics
from chain_replay_ml.strategy_simulator.store import StrategyRunStore

from .counterfactual import simulate_scenario
from .fold_quality import compute_fold_quality
from .prediction_analysis import analyze_prediction_rows
from .prediction_run_summary import (
    _analyze_losing_trade,
    _derive_filters,
    _normalize_recommendation,
    _pct_distribution,
    _premium_threshold,
    _trade_passes_filters,
)
from .regime_iv import analyze_iv_regimes, load_validation_feature_map


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _score_to_grade(score: int) -> str:
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 67:
        return "D+"
    return "D"


def _run_score(baseline: dict[str, Any], *, model_failure_pct: float) -> int:
    pf = _num(baseline.get("profit_factor")) or 1.0
    wr = _num(baseline.get("win_rate_pct")) or 50.0
    pf_part = min(pf / 2.5, 1.0) * 40
    wr_part = min(wr / 85.0, 1.0) * 35
    model_part = max(0, 25 - model_failure_pct * 0.4)
    return int(max(35, min(98, pf_part + wr_part + model_part)))


def _filter_scenarios(premium_threshold: float) -> list[dict[str, Any]]:
    return [
        {"key": "stop_7", "label": "If Stop = 7%", "stop_pct": 7.0},
        {"key": "confidence_70", "label": "If Confidence > 70%", "min_confidence": 70.0},
        {"key": "premium_floor", "label": f"If Premium > ₹{int(premium_threshold)}", "min_premium": premium_threshold},
    ]


def _scenario_filters(key: str, premium_threshold: float) -> dict[str, Any]:
    if key == "stop_7":
        return {}
    if key == "confidence_70":
        return {"min_confidence": 70.0}
    if key == "premium_floor":
        return {"min_premium": premium_threshold}
    return {}


def _star_rating(observed: int, pf_delta: float) -> int:
    if observed >= 2000 and pf_delta >= 0.15:
        return 5
    if observed >= 1200 and pf_delta >= 0.10:
        return 4
    if observed >= 600 and pf_delta >= 0.06:
        return 3
    if observed >= 200 and pf_delta >= 0.03:
        return 2
    return 1


def _collect_run_context(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None,
) -> dict[str, Any]:
    from .service import _list_strategy_runs_for_fold, _load_fold_trades

    with PredictionRunStore(data_dir) as store:
        pred_run = store.get_run(prediction_run_id)
        if not pred_run:
            return {"ok": False, "error": "prediction run not found"}
        folds = sorted(store.list_folds(prediction_run_id), key=lambda f: int(f.get("fold_number") or 0))

    selected_run = strategy_run_id
    enriched: list[dict[str, Any]] = []
    fold_ctx: list[dict[str, Any]] = []
    trading_days: set[str] = set()

    for fold in folds:
        fold_id = str(fold.get("fold_id") or "")
        with PredictionRunStore(data_dir) as store:
            rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)
        for row in rows:
            day = row.get("trading_day")
            if day:
                trading_days.add(str(day))

        run_id = selected_run
        if not run_id:
            available = _list_strategy_runs_for_fold(data_dir, prediction_run_id, fold_id)
            run_id = available[0].get("strategy_run_id") if available else None
        if not run_id:
            fold_ctx.append({"fold": fold, "fold_id": fold_id, "rows": rows, "trades": [], "feature_rows": {}})
            continue

        trades = _load_fold_trades(data_dir, str(run_id), fold_id)
        feature_map = load_validation_feature_map(data_dir, run=pred_run, fold=fold)
        pq = analyze_prediction_rows(rows)
        regime = analyze_iv_regimes(rows, feature_map)
        fq = compute_fold_quality(
            prediction_quality=pq,
            trading_metrics=compute_trade_metrics(trades) if trades else None,
            regime_analysis=regime,
            trade_count=len(trades),
        )
        fold_ctx.append({
            "fold": fold,
            "fold_id": fold_id,
            "rows": rows,
            "trades": trades,
            "feature_rows": feature_map,
            "fold_quality": fq,
            "prediction_quality": pq,
            "regime_analysis": regime,
        })
        for t in trades:
            enriched.append({
                "trade": dict(t),
                "fold_id": fold_id,
                "fold_number": fold.get("fold_number"),
                "rows": rows,
                "feature_rows": feature_map,
            })
        if not selected_run and run_id:
            selected_run = str(run_id)

    strategy_label = "—"
    if selected_run:
        with StrategyRunStore(data_dir) as sstore:
            sr = sstore.get_run(str(selected_run)) or {}
        strategy_label = str(sr.get("strategy_id") or sr.get("display_name") or selected_run[:12])

    return {
        "ok": True,
        "prediction_run": pred_run,
        "strategy_run_id": selected_run,
        "strategy_label": strategy_label,
        "folds": folds,
        "fold_ctx": fold_ctx,
        "enriched": enriched,
        "trading_days": sorted(trading_days),
    }


def build_research_report(
    data_dir: str,
    prediction_run_id: str,
    *,
    strategy_run_id: str | None = None,
) -> dict[str, Any]:
    ctx = _collect_run_context(data_dir, prediction_run_id, strategy_run_id=strategy_run_id)
    if not ctx.get("ok"):
        return ctx

    pred_run = ctx["prediction_run"]
    enriched = ctx["enriched"]
    all_trades = [e["trade"] for e in enriched]

    if not all_trades:
        return {
            "ok": True,
            "prediction_run_id": prediction_run_id,
            "prediction_run": pred_run,
            "strategy_run_id": ctx.get("strategy_run_id"),
            "executive_summary": {
                "model_id": pred_run.get("model_id"),
                "target": pred_run.get("target"),
                "strategy": ctx.get("strategy_label"),
                "trading_days": len(ctx.get("trading_days") or []),
                "fold_count": len(ctx.get("folds") or []),
                "trade_count": 0,
                "overall_grade": "—",
                "overall_score": None,
                "recommendation_flags": {
                    "worth_improving": False,
                    "not_production_ready": True,
                    "ready_for_live": False,
                },
            },
            "root_cause_analysis": {"items": []},
            "opportunity_analysis": {},
            "fold_ranking": [],
            "recommendations": [],
            "action_plan": {},
            "note": "No strategy trades found for this prediction run.",
        }

    baseline = compute_trade_metrics(all_trades)
    winners = [e for e in enriched if float(e["trade"].get("net_pnl") or 0) > 0]
    losers = [e for e in enriched if float(e["trade"].get("net_pnl") or 0) <= 0]
    premium_threshold = _premium_threshold(all_trades)
    direction = "long"

    root_counter: Counter[str] = Counter()
    root_trades: dict[str, list[str]] = defaultdict(list)
    root_folds: dict[str, set[str]] = defaultdict(set)
    fold_failure_counter: dict[str, Counter[str]] = defaultdict(Counter)
    trade_insights: dict[str, dict[str, Any]] = {}

    for item in losers:
        trade = item["trade"]
        tid = str(trade.get("trade_id") or "")
        fid = str(item["fold_id"] or "")
        try:
            insight = _analyze_losing_trade(
                trade,
                item["rows"],
                feature_rows=item["feature_rows"],
                direction=direction,
                premium_threshold=premium_threshold,
            )
        except Exception:
            bucket = "Others"
            root_counter[bucket] += 1
            root_trades[bucket].append(tid)
            root_folds[bucket].add(fid)
            fold_failure_counter[fid][bucket] += 1
            continue
        bucket = insight["root_cause_bucket"]
        if bucket == "Other":
            bucket = "Others"
        elif bucket == "Premium too low":
            bucket = f"Premium Below ₹{int(premium_threshold)}"
        elif bucket == "Wrong direction":
            bucket = "Wrong Prediction Direction"
        elif bucket == "Theta decay":
            bucket = "Theta Decay"
        elif bucket == "Range regime":
            bucket = "Range Market"
        trade_insights[tid] = insight
        root_counter[bucket] += 1
        root_trades[bucket].append(tid)
        root_folds[bucket].add(fid)
        fold_failure_counter[fid][bucket] += 1

    model_failures = sum(1 for i in trade_insights.values() if (i.get("classification") or {}).get("primary") == "Model Failure")
    model_failure_pct = (model_failures / len(losers) * 100.0) if losers else 0.0
    overall_score = _run_score(baseline, model_failure_pct=model_failure_pct)
    overall_grade = _score_to_grade(overall_score)

    pf = _num(baseline.get("profit_factor")) or 0
    wr = _num(baseline.get("win_rate_pct")) or 0

    recoverable = 0
    unrecoverable = 0
    for item in losers:
        trade = item["trade"]
        actual = _num(trade.get("net_pnl")) or 0
        entry = _num(trade.get("entry_price"))
        entry_ts = _num(trade.get("entry_ts"))
        path = [
            {"timestamp": entry_ts, "value": entry},
            {"timestamp": trade.get("exit_ts"), "value": trade.get("exit_price")},
        ]
        improved = False
        if entry and entry_ts:
            sim = simulate_scenario(path, entry_price=entry, entry_ts=entry_ts, direction=direction, stop_pct=7.0)
            if (_num(sim.get("profit")) or 0) > actual + 0.5:
                improved = True
        if not improved and _trade_passes_filters(
            trade, item["rows"], feature_rows=item["feature_rows"],
            filters={"min_confidence": 70.0},
        ) is False and actual < 0:
            improved = True
        if improved:
            recoverable += 1
        else:
            unrecoverable += 1

    scenario_rows: list[dict[str, Any]] = []
    for spec in _filter_scenarios(premium_threshold):
        key = spec["key"]
        if key == "stop_7":
            profit_delta = 0.0
            for item in losers:
                trade = item["trade"]
                actual = _num(trade.get("net_pnl")) or 0
                entry = _num(trade.get("entry_price"))
                entry_ts = _num(trade.get("entry_ts"))
                if not entry or not entry_ts:
                    continue
                path = [
                    {"timestamp": entry_ts, "value": entry},
                    {"timestamp": trade.get("exit_ts"), "value": trade.get("exit_price")},
                ]
                sim = simulate_scenario(path, entry_price=entry, entry_ts=entry_ts, direction=direction, stop_pct=7.0)
                profit_delta += max(0.0, (_num(sim.get("profit")) or 0) - actual)
            scenario_rows.append({
                "key": key,
                "label": spec["label"],
                "profit_delta": round(profit_delta, 2),
            })
            continue
        filt = _scenario_filters(key, premium_threshold)
        kept = [
            item["trade"] for item in enriched
            if _trade_passes_filters(item["trade"], item["rows"], feature_rows=item["feature_rows"], filters=filt)
        ]
        m = compute_trade_metrics(kept)
        profit_delta = (_num(m.get("profit")) or 0) - (_num(baseline.get("profit")) or 0)
        scenario_rows.append({
            "key": key,
            "label": spec["label"],
            "profit_delta": round(profit_delta, 2),
            "trade_count": len(kept),
        })

    combined_filters: dict[str, Any] = {
        "min_confidence": 70.0,
        "min_premium": premium_threshold,
        "max_abs_theta": 0.45,
        "skip_range": True,
    }
    combined_kept = [
        item["trade"] for item in enriched
        if _trade_passes_filters(item["trade"], item["rows"], feature_rows=item["feature_rows"], filters=combined_filters)
    ]
    combined_metrics = compute_trade_metrics(combined_kept)

    root_items = []
    total_losers = len(losers) or 1
    for label, count in root_counter.most_common():
        root_items.append({
            "label": label,
            "count": count,
            "pct": round(count / total_losers * 100.0, 1),
            "trade_ids": root_trades[label][:500],
            "fold_ids": sorted(root_folds[label]),
        })

    fold_ranking: list[dict[str, Any]] = []
    for fc in ctx["fold_ctx"]:
        fold = fc["fold"]
        fid = fc["fold_id"]
        fq = fc.get("fold_quality") or {}
        score = fq.get("total") or 0
        dom = fold_failure_counter.get(fid)
        reason = dom.most_common(1)[0][0] if dom else (fq.get("note") or "Stable")
        if reason == "Others" and fq.get("note"):
            reason = str(fq.get("note")).split("—")[0].strip() or reason
        fold_ranking.append({
            "fold_id": fid,
            "fold_number": fold.get("fold_number"),
            "score": score,
            "grade": _score_to_grade(int(score)),
            "reason": reason,
            "trade_count": len(fc.get("trades") or []),
        })
    fold_ranking.sort(key=lambda r: r["score"], reverse=True)

    rec_counter: Counter[str] = Counter()
    for item in losers:
        insight = trade_insights.get(str(item["trade"].get("trade_id") or ""))
        if not insight:
            continue
        for rec in (insight.get("research_conclusion") or {}).get("recommendations") or []:
            norm = _normalize_recommendation(str(rec), premium_threshold=premium_threshold)
            if norm:
                rec_counter[norm] += 1

    ranked_recs: list[dict[str, Any]] = []
    for text, observed in rec_counter.most_common(8):
        filt = _derive_filters([{"text": text}], premium_threshold=premium_threshold)
        kept = [item["trade"] for item in enriched if _trade_passes_filters(
            item["trade"], item["rows"], feature_rows=item["feature_rows"], filters=filt,
        )]
        m = compute_trade_metrics(kept)
        pf_before = _num(baseline.get("profit_factor")) or 1.0
        pf_after = _num(m.get("profit_factor")) or pf_before
        pf_delta = round(pf_after - pf_before, 2)
        ranked_recs.append({
            "text": text,
            "stars": _star_rating(observed, pf_delta),
            "observed_trades": observed,
            "expected_pf_delta": pf_delta,
        })
    ranked_recs.sort(key=lambda r: (r["stars"], r["observed_trades"], r["expected_pf_delta"]), reverse=True)

    flags = {
        "worth_improving": recoverable > len(losers) * 0.15 if losers else False,
        "not_production_ready": pf < 1.35 or wr < 60,
        "ready_for_live": pf >= 1.5 and wr >= 68 and overall_score >= 90,
    }

    action_items = [r["text"] for r in ranked_recs[:4]]
    if model_failure_pct > 20:
        action_items.append("Retrain model with theta / regime features")

    return {
        "ok": True,
        "prediction_run_id": prediction_run_id,
        "prediction_run": pred_run,
        "strategy_run_id": ctx.get("strategy_run_id"),
        "executive_summary": {
            "model_id": pred_run.get("model_id"),
            "target": pred_run.get("target"),
            "strategy": ctx.get("strategy_label"),
            "trading_days": len(ctx.get("trading_days") or []),
            "fold_count": len(ctx.get("folds") or []),
            "trade_count": baseline.get("trade_count"),
            "overall_grade": overall_grade,
            "overall_score": overall_score,
            "recommendation_flags": flags,
        },
        "root_cause_analysis": {"items": root_items, "losing_trade_count": len(losers)},
        "opportunity_analysis": {
            "total_trades": baseline.get("trade_count"),
            "winning": len(winners),
            "losing": len(losers),
            "recoverable": recoverable,
            "unrecoverable": unrecoverable,
            "scenarios": scenario_rows,
            "combined": {
                "label": "If all combined",
                "profit_factor_before": baseline.get("profit_factor"),
                "profit_factor_after": combined_metrics.get("profit_factor"),
                "trade_count": combined_metrics.get("trade_count"),
            },
        },
        "fold_ranking": fold_ranking,
        "recommendations": ranked_recs,
        "action_plan": {
            "next_experiment": action_items,
            "estimated_improvement": {
                "profit_factor_before": baseline.get("profit_factor"),
                "profit_factor_after": combined_metrics.get("profit_factor"),
                "win_rate_before_pct": baseline.get("win_rate_pct"),
                "win_rate_after_pct": combined_metrics.get("win_rate_pct"),
            },
        },
        "baseline_metrics": baseline,
        "premium_threshold": premium_threshold,
    }
