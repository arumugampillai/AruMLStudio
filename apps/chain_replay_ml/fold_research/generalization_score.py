"""Generalization Score — multi-slice anti-overfitting gate (Phase D3)."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics
from chain_replay_ml.strategy_simulator.store import StrategyRunStore

from .experiment_pipeline_store import ExperimentPipelineStore


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _label_score(overall: int) -> str:
    if overall >= 85:
        return "Excellent"
    if overall >= 70:
        return "Good"
    if overall >= 50:
        return "Fair"
    return "Poor"


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / abs(mean)


def _slice_consistency_score(values: list[float], *, floor: float = 1.0) -> float:
    """Higher when all slices are above floor and stable."""
    if not values:
        return 40.0
    clean = [v for v in values if v is not None and v == v]
    if not clean:
        return 40.0
    above = sum(1 for v in clean if v >= floor) / len(clean)
    cv = _coefficient_of_variation(clean)
    stability = 100.0 if cv is None else max(0.0, 100.0 - cv * 80.0)
    level = min(100.0, statistics.mean(clean) / max(floor, 0.5) * 35.0)
    return round(above * 40.0 + stability * 0.35 + level * 0.25, 1)


def _ts_to_month_key(ts: Any) -> str | None:
    val = _num(ts)
    if val is None:
        return None
    if val > 1e12:
        val = val / 1000.0
    try:
        dt = datetime.fromtimestamp(val, tz=timezone.utc)
        return dt.strftime("%Y-%m")
    except (OSError, ValueError, OverflowError):
        return None


def extract_generalization_slices(data_dir: str, job: dict[str, Any]) -> dict[str, Any]:
    """Build PF slices across folds, months, regimes, and expiry proxies."""
    outputs = job.get("outputs") or {}
    strategy_run_id = str(outputs.get("strategy_run_id") or "")
    prediction_run_id = str(outputs.get("prediction_run_id") or outputs.get("baseline_prediction_run_id") or "")

    slices: dict[str, Any] = {
        "walk_forward_folds": [],
        "calendar_months": [],
        "volatility_regimes": [],
        "expiry_weeks": [],
    }

    if not strategy_run_id:
        return slices

    with StrategyRunStore(data_dir) as store:
        all_trades = store.list_trades(strategy_run_id, limit=10000)

    if not all_trades:
        return slices

    # Walk-forward folds
    fold_trades: dict[str, list[dict[str, Any]]] = {}
    for t in all_trades:
        fid = str(t.get("fold_id") or "unknown")
        fold_trades.setdefault(fid, []).append(t)

    fold_meta: dict[str, dict[str, Any]] = {}
    if prediction_run_id:
        with PredictionRunStore(data_dir) as store:
            for fold in store.list_folds(prediction_run_id):
                fold_meta[str(fold.get("fold_id") or "")] = fold

    for fid, trades in sorted(fold_trades.items(), key=lambda x: int((fold_meta.get(x[0]) or {}).get("fold_number") or 0)):
        m = compute_trade_metrics(trades)
        pf = _num(m.get("profit_factor"))
        meta = fold_meta.get(fid) or {}
        slices["walk_forward_folds"].append({
            "key": fid,
            "label": f"Fold {meta.get('fold_number') or fid[:6]}",
            "profit_factor": pf,
            "trade_count": m.get("trade_count"),
        })

    # Calendar months from exit timestamps
    month_trades: dict[str, list[dict[str, Any]]] = {}
    for t in all_trades:
        mk = _ts_to_month_key(t.get("exit_ts") or t.get("entry_ts"))
        if mk:
            month_trades.setdefault(mk, []).append(t)
    for mk in sorted(month_trades.keys()):
        m = compute_trade_metrics(month_trades[mk])
        slices["calendar_months"].append({
            "key": mk,
            "label": mk,
            "profit_factor": _num(m.get("profit_factor")),
            "trade_count": m.get("trade_count"),
        })

    # Volatility regime proxy — high vs low premium buckets
    regime_trades: dict[str, list[dict[str, Any]]] = {"low_premium": [], "mid_premium": [], "high_premium": []}
    for t in all_trades:
        prem = _num(t.get("entry_premium") or t.get("premium") or t.get("entry_price"))
        if prem is None:
            regime_trades["mid_premium"].append(t)
        elif prem < 20:
            regime_trades["low_premium"].append(t)
        elif prem >= 35:
            regime_trades["high_premium"].append(t)
        else:
            regime_trades["mid_premium"].append(t)
    for key, label in (
        ("low_premium", "Low premium"),
        ("mid_premium", "Mid premium"),
        ("high_premium", "High premium"),
    ):
        trades = regime_trades[key]
        if not trades:
            continue
        m = compute_trade_metrics(trades)
        slices["volatility_regimes"].append({
            "key": key,
            "label": label,
            "profit_factor": _num(m.get("profit_factor")),
            "trade_count": m.get("trade_count"),
        })

    # Expiry week proxy — bucket by fold train_end week label
    expiry_trades: dict[str, list[dict[str, Any]]] = {}
    for fid, trades in fold_trades.items():
        meta = fold_meta.get(fid) or {}
        end = str(meta.get("train_end") or meta.get("val_end") or fid)[:10]
        expiry_trades.setdefault(end or fid, []).extend(trades)
    for key in sorted(expiry_trades.keys())[:12]:
        m = compute_trade_metrics(expiry_trades[key])
        slices["expiry_weeks"].append({
            "key": key,
            "label": key,
            "profit_factor": _num(m.get("profit_factor")),
            "trade_count": m.get("trade_count"),
        })

    return slices


def compute_generalization_score(
    slices: dict[str, Any],
    *,
    baseline_pf: float | None = None,
) -> dict[str, Any]:
    """Multi-dimensional generalization score (domain model v1.1 weights)."""
    floor = max(1.0, (baseline_pf or 1.0) * 0.85)

    def _pfs(key: str) -> list[float]:
        return [float(x["profit_factor"]) for x in (slices.get(key) or []) if _num(x.get("profit_factor")) is not None]

    fold_pfs = _pfs("walk_forward_folds")
    month_pfs = _pfs("calendar_months")
    regime_pfs = _pfs("volatility_regimes")
    expiry_pfs = _pfs("expiry_weeks")

    dim_scores = {
        "walk_forward_folds": _slice_consistency_score(fold_pfs, floor=floor),
        "calendar_dates": _slice_consistency_score(month_pfs, floor=floor),
        "volatility_regimes": _slice_consistency_score(regime_pfs, floor=floor),
        "expiry_weeks": _slice_consistency_score(expiry_pfs, floor=floor),
    }

    all_pfs = fold_pfs + month_pfs + regime_pfs + expiry_pfs
    cv = _coefficient_of_variation(all_pfs) if len(all_pfs) >= 2 else None
    pf_stability = 75.0 if cv is None else max(0.0, min(100.0, 100.0 - cv * 100.0))

    weights = {
        "walk_forward_folds": 0.30,
        "calendar_dates": 0.25,
        "volatility_regimes": 0.20,
        "expiry_weeks": 0.15,
        "pf_stability": 0.10,
    }

    available_weight = 0.0
    weighted = 0.0
    for key, weight in weights.items():
        if key == "pf_stability":
            weighted += pf_stability * weight
            available_weight += weight
            continue
        vals = slices.get(key) or []
        if vals:
            weighted += dim_scores[key] * weight
            available_weight += weight

    overall = int(round(weighted / available_weight)) if available_weight > 0 else 0

    return {
        "overall": overall,
        "label": _label_score(overall),
        "promote_recommended": overall >= 70,
        "dimensions": {
            "walk_forward_folds": {
                "score": dim_scores["walk_forward_folds"],
                "weight_pct": 30,
                "slices": slices.get("walk_forward_folds") or [],
            },
            "calendar_dates": {
                "score": dim_scores["calendar_dates"],
                "weight_pct": 25,
                "slices": slices.get("calendar_months") or [],
            },
            "volatility_regimes": {
                "score": dim_scores["volatility_regimes"],
                "weight_pct": 20,
                "slices": slices.get("volatility_regimes") or [],
            },
            "expiry_weeks": {
                "score": dim_scores["expiry_weeks"],
                "weight_pct": 15,
                "slices": slices.get("expiry_weeks") or [],
            },
            "pf_stability": {
                "score": round(pf_stability, 1),
                "weight_pct": 10,
                "coefficient_of_variation": round(cv, 4) if cv is not None else None,
            },
        },
        "slice_summary": {
            "fold_pf_values": fold_pfs,
            "month_pf_values": month_pfs,
        },
    }


def compute_job_generalization(
    data_dir: str,
    job_id: str,
    *,
    baseline_pf: float | None = None,
) -> dict[str, Any]:
    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id)
    if not job:
        return {"ok": False, "error": "job not found"}
    if job.get("status") not in ("complete", "completed"):
        return {"ok": False, "error": "job is not complete"}

    comparison = job.get("comparison") or {}
    base_pf = baseline_pf or _num(comparison.get("baseline_pf"))
    slices = extract_generalization_slices(data_dir, job)
    score = compute_generalization_score(slices, baseline_pf=base_pf)
    return {"ok": True, "job_id": job_id, "slices": slices, **score}


def evaluate_campaign_best_generalization(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}

    memory = campaign.get("memory") or {}
    job_id = str(memory.get("best_job_id") or "")
    if not job_id:
        return {"ok": False, "error": "no best job recorded yet"}

    gen = compute_job_generalization(data_dir, job_id)
    if not gen.get("ok"):
        return gen

    memory = dict(memory)
    memory["best_generalization"] = {
        "overall": gen.get("overall"),
        "label": gen.get("label"),
        "promote_recommended": gen.get("promote_recommended"),
        "job_id": job_id,
        "dimensions": gen.get("dimensions"),
    }

    from .research_cycle import apply_cycle_decision, infer_exploration_stage

    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id) or {}
    comparison = job.get("comparison") or {}
    verdict = (job.get("results") or {}).get("verdict") or {}
    memory = apply_cycle_decision(
        memory,
        comparison=comparison,
        verdict=verdict,
        objective=campaign.get("resolved_objective"),
        generalization=gen,
    )
    memory["exploration_stage"] = infer_exploration_stage(memory)
    update_research_campaign(data_dir, campaign_id, memory=memory)
    return {"ok": True, "generalization": gen, "memory": memory}
